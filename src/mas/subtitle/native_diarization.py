import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
import wave
from array import array
from datetime import datetime, timezone
from pathlib import Path

from ..reliability import IntegrityError, atomic_json, digest, file_digest


class ModelConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("model_path", ctypes.c_char_p),
                ("gpu", ctypes.c_int32), ("preset", ctypes.c_char_p),
                ("chunk_frames", ctypes.c_int32), ("right_context_frames", ctypes.c_int32),
                ("left_context_frames", ctypes.c_int32), ("fifo_frames", ctypes.c_int32),
                ("spkcache_frames", ctypes.c_int32), ("update_period_frames", ctypes.c_int32)]


class Segment(ctypes.Structure):
    _fields_ = [("start_time", ctypes.c_double), ("end_time", ctypes.c_double),
                ("speaker", ctypes.c_int32)]


class NativeAPI:
    def __init__(self, runtime_dir):
        runtime_dir = Path(runtime_dir).resolve()
        self.dll_directory = None
        if os.name == "nt":
            self.dll_directory = os.add_dll_directory(str(runtime_dir))
        self.lib = ctypes.CDLL(str(runtime_dir / "nemo_speech_asr_c.dll"))
        void_p = ctypes.c_void_p
        self.lib.nemo_speech_diar_create.argtypes = [ctypes.POINTER(ModelConfig), ctypes.POINTER(void_p)]
        self.lib.nemo_speech_diar_create.restype = ctypes.c_int
        self.lib.nemo_speech_diar_destroy.argtypes = [void_p]
        self.lib.nemo_speech_diar_destroy.restype = None
        self.lib.nemo_speech_diar_stream_open.argtypes = [void_p, ctypes.POINTER(void_p)]
        self.lib.nemo_speech_diar_stream_open.restype = ctypes.c_int
        self.lib.nemo_speech_diar_stream_push_f32.argtypes = [void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32]
        self.lib.nemo_speech_diar_stream_push_f32.restype = ctypes.c_int
        self.lib.nemo_speech_diar_stream_finish.argtypes = [void_p]
        self.lib.nemo_speech_diar_stream_finish.restype = ctypes.c_int
        self.lib.nemo_speech_diar_stream_close.argtypes = [void_p]
        self.lib.nemo_speech_diar_stream_close.restype = None
        self.lib.nemo_speech_diar_num_speakers.argtypes = [void_p]
        self.lib.nemo_speech_diar_num_speakers.restype = ctypes.c_int32
        self.lib.nemo_speech_diar_seconds_per_frame.argtypes = [void_p]
        self.lib.nemo_speech_diar_seconds_per_frame.restype = ctypes.c_double
        self.lib.nemo_speech_diar_frame_count.argtypes = [void_p]
        self.lib.nemo_speech_diar_frame_count.restype = ctypes.c_int64
        self.lib.nemo_speech_diar_frame_probs_start.argtypes = [void_p]
        self.lib.nemo_speech_diar_frame_probs_start.restype = ctypes.c_int64
        self.lib.nemo_speech_diar_frame_probs.argtypes = [void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self.lib.nemo_speech_diar_frame_probs.restype = ctypes.c_int
        self.lib.nemo_speech_diar_segments.argtypes = [void_p, void_p, ctypes.POINTER(Segment), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nemo_speech_diar_segments.restype = ctypes.c_int
        self.lib.nemo_speech_asr_last_error.restype = ctypes.c_char_p

    def check(self, status):
        if status:
            message = self.lib.nemo_speech_asr_last_error()
            raise IntegrityError((message or b"native diarization failed").decode("utf-8", "replace"))


def _runtime_identity(runtime_dir):
    paths = sorted((path for path in Path(runtime_dir).iterdir()
                    if path.is_file() and path.suffix.lower() in (".dll", ".exe")),
                   key=lambda path: path.name.lower())
    if not paths:
        raise IntegrityError("native diarization runtime files are missing")
    return [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_digest(path)}
            for path in paths]


def _read_wav(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise IntegrityError("native diarization requires mono 16000 Hz PCM16 WAV")
        rate, frames = stream.getframerate(), stream.getnframes()
        if not 0 < frames / rate <= 120:
            raise IntegrityError("native diarization pilot clip must be at most 120 seconds")
        payload = stream.readframes(frames)
        if len(payload) != frames * 2:
            raise IntegrityError("native diarization WAV payload is truncated")
        pcm = array("h")
        pcm.frombytes(payload)
        if sys.byteorder != "little":
            pcm.byteswap()
    return rate, [sample / 32768.0 for sample in pcm]


def _native_result(api, model_path, samples, sample_rate):
    model, stream = ctypes.c_void_p(), ctypes.c_void_p()
    encoded = str(Path(model_path).resolve()).encode("utf-8")
    config = ModelConfig(ctypes.sizeof(ModelConfig), encoded, -1, b"streaming", 0, 0, -1, 0, 0, 0)
    try:
        api.check(api.lib.nemo_speech_diar_create(ctypes.byref(config), ctypes.byref(model)))
        if api.lib.nemo_speech_diar_num_speakers(model) != 4:
            raise IntegrityError("native diarization model must expose four speaker columns")
        frame_seconds = api.lib.nemo_speech_diar_seconds_per_frame(model)
        if not math.isclose(frame_seconds, 0.08, abs_tol=1e-6):
            raise IntegrityError("native diarization frame duration is not 80 ms")
        api.check(api.lib.nemo_speech_diar_stream_open(model, ctypes.byref(stream)))
        audio = (ctypes.c_float * len(samples))(*samples)
        api.check(api.lib.nemo_speech_diar_stream_push_f32(stream, audio, len(samples), sample_rate))
        api.check(api.lib.nemo_speech_diar_stream_finish(stream))
        frame_count = api.lib.nemo_speech_diar_frame_count(stream)
        frame_start = api.lib.nemo_speech_diar_frame_probs_start(stream)
        maximum_frames = math.ceil(len(samples) / sample_rate / 0.08) + 2
        if frame_count <= 0 or frame_count > maximum_frames or frame_start != 0:
            raise IntegrityError("native diarization did not retain all pilot probabilities")
        values = (ctypes.c_float * (frame_count * 4))()
        api.check(api.lib.nemo_speech_diar_frame_probs(stream, values, len(values)))
        probabilities = [list(values[index:index + 4]) for index in range(0, len(values), 4)]
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for row in probabilities for value in row):
            raise IntegrityError("native diarization probabilities are invalid")
        count = ctypes.c_size_t()
        api.check(api.lib.nemo_speech_diar_segments(stream, None, None, 0, ctypes.byref(count)))
        native_segments = (Segment * count.value)()
        api.check(api.lib.nemo_speech_diar_segments(stream, None, native_segments,
                                                    count.value, ctypes.byref(count)))
        segments = [{"start": item.start_time, "end": item.end_time,
                     "speaker": item.speaker} for item in native_segments]
        return frame_seconds, probabilities, segments
    finally:
        if stream.value:
            api.lib.nemo_speech_diar_stream_close(stream)
        if model.value:
            api.lib.nemo_speech_diar_destroy(model)


def worker(request_path, api_factory=NativeAPI):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    audio, model, runtime_dir = map(Path, (request["audio"], request["model"], request["runtime_dir"]))
    before = file_digest(audio)
    if before != request["input_sha256"] or file_digest(model) != request["model_sha256"]:
        raise IntegrityError("native diarization input or model changed")
    runtime_before = _runtime_identity(runtime_dir)
    if runtime_before != request["runtime"]:
        raise IntegrityError("native diarization runtime differs from request")
    rate, samples = _read_wav(audio)
    started_at, started = datetime.now(timezone.utc).isoformat(), time.monotonic()
    frame_seconds, probabilities, segments = _native_result(api_factory(runtime_dir), model, samples, rate)
    if file_digest(audio) != before or file_digest(model) != request["model_sha256"]:
        raise IntegrityError("native diarization source or model changed during inference")
    if _runtime_identity(runtime_dir) != runtime_before:
        raise IntegrityError("native diarization runtime changed during inference")
    if (not math.isclose(frame_seconds, 0.08, abs_tol=1e-6) or not probabilities
            or len(probabilities) > math.ceil(len(samples) / rate / 0.08) + 2
            or any(len(row) != 4 for row in probabilities)
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for row in probabilities for value in row)):
        raise IntegrityError("native diarization frame probabilities are invalid")
    if any(not math.isfinite(item["start"]) or not math.isfinite(item["end"])
           or item["start"] < 0 or item["end"] <= item["start"]
           or item["speaker"] not in (1, 2, 3, 4) for item in segments):
        raise IntegrityError("native diarization segments are invalid")
    body = {"format": "mas-native-diarization-pilot-1", "status": "REVIEW_REQUIRED",
            "production_acceptance": False, "episode": request["episode"],
            "clip_identity": request["clip_identity"], "clip_lineage": "UNVERIFIED",
            "input_sha256": before,
            "model_sha256": request["model_sha256"], "runtime": runtime_before,
            "config": {"device": "cpu", "preset": "streaming", "max_duration_seconds": 120},
            "started_at_utc": started_at, "elapsed_seconds": time.monotonic() - started,
            "sample_rate": rate, "frame_seconds": frame_seconds,
            "frame_count": len(probabilities), "speaker_columns": 4,
            "probabilities": probabilities, "segments": segments}
    atomic_json(request["output"], {"data": body, "sha256": digest(body)})


def export_native(audio, model, runtime_dir, output, *, episode, clip_identity, timeout=180):
    audio, model, output = Path(audio), Path(model), Path(output)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 180:
        raise ValueError("native diarization timeout must be finite and in (0, 180] seconds")
    if output.exists() or output.with_suffix(".request.json").exists():
        raise IntegrityError("native diarization output exists; choose a new path")
    runtime_identity = _runtime_identity(runtime_dir)
    request = {"audio": str(audio.resolve()), "model": str(model.resolve()),
               "runtime_dir": str(Path(runtime_dir).resolve()), "output": str(output.resolve()),
               "episode": episode, "clip_identity": clip_identity,
               "input_sha256": file_digest(audio), "model_sha256": file_digest(model),
               "runtime": runtime_identity}
    request_path = output.with_suffix(".request.json")
    atomic_json(request_path, request)
    process = subprocess.Popen([sys.executable, "-m", "mas.subtitle.native_diarization",
                                "--worker", str(request_path)], stdin=subprocess.DEVNULL,
                               start_new_session=(os.name == "posix"),
                               creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process(process)
        raise TimeoutError(f"native diarization exceeded {timeout} seconds")
    except BaseException:
        _kill_process(process)
        raise
    if process.returncode:
        raise RuntimeError(f"native diarization worker exited with code {process.returncode}")
    wrapped = json.loads(output.read_text(encoding="utf-8"))
    body = wrapped.get("data")
    if (not isinstance(body, dict) or wrapped.get("sha256") != digest(body)
            or body.get("input_sha256") != request["input_sha256"]
            or body.get("model_sha256") != request["model_sha256"]
            or body.get("runtime") != runtime_identity
            or body.get("episode") != episode or body.get("clip_identity") != clip_identity
            or body.get("status") != "REVIEW_REQUIRED" or body.get("production_acceptance") is not False):
        raise IntegrityError("native diarization output binding or checksum mismatch")
    return wrapped


def _kill_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker)


if __name__ == "__main__":
    main()
