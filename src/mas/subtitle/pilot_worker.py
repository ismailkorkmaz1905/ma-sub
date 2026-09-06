import importlib.metadata
import json
import os
import sys
import time
import wave
from pathlib import Path

from ..reliability import IntegrityError, atomic_json, digest, file_digest
from .pilot import build_pilot, write_pilot


def model_identity(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise IntegrityError("pilot models must already exist in local directories")
    files = {str(path.relative_to(directory)).replace("\\", "/"): file_digest(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    if not files:
        raise IntegrityError("model directory is empty")
    return digest(files)


def cached_stage(path, binding, operation):
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        body = saved.get("data")
        if (not isinstance(body, dict) or saved.get("sha256") != digest(body)
                or body.get("binding") != binding):
            raise IntegrityError("pilot stage cache binding mismatch; preserve prior results")
        return body["result"]
    started = time.monotonic()
    result = operation()
    body = {"binding": binding, "result": result,
            "elapsed_seconds": round(time.monotonic() - started, 3)}
    atomic_json(path, {"data": body, "sha256": digest(body)})
    return result


def verify_source_clip(audio, source, source_sha256, offset_ms):
    if file_digest(Path(source)) != source_sha256:
        raise IntegrityError("source audio SHA mismatch")
    with wave.open(str(source), "rb") as original, wave.open(str(audio), "rb") as clip:
        if (original.getnchannels(), original.getsampwidth(), original.getframerate()) != (
                clip.getnchannels(), clip.getsampwidth(), clip.getframerate()):
            raise IntegrityError("sample and source WAV formats differ")
        frame = offset_ms * original.getframerate() // 1000
        if frame + clip.getnframes() > original.getnframes():
            raise IntegrityError("sample exceeds source audio")
        original.setpos(frame)
        if original.readframes(clip.getnframes()) != clip.readframes(clip.getnframes()):
            raise IntegrityError("sample bytes do not match source offset")


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    audio, output = Path(request["audio"]), Path(request["output"])
    if file_digest(audio) != request["audio_sha256"]:
        raise IntegrityError("sample changed after preflight")
    verify_source_clip(audio, request["source_audio"], request["source_sha256"], request["offset_ms"])
    if (model_identity(request["model_dir"]) != request["model_sha256"]
            or model_identity(request["diarization_model_dir"]) != request["diarization_sha256"]):
        raise IntegrityError("model files changed after preflight")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    import torch
    import stable_whisper
    from pyannote.audio import Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires CUDA; no CPU fallback and no Pod startup")
    expected = {"stable-ts": "2.19.1", "pyannote.audio": "4.0.0", "faster-whisper": "1.2.1"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError("pilot dependency versions do not match requirements-pilot.lock")
    with wave.open(str(audio), "rb") as stream:
        duration = round(stream.getnframes() * 1000 / stream.getframerate())
    if not 0 < duration <= 120_000:
        raise IntegrityError("pilot accepts at most 120 seconds of WAV audio")
    binding = {"audio_sha256": file_digest(audio), "versions": versions, "device": "cuda",
               "worker_sha256": file_digest(Path(__file__)),
               "model_sha256": model_identity(request["model_dir"]),
               "diarization_sha256": model_identity(request["diarization_model_dir"])}

    def transcribe():
        model = stable_whisper.load_faster_whisper(
            request["model_dir"], device="cuda", compute_type="float16", local_files_only=True)
        result = model.transcribe(str(audio), language="tr", word_timestamps=True,
                                  regroup=False, suppress_silence=False, vad_filter=False,
                                  condition_on_previous_text=False, beam_size=5,
                                  temperature=0.0, verbose=False)
        data = result.to_dict()
        del model
        torch.cuda.empty_cache()
        return data

    asr_binding = {key: value for key, value in binding.items() if key != "diarization_sha256"}
    asr_binding["versions"] = {key: value for key, value in versions.items() if key != "pyannote.audio"}
    cache_root = output.parent / "acoustic-cache"
    transcript = cached_stage(cache_root / ("asr-" + digest(asr_binding) + ".json"), asr_binding, transcribe)

    def diarize():
        pipeline = Pipeline.from_pretrained(request["diarization_model_dir"])
        pipeline.to(torch.device("cuda"))
        result = pipeline(str(audio))
        turns = [{"start": turn.start, "end": turn.end, "speaker": speaker}
                 for turn, speaker in result.speaker_diarization]
        del pipeline
        torch.cuda.empty_cache()
        return turns

    diarization_binding = {key: value for key, value in binding.items() if key != "model_sha256"}
    diarization_binding["versions"] = {"pyannote.audio": versions["pyannote.audio"]}
    turns = cached_stage(cache_root / ("speakers-" + digest(diarization_binding) + ".json"),
                         diarization_binding, diarize)
    if file_digest(audio) != binding["audio_sha256"]:
        raise IntegrityError("sample changed during inference")
    verify_source_clip(audio, request["source_audio"], request["source_sha256"], request["offset_ms"])
    body = {"format": "mas-acoustic-pilot-1", "episode": request["episode"],
            "source_sha256": request["source_sha256"], "audio_sha256": binding["audio_sha256"],
            "offset_ms": request["offset_ms"], "duration_ms": duration,
            "segments": transcript["segments"], "speaker_turns": turns,
            "provenance": binding}
    evidence = {"data": body, "sha256": digest(body)}
    atomic_json(output / "evidence.json", evidence)

    def regroup(words):
        result = stable_whisper.WhisperResult([{"words": words}])
        result.split_by_punctuation([".", "?", "!"])
        result.split_by_gap(0.5)
        result.split_by_length(max_chars=84)
        return [segment["words"] for segment in result.to_dict()["segments"]]

    write_pilot(output, build_pilot(evidence, regroup=regroup))


if __name__ == "__main__":
    run_worker(sys.argv[1])
