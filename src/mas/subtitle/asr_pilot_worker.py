import importlib.metadata
import json
import math
import os
import re
import sys
import time
import wave
from pathlib import Path

from ..reliability import IntegrityError, atomic_json, digest, file_digest
from ..engine.primary_checkpoint import model_identity, producer_identity


EXPECTED_VERSIONS = {
    "stable-ts": "2.19.1",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "torch": "2.8.0",
}
TRANSCRIBE_ARGS = {
    "language": "tr",
    "word_timestamps": True,
    "regroup": False,
    "suppress_silence": False,
    "vad_filter": False,
    "condition_on_previous_text": False,
    "beam_size": 5,
    "temperature": 0.0,
    "verbose": False,
}


def _manifest(path):
    wrapped = json.loads(Path(path).read_text(encoding="utf-8"))
    body = wrapped.get("data")
    if (not isinstance(body, dict) or wrapped.get("sha256") != digest(body)
            or body.get("format") != "mas-contextual-pilot-samples-1"
            or body.get("episode") != 11 or not isinstance(body.get("samples"), list)
            or not body["samples"] or len(body["samples"]) > 8):
        raise IntegrityError("ASR pilot sample manifest is invalid")
    names = []
    for sample in body["samples"]:
        names.append(sample.get("name"))
        if (sample.get("source_sha256") != body.get("source_sha256")
                or sample.get("exact_source_pcm_verified") is not True
                or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", str(sample.get("name", "")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(sample.get("audio_sha256", "")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(sample.get("source_sha256", "")))
                or isinstance(sample.get("offset_ms"), bool)
                or not isinstance(sample.get("offset_ms"), int) or sample["offset_ms"] < 0
                or isinstance(sample.get("duration_ms"), bool)
                or not isinstance(sample.get("duration_ms"), int)
                or not 0 < sample["duration_ms"] <= 120000):
            raise IntegrityError("ASR pilot sample attestation is invalid")
    if len(names) != len(set(names)):
        raise IntegrityError("ASR pilot sample names must be unique")
    return wrapped


def _checkpoint_binding(request, sample, audio_sha256, model_sha256, versions, producer):
    return {"episode": 11, "manifest_sha256": request["manifest_sha256"],
            "source_sha256": sample["source_sha256"], "clip_name": sample["name"],
            "clip_sha256": audio_sha256, "offset_ms": sample["offset_ms"],
            "duration_ms": sample["duration_ms"], "model_sha256": model_sha256,
            "versions": versions, "producer": producer,
            "transcribe_args": TRANSCRIBE_ARGS, "device": "cuda", "compute_type": "float16"}


def _wav_attestation(path, duration_ms):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise IntegrityError("ASR pilot clip must be mono 16000 Hz PCM16 WAV")
        frames = stream.getnframes()
        payload = stream.readframes(frames)
        if len(payload) != frames * 2 or round(frames * 1000 / 16000) != duration_ms:
            raise IntegrityError("ASR pilot remote WAV duration or bytes differ from attestation")


def _producer():
    identity = producer_identity([execute, _checkpoint_binding])
    distribution = importlib.metadata.distribution("stable-ts")
    files = {str(item): file_digest(Path(distribution.locate_file(item)))
             for item in distribution.files or ()
             if not str(item).endswith(".pyc") and Path(distribution.locate_file(item)).is_file()}
    if not files:
        raise IntegrityError("cannot bind stable-ts pilot runtime")
    identity["stable-ts"] = {"version": distribution.version,
                             "files_sha256": digest(files)}
    return identity


def execute(request, transcribe, *, clock=time.monotonic, producer=None):
    manifest = _manifest(request["manifest"])
    if manifest["sha256"] != request["manifest_sha256"]:
        raise IntegrityError("ASR pilot manifest binding mismatch")
    model = model_identity(request["model_dir"])
    if model["sha256"] != request["model_sha256"]:
        raise IntegrityError("ASR pilot model changed")
    versions = request["versions"]
    if versions != EXPECTED_VERSIONS:
        raise IntegrityError("ASR pilot dependency versions differ")
    maximum_seconds = request.get("maximum_seconds")
    if (isinstance(maximum_seconds, bool) or not isinstance(maximum_seconds, (int, float))
            or not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= 180):
        raise IntegrityError("ASR pilot worker deadline is invalid")
    started = clock()
    producer = producer or _producer()
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    paths = request.get("clips")
    if not isinstance(paths, dict) or set(paths) != {item["name"] for item in manifest["data"]["samples"]}:
        raise IntegrityError("ASR pilot remote clip set differs from manifest")
    for sample in manifest["data"]["samples"]:
        if clock() - started >= maximum_seconds:
            raise TimeoutError("ASR pilot worker deadline expired between clips")
        audio = Path(paths[sample["name"]])
        audio_sha256 = file_digest(audio)
        if audio_sha256 != sample["audio_sha256"]:
            raise IntegrityError("ASR pilot remote PCM differs from locally verified clip")
        _wav_attestation(audio, sample["duration_ms"])
        binding = _checkpoint_binding(request, sample, audio_sha256, model["sha256"], versions, producer)
        checkpoint = output / (sample["name"] + ".raw.json")
        if checkpoint.exists():
            wrapped = json.loads(checkpoint.read_text(encoding="utf-8"))
            body = wrapped.get("data")
            if (not isinstance(body, dict) or wrapped.get("sha256") != digest(body)
                    or body.get("binding") != binding):
                raise IntegrityError("ASR pilot raw checkpoint binding mismatch")
        else:
            raw = transcribe(audio)
            if file_digest(audio) != audio_sha256 or model_identity(request["model_dir"]) != model:
                raise IntegrityError("ASR pilot audio or model changed during transcription")
            body = {"format": "mas-asr-pilot-raw-1", "binding": binding, "raw": raw,
                    "status": "REVIEW_REQUIRED"}
            atomic_json(checkpoint, {"data": body, "sha256": digest(body)})
        completed.append({"name": sample["name"], "checkpoint": checkpoint.name,
                          "sha256": file_digest(checkpoint)})
        atomic_json(output / "progress.json",
                    {"completed_units": len(completed),
                     "artifact_bytes": sum((output / item["checkpoint"]).stat().st_size
                                           for item in completed)})
    result = {"format": "mas-asr-pilot-result-1", "status": "REVIEW_REQUIRED",
              "production_acceptance": False, "manifest_sha256": manifest["sha256"],
              "model_sha256": model["sha256"], "versions": versions,
              "producer": producer, "clips": completed}
    atomic_json(output / "result.json", {"data": result, "sha256": digest(result)})


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import stable_whisper
    if not torch.cuda.is_available():
        raise RuntimeError("ASR pilot requires CUDA; no CPU fallback")
    versions = {name: importlib.metadata.version(name).split("+", 1)[0]
                for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS or request.get("versions") != versions:
        raise RuntimeError("ASR pilot dependency versions do not match lock")
    manifest = _manifest(request["manifest"])
    if manifest["sha256"] != request.get("manifest_sha256"):
        raise IntegrityError("ASR pilot manifest binding mismatch")
    identity = model_identity(request["model_dir"])
    if identity["sha256"] != request.get("model_sha256"):
        raise IntegrityError("ASR pilot model changed before load")
    producer = _producer()
    model = stable_whisper.load_faster_whisper(
        request["model_dir"], device="cuda", compute_type="float16", local_files_only=True)
    try:
        execute(request, lambda audio: model.transcribe(str(audio), **TRANSCRIBE_ARGS).to_dict(),
                producer=producer)
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_worker(sys.argv[1])
