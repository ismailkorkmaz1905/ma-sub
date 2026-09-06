import json
import wave

import pytest

from mas.reliability import IntegrityError, digest, file_digest
from mas.subtitle.asr_pilot_worker import EXPECTED_VERSIONS, execute
from mas.engine.primary_checkpoint import model_identity


def setup_request(tmp_path):
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    source_sha = "1" * 64
    manifest_body = {"format": "mas-contextual-pilot-samples-1", "episode": 11,
                     "source_sha256": source_sha, "samples": [{
                         "name": "clip", "audio_sha256": file_digest(audio),
                         "source_sha256": source_sha, "offset_ms": 1000,
                         "duration_ms": 1000, "exact_source_pcm_verified": True}]}
    manifest = {"data": manifest_body, "sha256": digest(manifest_body)}
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(json.dumps(manifest))
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    return {"manifest": str(manifest_path), "manifest_sha256": manifest["sha256"],
            "model_dir": str(model), "model_sha256": model_identity(model)["sha256"],
            "versions": EXPECTED_VERSIONS, "maximum_seconds": 180,
            "clips": {"clip": str(audio)}, "output": str(tmp_path / "output")}


def test_execute_checkpoints_raw_before_result_and_resumes(tmp_path):
    request = setup_request(tmp_path)
    calls = []
    execute(request, lambda path: calls.append(path) or {"segments": [{"text": " Merhaba"}]},
            producer={"runtime": "test"})
    output = tmp_path / "output"
    raw = json.loads((output / "clip.raw.json").read_text())
    result = json.loads((output / "result.json").read_text())
    assert raw["sha256"] == digest(raw["data"])
    assert raw["data"]["raw"]["segments"][0]["text"] == " Merhaba"
    assert raw["data"]["binding"]["source_sha256"] == "1" * 64
    assert result["data"]["status"] == "REVIEW_REQUIRED"
    execute(request, lambda path: pytest.fail("completed raw clip must resume"),
            producer={"runtime": "test"})
    assert len(calls) == 1


def test_execute_rejects_remote_pcm_mismatch(tmp_path):
    request = setup_request(tmp_path)
    with open(request["clips"]["clip"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(IntegrityError, match="remote PCM"):
        execute(request, lambda path: {}, producer={"runtime": "test"})


def test_execute_res_deadline_between_clips(tmp_path):
    request = setup_request(tmp_path)
    request["maximum_seconds"] = 1
    ticks = iter([0, 2])
    with pytest.raises(TimeoutError, match="between clips"):
        execute(request, lambda path: {}, clock=lambda: next(ticks), producer={"runtime": "test"})


def test_manifest_attestation_is_not_remote_verification(tmp_path):
    request = setup_request(tmp_path)
    wrapped = json.loads(open(request["manifest"], encoding="utf-8").read())
    wrapped["data"]["samples"][0]["exact_source_pcm_verified"] = False
    wrapped["sha256"] = digest(wrapped["data"])
    open(request["manifest"], "w", encoding="utf-8").write(json.dumps(wrapped))
    request["manifest_sha256"] = wrapped["sha256"]
    with pytest.raises(IntegrityError, match="attestation"):
        execute(request, lambda path: {}, producer={"runtime": "test"})


@pytest.mark.parametrize("name", ["../escape", "Bad Name", "", "a" * 65])
def test_manifest_rejects_unsafe_clip_names(tmp_path, name):
    request = setup_request(tmp_path)
    wrapped = json.loads(open(request["manifest"], encoding="utf-8").read())
    wrapped["data"]["samples"][0]["name"] = name
    wrapped["sha256"] = digest(wrapped["data"])
    open(request["manifest"], "w", encoding="utf-8").write(json.dumps(wrapped))
    request["manifest_sha256"] = wrapped["sha256"]
    request["clips"] = {name: request["clips"]["clip"]}
    with pytest.raises(IntegrityError, match="attestation"):
        execute(request, lambda path: {}, producer={"runtime": "test"})


@pytest.mark.parametrize("target", ["audio", "model"])
def test_mutation_during_transcription_prevents_checkpoint(tmp_path, target):
    request = setup_request(tmp_path)
    def transcribe(path):
        mutation = path if target == "audio" else Path(request["model_dir"]) / "model.bin"
        with mutation.open("ab") as stream:
            stream.write(b"changed")
        return {"segments": []}
    from pathlib import Path
    with pytest.raises(IntegrityError, match="changed during"):
        execute(request, transcribe, producer={"runtime": "test"})
    assert not (tmp_path / "output" / "clip.raw.json").exists()
