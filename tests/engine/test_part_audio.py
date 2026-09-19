import copy
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from mas.engine import part_audio, part_scope
from mas.engine.part_audio import (
    extract_part_audio, load_part_plan, prepare_episode_parts, validate_part_audio,
)
from mas.engine.part_scope import PartScopeError
from mas.reliability import atomic_json, digest


@pytest.fixture
def synthetic_parents(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("local ffmpeg/ffprobe unavailable")
    monkeypatch.setattr(part_scope, "TARGET_DURATION_MS", 2000)
    part_audio._VERIFIED_AUDIO.clear()
    source = tmp_path / "source" / "episode.mp4"
    audio = tmp_path / "work" / "audio.wav"
    source.parent.mkdir()
    audio.parent.mkdir()
    source.write_bytes(b"synthetic immutable video identity")
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 40000 + b"\x02\x00" * 40007)
    captions = source.parent / "captions.vtt"
    captions.write_text("WEBVTT\n\n00:00:00.500 --> 00:00:01.000\nZeytin.\n\n"
                        "00:00:04.000 --> 00:00:04.900\nGökyüzü.\n", encoding="utf-8")
    monkeypatch.setattr(part_audio, "_vad_model_identity", lambda: {"fixture_sha256": "c" * 64})
    _, config = part_audio._audit_config()
    calls = []
    actual_run = part_audio._run

    def runner(command, root, name, deadline, **kwargs):
        calls.append(name)
        if name == "part-vad":
            part_audio._write_envelope(Path(command[-2]), {
                "independent_vad": True, "audio_sha256": part_audio._sha(audio),
                "sample_count": 16000 * 5 + 7, "config": config,
                "model": {"fixture_sha256": "c" * 64},
                "producer_sha256": part_audio._producer_sha(),
                "regions": [
                    {"vad_region_index": 1, "start_ms": 500, "end_ms": 1000, "source": "silero_vad"},
                    {"vad_region_index": 2, "start_ms": 4000, "end_ms": 4900, "source": "silero_vad"},
                ],
            })
        else:
            actual_run(command, root, name, deadline, **kwargs)

    monkeypatch.setattr(part_audio, "_run", runner)
    return tmp_path, source, audio, captions, calls


def test_real_derived_audio_exact_samples_portable_lineage_and_cache(synthetic_parents, tmp_path):
    root, source, audio, captions, calls = synthetic_parents
    original = source.read_bytes(), audio.read_bytes(), captions.read_bytes()
    plan = prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    assert len(plan["parts"]) == 2
    assert plan["parts"][0]["end_ms"] == 3880
    first = extract_part_audio(root, 14, "part-001", total_timeout=60)
    second = extract_part_audio(root, 14, "part-002", total_timeout=60)
    assert first["lineage"]["sample_count"] == 62080
    assert second["lineage"]["sample_count"] == 17927
    assert first["audio_path"].suffix == ".flac"
    assert second["parent_vad_regions"][0]["start_ms"] == 120
    assert "00:00:00.120" in second["captions_path"].read_text(encoding="utf-8")
    assert not (first["audio_path"].parent / "audio.done.json").exists()
    assert not (first["audio_path"].parent / "download.done.json").exists()
    before = list(calls)
    assert prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60) == plan
    assert extract_part_audio(root, 14, "part-001", total_timeout=60)["lineage"] == first["lineage"]
    assert calls[len(before):] == []
    moved = root.parent / (root.name + "-moved")
    shutil.copytree(root, moved)
    assert validate_part_audio(moved, 14, "part-002")["lineage"] == second["lineage"]
    assert original == (source.read_bytes(), audio.read_bytes(), captions.read_bytes())


def test_manifest_only_loading_does_not_require_transferred_media(synthetic_parents):
    root, source, audio, captions, _ = synthetic_parents
    plan = prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    source.unlink()
    assert load_part_plan(root, 14, verify_files=False) == plan
    with pytest.raises(PartScopeError, match="absent"):
        load_part_plan(root, 14)


def test_changed_parent_or_rehashed_omission_cannot_resume(synthetic_parents):
    root, source, audio, captions, calls = synthetic_parents
    prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    saved = part_audio._read_envelope(root / "work/part-plan.json")
    changed = copy.deepcopy(saved)
    changed["parts"][0]["vad_region_indices"] = []
    atomic_json(root / "work/part-plan.json", {"data": changed, "sha256": digest(changed)})
    with pytest.raises(PartScopeError, match="omits"):
        extract_part_audio(root, 14, "part-001", total_timeout=60)
    part_audio._write_envelope(root / "work/part-plan.json", saved)
    source.write_bytes(b"changed source")
    with pytest.raises(PartScopeError, match="identity mismatch"):
        extract_part_audio(root, 14, "part-001", total_timeout=60)
    assert not any(name.endswith("-extract") for name in calls)


def test_failed_audio_producer_records_failure_and_does_not_repeat(synthetic_parents, monkeypatch):
    root, source, audio, captions, _ = synthetic_parents
    prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    prior = part_audio._run
    count = []

    def fail(command, root, name, deadline, **kwargs):
        if name.endswith("-extract"):
            count.append(name)
            raise TimeoutError("fixture extraction deadline")
        return prior(command, root, name, deadline, **kwargs)

    monkeypatch.setattr(part_audio, "_run", fail)
    with pytest.raises(TimeoutError):
        extract_part_audio(root, 14, "part-001", total_timeout=60)
    with pytest.raises(PartScopeError, match="unchanged derived-audio failure"):
        extract_part_audio(root, 14, "part-001", total_timeout=60)
    assert count == ["part-001-extract"]
    assert not (root / "parts/part-001/work/audio-part.done.json").exists()


def test_tampered_child_audio_and_caption_fail_readback(synthetic_parents):
    root, source, audio, captions, _ = synthetic_parents
    prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    result = extract_part_audio(root, 14, "part-001", total_timeout=60)
    result["captions_path"].write_text("WEBVTT\n\n", encoding="utf-8")
    with pytest.raises(PartScopeError, match="identity mismatch"):
        validate_part_audio(root, 14, "part-001")


def test_rehashed_wrong_audio_of_same_length_fails_parent_pcm_readback(synthetic_parents):
    root, source, audio, captions, _ = synthetic_parents
    prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    result = extract_part_audio(root, 14, "part-001", total_timeout=60)
    subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3.88",
                    "-c:a", "flac", str(result["audio_path"])],
                   check=True, capture_output=True, timeout=30)
    changed = copy.deepcopy(result["lineage"])
    changed["audio"] = part_audio._record(root, result["audio_path"])
    part_audio._write_envelope(result["lineage_path"], changed)
    with pytest.raises(PartScopeError, match="decoded PCM differs"):
        validate_part_audio(root, 14, "part-001")


def test_standalone_parent_vad_fallback_cannot_create_receipt(tmp_path, monkeypatch):
    from mas.engine import transcribe
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"fixture bytes, decoder is not called")
    monkeypatch.setattr(part_audio, "_vad_model_identity", lambda: {"fixture": True})
    monkeypatch.setattr(transcribe, "_extract_vad_regions",
                        lambda *args: ([{"source": "fallback"}], "fixture fallback"))
    with pytest.raises(PartScopeError, match="fallback cannot define"):
        part_audio._scan_vad(audio, tmp_path / "vad.json", 16000)
    assert not (tmp_path / "vad.json").exists()


def test_derived_caption_serialization_preserves_literal_entities(tmp_path):
    from mas.engine.transcribe import load_vtt_captions
    records = [{"caption_index": 1, "start_ms": 1000, "end_ms": 2000,
                "text": "Allah & <bir ifade>"}]
    path = tmp_path / "captions.vtt"
    path.write_text(part_audio._caption_text(records), encoding="utf-8")
    assert load_vtt_captions(path) == records


def test_heavy_subprocess_is_deadline_bound(tmp_path):
    import sys
    from mas.reliability import OperationFailed
    with pytest.raises(OperationFailed, match="timed out"):
        part_audio._run([sys.executable, "-c", "import time; time.sleep(30)"],
                        tmp_path, "fixture-deadline", part_audio._deadline(.05))


def test_pcm_proof_reuse_is_bounded_and_binding_sensitive(synthetic_parents):
    root, source, audio, captions, calls = synthetic_parents
    plan = prepare_episode_parts(root, 14, source, audio, captions, total_timeout=60)
    result = extract_part_audio(root, 14, "part-001", total_timeout=60)
    before = len(calls)
    validate_part_audio(root, 14, "part-001")
    assert calls[before:] == []
    changed = copy.deepcopy(result["lineage"])
    changed["producer_sha256"] = "d" * 64
    part_audio._write_envelope(result["lineage_path"], changed)
    validate_part_audio(root, 14, "part-001")
    assert calls[before:] == ["part-001-probe", "part-001-parent-pcm", "part-001-child-pcm"]
    for index in range(100):
        part_audio._remember_verified(str(index))
    assert len(part_audio._VERIFIED_AUDIO) == 64
    assert "0" not in part_audio._VERIFIED_AUDIO


@pytest.mark.parametrize("name,expected", [("part-vad", 600), ("part-001-extract", 600),
                                          ("part-001-parent-pcm", 600), ("part-001-probe", 60)])
def test_heavy_stage_cannot_take_whole_episode_deadline(name, expected, tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(part_audio, "run_command", lambda *args, **kwargs: observed.append(kwargs))
    part_audio._run(["fixture"], tmp_path, name, part_audio._deadline(21600))
    assert observed[0]["timeout_seconds"] == expected


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_producer_deadlines_cannot_be_unbounded(value, tmp_path):
    with pytest.raises(PartScopeError, match="finite and positive"):
        prepare_episode_parts(tmp_path, 14, "none", "none", total_timeout=value)
