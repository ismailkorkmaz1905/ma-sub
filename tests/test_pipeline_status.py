import json

import pytest

from mas import pipeline, partial_delivery
from mas.engine import part_audio
from mas.engine.episode_archive import file_record
from mas.engine.part_scope import build_part_plan
from mas.hashing import sha256_file
from mas.reliability import atomic_json, digest


@pytest.fixture
def stored_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "episode_dir", lambda episode: tmp_path)
    source = {"relative_path": "source/movie.mp4", "size_bytes": 10, "sha256": "1" * 64}
    audio = {"relative_path": "prepare/audio.flac", "size_bytes": 10, "sha256": "2" * 64,
             "sample_rate_hz": 16000, "channels": 1, "sample_count": 7200000 * 16}
    regions = [{"vad_region_index": index, "start_ms": start, "end_ms": end, "source": "silero_vad"}
               for index, (start, end) in enumerate([(1000, 2500), (3600000, 3601000),
                                                    (3603000, 3605000), (7197000, 7199000)], 1)]
    plan = build_part_plan(episode=14, source=source, audio=audio,
        vad={"independent_vad": True, "audio_sha256": audio["sha256"], "sample_count": audio["sample_count"],
             "config": {"policy": "canonical"}, "model": {"name": "silero"},
             "producer_sha256": "3" * 64, "regions": regions})
    atomic_json(tmp_path / "work/part-plan.json", {"data": plan, "sha256": digest(plan)})
    monkeypatch.setattr(part_audio, "validate_part_audio", lambda *args, **kwargs: pytest.fail("no audio proof"))
    monkeypatch.setattr(partial_delivery, "validate_published_part", lambda *args, **kwargs: pytest.fail("no live proof"))
    return tmp_path, plan


def status_json(capsys):
    assert pipeline.status(14) == 0
    return json.loads(capsys.readouterr().out)


def test_legacy_status_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "episode_dir", lambda episode: tmp_path)
    state = {"episode": 14, "stages": {"audio": {"status": "pass"}}}
    atomic_json(tmp_path / "work/state.json", state)
    assert status_json(capsys) == state


def test_part_status_shows_exact_plan_current_child_and_stored_handoff(stored_parts, capsys):
    root, plan = stored_parts
    plan_sha = sha256_file(root / "work/part-plan.json")
    child = root / "parts/part-001"
    state = {"episode": 14, "part_id": "part-001", "part_plan_sha256": plan_sha,
             "stages": {"raw_asr": {"status": "pass", "updated_at": "2026-09-14T01:00:00"},
                        "tr_return": {"status": "blocked", "updated_at": "2026-09-14T01:01:00"}}}
    atomic_json(child / "work/state.json", state)
    pointer = {"format": "mas-current-part-1", "episode": 14, "part_id": "part-001", "stage": "tr_return",
               "part_plan_sha256": plan_sha, "state": file_record(child / "work/state.json", root)}
    atomic_json(root / "work/current-part.json", {"data": pointer, "sha256": digest(pointer)})
    handoff = {"format": "mas-partial-handoff-1", "episode": 14, "part_id": "part-001", "kind": "tr",
               "plan_sha256": plan_sha,
               "expected_return": "parts/part-001/translation_output/Muhtemel Ask 14.Bolum_TR_TEXT_CORRECTED.zip"}
    for filename in ("partial-handoff.json", "partial-handoff-tr.json"):
        atomic_json(child / "work" / filename, {"data": handoff, "sha256": digest(handoff)})
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    progress = status_json(capsys)["progressive"]
    assert progress["current_part"] == "part-001"
    assert progress["recorded_only"] is True and progress["live_verification"] is False
    first, second = progress["parts"]
    assert first["source_end_ms"] == plan["parts"][0]["end_ms"] == second["source_start_ms"]
    assert first["stage"]["name"] == "tr_return" and first["stage"]["status"] == "blocked"
    assert first["handoff"]["kind"] == "tr" and first["handoff"]["present"] is True
    assert first["handoff"]["metadata_status"] == "STORED_METADATA_BOUND"
    assert second["stage"] is None and second["delivery"]["present"] is False
    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before


def write_stored_receipt(root, plan):
    child = root / "parts/part-001"
    receipt = {"format": "mas-part-delivery-1", "status": "PASS_PARTIAL", "episode": 14, "part_id": "part-001",
               "plan_sha256": sha256_file(root / "work/part-plan.json")}
    path = child / "final/drive_readback_receipt.json"
    atomic_json(path, {"data": receipt, "sha256": digest(receipt)})
    return path, receipt


def test_delivery_status_is_explicitly_stored_and_needs_no_media(stored_parts, capsys):
    root, plan = stored_parts
    write_stored_receipt(root, plan)
    result = status_json(capsys)
    delivery = result["progressive"]["parts"][0]["delivery"]
    assert delivery["present"] is True and delivery["stored_status"] == "PASS_PARTIAL"
    assert delivery["metadata_status"] == "STORED_METADATA_BOUND"
    assert result["progressive"]["recorded_only"] is True
    assert result["progressive"]["live_verification"] is False
    assert not list(root.rglob("*.mp4")) and not list(root.rglob("*.flac"))


@pytest.mark.parametrize("change", ["checksum", "part", "plan", "episode"])
def test_filename_or_rebound_wrong_receipt_cannot_claim_stored_pass(stored_parts, capsys, change):
    root, plan = stored_parts
    path, receipt = write_stored_receipt(root, plan)
    if change == "checksum":
        atomic_json(path, {"data": receipt, "sha256": "0" * 64})
    else:
        if change == "part":
            receipt["part_id"] = "part-002"
        elif change == "plan":
            receipt["plan_sha256"] = "0" * 64
        else:
            receipt["episode"] = 13
        atomic_json(path, {"data": receipt, "sha256": digest(receipt)})
    delivery = status_json(capsys)["progressive"]["parts"][0]["delivery"]
    assert delivery == {"present": True, "stored_status": None, "metadata_status": "INVALID_LOCAL_METADATA"}


def test_wrong_child_state_and_current_pointer_are_not_trusted(stored_parts, capsys):
    root, _ = stored_parts
    atomic_json(root / "parts/part-001/work/state.json", {"episode": 14, "part_id": "part-002", "stages": {}})
    atomic_json(root / "work/current-part.json", {"data": {"part_id": "part-001"}, "sha256": "0" * 64})
    progress = status_json(capsys)["progressive"]
    assert progress["current_part"] is None and progress["current_part_metadata"] == "INVALID_LOCAL_METADATA"
    assert progress["parts"][0]["stage"] == {"metadata_status": "INVALID_LOCAL_METADATA"}
