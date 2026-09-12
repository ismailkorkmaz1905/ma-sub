import json
from pathlib import Path

import pytest

from mas.emergency_segment import (
    EmergencySegmentError,
    _translation_quality_issues,
    finalize_emergency_segment,
    prepare_emergency_segment,
)
from mas.engine.tr_correction import create_tr_correction_output, create_tr_correction_pack


def _record(uid, index, start, end, text):
    return {
        "utterance_uid": uid,
        "utterance_index": index,
        "coarse_start_ms": start,
        "coarse_end_ms": end,
        "asr_text": text,
        "youtube_text": "",
        "context_before": "",
        "context_after": "",
        "risk_flags": [],
        "asr_audit": {
            "avg_logprob": -0.1,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.1,
            "temperature": 0.0,
        },
    }


def _episode(tmp_path):
    root = tmp_path / "episode"
    input_dir = root / "translation_input"
    input_dir.mkdir(parents=True)
    pack = input_dir / "Muhtemel Ask 1.Bolum_TR_CORRECTION_PACK.zip"
    utterances = [
        _record("u1", 1, 1000, 1800, "Merhaba."),
        _record("u2", 2, 1600, 2200, "Nasılsın?"),
        _record("u3", 3, 5000, 5700, "Güle güle."),
        _record("u4", 4, 6000, 6200, "Altyazı M.K."),
    ]
    create_tr_correction_pack(utterances, [], pack, episode=1)
    corrected = root / "corrected.zip"
    records = []
    for source in utterances:
        records.append(
            {
                **source,
                "tr_corrected": source["asr_text"],
                "non_dialogue": False,
                "review_required": False,
                "audio_reviewed": False,
                "review_disposition": "not_applicable",
                "note": "",
            }
        )
    create_tr_correction_output(pack, records, corrected)
    return root, corrected


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_prepare_and_finalize_emergency_segment_without_strict_outputs(tmp_path):
    root, corrected = _episode(tmp_path)
    manifest_path = prepare_emergency_segment(1, corrected, root=root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = root / "emergency" / "segment-timing"
    work = _load_jsonl(output / "translation_work.jsonl")

    assert [record["utterance_uid"] for record in work] == ["u1", "u2", "u3"]
    assert work[1]["start_ms"] >= work[0]["end_ms"]
    assert work[1]["original_coarse_start_ms"] == 1600
    assert manifest["mode"] == "emergency"
    assert manifest["strict_eligible"] is False
    assert manifest["adjustment_count"] >= 1
    assert manifest["quarantine_count"] == 1
    assert _load_jsonl(output / "metadata_quarantine.jsonl")[0]["utterance_uid"] == "u4"

    for record in work:
        record["id_translation"] = "Ya."
    returned = output / "id_return.jsonl"
    returned.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in work),
        encoding="utf-8",
    )
    receipt_path = finalize_emergency_segment(1, returned, root=root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "EMERGENCY_SUBTITLES_READY"
    assert receipt["strict_eligible"] is False
    assert (output / "Muhtemel Ask 1.Bolum-tr.srt").is_file()
    assert (output / "Muhtemel Ask 1.Bolum-id.srt").is_file()
    assert not (root / "final").exists()


def test_finalize_rejects_changed_binding_and_extends_unreadable_translation(tmp_path):
    root, corrected = _episode(tmp_path)
    prepare_emergency_segment(1, corrected, root=root)
    output = root / "emergency" / "segment-timing"
    work = _load_jsonl(output / "translation_work.jsonl")
    for record in work:
        record["id_translation"] = "Ya."
    work[0]["tr_corrected"] = "forged"
    returned = output / "forged.jsonl"
    returned.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in work),
        encoding="utf-8",
    )
    with pytest.raises(EmergencySegmentError, match="immutable field"):
        finalize_emergency_segment(1, returned, root=root)

    work = _load_jsonl(output / "translation_work.jsonl")
    for record in work:
        record["id_translation"] = "Ya."
    work[0]["id_translation"] = "x" * 40
    returned.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in work),
        encoding="utf-8",
    )
    receipt_path = finalize_emergency_segment(1, returned, root=root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["maximum_observed_cps"] <= 20.0
    assert receipt["timing_adjustment_count"] >= 1
    assert (output / "indonesian_timing_adjustments.jsonl").is_file()


def test_finalize_accepts_relative_translation_path_inside_episode(tmp_path, monkeypatch):
    root, corrected = _episode(tmp_path)
    prepare_emergency_segment(1, corrected, root=root)
    output = root / "emergency" / "segment-timing"
    work = _load_jsonl(output / "translation_work.jsonl")
    for record in work:
        record["id_translation"] = "Ya."
    returned = output / "id_return.jsonl"
    returned.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in work),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    relative = returned.relative_to(tmp_path)
    receipt_path = finalize_emergency_segment(1, relative, root=root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["translation_return"]["path"] == (
        "emergency/segment-timing/id_return.jsonl"
    )


def test_translation_quality_checks_numbers_names_and_religious_terms():
    issues = _translation_quality_issues(
        "Allah'ım, Kadir 10:30'da gelecek.",
        "Semoga dia datang pukul 11:30.",
    )
    assert "numbers changed: ['10:30'] -> ['11:30']" in issues
    assert "canonical name missing: Kadir" in issues
    assert "religious expression must contain: ya allah" in issues

    assert not _translation_quality_issues(
        "Allah'ım, Kadir 10:30'da gelecek.",
        "Ya Allah, Kadir akan datang pukul 10:30.",
    )
    assert not _translation_quality_issues("Vallahi geleceğim.", "Sumpah, aku akan datang.")
    assert "Allah was not preserved" in _translation_quality_issues("Allah", "Ya...")
