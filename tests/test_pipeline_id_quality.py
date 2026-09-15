from types import SimpleNamespace

import pytest

from mas import pipeline
from mas.engine.id_translation import (
    build_id_translation_records,
    validate_aligned_turkish_schema,
    validate_id_translation_records,
)
from mas.engine.subtitle_qa import SubtitleQAError
from mas.engine.timing_qa import TimingQAV2Error


def quality_case(source, target, *, duration=5000, review=False, acoustic_start=1000):
    digest = "a" * 64
    schema = validate_aligned_turkish_schema({
        "schema_version": "2.0",
        "episode": 13,
        "blocks": [{
            "block_uid": "scene-1",
            "block_index": 1,
            "start_ms": 1000,
            "end_ms": 1000 + duration,
            "tr_text": source,
            "alignment_provenance": {
                "timing_source": "whisperx_ctc_forced_alignment",
                "alignment_sha256": digest,
                "first_word_start_ms": 1000,
                "last_word_end_ms": 1000 + duration - 220,
            },
        }],
    })
    alignment = {"alignment_sha256": digest}
    for field in (
        "unaligned_word_count", "synthetic_timing_count", "negative_word_gap_count",
        "missing_alignment_score_count", "low_alignment_score_count",
        "overlong_alignment_word_count", "outward_drift_violation_count",
        "low_score_edited_token_count", "unreviewed_deleted_token_count",
        "review_alignment_score_count",
    ):
        alignment[field] = 0
    alignment = {
        "alignment_sha256": digest,
        "report": alignment,
        "words": [{"text": source, "start_ms": acoustic_start,
                   "end_ms": 1000 + duration - 220, "utterance_uid": "source-1"}],
    }
    artifacts = SimpleNamespace(
        schema=schema,
        speech_coverage_report={"unresolved_speech_region_count": 0},
        alignment_report=alignment,
    )
    records = build_id_translation_records(schema)
    records[0].update(id_final=target, review_required=review, note="")
    validation = validate_id_translation_records(schema, records)
    _, series, names, religious = pipeline._load_configs()
    return pipeline._validate_id_quality(artifacts, validation, series, names, religious)


@pytest.mark.parametrize("source,target", [
    ("Seni seviyorum.", "Aku sayang kamu."),
    ("Kadir, Allah aşkına, 25 lira.", "Kadir, demi Allah, 25 lira."),
    ("Allah'ım!", "Ya Allah!"),
    ("İnşallah.", "Insyaallah."),
    ("Maşallah.", "Masyaallah."),
    ("Vallahi, eyvallah.", "Serius, makasih."),
])
def test_id_return_quality_accepts_canonical_dialogue(source, target):
    assert quality_case(source, target)["passed"]


@pytest.mark.parametrize("source,target", [
    ("Allah aşkına.", "Semoga."),
    ("Allah'ım!", "Demi Allah!"),
    ("Kadir, gel.", "Ayo, kemari."),
    ("25 lira.", "20 lira."),
])
def test_id_return_rejects_quality_failure_before_finalization(source, target):
    with pytest.raises(SubtitleQAError):
        quality_case(source, target)


def test_id_return_never_extends_time_to_fit_translation():
    with pytest.raises(TimingQAV2Error, match="high_cps_id"):
        quality_case("Gel.", "Tolong segera datang ke sini sekarang.", duration=1000)


def test_id_return_rejects_early_display_despite_matching_frozen_metadata():
    with pytest.raises(TimingQAV2Error, match="invalid_timing_count"):
        quality_case("Gökyüzü kapalı.", "Langit mendung.", acoustic_start=5000)


def test_id_return_blocks_unresolved_review():
    with pytest.raises(RuntimeError, match="still requires review"):
        quality_case("Gel.", "Kemari.", review=True)


def test_id_return_rejects_overlong_cue():
    with pytest.raises(TimingQAV2Error, match="invalid_timing_count"):
        quality_case("Gel.", "Kemari.", duration=7001)


def test_production_readability_has_one_42_character_line_limit():
    _, series, _, _ = pipeline._load_configs()
    assert series["subtitle"]["target_chars_per_line"] == 42
    assert series["subtitle"]["qa_max_chars_per_line"] == 42
    with pytest.raises(SubtitleQAError):
        quality_case("Gel.", "x" * 43)
