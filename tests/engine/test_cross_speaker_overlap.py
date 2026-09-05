from __future__ import annotations

import copy

import pytest

from mas.engine.forced_align import validate_coarse_segments
from mas.engine.id_translation import IDTranslationError, validate_aligned_turkish_schema
from mas.engine.aligned_schema import SchemaV2Error, build_aligned_turkish_schema
from mas.engine.segmentation import SegmentationError, build_blocks
from mas.engine.timing_qa import run_timing_qa_v2


ALIGNMENT_SHA = "a" * 64
AUDIO_SHA = "b" * 64


def alignment_report() -> dict:
    return {
        "alignment_sha256": ALIGNMENT_SHA,
        "unaligned_word_count": 0,
        "synthetic_timing_count": 0,
        "negative_word_gap_count": 0,
        "missing_alignment_score_count": 0,
        "low_alignment_score_count": 0,
        "overlong_alignment_word_count": 0,
        "outward_drift_violation_count": 0,
        "low_score_edited_token_count": 0,
        "unreviewed_deleted_token_count": 0,
        "review_alignment_score_count": 0,
    }


def word(index: int, start: int, end: int, text: str, speaker: str | None) -> dict:
    value = {
        "word_index": index,
        "segment_id": index,
        "utterance_uid": f"utt-{index}",
        "start_ms": start,
        "end_ms": end,
        "text": text,
        "probability": 0.99,
        "timing_source": "whisperx_ctc_forced_alignment",
        "alignment_sha256": ALIGNMENT_SHA,
    }
    if speaker is not None:
        value["speaker_id"] = speaker
    return value


def overlapping_blocks() -> list[dict]:
    return build_blocks(
        {
            "words": [
                word(1, 0, 900, "Merhaba.", "speaker-a"),
                word(2, 200, 1100, "Selam.", "speaker-b"),
            ]
        },
        episode=12,
    )


def test_different_explicit_speakers_remain_separate_through_schema_and_qa() -> None:
    blocks = overlapping_blocks()
    assert [block["speaker_id"] for block in blocks] == ["speaker-a", "speaker-b"]
    assert [block["primary_text"] for block in blocks] == ["Merhaba.", "Selam."]
    assert blocks[1]["start_ms"] < blocks[0]["end_ms"]

    schema = build_aligned_turkish_schema(
        blocks,
        episode=12,
        alignment_report=alignment_report(),
        speech_coverage_report={"unresolved_speech_region_count": 0},
    )
    assert validate_aligned_turkish_schema(schema) == schema
    qa = run_timing_qa_v2(
        schema["blocks"],
        speech_coverage_report={"unresolved_speech_region_count": 0},
        alignment_report=alignment_report(),
    )
    assert qa["overlap_count"] == 0
    assert qa["passed"]

    repeated = copy.deepcopy(schema["blocks"])
    repeated[1]["tr_text"] = repeated[0]["tr_text"]
    repeated_qa = run_timing_qa_v2(
        repeated,
        speech_coverage_report={"unresolved_speech_region_count": 0},
        alignment_report=alignment_report(),
    )
    assert repeated_qa["adjacent_short_duplicate_count"] == 0
    assert repeated_qa["passed"]


@pytest.mark.parametrize("speakers", [("speaker-a", "speaker-a"), (None, None)])
def test_same_or_unknown_speaker_word_overlap_is_rejected(speakers) -> None:
    with pytest.raises(SegmentationError, match="overlapping word intervals"):
        build_blocks(
            {
                "words": [
                    word(1, 0, 900, "Merhaba.", speakers[0]),
                    word(2, 200, 1100, "Selam.", speakers[1]),
                ]
            },
            episode=12,
        )


def test_same_speaker_block_overlap_is_rejected_at_schema_and_id_boundary() -> None:
    blocks = overlapping_blocks()
    blocks[1]["speaker_id"] = "speaker-a"
    with pytest.raises(SchemaV2Error, match="unsafe timing"):
        build_aligned_turkish_schema(
            blocks,
            episode=12,
            alignment_report=alignment_report(),
            speech_coverage_report={"unresolved_speech_region_count": 0},
        )

    schema = build_aligned_turkish_schema(
        overlapping_blocks(),
        episode=12,
        alignment_report=alignment_report(),
        speech_coverage_report={"unresolved_speech_region_count": 0},
    )
    unsafe = copy.deepcopy(schema)
    unsafe.pop("schema_sha256")
    unsafe["blocks"][1]["speaker_id"] = "speaker-a"
    with pytest.raises(IDTranslationError, match="same/unknown-speaker overlap"):
        validate_aligned_turkish_schema(unsafe)


def test_coarse_alignment_preserves_explicit_speaker_identity() -> None:
    source = validate_coarse_segments(
        [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "text": "Merhaba.",
                "asr_text": "Merhaba.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-1",
                "speaker_id": "speaker-a",
            }
        ]
    )
    assert source[0]["speaker_id"] == "speaker-a"


def test_nonoverlapping_known_and_unknown_speakers_keep_separate_text() -> None:
    blocks = build_blocks(
        {
            "words": [
                word(1, 0, 900, "Merhaba.", "speaker-a"),
                word(2, 1500, 2400, "Selam.", None),
            ]
        },
        episode=12,
    )
    assert [block["primary_text"] for block in blocks] == ["Merhaba.", "Selam."]
