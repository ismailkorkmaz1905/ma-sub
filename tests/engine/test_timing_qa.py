from __future__ import annotations

import unittest
import copy

import pytest

from mas.engine.timing_qa import TimingQAV2Error, assert_timing_qa_v2, run_timing_qa_v2


ALIGNMENT_SHA = "a" * 64


def block(index: int, start: int, end: int, text: str) -> dict:
    return {
        "block_uid": f"uid-{index}",
        "block_index": index,
        "start_ms": start,
        "end_ms": end,
        "primary_text": text,
        "vad_info": {"alignment_provenance": {
            "word_timing_sources": ["whisperx_ctc_forced_alignment"],
            "alignment_sha256": ALIGNMENT_SHA,
        }},
    }


def coverage(unresolved: int = 0) -> dict:
    return {"unresolved_speech_region_count": unresolved}


def alignment(**overrides: int | str) -> dict:
    value = {
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
    value.update(overrides)
    return value


def full_alignment_output(**counter_overrides: int) -> dict:
    report = alignment(**counter_overrides)
    return {
        "format_version": "1.0",
        "alignment_sha256": ALIGNMENT_SHA,
        "timing_source": "whisperx_ctc_forced_alignment",
        "segments": [],
        "words": [{"text": "Merhaba.", "start_ms": 1000, "end_ms": 1780}],
        "report": report,
    }


class TimingQAV2Tests(unittest.TestCase):
    def test_clean_alignment_backed_timeline_passes(self) -> None:
        blocks = [block(1, 1000, 2000, "Merhaba."), block(2, 2200, 3300, "Nasılsın?")]
        report = run_timing_qa_v2(
            blocks,
            speech_coverage_report=coverage(),
            alignment_report=alignment(),
            id_text_by_uid={"uid-1": "Halo.", "uid-2": "Apa kabar?"},
        )
        self.assertTrue(report["passed"])
        assert_timing_qa_v2(report)

    def test_full_forced_alignment_output_is_accepted(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 2000, "Merhaba.")],
            speech_coverage_report=coverage(),
            alignment_report=full_alignment_output(),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["alignment_sha256"], ALIGNMENT_SHA)
        self.assertTrue(report["word_ownership_checked"])

    def test_full_output_reads_nested_failures_and_rejects_conflicts(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 2000, "Merhaba.")],
            speech_coverage_report=coverage(),
            alignment_report=full_alignment_output(unaligned_word_count=2),
        )
        self.assertEqual(report["unaligned_word_count"], 2)
        self.assertFalse(report["passed"])

        conflicting = full_alignment_output()
        conflicting["synthetic_timing_count"] = 1
        with self.assertRaisesRegex(
            TimingQAV2Error, "conflicting synthetic_timing_count"
        ):
            run_timing_qa_v2(
                [block(1, 1000, 2000, "Merhaba.")],
                speech_coverage_report=coverage(),
                alignment_report=conflicting,
            )

    def test_nineteen_millisecond_fragment_is_hard_failure(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 1019, "Sizin...")],
            speech_coverage_report=coverage(),
            alignment_report=alignment(),
        )
        self.assertEqual(report["short_cue_count"], 1)
        self.assertGreater(report["high_cps_tr_count"], 0)
        with self.assertRaises(TimingQAV2Error):
            assert_timing_qa_v2(report)

    def test_uncovered_speech_and_unaligned_words_cannot_pass(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 2000, "Merhaba.")],
            speech_coverage_report=coverage(2),
            alignment_report=alignment(unaligned_word_count=1),
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["unresolved_speech_region_count"], 2)
        self.assertEqual(report["unaligned_word_count"], 1)

    def test_missing_or_low_alignment_score_cannot_pass(self) -> None:
        for field in (
            "missing_alignment_score_count",
            "low_alignment_score_count",
        ):
            with self.subTest(field=field):
                report = run_timing_qa_v2(
                    [block(1, 1000, 2000, "Merhaba.")],
                    speech_coverage_report=coverage(),
                    alignment_report=alignment(**{field: 1}),
                )
                self.assertFalse(report["passed"])
                self.assertEqual(report[field], 1)
                with self.assertRaisesRegex(TimingQAV2Error, field):
                    assert_timing_qa_v2(report)

    def test_duration_and_drift_alignment_violations_cannot_pass(self) -> None:
        for field in (
            "overlong_alignment_word_count",
            "outward_drift_violation_count",
            "low_score_edited_token_count",
            "unreviewed_deleted_token_count",
        ):
            with self.subTest(field=field):
                report = run_timing_qa_v2(
                    [block(1, 1000, 2000, "Merhaba.")],
                    speech_coverage_report=coverage(),
                    alignment_report=alignment(**{field: 1}),
                )
                self.assertFalse(report["passed"])
                self.assertEqual(report[field], 1)
                with self.assertRaisesRegex(TimingQAV2Error, field):
                    assert_timing_qa_v2(report)

    def test_review_score_counter_is_warning_not_hard_failure(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 2000, "Merhaba.")],
            speech_coverage_report=coverage(),
            alignment_report=alignment(review_alignment_score_count=1),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["review_alignment_score_count"], 1)
        assert_timing_qa_v2(report)

    def test_self_consistent_but_unproven_timing_fails_provenance(self) -> None:
        value = block(1, 1000, 2000, "Merhaba.")
        value["vad_info"]["alignment_provenance"]["word_timing_sources"] = ["faster_whisper_word_timestamp"]
        report = run_timing_qa_v2(
            [value],
            speech_coverage_report=coverage(),
            alignment_report=alignment(),
        )
        self.assertEqual(report["alignment_provenance_mismatch_count"], 1)

    def test_high_indonesian_cps_is_hard_failure(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 2000, "Evet.")],
            speech_coverage_report=coverage(),
            alignment_report=alignment(),
            id_text_by_uid={"uid-1": "Ini kalimat Indonesia yang terlalu panjang sekali."},
        )
        self.assertEqual(report["high_cps_id_count"], 1)

    def test_adjacent_short_duplicate_is_hard_failure(self) -> None:
        report = run_timing_qa_v2(
            [block(1, 1000, 1800, "Ne oldu?"), block(2, 1900, 2700, "Ne oldu?")],
            speech_coverage_report=coverage(),
            alignment_report=alignment(),
        )
        self.assertEqual(report["adjacent_short_duplicate_count"], 1)


@pytest.mark.parametrize("words", [
    ("Zeytin", "Gökyüzü"),
    ("Bekledik", "Döneceğiz"),
    ("Tamam", "Tamam"),
])
@pytest.mark.parametrize("boundary", ["early_start", "late_start", "early_end"])
def test_timing_qa_binds_each_scene_to_actual_words_not_mutable_metadata(words, boundary):
    values = [block(1, 1000, 2000, words[0]), block(2, 32_000, 33_000, words[1])]
    report_source = full_alignment_output()
    report_source["words"] = [
        {"text": words[0], "start_ms": 1000, "end_ms": 1780, "utterance_uid": "source-a"},
        {"text": words[1], "start_ms": 32_000, "end_ms": 32_900,
         "utterance_uid": "source-b"},
    ]
    if boundary == "early_start":
        values[1]["start_ms"] = 28_000
    elif boundary == "late_start":
        values[1]["start_ms"] = 32_050
    else:
        values[1]["end_ms"] = 32_850
    values[1]["vad_info"].update(
        first_word_start_ms=values[1]["start_ms"], last_word_end_ms=values[1]["end_ms"]
    )
    original = copy.deepcopy(values)
    report = run_timing_qa_v2(
        values, speech_coverage_report=coverage(), alignment_report=report_source
    )
    assert not report["passed"]
    assert report["invalid_timing_count"] == 1
    boundary_issues = [issue for issue in report["issues"]
                       if issue["code"] == "cue_acoustic_boundary_mismatch"]
    assert boundary_issues == [{
        "code": "cue_acoustic_boundary_mismatch", "block_index": 2, "block_uid": "uid-2",
        "actual_start_ms": values[1]["start_ms"], "actual_end_ms": values[1]["end_ms"],
        "first_word_start_ms": 32_000, "last_word_end_ms": 32_900,
        "utterance_uids": ["source-b"],
    }]
    assert all(issue["block_uid"] == "uid-2" for issue in report["issues"])
    assert values == original


def test_timing_qa_early_start_one_ms_is_not_hidden_by_legacy_padding():
    report = run_timing_qa_v2(
        [block(1, 999, 2000, "Merhaba.")],
        speech_coverage_report=coverage(), alignment_report=full_alignment_output(),
    )
    assert not report["passed"]
    assert report["issues"][0]["first_word_start_ms"] == 1000


def test_full_alignment_without_acoustic_words_cannot_prove_cue_ownership():
    value = full_alignment_output()
    value["words"] = []
    report = run_timing_qa_v2(
        [block(1, 1000, 2000, "Merhaba.")],
        speech_coverage_report=coverage(), alignment_report=value,
    )
    assert not report["passed"]
    assert report["issues"][0]["code"] == "acoustic_text_mismatch"


def test_timing_qa_rejects_generic_word_moved_to_previous_scene_with_total_text_preserved():
    values = [block(1, 1000, 2500, "Geldik. Yarın"), block(2, 32_300, 33_300, "döneriz.")]
    source = full_alignment_output()
    source["words"] = [
        {"text": "Geldik.", "start_ms": 1000, "end_ms": 2000},
        {"text": "Yarın", "start_ms": 32_000, "end_ms": 32_280},
        {"text": "döneriz.", "start_ms": 32_300, "end_ms": 32_900},
    ]
    report = run_timing_qa_v2(
        values, speech_coverage_report=coverage(), alignment_report=source,
    )
    assert not report["passed"]
    assert report["invalid_timing_count"] == 1
    assert report["issues"][0]["block_uid"] == "uid-1"
    assert report["issues"][0]["last_word_end_ms"] == 32_280


def test_timing_qa_keeps_different_known_speaker_ownership_with_overlap():
    values = [block(1, 1000, 2200, "Zeytin."), block(2, 1500, 2600, "Gökyüzü.")]
    source = full_alignment_output()
    source["words"] = [
        {"text": "Zeytin.", "start_ms": 1000, "end_ms": 2000, "speaker_id": "A"},
        {"text": "Gökyüzü.", "start_ms": 1500, "end_ms": 2300, "speaker_id": "B"},
    ]
    for value, speaker in zip(values, ("A", "B"), strict=True):
        value["speaker_id"] = speaker
    report = run_timing_qa_v2(
        values, speech_coverage_report=coverage(), alignment_report=source,
    )
    assert report["passed"]


if __name__ == "__main__":
    unittest.main()
