from __future__ import annotations

import unittest

from src.timing_qa_v2 import TimingQAV2Error, assert_timing_qa_v2, run_timing_qa_v2


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
        "words": [],
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


if __name__ == "__main__":
    unittest.main()
