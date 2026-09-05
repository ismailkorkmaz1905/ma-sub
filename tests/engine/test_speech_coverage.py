from __future__ import annotations

import json
import unittest

from mas.engine.speech_coverage import (
    SpeechCoverageConfig,
    SpeechCoverageError,
    V2_BETA_SPEECH_COVERAGE_CONFIG,
    analyze_speech_coverage,
    build_rescue_spans,
    require_v2_beta_speech_coverage_config,
)


def _config(**overrides: object) -> SpeechCoverageConfig:
    values: dict[str, object] = {
        "vad_merge_gap_ms": 0,
        "word_padding_ms": 0,
        "word_merge_gap_ms": 0,
        "min_hole_ms": 750,
        "min_region_coverage_ratio": 0.50,
        "rescue_padding_ms": 200,
        "rescue_merge_gap_ms": 0,
        "max_word_outside_speech_ms": 120,
    }
    values.update(overrides)
    return SpeechCoverageConfig(**values)  # type: ignore[arg-type]


class SpeechCoverageAnalysisTests(unittest.TestCase):
    def test_fully_covered_speech_passes_with_exact_metrics(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 5_000}],
            [
                {"start_ms": 1_000, "end_ms": 2_500, "text": "Bir"},
                {"start_ms": 2_500, "end_ms": 5_000, "text": "cümle"},
            ],
            config=_config(),
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["unresolved_speech_region_count"], 0)
        self.assertEqual(report["speech_coverage_ratio"], 1.0)
        self.assertEqual(report["metrics"]["covered_speech_ms"], 4_000)
        self.assertEqual(report["metrics"]["uncovered_speech_ms"], 0)
        self.assertEqual(report["speech_regions"][0]["covered_intervals"], [
            {"start_ms": 1_000, "end_ms": 5_000, "duration_ms": 4_000}
        ])
        self.assertEqual(report["coverage_issues"], [])
        self.assertEqual(report["rescue_spans"], [])

    def test_long_uncovered_speech_hole_fails_and_builds_padded_rescue(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 5_000}],
            [
                {"start_ms": 1_000, "end_ms": 1_800},
                {"start_ms": 3_000, "end_ms": 5_000},
            ],
            config=_config(),
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["unresolved_speech_region_count"], 1)
        issue = report["coverage_issues"][0]
        self.assertEqual(issue["issue_kind"], "speech_hole")
        self.assertEqual(issue["classification"], "unresolved_speech")
        self.assertEqual((issue["start_ms"], issue["end_ms"]), (1_800, 3_000))
        rescue = report["rescue_spans"][0]
        self.assertEqual((rescue["start_ms"], rescue["end_ms"]), (1_600, 3_200))
        self.assertEqual(rescue["issue_ids"], [issue["issue_id"]])

    def test_low_coverage_flags_only_actual_gaps_when_each_gap_is_short(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 4_000}],
            [
                {"start_ms": 0, "end_ms": 500},
                {"start_ms": 900, "end_ms": 1_400},
                {"start_ms": 1_800, "end_ms": 2_300},
                {"start_ms": 2_700, "end_ms": 3_200},
                {"start_ms": 3_500, "end_ms": 4_000},
            ],
            config=_config(
                min_hole_ms=500,
                min_region_coverage_ratio=0.80,
            ),
        )

        self.assertEqual(report["status"], "FAIL")
        issues = report["coverage_issues"]
        self.assertEqual(
            [(issue["start_ms"], issue["end_ms"]) for issue in issues],
            [(500, 900), (1_400, 1_800), (2_300, 2_700), (3_200, 3_500)],
        )
        self.assertTrue(
            all(issue["issue_kind"] == "low_coverage_speech_hole" for issue in issues)
        )
        self.assertTrue(
            all(
                issue["reasons"] == ["region_coverage_below_threshold"]
                for issue in issues
            )
        )
        self.assertEqual(report["speech_regions"][0]["significant_hole_count"], 0)

    def test_partial_words_with_missing_tail_emit_only_the_uncovered_tail(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 4_000}],
            [{"start_ms": 0, "end_ms": 1_000}],
            config=_config(min_region_coverage_ratio=0.80),
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["coverage_issues"]), 1)
        issue = report["coverage_issues"][0]
        self.assertEqual(issue["issue_kind"], "low_coverage_speech_hole")
        self.assertEqual((issue["start_ms"], issue["end_ms"]), (1_000, 4_000))
        self.assertEqual(issue["duration_ms"], 3_000)
        self.assertEqual(
            report["speech_regions"][0]["covered_intervals"],
            [{"start_ms": 0, "end_ms": 1_000, "duration_ms": 1_000}],
        )

    def test_word_outside_vad_cannot_gain_coverage_only_from_padding(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_200, "end_ms": 1_450}],
            [{"start_ms": 1_000, "end_ms": 1_200}],
            config=_config(
                word_padding_ms=60,
                word_merge_gap_ms=100,
                min_hole_ms=400,
                min_region_coverage_ratio=0.20,
            ),
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["speech_coverage_ratio"], 0.0)
        self.assertEqual(
            report["metrics"]["word_interval_wholly_outside_speech_count"],
            1,
        )
        self.assertEqual(
            report["word_intervals_wholly_outside_speech"],
            [{"start_ms": 1_000, "end_ms": 1_200, "duration_ms": 200}],
        )
        self.assertEqual(
            [
                (issue["start_ms"], issue["end_ms"])
                for issue in report["coverage_issues"]
            ],
            [(1_200, 1_450)],
        )

    def test_word_with_one_ms_vad_touch_fails_per_word_containment_gate(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 2_500}],
            [
                {"start_ms": 100, "end_ms": 1_001, "text": "erken"},
                {"start_ms": 1_100, "end_ms": 2_450, "text": "konuşma"},
            ],
            config=_config(
                min_hole_ms=750,
                min_region_coverage_ratio=0.80,
            ),
        )

        self.assertEqual(report["unresolved_speech_region_count"], 0)
        self.assertEqual(report["coverage_issues"], [])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(
            report["metrics"]["word_interval_outside_speech_violation_count"],
            1,
        )
        violation = report[
            "word_intervals_exceeding_outside_speech_tolerance"
        ][0]
        self.assertEqual(violation["inside_speech_ms"], 1)
        self.assertEqual(violation["outside_speech_ms"], 900)
        self.assertEqual(
            violation["maximum_contiguous_outside_speech_ms"], 900
        )

    def test_word_outside_speech_tolerance_boundary_is_exact(self) -> None:
        accepted = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 2_000}],
            [{"start_ms": 880, "end_ms": 2_000}],
            config=_config(min_region_coverage_ratio=0.90),
        )
        rejected = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 2_000}],
            [{"start_ms": 879, "end_ms": 2_000}],
            config=_config(min_region_coverage_ratio=0.90),
        )

        self.assertEqual(accepted["status"], "PASS")
        self.assertEqual(
            accepted["metrics"]["word_interval_outside_speech_violation_count"],
            0,
        )
        self.assertEqual(rejected["status"], "FAIL")
        self.assertEqual(
            rejected["metrics"]["word_interval_outside_speech_violation_count"],
            1,
        )

    def test_exact_reviewed_dialogue_extends_speech_inventory(self) -> None:
        dialogue_review = {
            "review_id": "candidate-audio:1",
            "start_ms": 3_000,
            "end_ms": 3_200,
            "classification": "dialogue",
            "review_status": "reviewed",
            "reason": "Exact contextual WAV contains a spoken interjection.",
        }
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 2_000}],
            [
                {"start_ms": 1_000, "end_ms": 2_000},
                {"start_ms": 3_000, "end_ms": 3_200},
            ],
            config=_config(min_hole_ms=150),
            reviewed_dialogue=[dialogue_review],
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["unresolved_speech_region_count"], 0)
        self.assertEqual(
            report["metrics"]["word_interval_wholly_outside_speech_count"], 0
        )
        self.assertEqual(report["metrics"]["reviewed_dialogue_record_count"], 1)
        self.assertEqual(report["reviewed_dialogue_records"], [
            {**dialogue_review, "duration_ms": 200}
        ])

    def test_reviewed_dialogue_without_aligned_words_still_fails(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 2_000}],
            [{"start_ms": 1_000, "end_ms": 2_000}],
            config=_config(min_hole_ms=150),
            reviewed_dialogue=[
                {
                    "review_id": "candidate-audio:missing",
                    "start_ms": 3_000,
                    "end_ms": 3_200,
                    "classification": "dialogue",
                    "review_status": "reviewed",
                    "reason": "Reviewer heard speech, so alignment is mandatory.",
                }
            ],
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["unresolved_speech_region_count"], 1)
        self.assertEqual(
            [(item["start_ms"], item["end_ms"]) for item in report["coverage_issues"]],
            [(3_000, 3_200)],
        )

    def test_explicit_reviewed_non_dialogue_resolves_matching_issue(self) -> None:
        review = {
            "review_id": "noise-1",
            "start_ms": 1_800,
            "end_ms": 3_000,
            "classification": "non_dialogue",
            "review_status": "reviewed",
            "reason": "Door slam and score music; no spoken dialogue.",
        }
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 5_000}],
            [
                {"start_ms": 1_000, "end_ms": 1_800},
                {"start_ms": 3_000, "end_ms": 5_000},
            ],
            config=_config(),
            reviewed_non_dialogue=[review],
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["unresolved_speech_region_count"], 0)
        self.assertEqual(report["metrics"]["reviewed_non_dialogue_issue_count"], 1)
        self.assertEqual(report["coverage_issues"][0]["classification"], "reviewed_non_dialogue")
        self.assertEqual(report["coverage_issues"][0]["review_id"], "noise-1")
        self.assertEqual(report["rescue_spans"], [])
        self.assertEqual(
            report["reviewed_non_dialogue_records"][0]["matched_issue_ids"],
            [report["coverage_issues"][0]["issue_id"]],
        )

    def test_broad_review_cannot_blanket_clear_a_smaller_speech_hole(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 4_000}],
            [
                {"start_ms": 0, "end_ms": 1_000},
                {"start_ms": 3_000, "end_ms": 4_000},
            ],
            config=_config(min_hole_ms=500, min_region_coverage_ratio=0.40),
            reviewed_non_dialogue=[
                {
                    "review_id": "unsafe-broad-review",
                    "start_ms": 500,
                    "end_ms": 3_500,
                    "classification": "non_dialogue",
                    "review_status": "reviewed",
                    "reason": "This interval is broader than the actual hole.",
                }
            ],
        )

        issue = report["coverage_issues"][0]
        review = report["reviewed_non_dialogue_records"][0]
        self.assertEqual((issue["start_ms"], issue["end_ms"]), (1_000, 3_000))
        self.assertEqual(issue["classification"], "unresolved_speech")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["unresolved_speech_region_count"], 1)
        self.assertEqual(review["matched_issue_ids"], [])
        self.assertEqual(review["partially_overlapping_issue_ids"], [issue["issue_id"]])

    def test_partial_review_does_not_hide_unresolved_speech(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 4_000}],
            [
                {"start_ms": 0, "end_ms": 1_000},
                {"start_ms": 3_000, "end_ms": 4_000},
            ],
            config=_config(min_region_coverage_ratio=0.40),
            reviewed_non_dialogue=[
                {
                    "review_id": "partial",
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "classification": "non_dialogue",
                    "review_status": "reviewed",
                    "reason": "Only this first half was checked.",
                }
            ],
        )

        issue = report["coverage_issues"][0]
        review = report["reviewed_non_dialogue_records"][0]
        self.assertEqual(issue["classification"], "unresolved_speech")
        self.assertEqual(review["matched_issue_ids"], [])
        self.assertEqual(review["partially_overlapping_issue_ids"], [issue["issue_id"]])
        self.assertEqual(report["metrics"]["unused_non_dialogue_review_count"], 1)

    def test_nearby_vad_regions_are_merged_without_mutating_sources(self) -> None:
        vad = [
            {"start_ms": 0, "end_ms": 1_000},
            {"start_ms": 1_100, "end_ms": 2_000},
        ]
        report = analyze_speech_coverage(
            vad,
            [{"start_ms": 0, "end_ms": 2_000}],
            config=_config(vad_merge_gap_ms=100),
        )

        self.assertEqual(vad[0]["end_ms"], 1_000)
        self.assertEqual(report["metrics"]["input_vad_region_count"], 2)
        self.assertEqual(report["metrics"]["merged_speech_region_count"], 1)
        self.assertEqual(
            report["speech_regions"][0]["source_vad_record_indices"], [1, 2]
        )
        self.assertEqual(report["status"], "PASS")

    def test_word_padding_and_short_gap_merge_are_explicit_in_report(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 2_000}],
            [
                {"start_ms": 1_100, "end_ms": 1_300},
                {"start_ms": 1_600, "end_ms": 1_900},
            ],
            config=_config(
                word_padding_ms=100,
                word_merge_gap_ms=100,
                min_hole_ms=200,
                min_region_coverage_ratio=0.50,
            ),
        )

        self.assertEqual(report["config"]["word_padding_ms"], 100)
        self.assertEqual(report["config"]["word_merge_gap_ms"], 100)
        self.assertEqual(report["metrics"]["covered_speech_ms"], 1_000)
        self.assertEqual(report["metrics"]["raw_word_covered_speech_ms"], 500)
        self.assertEqual(report["status"], "PASS")

    def test_report_contains_only_json_ready_values(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 1_000}],
            [],
            config=_config(),
        )
        encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
        self.assertIn('"unresolved_speech_region_count": 1', encoded)


class RescueSpanTests(unittest.TestCase):
    def test_padding_overlap_is_merged_into_nonoverlapping_rescue_span(self) -> None:
        issues = [
            {
                "issue_id": "a",
                "speech_region_index": 1,
                "start_ms": 1_000,
                "end_ms": 1_500,
                "classification": "unresolved_speech",
                "reasons": ["speech_hole_above_min_duration"],
            },
            {
                "issue_id": "b",
                "speech_region_index": 2,
                "start_ms": 1_800,
                "end_ms": 2_200,
                "classification": "unresolved_speech",
                "reasons": ["region_coverage_below_threshold"],
            },
        ]

        spans = build_rescue_spans(issues, padding_ms=200, merge_gap_ms=0)

        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0]["start_ms"], spans[0]["end_ms"]), (800, 2_400))
        self.assertEqual(spans[0]["issue_ids"], ["a", "b"])
        self.assertEqual(spans[0]["speech_region_indices"], [1, 2])

    def test_reviewed_issue_is_excluded_from_rescue(self) -> None:
        spans = build_rescue_spans(
            [
                {
                    "issue_id": "reviewed",
                    "speech_region_index": 1,
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "classification": "reviewed_non_dialogue",
                    "reasons": ["speech_hole_above_min_duration"],
                }
            ]
        )
        self.assertEqual(spans, [])


class SpeechCoverageValidationTests(unittest.TestCase):
    def test_canonical_policy_ignores_normal_between_word_pause(self) -> None:
        report = analyze_speech_coverage(
            [{"start_ms": 1_000, "end_ms": 3_000}],
            [
                {"start_ms": 1_000, "end_ms": 1_350},
                {"start_ms": 2_050, "end_ms": 2_400},
                {"start_ms": 2_650, "end_ms": 3_000},
            ],
            config=V2_BETA_SPEECH_COVERAGE_CONFIG,
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["metrics"]["unresolved_coverage_issue_count"], 0)
        self.assertEqual(report["rescue_spans"], [])

    def test_canonical_beta_config_constructs_at_module_import(self) -> None:
        self.assertIsInstance(
            V2_BETA_SPEECH_COVERAGE_CONFIG,
            SpeechCoverageConfig,
        )
        self.assertIs(
            require_v2_beta_speech_coverage_config(
                V2_BETA_SPEECH_COVERAGE_CONFIG
            ),
            V2_BETA_SPEECH_COVERAGE_CONFIG,
        )
        with self.assertRaisesRegex(
            SpeechCoverageError, "canonical V2 beta publication policy"
        ):
            require_v2_beta_speech_coverage_config(
                SpeechCoverageConfig(
                    min_hole_ms=10_000,
                    min_region_coverage_ratio=0.0,
                )
            )

    def test_malformed_timing_is_rejected_instead_of_coerced(self) -> None:
        invalid_values = (True, 1_000.0, "1000", -1)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(SpeechCoverageError):
                    analyze_speech_coverage(
                        [{"start_ms": value, "end_ms": 2_000}],
                        [],
                    )

    def test_missing_and_zero_length_timing_are_rejected(self) -> None:
        with self.assertRaisesRegex(SpeechCoverageError, "missing end_ms"):
            analyze_speech_coverage([{"start_ms": 0}], [])
        with self.assertRaisesRegex(SpeechCoverageError, "greater than start_ms"):
            analyze_speech_coverage([{"start_ms": 500, "end_ms": 500}], [])

    def test_out_of_order_vad_is_rejected(self) -> None:
        with self.assertRaisesRegex(SpeechCoverageError, "out of chronological order"):
            analyze_speech_coverage(
                [
                    {"start_ms": 2_000, "end_ms": 3_000},
                    {"start_ms": 1_000, "end_ms": 1_500},
                ],
                [],
            )

    def test_overlapping_vad_is_rejected(self) -> None:
        with self.assertRaisesRegex(SpeechCoverageError, "overlaps"):
            analyze_speech_coverage(
                [
                    {"start_ms": 1_000, "end_ms": 2_000},
                    {"start_ms": 1_900, "end_ms": 3_000},
                ],
                [],
            )

    def test_overlapping_words_are_rejected(self) -> None:
        with self.assertRaisesRegex(SpeechCoverageError, r"words\[2\] overlaps"):
            analyze_speech_coverage(
                [{"start_ms": 0, "end_ms": 4_000}],
                [
                    {"start_ms": 500, "end_ms": 1_500},
                    {"start_ms": 1_499, "end_ms": 2_000},
                ],
            )

    def test_review_requires_explicit_classification_status_and_reason(self) -> None:
        base = {
            "start_ms": 1_000,
            "end_ms": 2_000,
            "classification": "non_dialogue",
            "review_status": "reviewed",
            "reason": "Music only.",
        }
        for missing_field in ("classification", "review_status", "reason"):
            review = dict(base)
            del review[missing_field]
            with self.subTest(missing_field=missing_field):
                with self.assertRaises(SpeechCoverageError):
                    analyze_speech_coverage(
                        [{"start_ms": 0, "end_ms": 3_000}],
                        [],
                        reviewed_non_dialogue=[review],
                    )

    def test_overlapping_reviews_are_rejected(self) -> None:
        with self.assertRaisesRegex(SpeechCoverageError, "overlaps"):
            analyze_speech_coverage(
                [{"start_ms": 0, "end_ms": 4_000}],
                [],
                reviewed_non_dialogue=[
                    {
                        "start_ms": 500,
                        "end_ms": 2_000,
                        "classification": "non_dialogue",
                        "review_status": "reviewed",
                        "reason": "Noise.",
                    },
                    {
                        "start_ms": 1_900,
                        "end_ms": 3_000,
                        "classification": "non_dialogue",
                        "review_status": "reviewed",
                        "reason": "Music.",
                    },
                ],
            )

    def test_review_cannot_cross_or_escape_merged_speech_region(self) -> None:
        with self.assertRaisesRegex(SpeechCoverageError, "fully contained"):
            analyze_speech_coverage(
                [
                    {"start_ms": 0, "end_ms": 1_000},
                    {"start_ms": 2_000, "end_ms": 3_000},
                ],
                [],
                reviewed_non_dialogue=[
                    {
                        "start_ms": 500,
                        "end_ms": 2_500,
                        "classification": "non_dialogue",
                        "review_status": "reviewed",
                        "reason": "Invalid blanket review.",
                    }
                ],
            )

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(SpeechCoverageError):
            SpeechCoverageConfig(min_hole_ms=0)
        with self.assertRaises(SpeechCoverageError):
            SpeechCoverageConfig(min_region_coverage_ratio=1.01)
        with self.assertRaises(SpeechCoverageError):
            SpeechCoverageConfig(word_padding_ms=True)


if __name__ == "__main__":
    unittest.main()
