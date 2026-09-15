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
    def test_nested_required_sources_do_not_label_larger_generic_hole_mandatory(self) -> None:
        required = [{"start_ms": 4000, "end_ms": 6000, "source_id": "outer", "source_sha256": "a" * 64},
                    {"start_ms": 5000, "end_ms": 5500, "source_id": "inner", "source_sha256": "b" * 64}]
        for outside_enabled in (False, True):
            with self.subTest(outside_enabled=outside_enabled):
                report = analyze_speech_coverage([{"start_ms": 0, "end_ms": 30000}], [],
                    required_uncovered_intervals=required, include_required_outside_vad=outside_enabled,
                    config=_config(rescue_padding_ms=0))
                for collection in (report["coverage_issues"], report["rescue_spans"]):
                    self.assertEqual([(r["start_ms"], r["end_ms"],
                                      [s["source_id"] for s in r.get("required_source_records", [])]) for r in collection],
                        [(0, 4000, []), (4000, 5000, ["outer"]), (5000, 5500, ["inner", "outer"]),
                         (5500, 6000, ["outer"]), (6000, 30000, [])])
                for item in (report["coverage_issues"][0], report["coverage_issues"][-1]):
                    self.assertNotIn("required", item["issue_kind"])
                    self.assertFalse(any(reason.startswith("excluded_incomplete") for reason in item["reasons"]))

    def test_small_required_overlap_cannot_unlock_ordinary_192_span_cap(self) -> None:
        from mas.engine.raw_asr import _bounded_rescue_batches
        from mas.engine.transcribe import TranscriptionError

        vad = [{"start_ms": index * 10000, "end_ms": index * 10000 + 3000} for index in range(192)]
        report = analyze_speech_coverage(vad, [], config=_config(rescue_padding_ms=0),
            required_uncovered_intervals=[{"start_ms": 1000, "end_ms": 2000,
                                           "source_id": "small", "source_sha256": "a" * 64}],
            include_required_outside_vad=True)
        spans = report["rescue_spans"]
        self.assertEqual(sum(not span.get("required_source_records") for span in spans), 193)
        self.assertEqual(sum(bool(span.get("required_source_records")) for span in spans), 1)
        with self.assertRaisesRegex(TranscriptionError, "193 spans, above hard limit 192"):
            _bounded_rescue_batches(spans, limit=173, hard_limit=192, mandatory_hard_limit=256)

    def test_every_mandatory_issue_and_actual_rescue_is_bounded_inside_or_outside_vad(self) -> None:
        required = [{"start_ms": 0, "end_ms": 184000, "source_id": "incomplete", "source_sha256": "a" * 64}]
        words = [{"start_ms": 60000, "end_ms": 61000}]
        for vad in ([{"start_ms": 0, "end_ms": 184000}], [{"start_ms": 60000, "end_ms": 61000}]):
            with self.subTest(vad=vad):
                report = analyze_speech_coverage(vad, words, required_uncovered_intervals=required,
                    include_required_outside_vad=True)
                for collection in (report["coverage_issues"], report["rescue_spans"]):
                    self.assertTrue(collection)
                    self.assertTrue(all(item["end_ms"] - item["start_ms"] <= 10000 for item in collection))
                    self.assertFalse(any(item["start_ms"] < 61000 and 60000 < item["end_ms"] for item in collection))
                self.assertEqual(sum(item["duration_ms"] for item in report["coverage_issues"]), 183000)

    def test_outside_required_evidence_is_not_fake_vad_and_needs_exact_frozen_review(self) -> None:
        vad = [{"start_ms": 0, "end_ms": 1000}]
        words = [{"start_ms": 0, "end_ms": 1000}]
        required = [{"start_ms": 7000, "end_ms": 8000, "source_id": "excluded", "source_sha256": "a" * 64}]
        kwargs = {"required_uncovered_intervals": required, "include_required_outside_vad": True}
        report = analyze_speech_coverage(vad, words, **kwargs)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["metrics"]["speech_duration_ms"], 1000)
        self.assertEqual(report["metrics"]["independent_vad_speech_coverage_ratio"], 1.0)
        self.assertEqual(report["metrics"]["required_evidence_ms"], 1000)
        self.assertEqual(report["metrics"]["required_uncovered_ms"], 1000)
        self.assertEqual(report["metrics"]["required_outside_vad_evidence_ms"], 1000)
        review = {"start_ms": 7000, "end_ms": 8000, "classification": "non_dialogue",
                  "review_status": "reviewed", "reason": "Exact bounded acoustic review: no dialogue."}
        cleared = analyze_speech_coverage(vad, words, reviewed_non_dialogue=[review],
                    required_review_targets=[{"start_ms": 7000, "end_ms": 8000}], **kwargs)
        self.assertEqual(cleared["status"], "PASS")
        self.assertEqual(cleared["metrics"]["required_uncovered_ms"], 0)
        missing_target = analyze_speech_coverage(vad, words, reviewed_non_dialogue=[review],
                    required_review_targets=[{"start_ms": 0, "end_ms": 1000}], **kwargs)
        self.assertEqual(missing_target["status"], "FAIL")

    def test_confirmed_short_dialogue_clears_only_outside_target_with_real_word(self) -> None:
        required = [{"start_ms": 0, "end_ms": 1000, "source_id": "vad", "source_sha256": "a" * 64},
                    {"start_ms": 7000, "end_ms": 8000, "source_id": "outside", "source_sha256": "b" * 64}]
        review = {"start_ms": 7000, "end_ms": 8000, "classification": "dialogue", "review_id": "exact",
                  "review_status": "reviewed", "reason": "An exact review heard one short word."}
        kwargs = dict(required_uncovered_intervals=required, include_required_outside_vad=True,
                      reviewed_dialogue=[review], required_review_targets=[{"start_ms": 7000, "end_ms": 8000}])
        result = analyze_speech_coverage([{"start_ms": 0, "end_ms": 1000}],
                    [{"start_ms": 7200, "end_ms": 7400}], **kwargs)
        self.assertEqual([(i["start_ms"], i["end_ms"]) for i in result["coverage_issues"]], [(0, 1000)])
        self.assertEqual(result["status"], "FAIL")
        missing_word = analyze_speech_coverage([{"start_ms": 0, "end_ms": 1000}], [], **kwargs)
        self.assertEqual(missing_word["metrics"]["required_uncovered_ms"], 2000)

    def test_required_metrics_do_not_double_count_generic_hole_and_padding(self) -> None:
        required = {"start_ms": 4500, "end_ms": 5000, "source_id": "excluded", "source_sha256": "a" * 64}
        result = analyze_speech_coverage([{"start_ms": 0, "end_ms": 5000}],
                    [{"start_ms": 0, "end_ms": 4500}], required_uncovered_intervals=[required, dict(required)],
                    include_required_outside_vad=True)
        self.assertLess(result["speech_coverage_ratio"], 1.0)
        self.assertEqual(result["metrics"]["required_uncovered_ms"], 500)
        self.assertEqual(result["metrics"]["required_evidence_ms"], 500)
        self.assertEqual(result["metrics"]["required_inside_vad_evidence_ms"], 500)

    def test_required_rescue_never_crosses_trusted_words_and_keeps_outside_word_boundary(self) -> None:
        required = [{"start_ms": 0, "end_ms": 1000, "source_id": "one", "source_sha256": "a" * 64},
                    {"start_ms": 2000, "end_ms": 3000, "source_id": "two", "source_sha256": "b" * 64}]
        words = [{"start_ms": 1300, "end_ms": 1700}]
        report = analyze_speech_coverage([{"start_ms": 1300, "end_ms": 1700}], words,
                    required_uncovered_intervals=required, include_required_outside_vad=True,
                    config=_config(rescue_padding_ms=1000, rescue_merge_gap_ms=2000))
        self.assertEqual([(i["start_ms"], i["end_ms"]) for i in report["coverage_issues"]], [(0, 1000), (2000, 3000)])
        for span in report["rescue_spans"]:
            self.assertLessEqual(span["duration_ms"], 10000)
            self.assertFalse(span["start_ms"] < 1700 and 1300 < span["end_ms"])
            self.assertTrue(span["required_source_records"])

    def test_excluded_short_line_requires_review_despite_high_generic_coverage(self) -> None:
        vad = [{"start_ms": 0, "end_ms": 5000}]
        for start in (4500, 4900):
            with self.subTest(duration_ms=5000-start):
                words = [{"start_ms": 0, "end_ms": start}]
                generic = analyze_speech_coverage(vad, words)
                self.assertEqual(generic["status"], "PASS")
                required = [{"start_ms": start, "end_ms": 5000,
                             "source_id": "excluded-Evet", "source_sha256": "a" * 64}]
                result = analyze_speech_coverage(vad, words, required_uncovered_intervals=required)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(len(result["coverage_issues"]), 1)
                issue = result["coverage_issues"][0]
                self.assertEqual((issue["start_ms"], issue["end_ms"]), (start, 5000))
                self.assertEqual(issue["required_source_records"], [{"source_id": "excluded-Evet", "source_sha256": "a" * 64}])
                self.assertEqual(len(result["rescue_spans"]), 1)
                self.assertEqual(generic, analyze_speech_coverage(vad, words, required_uncovered_intervals=[]))

    def test_required_holes_merge_with_generic_and_duplicate_source_requirements(self) -> None:
        required = {"start_ms": 4000, "end_ms": 5000,
                    "source_id": "excluded-1", "source_sha256": "b" * 64}
        result = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 5000}], [{"start_ms": 0, "end_ms": 4000}],
            required_uncovered_intervals=[required, dict(required)],
        )
        self.assertEqual(len(result["coverage_issues"]), 1)
        issue = result["coverage_issues"][0]
        self.assertEqual((issue["start_ms"], issue["end_ms"]), (4000, 5000))
        self.assertIn("speech_hole_above_min_duration", issue["reasons"])
        self.assertEqual(len(issue["required_source_records"]), 1)
        self.assertEqual(result["metrics"]["unresolved_coverage_issue_count"], 1)

    def test_required_holes_cannot_invent_speech_in_vad_merge_gap_or_ignore_real_words(self) -> None:
        required = [{"start_ms": 1000, "end_ms": 1200,
                     "source_id": "excluded-1", "source_sha256": "b" * 64}]
        result = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 1100, "end_ms": 2000}],
            [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 1200, "end_ms": 2000}],
            required_uncovered_intervals=required,
        )
        self.assertEqual([(i["start_ms"], i["end_ms"]) for i in result["coverage_issues"]], [(1100, 1200)])
        covered = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 2000}], [{"start_ms": 0, "end_ms": 2000}],
            required_uncovered_intervals=required,
        )
        self.assertEqual(covered["status"], "PASS")
        with self.assertRaisesRegex(SpeechCoverageError, "SHA-256"):
            analyze_speech_coverage([], [], required_uncovered_intervals=[dict(required[0], source_sha256="bad")])

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
