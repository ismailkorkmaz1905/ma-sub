from __future__ import annotations

import copy
import unittest

from src.schema_v2 import SchemaV2Error, build_aligned_turkish_schema


ALIGNMENT_SHA = "b" * 64
AUDIO_SHA = "c" * 64


def alignment() -> dict:
    return {
        "alignment_sha256": ALIGNMENT_SHA,
        "audio_sha256": AUDIO_SHA,
        "unaligned_word_count": 0,
        "synthetic_timing_count": 0,
        "negative_word_gap_count": 0,
        "missing_alignment_score_count": 0,
        "low_alignment_score_count": 0,
        "overlong_alignment_word_count": 0,
        "outward_drift_violation_count": 0,
        "low_score_edited_token_count": 0,
        "unreviewed_deleted_token_count": 0,
    }


def full_alignment_output(**counter_overrides: int) -> dict:
    report = alignment()
    report.update(counter_overrides)
    return {
        "format_version": "1.0",
        "alignment_sha256": ALIGNMENT_SHA,
        "audio_sha256": AUDIO_SHA,
        "timing_source": "whisperx_ctc_forced_alignment",
        "segments": [],
        "words": [],
        "report": report,
    }


def coverage() -> dict:
    return {"unresolved_speech_region_count": 0, "speech_coverage_ratio": 0.99}


def block() -> dict:
    return {
        "start_ms": 1000,
        "end_ms": 2000,
        "primary_text": "Merhaba.",
        "context_before": "",
        "context_after": "Nasılsın?",
        "risk_flags": [],
        "vad_info": {
            "alignment_provenance": {
                "alignment_sha256": ALIGNMENT_SHA,
                "audio_sha256": AUDIO_SHA,
                "word_timing_sources": ["whisperx_ctc_forced_alignment"],
            }
        },
    }


class SchemaV2Tests(unittest.TestCase):
    def test_schema_binds_alignment_and_coverage(self) -> None:
        schema = build_aligned_turkish_schema(
            [block()], episode=12, alignment_report=alignment(), speech_coverage_report=coverage()
        )
        self.assertEqual(schema["schema_version"], "2.0")
        self.assertEqual(schema["blocks"][0]["tr_text"], "Merhaba.")
        self.assertEqual(schema["audio_sha256"], AUDIO_SHA)
        self.assertEqual(len(schema["schema_sha256"]), 64)

    def test_full_forced_alignment_output_matches_flat_report(self) -> None:
        flat = build_aligned_turkish_schema(
            [block()],
            episode=12,
            alignment_report=alignment(),
            speech_coverage_report=coverage(),
        )
        full = build_aligned_turkish_schema(
            [block()],
            episode=12,
            alignment_report=full_alignment_output(),
            speech_coverage_report=coverage(),
        )

        self.assertEqual(full, flat)

    def test_full_output_uses_nested_counters_and_rejects_conflicts(self) -> None:
        with self.assertRaisesRegex(SchemaV2Error, "unaligned words"):
            build_aligned_turkish_schema(
                [block()],
                episode=12,
                alignment_report=full_alignment_output(unaligned_word_count=1),
                speech_coverage_report=coverage(),
            )
        conflicting = full_alignment_output()
        conflicting["unaligned_word_count"] = 1
        with self.assertRaisesRegex(SchemaV2Error, "conflicting unaligned_word_count"):
            build_aligned_turkish_schema(
                [block()],
                episode=12,
                alignment_report=conflicting,
                speech_coverage_report=coverage(),
            )

    def test_missing_or_low_alignment_scores_cannot_lock(self) -> None:
        for field_name, message in (
            ("missing_alignment_score_count", "missing alignment scores"),
            ("low_alignment_score_count", "low alignment scores"),
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(SchemaV2Error, message):
                    build_aligned_turkish_schema(
                        [block()],
                        episode=12,
                        alignment_report=full_alignment_output(
                            **{field_name: 1}
                        ),
                        speech_coverage_report=coverage(),
                    )

    def test_duration_and_outward_drift_violations_cannot_lock(self) -> None:
        for field_name, message in (
            ("overlong_alignment_word_count", "overlong aligned words"),
            ("outward_drift_violation_count", "excessive outward drift"),
            ("low_score_edited_token_count", "low-score edited tokens"),
            ("unreviewed_deleted_token_count", "unreviewed deleted tokens"),
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(SchemaV2Error, message):
                    build_aligned_turkish_schema(
                        [block()],
                        episode=12,
                        alignment_report=full_alignment_output(
                            **{field_name: 1}
                        ),
                        speech_coverage_report=coverage(),
                    )

    def test_boolean_or_missing_hard_counter_cannot_lock(self) -> None:
        for field_name in (
            "overlong_alignment_word_count",
            "outward_drift_violation_count",
        ):
            with self.subTest(field=field_name, value=False):
                bad = alignment()
                bad[field_name] = False
                with self.assertRaises(SchemaV2Error):
                    build_aligned_turkish_schema(
                        [block()],
                        episode=12,
                        alignment_report=bad,
                        speech_coverage_report=coverage(),
                    )
            with self.subTest(field=field_name, value="missing"):
                bad = alignment()
                del bad[field_name]
                with self.assertRaises(SchemaV2Error):
                    build_aligned_turkish_schema(
                        [block()],
                        episode=12,
                        alignment_report=bad,
                        speech_coverage_report=coverage(),
                    )
    def test_coverage_change_changes_schema_identity(self) -> None:
        first = build_aligned_turkish_schema(
            [block()], episode=12, alignment_report=alignment(), speech_coverage_report=coverage()
        )
        changed_coverage = coverage()
        changed_coverage["speech_coverage_ratio"] = 0.98
        second = build_aligned_turkish_schema(
            [block()], episode=12, alignment_report=alignment(), speech_coverage_report=changed_coverage
        )
        self.assertNotEqual(first["schema_sha256"], second["schema_sha256"])

    def test_unresolved_speech_cannot_lock(self) -> None:
        bad = coverage()
        bad["unresolved_speech_region_count"] = 1
        with self.assertRaises(SchemaV2Error):
            build_aligned_turkish_schema(
                [block()], episode=12, alignment_report=alignment(), speech_coverage_report=bad
            )

    def test_block_alignment_digest_must_match(self) -> None:
        bad_block = copy.deepcopy(block())
        bad_block["vad_info"]["alignment_provenance"]["alignment_sha256"] = "c" * 64
        with self.assertRaises(SchemaV2Error):
            build_aligned_turkish_schema(
                [bad_block], episode=12, alignment_report=alignment(), speech_coverage_report=coverage()
            )


if __name__ == "__main__":
    unittest.main()
