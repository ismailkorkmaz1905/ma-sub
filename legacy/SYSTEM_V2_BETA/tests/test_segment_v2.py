from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from helpers import (
    make_schema,
    raw_blocks,
    synthetic_transcription,
    translation_records,
    write_translated_zip,
)
from src.schema import build_episode_schema
from src.segment_v2 import SegmentationError, build_blocks, validate_segmentation
from src.translation_validation import (
    TranslationValidationError,
    load_and_validate_translated_zip,
    validate_translation_records,
)


class SegmentationDriftRegressionTests(unittest.TestCase):
    def test_segmentation_preserves_all_words_and_hard_silence_boundaries(self) -> None:
        transcription = synthetic_transcription(3)
        blocks = build_blocks(transcription, episode=12)
        report = validate_segmentation(blocks, transcription["words"])

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(len(blocks), 3)
        self.assertEqual([block["block_index"] for block in blocks], [1, 2, 3])
        self.assertEqual([block["start_ms"] for block in blocks], [1_000, 5_000, 9_000])
        self.assertTrue(
            all(
                current["end_ms"] < following["start_ms"]
                for current, following in zip(blocks, blocks[1:])
            )
        )
        self.assertTrue(
            all(block["vad_info"]["max_internal_gap_ms"] <= 1_000 for block in blocks)
        )
        self.assertEqual(
            " ".join(block["timing_text"] for block in blocks),
            " ".join(word["text"] for word in transcription["words"]),
        )
        self.assertEqual(build_episode_schema(blocks, episode=12)["block_count"], 3)

    def test_explicit_dialogue_turn_starts_a_new_block(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 1_000,
                    "end_ms": 1_300,
                    "text": "Merhaba.",
                    "probability": 0.99,
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 1_900,
                    "end_ms": 2_200,
                    "text": "-Ben",
                    "probability": 0.99,
                },
                {
                    "word_index": 3,
                    "segment_id": 1,
                    "start_ms": 2_250,
                    "end_ms": 2_550,
                    "text": "geldim.",
                    "probability": 0.99,
                },
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 950,
                    "end_ms": 2_600,
                    "source": "synthetic-test",
                }
            ],
        }
        blocks = build_blocks(transcription, episode=12)

        self.assertEqual([block["primary_text"] for block in blocks], ["Merhaba.", "-Ben geldim."])
        self.assertNotIn("-Ben", blocks[0]["primary_text"])
        self.assertIn("speaker_change_boundary_before", blocks[1]["risk_flags"])

    def test_19ms_decoder_fragment_is_resegmented_across_segment_boundaries(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": index,
                    "segment_id": segment_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "probability": 0.95,
                }
                for index, (segment_id, start_ms, end_ms, text) in enumerate(
                    (
                        (1, 0, 380, "Bunu"),
                        (1, 400, 820, "gerçekten"),
                        (2, 839, 858, "de"),
                        (3, 880, 1_250, "istemiyorum."),
                    ),
                    start=1,
                )
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_300,
                    "source": "synthetic-test",
                }
            ],
        }

        blocks = build_blocks(transcription, episode=12)

        report = validate_segmentation(blocks, transcription["words"])

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(
            " ".join(block["primary_text"] for block in blocks),
            "Bunu gerçekten de istemiyorum.",
        )
        self.assertNotIn("de", [block["primary_text"] for block in blocks])
        self.assertTrue(all(block["end_ms"] - block["start_ms"] >= 700 for block in blocks))
        crossing_blocks = [
            block
            for block in blocks
            if block["vad_info"]["alignment_provenance"][
                "decoder_segment_crossing_count"
            ]
        ]
        self.assertTrue(crossing_blocks)
        self.assertIn("decoder_segment_crossing", crossing_blocks[0]["risk_flags"])
        self.assertNotIn(
            "whisper_segment_boundary",
            crossing_blocks[0]["vad_info"].get("boundary_before", []),
        )

    def test_999ms_aligned_gap_is_a_hard_boundary_not_an_early_display(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 1_000,
                    "end_ms": 1_300,
                    "text": "Bekle",
                    "probability": 0.96,
                    "timing_source": "whisperx_ctc_forced_alignment",
                    "alignment_sha256": "a" * 64,
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 2_299,
                    "end_ms": 2_650,
                    "text": "geldim.",
                    "probability": 0.97,
                    "timing_source": "whisperx_ctc_forced_alignment",
                    "alignment_sha256": "a" * 64,
                },
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 950,
                    "end_ms": 2_700,
                    "source": "silero_vad",
                }
            ],
        }

        blocks = build_blocks(transcription, episode=12)
        report = validate_segmentation(blocks, transcription["words"])

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(
            [block["primary_text"] for block in blocks],
            ["Bekle", "geldim."],
        )
        self.assertEqual(blocks[1]["start_ms"], 2_299)
        self.assertIn(
            "likely_speaker_gap",
            blocks[1]["vad_info"]["boundary_before"],
        )
        self.assertTrue(
            all(
                block["vad_info"]["max_internal_gap_ms"] < 650
                for block in blocks
            )
        )

    def test_validator_hard_fails_internal_gap_at_likely_speaker_threshold(self) -> None:
        unsafe = [
            {
                "block_uid": "",
                "episode": 12,
                "block_index": 1,
                "start_ms": 1_000,
                "end_ms": 3_000,
                "timing_text": "Bekle geldim.",
                "primary_text": "Bekle geldim.",
                "verification_text": "",
                "youtube_text": "",
                "context_before": "",
                "context_after": "",
                "vad_info": {"max_internal_gap_ms": 650},
                "risk_flags": [],
            }
        ]

        report = validate_segmentation(unsafe)

        self.assertFalse(report.valid)
        self.assertEqual(report.unresolved_internal_gap_count, 1)
        self.assertTrue(any("hard 650 ms threshold" in item for item in report.errors))

    def test_unpunctuated_gap_threshold_is_exactly_650ms(self) -> None:
        def transcription(gap_ms: int) -> dict:
            return {
                "words": [
                    {
                        "word_index": 1,
                        "segment_id": 1,
                        "start_ms": 1_000,
                        "end_ms": 1_300,
                        "text": "Bekle",
                        "probability": 0.96,
                    },
                    {
                        "word_index": 2,
                        "segment_id": 1,
                        "start_ms": 1_300 + gap_ms,
                        "end_ms": 1_650 + gap_ms,
                        "text": "geldim.",
                        "probability": 0.97,
                    },
                ],
                "vad_regions": [
                    {
                        "vad_region_index": 1,
                        "start_ms": 950,
                        "end_ms": 2_400,
                        "source": "silero_vad",
                    }
                ],
            }

        below = build_blocks(transcription(649), episode=12)
        at_threshold = build_blocks(transcription(650), episode=12)

        self.assertEqual(len(below), 1)
        self.assertEqual(len(at_threshold), 2)
        self.assertIn(
            "likely_speaker_gap",
            at_threshold[1]["vad_info"]["boundary_before"],
        )

    def test_distinct_vad_islands_force_boundary_below_gap_threshold(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 1_000,
                    "end_ms": 1_300,
                    "text": "Bekle",
                    "probability": 0.96,
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 1_850,
                    "end_ms": 2_200,
                    "text": "geldim.",
                    "probability": 0.97,
                },
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 950,
                    "end_ms": 1_350,
                    "source": "silero_vad",
                },
                {
                    "vad_region_index": 2,
                    "start_ms": 1_800,
                    "end_ms": 2_250,
                    "source": "silero_vad",
                },
            ],
        }

        blocks = build_blocks(transcription, episode=12)

        self.assertEqual([block["primary_text"] for block in blocks], ["Bekle", "geldim."])
        self.assertEqual(blocks[1]["start_ms"], 1_850)
        self.assertIn(
            "vad_region_transition",
            blocks[1]["vad_info"]["boundary_before"],
        )

    def test_short_sentence_before_next_speech_fails_without_retiming(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 800,
                    "end_ms": 1_000,
                    "text": "Evet.",
                    "probability": 0.99,
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 1_300,
                    "end_ms": 1_650,
                    "text": "Geldim.",
                    "probability": 0.99,
                },
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 750,
                    "end_ms": 1_700,
                    "source": "silero_vad",
                }
            ],
        }

        with self.assertRaisesRegex(SegmentationError, "No safe segmentation"):
            build_blocks(transcription, episode=12)

    def test_synthetic_repair_timing_is_rejected_as_canonical_input(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 1_000,
                    "end_ms": 1_040,
                    "text": " Sizi",
                    "probability": 0.79,
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 1_040,
                    "end_ms": 1_041,
                    "text": " istiyor",
                    "probability": 0.0,
                    "timing_source": "synthetic_segment_word_repair",
                },
                {
                    "word_index": 3,
                    "segment_id": 1,
                    "start_ms": 1_040,
                    "end_ms": 1_220,
                    "text": " belki",
                    "probability": 0.99,
                },
                {
                    "word_index": 4,
                    "segment_id": 1,
                    "start_ms": 1_220,
                    "end_ms": 1_420,
                    "text": " de",
                    "probability": 0.99,
                },
                {
                    "word_index": 5,
                    "segment_id": 1,
                    "start_ms": 1_420,
                    "end_ms": 1_660,
                    "text": " beyim.",
                    "probability": 0.90,
                },
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 900,
                    "end_ms": 1_700,
                    "source": "synthetic-test",
                }
            ],
        }

        with self.assertRaisesRegex(
            SegmentationError, "synthetic_segment_word_repair"
        ):
            build_blocks(transcription, episode=12)

    def test_forced_alignment_digest_is_preserved_without_false_repair_risk(self) -> None:
        digest = "a" * 64
        transcription = {
            "words": [
                {
                    "word_index": index,
                    "segment_id": 1,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "probability": 0.99,
                    "timing_source": "whisperx_ctc_forced_alignment",
                    "alignment_sha256": digest,
                }
                for index, (start_ms, end_ms, text) in enumerate(
                    ((0, 380, "Merhaba"), (400, 800, "dünya.")), start=1
                )
            ],
            "vad_regions": [],
        }

        blocks = build_blocks(transcription, episode=12)
        provenance = blocks[0]["vad_info"]["alignment_provenance"]

        self.assertEqual(provenance["alignment_sha256"], digest)
        self.assertEqual(provenance["repaired_word_count"], 0)
        self.assertEqual(
            provenance["word_timing_sources"], ["whisperx_ctc_forced_alignment"]
        )
        self.assertNotIn("repaired_word_provenance", blocks[0]["risk_flags"])

    def test_multiple_alignment_digests_hard_fail(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 0,
                    "end_ms": 380,
                    "text": "Merhaba",
                    "timing_source": "whisperx_ctc_forced_alignment",
                    "alignment_sha256": "a" * 64,
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 400,
                    "end_ms": 800,
                    "text": "dünya.",
                    "timing_source": "whisperx_ctc_forced_alignment",
                    "alignment_sha256": "b" * 64,
                },
            ]
        }

        with self.assertRaisesRegex(SegmentationError, "multiple alignment_sha256"):
            build_blocks(transcription, episode=12)

    def test_negative_and_overlapping_word_intervals_hard_fail(self) -> None:
        negative = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": -1,
                    "end_ms": 200,
                    "text": "Hayır.",
                }
            ]
        }
        with self.assertRaisesRegex(SegmentationError, "Invalid word interval"):
            build_blocks(negative, episode=12)

        overlapping = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 100,
                    "end_ms": 500,
                    "text": "Bunu",
                },
                {
                    "word_index": 2,
                    "segment_id": 1,
                    "start_ms": 450,
                    "end_ms": 800,
                    "text": "istemem.",
                },
            ]
        }
        with self.assertRaisesRegex(SegmentationError, "Overlapping word intervals"):
            build_blocks(overlapping, episode=12)

    def test_short_or_high_cps_final_cue_is_a_hard_validation_error(self) -> None:
        short = [
            {
                "block_uid": "",
                "episode": 12,
                "block_index": 1,
                "start_ms": 1_000,
                "end_ms": 1_019,
                "timing_text": "de",
                "primary_text": "de",
                "verification_text": "",
                "youtube_text": "",
                "context_before": "",
                "context_after": "",
                "vad_info": {"max_internal_gap_ms": 0},
                "risk_flags": [],
            }
        ]
        report = validate_segmentation(short)

        self.assertFalse(report.valid)
        self.assertTrue(any("below required minimum" in error for error in report.errors))
        self.assertTrue(any("source CPS" in error for error in report.errors))

    def test_impossible_reading_speed_fails_instead_of_warning(self) -> None:
        transcription = {
            "words": [
                {
                    "word_index": 1,
                    "segment_id": 1,
                    "start_ms": 0,
                    "end_ms": 100,
                    "text": "x" * 141,
                    "probability": 0.99,
                }
            ],
            "vad_regions": [],
        }

        with self.assertRaisesRegex(SegmentationError, "No safe segmentation"):
            build_blocks(transcription, episode=12)

    def test_segmentation_validator_rejects_mixed_turn_and_internal_gap(self) -> None:
        unsafe = [
            {
                "block_uid": "",
                "episode": 12,
                "block_index": 1,
                "start_ms": 1_000,
                "end_ms": 4_000,
                "timing_text": "-Ben geldim.\n-Sen gittin.",
                "primary_text": "-Ben geldim.\n-Sen gittin.",
                "verification_text": "",
                "youtube_text": "",
                "context_before": "",
                "context_after": "",
                "vad_info": {"max_internal_gap_ms": 1_001},
                "risk_flags": [],
            }
        ]
        report = validate_segmentation(unsafe)

        self.assertFalse(report.valid)
        self.assertEqual(report.mixed_speaker_block_count, 1)
        self.assertEqual(report.unresolved_internal_gap_count, 1)

    def test_11_old_2505_block_schema_cannot_be_reused_for_2470_blocks(self) -> None:
        schema_a = make_schema(2_505, episode=12, schema_version="1.0")
        old_translations = translation_records(schema_a)

        # Schema B keeps 1-11, drops old block 12, then shifts its segmentation.
        old_raw = raw_blocks(2_505, episode=12)
        schema_b_raw = copy.deepcopy(old_raw[:11])
        for new_index in range(12, 2_471):
            donor = copy.deepcopy(old_raw[new_index])  # old block new_index + 1
            donor["block_index"] = new_index
            schema_b_raw.append(donor)
        schema_b = build_episode_schema(schema_b_raw, episode=12, schema_version="1.0")

        self.assertEqual(schema_a["block_count"], 2_505)
        self.assertEqual(schema_b["block_count"], 2_470)
        self.assertEqual(
            [block["block_uid"] for block in schema_a["blocks"][:11]],
            [block["block_uid"] for block in schema_b["blocks"][:11]],
        )
        self.assertNotEqual(
            schema_a["blocks"][11]["block_uid"],
            schema_b["blocks"][11]["block_uid"],
        )

        diagnostic = validate_translation_records(
            schema_b, old_translations, raise_on_error=False
        )
        self.assertFalse(diagnostic.ok)
        self.assertGreater(diagnostic.report.schema_mismatch_count, 0)
        self.assertGreater(diagnostic.report.missing_translation_count, 0)
        self.assertGreater(diagnostic.report.extra_translation_count, 0)
        self.assertEqual(diagnostic.records_by_uid, {})
        with self.assertRaises(TranslationValidationError):
            validate_translation_records(schema_b, old_translations)

    def test_26_translation_batch_from_wrong_episode_hard_fails(self) -> None:
        expected_schema = make_schema(4, episode=12)
        wrong_episode_schema = make_schema(4, episode=13)
        records = translation_records(wrong_episode_schema)

        result = validate_translation_records(
            expected_schema, records, raise_on_error=False
        )
        self.assertGreater(result.report.episode_mismatch_count, 0)
        self.assertGreater(result.report.schema_mismatch_count, 0)
        self.assertEqual(result.records_by_uid, {})
        with self.assertRaises(TranslationValidationError):
            validate_translation_records(expected_schema, records)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "wrong-episode.zip"
            write_translated_zip(archive, wrong_episode_schema, records)
            with self.assertRaises(TranslationValidationError):
                load_and_validate_translated_zip(expected_schema, archive)

    def test_27_translation_batch_from_wrong_schema_version_hard_fails(self) -> None:
        schema_v1 = make_schema(4, episode=12, schema_version="1.0")
        schema_v2 = make_schema(4, episode=12, schema_version="2.0")
        records_v2 = translation_records(schema_v2)

        result = validate_translation_records(
            schema_v1, records_v2, raise_on_error=False
        )
        self.assertGreater(result.report.schema_mismatch_count, 0)
        self.assertGreater(result.report.missing_translation_count, 0)
        self.assertGreater(result.report.extra_translation_count, 0)
        self.assertEqual(result.records_by_uid, {})
        with self.assertRaises(TranslationValidationError):
            validate_translation_records(schema_v1, records_v2)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "wrong-schema-version.zip"
            write_translated_zip(archive, schema_v2, records_v2)
            with self.assertRaises(TranslationValidationError):
                load_and_validate_translated_zip(schema_v1, archive)


if __name__ == "__main__":
    unittest.main()
