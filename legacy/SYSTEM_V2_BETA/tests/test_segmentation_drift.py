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
from src.segment import build_blocks, validate_segmentation
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
                    "start_ms": 1_400,
                    "end_ms": 1_700,
                    "text": "-Ben",
                    "probability": 0.99,
                },
                {
                    "word_index": 3,
                    "segment_id": 1,
                    "start_ms": 1_750,
                    "end_ms": 2_050,
                    "text": "geldim.",
                    "probability": 0.99,
                },
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 950,
                    "end_ms": 2_100,
                    "source": "synthetic-test",
                }
            ],
        }
        blocks = build_blocks(transcription, episode=12)

        self.assertEqual([block["primary_text"] for block in blocks], ["Merhaba.", "-Ben geldim."])
        self.assertNotIn("-Ben", blocks[0]["primary_text"])
        self.assertIn("speaker_change_boundary_before", blocks[1]["risk_flags"])

    def test_short_whisper_segment_change_is_a_neutral_hard_boundary(self) -> None:
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
                        (1, 0, 300, "Ben"),
                        (1, 320, 600, "bunu"),
                        (1, 620, 1_050, "istemiyorum"),
                        (2, 1_200, 1_450, "Ama"),
                        (2, 1_470, 1_730, "neden"),
                        (2, 1_750, 2_050, "böyle?"),
                    ),
                    start=1,
                )
            ],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 2_100,
                    "source": "synthetic-test",
                }
            ],
        }

        blocks = build_blocks(transcription, episode=12)

        self.assertEqual(
            [block["primary_text"] for block in blocks],
            ["Ben bunu istemiyorum", "Ama neden böyle?"],
        )
        self.assertIn(
            "whisper_segment_boundary", blocks[0]["vad_info"]["boundary_after"]
        )
        self.assertIn(
            "whisper_segment_boundary", blocks[1]["vad_info"]["boundary_before"]
        )
        self.assertNotIn("possible_speaker_change", blocks[0]["risk_flags"])
        self.assertNotIn("possible_speaker_change", blocks[1]["risk_flags"])

    def test_synthetic_timing_words_are_not_split_into_impossible_cues(self) -> None:
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

        blocks = build_blocks(transcription, episode=12)
        report = validate_segmentation(blocks, transcription["words"])

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["primary_text"], "Sizi istiyor belki de beyim.")
        self.assertLess(blocks[0]["start_ms"], blocks[0]["end_ms"])

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

