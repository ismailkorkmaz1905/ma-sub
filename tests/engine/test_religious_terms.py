from __future__ import annotations

import unittest

from helpers import raw_blocks
from mas.engine.schema import build_episode_schema
from mas.engine.subtitle_qa import SubtitleQAError, assert_final_qa, run_subtitle_qa
from mas.engine.translation_validation import validate_translation_records


def _single_block_case(tr_text: str, id_text: str) -> dict:
    raw = raw_blocks(1)
    for field in ("timing_text", "primary_text", "verification_text", "youtube_text"):
        raw[0][field] = tr_text
    schema = build_episode_schema(raw, episode=12)
    block = schema["blocks"][0]
    record = {
        "block_uid": block["block_uid"],
        "schema_sha256": schema["schema_sha256"],
        "tr_final": tr_text,
        "id_final": id_text,
        "review_required": False,
        "note": "",
    }
    return run_subtitle_qa(schema["blocks"], [record])


class ReligiousAndAnchorTests(unittest.TestCase):
    def test_15_required_religious_phrase_preservation(self) -> None:
        cases = (
            ("Allah aşkına", "Demi Allah"),
            ("Allah Allah", "Ya Allah"),
            ("Allah'ım", "Ya Allah"),
            ("Ya Rabbim", "Ya Rabb"),
            ("İnşallah", "Insyaallah"),
            ("Maşallah", "Masyaallah"),
            ("Estağfurullah", "Astagfirullah"),
            ("Tövbe estağfurullah", "Tobat, astagfirullah"),
            (
                "La havle vela kuvvete illa billah",
                "La hawla wala quwwata illa billah",
            ),
        )
        for turkish, indonesian in cases:
            with self.subTest(turkish=turkish):
                report = _single_block_case(turkish, indonesian)
                self.assertEqual(report["religious_expression_mismatch_count"], 0)
                self.assertEqual(report["allah_preservation_failure_count"], 0)
                assert_final_qa(report)

    def test_16_allah_cannot_be_replaced_only_with_semoga(self) -> None:
        report = _single_block_case(
            "Allah aşkına bize yardım et.",
            "Semoga kamu membantu kami.",
        )

        self.assertEqual(report["allah_preservation_failure_count"], 1)
        self.assertTrue(
            any(issue["code"] == "allah_replaced_by_semoga" for issue in report["issues"])
        )

    def test_17_special_name_spelling_is_enforced(self) -> None:
        report = _single_block_case(
            "Emindağ, Levent Bartıner'i aradı.",
            "Emin Dağ menelepon Levent Bartiner.",
        )

        self.assertGreater(report["special_name_mismatch_count"], 0)

    def test_verification_or_context_name_is_not_a_current_block_anchor(self) -> None:
        raw = raw_blocks(1)
        raw[0].update(
            {
                "timing_text": "Şu odayı eski hâline getirin.",
                "primary_text": "Şu odayı eski hâline getirin.",
                "verification_text": "Selim, şu odayı eski hâline getirin.",
                "youtube_text": "",
                "context_before": "Ne diyeceğim Selim,",
                "context_after": "Tamam, hallederim.",
            }
        )
        schema = build_episode_schema(raw, episode=12)
        block = schema["blocks"][0]
        record = {
            "block_uid": block["block_uid"],
            "schema_sha256": schema["schema_sha256"],
            "tr_final": "Şu odayı eski hâline getirin.",
            "id_final": "Tolong kembalikan kamar ini seperti semula.",
            "review_required": False,
            "note": "",
        }

        report = run_subtitle_qa(schema["blocks"], [record])

        self.assertEqual(report["special_name_mismatch_count"], 0)
        assert_final_qa(report)

    def test_same_block_primary_name_remains_a_required_anchor(self) -> None:
        raw = raw_blocks(1)
        raw[0].update(
            {
                "timing_text": "Selim, şu odayı eski hâline getir.",
                "primary_text": "Selim, şu odayı eski hâline getir.",
                "verification_text": "",
                "youtube_text": "",
                "context_before": "",
                "context_after": "",
            }
        )
        schema = build_episode_schema(raw, episode=12)
        block = schema["blocks"][0]
        record = {
            "block_uid": block["block_uid"],
            "schema_sha256": schema["schema_sha256"],
            "tr_final": "Şu odayı eski hâline getir.",
            "id_final": "Kembalikan kamar ini seperti semula.",
            "review_required": False,
            "note": "",
        }

        report = run_subtitle_qa(schema["blocks"], [record])

        self.assertEqual(report["special_name_mismatch_count"], 1)
        self.assertFalse(report["passed"])
        with self.assertRaises(SubtitleQAError):
            assert_final_qa(report)

    def test_review_flag_alone_cannot_bypass_primary_name_anchor(self) -> None:
        raw = raw_blocks(1)
        raw[0].update(
            {
                "timing_text": "Zeyno.",
                "primary_text": "Zeyno.",
                "verification_text": "",
                "youtube_text": "",
            }
        )
        schema = build_episode_schema(raw, episode=12)
        block = schema["blocks"][0]
        record = {
            "block_uid": block["block_uid"],
            "schema_sha256": schema["schema_sha256"],
            "tr_final": "Zeliha.",
            "id_final": "Zeliha.",
            "review_required": True,
            "note": "Konuşmacı bağlamı Zeliha'yı destekliyor; ham ASR Zeyno diyor.",
        }

        report = run_subtitle_qa(schema["blocks"], [record])

        self.assertEqual(report["special_name_mismatch_count"], 1)
        self.assertFalse(report["passed"])

    def test_trusted_reviewed_source_name_correction_passes_final_qa(self) -> None:
        raw = raw_blocks(1)
        raw[0].update(
            {
                "timing_text": "Zeyno.",
                "primary_text": "Zeyno.",
                "verification_text": "",
                "youtube_text": "",
            }
        )
        schema = build_episode_schema(raw, episode=12)
        block = schema["blocks"][0]
        record = {
            "block_uid": block["block_uid"],
            "schema_sha256": schema["schema_sha256"],
            "tr_final": "Zeliha.",
            "id_final": "Zeliha.",
            "review_required": True,
            "note": "Konuşmacı bağlamı Zeliha'yı destekliyor; ham ASR Zeyno diyor.",
        }
        translation_report = {
            "schema_sha256": schema["schema_sha256"],
            "total_input_blocks": 1,
            "total_output_blocks": 1,
            "missing_block_count": 0,
            "duplicate_block_count": 0,
            "review_required_count": 1,
        }
        validated = validate_translation_records(
            schema,
            [record],
            translation_report=translation_report,
        )

        report = run_subtitle_qa(schema["blocks"], validated)

        self.assertEqual(report["special_name_mismatch_count"], 0)
        self.assertEqual(report["anchor_mismatch_count"], 0)
        assert_final_qa(report)

    def test_18_numeric_and_money_values_cannot_change(self) -> None:
        report = _single_block_case(
            "Defne 25 kutu için 1.250 TL ödedi.",
            "Defne membayar 1.500 TL untuk 24 kotak.",
        )

        self.assertEqual(report["numeric_mismatch_count"], 1)
        self.assertEqual(report["money_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()

