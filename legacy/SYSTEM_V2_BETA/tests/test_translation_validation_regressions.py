from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from helpers import raw_blocks, write_translated_zip
from src.batches import create_translation_pack
from src.schema import build_episode_schema
from src.translation_validation import (
    DEFAULT_SOURCE_NAME_VARIANTS,
    load_and_validate_translated_zip,
    validate_translation_records,
)


def _schema_with_candidates(
    rows: Sequence[Mapping[str, str]],
) -> dict:
    raw = raw_blocks(len(rows))
    for block, row in zip(raw, rows):
        for field in (
            "timing_text",
            "primary_text",
            "verification_text",
            "youtube_text",
        ):
            block[field] = row.get(field, row["primary_text"])
    return build_episode_schema(raw, episode=12)


def _records(
    schema: Mapping,
    translations: Sequence[tuple[str, str]],
) -> list[dict]:
    return [
        {
            "block_uid": block["block_uid"],
            "schema_sha256": schema["schema_sha256"],
            "tr_final": tr_final,
            "id_final": id_final,
            "review_required": False,
            "note": "",
        }
        for block, (tr_final, id_final) in zip(schema["blocks"], translations)
    ]


class TranslationValidationRegressionTests(unittest.TestCase):
    def test_default_source_variants_match_names_config(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "names.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        configured = {
            canonical: tuple(variants)
            for canonical, variants in config["source_variants"].items()
        }

        self.assertEqual(configured, DEFAULT_SOURCE_NAME_VARIANTS)

    def test_id_final_shift_is_detected_from_preserved_anchors(self) -> None:
        sources = (
            "Defne 11 kutu aldı.",
            "Kadir 22 kutu aldı.",
            "Tolga 33 kutu aldı.",
        )
        schema = _schema_with_candidates(
            [{"primary_text": source} for source in sources]
        )
        indonesian = (
            "Defne membeli 11 kotak.",
            "Kadir membeli 22 kotak.",
            "Tolga membeli 33 kotak.",
        )
        shifted = indonesian[1:] + indonesian[:1]
        records = _records(schema, list(zip(sources, shifted)))

        result = validate_translation_records(
            schema,
            records,
            raise_on_error=False,
        )

        self.assertGreaterEqual(
            result.report.positional_translation_mismatch_count,
            3,
        )
        id_issues = [
            issue
            for issue in result.report.errors
            if issue.code == "positional_translation_mismatch"
            and isinstance(issue.actual, dict)
            and issue.actual.get("field") == "id_final"
        ]
        self.assertEqual(len(id_issues), 3)

    def test_anchor_free_id_final_is_not_pseudo_aligned_cross_lingually(self) -> None:
        sources = (
            "Bugün eve dönüyorum.",
            "Kapıyı sessizce kapattı.",
            "Yarın erken kalkacağız.",
        )
        schema = _schema_with_candidates(
            [{"primary_text": source} for source in sources]
        )
        indonesian = (
            "Aku pulang hari ini.",
            "Dia menutup pintu dengan pelan.",
            "Besok kami akan bangun pagi.",
        )
        shifted = indonesian[1:] + indonesian[:1]
        records = _records(schema, list(zip(sources, shifted)))

        result = validate_translation_records(schema, records)

        self.assertTrue(result.ok)
        self.assertEqual(result.report.positional_translation_mismatch_count, 0)

    def test_isolated_remote_exact_text_is_not_position_evidence(self) -> None:
        schema = _schema_with_candidates(
            [
                {"primary_text": "Selom."},
                {"primary_text": "Bugün eve geldim."},
                {"primary_text": "Yarın işe gideceğim."},
                {"primary_text": "Selam."},
            ]
        )
        records = _records(
            schema,
            [
                ("Selam.", "Hai."),
                ("Bugün eve geldim.", "Hari ini aku pulang."),
                ("Yarın işe gideceğim.", "Besok aku akan bekerja."),
                ("Selam.", "Hai."),
            ],
        )

        result = validate_translation_records(
            schema,
            records,
            semantic_window=1,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.report.positional_translation_mismatch_count, 0)

    def test_invented_quantity_and_known_name_are_rejected(self) -> None:
        source = "Bugün eve gidiyoruz."
        schema = _schema_with_candidates([{"primary_text": source}])
        records = _records(
            schema,
            [
                (
                    "Bugün eve Kadir ile gidiyoruz 99.",
                    "Hari ini kami pulang bersama Kadir 99.",
                )
            ],
        )

        result = validate_translation_records(
            schema,
            records,
            raise_on_error=False,
        )

        self.assertEqual(result.report.numeric_mismatch_count, 1)
        self.assertEqual(result.report.special_name_mismatch_count, 1)

    def test_conflicting_name_candidates_allow_one_supported_correction(self) -> None:
        schema = _schema_with_candidates(
            [
                {
                    "primary_text": "Selim bugün geldi.",
                    "verification_text": "Selma bugün geldi.",
                    "youtube_text": "Selim bugün geldi.",
                }
            ]
        )
        for selected in ("Selim bugün geldi.", "Selma bugün geldi."):
            with self.subTest(selected=selected):
                name = selected.split()[0]
                records = _records(
                    schema,
                    [(selected, f"{name} datang hari ini.")],
                )
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_conflicting_religious_candidates_do_not_impose_their_union(self) -> None:
        schema = _schema_with_candidates(
            [
                {
                    "primary_text": "İnşallah bugün gelir.",
                    "verification_text": "Bugün gelir.",
                    "youtube_text": "Bugün gelir.",
                }
            ]
        )
        cases = (
            ("Bugün gelir.", "Dia datang hari ini."),
            ("İnşallah bugün gelir.", "Insyaallah dia datang hari ini."),
        )
        for translation in cases:
            with self.subTest(translation=translation):
                records = _records(schema, [translation])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_supported_correction_is_not_rejected_for_matching_other_block(self) -> None:
        schema = _schema_with_candidates(
            [
                {
                    "primary_text": "Ben gerçekten çok iyiyim.",
                    "verification_text": "Ben iyiyim artık.",
                    "youtube_text": "Ben gerçekten çok iyiyim.",
                },
                {"primary_text": "Ben iyiyim."},
            ]
        )
        records = _records(
            schema,
            [
                ("Ben iyiyim.", "Saya baik."),
                ("Ben iyiyim.", "Saya baik."),
            ],
        )

        result = validate_translation_records(schema, records)

        self.assertTrue(result.ok)
        self.assertEqual(result.report.positional_translation_mismatch_count, 0)

    def test_numeric_tokens_preserve_decimal_and_whitespace_boundaries(self) -> None:
        cases = (
            ("Bu miktar 1,20 TL.", "Bu miktar 120 TL.", "Jumlahnya 120 TL."),
            ("Kod 1 2 olarak yazıldı.", "Kod 12 olarak yazıldı.", "Kodenya 12."),
        )
        for source, tr_final, id_final in cases:
            with self.subTest(source=source):
                schema = _schema_with_candidates([{"primary_text": source}])
                records = _records(schema, [(tr_final, id_final)])
                result = validate_translation_records(
                    schema,
                    records,
                    raise_on_error=False,
                )
                self.assertEqual(result.report.numeric_mismatch_count, 1)

    def test_thousands_separator_formatting_can_change_without_changing_value(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Toplam 1.250 TL ödedi."}]
        )
        records = _records(
            schema,
            [("Toplam 1250 TL ödedi.", "Dia membayar total 1250 TL.")],
        )

        self.assertTrue(validate_translation_records(schema, records).ok)

    def test_known_asr_name_variant_can_be_corrected_to_canonical_spelling(self) -> None:
        cases = (
            (
                "Emin Dağ bugün aradı.",
                "Emindağ bugün aradı.",
                "Emindağ menelepon hari ini.",
            ),
            (
                "Levent Bartiner bugün aradı.",
                "Levent Bartıner bugün aradı.",
                "Levent Bartıner menelepon hari ini.",
            ),
        )
        for source, tr_final, id_final in cases:
            with self.subTest(source=source):
                schema = _schema_with_candidates([{"primary_text": source}])
                records = _records(schema, [(tr_final, id_final)])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_source_name_variant_supports_canonical_final_name(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Selam bugün geldi."}]
        )
        records = _records(
            schema,
            [("Selim bugün geldi.", "Selim datang hari ini.")],
        )

        result = validate_translation_records(
            schema,
            records,
            source_name_variants={"Selim": ("Selam",)},
        )

        self.assertTrue(result.ok)
        self.assertIn("Selam", DEFAULT_SOURCE_NAME_VARIANTS["Selim"])

    def test_source_name_variant_correction_is_not_misaligned_to_exact_name(self) -> None:
        schema = _schema_with_candidates(
            [
                {"primary_text": "Selam."},
                {"primary_text": "Selim."},
            ]
        )
        records = _records(
            schema,
            [
                ("Selim.", "Selim."),
                ("Selim.", "Selim."),
            ],
        )

        result = validate_translation_records(schema, records)

        self.assertTrue(result.ok)
        self.assertEqual(result.report.positional_translation_mismatch_count, 0)

    def test_source_name_variant_is_not_automatically_forbidden_in_final(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Herkese selam."}]
        )
        records = _records(
            schema,
            [("Herkese selam.", "Salam untuk semuanya.")],
        )

        result = validate_translation_records(
            schema,
            records,
            source_name_variants={"Selim": ("Selam",)},
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.report.special_name_mismatch_count, 0)

    def test_ambiguous_source_variant_allows_literal_and_name_readings(self) -> None:
        cases = (
            (
                "Gelirse beni ara.",
                "Gelirse beni ara.",
                "Kalau dia datang, telepon aku.",
            ),
            (
                "Gelirse aradım.",
                "Melis'i aradım.",
                "Aku menelepon Melis.",
            ),
        )
        for source, tr_final, id_final in cases:
            with self.subTest(tr_final=tr_final):
                schema = _schema_with_candidates([{"primary_text": source}])
                records = _records(schema, [(tr_final, id_final)])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_source_variants_support_natural_turkish_address_forms(self) -> None:
        cases = (
            ("Defneciğim, dinle.", "Defneciğim, dinle.", "Defne, dengarkan."),
            ("Tolgacım, dinle.", "Tolgacığım, dinle.", "Tolga, dengarkan."),
        )
        for source, tr_final, id_final in cases:
            with self.subTest(tr_final=tr_final):
                schema = _schema_with_candidates([{"primary_text": source}])
                records = _records(schema, [(tr_final, id_final)])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_turkish_name_suffixes_preserve_canonical_identity(self) -> None:
        cases = (
            ("Defneyle gel.", "Defne'yle gel.", "Datang bersama Defne."),
            ("Özlemciğim, dinle.", "Özlemciğim, dinle.", "Özlem, dengarkan."),
            ("Selimciğim, dinle.", "Selimciğim, dinle.", "Selim, dengarkan."),
            ("Kaderle gel.", "Kadir'le gel.", "Datang bersama Kadir."),
            ("Emindağlar geldi.", "Emindağlar geldi.", "Keluarga Emindağ datang."),
        )
        for source, tr_final, id_final in cases:
            with self.subTest(source=source):
                schema = _schema_with_candidates([{"primary_text": source}])
                records = _records(schema, [(tr_final, id_final)])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_reviewed_source_ambiguity_passes_but_tr_id_name_parity_stays_strict(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Beni anlamıyor."}]
        )
        records = _records(
            schema,
            [("Mine Hanım.", "Bu Mine.")],
        )
        records[0]["review_required"] = True
        records[0]["note"] = "Ses belirsiz; sahne bağlamı Mine Hanım'ı destekliyor."

        self.assertTrue(validate_translation_records(schema, records).ok)

        records[0]["id_final"] = "Pak Kadir."
        rejected = validate_translation_records(
            schema,
            records,
            raise_on_error=False,
        )
        self.assertEqual(rejected.report.special_name_mismatch_count, 1)

    def test_review_flag_without_note_does_not_bypass_source_evidence(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Beni anlamıyor."}]
        )
        records = _records(schema, [("Mine Hanım.", "Bu Mine.")])
        records[0]["review_required"] = True

        result = validate_translation_records(
            schema,
            records,
            raise_on_error=False,
        )

        self.assertEqual(result.report.special_name_mismatch_count, 1)

    def test_overlapping_source_variant_preserves_literal_name_reading(self) -> None:
        cases = (
            ("Kadir Emin geldi.", "Kadir Emin datang."),
            ("Kadir Emindağ geldi.", "Kadir Emindağ datang."),
            ("Emindağ geldi.", "Emindağ datang."),
        )
        for tr_final, id_final in cases:
            with self.subTest(tr_final=tr_final):
                schema = _schema_with_candidates(
                    [{"primary_text": "Kadir Emin geldi."}]
                )
                records = _records(schema, [(tr_final, id_final)])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_explicit_empty_source_variants_do_not_use_defaults(self) -> None:
        schema = _schema_with_candidates([{"primary_text": "Selam geldi."}])
        records = _records(
            schema,
            [("Selim geldi.", "Selim datang.")],
        )

        result = validate_translation_records(
            schema,
            records,
            source_name_variants={},
            raise_on_error=False,
        )

        self.assertFalse(result.ok)
        self.assertGreater(result.report.special_name_mismatch_count, 0)

    def test_zip_loader_forwards_source_name_variants(self) -> None:
        schema = _schema_with_candidates([{"primary_text": "Selam geldi."}])
        records = _records(
            schema,
            [("Selim geldi.", "Selim datang.")],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack_path = root / "Muhtemel Ask 12.Bolum_TRANSLATION_PACK.zip"
            translated_path = root / "Muhtemel Ask 12.Bolum_TRANSLATED.zip"
            manifest = create_translation_pack(schema, pack_path)
            write_translated_zip(translated_path, schema, records)

            result = load_and_validate_translated_zip(
                schema,
                translated_path,
                input_manifest=manifest,
                source_name_variants={"Selim": ("Selam",)},
            )

        self.assertTrue(result.ok)

    def test_bartiner_remains_forbidden_in_final_output(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Levent Bartiner bugün aradı."}]
        )
        records = _records(
            schema,
            [
                (
                    "Levent Bartiner bugün aradı.",
                    "Levent Bartiner menelepon hari ini.",
                )
            ],
        )

        result = validate_translation_records(
            schema,
            records,
            raise_on_error=False,
        )

        name_issue = next(
            issue
            for issue in result.report.errors
            if issue.code == "special_name_mismatch"
        )
        self.assertIn(
            {"canonical": "Bartıner", "forbidden_variant": "Bartiner"},
            name_issue.actual["forbidden_variants"],
        )

    def test_apostrophe_and_joined_religious_source_forms_are_equivalent(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Allahım yarabbim."}]
        )
        records = _records(
            schema,
            [("Allah'ım, Ya Rabbim.", "Ya Allah, Ya Rabb.")],
        )

        result = validate_translation_records(schema, records)

        self.assertTrue(result.ok)

    def test_apostrophized_rabbim_and_vallahi_oath_align(self) -> None:
        cases = (
            (
                "Allah 'ım yarabbim.",
                "Allah'ım ya Rabb'im.",
                "Ya Allah, ya Rabb.",
            ),
            (
                "Vallahi bin pişmanım.",
                "Vallahi bin pişmanım.",
                "Demi Allah, aku benar-benar menyesal.",
            ),
        )
        for source, tr_final, id_final in cases:
            with self.subTest(source=source):
                schema = _schema_with_candidates([{"primary_text": source}])
                records = _records(schema, [(tr_final, id_final)])
                self.assertTrue(validate_translation_records(schema, records).ok)

    def test_religious_source_normalization_keeps_id_mapping_strict(self) -> None:
        schema = _schema_with_candidates(
            [{"primary_text": "Allahım yarabbim."}]
        )
        records = _records(
            schema,
            [("Allah'ım, Ya Rabbim.", "Ya Allah.")],
        )

        result = validate_translation_records(
            schema,
            records,
            raise_on_error=False,
        )

        issue_codes = [issue.code for issue in result.report.errors]
        self.assertIn("religious_expression_mismatch", issue_codes)
        self.assertNotIn("religious_source_evidence_mismatch", issue_codes)


if __name__ == "__main__":
    unittest.main()
