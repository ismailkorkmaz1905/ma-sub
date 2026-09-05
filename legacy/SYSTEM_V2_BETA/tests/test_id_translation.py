from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from src.id_translation import (
    ID_TRANSLATION_INSTRUCTIONS,
    IDTranslationError,
    build_id_translation_records,
    create_id_translation_output_zip,
    create_id_translation_pack,
    load_and_validate_id_translation_zip,
    validate_aligned_turkish_schema,
    validate_id_translation_glossary,
    validate_id_translation_pack,
    validate_id_translation_records,
)


def _glossary() -> dict:
    return {
        "canonical_names": ["Defne", "Kadir"],
        "forbidden_name_variants": {"Defne": ["Defnee"]},
        "source_name_variants": {"Kadir": ["Kader"]},
        "religious_terms": [
            {
                "source": "Allah aşkına",
                "preferred_indonesian": ["Demi Allah"],
                "must_preserve_allah": True,
            },
            {
                "source": "İnşallah",
                "preferred_indonesian": ["Insyaallah"],
            },
            {
                "source": "Maşallah",
                "preferred_indonesian": ["Masyaallah"],
            },
        ],
        "allah_only_semoga_is_invalid": True,
    }


def _schema(block_count: int = 5) -> dict:
    blocks = []
    for offset in range(block_count):
        index = offset + 1
        start_ms = 1_000 + offset * 2_000
        blocks.append(
            {
                "block_uid": f"MA12-V2-{index:06d}-abcdef{index:02d}",
                "block_index": index,
                "start_ms": start_ms,
                "end_ms": start_ms + 1_250,
                "tr_text": f"Bu düzeltilmiş Türkçe cümle {index}.",
                "alignment_provenance": {
                    "timing_source": "whisperx_forced_alignment",
                    "alignment_model": "tr-test-model",
                    "whisperx_version": "test",
                    "first_word_start_ms": start_ms + 30,
                    "last_word_end_ms": start_ms + 1_180,
                },
            }
        )
    return {
        "schema_version": "2.0",
        "episode": 12,
        "block_count": block_count,
        "pipeline": "correct-tr-then-align-then-id",
        "blocks": blocks,
    }


def _translated_records(schema: dict) -> list[dict]:
    records = build_id_translation_records(schema)
    for position, record in enumerate(records, start=1):
        record["id_final"] = f"Kalimat bahasa Indonesia {position}."
        record["review_required"] = position == 2
        record["note"] = "Periksa sapaan." if position == 2 else ""
    return records


class IDTranslationPackTests(unittest.TestCase):
    def test_pack_is_deterministic_and_hashes_every_payload(self) -> None:
        schema = _schema(5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.zip"
            second = root / "second.zip"
            manifest_a = create_id_translation_pack(
                schema,
                first,
                batch_size=2,
                glossary=_glossary(),
            )
            manifest_b = create_id_translation_pack(
                schema,
                second,
                batch_size=2,
                glossary=_glossary(),
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["batch_count"], 3)
            self.assertEqual(
                manifest_a,
                validate_id_translation_pack(
                    first,
                    expected_schema=schema,
                    expected_glossary=_glossary(),
                ),
            )
            with zipfile.ZipFile(first, "r") as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "manifest.json",
                        "schema.json",
                        "glossary.json",
                        "ID_TRANSLATION_INSTRUCTIONS.md",
                        "batch_001.jsonl",
                        "batch_002.jsonl",
                        "batch_003.jsonl",
                    ],
                )
                for name, digest in manifest_a["file_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                first_record = json.loads(
                    archive.read("batch_001.jsonl").decode("utf-8").splitlines()[0]
                )
                self.assertNotIn("id_final", first_record)
                self.assertEqual(first_record["tr_text"], schema["blocks"][0]["tr_text"])
                self.assertIn("alignment_provenance", first_record)
                self.assertEqual(
                    json.loads(archive.read("glossary.json")),
                    validate_id_translation_glossary(_glossary()),
                )
                instructions = archive.read(
                    "ID_TRANSLATION_INSTRUCTIONS.md"
                ).decode("utf-8")
                self.assertEqual(instructions, ID_TRANSLATION_INSTRUCTIONS)
                for required_rule in (
                    "natural, conversational Indonesian",
                    "not literal or",
                    "`aku`,",
                    "`saya`, `Anda`, `Pak` and `Bu`",
                    "Preserve romance, anger, sarcasm, comedy",
                    "Do not censor, soften, explain",
                    "canonical spelling",
                    "every repeated",
                    "literal word Allah",
                    "preferred_indonesian",
                ):
                    self.assertIn(required_rule, instructions)

    def test_glossary_is_required_hash_bound_and_expected_exactly(self) -> None:
        schema = _schema(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.zip"
            missing = root / "missing-glossary.zip"
            tampered = root / "tampered-glossary.zip"
            create_id_translation_pack(
                schema,
                valid,
                glossary=_glossary(),
            )
            with zipfile.ZipFile(valid, "r") as archive:
                payloads = {
                    info.filename: archive.read(info.filename)
                    for info in archive.infolist()
                }
            with zipfile.ZipFile(missing, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, payload in payloads.items():
                    if name != "glossary.json":
                        archive.writestr(name, payload)
            with self.assertRaisesRegex(IDTranslationError, "missing required"):
                validate_id_translation_pack(missing, expected_schema=schema)

            changed = _glossary()
            changed["canonical_names"].append("Tolga")
            payloads["glossary.json"] = (
                json.dumps(changed, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            ).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, payload in payloads.items():
                    archive.writestr(name, payload)
            with self.assertRaisesRegex(IDTranslationError, "file_sha256"):
                validate_id_translation_pack(tampered, expected_schema=schema)

            expected_other = _glossary()
            expected_other["canonical_names"].append("Tolga")
            with self.assertRaisesRegex(IDTranslationError, "differs from expected"):
                validate_id_translation_pack(
                    valid,
                    expected_schema=schema,
                    expected_glossary=expected_other,
                )

    def test_tampered_pack_hash_is_rejected(self) -> None:
        schema = _schema(2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pack.zip"
            create_id_translation_pack(schema, path, batch_size=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "a") as archive:
                    # Duplicate names are forbidden, independent of CRC.
                    archive.writestr("batch_001.jsonl", b"{}\n")
            with self.assertRaises(IDTranslationError):
                validate_id_translation_pack(path, expected_schema=schema)

    def test_schema_requires_v2_alignment_provenance_and_valid_digest(self) -> None:
        schema = _schema(2)
        del schema["blocks"][0]["alignment_provenance"]
        with self.assertRaises(IDTranslationError):
            validate_aligned_turkish_schema(schema)

        trusted = validate_aligned_turkish_schema(_schema(2))
        trusted["blocks"][0]["tr_text"] = "Sonradan değiştirildi."
        with self.assertRaises(IDTranslationError):
            validate_aligned_turkish_schema(trusted)


class IDTranslationOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = _schema(5)
        self.records = _translated_records(self.schema)

    def _diagnose(self, records: list[dict]):
        return validate_id_translation_records(
            self.schema,
            records,
            raise_on_error=False,
        )

    def test_valid_id_only_records_are_exposed_in_schema_order(self) -> None:
        result = validate_id_translation_records(self.schema, self.records)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.records_by_uid), 5)
        self.assertEqual(result.ordered_records(self.schema), self.records)
        with self.assertRaises(TypeError):
            result.records_by_uid[self.records[0]["block_uid"]]["id_final"] = "X"

    def test_optional_review_fields_are_normalized(self) -> None:
        records = build_id_translation_records(self.schema)
        for record in records:
            record["id_final"] = "Terjemahan yang tidak kosong."
        result = validate_id_translation_records(self.schema, records)
        ordered = result.ordered_records(self.schema)
        self.assertTrue(all(record["review_required"] is False for record in ordered))
        self.assertTrue(all(record["note"] == "" for record in ordered))

    def test_changed_timing_turkish_or_provenance_hard_fails(self) -> None:
        mutations = (
            ("start_ms", 999_999),
            ("end_ms", 999_999),
            ("tr_text", "Yanlış Türkçe"),
            ("alignment_provenance", {"timing_source": "guessed"}),
        )
        for field_name, value in mutations:
            with self.subTest(field=field_name):
                records = copy.deepcopy(self.records)
                records[1][field_name] = value
                result = self._diagnose(records)
                self.assertFalse(result.ok)
                self.assertEqual(result.records_by_uid, {})
                self.assertTrue(
                    any(field_name in issue for issue in result.issues),
                    result.issues,
                )
                with self.assertRaises(IDTranslationError):
                    validate_id_translation_records(self.schema, records)

    def test_missing_duplicate_extra_and_reorder_hard_fail(self) -> None:
        cases = []
        missing = copy.deepcopy(self.records[:-1])
        cases.append(missing)
        duplicate = copy.deepcopy(self.records)
        duplicate.append(copy.deepcopy(duplicate[0]))
        cases.append(duplicate)
        extra = copy.deepcopy(self.records)
        extra[-1]["block_uid"] = "EXTRA-UID"
        cases.append(extra)
        reordered = copy.deepcopy(self.records)
        reordered[1], reordered[2] = reordered[2], reordered[1]
        cases.append(reordered)
        for records in cases:
            with self.subTest(record_count=len(records)):
                result = self._diagnose(records)
                self.assertFalse(result.ok)
                self.assertEqual(result.records_by_uid, {})

    def test_empty_id_and_any_unlisted_output_field_hard_fail(self) -> None:
        empty = copy.deepcopy(self.records)
        empty[0]["id_final"] = "  "
        self.assertFalse(self._diagnose(empty).ok)

        extra = copy.deepcopy(self.records)
        extra[0]["tr_final"] = extra[0]["tr_text"]
        result = self._diagnose(extra)
        self.assertFalse(result.ok)
        self.assertTrue(any("forbidden fields" in issue for issue in result.issues))

    def test_output_zip_round_trip_and_report_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "pack.zip"
            manifest = create_id_translation_pack(self.schema, pack, batch_size=2)
            first = root / "translated-a.zip"
            second = root / "translated-b.zip"
            create_id_translation_output_zip(
                self.schema,
                self.records,
                first,
                input_manifest=manifest,
            )
            create_id_translation_output_zip(
                self.schema,
                self.records,
                second,
                input_manifest=manifest,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            result = load_and_validate_id_translation_zip(
                self.schema,
                first,
                input_manifest=manifest,
            )
            self.assertTrue(result.ok)
            with zipfile.ZipFile(first, "r") as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "translated_batch_001.jsonl",
                        "translated_batch_002.jsonl",
                        "translated_batch_003.jsonl",
                        "translation_report.json",
                    ],
                )

    def test_output_zip_rejects_wrong_batch_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "pack.zip"
            manifest = create_id_translation_pack(self.schema, pack, batch_size=2)
            translated = root / "translated.zip"
            create_id_translation_output_zip(
                self.schema,
                self.records,
                translated,
                input_manifest=manifest,
            )
            with zipfile.ZipFile(translated, "a") as archive:
                archive.writestr("unexpected.json", b"{}\n")
            with self.assertRaises(IDTranslationError):
                load_and_validate_id_translation_zip(
                    self.schema,
                    translated,
                    input_manifest=manifest,
                )


if __name__ == "__main__":
    unittest.main()
