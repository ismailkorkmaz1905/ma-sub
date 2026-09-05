from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from helpers import make_schema, translation_records, write_translated_zip
from src.batches import BatchError, create_translation_pack, validate_translation_pack
from src.translation_validation import (
    MAX_TRANSLATED_ZIP_MEMBER_BYTES,
    TranslationValidationError,
    load_and_validate_translated_zip,
)


def _patch_central_directory_sizes(
    path: Path,
    member: str,
    *,
    uncompressed_size: int,
) -> None:
    """Alter only central-directory metadata; never allocate the declared size."""

    payload = bytearray(path.read_bytes())
    signature = b"PK\x01\x02"
    offset = 0
    while True:
        offset = payload.find(signature, offset)
        if offset < 0:
            raise AssertionError(f"Central-directory member not found: {member}")
        name_length = struct.unpack_from("<H", payload, offset + 28)[0]
        extra_length = struct.unpack_from("<H", payload, offset + 30)[0]
        comment_length = struct.unpack_from("<H", payload, offset + 32)[0]
        name_start = offset + 46
        name = bytes(payload[name_start : name_start + name_length]).decode("utf-8")
        if name == member:
            struct.pack_into("<I", payload, offset + 24, uncompressed_size)
            path.write_bytes(payload)
            return
        offset = name_start + name_length + extra_length + comment_length


class ArchiveSafetyTests(unittest.TestCase):
    def test_translation_pack_rejects_zero_record_batch(self) -> None:
        schema = make_schema(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "Muhtemel Ask 12.Bolum_TRANSLATION_PACK.zip"
            crafted = root / "crafted-empty-batch.zip"
            create_translation_pack(schema, valid)
            with zipfile.ZipFile(valid, "r") as archive:
                payloads = {
                    info.filename: archive.read(info.filename)
                    for info in archive.infolist()
                }

            manifest = json.loads(payloads["manifest.json"].decode("utf-8"))
            empty_payload = b""
            manifest["batch_count"] = 2
            manifest["batches"].append(
                {
                    "input_file": "batch_002.jsonl",
                    "output_file": "translated_batch_002.jsonl",
                    "block_count": 0,
                    "first_block_uid": None,
                    "last_block_uid": None,
                    "sha256": hashlib.sha256(empty_payload).hexdigest(),
                }
            )
            payloads["manifest.json"] = (
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            payloads["batch_002.jsonl"] = empty_payload
            with zipfile.ZipFile(crafted, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, payload in payloads.items():
                    archive.writestr(name, payload)

            with self.assertRaisesRegex(BatchError, "positive integer|zero records"):
                validate_translation_pack(crafted, expected_schema=schema)

    def test_translated_zip_rejects_zero_record_batch(self) -> None:
        schema = make_schema(1)
        report = {
            "schema_sha256": schema["schema_sha256"],
            "total_input_blocks": 1,
            "total_output_blocks": 0,
            "missing_block_count": 1,
            "duplicate_block_count": 0,
            "review_required_count": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "translated.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("translated_batch_001.jsonl", b"")
                archive.writestr(
                    "translation_report.json",
                    json.dumps(report, sort_keys=True).encode("utf-8"),
                )
            with self.assertRaisesRegex(
                TranslationValidationError, "zero records"
            ):
                load_and_validate_translated_zip(schema, archive_path)

    def test_oversize_metadata_is_rejected_before_crc_or_read(self) -> None:
        schema = make_schema(1)
        records = translation_records(schema)
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "translated.zip"
            write_translated_zip(archive_path, schema, records)
            _patch_central_directory_sizes(
                archive_path,
                "translated_batch_001.jsonl",
                uncompressed_size=MAX_TRANSLATED_ZIP_MEMBER_BYTES + 1,
            )
            with mock.patch.object(
                zipfile.ZipFile,
                "testzip",
                side_effect=AssertionError("CRC scan must not run before preflight"),
            ), mock.patch.object(
                zipfile.ZipFile,
                "read",
                side_effect=AssertionError("member read must not run before preflight"),
            ):
                with self.assertRaisesRegex(
                    TranslationValidationError, "member is too large"
                ):
                    load_and_validate_translated_zip(schema, archive_path)

    def test_suspicious_ratio_metadata_is_rejected_before_crc_or_read(self) -> None:
        schema = make_schema(1)
        records = translation_records(schema)
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "translated.zip"
            write_translated_zip(archive_path, schema, records)
            _patch_central_directory_sizes(
                archive_path,
                "translated_batch_001.jsonl",
                uncompressed_size=4 * 1024 * 1024,
            )
            with mock.patch.object(
                zipfile.ZipFile,
                "testzip",
                side_effect=AssertionError("CRC scan must not run before preflight"),
            ), mock.patch.object(
                zipfile.ZipFile,
                "read",
                side_effect=AssertionError("member read must not run before preflight"),
            ):
                with self.assertRaisesRegex(
                    TranslationValidationError, "suspicious compression ratio"
                ):
                    load_and_validate_translated_zip(schema, archive_path)

    def test_extra_media_is_rejected_before_crc_or_read(self) -> None:
        schema = make_schema(1)
        records = translation_records(schema)
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "translated.zip"
            write_translated_zip(archive_path, schema, records)
            with zipfile.ZipFile(
                archive_path, "a", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("episode.mp4", b"not-media-but-still-forbidden")
            with mock.patch.object(
                zipfile.ZipFile,
                "testzip",
                side_effect=AssertionError("CRC scan must not run for extra media"),
            ), mock.patch.object(
                zipfile.ZipFile,
                "read",
                side_effect=AssertionError("member read must not run for extra media"),
            ):
                with self.assertRaisesRegex(
                    TranslationValidationError, "member mismatch"
                ):
                    load_and_validate_translated_zip(schema, archive_path)


if __name__ == "__main__":
    unittest.main()

