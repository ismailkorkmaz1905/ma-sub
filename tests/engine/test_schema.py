from __future__ import annotations

import copy
import hashlib
import unittest

from helpers import make_schema, raw_blocks
from mas.engine.schema import (
    SchemaError,
    build_episode_schema,
    canonical_json_bytes,
    compute_schema_sha256,
    generate_block_uid,
    validate_episode_schema,
)


class BlockIdentityTests(unittest.TestCase):
    @staticmethod
    def _raw_declared_sha(schema: dict) -> str:
        """Mimic an attacker recomputing the envelope hash without validation."""

        digest_input = {
            "schema_version": schema["schema_version"],
            "episode": schema["episode"],
            "blocks": schema["blocks"],
        }
        return hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()

    def test_01_stable_block_uid_generation(self) -> None:
        arguments = (12, 17, 31_250, 34_700, "  Allah\u00a0aşkına, Defne!  ", "1.0")
        first = generate_block_uid(*arguments)
        second = generate_block_uid(*arguments)

        self.assertEqual(first, second)
        self.assertRegex(first, r"^MA12-000017-[0-9a-f]{12}$")

    def test_02_same_input_creates_same_block_uid(self) -> None:
        first = build_episode_schema(raw_blocks(3), episode=12)
        second = build_episode_schema(copy.deepcopy(raw_blocks(3)), episode=12)

        self.assertEqual(
            [block["block_uid"] for block in first["blocks"]],
            [block["block_uid"] for block in second["blocks"]],
        )

    def test_03_changed_timing_creates_different_uid(self) -> None:
        baseline = raw_blocks(1)
        retimed = copy.deepcopy(baseline)
        retimed[0]["start_ms"] += 1

        first = build_episode_schema(baseline, episode=12)
        second = build_episode_schema(retimed, episode=12)

        self.assertNotEqual(
            first["blocks"][0]["block_uid"], second["blocks"][0]["block_uid"]
        )

    def test_04_schema_sha_consistency_and_tamper_detection(self) -> None:
        schema = make_schema(5)

        self.assertEqual(
            schema["schema_sha256"],
            compute_schema_sha256(
                schema["blocks"],
                episode=schema["episode"],
                schema_version=schema["schema_version"],
            ),
        )
        self.assertEqual(schema, validate_episode_schema(schema))

        tampered = copy.deepcopy(schema)
        tampered["blocks"][0]["primary_text"] = "Başka bir kaynak cümlesi."
        with self.assertRaises(SchemaError):
            validate_episode_schema(tampered)

    def test_schema_version_changes_block_identity(self) -> None:
        first = make_schema(2, schema_version="1.0")
        second = make_schema(2, schema_version="2.0")
        self.assertNotEqual(first["schema_sha256"], second["schema_sha256"])
        self.assertNotEqual(
            first["blocks"][0]["block_uid"], second["blocks"][0]["block_uid"]
        )

    def test_schema_rejects_translation_fields(self) -> None:
        blocks = raw_blocks(1)
        blocks[0]["tr_final"] = "Bu alan immutable şemaya ait değil."
        with self.assertRaises(SchemaError):
            build_episode_schema(blocks, episode=12)

    def test_missing_per_block_episode_cannot_be_hidden_by_recomputed_sha(self) -> None:
        tampered = copy.deepcopy(make_schema(2))
        del tampered["blocks"][1]["episode"]

        with self.assertRaisesRegex(SchemaError, "missing immutable fields: episode"):
            validate_episode_schema(tampered)
        with self.assertRaisesRegex(SchemaError, "missing immutable fields: episode"):
            compute_schema_sha256(
                tampered["blocks"],
                episode=tampered["episode"],
                schema_version=tampered["schema_version"],
            )

        tampered["schema_sha256"] = self._raw_declared_sha(tampered)
        with self.assertRaisesRegex(SchemaError, "missing immutable fields: episode"):
            validate_episode_schema(tampered)

    def test_adjacent_overlap_hard_fails_build_and_tamper_validation(self) -> None:
        overlapping_raw = raw_blocks(2)
        overlapping_raw[1]["start_ms"] = overlapping_raw[0]["end_ms"] - 1
        with self.assertRaisesRegex(SchemaError, "overlaps the preceding block"):
            build_episode_schema(overlapping_raw, episode=12)

        tampered = copy.deepcopy(make_schema(2))
        second = tampered["blocks"][1]
        second["start_ms"] = tampered["blocks"][0]["end_ms"] - 1
        second["block_uid"] = generate_block_uid(
            tampered["episode"],
            second["block_index"],
            second["start_ms"],
            second["end_ms"],
            second["timing_text"],
            tampered["schema_version"],
        )

        with self.assertRaisesRegex(SchemaError, "overlaps the preceding block"):
            compute_schema_sha256(
                tampered["blocks"],
                episode=tampered["episode"],
                schema_version=tampered["schema_version"],
            )
        tampered["schema_sha256"] = self._raw_declared_sha(tampered)
        with self.assertRaisesRegex(SchemaError, "overlaps the preceding block"):
            validate_episode_schema(tampered)


if __name__ == "__main__":
    unittest.main()
