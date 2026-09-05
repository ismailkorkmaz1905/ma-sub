from __future__ import annotations

import copy
import unittest

from helpers import clone_records, make_schema, translation_records
from mas.engine.translation_validation import (
    TranslationValidationError,
    validate_translation_records,
)


class TranslationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = make_schema(20)
        self.records = translation_records(self.schema)

    def _diagnose(self, records: list[dict]) -> object:
        return validate_translation_records(
            self.schema,
            records,
            raise_on_error=False,
        )

    def _assert_hard_fail(self, records: list[dict]) -> TranslationValidationError:
        with self.assertRaises(TranslationValidationError) as caught:
            validate_translation_records(self.schema, records)
        self.assertIsNotNone(caught.exception.result)
        self.assertFalse(caught.exception.result.ok)
        self.assertEqual(caught.exception.result.records_by_uid, {})
        return caught.exception

    def test_valid_translation_contract_exposes_uid_map(self) -> None:
        result = validate_translation_records(self.schema, self.records)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.records_by_uid), self.schema["block_count"])
        self.assertEqual(
            [record["block_uid"] for record in result.ordered_records(self.schema)],
            [block["block_uid"] for block in self.schema["blocks"]],
        )

    def test_05_missing_translation_uid_hard_fails(self) -> None:
        records = clone_records(self.records[:-1])
        result = self._diagnose(records)
        self.assertEqual(result.report.missing_translation_count, 1)
        self.assertEqual(result.records_by_uid, {})
        self._assert_hard_fail(records)

    def test_06_duplicate_translation_uid_hard_fails(self) -> None:
        records = clone_records(self.records)
        records.append(copy.deepcopy(records[4]))
        result = self._diagnose(records)
        self.assertEqual(result.report.duplicate_translation_count, 1)
        self.assertTrue(
            any(issue.code == "duplicate_translation_uid" for issue in result.report.errors)
        )
        self._assert_hard_fail(records)

    def test_07_extra_translation_uid_hard_fails(self) -> None:
        records = clone_records(self.records)
        extra = copy.deepcopy(records[-1])
        extra["block_uid"] = "MA12-999999-deadbeef0000"
        records.append(extra)
        result = self._diagnose(records)
        self.assertEqual(result.report.extra_translation_count, 1)
        self._assert_hard_fail(records)

    def test_08_reordered_translation_records_hard_fail(self) -> None:
        records = clone_records(self.records)
        records[3], records[4] = records[4], records[3]
        result = self._diagnose(records)
        self.assertEqual(result.report.order_mismatch_count, 2)
        self._assert_hard_fail(records)

    def test_09_changed_schema_sha_hard_fails(self) -> None:
        records = clone_records(self.records)
        records[7]["schema_sha256"] = "0" * 64
        result = self._diagnose(records)
        self.assertEqual(result.report.schema_mismatch_count, 1)
        issue = next(
            issue
            for issue in result.report.errors
            if issue.code == "schema_sha256_mismatch"
        )
        self.assertEqual(issue.block_uid, records[7]["block_uid"])
        self.assertEqual(issue.batch_file, "<memory>")
        self._assert_hard_fail(records)

    def test_10_wrong_block_number_mapping_hard_fails(self) -> None:
        records = translation_records(self.schema, include_optional_echoes=True)
        records[5]["block_index"] = 7
        result = self._diagnose(records)
        self.assertEqual(result.report.block_index_mismatch_count, 1)
        self.assertGreater(result.report.record_shape_mismatch_count, 0)
        self._assert_hard_fail(records)

    def test_12_shifted_translation_beginning_at_block_12_is_detected(self) -> None:
        records = clone_records(self.records)
        shifted_texts = [record["tr_final"] for record in records[11:]]
        shifted_texts = shifted_texts[1:] + shifted_texts[:1]
        for record, shifted_text in zip(records[11:], shifted_texts):
            record["tr_final"] = shifted_text
        result = self._diagnose(records)
        self.assertGreater(result.report.positional_translation_mismatch_count, 0)
        shifted_issues = [
            issue
            for issue in result.report.errors
            if issue.code == "positional_translation_mismatch"
        ]
        self.assertTrue(
            any(issue.expected["block_index"] == 12 for issue in shifted_issues)
        )
        self._assert_hard_fail(records)

    def test_13_split_block_attempt_is_rejected(self) -> None:
        records = clone_records(self.records)
        first_half = copy.deepcopy(records[9])
        second_half = copy.deepcopy(records[9])
        first_half["tr_final"] = "Bu benzersiz cümle 10"
        second_half["tr_final"] = "numaralı sahneye aittir."
        records[9:10] = [first_half, second_half]
        result = self._diagnose(records)
        self.assertEqual(result.report.duplicate_translation_count, 1)
        self.assertGreater(result.report.block_count_mismatch_count, 0)
        self._assert_hard_fail(records)

    def test_14_merge_block_attempt_is_rejected(self) -> None:
        records = clone_records(self.records)
        records[8]["tr_final"] += " " + records[9]["tr_final"]
        del records[9]
        result = self._diagnose(records)
        self.assertEqual(result.report.missing_translation_count, 1)
        self.assertGreater(result.report.block_count_mismatch_count, 0)
        self._assert_hard_fail(records)


if __name__ == "__main__":
    unittest.main()
