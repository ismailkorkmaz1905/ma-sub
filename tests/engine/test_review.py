from __future__ import annotations

import unittest

from mas.engine.review import collect_review_rows


def _block(index: int, *, flagged: bool = True) -> dict[str, object]:
    return {
        "block_uid": f"uid-{index:03d}",
        "block_index": index,
        "start_ms": index * 1_000,
        "end_ms": index * 1_000 + 900,
        "primary_text": f"Türkçe {index}",
        "youtube_text": "",
        "risk_flags": ["low_word_confidence"] if flagged else [],
    }


def _record(index: int, *, mandatory: bool = False) -> dict[str, object]:
    return {
        "block_uid": f"uid-{index:03d}",
        "tr_final": f"Türkçe {index}",
        "id_final": f"Indonesia {index}",
        "review_required": mandatory,
        "note": "Work Ultra review" if mandatory else "",
    }


class ReviewSelectionTests(unittest.TestCase):
    def test_five_percent_is_total_cap_when_explicit_rows_fit(self) -> None:
        blocks = [_block(index) for index in range(1, 101)]
        records = {
            f"uid-{index:03d}": _record(index, mandatory=index <= 4)
            for index in range(1, 101)
        }

        rows = collect_review_rows(blocks, records, [], max_fraction=0.05)

        self.assertEqual(len(rows), 5)
        self.assertEqual([row["SRT Blok"] for row in rows], [1, 2, 3, 4, 5])

    def test_explicit_rows_are_never_dropped_when_they_exceed_cap(self) -> None:
        blocks = [_block(index) for index in range(1, 101)]
        records = {
            f"uid-{index:03d}": _record(index, mandatory=index <= 8)
            for index in range(1, 101)
        }

        rows = collect_review_rows(blocks, records, [], max_fraction=0.05)

        self.assertEqual(len(rows), 8)
        self.assertEqual([row["SRT Blok"] for row in rows], list(range(1, 9)))


if __name__ == "__main__":
    unittest.main()
