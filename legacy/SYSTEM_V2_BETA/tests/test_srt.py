from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from helpers import make_schema, records_by_uid, translation_records
from src.srt import (
    SRTError,
    SubtitleEntry,
    assert_srt_roundtrip,
    build_entries,
    parse_srt,
    render_srt,
    write_srt,
)
from src.subtitle_qa import run_subtitle_qa


class SRTTests(unittest.TestCase):
    def test_19_overlap_is_reported_and_never_silently_retimed(self) -> None:
        schema = make_schema(2)
        records = translation_records(schema)
        blocks = copy.deepcopy(schema["blocks"])
        blocks[1]["start_ms"] = blocks[0]["end_ms"] - 1

        report = run_subtitle_qa(blocks, records)

        self.assertEqual(report["overlap_count"], 1)
        rendered = render_srt(
            [
                SubtitleEntry(1, 1_000, 2_000, "Bir"),
                SubtitleEntry(2, 1_999, 2_500, "İki"),
            ]
        )
        self.assertIn("00:00:01,999 --> 00:00:02,500", rendered)

    def test_20_more_than_two_visible_lines_fails_qa(self) -> None:
        schema = make_schema(1)
        records = translation_records(schema)
        records[0]["id_final"] = "satır satu\nsatır dua\nsatır tiga"

        report = run_subtitle_qa(schema["blocks"], records)

        self.assertEqual(report["more_than_two_lines_count"], 1)
        with self.assertRaises(SRTError):
            build_entries(schema["blocks"], records_by_uid(records), "id")

    def test_21_line_over_84_characters_fails_qa(self) -> None:
        schema = make_schema(1)
        records = translation_records(schema)
        records[0]["id_final"] = "x" * 85

        report = run_subtitle_qa(schema["blocks"], records)

        self.assertEqual(report["line_over_84_count"], 1)
        with self.assertRaises(SRTError):
            build_entries(schema["blocks"], records_by_uid(records), "id")

    def test_22_utf8_srt_round_trip(self) -> None:
        entries = [
            SubtitleEntry(1, 1_234, 3_456, "Özlem: Şimdi görüşürüz."),
            SubtitleEntry(2, 4_000, 5_500, "Masyaallah, kamu baik-baik saja."),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Muhtemel Ask 12.Bolum.id-final.srt"
            write_srt(path, entries)

            self.assertEqual(path.read_bytes().decode("utf-8"), render_srt(entries))
            self.assertEqual(parse_srt(path), entries)
            assert_srt_roundtrip(path, entries)

    def test_23_timing_round_trip_is_exact_to_one_millisecond(self) -> None:
        entries = [
            SubtitleEntry(1, 0, 1, "A"),
            SubtitleEntry(2, 3_599_999, 3_600_001, "B"),
            SubtitleEntry(3, 7_234_567, 7_239_999, "C"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timing.srt"
            write_srt(path, entries)
            reparsed = parse_srt(path)

        self.assertEqual(
            [(entry.start_ms, entry.end_ms) for entry in reparsed],
            [(entry.start_ms, entry.end_ms) for entry in entries],
        )

    def test_25_empty_translation_is_rejected(self) -> None:
        schema = make_schema(1)
        records = translation_records(schema)
        records[0]["id_final"] = "   "

        report = run_subtitle_qa(schema["blocks"], records)

        self.assertEqual(report["empty_text_count"], 1)
        with self.assertRaises(SRTError):
            build_entries(schema["blocks"], records_by_uid(records), "id")


if __name__ == "__main__":
    unittest.main()
