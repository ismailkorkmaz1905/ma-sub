from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

from src.segment import _join_words
from src.transcribe import (
    TranscriptionError,
    _consume_segments,
    _validate_transcription_data,
    load_vtt_captions,
)


def _word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end, probability=0.99)


def _segment(text: str, words: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        start=0.0,
        end=1.0,
        text=text,
        words=words,
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        compression_ratio=1.0,
        temperature=0.0,
    )


class TranscriptionCoverageTests(unittest.TestCase):
    def test_youtube_rolling_vtt_drops_snapshots_and_display_tail(self) -> None:
        payload = """WEBVTT

00:00:01.000 --> 00:00:03.000
Merhaba<00:00:01.500><c> dünya</c>

00:00:03.000 --> 00:00:03.010
Merhaba dünya

00:00:03.010 --> 00:00:10.000
Merhaba dünya
Nasılsın<00:00:03.500><c> bugün?</c>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "youtube.tr.vtt"
            path.write_text(payload, encoding="utf-8")
            records = load_vtt_captions(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "Merhaba dünya")
        self.assertEqual(
            (records[0]["start_ms"], records[0]["end_ms"]),
            (1_000, 2_000),
        )
        self.assertEqual(records[1]["text"], "Nasılsın bugün?")
        self.assertEqual(
            (records[1]["start_ms"], records[1]["end_ms"]),
            (3_010, 4_000),
        )

    def test_standard_vtt_keeps_authored_interval(self) -> None:
        payload = """WEBVTT

00:00:01.000 --> 00:00:05.000
Merhaba dünya
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.vtt"
            path.write_text(payload, encoding="utf-8")
            records = load_vtt_captions(path)

        self.assertEqual(
            records,
            [
                {
                    "caption_index": 1,
                    "start_ms": 1_000,
                    "end_ms": 5_000,
                    "text": "Merhaba dünya",
                }
            ],
        )

    def test_nonempty_segment_without_timed_words_hard_fails(self) -> None:
        with self.assertRaisesRegex(TranscriptionError, "no valid timed words"):
            _consume_segments([_segment("Kadir geldi.", [])])

    def test_segment_text_replacement_still_hard_fails(self) -> None:
        segment = _segment(
            "Kadir geldi.",
            [_word(" Kadir", 0.0, 0.4), _word(" gitti.", 0.45, 0.9)],
        )

        with self.assertRaisesRegex(TranscriptionError, "not fully covered"):
            _consume_segments([segment])

    def test_missing_leading_dotted_initial_is_repaired_in_place(self) -> None:
        segments, words = _consume_segments(
            [
                _segment(
                    "Altyazı M.K.",
                    [
                        _word(" Altyazı", 0.0, 0.45),
                        _word(" .K.", 0.50, 0.95),
                    ],
                )
            ]
        )

        self.assertEqual(
            [word["text"].strip() for word in words], ["Altyazı", "M.K."]
        )
        self.assertEqual((words[1]["start_ms"], words[1]["end_ms"]), (500, 950))
        self.assertEqual(words[1]["probability"], 0.0)
        self.assertEqual(
            words[1]["timing_source"], "dotted_initialism_surface_repair"
        )
        self.assertEqual(segments[0]["words"], words)
        self.assertEqual(_join_words(words), "Altyazı M.K.")
        data = {
            "format_version": "1.0",
            "audio_sha256": "c" * 64,
            "language": "tr",
            "segments": segments,
            "words": words,
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 950,
                    "source": "synthetic-test",
                }
            ],
        }
        self.assertEqual(_validate_transcription_data(data, "c" * 64), [])

    def test_non_initialism_partial_word_still_hard_fails(self) -> None:
        segment = _segment(
            "Kadir geldi.",
            [_word(" Kadir", 0.0, 0.4), _word(" eldi.", 0.45, 0.9)],
        )

        with self.assertRaisesRegex(TranscriptionError, "not fully covered"):
            _consume_segments([segment])

    def test_changed_dotted_initial_still_hard_fails(self) -> None:
        segment = _segment(
            "Altyazı M.K.",
            [_word(" Altyazı", 0.0, 0.4), _word(" N.K.", 0.45, 0.9)],
        )

        with self.assertRaisesRegex(TranscriptionError, "not fully covered"):
            _consume_segments([segment])

    def test_missing_leading_word_gets_explicit_synthetic_timing(self) -> None:
        segments, words = _consume_segments(
            [
                _segment(
                    "Ne yapıyorsun bakalım?",
                    [
                        _word(" yapıyorsun", 0.20, 0.55),
                        _word(" bakalım?", 0.60, 0.95),
                    ],
                )
            ]
        )

        self.assertEqual([word["text"].strip() for word in words], ["Ne", "yapıyorsun", "bakalım?"])
        self.assertEqual(words[0]["timing_source"], "synthetic_segment_word_repair")
        self.assertEqual(words[0]["probability"], 0.0)
        self.assertLess(words[0]["start_ms"], words[0]["end_ms"])
        self.assertEqual(segments[0]["words"], words)

    def test_missing_trailing_word_gets_explicit_synthetic_timing(self) -> None:
        segments, words = _consume_segments(
            [_segment("Kadir geldi.", [_word(" Kadir", 0.0, 0.4)])]
        )

        self.assertEqual([word["text"].strip() for word in words], ["Kadir", "geldi."])
        self.assertEqual(words[-1]["timing_source"], "synthetic_segment_word_repair")
        self.assertEqual(segments[0]["words"], words)

    def test_internal_omissions_get_explicit_synthetic_timing(self) -> None:
        segments, words = _consume_segments(
            [
                _segment(
                    "Ya bırak Allah Allah Allah.",
                    [
                        _word(" Ya", 0.0, 0.2),
                        _word(" Allah", 0.45, 0.65),
                        _word(" Allah.", 0.70, 0.90),
                    ],
                )
            ]
        )

        self.assertEqual(
            [word["text"].strip() for word in words],
            ["Ya", "bırak", "Allah", "Allah", "Allah."],
        )
        repaired = [word for word in words if word.get("timing_source")]
        self.assertEqual([word["text"].strip() for word in repaired], ["bırak", "Allah"])
        self.assertTrue(all(word["probability"] == 0.0 for word in repaired))
        self.assertEqual(segments[0]["words"], words)

    def test_combined_timed_word_is_split_before_omission_repair(self) -> None:
        segments, words = _consume_segments(
            [
                _segment(
                    "Ben de buradan teşekkür ederim.",
                    [
                        # The real failure has no gap before the first timed
                        # record, so both omitted prefix tokens must safely
                        # share its boundary.
                        _word(" buradan teşekkür", 0.0, 0.65),
                        _word(" ederim.", 0.70, 0.95),
                    ],
                )
            ]
        )

        self.assertEqual(
            [word["text"].strip() for word in words],
            ["Ben", "de", "buradan", "teşekkür", "ederim."],
        )
        self.assertEqual(words[0]["timing_source"], "synthetic_segment_word_repair")
        self.assertEqual(words[2]["timing_source"], "split_combined_whisper_word")
        self.assertEqual(words[3]["timing_source"], "split_combined_whisper_word")
        self.assertEqual(
            [word["start_ms"] for word in words],
            sorted(word["start_ms"] for word in words),
        )
        self.assertEqual(segments[0]["words"], words)

    def test_internal_omission_without_gap_preserves_text_order(self) -> None:
        segments, words = _consume_segments(
            [
                _segment(
                    "Ya bırak Allah.",
                    [
                        _word(" Ya", 0.0, 0.4),
                        _word(" Allah.", 0.3, 0.9),
                    ],
                )
            ]
        )

        self.assertEqual(
            [word["text"].strip() for word in words], ["Ya", "bırak", "Allah."]
        )
        self.assertEqual(
            [word["start_ms"] for word in words],
            sorted(word["start_ms"] for word in words),
        )
        self.assertEqual(segments[0]["words"], words)

    def test_trailing_omission_without_gap_preserves_text_order(self) -> None:
        segments, words = _consume_segments(
            [_segment("Kadir geldi şimdi.", [_word(" Kadir", 0.0, 1.0)])]
        )

        self.assertEqual(
            [word["text"].strip() for word in words], ["Kadir", "geldi", "şimdi."]
        )
        self.assertEqual(
            [word["start_ms"] for word in words],
            sorted(word["start_ms"] for word in words),
        )
        self.assertEqual(segments[0]["words"], words)

    def test_persisted_transcription_rejects_silent_segment_loss(self) -> None:
        valid_word = {
            "word_index": 1,
            "segment_id": 1,
            "start_ms": 0,
            "end_ms": 500,
            "text": "Merhaba",
            "probability": 0.99,
        }
        data = {
            "format_version": "1.0",
            "audio_sha256": "a" * 64,
            "language": "tr",
            "segments": [
                {
                    "segment_id": 1,
                    "start_ms": 0,
                    "end_ms": 500,
                    "text": "Merhaba",
                    "words": [valid_word],
                },
                {
                    "segment_id": 2,
                    "start_ms": 600,
                    "end_ms": 1_100,
                    "text": "Kadir geldi.",
                    "words": [],
                },
            ],
            "words": [valid_word],
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_100,
                    "source": "synthetic-test",
                }
            ],
        }

        errors = _validate_transcription_data(data, "a" * 64)

        self.assertIn(
            "ASR segment 2 contains text but has no valid timed words", errors
        )

    def test_normalized_punctuation_and_spacing_preserve_full_coverage(self) -> None:
        segments, words = _consume_segments(
            [
                _segment(
                    "Allah'ım, geldim!",
                    [
                        _word(" Allah'ım,", 0.0, 0.4),
                        _word(" geldim!", 0.45, 0.9),
                    ],
                )
            ]
        )
        data = {
            "format_version": "1.0",
            "audio_sha256": "b" * 64,
            "language": "tr",
            "segments": segments,
            "words": words,
            "vad_regions": [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 900,
                    "source": "synthetic-test",
                }
            ],
        }

        self.assertEqual(_validate_transcription_data(data, "b" * 64), [])


if __name__ == "__main__":
    unittest.main()
