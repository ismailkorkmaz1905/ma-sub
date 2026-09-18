from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

import mas.engine.forced_align as forced_align
from mas.engine.forced_align import (
    ALIGNMENT_TEXT_NORMALIZATION,
    AUDIO_REVIEW_SCORE_CONTEXT,
    CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE,
    DEFAULT_MAX_OUTWARD_DRIFT_MS,
    DEFAULT_MAX_WORD_DURATION_MS,
    DEFAULT_MIN_WORD_SCORE,
    DEFAULT_TURKISH_ALIGNMENT_MODEL,
    DURATION_VAD_CONTEXT,
    EDITED_TOKEN_MIN_WORD_SCORE,
    REVIEW_WORD_SCORE,
    SUPPORTED_WHISPERX_VERSION,
    TIMING_SOURCE,
    ForcedAlignmentError,
    align_corrected_segments,
    correction_deletes_lexical_tokens,
    validate_coarse_segments,
    validate_forced_alignment_data,
)
from mas.engine.segmentation import build_blocks


class _FakeWhisperX:
    __version__ = "3.8.6"

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = list(results)
        self.audio_calls: list[str] = []
        self.model_calls: list[dict[str, Any]] = []
        self.align_calls: list[dict[str, Any]] = []

    def load_audio(self, path: str) -> object:
        self.audio_calls.append(path)
        return object()

    def load_align_model(
        self, *, language_code: str, device: str, model_name: str
    ) -> tuple[object, dict[str, str]]:
        self.model_calls.append(
            {
                "language_code": language_code,
                "device": device,
                "model_name": model_name,
            }
        )
        return object(), {"language": language_code, "type": "huggingface"}

    def align(
        self,
        transcript: list[dict[str, Any]],
        model: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
    ) -> dict[str, Any]:
        self.align_calls.append(
            {
                "transcript": transcript,
                "interpolate_method": interpolate_method,
                "return_char_alignments": return_char_alignments,
                "print_progress": print_progress,
            }
        )
        return self.results.pop(0)


class _NoneInterpolationWhisperX(_FakeWhisperX):
    def align(
        self,
        transcript: list[dict[str, Any]],
        model: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        interpolate_method: str | None = None,
        return_char_alignments: bool = False,
        print_progress: bool = False,
    ) -> dict[str, Any]:
        return super().align(
            transcript,
            model,
            metadata,
            audio,
            device,
            interpolate_method=interpolate_method,  # type: ignore[arg-type]
            return_char_alignments=return_char_alignments,
            print_progress=print_progress,
        )


class _MutatingWhisperX(_FakeWhisperX):
    def align(
        self,
        transcript: list[dict[str, Any]],
        model: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
    ) -> dict[str, Any]:
        Path(self.audio_calls[-1]).write_bytes(b"changed-during-alignment")
        return super().align(
            transcript,
            model,
            metadata,
            audio,
            device,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments,
            print_progress=print_progress,
        )


class _OverlapWhisperX(_FakeWhisperX):
    def __init__(self, *, joint_resolves: bool) -> None:
        super().__init__([])
        self.joint_resolves = joint_resolves

    def align(
        self,
        transcript: list[dict[str, Any]],
        model: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
    ) -> dict[str, Any]:
        text = transcript[0]["text"]
        if "Merhaba" in text and "Selam" in text:
            second_start = 1.5 if self.joint_resolves else 1.3
            return _result(
                [
                    {"word": "Merhaba.", "start": 1.1, "end": 1.4},
                    {"word": "Selam.", "start": second_start, "end": 2.2},
                ]
            )
        if "Merhaba" in text:
            return _result([{"word": "Merhaba.", "start": 1.1, "end": 2.0}])
        return _result([{"word": "Selam.", "start": 1.3, "end": 2.2}])


class _TwoPairOverlapWhisperX(_FakeWhisperX):
    def __init__(self) -> None:
        super().__init__([])

    def align(self, transcript, *args, **kwargs):
        text = transcript[0]["text"]
        self.align_calls.append({"text": text})
        results = {
            "Alpha": [("Alpha", 1.1, 2.0)],
            "Bravo": [("Bravo", 1.3, 2.2)],
            "Charlie": [("Charlie", 10.1, 11.0)],
            "Delta": [("Delta", 10.3, 11.2)],
            "Alpha Bravo": [("Alpha", 1.1, 1.4), ("Bravo", 1.5, 1.8)],
            "Charlie Delta": [("Charlie", 10.1, 10.4), ("Delta", 10.5, 10.8)],
        }
        return _result([
            {"word": word, "start": start, "end": end}
            for word, start, end in results[text]
        ])


class _PartialJointOverlapWhisperX(_FakeWhisperX):
    def __init__(self) -> None:
        super().__init__([])

    def align(self, transcript, *args, **kwargs):
        text = transcript[0]["text"]
        self.align_calls.append({"text": text})
        results = {
            "Alpha": [("Alpha", 1.1, 2.0, 0.9)],
            "Bravo": [("Bravo", 1.3, 2.2, 0.9)],
            "Alpha Bravo": [
                ("Alpha", 1.1, 1.4, 0.1),
                ("Bravo", 2.1, 2.2, 0.9),
            ],
        }
        return _result([
            {"word": word, "start": start, "end": end, "score": score}
            for word, start, end, score in results[text]
        ])


class _ContextualOverlapWhisperX(_FakeWhisperX):
    def __init__(self, *, invalid_context: bool = False) -> None:
        super().__init__([])
        self.invalid_context = invalid_context

    def align(
        self,
        transcript: list[dict[str, Any]],
        model: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
    ) -> dict[str, Any]:
        text = transcript[0]["text"]
        self.align_calls.append({"text": text})
        if text == "Once Ben ona ne yaptim Hicbir sey yapmadim Ben ona ne yaptim":
            return _result(
                [
                    {"word": "Once", "start": 0.45, "end": 0.75,
                     "score": 0.01 if self.invalid_context else 0.90},
                    {"word": "Ben", "start": 1.1, "end": 1.2},
                    {"word": "ona", "start": 1.2, "end": 1.3},
                    {"word": "ne", "start": 1.3, "end": 1.4},
                    {"word": "yaptim", "start": 1.4, "end": 1.5},
                    {"word": "Hicbir", "start": 1.6, "end": 1.7},
                    {"word": "sey", "start": 1.7, "end": 1.8},
                    {"word": "yapmadim", "start": 1.8, "end": 1.9},
                    {"word": "Ben", "start": 2.45, "end": 2.55},
                    {"word": "ona", "start": 2.55, "end": 2.65},
                    {"word": "ne", "start": 2.65, "end": 2.75},
                    {"word": "yaptim", "start": 2.75, "end": 2.85},
                ]
            )
        if text == "Ben ona ne yaptim Hicbir sey yapmadim":
            return _result(
                [
                    {"word": "Ben", "start": 1.1, "end": 1.2},
                    {"word": "ona", "start": 1.2, "end": 1.3},
                    {"word": "ne", "start": 1.3, "end": 1.4},
                    {"word": "yaptim", "start": 1.4, "end": 1.5},
                    {"word": "Hicbir", "start": 2.3, "end": 2.4},
                    {"word": "sey", "start": 2.4, "end": 2.5},
                    {"word": "yapmadim", "start": 2.5, "end": 2.6},
                ]
            )
        words = text.split()
        if text == "Once":
            start, end = 0.5, 0.8
        elif text == "Hicbir sey yapmadim":
            start, end = 1.3, 2.2
        elif len(self.align_calls) == 2:
            start, end = 1.1, 2.0
        else:
            start, end = 2.4, 2.8
        duration = (end - start) / len(words)
        return _result(
            [
                {"word": word, "start": start + index * duration,
                 "end": start + (index + 1) * duration}
                for index, word in enumerate(words)
            ]
        )


class _ResidualEarlyStopWhisperX(_FakeWhisperX):
    def __init__(self) -> None:
        super().__init__([])

    def align(self, transcript, *args, **kwargs):
        text = transcript[0]["text"]
        self.align_calls.append({"text": text})
        if text == "Alpha Bravo":
            return _result([{"word": "Alpha", "start": 1.1, "end": 1.9}])
        if text == "Context Alpha Bravo":
            return _result(
                [
                    {"word": "Context", "start": 0.5, "end": 0.8},
                    {"word": "Alpha", "start": 1.1, "end": 1.4},
                    {"word": "Bravo", "start": 1.5, "end": 1.8},
                ]
            )
        start, end = {
            "Context": (0.5, 0.8),
            "Alpha": (1.1, 2.0),
            "Bravo": (1.3, 2.2),
        }[text]
        return _result([{"word": text, "start": start, "end": end}])


class _OverBudgetContextWhisperX(_FakeWhisperX):
    def __init__(self) -> None:
        super().__init__([])

    def align(self, transcript, *args, **kwargs):
        text = transcript[0]["text"]
        self.align_calls.append({"text": text})
        if text == "A B":
            return _result([{"word": "A", "start": 1.1, "end": 2.0}])
        times = {
            "A": (1.1, 2.0),
            "B": (1.3, 2.2),
            "C": (2.35, 2.55),
            "D": (4.1, 4.3),
            "E": (5.1, 5.3),
        }
        words = text.split()
        if len(words) >= 3 and words[:3] == ["A", "B", "C"]:
            times["A"] = (2.4, 2.5)
            times["C"] = (3.0, 3.2)
            if "D" in words:
                times["D"] = (4.2, 4.25)
        return _result(
            [
                {"word": word, "start": times[word][0], "end": times[word][1]}
                for word in words
            ]
        )


def _result(
    words: list[dict[str, Any]], *, add_default_scores: bool = True
) -> dict[str, Any]:
    normalized = copy.deepcopy(words)
    if add_default_scores:
        for word in normalized:
            word.setdefault("score", 0.90)
    return {
        "segments": [{"text": "unused", "words": normalized}],
        "word_segments": normalized,
    }


def _coarse() -> list[dict[str, Any]]:
    return [
        {
            "start_ms": 1_000,
            "end_ms": 3_000,
            "text": "Merhaba, dünya!",
            "asr_text": "Merhaba dünya",
            "deletion_audio_reviewed": False,
            "utterance_uid": "utt-1",
        },
        {
            "start_ms": 4_000,
            "end_ms": 5_500,
            "text": "Nasılsın?",
            "asr_text": "Nasılsın",
            "deletion_audio_reviewed": False,
            "utterance_uid": "utt-2",
        },
    ]


class ForcedAlignmentTests(unittest.TestCase):
    def _audio(self, directory: str) -> Path:
        path = Path(directory, "audio.flac")
        path.write_bytes(b"synthetic-audio-placeholder")
        return path

    def _two_pair_coarse(self):
        return [
            {
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "asr_text": text,
                "deletion_audio_reviewed": False,
                "utterance_uid": f"utt-{text.lower()}",
                "speaker_id": "speaker-a",
            }
            for start, end, text in (
                (1000, 2500, "Alpha"),
                (1200, 2800, "Bravo"),
                (10000, 11500, "Charlie"),
                (10200, 11800, "Delta"),
            )
        ]

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_component_checkpoint_reuses_postprocess_with_identical_output(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            cold = align_corrected_segments(
                audio, self._two_pair_coarse(),
                whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
            )
            reused = _TwoPairOverlapWhisperX()
            with patch.object(
                forced_align, "_normalize_aligned_words",
                wraps=forced_align._normalize_aligned_words,
            ) as normalize:
                warm = align_corrected_segments(
                    audio, self._two_pair_coarse(), whisperx_module=reused,
                    checkpoint_dir=checkpoint,
                )

        self.assertEqual(reused.align_calls, [])
        self.assertEqual(normalize.call_count, 4)
        self.assertEqual(cold, warm)
        self.assertEqual(json.dumps(cold, sort_keys=True), json.dumps(warm, sort_keys=True))
        validate_forced_alignment_data(warm)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_component_checkpoint_changed_cue_invalidates_only_its_component(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            coarse = self._two_pair_coarse()
            align_corrected_segments(
                audio, coarse, whisperx_module=_TwoPairOverlapWhisperX(),
                checkpoint_dir=checkpoint,
            )
            coarse[0]["end_ms"] = 2400
            changed = _TwoPairOverlapWhisperX()
            with patch.object(
                forced_align, "_normalize_aligned_words",
                wraps=forced_align._normalize_aligned_words,
            ) as normalize:
                data = align_corrected_segments(
                    audio, coarse, whisperx_module=changed, checkpoint_dir=checkpoint,
                )
            fresh = align_corrected_segments(
                audio, coarse, whisperx_module=_TwoPairOverlapWhisperX(),
                checkpoint_dir=Path(directory) / "cold-units",
            )

        self.assertEqual(changed.align_calls, [{"text": "Alpha"}])
        self.assertEqual(normalize.call_count, 6)
        self.assertEqual(data, fresh)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_component_checkpoint_binds_joint_raw_result(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            cold = align_corrected_segments(
                audio, self._two_pair_coarse(),
                whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
            )
            for path in checkpoint.glob("*/*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                raw = record["data"]["result"]["word_segments"]
                if [word["word"] for word in raw] != ["Alpha", "Bravo"]:
                    continue
                raw[0]["score"] = 0.85
                record["sha256"] = forced_align.digest(record["data"])
                path.write_text(json.dumps(record), encoding="utf-8")
                break
            else:
                self.fail("missing joint raw checkpoint")
            reused = _TwoPairOverlapWhisperX()
            with patch.object(
                forced_align, "_normalize_aligned_words",
                wraps=forced_align._normalize_aligned_words,
            ) as normalize:
                data = align_corrected_segments(
                    audio, self._two_pair_coarse(), whisperx_module=reused,
                    checkpoint_dir=checkpoint,
                )

        self.assertEqual(reused.align_calls, [])
        self.assertEqual(normalize.call_count, 6)
        self.assertNotEqual(data["alignment_sha256"], cold["alignment_sha256"])
        self.assertEqual(data["segments"][0]["words"][0]["score"], 0.85)
        validate_forced_alignment_data(data)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_component_checkpoint_local_vad_change_and_global_policy_invalidation(self, _model):
        regions = [
            {"start_ms": 1000, "end_ms": 2800, "vad_region_index": 1,
             "source": "silero_vad"},
            {"start_ms": 10000, "end_ms": 11800, "vad_region_index": 2,
             "source": "silero_vad"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            align_corrected_segments(
                audio, self._two_pair_coarse(), vad_regions=regions,
                whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
            )
            regions[0]["end_ms"] = 2700
            with patch.object(
                forced_align, "_normalize_aligned_words",
                wraps=forced_align._normalize_aligned_words,
            ) as normalize:
                data = align_corrected_segments(
                    audio, self._two_pair_coarse(), vad_regions=regions,
                    whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
                )
                self.assertEqual(normalize.call_count, 6)
                normalize.reset_mock()
                data = align_corrected_segments(
                    audio, self._two_pair_coarse(), vad_regions=regions,
                    min_word_score=0.35,
                    whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
                )
                self.assertEqual(normalize.call_count, 8)

        validate_forced_alignment_data(data)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_malformed_component_payload_recomputes_from_raw_evidence(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            cold = align_corrected_segments(
                audio, self._two_pair_coarse(),
                whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
            )
            for path in (checkpoint / "components").glob("*/*.json"):
                original = json.loads(path.read_text(encoding="utf-8"))
                if "utt-alpha" in original["data"]["result"]["selected"]:
                    break
            else:
                self.fail("missing component checkpoint")
            for field, value in (
                ("selected", None), ("diagnostics", {}), ("joint_calls", [1]),
                ("options", None),
            ):
                with self.subTest(field=field):
                    record = copy.deepcopy(original)
                    if field == "options":
                        del record["data"]["result"][field]
                    else:
                        record["data"]["result"][field] = value
                    record["sha256"] = forced_align.digest(record["data"])
                    path.write_text(json.dumps(record), encoding="utf-8")
                    fake = _TwoPairOverlapWhisperX()
                    with patch.object(
                        forced_align, "_normalize_aligned_words",
                        wraps=forced_align._normalize_aligned_words,
                    ) as normalize:
                        data = align_corrected_segments(
                            audio, self._two_pair_coarse(), whisperx_module=fake,
                            checkpoint_dir=checkpoint,
                        )
                    self.assertEqual(fake.align_calls, [])
                    self.assertEqual(normalize.call_count, 6)
                    self.assertEqual(data, cold)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_component_checkpoint_cannot_bypass_final_acoustic_score_gate(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            align_corrected_segments(
                audio, self._two_pair_coarse(),
                whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
            )
            for path in (checkpoint / "components").glob("*/*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                selected = record["data"]["result"]["selected"]
                if "utt-alpha" not in selected:
                    continue
                selected["utt-alpha"][0]["score"] = 0.1
                selected["utt-alpha"][0]["probability"] = 0.1
                record["sha256"] = forced_align.digest(record["data"])
                path.write_text(json.dumps(record), encoding="utf-8")
                break
            else:
                self.fail("missing component checkpoint")
            with self.assertRaisesRegex(ForcedAlignmentError, "score is below"):
                align_corrected_segments(
                    audio, self._two_pair_coarse(),
                    whisperx_module=_TwoPairOverlapWhisperX(), checkpoint_dir=checkpoint,
                )

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_residual_normalization_replays_without_reprocessing_or_model_calls(self, _model):
        coarse = [
            {"start_ms": start, "end_ms": end, "text": text, "asr_text": text,
             "deletion_audio_reviewed": False, "utterance_uid": f"utt-{text}",
             "speaker_id": "speaker-a"}
            for start, end, text in (
                (400, 900, "Context"), (1000, 2500, "Alpha"), (1200, 2800, "Bravo"),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            cold = align_corrected_segments(
                audio, coarse, whisperx_module=_ResidualEarlyStopWhisperX(),
                checkpoint_dir=checkpoint,
            )
            fake = _ResidualEarlyStopWhisperX()
            with patch.object(forced_align, "_normalize_aligned_words_uncached",
                              side_effect=AssertionError("unchanged candidate normalized again")):
                warm = align_corrected_segments(
                    audio, coarse, whisperx_module=fake, checkpoint_dir=checkpoint,
                )
        self.assertEqual(fake.align_calls, [])
        self.assertEqual(cold, warm)
        validate_forced_alignment_data(warm)

    def test_residual_selection_checkpoint_binds_options_and_live_neighbors(self):
        def word(uid, start, end):
            return {"utterance_uid": uid, "text": uid, "start_ms": start, "end_ms": end}

        options = {
            "a": [("independent", [word("a", 1000, 2000)]),
                  ("pair-edge-right", [word("a", 1000, 1400)])],
            "b": [("independent", [word("b", 1500, 2200)])],
        }
        selected = {"a": options["a"][0][1], "b": options["b"][0][1],
                    "c": [word("c", 32000, 33000)]}
        with tempfile.TemporaryDirectory() as directory:
            journal = forced_align.UnitJournal(Path(directory), {"policy": "fixture"})
            cold = forced_align._strict_existing_option_selection(
                ["a", "b"], options, selected, {"a": 0, "b": 1, "c": 2},
                checkpoint_journal=journal,
            )
            with patch.object(forced_align, "_strict_existing_option_selection_uncached",
                              wraps=forced_align._strict_existing_option_selection_uncached) as search:
                warm = forced_align._strict_existing_option_selection(
                    ["a", "b"], options, selected, {"a": 0, "b": 1, "c": 2},
                    checkpoint_journal=journal,
                )
                self.assertEqual(search.call_count, 0)
                self.assertEqual(cold, warm)
                selected["c"] = [word("c", 1600, 1700)]
                self.assertIsNone(forced_align._strict_existing_option_selection(
                    ["a", "b"], options, selected, {"a": 0, "b": 1, "c": 2},
                    checkpoint_journal=journal,
                ))
                self.assertEqual(search.call_count, 1)
                self.assertIsNone(forced_align._strict_existing_option_selection(
                    ["a", "b"], options, selected, {"a": 0, "b": 1, "c": 2},
                    checkpoint_journal=journal,
                ))
                self.assertEqual(search.call_count, 1)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_conflict_call_budget_persists_exact_uids_and_blocks_unchanged_retry(self, _model):
        class CountingOverlap(_OverlapWhisperX):
            def align(self, transcript, *args, **kwargs):
                self.align_calls.append(transcript)
                return super().align(transcript, *args, **kwargs)

        coarse = self._two_pair_coarse()[:2]
        for item, text in zip(coarse, ("Merhaba.", "Selam.")):
            item["text"] = item["asr_text"] = text
            item.pop("speaker_id")
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            fake = CountingOverlap(joint_resolves=False)
            with patch.object(forced_align, "MAX_CONFLICT_ALIGNMENT_CALLS", 2):
                with self.assertRaises(forced_align.AlignmentConflictBlocked) as failed:
                    align_corrected_segments(audio, coarse, whisperx_module=fake,
                                             checkpoint_dir=checkpoint)
                self.assertEqual(len(fake.align_calls), 4)
                details = failed.exception.details
                self.assertEqual(details["component_uids"], ["utt-alpha", "utt-bravo"])
                self.assertEqual(details["work"]["alignment_calls"], 2)
                self.assertEqual(details["reason"], "conflict_alignment_budget")
                receipt = json.loads((checkpoint / "components" / "latest-conflict-failure.json")
                                     .read_text(encoding="utf-8"))
                self.assertEqual(receipt["sha256"], forced_align.digest(receipt["data"]))
                resumed = CountingOverlap(joint_resolves=False)
                with self.assertRaises(forced_align.AlignmentConflictBlocked) as repeated:
                    align_corrected_segments(audio, coarse, whisperx_module=resumed,
                                             checkpoint_dir=checkpoint)
                self.assertEqual(resumed.align_calls, [])
                self.assertEqual(repeated.exception.details["failure_invariant"],
                                 details["failure_invariant"])

    def test_conflict_time_budget_stops_before_first_fresh_recovery_call(self):
        source = [{"utterance_uid": uid, "text": uid, "start_ms": 0, "end_ms": 3000,
                   "coarse_start_ms": 1000, "coarse_end_ms": 2500} for uid in ("a", "b")]
        independent = {uid: [{"utterance_uid": uid, "text": uid,
                              "start_ms": 1000, "end_ms": 2000}] for uid in ("a", "b")}
        with patch.object(forced_align.time, "monotonic", side_effect=[0.0, 121.0, 121.0]):
            with self.assertRaises(forced_align.AlignmentConflictBlocked) as failed:
                forced_align._resolve_alignment_overlaps(
                    source, independent, align=lambda *args, **kwargs: self.fail("fresh call"),
                    align_model=object(), align_metadata={}, audio=object(), device="cuda",
                    call_kwargs={}, min_word_score=0.3, max_word_duration_ms=2500,
                    max_outward_drift_ms=500, vad_regions=[],
                )
        self.assertEqual(failed.exception.details["reason"], "conflict_time_budget")
        self.assertEqual(failed.exception.details["work"]["alignment_calls"], 0)

    def test_scoped_recovery_budget_resume_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "units"
            journal = forced_align.UnitJournal(checkpoint, {"raw": "fixture"})
            raw_key = forced_align.digest({"call": "cached"})
            raw_result = {"segments": [], "word_segments": []}
            journal.write(raw_key, raw_result)
            source = [
                {
                    "utterance_uid": uid,
                    "start_ms": index * 1000,
                    "end_ms": index * 1000 + 900,
                    "coarse_start_ms": index * 1000,
                    "coarse_end_ms": index * 1000 + 900,
                    "text": uid,
                }
                for index, uid in enumerate(("a", "b", "c"), 1)
            ]
            scope = {
                "stage": "forced_alignment",
                "target_uids": ["b"],
                "context_uids": ["a", "b", "c"],
                "audio_sha256": "a" * 64,
                "model_state_sha256": "b" * 64,
                "raw_alignment_binding": journal.binding,
                "source_sha256": forced_align.digest(source),
            }
            details = {
                "status": "BLOCKED",
                "reason": "recovery_time_budget",
                "component_uids": ["b"],
                "source_sha256": forced_align.digest([source[1]]),
                "raw_results": {raw_key: forced_align.digest(raw_result)},
                "total_recovery_seconds": 900.1,
                "limits": {"total_recovery_seconds": 900},
            }
            failure = checkpoint / "components/latest-conflict-failure.json"
            forced_align.atomic_json(
                failure,
                {"data": details, "sha256": forced_align.digest(details)},
            )

            archive = forced_align._archive_authorized_recovery_failure(
                checkpoint, scope, source, journal
            )

            self.assertFalse(failure.exists())
            saved = json.loads(archive.read_text(encoding="utf-8"))
            self.assertEqual(saved["sha256"], forced_align.digest(details))
            authorization = json.loads(
                archive.with_suffix(".authorization.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                authorization["sha256"],
                forced_align.digest(authorization["data"]),
            )

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_independent_context_timeout_is_persisted_and_not_retried(self, _model):
        clock = {"now": 0.0}

        class SlowContext(_FakeWhisperX):
            def align(self, transcript, *args, **kwargs):
                text = transcript[0]["text"]
                self.align_calls.append(text)
                if text == "Alpha Bravo":
                    clock["now"] += 121.0
                    return _result([{"word": "Alpha", "start": 1.1, "end": 1.5},
                                    {"word": "Bravo", "start": 2.1, "end": 2.5}])
                return _result([{"word": text, "start": 1.1 if text == "Alpha" else 2.1,
                                 "end": 1.5 if text == "Alpha" else 2.5,
                                 "score": 0.9 if text == "Alpha" else 0.218}])

        coarse = [{"utterance_uid": uid, "text": text, "asr_text": text,
                   "start_ms": start, "end_ms": end, "deletion_audio_reviewed": False}
                  for uid, text, start, end in (("a", "Alpha", 1000, 2000),
                                               ("b", "Bravo", 2000, 3000))]
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            with patch.object(forced_align.time, "monotonic", side_effect=lambda: clock["now"]):
                fake = SlowContext([])
                with self.assertRaises(forced_align.AlignmentConflictBlocked) as failed:
                    align_corrected_segments(audio, coarse, whisperx_module=fake,
                                             checkpoint_dir=checkpoint)
                self.assertEqual(fake.align_calls, ["Alpha", "Bravo", "Alpha Bravo"])
                self.assertEqual(failed.exception.details["phase"], "independent-context")
                self.assertEqual(failed.exception.details["component_uids"], ["b"])
                resumed = SlowContext([])
                with self.assertRaises(forced_align.AlignmentConflictBlocked):
                    align_corrected_segments(audio, coarse, whisperx_module=resumed,
                                             checkpoint_dir=checkpoint)
                self.assertEqual(resumed.align_calls, [])

    def test_conflict_search_budget_is_shared_across_candidate_attempts(self):
        coarse = self._two_pair_coarse()[:2]
        for item, text in zip(coarse, ("Merhaba.", "Selam.")):
            item["text"] = item["asr_text"] = text
            item.pop("speaker_id")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(forced_align, "MAX_CONFLICT_SEARCH_CHECKS", 1):
                with self.assertRaises(forced_align.AlignmentConflictBlocked) as failed:
                    align_corrected_segments(self._audio(directory), coarse,
                                             whisperx_module=_OverlapWhisperX(joint_resolves=True))
        self.assertEqual(failed.exception.details["reason"], "conflict_search_budget")
        self.assertEqual(failed.exception.details["work"]["search_checks"], 1)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_resume_scope_allows_cached_neighbors_but_blocks_uncached_outside_target(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            coarse = self._two_pair_coarse()
            cold = align_corrected_segments(audio, coarse, whisperx_module=_TwoPairOverlapWhisperX(),
                                            checkpoint_dir=checkpoint)
            identity = json.loads((checkpoint / "resume-identity.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["sha256"], forced_align.digest(identity["data"]))
            scope = {key: identity["data"][key] for key in (
                "stage", "audio_sha256", "model_state_sha256", "raw_alignment_binding", "source_sha256",
            )}
            scope.update(target_uids=["utt-alpha"], context_uids=["utt-alpha", "utt-bravo"])
            for missing_text, allowed in (("Alpha", True), ("Charlie", False)):
                for path in checkpoint.glob("*/*.json"):
                    record = json.loads(path.read_text(encoding="utf-8"))
                    words = record["data"]["result"].get("word_segments")
                    if words and [word["word"] for word in words] == [missing_text]:
                        path.unlink()
                        break
                else:
                    self.fail(f"missing raw fixture {missing_text}")
                for raw_path in checkpoint.glob("*/*.json"):
                    raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
                    raw_result = raw_record.get("data", {}).get("result")
                    if isinstance(raw_result, dict):
                        break
                else:
                    self.fail("missing retained raw alignment fixture")
                source = identity["data"]["source"]
                details = {
                    "status": "BLOCKED",
                    "reason": "recovery_time_budget",
                    "component_uids": ["utt-alpha"],
                    "source_sha256": forced_align.digest([source[0]]),
                    "raw_results": {
                        raw_record["data"]["uid"]: forced_align.digest(raw_result),
                    },
                    "total_recovery_seconds": 900.1,
                    "limits": {"total_recovery_seconds": 900},
                }
                forced_align.atomic_json(
                    checkpoint / "components/latest-conflict-failure.json",
                    {"data": details, "sha256": forced_align.digest(details)},
                )
                fake = _TwoPairOverlapWhisperX()
                if allowed:
                    warm = align_corrected_segments(audio, coarse, whisperx_module=fake,
                                                    checkpoint_dir=checkpoint, resume_scope=scope)
                    self.assertEqual(warm, cold)
                    self.assertEqual(fake.align_calls, [{"text": "Alpha"}])
                else:
                    with self.assertRaisesRegex(forced_align.AlignmentConflictBlocked, "outside authorized"):
                        align_corrected_segments(audio, coarse, whisperx_module=fake,
                                                 checkpoint_dir=checkpoint, resume_scope=scope)
                    self.assertEqual(fake.align_calls, [])

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_resume_scope_allows_signed_bounded_continuation_component(self, _model):
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            coarse = self._two_pair_coarse()
            cold = align_corrected_segments(
                audio,
                coarse,
                whisperx_module=_TwoPairOverlapWhisperX(),
                checkpoint_dir=checkpoint,
            )
            identity = json.loads(
                (checkpoint / "resume-identity.json").read_text(encoding="utf-8")
            )["data"]
            scope = {
                key: identity[key]
                for key in (
                    "stage",
                    "audio_sha256",
                    "model_state_sha256",
                    "raw_alignment_binding",
                    "source_sha256",
                )
            }
            scope.update(
                target_uids=["utt-alpha"],
                context_uids=["utt-alpha", "utt-bravo"],
                continuation_component_limit=1,
                recovery_seconds_limit=1800,
            )
            retained = None
            for path in checkpoint.glob("*/*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                result = record.get("data", {}).get("result")
                words = result.get("word_segments") if isinstance(result, dict) else None
                if words and [word["word"] for word in words] == ["Charlie", "Delta"]:
                    path.unlink()
                elif retained is None and isinstance(result, dict):
                    retained = (record["data"]["uid"], result)
            self.assertIsNotNone(retained)
            source = identity["source"]
            details = {
                "status": "BLOCKED",
                "reason": "recovery_time_budget",
                "component_uids": ["utt-alpha"],
                "unresolved_components": [
                    ["utt-alpha"],
                    ["utt-charlie", "utt-delta"],
                ],
                "source_sha256": forced_align.digest([source[0]]),
                "raw_results": {retained[0]: forced_align.digest(retained[1])},
                "total_recovery_seconds": 900.1,
                "limits": {"total_recovery_seconds": 900},
            }
            forced_align.atomic_json(
                checkpoint / "components/latest-conflict-failure.json",
                {"data": details, "sha256": forced_align.digest(details)},
            )
            fake = _TwoPairOverlapWhisperX()

            warm = align_corrected_segments(
                audio,
                coarse,
                whisperx_module=fake,
                checkpoint_dir=checkpoint,
                resume_scope=scope,
            )

            self.assertEqual(warm, cold)
            self.assertEqual(fake.align_calls, [{"text": "Charlie Delta"}])
            receipt = json.loads(
                (checkpoint / "components/authorized-continuation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                receipt["sha256"], forced_align.digest(receipt["data"])
            )
            self.assertEqual(
                receipt["data"]["requests"][0]["component_uids"],
                ["utt-charlie", "utt-delta"],
            )

    def test_generic_resume_context_never_joins_words_across_thirty_second_silence(self):
        for words in (("Zeytin", "Gökyüzü"), ("Buralarda", "Dönüyoruz")):
            source = forced_align.validate_coarse_segments([
                {"utterance_uid": uid, "start_ms": start, "end_ms": end,
                 "text": word, "asr_text": word, "deletion_audio_reviewed": False}
                for uid, word, start, end in (("a", words[0], 1000, 2000),
                                             ("b", words[1], 32000, 33000))
            ])
            identity = {key: "a" * 64 for key in (
                "audio_sha256", "model_state_sha256", "raw_alignment_binding", "source_sha256",
            )}
            groups = forced_align._alignment_resume_groups(
                {"stage": "forced_alignment", "target_uids": ["a", "b"],
                 "context_uids": ["a", "b"], **identity}, source, identity,
            )
            self.assertEqual([group["text"] for group in groups],
                             [forced_align._alignment_model_text(word) for word in words])
            with self.assertRaisesRegex(ForcedAlignmentError, "unrelated scene"):
                forced_align._alignment_resume_groups(
                    {"stage": "forced_alignment", "target_uids": ["a"],
                     "context_uids": ["a", "b"], **identity}, source, identity,
                )

    def test_resume_scope_allows_active_target_to_realign_exact_context_cue(self):
        source = forced_align.validate_coarse_segments([
            {"utterance_uid": uid, "start_ms": start, "end_ms": end,
             "text": text, "asr_text": text, "deletion_audio_reviewed": False}
            for uid, text, start, end in (
                ("before", "Once", 1000, 1400),
                ("target", "Hedef", 1500, 1900),
                ("after", "Sonra", 2000, 2400),
            )
        ])
        identity = {key: "a" * 64 for key in (
            "audio_sha256", "model_state_sha256", "raw_alignment_binding", "source_sha256",
        )}
        scope = {
            "stage": "forced_alignment",
            "target_uids": ["target"],
            "context_uids": ["before", "target", "after"],
            **identity,
        }
        request = [{"start": 1.0, "end": 1.4, "text": "Once"}]

        self.assertTrue(forced_align._alignment_request_matches_scope(
            request, ["target"], source, scope, identity,
        ))
        self.assertFalse(forced_align._alignment_request_matches_scope(
            request, ["before"], source, scope, identity,
        ))

    def test_resume_scope_allows_full_signed_context_for_adjacent_targets(self):
        source = forced_align.validate_coarse_segments([
            {"utterance_uid": f"u{index:02d}", "start_ms": index * 500,
             "end_ms": index * 500 + 400, "text": f"Soz{index}",
             "asr_text": f"Soz{index}", "deletion_audio_reviewed": False}
            for index in range(18)
        ])
        identity = {key: "a" * 64 for key in (
            "audio_sha256", "model_state_sha256", "raw_alignment_binding", "source_sha256",
        )}
        scope = {
            "stage": "forced_alignment",
            "target_uids": ["u08", "u09"],
            "context_uids": [f"u{index:02d}" for index in range(18)],
            **identity,
        }
        request = [{
            "start": 0.0,
            "end": 8.9,
            "text": forced_align._alignment_model_text(
                " ".join(item["text"] for item in source)
            ),
        }]

        self.assertTrue(forced_align._alignment_request_matches_scope(
            request, ["u08", "u09"], source, scope, identity,
        ))

    def test_signed_continuation_can_merge_only_uids_from_its_exact_context(self):
        source = forced_align.validate_coarse_segments([
            {"utterance_uid": uid, "start_ms": index * 500,
             "end_ms": index * 500 + 400, "text": uid,
             "asr_text": uid, "deletion_audio_reviewed": False}
            for index, uid in enumerate(("before", "a", "b", "c", "outside"))
        ])
        scope = {
            "stage": "forced_alignment",
            "target_uids": ["a", "b", "c"],
            "context_uids": ["before", "a", "b", "c"],
            **{key: "a" * 64 for key in (
                "audio_sha256", "model_state_sha256", "raw_alignment_binding",
                "source_sha256",
            )},
        }

        expanded = forced_align._expanded_continuation_scope(
            ["before", "a", "b", "c"], scope, source,
        )

        self.assertEqual(expanded["target_uids"], ["before", "a", "b", "c"])
        self.assertIsNone(forced_align._expanded_continuation_scope(
            ["a", "b", "c", "outside"], scope, source,
        ))

    def test_resume_request_metadata_allows_only_bounded_component_windows(self):
        source = forced_align.validate_coarse_segments(self._two_pair_coarse())
        identity = {
            "audio_sha256": "a" * 64,
            "model_state_sha256": "b" * 64,
            "raw_alignment_binding": "c" * 64,
            "source_sha256": forced_align.digest(source),
        }
        scope = {
            "stage": "forced_alignment",
            "target_uids": ["utt-alpha"],
            "context_uids": ["utt-alpha", "utt-bravo"],
            **identity,
        }
        alpha = source[0]
        expanded = [{
            "start": source[0]["start_ms"] / 1000.0,
            "end": source[1]["end_ms"] / 1000.0,
            "text": forced_align._alignment_model_text(alpha["text"]),
        }]
        self.assertTrue(forced_align._alignment_request_matches_scope(
            expanded, ["utt-alpha"], source, scope, identity
        ))
        self.assertFalse(forced_align._alignment_request_matches_scope(
            expanded, ["utt-charlie"], source, scope, identity
        ))
        discovered = {
            **scope,
            "target_uids": ["utt-charlie"],
            "context_uids": ["utt-charlie", "utt-delta"],
        }
        charlie = source[2]
        request = [{
            "start": charlie["start_ms"] / 1000.0,
            "end": source[3]["end_ms"] / 1000.0,
            "text": forced_align._alignment_model_text(charlie["text"]),
        }]
        self.assertTrue(forced_align._alignment_request_matches_scope(
            request, ["utt-charlie"], source, discovered, identity
        ))

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_interrupted_alignment_reuses_only_hash_bound_model_calls(self, model_hash):
        results = [
            _result([
                {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                {"word": "d\u00fcnya!", "start": 1.5, "end": 1.9},
            ]),
            _result([{"word": "Nas\u0131ls\u0131n?", "start": 4.1, "end": 4.7}]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            interrupted = _FakeWhisperX(results[:1])
            with self.assertRaisesRegex(ForcedAlignmentError, "alignment failed for utt-2"):
                align_corrected_segments(audio, _coarse(), whisperx_module=interrupted,
                                         checkpoint_dir=checkpoint)
            resumed = _FakeWhisperX(results[1:])
            data = align_corrected_segments(audio, _coarse(), whisperx_module=resumed,
                                            checkpoint_dir=checkpoint)
            self.assertEqual(len(resumed.align_calls), 1)
            validate_forced_alignment_data(data)
            self.assertEqual(data["provenance"]["model_state_sha256"], "a" * 64)
            model_hash.return_value = "b" * 64
            changed_model = _FakeWhisperX(results)
            align_corrected_segments(audio, _coarse(), whisperx_module=changed_model,
                                     checkpoint_dir=checkpoint)
            self.assertEqual(len(changed_model.align_calls), 2)
            audio.write_bytes(b"different immutable input")
            fresh = _FakeWhisperX(results)
            align_corrected_segments(audio, _coarse(), whisperx_module=fresh,
                                     checkpoint_dir=checkpoint)
            self.assertEqual(len(fresh.align_calls), 2)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_selection_only_change_reuses_raw_alignment_calls(self, _model_hash):
        results = [
            _result([
                {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                {"word": "dünya!", "start": 1.5, "end": 1.9},
            ]),
            _result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            align_corrected_segments(
                audio,
                _coarse(),
                whisperx_module=_FakeWhisperX(results),
                checkpoint_dir=checkpoint,
            )
            original = forced_align._resolve_alignment_overlaps
            reused = _FakeWhisperX([])
            with patch(
                "mas.engine.forced_align._resolve_alignment_overlaps",
                wraps=original,
            ) as selection:
                data = align_corrected_segments(
                    audio,
                    _coarse(),
                    whisperx_module=reused,
                    checkpoint_dir=checkpoint,
                )

        self.assertEqual(reused.align_calls, [])
        selection.assert_called_once()
        validate_forced_alignment_data(data)

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_checkpointed_alignment_marks_each_ctc_call(self, _model_hash):
        results = [
            _result([
                {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                {"word": "dünya!", "start": 1.5, "end": 1.9},
            ]),
            _result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}]),
        ]
        marks = []
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "mas.engine.forced_align.mark_work_progress",
                side_effect=lambda stage, **kwargs: marks.append((stage, kwargs)),
            ):
                align_corrected_segments(
                    self._audio(directory),
                    _coarse(),
                    whisperx_module=_FakeWhisperX(results),
                    checkpoint_dir=Path(directory) / "units",
                )

        self.assertEqual(
            [
                kwargs["completed"]
                for stage, kwargs in marks
                if stage == "forced_alignment:ctc"
            ],
            [1, 2],
        )

    @patch("mas.engine.forced_align._model_state_sha256", return_value="a" * 64)
    def test_changed_raw_alignment_producer_invalidates_cached_calls(
        self, _model_hash
    ):
        results = [
            _result([
                {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                {"word": "dünya!", "start": 1.5, "end": 1.9},
            ]),
            _result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio = self._audio(directory)
            checkpoint = Path(directory) / "units"
            with patch(
                "mas.engine.forced_align._raw_alignment_producer_sha256",
                return_value="a" * 64,
            ):
                align_corrected_segments(
                    audio,
                    _coarse(),
                    whisperx_module=_FakeWhisperX(results),
                    checkpoint_dir=checkpoint,
                )
            changed = _FakeWhisperX(results)
            with patch(
                "mas.engine.forced_align._raw_alignment_producer_sha256",
                return_value="b" * 64,
            ):
                align_corrected_segments(
                    audio,
                    _coarse(),
                    whisperx_module=changed,
                    checkpoint_dir=checkpoint,
                )

        self.assertEqual(len(changed.align_calls), 2)

    def test_raw_alignment_producer_identity_excludes_selection_helpers(self):
        expected = forced_align._raw_alignment_producer_sha256()
        with patch("mas.engine.forced_align._resolve_alignment_overlaps"):
            self.assertEqual(
                forced_align._raw_alignment_producer_sha256(), expected
            )

        getsource = forced_align.inspect.getsource

        def changed_source(function):
            source = getsource(function)
            if function is forced_align._alignment_model_text:
                return source + "\n# changed producer"
            return source

        with patch(
            "mas.engine.forced_align.inspect.getsource",
            side_effect=changed_source,
        ):
            self.assertNotEqual(
                forced_align._raw_alignment_producer_sha256(), expected
            )

    def test_model_digest_binds_weights_metadata_and_config(self):
        from types import SimpleNamespace
        import struct
        from mas.engine.forced_align import _model_state_sha256

        class Tensor:
            def __init__(self, values):
                self.values = values
                self.dtype = "float32"
                self.shape = (len(values),)

            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self
            def tobytes(self): return struct.pack(f"<{len(self.values)}f", *self.values)

        state = {"weight": Tensor([1, 2])}
        config = {"vocab_size": 2}
        model = SimpleNamespace(state_dict=lambda: state,
                                config=SimpleNamespace(to_dict=lambda: config))
        metadata = {"language": "tr", "dictionary": {"a": 1}}
        original = _model_state_sha256(model, metadata)
        self.assertEqual(original, _model_state_sha256(model, metadata))
        state["weight"] = Tensor([1, 3])
        self.assertNotEqual(original, _model_state_sha256(model, metadata))
        state["weight"] = Tensor([1, 2])
        config["vocab_size"] = 3
        self.assertNotEqual(original, _model_state_sha256(model, metadata))
        config["vocab_size"] = 2
        metadata["dictionary"]["a"] = 2
        self.assertNotEqual(original, _model_state_sha256(model, metadata))
        with self.assertRaisesRegex(ForcedAlignmentError, "actual model weights"):
            _model_state_sha256(object(), metadata)

    def test_explicit_cross_speaker_overlap_is_preserved(self) -> None:
        coarse = [
            {
                "start_ms": 1000,
                "end_ms": 2500,
                "text": "Merhaba.",
                "asr_text": "Merhaba.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-a",
                "speaker_id": "speaker-a",
            },
            {
                "start_ms": 1200,
                "end_ms": 2800,
                "text": "Selam.",
                "asr_text": "Selam.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-b",
                "speaker_id": "speaker-b",
            },
        ]
        fake = _FakeWhisperX(
            [
                _result([{"word": "Merhaba.", "start": 1.1, "end": 2.0}]),
                _result([{"word": "Selam.", "start": 1.3, "end": 2.2}]),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )

        self.assertEqual(
            [segment["speaker_id"] for segment in data["segments"]],
            ["speaker-a", "speaker-b"],
        )
        self.assertEqual(
            [word["speaker_id"] for word in data["words"]],
            ["speaker-a", "speaker-b"],
        )
        validate_forced_alignment_data(data)

    def test_large_overlap_component_hits_budget_before_exponential_search(self):
        coarse = [
            {"start_ms": 1000, "end_ms": 2500, "text": "Merhaba.",
             "asr_text": "Merhaba.", "deletion_audio_reviewed": False,
             "utterance_uid": f"utt-{i}"}
            for i in range(22)
        ]
        independent = _result([{"word": "Merhaba.", "start": 1.1, "end": 2.0}])
        joint = _result([
            {"word": "Merhaba.", "start": 1.1, "end": 1.4} for _ in coarse
        ])
        fake = _FakeWhisperX([independent] * len(coarse) + [joint])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "4194304 combinations.*limit 2097152"
            ):
                align_corrected_segments(self._audio(directory), coarse, whisperx_module=fake)
        self.assertEqual(len(fake.align_calls), 23)

    def test_all_independent_utterance_failures_are_reported_in_one_pass(self):
        fake = _FakeWhisperX([
            _result([{"word": "Merhaba,", "start": 1.1, "end": 1.4, "score": 0.01}]),
            _result([{"word": "Nas\u0131ls\u0131n?", "start": 4.1, "end": 4.7, "score": 0.01}]),
        ])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ForcedAlignmentError) as raised:
                align_corrected_segments(self._audio(directory), _coarse(), whisperx_module=fake)
        self.assertIn("failed for 2 utterances", str(raised.exception))
        self.assertIn("utt-1", str(raised.exception))
        self.assertIn("utt-2", str(raised.exception))
        self.assertEqual(len(fake.align_calls), 3)

    def test_low_independent_score_recovers_from_disjoint_joint_context(self):
        coarse = [
            {
                "start_ms": 500,
                "end_ms": 1_500,
                "text": "Once",
                "asr_text": "Once",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-before",
            },
            {
                "start_ms": 1_500,
                "end_ms": 3_000,
                "text": "Beni anliyor",
                "asr_text": "Beni anliyor",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-target",
            },
            {
                "start_ms": 3_000,
                "end_ms": 4_000,
                "text": "Sonra",
                "asr_text": "Sonra",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-after",
            },
        ]
        fake = _FakeWhisperX(
            [
                _result([{"word": "Once", "start": 0.6, "end": 1.0}]),
                _result([
                    {"word": "Beni", "start": 1.7, "end": 2.0},
                    {"word": "anliyor", "start": 2.1, "end": 2.6, "score": 0.218},
                ]),
                _result([{"word": "Sonra", "start": 3.2, "end": 3.6}]),
                _result([
                    {"word": "Once", "start": 0.6, "end": 1.0},
                    {"word": "Beni", "start": 1.7, "end": 2.0},
                    {"word": "anliyor", "start": 2.1, "end": 2.6},
                ]),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )

        self.assertEqual(len(fake.align_calls), 4)
        self.assertEqual(
            data["provenance"]["overlap_resolution"]["selected_mode_counts"],
            {"independent": 2, "joint-recovery": 1},
        )
        self.assertEqual(data["segments"][1]["end_ms"], 2_600)
        validate_forced_alignment_data(data)

    def test_low_score_recovery_does_not_serialize_overlapping_context(self):
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 2_500,
                "text": "Merhaba",
                "asr_text": "Merhaba",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-a",
                "speaker_id": "speaker-a",
            },
            {
                "start_ms": 1_200,
                "end_ms": 2_800,
                "text": "Anliyor",
                "asr_text": "Anliyor",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-b",
                "speaker_id": "speaker-b",
            },
        ]
        fake = _FakeWhisperX(
            [
                _result([{"word": "Merhaba", "start": 1.1, "end": 2.0}]),
                _result([
                    {"word": "Anliyor", "start": 1.3, "end": 2.2, "score": 0.218}
                ]),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "below the required minimum"):
                align_corrected_segments(
                    self._audio(directory), coarse, whisperx_module=fake
                )
        self.assertEqual(len(fake.align_calls), 2)

    def test_joint_alignment_resolves_unknown_speaker_overlap(self) -> None:
        coarse = [
            {
                "start_ms": 1000,
                "end_ms": 2500,
                "text": "Merhaba.",
                "asr_text": "Merhaba.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-a",
            },
            {
                "start_ms": 1200,
                "end_ms": 2800,
                "text": "Selam.",
                "asr_text": "Selam.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-b",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                forced_align,
                "_overlap_components",
                wraps=forced_align._overlap_components,
            ) as components:
                data = align_corrected_segments(
                    self._audio(directory),
                    coarse,
                    whisperx_module=_OverlapWhisperX(joint_resolves=True),
                )

        resolution = data["provenance"]["overlap_resolution"]
        self.assertEqual(components.call_count, 3)
        self.assertEqual(resolution["initial_overlap_count"], 1)
        self.assertEqual(resolution["final_overlap_count"], 0)
        self.assertEqual(resolution["acoustic_component_count"], 0)
        self.assertNotIn("speaker_id", data["segments"][0])
        validate_forced_alignment_data(data)

        forged = copy.deepcopy(data)
        forged["provenance"]["overlap_resolution"]["acoustic_component_count"] = 1
        forged["provenance"]["overlap_resolution"]["acoustic_components"] = [
            {"component_index": 1, "lanes": [
                {"utterance_uid": "utt-a", "speaker_id": "acoustic-overlap-0001-lane-01"},
                {"utterance_uid": "utt-b", "speaker_id": "acoustic-overlap-0001-lane-02"},
            ]}
        ]
        with self.assertRaisesRegex(ForcedAlignmentError, "not independent speaker evidence"):
            validate_forced_alignment_data(forged)

        with tempfile.TemporaryDirectory() as directory:
            with patch("mas.engine.forced_align.MAX_OVERLAP_COMBINATIONS", 1):
                with self.assertRaisesRegex(ForcedAlignmentError, "candidate budget exceeded"):
                    align_corrected_segments(
                        self._audio(directory), coarse,
                        whisperx_module=_OverlapWhisperX(joint_resolves=True),
                    )

    def test_matching_joint_candidates_are_selected_atomically(self) -> None:
        coarse = [
            {
                "start_ms": 1000,
                "end_ms": 2500,
                "text": "Merhaba.",
                "asr_text": "Merhaba.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-a",
                "speaker_id": "speaker-a",
            },
            {
                "start_ms": 1200,
                "end_ms": 2800,
                "text": "Selam.",
                "asr_text": "Selam.",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-b",
                "speaker_id": "speaker-a",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("mas.engine.forced_align.MAX_OVERLAP_COMBINATIONS", 1):
                data = align_corrected_segments(
                    self._audio(directory),
                    coarse,
                    whisperx_module=_OverlapWhisperX(joint_resolves=True),
                )

        resolution = data["provenance"]["overlap_resolution"]
        self.assertEqual(resolution["selected_mode_counts"], {"joint": 2})
        self.assertEqual(resolution["final_overlap_count"], 0)
        validate_forced_alignment_data(data)

    def test_atomic_selection_ignores_other_disconnected_component(self) -> None:
        coarse = [
            {
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "asr_text": text,
                "deletion_audio_reviewed": False,
                "utterance_uid": f"utt-{text.lower()}",
                "speaker_id": "speaker-a",
            }
            for start, end, text in (
                (1000, 2500, "Alpha"),
                (1200, 2800, "Bravo"),
                (10000, 11500, "Charlie"),
                (10200, 11800, "Delta"),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory),
                coarse,
                whisperx_module=_TwoPairOverlapWhisperX(),
            )

        resolution = data["provenance"]["overlap_resolution"]
        self.assertEqual(resolution["selected_mode_counts"], {"joint": 4})
        self.assertEqual(resolution["final_overlap_count"], 0)
        validate_forced_alignment_data(data)

    def test_valid_unknown_joint_subset_is_selected_atomically(self) -> None:
        coarse = [
            {
                "start_ms": 1000,
                "end_ms": 2500,
                "text": "Alpha",
                "asr_text": "Alpha",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-alpha",
            },
            {
                "start_ms": 1200,
                "end_ms": 2800,
                "text": "Bravo",
                "asr_text": "Bravo",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-bravo",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("mas.engine.forced_align.MAX_OVERLAP_COMBINATIONS", 1):
                data = align_corrected_segments(
                    self._audio(directory),
                    coarse,
                    whisperx_module=_PartialJointOverlapWhisperX(),
                )

        resolution = data["provenance"]["overlap_resolution"]
        self.assertEqual(
            resolution["selected_mode_counts"],
            {"independent": 1, "joint": 1},
        )
        self.assertEqual(resolution["final_overlap_count"], 0)
        validate_forced_alignment_data(data)

    def test_residual_overlap_uses_adjacent_joint_context(self) -> None:
        coarse = [
            {
                "start_ms": 400,
                "end_ms": 900,
                "text": "Once",
                "asr_text": "Once",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-context",
            },
            {
                "start_ms": 1000,
                "end_ms": 3000,
                "text": "Ben ona ne yaptim",
                "asr_text": "Ben ona ne yaptim",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-repeat-a",
            },
            {
                "start_ms": 1100,
                "end_ms": 3100,
                "text": "Hicbir sey yapmadim",
                "asr_text": "Hicbir sey yapmadim",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-middle",
            },
            {
                "start_ms": 2000,
                "end_ms": 3500,
                "text": "Ben ona ne yaptim",
                "asr_text": "Ben ona ne yaptim",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-repeat-b",
            },
        ]
        fake = _ContextualOverlapWhisperX()
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory),
                coarse,
                whisperx_module=fake,
            )

        self.assertEqual(
            [(segment["start_ms"], segment["end_ms"]) for segment in data["segments"]],
            [(450, 750), (1100, 1500), (2300, 2600), (2655, 2800)],
        )
        self.assertIn(
            "Once Ben ona ne yaptim Hicbir sey yapmadim Ben ona ne yaptim",
            [call["text"] for call in fake.align_calls],
        )
        self.assertIn(
            "Hicbir sey yapmadim Ben ona ne yaptim",
            [call["text"] for call in fake.align_calls],
        )
        self.assertEqual(
            [call["text"] for call in fake.align_calls].count(
                "Hicbir sey yapmadim Ben ona ne yaptim"
            ),
            1,
        )
        resolution = data["provenance"]["overlap_resolution"]
        self.assertEqual(resolution["selected_mode_counts"], {"joint": 4})
        self.assertEqual(resolution["final_overlap_count"], 0)
        validate_forced_alignment_data(data)

        partial_fake = _ContextualOverlapWhisperX(invalid_context=True)
        with tempfile.TemporaryDirectory() as directory:
            partial = align_corrected_segments(
                self._audio(directory),
                coarse,
                whisperx_module=partial_fake,
            )

        self.assertEqual(
            partial["provenance"]["overlap_resolution"]["selected_mode_counts"],
            {"independent": 1, "joint": 3},
        )
        self.assertEqual(
            [(segment["start_ms"], segment["end_ms"]) for segment in partial["segments"]],
            [(500, 800), (1100, 1500), (2300, 2600), (2655, 2800)],
        )
        validate_forced_alignment_data(partial)

    def test_residual_joint_recovery_stops_after_component_is_resolved(self) -> None:
        coarse = [
            {
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "asr_text": text,
                "deletion_audio_reviewed": False,
                "utterance_uid": f"utt-{text.lower()}",
                "speaker_id": "speaker-a",
            }
            for start, end, text in (
                (400, 900, "Context"),
                (1000, 2500, "Alpha"),
                (1200, 2800, "Bravo"),
            )
        ]
        fake = _ResidualEarlyStopWhisperX()
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory),
                coarse,
                whisperx_module=fake,
            )

        texts = [call["text"] for call in fake.align_calls]
        self.assertEqual(texts.count("Context Alpha Bravo"), 1)
        self.assertNotIn("Context Alpha", texts)
        self.assertEqual(
            data["provenance"]["overlap_resolution"]["final_overlap_count"], 0
        )
        validate_forced_alignment_data(data)

    def test_over_budget_context_search_keeps_neighbor_options(self) -> None:
        coarse = [
            {
                "start_ms": 0,
                "end_ms": 6000,
                "coarse_start_ms": start,
                "coarse_end_ms": end,
                "text": text,
                "asr_text": text,
                "deletion_audio_reviewed": False,
                "utterance_uid": f"utt-{text.lower()}",
            }
            for text, start, end in (
                ("A", 1000, 2500),
                ("B", 1200, 2800),
                ("C", 2300, 3300),
                ("D", 4000, 4500),
                ("E", 5000, 5500),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("mas.engine.forced_align.MAX_OVERLAP_COMBINATIONS", 5):
                data = align_corrected_segments(
                    self._audio(directory),
                    coarse,
                    whisperx_module=_OverBudgetContextWhisperX(),
                )

        self.assertEqual(
            data["provenance"]["overlap_resolution"]["final_overlap_count"], 0
        )
        self.assertEqual(
            [(segment["start_ms"], segment["end_ms"]) for segment in data["segments"]],
            [(2400, 2500), (1300, 2200), (3000, 3200), (4200, 4250), (5100, 5300)],
        )
        validate_forced_alignment_data(data)

    def test_edge_partition_preserves_long_cue_tail_context(self) -> None:
        source = [
            {
                "start_ms": 500,
                "end_ms": 5400,
                "coarse_start_ms": 1000,
                "coarse_end_ms": 5000,
                "text": "Left",
                "utterance_uid": "utt-left",
            },
            {
                "start_ms": 4500,
                "end_ms": 5900,
                "coarse_start_ms": 5000,
                "coarse_end_ms": 5500,
                "text": "Right",
                "utterance_uid": "utt-right",
            },
        ]
        independent = {
            "utt-left": [
                {
                    "utterance_uid": "utt-left",
                    "word": "Left",
                    "start_ms": 4800,
                    "end_ms": 5200,
                }
            ],
            "utt-right": [
                {
                    "utterance_uid": "utt-right",
                    "word": "Right",
                    "start_ms": 5100,
                    "end_ms": 5300,
                }
            ],
        }
        edge_windows = {
            "utt-left": (500, 5000, 4800, 4950),
            "utt-right": (5000, 5900, 5050, 5300),
        }
        calls = []

        def candidate(item, *, window_start_ms, window_end_ms, **kwargs):
            calls.append((item["utterance_uid"], window_start_ms, window_end_ms))
            expected_start, expected_end, word_start, word_end = edge_windows[
                item["utterance_uid"]
            ]
            if (window_start_ms, window_end_ms) != (expected_start, expected_end):
                raise ForcedAlignmentError("synthetic non-edge candidate rejected")
            return [
                {
                    "utterance_uid": item["utterance_uid"],
                    "word": item["text"],
                    "start_ms": word_start,
                    "end_ms": word_end,
                }
            ]

        with patch("mas.engine.forced_align._alignment_candidate", side_effect=candidate):
            selected, _, resolution = forced_align._resolve_alignment_overlaps(
                source,
                independent,
                align=lambda *args, **kwargs: _result(
                    [{"word": "Left", "start": 4.8, "end": 5.2}]
                ),
                align_model=object(),
                align_metadata={},
                audio=object(),
                device="cuda",
                call_kwargs={},
                min_word_score=DEFAULT_MIN_WORD_SCORE,
                max_word_duration_ms=DEFAULT_MAX_WORD_DURATION_MS,
                max_outward_drift_ms=DEFAULT_MAX_OUTWARD_DRIFT_MS,
                vad_regions=[],
            )

        self.assertEqual(resolution["final_overlap_count"], 0)
        self.assertEqual(len(calls), 12)
        self.assertEqual(calls[-2:], [
            ("utt-left", 500, 5000), ("utt-right", 5000, 5900),
        ])
        self.assertEqual(
            [(word["start_ms"], word["end_ms"]) for word in selected["utt-left"]],
            [(4800, 4950)],
        )
        self.assertEqual(
            [(word["start_ms"], word["end_ms"]) for word in selected["utt-right"]],
            [(5100, 5300)],
        )

    def test_pair_edge_excludes_overlapping_coarse_region(self) -> None:
        source = [
            {
                "start_ms": 0,
                "end_ms": 3000,
                "coarse_start_ms": 1000,
                "coarse_end_ms": 2900,
                "text": "Anchor",
                "utterance_uid": "utt-anchor",
            },
            {
                "start_ms": 2000,
                "end_ms": 6000,
                "coarse_start_ms": 3000,
                "coarse_end_ms": 5000,
                "text": "Left",
                "utterance_uid": "utt-left",
            },
            {
                "start_ms": 4500,
                "end_ms": 6500,
                "coarse_start_ms": 4800,
                "coarse_end_ms": 6000,
                "text": "Right",
                "utterance_uid": "utt-right",
            },
        ]
        independent = {
            "utt-anchor": [
                {
                    "utterance_uid": "utt-anchor",
                    "word": "Anchor",
                    "start_ms": 1000,
                    "end_ms": 1500,
                }
            ],
            "utt-left": [
                {
                    "utterance_uid": "utt-left",
                    "word": "Left",
                    "start_ms": 4800,
                    "end_ms": 5200,
                }
            ],
            "utt-right": [
                {
                    "utterance_uid": "utt-right",
                    "word": "Right",
                    "start_ms": 5100,
                    "end_ms": 5500,
                }
            ],
        }
        pair_windows = {
            "utt-left": (2000, 4800, 4500, 4700),
            "utt-right": (5000, 6500, 5100, 5500),
        }

        def candidate(item, *, window_start_ms, window_end_ms, **kwargs):
            if item["utterance_uid"] not in pair_windows:
                raise ForcedAlignmentError("synthetic unrelated candidate rejected")
            expected_start, expected_end, word_start, word_end = pair_windows[
                item["utterance_uid"]
            ]
            if (window_start_ms, window_end_ms) != (expected_start, expected_end):
                raise ForcedAlignmentError("synthetic non-pair-edge candidate rejected")
            return [
                {
                    "utterance_uid": item["utterance_uid"],
                    "word": item["text"],
                    "start_ms": word_start,
                    "end_ms": word_end,
                }
            ]

        with patch("mas.engine.forced_align._alignment_candidate", side_effect=candidate):
            selected, _, resolution = forced_align._resolve_alignment_overlaps(
                source,
                independent,
                align=lambda *args, **kwargs: _result(
                    [{"word": "Left", "start": 4.8, "end": 5.2}]
                ),
                align_model=object(),
                align_metadata={},
                audio=object(),
                device="cuda",
                call_kwargs={},
                min_word_score=DEFAULT_MIN_WORD_SCORE,
                max_word_duration_ms=DEFAULT_MAX_WORD_DURATION_MS,
                max_outward_drift_ms=DEFAULT_MAX_OUTWARD_DRIFT_MS,
                vad_regions=[],
            )

        self.assertEqual(resolution["final_overlap_count"], 0)
        self.assertEqual(
            [(word["start_ms"], word["end_ms"]) for word in selected["utt-left"]],
            [(4500, 4700)],
        )
        self.assertEqual(
            [(word["start_ms"], word["end_ms"]) for word in selected["utt-right"]],
            [(5100, 5500)],
        )

    def test_component_edge_gives_middle_cue_both_incident_cuts(self) -> None:
        source = [
            {
                "start_ms": start,
                "end_ms": end,
                "coarse_start_ms": coarse_start,
                "coarse_end_ms": coarse_end,
                "text": text,
                "utterance_uid": uid,
            }
            for uid, text, start, end, coarse_start, coarse_end in (
                ("utt-left", "Left", 0, 4000, 1000, 3000),
                ("utt-middle", "Middle", 2000, 6000, 2800, 5000),
                ("utt-right", "Right", 4500, 7000, 4800, 6000),
            )
        ]
        independent = {
            "utt-left": [
                {
                    "utterance_uid": "utt-left",
                    "word": "Left",
                    "start_ms": 2700,
                    "end_ms": 3100,
                }
            ],
            "utt-middle": [
                {
                    "utterance_uid": "utt-middle",
                    "word": "Middle",
                    "start_ms": 2900,
                    "end_ms": 5100,
                }
            ],
            "utt-right": [
                {
                    "utterance_uid": "utt-right",
                    "word": "Right",
                    "start_ms": 4900,
                    "end_ms": 5300,
                }
            ],
        }
        component_windows = {
            "utt-left": (0, 2800, 2500, 2700),
            "utt-middle": (3000, 4800, 3300, 4500),
            "utt-right": (5000, 7000, 5100, 5300),
        }
        calls = []

        def candidate(item, *, window_start_ms, window_end_ms, **kwargs):
            uid = item["utterance_uid"]
            calls.append((uid, window_start_ms, window_end_ms))
            expected_start, expected_end, word_start, word_end = component_windows[uid]
            if (window_start_ms, window_end_ms) != (expected_start, expected_end):
                raise ForcedAlignmentError("synthetic non-component candidate rejected")
            return [
                {
                    "utterance_uid": uid,
                    "word": item["text"],
                    "start_ms": word_start,
                    "end_ms": word_end,
                }
            ]

        with patch("mas.engine.forced_align._alignment_candidate", side_effect=candidate):
            selected, _, resolution = forced_align._resolve_alignment_overlaps(
                source,
                independent,
                align=lambda *args, **kwargs: _result(
                    [{"word": "Left", "start": 2.7, "end": 3.1}]
                ),
                align_model=object(),
                align_metadata={},
                audio=object(),
                device="cuda",
                call_kwargs={},
                min_word_score=DEFAULT_MIN_WORD_SCORE,
                max_word_duration_ms=DEFAULT_MAX_WORD_DURATION_MS,
                max_outward_drift_ms=DEFAULT_MAX_OUTWARD_DRIFT_MS,
                vad_regions=[],
            )

        self.assertIn(("utt-middle", 3000, 4800), calls)
        self.assertEqual(calls[-3:], [
            ("utt-left", 0, 2800), ("utt-middle", 3000, 4800),
            ("utt-right", 5000, 7000),
        ])
        self.assertEqual(resolution["final_overlap_count"], 0)
        self.assertEqual(
            [(word["start_ms"], word["end_ms"]) for word in selected["utt-middle"]],
            [(3300, 4500)],
        )

    def test_pair_edge_recomputes_residual_before_next_pair(self):
        source = [
            {"utterance_uid": uid, "text": text, "start_ms": start,
             "end_ms": end, "coarse_start_ms": coarse_start,
             "coarse_end_ms": coarse_end}
            for uid, text, start, end, coarse_start, coarse_end in (
                ("utt-a", "Alpha", 0, 4600, 1000, 4100),
                ("utt-b", "Bravo", 2800, 4500, 3000, 4000),
                ("utt-c", "Charlie", 3500, 5000, 4000, 4500),
            )
        ]
        independent = {
            uid: [{"utterance_uid": uid, "word": text,
                   "start_ms": start, "end_ms": end}]
            for uid, text, start, end in (
                ("utt-a", "Alpha", 2500, 4500),
                ("utt-b", "Bravo", 3100, 3200),
                ("utt-c", "Charlie", 4100, 4200),
            )
        }
        calls = []

        def candidate(item, *, window_start_ms, window_end_ms, **kwargs):
            call = (item["utterance_uid"], window_start_ms, window_end_ms)
            calls.append(call)
            if call != ("utt-a", 0, 3000):
                raise ForcedAlignmentError("synthetic non-edge candidate rejected")
            return [{"utterance_uid": "utt-a", "word": "Alpha",
                     "start_ms": 2500, "end_ms": 2700}]

        with patch.object(forced_align, "_alignment_candidate", side_effect=candidate):
            selected, _, resolution = forced_align._resolve_alignment_overlaps(
                source, independent, align=lambda *args, **kwargs: _result([
                    {"word": "Alpha", "start": 2.5, "end": 4.5},
                ]), align_model=object(), align_metadata={}, audio=object(),
                device="cuda", call_kwargs={}, min_word_score=DEFAULT_MIN_WORD_SCORE,
                max_word_duration_ms=DEFAULT_MAX_WORD_DURATION_MS,
                max_outward_drift_ms=DEFAULT_MAX_OUTWARD_DRIFT_MS, vad_regions=[],
            )

        self.assertEqual(resolution["final_overlap_count"], 0)
        self.assertEqual(calls[-2:], [("utt-a", 0, 3000), ("utt-b", 4100, 4500)])
        self.assertEqual(selected["utt-c"], independent["utt-c"])

    def test_bounded_overlap_search_forward_checks_late_conflict(self) -> None:
        component_uids = [f"utt-{index}" for index in range(8)]
        options = {}
        for index, uid in enumerate(component_uids):
            if index == 0:
                starts = list(range(850, 857)) + [0]
            elif index == 7:
                starts = list(range(840, 848))
            else:
                starts = [index * 100 + option for option in range(8)]
            options[uid] = [
                (
                    "joint",
                    [
                        {
                            "utterance_uid": uid,
                            "start_ms": start,
                            "end_ms": start + 20,
                        }
                    ],
                )
                for start in starts
            ]

        chosen, attempts = forced_align._bounded_overlap_search(
            component_uids,
            options,
            {uid: option_set[0][1] for uid, option_set in options.items()},
            {uid: index for index, uid in enumerate(component_uids)},
            700,
        )

        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen["utt-0"][1][0]["start_ms"], 0)
        self.assertEqual(attempts, 695)

    def test_final_stabilization_rejects_new_overlap_relation(self) -> None:
        order = {"utt-a": 0, "utt-b": 1, "utt-c": 2}

        def word(uid, start, end):
            return [{"utterance_uid": uid, "start_ms": start, "end_ms": end}]

        selected = {
            "utt-a": word("utt-a", 0, 100),
            "utt-b": word("utt-b", 50, 150),
            "utt-c": word("utt-c", 200, 300),
        }
        options = {
            "utt-a": [
                ("independent", selected["utt-a"]),
                ("edge-partition", word("utt-a", 220, 260)),
            ],
            "utt-b": [("independent", selected["utt-b"])],
            "utt-c": [("independent", selected["utt-c"])],
        }
        modes = {uid: "independent" for uid in selected}

        forced_align._stabilize_overlap_selection(
            selected,
            options,
            modes,
            order,
            lambda seed: list(seed),
        )

        self.assertEqual(
            forced_align._overlap_pair_signature(selected, order),
            frozenset({("utt-a", "utt-b")}),
        )
        self.assertEqual(selected["utt-a"][0]["start_ms"], 0)

    def test_final_stabilization_resolves_after_overlapping_context_selections(self) -> None:
        order = {"utt-a": 0, "utt-b": 1, "utt-c": 2}

        def word(uid, start, end):
            return [{"utterance_uid": uid, "start_ms": start, "end_ms": end}]

        selected = {
            "utt-a": word("utt-a", 0, 100),
            "utt-b": word("utt-b", 90, 190),
            "utt-c": word("utt-c", 180, 280),
        }
        options = {
            "utt-a": [("context-left", selected["utt-a"])],
            "utt-b": [
                ("context-right", selected["utt-b"]),
                ("edge-partition", word("utt-b", 105, 175)),
            ],
            "utt-c": [("context-right", selected["utt-c"])],
        }
        modes = {
            "utt-a": "context-left",
            "utt-b": "context-right",
            "utt-c": "context-right",
        }

        forced_align._stabilize_overlap_selection(
            selected,
            options,
            modes,
            order,
            lambda seed: ["utt-a", "utt-b", "utt-c"],
        )

        self.assertFalse(forced_align._overlap_pair_signature(selected, order))
        self.assertEqual(
            (selected["utt-b"][0]["start_ms"], selected["utt-b"][0]["end_ms"]),
            (105, 175),
        )
        self.assertEqual(modes["utt-b"], "edge-partition")

    def test_final_residual_single_path_resolves_adjacent_disjoint_cues(self) -> None:
        source = validate_coarse_segments(
            [
                {
                    "start_ms": 500,
                    "end_ms": 2200,
                    "coarse_start_ms": 1000,
                    "coarse_end_ms": 1500,
                    "text": "Alpha",
                    "asr_text": "Alpha",
                    "deletion_audio_reviewed": False,
                    "utterance_uid": "utt-alpha",
                },
                {
                    "start_ms": 800,
                    "end_ms": 2800,
                    "coarse_start_ms": 1500,
                    "coarse_end_ms": 2000,
                    "text": "Bravo",
                    "asr_text": "Bravo",
                    "deletion_audio_reviewed": False,
                    "utterance_uid": "utt-bravo",
                },
            ]
        )
        independent = {
            "utt-alpha": [
                {
                    "utterance_uid": "utt-alpha",
                    "text": "Alpha",
                    "start_ms": 1100,
                    "end_ms": 1700,
                }
            ],
            "utt-bravo": [
                {
                    "utterance_uid": "utt-bravo",
                    "text": "Bravo",
                    "start_ms": 1600,
                    "end_ms": 1900,
                }
            ],
        }
        joint_calls = []

        def align(transcript, *args, **kwargs):
            call = (
                transcript[0]["text"],
                transcript[0]["start"],
                transcript[0]["end"],
            )
            joint_calls.append(call)
            if call != ("Alpha Bravo", 0.8, 2.2):
                return _result(
                    [
                        {"word": "Alpha", "start": 1.1, "end": 1.7},
                        {"word": "Bravo", "start": 1.6, "end": 1.9},
                    ]
                )
            return _result(
                [
                    {"word": "Alpha", "start": 1.1, "end": 1.4},
                    {"word": "Bravo", "start": 1.6, "end": 1.9},
                ]
            )

        with patch.object(
            forced_align, "_alignment_candidate", side_effect=ForcedAlignmentError
        ):
            selected, _, resolution = forced_align._resolve_alignment_overlaps(
                source,
                independent,
                align=align,
                align_model=object(),
                align_metadata={},
                audio=object(),
                device="cuda",
                call_kwargs={},
                min_word_score=DEFAULT_MIN_WORD_SCORE,
                max_word_duration_ms=DEFAULT_MAX_WORD_DURATION_MS,
                max_outward_drift_ms=DEFAULT_MAX_OUTWARD_DRIFT_MS,
                vad_regions=[],
            )

        self.assertEqual(joint_calls[-1], ("Alpha Bravo", 0.8, 2.2))
        self.assertEqual(len(joint_calls), len(set(joint_calls)))
        self.assertEqual(resolution["selected_mode_counts"], {"joint": 2})
        self.assertEqual(
            (selected["utt-alpha"][0]["start_ms"], selected["utt-alpha"][0]["end_ms"]),
            (1100, 1400),
        )
        self.assertEqual(
            (selected["utt-bravo"][0]["start_ms"], selected["utt-bravo"][0]["end_ms"]),
            (1600, 1900),
        )

    def test_final_residual_single_path_rejects_mismatch_and_low_score(self) -> None:
        source = validate_coarse_segments(
            [
                {
                    "start_ms": 500,
                    "end_ms": 2200,
                    "coarse_start_ms": 1000,
                    "coarse_end_ms": 1500,
                    "text": "Alpha",
                    "asr_text": "Alpha",
                    "deletion_audio_reviewed": False,
                    "utterance_uid": "utt-alpha",
                },
                {
                    "start_ms": 800,
                    "end_ms": 2800,
                    "coarse_start_ms": 1500,
                    "coarse_end_ms": 2000,
                    "text": "Bravo",
                    "asr_text": "Bravo",
                    "deletion_audio_reviewed": False,
                    "utterance_uid": "utt-bravo",
                },
            ]
        )
        independent = {
            "utt-alpha": [
                {
                    "utterance_uid": "utt-alpha",
                    "text": "Alpha",
                    "start_ms": 1100,
                    "end_ms": 1700,
                }
            ],
            "utt-bravo": [
                {
                    "utterance_uid": "utt-bravo",
                    "text": "Bravo",
                    "start_ms": 1600,
                    "end_ms": 1900,
                }
            ],
        }
        rejection_cases = {
            "lexical alignment coverage failed": [
                {"word": "Bravo", "start": 1.1, "end": 1.4},
                {"word": "Alpha", "start": 1.6, "end": 1.9},
            ],
            "below the required minimum": [
                {"word": "Alpha", "start": 1.1, "end": 1.4},
                {"word": "Bravo", "start": 1.6, "end": 1.9, "score": 0.01},
            ],
        }
        for expected_error, late_words in rejection_cases.items():
            with self.subTest(expected_error=expected_error):
                joint_calls = []

                def align(transcript, *args, **kwargs):
                    call = (
                        transcript[0]["text"],
                        transcript[0]["start"],
                        transcript[0]["end"],
                    )
                    joint_calls.append(call)
                    if call != ("Alpha Bravo", 0.8, 2.2):
                        return _result(
                            [
                                {"word": "Alpha", "start": 1.1, "end": 1.7},
                                {"word": "Bravo", "start": 1.6, "end": 1.9},
                            ]
                        )
                    return _result(late_words)

                with patch.object(
                    forced_align,
                    "_alignment_candidate",
                    side_effect=ForcedAlignmentError,
                ):
                    with self.assertRaisesRegex(
                        ForcedAlignmentError, expected_error
                    ):
                        forced_align._resolve_alignment_overlaps(
                            source,
                            independent,
                            align=align,
                            align_model=object(),
                            align_metadata={},
                            audio=object(),
                            device="cuda",
                            call_kwargs={},
                            min_word_score=DEFAULT_MIN_WORD_SCORE,
                            max_word_duration_ms=DEFAULT_MAX_WORD_DURATION_MS,
                            max_outward_drift_ms=DEFAULT_MAX_OUTWARD_DRIFT_MS,
                            vad_regions=[],
                        )
                self.assertEqual(joint_calls[-1], ("Alpha Bravo", 0.8, 2.2))
                self.assertEqual(len(joint_calls), len(set(joint_calls)))

    def test_reviewed_dialogue_does_not_establish_distinct_speakers(self) -> None:
        coarse = [
            {
                "start_ms": 1000,
                "end_ms": 2500,
                "text": "Merhaba.",
                "asr_text": "Merhaba.",
                "deletion_audio_reviewed": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "utterance_uid": "utt-a",
            },
            {
                "start_ms": 1200,
                "end_ms": 2800,
                "text": "Selam.",
                "asr_text": "Selam.",
                "deletion_audio_reviewed": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "utterance_uid": "utt-b",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "unknown-speaker alignment overlap"):
                align_corrected_segments(
                    self._audio(directory),
                    coarse,
                    whisperx_module=_OverlapWhisperX(joint_resolves=False),
                )
        self.assertTrue(all("speaker_id" not in item for item in coarse))

    def test_success_is_json_ready_and_preserves_provenance(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.12, "end": 1.55, "score": 0.94},
                        {"word": "dünya!", "start": 1.60, "end": 2.08, "score": 0.91},
                    ]
                ),
                _result(
                    [{"word": "Nasılsın?", "start": 4.18, "end": 4.77, "score": 0.93}]
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), _coarse(), whisperx_module=fake
            )

        self.assertEqual(data["timing_source"], TIMING_SOURCE)
        self.assertEqual(
            data["audio_sha256"],
            hashlib.sha256(b"synthetic-audio-placeholder").hexdigest(),
        )
        self.assertEqual(data["provenance"]["whisperx_version"], "3.8.6")
        self.assertEqual(
            data["provenance"]["whisperx_version"], SUPPORTED_WHISPERX_VERSION
        )
        self.assertEqual(
            data["provenance"]["model_name"], DEFAULT_TURKISH_ALIGNMENT_MODEL
        )
        self.assertEqual(data["provenance"]["interpolation"], "disabled:ignore")
        self.assertEqual(
            data["provenance"]["text_normalization"],
            ALIGNMENT_TEXT_NORMALIZATION,
        )
        self.assertEqual(
            data["provenance"]["min_word_score"], DEFAULT_MIN_WORD_SCORE
        )
        self.assertEqual(
            data["provenance"]["review_word_score"], REVIEW_WORD_SCORE
        )
        self.assertEqual(
            data["provenance"]["edited_token_min_word_score"],
            EDITED_TOKEN_MIN_WORD_SCORE,
        )
        self.assertEqual(
            data["provenance"]["max_word_duration_ms"],
            DEFAULT_MAX_WORD_DURATION_MS,
        )
        self.assertEqual(
            data["provenance"]["max_outward_drift_ms"],
            DEFAULT_MAX_OUTWARD_DRIFT_MS,
        )
        self.assertEqual([segment["utterance_uid"] for segment in data["segments"]], ["utt-1", "utt-2"])
        self.assertEqual([word["word_index"] for word in data["words"]], [1, 2, 3])
        self.assertRegex(data["alignment_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(
            all(
                word["alignment_sha256"] == data["alignment_sha256"]
                for word in data["words"]
            )
        )
        self.assertEqual(data["segments"][0]["start_ms"], 1_120)
        self.assertEqual(data["segments"][0]["end_ms"], 2_080)
        self.assertEqual(data["report"]["unaligned_lexical_token_count"], 0)
        self.assertEqual(data["report"]["unaligned_word_count"], 0)
        self.assertEqual(data["report"]["negative_word_gap_count"], 0)
        self.assertEqual(
            data["report"]["alignment_sha256"], data["alignment_sha256"]
        )
        self.assertEqual(data["report"]["synthetic_timing_count"], 0)
        self.assertEqual(data["report"]["missing_alignment_score_count"], 0)
        self.assertEqual(data["report"]["low_alignment_score_count"], 0)
        self.assertEqual(data["report"]["overlong_alignment_word_count"], 0)
        self.assertEqual(data["report"]["outward_drift_violation_count"], 0)
        self.assertEqual(data["report"]["low_score_edited_token_count"], 0)
        self.assertEqual(data["report"]["unreviewed_deleted_token_count"], 0)
        self.assertEqual(data["report"]["edited_token_count"], 0)
        self.assertEqual(data["report"]["edited_token_ratio"], 0.0)
        self.assertEqual(data["report"]["review_alignment_score_count"], 0)
        self.assertEqual(data["report"]["minimum_alignment_score"], 0.91)
        self.assertEqual(len(fake.audio_calls), 1)
        self.assertEqual(len(fake.model_calls), 1)
        self.assertEqual(len(fake.align_calls), 2)
        self.assertTrue(
            all(call["interpolate_method"] == "ignore" for call in fake.align_calls)
        )
        self.assertEqual(
            [call["transcript"][0]["text"] for call in fake.align_calls],
            ["Merhaba, dunya!", "Nasilsin?"],
        )
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        self.assertEqual(validate_forced_alignment_data(data)["aligned_word_count"], 3)

        tampered = copy.deepcopy(data)
        tampered["provenance"].pop("text_normalization")
        with self.assertRaisesRegex(ForcedAlignmentError, "text_normalization"):
            validate_forced_alignment_data(tampered)

    def test_none_is_used_when_align_api_explicitly_supports_it(self) -> None:
        fake = _NoneInterpolationWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                        {"word": "dünya!", "start": 1.5, "end": 1.9},
                    ]
                )
            ]
        )
        coarse = [_coarse()[0]]
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )

        self.assertIsNone(fake.align_calls[0]["interpolate_method"])
        self.assertEqual(data["provenance"]["interpolation"], "disabled:none")

    def test_audio_mutation_during_alignment_hard_fails(self) -> None:
        fake = _MutatingWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                        {"word": "dünya!", "start": 1.5, "end": 1.9},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "audio changed while forced alignment"
            ):
                align_corrected_segments(
                    self._audio(directory), [_coarse()[0]], whisperx_module=fake
                )

    def test_missing_low_and_out_of_range_scores_hard_fail(self) -> None:
        cases = (
            (
                _result(
                    [{"word": "Nasılsın?", "start": 4.1, "end": 4.7}],
                    add_default_scores=False,
                ),
                "alignment score is required",
            ),
            (
                _result(
                    [
                        {
                            "word": "Nasılsın?",
                            "start": 4.1,
                            "end": 4.7,
                            "score": 0.149,
                        }
                    ]
                ),
                "below the required minimum",
            ),
            (
                _result(
                    [
                        {
                            "word": "Nasılsın?",
                            "start": 4.1,
                            "end": 4.7,
                            "score": 1.01,
                        }
                    ]
                ),
                "within \\[0, 1\\]",
            ),
        )
        for raw_result, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(ForcedAlignmentError, message):
                        align_corrected_segments(
                            self._audio(directory),
                            [_coarse()[1]],
                            whisperx_module=_FakeWhisperX([raw_result]),
                        )

    def test_configurable_minimum_score_is_digest_bound(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {
                            "word": "Nasılsın?",
                            "start": 4.1,
                            "end": 4.7,
                            "score": 0.40,
                        }
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory),
                [_coarse()[1]],
                min_word_score=0.35,
                whisperx_module=fake,
            )
        self.assertEqual(data["provenance"]["min_word_score"], 0.35)
        self.assertEqual(data["report"]["minimum_alignment_score"], 0.40)
        self.assertEqual(data["report"]["review_alignment_score_count"], 1)

        tampered = copy.deepcopy(data)
        tampered["provenance"]["min_word_score"] = 0.41
        with self.assertRaisesRegex(ForcedAlignmentError, "below provenance"):
            validate_forced_alignment_data(tampered)

    def test_canonical_score_floor_cannot_be_lowered(self) -> None:
        fake = _FakeWhisperX(
            [_result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}])]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "min_word_score must be within"
            ):
                align_corrected_segments(
                    self._audio(directory),
                    [_coarse()[1]],
                    min_word_score=0.20,
                    whisperx_module=fake,
                )

        valid = _FakeWhisperX(
            [_result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}])]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), [_coarse()[1]], whisperx_module=valid
            )
        tampered = copy.deepcopy(data)
        tampered["provenance"]["min_word_score"] = 0.20
        with self.assertRaisesRegex(
            ForcedAlignmentError, "provenance min_word_score must be within"
        ):
            validate_forced_alignment_data(tampered)

    def test_score_below_beta_floor_hard_fails(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {
                            "word": "Nasılsın?",
                            "start": 4.1,
                            "end": 4.7,
                            "score": 0.20,
                        }
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "below the required minimum"
            ):
                align_corrected_segments(
                    self._audio(directory), [_coarse()[1]], whisperx_module=fake
                )

    def test_hash_bound_audio_review_supports_low_alignment_score(self) -> None:
        coarse = [_coarse()[1]]
        coarse[0]["audio_reviewed"] = True
        coarse[0]["review_disposition"] = "confirmed_dialogue"
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {
                            "word": coarse[0]["text"],
                            "start": 4.1,
                            "end": 4.7,
                            "score": 0.0,
                        }
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )

        self.assertTrue(data["segments"][0]["audio_reviewed"])
        self.assertEqual(
            data["segments"][0]["review_disposition"], "confirmed_dialogue"
        )
        self.assertEqual(data["words"][0]["score_context"], AUDIO_REVIEW_SCORE_CONTEXT)
        validate_forced_alignment_data(data)

        tampered = copy.deepcopy(data)
        tampered["words"][0]["score_context"] = "adjacent_unchanged_word"
        tampered["segments"][0]["words"][0]["score_context"] = (
            "adjacent_unchanged_word"
        )
        with self.assertRaisesRegex(ForcedAlignmentError, "unexpected score_context"):
            validate_forced_alignment_data(tampered)

    def test_hash_bound_audio_review_supports_low_edited_token_score(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "Merhaba dunya",
                "asr_text": "Merhaba",
                "deletion_audio_reviewed": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "utterance_uid": "utt-reviewed-insert",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.1, "end": 1.4, "score": 0.9},
                        {"word": "dunya", "start": 1.5, "end": 1.9, "score": 0.3},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )

        self.assertEqual(data["words"][1]["edit_kind"], "inserted")
        self.assertEqual(data["words"][1]["score_context"], AUDIO_REVIEW_SCORE_CONTEXT)
        validate_forced_alignment_data(data)

    def test_inconsistent_audio_review_fields_hard_fail(self) -> None:
        coarse = [_coarse()[1]]
        coarse[0]["audio_reviewed"] = True
        with self.assertRaisesRegex(ForcedAlignmentError, "fields are inconsistent"):
            validate_coarse_segments(coarse)

    def test_unchanged_word_uses_bounded_adjacent_score_context(self) -> None:
        coarse = [
            {
                "start_ms": 100,
                "end_ms": 3_000,
                "coarse_start_ms": 1_000,
                "coarse_end_ms": 2_500,
                "text": "Ağabeyim Ağabey",
                "asr_text": "Ağabeyim Ağabey",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-context-score",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Ağabeyim", "start": 0.3, "end": 0.8, "score": 0.275},
                        {"word": "Ağabey", "start": 1.1, "end": 1.4, "score": 0.80},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )
        self.assertEqual(
            data["provenance"]["contextual_unchanged_min_word_score"],
            CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE,
        )
        self.assertEqual(
            data["words"][0]["score_context"], "adjacent_unchanged_word"
        )
        self.assertNotIn("score_context", data["words"][1])
        self.assertEqual(
            data["segments"][0]["drift_context"],
            "contextual_low_score_alignment_window",
        )

    def test_inserted_token_requires_edited_token_acoustic_floor(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "Merhaba dünya",
                "asr_text": "Merhaba",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-inserted",
            }
        ]
        low_insert = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.1, "end": 1.4, "score": 0.9},
                        {"word": "dünya", "start": 1.5, "end": 1.9, "score": 0.549},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "edited token 'dünya'.*edited-token minimum"
            ):
                align_corrected_segments(
                    self._audio(directory), coarse, whisperx_module=low_insert
                )

        accepted = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.1, "end": 1.4, "score": 0.9},
                        {"word": "dünya", "start": 1.5, "end": 1.9, "score": 0.55},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=accepted
            )
        self.assertEqual(
            [word["edit_kind"] for word in data["words"]],
            ["unchanged", "inserted"],
        )
        self.assertEqual(data["segments"][0]["edit_audit"]["inserted_token_count"], 1)
        self.assertEqual(data["segments"][0]["edit_audit"]["edited_token_ratio"], 0.5)
        self.assertEqual(data["report"]["edited_token_count"], 1)
        self.assertEqual(data["report"]["low_score_edited_token_count"], 0)

    def test_replaced_token_requires_edited_token_acoustic_floor(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "Merhaba dünya",
                "asr_text": "Meraba dünya",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-replaced",
            }
        ]
        low_replacement = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.1, "end": 1.4, "score": 0.549},
                        {"word": "dünya", "start": 1.5, "end": 1.9, "score": 0.90},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "edited token"):
                align_corrected_segments(
                    self._audio(directory), coarse, whisperx_module=low_replacement
                )

        accepted = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.1, "end": 1.4, "score": 0.55},
                        {"word": "dünya", "start": 1.5, "end": 1.9, "score": 0.90},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=accepted
            )
        self.assertEqual(
            [word["edit_kind"] for word in data["words"]],
            ["replaced", "unchanged"],
        )
        self.assertEqual(data["segments"][0]["edit_audit"]["replaced_token_count"], 1)

    def test_empty_asr_evidence_marks_every_corrected_token_inserted(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "Duydum seni",
                "asr_text": "",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-audio-review-only",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Duydum", "start": 1.1, "end": 1.4, "score": 0.55},
                        {"word": "seni", "start": 1.5, "end": 1.9, "score": 0.55},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )
        self.assertEqual(
            [word["edit_kind"] for word in data["words"]],
            ["inserted", "inserted"],
        )
        self.assertEqual(data["segments"][0]["edit_audit"]["edited_token_ratio"], 1.0)

    def test_blank_corrected_text_reports_lexical_deletion(self) -> None:
        self.assertTrue(correction_deletes_lexical_tokens("Yanlis altyazi", ""))
        self.assertFalse(correction_deletes_lexical_tokens("", ""))

    def test_punctuation_and_case_only_change_is_not_lexical_edit(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "Merhaba, DÜNYA!",
                "asr_text": "merhaba dünya",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-surface-only",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "merhaba", "start": 1.1, "end": 1.4, "score": 0.30},
                        {"word": "dünya", "start": 1.5, "end": 1.9, "score": 0.30},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )
        self.assertEqual(
            [word["edit_kind"] for word in data["words"]],
            ["unchanged", "unchanged"],
        )
        self.assertEqual(data["report"]["edited_token_count"], 0)
        self.assertEqual(data["report"]["review_alignment_score_count"], 2)

    def test_turkish_dotted_and_dotless_i_case_changes_are_unchanged(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "ışık için",
                "asr_text": "IŞIK İÇİN",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-turkish-case",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "ışık", "start": 1.1, "end": 1.4, "score": 0.30},
                        {"word": "için", "start": 1.5, "end": 1.9, "score": 0.30},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )
        self.assertEqual(
            [word["edit_kind"] for word in data["words"]],
            ["unchanged", "unchanged"],
        )

    def test_repeated_token_edits_are_conservative_and_deterministic(self) -> None:
        insertion = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "evet evet",
                "asr_text": "evet",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-repeat-insert",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "evet", "start": 1.1, "end": 1.3, "score": 0.40},
                        {"word": "evet", "start": 1.4, "end": 1.6, "score": 0.90},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "edited token"):
                align_corrected_segments(
                    self._audio(directory), insertion, whisperx_module=fake
                )

        deletion = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "evet",
                "asr_text": "evet evet",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-repeat-delete",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "lexical deletions without exact audio-reviewed"
            ):
                align_corrected_segments(
                    self._audio(directory),
                    deletion,
                    whisperx_module=_FakeWhisperX(
                        [_result([{"word": "evet", "start": 1.1, "end": 1.4}])]
                    ),
                )

    def test_deleted_token_requires_exact_audio_review_flag(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "Merhaba dünya",
                "asr_text": "Merhaba güzel dünya",
                "deletion_audio_reviewed": True,
                "utterance_uid": "utt-reviewed-delete",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.1, "end": 1.4},
                        {"word": "dünya", "start": 1.5, "end": 1.9},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )
        audit = data["segments"][0]["edit_audit"]
        self.assertEqual(audit["deleted_asr_token_count"], 1)
        self.assertEqual(audit["unreviewed_deleted_token_count"], 0)
        self.assertEqual(data["report"]["deleted_asr_token_count"], 1)

        tampered = copy.deepcopy(data)
        tampered["segments"][0]["deletion_audio_reviewed"] = False
        with self.assertRaisesRegex(ForcedAlignmentError, "unreviewed deleted"):
            validate_forced_alignment_data(tampered)

    def test_word_duration_policy_is_hard_and_cannot_be_weakened(self) -> None:
        long_coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 5_000,
                "text": "Merhaba",
                "asr_text": "Merhaba",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-long",
            }
        ]
        fake = _FakeWhisperX(
            [_result([{"word": "Merhaba", "start": 1.1, "end": 3.601}])]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "exceeds the maximum"):
                align_corrected_segments(
                    self._audio(directory), long_coarse, whisperx_module=fake
                )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "max_word_duration_ms cannot exceed"
            ):
                align_corrected_segments(
                    self._audio(directory),
                    [_coarse()[1]],
                    max_word_duration_ms=2_501,
                    whisperx_module=_FakeWhisperX(
                        [_result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}])]
                    ),
                )

    def test_reviewed_overlong_ctc_word_uses_exact_independent_vad_boundary(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 6_000,
                "text": "Allah'im ya",
                "asr_text": "Allah'im ya",
                "deletion_audio_reviewed": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "utterance_uid": "utt-vad-duration",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Allah'im", "start": 1.1, "end": 5.0},
                        {"word": "ya", "start": 5.1, "end": 5.4},
                    ]
                )
            ]
        )
        vad_regions = [
            {
                "vad_region_index": 7,
                "start_ms": 1_000,
                "end_ms": 1_700,
                "source": "silero_vad",
            },
            {
                "vad_region_index": 8,
                "start_ms": 5_050,
                "end_ms": 5_500,
                "source": "silero_vad",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory),
                coarse,
                vad_regions=vad_regions,
                whisperx_module=fake,
            )

        word = data["segments"][0]["words"][0]
        self.assertEqual(word["start_ms"], 1_100)
        self.assertEqual(word["end_ms"], 1_700)
        self.assertEqual(word["raw_end_ms"], 5_000)
        self.assertEqual(word["duration_context"], DURATION_VAD_CONTEXT)
        self.assertEqual(data["report"]["maximum_alignment_word_duration_ms"], 600)

        tampered = copy.deepcopy(data)
        tampered["segments"][0]["words"][0]["vad_end_ms"] = 1_701
        with self.assertRaisesRegex(ForcedAlignmentError, "does not bind"):
            validate_forced_alignment_data(tampered)

    def test_unpadded_coarse_bounds_and_outward_drift_are_hard_bound(self) -> None:
        coarse = [
            {
                "start_ms": 1_100,
                "end_ms": 4_900,
                "coarse_start_ms": 2_000,
                "coarse_end_ms": 4_000,
                "text": "Merhaba dünya",
                "asr_text": "Merhaba dünya",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-drift",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.5, "end": 2.0},
                        {"word": "dünya", "start": 4.0, "end": 4.5},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )
        segment = data["segments"][0]
        self.assertEqual(segment["coarse_start_ms"], 2_000)
        self.assertEqual(segment["coarse_end_ms"], 4_000)
        self.assertEqual(segment["drift_audit"]["early_outward_drift_ms"], 500)
        self.assertEqual(segment["drift_audit"]["late_outward_drift_ms"], 500)
        self.assertEqual(data["report"]["maximum_early_outward_drift_ms"], 500)
        self.assertEqual(data["report"]["maximum_late_outward_drift_ms"], 500)

        too_early = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 1.499, "end": 2.0},
                        {"word": "dünya", "start": 4.0, "end": 4.5},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "exceeds max_outward_drift_ms"
            ):
                align_corrected_segments(
                    self._audio(directory), coarse, whisperx_module=too_early
                )

        boundary_coarse = [
            {
                **coarse[0],
                "text": "Merhaba dunya",
                "asr_text": "Merhaba dunya",
            }
        ]
        out_of_bounds_boundary = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba", "start": 2.1, "end": 2.6, "score": 0.9},
                        {"word": "dunya", "start": 3.8, "end": 4.739, "score": 0.608},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "exceeds max_outward_drift_ms"
            ):
                align_corrected_segments(
                    self._audio(directory),
                    boundary_coarse,
                    whisperx_module=out_of_bounds_boundary,
                )

        tampered = copy.deepcopy(data)
        tampered["provenance"]["max_outward_drift_ms"] = 501
        with self.assertRaisesRegex(
            ForcedAlignmentError, "max_outward_drift_ms cannot exceed"
        ):
            validate_forced_alignment_data(tampered)

    def test_missing_lexical_timestamp_hard_fails(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                        {"word": "dünya!"},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "finite number"):
                align_corrected_segments(
                    self._audio(directory), [_coarse()[0]], whisperx_module=fake
                )

    def test_missing_or_changed_lexical_token_hard_fails(self) -> None:
        fake = _FakeWhisperX(
            [_result([{"word": "Merhaba,", "start": 1.1, "end": 1.4}])]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "coverage failed"):
                align_corrected_segments(
                    self._audio(directory), [_coarse()[0]], whisperx_module=fake
                )

    def test_punctuation_only_word_does_not_need_timestamp(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                        {"word": "--"},
                        {"word": "dünya!", "start": 1.5, "end": 1.9},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), [_coarse()[0]], whisperx_module=fake
            )

        self.assertEqual([word["text"] for word in data["words"]], ["Merhaba,", "dünya!"])
        self.assertEqual(data["report"]["raw_punctuation_only_word_count"], 1)

    def test_corrected_surface_not_whisperx_surface_is_canonical(self) -> None:
        coarse = [
            {
                "start_ms": 1_000,
                "end_ms": 3_000,
                "text": "— Allah'ım, Dünya!",
                "asr_text": "allahım dünya",
                "deletion_audio_reviewed": False,
                "utterance_uid": "utt-canonical-surface",
            }
        ]
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "allahım", "start": 1.1, "end": 1.5},
                        {"word": "dünya", "start": 1.6, "end": 2.0},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), coarse, whisperx_module=fake
            )

        self.assertEqual(
            [word["text"] for word in data["words"]],
            ["— Allah'ım,", "Dünya!"],
        )
        self.assertEqual(data["segments"][0]["text"], coarse[0]["text"])
        self.assertEqual(validate_forced_alignment_data(data)["aligned_word_count"], 2)
        blocks = build_blocks(
            {
                "words": data["words"],
                "vad_regions": [
                    {
                        "vad_region_index": 1,
                        "start_ms": 1_000,
                        "end_ms": 3_000,
                        "source": "silero_vad",
                    }
                ],
            },
            episode=12,
        )
        self.assertEqual(
            " ".join(block["primary_text"] for block in blocks), coarse[0]["text"]
        )

        noncanonical = copy.deepcopy(data)
        noncanonical["segments"][0]["words"][0]["text"] = "allahım"
        noncanonical["words"][0]["text"] = "allahım"
        with self.assertRaisesRegex(ForcedAlignmentError, "preserve corrected"):
            validate_forced_alignment_data(noncanonical)

    def test_overlapping_word_interval_hard_fails(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.1, "end": 1.7},
                        {"word": "dünya!", "start": 1.6, "end": 1.9},
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ForcedAlignmentError, "overlapping/backward"):
                align_corrected_segments(
                    self._audio(directory), [_coarse()[0]], whisperx_module=fake
                )

    def test_nonfinite_and_synthetic_timing_hard_fail(self) -> None:
        bad_words = (
            [{"word": "Nasılsın?", "start": math.nan, "end": 4.7}],
            [
                {
                    "word": "Nasılsın?",
                    "start": 4.1,
                    "end": 4.7,
                    "interpolated": True,
                }
            ],
            [
                {
                    "word": "Nasılsın?",
                    "start": 4.1,
                    "end": 4.7,
                    "timing_source": "synthetic_segment_word_repair",
                }
            ],
        )
        for words in bad_words:
            with self.subTest(words=words):
                fake = _FakeWhisperX([_result(words)])
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ForcedAlignmentError):
                        align_corrected_segments(
                            self._audio(directory), [_coarse()[1]], whisperx_module=fake
                        )

    def test_cross_segment_overlap_hard_fails_in_pure_validator(self) -> None:
        fake = _FakeWhisperX(
            [
                _result(
                    [
                        {"word": "Merhaba,", "start": 1.1, "end": 1.4},
                        {"word": "dünya!", "start": 1.5, "end": 1.9},
                    ]
                ),
                _result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}]),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), _coarse(), whisperx_module=fake
            )
        corrupt = copy.deepcopy(data)
        corrupt["segments"][1]["alignment_window_start_ms"] = 1_800
        corrupt["segments"][1]["start_ms"] = 1_850
        corrupt["segments"][1]["end_ms"] = 2_250
        corrupt["segments"][1]["words"][0]["start_ms"] = 1_850
        corrupt["segments"][1]["words"][0]["end_ms"] = 2_250
        corrupt["words"][2]["start_ms"] = 1_850
        corrupt["words"][2]["end_ms"] = 2_250

        with self.assertRaisesRegex(ForcedAlignmentError, "overlapping"):
            validate_forced_alignment_data(corrupt)

    def test_unsupported_language_and_bad_coarse_input_fail_early(self) -> None:
        with self.assertRaisesRegex(ForcedAlignmentError, "only 'tr'"):
            align_corrected_segments(
                "/does/not/matter.flac", _coarse(), language="id", whisperx_module=object()
            )
        with self.assertRaisesRegex(ForcedAlignmentError, "duplicate utterance_uid"):
            validate_coarse_segments([_coarse()[0], {**_coarse()[1], "utterance_uid": "utt-1"}])
        with self.assertRaisesRegex(ForcedAlignmentError, "preceding segment"):
            validate_coarse_segments([_coarse()[1], _coarse()[0]])

    def test_unpinned_whisperx_version_hard_fails(self) -> None:
        fake = _FakeWhisperX(
            [_result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}])]
        )
        fake.__version__ = "3.8.5"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ForcedAlignmentError, "WhisperX 3.8.6 is required"
            ):
                align_corrected_segments(
                    self._audio(directory), [_coarse()[1]], whisperx_module=fake
                )

    def test_pure_validator_rejects_report_and_provenance_tampering(self) -> None:
        fake = _FakeWhisperX(
            [_result([{"word": "Nasılsın?", "start": 4.1, "end": 4.7}])]
        )
        with tempfile.TemporaryDirectory() as directory:
            data = align_corrected_segments(
                self._audio(directory), [_coarse()[1]], whisperx_module=fake
            )

        bad_report = copy.deepcopy(data)
        bad_report["report"]["aligned_word_count"] = 2
        with self.assertRaisesRegex(ForcedAlignmentError, "report aligned_word_count"):
            validate_forced_alignment_data(bad_report)
        bad_source = copy.deepcopy(data)
        bad_source["provenance"]["interpolation"] = "nearest"
        with self.assertRaisesRegex(ForcedAlignmentError, "not disabled"):
            validate_forced_alignment_data(bad_source)
        bad_digest = copy.deepcopy(data)
        bad_digest["alignment_sha256"] = "0" * 64
        for word in bad_digest["words"]:
            word["alignment_sha256"] = "0" * 64
        with self.assertRaisesRegex(ForcedAlignmentError, "alignment_sha256 mismatch"):
            validate_forced_alignment_data(bad_digest)
        bad_audio = copy.deepcopy(data)
        bad_audio["audio_sha256"] = "0" * 64
        with self.assertRaisesRegex(ForcedAlignmentError, "alignment_sha256 mismatch"):
            validate_forced_alignment_data(bad_audio)

        bad_edit_kind = copy.deepcopy(data)
        bad_edit_kind["segments"][0]["words"][0]["edit_kind"] = "inserted"
        bad_edit_kind["words"][0]["edit_kind"] = "inserted"
        with self.assertRaisesRegex(ForcedAlignmentError, "edit_kind mismatch"):
            validate_forced_alignment_data(bad_edit_kind)

        bad_asr_index = copy.deepcopy(data)
        bad_asr_index["segments"][0]["words"][0]["asr_token_index"] = None
        bad_asr_index["words"][0]["asr_token_index"] = None
        with self.assertRaisesRegex(ForcedAlignmentError, "asr_token_index mismatch"):
            validate_forced_alignment_data(bad_asr_index)

        bad_edit_audit = copy.deepcopy(data)
        bad_edit_audit["segments"][0]["edit_audit"]["edited_token_count"] = 1
        with self.assertRaisesRegex(ForcedAlignmentError, "edit audit mismatch"):
            validate_forced_alignment_data(bad_edit_audit)

        bad_edited_threshold = copy.deepcopy(data)
        bad_edited_threshold["provenance"]["edited_token_min_word_score"] = 0.54
        with self.assertRaisesRegex(
            ForcedAlignmentError, "edited_token_min_word_score must equal"
        ):
            validate_forced_alignment_data(bad_edited_threshold)


def _reference_component_selection(component_uids, options, selected, objective):
    best = best_score = None
    outside = [word for uid, words in selected.items() if uid not in component_uids for word in words]
    for combination in itertools.product(*(options[uid] for uid in component_uids)):
        trial = outside + [word for _, words in combination for word in words]
        if any(left["utterance_uid"] in component_uids or right["utterance_uid"] in component_uids
               for left, right in forced_align._unsafe_word_overlaps(trial)):
            continue
        score = (sum(mode == "joint" for mode, _ in combination),)
        if objective == "preferred":
            score += (sum(mode == "independent" for mode, _ in combination),
                      sum(mode.startswith("padding-") for mode, _ in combination))
        if best_score is None or score > best_score:
            best, best_score = dict(zip(component_uids, combination)), score
    return best


def test_exact_selector_matches_original_exhaustive_winners_and_ties():
    rng = random.Random(20260914)
    for case in range(200):
        uids = [f"uid-{index}" for index in range(rng.randint(1, 4))]
        options = {}
        for uid in uids:
            options[uid] = []
            for index in range(rng.randint(1, 4)):
                start = rng.randrange(16) * 100
                speaker = rng.choice([None, "speaker-a", "speaker-b"])
                words = [{"utterance_uid": uid, "start_ms": start, "end_ms": start + rng.randint(1, 5) * 100}]
                if speaker is not None:
                    words[0]["speaker_id"] = speaker
                if rng.random() < .25:
                    words.append({**words[0], "start_ms": start + 50, "end_ms": start + 200})
                options[uid].append((rng.choice(["joint", "independent", "padding-200", "edge"]), words))
        selected = {uid: options[uid][0][1] for uid in uids}
        selected["outside-a"] = [{"utterance_uid": "outside-a", "start_ms": 500, "end_ms": 600}]
        selected["outside-b"] = [{"utterance_uid": "outside-b", "start_ms": 550, "end_ms": 650}]
        for objective in ("joint", "preferred"):
            expected = _reference_component_selection(uids, options, selected, objective)
            actual = forced_align._strict_existing_option_selection_uncached(
                uids, options, selected, dict(zip(uids, range(len(uids)))), objective=objective,
            )
            assert actual == expected, (case, objective)


def test_exact_selector_prunes_only_provably_dominated_combinations_and_reuses_conflicts():
    uids = [f"uid-{index}" for index in range(5)]
    options = {uid: [("joint", [{"utterance_uid": uid, "start_ms": index * 1000 + candidate,
                               "end_ms": index * 1000 + 100 + candidate}])
                     for candidate in range(6)] for index, uid in enumerate(uids)}
    selected = {uid: items[0][1] for uid, items in options.items()}
    expected = _reference_component_selection(uids, options, selected, "preferred")
    work = []
    with patch.object(forced_align, "_timings_conflict", wraps=forced_align._timings_conflict) as conflicts:
        with patch.object(forced_align, "_unsafe_word_overlaps", wraps=forced_align._unsafe_word_overlaps) as whole_trial:
            actual = forced_align._strict_existing_option_selection_uncached(
                uids, options, selected, dict(zip(uids, range(5))),
                work_check=lambda **counts: work.append(counts),
            )
    assert actual == expected
    assert whole_trial.call_count == 0
    assert conflicts.call_count <= 120
    assert len(work) < 200
    assert 6 ** 5 == 7776  # Original exhaustive selector rebuilt this many complete trials.


def test_selector_keeps_non_neighbor_and_known_speaker_constraints():
    def words(uid, start, end, speaker):
        return [{"utterance_uid": uid, "start_ms": start, "end_ms": end, "speaker_id": speaker}]
    options = {
        "a": [("joint", words("a", 0, 100, "speaker-a"))],
        "b": [("joint", words("b", 40, 150, "speaker-b"))],
        "c": [("joint", words("c", 50, 120, "speaker-a")),
              ("padding-200", words("c", 100, 180, "speaker-a"))],
    }
    chosen = forced_align._strict_existing_option_selection_uncached(
        list(options), options, {uid: items[0][1] for uid, items in options.items()},
        {"a": 0, "b": 1, "c": 2},
    )
    assert chosen["c"][0] == "padding-200"
    assert chosen["a"][1][0]["end_ms"] > chosen["b"][1][0]["start_ms"]


def test_component_selection_cache_binds_objective_and_preserves_first_tie(tmp_path):
    word = {"utterance_uid": "a", "start_ms": 100, "end_ms": 200}
    options = {"a": [("edge", [word]), ("independent", [dict(word)])]}
    journal = forced_align.UnitJournal(tmp_path, {"fixture": "objective"})
    first = forced_align._strict_existing_option_selection(
        ["a"], options, {"a": [word]}, {"a": 0}, checkpoint_journal=journal, objective="joint",
    )
    second = forced_align._strict_existing_option_selection(
        ["a"], options, {"a": [word]}, {"a": 0}, checkpoint_journal=journal, objective="preferred",
    )
    assert first["a"][0] == "edge"
    assert second["a"][0] == "independent"
    with patch.object(forced_align, "_strict_existing_option_selection_uncached",
                      side_effect=AssertionError("unchanged objective must reuse")):
        assert forced_align._strict_existing_option_selection(
            ["a"], options, {"a": [word]}, {"a": 0}, checkpoint_journal=journal, objective="joint",
        ) == first


def test_scoped_recovery_budget_resume_bound_fixture(record_property):
    ForcedAlignmentTests(
        "test_scoped_recovery_budget_resume_preserves_failure_evidence"
    ).test_scoped_recovery_budget_resume_preserves_failure_evidence()
    diagnostics_json = os.getenv("MAS_RETRY_DIAGNOSTICS_JSON")
    if diagnostics_json:
        records = json.loads(diagnostics_json)
        scope = json.loads(os.environ["MAS_RETRY_SCOPE_JSON"])
        root = Path(os.environ["MAS_RETRY_EPISODE_ROOT"])
        paths = {Path(record["relative_path"]).name: record for record in records}
        assert set(paths) == {"resume-identity.json", "latest-conflict-failure.json"}
        failure = json.loads(
            (root / paths["latest-conflict-failure.json"]["relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert failure["sha256"] == forced_align.digest(failure["data"])
        assert failure["data"]["reason"] == "recovery_time_budget"
        assert failure["data"]["component_uids"] == scope["target_uids"]
        for name in (
            "mas_retry_failure_sha256",
            "mas_retry_scope_sha256",
            "mas_retry_diagnostics_sha256",
        ):
            record_property(name, os.environ[name.upper()])


if __name__ == "__main__":
    unittest.main()
