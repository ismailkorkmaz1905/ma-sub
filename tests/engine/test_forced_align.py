from __future__ import annotations

import copy
import hashlib
import json
import math
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


class _ContextualOverlapWhisperX(_FakeWhisperX):
    def __init__(self) -> None:
        super().__init__([])

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
                    {"word": "Once", "start": 0.5, "end": 0.8},
                    {"word": "Ben", "start": 1.1, "end": 1.2},
                    {"word": "ona", "start": 1.2, "end": 1.3},
                    {"word": "ne", "start": 1.3, "end": 1.4},
                    {"word": "yaptim", "start": 1.4, "end": 1.5},
                    {"word": "Hicbir", "start": 1.6, "end": 1.7},
                    {"word": "sey", "start": 1.7, "end": 1.8},
                    {"word": "yapmadim", "start": 1.8, "end": 1.9},
                    {"word": "Ben", "start": 2.4, "end": 2.5},
                    {"word": "ona", "start": 2.5, "end": 2.6},
                    {"word": "ne", "start": 2.6, "end": 2.7},
                    {"word": "yaptim", "start": 2.7, "end": 2.8},
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
        self.assertEqual(components.call_count, 2)
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
            [(500, 800), (1100, 1500), (1600, 1900), (2400, 2800)],
        )
        self.assertIn(
            "Once Ben ona ne yaptim Hicbir sey yapmadim Ben ona ne yaptim",
            [call["text"] for call in fake.align_calls],
        )
        resolution = data["provenance"]["overlap_resolution"]
        self.assertEqual(resolution["selected_mode_counts"], {"independent": 2, "joint": 2})
        self.assertEqual(resolution["final_overlap_count"], 0)
        validate_forced_alignment_data(data)

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


if __name__ == "__main__":
    unittest.main()
