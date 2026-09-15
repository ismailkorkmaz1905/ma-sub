from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mas.engine.forced_align import align_corrected_segments
from mas.engine.audio_review import AUDIO_REVIEW_V2_FORMAT, AUDIO_REVIEW_V2_VERSION
from mas.engine.download import sha256_json
from mas.engine.id_translation import validate_id_translation_pack
from mas.engine import raw_asr as raw_asr_module
from mas.engine.raw_asr import RawASRV2Config, required_acoustic_review_regions
from mas.engine.tr_correction import compute_output_sha256
from mas.engine.speech_coverage import SpeechCoverageConfig
from mas.engine.speaker_evidence import build_speaker_evidence
from mas.reliability import digest
from mas.engine.workflow import (
    DEFAULT_ALIGNMENT_PADDING_MS,
    V2PipelineError,
    build_strict_v2_artifacts,
    correction_records_to_alignment_inputs,
    create_v2_id_translation_pack,
    recompute_final_speech_coverage,
)


AUDIO_BYTES = b"test-audio"
AUDIO_SHA = hashlib.sha256(AUDIO_BYTES).hexdigest()


class _FakeWhisperX:
    __version__ = "3.8.6"

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = list(results)

    def load_audio(self, path: str) -> object:
        return object()

    def load_align_model(
        self,
        *,
        language_code: str,
        device: str,
        model_name: str,
    ) -> tuple[object, dict[str, str]]:
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
        return self.results.pop(0)


def _alignment_result(words: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "segments": [{"text": "unused", "words": words}],
        "word_segments": words,
    }


def _input_utterances() -> list[dict[str, Any]]:
    return [
        {
            "utterance_uid": "utt-1",
            "utterance_index": 1,
            "coarse_start_ms": 1_000,
            "coarse_end_ms": 2_500,
            "asr_text": "Meraba dunya.",
            "youtube_text": "Merhaba dünya.",
            "context_before": "",
            "context_after": "Ses efekti.",
            "risk_flags": [],
            "asr_audit": {
                "avg_logprob": -0.1,
                "no_speech_prob": 0.01,
                "compression_ratio": 1.0,
                "temperature": 0.0,
            },
        },
        {
            "utterance_uid": "utt-2",
            "utterance_index": 2,
            "coarse_start_ms": 3_000,
            "coarse_end_ms": 4_000,
            "asr_text": "",
            "youtube_text": "",
            "context_before": "Meraba dunya.",
            "context_after": "Bugun nasilsin?",
            "risk_flags": [
                "unresolved_vad_speech",
                "manual_audio_review_required",
            ],
            "asr_audit": {
                "avg_logprob": None,
                "no_speech_prob": None,
                "compression_ratio": None,
                "temperature": None,
            },
        },
        {
            "utterance_uid": "utt-3",
            "utterance_index": 3,
            "coarse_start_ms": 4_500,
            "coarse_end_ms": 6_000,
            "asr_text": "Bugun nasilsin?",
            "youtube_text": "Bugün nasılsın?",
            "context_before": "Ses efekti.",
            "context_after": "",
            "risk_flags": [],
            "asr_audit": {
                "avg_logprob": -0.1,
                "no_speech_prob": 0.01,
                "compression_ratio": 1.0,
                "temperature": 0.0,
            },
        },
    ]


def _corrections() -> list[dict[str, Any]]:
    records = copy.deepcopy(_input_utterances())
    records[0].update(
        {
            "tr_corrected": "Merhaba dünya.",
            "non_dialogue": False,
            "review_required": False,
            "note": "",
            "audio_reviewed": False,
            "review_disposition": "not_applicable",
        }
    )
    records[1].update(
        {
            "tr_corrected": "",
            "non_dialogue": True,
            "review_required": False,
            "note": "Dinlenerek kapı sesi olduğu doğrulandı.",
            "audio_reviewed": True,
            "review_disposition": "reviewed_non_dialogue",
        }
    )
    records[2].update(
        {
            "tr_corrected": "Bugün nasılsın?",
            "non_dialogue": False,
            "review_required": False,
            "note": "",
            "audio_reviewed": False,
            "review_disposition": "not_applicable",
        }
    )
    return records


def _speech_holes() -> list[dict[str, Any]]:
    return [
        {
            "hole_uid": "utt-2",
            "hole_index": 1,
            "start_ms": 3_000,
            "end_ms": 4_000,
            "clip_start_ms": 3_000,
            "clip_end_ms": 4_000,
            "reason": "unresolved speech hole",
            "context_before": "Meraba dunya.",
            "context_after": "Bugun nasilsin?",
            "risk_flags": [
                "unresolved_vad_speech",
                "manual_audio_review_required",
            ],
            "audio_member": "speech_hole_audio/utt-2.wav",
            "audio_sha256": "a" * 64,
            "audio_size_bytes": 1_024,
        }
    ]


def _hallucination_routing_fixture(
    *, discard: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = _input_utterances()
    corrections = _corrections()
    for record in (inputs[0], corrections[0]):
        record["risk_flags"].append("suspected_asr_hallucination")
    candidate_uid = "AH12-000001"
    candidates = [
        {
            "candidate_uid": candidate_uid,
            "candidate_index": 1,
            "utterance_uid": inputs[0]["utterance_uid"],
            "utterance_index": inputs[0]["utterance_index"],
            "start_ms": inputs[0]["coarse_start_ms"],
            "end_ms": inputs[0]["coarse_end_ms"],
            "clip_start_ms": 250,
            "clip_end_ms": 3_250,
            "reason": "ASR confidence/VAD evidence requires exact WAV review",
            "asr_text": inputs[0]["asr_text"],
            "youtube_text": inputs[0]["youtube_text"],
            "context_before": inputs[0]["context_before"],
            "context_after": inputs[0]["context_after"],
            "risk_flags": copy.deepcopy(inputs[0]["risk_flags"]),
            "asr_audit": copy.deepcopy(inputs[0]["asr_audit"]),
            "audio_member": f"asr_hallucination_audio/{candidate_uid}.wav",
            "audio_sha256": "b" * 64,
            "audio_size_bytes": 96_044,
        }
    ]
    corrections[0].update(
        {
            "tr_corrected": "" if discard else "Merhaba dünya.",
            "non_dialogue": discard,
            "review_required": False,
            "audio_reviewed": True,
            "review_disposition": (
                "discarded_asr_hallucination"
                if discard
                else "confirmed_dialogue"
            ),
            "note": (
                "WAV dinlendi; hedefte konuşma yok."
                if discard
                else "WAV dinlendi; diyalog doğrulandı."
            ),
        }
    )
    return inputs, corrections, candidates


def _raw() -> dict[str, Any]:
    raw = {
        "format_version": "2.0",
        "status": "completed",
        "input_sha256": "e" * 64,
        "episode": 12,
        "audio_sha256": AUDIO_SHA,
        "language": "tr",
        "segments": [
            {
                "segment_id": "main-1",
                "start_ms": 1_000,
                "end_ms": 2_500,
                "text": "Meraba dunya.",
                "source": "main",
                "word_timing_complete": True,
                "words": [
                    {"start_ms": 1_080, "end_ms": 1_550, "text": " Meraba"},
                    {"start_ms": 1_620, "end_ms": 2_350, "text": " dunya."},
                ],
            },
            {
                "segment_id": "main-2",
                "start_ms": 4_500,
                "end_ms": 6_000,
                "text": "Bugun nasilsin?",
                "source": "main",
                "word_timing_complete": True,
                "words": [
                    {"start_ms": 4_580, "end_ms": 5_050, "text": " Bugun"},
                    {"start_ms": 5_120, "end_ms": 5_850, "text": " nasilsin?"},
                ],
            },
        ],
        "required_acoustic_review_regions": [],
        "synthetic_timing_count": 0,
        "vad_fallback_reason": None,
        "independent_vad": True,
        "model": {"settings": asdict(RawASRV2Config())},
        "hallucination_review_utterance_uids": [],
        "vad_regions": [
            {
                "vad_region_index": 1,
                "start_ms": 1_000,
                "end_ms": 2_500,
                "source": "silero_vad",
            },
            {
                "vad_region_index": 2,
                "start_ms": 3_000,
                "end_ms": 4_000,
                "source": "silero_vad",
            },
            {
                "vad_region_index": 3,
                "start_ms": 4_500,
                "end_ms": 6_000,
                "source": "silero_vad",
            },
        ],
        "speech_coverage": {
            "config": {
                "vad_merge_gap_ms": 150,
                "word_padding_ms": 120,
                "word_merge_gap_ms": 300,
                "min_hole_ms": 750,
                "min_region_coverage_ratio": 0.70,
                "rescue_padding_ms": 500,
                "rescue_merge_gap_ms": 250,
                "max_word_outside_speech_ms": 120,
            }
        },
        "youtube_captions": [],
        "correction_utterances": _input_utterances(),
        "speech_hole_records": _speech_holes(),
        "asr_hallucination_records": [],
    }
    _bind_canonical_rescue_plan(raw)
    return raw


def _required_outside_review_fixture(
    disposition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _raw()
    excluded_segment = {
        "segment_id": "main-outside",
        "start_ms": 7_000,
        "end_ms": 9_000,
        "text": "Evet, tamam.",
        "source": "main",
        "word_timing_complete": False,
        "words": [
            {
                "start_ms": 7_000,
                "end_ms": 8_000,
                "text": " Evet,",
            }
        ],
        "asr_audit": {
            "avg_logprob": -0.2,
            "no_speech_prob": 0.02,
            "compression_ratio": 1.0,
            "temperature": 0.0,
        },
    }
    raw["segments"] = [excluded_segment]
    raw["required_acoustic_review_regions"] = required_acoustic_review_regions(
        raw["segments"], raw["vad_regions"]
    )
    _bind_canonical_rescue_plan(raw)
    hole = {
        "hole_uid": "utt-required-outside",
        "hole_index": 2,
        "start_ms": 7_000,
        "end_ms": 9_000,
        "clip_start_ms": 6_500,
        "clip_end_ms": 9_500,
        "reason": "excluded incomplete ASR hypothesis requires exact review",
        "context_before": "Bugun nasilsin?",
        "context_after": "",
        "risk_flags": [
            "unresolved_vad_speech",
            "manual_audio_review_required",
        ],
        "audio_member": "speech_hole_audio/utt-required-outside.wav",
        "audio_sha256": "d" * 64,
        "audio_size_bytes": 64_044,
    }
    raw["speech_hole_records"].append(hole)
    correction_input = {
        "utterance_uid": hole["hole_uid"],
        "utterance_index": 4,
        "coarse_start_ms": hole["start_ms"],
        "coarse_end_ms": hole["end_ms"],
        "asr_text": "",
        "youtube_text": "",
        "context_before": hole["context_before"],
        "context_after": hole["context_after"],
        "risk_flags": copy.deepcopy(hole["risk_flags"]),
        "asr_audit": {
            "avg_logprob": None,
            "no_speech_prob": None,
            "compression_ratio": None,
            "temperature": None,
        },
    }
    raw["correction_utterances"].append(correction_input)
    corrections = _corrections()
    correction = copy.deepcopy(correction_input)
    non_dialogue = disposition == "reviewed_non_dialogue"
    correction.update(
        {
            "tr_corrected": "" if non_dialogue else "Evet.",
            "non_dialogue": non_dialogue,
            "review_required": False,
            "note": "Exact WAV targeti dinlendi.",
            "audio_reviewed": True,
            "review_disposition": disposition,
        }
    )
    corrections.append(correction)
    return raw, corrections


def _bind_canonical_rescue_plan(raw: dict[str, Any]) -> None:
    words = []
    for segment in raw["segments"]:
        for word in segment.get("words", []):
            words.append(
                {
                    **copy.deepcopy(word),
                    "segment_id": segment["segment_id"],
                }
            )
    raw["words"] = words
    settings = RawASRV2Config()
    primary_segments = [
        segment
        for segment in raw["segments"]
        if not str(segment.get("source", "main")).startswith("rescue")
    ]
    initial = raw_asr_module._analyze_raw_speech_coverage(
        raw["vad_regions"],
        primary_segments,
        words,
        config=settings.speech_coverage_config(),
    )
    batches, budget = raw_asr_module._plan_rescue_batches(
        initial,
        raw["vad_regions"],
        primary_segments,
        words,
        settings,
    )
    raw["initial_speech_coverage"] = initial
    raw["rescue_batches"] = batches
    raw["rescue_budget_audit"] = budget


def _candidate_utterances() -> list[dict[str, Any]]:
    audit = {
        "avg_logprob": -0.2,
        "no_speech_prob": 0.02,
        "compression_ratio": 1.0,
        "temperature": 0.0,
    }
    return [
        {
            "utterance_uid": "utt-candidate",
            "utterance_index": 1,
            "coarse_start_ms": 100,
            "coarse_end_ms": 1_000,
            "asr_text": "Altyazı",
            "youtube_text": "",
            "context_before": "",
            "context_after": "Geldim.",
            "risk_flags": ["suspected_asr_hallucination"],
            "asr_audit": audit,
        },
        {
            "utterance_uid": "utt-real",
            "utterance_index": 2,
            "coarse_start_ms": 1_400,
            "coarse_end_ms": 2_300,
            "asr_text": "Geldim.",
            "youtube_text": "Geldim.",
            "context_before": "Altyazı",
            "context_after": "",
            "risk_flags": [],
            "asr_audit": audit,
        },
    ]


def _candidate_evidence() -> list[dict[str, Any]]:
    utterance = _candidate_utterances()[0]
    return [
        {
            "candidate_uid": "candidate-1",
            "candidate_index": 1,
            "utterance_uid": utterance["utterance_uid"],
            "utterance_index": utterance["utterance_index"],
            "start_ms": utterance["coarse_start_ms"],
            "end_ms": utterance["coarse_end_ms"],
            "clip_start_ms": 0,
            "clip_end_ms": 1_750,
            "reason": "ASR marker outside independent VAD",
            "asr_text": utterance["asr_text"],
            "youtube_text": utterance["youtube_text"],
            "context_before": utterance["context_before"],
            "context_after": utterance["context_after"],
            "risk_flags": utterance["risk_flags"],
            "asr_audit": utterance["asr_audit"],
            "audio_member": "asr_hallucination_audio/candidate-1.wav",
            "audio_sha256": "c" * 64,
            "audio_size_bytes": 1_024,
        }
    ]


def _candidate_corrections(disposition: str) -> list[dict[str, Any]]:
    records = copy.deepcopy(_candidate_utterances())
    discarded = disposition == "discarded_asr_hallucination"
    records[0].update(
        {
            "tr_corrected": "" if discarded else "Altyazı",
            "non_dialogue": discarded,
            "review_required": False,
            "audio_reviewed": True,
            "review_disposition": disposition,
            "note": "Exact review WAV dinlendi.",
        }
    )
    records[1].update(
        {
            "tr_corrected": "Geldim.",
            "non_dialogue": False,
            "review_required": False,
            "audio_reviewed": False,
            "review_disposition": "not_applicable",
            "note": "",
        }
    )
    return records


def _candidate_raw(*, include_candidate_vad: bool = False) -> dict[str, Any]:
    raw = _raw()
    raw["correction_utterances"] = _candidate_utterances()
    raw["speech_hole_records"] = []
    raw["asr_hallucination_records"] = _candidate_evidence()
    raw["vad_regions"] = []
    if include_candidate_vad:
        raw["vad_regions"].append(
            {
                "vad_region_index": 1,
                "start_ms": 100,
                "end_ms": 1_000,
                "source": "silero_vad",
            }
        )
    raw["vad_regions"].append(
        {
            "vad_region_index": len(raw["vad_regions"]) + 1,
            "start_ms": 1_400,
            "end_ms": 2_300,
            "source": "silero_vad",
        }
    )
    raw["segments"] = [
        {
            "segment_id": "main-candidate",
            "start_ms": 100,
            "end_ms": 1_000,
            "text": "Altyazi",
            "source": "main",
            "word_timing_complete": True,
            "words": [
                {"start_ms": 100, "end_ms": 1_000, "text": " Altyazi"}
            ],
        },
        {
            "segment_id": "main-real",
            "start_ms": 1_400,
            "end_ms": 2_300,
            "text": "Geldim.",
            "source": "main",
            "word_timing_complete": True,
            "words": [
                {"start_ms": 1_400, "end_ms": 2_300, "text": " Geldim."}
            ],
        },
    ]
    raw["required_acoustic_review_regions"] = required_acoustic_review_regions(
        raw["segments"], raw["vad_regions"]
    )
    _bind_canonical_rescue_plan(raw)
    return raw


def _candidate_alignment(
    directory: str,
    alignment_inputs: tuple[dict[str, Any], ...],
    *,
    candidate_start: float = 0.1,
    candidate_end: float = 0.9,
) -> dict[str, Any]:
    audio = Path(directory, "candidate-audio.flac")
    audio.write_bytes(AUDIO_BYTES)
    results: list[dict[str, Any]] = []
    if len(alignment_inputs) == 2:
        results.append(
            _alignment_result(
                [
                    {
                        "word": "Altyazı",
                        "start": candidate_start,
                        "end": candidate_end,
                        "score": 0.96,
                    }
                ]
            )
        )
    results.append(
        _alignment_result(
            [
                {
                    "word": "Geldim.",
                    "start": 1.45,
                    "end": 2.25,
                    "score": 0.97,
                }
            ]
        )
    )
    return align_corrected_segments(
        audio,
        alignment_inputs,
        whisperx_module=_FakeWhisperX(results),
        device="cpu",
    )


def _forced_alignment(
    directory: str,
    alignment_inputs: tuple[dict[str, Any], ...],
    *,
    second_text: str = "Bugün nasılsın?",
) -> dict[str, Any]:
    audio = Path(directory, "audio.flac")
    audio.write_bytes(AUDIO_BYTES)
    fake = _FakeWhisperX(
        [
            _alignment_result(
                [
                    {"word": "Merhaba", "start": 1.08, "end": 1.55, "score": 0.96},
                    {"word": "dünya.", "start": 1.62, "end": 2.35, "score": 0.95},
                ]
            ),
            _alignment_result(
                [
                    {"word": second_text.split()[0], "start": 4.58, "end": 5.05, "score": 0.94},
                    {"word": second_text.split()[1], "start": 5.12, "end": 5.85, "score": 0.95},
                ]
            ),
        ]
    )
    return align_corrected_segments(
        audio,
        alignment_inputs,
        whisperx_module=fake,
        device="cpu",
    )


def _required_outside_alignment(
    directory: str,
    alignment_inputs: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    audio = Path(directory, "required-audio.flac")
    audio.write_bytes(AUDIO_BYTES)
    fake = _FakeWhisperX(
        [
            _alignment_result(
                [
                    {"word": "Merhaba", "start": 1.08, "end": 1.55, "score": 0.96},
                    {"word": "dunya.", "start": 1.62, "end": 2.35, "score": 0.95},
                ]
            ),
            _alignment_result(
                [
                    {"word": "Bugun", "start": 4.58, "end": 5.05, "score": 0.94},
                    {"word": "nasilsin?", "start": 5.12, "end": 5.85, "score": 0.95},
                ]
            ),
            _alignment_result(
                [{"word": "Evet.", "start": 7.2, "end": 7.5, "score": 0.97}]
            ),
        ]
    )
    return align_corrected_segments(
        audio,
        alignment_inputs,
        whisperx_module=fake,
        device="cpu",
    )


class V2CorrectionRoutingTests(unittest.TestCase):
    def test_hash_bound_speaker_evidence_is_attached_without_changing_corrections(self) -> None:
        inputs = _input_utterances()
        corrections = _corrections()
        probabilities = [[0.0, 0.0, 0.0, 0.0] for _ in range(82)]
        for index in range(12, 31):
            probabilities[index] = [0.98, 0.01, 0.0, 0.0]
        for index in range(56, 75):
            probabilities[index] = [0.01, 0.98, 0.0, 0.0]
        pilot_data = {
            "format": "mas-native-diarization-pilot-1",
            "status": "REVIEW_REQUIRED",
            "production_acceptance": False,
            "episode": 12,
            "clip_identity": "ep12-residual-001-0-6500",
            "input_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "runtime": [],
            "sample_rate": 16000,
            "frame_seconds": 0.08,
            "frame_count": len(probabilities),
            "speaker_columns": 4,
            "probabilities": probabilities,
        }
        pilot = {"data": pilot_data, "sha256": digest(pilot_data)}
        evidence = build_speaker_evidence(
            episode=12,
            audio_sha256=AUDIO_SHA,
            input_utterances=inputs,
            intervals=[{"start": 0, "end": 6500}],
            pilot_wrappers=[pilot],
        )

        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=_speech_holes(),
            speaker_evidence=evidence,
            episode=12,
            audio_sha256=AUDIO_SHA,
        )

        self.assertTrue(bundle.alignment_inputs[0]["speaker_id"].endswith("column-1"))
        self.assertTrue(bundle.alignment_inputs[1]["speaker_id"].endswith("column-2"))
        self.assertEqual(
            bundle.window_audit[0]["speaker_evidence_sha256"], evidence["sha256"]
        )

    def test_padding_prevents_clipping_without_changing_uid_or_text(self) -> None:
        inputs = _input_utterances()
        corrections = _corrections()
        original_inputs = copy.deepcopy(inputs)
        original_corrections = copy.deepcopy(corrections)
        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=_speech_holes(),
        )

        self.assertEqual(DEFAULT_ALIGNMENT_PADDING_MS, 500)
        self.assertEqual(
            [item["utterance_uid"] for item in bundle.alignment_inputs],
            ["utt-1", "utt-3"],
        )
        self.assertEqual(
            [item["text"] for item in bundle.alignment_inputs],
            ["Merhaba dünya.", "Bugün nasılsın?"],
        )
        self.assertEqual(
            (bundle.alignment_inputs[0]["start_ms"], bundle.alignment_inputs[0]["end_ms"]),
            (500, 3_000),
        )
        self.assertEqual(
            (bundle.alignment_inputs[1]["start_ms"], bundle.alignment_inputs[1]["end_ms"]),
            (4_000, 6_500),
        )
        self.assertEqual(bundle.window_audit[0]["coarse_start_ms"], 1_000)
        self.assertEqual(bundle.window_audit[0]["alignment_start_ms"], 500)
        self.assertEqual(bundle.reviewed_non_dialogue[0]["start_ms"], 3_000)
        self.assertEqual(bundle.reviewed_non_dialogue[0]["end_ms"], 4_000)
        self.assertEqual(inputs, original_inputs)
        self.assertEqual(corrections, original_corrections)

    def test_left_padding_is_clamped_and_overlapping_windows_are_allowed(self) -> None:
        inputs = _input_utterances()
        corrections = _corrections()
        inputs[0]["coarse_start_ms"] = 200
        inputs[0]["coarse_end_ms"] = 1_800
        corrections[0]["coarse_start_ms"] = 200
        corrections[0]["coarse_end_ms"] = 1_800
        inputs[1]["coarse_start_ms"] = 2_000
        inputs[1]["coarse_end_ms"] = 3_000
        corrections[1]["coarse_start_ms"] = 2_000
        corrections[1]["coarse_end_ms"] = 3_000
        inputs[2]["coarse_start_ms"] = 3_200
        inputs[2]["coarse_end_ms"] = 4_500
        corrections[2]["coarse_start_ms"] = 3_200
        corrections[2]["coarse_end_ms"] = 4_500
        holes = _speech_holes()
        holes[0]["start_ms"] = 2_000
        holes[0]["end_ms"] = 3_000
        holes[0]["clip_start_ms"] = 2_000
        holes[0]["clip_end_ms"] = 3_000
        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=holes,
            alignment_padding_ms=1_000,
        )
        self.assertEqual(bundle.alignment_inputs[0]["start_ms"], 0)
        self.assertLess(
            bundle.alignment_inputs[1]["start_ms"],
            bundle.alignment_inputs[0]["end_ms"],
        )

    def test_any_pending_turkish_correction_blocks_alignment_with_count(self) -> None:
        corrections = _corrections()
        corrections[0]["review_required"] = True
        corrections[1]["review_required"] = True
        corrections[1]["non_dialogue"] = False
        corrections[1]["audio_reviewed"] = False
        corrections[1]["review_disposition"] = "pending_audio_review"
        with self.assertRaisesRegex(
            V2PipelineError,
            r"review_required_count=2.*utt-1.*utt-2",
        ):
            correction_records_to_alignment_inputs(
                _input_utterances(),
                corrections,
                speech_hole_records=_speech_holes(),
            )

    def test_ordinary_asr_utterance_cannot_be_declared_non_dialogue(self) -> None:
        corrections = _corrections()
        corrections[0]["tr_corrected"] = ""
        corrections[0]["non_dialogue"] = True
        corrections[0]["note"] = "Yanlışlıkla non-dialogue işaretlendi."
        with self.assertRaisesRegex(V2PipelineError, "audio-reviewed speech-hole"):
            correction_records_to_alignment_inputs(
                _input_utterances(),
                corrections,
                speech_hole_records=_speech_holes(),
            )

    def test_non_dialogue_must_preserve_exact_speech_hole_bounds(self) -> None:
        holes = _speech_holes()
        holes[0]["start_ms"] += 1
        with self.assertRaisesRegex(V2PipelineError, "changed immutable review bounds"):
            correction_records_to_alignment_inputs(
                _input_utterances(),
                _corrections(),
                speech_hole_records=holes,
            )

    def test_changed_immutable_correction_evidence_is_rejected(self) -> None:
        corrections = _corrections()
        corrections[0]["coarse_start_ms"] = 999
        with self.assertRaises(Exception):
            correction_records_to_alignment_inputs(
                _input_utterances(),
                corrections,
                speech_hole_records=_speech_holes(),
            )

    def test_discarded_hallucination_is_removed_but_never_clears_vad(self) -> None:
        bundle = correction_records_to_alignment_inputs(
            _candidate_utterances(),
            _candidate_corrections("discarded_asr_hallucination"),
            asr_hallucination_records=_candidate_evidence(),
        )

        self.assertEqual(
            [item["utterance_uid"] for item in bundle.alignment_inputs],
            ["utt-real"],
        )
        self.assertEqual(bundle.reviewed_non_dialogue, ())
        self.assertEqual(len(bundle.discarded_asr_hallucinations), 1)

    def test_discard_requires_exact_immutable_wav_candidate(self) -> None:
        with self.assertRaisesRegex(
            V2PipelineError,
            "candidates|audio-reviewed|immutable",
        ):
            correction_records_to_alignment_inputs(
                _candidate_utterances(),
                _candidate_corrections("discarded_asr_hallucination"),
                asr_hallucination_records=[],
            )

    def test_confirmed_hallucination_candidate_routes_to_reviewed_dialogue(self) -> None:
        inputs, corrections, candidates = _hallucination_routing_fixture(
            discard=False
        )
        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=_speech_holes(),
            asr_hallucination_records=candidates,
        )
        self.assertEqual(
            [item["utterance_uid"] for item in bundle.alignment_inputs],
            ["utt-1", "utt-3"],
        )
        self.assertEqual(len(bundle.reviewed_dialogue), 1)
        self.assertEqual(
            bundle.reviewed_dialogue[0]["classification"], "dialogue"
        )
        self.assertEqual(bundle.discarded_asr_hallucinations, ())
        self.assertTrue(bundle.alignment_inputs[0]["deletion_audio_reviewed"])

    def test_incomplete_provisional_dialogue_is_rejected_after_audio_review(self) -> None:
        inputs, corrections, candidates = _hallucination_routing_fixture(
            discard=False
        )
        for record in (inputs[0], corrections[0], candidates[0]):
            record["risk_flags"].append("incomplete_provisional_word_timing")

        with self.assertRaisesRegex(
            V2PipelineError,
            "incomplete provisional word timing.*utterance_uids=\\['utt-1'\\]",
        ):
            correction_records_to_alignment_inputs(
                inputs,
                corrections,
                speech_hole_records=_speech_holes(),
                asr_hallucination_records=candidates,
            )

    def test_incomplete_provisional_hallucination_discard_still_passes(self) -> None:
        inputs, corrections, candidates = _hallucination_routing_fixture(
            discard=True
        )
        for record in (inputs[0], corrections[0], candidates[0]):
            record["risk_flags"].append("incomplete_provisional_word_timing")

        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=_speech_holes(),
            asr_hallucination_records=candidates,
        )

        self.assertEqual(
            [item["utterance_uid"] for item in bundle.alignment_inputs],
            ["utt-3"],
        )
        self.assertEqual(len(bundle.discarded_asr_hallucinations), 1)

    def test_complete_speech_hole_rescue_dialogue_still_passes(self) -> None:
        inputs, corrections, candidates = _hallucination_routing_fixture(
            discard=False
        )
        for record in (inputs[0], corrections[0], candidates[0]):
            record["risk_flags"] = [
                "suspected_asr_hallucination",
                "speech_hole_rescue_asr",
            ]

        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=_speech_holes(),
            asr_hallucination_records=candidates,
        )

        self.assertEqual(
            [item["utterance_uid"] for item in bundle.alignment_inputs],
            ["utt-1", "utt-3"],
        )
        self.assertTrue(bundle.alignment_inputs[0]["deletion_audio_reviewed"])

    def test_discarded_hallucination_is_not_vad_clearing_review(self) -> None:
        inputs, corrections, candidates = _hallucination_routing_fixture(
            discard=True
        )
        bundle = correction_records_to_alignment_inputs(
            inputs,
            corrections,
            speech_hole_records=_speech_holes(),
            asr_hallucination_records=candidates,
        )
        self.assertEqual(
            [item["utterance_uid"] for item in bundle.alignment_inputs],
            ["utt-3"],
        )
        self.assertEqual(bundle.reviewed_dialogue, ())
        self.assertEqual(len(bundle.discarded_asr_hallucinations), 1)
        self.assertEqual(
            bundle.discarded_asr_hallucinations[0]["disposition"],
            "discarded_asr_hallucination",
        )
        self.assertEqual(
            [item["review_id"] for item in bundle.reviewed_non_dialogue],
            ["tr-correction:utt-2"],
        )


class V2PipelineIntegrationTests(unittest.TestCase):
    def test_partial_vad_ceiling_does_not_relax_exact_word_or_review_bounds(self):
        from types import SimpleNamespace
        from mas.engine.workflow import _parent_part_vad_gate

        regions = [{"vad_region_index": 1, "start_ms": 100, "end_ms": 1001, "source": "silero_vad"}]
        raw = {"episode": 14, "audio_sha256": AUDIO_SHA, "vad_regions": copy.deepcopy(regions)}
        aligned = {"words": [{"start_ms": 100, "end_ms": 1000}], "alignment_sha256": "4" * 64}
        preparation = SimpleNamespace(reviewed_dialogue=[], reviewed_non_dialogue=[])
        lineage = {"episode": 14, "part_id": "part-001", "audio": {"sha256": AUDIO_SHA},
                   "sample_rate_hz": 16000, "start_sample": 0, "end_sample": 16009,
                   "sample_count": 16009, "plan_sha256": "1" * 64,
                   "parent_source_sha256": "2" * 64, "parent_audio_sha256": "3" * 64,
                   "parent_vad_sha256": digest(regions)}
        report = _parent_part_vad_gate(raw, aligned, preparation, regions, lineage, SpeechCoverageConfig())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(aligned["words"][0]["end_ms"], 1000)
        invalid_regions = [{**regions[0], "end_ms": 1002}]
        with self.assertRaisesRegex(V2PipelineError, "exact child sample range"):
            _parent_part_vad_gate(raw, aligned, preparation, invalid_regions,
                {**lineage, "parent_vad_sha256": digest(invalid_regions)}, SpeechCoverageConfig())
        for key in ("words", "reviewed_dialogue", "reviewed_non_dialogue"):
            with self.subTest(key=key):
                changed_aligned = copy.deepcopy(aligned)
                changed_preparation = copy.deepcopy(preparation)
                if key == "words":
                    changed_aligned[key][0]["end_ms"] = 1001
                else:
                    setattr(changed_preparation, key, [{"start_ms": 100, "end_ms": 1001}])
                with self.assertRaisesRegex(V2PipelineError, "exact child sample range"):
                    _parent_part_vad_gate(raw, changed_aligned, changed_preparation, regions, lineage,
                                         SpeechCoverageConfig())

    def test_partial_parent_vad_is_independently_checked_and_bound_before_schema_freeze(self):
        raw, corrections = _raw(), _corrections()
        prepared = correction_records_to_alignment_inputs(
            raw["correction_utterances"], corrections, speech_hole_records=raw["speech_hole_records"])
        parent_vad = copy.deepcopy(raw["vad_regions"])
        parent_vad[0]["start_ms"] -= 20
        parent_vad[0]["end_ms"] += 20
        lineage = {"episode": 12, "part_id": "part-001", "audio": {"sha256": AUDIO_SHA},
                   "sample_rate_hz": 16000, "start_sample": 0, "end_sample": 10000 * 16,
                   "sample_count": 10000 * 16, "plan_sha256": "1" * 64,
                   "parent_source_sha256": "2" * 64, "parent_audio_sha256": "3" * 64,
                   "parent_vad_sha256": digest(parent_vad)}
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, prepared.alignment_inputs)
            whole = build_strict_v2_artifacts(raw, corrections, alignment)
            part = build_strict_v2_artifacts(raw, corrections, alignment,
                parent_vad_regions=parent_vad, part_lineage=lineage)
            self.assertNotEqual(whole.schema["schema_sha256"], part.schema["schema_sha256"])
            self.assertNotIn("parent_part_vad_v1", whole.speech_coverage_report)
            evidence = part.speech_coverage_report["parent_part_vad_v1"]
            self.assertEqual(evidence["coverage"]["config"], raw["speech_coverage"]["config"])
            self.assertEqual(evidence["parent_vad_sha256"], digest(parent_vad))
            self.assertEqual(evidence["coverage"]["metrics"]["input_vad_region_count"], len(parent_vad))
            missing = parent_vad + [{"vad_region_index": 4, "start_ms": 8000, "end_ms": 9000, "source": "silero_vad"}]
            with self.assertRaisesRegex(V2PipelineError, "complete projected parent VAD"):
                build_strict_v2_artifacts(raw, corrections, alignment, parent_vad_regions=missing,
                    part_lineage={**lineage, "parent_vad_sha256": digest(missing)})
            with self.assertRaisesRegex(V2PipelineError, "Projected parent VAD identity changed"):
                build_strict_v2_artifacts(raw, corrections, alignment, parent_vad_regions=missing, part_lineage=lineage)
            with self.assertRaisesRegex(V2PipelineError, "Partial audio lineage"):
                build_strict_v2_artifacts(raw, corrections, alignment, parent_vad_regions=parent_vad,
                    part_lineage={**lineage, "audio": {"sha256": "f" * 64}})
            with self.assertRaisesRegex(V2PipelineError, "exact child sample range"):
                build_strict_v2_artifacts(raw, corrections, alignment, parent_vad_regions=parent_vad,
                    part_lineage={**lineage, "end_sample": 5900 * 16, "sample_count": 5900 * 16})
            with self.assertRaisesRegex(V2PipelineError, "Partial audio lineage"):
                build_strict_v2_artifacts(raw, corrections, alignment, parent_vad_regions=parent_vad)

    def test_production_policy_binds_segmentation_timing_schema_and_pack(self):
        import yaml
        from dataclasses import replace
        from mas.engine.id_translation import build_production_translation_policy
        from mas.engine.segmentation import SegmentationConfig

        config_dir = Path(__file__).resolve().parents[2] / "config" / "production"
        configs = [yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
                   for name in ("series.yaml", "names.yaml", "religious_terms.yaml")]
        policy = build_production_translation_policy(*configs)
        raw, corrections = _raw(), _corrections()
        prepared = correction_records_to_alignment_inputs(
            raw["correction_utterances"], corrections, speech_hole_records=raw["speech_hole_records"])
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, prepared.alignment_inputs)
            artifacts = build_strict_v2_artifacts(raw, corrections, alignment, production_policy=policy)
            self.assertEqual(artifacts.schema["production_policy"], policy)
            self.assertEqual(artifacts.timing_qa_report["config"], policy["timing_qa"])
            manifest = create_v2_id_translation_pack(artifacts, Path(temporary) / "input.zip",
                                                      glossary=policy["glossary"])
            self.assertEqual(manifest["schema_sha256"], artifacts.schema["schema_sha256"])
            with self.assertRaisesRegex(V2PipelineError, "differs from frozen production policy"):
                build_strict_v2_artifacts(raw, corrections, alignment, production_policy=policy,
                                         segmentation_config=SegmentationConfig(target_chars_per_line=40))
            changed_report = copy.deepcopy(artifacts.timing_qa_report)
            changed_report["config"]["maximum_cps"] = 19
            with self.assertRaisesRegex(V2PipelineError, "Pre-ID timing QA differs"):
                create_v2_id_translation_pack(replace(artifacts, timing_qa_report=changed_report),
                                             Path(temporary) / "bad.zip", glossary=policy["glossary"])

    def test_machine_audio_decision_requires_and_binds_external_audit(self) -> None:
        raw = _candidate_raw()
        corrections = _candidate_corrections("confirmed_dialogue")
        corrections[0]["note"] = "machine_audio_review_v2: corroborated"
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            asr_hallucination_records=raw["asr_hallucination_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _candidate_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(V2PipelineError, "hash-bound"):
                build_strict_v2_artifacts(raw, corrections, alignment)
            evidence = raw["asr_hallucination_records"][0]
            audit = {
                "format": AUDIO_REVIEW_V2_FORMAT,
                "format_version": AUDIO_REVIEW_V2_VERSION,
                "status": "PASS",
                "pending_count": 0,
                "correction_output_sha256": compute_output_sha256(corrections),
                "outcomes": [
                    {
                        "utterance_uid": corrections[0]["utterance_uid"],
                        "evidence_kind": "asr_caption_candidate",
                        "evidence_uid": evidence["candidate_uid"],
                        "audio_member": evidence["audio_member"],
                        "audio_sha256": evidence["audio_sha256"],
                        "decision": "confirmed_dialogue",
                        "source": "secondary_asr",
                        "tr_corrected_sha256": hashlib.sha256(
                            corrections[0]["tr_corrected"].encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            }
            audit["audio_review_sha256"] = sha256_json(audit)
            artifacts = build_strict_v2_artifacts(
                raw,
                corrections,
                alignment,
                acoustic_audio_review=audit,
            )
        self.assertEqual(
            artifacts.audio_review_report["outcomes"][0]["review_source"],
            "secondary_asr",
        )
        self.assertEqual(
            artifacts.audio_review_report["outcomes"][0][
                "acoustic_audit_sha256"
            ],
            audit["audio_review_sha256"],
        )

    def test_alignment_raw_coverage_schema_and_id_pack_round_trip(self) -> None:
        raw = _raw()
        corrections = _corrections()
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            coverage = recompute_final_speech_coverage(
                raw,
                alignment,
                preparation.reviewed_non_dialogue,
                expected_alignment_inputs=preparation.alignment_inputs,
            )
            self.assertEqual(coverage["status"], "PASS")
            self.assertEqual(coverage["unresolved_speech_region_count"], 0)
            self.assertEqual(coverage["config"]["min_region_coverage_ratio"], 0.70)
            self.assertEqual(
                coverage["metrics"]["reviewed_non_dialogue_issue_count"],
                1,
            )

            artifacts = build_strict_v2_artifacts(
                raw,
                corrections,
                alignment,
            )
            self.assertEqual(artifacts.episode, 12)
            self.assertTrue(artifacts.timing_qa_report["passed"])
            self.assertTrue(artifacts.timing_qa_report["word_ownership_checked"])
            self.assertEqual(artifacts.alignment_report["words"], alignment["words"])
            self.assertEqual(artifacts.schema["schema_version"], "2.0")
            self.assertEqual(artifacts.schema["block_count"], 2)
            self.assertEqual(artifacts.schema["audio_sha256"], AUDIO_SHA)
            self.assertEqual(len(artifacts.schema["schema_sha256"]), 64)
            self.assertTrue(
                all(
                    block["alignment_provenance"]["alignment_sha256"]
                    == alignment["alignment_sha256"]
                    for block in artifacts.schema["blocks"]
                )
            )
            self.assertTrue(
                all(
                    block["alignment_provenance"]["model_name"]
                    == alignment["provenance"]["model_name"]
                    for block in artifacts.schema["blocks"]
                )
            )
            self.assertIn("segments", artifacts.alignment_report)

            first_pack = Path(temporary, "id-pack-a.zip")
            second_pack = Path(temporary, "id-pack-b.zip")
            manifest_a = create_v2_id_translation_pack(
                artifacts,
                first_pack,
                batch_size=1,
            )
            manifest_b = create_v2_id_translation_pack(
                artifacts,
                second_pack,
                batch_size=1,
            )
            self.assertEqual(first_pack.read_bytes(), second_pack.read_bytes())
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(
                validate_id_translation_pack(
                    first_pack,
                    expected_schema=artifacts.schema,
                ),
                manifest_a,
            )

    def test_forced_alignment_must_be_bound_to_exact_corrected_windows(self) -> None:
        raw = _raw()
        corrections = _corrections()
        expected = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            speech_hole_records=raw["speech_hole_records"],
        )
        changed_corrections = _corrections()
        changed_corrections[2]["tr_corrected"] = "Yarın nasılsın?"
        changed = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            changed_corrections,
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(
                temporary,
                changed.alignment_inputs,
                second_text="Yarın nasılsın?",
            )
            with self.assertRaisesRegex(V2PipelineError, "changed text"):
                recompute_final_speech_coverage(
                    raw,
                    alignment,
                    expected.reviewed_non_dialogue,
                    expected_alignment_inputs=expected.alignment_inputs,
                )

    def test_uncovered_independent_vad_speech_cannot_lock_schema(self) -> None:
        raw = _raw()
        raw["vad_regions"].append(
            {
                "vad_region_index": 4,
                "start_ms": 8_000,
                "end_ms": 9_500,
                "source": "silero_vad",
            }
        )
        _bind_canonical_rescue_plan(raw)
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(V2PipelineError, "do not safely match"):
                build_strict_v2_artifacts(raw, _corrections(), alignment)

            loose_override = SpeechCoverageConfig(
                min_hole_ms=10_000,
                min_region_coverage_ratio=0.0,
            )
            with self.assertRaisesRegex(
                V2PipelineError,
                "must exactly match the raw-ASR speech coverage policy",
            ):
                build_strict_v2_artifacts(
                    raw,
                    _corrections(),
                    alignment,
                    coverage_config=loose_override,
                )

    def test_required_outside_vad_confirmed_dialogue_aligns_and_covers(self) -> None:
        raw, corrections = _required_outside_review_fixture(
            "confirmed_dialogue"
        )
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            speech_hole_records=raw["speech_hole_records"],
        )

        self.assertEqual(
            [item["review_id"] for item in preparation.reviewed_dialogue],
            ["tr-correction:utt-required-outside"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _required_outside_alignment(
                temporary, preparation.alignment_inputs
            )
            artifacts = build_strict_v2_artifacts(raw, corrections, alignment)

        self.assertEqual(artifacts.speech_coverage_report["status"], "PASS")
        self.assertEqual(
            artifacts.speech_coverage_report["unresolved_speech_region_count"],
            0,
        )

    def test_required_outside_vad_exact_non_dialogue_review_clears_issue(self) -> None:
        raw, corrections = _required_outside_review_fixture(
            "reviewed_non_dialogue"
        )
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(
                temporary, preparation.alignment_inputs
            )
            artifacts = build_strict_v2_artifacts(raw, corrections, alignment)

        self.assertEqual(artifacts.speech_coverage_report["status"], "PASS")
        required_issues = [
            issue
            for issue in artifacts.speech_coverage_report["coverage_issues"]
            if issue["issue_kind"] == "required_uncovered_speech"
        ]
        self.assertTrue(required_issues)
        self.assertTrue(
            all(
                issue["classification"] == "reviewed_non_dialogue"
                for issue in required_issues
            )
        )
        for required_target in (None, {"start_ms": 7_000, "end_ms": 8_001}):
            with self.subTest(required_target=required_target):
                changed = copy.deepcopy(raw)
                changed["speech_hole_records"] = _speech_holes()
                if required_target is not None:
                    changed["speech_hole_records"].append(
                        {
                            **raw["speech_hole_records"][-1],
                            **required_target,
                        }
                    )
                report = recompute_final_speech_coverage(
                    changed,
                    alignment,
                    preparation.reviewed_non_dialogue,
                    expected_alignment_inputs=preparation.alignment_inputs,
                )
                self.assertEqual(report["status"], "FAIL")

    def test_discarded_hallucination_cannot_clear_required_outside_vad(self) -> None:
        raw, _corrections_with_required = _required_outside_review_fixture(
            "confirmed_dialogue"
        )
        base_preparation = correction_records_to_alignment_inputs(
            _input_utterances(),
            _corrections(),
            speech_hole_records=_speech_holes(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(
                temporary, base_preparation.alignment_inputs
            )
            report = recompute_final_speech_coverage(
                raw,
                alignment,
                base_preparation.reviewed_non_dialogue,
                expected_alignment_inputs=base_preparation.alignment_inputs,
            )

        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["unresolved_speech_region_count"], 0)

    def test_required_acoustic_review_regions_are_mandatory_and_immutable(self) -> None:
        raw, _corrections_with_required = _required_outside_review_fixture(
            "confirmed_dialogue"
        )
        for mutation, message in (
            (
                lambda value: value.pop("required_acoustic_review_regions"),
                "must be a list",
            ),
            (
                lambda value: value["required_acoustic_review_regions"][0].update(
                    end_ms=8_001
                ),
                "does not match",
            ),
            (
                lambda value: value["required_acoustic_review_regions"][0].update(
                    source_sha256="0" * 64
                ),
                "does not match",
            ),
        ):
            with self.subTest(message=message):
                changed = copy.deepcopy(raw)
                mutation(changed)
                with self.assertRaisesRegex(V2PipelineError, message):
                    recompute_final_speech_coverage(changed, {}, [])

    def test_required_outside_inventory_keeps_wordless_parent_and_partial_word_tail(self) -> None:
        from mas.engine.workflow import _validate_raw_vad_inventory

        cases = (
            (
                {
                    "segment_id": "main-wordless",
                    "start_ms": 7_000,
                    "end_ms": 9_000,
                    "text": "Eksik hipotez",
                    "words": [],
                },
                [{"vad_region_index": 1, "start_ms": 0, "end_ms": 1_000, "source": "silero_vad"}],
                (7_000, 9_000),
            ),
            (
                {
                    "segment_id": "main-partial-word",
                    "start_ms": 500,
                    "end_ms": 1_500,
                    "text": "Yarim eksik",
                    "words": [
                        {"start_ms": 900, "end_ms": 1_100, "text": " Yarim"}
                    ],
                },
                [{"vad_region_index": 1, "start_ms": 0, "end_ms": 1_000, "source": "silero_vad"}],
                (1_000, 1_500),
            ),
        )
        for segment, vad_regions, expected_bounds in cases:
            with self.subTest(segment_id=segment["segment_id"]):
                raw = _raw()
                raw["segments"] = [segment]
                raw["vad_regions"] = vad_regions
                raw["required_acoustic_review_regions"] = (
                    required_acoustic_review_regions(raw["segments"], vad_regions)
                )
                _bind_canonical_rescue_plan(raw)
                trusted = _validate_raw_vad_inventory(raw)
                self.assertEqual(
                    (
                        trusted["required_acoustic_review_regions"][0]["start_ms"],
                        trusted["required_acoustic_review_regions"][0]["end_ms"],
                    ),
                    expected_bounds,
                )
                raw["required_acoustic_review_regions"][0]["start_ms"] += 1
                with self.assertRaisesRegex(V2PipelineError, "does not match"):
                    _validate_raw_vad_inventory(raw)

    def test_rescue_plan_reports_and_batches_are_recomputed_downstream(self) -> None:
        from mas.engine.workflow import _validate_raw_vad_inventory

        raw, _corrections_with_required = _required_outside_review_fixture(
            "confirmed_dialogue"
        )
        mutations = (
            lambda value: value["initial_speech_coverage"]["metrics"].update(
                coverage_issue_count=999
            ),
            lambda value: value["rescue_batches"][0].update(end_ms=8_999),
            lambda value: value["rescue_budget_audit"].update(
                actual_batch_count=999
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(raw)
                mutation(changed)
                with self.assertRaisesRegex(
                    V2PipelineError,
                    "rescue execution proof is invalid",
                ):
                    _validate_raw_vad_inventory(changed)

    def test_current_policy_raw_rejects_deleted_segment_inventory(self) -> None:
        raw = _raw()
        raw.pop("segments", None)
        raw.pop("required_acoustic_review_regions", None)
        with self.assertRaisesRegex(
            V2PipelineError,
            "segments must be a list",
        ):
            recompute_final_speech_coverage(raw, {}, [])

    def test_missing_segments_fail_when_incomplete_evidence_exists(self) -> None:
        def excluded(raw: dict[str, Any]) -> None:
            raw["excluded_incomplete_segments"] = [{"segment_id": "main-1"}]

        def correction_flag(raw: dict[str, Any]) -> None:
            raw["correction_utterances"][0]["risk_flags"].append(
                "incomplete_provisional_word_timing"
            )

        def coverage_source(raw: dict[str, Any]) -> None:
            raw["speech_coverage"]["coverage_issues"] = [
                {
                    "issue_kind": "required_uncovered_speech",
                    "required_source_records": [
                        {"source_id": "main-1", "source_sha256": "a" * 64}
                    ],
                }
            ]

        def required_region(raw: dict[str, Any]) -> None:
            raw["required_acoustic_review_regions"] = [
                {
                    "start_ms": 100,
                    "end_ms": 200,
                    "source_id": "main-1",
                    "source_sha256": "a" * 64,
                }
            ]

        for mutation in (excluded, correction_flag, coverage_source, required_region):
            with self.subTest(mutation=mutation.__name__):
                raw = _raw()
                raw.pop("segments", None)
                raw.pop("required_acoustic_review_regions", None)
                mutation(raw)
                with self.assertRaisesRegex(
                    V2PipelineError,
                    "segments must be a list",
                ):
                    recompute_final_speech_coverage(raw, {}, [])

    def test_aligned_word_wholly_outside_vad_cannot_lock_schema(self) -> None:
        raw = _raw()
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary, "audio.flac")
            audio.write_bytes(AUDIO_BYTES)
            fake = _FakeWhisperX(
                [
                    _alignment_result(
                        [
                            {
                                "word": "Merhaba",
                                "start": 0.50,
                                "end": 1.001,
                                "score": 0.96,
                            },
                            {
                                "word": "dünya.",
                                "start": 1.10,
                                "end": 2.35,
                                "score": 0.95,
                            },
                        ]
                    ),
                    _alignment_result(
                        [
                            {
                                "word": "Bugün",
                                "start": 4.58,
                                "end": 5.05,
                                "score": 0.94,
                            },
                            {
                                "word": "nasılsın?",
                                "start": 5.12,
                                "end": 5.85,
                                "score": 0.95,
                            },
                        ]
                    ),
                ]
            )
            alignment = align_corrected_segments(
                audio,
                preparation.alignment_inputs,
                whisperx_module=fake,
                device="cpu",
            )
            with self.assertRaisesRegex(
                V2PipelineError,
                "strict_word_vad_unsafe_word_count=1",
            ):
                build_strict_v2_artifacts(raw, _corrections(), alignment)

    def test_asr_word_derived_vad_fallback_is_rejected(self) -> None:
        raw = _raw()
        raw["vad_fallback_reason"] = "silero_vad_returned_no_regions"
        raw["independent_vad"] = False
        raw["vad_regions"][0]["source"] = "vad_filtered_word_timing_fallback"
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(V2PipelineError, "non-publishable"):
                recompute_final_speech_coverage(
                    raw,
                    alignment,
                    preparation.reviewed_non_dialogue,
                )

    def test_debug_override_artifact_is_explicitly_non_publishable(self) -> None:
        raw = _raw()
        raw["independent_vad"] = False
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(V2PipelineError, "non-publishable"):
                build_strict_v2_artifacts(raw, _corrections(), alignment)

    def test_stale_audio_alignment_is_rejected_before_coverage_or_schema(self) -> None:
        raw = _raw()
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            raw["audio_sha256"] = "d" * 64
            with self.assertRaisesRegex(V2PipelineError, "stale or different audio"):
                build_strict_v2_artifacts(raw, _corrections(), alignment)

    def test_missing_or_mutated_raw_coverage_policy_is_rejected(self) -> None:
        raw = _raw()
        raw["speech_coverage"]["config"]["unexpected"] = 1
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(V2PipelineError, "config field mismatch"):
                recompute_final_speech_coverage(
                    raw,
                    alignment,
                    preparation.reviewed_non_dialogue,
                )

    def test_missing_raw_coverage_config_cannot_restore_loose_defaults(self) -> None:
        raw = _raw()
        raw["speech_coverage"].pop("config")
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(
                V2PipelineError,
                "does not preserve the exact speech coverage config",
            ):
                recompute_final_speech_coverage(
                    raw,
                    alignment,
                    preparation.reviewed_non_dialogue,
                )

    def test_loose_hash_bound_raw_policy_cannot_publish(self) -> None:
        raw = _raw()
        raw["model"]["settings"]["rescue_min_hole_ms"] = 10_000
        raw["model"]["settings"]["rescue_min_coverage_ratio"] = 0.0
        raw["speech_coverage"]["config"]["min_hole_ms"] = 10_000
        raw["speech_coverage"]["config"]["min_region_coverage_ratio"] = 0.0
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(
                V2PipelineError,
                "canonical V2 beta policy",
            ):
                build_strict_v2_artifacts(raw, _corrections(), alignment)

    def test_embedded_coverage_policy_must_match_canonical_model_policy(self) -> None:
        raw = _raw()
        raw["speech_coverage"]["config"]["min_hole_ms"] = 10_000
        raw["speech_coverage"]["config"]["min_region_coverage_ratio"] = 0.0
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            _corrections(),
            speech_hole_records=raw["speech_hole_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _forced_alignment(temporary, preparation.alignment_inputs)
            with self.assertRaisesRegex(
                V2PipelineError,
                "canonical V2 beta publication policy",
            ):
                build_strict_v2_artifacts(raw, _corrections(), alignment)

    def test_reviewed_discard_is_omitted_from_schema_and_not_vad_clearance(self) -> None:
        raw = _candidate_raw()
        corrections = _candidate_corrections("discarded_asr_hallucination")
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            asr_hallucination_records=raw["asr_hallucination_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _candidate_alignment(
                temporary, preparation.alignment_inputs
            )
            artifacts = build_strict_v2_artifacts(raw, corrections, alignment)

        self.assertEqual(artifacts.schema["block_count"], 1)
        self.assertNotIn("Altyazı", artifacts.schema["blocks"][0]["tr_text"])
        self.assertEqual(
            artifacts.audio_review_report[
                "discarded_asr_hallucination_count"
            ],
            1,
        )
        self.assertEqual(
            artifacts.speech_coverage_report["reviewed_non_dialogue_records"],
            [],
        )

        raw_with_candidate_vad = _candidate_raw(include_candidate_vad=True)
        with self.assertRaisesRegex(V2PipelineError, "do not safely match"):
            build_strict_v2_artifacts(
                raw_with_candidate_vad,
                corrections,
                alignment,
            )

    def test_confirmed_dialogue_is_exact_exception_and_vad_boundary(self) -> None:
        raw = _candidate_raw()
        corrections = _candidate_corrections("confirmed_dialogue")
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            asr_hallucination_records=raw["asr_hallucination_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _candidate_alignment(
                temporary, preparation.alignment_inputs
            )
            artifacts = build_strict_v2_artifacts(raw, corrections, alignment)

        self.assertEqual(artifacts.schema["block_count"], 2)
        self.assertEqual(
            [block["tr_text"] for block in artifacts.schema["blocks"]],
            ["Altyazı", "Geldim."],
        )
        self.assertEqual(artifacts.schema["blocks"][1]["start_ms"], 1_450)
        self.assertIn(
            "vad_region_transition",
            artifacts.segmented_blocks[1]["vad_info"]["boundary_before"],
        )
        self.assertEqual(
            artifacts.strict_word_vad_report[
                "confirmed_dialogue_exception_word_count"
            ],
            1,
        )
        self.assertEqual(
            artifacts.strict_word_vad_report[
                "confirmed_dialogue_coverage_fail_count"
            ],
            0,
        )

    def test_confirmed_dialogue_target_still_requires_complete_coverage(self) -> None:
        raw = _candidate_raw()
        corrections = _candidate_corrections("confirmed_dialogue")
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            corrections,
            asr_hallucination_records=raw["asr_hallucination_records"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            alignment = _candidate_alignment(
                temporary,
                preparation.alignment_inputs,
                candidate_start=0.9,
                candidate_end=1.0,
            )
            with self.assertRaisesRegex(
                V2PipelineError,
                "confirmed_dialogue_coverage_fail_count=1",
            ):
                build_strict_v2_artifacts(raw, corrections, alignment)

    def test_word_spanning_two_vad_islands_cannot_use_tiny_edge_contacts(self) -> None:
        raw = _candidate_raw()
        raw["correction_utterances"] = [
            {
                **_candidate_utterances()[1],
                "utterance_uid": "utt-real",
                "utterance_index": 1,
                "coarse_start_ms": 100,
                "coarse_end_ms": 1_100,
                "asr_text": "Geldim.",
                "youtube_text": "Geldim.",
                "context_before": "",
            }
        ]
        raw["asr_hallucination_records"] = []
        raw["vad_regions"] = [
            {
                "vad_region_index": 1,
                "start_ms": 0,
                "end_ms": 300,
                "source": "silero_vad",
            },
            {
                "vad_region_index": 2,
                "start_ms": 900,
                "end_ms": 1_200,
                "source": "silero_vad",
            },
        ]
        _bind_canonical_rescue_plan(raw)
        correction = copy.deepcopy(raw["correction_utterances"])
        correction[0].update(
            {
                "tr_corrected": "Geldim.",
                "non_dialogue": False,
                "review_required": False,
                "audio_reviewed": False,
                "review_disposition": "not_applicable",
                "note": "",
            }
        )
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"], correction
        )
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary, "islands.flac")
            audio.write_bytes(AUDIO_BYTES)
            alignment = align_corrected_segments(
                audio,
                preparation.alignment_inputs,
                whisperx_module=_FakeWhisperX(
                    [
                        _alignment_result(
                            [
                                {
                                    "word": "Geldim.",
                                    "start": 0.1,
                                    "end": 1.1,
                                    "score": 0.97,
                                }
                            ]
                        )
                    ]
                ),
                device="cpu",
            )
            with self.assertRaisesRegex(
                V2PipelineError,
                "strict_word_vad_unsafe_word_count=1",
            ):
                build_strict_v2_artifacts(raw, correction, alignment)


if __name__ == "__main__":
    unittest.main()
