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
from mas.engine.raw_asr import RawASRV2Config
from mas.engine.tr_correction import compute_output_sha256
from mas.engine.speech_coverage import SpeechCoverageConfig
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
    return {
        "format_version": "2.0",
        "status": "completed",
        "input_sha256": "e" * 64,
        "episode": 12,
        "audio_sha256": AUDIO_SHA,
        "language": "tr",
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


class V2CorrectionRoutingTests(unittest.TestCase):
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

        self.assertEqual(DEFAULT_ALIGNMENT_PADDING_MS, 900)
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
            (100, 3_400),
        )
        self.assertEqual(
            (bundle.alignment_inputs[1]["start_ms"], bundle.alignment_inputs[1]["end_ms"]),
            (3_600, 6_900),
        )
        self.assertEqual(bundle.window_audit[0]["coarse_start_ms"], 1_000)
        self.assertEqual(bundle.window_audit[0]["alignment_start_ms"], 100)
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
