from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from src.audio_review_v2 import (
    _FasterWhisperReviewDecoder,
    AudioReviewV2Config,
    AudioReviewV2Error,
    resolve_tr_audio_reviews_v2,
    validate_audio_review_v2_report,
)
from src.tr_correction import (
    create_tr_correction_output,
    create_tr_correction_pack,
    validate_tr_correction_output,
)


AUDIT = {
    "avg_logprob": -0.2,
    "no_speech_prob": 0.02,
    "compression_ratio": 1.0,
    "temperature": 0.0,
}
BLANK_AUDIT = {key: None for key in AUDIT}


def _wav(duration_ms: int = 1_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * round(16_000 * duration_ms / 1000))
    return output.getvalue()


class FakeDecoder:
    def __init__(self, results: list[Mapping[str, Any]]) -> None:
        self.results = list(results)
        self.call_count = 0

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {"engine": "fake", "call_count": self.call_count}

    def decode(
        self, audio_path: Path, *, initial_prompt: str | None = None
    ) -> Mapping[str, Any]:
        self.call_count += 1
        return self.results.pop(0)

    def close(self) -> None:
        pass


def _decoded(text: str = "", start_ms: int = 100, end_ms: int = 500) -> dict:
    if not text:
        return {"segments": [], "words": []}
    return {
        "segments": [{"text": text, "start_ms": start_ms, "end_ms": end_ms}],
        "words": [
            {
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "probability": 0.95,
            }
        ],
    }


def _candidate(*, orphan: bool = False) -> tuple[dict, dict]:
    utterance = {
        "utterance_uid": "candidate-1",
        "utterance_index": 1,
        "coarse_start_ms": 100,
        "coarse_end_ms": 500,
        "asr_text": "" if orphan else "Merhaba",
        "youtube_text": "Merhaba" if orphan else "",
        "context_before": "",
        "context_after": "Geldim",
        "risk_flags": [
            "orphan_youtube_caption" if orphan else "suspected_asr_hallucination",
            "manual_audio_review_required",
        ],
        "asr_audit": copy.deepcopy(BLANK_AUDIT if orphan else AUDIT),
    }
    evidence = {
        "candidate_uid": "candidate-evidence-1",
        "candidate_index": 1,
        "utterance_uid": utterance["utterance_uid"],
        "utterance_index": utterance["utterance_index"],
        "start_ms": utterance["coarse_start_ms"],
        "end_ms": utterance["coarse_end_ms"],
        "clip_start_ms": 0,
        "clip_end_ms": 1_000,
        "reason": (
            "orphan_youtube_caption_without_asr_or_vad_overlap"
            if orphan
            else "zero_unpadded_independent_vad_overlap"
        ),
        "asr_text": utterance["asr_text"],
        "youtube_text": utterance["youtube_text"],
        "context_before": utterance["context_before"],
        "context_after": utterance["context_after"],
        "risk_flags": copy.deepcopy(utterance["risk_flags"]),
        "asr_audit": copy.deepcopy(utterance["asr_audit"]),
        "audio_member": "asr_hallucination_audio/candidate-evidence-1.wav",
        "audio_sha256": hashlib.sha256(_wav()).hexdigest(),
        "audio_size_bytes": len(_wav()),
    }
    return utterance, evidence


def _hole(index: int = 2) -> tuple[dict, dict]:
    utterance = {
        "utterance_uid": "hole-1",
        "utterance_index": index,
        "coarse_start_ms": 1_000,
        "coarse_end_ms": 1_400,
        "asr_text": "",
        "youtube_text": "",
        "context_before": "Merhaba",
        "context_after": "Nasılsın",
        "risk_flags": ["unresolved_vad_speech", "manual_audio_review_required"],
        "asr_audit": copy.deepcopy(BLANK_AUDIT),
    }
    evidence = {
        "hole_uid": utterance["utterance_uid"],
        "hole_index": 1,
        "start_ms": utterance["coarse_start_ms"],
        "end_ms": utterance["coarse_end_ms"],
        "clip_start_ms": 800,
        "clip_end_ms": 1_800,
        "reason": "unresolved_speech",
        "context_before": utterance["context_before"],
        "context_after": utterance["context_after"],
        "risk_flags": copy.deepcopy(utterance["risk_flags"]),
        "audio_member": "speech_hole_audio/hole-1.wav",
        "audio_sha256": hashlib.sha256(_wav()).hexdigest(),
        "audio_size_bytes": len(_wav()),
    }
    return utterance, evidence


def _pending_output(utterances: list[dict]) -> list[dict]:
    records = []
    for utterance in utterances:
        text = utterance["asr_text"] or utterance["youtube_text"]
        records.append(
            {
                **copy.deepcopy(utterance),
                "tr_corrected": text,
                "non_dialogue": False,
                "review_required": True,
                "audio_reviewed": False,
                "review_disposition": "pending_audio_review",
                "note": "",
            }
        )
    return records


def _make_files(
    root: Path,
    *,
    orphan: bool = False,
    include_hole: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    candidate, candidate_evidence = _candidate(orphan=orphan)
    utterances = [candidate]
    holes: list[dict] = []
    if include_hole:
        hole, hole_evidence = _hole()
        utterances.append(hole)
        holes.append(hole_evidence)
    candidate_path = root / candidate_evidence["audio_member"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(_wav())
    for hole_evidence in holes:
        hole_path = root / hole_evidence["audio_member"]
        hole_path.parent.mkdir(parents=True, exist_ok=True)
        hole_path.write_bytes(_wav())
    input_zip = root / "input.zip"
    provisional_zip = root / "provisional.zip"
    final_zip = root / "final.zip"
    report = root / "audio_review_v2.json"
    recovery = root / "audio_review_v2.recovery.json"
    create_tr_correction_pack(
        utterances,
        holes,
        input_zip,
        episode=12,
        speech_hole_audio_root=root,
        asr_hallucination_records=[candidate_evidence],
        asr_hallucination_audio_root=root,
    )
    create_tr_correction_output(
        input_zip, _pending_output(utterances), provisional_zip
    )
    return input_zip, provisional_zip, final_zip, report, recovery


class AudioReviewV2Tests(unittest.TestCase):
    def test_runtime_decoder_keeps_zero_duration_word_without_aborting(self) -> None:
        class RuntimeModel:
            def transcribe(self, *_args: Any, **_kwargs: Any) -> tuple[Any, None]:
                segment = SimpleNamespace(
                    text="İyi misin?",
                    start=0.10,
                    end=0.50,
                    words=[
                        SimpleNamespace(
                            word="İyi", start=0.10, end=0.30, probability=0.95
                        ),
                        SimpleNamespace(
                            word="misin?", start=0.30, end=0.30, probability=0.95
                        ),
                    ],
                )
                return iter([segment]), None

        decoder = _FasterWhisperReviewDecoder.__new__(
            _FasterWhisperReviewDecoder
        )
        decoder.config = AudioReviewV2Config()
        decoder._model = RuntimeModel()

        decoded = decoder._decode_once(Path("unused.wav"), initial_prompt=None)

        self.assertEqual(
            [(word["start_ms"], word["end_ms"]) for word in decoded["words"]],
            [(100, 300), (300, 301)],
        )

    def test_secondary_asr_confirms_candidate_and_fills_speech_hole(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), include_hole=True)
            decoder = FakeDecoder([_decoded("Merhaba"), _decoded("Geldim", 250, 550)])
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["review_count"], 2)
            output = validate_tr_correction_output(paths[0], paths[2])
            self.assertEqual(
                [record["review_disposition"] for record in output.records],
                ["confirmed_dialogue", "confirmed_dialogue"],
            )
            self.assertEqual(output.records[1]["tr_corrected"], "Geldim")
            self.assertEqual(decoder.call_count, 2)

    def test_silent_structural_asr_candidate_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            self.assertEqual(report["status"], "PASS")
            output = validate_tr_correction_output(paths[0], paths[2])
            self.assertTrue(output.records[0]["non_dialogue"])
            self.assertEqual(
                output.records[0]["review_disposition"],
                "discarded_asr_hallucination",
            )

    def test_ambiguous_orphan_fails_closed_and_writes_remainder_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), orphan=True)
            with self.assertRaisesRegex(
                AudioReviewV2Error, "pending_count=1"
            ):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([_decoded(), _decoded()]),
                    progress=None,
                )
            self.assertFalse(paths[2].exists())
            report = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NEEDS_MANUAL_REVIEW")
            self.assertEqual(report["pending_utterance_uids"], ["candidate-1"])

    def test_prompted_second_pass_can_confirm_but_remains_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), orphan=True)
            decoder = FakeDecoder([_decoded(), _decoded("Merhaba")])
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(decoder.call_count, 2)
            outcome = report["outcomes"][0]
            self.assertEqual(
                outcome["source"], "secondary_asr_prompted_confirmation"
            )
            self.assertEqual(len(outcome["evidence_prompt_sha256"]), 64)
            self.assertEqual(outcome["blind_target_decode"]["source"], "none")

    def test_checkpoint_is_reused_when_manual_override_closes_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), orphan=True)
            with self.assertRaises(AudioReviewV2Error):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([_decoded(), _decoded()]),
                    progress=None,
                )
            resumed_decoder = FakeDecoder([])
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=resumed_decoder,
                manual_overrides={
                    "candidate-1": {
                        "disposition": "confirmed_dialogue",
                        "tr_corrected": "Merhaba",
                        "note": "Exact WAV dinlendi; konuşma doğrulandı.",
                    }
                },
                progress=None,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(resumed_decoder.call_count, 0)

    def test_report_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            report = json.loads(paths[3].read_text(encoding="utf-8"))
            report["outcomes"][0]["audio_sha256"] = "0" * 64
            paths[3].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(AudioReviewV2Error, "digest mismatch"):
                validate_audio_review_v2_report(paths[0], paths[1], paths[2], paths[3])

    def test_text_model_cannot_preclaim_audio_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _make_files(root)
            provisional = validate_tr_correction_output(paths[0], paths[1])
            records = [copy.deepcopy(dict(item)) for item in provisional.records]
            records[0].update(
                {
                    "review_required": False,
                    "audio_reviewed": True,
                    "review_disposition": "confirmed_dialogue",
                    "note": "Model claimed it listened.",
                }
            )
            create_tr_correction_output(paths[0], records, paths[1])
            with self.assertRaisesRegex(AudioReviewV2Error, "must leave.*pending"):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([]),
                    progress=None,
                )

    def test_config_rejects_unbounded_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            AudioReviewV2Config(device="internet")
        with self.assertRaises(ValueError):
            AudioReviewV2Config(target_tolerance_ms=501)


if __name__ == "__main__":
    unittest.main()
