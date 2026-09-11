from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from mas.engine.audio_review import (
    _EP13_C0E3_AUDIO_REVIEW_CODE_SHA256,
    _FasterWhisperReviewDecoder,
    _report_sha,
    _write_exact_target_crop,
    AudioReviewV2Config,
    AudioReviewV2Error,
    resolve_tr_audio_reviews_v2,
    validate_audio_review_v2_report,
)
from mas.engine.tr_correction import (
    create_tr_correction_output,
    create_tr_correction_pack,
    read_tr_correction_pack,
    validate_tr_correction_output,
)
from mas.hashing import sha256_json


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


def _decoded_words(*words: tuple[str, int, int]) -> dict:
    return {
        "segments": [],
        "words": [
            {
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "probability": 0.95,
            }
            for text, start_ms, end_ms in words
        ],
    }


def _candidate(
    *,
    orphan: bool = False,
    reason: str | None = None,
    context_after: str = "Geldim",
    asr_text: str = "Merhaba",
) -> tuple[dict, dict]:
    utterance = {
        "utterance_uid": "candidate-1",
        "utterance_index": 1,
        "coarse_start_ms": 100,
        "coarse_end_ms": 500,
        "asr_text": "" if orphan else asr_text,
        "youtube_text": "Merhaba" if orphan else "",
        "context_before": "",
        "context_after": context_after,
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
        "reason": reason
        or (
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
    episode: int = 12,
    orphan: bool = False,
    include_hole: bool = False,
    candidate_reason: str | None = None,
    candidate_context_after: str = "Geldim",
    candidate_asr_text: str = "Merhaba",
) -> tuple[Path, Path, Path, Path, Path]:
    candidate, candidate_evidence = _candidate(
        orphan=orphan,
        reason=candidate_reason,
        context_after=candidate_context_after,
        asr_text=candidate_asr_text,
    )
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
        episode=episode,
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
                    decoder=FakeDecoder([_decoded(), _decoded(), _decoded()]),
                    progress=None,
                )
            self.assertFalse(paths[2].exists())
            report = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "NEEDS_MANUAL_REVIEW")
            self.assertEqual(report["pending_utterance_uids"], ["candidate-1"])

    def test_prompted_second_pass_can_confirm_but_remains_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory), candidate_reason="weak_confidence_candidate"
            )
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

    def test_exact_target_crop_can_confirm_after_context_decode_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory), candidate_reason="weak_confidence_candidate"
            )
            decoder = FakeDecoder([_decoded(), _decoded(), _decoded("Merhaba")])
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(decoder.call_count, 3)
            outcome = report["outcomes"][0]
            self.assertEqual(outcome["source"], "secondary_asr_exact_target_crop")
            self.assertEqual(outcome["exact_target_crop"]["duration_ms"], 400)
            self.assertEqual(
                outcome["exact_target_crop_decode"]["transcript"], "Merhaba"
            )

    def test_contextual_policy_retains_non_orphan_asr_boundary_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory), candidate_reason="weak_confidence_candidate"
            )
            decoder = FakeDecoder([_decoded(), _decoded(), _decoded()])

            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            record = output.records[0]
            outcome = report["outcomes"][0]
            self.assertEqual(record["review_disposition"], "confirmed_dialogue")
            self.assertEqual(record["tr_corrected"], "Merhaba")
            self.assertEqual(outcome["source"], "contextual_boundary_policy")
            self.assertTrue(outcome["forced_alignment_required"])
            self.assertEqual(
                [item["stage"] for item in outcome["contextual_boundary_audit"]["bounded_decodes"]],
                ["blind_padded", "prompted_padded", "exact_target_crop"],
            )
            self.assertNotIn("manual_audio_review_v2", record["note"])
            resumed_decoder = FakeDecoder([])
            resumed = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=resumed_decoder,
                progress=None,
            )
            self.assertEqual(resumed_decoder.call_count, 0)
            self.assertEqual(
                resumed["outcomes"][0]["source"], "contextual_boundary_policy"
            )

    def test_contextual_policy_closes_blank_hole_without_inventing_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory),
                include_hole=True,
                candidate_reason="weak_confidence_candidate",
            )
            decoder = FakeDecoder(
                [
                    _decoded("Merhaba"),
                    _decoded(),
                    _decoded(),
                    _decoded(),
                ]
            )

            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            hole = output.records[1]
            outcome = report["outcomes"][1]
            self.assertEqual(hole["review_disposition"], "reviewed_non_dialogue")
            self.assertEqual(hole["tr_corrected"], "")
            self.assertTrue(hole["non_dialogue"])
            self.assertEqual(outcome["source"], "contextual_boundary_policy")
            self.assertTrue(outcome["forced_alignment_required"])

    def test_prompted_hole_word_outside_exact_target_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), include_hole=True)
            decoder = FakeDecoder(
                [
                    _decoded("Merhaba"),
                    _decoded(),
                    _decoded("Geldim", 620, 700),
                    _decoded(),
                ]
            )

            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            hole = output.records[1]
            outcome = report["outcomes"][1]
            self.assertEqual(hole["review_disposition"], "reviewed_non_dialogue")
            self.assertEqual(hole["tr_corrected"], "")
            prompted = outcome["contextual_boundary_audit"]["bounded_decodes"][1]
            self.assertEqual(prompted["stage"], "prompted_padded")
            self.assertEqual(prompted["exact_word_overlap_ms"], 0)
            self.assertFalse(prompted["usable_target_text"])

    def test_overlapping_rescue_word_is_deduplicated_and_restores_parent_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, candidate_evidence = _candidate(
                asr_text="Sen sabah eve geldim"
            )
            candidate.update({"coarse_start_ms": 100, "coarse_end_ms": 900})
            candidate_evidence.update({"start_ms": 100, "end_ms": 900})
            hole, hole_evidence = _hole()
            hole.update({"coarse_start_ms": 300, "coarse_end_ms": 500})
            hole_evidence.update(
                {
                    "start_ms": 300,
                    "end_ms": 500,
                    "clip_start_ms": 0,
                    "clip_end_ms": 1_000,
                }
            )
            for evidence in (candidate_evidence, hole_evidence):
                path = root / evidence["audio_member"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_wav())
            paths = (
                root / "input.zip",
                root / "provisional.zip",
                root / "final.zip",
                root / "audio_review_v2.json",
                root / "audio_review_v2.recovery.json",
            )
            create_tr_correction_pack(
                [candidate, hole],
                [hole_evidence],
                paths[0],
                episode=12,
                speech_hole_audio_root=root,
                asr_hallucination_records=[candidate_evidence],
                asr_hallucination_audio_root=root,
            )
            create_tr_correction_output(
                paths[0], _pending_output([candidate, hole]), paths[1]
            )
            decoder = FakeDecoder(
                [
                    _decoded_words(
                        ("Sen", 100, 250),
                        ("beni", 300, 500),
                        ("sabah", 510, 600),
                        ("eve", 610, 680),
                        ("geldim", 690, 800),
                    ),
                    _decoded_words(("bence", 300, 500)),
                ]
            )

            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            self.assertEqual(
                output.records[0]["tr_corrected"], "Sen beni sabah eve geldim"
            )
            self.assertTrue(output.records[1]["non_dialogue"])
            self.assertEqual(
                output.records[1]["review_disposition"], "reviewed_non_dialogue"
            )
            self.assertEqual(report["duplicate_resolution_count"], 1)
            self.assertEqual(
                report["duplicate_resolutions"][0]["duplicate_of_utterance_uid"],
                "candidate-1",
            )

    def test_contextual_policy_missing_context_remains_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory),
                candidate_reason="weak_confidence_candidate",
                candidate_context_after="",
            )
            with self.assertRaisesRegex(AudioReviewV2Error, "pending_count=1"):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([_decoded(), _decoded(), _decoded()]),
                    progress=None,
                )

            report = json.loads(paths[3].read_text(encoding="utf-8"))
            self.assertEqual(report["pending_utterance_uids"], ["candidate-1"])

    def test_contextual_policy_keeps_exact_crop_speech_hole_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), include_hole=True)
            decoder = FakeDecoder(
                [
                    _decoded("Merhaba"),
                    _decoded(),
                    _decoded(),
                    _decoded("Duydum"),
                ]
            )
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )
            output = validate_tr_correction_output(paths[0], paths[2])
            self.assertEqual(output.records[1]["review_disposition"], "confirmed_dialogue")
            self.assertEqual(output.records[1]["tr_corrected"], "Duydum")
            self.assertEqual(
                report["outcomes"][1]["source"], "contextual_boundary_policy"
            )

    def test_contextual_policy_rejects_known_short_clip_hallucination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), include_hole=True)
            decoder = FakeDecoder(
                [
                    _decoded("Merhaba"),
                    _decoded(),
                    _decoded(),
                    _decoded("Altyazı M.K."),
                ]
            )

            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            hole = output.records[1]
            outcome = report["outcomes"][1]
            self.assertEqual(hole["review_disposition"], "reviewed_non_dialogue")
            self.assertEqual(hole["tr_corrected"], "")
            self.assertTrue(hole["non_dialogue"])
            exact_audit = outcome["contextual_boundary_audit"]["bounded_decodes"][-1]
            self.assertTrue(exact_audit["known_short_clip_hallucination"])
            self.assertFalse(exact_audit["usable_target_text"])

    def test_repetitive_source_loop_is_discarded_without_merging_context(self) -> None:
        repeated = " ".join(["Nefes al"] * 24)
        decoded = _decoded(
            "Nefes al. Tolga nereye gidiyorsun? Bir dur. Noter kağıdı geldi."
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory),
                candidate_reason="weak_confidence_candidate",
                candidate_asr_text=repeated,
            )
            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([decoded, decoded, decoded]),
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            record = output.records[0]
            outcome = report["outcomes"][0]
            self.assertTrue(record["non_dialogue"])
            self.assertEqual(record["tr_corrected"], "")
            self.assertEqual(
                record["review_disposition"], "discarded_asr_hallucination"
            )
            self.assertEqual(outcome["source"], "contextual_boundary_policy")
            validate_audio_review_v2_report(*paths[:4])

    def test_known_source_hallucination_is_discarded_without_acoustic_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(
                Path(directory),
                candidate_reason="known_subtitle_hallucination_signature",
                candidate_asr_text="Altyazı M.K.",
            )
            decoder = FakeDecoder(
                [
                    _decoded(),
                    _decoded(),
                    _decoded("Altyazı M .K."),
                ]
            )

            report = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=decoder,
                progress=None,
            )

            output = validate_tr_correction_output(paths[0], paths[2])
            record = output.records[0]
            self.assertEqual(
                record["review_disposition"], "discarded_asr_hallucination"
            )
            self.assertEqual(record["tr_corrected"], "")
            self.assertTrue(record["non_dialogue"])
            self.assertEqual(
                report["outcomes"][0]["source"], "contextual_boundary_policy"
            )

    def test_exact_target_crop_uses_only_declared_sample_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            destination = root / "target.wav"
            samples = list(range(1000))
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(1000)
                handle.writeframes(b"".join(value.to_bytes(2, "little") for value in samples))
            evidence = {
                "clip_start_ms": 800,
                "start_ms": 1000,
                "end_ms": 1400,
            }
            crop = _write_exact_target_crop(source, destination, evidence)
            with wave.open(str(destination), "rb") as handle:
                frames = handle.readframes(handle.getnframes())
            values = [
                int.from_bytes(frames[offset : offset + 2], "little")
                for offset in range(0, len(frames), 2)
            ]
            self.assertEqual(values, samples[200:600])
            self.assertEqual(crop["start_frame"], 200)
            self.assertEqual(crop["end_frame"], 600)

    def test_checkpoint_is_reused_when_manual_override_closes_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), orphan=True)
            with self.assertRaises(AudioReviewV2Error):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([_decoded(), _decoded(), _decoded()]),
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

    def test_completed_review_hydrates_without_recovery_or_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            completed = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            paths[4].unlink()
            resumed_decoder = FakeDecoder([])

            hydrated = resolve_tr_audio_reviews_v2(
                *paths,
                decoder=resumed_decoder,
                progress=None,
            )

            self.assertEqual(hydrated, completed)
            self.assertEqual(resumed_decoder.call_count, 0)

    def test_completed_ep13_review_hydrates_from_c0e3_producer_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory), episode=13)
            settings = AudioReviewV2Config(
                model_name="large-v3", device="cuda", allow_cpu_fallback=False
            )
            resolve_tr_audio_reviews_v2(
                *paths,
                config=settings,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            pack = read_tr_correction_pack(paths[0])
            provisional = validate_tr_correction_output(paths[0], paths[1])
            report = json.loads(paths[3].read_text(encoding="utf-8"))
            report["review_input_sha256"] = sha256_json(
                {
                    "correction_input_sha256": pack.manifest["input_sha256"],
                    "provisional_output_sha256": provisional.output_sha256,
                    "config": asdict(settings),
                    "code_sha256": _EP13_C0E3_AUDIO_REVIEW_CODE_SHA256,
                }
            )
            report["audio_review_sha256"] = _report_sha(report)
            paths[3].write_text(json.dumps(report), encoding="utf-8")
            report_before = paths[3].read_bytes()
            final_before = paths[2].read_bytes()
            resumed_decoder = FakeDecoder([])

            hydrated = resolve_tr_audio_reviews_v2(
                *paths,
                config=settings,
                decoder=resumed_decoder,
                progress=None,
            )

            self.assertEqual(hydrated, report)
            self.assertEqual(resumed_decoder.call_count, 0)
            self.assertEqual(paths[2].read_bytes(), final_before)
            self.assertEqual(paths[3].read_bytes(), report_before)

    def test_c0e3_producer_identity_is_not_accepted_for_other_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            pack = read_tr_correction_pack(paths[0])
            provisional = validate_tr_correction_output(paths[0], paths[1])
            report = json.loads(paths[3].read_text(encoding="utf-8"))
            identity = {
                "correction_input_sha256": pack.manifest["input_sha256"],
                "provisional_output_sha256": provisional.output_sha256,
                "config": asdict(AudioReviewV2Config()),
                "code_sha256": _EP13_C0E3_AUDIO_REVIEW_CODE_SHA256,
            }
            report["review_input_sha256"] = sha256_json(identity)
            report["audio_review_sha256"] = _report_sha(report)
            paths[3].write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(
                AudioReviewV2Error, "different inputs, config, or code"
            ):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([]),
                    progress=None,
                )

    def test_completed_review_rejects_changed_hydration_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            final_before = paths[2].read_bytes()
            report_before = paths[3].read_bytes()

            with self.assertRaisesRegex(
                AudioReviewV2Error, "different inputs, config, or code"
            ):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    config=AudioReviewV2Config(beam_size=4),
                    decoder=FakeDecoder([]),
                    progress=None,
                )

            self.assertEqual(paths[2].read_bytes(), final_before)
            self.assertEqual(paths[3].read_bytes(), report_before)

    def test_completed_review_rejects_missing_bound_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            final_before = paths[2].read_bytes()
            paths[3].unlink()

            with self.assertRaisesRegex(AudioReviewV2Error, "no safe bound report"):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    decoder=FakeDecoder([]),
                    progress=None,
                )

            self.assertEqual(paths[2].read_bytes(), final_before)

    def test_completed_review_rejects_changed_manual_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _make_files(Path(directory))
            resolve_tr_audio_reviews_v2(
                *paths,
                decoder=FakeDecoder([_decoded()]),
                progress=None,
            )
            final_before = paths[2].read_bytes()
            report_before = paths[3].read_bytes()

            with self.assertRaisesRegex(AudioReviewV2Error, "different manual overrides"):
                resolve_tr_audio_reviews_v2(
                    *paths,
                    manual_overrides={
                        "candidate-1": {
                            "disposition": "confirmed_dialogue",
                            "tr_corrected": "Merhaba",
                            "note": "Exact WAV yeniden doğrulandı.",
                        }
                    },
                    decoder=FakeDecoder([]),
                    progress=None,
                )

            self.assertEqual(paths[2].read_bytes(), final_before)
            self.assertEqual(paths[3].read_bytes(), report_before)

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
