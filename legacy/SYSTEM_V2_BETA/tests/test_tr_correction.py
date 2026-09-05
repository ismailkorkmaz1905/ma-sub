from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import warnings
import wave
import zipfile
from pathlib import Path

from src.tr_correction import (
    TRCorrectionError,
    compute_input_sha256,
    create_tr_correction_output,
    create_tr_correction_pack,
    read_tr_correction_pack,
    validate_tr_correction_output,
    validate_tr_correction_pack,
    validate_tr_correction_records,
    validate_input_utterances,
)


ASR_AUDIT = {
    "avg_logprob": -0.25,
    "no_speech_prob": 0.08,
    "compression_ratio": 1.1,
    "temperature": 0.0,
}
BLANK_ASR_AUDIT = {
    "avg_logprob": None,
    "no_speech_prob": None,
    "compression_ratio": None,
    "temperature": None,
}


def _utterances(count: int = 4) -> list[dict]:
    records: list[dict] = []
    for offset in range(count):
        index = offset + 1
        start_ms = 1_000 + offset * 2_000
        records.append(
            {
                "utterance_uid": f"U12-{index:06d}-evidence",
                "utterance_index": index,
                "coarse_start_ms": start_ms,
                "coarse_end_ms": start_ms + 1_500,
                "asr_text": f"Ham cümle {index}",
                "youtube_text": f"YouTube cümlesi {index}",
                "context_before": "" if index == 1 else f"Önce {index - 1}",
                "context_after": "" if index == count else f"Sonra {index + 1}",
                "risk_flags": [] if index % 2 else ["low_asr_confidence"],
                "asr_audit": copy.deepcopy(ASR_AUDIT),
            }
        )
    records.append(
        {
            "utterance_uid": "H12-000001",
            "utterance_index": len(records) + 1,
            "coarse_start_ms": 9_100,
            "coarse_end_ms": 9_900,
            "asr_text": "",
            "youtube_text": "",
            "context_before": "Önceki söz",
            "context_after": "Sonraki söz",
            "risk_flags": ["unresolved_vad_speech", "speech_without_asr"],
            "asr_audit": copy.deepcopy(BLANK_ASR_AUDIT),
        }
    )
    return records


def _wav_bytes(duration_ms: int = 800) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * round(16_000 * duration_ms / 1000))
    return target.getvalue()


HOLE_WAV = _wav_bytes()
HALLUCINATION_WAV = _wav_bytes(1_519)


def _holes() -> list[dict]:
    hole_uid = "H12-000001"
    return [
        {
            "hole_uid": hole_uid,
            "hole_index": 1,
            "start_ms": 9_100,
            "end_ms": 9_900,
            "clip_start_ms": 9_100,
            "clip_end_ms": 9_900,
            "reason": "VAD speech has no ASR words",
            "context_before": "Önceki söz",
            "context_after": "Sonraki söz",
            "risk_flags": ["unresolved_vad_speech", "speech_without_asr"],
            "audio_member": f"speech_hole_audio/{hole_uid}.wav",
            "audio_sha256": hashlib.sha256(HOLE_WAV).hexdigest(),
            "audio_size_bytes": len(HOLE_WAV),
        }
    ]


def _write_hole_audio(root: Path, holes: list[dict]) -> None:
    for hole in holes:
        path = root.joinpath(*Path(hole["audio_member"]).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(HOLE_WAV)


def _hallucination_fixture() -> tuple[list[dict], list[dict]]:
    inputs = copy.deepcopy(_utterances()[:-1])
    inputs[0]["coarse_end_ms"] = 1_019
    inputs[0]["risk_flags"].append("suspected_asr_hallucination")
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
            "clip_end_ms": 1_769,
            "reason": "ASR target has no independent-VAD overlap",
            "asr_text": inputs[0]["asr_text"],
            "youtube_text": inputs[0]["youtube_text"],
            "context_before": inputs[0]["context_before"],
            "context_after": inputs[0]["context_after"],
            "risk_flags": copy.deepcopy(inputs[0]["risk_flags"]),
            "asr_audit": copy.deepcopy(inputs[0]["asr_audit"]),
            "audio_member": f"asr_hallucination_audio/{candidate_uid}.wav",
            "audio_sha256": hashlib.sha256(HALLUCINATION_WAV).hexdigest(),
            "audio_size_bytes": len(HALLUCINATION_WAV),
        }
    ]
    return inputs, candidates


def _write_hallucination_audio(root: Path, candidates: list[dict]) -> None:
    for candidate in candidates:
        path = root.joinpath(*Path(candidate["audio_member"]).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(HALLUCINATION_WAV)


def _outputs(inputs: list[dict]) -> list[dict]:
    outputs: list[dict] = []
    for position, record in enumerate(inputs, start=1):
        needs_audio = "unresolved_vad_speech" in record["risk_flags"]
        outputs.append(
            {
                **copy.deepcopy(record),
                "tr_corrected": f"Düzeltilmiş cümle {position}",
                "non_dialogue": False,
                "review_required": False,
                "audio_reviewed": needs_audio,
                "review_disposition": (
                    "confirmed_dialogue" if needs_audio else "not_applicable"
                ),
                "note": (
                    "WAV dinlendi; konuşma doğrulandı." if needs_audio else ""
                ),
            }
        )
    return outputs


def _rewrite_zip(
    source: Path,
    destination: Path,
    *,
    replacements: dict[str, bytes] | None = None,
    additions: list[tuple[str, bytes]] | None = None,
) -> None:
    replacements = replacements or {}
    additions = additions or []
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as changed:
        for info in original.infolist():
            changed.writestr(info.filename, replacements.get(info.filename, original.read(info)))
        for name, payload in additions:
            changed.writestr(name, payload)


class TRCorrectionTests(unittest.TestCase):
    def test_unresolved_vad_utterance_can_be_reviewed_without_asr_text(self) -> None:
        utterance = copy.deepcopy(self.inputs[0])
        utterance["asr_text"] = ""
        utterance["youtube_text"] = ""
        utterance["risk_flags"] = ["unresolved_vad_speech"]
        validated = validate_input_utterances([utterance])
        self.assertEqual(validated[0]["asr_text"], "")

    def setUp(self) -> None:
        self.inputs = _utterances()
        self.holes = _holes()

    def _create_pack(
        self,
        path: Path,
        *,
        inputs: list[dict] | None = None,
        holes: list[dict] | None = None,
        hallucinations: list[dict] | None = None,
        batch_size: int = 250,
        episode: int = 12,
    ) -> dict:
        selected_holes = self.holes if holes is None else holes
        selected_hallucinations = [] if hallucinations is None else hallucinations
        _write_hole_audio(path.parent, selected_holes)
        _write_hallucination_audio(path.parent, selected_hallucinations)
        return create_tr_correction_pack(
            self.inputs if inputs is None else inputs,
            selected_holes,
            path,
            episode=episode,
            batch_size=batch_size,
            speech_hole_audio_root=path.parent,
            asr_hallucination_records=selected_hallucinations,
            asr_hallucination_audio_root=path.parent,
        )

    def test_input_pack_is_deterministic_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.zip"
            second = root / "second.zip"
            first_manifest = self._create_pack(first, batch_size=2)
            second_manifest = self._create_pack(second, batch_size=2)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                first_manifest["input_sha256"],
                compute_input_sha256(self.inputs, self.holes, episode=12),
            )
            data = read_tr_correction_pack(first)
            self.assertEqual(list(data.utterances), self.inputs)
            self.assertEqual(list(data.speech_holes), self.holes)
            self.assertIn("Turkish correction pass", data.instructions)
            self.assertEqual(
                first_manifest["speech_hole_audio"],
                [
                    {
                        "hole_uid": self.holes[0]["hole_uid"],
                        "member": self.holes[0]["audio_member"],
                        "sha256": self.holes[0]["audio_sha256"],
                        "size_bytes": self.holes[0]["audio_size_bytes"],
                    }
                ],
            )
            with zipfile.ZipFile(first, "r") as archive:
                self.assertEqual(
                    archive.read(self.holes[0]["audio_member"]), HOLE_WAV
                )

    def test_missing_or_tampered_speech_hole_audio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.zip"
            with self.assertRaisesRegex(TRCorrectionError, "missing"):
                create_tr_correction_pack(
                    self.inputs,
                    self.holes,
                    missing,
                    episode=12,
                    speech_hole_audio_root=root,
                )

            source = root / "source.zip"
            self._create_pack(source)
            changed_audio = bytearray(HOLE_WAV)
            changed_audio[-1] ^= 0x01
            tampered = root / "tampered.zip"
            _rewrite_zip(
                source,
                tampered,
                replacements={
                    self.holes[0]["audio_member"]: bytes(changed_audio)
                },
            )
            with self.assertRaisesRegex(TRCorrectionError, "audio SHA-256 mismatch"):
                validate_tr_correction_pack(tampered)

    def test_speech_hole_audio_member_path_is_exact_and_safe(self) -> None:
        holes = copy.deepcopy(self.holes)
        holes[0]["audio_member"] = "../outside.wav"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TRCorrectionError, "audio_member must be"):
                create_tr_correction_pack(
                    self.inputs,
                    holes,
                    Path(directory) / "bad.zip",
                    episode=12,
                    speech_hole_audio_root=directory,
                )

    def test_valid_output_round_trips_and_exposes_both_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_zip = root / "input.zip"
            output_zip = root / "output.zip"
            self._create_pack(input_zip, batch_size=2)
            output_manifest = create_tr_correction_output(
                input_zip, _outputs(self.inputs), output_zip, batch_size=3
            )
            result = validate_tr_correction_output(input_zip, output_zip)
            self.assertEqual(result.input_sha256, output_manifest["input_sha256"])
            self.assertEqual(result.output_sha256, output_manifest["output_sha256"])
            self.assertEqual(len(result.records), len(self.inputs))

    def test_pack_without_speech_holes_needs_no_audio_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no-holes.zip"
            manifest = create_tr_correction_pack(
                self.inputs[:-1], [], path, episode=12
            )
            self.assertEqual(manifest["speech_hole_count"], 0)
            self.assertEqual(manifest["speech_hole_audio"], [])
            self.assertEqual(read_tr_correction_pack(path).speech_holes, ())

    def test_blank_unresolved_utterance_requires_matching_hole_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unlinked.zip"
            with self.assertRaisesRegex(
                TRCorrectionError, "Blank unresolved utterances"
            ):
                create_tr_correction_pack(
                    self.inputs, [], path, episode=12
                )

            changed_holes = copy.deepcopy(self.holes)
            changed_holes[0]["start_ms"] += 1
            with self.assertRaisesRegex(TRCorrectionError, "coarse_start_ms"):
                create_tr_correction_pack(
                    self.inputs,
                    changed_holes,
                    path,
                    episode=12,
                    speech_hole_audio_root=directory,
                )

    def test_non_dialogue_requires_empty_text_and_note(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[-1]["non_dialogue"] = True
        outputs[-1]["tr_corrected"] = ""
        outputs[-1]["audio_reviewed"] = True
        outputs[-1]["review_disposition"] = "reviewed_non_dialogue"
        outputs[-1]["note"] = "WAV dinlendi; yalnız kapı sesi var."
        validated = validate_tr_correction_records(
            self.inputs, outputs, speech_holes=self.holes
        )
        self.assertTrue(validated[-1]["non_dialogue"])

        no_note = copy.deepcopy(outputs)
        no_note[-1]["note"] = ""
        with self.assertRaisesRegex(TRCorrectionError, "requires a concrete note"):
            validate_tr_correction_records(
                self.inputs, no_note, speech_holes=self.holes
            )

        with_text = copy.deepcopy(outputs)
        with_text[-1]["tr_corrected"] = "Müzik"
        with self.assertRaisesRegex(TRCorrectionError, "must have empty"):
            validate_tr_correction_records(
                self.inputs, with_text, speech_holes=self.holes
            )

    def test_ordinary_asr_record_cannot_be_marked_non_dialogue(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[0]["non_dialogue"] = True
        outputs[0]["tr_corrected"] = ""
        outputs[0]["note"] = "Yanlışlıkla ses sayıldı."
        with self.assertRaisesRegex(
            TRCorrectionError, "WAV-reviewed unresolved_vad_speech"
        ):
            validate_tr_correction_records(
                self.inputs, outputs, speech_holes=self.holes
            )

    def test_non_dialogue_requires_trusted_exact_hole_identity_and_bounds(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[-1]["non_dialogue"] = True
        outputs[-1]["tr_corrected"] = ""
        outputs[-1]["audio_reviewed"] = True
        outputs[-1]["review_disposition"] = "reviewed_non_dialogue"
        outputs[-1]["note"] = "WAV dinlendi; yalnız kapı sesi var."

        with self.assertRaisesRegex(
            TRCorrectionError, "WAV-reviewed unresolved_vad_speech"
        ):
            without_trusted_audio = copy.deepcopy(outputs)
            without_trusted_audio[-1]["audio_reviewed"] = False
            without_trusted_audio[-1]["review_disposition"] = "not_applicable"
            validate_tr_correction_records(self.inputs, without_trusted_audio)

        shifted = copy.deepcopy(self.holes)
        shifted[0]["start_ms"] += 1
        with self.assertRaisesRegex(TRCorrectionError, "coarse_start_ms"):
            validate_tr_correction_records(
                self.inputs, outputs, speech_holes=shifted
            )

        unknown = copy.deepcopy(self.holes)
        unknown[0]["hole_uid"] = "H12-999999"
        unknown[0]["audio_member"] = "speech_hole_audio/H12-999999.wav"
        with self.assertRaisesRegex(
            TRCorrectionError, "Blank unresolved utterances"
        ):
            validate_tr_correction_records(
                self.inputs, outputs, speech_holes=unknown
            )

    def test_contextual_wav_hallucination_candidate_round_trips_and_discards(self) -> None:
        inputs, candidates = _hallucination_fixture()
        self.assertEqual(
            inputs[0]["coarse_end_ms"] - inputs[0]["coarse_start_ms"], 19
        )
        self.assertEqual(
            candidates[0]["clip_end_ms"] - candidates[0]["clip_start_ms"],
            1_519,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_zip = root / "candidate-input.zip"
            output_zip = root / "candidate-output.zip"
            manifest = self._create_pack(
                input_zip,
                inputs=inputs,
                holes=[],
                hallucinations=candidates,
            )
            self.assertEqual(manifest["asr_hallucination_count"], 1)
            pack = read_tr_correction_pack(input_zip)
            self.assertEqual(
                list(pack.asr_hallucination_records), candidates
            )
            with zipfile.ZipFile(input_zip, "r") as archive:
                self.assertEqual(
                    archive.read(candidates[0]["audio_member"]),
                    HALLUCINATION_WAV,
                )

            outputs = _outputs(inputs)
            outputs[0].update(
                {
                    "tr_corrected": "",
                    "non_dialogue": True,
                    "review_required": False,
                    "audio_reviewed": True,
                    "review_disposition": "discarded_asr_hallucination",
                    "note": "Bağlı WAV dinlendi; hedefte konuşma yok.",
                }
            )
            create_tr_correction_output(input_zip, outputs, output_zip)
            validated = validate_tr_correction_output(input_zip, output_zip)
            self.assertEqual(
                validated.records[0]["review_disposition"],
                "discarded_asr_hallucination",
            )

    def test_hallucination_candidate_requires_explicit_audio_disposition(self) -> None:
        inputs, candidates = _hallucination_fixture()
        outputs = _outputs(inputs)

        outputs[0].update(
            {
                "review_required": True,
                "audio_reviewed": False,
                "review_disposition": "pending_audio_review",
            }
        )
        validate_tr_correction_records(
            inputs,
            outputs,
            asr_hallucination_records=candidates,
        )

        outputs[0].update(
            {
                "review_required": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "note": "Bağlı WAV dinlendi; Türkçe konuşma doğrulandı.",
            }
        )
        validate_tr_correction_records(
            inputs,
            outputs,
            asr_hallucination_records=candidates,
        )

        missing_audit = copy.deepcopy(outputs)
        missing_audit[0]["audio_reviewed"] = False
        missing_audit[0]["review_disposition"] = "not_applicable"
        with self.assertRaisesRegex(TRCorrectionError, "audio_reviewed=true"):
            validate_tr_correction_records(
                inputs,
                missing_audit,
                asr_hallucination_records=candidates,
            )

    def test_orphan_youtube_caption_reuses_exact_audio_review_candidate(self) -> None:
        inputs, candidates = _hallucination_fixture()
        inputs[0]["asr_text"] = ""
        inputs[0]["risk_flags"] = ["orphan_youtube_caption"]
        inputs[0]["asr_audit"] = copy.deepcopy(BLANK_ASR_AUDIT)
        candidates[0]["asr_text"] = ""
        candidates[0]["risk_flags"] = ["orphan_youtube_caption"]
        candidates[0]["asr_audit"] = copy.deepcopy(BLANK_ASR_AUDIT)

        outputs = _outputs(inputs)
        outputs[0].update(
            {
                "tr_corrected": "",
                "non_dialogue": True,
                "review_required": False,
                "audio_reviewed": True,
                "review_disposition": "discarded_asr_hallucination",
                "note": "WAV dinlendi; YouTube metnine karşılık konuşma yok.",
            }
        )
        validated = validate_tr_correction_records(
            inputs,
            outputs,
            asr_hallucination_records=candidates,
        )
        self.assertTrue(validated[0]["non_dialogue"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_zip = root / "orphan-input.zip"
            output_zip = root / "orphan-output.zip"
            self._create_pack(
                input_zip,
                inputs=inputs,
                holes=[],
                hallucinations=candidates,
            )
            create_tr_correction_output(input_zip, outputs, output_zip)
            round_trip = validate_tr_correction_output(input_zip, output_zip)
            self.assertEqual(
                round_trip.records[0]["risk_flags"],
                ["orphan_youtube_caption"],
            )

        invalid = copy.deepcopy(candidates)
        invalid[0]["risk_flags"] = ["suspected_asr_hallucination"]
        with self.assertRaisesRegex(TRCorrectionError, "non-empty ASR evidence"):
            validate_tr_correction_records(
                inputs,
                outputs,
                asr_hallucination_records=invalid,
            )

    def test_lexical_deletion_needs_pending_rerun_then_exact_audio_review(self) -> None:
        inputs = copy.deepcopy(self.inputs[:-1])
        inputs[0]["coarse_end_ms"] = inputs[0]["coarse_start_ms"] + 150
        inputs[0]["asr_text"] = "evet geliyorum"
        outputs = _outputs(inputs)
        outputs[0]["tr_corrected"] = "geliyorum"

        with self.assertRaisesRegex(
            TRCorrectionError,
            r"U12-000001-evidence.*EXTRA_AUDIO_REVIEW_UIDS",
        ):
            validate_tr_correction_records(inputs, outputs)

        outputs[0]["review_required"] = True
        outputs[0]["review_disposition"] = "pending_audio_review"
        validate_tr_correction_records(inputs, outputs)

        inputs[0]["risk_flags"].append("suspected_asr_hallucination")
        candidate_uid = "AH12-000001-delete"
        candidates = [
            {
                "candidate_uid": candidate_uid,
                "candidate_index": 1,
                "utterance_uid": inputs[0]["utterance_uid"],
                "utterance_index": inputs[0]["utterance_index"],
                "start_ms": inputs[0]["coarse_start_ms"],
                "end_ms": inputs[0]["coarse_end_ms"],
                "clip_start_ms": 250,
                "clip_end_ms": 1_900,
                "reason": "Explicit lexical-deletion audio review request",
                "asr_text": inputs[0]["asr_text"],
                "youtube_text": inputs[0]["youtube_text"],
                "context_before": inputs[0]["context_before"],
                "context_after": inputs[0]["context_after"],
                "risk_flags": copy.deepcopy(inputs[0]["risk_flags"]),
                "asr_audit": copy.deepcopy(inputs[0]["asr_audit"]),
                "audio_member": f"asr_hallucination_audio/{candidate_uid}.wav",
                "audio_sha256": "a" * 64,
                "audio_size_bytes": 1_024,
            }
        ]
        outputs = _outputs(inputs)
        outputs[0].update(
            {
                "tr_corrected": "geliyorum",
                "review_required": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "note": "WAV dinlendi; ilk ASR tokeni hedefte söylenmiyor.",
            }
        )
        validate_tr_correction_records(
            inputs,
            outputs,
            asr_hallucination_records=candidates,
        )

        punctuation_only = _outputs(copy.deepcopy(self.inputs[:-1]))
        punctuation_only[0]["tr_corrected"] = self.inputs[0]["asr_text"].upper() + "!"
        validate_tr_correction_records(self.inputs[:-1], punctuation_only)

    def test_review_only_pack_refresh_rebinds_existing_text_output(self) -> None:
        inputs = copy.deepcopy(self.inputs[:-1])
        inputs[0]["coarse_end_ms"] = inputs[0]["coarse_start_ms"] + 19
        outputs = _outputs(inputs)
        outputs[0].update(
            {
                "tr_corrected": "cümle 1",
                "review_required": True,
                "audio_reviewed": False,
                "review_disposition": "pending_audio_review",
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack_path = root / "Muhtemel Ask 12.Bolum_TR_CORRECTION_PACK.zip"
            output_path = root / "Muhtemel Ask 12.Bolum_TR_TEXT_CORRECTED.zip"
            old_manifest = self._create_pack(
                pack_path, inputs=inputs, holes=[], hallucinations=[]
            )
            create_tr_correction_output(pack_path, outputs, output_path)

            refreshed_inputs = copy.deepcopy(inputs)
            refreshed_inputs[0]["risk_flags"].extend(
                ["suspected_asr_hallucination", "manual_audio_review_required"]
            )
            candidate_uid = "AH12-review-only-refresh"
            candidate = {
                "candidate_uid": candidate_uid,
                "candidate_index": 1,
                "utterance_uid": refreshed_inputs[0]["utterance_uid"],
                "utterance_index": refreshed_inputs[0]["utterance_index"],
                "start_ms": refreshed_inputs[0]["coarse_start_ms"],
                "end_ms": refreshed_inputs[0]["coarse_end_ms"],
                "clip_start_ms": 250,
                "clip_end_ms": 1_769,
                "reason": "explicit_uid_review_request",
                "asr_text": refreshed_inputs[0]["asr_text"],
                "youtube_text": refreshed_inputs[0]["youtube_text"],
                "context_before": refreshed_inputs[0]["context_before"],
                "context_after": refreshed_inputs[0]["context_after"],
                "risk_flags": copy.deepcopy(refreshed_inputs[0]["risk_flags"]),
                "asr_audit": copy.deepcopy(refreshed_inputs[0]["asr_audit"]),
                "audio_member": f"asr_hallucination_audio/{candidate_uid}.wav",
                "audio_sha256": hashlib.sha256(HALLUCINATION_WAV).hexdigest(),
                "audio_size_bytes": len(HALLUCINATION_WAV),
            }
            _write_hallucination_audio(root, [candidate])
            new_manifest = create_tr_correction_pack(
                refreshed_inputs,
                [],
                pack_path,
                episode=12,
                speech_hole_audio_root=root,
                asr_hallucination_records=[candidate],
                asr_hallucination_audio_root=root,
                rebind_text_output_path=output_path,
            )

            self.assertNotEqual(
                old_manifest["input_sha256"], new_manifest["input_sha256"]
            )
            rebound = validate_tr_correction_output(pack_path, output_path)
            self.assertEqual(rebound.input_sha256, new_manifest["input_sha256"])
            self.assertEqual(rebound.records[0]["tr_corrected"], "cümle 1")
            self.assertEqual(
                rebound.records[0]["risk_flags"],
                ["suspected_asr_hallucination", "manual_audio_review_required"],
            )
            self.assertTrue(rebound.records[0]["review_required"])
            self.assertFalse(rebound.records[0]["audio_reviewed"])
            self.assertEqual(
                rebound.records[0]["review_disposition"],
                "pending_audio_review",
            )

    def test_hallucination_candidate_binding_and_audio_tampering_are_rejected(self) -> None:
        inputs, candidates = _hallucination_fixture()
        outputs = _outputs(inputs)
        outputs[0].update(
            {
                "review_required": False,
                "audio_reviewed": True,
                "review_disposition": "confirmed_dialogue",
                "note": "WAV dinlendi; diyalog doğrulandı.",
            }
        )
        shifted = copy.deepcopy(candidates)
        shifted[0]["start_ms"] += 1
        with self.assertRaisesRegex(TRCorrectionError, "coarse_start_ms"):
            validate_tr_correction_records(
                inputs,
                outputs,
                asr_hallucination_records=shifted,
            )

        unknown = copy.deepcopy(candidates)
        unknown[0]["utterance_uid"] = "U12-999999-unknown"
        with self.assertRaisesRegex(TRCorrectionError, "missing_candidates"):
            validate_tr_correction_records(
                inputs,
                outputs,
                asr_hallucination_records=unknown,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            self._create_pack(
                source,
                inputs=inputs,
                holes=[],
                hallucinations=candidates,
            )
            tampered = root / "tampered.zip"
            changed = bytearray(HALLUCINATION_WAV)
            changed[-1] ^= 0x01
            _rewrite_zip(
                source,
                tampered,
                replacements={candidates[0]["audio_member"]: bytes(changed)},
            )
            with self.assertRaisesRegex(
                TRCorrectionError, "ASR hallucination audio SHA-256 mismatch"
            ):
                validate_tr_correction_pack(tampered)

    def test_dialogue_requires_nonempty_corrected_turkish(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[2]["tr_corrected"] = "  "
        with self.assertRaisesRegex(TRCorrectionError, "requires non-empty"):
            validate_tr_correction_records(self.inputs, outputs)

    def test_changed_time_or_evidence_is_rejected(self) -> None:
        for field, replacement in (
            ("coarse_start_ms", 999),
            ("asr_text", "Başka ham metin"),
            ("youtube_text", "Başka kanıt"),
            ("context_before", "Başka bağlam"),
            ("risk_flags", ["başka"]),
        ):
            with self.subTest(field=field):
                outputs = _outputs(self.inputs)
                outputs[1][field] = replacement
                with self.assertRaisesRegex(TRCorrectionError, "changed immutable"):
                    validate_tr_correction_records(self.inputs, outputs)

    def test_extra_output_field_is_rejected(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[0]["aligned_start_ms"] = 42
        with self.assertRaisesRegex(TRCorrectionError, "field mismatch"):
            validate_tr_correction_records(self.inputs, outputs)

    def test_duplicate_output_uid_is_rejected_before_mapping(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[2] = copy.deepcopy(outputs[1])
        with self.assertRaisesRegex(TRCorrectionError, "duplicates=.*U12-000002"):
            validate_tr_correction_records(self.inputs, outputs)

    def test_missing_and_extra_output_uids_are_rejected(self) -> None:
        outputs = _outputs(self.inputs)
        outputs.pop(1)
        with self.assertRaisesRegex(TRCorrectionError, "missing=.*U12-000002"):
            validate_tr_correction_records(self.inputs, outputs)

        outputs = _outputs(self.inputs)
        outputs[-1]["utterance_uid"] = "U12-999999-extra"
        with self.assertRaisesRegex(TRCorrectionError, "extra=.*U12-999999-extra"):
            validate_tr_correction_records(self.inputs, outputs)

    def test_reordered_output_records_are_rejected(self) -> None:
        outputs = _outputs(self.inputs)
        outputs[1], outputs[2] = outputs[2], outputs[1]
        with self.assertRaisesRegex(TRCorrectionError, "order mismatch"):
            validate_tr_correction_records(self.inputs, outputs)

    def test_duplicate_input_uid_and_bad_input_order_are_rejected(self) -> None:
        duplicate = copy.deepcopy(self.inputs)
        duplicate[1]["utterance_uid"] = duplicate[0]["utterance_uid"]
        with self.assertRaisesRegex(TRCorrectionError, "Duplicate utterance_uid"):
            compute_input_sha256(duplicate, self.holes, episode=12)

        reordered = copy.deepcopy(self.inputs)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(TRCorrectionError, "order/index mismatch"):
            compute_input_sha256(reordered, self.holes, episode=12)

    def test_failed_creation_does_not_overwrite_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pack.zip"
            destination.write_bytes(b"keep-me")
            invalid = copy.deepcopy(self.inputs)
            invalid[0].pop("asr_text")
            with self.assertRaises(TRCorrectionError):
                self._create_pack(destination, inputs=invalid)
            self.assertEqual(destination.read_bytes(), b"keep-me")

    def test_pack_rejects_path_traversal_and_extra_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            self._create_pack(source)
            malicious = root / "malicious.zip"
            _rewrite_zip(source, malicious, additions=[("../escape", b"bad")])
            with self.assertRaisesRegex(TRCorrectionError, "Unsafe ZIP member"):
                validate_tr_correction_pack(malicious)

            extra = root / "extra.zip"
            _rewrite_zip(source, extra, additions=[("surprise.txt", b"bad")])
            with self.assertRaisesRegex(TRCorrectionError, "member mismatch"):
                validate_tr_correction_pack(extra)

    def test_pack_rejects_duplicate_zip_member_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            self._create_pack(source)
            duplicate = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _rewrite_zip(
                    source,
                    duplicate,
                    additions=[("manifest.json", b"{}")],
                )
            with self.assertRaisesRegex(TRCorrectionError, "duplicate member"):
                validate_tr_correction_pack(duplicate)

    def test_pack_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            self._create_pack(source)
            duplicate_key = root / "duplicate-key.zip"
            manifest_payload = b'{"format":"a","format":"b"}\n'
            _rewrite_zip(
                source,
                duplicate_key,
                replacements={"manifest.json": manifest_payload},
            )
            with self.assertRaisesRegex(TRCorrectionError, "Duplicate JSON object key"):
                validate_tr_correction_pack(duplicate_key)

    def test_pack_and_output_hash_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_zip = root / "input.zip"
            output_zip = root / "output.zip"
            self._create_pack(input_zip)
            create_tr_correction_output(
                input_zip, _outputs(self.inputs), output_zip
            )

            bad_input = root / "bad-input.zip"
            with zipfile.ZipFile(input_zip, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
            manifest["input_sha256"] = "0" * 64
            _rewrite_zip(
                input_zip,
                bad_input,
                replacements={"manifest.json": json.dumps(manifest).encode()},
            )
            with self.assertRaisesRegex(TRCorrectionError, "input_sha256 mismatch"):
                validate_tr_correction_pack(bad_input)

            bad_output = root / "bad-output.zip"
            with zipfile.ZipFile(output_zip, "r") as archive:
                output_manifest = json.loads(archive.read("manifest.json"))
            output_manifest["output_sha256"] = "f" * 64
            _rewrite_zip(
                output_zip,
                bad_output,
                replacements={
                    "manifest.json": json.dumps(output_manifest).encode()
                },
            )
            with self.assertRaisesRegex(TRCorrectionError, "output_sha256 mismatch"):
                validate_tr_correction_output(input_zip, bad_output)

    def test_input_sha_binds_episode_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            changed = root / "changed-episode.zip"
            self._create_pack(source)
            with zipfile.ZipFile(source, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
            manifest["episode"] = 13
            _rewrite_zip(
                source,
                changed,
                replacements={"manifest.json": json.dumps(manifest).encode()},
            )
            with self.assertRaisesRegex(TRCorrectionError, "input_sha256 mismatch"):
                validate_tr_correction_pack(changed)

    def test_output_is_bound_to_exact_input_pack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "first-input.zip"
            second_input = root / "second-input.zip"
            output = root / "output.zip"
            self._create_pack(first_input)
            create_tr_correction_output(
                first_input, _outputs(self.inputs), output
            )
            changed_inputs = copy.deepcopy(self.inputs)
            changed_inputs[0]["asr_text"] += " değişti"
            self._create_pack(second_input, inputs=changed_inputs)
            with self.assertRaisesRegex(TRCorrectionError, "input_sha256 mismatch"):
                validate_tr_correction_output(second_input, output)


if __name__ == "__main__":
    unittest.main()
