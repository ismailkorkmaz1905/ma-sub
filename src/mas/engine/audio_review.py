"""Resumable, evidence-bound acoustic review of correction WAVs.

The language model that corrects Turkish is intentionally text-only.  This
module owns the separate acoustic pass: it decodes only the short WAV members
already bound into the correction pack, records every decision, and refuses to
publish a final correction ZIP while any result is ambiguous.

The second pass is supporting evidence, not final subtitle timing.  WhisperX
forced alignment remains the timing authority later in the pipeline.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .download import atomic_write_json, sha256_file, sha256_json, utc_now_iso
from .transcribe import (
    TranscriptionConfig,
    TranscriptionError,
    _import_whisper,
    _is_cuda_runtime_error,
    _select_device,
)
from .tr_correction import (
    TRCorrectionPackData,
    create_tr_correction_output,
    read_tr_correction_pack,
    validate_tr_correction_output,
)


AUDIO_REVIEW_V2_FORMAT = "muhtemel-ask-audio-review-v2"
AUDIO_REVIEW_V2_VERSION = "1.0"
AUDIO_REVIEW_V2_RECOVERY_FORMAT = "audio-review-v2-recovery-1"
MACHINE_NOTE_PREFIX = "machine_audio_review_v2:"
MANUAL_NOTE_PREFIX = "manual_audio_review_v2:"


class AudioReviewV2Error(RuntimeError):
    """Raised when bounded audio review cannot safely reach a final decision."""


class ReviewDecoder(Protocol):
    """Small injectable interface used by the runtime and deterministic tests."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def decode(
        self, audio_path: Path, *, initial_prompt: str | None = None
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class AudioReviewV2Config:
    """Conservative settings for the short-clip acoustic audit."""

    model_name: str = "large-v3"
    language: str = "tr"
    device: str = "auto"
    beam_size: int = 8
    best_of: int = 8
    compute_type_gpu: str = "float16"
    compute_type_cpu: str = "int8"
    allow_cpu_fallback: bool = True
    target_tolerance_ms: int = 160
    min_reference_similarity: float = 0.48
    min_shared_tokens: int = 1
    checkpoint_every: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.language != "tr":
            raise ValueError("audio review language must be 'tr'")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.beam_size < 1 or self.best_of < 1:
            raise ValueError("beam_size and best_of must be positive")
        if (
            isinstance(self.target_tolerance_ms, bool)
            or not isinstance(self.target_tolerance_ms, int)
            or not 0 <= self.target_tolerance_ms <= 500
        ):
            raise ValueError("target_tolerance_ms must be within [0, 500]")
        if (
            isinstance(self.min_reference_similarity, bool)
            or not isinstance(self.min_reference_similarity, (int, float))
            or not math.isfinite(float(self.min_reference_similarity))
            or not 0.0 <= float(self.min_reference_similarity) <= 1.0
        ):
            raise ValueError("min_reference_similarity must be within [0, 1]")
        if (
            isinstance(self.min_shared_tokens, bool)
            or not isinstance(self.min_shared_tokens, int)
            or self.min_shared_tokens < 1
        ):
            raise ValueError("min_shared_tokens must be positive")
        if (
            isinstance(self.checkpoint_every, bool)
            or not isinstance(self.checkpoint_every, int)
            or self.checkpoint_every < 1
        ):
            raise ValueError("checkpoint_every must be positive")


def _normalized_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).translate(
        str.maketrans({"I": "ı", "İ": "i"})
    ).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _tokens(value: Any) -> list[str]:
    normalized = _normalized_text(value)
    return normalized.split() if normalized else []


def _text_similarity(first: Any, second: Any) -> float:
    left = _normalized_text(first)
    right = _normalized_text(second)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _shared_token_count(first: Any, second: Any) -> int:
    return len(set(_tokens(first)).intersection(_tokens(second)))


def _finite_milliseconds(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioReviewV2Error(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise AudioReviewV2Error(f"{label} must be a finite number")
    return round(number * 1000)


def _normalize_decoder_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AudioReviewV2Error("review decoder result must be an object")
    words: list[dict[str, Any]] = []
    raw_words = value.get("words", [])
    if isinstance(raw_words, (str, bytes, bytearray)) or not isinstance(
        raw_words, Sequence
    ):
        raise AudioReviewV2Error("review decoder words must be a sequence")
    for position, raw in enumerate(raw_words, start=1):
        if not isinstance(raw, Mapping):
            raise AudioReviewV2Error(
                f"review decoder word {position} must be an object"
            )
        text = str(raw.get("text", "")).strip()
        start_ms = raw.get("start_ms")
        end_ms = raw.get("end_ms")
        if (
            not text
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or start_ms < 0
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
        ):
            raise AudioReviewV2Error(
                f"review decoder word {position} is malformed"
            )
        probability = raw.get("probability")
        if probability is not None and (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
        ):
            raise AudioReviewV2Error(
                f"review decoder word {position} probability is invalid"
            )
        words.append(
            {
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "probability": (
                    None if probability is None else float(probability)
                ),
            }
        )
    words.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["text"]))

    segments: list[dict[str, Any]] = []
    raw_segments = value.get("segments", [])
    if isinstance(raw_segments, (str, bytes, bytearray)) or not isinstance(
        raw_segments, Sequence
    ):
        raise AudioReviewV2Error("review decoder segments must be a sequence")
    for position, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, Mapping):
            raise AudioReviewV2Error(
                f"review decoder segment {position} must be an object"
            )
        text = str(raw.get("text", "")).strip()
        start_ms = raw.get("start_ms")
        end_ms = raw.get("end_ms")
        if (
            not text
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or start_ms < 0
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
        ):
            continue
        segments.append(
            {"text": text, "start_ms": start_ms, "end_ms": end_ms}
        )
    segments.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
    return {"words": words, "segments": segments}


class _FasterWhisperReviewDecoder:
    """Load one pinned model and reuse it for every bounded review WAV."""

    def __init__(self, config: AudioReviewV2Config) -> None:
        self.config = config
        self._model: Any | None = None
        self._device = ""
        self._compute_type = ""
        self._fallback_reason: str | None = None
        self._load_requested_device(config.device)

    def _load_requested_device(self, requested: str) -> None:
        WhisperModel, ctranslate2 = _import_whisper()
        if requested == "auto":
            selection = TranscriptionConfig(
                model_name=self.config.model_name,
                language=self.config.language,
                compute_type_gpu=self.config.compute_type_gpu,
                compute_type_cpu=self.config.compute_type_cpu,
                allow_cpu_fallback=self.config.allow_cpu_fallback,
            )
            device, compute_type = _select_device(selection, ctranslate2)
        elif requested == "cuda":
            try:
                cuda_count = int(ctranslate2.get_cuda_device_count())
            except Exception:
                cuda_count = 0
            if cuda_count < 1:
                raise AudioReviewV2Error(
                    "CUDA audio review was requested but no compatible GPU exists"
                )
            device, compute_type = "cuda", self.config.compute_type_gpu
        else:
            device, compute_type = "cpu", self.config.compute_type_cpu
        try:
            self._model = WhisperModel(
                self.config.model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=max(1, os.cpu_count() or 1),
                num_workers=1,
            )
        except Exception as exc:
            if (
                device == "cuda"
                and self.config.allow_cpu_fallback
                and _is_cuda_runtime_error(exc)
            ):
                self._fallback_reason = f"{type(exc).__name__}: {exc}"
                self._model = WhisperModel(
                    self.config.model_name,
                    device="cpu",
                    compute_type=self.config.compute_type_cpu,
                    cpu_threads=max(1, os.cpu_count() or 1),
                    num_workers=1,
                )
                device, compute_type = "cpu", self.config.compute_type_cpu
            else:
                raise AudioReviewV2Error(
                    f"Could not load bounded audio-review model: {exc}"
                ) from exc
        self._device = device
        self._compute_type = compute_type

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "engine": "faster-whisper",
            "model_name": self.config.model_name,
            "language": self.config.language,
            "device": self._device,
            "compute_type": self._compute_type,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "condition_on_previous_text": False,
            "initial_prompt_policy": "blind_first_then_evidence_prompt_on_ambiguity",
            "vad_filter": False,
            "fallback_reason": self._fallback_reason,
        }

    def _decode_once(
        self, audio_path: Path, *, initial_prompt: str | None
    ) -> dict[str, Any]:
        if self._model is None:
            raise AudioReviewV2Error("audio-review model is closed")
        iterator, _ = self._model.transcribe(
            str(audio_path),
            language=self.config.language,
            task="transcribe",
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            vad_filter=False,
        )
        segments: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        for segment in iterator:
            segment_text = str(getattr(segment, "text", "")).strip()
            segment_start = getattr(segment, "start", None)
            segment_end = getattr(segment, "end", None)
            if segment_text and segment_start is not None and segment_end is not None:
                segments.append(
                    {
                        "text": segment_text,
                        "start_ms": _finite_milliseconds(
                            segment_start, "secondary segment start"
                        ),
                        "end_ms": _finite_milliseconds(
                            segment_end, "secondary segment end"
                        ),
                    }
                )
            for word in getattr(segment, "words", None) or []:
                word_text = str(getattr(word, "word", "")).strip()
                word_start = getattr(word, "start", None)
                word_end = getattr(word, "end", None)
                if not word_text or word_start is None or word_end is None:
                    continue
                word_start_ms = _finite_milliseconds(
                    word_start, "secondary word start"
                )
                word_end_ms = _finite_milliseconds(
                    word_end, "secondary word end"
                )
                # faster-whisper can emit a zero-duration word at a segment
                # boundary. Millisecond rounding then makes start == end,
                # which must not abort the remaining bounded review clips.
                if word_end_ms < 0 or word_end_ms < word_start_ms:
                    continue
                word_start_ms = max(0, word_start_ms)
                word_end_ms = max(0, word_end_ms)
                if word_end_ms == word_start_ms:
                    word_end_ms += 1
                probability = getattr(word, "probability", None)
                words.append(
                    {
                        "text": word_text,
                        "start_ms": word_start_ms,
                        "end_ms": word_end_ms,
                        "probability": (
                            None if probability is None else float(probability)
                        ),
                    }
                )
        return _normalize_decoder_result({"segments": segments, "words": words})

    def decode(
        self, audio_path: Path, *, initial_prompt: str | None = None
    ) -> Mapping[str, Any]:
        try:
            return self._decode_once(audio_path, initial_prompt=initial_prompt)
        except Exception as exc:
            if (
                self._device == "cuda"
                and self.config.allow_cpu_fallback
                and _is_cuda_runtime_error(exc)
            ):
                self._fallback_reason = f"{type(exc).__name__}: {exc}"
                self.close()
                self._load_requested_device("cpu")
                return self._decode_once(audio_path, initial_prompt=initial_prompt)
            if isinstance(exc, AudioReviewV2Error):
                raise
            raise AudioReviewV2Error(
                f"Short review WAV transcription failed: {exc}"
            ) from exc

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:  # pragma: no cover - torch is optional outside Colab
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _review_inventory(
    pack: TRCorrectionPackData,
) -> list[tuple[str, str, Mapping[str, Any]]]:
    holes = {str(item["hole_uid"]): item for item in pack.speech_holes}
    candidates = {
        str(item["utterance_uid"]): item
        for item in pack.asr_hallucination_records
    }
    inventory: list[tuple[str, str, Mapping[str, Any]]] = []
    for utterance in pack.utterances:
        uid = str(utterance["utterance_uid"])
        if uid in holes:
            inventory.append((uid, "speech_hole", holes[uid]))
        elif uid in candidates:
            inventory.append((uid, "asr_caption_candidate", candidates[uid]))
    expected = len(holes) + len(candidates)
    if len(inventory) != expected:
        raise AudioReviewV2Error(
            "Correction pack review inventory is not one-to-one"
        )
    return inventory


def _target_decode(
    decoded: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    tolerance_ms: int,
) -> dict[str, Any]:
    target_start = int(evidence["start_ms"]) - int(evidence["clip_start_ms"])
    target_end = int(evidence["end_ms"]) - int(evidence["clip_start_ms"])
    lower = max(0, target_start - tolerance_ms)
    upper = target_end + tolerance_ms
    selected_words: list[dict[str, Any]] = []
    exact_overlap_ms = 0
    for word in decoded["words"]:
        overlap = max(
            0,
            min(target_end, int(word["end_ms"]))
            - max(target_start, int(word["start_ms"])),
        )
        center = (int(word["start_ms"]) + int(word["end_ms"])) / 2
        if overlap > 0 or lower <= center <= upper:
            selected_words.append(dict(word))
            exact_overlap_ms += overlap
    transcript = " ".join(item["text"].strip() for item in selected_words).strip()
    # Segment-level text can include the 750 ms context on either side and is
    # therefore never a safe target decision. Missing word timing fails closed.
    source = "word_timestamps" if transcript else "none"
    return {
        "transcript": transcript,
        "source": source,
        "exact_word_overlap_ms": exact_overlap_ms,
        "selected_word_count": len(selected_words),
        "target_start_in_clip_ms": target_start,
        "target_end_in_clip_ms": target_end,
        "words": selected_words,
    }


def _machine_decision(
    kind: str,
    evidence: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
    target_decode: Mapping[str, Any],
    config: AudioReviewV2Config,
) -> dict[str, Any]:
    transcript = str(target_decode["transcript"]).strip()
    references = {
        "tr_corrected": str(provisional_record.get("tr_corrected", "")),
        "asr_text": str(evidence.get("asr_text", "")),
        "youtube_text": str(evidence.get("youtube_text", "")),
    }
    scores = {
        name: _text_similarity(transcript, text)
        for name, text in references.items()
        if text.strip()
    }
    shared = {
        name: _shared_token_count(transcript, text)
        for name, text in references.items()
        if text.strip()
    }
    best_similarity = max(scores.values(), default=0.0)
    best_shared = max(shared.values(), default=0)
    target_token_count = len(_tokens(transcript))
    longest_reference_token_count = max(
        (len(_tokens(text)) for text in references.values() if text.strip()),
        default=0,
    )
    required_shared_tokens = max(
        config.min_shared_tokens,
        2 if target_token_count > 1 and longest_reference_token_count > 1 else 1,
    )
    base = {
        "secondary_transcript": transcript,
        "target_decode": copy.deepcopy(dict(target_decode)),
        "reference_similarity": scores,
        "shared_token_count": shared,
        "best_reference_similarity": best_similarity,
        "best_shared_token_count": best_shared,
        "required_shared_token_count": required_shared_tokens,
        "source": "secondary_asr",
    }
    if kind == "speech_hole":
        context_matches = max(
            _text_similarity(transcript, evidence.get("context_before", "")),
            _text_similarity(transcript, evidence.get("context_after", "")),
        )
        if transcript and context_matches < 0.85:
            return {
                **base,
                "decision": "confirmed_dialogue",
                "tr_corrected": transcript,
                "reason": "blind secondary ASR found target dialogue",
            }
        return {
            **base,
            "decision": "pending_audio_review",
            "tr_corrected": "",
            "reason": "speech hole has no reliable target transcript",
        }

    confirmed = transcript and (
        best_similarity >= float(config.min_reference_similarity)
        or best_shared >= required_shared_tokens
    )
    if confirmed:
        return {
            **base,
            "decision": "confirmed_dialogue",
            "tr_corrected": str(provisional_record["tr_corrected"]).strip(),
            "reason": "blind secondary ASR corroborated candidate text",
        }

    asr_text = str(evidence.get("asr_text", "")).strip()
    youtube_text = str(evidence.get("youtube_text", "")).strip()
    independent_text_agreement = (
        bool(asr_text and youtube_text)
        and _text_similarity(asr_text, youtube_text) >= 0.70
    )
    reason = str(evidence.get("reason", "")).casefold()
    # Automatic deletion needs the strongest structural case: no unpadded
    # independent VAD at all, no blind secondary transcript and no agreeing
    # YouTube text. Merely low overlap or a weak confidence flag stays pending.
    structural_silence = "zero_unpadded_independent_vad_overlap" in reason
    is_orphan = "orphan_youtube_caption" in set(
        evidence.get("risk_flags", [])
    )
    if not transcript and structural_silence and not independent_text_agreement and not is_orphan:
        return {
            **base,
            "decision": "discarded_asr_hallucination",
            "tr_corrected": "",
            "reason": "no blind secondary-ASR target speech and no corroborating caption",
        }
    return {
        **base,
        "decision": "pending_audio_review",
        "tr_corrected": str(provisional_record["tr_corrected"]).strip(),
        "reason": "candidate evidence remains acoustically ambiguous",
    }


def _evidence_prompt(
    evidence: Mapping[str, Any], provisional_record: Mapping[str, Any]
) -> str | None:
    """Return bounded Turkish context for a second, non-authoritative decode."""

    parts = [
        str(evidence.get("context_before", "")).strip(),
        str(provisional_record.get("tr_corrected", "")).strip(),
        str(evidence.get("context_after", "")).strip(),
    ]
    compact: list[str] = []
    for part in parts:
        if part and part not in compact:
            compact.append(part)
    prompt = " ".join(compact).strip()
    return prompt[:1_000] or None


def _manual_decision(
    uid: str,
    kind: str,
    raw: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AudioReviewV2Error(f"Manual audio review {uid} must be an object")
    allowed = {"disposition", "tr_corrected", "note"}
    extra = sorted(set(raw).difference(allowed))
    if extra:
        raise AudioReviewV2Error(
            f"Manual audio review {uid} has unsupported fields: {extra}"
        )
    disposition = raw.get("disposition")
    if not isinstance(disposition, str):
        raise AudioReviewV2Error(
            f"Manual audio review {uid} disposition must be a string"
        )
    allowed_dispositions = (
        {"confirmed_dialogue", "reviewed_non_dialogue"}
        if kind == "speech_hole"
        else {"confirmed_dialogue", "discarded_asr_hallucination"}
    )
    if disposition not in allowed_dispositions:
        raise AudioReviewV2Error(
            f"Manual audio review {uid} disposition must be one of "
            f"{sorted(allowed_dispositions)}"
        )
    note = raw.get("note")
    if not isinstance(note, str) or not note.strip():
        raise AudioReviewV2Error(
            f"Manual audio review {uid} requires a concrete listening note"
        )
    corrected = raw.get("tr_corrected", provisional_record.get("tr_corrected", ""))
    if not isinstance(corrected, str):
        raise AudioReviewV2Error(
            f"Manual audio review {uid} tr_corrected must be a string"
        )
    if disposition == "confirmed_dialogue" and not corrected.strip():
        raise AudioReviewV2Error(
            f"Manual audio review {uid} confirmed dialogue requires Turkish text"
        )
    if disposition != "confirmed_dialogue" and corrected.strip():
        raise AudioReviewV2Error(
            f"Manual audio review {uid} non-dialogue decision requires empty text"
        )
    return {
        "decision": disposition,
        "tr_corrected": corrected.strip(),
        "reason": note.strip(),
        "source": "manual",
        "secondary_transcript": "",
        "target_decode": {
            "transcript": "",
            "source": "manual",
            "exact_word_overlap_ms": 0,
            "selected_word_count": 0,
            "target_start_in_clip_ms": 0,
            "target_end_in_clip_ms": 0,
            "words": [],
        },
        "reference_similarity": {},
        "shared_token_count": {},
        "best_reference_similarity": 0.0,
        "best_shared_token_count": 0,
    }


def _report_sha(report: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in report.items() if key != "audio_review_sha256"}
    )


def _write_report(path: Path, draft: Mapping[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(dict(draft))
    report["audio_review_sha256"] = _report_sha(report)
    atomic_write_json(path, report)
    return report


def _outcome(
    uid: str,
    kind: str,
    evidence: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_uid = (
        str(evidence["hole_uid"])
        if kind == "speech_hole"
        else str(evidence["candidate_uid"])
    )
    corrected = str(decision.get("tr_corrected", ""))
    outcome = {
        "utterance_uid": uid,
        "evidence_kind": kind,
        "evidence_uid": evidence_uid,
        "start_ms": int(evidence["start_ms"]),
        "end_ms": int(evidence["end_ms"]),
        "clip_start_ms": int(evidence["clip_start_ms"]),
        "clip_end_ms": int(evidence["clip_end_ms"]),
        "audio_member": str(evidence["audio_member"]),
        "audio_sha256": str(evidence["audio_sha256"]),
        "decision": str(decision["decision"]),
        "source": str(decision["source"]),
        "secondary_transcript": str(decision.get("secondary_transcript", "")),
        "target_decode": copy.deepcopy(dict(decision["target_decode"])),
        "reference_similarity": copy.deepcopy(
            dict(decision.get("reference_similarity", {}))
        ),
        "shared_token_count": copy.deepcopy(
            dict(decision.get("shared_token_count", {}))
        ),
        "best_reference_similarity": float(
            decision.get("best_reference_similarity", 0.0)
        ),
        "best_shared_token_count": int(
            decision.get("best_shared_token_count", 0)
        ),
        "required_shared_token_count": int(
            decision.get("required_shared_token_count", 1)
        ),
        "tr_corrected_sha256": hashlib.sha256(
            corrected.encode("utf-8")
        ).hexdigest(),
        "reason": str(decision["reason"]),
    }
    if "blind_target_decode" in decision:
        outcome["blind_target_decode"] = copy.deepcopy(
            dict(decision["blind_target_decode"])
        )
        outcome["evidence_prompt_sha256"] = decision.get(
            "evidence_prompt_sha256"
        )
    return outcome


def validate_audio_review_v2_report(
    input_pack_path: str | os.PathLike[str],
    provisional_output_path: str | os.PathLike[str],
    final_output_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the complete machine/manual audit against both correction ZIPs."""

    pack = read_tr_correction_pack(input_pack_path)
    provisional = validate_tr_correction_output(
        input_pack_path, provisional_output_path
    )
    final = validate_tr_correction_output(input_pack_path, final_output_path)
    try:
        with Path(report_path).open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AudioReviewV2Error(f"Cannot read audio-review report: {exc}") from exc
    if not isinstance(report, Mapping):
        raise AudioReviewV2Error("audio-review report must be an object")
    if report.get("format") != AUDIO_REVIEW_V2_FORMAT:
        raise AudioReviewV2Error("wrong audio-review report format")
    if report.get("format_version") != AUDIO_REVIEW_V2_VERSION:
        raise AudioReviewV2Error("unsupported audio-review report version")
    if report.get("status") != "PASS" or report.get("pending_count") != 0:
        raise AudioReviewV2Error("audio-review report is not complete")
    if report.get("episode") != pack.manifest["episode"]:
        raise AudioReviewV2Error("audio-review report episode mismatch")
    if report.get("correction_input_sha256") != pack.manifest["input_sha256"]:
        raise AudioReviewV2Error("audio-review input SHA-256 mismatch")
    if report.get("provisional_output_sha256") != provisional.output_sha256:
        raise AudioReviewV2Error("audio-review provisional SHA-256 mismatch")
    if report.get("correction_output_sha256") != final.output_sha256:
        raise AudioReviewV2Error("audio-review final SHA-256 mismatch")
    digest = report.get("audio_review_sha256")
    if not isinstance(digest, str) or digest != _report_sha(report):
        raise AudioReviewV2Error("audio-review report digest mismatch")
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        raise AudioReviewV2Error("audio-review outcomes must be a list")
    inventory = _review_inventory(pack)
    expected_uids = [uid for uid, _kind, _evidence in inventory]
    actual_uids = [
        item.get("utterance_uid") if isinstance(item, Mapping) else None
        for item in outcomes
    ]
    if actual_uids != expected_uids:
        raise AudioReviewV2Error("audio-review outcome identity/order mismatch")
    final_by_uid = {str(item["utterance_uid"]): item for item in final.records}
    for (uid, kind, evidence), raw in zip(inventory, outcomes):
        if not isinstance(raw, Mapping):
            raise AudioReviewV2Error(f"audio-review outcome {uid} is malformed")
        evidence_uid = (
            str(evidence["hole_uid"])
            if kind == "speech_hole"
            else str(evidence["candidate_uid"])
        )
        comparisons = {
            "evidence_kind": kind,
            "evidence_uid": evidence_uid,
            "audio_member": evidence["audio_member"],
            "audio_sha256": evidence["audio_sha256"],
            "start_ms": evidence["start_ms"],
            "end_ms": evidence["end_ms"],
            "clip_start_ms": evidence["clip_start_ms"],
            "clip_end_ms": evidence["clip_end_ms"],
        }
        for field, expected in comparisons.items():
            if raw.get(field) != expected:
                raise AudioReviewV2Error(
                    f"audio-review outcome {uid} changed {field}"
                )
        record = final_by_uid[uid]
        if raw.get("decision") != record["review_disposition"]:
            raise AudioReviewV2Error(
                f"audio-review outcome {uid} disposition mismatch"
            )
        corrected_sha = hashlib.sha256(
            str(record["tr_corrected"]).encode("utf-8")
        ).hexdigest()
        if raw.get("tr_corrected_sha256") != corrected_sha:
            raise AudioReviewV2Error(
                f"audio-review outcome {uid} corrected-text mismatch"
            )
        if record["audio_reviewed"] is not True or record["review_required"] is not False:
            raise AudioReviewV2Error(
                f"audio-review outcome {uid} did not close the correction gate"
            )
    if report.get("review_count") != len(outcomes):
        raise AudioReviewV2Error("audio-review report count mismatch")
    return copy.deepcopy(dict(report))


def resolve_tr_audio_reviews_v2(
    input_pack_path: str | os.PathLike[str],
    provisional_output_path: str | os.PathLike[str],
    final_output_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    recovery_path: str | os.PathLike[str],
    *,
    config: AudioReviewV2Config | None = None,
    manual_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    force: bool = False,
    decoder: ReviewDecoder | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Resolve only immutable review WAVs and create the strict final TR ZIP.

    A checkpoint is written every few clips.  Ambiguous evidence produces a
    durable report and raises without creating or overwriting the final output.
    """

    settings = config or AudioReviewV2Config()
    if not isinstance(settings, AudioReviewV2Config):
        raise AudioReviewV2Error("config must be AudioReviewV2Config")
    overrides = dict(manual_overrides or {})
    if any(not isinstance(uid, str) or not uid for uid in overrides):
        raise AudioReviewV2Error("manual override UIDs must be non-empty strings")
    pack = read_tr_correction_pack(input_pack_path)
    provisional = validate_tr_correction_output(
        input_pack_path, provisional_output_path
    )
    inventory = _review_inventory(pack)
    review_uids = {uid for uid, _kind, _evidence in inventory}
    unknown_overrides = sorted(set(overrides).difference(review_uids))
    if unknown_overrides:
        raise AudioReviewV2Error(
            f"Manual overrides name unknown review UIDs: {unknown_overrides}"
        )
    provisional_by_uid = {
        str(record["utterance_uid"]): record for record in provisional.records
    }
    for uid in review_uids:
        record = provisional_by_uid[uid]
        if not (
            record["review_required"] is True
            and record["audio_reviewed"] is False
            and record["review_disposition"] == "pending_audio_review"
            and record["non_dialogue"] is False
        ):
            raise AudioReviewV2Error(
                f"Text-only correction must leave immutable WAV decision pending: {uid}"
            )
    unsupported_pending = sorted(
        str(record["utterance_uid"])
        for record in provisional.records
        if record["review_required"] is True
        and str(record["utterance_uid"]) not in review_uids
    )
    if unsupported_pending:
        raise AudioReviewV2Error(
            "Turkish correction deleted lexical ASR tokens that have no review "
            "WAV. Rerun 01_PREPARE_TR with these EXTRA_AUDIO_REVIEW_UIDS: "
            f"{unsupported_pending}"
        )

    code_path = Path(__file__)
    identity = {
        "correction_input_sha256": pack.manifest["input_sha256"],
        "provisional_output_sha256": provisional.output_sha256,
        "config": asdict(settings),
        "code_sha256": sha256_file(code_path),
    }
    review_input_sha256 = sha256_json(identity)
    recovery_file = Path(recovery_path)
    cached: dict[str, dict[str, Any]] = {}
    if not force and recovery_file.is_file():
        try:
            with recovery_file.open(encoding="utf-8") as handle:
                recovery = json.load(handle)
            if (
                isinstance(recovery, Mapping)
                and recovery.get("format") == AUDIO_REVIEW_V2_RECOVERY_FORMAT
                and recovery.get("input_sha256") == review_input_sha256
                and isinstance(recovery.get("machine_decisions"), Mapping)
                and recovery.get("recovery_sha256")
                == sha256_json(
                    {
                        key: value
                        for key, value in recovery.items()
                        if key != "recovery_sha256"
                    }
                )
            ):
                cached = {
                    str(uid): copy.deepcopy(dict(value))
                    for uid, value in recovery["machine_decisions"].items()
                    if isinstance(uid, str) and isinstance(value, Mapping)
                }
        except (OSError, UnicodeError, json.JSONDecodeError):
            cached = {}

    runtime_decoder = decoder
    owns_decoder = decoder is None
    machine_decisions: dict[str, dict[str, Any]] = {}

    def save_recovery() -> None:
        recovery = {
            "format": AUDIO_REVIEW_V2_RECOVERY_FORMAT,
            "input_sha256": review_input_sha256,
            "updated_at": utc_now_iso(),
            "machine_decisions": machine_decisions,
        }
        recovery["recovery_sha256"] = sha256_json(recovery)
        atomic_write_json(
            recovery_file,
            recovery,
        )

    try:
        with zipfile.ZipFile(input_pack_path, "r") as archive, tempfile.TemporaryDirectory(
            prefix="ma-audio-review-v2-"
        ) as temporary:
            temporary_root = Path(temporary)
            for position, (uid, kind, evidence) in enumerate(inventory, start=1):
                if uid in cached:
                    cached_decision = cached[uid]
                    try:
                        normalized_cached_words = _normalize_decoder_result(
                            {
                                "words": cached_decision["target_decode"]["words"],
                                "segments": [],
                            }
                        )["words"]
                        cached_target_decode = {
                            key: cached_decision["target_decode"][key]
                            for key in (
                                "transcript",
                                "source",
                                "exact_word_overlap_ms",
                                "selected_word_count",
                                "target_start_in_clip_ms",
                                "target_end_in_clip_ms",
                            )
                        }
                        cached_target_decode["words"] = normalized_cached_words
                        recomputed = _machine_decision(
                            kind,
                            evidence,
                            provisional_by_uid[uid],
                            cached_target_decode,
                            settings,
                        )
                        if "blind_target_decode" in cached_decision:
                            expected_prompt = _evidence_prompt(
                                evidence, provisional_by_uid[uid]
                            )
                            if expected_prompt is None:
                                raise AudioReviewV2Error(
                                    "cached prompted decision has no reproducible prompt"
                                )
                            recomputed["source"] = (
                                "secondary_asr_prompted_confirmation"
                            )
                            recomputed["reason"] = (
                                "evidence-conditioned secondary ASR produced corroborating "
                                "target text; final CTC alignment remains mandatory"
                            )
                            recomputed["blind_target_decode"] = copy.deepcopy(
                                cached_decision["blind_target_decode"]
                            )
                            recomputed["evidence_prompt_sha256"] = hashlib.sha256(
                                expected_prompt.encode("utf-8")
                            ).hexdigest()
                    except (KeyError, TypeError, ValueError, AudioReviewV2Error):
                        recomputed = None
                    if recomputed is not None and sha256_json(recomputed) == sha256_json(
                        cached_decision
                    ):
                        machine_decisions[uid] = cached_decision
                        if progress is not None:
                            progress(
                                f"Audio review {position}/{len(inventory)} resumed: {uid}"
                            )
                        continue
                member = str(evidence["audio_member"])
                payload = archive.read(member)
                if hashlib.sha256(payload).hexdigest() != evidence["audio_sha256"]:
                    raise AudioReviewV2Error(
                        f"Review WAV hash changed before decode: {member}"
                    )
                clip_path = temporary_root / f"{position:03d}.wav"
                clip_path.write_bytes(payload)
                if runtime_decoder is None:
                    runtime_decoder = _FasterWhisperReviewDecoder(settings)
                decoded = _normalize_decoder_result(
                    runtime_decoder.decode(clip_path, initial_prompt=None)
                )
                target = _target_decode(
                    decoded,
                    evidence,
                    tolerance_ms=settings.target_tolerance_ms,
                )
                decision = _machine_decision(
                    kind,
                    evidence,
                    provisional_by_uid[uid],
                    target,
                    settings,
                )
                if decision["decision"] == "pending_audio_review":
                    evidence_prompt = _evidence_prompt(
                        evidence, provisional_by_uid[uid]
                    )
                    if evidence_prompt:
                        prompted_decoded = _normalize_decoder_result(
                            runtime_decoder.decode(
                                clip_path, initial_prompt=evidence_prompt
                            )
                        )
                        prompted_target = _target_decode(
                            prompted_decoded,
                            evidence,
                            tolerance_ms=settings.target_tolerance_ms,
                        )
                        prompted_decision = _machine_decision(
                            kind,
                            evidence,
                            provisional_by_uid[uid],
                            prompted_target,
                            settings,
                        )
                        if prompted_decision["decision"] == "confirmed_dialogue":
                            prompted_decision["source"] = (
                                "secondary_asr_prompted_confirmation"
                            )
                            prompted_decision["reason"] = (
                                "evidence-conditioned secondary ASR produced corroborating "
                                "target text; final CTC alignment remains mandatory"
                            )
                            prompted_decision["blind_target_decode"] = target
                            prompted_decision["evidence_prompt_sha256"] = hashlib.sha256(
                                evidence_prompt.encode("utf-8")
                            ).hexdigest()
                            decision = prompted_decision
                machine_decisions[uid] = decision
                if (
                    position % settings.checkpoint_every == 0
                    or position == len(inventory)
                ):
                    save_recovery()
                if progress is not None:
                    progress(
                        f"Audio review {position}/{len(inventory)}: "
                        f"{machine_decisions[uid]['decision']}"
                    )
    finally:
        if owns_decoder and runtime_decoder is not None:
            runtime_decoder.close()

    outcomes: list[dict[str, Any]] = []
    resolved_records = [copy.deepcopy(dict(record)) for record in provisional.records]
    resolved_by_uid = {
        str(record["utterance_uid"]): record for record in resolved_records
    }
    pending_uids: list[str] = []
    for uid, kind, evidence in inventory:
        decision = machine_decisions[uid]
        if uid in overrides:
            decision = _manual_decision(
                uid, kind, overrides[uid], provisional_by_uid[uid]
            )
        record = resolved_by_uid[uid]
        disposition = str(decision["decision"])
        if disposition == "pending_audio_review":
            pending_uids.append(uid)
            outcomes.append(_outcome(uid, kind, evidence, decision))
            continue
        record["review_required"] = False
        record["audio_reviewed"] = True
        record["review_disposition"] = disposition
        record["tr_corrected"] = str(decision["tr_corrected"]).strip()
        record["non_dialogue"] = disposition in {
            "reviewed_non_dialogue",
            "discarded_asr_hallucination",
        }
        prefix = (
            MANUAL_NOTE_PREFIX
            if decision["source"] == "manual"
            else MACHINE_NOTE_PREFIX
        )
        record["note"] = f"{prefix} {str(decision['reason']).strip()}"
        outcomes.append(_outcome(uid, kind, evidence, decision))

    decoder_provenance = (
        copy.deepcopy(dict(runtime_decoder.provenance))
        if runtime_decoder is not None
        else {
            "engine": "checkpoint-only",
            "model_name": settings.model_name,
            "language": settings.language,
        }
    )
    base_report: dict[str, Any] = {
        "format": AUDIO_REVIEW_V2_FORMAT,
        "format_version": AUDIO_REVIEW_V2_VERSION,
        "status": "NEEDS_MANUAL_REVIEW" if pending_uids else "PASS",
        "created_at": utc_now_iso(),
        "episode": pack.manifest["episode"],
        "correction_input_sha256": pack.manifest["input_sha256"],
        "provisional_output_sha256": provisional.output_sha256,
        "correction_output_sha256": None,
        "review_input_sha256": review_input_sha256,
        "manual_overrides_sha256": sha256_json(overrides),
        "config": asdict(settings),
        "decoder": decoder_provenance,
        "review_count": len(outcomes),
        "resolved_count": len(outcomes) - len(pending_uids),
        "pending_count": len(pending_uids),
        "pending_utterance_uids": pending_uids,
        "outcomes": outcomes,
    }
    report_file = Path(report_path)
    if pending_uids:
        _write_report(report_file, base_report)
        raise AudioReviewV2Error(
            "Bounded Colab audio review left ambiguous items; "
            f"pending_count={len(pending_uids)}, report={report_file}. "
            "Listen only to the listed audio_member files and fill "
            "MANUAL_AUDIO_REVIEW, then rerun this cell."
        )

    final_manifest = create_tr_correction_output(
        input_pack_path, resolved_records, final_output_path
    )
    base_report["correction_output_sha256"] = final_manifest["output_sha256"]
    _write_report(report_file, base_report)
    validated = validate_audio_review_v2_report(
        input_pack_path,
        provisional_output_path,
        final_output_path,
        report_file,
    )
    return validated


__all__ = [
    "AUDIO_REVIEW_V2_FORMAT",
    "AUDIO_REVIEW_V2_RECOVERY_FORMAT",
    "AUDIO_REVIEW_V2_VERSION",
    "AudioReviewV2Config",
    "AudioReviewV2Error",
    "MACHINE_NOTE_PREFIX",
    "MANUAL_NOTE_PREFIX",
    "resolve_tr_audio_reviews_v2",
    "validate_audio_review_v2_report",
]
