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
import wave
import zipfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import primary_checkpoint
from .download import atomic_write_json, sha256_file, sha256_json, utc_now_iso
from ..progress import mark_work_progress
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
CONTEXTUAL_BOUNDARY_POLICY = "contextual-boundary-policy-3"
ACOUSTIC_DUPLICATE_POLICY = "acoustic-duplicate-policy-2"


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


def _has_one_millisecond_word_timing(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    words = value.get("words")
    if isinstance(words, (str, bytes, bytearray)) or not isinstance(
        words, Sequence
    ):
        return False
    return any(
        isinstance(word, Mapping)
        and isinstance(word.get("start_ms"), int)
        and not isinstance(word.get("start_ms"), bool)
        and isinstance(word.get("end_ms"), int)
        and not isinstance(word.get("end_ms"), bool)
        and int(word["end_ms"]) - int(word["start_ms"]) == 1
        for word in words
    )


def _is_known_short_clip_hallucination(
    value: Any,
    *,
    stage: str | None = None,
    decoded: Mapping[str, Any] | None = None,
    staged_decodes: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> bool:
    normalized = _normalized_text(value)
    if normalized == "altyazı m k":
        return True
    if (
        normalized != "izlediğiniz için teşekkür ederim"
        or stage != "exact_target_crop"
        or not _has_one_millisecond_word_timing(decoded)
    ):
        return False
    transcripts = {
        name: str(candidate.get("transcript", "")).strip()
        for name, candidate in staged_decodes
        if isinstance(name, str) and isinstance(candidate, Mapping)
    }
    return (
        "blind_padded" in transcripts
        and "prompted_padded" in transcripts
        and not transcripts["blind_padded"]
        and not transcripts["prompted_padded"]
    )


def _is_repetitive_asr_hallucination(value: Any) -> bool:
    tokens = _tokens(value)
    return len(tokens) >= 20 and len(set(tokens)) <= max(2, len(tokens) // 10)


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

    def __init__(
        self,
        config: AudioReviewV2Config,
        *,
        model_dir: Path | None = None,
        cache_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._model_dir = model_dir
        self._cache_identity = copy.deepcopy(dict(cache_identity or {}))
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
                str(self._model_dir or self.config.model_name),
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
                    str(self._model_dir or self.config.model_name),
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
            "cache_identity": copy.deepcopy(self._cache_identity),
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


def _write_exact_target_crop(
    source_path: Path,
    destination_path: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    target_start_ms = int(evidence["start_ms"]) - int(evidence["clip_start_ms"])
    target_end_ms = int(evidence["end_ms"]) - int(evidence["clip_start_ms"])
    if target_start_ms < 0 or target_end_ms <= target_start_ms:
        raise AudioReviewV2Error("review target interval is invalid")
    try:
        with wave.open(str(source_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            compression = source.getcomptype()
            frame_count = source.getnframes()
            if channels < 1 or sample_width < 1 or sample_rate < 1:
                raise AudioReviewV2Error("review WAV format is invalid")
            if compression != "NONE":
                raise AudioReviewV2Error("review WAV must be uncompressed PCM")
            start_frame = round(target_start_ms * sample_rate / 1000)
            end_frame = round(target_end_ms * sample_rate / 1000)
            if start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
                raise AudioReviewV2Error("review target crop exceeds WAV bounds")
            source.setpos(start_frame)
            frames = source.readframes(end_frame - start_frame)
        if len(frames) != (end_frame - start_frame) * channels * sample_width:
            raise AudioReviewV2Error("review target crop is truncated")
        with wave.open(str(destination_path), "wb") as destination:
            destination.setnchannels(channels)
            destination.setsampwidth(sample_width)
            destination.setframerate(sample_rate)
            destination.writeframes(frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioReviewV2Error(f"Could not create exact review crop: {exc}") from exc
    return {
        "audio_sha256": sha256_file(destination_path),
        "sample_rate_hz": sample_rate,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_ms": round((end_frame - start_frame) * 1000 / sample_rate),
    }


def _whole_clip_target_decode(
    decoded: Mapping[str, Any], *, duration_ms: int
) -> dict[str, Any]:
    return _target_decode(
        decoded,
        {"start_ms": 0, "end_ms": duration_ms, "clip_start_ms": 0},
        tolerance_ms=0,
    )


def _exact_target_crop_confirms(
    kind: str,
    evidence: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> bool:
    if (
        kind == "speech_hole"
        or decision.get("decision") != "confirmed_dialogue"
        or "orphan_youtube_caption" in set(evidence.get("risk_flags", []))
        or not (
            str(evidence.get("context_before", "")).strip()
            or str(evidence.get("context_after", "")).strip()
        )
    ):
        return False
    transcript = _normalized_text(
        decision.get("target_decode", {}).get("transcript", "")
    )
    references = (
        provisional_record.get("tr_corrected", ""),
        evidence.get("asr_text", ""),
        evidence.get("youtube_text", ""),
    )
    return bool(transcript) and any(
        transcript == _normalized_text(reference)
        for reference in references
        if str(reference).strip()
    )


def _machine_decision(
    kind: str,
    evidence: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
    target_decode: Mapping[str, Any],
    config: AudioReviewV2Config,
) -> dict[str, Any]:
    transcript = str(target_decode["transcript"]).strip()
    known_short_clip_hallucination = _is_known_short_clip_hallucination(
        transcript
    )
    acoustic_transcript = "" if known_short_clip_hallucination else transcript
    references = {
        "tr_corrected": str(provisional_record.get("tr_corrected", "")),
        "asr_text": str(evidence.get("asr_text", "")),
        "youtube_text": str(evidence.get("youtube_text", "")),
    }
    scores = {
        name: _text_similarity(acoustic_transcript, text)
        for name, text in references.items()
        if text.strip()
    }
    shared = {
        name: _shared_token_count(acoustic_transcript, text)
        for name, text in references.items()
        if text.strip()
    }
    best_similarity = max(scores.values(), default=0.0)
    best_shared = max(shared.values(), default=0)
    target_token_count = len(_tokens(acoustic_transcript))
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
        "known_short_clip_hallucination": known_short_clip_hallucination,
    }
    if {"overlapping_rescue_word_evidence", "incomplete_provisional_word_timing"}.intersection(
        evidence.get("risk_flags", [])
    ):
        return {
            **base,
            "decision": "pending_audio_review",
            "tr_corrected": str(provisional_record["tr_corrected"]).strip(),
            "reason": (
                "incomplete or overlapping source requires bounded replacement "
                "evidence or manual review"
            ),
        }
    has_adjacent_context = bool(
        str(evidence.get("context_before", "")).strip()
        or str(evidence.get("context_after", "")).strip()
    )
    is_orphan = "orphan_youtube_caption" in set(
        evidence.get("risk_flags", [])
    )
    if not has_adjacent_context or is_orphan:
        return {
            **base,
            "decision": "pending_audio_review",
            "tr_corrected": str(provisional_record.get("tr_corrected", "")).strip(),
            "reason": (
                "orphan caption or missing adjacent context remains fail-closed"
            ),
        }
    if kind == "speech_hole":
        context_matches = max(
            _text_similarity(
                acoustic_transcript, evidence.get("context_before", "")
            ),
            _text_similarity(
                acoustic_transcript, evidence.get("context_after", "")
            ),
        )
        if (
            acoustic_transcript
            and int(target_decode.get("exact_word_overlap_ms", 0)) > 0
            and context_matches < 0.85
        ):
            return {
                **base,
                "decision": "confirmed_dialogue",
                "tr_corrected": acoustic_transcript,
                "reason": "blind secondary ASR found target dialogue",
            }
        return {
            **base,
            "decision": "pending_audio_review",
            "tr_corrected": "",
            "reason": "speech hole has no reliable target transcript",
        }

    repetitive_source_hallucination = (
        kind == "asr_caption_candidate"
        and _is_repetitive_asr_hallucination(evidence.get("asr_text", ""))
    )
    confirmed = acoustic_transcript and not repetitive_source_hallucination and (
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
    if (
        not acoustic_transcript
        and structural_silence
        and not independent_text_agreement
        and not is_orphan
    ):
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


def _contextual_boundary_decision(
    kind: str,
    evidence: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
    pending_decision: Mapping[str, Any],
    staged_decodes: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if pending_decision.get("decision") != "pending_audio_review":
        return copy.deepcopy(dict(pending_decision))
    if "overlapping_rescue_word_evidence" in set(
        evidence.get("risk_flags", [])
    ):
        return copy.deepcopy(dict(pending_decision))
    context_before = str(evidence.get("context_before", "")).strip()
    context_after = str(evidence.get("context_after", "")).strip()
    if not context_before and not context_after:
        return copy.deepcopy(dict(pending_decision))
    is_orphan = "orphan_youtube_caption" in set(evidence.get("risk_flags", []))
    decode_audit = []
    has_usable_target_text = False
    for stage, decoded in staged_decodes:
        transcript = str(decoded.get("transcript", "")).strip()
        exact_word_overlap_ms = int(decoded.get("exact_word_overlap_ms", 0))
        known_short_clip_hallucination = _is_known_short_clip_hallucination(
            transcript,
            stage=stage,
            decoded=decoded,
            staged_decodes=staged_decodes,
        )
        one_millisecond_word_timing = (
            stage == "exact_target_crop"
            and _has_one_millisecond_word_timing(decoded)
        )
        context_similarity = max(
            _text_similarity(transcript, context_before),
            _text_similarity(transcript, context_after),
        )
        usable_target_text = (
            bool(transcript)
            and not known_short_clip_hallucination
            and context_similarity < 0.85
            and (stage != "prompted_padded" or exact_word_overlap_ms > 0)
        )
        has_usable_target_text = has_usable_target_text or usable_target_text
        decode_audit.append(
            {
                "stage": stage,
                "transcript": transcript,
                "transcript_sha256": hashlib.sha256(
                    transcript.encode("utf-8")
                ).hexdigest(),
                "transcript_present": bool(transcript),
                "usable_target_text": usable_target_text,
                "known_short_clip_hallucination": known_short_clip_hallucination,
                "one_millisecond_word_timing": one_millisecond_word_timing,
                "exact_word_overlap_ms": exact_word_overlap_ms,
            }
        )
    audit = {
        "policy": CONTEXTUAL_BOUNDARY_POLICY,
        "context_before_present": bool(context_before),
        "context_after_present": bool(context_after),
        "context_sha256": sha256_json(
            {"before": context_before, "after": context_after}
        ),
        "orphan_caption": is_orphan,
        "bounded_decodes": decode_audit,
    }
    corrected = str(provisional_record.get("tr_corrected", "")).strip()
    known_source_hallucination = _is_known_short_clip_hallucination(
        evidence.get("asr_text", "")
    )
    repetitive_source_hallucination = _is_repetitive_asr_hallucination(
        evidence.get("asr_text", "")
    )
    usable_decodes = [
        item for item in decode_audit if item["usable_target_text"] is True
    ]
    if kind == "asr_caption_candidate" and repetitive_source_hallucination:
        return {
            **copy.deepcopy(dict(pending_decision)),
            "decision": "discarded_asr_hallucination",
            "tr_corrected": "",
            "reason": (
                "repetitive source-ASR loop was not retained as one long dialogue cue"
            ),
            "source": "contextual_boundary_policy",
            "forced_alignment_required": True,
            "contextual_boundary_audit": audit,
        }
    if "incomplete_provisional_word_timing" in set(evidence.get("risk_flags", [])):
        return copy.deepcopy(dict(pending_decision))
    if kind == "asr_caption_candidate" and known_source_hallucination:
        if usable_decodes:
            selected = usable_decodes[-1]
            return {
                **copy.deepcopy(dict(pending_decision)),
                "decision": "confirmed_dialogue",
                "tr_corrected": selected["transcript"],
                "reason": (
                    "known subtitle hallucination was replaced by bounded target "
                    "audio; final forced alignment is required"
                ),
                "source": "contextual_boundary_policy",
                "forced_alignment_required": True,
                "contextual_boundary_audit": audit,
            }
        return {
            **copy.deepcopy(dict(pending_decision)),
            "decision": "discarded_asr_hallucination",
            "tr_corrected": "",
            "reason": (
                "known subtitle hallucination had no usable bounded target audio"
            ),
            "source": "contextual_boundary_policy",
            "forced_alignment_required": True,
            "contextual_boundary_audit": audit,
        }
    if (
        kind == "asr_caption_candidate"
        and not is_orphan
        and corrected
        and str(evidence.get("asr_text", "")).strip()
    ):
        if "provisional_word_gap_requires_audio_review" in set(evidence.get("risk_flags", [])):
            return copy.deepcopy(dict(pending_decision))
        return {
            **copy.deepcopy(dict(pending_decision)),
            "decision": "confirmed_dialogue",
            "tr_corrected": corrected,
            "reason": (
                "non-orphan ASR boundary fragment retained from corrected text; "
                "adjacent transcript context is present and final forced alignment "
                "is required"
            ),
            "source": "contextual_boundary_policy",
            "forced_alignment_required": True,
            "contextual_boundary_audit": audit,
        }
    if kind == "speech_hole" and not corrected and has_usable_target_text:
        selected = next(
            item for item in reversed(decode_audit) if item["usable_target_text"]
        )
        return {
            **copy.deepcopy(dict(pending_decision)),
            "decision": "confirmed_dialogue",
            "tr_corrected": selected["transcript"],
            "reason": (
                "bounded target audio produced usable dialogue in adjacent "
                "subtitle context; final forced alignment is required"
            ),
            "source": "contextual_boundary_policy",
            "forced_alignment_required": True,
            "contextual_boundary_audit": audit,
        }
    if kind == "speech_hole" and not corrected:
        return {
            **copy.deepcopy(dict(pending_decision)),
            "decision": "reviewed_non_dialogue",
            "tr_corrected": "",
            "reason": (
                "blank bounded speech-hole interval had adjacent transcript context "
                "but no usable target text; no dialogue text was invented and final "
                "forced alignment is required"
            ),
            "source": "contextual_boundary_policy",
            "forced_alignment_required": True,
            "contextual_boundary_audit": audit,
        }
    return copy.deepcopy(dict(pending_decision))


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


_EP13_LEGACY_AUDIO_REVIEW_REPORT_SHA256 = (
    "47540a5da0ca4f65b6c09b1865795d3690aa545ff3b1426cac5c0e512bc42138"
)
_EP13_LEGACY_TR_CORRECTED_ZIP_SHA256 = (
    "b026480b2b8a08a68f4c068bb669f55a4753701301d682d21461f6b8df51471f"
)
_EP13_LEGACY_TR_PACK_SHA256 = (
    "7d74ddce943e89702ef60cf10aebc02921793702b2307ef62b91cd08117d53a9"
)
_EP13_LEGACY_PROVISIONAL_ZIP_SHA256 = (
    "6ef69291a806531a1eb544afafc9c2c6452d1f603877418abe59da08b6e415ed"
)
_EP13_LEGACY_RECOVERY_SHA256 = (
    "412c3baabfe2159cce4696877530a78d3db14cc849aca4b276d004401a4b1991"
)


def _validated_cached_decode(
    raw: Mapping[str, Any], *, target_start_ms: int, target_end_ms: int,
    clip_duration_ms: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AudioReviewV2Error("legacy decode evidence must be an object")
    words = _normalize_decoder_result(
        {"words": raw.get("words"), "segments": []}
    )["words"]
    transcript = " ".join(str(word["text"]).strip() for word in words).strip()
    if (
        raw.get("words") != words
        or raw.get("transcript") != transcript
        or raw.get("selected_word_count") != len(words)
        or raw.get("target_start_in_clip_ms") != target_start_ms
        or raw.get("target_end_in_clip_ms") != target_end_ms
        or raw.get("source") != ("word_timestamps" if transcript else "none")
        or any(
            int(word["start_ms"]) < 0
            or int(word["end_ms"]) > clip_duration_ms
            for word in words
        )
    ):
        raise AudioReviewV2Error("legacy decode transcript/timing evidence mismatch")
    exact_overlap_ms = sum(
        max(
            0,
            min(target_end_ms, int(word["end_ms"]))
            - max(target_start_ms, int(word["start_ms"])),
        )
        for word in words
    )
    if raw.get("exact_word_overlap_ms") != exact_overlap_ms:
        raise AudioReviewV2Error("legacy decode overlap evidence mismatch")
    return {
        "transcript": transcript,
        "source": raw["source"],
        "exact_word_overlap_ms": exact_overlap_ms,
        "selected_word_count": len(words),
        "target_start_in_clip_ms": target_start_ms,
        "target_end_in_clip_ms": target_end_ms,
        "words": words,
    }


def _validate_legacy_decision_evidence(
    decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
    clip_path: Path,
    crop_path: Path,
) -> None:
    clip_duration_ms = int(evidence["clip_end_ms"]) - int(
        evidence["clip_start_ms"]
    )
    target_start_ms = int(evidence["start_ms"]) - int(evidence["clip_start_ms"])
    target_end_ms = int(evidence["end_ms"]) - int(evidence["clip_start_ms"])
    target = _validated_cached_decode(
        decision.get("target_decode", {}),
        target_start_ms=target_start_ms,
        target_end_ms=target_end_ms,
        clip_duration_ms=clip_duration_ms,
    )
    blind = None
    if "blind_target_decode" in decision:
        blind = _validated_cached_decode(
            decision["blind_target_decode"],
            target_start_ms=target_start_ms,
            target_end_ms=target_end_ms,
            clip_duration_ms=clip_duration_ms,
        )
    exact = None
    if "exact_target_crop_decode" in decision:
        crop = _write_exact_target_crop(clip_path, crop_path, evidence)
        if decision.get("exact_target_crop") != crop:
            raise AudioReviewV2Error("legacy exact crop evidence mismatch")
        exact = _validated_cached_decode(
            decision["exact_target_crop_decode"],
            target_start_ms=0,
            target_end_ms=int(crop["duration_ms"]),
            clip_duration_ms=int(crop["duration_ms"]),
        )
    audit = decision.get("contextual_boundary_audit")
    if decision.get("source") == "contextual_boundary_policy":
        if not isinstance(audit, Mapping) or audit.get("policy") != "contextual-boundary-policy-1":
            raise AudioReviewV2Error("legacy contextual decode audit mismatch")
        stages = audit.get("bounded_decodes")
        if not isinstance(stages, list) or not stages:
            raise AudioReviewV2Error("legacy contextual decode stages are missing")
        names = []
        for item in stages:
            if not isinstance(item, Mapping):
                raise AudioReviewV2Error("legacy contextual decode stage is malformed")
            stage = item.get("stage")
            transcript = item.get("transcript")
            if (
                not isinstance(stage, str)
                or not isinstance(transcript, str)
                or item.get("transcript_sha256")
                != hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                or isinstance(item.get("exact_word_overlap_ms"), bool)
                or not isinstance(item.get("exact_word_overlap_ms"), int)
                or item["exact_word_overlap_ms"] < 0
            ):
                raise AudioReviewV2Error("legacy contextual decode stage evidence mismatch")
            names.append(stage)
            expected = blind if stage == "blind_padded" else exact if stage == "exact_target_crop" else None
            if expected is not None and (
                transcript != expected["transcript"]
                or item["exact_word_overlap_ms"] != expected["exact_word_overlap_ms"]
            ):
                raise AudioReviewV2Error("legacy contextual decode stage binding mismatch")
        if len(names) != len(set(names)) or not {"blind_padded", "exact_target_crop"}.issubset(names):
            raise AudioReviewV2Error("legacy contextual decode stage coverage mismatch")
    elif audit is not None:
        raise AudioReviewV2Error("legacy contextual audit has invalid source")
    if decision.get("source") == "secondary_asr_prompted_confirmation":
        prompt = _evidence_prompt(evidence, provisional_record)
        if prompt is None or decision.get("evidence_prompt_sha256") != hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest():
            raise AudioReviewV2Error("legacy prompted decode evidence mismatch")


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
    if "exact_target_crop_decode" in decision:
        outcome["exact_target_crop_decode"] = copy.deepcopy(
            dict(decision["exact_target_crop_decode"])
        )
        outcome["exact_target_crop"] = copy.deepcopy(
            dict(decision["exact_target_crop"])
        )
    if decision.get("source") == "contextual_boundary_policy":
        outcome["forced_alignment_required"] = bool(
            decision.get("forced_alignment_required")
        )
        outcome["contextual_boundary_audit"] = copy.deepcopy(
            dict(decision["contextual_boundary_audit"])
        )
    for field in (
        "duplicate_of_utterance_uid",
        "duplicate_evidence",
        "acoustic_correction_evidence",
    ):
        if field in decision:
            outcome[field] = copy.deepcopy(decision[field])
    return outcome


def _exact_target_words(
    decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    allow_unconfirmed: bool = False,
) -> list[dict[str, Any]]:
    if decision.get("decision") != "confirmed_dialogue" and not allow_unconfirmed:
        return []
    if decision.get("source") == "secondary_asr_exact_target_crop":
        decode = decision.get("exact_target_crop_decode")
        offset_ms = int(evidence["start_ms"])
    elif decision.get("source") == "contextual_boundary_policy":
        return []
    else:
        decode = decision.get("target_decode")
        offset_ms = int(evidence["clip_start_ms"])
    if not isinstance(decode, Mapping):
        return []
    raw_words = decode.get("words")
    if not isinstance(raw_words, list):
        return []
    target_start = int(evidence["start_ms"])
    target_end = int(evidence["end_ms"])
    words = []
    for raw in raw_words:
        if not isinstance(raw, Mapping):
            continue
        start_ms = offset_ms + int(raw["start_ms"])
        end_ms = offset_ms + int(raw["end_ms"])
        overlap_ms = max(
            0,
            min(target_end, end_ms) - max(target_start, start_ms),
        )
        normalized = _normalized_text(raw.get("text", ""))
        if overlap_ms > 0 and normalized:
            words.append(
                {
                    "text": str(raw["text"]),
                    "normalized": normalized,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "target_overlap_ms": overlap_ms,
                }
            )
    return words


def _contiguous_token_match(child: Sequence[str], parent: Sequence[str]) -> int | None:
    if not child or len(child) > len(parent):
        return None
    for offset in range(len(parent) - len(child) + 1):
        if list(parent[offset : offset + len(child)]) == list(child):
            return offset
    return None


def _apply_acoustic_duplicate_policy(
    inventory: Sequence[tuple[str, str, Mapping[str, Any]]],
    provisional_by_uid: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    override_uids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    adjusted = {
        uid: copy.deepcopy(dict(decision)) for uid, decision in decisions.items()
    }
    evidence_by_uid = {
        uid: (kind, evidence) for uid, kind, evidence in inventory
    }
    words_by_uid = {
        uid: _exact_target_words(
            adjusted[uid],
            evidence,
            allow_unconfirmed=(
                "overlapping_rescue_word_evidence"
                in set(provisional_by_uid[uid].get("risk_flags", []))
            ),
        )
        for uid, _kind, evidence in inventory
    }
    resolutions = []
    for child_uid, child_kind, child_evidence in inventory:
        if child_uid in override_uids:
            continue
        child_record = provisional_by_uid[child_uid]
        child_flags = set(child_record.get("risk_flags", []))
        overlapping_rescue = "overlapping_rescue_word_evidence" in child_flags
        if (
            child_kind != "speech_hole"
            and "speech_hole_rescue_asr" not in child_flags
            and not overlapping_rescue
        ):
            continue
        child_words = words_by_uid[child_uid]
        child_tokens = [word["normalized"] for word in child_words]
        if not child_tokens or (len(child_tokens) > 3 and not overlapping_rescue):
            continue
        child_source_tokens = _tokens(child_record.get("asr_text", ""))
        if overlapping_rescue and len(child_source_tokens) < 2:
            continue
        child_start = int(child_record["coarse_start_ms"])
        child_end = int(child_record["coarse_end_ms"])
        child_duration = child_end - child_start
        matches = []
        for parent_uid, _parent_kind, parent_evidence in inventory:
            if parent_uid == child_uid or parent_uid in override_uids:
                continue
            parent_record = provisional_by_uid[parent_uid]
            parent_flags = set(parent_record.get("risk_flags", []))
            parent_start = int(parent_record["coarse_start_ms"])
            parent_end = int(parent_record["coarse_end_ms"])
            coarse_overlap_ms = max(
                0,
                min(child_end, parent_end) - max(child_start, parent_start),
            )
            if overlapping_rescue:
                if (
                    _parent_kind != "asr_caption_candidate"
                    or "speech_hole_rescue_asr" in parent_flags
                    or "overlapping_rescue_word_evidence" in parent_flags
                    or parent_end - parent_start <= child_duration
                    or coarse_overlap_ms * 2 < child_duration
                ):
                    continue
                parent_source_tokens = _tokens(parent_record.get("asr_text", ""))
                source_match_offset = _contiguous_token_match(
                    child_source_tokens, parent_source_tokens
                )
                if source_match_offset is None:
                    continue
            elif (
                parent_start > child_start
                or parent_end < child_end
                or parent_end - parent_start < child_duration * 2
            ):
                continue
            parent_words = words_by_uid[parent_uid]
            parent_tokens = [word["normalized"] for word in parent_words]
            match_offset = _contiguous_token_match(child_tokens, parent_tokens)
            match_kind = "exact_contiguous"
            similarity = 1.0
            if match_offset is None and len(child_tokens) == 1:
                scored = [
                    (
                        SequenceMatcher(None, child_tokens[0], token).ratio(),
                        index,
                    )
                    for index, token in enumerate(parent_tokens)
                ]
                if not scored:
                    continue
                similarity, match_offset = max(scored)
                if similarity < 0.65:
                    continue
                match_kind = "single_token_acoustic_fuzzy"
            if match_offset is None:
                continue
            matched_parent_words = parent_words[
                match_offset : match_offset + len(child_words)
            ]
            acoustic_overlap_ms = sum(
                max(
                    0,
                    min(child_word["end_ms"], parent_word["end_ms"])
                    - max(child_word["start_ms"], parent_word["start_ms"]),
                )
                for child_word, parent_word in zip(
                    child_words, matched_parent_words
                )
            )
            if acoustic_overlap_ms <= 0:
                continue
            if overlapping_rescue and any(
                max(
                    0,
                    min(child_word["end_ms"], parent_word["end_ms"])
                    - max(child_word["start_ms"], parent_word["start_ms"]),
                )
                * 2
                < min(
                    child_word["end_ms"] - child_word["start_ms"],
                    parent_word["end_ms"] - parent_word["start_ms"],
                )
                for child_word, parent_word in zip(
                    child_words, matched_parent_words
                )
            ):
                continue
            matches.append(
                (
                    parent_end - parent_start,
                    acoustic_overlap_ms,
                    similarity,
                    parent_uid,
                    match_kind,
                    match_offset,
                    coarse_overlap_ms,
                )
            )
        if not matches:
            continue
        (
            _parent_duration,
            acoustic_overlap_ms,
            similarity,
            parent_uid,
            match_kind,
            match_offset,
            coarse_overlap_ms,
        ) = max(matches)
        parent_kind, parent_evidence = evidence_by_uid[parent_uid]
        parent_words = words_by_uid[parent_uid]
        evidence = {
            "policy": ACOUSTIC_DUPLICATE_POLICY,
            "duplicate_of_utterance_uid": parent_uid,
            "match_kind": match_kind,
            "token_similarity": round(similarity, 6),
            "acoustic_word_overlap_ms": acoustic_overlap_ms,
            "child_audio_sha256": str(child_evidence["audio_sha256"]),
            "parent_audio_sha256": str(parent_evidence["audio_sha256"]),
            "child_target_transcript_sha256": hashlib.sha256(
                str(adjusted[child_uid]["target_decode"]["transcript"]).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "parent_target_transcript_sha256": hashlib.sha256(
                str(adjusted[parent_uid]["target_decode"]["transcript"]).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        if overlapping_rescue:
            evidence.update(
                {
                    "provenance_flag": "overlapping_rescue_word_evidence",
                    "source_match_kind": "exact_contiguous",
                    "source_matched_token_count": len(child_source_tokens),
                    "source_child_text_sha256": hashlib.sha256(
                        str(child_record["asr_text"]).encode("utf-8")
                    ).hexdigest(),
                    "source_parent_text_sha256": hashlib.sha256(
                        str(provisional_by_uid[parent_uid]["asr_text"]).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "coarse_overlap_ms": coarse_overlap_ms,
                }
            )
        child_decision = adjusted[child_uid]
        child_decision["decision"] = (
            "reviewed_non_dialogue"
            if child_kind == "speech_hole"
            else "discarded_asr_hallucination"
        )
        child_decision["tr_corrected"] = ""
        child_decision["source"] = "acoustic_duplicate_policy"
        child_decision["reason"] = (
            f"target words duplicate acoustically overlapping record {parent_uid}"
        )
        child_decision["duplicate_of_utterance_uid"] = parent_uid
        child_decision["duplicate_evidence"] = evidence
        resolutions.append(
            {
                "utterance_uid": child_uid,
                "evidence_kind": child_kind,
                **copy.deepcopy(evidence),
            }
        )

        if match_kind != "single_token_acoustic_fuzzy":
            continue
        parent_decision = adjusted[parent_uid]
        parent_target = str(
            parent_decision.get("target_decode", {}).get("transcript", "")
        ).strip()
        corrected = str(provisional_by_uid[parent_uid]["tr_corrected"]).strip()
        target_tokens = _tokens(parent_target)
        corrected_tokens = _tokens(corrected)
        matched_token = parent_words[match_offset]["normalized"]
        if (
            parent_target
            and len(corrected_tokens) >= 4
            and len(target_tokens) == len(corrected_tokens) + 1
            and matched_token not in corrected_tokens
            and _text_similarity(parent_target, corrected) >= 0.85
        ):
            parent_decision["tr_corrected"] = parent_target
            parent_decision["source"] = "acoustic_duplicate_parent_correction"
            parent_decision["reason"] = (
                f"target-only word corroborated by overlapping record {child_uid}"
            )
            parent_decision["acoustic_correction_evidence"] = {
                "policy": ACOUSTIC_DUPLICATE_POLICY,
                "corroborating_utterance_uid": child_uid,
                "inserted_token": matched_token,
                "token_similarity": round(similarity, 6),
                "acoustic_word_overlap_ms": acoustic_overlap_ms,
            }
    return adjusted, resolutions


def _validate_contextual_boundary_outcome(
    kind: str,
    evidence: Mapping[str, Any],
    provisional_record: Mapping[str, Any],
    final_record: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> None:
    if (outcome.get("decision") == "confirmed_dialogue"
            and "incomplete_provisional_word_timing" in set(evidence.get("risk_flags", []))):
        raise AudioReviewV2Error("incomplete provisional inventory cannot confirm one coarse dialogue record")
    if outcome.get("forced_alignment_required") is not True:
        raise AudioReviewV2Error(
            "contextual boundary outcome must require final forced alignment"
        )
    audit = outcome.get("contextual_boundary_audit")
    if not isinstance(audit, Mapping):
        raise AudioReviewV2Error("contextual boundary audit is missing")
    context_before = str(evidence.get("context_before", "")).strip()
    context_after = str(evidence.get("context_after", "")).strip()
    expected_context_sha256 = sha256_json(
        {"before": context_before, "after": context_after}
    )
    if (
        audit.get("policy") != CONTEXTUAL_BOUNDARY_POLICY
        or audit.get("context_before_present") is not bool(context_before)
        or audit.get("context_after_present") is not bool(context_after)
        or audit.get("context_sha256") != expected_context_sha256
        or not (context_before or context_after)
    ):
        raise AudioReviewV2Error("contextual boundary context audit mismatch")
    bounded_decodes = audit.get("bounded_decodes")
    if not isinstance(bounded_decodes, list) or not bounded_decodes:
        raise AudioReviewV2Error("contextual boundary decode audit is missing")
    exact_decode = outcome.get("exact_target_crop_decode")
    if not isinstance(exact_decode, Mapping):
        raise AudioReviewV2Error(
            "contextual boundary exact target decode is missing"
        )
    try:
        normalized_exact_words = _normalize_decoder_result(
            {"words": exact_decode.get("words"), "segments": []}
        )["words"]
    except AudioReviewV2Error as exc:
        raise AudioReviewV2Error(
            "contextual boundary exact target word evidence is invalid"
        ) from exc
    exact_transcript = " ".join(
        str(word["text"]).strip() for word in normalized_exact_words
    ).strip()
    if (
        exact_decode.get("words") != normalized_exact_words
        or exact_decode.get("transcript") != exact_transcript
        or exact_decode.get("selected_word_count") != len(normalized_exact_words)
    ):
        raise AudioReviewV2Error(
            "contextual boundary exact target word evidence mismatch"
        )
    semantic_staged_decodes = [
        (
            str(item.get("stage", "")),
            exact_decode
            if item.get("stage") == "exact_target_crop"
            else {
                "transcript": item.get("transcript", ""),
                "words": [],
            },
        )
        for item in bounded_decodes
        if isinstance(item, Mapping)
    ]
    stages = []
    for item in bounded_decodes:
        if not isinstance(item, Mapping):
            raise AudioReviewV2Error("contextual boundary decode audit is malformed")
        stage = item.get("stage")
        if not isinstance(stage, str) or not stage:
            raise AudioReviewV2Error("contextual boundary decode stage is invalid")
        stages.append(stage)
        digest = item.get("transcript_sha256")
        transcript = item.get("transcript")
        if (
            not isinstance(transcript, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest != hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        ):
            raise AudioReviewV2Error("contextual boundary transcript digest is invalid")
        if (
            not isinstance(item.get("transcript_present"), bool)
            or not isinstance(item.get("usable_target_text"), bool)
            or not isinstance(item.get("known_short_clip_hallucination"), bool)
            or not isinstance(item.get("one_millisecond_word_timing"), bool)
            or isinstance(item.get("exact_word_overlap_ms"), bool)
            or not isinstance(item.get("exact_word_overlap_ms"), int)
            or item.get("exact_word_overlap_ms") < 0
        ):
            raise AudioReviewV2Error("contextual boundary decode flags are invalid")
        context_similarity = max(
            _text_similarity(transcript, context_before),
            _text_similarity(transcript, context_after),
        )
        known_short_clip_hallucination = _is_known_short_clip_hallucination(
            transcript,
            stage=stage,
            decoded=(
                exact_decode
                if stage == "exact_target_crop"
                else {"transcript": transcript, "words": []}
            ),
            staged_decodes=semantic_staged_decodes,
        )
        one_millisecond_word_timing = (
            stage == "exact_target_crop"
            and _has_one_millisecond_word_timing(exact_decode)
        )
        if (
            item.get("transcript_present") is not bool(transcript.strip())
            or item.get("known_short_clip_hallucination")
            is not known_short_clip_hallucination
            or item.get("one_millisecond_word_timing")
            is not one_millisecond_word_timing
            or item.get("usable_target_text")
            is not (
                bool(transcript.strip())
                and not known_short_clip_hallucination
                and context_similarity < 0.85
                and (
                    stage != "prompted_padded"
                    or item.get("exact_word_overlap_ms") > 0
                )
            )
        ):
            raise AudioReviewV2Error("contextual boundary decode audit mismatch")
    if len(stages) != len(set(stages)) or not {
        "blind_padded",
        "exact_target_crop",
    }.issubset(stages):
        raise AudioReviewV2Error("contextual boundary decode coverage is incomplete")
    is_orphan = "orphan_youtube_caption" in set(evidence.get("risk_flags", []))
    if audit.get("orphan_caption") is not is_orphan:
        raise AudioReviewV2Error("contextual boundary orphan audit mismatch")
    if kind == "asr_caption_candidate":
        corrected = str(provisional_record.get("tr_corrected", "")).strip()
        known_source_hallucination = _is_known_short_clip_hallucination(
            evidence.get("asr_text", "")
        )
        repetitive_source_hallucination = _is_repetitive_asr_hallucination(
            evidence.get("asr_text", "")
        )
        usable = [
            item for item in bounded_decodes if item.get("usable_target_text") is True
        ]
        if repetitive_source_hallucination:
            if (
                is_orphan
                or outcome.get("decision") != "discarded_asr_hallucination"
                or str(final_record.get("tr_corrected", "")).strip()
            ):
                raise AudioReviewV2Error(
                    "repetitive source hallucination policy was misapplied"
                )
            return
        if known_source_hallucination:
            if usable:
                valid = (
                    outcome.get("decision") == "confirmed_dialogue"
                    and str(final_record.get("tr_corrected", "")).strip()
                    == str(usable[-1]["transcript"]).strip()
                )
            else:
                valid = (
                    outcome.get("decision") == "discarded_asr_hallucination"
                    and not str(final_record.get("tr_corrected", "")).strip()
                )
            if is_orphan or not valid:
                raise AudioReviewV2Error(
                    "known subtitle hallucination policy was misapplied"
                )
            return
        if (
            is_orphan
            or "provisional_word_gap_requires_audio_review" in set(evidence.get("risk_flags", []))
            or not corrected
            or not str(evidence.get("asr_text", "")).strip()
            or outcome.get("decision") != "confirmed_dialogue"
            or str(final_record.get("tr_corrected", "")).strip() != corrected
        ):
            raise AudioReviewV2Error("contextual ASR boundary policy was misapplied")
    else:
        if str(provisional_record.get("tr_corrected", "")).strip():
            raise AudioReviewV2Error("contextual blank-hole policy was misapplied")
        usable = [
            item for item in bounded_decodes if item.get("usable_target_text") is True
        ]
        if usable:
            if (
                outcome.get("decision") != "confirmed_dialogue"
                or str(final_record.get("tr_corrected", "")).strip()
                != str(usable[-1]["transcript"]).strip()
            ):
                raise AudioReviewV2Error("contextual speech-hole text mismatch")
        elif (
            outcome.get("decision") != "reviewed_non_dialogue"
            or str(final_record.get("tr_corrected", "")).strip()
        ):
            raise AudioReviewV2Error("contextual blank-hole policy was misapplied")


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
    provisional_by_uid = {
        str(item["utterance_uid"]): item for item in provisional.records
    }
    final_by_uid = {str(item["utterance_uid"]): item for item in final.records}
    outcome_by_uid = {
        str(item["utterance_uid"]): item
        for item in outcomes
        if isinstance(item, Mapping)
        and isinstance(item.get("utterance_uid"), str)
    }
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
        if (
            "overlapping_rescue_word_evidence"
            in set(provisional_by_uid[uid].get("risk_flags", []))
            and raw.get("source") not in {"manual", "acoustic_duplicate_policy"}
        ):
            raise AudioReviewV2Error(
                "overlapping rescue requires acoustic duplicate or manual evidence"
            )
        if raw.get("source") == "contextual_boundary_policy":
            _validate_contextual_boundary_outcome(
                kind,
                evidence,
                provisional_by_uid[uid],
                record,
                raw,
            )
    duplicate_resolutions = report.get("duplicate_resolutions")
    if not isinstance(duplicate_resolutions, list):
        raise AudioReviewV2Error("audio-review duplicate resolutions must be a list")
    if report.get("duplicate_resolution_count") != len(duplicate_resolutions):
        raise AudioReviewV2Error("audio-review duplicate resolution count mismatch")
    duplicate_uids = []
    for resolution in duplicate_resolutions:
        if not isinstance(resolution, Mapping):
            raise AudioReviewV2Error("audio-review duplicate resolution is malformed")
        uid = resolution.get("utterance_uid")
        parent_uid = resolution.get("duplicate_of_utterance_uid")
        if (
            not isinstance(uid, str)
            or not isinstance(parent_uid, str)
            or uid == parent_uid
            or resolution.get("policy") != ACOUSTIC_DUPLICATE_POLICY
            or uid not in outcome_by_uid
            or parent_uid not in outcome_by_uid
        ):
            raise AudioReviewV2Error("audio-review duplicate identity is invalid")
        raw = outcome_by_uid[uid]
        parent = outcome_by_uid[parent_uid]
        if (
            raw.get("source") != "acoustic_duplicate_policy"
            or raw.get("duplicate_of_utterance_uid") != parent_uid
            or raw.get("duplicate_evidence") != {
                key: value
                for key, value in resolution.items()
                if key not in {"utterance_uid", "evidence_kind"}
            }
            or parent.get("decision") != "confirmed_dialogue"
            or resolution.get("child_audio_sha256") != raw.get("audio_sha256")
            or resolution.get("parent_audio_sha256") != parent.get("audio_sha256")
            or not isinstance(resolution.get("acoustic_word_overlap_ms"), int)
            or resolution.get("acoustic_word_overlap_ms") <= 0
        ):
            raise AudioReviewV2Error("audio-review duplicate evidence mismatch")
        duplicate_uids.append(uid)
    expected_duplicate_uids = [
        str(item["utterance_uid"])
        for item in outcomes
        if isinstance(item, Mapping)
        and item.get("source") == "acoustic_duplicate_policy"
    ]
    if duplicate_uids != expected_duplicate_uids:
        raise AudioReviewV2Error("audio-review duplicate outcome coverage mismatch")
    overlapping_rescue_uids = {
        uid
        for uid, record in provisional_by_uid.items()
        if "overlapping_rescue_word_evidence" in set(record.get("risk_flags", []))
    }
    if overlapping_rescue_uids:
        _recomputed_decisions, recomputed_resolutions = (
            _apply_acoustic_duplicate_policy(
                inventory,
                provisional_by_uid,
                outcome_by_uid,
                override_uids={
                    uid
                    for uid, outcome in outcome_by_uid.items()
                    if outcome.get("source") == "manual"
                },
            )
        )
        expected_resolutions = [
            resolution
            for resolution in recomputed_resolutions
            if resolution["utterance_uid"] in overlapping_rescue_uids
        ]
        persisted_resolutions = [
            resolution
            for resolution in duplicate_resolutions
            if resolution.get("utterance_uid") in overlapping_rescue_uids
        ]
        if expected_resolutions != persisted_resolutions:
            raise AudioReviewV2Error(
                "overlapping rescue acoustic duplicate evidence mismatch"
            )
    for uid, raw in outcome_by_uid.items():
        if raw.get("source") != "acoustic_duplicate_parent_correction":
            continue
        evidence = raw.get("acoustic_correction_evidence")
        if not isinstance(evidence, Mapping):
            raise AudioReviewV2Error("audio-review acoustic correction evidence is missing")
        child_uid = evidence.get("corroborating_utterance_uid")
        if (
            evidence.get("policy") != ACOUSTIC_DUPLICATE_POLICY
            or not isinstance(child_uid, str)
            or child_uid not in duplicate_uids
            or outcome_by_uid[child_uid].get("duplicate_of_utterance_uid") != uid
            or str(final_by_uid[uid]["tr_corrected"]).strip()
            != str(raw.get("target_decode", {}).get("transcript", "")).strip()
        ):
            raise AudioReviewV2Error("audio-review acoustic correction mismatch")
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
    require_resume: bool = False,
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
    incomplete = [uid for uid, _kind, evidence in inventory
                  if "incomplete_provisional_word_timing" in evidence.get("risk_flags", [])]
    if incomplete:
        raise AudioReviewV2Error(
            f"Incomplete ASR hypotheses require raw coverage recovery before acoustic review: {incomplete}"
        )
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

    runtime_decoder = decoder
    owns_decoder = decoder is None
    resolved_model_dir: Path | None = None
    if decoder is not None:
        try:
            decoder_identity = copy.deepcopy(dict(decoder.provenance))
            sha256_json(decoder_identity)
        except Exception as exc:
            raise AudioReviewV2Error(
                "Injected audio-review decoder identity is unavailable"
            ) from exc
        if not decoder_identity:
            raise AudioReviewV2Error(
                "Injected audio-review decoder identity is unavailable"
            )
    elif not inventory:
        decoder_identity = {
            "engine": "not-used",
            "reason": "empty-review-inventory",
        }
    else:
        try:
            resolved_model_dir = primary_checkpoint.resolve_model(settings.model_name)
            decoder_identity = {
                "engine": "faster-whisper",
                "model": primary_checkpoint.model_identity(resolved_model_dir),
                "runtime": primary_checkpoint.producer_identity(
                    (
                        _FasterWhisperReviewDecoder._decode_once,
                        _normalize_decoder_result,
                        _finite_milliseconds,
                    )
                ),
            }
        except Exception as exc:
            raise AudioReviewV2Error(
                "Production audio-review decoder identity is unavailable"
            ) from exc

    code_path = Path(__file__)
    identity = {
        "correction_input_sha256": pack.manifest["input_sha256"],
        "provisional_output_sha256": provisional.output_sha256,
        "config": asdict(settings),
        "code_sha256": sha256_file(code_path),
        "decoder": decoder_identity,
    }
    review_input_sha256 = sha256_json(identity)
    final_output_file = Path(final_output_path)
    report_file = Path(report_path)
    recovery_file = Path(recovery_path)
    legacy_migration: dict[str, Any] | None = None
    if (
        not force
        and final_output_file.is_file()
        and not final_output_file.is_symlink()
        and report_file.is_file()
        and not report_file.is_symlink()
        and sha256_file(report_file) == _EP13_LEGACY_AUDIO_REVIEW_REPORT_SHA256
        and sha256_file(final_output_file) == _EP13_LEGACY_TR_CORRECTED_ZIP_SHA256
    ):
        required_hashes = (
            (Path(input_pack_path), _EP13_LEGACY_TR_PACK_SHA256),
            (Path(provisional_output_path), _EP13_LEGACY_PROVISIONAL_ZIP_SHA256),
            (recovery_file, _EP13_LEGACY_RECOVERY_SHA256),
        )
        if (
            pack.manifest["episode"] != 13
            or any(
                not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != expected
                for path, expected in required_hashes
            )
        ):
            raise AudioReviewV2Error(
                "Episode 13 legacy decode migration artifact binding mismatch"
            )
        try:
            legacy_report = json.loads(report_file.read_text(encoding="utf-8"))
            legacy_recovery = json.loads(
                recovery_file.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AudioReviewV2Error(
                "Episode 13 legacy decode migration evidence is unreadable"
            ) from exc
        legacy_final = validate_tr_correction_output(
            input_pack_path, final_output_file
        )
        inventory_uids = [uid for uid, _kind, _evidence in inventory]
        outcome_uids = [
            item.get("utterance_uid")
            for item in legacy_report.get("outcomes", [])
            if isinstance(item, Mapping)
        ]
        recovery_decisions = legacy_recovery.get("machine_decisions")
        if (
            legacy_report.get("audio_review_sha256") != _report_sha(legacy_report)
            or legacy_report.get("status") != "PASS"
            or legacy_report.get("pending_count") != 0
            or legacy_report.get("config") != asdict(settings)
            or legacy_report.get("correction_input_sha256")
            != pack.manifest["input_sha256"]
            or legacy_report.get("provisional_output_sha256")
            != provisional.output_sha256
            or legacy_report.get("correction_output_sha256")
            != legacy_final.output_sha256
            or legacy_report.get("manual_overrides_sha256")
            != sha256_json(overrides)
            or outcome_uids != inventory_uids
            or legacy_recovery.get("format") != AUDIO_REVIEW_V2_RECOVERY_FORMAT
            or legacy_recovery.get("input_sha256")
            != legacy_report.get("review_input_sha256")
            or legacy_recovery.get("recovery_sha256")
            != sha256_json(
                {
                    key: value
                    for key, value in legacy_recovery.items()
                    if key != "recovery_sha256"
                }
            )
            or not isinstance(recovery_decisions, Mapping)
            or set(recovery_decisions) != set(inventory_uids)
            or any(
                not isinstance(value, Mapping)
                for value in recovery_decisions.values()
            )
        ):
            raise AudioReviewV2Error(
                "Episode 13 legacy decode migration evidence mismatch"
            )
        legacy_migration = {
            "machine_decisions": {
                str(uid): copy.deepcopy(dict(value))
                for uid, value in recovery_decisions.items()
                if isinstance(uid, str) and isinstance(value, Mapping)
            },
            "decoder": copy.deepcopy(dict(legacy_report.get("decoder", {}))),
        }
    if not force and (final_output_file.exists() or final_output_file.is_symlink()):
        if legacy_migration is not None:
            pass
        else:
            if final_output_file.is_symlink() or not final_output_file.is_file():
                raise AudioReviewV2Error(
                    "Completed audio-review output path is unsafe; preserve it"
                )
            if not report_file.is_file() or report_file.is_symlink():
                raise AudioReviewV2Error(
                    "Completed audio-review output has no safe bound report; preserve it"
                )
            completed = validate_audio_review_v2_report(
                input_pack_path,
                provisional_output_path,
                final_output_file,
                report_file,
            )
            completed_input_sha256 = completed.get("review_input_sha256")
            if completed_input_sha256 != review_input_sha256:
                raise AudioReviewV2Error(
                    "Completed audio-review output belongs to different inputs, config, or code; preserve it"
                )
            if completed.get("manual_overrides_sha256") != sha256_json(overrides):
                raise AudioReviewV2Error(
                    "Completed audio-review output belongs to different manual overrides; preserve it"
                )
            return completed
    if require_resume:
        raise AudioReviewV2Error("Scoped alignment retry requires a validated completed audio-review checkpoint")
    cached: dict[str, dict[str, Any]] = (
        legacy_migration["machine_decisions"] if legacy_migration else {}
    )
    if not legacy_migration and not force and recovery_file.is_file():
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
                        if legacy_migration is not None:
                            member = str(evidence["audio_member"])
                            payload = archive.read(member)
                            if hashlib.sha256(payload).hexdigest() != evidence["audio_sha256"]:
                                raise AudioReviewV2Error(
                                    f"Review WAV hash changed before migration: {member}"
                                )
                            clip_path = temporary_root / f"{position:03d}.wav"
                            crop_path = temporary_root / f"{position:03d}-target.wav"
                            clip_path.write_bytes(payload)
                            _validate_legacy_decision_evidence(
                                cached_decision,
                                evidence,
                                provisional_by_uid[uid],
                                clip_path,
                                crop_path,
                            )
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
                        if (
                            cached_decision.get("source")
                            == "contextual_boundary_policy"
                        ):
                            cached_audit = cached_decision[
                                "contextual_boundary_audit"
                            ]
                            cached_staged_decodes = [
                                (
                                    str(item["stage"]),
                                    copy.deepcopy(
                                        cached_decision["exact_target_crop_decode"]
                                    )
                                    if item["stage"] == "exact_target_crop"
                                    else {
                                        "transcript": str(item["transcript"]),
                                        "exact_word_overlap_ms": int(
                                            item["exact_word_overlap_ms"]
                                        ),
                                        "words": [],
                                    },
                                )
                                for item in cached_audit["bounded_decodes"]
                            ]
                            recomputed = _contextual_boundary_decision(
                                kind,
                                evidence,
                                provisional_by_uid[uid],
                                recomputed,
                                cached_staged_decodes,
                            )
                            if (
                                recomputed.get("source")
                                != "contextual_boundary_policy"
                            ):
                                raise AudioReviewV2Error(
                                    "cached contextual decision no longer satisfies policy"
                                )
                            recomputed["blind_target_decode"] = copy.deepcopy(
                                cached_decision["blind_target_decode"]
                            )
                            recomputed["exact_target_crop_decode"] = copy.deepcopy(
                                cached_decision["exact_target_crop_decode"]
                            )
                            recomputed["exact_target_crop"] = copy.deepcopy(
                                cached_decision["exact_target_crop"]
                            )
                        elif "exact_target_crop_decode" in cached_decision:
                            recomputed = _machine_decision(
                                kind,
                                evidence,
                                provisional_by_uid[uid],
                                cached_decision["exact_target_crop_decode"],
                                settings,
                            )
                            if not _exact_target_crop_confirms(
                                kind,
                                evidence,
                                provisional_by_uid[uid],
                                recomputed,
                            ):
                                raise AudioReviewV2Error(
                                    "cached exact target crop no longer satisfies policy"
                                )
                            recomputed["source"] = "secondary_asr_exact_target_crop"
                            recomputed["reason"] = (
                                "blind secondary ASR on the exact target-only crop "
                                "satisfied the existing acoustic decision policy"
                            )
                            recomputed["blind_target_decode"] = copy.deepcopy(
                                cached_decision["blind_target_decode"]
                            )
                            recomputed["exact_target_crop_decode"] = copy.deepcopy(
                                cached_decision["exact_target_crop_decode"]
                            )
                            recomputed["exact_target_crop"] = copy.deepcopy(
                                cached_decision["exact_target_crop"]
                            )
                        elif "blind_target_decode" in cached_decision:
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
                    except (KeyError, TypeError, ValueError, AudioReviewV2Error) as exc:
                        if legacy_migration is not None:
                            raise AudioReviewV2Error(
                                f"Episode 13 legacy decode evidence is invalid: {uid}"
                            ) from exc
                        recomputed = None
                    if recomputed is not None and (
                        legacy_migration is not None
                        or sha256_json(recomputed) == sha256_json(cached_decision)
                    ):
                        machine_decisions[uid] = (
                            recomputed
                            if legacy_migration is not None
                            else cached_decision
                        )
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
                    runtime_decoder = _FasterWhisperReviewDecoder(
                        settings,
                        model_dir=resolved_model_dir,
                        cache_identity=decoder_identity,
                    )
                decoded = _normalize_decoder_result(
                    runtime_decoder.decode(clip_path, initial_prompt=None)
                )
                target = _target_decode(
                    decoded,
                    evidence,
                    tolerance_ms=settings.target_tolerance_ms,
                )
                staged_decodes = [("blind_padded", target)]
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
                        staged_decodes.append(("prompted_padded", prompted_target))
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
                if decision["decision"] == "pending_audio_review":
                    crop_path = temporary_root / f"{position:03d}-target.wav"
                    crop = _write_exact_target_crop(clip_path, crop_path, evidence)
                    crop_decoded = _normalize_decoder_result(
                        runtime_decoder.decode(crop_path, initial_prompt=None)
                    )
                    crop_target = _whole_clip_target_decode(
                        crop_decoded, duration_ms=int(crop["duration_ms"])
                    )
                    staged_decodes.append(("exact_target_crop", crop_target))
                    crop_decision = _machine_decision(
                        kind,
                        evidence,
                        provisional_by_uid[uid],
                        crop_target,
                        settings,
                    )
                    if _exact_target_crop_confirms(
                        kind,
                        evidence,
                        provisional_by_uid[uid],
                        crop_decision,
                    ):
                        crop_decision["source"] = "secondary_asr_exact_target_crop"
                        crop_decision["reason"] = (
                            "blind secondary ASR on the exact target-only crop "
                            "satisfied the existing acoustic decision policy"
                        )
                        crop_decision["blind_target_decode"] = target
                        crop_decision["exact_target_crop_decode"] = crop_target
                        crop_decision["exact_target_crop"] = crop
                        decision = crop_decision
                    if decision["decision"] == "pending_audio_review":
                        contextual_decision = _contextual_boundary_decision(
                            kind,
                            evidence,
                            provisional_by_uid[uid],
                            decision,
                            staged_decodes,
                        )
                        if (
                            contextual_decision.get("source")
                            == "contextual_boundary_policy"
                        ):
                            contextual_decision["blind_target_decode"] = target
                            contextual_decision["exact_target_crop_decode"] = crop_target
                            contextual_decision["exact_target_crop"] = crop
                        decision = contextual_decision
                machine_decisions[uid] = decision
                mark_work_progress("audio_review", completed=position)
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

    if decoder is not None:
        try:
            current_decoder_identity = copy.deepcopy(dict(decoder.provenance))
        except Exception as exc:
            raise AudioReviewV2Error(
                "Injected audio-review decoder identity is unavailable after review"
            ) from exc
        if current_decoder_identity != decoder_identity:
            raise AudioReviewV2Error(
                "Injected audio-review decoder identity changed during review"
            )
    elif resolved_model_dir is not None:
        try:
            current_model_identity = primary_checkpoint.model_identity(
                resolved_model_dir
            )
        except Exception as exc:
            raise AudioReviewV2Error(
                "Production audio-review model identity is unavailable after review"
            ) from exc
        if current_model_identity != decoder_identity["model"]:
            raise AudioReviewV2Error(
                "Production audio-review model changed during review"
            )

    if legacy_migration is not None:
        save_recovery()

    final_decisions = {
        uid: copy.deepcopy(decision) for uid, decision in machine_decisions.items()
    }
    for uid, kind, _evidence in inventory:
        if uid in overrides:
            final_decisions[uid] = _manual_decision(
                uid, kind, overrides[uid], provisional_by_uid[uid]
            )
    final_decisions, duplicate_resolutions = _apply_acoustic_duplicate_policy(
        inventory,
        provisional_by_uid,
        final_decisions,
        override_uids=set(overrides),
    )
    outcomes: list[dict[str, Any]] = []
    resolved_records = [copy.deepcopy(dict(record)) for record in provisional.records]
    resolved_by_uid = {
        str(record["utterance_uid"]): record for record in resolved_records
    }
    pending_uids: list[str] = []
    for uid, kind, evidence in inventory:
        decision = final_decisions[uid]
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
        else copy.deepcopy(legacy_migration["decoder"])
        if legacy_migration is not None
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
        "decoder_identity": decoder_identity,
        "manual_overrides_sha256": sha256_json(overrides),
        "config": asdict(settings),
        "decoder": decoder_provenance,
        "review_count": len(outcomes),
        "resolved_count": len(outcomes) - len(pending_uids),
        "pending_count": len(pending_uids),
        "pending_utterance_uids": pending_uids,
        "duplicate_resolution_count": len(duplicate_resolutions),
        "duplicate_resolutions": duplicate_resolutions,
        "outcomes": outcomes,
    }
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
