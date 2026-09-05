"""High-quality Turkish ASR with targeted, evidence-preserving verification.

One full ``large-v3`` transcription is performed with word timestamps and VAD.
Only ranked suspicious spans are decoded again.  The second pass is deliberately
bounded by both span count and total duration; this module never performs blind
full-episode retranscriptions.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .download import (
    atomic_write_json,
    load_valid_stage_marker,
    sha256_file,
    sha256_json,
    strip_vtt_markup,
    utc_now_iso,
    write_stage_marker,
)

LOGGER = logging.getLogger(__name__)
TRANSCRIPTION_FORMAT_VERSION = "1.0"


class TranscriptionError(RuntimeError):
    """Raised when ASR output cannot be produced or structurally validated."""


@dataclass(frozen=True)
class TranscriptionConfig:
    """Conservative defaults for a roughly two-hour Turkish episode."""

    model_name: str = "large-v3"
    language: str = "tr"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: float = 0.0
    condition_on_previous_text: bool = True
    compute_type_gpu: str = "float16"
    compute_type_cpu: str = "int8"
    cpu_threads: int = 0
    num_workers: int = 1
    allow_cpu_fallback: bool = True
    vad_threshold: float = 0.50
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    suspicious_word_probability: float = 0.55
    suspicious_segment_avg_logprob: float = -0.90
    suspicious_no_speech_probability: float = 0.45
    suspicious_caption_similarity: float = 0.28
    suspicious_gap_ms: int = 1000
    targeted_verification: bool = True
    verification_beam_size: int = 8
    verification_padding_ms: int = 800
    verification_max_spans: int = 24
    verification_max_total_seconds: int = 720
    verification_merge_gap_ms: int = 1200
    verification_max_span_ms: int = 45000
    extract_explicit_vad_regions: bool = True

    def __post_init__(self) -> None:
        if self.language != "tr":
            raise ValueError("This workflow requires Turkish ASR (language='tr')")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if self.beam_size < 1 or self.best_of < 1 or self.verification_beam_size < 1:
            raise ValueError("beam sizes and best_of must be positive")
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be between 0 and 1")
        if not 0.0 <= self.suspicious_word_probability <= 1.0:
            raise ValueError("suspicious_word_probability must be between 0 and 1")
        if self.verification_max_spans < 0 or self.verification_max_total_seconds < 0:
            raise ValueError("verification limits cannot be negative")

    @property
    def vad_parameters(self) -> dict[str, Any]:
        return {
            "threshold": self.vad_threshold,
            "min_speech_duration_ms": self.vad_min_speech_ms,
            "min_silence_duration_ms": self.vad_min_silence_ms,
            "speech_pad_ms": self.vad_speech_pad_ms,
        }

    def marker_settings(self) -> dict[str, Any]:
        """Return every option that can change persisted ASR output."""

        return asdict(self)


@dataclass(frozen=True)
class TranscriptionResult:
    """Verified persistent output from :func:`transcribe_audio`."""

    output_path: Path
    marker_path: Path
    data: dict[str, Any]
    resumed: bool = False


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_whisper() -> tuple[Any, Any]:
    try:
        import ctranslate2  # type: ignore
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise TranscriptionError(
            "faster-whisper/ctranslate2 is missing. Run the PREPARE dependency cell first."
        ) from exc
    return WhisperModel, ctranslate2


def _select_device(config: TranscriptionConfig, ctranslate2: Any) -> tuple[str, str]:
    cuda_count = 0
    try:
        cuda_count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        cuda_count = 0
    if cuda_count > 0:
        return "cuda", config.compute_type_gpu
    if not config.allow_cpu_fallback:
        raise TranscriptionError(
            "No compatible GPU is available. Enable a Colab GPU runtime or set "
            "allow_cpu_fallback=True."
        )
    LOGGER.warning(
        "No compatible GPU detected. Falling back to CPU/%s; large-v3 may take several hours.",
        config.compute_type_cpu,
    )
    return "cpu", config.compute_type_cpu


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    """Recognize CUDA-library/device failures without masking download errors."""

    message = f"{type(exc).__name__}: {exc}".casefold()
    cuda_indicators = (
        "cuda",
        "cudnn",
        "cublas",
        "libcudnn",
        "libcublas",
        "nvidia driver",
        "compute capability",
        "gpu device",
    )
    clearly_unrelated = (
        "http error",
        "connectionerror",
        "connection error",
        "name resolution",
        "not found in the cached files",
        "repository not found",
        "unauthorized",
        "forbidden",
    )
    return any(token in message for token in cuda_indicators) and not any(
        token in message for token in clearly_unrelated
    )


_TIMING_LINE = re.compile(
    r"^(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})"
)
_INLINE_VTT_TIMESTAMP = re.compile(
    r"<(?P<time>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})>"
)
_ROLLING_CAPTION_FINAL_WORD_TAIL_MS = 500


def _timestamp_to_ms(value: str) -> int:
    pieces = value.replace(",", ".").split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        raise ValueError(f"Unsupported caption timestamp: {value}")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def load_vtt_captions(path: str | Path | None) -> list[dict[str, Any]]:
    """Parse YouTube WebVTT into UTF-8 timed evidence records.

    YouTube rolling captions often repeat the same text. Exact duplicate cues
    with identical timing and text are removed, but no linguistic content is
    otherwise rewritten.
    """

    if path is None:
        return []
    caption_path = Path(path)
    if not caption_path.is_file():
        raise TranscriptionError(f"Caption file does not exist: {caption_path}")
    try:
        lines = caption_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TranscriptionError(f"Cannot read caption file {caption_path}: {exc}") from exc

    raw_cues: list[tuple[int, int, list[str]]] = []
    line_number = 0
    while line_number < len(lines):
        match = _TIMING_LINE.match(lines[line_number].strip())
        if not match:
            line_number += 1
            continue
        start_ms = _timestamp_to_ms(match.group("start"))
        end_ms = _timestamp_to_ms(match.group("end"))
        line_number += 1
        text_lines: list[str] = []
        while line_number < len(lines) and lines[line_number].strip():
            text_lines.append(lines[line_number])
            line_number += 1
        raw_cues.append((start_ms, end_ms, text_lines))

    # YouTube automatic VTT is a rolling display format, not a sequence of
    # speech-bounded captions. A cue commonly repeats the preceding line and
    # then remains open through many seconds of silence. Inline word timestamps
    # identify this format. Keep only the newest visible line, discard 10 ms
    # snapshot cues, and cap the post-word display tail before the captions are
    # used as timing evidence.
    rolling = any(
        _INLINE_VTT_TIMESTAMP.search(line)
        for _start_ms, _end_ms, text_lines in raw_cues
        for line in text_lines
    )

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for start_ms, original_end_ms, text_lines in raw_cues:
        end_ms = original_end_ms
        if rolling:
            visible_lines = [
                line for line in text_lines if strip_vtt_markup(line).strip()
            ]
            if not visible_lines:
                continue
            current_line = visible_lines[-1]
            inline_times = [
                _timestamp_to_ms(match.group("time"))
                for match in _INLINE_VTT_TIMESTAMP.finditer(current_line)
            ]
            if not inline_times and original_end_ms - start_ms <= 50:
                continue
            text = strip_vtt_markup(current_line)
            if inline_times:
                end_ms = min(
                    original_end_ms,
                    max(
                        start_ms + 1,
                        inline_times[-1]
                        + _ROLLING_CAPTION_FINAL_WORD_TAIL_MS,
                    ),
                )
            else:
                end_ms = min(
                    original_end_ms,
                    start_ms + _ROLLING_CAPTION_FINAL_WORD_TAIL_MS,
                )
        else:
            text = strip_vtt_markup(" ".join(text_lines))
        key = (start_ms, end_ms, text)
        if text and end_ms > start_ms and key not in seen:
            seen.add(key)
            records.append(
                {
                    "caption_index": len(records) + 1,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                }
            )
    return records


def _normalise_for_compare(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\wçğıöşü]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _word_text_for_coverage(words: Sequence[Mapping[str, Any]]) -> str:
    """Reconstruct decoded word text without depending on tokenizer spacing.

    faster-whisper normally includes leading whitespace in word tokens, while
    synthetic/test objects and some tokenizer edge cases do not.  Preserve the
    native concatenation when those leading spaces are present; otherwise join
    complete word records with one space.
    """

    pieces = [
        str(word.get("text", ""))
        for word in words
        if str(word.get("text", ""))
    ]
    if not pieces:
        return ""
    if any(piece[:1].isspace() for piece in pieces[1:]):
        return "".join(pieces).strip()
    return " ".join(piece.strip() for piece in pieces if piece.strip())


def _asr_text_coverage_matches(segment_text: str, words: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether timed words cover the same normalized decoded content."""

    segment_normalized = _normalise_for_compare(segment_text)
    words_normalized = _normalise_for_compare(_word_text_for_coverage(words))
    if segment_normalized == words_normalized:
        return True
    # Tokenizers can expose a compound as adjacent word records even though the
    # segment renderer removes that boundary.  Equal ordered characters still
    # prove full coverage; missing, additional or reordered content does not.
    return segment_normalized.replace(" ", "") == words_normalized.replace(" ", "")


def _overlapping_caption_text(
    start_ms: int, end_ms: int, captions: Sequence[Mapping[str, Any]]
) -> str:
    pieces: list[str] = []
    for caption in captions:
        caption_start = int(caption.get("start_ms", -1))
        caption_end = int(caption.get("end_ms", -1))
        if min(end_ms, caption_end) - max(start_ms, caption_start) > 0:
            text = str(caption.get("text", "")).strip()
            if text and (not pieces or text != pieces[-1]):
                pieces.append(text)
    # Rolling auto-captions can repeat a prior phrase. Exact sequence
    # de-duplication prevents that repetition from dominating similarity.
    compact: list[str] = []
    for piece in pieces:
        if compact and piece.startswith(compact[-1]):
            compact[-1] = piece
        elif not compact or piece not in compact[-1]:
            compact.append(piece)
    return " ".join(compact).strip()


def _has_strange_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    if not tokens:
        return True
    letters = sum(character.isalpha() for character in normalized)
    visible = sum(not character.isspace() for character in normalized)
    if visible and letters / visible < 0.45 and "[Müzik]" not in normalized:
        return True
    # Common Whisper loop failure: one token repeated four or more times.
    lowered = [_normalise_for_compare(token) for token in tokens]
    return any(
        lowered[index]
        and len(set(lowered[index : index + 4])) == 1
        for index in range(max(0, len(lowered) - 3))
    )


def _contains_sensitive_term(
    text: str,
    canonical_names: Sequence[str],
    religious_terms: Sequence[str],
) -> bool:
    normal = f" {_normalise_for_compare(text)} "
    terms = [*canonical_names, *religious_terms]
    return any(
        term and f" {_normalise_for_compare(term)} " in normal
        for term in terms
    )


def _possible_speaker_change(text: str) -> bool:
    return bool(
        re.search(r"(?:^|\n)\s*[-–—]\s*\S", text)
        or len(re.findall(r"\s[-–—]\s", text)) >= 1
    )


def _span_priority(reasons: Sequence[str]) -> int:
    weights = {
        "very_low_word_confidence": 100,
        "low_segment_logprob": 90,
        "youtube_caption_conflict": 80,
        "strange_turkish": 75,
        "large_internal_gap": 70,
        "music_speech_uncertainty": 65,
        "possible_speaker_change": 60,
        "name_or_religious_expression": 55,
        "low_word_confidence": 50,
        "high_no_speech_probability": 45,
    }
    return max((weights.get(reason, 10) for reason in reasons), default=0)


def find_suspicious_spans(
    transcription: Mapping[str, Any],
    captions: Sequence[Mapping[str, Any]] = (),
    canonical_names: Sequence[str] = (),
    religious_terms: Sequence[str] = (),
    config: TranscriptionConfig | None = None,
) -> list[dict[str, Any]]:
    """Rank suspicious main-pass spans for optional targeted verification."""

    settings = config or TranscriptionConfig()
    spans: list[dict[str, Any]] = []
    for segment in transcription.get("segments") or []:
        if not isinstance(segment, Mapping):
            continue
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        text = str(segment.get("text", "")).strip()
        words = [word for word in segment.get("words") or [] if isinstance(word, Mapping)]
        probabilities = [
            float(word["probability"])
            for word in words
            if word.get("probability") is not None
        ]
        reasons: list[str] = []
        if probabilities and min(probabilities) < settings.suspicious_word_probability:
            reasons.append("low_word_confidence")
        if probabilities and min(probabilities) < max(0.25, settings.suspicious_word_probability - 0.25):
            reasons.append("very_low_word_confidence")
        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is not None and float(avg_logprob) < settings.suspicious_segment_avg_logprob:
            reasons.append("low_segment_logprob")
        no_speech = segment.get("no_speech_probability")
        if text and no_speech is not None and float(no_speech) > settings.suspicious_no_speech_probability:
            reasons.append("high_no_speech_probability")
        gaps = [
            int(next_word["start_ms"]) - int(word["end_ms"])
            for word, next_word in zip(words, words[1:])
        ]
        if any(gap > settings.suspicious_gap_ms for gap in gaps):
            reasons.append("large_internal_gap")
        if _has_strange_text(text):
            reasons.append("strange_turkish")
        if _possible_speaker_change(text):
            reasons.append("possible_speaker_change")
        if re.search(r"\[(?:müzik|music|şarkı|alkış)", text, flags=re.IGNORECASE):
            reasons.append("music_speech_uncertainty")
        if _contains_sensitive_term(text, canonical_names, religious_terms):
            reasons.append("name_or_religious_expression")

        youtube_text = _overlapping_caption_text(start_ms, end_ms, captions)
        caption_similarity: float | None = None
        if youtube_text and text:
            caption_similarity = SequenceMatcher(
                None,
                _normalise_for_compare(text),
                _normalise_for_compare(youtube_text),
            ).ratio()
            if caption_similarity < settings.suspicious_caption_similarity:
                reasons.append("youtube_caption_conflict")

        if reasons:
            spans.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "source_segment_ids": [int(segment["segment_id"])],
                    "primary_text": text,
                    "youtube_text": youtube_text,
                    "caption_similarity": caption_similarity,
                    "reasons": sorted(set(reasons)),
                    "priority": _span_priority(reasons),
                }
            )
    return _merge_suspicious_spans(spans, settings)


def _merge_suspicious_spans(
    spans: Sequence[Mapping[str, Any]], config: TranscriptionConfig
) -> list[dict[str, Any]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (int(span["start_ms"]), int(span["end_ms"])))
    merged: list[dict[str, Any]] = []
    for source in ordered:
        current = dict(source)
        if (
            merged
            and int(current["start_ms"]) - int(merged[-1]["end_ms"])
            <= config.verification_merge_gap_ms
            and max(int(current["end_ms"]), int(merged[-1]["end_ms"]))
            - min(int(current["start_ms"]), int(merged[-1]["start_ms"]))
            <= config.verification_max_span_ms
        ):
            target = merged[-1]
            target["start_ms"] = min(int(target["start_ms"]), int(current["start_ms"]))
            target["end_ms"] = max(int(target["end_ms"]), int(current["end_ms"]))
            target["source_segment_ids"] = sorted(
                set(target.get("source_segment_ids", []))
                | set(current.get("source_segment_ids", []))
            )
            target["reasons"] = sorted(
                set(target.get("reasons", [])) | set(current.get("reasons", []))
            )
            target["priority"] = max(int(target.get("priority", 0)), int(current.get("priority", 0)))
            target["primary_text"] = " ".join(
                part for part in (target.get("primary_text", ""), current.get("primary_text", "")) if part
            )
            target["youtube_text"] = " ".join(
                part for part in (target.get("youtube_text", ""), current.get("youtube_text", "")) if part
            )
            target["caption_similarity"] = None
        else:
            merged.append(current)
    for index, span in enumerate(merged, start=1):
        span["suspicious_span_index"] = index
    return merged


def _select_verification_plan(
    spans: Sequence[Mapping[str, Any]],
    duration_ms: int,
    config: TranscriptionConfig,
) -> list[dict[str, Any]]:
    if not config.targeted_verification:
        return []
    ranked = sorted(
        spans,
        key=lambda span: (
            -int(span.get("priority", 0)),
            int(span["start_ms"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    total_ms = 0
    maximum_total_ms = config.verification_max_total_seconds * 1000
    for span in ranked:
        start = max(0, int(span["start_ms"]) - config.verification_padding_ms)
        end = min(duration_ms, int(span["end_ms"]) + config.verification_padding_ms)
        if end <= start:
            continue
        span_duration = end - start
        if len(selected) >= config.verification_max_spans:
            break
        if total_ms + span_duration > maximum_total_ms:
            continue
        plan = dict(span)
        plan["clip_start_ms"] = start
        plan["clip_end_ms"] = end
        plan["clip_duration_ms"] = span_duration
        selected.append(plan)
        total_ms += span_duration
    selected.sort(key=lambda span: int(span["clip_start_ms"]))
    for index, span in enumerate(selected, start=1):
        span["verification_index"] = index
    return selected


def _word_record(word: Any, segment_id: int, word_index: int, offset_ms: int = 0) -> dict[str, Any] | None:
    start = getattr(word, "start", None)
    end = getattr(word, "end", None)
    if start is None or end is None:
        return None
    start_ms = round(float(start) * 1000) + offset_ms
    end_ms = round(float(end) * 1000) + offset_ms
    if end_ms <= start_ms:
        return None
    probability = getattr(word, "probability", None)
    return {
        "word_index": word_index,
        "segment_id": segment_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": str(getattr(word, "word", "")),
        "probability": float(probability) if probability is not None else None,
    }


def _compact_word_token(text: str) -> str:
    """Normalize one rendered/timed word without depending on punctuation."""

    return _normalise_for_compare(text).replace(" ", "")


def _synthetic_words(
    tokens: Sequence[str],
    *,
    segment_id: int,
    start_ms: int,
    end_ms: int,
    leading_space: bool,
) -> list[dict[str, Any]]:
    """Give omitted edge tokens bounded, explicit synthetic timings.

    The interval normally comes from the gap between the segment boundary and
    the first/last timed word.  When faster-whisper reports no such gap, the
    caller supplies a small overlapping slice of the adjacent real word.  The
    zero probability makes the repaired segment eligible for targeted
    verification instead of presenting the estimate as model confidence.
    """

    if not tokens:
        return []
    interval_start = max(0, int(start_ms))
    interval_end = max(interval_start + 1, int(end_ms))
    duration = interval_end - interval_start
    count = len(tokens)
    records: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        token_start = interval_start + (duration * index) // count
        token_end = interval_start + (duration * (index + 1)) // count
        if token_end <= token_start:
            token_end = token_start + 1
        text = token.strip()
        if leading_space or index > 0:
            text = " " + text
        records.append(
            {
                "word_index": 0,
                "segment_id": segment_id,
                "start_ms": token_start,
                "end_ms": token_end,
                "text": text,
                "probability": 0.0,
                "timing_source": "synthetic_segment_word_repair",
            }
        )
    return records


def _split_combined_timed_word(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split a rare multi-word Whisper record inside its reported time span."""

    original_text = str(record.get("text", ""))
    tokens = [token for token in re.findall(r"\S+", original_text) if _compact_word_token(token)]
    if len(tokens) <= 1:
        return [dict(record)]
    start_ms = int(record["start_ms"])
    end_ms = int(record["end_ms"])
    duration = max(1, end_ms - start_ms)
    split: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        token_start = start_ms + (duration * index) // len(tokens)
        token_end = start_ms + (duration * (index + 1)) // len(tokens)
        if token_end <= token_start:
            token_end = token_start + 1
        child = dict(record)
        child.update(
            {
                "start_ms": token_start,
                "end_ms": token_end,
                "text": (
                    (" " if original_text[:1].isspace() or index > 0 else "")
                    + token.strip()
                ),
                "probability": 0.0,
                "timing_source": "split_combined_whisper_word",
            }
        )
        split.append(child)
    return split


def _repair_dotted_initialism_word_surfaces(
    segment_text: str, words: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]] | None:
    """Repair one missing leading letter in a dotted initialism token.

    faster-whisper can rarely render a segment such as ``M.K.`` while exposing
    the corresponding timed word as ``.K.``.  The timing still belongs to the
    whole decoded token, so preserve that interval and restore only the exact
    rendered surface.  Keep this deliberately narrow: token counts must match,
    every other token must already match, and the malformed timed token must be
    the rendered dotted initialism with exactly its first letter missing.
    """

    segment_tokens = re.findall(r"\S+", segment_text)
    if len(segment_tokens) != len(words) or not segment_tokens:
        return None

    repaired: list[dict[str, Any]] = []
    repair_count = 0
    letter_pattern = r"[^\W\d_]"
    rendered_pattern = re.compile(rf"(?:{letter_pattern}\.){{2,}}", re.UNICODE)
    missing_leading_pattern = re.compile(
        rf"\.(?:{letter_pattern}\.)+", re.UNICODE
    )

    for segment_token, word in zip(segment_tokens, words):
        record = dict(word)
        original_text = str(record.get("text", ""))
        timed_token = original_text.strip()
        if _compact_word_token(segment_token) == _compact_word_token(timed_token):
            repaired.append(record)
            continue

        if repair_count:
            return None
        rendered_surface = unicodedata.normalize("NFKC", segment_token)
        timed_surface = unicodedata.normalize("NFKC", timed_token)
        if not rendered_pattern.fullmatch(
            rendered_surface
        ) or not missing_leading_pattern.fullmatch(timed_surface):
            return None
        rendered_letters = "".join(
            re.findall(letter_pattern, rendered_surface)
        ).casefold()
        timed_letters = "".join(
            re.findall(letter_pattern, timed_surface)
        ).casefold()
        if (
            len(rendered_letters) != len(timed_letters) + 1
            or rendered_letters[1:] != timed_letters
        ):
            return None

        leading_space = original_text[
            : len(original_text) - len(original_text.lstrip())
        ]
        record.update(
            {
                "text": leading_space + segment_token,
                "probability": 0.0,
                "timing_source": "dotted_initialism_surface_repair",
            }
        )
        repaired.append(record)
        repair_count += 1

    if repair_count == 0:
        return None
    return repaired if _asr_text_coverage_matches(segment_text, repaired) else None


def _repair_segment_word_omissions(
    segment_text: str,
    words: Sequence[Mapping[str, Any]],
    *,
    segment_id: int,
    segment_start_ms: int,
    segment_end_ms: int,
) -> list[dict[str, Any]] | None:
    """Repair only provable omissions from an ordered timed-word subsequence.

    A repair is allowed when every existing timed-word token appears in the
    rendered segment in the same order.  Missing leading, internal or trailing
    tokens receive explicit synthetic timings.  Replacements, additions and
    reordered timed text still hard fail.  Repeated tokens are aligned to the
    earliest valid occurrence; their estimated timing remains visibly marked
    and zero-confidence for targeted verification.
    """

    segment_tokens = [
        token for token in re.findall(r"\S+", segment_text) if _compact_word_token(token)
    ]
    timed_tokens = [str(word.get("text", "")).strip() for word in words]
    if not segment_tokens or not timed_tokens or len(timed_tokens) >= len(segment_tokens):
        return None
    segment_keys = [_compact_word_token(token) for token in segment_tokens]
    timed_keys = [_compact_word_token(token) for token in timed_tokens]
    if any(not key for key in timed_keys):
        return None
    def remaining_fits(start: int, remaining: Sequence[str]) -> bool:
        cursor = start
        for key in remaining:
            try:
                cursor = segment_keys.index(key, cursor) + 1
            except ValueError:
                return False
        return True

    matched_positions: list[int] = []
    search_from = 0
    for timed_index, timed_key in enumerate(timed_keys):
        candidates = [
            position
            for position in range(search_from, len(segment_keys))
            if segment_keys[position] == timed_key
            and remaining_fits(position + 1, timed_keys[timed_index + 1 :])
        ]
        if not candidates:
            return None
        timed_surface = unicodedata.normalize("NFKC", timed_tokens[timed_index]).casefold()
        position = min(
            candidates,
            key=lambda candidate: (
                unicodedata.normalize(
                    "NFKC", segment_tokens[candidate]
                ).casefold()
                != timed_surface,
                candidate,
            ),
        )
        matched_positions.append(position)
        search_from = position + 1

    timed_records = [dict(word) for word in words]
    repaired: list[dict[str, Any]] = []
    previous_position = -1
    previous_word: Mapping[str, Any] | None = None
    for position, timed_word in zip(matched_positions, timed_records):
        omitted = segment_tokens[previous_position + 1 : position]
        if omitted:
            if previous_word is None:
                interval_start = min(segment_start_ms, int(timed_word["start_ms"]))
                interval_end = int(timed_word["start_ms"])
                if interval_end <= interval_start:
                    interval_start = int(timed_word["start_ms"])
                    # No leading gap exists.  Put every synthetic prefix token
                    # on the same one-millisecond boundary as the first real
                    # word.  Equal starts are valid and preserve textual order
                    # without falsifying or shifting the real word timing.
                    interval_end = interval_start + 1
            else:
                interval_start = int(previous_word["end_ms"])
                interval_end = int(timed_word["start_ms"])
                if interval_end <= interval_start:
                    interval_start = int(timed_word["start_ms"])
                    # Overlapping adjacent real words leave no internal gap.
                    # Share the next word's boundary rather than sorting the
                    # synthetic text to the wrong side of that real word.
                    interval_end = interval_start + 1
            repaired.extend(
                _synthetic_words(
                    omitted,
                    segment_id=segment_id,
                    start_ms=interval_start,
                    end_ms=interval_end,
                    leading_space=previous_word is not None,
                )
            )
        repaired.append(timed_word)
        previous_word = timed_word
        previous_position = position

    suffix = segment_tokens[previous_position + 1 :]
    if suffix and previous_word is not None:
        interval_start = int(previous_word["end_ms"])
        interval_end = max(segment_end_ms, interval_start)
        if interval_end <= interval_start:
            # No trailing gap exists.  Reuse the final millisecond of the last
            # real word so appended synthetic tokens remain temporally ordered.
            interval_end = int(previous_word["end_ms"])
            interval_start = max(int(previous_word["start_ms"]), interval_end - 1)
        repaired.extend(
            _synthetic_words(
                suffix,
                segment_id=segment_id,
                start_ms=interval_start,
                end_ms=interval_end,
                leading_space=True,
            )
        )
    # Records are deliberately assembled in decoded text order.  Sorting equal
    # or overlapping fallback timings can move a real word between synthetic
    # tokens and silently recreate the coverage failure we are repairing.
    if any(
        int(current["start_ms"]) < int(previous["start_ms"])
        for previous, current in zip(repaired, repaired[1:])
    ):
        return None
    return repaired if _asr_text_coverage_matches(segment_text, repaired) else None


def _consume_segments(
    segment_iterator: Iterable[Any], *, offset_ms: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []
    word_index = 1
    for ordinal, segment in enumerate(segment_iterator, start=1):
        segment_id = int(getattr(segment, "id", ordinal))
        segment_text = str(getattr(segment, "text", "")).strip()
        start_ms = round(float(getattr(segment, "start", 0.0)) * 1000) + offset_ms
        end_ms = round(float(getattr(segment, "end", 0.0)) * 1000) + offset_ms
        words: list[dict[str, Any]] = []
        for whisper_word in getattr(segment, "words", None) or []:
            record = _word_record(whisper_word, segment_id, 0, offset_ms)
            if record is not None and record["text"].strip():
                words.extend(_split_combined_timed_word(record))
        if segment_text and not words:
            raise TranscriptionError(
                f"ASR segment {segment_id} contains text but has no valid timed words: "
                f"{segment_text[:160]!r}"
            )
        if segment_text and not _asr_text_coverage_matches(segment_text, words):
            repaired_surfaces = _repair_dotted_initialism_word_surfaces(
                segment_text, words
            )
            if repaired_surfaces is not None:
                LOGGER.warning(
                    "Repaired faster-whisper segment %s with %d dotted-initialism "
                    "surface correction(s)",
                    segment_id,
                    sum(
                        word.get("timing_source")
                        == "dotted_initialism_surface_repair"
                        for word in repaired_surfaces
                    ),
                )
                words = repaired_surfaces
        if segment_text and not _asr_text_coverage_matches(segment_text, words):
            repaired = _repair_segment_word_omissions(
                segment_text,
                words,
                segment_id=segment_id,
                segment_start_ms=start_ms,
                segment_end_ms=end_ms,
            )
            if repaired is not None:
                LOGGER.warning(
                    "Repaired faster-whisper segment %s with %d synthetic word(s)",
                    segment_id,
                    sum(word.get("timing_source") is not None for word in repaired),
                )
                words = repaired
        if segment_text and not _asr_text_coverage_matches(segment_text, words):
            word_text = _word_text_for_coverage(words)
            raise TranscriptionError(
                f"ASR segment {segment_id} text is not fully covered by its timed words: "
                f"segment={segment_text[:160]!r}, words={word_text[:160]!r}"
            )
        for word in words:
            word["word_index"] = word_index
            flat_words.append(word)
            word_index += 1
        segments.append(
            {
                "segment_id": segment_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": segment_text,
                "avg_logprob": _finite_or_none(getattr(segment, "avg_logprob", None)),
                "no_speech_probability": _finite_or_none(
                    getattr(segment, "no_speech_prob", None)
                ),
                "compression_ratio": _finite_or_none(
                    getattr(segment, "compression_ratio", None)
                ),
                "temperature": _finite_or_none(getattr(segment, "temperature", None)),
                "words": words,
            }
        )
    flat_words.sort(key=lambda word: (int(word["start_ms"]), int(word["end_ms"]), int(word["word_index"])))
    for new_index, word in enumerate(flat_words, start=1):
        word["word_index"] = new_index
    return segments, flat_words


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _derive_word_speech_regions(
    words: Sequence[Mapping[str, Any]], *, merge_gap_ms: int = 500, pad_ms: int = 100
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for word in words:
        start = max(0, int(word["start_ms"]) - pad_ms)
        end = int(word["end_ms"]) + pad_ms
        if regions and start - int(regions[-1]["end_ms"]) <= merge_gap_ms:
            regions[-1]["end_ms"] = max(int(regions[-1]["end_ms"]), end)
        else:
            regions.append(
                {
                    "start_ms": start,
                    "end_ms": end,
                    "source": "vad_filtered_word_timing_fallback",
                }
            )
    for index, region in enumerate(regions, start=1):
        region["vad_region_index"] = index
    return regions


def _extract_vad_regions(
    audio_path: Path,
    config: TranscriptionConfig,
    fallback_words: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    if not config.extract_explicit_vad_regions:
        return _derive_word_speech_regions(fallback_words), "explicit_vad_disabled"
    try:
        from faster_whisper.audio import decode_audio  # type: ignore
        from faster_whisper.vad import VadOptions, get_speech_timestamps  # type: ignore

        audio = decode_audio(str(audio_path), sampling_rate=16000)
        options = VadOptions(**config.vad_parameters)
        chunks = get_speech_timestamps(audio, options, sampling_rate=16000)
        regions = [
            {
                "vad_region_index": index,
                "start_ms": round(int(chunk["start"]) / 16000 * 1000),
                "end_ms": round(int(chunk["end"]) / 16000 * 1000),
                "source": "silero_vad",
            }
            for index, chunk in enumerate(chunks, start=1)
            if int(chunk["end"]) > int(chunk["start"])
        ]
        del audio
        gc.collect()
        if regions:
            return regions, None
        return _derive_word_speech_regions(fallback_words), "silero_vad_returned_no_regions"
    except Exception as exc:
        LOGGER.warning("Explicit VAD region extraction failed; using VAD-filtered word regions: %s", exc)
        return _derive_word_speech_regions(fallback_words), f"{type(exc).__name__}: {exc}"


def _prompt_text(names: Sequence[str], religious_terms: Sequence[str]) -> str | None:
    terms: list[str] = []
    for value in [*names, *religious_terms]:
        clean = str(value).strip()
        if clean and clean not in terms:
            terms.append(clean)
    if not terms:
        return None
    return "Türkçe dizi diyaloğu. Özel yazımlar: " + ", ".join(terms[:80]) + "."


def _extract_clip(audio_path: Path, output_path: Path, start_ms: int, end_ms: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise TranscriptionError("ffmpeg is required for targeted span verification")
    duration_seconds = (end_ms - start_ms) / 1000.0
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_ms / 1000.0:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    process = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 44:
        diagnostic = process.stderr.strip()[-3000:]
        raise TranscriptionError(f"Targeted clip extraction failed: {diagnostic}")


def verify_spans(
    audio_path: str | Path,
    model: Any,
    spans: Sequence[Mapping[str, Any]],
    *,
    config: TranscriptionConfig,
    initial_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Decode only selected time ranges with a more conservative beam."""

    source = Path(audio_path)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="asr-verify-") as temporary_name:
        temporary_dir = Path(temporary_name)
        for ordinal, plan in enumerate(spans, start=1):
            start_ms = int(plan["clip_start_ms"])
            end_ms = int(plan["clip_end_ms"])
            clip_path = temporary_dir / f"clip_{ordinal:03d}.wav"
            try:
                _extract_clip(source, clip_path, start_ms, end_ms)
                iterator, info = model.transcribe(
                    str(clip_path),
                    language=config.language,
                    task="transcribe",
                    beam_size=config.verification_beam_size,
                    best_of=max(config.best_of, config.verification_beam_size),
                    patience=max(config.patience, 1.2),
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=config.vad_parameters,
                )
                segments, words = _consume_segments(iterator, offset_ms=start_ms)
                text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
                results.append(
                    {
                        "verification_index": int(plan.get("verification_index", ordinal)),
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "requested_span_start_ms": int(plan["start_ms"]),
                        "requested_span_end_ms": int(plan["end_ms"]),
                        "reasons": list(plan.get("reasons", [])),
                        "source_segment_ids": list(plan.get("source_segment_ids", [])),
                        "language": getattr(info, "language", config.language),
                        "language_probability": _finite_or_none(
                            getattr(info, "language_probability", None)
                        ),
                        "text": text,
                        "segments": segments,
                        "words": words,
                        "status": "completed",
                    }
                )
            except Exception as exc:
                LOGGER.warning(
                    "Targeted verification span %d failed and was flagged for review: %s",
                    ordinal,
                    exc,
                )
                results.append(
                    {
                        "verification_index": int(plan.get("verification_index", ordinal)),
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "requested_span_start_ms": int(plan["start_ms"]),
                        "requested_span_end_ms": int(plan["end_ms"]),
                        "reasons": list(plan.get("reasons", [])),
                        "source_segment_ids": list(plan.get("source_segment_ids", [])),
                        "text": "",
                        "segments": [],
                        "words": [],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return results


def _validate_transcription_data(data: Mapping[str, Any], audio_sha256: str) -> list[str]:
    errors: list[str] = []
    if data.get("format_version") != TRANSCRIPTION_FORMAT_VERSION:
        errors.append("unsupported transcription format_version")
    if data.get("audio_sha256") != audio_sha256:
        errors.append("audio_sha256 mismatch")
    if data.get("language") != "tr":
        errors.append("transcription language is not Turkish")
    words = data.get("words")
    if not isinstance(words, list) or not words:
        errors.append("transcription has no timed words")
        return errors
    expected_index = 1
    previous_start = -1
    flat_word_signatures: list[tuple[str, int, int, str]] = []
    for word in words:
        if not isinstance(word, Mapping):
            errors.append("word record is not an object")
            continue
        try:
            if int(word["word_index"]) != expected_index:
                errors.append(f"word_index is not sequential at {expected_index}")
            start = int(word["start_ms"])
            end = int(word["end_ms"])
            if start < 0 or end <= start:
                errors.append(f"invalid timing at word {expected_index}")
            if start < previous_start:
                errors.append(f"words are not ordered at word {expected_index}")
            if not str(word.get("text", "")).strip():
                errors.append(f"empty text at word {expected_index}")
            else:
                flat_word_signatures.append(
                    (
                        str(word.get("segment_id", "")),
                        start,
                        end,
                        str(word["text"]),
                    )
                )
            previous_start = start
        except (KeyError, TypeError, ValueError):
            errors.append(f"malformed word record at {expected_index}")
        expected_index += 1

    segments = data.get("segments")
    segment_word_signatures: list[tuple[str, int, int, str]] = []
    if not isinstance(segments, list) or not segments:
        errors.append("transcription has no ASR segments")
    else:
        for position, segment in enumerate(segments, start=1):
            if not isinstance(segment, Mapping):
                errors.append(f"ASR segment {position} is not an object")
                continue
            segment_id = str(segment.get("segment_id", position))
            segment_text = str(segment.get("text", "")).strip()
            segment_words_value = segment.get("words")
            if not isinstance(segment_words_value, list):
                errors.append(f"ASR segment {segment_id} words is not a list")
                continue
            valid_segment_words: list[Mapping[str, Any]] = []
            for word in segment_words_value:
                if not isinstance(word, Mapping):
                    continue
                try:
                    start = int(word["start_ms"])
                    end = int(word["end_ms"])
                    text = str(word.get("text", ""))
                except (KeyError, TypeError, ValueError):
                    continue
                if start < 0 or end <= start or not text.strip():
                    continue
                valid_segment_words.append(word)
                segment_word_signatures.append((segment_id, start, end, text))
            if segment_text and not valid_segment_words:
                errors.append(
                    f"ASR segment {segment_id} contains text but has no valid timed words"
                )
            elif valid_segment_words and not segment_text:
                errors.append(
                    f"ASR segment {segment_id} has timed words but empty segment text"
                )
            elif segment_text and not _asr_text_coverage_matches(
                segment_text, valid_segment_words
            ):
                errors.append(
                    f"ASR segment {segment_id} text differs from timed-word coverage"
                )

    if sorted(segment_word_signatures) != sorted(flat_word_signatures):
        errors.append("flat timed words do not exactly match the ASR segment word records")
    vad_regions = data.get("vad_regions")
    if not isinstance(vad_regions, list) or not vad_regions:
        errors.append("no speech/VAD regions were saved")
    return errors


def _resume_transcription(
    marker: Mapping[str, Any], marker_path: Path, audio_sha256: str
) -> TranscriptionResult | None:
    try:
        output_path = Path(marker["outputs"]["transcription"]["path"])
        with output_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or _validate_transcription_data(data, audio_sha256):
            return None
        return TranscriptionResult(
            output_path=output_path,
            marker_path=marker_path,
            data=data,
            resumed=True,
        )
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


def transcribe_audio(
    audio_path: str | Path,
    prepare_dir: str | Path,
    *,
    config: TranscriptionConfig | None = None,
    captions_path: str | Path | None = None,
    canonical_names: Sequence[str] = (),
    religious_terms: Sequence[str] = (),
    force: bool = False,
) -> TranscriptionResult:
    """Run/resume Turkish large-v3 ASR and bounded suspicious-span verification."""

    settings = config or TranscriptionConfig()
    source = Path(audio_path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise TranscriptionError(f"Audio input is missing or empty: {source}")
    destination = Path(prepare_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "transcription.json"
    marker_path = destination / "transcription.done.json"
    audio_hash = sha256_file(source)
    caption_file = Path(captions_path) if captions_path is not None else None
    captions_hash = sha256_file(caption_file) if caption_file and caption_file.is_file() else None
    input_descriptor = {
        "audio_sha256": audio_hash,
        "config": settings.marker_settings(),
        "captions_sha256": captions_hash,
        "canonical_names": list(canonical_names),
        "religious_terms": list(religious_terms),
        "format_version": TRANSCRIPTION_FORMAT_VERSION,
    }
    input_hash = sha256_json(input_descriptor)

    if not force:
        marker = load_valid_stage_marker(
            marker_path,
            stage="transcription",
            input_sha256=input_hash,
            required_output_keys=("transcription",),
            allowed_root=destination,
        )
        if marker is not None:
            result = _resume_transcription(marker, marker_path, audio_hash)
            if result is not None:
                LOGGER.info("Transcription resumed from verified marker: %s", marker_path)
                return result

    captions = load_vtt_captions(caption_file) if caption_file is not None else []
    WhisperModel, ctranslate2 = _import_whisper()
    device, compute_type = _select_device(settings, ctranslate2)
    cpu_threads = settings.cpu_threads if settings.cpu_threads > 0 else max(1, os.cpu_count() or 1)
    LOGGER.info(
        "Loading faster-whisper %s on %s (%s)", settings.model_name, device, compute_type
    )
    runtime_fallback_reason: str | None = None
    try:
        model = WhisperModel(
            settings.model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=settings.num_workers,
        )
    except Exception as exc:
        if device == "cuda" and settings.allow_cpu_fallback and _is_cuda_runtime_error(exc):
            LOGGER.warning(
                "CUDA model initialization failed (%s). Retrying once on CPU/%s.",
                exc,
                settings.compute_type_cpu,
            )
            runtime_fallback_reason = f"CUDA initialization: {type(exc).__name__}: {exc}"
            device = "cpu"
            compute_type = settings.compute_type_cpu
            try:
                model = WhisperModel(
                    settings.model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=settings.num_workers,
                )
            except Exception as cpu_exc:
                raise TranscriptionError(
                    "Whisper CUDA initialization failed and the one CPU/int8 fallback "
                    f"also failed: {cpu_exc}"
                ) from cpu_exc
        else:
            raise TranscriptionError(
                f"Could not load Whisper model {settings.model_name} on {device}/{compute_type}: {exc}"
            ) from exc

    initial_prompt = _prompt_text(canonical_names, religious_terms)

    def run_main_pass(whisper_model: Any) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        LOGGER.info("Starting the single full-episode Turkish ASR pass")
        iterator, pass_info = whisper_model.transcribe(
            str(source),
            language=settings.language,
            task="transcribe",
            beam_size=settings.beam_size,
            best_of=settings.best_of,
            patience=settings.patience,
            temperature=settings.temperature,
            condition_on_previous_text=settings.condition_on_previous_text,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=settings.vad_parameters,
        )
        pass_segments, pass_words = _consume_segments(iterator)
        return pass_info, pass_segments, pass_words

    try:
        info, segments, words = run_main_pass(model)
    except Exception as exc:
        if device == "cuda" and settings.allow_cpu_fallback and _is_cuda_runtime_error(exc):
            LOGGER.warning(
                "CUDA failed during first ASR inference (%s). Retrying the failed pass once "
                "on CPU/%s.",
                exc,
                settings.compute_type_cpu,
            )
            runtime_fallback_reason = f"CUDA inference: {type(exc).__name__}: {exc}"
            del model
            gc.collect()
            device = "cpu"
            compute_type = settings.compute_type_cpu
            try:
                model = WhisperModel(
                    settings.model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=settings.num_workers,
                )
                info, segments, words = run_main_pass(model)
            except Exception as cpu_exc:
                raise TranscriptionError(
                    "The CUDA ASR attempt failed and the one CPU/int8 fallback also failed: "
                    f"{cpu_exc}"
                ) from cpu_exc
        else:
            raise TranscriptionError(f"Full Turkish ASR pass failed: {exc}") from exc
    if not words:
        raise TranscriptionError("Whisper returned no word-level timestamps")

    reported_duration = _finite_or_none(getattr(info, "duration", None))
    duration_ms = round(reported_duration * 1000) if reported_duration else max(
        int(word["end_ms"]) for word in words
    )
    base: dict[str, Any] = {"segments": segments, "words": words}
    suspicious = find_suspicious_spans(
        base,
        captions,
        canonical_names,
        religious_terms,
        settings,
    )
    verification_plan = _select_verification_plan(suspicious, duration_ms, settings)
    LOGGER.info(
        "Suspicious spans: %d; targeted verification selected: %d (%.1f minutes)",
        len(suspicious),
        len(verification_plan),
        sum(int(span["clip_duration_ms"]) for span in verification_plan) / 60000.0,
    )
    verification_results = verify_spans(
        source,
        model,
        verification_plan,
        config=settings,
        initial_prompt=initial_prompt,
    )
    vad_regions, vad_fallback_reason = _extract_vad_regions(source, settings, words)
    model_versions = {
        "faster_whisper": _package_version("faster-whisper"),
        "ctranslate2": _package_version("ctranslate2"),
    }
    data: dict[str, Any] = {
        "format_version": TRANSCRIPTION_FORMAT_VERSION,
        "created_at": utc_now_iso(),
        "audio_path": str(source.resolve()),
        "audio_sha256": audio_hash,
        "duration_ms": duration_ms,
        "language": str(getattr(info, "language", settings.language)),
        "language_probability": _finite_or_none(getattr(info, "language_probability", None)),
        "model": {
            "name": settings.model_name,
            "device": device,
            "compute_type": compute_type,
            "versions": model_versions,
            "settings": settings.marker_settings(),
            "full_episode_passes": 1,
            "runtime_fallback_reason": runtime_fallback_reason,
        },
        "caption_source_sha256": captions_hash,
        "youtube_captions": captions,
        "segments": segments,
        "words": words,
        "vad_regions": vad_regions,
        "vad_fallback_reason": vad_fallback_reason,
        "suspicious_spans": suspicious,
        "verification_plan": verification_plan,
        "verification_results": verification_results,
        "verification_summary": {
            "suspicious_span_count": len(suspicious),
            "selected_span_count": len(verification_plan),
            "completed_span_count": sum(
                result.get("status") == "completed" for result in verification_results
            ),
            "failed_span_count": sum(
                result.get("status") == "failed" for result in verification_results
            ),
            "verified_audio_duration_ms": sum(
                int(span["clip_duration_ms"]) for span in verification_plan
            ),
            "bounded_targeted_only": True,
        },
    }
    validation_errors = _validate_transcription_data(data, audio_hash)
    if validation_errors:
        raise TranscriptionError(
            "Transcription validation failed: " + "; ".join(validation_errors[:20])
        )
    atomic_write_json(output_path, data)
    write_stage_marker(
        marker_path,
        stage="transcription",
        input_sha256=input_hash,
        outputs={"transcription": output_path},
        details={
            "audio_sha256": audio_hash,
            "model_name": settings.model_name,
            "model_versions": model_versions,
            "device": device,
            "compute_type": compute_type,
            "word_count": len(words),
            "segment_count": len(segments),
            "vad_region_count": len(vad_regions),
            "verification_summary": data["verification_summary"],
            "transcription_sha256": sha256_file(output_path),
        },
    )
    return TranscriptionResult(
        output_path=output_path,
        marker_path=marker_path,
        data=data,
        resumed=False,
    )


def validate_transcription_marker(
    marker_path: str | Path,
    *,
    audio_sha256: str,
    config: TranscriptionConfig | None = None,
    captions_sha256: str | None = None,
    canonical_names: Sequence[str] = (),
    religious_terms: Sequence[str] = (),
) -> bool:
    """Validate persisted ASR identity, artifact hash, and timed-word structure."""

    settings = config or TranscriptionConfig()
    marker_file = Path(marker_path)
    input_hash = sha256_json(
        {
            "audio_sha256": audio_sha256,
            "config": settings.marker_settings(),
            "captions_sha256": captions_sha256,
            "canonical_names": list(canonical_names),
            "religious_terms": list(religious_terms),
            "format_version": TRANSCRIPTION_FORMAT_VERSION,
        }
    )
    marker = load_valid_stage_marker(
        marker_file,
        stage="transcription",
        input_sha256=input_hash,
        required_output_keys=("transcription",),
        allowed_root=marker_file.parent,
    )
    return marker is not None and _resume_transcription(marker, marker_file, audio_sha256) is not None


__all__ = [
    "TranscriptionConfig",
    "TranscriptionError",
    "TranscriptionResult",
    "find_suspicious_spans",
    "load_vtt_captions",
    "transcribe_audio",
    "validate_transcription_marker",
    "verify_spans",
]
