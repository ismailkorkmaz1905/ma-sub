"""Strict Turkish CTC forced alignment for corrected subtitle text.

This module deliberately treats WhisperX as an optional runtime dependency.
Importing :mod:`src.forced_align` never imports torch, transformers, or
WhisperX.  The heavy dependency is loaded only when
:func:`align_corrected_segments` is called, and a compatible module can be
injected for deterministic unit tests.

No guessed timing is permitted.  WhisperX interpolation is disabled and every
lexical token (a whitespace-delimited token containing a Unicode letter or
number) must have a finite, positive, monotonically ordered CTC interval.
Every lexical token must also carry a finite WhisperX acoustic score in
``[0, 1]`` at or above the beta minimum (``0.30``), and no single aligned word
may exceed 2500 ms.
Punctuation-only tokens do not require their own timestamp because their
surface form remains in the segment text.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import inspect
import itertools
import json
import math
import re
import types
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence, Union, get_args, get_origin

from .speaker import overlap_is_unsafe, speaker_id
from ..reliability import UnitJournal, digest
from ..progress import mark_work_progress


FORCED_ALIGNMENT_FORMAT_VERSION = "1.0"
TIMING_SOURCE = "whisperx_ctc_forced_alignment"
SUPPORTED_LANGUAGE = "tr"
SUPPORTED_WHISPERX_VERSION = "3.8.6"
DEFAULT_TURKISH_ALIGNMENT_MODEL = (
    "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"
)
DEFAULT_MIN_WORD_SCORE = 0.30
CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE = 0.25
REVIEW_WORD_SCORE = 0.55
DEFAULT_MAX_WORD_DURATION_MS = 2_500
DEFAULT_MAX_OUTWARD_DRIFT_MS = 500
EDITED_TOKEN_MIN_WORD_SCORE = 0.55
AUDIO_REVIEW_SCORE_CONTEXT = "hash_bound_confirmed_dialogue_audio_review"
DURATION_VAD_CONTEXT = "hash_bound_independent_vad_boundary"
ALIGNMENT_TEXT_NORMALIZATION = "turkish_ascii_ctc_v1"
OVERLAP_RESOLUTION_POLICY = "ctc_joint_adaptive_partition_v6"
MAX_OVERLAP_COMBINATIONS = 2_097_152
DURATION_VAD_FIELDS = frozenset(
    {
        "duration_context",
        "raw_start_ms",
        "raw_end_ms",
        "vad_region_index",
        "vad_start_ms",
        "vad_end_ms",
        "vad_source",
    }
)


class ForcedAlignmentError(RuntimeError):
    """Raised when corrected text cannot be aligned without guessed timing."""


def _alignment_model_text(value: str) -> str:
    mapped = value.translate(
        str.maketrans(
            {
                "Ç": "C",
                "Ğ": "G",
                "İ": "I",
                "Ö": "O",
                "Ş": "S",
                "Ü": "U",
                "ç": "c",
                "ğ": "g",
                "ı": "i",
                "ö": "o",
                "ş": "s",
                "ü": "u",
            }
        )
    )
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", mapped)
        if not unicodedata.combining(character)
    )


def _require_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForcedAlignmentError(f"{field} must be an integer")
    if value < minimum:
        raise ForcedAlignmentError(f"{field} must be at least {minimum}")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForcedAlignmentError(f"{field} must be a non-empty string")
    return unicodedata.normalize("NFC", value)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForcedAlignmentError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ForcedAlignmentError(f"{field} must be a finite number")
    return number


def _lexeme(value: str) -> str:
    """Normalize a word for exact lexical coverage comparisons.

    Punctuation and combining marks are intentionally excluded.  This makes
    ``M.K.`` and ``M.K`` the same lexical token without making two different
    words equivalent.
    """

    # Python's locale-independent casefold maps ASCII ``I`` to ``i``.  Turkish
    # casing instead pairs ``I``/``ı`` and ``İ``/``i``; normalize those two
    # capitals before casefolding so a casing-only correction is not mislabeled
    # as an acoustic lexical replacement.
    normalized = unicodedata.normalize("NFKC", value).translate(
        str.maketrans({"I": "ı", "İ": "i"})
    ).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _surface_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text, flags=re.UNICODE)


def _lexical_tokens(text: str) -> list[str]:
    return [token for token in _surface_tokens(text) if _lexeme(token)]


def _canonical_lexical_surfaces(text: str) -> list[str]:
    """Bind each lexical token to its exact corrected-Turkish surface.

    WhisperX supplies acoustic intervals and scores, but its returned token
    spelling is not a correction authority.  Punctuation-only whitespace
    tokens are attached to their nearest preceding lexical token (or carried
    into the first token) so joining the resulting words reconstructs the
    human-corrected surface instead of WhisperX capitalization/punctuation.
    """

    surfaces: list[str] = []
    leading_punctuation: list[str] = []
    for token in _surface_tokens(text):
        if _lexeme(token):
            if leading_punctuation:
                surfaces.append(" ".join([*leading_punctuation, token]))
                leading_punctuation.clear()
            else:
                surfaces.append(token)
        elif surfaces:
            surfaces[-1] = f"{surfaces[-1]} {token}"
        else:
            leading_punctuation.append(token)
    return surfaces


def _token_edit_audit(
    asr_text: str,
    corrected_text: str,
    *,
    deletion_audio_reviewed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a deterministic lexical edit map and its complete audit.

    The map uses Levenshtein alignment with a stable tie-break: exact diagonal
    matches first, then replacements, deletions, and insertions.  Preferring a
    replacement on equal-cost paths keeps ordinary spelling corrections from
    being mislabeled as an unreviewed deletion plus insertion.  Punctuation,
    casing, and apostrophes are excluded by :func:`_lexeme` and therefore do
    not manufacture an acoustic edit.
    """

    asr_surfaces = _lexical_tokens(asr_text)
    corrected_surfaces = _canonical_lexical_surfaces(corrected_text)
    asr_lexemes = [_lexeme(token) for token in asr_surfaces]
    corrected_lexemes = [_lexeme(token) for token in corrected_surfaces]
    row_count = len(asr_lexemes) + 1
    column_count = len(corrected_lexemes) + 1
    distance = [[0] * column_count for _ in range(row_count)]
    for row in range(row_count):
        distance[row][0] = row
    for column in range(column_count):
        distance[0][column] = column
    for row in range(1, row_count):
        for column in range(1, column_count):
            substitution_cost = (
                0
                if asr_lexemes[row - 1] == corrected_lexemes[column - 1]
                else 1
            )
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + substitution_cost,
            )

    reversed_operations: list[dict[str, Any]] = []
    deleted_asr_token_count = 0
    row = len(asr_lexemes)
    column = len(corrected_lexemes)
    while row or column:
        if (
            row
            and column
            and asr_lexemes[row - 1] == corrected_lexemes[column - 1]
            and distance[row][column] == distance[row - 1][column - 1]
        ):
            reversed_operations.append(
                {
                    "corrected_token_index": column,
                    "corrected_lexeme": corrected_lexemes[column - 1],
                    "asr_token_index": row,
                    "edit_kind": "unchanged",
                }
            )
            row -= 1
            column -= 1
            continue
        if (
            row
            and column
            and distance[row][column] == distance[row - 1][column - 1] + 1
        ):
            reversed_operations.append(
                {
                    "corrected_token_index": column,
                    "corrected_lexeme": corrected_lexemes[column - 1],
                    "asr_token_index": row,
                    "edit_kind": "replaced",
                }
            )
            row -= 1
            column -= 1
            continue
        if row and distance[row][column] == distance[row - 1][column] + 1:
            deleted_asr_token_count += 1
            row -= 1
            continue
        if column and distance[row][column] == distance[row][column - 1] + 1:
            reversed_operations.append(
                {
                    "corrected_token_index": column,
                    "corrected_lexeme": corrected_lexemes[column - 1],
                    "asr_token_index": None,
                    "edit_kind": "inserted",
                }
            )
            column -= 1
            continue
        raise ForcedAlignmentError("lexical edit audit could not be reconstructed")

    token_edits = list(reversed(reversed_operations))
    unchanged_count = sum(
        operation["edit_kind"] == "unchanged" for operation in token_edits
    )
    replaced_count = sum(
        operation["edit_kind"] == "replaced" for operation in token_edits
    )
    inserted_count = sum(
        operation["edit_kind"] == "inserted" for operation in token_edits
    )
    edited_count = replaced_count + inserted_count
    corrected_count = len(corrected_lexemes)
    unreviewed_deleted_count = (
        0 if deletion_audio_reviewed else deleted_asr_token_count
    )
    audit: dict[str, Any] = {
        "asr_lexical_token_count": len(asr_lexemes),
        "corrected_lexical_token_count": corrected_count,
        "unchanged_token_count": unchanged_count,
        "replaced_token_count": replaced_count,
        "inserted_token_count": inserted_count,
        "edited_token_count": edited_count,
        "edited_token_ratio": (
            edited_count / corrected_count if corrected_count else 0.0
        ),
        "deleted_asr_token_count": deleted_asr_token_count,
        "deletion_audio_reviewed": deletion_audio_reviewed,
        "unreviewed_deleted_token_count": unreviewed_deleted_count,
        "token_edits": token_edits,
    }
    return token_edits, audit


def correction_deletes_lexical_tokens(asr_text: str, corrected_text: str) -> bool:
    _, audit = _token_edit_audit(
        asr_text,
        corrected_text,
        deletion_audio_reviewed=False,
    )
    return bool(audit["deleted_asr_token_count"])


def validate_coarse_segments(
    coarse_segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and copy ordered alignment windows.

    Each input record must contain ``start_ms``, ``end_ms``, corrected ``text``,
    immutable ``asr_text``, ``deletion_audio_reviewed`` and a unique
    ``utterance_uid``.  Windows must be ordered by start time; overlap is
    allowed here because the final word validator is the timing authority and
    rejects any actual overlapping word intervals.
    """

    if isinstance(coarse_segments, (str, bytes)) or not isinstance(
        coarse_segments, Sequence
    ):
        raise ForcedAlignmentError("coarse_segments must be a sequence")
    if not coarse_segments:
        raise ForcedAlignmentError("coarse_segments must not be empty")

    validated: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    previous_start = -1
    for index, raw in enumerate(coarse_segments, start=1):
        if not isinstance(raw, Mapping):
            raise ForcedAlignmentError(f"coarse segment {index} must be an object")
        start_ms = _require_integer(
            raw.get("start_ms"), f"coarse segment {index} start_ms"
        )
        end_ms = _require_integer(
            raw.get("end_ms"), f"coarse segment {index} end_ms", minimum=1
        )
        if end_ms <= start_ms:
            raise ForcedAlignmentError(
                f"coarse segment {index} end_ms must be greater than start_ms"
            )
        if start_ms < previous_start:
            raise ForcedAlignmentError(
                f"coarse segment {index} starts before the preceding segment"
            )
        previous_start = start_ms
        coarse_start_ms = _require_integer(
            raw.get("coarse_start_ms", start_ms),
            f"coarse segment {index} coarse_start_ms",
        )
        coarse_end_ms = _require_integer(
            raw.get("coarse_end_ms", end_ms),
            f"coarse segment {index} coarse_end_ms",
            minimum=1,
        )
        if (
            coarse_start_ms < start_ms
            or coarse_end_ms > end_ms
            or coarse_end_ms <= coarse_start_ms
        ):
            raise ForcedAlignmentError(
                f"coarse segment {index} unpadded bounds must be a positive "
                "interval inside its alignment window"
            )

        text = _require_nonempty_string(
            raw.get("text"), f"coarse segment {index} text"
        )
        if not _lexical_tokens(text):
            raise ForcedAlignmentError(
                f"coarse segment {index} contains no lexical token"
            )
        asr_text_value = raw.get("asr_text")
        if not isinstance(asr_text_value, str):
            raise ForcedAlignmentError(
                f"coarse segment {index} asr_text must be a string"
            )
        asr_text = unicodedata.normalize("NFC", asr_text_value)
        deletion_audio_reviewed = raw.get("deletion_audio_reviewed")
        if not isinstance(deletion_audio_reviewed, bool):
            raise ForcedAlignmentError(
                f"coarse segment {index} deletion_audio_reviewed must be boolean"
            )
        audio_reviewed = raw.get("audio_reviewed", False)
        review_disposition = raw.get("review_disposition", "not_applicable")
        if not isinstance(audio_reviewed, bool):
            raise ForcedAlignmentError(
                f"coarse segment {index} audio_reviewed must be boolean"
            )
        if review_disposition not in {"not_applicable", "confirmed_dialogue"}:
            raise ForcedAlignmentError(
                f"coarse segment {index} review_disposition is invalid for alignment"
            )
        if audio_reviewed != (review_disposition == "confirmed_dialogue"):
            raise ForcedAlignmentError(
                f"coarse segment {index} audio-review fields are inconsistent"
            )
        _, edit_audit = _token_edit_audit(
            asr_text,
            text,
            deletion_audio_reviewed=deletion_audio_reviewed,
        )
        if edit_audit["unreviewed_deleted_token_count"] != 0:
            raise ForcedAlignmentError(
                f"coarse segment {index} has lexical deletions without exact "
                "audio-reviewed evidence"
            )
        utterance_uid = _require_nonempty_string(
            raw.get("utterance_uid"), f"coarse segment {index} utterance_uid"
        )
        try:
            segment_speaker_id = speaker_id(raw, f"coarse segment {index}")
        except ValueError as exc:
            raise ForcedAlignmentError(str(exc)) from exc
        if utterance_uid in seen_uids:
            raise ForcedAlignmentError(
                f"duplicate utterance_uid in coarse segments: {utterance_uid}"
            )
        seen_uids.add(utterance_uid)
        segment = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "coarse_start_ms": coarse_start_ms,
                "coarse_end_ms": coarse_end_ms,
                "text": text,
                "asr_text": asr_text,
                "deletion_audio_reviewed": deletion_audio_reviewed,
                "audio_reviewed": audio_reviewed,
                "review_disposition": review_disposition,
                "utterance_uid": utterance_uid,
            }
        if segment_speaker_id is not None:
            segment["speaker_id"] = segment_speaker_id
        validated.append(segment)
    return validated


def _validate_word_record(
    raw: Mapping[str, Any],
    *,
    utterance_uid: str,
    raw_index: int,
    window_start_ms: int,
    window_end_ms: int,
) -> tuple[str, int, int, float] | None:
    text_value = raw.get("word", raw.get("text"))
    if not isinstance(text_value, str) or not text_value.strip():
        raise ForcedAlignmentError(
            f"{utterance_uid} aligned word {raw_index} has no text"
        )
    text = unicodedata.normalize("NFC", text_value.strip())
    if not _lexeme(text):
        # A standalone punctuation token is preserved in segment text and is
        # not allowed to manufacture a subtitle interval.
        return None

    if raw.get("interpolated") or raw.get("synthetic"):
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} uses synthetic/interpolated timing"
        )
    upstream_source = raw.get("timing_source")
    if upstream_source not in (None, TIMING_SOURCE):
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} has untrusted timing_source "
            f"{upstream_source!r}"
        )

    start_seconds = _finite_number(
        raw.get("start"), f"{utterance_uid} word {text!r} start"
    )
    end_seconds = _finite_number(
        raw.get("end"), f"{utterance_uid} word {text!r} end"
    )
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} has an invalid interval"
        )

    start_ms = round(start_seconds * 1000)
    end_ms = round(end_seconds * 1000)
    if end_ms <= start_ms:
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} collapses to a non-positive ms interval"
        )
    if start_ms < window_start_ms or end_ms > window_end_ms:
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} lies outside its coarse alignment window"
        )
    score_value = raw.get("score")
    if score_value is None:
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} alignment score is required"
        )
    score = _finite_number(
        score_value, f"{utterance_uid} word {text!r} score"
    )
    if not 0.0 <= score <= 1.0:
        raise ForcedAlignmentError(
            f"{utterance_uid} word {text!r} alignment score must be within [0, 1]"
        )
    return text, start_ms, end_ms, score


def _trusted_vad_regions(
    vad_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(vad_regions, (str, bytes)) or not isinstance(vad_regions, Sequence):
        raise ForcedAlignmentError("vad_regions must be a sequence")
    trusted: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    previous_end_ms = -1
    for position, raw in enumerate(vad_regions, start=1):
        if not isinstance(raw, Mapping):
            raise ForcedAlignmentError(f"vad_regions[{position}] must be an object")
        vad_index = _require_integer(
            raw.get("vad_region_index"),
            f"vad_regions[{position}] vad_region_index",
            minimum=1,
        )
        if vad_index in seen_indices:
            raise ForcedAlignmentError("vad_region_index values must be unique")
        seen_indices.add(vad_index)
        start_ms = _require_integer(
            raw.get("start_ms"), f"vad_regions[{position}] start_ms"
        )
        end_ms = _require_integer(
            raw.get("end_ms"), f"vad_regions[{position}] end_ms", minimum=1
        )
        if end_ms <= start_ms:
            raise ForcedAlignmentError(f"vad_regions[{position}] interval is invalid")
        if start_ms < previous_end_ms:
            raise ForcedAlignmentError("vad_regions must be chronological and disjoint")
        if raw.get("source") != "silero_vad":
            raise ForcedAlignmentError(
                f"vad_regions[{position}] is not independent Silero VAD evidence"
            )
        trusted.append(
            {
                "vad_region_index": vad_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "source": "silero_vad",
            }
        )
        previous_end_ms = end_ms
    return trusted


def _raw_aligned_words(result: Any, utterance_uid: str) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise ForcedAlignmentError(
            f"WhisperX returned a non-object result for {utterance_uid}"
        )
    word_segments = result.get("word_segments")
    if word_segments is None:
        segments = result.get("segments")
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
            raise ForcedAlignmentError(
                f"WhisperX returned no word_segments for {utterance_uid}"
            )
        flattened: list[Any] = []
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise ForcedAlignmentError(
                    f"WhisperX returned a malformed segment for {utterance_uid}"
                )
            nested_words = segment.get("words") or []
            if isinstance(nested_words, (str, bytes)) or not isinstance(
                nested_words, Sequence
            ):
                raise ForcedAlignmentError(
                    f"WhisperX returned malformed words for {utterance_uid}"
                )
            flattened.extend(nested_words)
        word_segments = flattened

    if isinstance(word_segments, (str, bytes)) or not isinstance(
        word_segments, Sequence
    ):
        raise ForcedAlignmentError(
            f"WhisperX word_segments is malformed for {utterance_uid}"
        )
    words: list[Mapping[str, Any]] = []
    for index, word in enumerate(word_segments, start=1):
        if not isinstance(word, Mapping):
            raise ForcedAlignmentError(
                f"WhisperX word {index} is malformed for {utterance_uid}"
            )
        words.append(word)
    return words


def _normalize_aligned_words(
    result: Any,
    coarse: Mapping[str, Any],
    *,
    segment_index: int,
    first_word_index: int,
    min_word_score: float,
    max_word_duration_ms: int,
    vad_regions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    utterance_uid = str(coarse["utterance_uid"])
    expected_tokens = _canonical_lexical_surfaces(str(coarse["text"]))
    token_edits, _ = _token_edit_audit(
        str(coarse["asr_text"]),
        str(coarse["text"]),
        deletion_audio_reviewed=bool(coarse["deletion_audio_reviewed"]),
    )
    raw_words = _raw_aligned_words(result, utterance_uid)
    audio_review_supported = (
        coarse["audio_reviewed"] is True
        and coarse["review_disposition"] == "confirmed_dialogue"
    )

    accepted: list[tuple[str, int, int, float | None]] = []
    punctuation_only_count = 0
    previous_end_ms: int | None = None
    for raw_index, raw in enumerate(raw_words, start=1):
        normalized = _validate_word_record(
            raw,
            utterance_uid=utterance_uid,
            raw_index=raw_index,
            window_start_ms=int(coarse["start_ms"]),
            window_end_ms=int(coarse["end_ms"]),
        )
        if normalized is None:
            punctuation_only_count += 1
            continue
        text, start_ms, end_ms, score = normalized
        if previous_end_ms is not None and start_ms < previous_end_ms:
            raise ForcedAlignmentError(
                f"{utterance_uid} has overlapping/backward aligned words at {text!r}"
            )
        previous_end_ms = end_ms
        accepted.append((text, start_ms, end_ms, score))

    duration_audits: dict[int, dict[str, Any]] = {}
    for offset, (_, start_ms, end_ms, _) in enumerate(accepted):
        if end_ms - start_ms <= max_word_duration_ms:
            continue
        next_start_ms = accepted[offset + 1][1] if offset + 1 < len(accepted) else None
        matching_regions = [
            region
            for region in vad_regions
            if int(region["start_ms"]) <= start_ms < int(region["end_ms"])
            and int(region["end_ms"]) < end_ms
            and next_start_ms is not None
            and int(region["end_ms"]) <= next_start_ms
        ]
        trimmed_end_ms = (
            int(matching_regions[0]["end_ms"])
            if audio_review_supported and len(matching_regions) == 1
            else end_ms
        )
        if (
            not audio_review_supported
            or len(matching_regions) != 1
            or trimmed_end_ms <= start_ms
            or trimmed_end_ms - start_ms > max_word_duration_ms
        ):
            raise ForcedAlignmentError(
                f"{utterance_uid} word {accepted[offset][0]!r} duration "
                f"{end_ms - start_ms} ms exceeds the maximum "
                f"{max_word_duration_ms} ms"
            )
        region = matching_regions[0]
        duration_audits[offset] = {
            "duration_context": DURATION_VAD_CONTEXT,
            "raw_start_ms": start_ms,
            "raw_end_ms": end_ms,
            "vad_region_index": int(region["vad_region_index"]),
            "vad_start_ms": int(region["start_ms"]),
            "vad_end_ms": trimmed_end_ms,
            "vad_source": "silero_vad",
        }
        text, _, _, score = accepted[offset]
        accepted[offset] = (text, start_ms, trimmed_end_ms, score)

    expected_lexemes = [
        _lexeme(_alignment_model_text(token)) for token in expected_tokens
    ]
    actual_lexemes = [
        _lexeme(_alignment_model_text(word[0])) for word in accepted
    ]
    if actual_lexemes != expected_lexemes:
        missing_at = next(
            (
                index
                for index, (expected, actual) in enumerate(
                    zip(expected_lexemes, actual_lexemes), start=1
                )
                if expected != actual
            ),
            min(len(expected_lexemes), len(actual_lexemes)) + 1,
        )
        raise ForcedAlignmentError(
            f"{utterance_uid} lexical alignment coverage failed at token "
            f"{missing_at}: expected {len(expected_lexemes)}, aligned "
            f"{len(actual_lexemes)}"
        )

    words: list[dict[str, Any]] = []
    for offset, ((_, start_ms, end_ms, score), canonical_text, token_edit) in enumerate(
        zip(accepted, expected_tokens, token_edits)
    ):
        edit_kind = str(token_edit["edit_kind"])
        score_context = None
        if score < min_word_score:
            neighbor_scores = [
                accepted[position][3]
                for position in (offset - 1, offset + 1)
                if 0 <= position < len(accepted)
            ]
            if audio_review_supported:
                score_context = AUDIO_REVIEW_SCORE_CONTEXT
            elif (
                score < CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE
                or edit_kind != "unchanged"
                or not any(
                    neighbor_score >= min_word_score
                    for neighbor_score in neighbor_scores
                )
            ):
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {canonical_text!r} alignment score "
                    f"{score:.6f} is below the required minimum "
                    f"{min_word_score:.6f} without adjacent unchanged-word support"
                )
            else:
                score_context = "adjacent_unchanged_word"
        if edit_kind in {"inserted", "replaced"} and score < EDITED_TOKEN_MIN_WORD_SCORE:
            if audio_review_supported:
                score_context = AUDIO_REVIEW_SCORE_CONTEXT
            else:
                raise ForcedAlignmentError(
                    f"{utterance_uid} edited token {canonical_text!r} alignment score "
                    f"{score:.6f} is below the edited-token minimum "
                    f"{EDITED_TOKEN_MIN_WORD_SCORE:.6f}"
                )
        words.append(
            {
                "word_index": first_word_index + offset,
                "segment_index": segment_index,
                "segment_id": segment_index,
                "utterance_uid": utterance_uid,
                "text": canonical_text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "score": score,
                "probability": score,
                "edit_kind": edit_kind,
                "asr_token_index": token_edit["asr_token_index"],
                "timing_source": TIMING_SOURCE,
                **duration_audits.get(offset, {}),
                **(
                    {"score_context": score_context}
                    if score_context is not None
                    else {}
                ),
                **(
                    {"speaker_id": coarse["speaker_id"]}
                    if "speaker_id" in coarse
                    else {}
                ),
            }
        )
    return words, punctuation_only_count


def _segment_drift_audit(
    words: Sequence[Mapping[str, Any]],
    *,
    coarse_start_ms: int,
    coarse_end_ms: int,
) -> dict[str, int]:
    """Describe aligned-word drift from the original unpadded ASR bounds."""

    return {
        "first_word_start_delta_ms": int(words[0]["start_ms"]) - coarse_start_ms,
        "last_word_end_delta_ms": int(words[-1]["end_ms"]) - coarse_end_ms,
        "early_outward_drift_ms": max(
            0, coarse_start_ms - int(words[0]["start_ms"])
        ),
        "late_outward_drift_ms": max(
            0, int(words[-1]["end_ms"]) - coarse_end_ms
        ),
        "word_wholly_before_coarse_count": sum(
            int(word["end_ms"]) <= coarse_start_ms for word in words
        ),
        "word_wholly_after_coarse_count": sum(
            int(word["start_ms"]) >= coarse_end_ms for word in words
        ),
        "word_crossing_coarse_start_count": sum(
            int(word["start_ms"]) < coarse_start_ms < int(word["end_ms"])
            for word in words
        ),
        "word_crossing_coarse_end_count": sum(
            int(word["start_ms"]) < coarse_end_ms < int(word["end_ms"])
            for word in words
        ),
    }


def _bounded_drift_context(
    words: Sequence[Mapping[str, Any]],
    drift_audit: Mapping[str, int],
    *,
    window_start_ms: int,
    window_end_ms: int,
    coarse_start_ms: int,
    coarse_end_ms: int,
    max_outward_drift_ms: int,
) -> str | None:
    early_drift = int(drift_audit["early_outward_drift_ms"])
    late_drift = int(drift_audit["late_outward_drift_ms"])
    if early_drift <= max_outward_drift_ms and late_drift <= max_outward_drift_ms:
        return None
    if (
        early_drift > coarse_start_ms - window_start_ms
        or late_drift > window_end_ms - coarse_end_ms
    ):
        return None
    if any(
        word.get("score_context") == "adjacent_unchanged_word" for word in words
    ):
        return "contextual_low_score_alignment_window"
    return None


def _computed_report(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    words = [word for segment in segments for word in segment["words"]]
    edit_audits = [segment["edit_audit"] for segment in segments]
    input_tokens = [
        token for segment in segments for token in _surface_tokens(str(segment["text"]))
    ]
    lexical_count = sum(bool(_lexeme(token)) for token in input_tokens)
    punctuation_count = len(input_tokens) - lexical_count
    edited_token_count = sum(
        int(audit["edited_token_count"]) for audit in edit_audits
    )
    return {
        "input_segment_count": len(segments),
        "aligned_segment_count": len(segments),
        "input_token_count": len(input_tokens),
        "input_lexical_token_count": lexical_count,
        "input_punctuation_only_token_count": punctuation_count,
        "aligned_word_count": len(words),
        "unaligned_lexical_token_count": 0,
        "unaligned_word_count": 0,
        "synthetic_timing_count": 0,
        "interpolated_timing_count": 0,
        "overlap_violation_count": 0,
        "negative_word_gap_count": 0,
        "missing_alignment_score_count": 0,
        "low_alignment_score_count": 0,
        "overlong_alignment_word_count": 0,
        "outward_drift_violation_count": 0,
        "low_score_edited_token_count": sum(
            word["edit_kind"] in {"inserted", "replaced"}
            and float(word["score"]) < EDITED_TOKEN_MIN_WORD_SCORE
            and word.get("score_context") != AUDIO_REVIEW_SCORE_CONTEXT
            for word in words
        ),
        "unreviewed_deleted_token_count": sum(
            int(audit["unreviewed_deleted_token_count"])
            for audit in edit_audits
        ),
        "unchanged_token_count": sum(
            int(audit["unchanged_token_count"]) for audit in edit_audits
        ),
        "replaced_token_count": sum(
            int(audit["replaced_token_count"]) for audit in edit_audits
        ),
        "inserted_token_count": sum(
            int(audit["inserted_token_count"]) for audit in edit_audits
        ),
        "edited_token_count": edited_token_count,
        "edited_token_ratio": edited_token_count / len(words),
        "deleted_asr_token_count": sum(
            int(audit["deleted_asr_token_count"]) for audit in edit_audits
        ),
        "review_alignment_score_count": sum(
            float(word["score"]) < REVIEW_WORD_SCORE for word in words
        ),
        "minimum_alignment_score": min(float(word["score"]) for word in words),
        "maximum_alignment_word_duration_ms": max(
            int(word["end_ms"]) - int(word["start_ms"]) for word in words
        ),
        "maximum_early_outward_drift_ms": max(
            int(segment["drift_audit"]["early_outward_drift_ms"])
            for segment in segments
        ),
        "maximum_late_outward_drift_ms": max(
            int(segment["drift_audit"]["late_outward_drift_ms"])
            for segment in segments
        ),
    }


def _without_alignment_digest(value: Any) -> Any:
    """Return a JSON-shaped copy with recursive digest fields removed."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_alignment_digest(item)
            for key, item in value.items()
            if key != "alignment_sha256"
        }
    if isinstance(value, (list, tuple)):
        return [_without_alignment_digest(item) for item in value]
    return value


def _alignment_sha256(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _without_alignment_digest(data),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _audio_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_forced_alignment_data(data: Mapping[str, Any]) -> dict[str, int]:
    """Purely validate a persisted forced-alignment object.

    The function performs no imports, model calls, file reads, or mutation.  It
    returns freshly computed counts and raises :class:`ForcedAlignmentError` on
    the first integrity violation.
    """

    if not isinstance(data, Mapping):
        raise ForcedAlignmentError("forced alignment data must be an object")
    if data.get("format_version") != FORCED_ALIGNMENT_FORMAT_VERSION:
        raise ForcedAlignmentError("unsupported forced alignment format_version")
    if data.get("language") != SUPPORTED_LANGUAGE:
        raise ForcedAlignmentError("forced alignment language must be 'tr'")
    if data.get("timing_source") != TIMING_SOURCE:
        raise ForcedAlignmentError("forced alignment timing_source is invalid")
    alignment_sha256 = data.get("alignment_sha256")
    if (
        not isinstance(alignment_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", alignment_sha256)
    ):
        raise ForcedAlignmentError("forced alignment alignment_sha256 is invalid")
    audio_sha256 = data.get("audio_sha256")
    if (
        not isinstance(audio_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", audio_sha256)
    ):
        raise ForcedAlignmentError("forced alignment audio_sha256 is invalid")

    provenance = data.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ForcedAlignmentError("forced alignment provenance must be an object")
    if provenance.get("timing_source") != TIMING_SOURCE:
        raise ForcedAlignmentError("provenance timing_source is invalid")
    if provenance.get("engine") != "whisperx":
        raise ForcedAlignmentError("provenance engine must be 'whisperx'")
    if provenance.get("language") != SUPPORTED_LANGUAGE:
        raise ForcedAlignmentError("provenance language must be 'tr'")
    _require_nonempty_string(provenance.get("device"), "provenance device")
    _require_nonempty_string(provenance.get("model_name"), "provenance model_name")
    persisted_whisperx_version = _require_nonempty_string(
        provenance.get("whisperx_version"), "provenance whisperx_version"
    )
    if persisted_whisperx_version != SUPPORTED_WHISPERX_VERSION:
        raise ForcedAlignmentError(
            "provenance whisperx_version must equal "
            f"{SUPPORTED_WHISPERX_VERSION}"
        )
    if provenance.get("interpolation") not in ("disabled:none", "disabled:ignore"):
        raise ForcedAlignmentError("forced alignment interpolation was not disabled")
    if provenance.get("text_normalization") != ALIGNMENT_TEXT_NORMALIZATION:
        raise ForcedAlignmentError("forced alignment text_normalization is invalid")
    min_word_score = _finite_number(
        provenance.get("min_word_score"), "provenance min_word_score"
    )
    if not DEFAULT_MIN_WORD_SCORE <= min_word_score <= 1.0:
        raise ForcedAlignmentError(
            f"provenance min_word_score must be within "
            f"[{DEFAULT_MIN_WORD_SCORE}, 1]"
        )
    contextual_unchanged_min_word_score = _finite_number(
        provenance.get("contextual_unchanged_min_word_score"),
        "provenance contextual_unchanged_min_word_score",
    )
    if (
        contextual_unchanged_min_word_score
        != CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE
    ):
        raise ForcedAlignmentError(
            "provenance contextual_unchanged_min_word_score must equal "
            f"{CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE}"
        )
    review_word_score = _finite_number(
        provenance.get("review_word_score"), "provenance review_word_score"
    )
    if review_word_score != REVIEW_WORD_SCORE:
        raise ForcedAlignmentError(
            f"provenance review_word_score must equal {REVIEW_WORD_SCORE}"
        )
    edited_token_min_word_score = _finite_number(
        provenance.get("edited_token_min_word_score"),
        "provenance edited_token_min_word_score",
    )
    if edited_token_min_word_score != EDITED_TOKEN_MIN_WORD_SCORE:
        raise ForcedAlignmentError(
            "provenance edited_token_min_word_score must equal "
            f"{EDITED_TOKEN_MIN_WORD_SCORE}"
        )
    max_word_duration_ms = _require_integer(
        provenance.get("max_word_duration_ms"),
        "provenance max_word_duration_ms",
        minimum=1,
    )
    if max_word_duration_ms > DEFAULT_MAX_WORD_DURATION_MS:
        raise ForcedAlignmentError(
            f"provenance max_word_duration_ms cannot exceed "
            f"{DEFAULT_MAX_WORD_DURATION_MS}"
        )
    max_outward_drift_ms = _require_integer(
        provenance.get("max_outward_drift_ms"),
        "provenance max_outward_drift_ms",
    )
    if max_outward_drift_ms > DEFAULT_MAX_OUTWARD_DRIFT_MS:
        raise ForcedAlignmentError(
            f"provenance max_outward_drift_ms cannot exceed "
            f"{DEFAULT_MAX_OUTWARD_DRIFT_MS}"
        )
    overlap_resolution = provenance.get("overlap_resolution")
    if not isinstance(overlap_resolution, Mapping):
        raise ForcedAlignmentError("provenance overlap_resolution must be an object")
    if overlap_resolution.get("policy") != OVERLAP_RESOLUTION_POLICY:
        raise ForcedAlignmentError("provenance overlap resolution policy is invalid")
    for field in (
        "initial_overlap_count",
        "initial_component_count",
        "adaptive_candidate_count",
        "resolved_component_count",
        "acoustic_component_count",
        "final_overlap_count",
    ):
        _require_integer(
            overlap_resolution.get(field),
            f"provenance overlap_resolution {field}",
        )
    if overlap_resolution.get("final_overlap_count") != 0:
        raise ForcedAlignmentError("provenance final overlap count must be zero")
    selected_mode_counts = overlap_resolution.get("selected_mode_counts")
    if not isinstance(selected_mode_counts, Mapping) or any(
        not isinstance(mode, str)
        or not mode
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for mode, count in selected_mode_counts.items()
    ):
        raise ForcedAlignmentError(
            "provenance overlap selected_mode_counts is invalid"
        )
    acoustic_components = overlap_resolution.get("acoustic_components")
    if isinstance(acoustic_components, (str, bytes)) or not isinstance(
        acoustic_components, Sequence
    ):
        raise ForcedAlignmentError(
            "provenance overlap acoustic_components must be a sequence"
        )
    if len(acoustic_components) != overlap_resolution.get("acoustic_component_count"):
        raise ForcedAlignmentError(
            "provenance overlap acoustic component count mismatch"
        )
    if acoustic_components:
        raise ForcedAlignmentError(
            "synthetic acoustic overlap lanes are not independent speaker evidence"
        )

    segments = data.get("segments")
    flat_words = data.get("words")
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
        raise ForcedAlignmentError("forced alignment segments must be a sequence")
    if isinstance(flat_words, (str, bytes)) or not isinstance(flat_words, Sequence):
        raise ForcedAlignmentError("forced alignment words must be a sequence")
    if not segments:
        raise ForcedAlignmentError("forced alignment segments must not be empty")

    active_top_level_words: list[tuple[int, str | None]] = []
    previous_top_level_start = -1
    for position, raw_word in enumerate(flat_words, start=1):
        if not isinstance(raw_word, Mapping):
            raise ForcedAlignmentError("top-level aligned word must be an object")
        word_start = _require_integer(
            raw_word.get("start_ms"), f"top-level word {position} start_ms"
        )
        word_end = _require_integer(
            raw_word.get("end_ms"), f"top-level word {position} end_ms", minimum=1
        )
        if word_start < previous_top_level_start:
            raise ForcedAlignmentError("top-level aligned words are not chronological")
        try:
            current_speaker_id = speaker_id(raw_word, f"top-level word {position}")
        except ValueError as exc:
            raise ForcedAlignmentError(str(exc)) from exc
        active_top_level_words = [
            item for item in active_top_level_words if item[0] > word_start
        ]
        if any(
            overlap_is_unsafe(prior_speaker, current_speaker_id)
            for _, prior_speaker in active_top_level_words
        ):
            raise ForcedAlignmentError("same/unknown-speaker overlapping aligned words")
        active_top_level_words.append((word_end, current_speaker_id))
        previous_top_level_start = word_start

    expected_segment_index = 1
    word_indices: set[int] = set()
    seen_uids: set[str] = set()
    flattened: list[Mapping[str, Any]] = []
    for raw_segment in segments:
        if not isinstance(raw_segment, Mapping):
            raise ForcedAlignmentError("aligned segment must be an object")
        segment_index = _require_integer(
            raw_segment.get("segment_index"), "aligned segment_index", minimum=1
        )
        if segment_index != expected_segment_index:
            raise ForcedAlignmentError("aligned segment_index sequence is not contiguous")
        expected_segment_index += 1
        utterance_uid = _require_nonempty_string(
            raw_segment.get("utterance_uid"), "aligned utterance_uid"
        )
        try:
            segment_speaker_id = speaker_id(raw_segment, utterance_uid)
        except ValueError as exc:
            raise ForcedAlignmentError(str(exc)) from exc
        if utterance_uid in seen_uids:
            raise ForcedAlignmentError(f"duplicate aligned utterance_uid: {utterance_uid}")
        seen_uids.add(utterance_uid)
        text = _require_nonempty_string(raw_segment.get("text"), "aligned text")
        asr_text_value = raw_segment.get("asr_text")
        if not isinstance(asr_text_value, str):
            raise ForcedAlignmentError(f"{utterance_uid} asr_text must be a string")
        asr_text = unicodedata.normalize("NFC", asr_text_value)
        deletion_audio_reviewed = raw_segment.get("deletion_audio_reviewed")
        if not isinstance(deletion_audio_reviewed, bool):
            raise ForcedAlignmentError(
                f"{utterance_uid} deletion_audio_reviewed must be boolean"
            )
        audio_reviewed = raw_segment.get("audio_reviewed", False)
        review_disposition = raw_segment.get(
            "review_disposition", "not_applicable"
        )
        if not isinstance(audio_reviewed, bool):
            raise ForcedAlignmentError(
                f"{utterance_uid} audio_reviewed must be boolean"
            )
        if review_disposition not in {"not_applicable", "confirmed_dialogue"}:
            raise ForcedAlignmentError(
                f"{utterance_uid} review_disposition is invalid for alignment"
            )
        audio_review_supported = (
            audio_reviewed is True
            and review_disposition == "confirmed_dialogue"
        )
        if audio_reviewed != audio_review_supported:
            raise ForcedAlignmentError(
                f"{utterance_uid} audio-review fields are inconsistent"
            )
        if segment_speaker_id and segment_speaker_id.startswith("acoustic-overlap-"):
            raise ForcedAlignmentError(
                f"{utterance_uid} synthetic lane is not independent speaker evidence"
            )
        expected_token_edits, expected_edit_audit = _token_edit_audit(
            asr_text,
            text,
            deletion_audio_reviewed=deletion_audio_reviewed,
        )
        if expected_edit_audit["unreviewed_deleted_token_count"] != 0:
            raise ForcedAlignmentError(
                f"{utterance_uid} contains unreviewed deleted ASR tokens"
            )
        if raw_segment.get("edit_audit") != expected_edit_audit:
            raise ForcedAlignmentError(f"{utterance_uid} lexical edit audit mismatch")
        if raw_segment.get("timing_source") != TIMING_SOURCE:
            raise ForcedAlignmentError(
                f"{utterance_uid} aligned segment timing_source is invalid"
            )
        start_ms = _require_integer(
            raw_segment.get("start_ms"), f"{utterance_uid} aligned start_ms"
        )
        end_ms = _require_integer(
            raw_segment.get("end_ms"), f"{utterance_uid} aligned end_ms", minimum=1
        )
        if end_ms <= start_ms:
            raise ForcedAlignmentError(f"{utterance_uid} aligned interval is invalid")
        window_start = _require_integer(
            raw_segment.get("alignment_window_start_ms"),
            f"{utterance_uid} alignment_window_start_ms",
        )
        window_end = _require_integer(
            raw_segment.get("alignment_window_end_ms"),
            f"{utterance_uid} alignment_window_end_ms",
            minimum=1,
        )
        if window_end <= window_start or start_ms < window_start or end_ms > window_end:
            raise ForcedAlignmentError(
                f"{utterance_uid} aligned segment lies outside its alignment window"
            )
        coarse_start = _require_integer(
            raw_segment.get("coarse_start_ms"),
            f"{utterance_uid} coarse_start_ms",
        )
        coarse_end = _require_integer(
            raw_segment.get("coarse_end_ms"),
            f"{utterance_uid} coarse_end_ms",
            minimum=1,
        )
        if (
            coarse_start < window_start
            or coarse_end > window_end
            or coarse_end <= coarse_start
        ):
            raise ForcedAlignmentError(
                f"{utterance_uid} unpadded coarse bounds are outside its window"
            )

        segment_words = raw_segment.get("words")
        if isinstance(segment_words, (str, bytes)) or not isinstance(
            segment_words, Sequence
        ):
            raise ForcedAlignmentError(f"{utterance_uid} words must be a sequence")
        if not segment_words:
            raise ForcedAlignmentError(f"{utterance_uid} has no aligned lexical words")
        if len(segment_words) != len(expected_token_edits):
            raise ForcedAlignmentError(
                f"{utterance_uid} lexical text/timing coverage mismatch"
            )
        actual_lexemes: list[str] = []
        actual_surfaces: list[str] = []
        for segment_word_offset, word in enumerate(segment_words):
            if not isinstance(word, Mapping):
                raise ForcedAlignmentError(f"{utterance_uid} word must be an object")
            word_index = _require_integer(
                word.get("word_index"), f"{utterance_uid} word_index", minimum=1
            )
            if word_index in word_indices:
                raise ForcedAlignmentError("aligned word_index values are not unique")
            word_indices.add(word_index)
            if word.get("segment_index") != segment_index:
                raise ForcedAlignmentError(f"{utterance_uid} word segment_index mismatch")
            if word.get("segment_id") != segment_index:
                raise ForcedAlignmentError(f"{utterance_uid} word segment_id mismatch")
            if word.get("utterance_uid") != utterance_uid:
                raise ForcedAlignmentError(f"{utterance_uid} word utterance_uid mismatch")
            try:
                word_speaker_id = speaker_id(word, f"{utterance_uid} word")
            except ValueError as exc:
                raise ForcedAlignmentError(str(exc)) from exc
            if word_speaker_id != segment_speaker_id:
                raise ForcedAlignmentError(f"{utterance_uid} word speaker_id mismatch")
            word_text = _require_nonempty_string(
                word.get("text"), f"{utterance_uid} word text"
            )
            lexeme = _lexeme(word_text)
            if not lexeme:
                raise ForcedAlignmentError(
                    f"{utterance_uid} contains a timed punctuation-only word"
                )
            actual_lexemes.append(lexeme)
            actual_surfaces.append(word_text)
            token_edit = expected_token_edits[segment_word_offset]
            if word.get("edit_kind") != token_edit["edit_kind"]:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} edit_kind mismatch"
                )
            if word.get("asr_token_index") != token_edit["asr_token_index"]:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} asr_token_index mismatch"
                )
            if word.get("timing_source") != TIMING_SOURCE:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} timing_source is invalid"
                )
            if word.get("alignment_sha256") != alignment_sha256:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} alignment_sha256 mismatch"
                )
            word_start = _require_integer(
                word.get("start_ms"), f"{utterance_uid} word start_ms"
            )
            word_end = _require_integer(
                word.get("end_ms"), f"{utterance_uid} word end_ms", minimum=1
            )
            if word_end <= word_start:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} interval is invalid"
                )
            if word_end - word_start > max_word_duration_ms:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} exceeds provenance "
                    "max_word_duration_ms"
                )
            duration_fields = DURATION_VAD_FIELDS.intersection(word)
            if duration_fields:
                if duration_fields != DURATION_VAD_FIELDS:
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} has incomplete "
                        "duration VAD evidence"
                    )
                if word.get("duration_context") != DURATION_VAD_CONTEXT:
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} has invalid "
                        "duration_context"
                    )
                if not audio_review_supported:
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} has unreviewed "
                        "duration VAD evidence"
                    )
                raw_start_ms = _require_integer(
                    word.get("raw_start_ms"),
                    f"{utterance_uid} word {word_text!r} raw_start_ms",
                )
                raw_end_ms = _require_integer(
                    word.get("raw_end_ms"),
                    f"{utterance_uid} word {word_text!r} raw_end_ms",
                    minimum=1,
                )
                vad_start_ms = _require_integer(
                    word.get("vad_start_ms"),
                    f"{utterance_uid} word {word_text!r} vad_start_ms",
                )
                vad_end_ms = _require_integer(
                    word.get("vad_end_ms"),
                    f"{utterance_uid} word {word_text!r} vad_end_ms",
                    minimum=1,
                )
                _require_integer(
                    word.get("vad_region_index"),
                    f"{utterance_uid} word {word_text!r} vad_region_index",
                    minimum=1,
                )
                if word.get("vad_source") != "silero_vad":
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} duration VAD source "
                        "is not independent"
                    )
                if (
                    raw_start_ms != word_start
                    or raw_end_ms <= word_end
                    or raw_end_ms - raw_start_ms <= max_word_duration_ms
                    or raw_end_ms > window_end
                    or vad_start_ms > raw_start_ms
                    or raw_start_ms >= vad_end_ms
                    or vad_end_ms != word_end
                ):
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} duration VAD evidence "
                        "does not bind the raw interval"
                    )
                next_word = (
                    segment_words[segment_word_offset + 1]
                    if segment_word_offset + 1 < len(segment_words)
                    else None
                )
                if (
                    not isinstance(next_word, Mapping)
                    or vad_end_ms > _require_integer(
                        next_word.get("start_ms"),
                        f"{utterance_uid} next word start_ms",
                    )
                    or raw_end_ms > int(next_word["start_ms"])
                ):
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} duration VAD boundary "
                        "is not bounded by the next aligned word"
                    )
            if word_start < window_start or word_end > window_end:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} is outside its window"
                )
            score = word.get("score")
            if score is None:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} alignment score is required"
                )
            score = _finite_number(score, f"{utterance_uid} word score")
            if not 0.0 <= score <= 1.0:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} alignment score must be "
                    "within [0, 1]"
                )
            expected_score_context = None
            if score < min_word_score:
                neighbor_scores = [
                    _finite_number(
                        segment_words[position].get("score"),
                        f"{utterance_uid} neighboring word score",
                    )
                    for position in (
                        segment_word_offset - 1,
                        segment_word_offset + 1,
                    )
                    if 0 <= position < len(segment_words)
                    and isinstance(segment_words[position], Mapping)
                ]
                if audio_review_supported:
                    expected_score_context = AUDIO_REVIEW_SCORE_CONTEXT
                elif (
                    score < contextual_unchanged_min_word_score
                    or token_edit["edit_kind"] != "unchanged"
                    or not any(
                        neighbor_score >= min_word_score
                        for neighbor_score in neighbor_scores
                    )
                ):
                    raise ForcedAlignmentError(
                        f"{utterance_uid} word {word_text!r} alignment score is below "
                        "provenance min_word_score without contextual support"
                    )
                else:
                    expected_score_context = "adjacent_unchanged_word"
            if (
                token_edit["edit_kind"] in {"inserted", "replaced"}
                and score < edited_token_min_word_score
            ):
                if audio_review_supported:
                    expected_score_context = AUDIO_REVIEW_SCORE_CONTEXT
                else:
                    raise ForcedAlignmentError(
                        f"{utterance_uid} edited token {word_text!r} alignment score "
                        "is below provenance edited_token_min_word_score"
                    )
            if word.get("score_context") != expected_score_context:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word {word_text!r} has unexpected score_context"
                )
            probability = word.get("probability")
            if probability != score:
                raise ForcedAlignmentError(
                    f"{utterance_uid} word probability/score mismatch"
                )
            flattened.append(word)

        expected_lexemes = [_lexeme(token) for token in _lexical_tokens(text)]
        if actual_lexemes != expected_lexemes:
            raise ForcedAlignmentError(
                f"{utterance_uid} lexical text/timing coverage mismatch"
            )
        if actual_surfaces != _canonical_lexical_surfaces(text):
            raise ForcedAlignmentError(
                f"{utterance_uid} aligned words do not preserve corrected text surface"
            )
        expected_drift_audit = _segment_drift_audit(
            segment_words,
            coarse_start_ms=coarse_start,
            coarse_end_ms=coarse_end,
        )
        if raw_segment.get("drift_audit") != expected_drift_audit:
            raise ForcedAlignmentError(
                f"{utterance_uid} alignment drift audit mismatch"
            )
        exceeds_default_drift = (
            expected_drift_audit["early_outward_drift_ms"] > max_outward_drift_ms
            or expected_drift_audit["late_outward_drift_ms"] > max_outward_drift_ms
        )
        expected_drift_context = _bounded_drift_context(
            segment_words,
            expected_drift_audit,
            window_start_ms=window_start,
            window_end_ms=window_end,
            coarse_start_ms=coarse_start,
            coarse_end_ms=coarse_end,
            max_outward_drift_ms=max_outward_drift_ms,
        )
        if raw_segment.get("drift_context") != expected_drift_context:
            raise ForcedAlignmentError(
                f"{utterance_uid} drift_context does not match acoustic evidence"
            )
        if exceeds_default_drift and expected_drift_context is None:
            raise ForcedAlignmentError(
                f"{utterance_uid} exceeds provenance max_outward_drift_ms"
            )
        if start_ms != segment_words[0]["start_ms"] or end_ms != segment_words[-1]["end_ms"]:
            raise ForcedAlignmentError(
                f"{utterance_uid} segment boundaries do not match its aligned words"
            )

    if word_indices != set(range(1, len(flattened) + 1)):
        raise ForcedAlignmentError("aligned word_index sequence is not contiguous")
    chronological = sorted(
        flattened,
        key=lambda word: (
            int(word["start_ms"]),
            int(word["end_ms"]),
            int(word["word_index"]),
        ),
    )
    if list(flat_words) != chronological:
        raise ForcedAlignmentError("top-level words do not match segment words")
    active_words: list[tuple[int, str | None]] = []
    for word in flat_words:
        word_start = int(word["start_ms"])
        word_end = int(word["end_ms"])
        current_speaker_id = speaker_id(word, "top-level word")
        active_words = [item for item in active_words if item[0] > word_start]
        if any(
            overlap_is_unsafe(prior_speaker, current_speaker_id)
            for _, prior_speaker in active_words
        ):
            raise ForcedAlignmentError("same/unknown-speaker overlapping aligned words")
        active_words.append((word_end, current_speaker_id))

    computed = _computed_report(segments)
    if computed["aligned_word_count"] != computed["input_lexical_token_count"]:
        raise ForcedAlignmentError("not every lexical token has an aligned word")
    report = data.get("report")
    if not isinstance(report, Mapping):
        raise ForcedAlignmentError("forced alignment report must be an object")
    if report.get("alignment_sha256") != alignment_sha256:
        raise ForcedAlignmentError("forced alignment report alignment_sha256 mismatch")
    for key, expected in computed.items():
        if report.get(key) != expected:
            raise ForcedAlignmentError(
                f"forced alignment report {key} mismatch: "
                f"expected {expected}, got {report.get(key)!r}"
            )
    if _alignment_sha256(data) != alignment_sha256:
        raise ForcedAlignmentError("forced alignment alignment_sha256 mismatch")
    try:
        json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ForcedAlignmentError(f"forced alignment data is not JSON-safe: {exc}") from exc
    return computed


def _import_whisperx() -> Any:
    try:
        import whisperx  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in the runtime
        raise ForcedAlignmentError(
            "WhisperX is missing. Install the pinned whisperx==3.8.6 dependency first."
        ) from exc
    return whisperx


def _whisperx_version(module: Any, explicit_version: str | None) -> str:
    if explicit_version is not None:
        return _require_nonempty_string(explicit_version, "whisperx_version")
    module_version = getattr(module, "__version__", None)
    if isinstance(module_version, str) and module_version.strip():
        return module_version.strip()
    try:
        return importlib.metadata.version("whisperx")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ForcedAlignmentError(
            "Cannot determine the installed WhisperX version for provenance"
        ) from exc


def _allows_none(parameter: inspect.Parameter) -> bool:
    if parameter.default is None:
        return True
    annotation = parameter.annotation
    if annotation is inspect.Parameter.empty:
        return False
    if isinstance(annotation, str):
        lowered = annotation.casefold()
        return "none" in lowered or "optional" in lowered
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and type(None) in get_args(annotation)


def _supports_keyword(signature: inspect.Signature, name: str) -> bool:
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _align_call_kwargs(align_function: Any) -> tuple[dict[str, Any], str]:
    try:
        signature = inspect.signature(align_function)
    except (TypeError, ValueError):
        # WhisperX 3.8.6 exposes a normal Python callable.  For unusual
        # compatible wrappers, ``ignore`` is its documented no-interpolation
        # mode and is safer than accepting the default ``nearest``.
        return {"interpolate_method": "ignore"}, "disabled:ignore"

    if not _supports_keyword(signature, "interpolate_method"):
        raise ForcedAlignmentError(
            "WhisperX align API cannot disable interpolation; refusing unsafe timing"
        )
    parameter = signature.parameters.get("interpolate_method")
    interpolation: str | None = None
    mode = "disabled:none"
    if parameter is None or not _allows_none(parameter):
        # WhisperX 3.8.6 uses the explicit ``ignore`` compatibility value.
        interpolation = "ignore"
        mode = "disabled:ignore"
    kwargs: dict[str, Any] = {"interpolate_method": interpolation}
    if _supports_keyword(signature, "return_char_alignments"):
        kwargs["return_char_alignments"] = False
    if _supports_keyword(signature, "print_progress"):
        kwargs["print_progress"] = False
    return kwargs, mode


def _unsafe_word_overlaps(
    words: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    ordered = sorted(
        words,
        key=lambda word: (int(word["start_ms"]), int(word["end_ms"])),
    )
    active: list[Mapping[str, Any]] = []
    overlaps: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for word in ordered:
        word_start = int(word["start_ms"])
        active = [prior for prior in active if int(prior["end_ms"]) > word_start]
        current_speaker = speaker_id(word, "aligned word")
        for prior in active:
            if overlap_is_unsafe(
                speaker_id(prior, "prior aligned word"), current_speaker
            ):
                overlaps.append((prior, word))
        active.append(word)
    return overlaps


def _overlap_components(
    overlaps: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    order: Mapping[str, int],
) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for prior, current in overlaps:
        prior_uid = str(prior["utterance_uid"])
        current_uid = str(current["utterance_uid"])
        if prior_uid == current_uid:
            continue
        graph.setdefault(prior_uid, set()).add(current_uid)
        graph.setdefault(current_uid, set()).add(prior_uid)
    pending = set(graph)
    result: list[list[str]] = []
    while pending:
        root = min(pending, key=order.__getitem__)
        pending.remove(root)
        component = {root}
        stack = [root]
        while stack:
            uid = stack.pop()
            for neighbor in graph[uid]:
                if neighbor in component:
                    continue
                component.add(neighbor)
                pending.discard(neighbor)
                stack.append(neighbor)
        result.append(sorted(component, key=order.__getitem__))
    return sorted(result, key=lambda component: order[component[0]])


def _alignment_candidate(
    coarse: Mapping[str, Any],
    *,
    window_start_ms: int,
    window_end_ms: int,
    segment_index: int,
    align: Any,
    align_model: Any,
    align_metadata: Mapping[str, Any],
    audio: Any,
    device: str,
    call_kwargs: Mapping[str, Any],
    min_word_score: float,
    max_word_duration_ms: int,
    max_outward_drift_ms: int,
    vad_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate = dict(coarse)
    candidate["start_ms"] = window_start_ms
    candidate["end_ms"] = window_end_ms
    raw_result = align(
        [
            {
                "start": window_start_ms / 1000.0,
                "end": window_end_ms / 1000.0,
                "text": _alignment_model_text(str(coarse["text"])),
            }
        ],
        align_model,
        align_metadata,
        audio,
        device,
        **call_kwargs,
    )
    words, _ = _normalize_aligned_words(
        raw_result,
        candidate,
        segment_index=segment_index,
        first_word_index=1,
        min_word_score=min_word_score,
        max_word_duration_ms=max_word_duration_ms,
        vad_regions=vad_regions,
    )
    drift = _segment_drift_audit(
        words,
        coarse_start_ms=int(coarse["coarse_start_ms"]),
        coarse_end_ms=int(coarse["coarse_end_ms"]),
    )
    exceeds = (
        drift["early_outward_drift_ms"] > max_outward_drift_ms
        or drift["late_outward_drift_ms"] > max_outward_drift_ms
    )
    context = _bounded_drift_context(
        words,
        drift,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        coarse_start_ms=int(coarse["coarse_start_ms"]),
        coarse_end_ms=int(coarse["coarse_end_ms"]),
        max_outward_drift_ms=max_outward_drift_ms,
    )
    if exceeds and context is None:
        raise ForcedAlignmentError(
            f"{coarse['utterance_uid']} adaptive alignment exceeds outward drift"
        )
    return words


def _resolve_alignment_overlaps(
    source: Sequence[Mapping[str, Any]],
    independent: Mapping[str, list[dict[str, Any]]],
    *,
    align: Any,
    align_model: Any,
    align_metadata: Mapping[str, Any],
    audio: Any,
    device: str,
    call_kwargs: Mapping[str, Any],
    min_word_score: float,
    max_word_duration_ms: int,
    max_outward_drift_ms: int,
    vad_regions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    by_uid = {str(item["utterance_uid"]): item for item in source}
    order = {str(item["utterance_uid"]): index for index, item in enumerate(source)}
    selected = dict(independent)
    initial_overlaps = _unsafe_word_overlaps(
        [word for words in selected.values() for word in words]
    )
    initial_components = _overlap_components(initial_overlaps, order)
    options: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        uid: [("independent", words)] for uid, words in independent.items()
    }
    selected_mode_by_uid = {uid: "independent" for uid in selected}
    joint_diagnostics: dict[str, Any] = {
        "attempts": 0,
        "align_failures": 0,
        "raw_word_count_mismatches": 0,
        "candidate_validation_failures": 0,
        "candidate_options_added": 0,
        "failure_examples": [],
    }

    def record_joint_failure(message: str) -> None:
        examples = joint_diagnostics["failure_examples"]
        if len(examples) < 5:
            examples.append(message)

    def add_option(uid: str, mode: str, words: list[dict[str, Any]]) -> None:
        signature = tuple(
            (int(word["start_ms"]), int(word["end_ms"])) for word in words
        )
        if any(
            signature
            == tuple(
                (int(word["start_ms"]), int(word["end_ms"]))
                for word in existing
            )
            for _, existing in options[uid]
        ):
            return
        options[uid].append((mode, words))

    def select_component(component_uids: Sequence[str]) -> bool:
        option_sets = [options[uid] for uid in component_uids]
        combination_count = math.prod(len(option_set) for option_set in option_sets)
        if combination_count > MAX_OVERLAP_COMBINATIONS:
            raise ForcedAlignmentError(
                f"overlap candidate budget exceeded: {combination_count} combinations "
                f"for {', '.join(component_uids)}; limit {MAX_OVERLAP_COMBINATIONS}"
            )
        envelope_start = min(
            int(word["start_ms"])
            for option_set in option_sets for _, words in option_set for word in words
        )
        envelope_end = max(
            int(word["end_ms"])
            for option_set in option_sets for _, words in option_set for word in words
        )
        outside_words = [
            word
            for uid, words in selected.items()
            if uid not in component_uids
            for word in words
            if int(word["end_ms"]) > envelope_start
            and int(word["start_ms"]) < envelope_end
        ]
        best: tuple[Any, ...] | None = None
        best_joint_count = -1
        for combination in itertools.product(*option_sets):
            trial_words = [word for _, words in combination for word in words]
            if _unsafe_word_overlaps(trial_words):
                continue
            trial_start = min(int(word["start_ms"]) for word in trial_words)
            trial_end = max(int(word["end_ms"]) for word in trial_words)
            nearby = [
                word
                for word in outside_words
                if int(word["end_ms"]) > trial_start
                and int(word["start_ms"]) < trial_end
            ]
            boundary_overlaps = [
                overlap
                for overlap in _unsafe_word_overlaps(nearby + trial_words)
                if str(overlap[0]["utterance_uid"]) in component_uids
                or str(overlap[1]["utterance_uid"]) in component_uids
            ]
            if boundary_overlaps:
                continue
            joint_count = sum(mode == "joint" for mode, _ in combination)
            if joint_count > best_joint_count:
                best = combination
                best_joint_count = joint_count
        if best is None:
            return False
        for uid, (mode, words) in zip(component_uids, best):
            selected[uid] = words
            selected_mode_by_uid[uid] = mode
        return True

    adaptive_candidate_count = 0

    def add_joint_component_options(
        target_uids: Sequence[str],
        context_uids: Sequence[str],
        component_index: str,
    ) -> None:
        nonlocal adaptive_candidate_count
        joint_diagnostics["attempts"] += 1
        target_uid_set = set(target_uids)
        group = [by_uid[uid] for uid in context_uids]
        joint_start = min(int(item["start_ms"]) for item in group)
        joint_end = max(int(item["end_ms"]) for item in group)
        joint_text = " ".join(str(item["text"]) for item in group)
        try:
            joint_raw = align(
                [
                    {
                        "start": joint_start / 1000.0,
                        "end": joint_end / 1000.0,
                        "text": _alignment_model_text(joint_text),
                    }
                ],
                align_model,
                align_metadata,
                audio,
                device,
                **call_kwargs,
            )
            raw_words = _raw_aligned_words(
                joint_raw, f"overlap-component-{component_index}"
            )
            expected_count = sum(
                len(_canonical_lexical_surfaces(str(item["text"])))
                for item in group
            )
            if len(raw_words) != expected_count:
                joint_diagnostics["raw_word_count_mismatches"] += 1
                record_joint_failure(
                    f"{component_index}: expected {expected_count} joint words, "
                    f"received {len(raw_words)}"
                )
                return
            offset = 0
            for item in group:
                uid = str(item["utterance_uid"])
                word_count = len(_canonical_lexical_surfaces(str(item["text"])))
                raw_word_slice = raw_words[offset : offset + word_count]
                offset += word_count
                if uid not in target_uid_set:
                    continue
                try:
                    expanded = dict(item)
                    expanded["start_ms"] = joint_start
                    expanded["end_ms"] = joint_end
                    words, _ = _normalize_aligned_words(
                        {"word_segments": raw_word_slice},
                        expanded,
                        segment_index=order[uid] + 1,
                        first_word_index=1,
                        min_word_score=min_word_score,
                        max_word_duration_ms=max_word_duration_ms,
                        vad_regions=vad_regions,
                    )
                    drift = _segment_drift_audit(
                        words,
                        coarse_start_ms=int(item["coarse_start_ms"]),
                        coarse_end_ms=int(item["coarse_end_ms"]),
                    )
                    context = _bounded_drift_context(
                        words,
                        drift,
                        window_start_ms=joint_start,
                        window_end_ms=joint_end,
                        coarse_start_ms=int(item["coarse_start_ms"]),
                        coarse_end_ms=int(item["coarse_end_ms"]),
                        max_outward_drift_ms=max_outward_drift_ms,
                    )
                    if (
                        drift["early_outward_drift_ms"] > max_outward_drift_ms
                        or drift["late_outward_drift_ms"] > max_outward_drift_ms
                    ) and context is None:
                        continue
                except Exception as exc:
                    joint_diagnostics["candidate_validation_failures"] += 1
                    record_joint_failure(f"{component_index}/{uid}: {exc}")
                    continue
                before = len(options[uid])
                add_option(uid, "joint", words)
                added = len(options[uid]) - before
                adaptive_candidate_count += added
                joint_diagnostics["candidate_options_added"] += added
        except Exception as exc:
            joint_diagnostics["align_failures"] += 1
            record_joint_failure(f"{component_index}: {exc}")
            return

    def contextual_component_uids(
        component_uids: Sequence[str], *, radius: int = 2
    ) -> list[str]:
        positions = [order[uid] for uid in component_uids]
        start = max(0, min(positions) - radius)
        end = min(len(source), max(positions) + radius + 1)
        return [str(item["utterance_uid"]) for item in source[start:end]]

    for component_index, component_uids in enumerate(initial_components, start=1):
        group = [by_uid[uid] for uid in component_uids]
        add_joint_component_options(
            component_uids, component_uids, str(component_index)
        )
        if select_component(component_uids):
            continue
        for item in group:
            uid = str(item["utterance_uid"])
            for padding_ms in (400, 300, 200, 100, 0):
                window_start = max(
                    int(item["start_ms"]),
                    int(item["coarse_start_ms"]) - padding_ms,
                )
                window_end = min(
                    int(item["end_ms"]),
                    int(item["coarse_end_ms"]) + padding_ms,
                )
                try:
                    words = _alignment_candidate(
                        item,
                        window_start_ms=window_start,
                        window_end_ms=window_end,
                        segment_index=order[uid] + 1,
                        align=align,
                        align_model=align_model,
                        align_metadata=align_metadata,
                        audio=audio,
                        device=device,
                        call_kwargs=call_kwargs,
                        min_word_score=min_word_score,
                        max_word_duration_ms=max_word_duration_ms,
                        max_outward_drift_ms=max_outward_drift_ms,
                        vad_regions=vad_regions,
                    )
                except Exception:
                    continue
                add_option(uid, f"padding-{padding_ms}", words)
                adaptive_candidate_count += 1
        select_component(component_uids)

    anchors: list[int] = []
    for item in source:
        anchor = (int(item["coarse_start_ms"]) + int(item["coarse_end_ms"])) // 2
        if anchors:
            anchor = max(anchor, anchors[-1] + 2)
        anchors.append(anchor)
    boundaries = [
        (anchors[position - 1] + anchors[position]) // 2
        for position in range(1, len(anchors))
    ]
    residual_before_partition = _unsafe_word_overlaps(
        [word for words in selected.values() for word in words]
    )
    residual_components = _overlap_components(residual_before_partition, order)
    for component_index, seed_uids in enumerate(residual_components, start=1):
        seen_contexts: set[tuple[str, ...]] = set()
        for radius in (1, 2, 4, 8):
            context_uids = contextual_component_uids(seed_uids, radius=radius)
            context_key = tuple(context_uids)
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)
            add_joint_component_options(
                context_uids,
                context_uids,
                f"residual-{component_index}-radius-{radius}",
            )
        for uid in sorted(seed_uids, key=order.__getitem__):
            position = order[uid]
            for start, end, label in (
                (max(0, position - 1), position + 1, "left"),
                (max(0, position - 1), min(len(source), position + 2), "center"),
                (position, min(len(source), position + 2), "right"),
            ):
                context_uids = [
                    str(item["utterance_uid"]) for item in source[start:end]
                ]
                context_key = tuple(context_uids)
                if len(context_uids) < 2 or context_key in seen_contexts:
                    continue
                seen_contexts.add(context_key)
                add_joint_component_options(
                    context_uids,
                    context_uids,
                    f"residual-{component_index}-{order[uid] + 1}-{label}",
                )
    conflict_uids = {
        str(word["utterance_uid"])
        for overlap in residual_before_partition
        for word in overlap
    }
    partition_uids = set(conflict_uids)
    for component_uids in residual_components:
        partition_uids.update(contextual_component_uids(component_uids))
    for uid in sorted(partition_uids, key=order.__getitem__):
        position = order[uid]
        item = by_uid[uid]
        window_start = max(
            int(item["start_ms"]), boundaries[position - 1] if position else 0
        )
        window_end = min(
            int(item["end_ms"]),
            boundaries[position] if position < len(boundaries) else int(item["end_ms"]),
        )
        if window_end <= window_start:
            continue
        try:
            words = _alignment_candidate(
                item,
                window_start_ms=window_start,
                window_end_ms=window_end,
                segment_index=position + 1,
                align=align,
                align_model=align_model,
                align_metadata=align_metadata,
                audio=audio,
                device=device,
                call_kwargs=call_kwargs,
                min_word_score=min_word_score,
                max_word_duration_ms=max_word_duration_ms,
                max_outward_drift_ms=max_outward_drift_ms,
                vad_regions=vad_regions,
            )
        except Exception:
            continue
        add_option(uid, "partition", words)
        adaptive_candidate_count += 1

    resolved_component_count = 0
    for seed_uids in residual_components:
        component_uids = contextual_component_uids(seed_uids)
        option_sets = [options[uid] for uid in component_uids]
        if math.prod(len(option_set) for option_set in option_sets) > MAX_OVERLAP_COMBINATIONS:
            component_uids = list(seed_uids)
            option_sets = [options[uid] for uid in component_uids]
        combination_count = math.prod(len(option_set) for option_set in option_sets)
        if combination_count > MAX_OVERLAP_COMBINATIONS:
            raise ForcedAlignmentError(
                f"overlap candidate budget exceeded: {combination_count} combinations "
                f"for {', '.join(component_uids)}; limit {MAX_OVERLAP_COMBINATIONS}"
            )
        envelope_start = min(
            int(word["start_ms"])
            for option_set in option_sets
            for _, words in option_set
            for word in words
        )
        envelope_end = max(
            int(word["end_ms"])
            for option_set in option_sets
            for _, words in option_set
            for word in words
        )
        outside_words = [
            word
            for uid, words in selected.items()
            if uid not in component_uids
            for word in words
            if int(word["end_ms"]) > envelope_start
            and int(word["start_ms"]) < envelope_end
        ]
        best: tuple[Any, ...] | None = None
        best_score: tuple[int, int, int] | None = None
        for combination in itertools.product(*option_sets):
            trial_words = [word for _, words in combination for word in words]
            if _unsafe_word_overlaps(trial_words):
                continue
            trial_start = min(int(word["start_ms"]) for word in trial_words)
            trial_end = max(int(word["end_ms"]) for word in trial_words)
            nearby = [
                word
                for word in outside_words
                if int(word["end_ms"]) > trial_start
                and int(word["start_ms"]) < trial_end
            ]
            boundary_overlaps = [
                overlap
                for overlap in _unsafe_word_overlaps(nearby + trial_words)
                if str(overlap[0]["utterance_uid"]) in component_uids
                or str(overlap[1]["utterance_uid"]) in component_uids
            ]
            if boundary_overlaps:
                continue
            score = (
                sum(mode == "joint" for mode, _ in combination),
                sum(mode == "independent" for mode, _ in combination),
                sum(mode.startswith("padding-") for mode, _ in combination),
            )
            if best_score is None or score > best_score:
                best = combination
                best_score = score
        if best is None:
            continue
        for uid, (mode, words) in zip(component_uids, best):
            selected[uid] = words
            selected_mode_by_uid[uid] = mode
        resolved_component_count += 1

    assigned_speakers: dict[str, str] = {}
    acoustic_components: list[dict[str, Any]] = []

    final_overlaps = _unsafe_word_overlaps(
        [word for words in selected.values() for word in words]
    )
    if final_overlaps:
        prior, current = final_overlaps[0]
        pairs = sorted({
            (str(left["utterance_uid"]), str(right["utterance_uid"]))
            for left, right in final_overlaps
        })
        raise ForcedAlignmentError(
            "same/unknown-speaker alignment overlap remains between "
            f"{prior['utterance_uid']} and {current['utterance_uid']}; "
            f"joint diagnostics: {joint_diagnostics}; "
            f"all {len(pairs)} unresolved UID pairs: {pairs}"
        )
    return selected, assigned_speakers, {
        "policy": OVERLAP_RESOLUTION_POLICY,
        "initial_overlap_count": len(initial_overlaps),
        "initial_component_count": len(initial_components),
        "adaptive_candidate_count": adaptive_candidate_count,
        "resolved_component_count": resolved_component_count,
        "acoustic_component_count": len(acoustic_components),
        "acoustic_components": acoustic_components,
        "selected_mode_counts": {
            mode: list(selected_mode_by_uid.values()).count(mode)
            for mode in sorted(set(selected_mode_by_uid.values()))
        },
        "final_overlap_count": 0,
    }


def _model_state_sha256(model, metadata):
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise ForcedAlignmentError("alignment checkpoint requires actual model weights")
    state = state_dict()
    if not state:
        raise ForcedAlignmentError("alignment model state is empty")
    config = getattr(model, "config", None)
    hasher = hashlib.sha256()
    hasher.update(digest({
        "metadata": dict(metadata),
        "config": config.to_dict() if config is not None else None,
    }).encode("ascii"))
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        hasher.update(digest({"name": name, "dtype": str(tensor.dtype),
                              "shape": list(tensor.shape)}).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def align_corrected_segments(
    audio_path: str | Path,
    coarse_segments: Sequence[Mapping[str, Any]],
    *,
    language: str = SUPPORTED_LANGUAGE,
    device: str = "cuda",
    model_name: str = DEFAULT_TURKISH_ALIGNMENT_MODEL,
    min_word_score: float = DEFAULT_MIN_WORD_SCORE,
    max_word_duration_ms: int = DEFAULT_MAX_WORD_DURATION_MS,
    max_outward_drift_ms: int = DEFAULT_MAX_OUTWARD_DRIFT_MS,
    vad_regions: Sequence[Mapping[str, Any]] = (),
    whisperx_module: Any | None = None,
    whisperx_version: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Force-align corrected Turkish text and return strict JSON-ready data.

    The model and audio are loaded once.  Each coarse utterance is aligned in a
    separate WhisperX call so sentence splitting cannot detach timestamps from
    its stable ``utterance_uid``.  The source audio is hashed before it is
    loaded and again after all model calls; any mid-run byte change rejects the
    entire result before an alignment digest can be issued.
    """

    if language != SUPPORTED_LANGUAGE:
        raise ForcedAlignmentError(
            f"unsupported alignment language {language!r}; only 'tr' is supported"
        )
    model_name = _require_nonempty_string(model_name, "model_name")
    device = _require_nonempty_string(device, "device")
    min_word_score = _finite_number(min_word_score, "min_word_score")
    if not DEFAULT_MIN_WORD_SCORE <= min_word_score <= 1.0:
        raise ForcedAlignmentError(
            f"min_word_score must be within [{DEFAULT_MIN_WORD_SCORE}, 1]"
        )
    max_word_duration_ms = _require_integer(
        max_word_duration_ms, "max_word_duration_ms", minimum=1
    )
    if max_word_duration_ms > DEFAULT_MAX_WORD_DURATION_MS:
        raise ForcedAlignmentError(
            f"max_word_duration_ms cannot exceed {DEFAULT_MAX_WORD_DURATION_MS}"
        )
    max_outward_drift_ms = _require_integer(
        max_outward_drift_ms, "max_outward_drift_ms"
    )
    if max_outward_drift_ms > DEFAULT_MAX_OUTWARD_DRIFT_MS:
        raise ForcedAlignmentError(
            f"max_outward_drift_ms cannot exceed {DEFAULT_MAX_OUTWARD_DRIFT_MS}"
        )
    trusted_vad_regions = _trusted_vad_regions(vad_regions)
    source = validate_coarse_segments(coarse_segments)
    path = Path(audio_path)
    if not path.is_file():
        raise ForcedAlignmentError(f"alignment audio does not exist: {path}")
    audio_sha256 = _audio_sha256(path)

    whisperx = whisperx_module if whisperx_module is not None else _import_whisperx()
    version = _whisperx_version(whisperx, whisperx_version)
    if version != SUPPORTED_WHISPERX_VERSION:
        raise ForcedAlignmentError(
            f"WhisperX {SUPPORTED_WHISPERX_VERSION} is required; got {version!r}"
        )
    load_audio = getattr(whisperx, "load_audio", None)
    load_align_model = getattr(whisperx, "load_align_model", None)
    align = getattr(whisperx, "align", None)
    if not callable(load_audio) or not callable(load_align_model) or not callable(align):
        raise ForcedAlignmentError(
            "WhisperX module must expose load_audio, load_align_model, and align"
        )

    try:
        audio = load_audio(str(path))
    except Exception as exc:
        raise ForcedAlignmentError(f"WhisperX could not load alignment audio: {exc}") from exc
    try:
        loaded = load_align_model(
            language_code=language,
            device=device,
            model_name=model_name,
        )
    except Exception as exc:
        raise ForcedAlignmentError(
            f"WhisperX could not load Turkish alignment model {model_name!r}: {exc}"
        ) from exc
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ForcedAlignmentError(
            "WhisperX load_align_model must return (model, metadata)"
        )
    align_model, align_metadata = loaded
    if not isinstance(align_metadata, Mapping):
        raise ForcedAlignmentError("WhisperX alignment metadata must be an object")
    metadata_language = align_metadata.get("language")
    if metadata_language != language:
        raise ForcedAlignmentError(
            f"WhisperX alignment model language mismatch: {metadata_language!r}"
        )

    call_kwargs, interpolation_mode = _align_call_kwargs(align)
    model_state_sha256 = None
    if checkpoint_dir is not None:
        model_state_sha256 = _model_state_sha256(align_model, align_metadata)
        journal = UnitJournal(Path(checkpoint_dir), {
            "audio_sha256": audio_sha256,
            "model_name": model_name,
            "model_state_sha256": model_state_sha256,
            "whisperx_version": version,
            "device": device,
            "language": language,
            "interpolation": interpolation_mode,
            "code_sha256": _audio_sha256(Path(__file__)),
            "dependencies_sha256": _audio_sha256(
                Path(__file__).resolve().parents[3] / "requirements.lock"
            ),
        })
        model_align = align

        def align(transcript, model, metadata, audio, device, **kwargs):
            key = digest({"transcript": transcript, "kwargs": kwargs})
            cached = journal.read(key)
            if cached is not None:
                return cached
            result = model_align(transcript, model, metadata, audio, device, **kwargs)
            _raw_aligned_words(result, "alignment checkpoint")
            journal.write(key, result)
            return result

    aligned_segments: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []
    independent_by_uid: dict[str, list[dict[str, Any]]] = {}
    raw_punctuation_only_count = 0
    next_word_index = 1
    alignment_issues = []
    for segment_index, coarse in enumerate(source, start=1):
        if segment_index > 1:
            mark_work_progress("forced_alignment", completed=segment_index - 1)
        transcript = [
            {
                "start": coarse["start_ms"] / 1000.0,
                "end": coarse["end_ms"] / 1000.0,
                "text": _alignment_model_text(str(coarse["text"])),
            }
        ]
        try:
            raw_result = align(
                transcript,
                align_model,
                align_metadata,
                audio,
                device,
                **call_kwargs,
            )
        except Exception as exc:
            raise ForcedAlignmentError(
                f"WhisperX alignment failed for {coarse['utterance_uid']}: {exc}"
            ) from exc
        try:
            words, punctuation_count = _normalize_aligned_words(
                raw_result,
                coarse,
                segment_index=segment_index,
                first_word_index=next_word_index,
                min_word_score=min_word_score,
                max_word_duration_ms=max_word_duration_ms,
                vad_regions=trusted_vad_regions,
            )
        except ForcedAlignmentError as exc:
            alignment_issues.append(f"{coarse['utterance_uid']}: {exc}")
            continue
        raw_punctuation_only_count += punctuation_count
        next_word_index += len(words)
        independent_by_uid[str(coarse["utterance_uid"])] = words
        flat_words.extend(words)
        drift_audit = _segment_drift_audit(
            words,
            coarse_start_ms=coarse["coarse_start_ms"],
            coarse_end_ms=coarse["coarse_end_ms"],
        )
        exceeds_default_drift = (
            drift_audit["early_outward_drift_ms"] > max_outward_drift_ms
            or drift_audit["late_outward_drift_ms"] > max_outward_drift_ms
        )
        drift_context = _bounded_drift_context(
            words,
            drift_audit,
            window_start_ms=coarse["start_ms"],
            window_end_ms=coarse["end_ms"],
            coarse_start_ms=coarse["coarse_start_ms"],
            coarse_end_ms=coarse["coarse_end_ms"],
            max_outward_drift_ms=max_outward_drift_ms,
        )
        _, edit_audit = _token_edit_audit(
            str(coarse["asr_text"]),
            str(coarse["text"]),
            deletion_audio_reviewed=bool(coarse["deletion_audio_reviewed"]),
        )
        if exceeds_default_drift and drift_context is None:
            alignment_issues.append(
                f"{coarse['utterance_uid']} exceeds max_outward_drift_ms "
                f"{max_outward_drift_ms}: {drift_audit}"
            )
            continue
        aligned_segments.append(
            {
                "segment_index": segment_index,
                "utterance_uid": coarse["utterance_uid"],
                "start_ms": words[0]["start_ms"],
                "end_ms": words[-1]["end_ms"],
                "alignment_window_start_ms": coarse["start_ms"],
                "alignment_window_end_ms": coarse["end_ms"],
                "coarse_start_ms": coarse["coarse_start_ms"],
                "coarse_end_ms": coarse["coarse_end_ms"],
                "drift_audit": drift_audit,
                **(
                    {"drift_context": drift_context}
                    if drift_context is not None
                    else {}
                ),
                "text": coarse["text"],
                "asr_text": coarse["asr_text"],
                "deletion_audio_reviewed": coarse["deletion_audio_reviewed"],
                "audio_reviewed": coarse["audio_reviewed"],
                "review_disposition": coarse["review_disposition"],
                "edit_audit": edit_audit,
                "timing_source": TIMING_SOURCE,
                "words": words,
                **(
                    {"speaker_id": coarse["speaker_id"]}
                    if "speaker_id" in coarse
                    else {}
                ),
            }
        )

    if alignment_issues:
        raise ForcedAlignmentError(
            f"independent alignment failed for {len(alignment_issues)} utterances:\n"
            + "\n".join(alignment_issues)
        )

    selected_by_uid, assigned_speakers, overlap_resolution = (
        _resolve_alignment_overlaps(
            source,
            independent_by_uid,
            align=align,
            align_model=align_model,
            align_metadata=align_metadata,
            audio=audio,
            device=device,
            call_kwargs=call_kwargs,
            min_word_score=min_word_score,
            max_word_duration_ms=max_word_duration_ms,
            max_outward_drift_ms=max_outward_drift_ms,
            vad_regions=trusted_vad_regions,
        )
    )
    aligned_segments = []
    flat_words = []
    for segment_index, coarse in enumerate(source, start=1):
        if segment_index > 1:
            mark_work_progress("forced_alignment", completed=segment_index - 1)
        utterance_uid = str(coarse["utterance_uid"])
        words = selected_by_uid[utterance_uid]
        segment_speaker = coarse.get("speaker_id", assigned_speakers.get(utterance_uid))
        if segment_speaker is not None:
            for word in words:
                word["speaker_id"] = segment_speaker
        drift_audit = _segment_drift_audit(
            words,
            coarse_start_ms=int(coarse["coarse_start_ms"]),
            coarse_end_ms=int(coarse["coarse_end_ms"]),
        )
        alignment_window_start_ms = min(
            int(coarse["start_ms"]), int(words[0]["start_ms"])
        )
        alignment_window_end_ms = max(
            int(coarse["end_ms"]), int(words[-1]["end_ms"])
        )
        drift_context = _bounded_drift_context(
            words,
            drift_audit,
            window_start_ms=alignment_window_start_ms,
            window_end_ms=alignment_window_end_ms,
            coarse_start_ms=int(coarse["coarse_start_ms"]),
            coarse_end_ms=int(coarse["coarse_end_ms"]),
            max_outward_drift_ms=max_outward_drift_ms,
        )
        exceeds_default_drift = (
            drift_audit["early_outward_drift_ms"] > max_outward_drift_ms
            or drift_audit["late_outward_drift_ms"] > max_outward_drift_ms
        )
        if exceeds_default_drift and drift_context is None:
            raise ForcedAlignmentError(
                f"{utterance_uid} exceeds max_outward_drift_ms "
                f"{max_outward_drift_ms}: {drift_audit}"
            )
        _, edit_audit = _token_edit_audit(
            str(coarse["asr_text"]),
            str(coarse["text"]),
            deletion_audio_reviewed=bool(coarse["deletion_audio_reviewed"]),
        )
        aligned_segments.append(
            {
                "segment_index": segment_index,
                "utterance_uid": utterance_uid,
                "start_ms": words[0]["start_ms"],
                "end_ms": words[-1]["end_ms"],
                "alignment_window_start_ms": alignment_window_start_ms,
                "alignment_window_end_ms": alignment_window_end_ms,
                "coarse_start_ms": coarse["coarse_start_ms"],
                "coarse_end_ms": coarse["coarse_end_ms"],
                "drift_audit": drift_audit,
                **(
                    {"drift_context": drift_context}
                    if drift_context is not None
                    else {}
                ),
                "text": coarse["text"],
                "asr_text": coarse["asr_text"],
                "deletion_audio_reviewed": coarse["deletion_audio_reviewed"],
                "audio_reviewed": coarse["audio_reviewed"],
                "review_disposition": coarse["review_disposition"],
                "edit_audit": edit_audit,
                "timing_source": TIMING_SOURCE,
                "words": words,
                **(
                    {"speaker_id": segment_speaker}
                    if segment_speaker is not None
                    else {}
                ),
            }
        )
        flat_words.extend(words)

    flat_words.sort(
        key=lambda word: (
            int(word["start_ms"]),
            int(word["end_ms"]),
            int(word["word_index"]),
        )
    )
    for word_index, word in enumerate(flat_words, start=1):
        word["word_index"] = word_index

    if _audio_sha256(path) != audio_sha256:
        raise ForcedAlignmentError(
            "alignment audio changed while forced alignment was running"
        )

    report = _computed_report(aligned_segments)
    report["raw_punctuation_only_word_count"] = raw_punctuation_only_count
    data: dict[str, Any] = {
        "format_version": FORCED_ALIGNMENT_FORMAT_VERSION,
        "language": language,
        "audio_sha256": audio_sha256,
        "timing_source": TIMING_SOURCE,
        "provenance": {
            "timing_source": TIMING_SOURCE,
            "engine": "whisperx",
            "whisperx_version": version,
            "model_name": model_name,
            "model_type": align_metadata.get("type"),
            "model_state_sha256": model_state_sha256,
            "language": language,
            "device": device,
            "interpolation": interpolation_mode,
            "text_normalization": ALIGNMENT_TEXT_NORMALIZATION,
            "min_word_score": min_word_score,
            "contextual_unchanged_min_word_score": (
                CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE
            ),
            "review_word_score": REVIEW_WORD_SCORE,
            "edited_token_min_word_score": EDITED_TOKEN_MIN_WORD_SCORE,
            "max_word_duration_ms": max_word_duration_ms,
            "max_outward_drift_ms": max_outward_drift_ms,
            "overlap_resolution": overlap_resolution,
        },
        "segments": aligned_segments,
        "words": flat_words,
        "report": report,
    }
    alignment_sha256 = _alignment_sha256(data)
    data["alignment_sha256"] = alignment_sha256
    report["alignment_sha256"] = alignment_sha256
    for word in flat_words:
        word["alignment_sha256"] = alignment_sha256
    validate_forced_alignment_data(data)
    return data


__all__ = [
    "ALIGNMENT_TEXT_NORMALIZATION",
    "AUDIO_REVIEW_SCORE_CONTEXT",
    "CONTEXTUAL_UNCHANGED_WORD_MIN_SCORE",
    "DEFAULT_MAX_OUTWARD_DRIFT_MS",
    "DEFAULT_MAX_WORD_DURATION_MS",
    "DEFAULT_MIN_WORD_SCORE",
    "DEFAULT_TURKISH_ALIGNMENT_MODEL",
    "EDITED_TOKEN_MIN_WORD_SCORE",
    "FORCED_ALIGNMENT_FORMAT_VERSION",
    "ForcedAlignmentError",
    "REVIEW_WORD_SCORE",
    "SUPPORTED_LANGUAGE",
    "SUPPORTED_WHISPERX_VERSION",
    "TIMING_SOURCE",
    "align_corrected_segments",
    "correction_deletes_lexical_tokens",
    "validate_coarse_segments",
    "validate_forced_alignment_data",
]
