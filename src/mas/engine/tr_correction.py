"""Atomic Turkish-correction interchange for the subtitle pipeline.

The correction pass happens before word alignment and subtitle segmentation.
It therefore operates on coarse utterances, never on final subtitle blocks.  A
correction output must echo every input evidence field exactly; only
``tr_corrected``, ``non_dialogue``, ``review_required`` and ``note`` are
editable.  This module deliberately has no dependency on the V1 schema.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


FORMAT_VERSION = "2.0"
INPUT_FORMAT = "muhtemel-ask-tr-correction-input"
OUTPUT_FORMAT = "muhtemel-ask-tr-correction-output"
DEFAULT_BATCH_SIZE = 250

IMMUTABLE_UTTERANCE_FIELDS: tuple[str, ...] = (
    "utterance_uid",
    "utterance_index",
    "coarse_start_ms",
    "coarse_end_ms",
    "asr_text",
    "youtube_text",
    "context_before",
    "context_after",
    "risk_flags",
    "asr_audit",
)

ASR_AUDIT_FIELDS = frozenset(
    {"avg_logprob", "no_speech_prob", "compression_ratio", "temperature"}
)

EDITABLE_CORRECTION_FIELDS: tuple[str, ...] = (
    "tr_corrected",
    "non_dialogue",
    "review_required",
    "audio_reviewed",
    "review_disposition",
    "note",
)

REVIEW_DISPOSITIONS = frozenset(
    {
        "not_applicable",
        "pending_audio_review",
        "confirmed_dialogue",
        "reviewed_non_dialogue",
        "discarded_asr_hallucination",
    }
)

OUTPUT_RECORD_FIELDS = frozenset(
    (*IMMUTABLE_UTTERANCE_FIELDS, *EDITABLE_CORRECTION_FIELDS)
)

SPEECH_HOLE_FIELDS = frozenset(
    {
        "hole_uid",
        "hole_index",
        "start_ms",
        "end_ms",
        "clip_start_ms",
        "clip_end_ms",
        "reason",
        "context_before",
        "context_after",
        "risk_flags",
        "audio_member",
        "audio_sha256",
        "audio_size_bytes",
    }
)

ASR_HALLUCINATION_FIELDS = frozenset(
    {
        "candidate_uid",
        "candidate_index",
        "utterance_uid",
        "utterance_index",
        "start_ms",
        "end_ms",
        "clip_start_ms",
        "clip_end_ms",
        "reason",
        "asr_text",
        "youtube_text",
        "context_before",
        "context_after",
        "risk_flags",
        "asr_audit",
        "audio_member",
        "audio_sha256",
        "audio_size_bytes",
    }
)

INPUT_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "episode",
        "input_sha256",
        "utterance_count",
        "speech_hole_count",
        "asr_hallucination_count",
        "batch_count",
        "batches",
        "speech_hole_audio",
        "asr_hallucination_audio",
        "file_sha256",
    }
)
INPUT_BATCH_DESCRIPTOR_FIELDS = frozenset(
    {
        "input_file",
        "output_file",
        "utterance_count",
        "first_utterance_uid",
        "last_utterance_uid",
        "sha256",
    }
)
SPEECH_HOLE_AUDIO_DESCRIPTOR_FIELDS = frozenset(
    {"hole_uid", "member", "sha256", "size_bytes"}
)
ASR_HALLUCINATION_AUDIO_DESCRIPTOR_FIELDS = frozenset(
    {"candidate_uid", "utterance_uid", "member", "sha256", "size_bytes"}
)
OUTPUT_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "episode",
        "input_sha256",
        "output_sha256",
        "utterance_count",
        "batch_count",
        "batches",
    }
)
OUTPUT_BATCH_DESCRIPTOR_FIELDS = frozenset(
    {
        "output_file",
        "utterance_count",
        "first_utterance_uid",
        "last_utterance_uid",
        "sha256",
    }
)

# Bounds are generous for a feature-length episode but make all decompression
# bounded before any member is read.
MAX_ZIP_MEMBERS = 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250.0
MIN_RATIO_CHECK_BYTES = 1024 * 1024
MAX_SPEECH_HOLE_AUDIO_FILES = 192
MAX_SPEECH_HOLE_AUDIO_BYTES = 16 * 1024 * 1024
MAX_SPEECH_HOLE_AUDIO_TOTAL_BYTES = 128 * 1024 * 1024
# Keep heuristic detector candidates at the original conservative bound.
# Mandatory structural reviews and explicit requests remain governed by the
# aggregate file and byte bounds without expanding the detector budget.
MAX_AUTOMATIC_ASR_HALLUCINATION_AUDIO_FILES = 192
MAX_EXPLICIT_ASR_HALLUCINATION_AUDIO_FILES = 832
MAX_ASR_HALLUCINATION_AUDIO_FILES = (
    MAX_AUTOMATIC_ASR_HALLUCINATION_AUDIO_FILES
    + MAX_EXPLICIT_ASR_HALLUCINATION_AUDIO_FILES
)
MAX_ASR_HALLUCINATION_AUDIO_BYTES = 16 * 1024 * 1024
MAX_ASR_HALLUCINATION_AUDIO_TOTAL_BYTES = 128 * 1024 * 1024
MAX_REVIEW_AUDIO_FILES = 1000
MAX_REVIEW_AUDIO_TOTAL_BYTES = 128 * 1024 * 1024


DEFAULT_INSTRUCTIONS = """# Turkish correction pass

Correct the Turkish transcript using the ASR text, YouTube text and context as
evidence. Do not translate it and do not change, remove, split, merge or reorder
records. Copy every immutable field exactly.

This is a text-only Turkish correction pass. Do not open, inspect, transcribe or
listen to any WAV member. The next Colab stage owns all audio decisions. Do not
browse or search the web. Do not search Hugging Face, GitHub or model
repositories; download or install an ASR model; look for a transcription plugin;
or call any external transcription or translation service. Perform the Turkish
language correction yourself using only the JSON text evidence in this ZIP.
The only permitted external connector is Google Drive for reading the exact
input ZIP and writing the exact output ZIP. Local ZIP/JSON processing is allowed,
but network access, package installation and model/tool discovery are forbidden.

Return one ZIP named exactly like the input episode with the suffix
`_TR_TEXT_CORRECTED.zip`. For every record linked to `speech_holes.jsonl` or
`asr_hallucinations.jsonl`, keep `non_dialogue=false`,
`review_required=true`, `audio_reviewed=false`, and
`review_disposition=pending_audio_review`. Do not claim an audio decision. For a
speech hole whose ASR and YouTube text are both empty, keep `tr_corrected` empty.
For an ASR/caption candidate, correct its available Turkish text normally while
leaving the audio fields pending. Continue through every record; the Colab audio
gate will create the final `_TR_CORRECTED.zip` later.

For each input record, add exactly these fields:

- `tr_corrected`: corrected Turkish dialogue.
- `non_dialogue`: always `false` in this text-only pass.
- `review_required`: `true` only for a linked audio-review record or an
  ordinary correction that would delete a lexical ASR token.
- `audio_reviewed`: always `false` in this text-only pass.
- `review_disposition`: `pending_audio_review` when `review_required=true`;
  otherwise `not_applicable`. Never emit a final acoustic disposition.
- `note`: optional text note. Never claim that audio was heard or reviewed.

Every non-flagged ASR record is dialogue and requires non-empty `tr_corrected`;
do not discard it as noise. Return every record once in the original order.

An input whose ASR and YouTube text are both empty represents an unresolved VAD
speech hole and is marked `unresolved_vad_speech`. Do not infer its words from
neighboring context and do not open its WAV. Keep `tr_corrected` empty and its
audio decision pending for Colab.

An ordinary record marked `suspected_asr_hallucination` or an ASR-empty record
marked `orphan_youtube_caption` has an exact immutable entry in
`asr_hallucinations.jsonl` and a contextual WAV named by its `audio_member`.
The target is `start_ms`/`end_ms`; `clip_start_ms`/`clip_end_ms` are Colab-only
acoustic context. Correct the available ASR/YouTube Turkish text, but keep
`review_required=true`, `audio_reviewed=false`, and
`review_disposition=pending_audio_review`. Never use `non_dialogue=true` in this
pass. Copy `asr_audit` and every other immutable field exactly; never invent a
risk flag.

Do not delete lexical ASR tokens from an ordinary record without exact audio
review. Punctuation-only and case-only edits are not lexical deletion. If a
correction needs to remove a spoken-word token but the record has no immutable
review entry, return it with `review_required=true`, `audio_reviewed=false`, and
`review_disposition=pending_audio_review`; then rerun `01_PREPARE_TR` with that
exact UID in `EXTRA_AUDIO_REVIEW_UIDS`. The notebook may reuse the already
validated text edits only when it proves that the refreshed pack differs solely
by added hash-bound audio-review evidence; otherwise a fresh correction is
required.
"""


class TRCorrectionError(ValueError):
    """Raised when a correction pack or output violates its trust contract."""


@dataclass(frozen=True)
class TRCorrectionPackData:
    """Validated contents of an input correction pack."""

    manifest: dict[str, Any]
    utterances: tuple[dict[str, Any], ...]
    speech_holes: tuple[dict[str, Any], ...]
    asr_hallucination_records: tuple[dict[str, Any], ...]
    instructions: str


@dataclass(frozen=True)
class TRCorrectionOutputData:
    """Validated correction output bound to one input pack."""

    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    input_sha256: str
    output_sha256: str


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TRCorrectionError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise TRCorrectionError(f"Non-finite JSON number is forbidden: {value}")


def _require_exact_fields(
    record: Mapping[str, Any], expected: frozenset[str] | set[str], label: str
) -> None:
    actual = set(record)
    missing = sorted(set(expected).difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise TRCorrectionError(
            f"{label} field mismatch; missing={missing}, extra={extra}"
        )


def _require_string(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TRCorrectionError(f"{field} must be a string")
    if nonempty and not value.strip():
        raise TRCorrectionError(f"{field} must not be empty")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TRCorrectionError(f"{field} must be an integer")
    if value < minimum:
        raise TRCorrectionError(f"{field} must be at least {minimum}")
    return value


def _validate_flags(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TRCorrectionError(f"{field} must be a list of strings")
    flags: list[str] = []
    seen: set[str] = set()
    for offset, flag in enumerate(value):
        flag = _require_string(flag, f"{field}[{offset}]", nonempty=True)
        if flag in seen:
            raise TRCorrectionError(f"{field} contains duplicate value: {flag!r}")
        seen.add(flag)
        flags.append(flag)
    return flags


def _validate_asr_audit(value: Any, field: str = "asr_audit") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TRCorrectionError(f"{field} must be a JSON object")
    _require_exact_fields(value, ASR_AUDIT_FIELDS, field)
    trusted: dict[str, Any] = {}
    for name in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
        item = value.get(name)
        if item is None:
            trusted[name] = None
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TRCorrectionError(f"{field}.{name} must be a finite number or null")
        if not math.isfinite(float(item)):
            raise TRCorrectionError(f"{field}.{name} must be finite")
        trusted[name] = item
    return trusted


def _lexical_tokens(text: str) -> list[str]:
    """Tokenize words while ignoring punctuation and case-only differences."""

    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def validate_input_utterances(
    utterances: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and copy ordered coarse-ASR utterances."""

    if isinstance(utterances, (str, bytes, bytearray)) or not isinstance(
        utterances, Sequence
    ):
        raise TRCorrectionError("utterances must be a sequence of JSON objects")
    if not utterances:
        raise TRCorrectionError("At least one correction utterance is required")
    expected_fields = frozenset(IMMUTABLE_UTTERANCE_FIELDS)
    trusted: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    previous_start = -1
    for position, raw in enumerate(utterances, start=1):
        if not isinstance(raw, Mapping):
            raise TRCorrectionError(f"Utterance {position} must be a JSON object")
        _require_exact_fields(raw, expected_fields, f"Utterance {position}")
        uid = _require_string(
            raw.get("utterance_uid"), "utterance_uid", nonempty=True
        )
        if uid in seen_uids:
            raise TRCorrectionError(f"Duplicate utterance_uid: {uid}")
        seen_uids.add(uid)
        index = _require_int(
            raw.get("utterance_index"), "utterance_index", minimum=1
        )
        if index != position:
            raise TRCorrectionError(
                f"Utterance order/index mismatch at position {position}: got {index}"
            )
        start_ms = _require_int(raw.get("coarse_start_ms"), "coarse_start_ms")
        end_ms = _require_int(raw.get("coarse_end_ms"), "coarse_end_ms", minimum=1)
        if end_ms <= start_ms:
            raise TRCorrectionError(
                f"Utterance {uid} coarse_end_ms must exceed coarse_start_ms"
            )
        if start_ms < previous_start:
            raise TRCorrectionError("Utterances must be ordered by coarse_start_ms")
        previous_start = start_ms
        asr_text = _require_string(raw.get("asr_text"), "asr_text")
        youtube_text = _require_string(raw.get("youtube_text"), "youtube_text")
        context_before = _require_string(
            raw.get("context_before"), "context_before"
        )
        context_after = _require_string(raw.get("context_after"), "context_after")
        flags = _validate_flags(raw.get("risk_flags"), "risk_flags")
        asr_audit = _validate_asr_audit(raw.get("asr_audit"))
        if (
            not asr_text.strip()
            and not youtube_text.strip()
            and "unresolved_vad_speech" not in flags
        ):
            raise TRCorrectionError(
                f"Utterance {uid} needs non-empty ASR/YouTube evidence or the "
                "explicit unresolved_vad_speech risk flag"
            )
        trusted.append(
            {
                "utterance_uid": uid,
                "utterance_index": index,
                "coarse_start_ms": start_ms,
                "coarse_end_ms": end_ms,
                "asr_text": asr_text,
                "youtube_text": youtube_text,
                "context_before": context_before,
                "context_after": context_after,
                "risk_flags": flags,
                "asr_audit": asr_audit,
            }
        )
    return trusted


def validate_speech_holes(
    speech_holes: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate read-only speech regions that received no coarse ASR text."""

    if speech_holes is None:
        return []
    if isinstance(speech_holes, (str, bytes, bytearray)) or not isinstance(
        speech_holes, Sequence
    ):
        raise TRCorrectionError("speech_holes must be a sequence of JSON objects")
    trusted: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_audio_members: set[str] = set()
    previous_start = -1
    total_audio_bytes = 0
    if len(speech_holes) > MAX_SPEECH_HOLE_AUDIO_FILES:
        raise TRCorrectionError(
            "speech_holes exceeds the bounded audio-file count limit"
        )
    for position, raw in enumerate(speech_holes, start=1):
        if not isinstance(raw, Mapping):
            raise TRCorrectionError(f"Speech hole {position} must be a JSON object")
        _require_exact_fields(raw, SPEECH_HOLE_FIELDS, f"Speech hole {position}")
        uid = _require_string(raw.get("hole_uid"), "hole_uid", nonempty=True)
        if uid in {".", ".."} or any(
            not (character.isalnum() or character in "._-") for character in uid
        ):
            raise TRCorrectionError(
                f"Speech hole {position} hole_uid contains unsafe characters"
            )
        if uid in seen_uids:
            raise TRCorrectionError(f"Duplicate hole_uid: {uid}")
        seen_uids.add(uid)
        index = _require_int(raw.get("hole_index"), "hole_index", minimum=1)
        if index != position:
            raise TRCorrectionError(
                f"Speech-hole order/index mismatch at position {position}: got {index}"
            )
        start_ms = _require_int(raw.get("start_ms"), "start_ms")
        end_ms = _require_int(raw.get("end_ms"), "end_ms", minimum=1)
        if end_ms <= start_ms:
            raise TRCorrectionError(
                f"Speech hole {uid} end_ms must exceed start_ms"
            )
        clip_start_ms = _require_int(raw.get("clip_start_ms"), "clip_start_ms")
        clip_end_ms = _require_int(
            raw.get("clip_end_ms"), "clip_end_ms", minimum=1
        )
        if (
            clip_end_ms <= clip_start_ms
            or clip_start_ms > start_ms
            or clip_end_ms < end_ms
        ):
            raise TRCorrectionError(
                f"Speech hole {uid} clip bounds must contain its target bounds"
            )
        if start_ms < previous_start:
            raise TRCorrectionError("Speech holes must be ordered by start_ms")
        previous_start = start_ms
        risk_flags = _validate_flags(raw.get("risk_flags"), "risk_flags")
        if "unresolved_vad_speech" not in risk_flags:
            raise TRCorrectionError(
                f"Speech hole {uid} must include unresolved_vad_speech"
            )
        audio_member = _require_string(
            raw.get("audio_member"), "audio_member", nonempty=True
        )
        expected_audio_member = f"speech_hole_audio/{uid}.wav"
        if audio_member != expected_audio_member:
            raise TRCorrectionError(
                f"Speech hole {uid} audio_member must be {expected_audio_member!r}"
            )
        if audio_member in seen_audio_members:
            raise TRCorrectionError(f"Duplicate speech-hole audio member: {audio_member}")
        seen_audio_members.add(audio_member)
        audio_sha256 = _require_string(
            raw.get("audio_sha256"), "audio_sha256", nonempty=True
        )
        if len(audio_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in audio_sha256
        ):
            raise TRCorrectionError(
                f"Speech hole {uid} audio_sha256 must be lowercase SHA-256"
            )
        audio_size_bytes = _require_int(
            raw.get("audio_size_bytes"), "audio_size_bytes", minimum=45
        )
        if audio_size_bytes > MAX_SPEECH_HOLE_AUDIO_BYTES:
            raise TRCorrectionError(f"Speech hole {uid} audio exceeds size limit")
        total_audio_bytes += audio_size_bytes
        if total_audio_bytes > MAX_SPEECH_HOLE_AUDIO_TOTAL_BYTES:
            raise TRCorrectionError("Speech-hole audio total exceeds size limit")
        trusted.append(
            {
                "hole_uid": uid,
                "hole_index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "clip_start_ms": clip_start_ms,
                "clip_end_ms": clip_end_ms,
                "reason": _require_string(
                    raw.get("reason"), "reason", nonempty=True
                ),
                "context_before": _require_string(
                    raw.get("context_before"), "context_before"
                ),
                "context_after": _require_string(
                    raw.get("context_after"), "context_after"
                ),
                "risk_flags": risk_flags,
                "audio_member": audio_member,
                "audio_sha256": audio_sha256,
                "audio_size_bytes": audio_size_bytes,
            }
        )
    return trusted


def validate_asr_hallucination_records(
    records: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate immutable WAV-backed reviews for suspected ASR hallucinations."""

    if records is None:
        return []
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TRCorrectionError(
            "asr_hallucination_records must be a sequence of JSON objects"
        )
    if len(records) > MAX_ASR_HALLUCINATION_AUDIO_FILES:
        raise TRCorrectionError(
            "asr_hallucination_records exceeds the bounded audio-file count limit"
        )
    trusted: list[dict[str, Any]] = []
    seen_candidate_uids: set[str] = set()
    seen_utterance_uids: set[str] = set()
    seen_audio_members: set[str] = set()
    previous_utterance_index = 0
    previous_start = -1
    total_audio_bytes = 0
    for position, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            raise TRCorrectionError(
                f"ASR hallucination candidate {position} must be a JSON object"
            )
        _require_exact_fields(
            raw,
            ASR_HALLUCINATION_FIELDS,
            f"ASR hallucination candidate {position}",
        )
        candidate_uid = _require_string(
            raw.get("candidate_uid"), "candidate_uid", nonempty=True
        )
        if candidate_uid in {".", ".."} or any(
            not (character.isalnum() or character in "._-")
            for character in candidate_uid
        ):
            raise TRCorrectionError(
                f"ASR hallucination candidate {position} candidate_uid "
                "contains unsafe characters"
            )
        if candidate_uid in seen_candidate_uids:
            raise TRCorrectionError(f"Duplicate candidate_uid: {candidate_uid}")
        seen_candidate_uids.add(candidate_uid)
        candidate_index = _require_int(
            raw.get("candidate_index"), "candidate_index", minimum=1
        )
        if candidate_index != position:
            raise TRCorrectionError(
                "ASR hallucination candidate order/index mismatch at position "
                f"{position}: got {candidate_index}"
            )
        utterance_uid = _require_string(
            raw.get("utterance_uid"), "utterance_uid", nonempty=True
        )
        if utterance_uid in seen_utterance_uids:
            raise TRCorrectionError(
                f"Duplicate hallucination-candidate utterance_uid: {utterance_uid}"
            )
        seen_utterance_uids.add(utterance_uid)
        utterance_index = _require_int(
            raw.get("utterance_index"), "utterance_index", minimum=1
        )
        if utterance_index <= previous_utterance_index:
            raise TRCorrectionError(
                "ASR hallucination candidates must follow utterance order"
            )
        previous_utterance_index = utterance_index
        start_ms = _require_int(raw.get("start_ms"), "start_ms")
        end_ms = _require_int(raw.get("end_ms"), "end_ms", minimum=1)
        if end_ms <= start_ms:
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} end_ms must exceed start_ms"
            )
        clip_start_ms = _require_int(raw.get("clip_start_ms"), "clip_start_ms")
        clip_end_ms = _require_int(
            raw.get("clip_end_ms"), "clip_end_ms", minimum=1
        )
        if (
            clip_end_ms <= clip_start_ms
            or clip_start_ms > start_ms
            or clip_end_ms < end_ms
        ):
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} clip bounds must "
                "contain its target bounds"
            )
        if start_ms < previous_start:
            raise TRCorrectionError(
                "ASR hallucination candidates must be ordered by start_ms"
            )
        previous_start = start_ms
        asr_text = _require_string(raw.get("asr_text"), "asr_text")
        youtube_text = _require_string(raw.get("youtube_text"), "youtube_text")
        context_before = _require_string(
            raw.get("context_before"), "context_before"
        )
        context_after = _require_string(raw.get("context_after"), "context_after")
        risk_flags = _validate_flags(raw.get("risk_flags"), "risk_flags")
        asr_audit = _validate_asr_audit(
            raw.get("asr_audit"), "candidate.asr_audit"
        )
        is_asr_suspect = "suspected_asr_hallucination" in risk_flags
        is_orphan_caption = "orphan_youtube_caption" in risk_flags
        if not is_asr_suspect and not is_orphan_caption:
            raise TRCorrectionError(
                f"Audio-review candidate {candidate_uid} must include "
                "suspected_asr_hallucination or orphan_youtube_caption"
            )
        if is_orphan_caption and (
            asr_text.strip() or not youtube_text.strip()
        ):
            raise TRCorrectionError(
                f"Orphan YouTube candidate {candidate_uid} requires empty ASR "
                "and non-empty YouTube evidence"
            )
        if is_orphan_caption and any(
            value is not None for value in asr_audit.values()
        ):
            raise TRCorrectionError(
                f"Orphan YouTube candidate {candidate_uid} requires null ASR audit"
            )
        if not is_orphan_caption and not asr_text.strip():
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} requires non-empty "
                "ASR evidence"
            )
        if "unresolved_vad_speech" in risk_flags:
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} cannot also be an "
                "unresolved_vad_speech hole"
            )
        audio_member = _require_string(
            raw.get("audio_member"), "audio_member", nonempty=True
        )
        expected_audio_member = (
            f"asr_hallucination_audio/{candidate_uid}.wav"
        )
        if audio_member != expected_audio_member:
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} audio_member must be "
                f"{expected_audio_member!r}"
            )
        if audio_member in seen_audio_members:
            raise TRCorrectionError(
                f"Duplicate ASR hallucination audio member: {audio_member}"
            )
        seen_audio_members.add(audio_member)
        audio_sha256 = _require_string(
            raw.get("audio_sha256"), "audio_sha256", nonempty=True
        )
        if len(audio_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in audio_sha256
        ):
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} audio_sha256 must be "
                "lowercase SHA-256"
            )
        audio_size_bytes = _require_int(
            raw.get("audio_size_bytes"), "audio_size_bytes", minimum=45
        )
        if audio_size_bytes > MAX_ASR_HALLUCINATION_AUDIO_BYTES:
            raise TRCorrectionError(
                f"ASR hallucination candidate {candidate_uid} audio exceeds size limit"
            )
        total_audio_bytes += audio_size_bytes
        if total_audio_bytes > MAX_ASR_HALLUCINATION_AUDIO_TOTAL_BYTES:
            raise TRCorrectionError(
                "ASR hallucination audio total exceeds size limit"
            )
        trusted.append(
            {
                "candidate_uid": candidate_uid,
                "candidate_index": candidate_index,
                "utterance_uid": utterance_uid,
                "utterance_index": utterance_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "clip_start_ms": clip_start_ms,
                "clip_end_ms": clip_end_ms,
                "reason": _require_string(
                    raw.get("reason"), "reason", nonempty=True
                ),
                "asr_text": asr_text,
                "youtube_text": youtube_text,
                "context_before": context_before,
                "context_after": context_after,
                "risk_flags": risk_flags,
                "asr_audit": asr_audit,
                "audio_member": audio_member,
                "audio_sha256": audio_sha256,
                "audio_size_bytes": audio_size_bytes,
            }
        )
    return trusted


def _validate_hole_utterance_links(
    utterances: Sequence[Mapping[str, Any]],
    holes: Sequence[Mapping[str, Any]],
) -> None:
    blank_candidates = {
        str(record["utterance_uid"]): record
        for record in utterances
        if not str(record["asr_text"]).strip()
        and not str(record["youtube_text"]).strip()
        and "unresolved_vad_speech" in record["risk_flags"]
    }
    holes_by_uid = {str(hole["hole_uid"]): hole for hole in holes}
    if set(blank_candidates) != set(holes_by_uid):
        raise TRCorrectionError(
            "Blank unresolved utterances and speech-hole audit records do not "
            f"match; missing_holes={sorted(set(blank_candidates) - set(holes_by_uid))}, "
            f"missing_utterances={sorted(set(holes_by_uid) - set(blank_candidates))}"
        )
    for uid, hole in holes_by_uid.items():
        utterance = blank_candidates[uid]
        comparisons = {
            "coarse_start_ms": "start_ms",
            "coarse_end_ms": "end_ms",
            "context_before": "context_before",
            "context_after": "context_after",
            "risk_flags": "risk_flags",
        }
        for utterance_field, hole_field in comparisons.items():
            if utterance[utterance_field] != hole[hole_field]:
                raise TRCorrectionError(
                    f"Speech-hole {uid} differs from its blank utterance in "
                    f"{utterance_field}"
                )


def _validate_hallucination_utterance_links(
    utterances: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    flagged = {
        str(record["utterance_uid"]): record
        for record in utterances
        if (
            "suspected_asr_hallucination" in record["risk_flags"]
            or "orphan_youtube_caption" in record["risk_flags"]
        )
    }
    candidates_by_utterance_uid = {
        str(candidate["utterance_uid"]): candidate for candidate in candidates
    }
    if set(flagged) != set(candidates_by_utterance_uid):
        raise TRCorrectionError(
            "Flagged ASR hallucinations and immutable candidate audit records do "
            "not match; missing_candidates="
            f"{sorted(set(flagged) - set(candidates_by_utterance_uid))}, "
            "missing_utterances="
            f"{sorted(set(candidates_by_utterance_uid) - set(flagged))}"
        )
    comparisons = {
        "utterance_index": "utterance_index",
        "coarse_start_ms": "start_ms",
        "coarse_end_ms": "end_ms",
        "asr_text": "asr_text",
        "youtube_text": "youtube_text",
        "context_before": "context_before",
        "context_after": "context_after",
        "risk_flags": "risk_flags",
        "asr_audit": "asr_audit",
    }
    for uid, candidate in candidates_by_utterance_uid.items():
        utterance = flagged[uid]
        is_orphan_caption = "orphan_youtube_caption" in utterance["risk_flags"]
        if is_orphan_caption and (
            str(utterance["asr_text"]).strip()
            or not str(utterance["youtube_text"]).strip()
        ):
            raise TRCorrectionError(
                f"Orphan YouTube candidate {uid} has invalid text evidence"
            )
        if not is_orphan_caption and not str(utterance["asr_text"]).strip():
            raise TRCorrectionError(
                f"ASR hallucination candidate {uid} must bind non-empty ASR evidence"
            )
        if "unresolved_vad_speech" in utterance["risk_flags"]:
            raise TRCorrectionError(
                f"ASR hallucination candidate {uid} cannot bind a speech hole"
            )
        for utterance_field, candidate_field in comparisons.items():
            if utterance[utterance_field] != candidate[candidate_field]:
                raise TRCorrectionError(
                    f"ASR hallucination candidate {uid} differs from its utterance "
                    f"in {utterance_field}"
                )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TRCorrectionError(f"Value is not canonical JSON: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_review_wav(
    payload: bytes,
    *,
    member: str,
    expected_duration_ms: int,
    label: str,
) -> None:
    """Require the bounded mono 16 kHz PCM WAV produced by PREPARE."""

    try:
        with wave.open(io.BytesIO(payload), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            frame_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
            frame_payload = audio.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise TRCorrectionError(f"Invalid {label} WAV: {member}") from exc
    if (
        channels != 1
        or sample_width != 2
        or frame_rate != 16_000
        or compression != "NONE"
        or frame_count < 1
    ):
        raise TRCorrectionError(
            f"{label} WAV must be mono 16 kHz PCM s16le: {member}"
        )
    if len(frame_payload) != frame_count * channels * sample_width:
        raise TRCorrectionError(f"{label} WAV is truncated: {member}")
    actual_duration_ms = round(frame_count * 1000 / frame_rate)
    if abs(actual_duration_ms - expected_duration_ms) > 100:
        raise TRCorrectionError(
            f"{label} WAV duration mismatch for {member}: expected about "
            f"{expected_duration_ms} ms, got {actual_duration_ms} ms"
        )


def _speech_hole_audio_payloads(
    holes: Sequence[Mapping[str, Any]],
    audio_root: str | os.PathLike[str] | None,
) -> dict[str, bytes]:
    if not holes:
        return {}
    if audio_root is None:
        raise TRCorrectionError(
            "speech_hole_audio_root is required when speech holes are present"
        )
    root = Path(audio_root)
    if not root.is_dir():
        raise TRCorrectionError(f"Speech-hole audio root does not exist: {root}")
    if root.is_symlink():
        raise TRCorrectionError("Speech-hole audio root must not be a symlink")
    resolved_root = root.resolve()
    payloads: dict[str, bytes] = {}
    total_bytes = 0
    for hole in holes:
        member = str(hole["audio_member"])
        member_parts = PurePosixPath(member).parts
        candidate = root.joinpath(*member_parts)
        try:
            current = root
            for part in member_parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("symlink component")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TRCorrectionError(
                f"Speech-hole audio path escapes its root or is missing: {member}"
            ) from exc
        if not resolved.is_file():
            raise TRCorrectionError(
                f"Speech-hole audio must be a regular non-symlink file: {member}"
            )
        size = resolved.stat().st_size
        if size != hole["audio_size_bytes"]:
            raise TRCorrectionError(f"Speech-hole audio size mismatch: {member}")
        if size > MAX_SPEECH_HOLE_AUDIO_BYTES:
            raise TRCorrectionError(f"Speech-hole audio exceeds size limit: {member}")
        total_bytes += size
        if total_bytes > MAX_SPEECH_HOLE_AUDIO_TOTAL_BYTES:
            raise TRCorrectionError("Speech-hole audio total exceeds size limit")
        payload = resolved.read_bytes()
        if _sha256_bytes(payload) != hole["audio_sha256"]:
            raise TRCorrectionError(f"Speech-hole audio SHA-256 mismatch: {member}")
        _validate_review_wav(
            payload,
            member=member,
            expected_duration_ms=(
                int(hole["clip_end_ms"]) - int(hole["clip_start_ms"])
            ),
            label="Speech-hole",
        )
        payloads[member] = payload
    return payloads


def _asr_hallucination_audio_payloads(
    candidates: Sequence[Mapping[str, Any]],
    audio_root: str | os.PathLike[str] | None,
) -> dict[str, bytes]:
    if not candidates:
        return {}
    if audio_root is None:
        raise TRCorrectionError(
            "asr_hallucination_audio_root is required when candidates are present"
        )
    root = Path(audio_root)
    if not root.is_dir():
        raise TRCorrectionError(
            f"ASR hallucination audio root does not exist: {root}"
        )
    if root.is_symlink():
        raise TRCorrectionError(
            "ASR hallucination audio root must not be a symlink"
        )
    resolved_root = root.resolve()
    payloads: dict[str, bytes] = {}
    total_bytes = 0
    for candidate_record in candidates:
        member = str(candidate_record["audio_member"])
        member_parts = PurePosixPath(member).parts
        candidate_path = root.joinpath(*member_parts)
        try:
            current = root
            for part in member_parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("symlink component")
            resolved = candidate_path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TRCorrectionError(
                "ASR hallucination audio path escapes its root or is missing: "
                f"{member}"
            ) from exc
        if not resolved.is_file():
            raise TRCorrectionError(
                "ASR hallucination audio must be a regular non-symlink file: "
                f"{member}"
            )
        size = resolved.stat().st_size
        if size != candidate_record["audio_size_bytes"]:
            raise TRCorrectionError(
                f"ASR hallucination audio size mismatch: {member}"
            )
        if size > MAX_ASR_HALLUCINATION_AUDIO_BYTES:
            raise TRCorrectionError(
                f"ASR hallucination audio exceeds size limit: {member}"
            )
        total_bytes += size
        if total_bytes > MAX_ASR_HALLUCINATION_AUDIO_TOTAL_BYTES:
            raise TRCorrectionError(
                "ASR hallucination audio total exceeds size limit"
            )
        payload = resolved.read_bytes()
        if _sha256_bytes(payload) != candidate_record["audio_sha256"]:
            raise TRCorrectionError(
                f"ASR hallucination audio SHA-256 mismatch: {member}"
            )
        _validate_review_wav(
            payload,
            member=member,
            expected_duration_ms=(
                int(candidate_record["clip_end_ms"])
                - int(candidate_record["clip_start_ms"])
            ),
            label="ASR hallucination review",
        )
        payloads[member] = payload
    return payloads


def compute_input_sha256(
    utterances: Sequence[Mapping[str, Any]],
    speech_holes: Sequence[Mapping[str, Any]] | None = None,
    *,
    episode: int,
    asr_hallucination_records: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Hash the exact ordered immutable correction evidence."""

    episode = _require_int(episode, "episode", minimum=1)
    trusted_utterances = validate_input_utterances(utterances)
    trusted_holes = validate_speech_holes(speech_holes)
    trusted_hallucinations = validate_asr_hallucination_records(
        asr_hallucination_records
    )
    _validate_hole_utterance_links(trusted_utterances, trusted_holes)
    _validate_hallucination_utterance_links(
        trusted_utterances, trusted_hallucinations
    )
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "format_version": FORMAT_VERSION,
                "episode": episode,
                "utterances": trusted_utterances,
                "speech_holes": trusted_holes,
                "asr_hallucination_records": trusted_hallucinations,
            }
        )
    )


def compute_output_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash already validated, ordered correction output records."""

    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TRCorrectionError("records must be a sequence of JSON objects")
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "format_version": FORMAT_VERSION,
                "records": [dict(record) for record in records],
            }
        )
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _write_zip_atomic(
    destination: Path,
    member_order: Sequence[str],
    payloads: Mapping[str, bytes],
    *,
    validate_temporary: Callable[[Path], Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in member_order:
                archive.writestr(_zip_info(name), payloads[name], compresslevel=9)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        # Do not publish even a self-generated archive until it has passed the
        # same parser, CRC, hash and contract checks used for external input.
        validate_temporary(temporary)
        os.replace(temporary, destination)
        temporary = None
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _parse_json_bytes(payload: bytes, member: str) -> Any:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TRCorrectionError(f"{member} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise TRCorrectionError(
            f"Invalid JSON in {member} at line {exc.lineno}: {exc.msg}"
        ) from exc


def _parse_jsonl_bytes(payload: bytes, member: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TRCorrectionError(f"{member} is not valid UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise TRCorrectionError(
                f"Invalid JSONL in {member} line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise TRCorrectionError(
                f"{member} line {line_number} must be a JSON object"
            )
        records.append(value)
    return records


def _validate_zip_container(archive: zipfile.ZipFile) -> list[str]:
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise TRCorrectionError("ZIP contains too many members")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise TRCorrectionError("ZIP contains duplicate member names")
    total_size = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name)
        is_audio_member = (
            len(pure.parts) == 2
            and pure.parts[0]
            in {"speech_hole_audio", "asr_hallucination_audio"}
            and pure.parts[1].endswith(".wav")
        )
        if (
            not name
            or pure.is_absolute()
            or (len(pure.parts) != 1 and not is_audio_member)
            or ".." in pure.parts
            or "\\" in name
            or name.endswith("/")
        ):
            raise TRCorrectionError(f"Unsafe ZIP member path: {name}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = unix_mode & 0o170000
        if file_type not in (0, 0o100000):
            raise TRCorrectionError(f"ZIP member is not a regular file: {name}")
        if info.flag_bits & 0x1:
            raise TRCorrectionError(f"Encrypted ZIP member is forbidden: {name}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise TRCorrectionError(f"ZIP member is too large: {name}")
        if is_audio_member and info.file_size > max(
            MAX_SPEECH_HOLE_AUDIO_BYTES,
            MAX_ASR_HALLUCINATION_AUDIO_BYTES,
        ):
            raise TRCorrectionError(f"Review audio member is too large: {name}")
        total_size += info.file_size
        if total_size > MAX_TOTAL_BYTES:
            raise TRCorrectionError("ZIP uncompressed size exceeds safety limit")
        if info.file_size >= MIN_RATIO_CHECK_BYTES and not is_audio_member:
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > MAX_COMPRESSION_RATIO:
                raise TRCorrectionError(
                    f"ZIP member compression ratio is unsafe: {name}"
                )
    bad_member = archive.testzip()
    if bad_member is not None:
        raise TRCorrectionError(f"ZIP CRC failure: {bad_member}")
    return names


def _open_validated_zip(path_value: str | os.PathLike[str]) -> zipfile.ZipFile:
    path = Path(path_value)
    if not path.is_file():
        raise TRCorrectionError(f"ZIP not found: {path}")
    try:
        archive = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise TRCorrectionError(f"Corrupt ZIP: {path}") from exc
    try:
        _validate_zip_container(archive)
    except Exception:
        archive.close()
        raise
    return archive


def _batch_slices(
    records: Sequence[Mapping[str, Any]], batch_size: int
) -> list[list[Mapping[str, Any]]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise TRCorrectionError("batch_size must be a positive integer")
    return [
        list(records[offset : offset + batch_size])
        for offset in range(0, len(records), batch_size)
    ]


_REBINDABLE_AUDIO_REVIEW_FLAGS = frozenset(
    {"suspected_asr_hallucination", "manual_audio_review_required"}
)


def _rebind_text_only_output_to_refreshed_pack(
    previous_pack: TRCorrectionPackData,
    previous_output: TRCorrectionOutputData,
    refreshed_pack: TRCorrectionPackData,
    refreshed_pack_path: Path,
    output_path: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Carry verified text edits across a review-only pack augmentation."""

    if previous_pack.manifest["episode"] != refreshed_pack.manifest["episode"]:
        raise TRCorrectionError("Cannot rebind text correction across episodes")
    if any(
        record["non_dialogue"] is not False
        or record["audio_reviewed"] is not False
        for record in previous_output.records
    ):
        raise TRCorrectionError(
            "Only a provisional text-only correction output can be rebound"
        )
    previous_uids = [
        str(record["utterance_uid"]) for record in previous_pack.utterances
    ]
    refreshed_uids = [
        str(record["utterance_uid"]) for record in refreshed_pack.utterances
    ]
    if previous_uids != refreshed_uids:
        raise TRCorrectionError(
            "Cannot rebind text correction after utterance identity/order changed"
        )
    if list(previous_pack.speech_holes) != list(refreshed_pack.speech_holes):
        raise TRCorrectionError(
            "Cannot rebind text correction after speech-hole evidence changed"
        )

    previous_candidate_uids = {
        str(record["utterance_uid"])
        for record in previous_pack.asr_hallucination_records
    }
    refreshed_candidate_uids = {
        str(record["utterance_uid"])
        for record in refreshed_pack.asr_hallucination_records
    }
    if not previous_candidate_uids.issubset(refreshed_candidate_uids):
        raise TRCorrectionError(
            "Cannot rebind text correction after an audio-review candidate was removed"
        )

    review_only_changes: set[str] = set()
    for previous, refreshed in zip(
        previous_pack.utterances, refreshed_pack.utterances
    ):
        uid = str(previous["utterance_uid"])
        for field in IMMUTABLE_UTTERANCE_FIELDS:
            if field == "risk_flags":
                continue
            if type(previous[field]) is not type(refreshed[field]) or previous[
                field
            ] != refreshed[field]:
                raise TRCorrectionError(
                    "Cannot rebind text correction because immutable evidence "
                    f"changed for {uid}: {field}"
                )
        previous_flags = list(previous["risk_flags"])
        refreshed_flags = list(refreshed["risk_flags"])
        if refreshed_flags[: len(previous_flags)] != previous_flags:
            raise TRCorrectionError(
                f"Cannot rebind text correction because risk flags changed for {uid}"
            )
        added_flags = refreshed_flags[len(previous_flags) :]
        if any(flag not in _REBINDABLE_AUDIO_REVIEW_FLAGS for flag in added_flags):
            raise TRCorrectionError(
                f"Cannot rebind text correction across non-review risk changes for {uid}"
            )
        if added_flags:
            review_only_changes.add(uid)

    added_candidate_uids = refreshed_candidate_uids.difference(
        previous_candidate_uids
    )
    if added_candidate_uids != review_only_changes:
        raise TRCorrectionError(
            "Refreshed review records do not exactly match review-only flag changes"
        )

    refreshed_by_uid = {
        str(record["utterance_uid"]): record
        for record in refreshed_pack.utterances
    }
    refreshed_review_uids = refreshed_candidate_uids.union(
        str(record["hole_uid"]) for record in refreshed_pack.speech_holes
    )
    rebound_records: list[dict[str, Any]] = []
    for previous_record in previous_output.records:
        uid = str(previous_record["utterance_uid"])
        rebound = copy.deepcopy(dict(previous_record))
        refreshed = refreshed_by_uid[uid]
        for field in IMMUTABLE_UTTERANCE_FIELDS:
            rebound[field] = copy.deepcopy(refreshed[field])
        if uid in refreshed_review_uids:
            rebound["non_dialogue"] = False
            rebound["review_required"] = True
            rebound["audio_reviewed"] = False
            rebound["review_disposition"] = "pending_audio_review"
        rebound_records.append(rebound)

    return create_tr_correction_output(
        refreshed_pack_path,
        rebound_records,
        output_path,
        batch_size=batch_size,
    )


def create_tr_correction_pack(
    utterances: Sequence[Mapping[str, Any]],
    speech_holes: Sequence[Mapping[str, Any]] | None,
    out_zip: str | os.PathLike[str],
    *,
    episode: int,
    instructions: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    speech_hole_audio_root: str | os.PathLike[str] | None = None,
    asr_hallucination_records: Sequence[Mapping[str, Any]] | None = None,
    asr_hallucination_audio_root: str | os.PathLike[str] | None = None,
    rebind_text_output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Atomically create and reopen a deterministic V2 correction input ZIP."""

    episode = _require_int(episode, "episode", minimum=1)
    destination = Path(out_zip)
    rebind_output = (
        Path(rebind_text_output_path)
        if rebind_text_output_path is not None
        else None
    )
    rebind_snapshot: tuple[TRCorrectionPackData, TRCorrectionOutputData] | None = None
    if rebind_output is not None and rebind_output.is_file():
        if not destination.is_file():
            raise TRCorrectionError(
                "Cannot preserve text correction because its prior input pack is missing"
            )
        previous_pack = read_tr_correction_pack(destination)
        previous_output = validate_tr_correction_output(
            destination, rebind_output
        )
        rebind_snapshot = (previous_pack, previous_output)
    trusted_utterances = validate_input_utterances(utterances)
    trusted_holes = validate_speech_holes(speech_holes)
    trusted_hallucinations = validate_asr_hallucination_records(
        asr_hallucination_records
    )
    _validate_hole_utterance_links(trusted_utterances, trusted_holes)
    _validate_hallucination_utterance_links(
        trusted_utterances, trusted_hallucinations
    )
    if len(trusted_holes) + len(trusted_hallucinations) > MAX_REVIEW_AUDIO_FILES:
        raise TRCorrectionError("Combined review audio file count exceeds limit")
    instructions_text = DEFAULT_INSTRUCTIONS if instructions is None else instructions
    instructions_text = _require_string(
        instructions_text, "instructions", nonempty=True
    ).replace("\r\n", "\n").replace("\r", "\n")
    instructions_payload = instructions_text.encode("utf-8")
    holes_payload = _jsonl_bytes(trusted_holes)
    hallucinations_payload = _jsonl_bytes(trusted_hallucinations)
    hole_audio_payloads = _speech_hole_audio_payloads(
        trusted_holes, speech_hole_audio_root
    )
    hallucination_audio_payloads = _asr_hallucination_audio_payloads(
        trusted_hallucinations,
        asr_hallucination_audio_root,
    )
    if (
        sum(map(len, hole_audio_payloads.values()))
        + sum(map(len, hallucination_audio_payloads.values()))
        > MAX_REVIEW_AUDIO_TOTAL_BYTES
    ):
        raise TRCorrectionError("Combined review audio size exceeds limit")
    batches = _batch_slices(trusted_utterances, batch_size)

    payloads: dict[str, bytes] = {
        "TR_CORRECTION_INSTRUCTIONS.md": instructions_payload,
        "speech_holes.jsonl": holes_payload,
        "asr_hallucinations.jsonl": hallucinations_payload,
        **hole_audio_payloads,
        **hallucination_audio_payloads,
    }
    descriptors: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(batches, start=1):
        name = f"batch_{batch_number:03d}.jsonl"
        payload = _jsonl_bytes(batch)
        payloads[name] = payload
        descriptors.append(
            {
                "input_file": name,
                "output_file": f"corrected_batch_{batch_number:03d}.jsonl",
                "utterance_count": len(batch),
                "first_utterance_uid": batch[0]["utterance_uid"],
                "last_utterance_uid": batch[-1]["utterance_uid"],
                "sha256": _sha256_bytes(payload),
            }
        )
    input_sha = compute_input_sha256(
        trusted_utterances,
        trusted_holes,
        episode=episode,
        asr_hallucination_records=trusted_hallucinations,
    )
    manifest: dict[str, Any] = {
        "format": INPUT_FORMAT,
        "format_version": FORMAT_VERSION,
        "episode": episode,
        "input_sha256": input_sha,
        "utterance_count": len(trusted_utterances),
        "speech_hole_count": len(trusted_holes),
        "asr_hallucination_count": len(trusted_hallucinations),
        "batch_count": len(batches),
        "batches": descriptors,
        "speech_hole_audio": [
            {
                "hole_uid": hole["hole_uid"],
                "member": hole["audio_member"],
                "sha256": hole["audio_sha256"],
                "size_bytes": hole["audio_size_bytes"],
            }
            for hole in trusted_holes
        ],
        "asr_hallucination_audio": [
            {
                "candidate_uid": candidate["candidate_uid"],
                "utterance_uid": candidate["utterance_uid"],
                "member": candidate["audio_member"],
                "sha256": candidate["audio_sha256"],
                "size_bytes": candidate["audio_size_bytes"],
            }
            for candidate in trusted_hallucinations
        ],
        "file_sha256": {
            "TR_CORRECTION_INSTRUCTIONS.md": _sha256_bytes(instructions_payload),
            "speech_holes.jsonl": _sha256_bytes(holes_payload),
            "asr_hallucinations.jsonl": _sha256_bytes(hallucinations_payload),
        },
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    member_order = [
        "manifest.json",
        "TR_CORRECTION_INSTRUCTIONS.md",
        "speech_holes.jsonl",
        "asr_hallucinations.jsonl",
        *[hole["audio_member"] for hole in trusted_holes],
        *[
            candidate["audio_member"]
            for candidate in trusted_hallucinations
        ],
        *[descriptor["input_file"] for descriptor in descriptors],
    ]
    _write_zip_atomic(
        destination,
        member_order,
        payloads,
        validate_temporary=lambda path: validate_tr_correction_pack(
            path, expected_input_sha256=input_sha
        ),
    )
    validated = validate_tr_correction_pack(
        destination, expected_input_sha256=input_sha
    )
    if rebind_snapshot is not None:
        refreshed_pack = read_tr_correction_pack(
            destination, expected_input_sha256=input_sha
        )
        _rebind_text_only_output_to_refreshed_pack(
            rebind_snapshot[0],
            rebind_snapshot[1],
            refreshed_pack,
            destination,
            rebind_output,
            batch_size=batch_size,
        )
    return validated


def read_tr_correction_pack(
    pack_path: str | os.PathLike[str],
    *,
    expected_input_sha256: str | None = None,
) -> TRCorrectionPackData:
    """Read a fully validated correction input ZIP."""

    with _open_validated_zip(pack_path) as archive:
        names = [info.filename for info in archive.infolist()]
        required = {
            "manifest.json",
            "TR_CORRECTION_INSTRUCTIONS.md",
            "speech_holes.jsonl",
            "asr_hallucinations.jsonl",
        }
        if not required.issubset(names):
            raise TRCorrectionError(
                "Correction pack missing members: "
                + ", ".join(sorted(required.difference(names)))
            )
        manifest = _parse_json_bytes(archive.read("manifest.json"), "manifest.json")
        if not isinstance(manifest, dict):
            raise TRCorrectionError("manifest.json must contain an object")
        _require_exact_fields(manifest, INPUT_MANIFEST_FIELDS, "Input manifest")
        if manifest.get("format") != INPUT_FORMAT:
            raise TRCorrectionError("Wrong correction input format")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise TRCorrectionError("Unsupported correction input format_version")
        episode = _require_int(
            manifest.get("episode"), "manifest.episode", minimum=1
        )
        batch_count = _require_int(
            manifest.get("batch_count"), "manifest.batch_count", minimum=1
        )
        expected_batch_names = [
            f"batch_{number:03d}.jsonl" for number in range(1, batch_count + 1)
        ]
        audio_descriptors = manifest.get("speech_hole_audio")
        if not isinstance(audio_descriptors, list):
            raise TRCorrectionError("Manifest speech_hole_audio must be a list")
        if len(audio_descriptors) > MAX_SPEECH_HOLE_AUDIO_FILES:
            raise TRCorrectionError("Manifest has too many speech-hole audio files")
        audio_members: list[str] = []
        for position, descriptor in enumerate(audio_descriptors, start=1):
            if not isinstance(descriptor, Mapping):
                raise TRCorrectionError(
                    "Every speech-hole audio descriptor must be an object"
                )
            _require_exact_fields(
                descriptor,
                SPEECH_HOLE_AUDIO_DESCRIPTOR_FIELDS,
                f"Speech-hole audio descriptor {position}",
            )
            member = _require_string(
                descriptor.get("member"),
                f"speech_hole_audio[{position}].member",
                nonempty=True,
            )
            if member in audio_members:
                raise TRCorrectionError(
                    f"Duplicate speech-hole audio descriptor member: {member}"
                )
            audio_members.append(member)
        hallucination_audio_descriptors = manifest.get(
            "asr_hallucination_audio"
        )
        if not isinstance(hallucination_audio_descriptors, list):
            raise TRCorrectionError(
                "Manifest asr_hallucination_audio must be a list"
            )
        if (
            len(hallucination_audio_descriptors)
            > MAX_ASR_HALLUCINATION_AUDIO_FILES
        ):
            raise TRCorrectionError(
                "Manifest has too many ASR hallucination audio files"
            )
        hallucination_audio_members: list[str] = []
        for position, descriptor in enumerate(
            hallucination_audio_descriptors, start=1
        ):
            if not isinstance(descriptor, Mapping):
                raise TRCorrectionError(
                    "Every ASR hallucination audio descriptor must be an object"
                )
            _require_exact_fields(
                descriptor,
                ASR_HALLUCINATION_AUDIO_DESCRIPTOR_FIELDS,
                f"ASR hallucination audio descriptor {position}",
            )
            member = _require_string(
                descriptor.get("member"),
                f"asr_hallucination_audio[{position}].member",
                nonempty=True,
            )
            if member in hallucination_audio_members:
                raise TRCorrectionError(
                    "Duplicate ASR hallucination audio descriptor member: "
                    f"{member}"
                )
            hallucination_audio_members.append(member)
        if len(audio_members) + len(hallucination_audio_members) > MAX_REVIEW_AUDIO_FILES:
            raise TRCorrectionError(
                "Manifest combined review audio file count exceeds limit"
            )
        expected_members = required.union(
            expected_batch_names,
            audio_members,
            hallucination_audio_members,
        )
        if set(names) != expected_members:
            raise TRCorrectionError(
                "Correction pack member mismatch; "
                f"missing={sorted(expected_members.difference(names))}, "
                f"extra={sorted(set(names).difference(expected_members))}"
            )
        descriptors = manifest.get("batches")
        if not isinstance(descriptors, list) or len(descriptors) != batch_count:
            raise TRCorrectionError("Manifest batches do not match batch_count")
        utterances: list[dict[str, Any]] = []
        for offset, (expected_name, descriptor) in enumerate(
            zip(expected_batch_names, descriptors), start=1
        ):
            if not isinstance(descriptor, Mapping):
                raise TRCorrectionError("Every batch descriptor must be an object")
            _require_exact_fields(
                descriptor,
                INPUT_BATCH_DESCRIPTOR_FIELDS,
                f"Input batch descriptor {offset}",
            )
            if descriptor.get("input_file") != expected_name:
                raise TRCorrectionError(f"Batch descriptor order mismatch: {expected_name}")
            if descriptor.get("output_file") != f"corrected_batch_{offset:03d}.jsonl":
                raise TRCorrectionError(f"Batch output filename mismatch: {expected_name}")
            payload = archive.read(expected_name)
            if descriptor.get("sha256") != _sha256_bytes(payload):
                raise TRCorrectionError(f"Batch SHA-256 mismatch: {expected_name}")
            batch_records = _parse_jsonl_bytes(payload, expected_name)
            if not batch_records:
                raise TRCorrectionError(f"Empty correction batch: {expected_name}")
            declared_count = _require_int(
                descriptor.get("utterance_count"),
                f"{expected_name}.utterance_count",
                minimum=1,
            )
            if declared_count != len(batch_records):
                raise TRCorrectionError(f"Batch count mismatch: {expected_name}")
            if descriptor.get("first_utterance_uid") != batch_records[0].get(
                "utterance_uid"
            ):
                raise TRCorrectionError(f"Batch first UID mismatch: {expected_name}")
            if descriptor.get("last_utterance_uid") != batch_records[-1].get(
                "utterance_uid"
            ):
                raise TRCorrectionError(f"Batch last UID mismatch: {expected_name}")
            utterances.extend(batch_records)
        trusted_utterances = validate_input_utterances(utterances)
        utterance_count = _require_int(
            manifest.get("utterance_count"), "manifest.utterance_count", minimum=1
        )
        if utterance_count != len(trusted_utterances):
            raise TRCorrectionError("Manifest utterance_count mismatch")

        holes_payload = archive.read("speech_holes.jsonl")
        hallucinations_payload = archive.read("asr_hallucinations.jsonl")
        instructions_payload = archive.read("TR_CORRECTION_INSTRUCTIONS.md")
        file_hashes = manifest.get("file_sha256")
        if not isinstance(file_hashes, Mapping):
            raise TRCorrectionError("Manifest file_sha256 must be an object")
        if set(file_hashes) != {
            "TR_CORRECTION_INSTRUCTIONS.md",
            "speech_holes.jsonl",
            "asr_hallucinations.jsonl",
        }:
            raise TRCorrectionError("Manifest file_sha256 fields are invalid")
        if file_hashes.get("speech_holes.jsonl") != _sha256_bytes(holes_payload):
            raise TRCorrectionError("speech_holes.jsonl SHA-256 mismatch")
        if file_hashes.get("asr_hallucinations.jsonl") != _sha256_bytes(
            hallucinations_payload
        ):
            raise TRCorrectionError(
                "asr_hallucinations.jsonl SHA-256 mismatch"
            )
        if file_hashes.get("TR_CORRECTION_INSTRUCTIONS.md") != _sha256_bytes(
            instructions_payload
        ):
            raise TRCorrectionError("Instructions SHA-256 mismatch")
        trusted_holes = validate_speech_holes(
            _parse_jsonl_bytes(holes_payload, "speech_holes.jsonl")
        )
        trusted_hallucinations = validate_asr_hallucination_records(
            _parse_jsonl_bytes(
                hallucinations_payload, "asr_hallucinations.jsonl"
            )
        )
        speech_hole_count = _require_int(
            manifest.get("speech_hole_count"), "manifest.speech_hole_count"
        )
        if speech_hole_count != len(trusted_holes):
            raise TRCorrectionError("Manifest speech_hole_count mismatch")
        hallucination_count = _require_int(
            manifest.get("asr_hallucination_count"),
            "manifest.asr_hallucination_count",
        )
        if hallucination_count != len(trusted_hallucinations):
            raise TRCorrectionError(
                "Manifest asr_hallucination_count mismatch"
            )
        expected_audio_descriptors = [
            {
                "hole_uid": hole["hole_uid"],
                "member": hole["audio_member"],
                "sha256": hole["audio_sha256"],
                "size_bytes": hole["audio_size_bytes"],
            }
            for hole in trusted_holes
        ]
        if audio_descriptors != expected_audio_descriptors:
            raise TRCorrectionError(
                "Manifest speech_hole_audio does not match speech_holes.jsonl"
            )
        expected_hallucination_audio_descriptors = [
            {
                "candidate_uid": candidate["candidate_uid"],
                "utterance_uid": candidate["utterance_uid"],
                "member": candidate["audio_member"],
                "sha256": candidate["audio_sha256"],
                "size_bytes": candidate["audio_size_bytes"],
            }
            for candidate in trusted_hallucinations
        ]
        if (
            hallucination_audio_descriptors
            != expected_hallucination_audio_descriptors
        ):
            raise TRCorrectionError(
                "Manifest asr_hallucination_audio does not match "
                "asr_hallucinations.jsonl"
            )
        total_audio_bytes = 0
        for hole in trusted_holes:
            member = hole["audio_member"]
            payload = archive.read(member)
            if len(payload) != hole["audio_size_bytes"]:
                raise TRCorrectionError(
                    f"Packed speech-hole audio size mismatch: {member}"
                )
            total_audio_bytes += len(payload)
            if total_audio_bytes > MAX_SPEECH_HOLE_AUDIO_TOTAL_BYTES:
                raise TRCorrectionError("Packed speech-hole audio total exceeds limit")
            if _sha256_bytes(payload) != hole["audio_sha256"]:
                raise TRCorrectionError(
                    f"Packed speech-hole audio SHA-256 mismatch: {member}"
                )
            _validate_review_wav(
                payload,
                member=member,
                expected_duration_ms=(
                    hole["clip_end_ms"] - hole["clip_start_ms"]
                ),
                label="Speech-hole",
            )
        for candidate in trusted_hallucinations:
            member = candidate["audio_member"]
            payload = archive.read(member)
            if len(payload) != candidate["audio_size_bytes"]:
                raise TRCorrectionError(
                    f"Packed ASR hallucination audio size mismatch: {member}"
                )
            total_audio_bytes += len(payload)
            if total_audio_bytes > MAX_REVIEW_AUDIO_TOTAL_BYTES:
                raise TRCorrectionError(
                    "Packed combined review audio total exceeds limit"
                )
            if _sha256_bytes(payload) != candidate["audio_sha256"]:
                raise TRCorrectionError(
                    f"Packed ASR hallucination audio SHA-256 mismatch: {member}"
                )
            _validate_review_wav(
                payload,
                member=member,
                expected_duration_ms=(
                    candidate["clip_end_ms"] - candidate["clip_start_ms"]
                ),
                label="ASR hallucination review",
            )
        try:
            instructions = instructions_payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise TRCorrectionError("Instructions are not valid UTF-8") from exc
        if not instructions.strip():
            raise TRCorrectionError("Instructions are empty")
        input_sha = compute_input_sha256(
            trusted_utterances,
            trusted_holes,
            episode=episode,
            asr_hallucination_records=trusted_hallucinations,
        )
        if manifest.get("input_sha256") != input_sha:
            raise TRCorrectionError("Correction pack input_sha256 mismatch")
        if expected_input_sha256 is not None and expected_input_sha256 != input_sha:
            raise TRCorrectionError(
                f"Unexpected input_sha256: expected {expected_input_sha256}, got {input_sha}"
            )
        return TRCorrectionPackData(
            manifest=copy.deepcopy(manifest),
            utterances=tuple(copy.deepcopy(trusted_utterances)),
            speech_holes=tuple(copy.deepcopy(trusted_holes)),
            asr_hallucination_records=tuple(
                copy.deepcopy(trusted_hallucinations)
            ),
            instructions=instructions,
        )


def validate_tr_correction_pack(
    pack_path: str | os.PathLike[str],
    *,
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an input ZIP and return a defensive copy of its manifest."""

    return copy.deepcopy(
        read_tr_correction_pack(
            pack_path, expected_input_sha256=expected_input_sha256
        ).manifest
    )


def validate_tr_correction_records(
    input_utterances: Sequence[Mapping[str, Any]],
    output_records: Sequence[Mapping[str, Any]],
    *,
    speech_holes: Sequence[Mapping[str, Any]] | None = None,
    asr_hallucination_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate exact echoes, cardinality and semantics of correction records."""

    trusted_inputs = validate_input_utterances(input_utterances)
    trusted_holes = validate_speech_holes(speech_holes)
    trusted_hallucinations = validate_asr_hallucination_records(
        asr_hallucination_records
    )
    if speech_holes is not None:
        _validate_hole_utterance_links(trusted_inputs, trusted_holes)
    if asr_hallucination_records is not None:
        _validate_hallucination_utterance_links(
            trusted_inputs, trusted_hallucinations
        )
    holes_by_uid = {hole["hole_uid"]: hole for hole in trusted_holes}
    hallucinations_by_utterance_uid = {
        candidate["utterance_uid"]: candidate
        for candidate in trusted_hallucinations
    }
    if isinstance(output_records, (str, bytes, bytearray)) or not isinstance(
        output_records, Sequence
    ):
        raise TRCorrectionError("output_records must be a sequence of JSON objects")

    raw_outputs = list(output_records)
    expected_uids = [record["utterance_uid"] for record in trusted_inputs]
    actual_uids: list[str] = []
    invalid_uid_positions: list[int] = []
    for position, raw in enumerate(raw_outputs, start=1):
        if not isinstance(raw, Mapping):
            raise TRCorrectionError(
                f"Correction output {position} must be a JSON object"
            )
        uid = raw.get("utterance_uid")
        if isinstance(uid, str):
            actual_uids.append(uid)
        else:
            invalid_uid_positions.append(position)
    if invalid_uid_positions:
        raise TRCorrectionError(
            f"Correction records have invalid utterance_uid at positions {invalid_uid_positions}"
        )
    duplicates = sorted(
        uid for uid in set(actual_uids) if actual_uids.count(uid) > 1
    )
    expected_set = set(expected_uids)
    actual_set = set(actual_uids)
    missing = sorted(expected_set.difference(actual_set))
    extra = sorted(actual_set.difference(expected_set))
    if len(raw_outputs) != len(trusted_inputs) or duplicates or missing or extra:
        raise TRCorrectionError(
            "Correction output identity/cardinality mismatch; "
            f"expected={len(trusted_inputs)}, actual={len(raw_outputs)}, "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )
    if actual_uids != expected_uids:
        mismatch = next(
            position
            for position, (expected, actual) in enumerate(
                zip(expected_uids, actual_uids), start=1
            )
            if expected != actual
        )
        raise TRCorrectionError(
            f"Correction output order mismatch at position {mismatch}: "
            f"expected {expected_uids[mismatch - 1]}, got {actual_uids[mismatch - 1]}"
        )

    validated: list[dict[str, Any]] = []
    unsupported_lexical_deletions: list[str] = []
    for position, (trusted, raw) in enumerate(
        zip(trusted_inputs, raw_outputs), start=1
    ):
        _require_exact_fields(raw, OUTPUT_RECORD_FIELDS, f"Correction {position}")
        uid = trusted["utterance_uid"]
        for field in IMMUTABLE_UTTERANCE_FIELDS:
            if type(raw.get(field)) is not type(trusted[field]) or raw.get(field) != trusted[field]:
                raise TRCorrectionError(
                    f"Correction {uid} changed immutable field {field}: "
                    f"expected {trusted[field]!r}, got {raw.get(field)!r}"
                )
        corrected = _require_string(raw.get("tr_corrected"), "tr_corrected")
        non_dialogue = raw.get("non_dialogue")
        review_required = raw.get("review_required")
        if not isinstance(non_dialogue, bool):
            raise TRCorrectionError(f"Correction {uid} non_dialogue must be boolean")
        if not isinstance(review_required, bool):
            raise TRCorrectionError(
                f"Correction {uid} review_required must be boolean"
            )
        audio_reviewed = raw.get("audio_reviewed")
        if not isinstance(audio_reviewed, bool):
            raise TRCorrectionError(
                f"Correction {uid} audio_reviewed must be boolean"
            )
        review_disposition = _require_string(
            raw.get("review_disposition"),
            "review_disposition",
            nonempty=True,
        )
        if review_disposition not in REVIEW_DISPOSITIONS:
            raise TRCorrectionError(
                f"Correction {uid} has invalid review_disposition: "
                f"{review_disposition!r}"
            )
        note = _require_string(raw.get("note"), "note")
        hole = holes_by_uid.get(uid)
        hallucination = hallucinations_by_utterance_uid.get(uid)
        has_lexical_deletion = (
            len(_lexical_tokens(corrected)) < len(_lexical_tokens(trusted["asr_text"]))
        )
        if hole is not None:
            if review_required:
                if audio_reviewed or review_disposition != "pending_audio_review":
                    raise TRCorrectionError(
                        f"Correction {uid} pending speech-hole review must use "
                        "audio_reviewed=false and pending_audio_review"
                    )
                if non_dialogue:
                    raise TRCorrectionError(
                        f"Correction {uid} cannot be non_dialogue before WAV review"
                    )
            else:
                if not audio_reviewed:
                    raise TRCorrectionError(
                        f"Correction {uid} speech-hole resolution requires "
                        "audio_reviewed=true"
                    )
                expected_disposition = (
                    "reviewed_non_dialogue"
                    if non_dialogue
                    else "confirmed_dialogue"
                )
                if review_disposition != expected_disposition:
                    raise TRCorrectionError(
                        f"Correction {uid} speech-hole resolution requires "
                        f"review_disposition={expected_disposition!r}"
                    )
                if not note.strip():
                    raise TRCorrectionError(
                        f"Correction {uid} WAV review requires a concrete note"
                    )
            if non_dialogue and corrected.strip():
                raise TRCorrectionError(
                    f"Correction {uid} marked non_dialogue must have empty tr_corrected"
                )
            if not non_dialogue and not review_required and not corrected.strip():
                raise TRCorrectionError(
                    f"Correction {uid} confirmed dialogue requires non-empty tr_corrected"
                )
        elif hallucination is not None:
            if review_required:
                if audio_reviewed or review_disposition != "pending_audio_review":
                    raise TRCorrectionError(
                        f"Correction {uid} pending ASR-hallucination review must use "
                        "audio_reviewed=false and pending_audio_review"
                    )
                if non_dialogue:
                    raise TRCorrectionError(
                        f"Correction {uid} cannot be discarded before WAV review"
                    )
            else:
                if not audio_reviewed:
                    raise TRCorrectionError(
                        f"Correction {uid} ASR-hallucination decision requires "
                        "audio_reviewed=true"
                    )
                expected_disposition = (
                    "discarded_asr_hallucination"
                    if non_dialogue
                    else "confirmed_dialogue"
                )
                if review_disposition != expected_disposition:
                    raise TRCorrectionError(
                        f"Correction {uid} ASR-hallucination decision requires "
                        f"review_disposition={expected_disposition!r}"
                    )
                if not note.strip():
                    raise TRCorrectionError(
                        f"Correction {uid} WAV review requires a concrete note"
                    )
            if non_dialogue and corrected.strip():
                raise TRCorrectionError(
                    f"Correction {uid} discarded ASR hallucination must have empty "
                    "tr_corrected"
                )
            if not non_dialogue and not corrected.strip():
                raise TRCorrectionError(
                    f"Correction {uid} dialogue requires non-empty tr_corrected"
                )
        else:
            is_pending_audio_review = (
                review_required
                and not audio_reviewed
                and review_disposition == "pending_audio_review"
            )
            if not is_pending_audio_review and (
                audio_reviewed or review_disposition != "not_applicable"
            ):
                raise TRCorrectionError(
                    f"Correction {uid} has no immutable audio-review record and must "
                    "remain pending or use audio_reviewed=false, "
                    "review_disposition=not_applicable"
                )
            if non_dialogue:
                raise TRCorrectionError(
                    f"Correction {uid} non_dialogue is allowed only for an exact "
                    "WAV-reviewed unresolved_vad_speech or "
                    "suspected_asr_hallucination record"
                )
            if not corrected.strip():
                raise TRCorrectionError(
                    f"Correction {uid} dialogue requires non-empty tr_corrected"
                )
        if has_lexical_deletion and hallucination is None:
            if not (
                review_required
                and not audio_reviewed
                and review_disposition == "pending_audio_review"
            ):
                unsupported_lexical_deletions.append(uid)
        validated.append(copy.deepcopy(dict(raw)))
    if unsupported_lexical_deletions:
        raise TRCorrectionError(
            f"Corrections {unsupported_lexical_deletions!r} delete lexical ASR "
            "token(s) without an exact hash-bound "
            "audio review. Set review_required=true with pending_audio_review, "
            "then rerun preparation using EXTRA_AUDIO_REVIEW_UIDS="
            f"{unsupported_lexical_deletions!r}."
        )
    return validated


def create_tr_correction_output(
    input_pack_path: str | os.PathLike[str],
    output_records: Sequence[Mapping[str, Any]],
    out_zip: str | os.PathLike[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Atomically create a correction output ZIP bound to an input pack SHA."""

    pack = read_tr_correction_pack(input_pack_path)
    records = validate_tr_correction_records(
        pack.utterances,
        output_records,
        speech_holes=pack.speech_holes,
        asr_hallucination_records=pack.asr_hallucination_records,
    )
    batches = _batch_slices(records, batch_size)
    payloads: dict[str, bytes] = {}
    descriptors: list[dict[str, Any]] = []
    for number, batch in enumerate(batches, start=1):
        name = f"corrected_batch_{number:03d}.jsonl"
        payload = _jsonl_bytes(batch)
        payloads[name] = payload
        descriptors.append(
            {
                "output_file": name,
                "utterance_count": len(batch),
                "first_utterance_uid": batch[0]["utterance_uid"],
                "last_utterance_uid": batch[-1]["utterance_uid"],
                "sha256": _sha256_bytes(payload),
            }
        )
    output_sha = compute_output_sha256(records)
    manifest: dict[str, Any] = {
        "format": OUTPUT_FORMAT,
        "format_version": FORMAT_VERSION,
        "episode": pack.manifest["episode"],
        "input_sha256": pack.manifest["input_sha256"],
        "output_sha256": output_sha,
        "utterance_count": len(records),
        "batch_count": len(batches),
        "batches": descriptors,
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    destination = Path(out_zip)
    member_order = ["manifest.json", *[item["output_file"] for item in descriptors]]
    _write_zip_atomic(
        destination,
        member_order,
        payloads,
        validate_temporary=lambda path: validate_tr_correction_output(
            input_pack_path, path
        ),
    )
    validated = validate_tr_correction_output(input_pack_path, destination)
    return copy.deepcopy(validated.manifest)


def validate_tr_correction_output(
    input_pack_path: str | os.PathLike[str],
    output_zip_path: str | os.PathLike[str],
) -> TRCorrectionOutputData:
    """Validate an output ZIP against the exact immutable input evidence."""

    pack = read_tr_correction_pack(input_pack_path)
    with _open_validated_zip(output_zip_path) as archive:
        names = [info.filename for info in archive.infolist()]
        if "manifest.json" not in names:
            raise TRCorrectionError("Correction output missing manifest.json")
        manifest = _parse_json_bytes(archive.read("manifest.json"), "manifest.json")
        if not isinstance(manifest, dict):
            raise TRCorrectionError("Output manifest.json must contain an object")
        _require_exact_fields(manifest, OUTPUT_MANIFEST_FIELDS, "Output manifest")
        if manifest.get("format") != OUTPUT_FORMAT:
            raise TRCorrectionError("Wrong correction output format")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise TRCorrectionError("Unsupported correction output format_version")
        if manifest.get("episode") != pack.manifest["episode"]:
            raise TRCorrectionError("Correction output episode mismatch")
        input_sha = pack.manifest["input_sha256"]
        if manifest.get("input_sha256") != input_sha:
            raise TRCorrectionError("Correction output input_sha256 mismatch")
        batch_count = _require_int(
            manifest.get("batch_count"), "output manifest.batch_count", minimum=1
        )
        expected_names = [
            f"corrected_batch_{number:03d}.jsonl"
            for number in range(1, batch_count + 1)
        ]
        expected_members = {"manifest.json", *expected_names}
        if set(names) != expected_members:
            raise TRCorrectionError(
                "Correction output member mismatch; "
                f"missing={sorted(expected_members.difference(names))}, "
                f"extra={sorted(set(names).difference(expected_members))}"
            )
        descriptors = manifest.get("batches")
        if not isinstance(descriptors, list) or len(descriptors) != batch_count:
            raise TRCorrectionError("Output batches do not match batch_count")
        records: list[dict[str, Any]] = []
        for expected_name, descriptor in zip(expected_names, descriptors):
            if not isinstance(descriptor, Mapping):
                raise TRCorrectionError("Every output batch descriptor must be an object")
            _require_exact_fields(
                descriptor,
                OUTPUT_BATCH_DESCRIPTOR_FIELDS,
                f"Output batch descriptor {expected_name}",
            )
            if descriptor.get("output_file") != expected_name:
                raise TRCorrectionError(
                    f"Output batch descriptor order mismatch: {expected_name}"
                )
            payload = archive.read(expected_name)
            if descriptor.get("sha256") != _sha256_bytes(payload):
                raise TRCorrectionError(f"Output batch SHA-256 mismatch: {expected_name}")
            batch_records = _parse_jsonl_bytes(payload, expected_name)
            if not batch_records:
                raise TRCorrectionError(f"Empty output batch: {expected_name}")
            declared_count = _require_int(
                descriptor.get("utterance_count"),
                f"{expected_name}.utterance_count",
                minimum=1,
            )
            if declared_count != len(batch_records):
                raise TRCorrectionError(f"Output batch count mismatch: {expected_name}")
            if descriptor.get("first_utterance_uid") != batch_records[0].get(
                "utterance_uid"
            ):
                raise TRCorrectionError(f"Output first UID mismatch: {expected_name}")
            if descriptor.get("last_utterance_uid") != batch_records[-1].get(
                "utterance_uid"
            ):
                raise TRCorrectionError(f"Output last UID mismatch: {expected_name}")
            records.extend(batch_records)

        validated_records = validate_tr_correction_records(
            pack.utterances,
            records,
            speech_holes=pack.speech_holes,
            asr_hallucination_records=pack.asr_hallucination_records,
        )
        utterance_count = _require_int(
            manifest.get("utterance_count"),
            "output manifest.utterance_count",
            minimum=1,
        )
        if utterance_count != len(validated_records):
            raise TRCorrectionError("Output manifest utterance_count mismatch")
        output_sha = compute_output_sha256(validated_records)
        if manifest.get("output_sha256") != output_sha:
            raise TRCorrectionError("Correction output output_sha256 mismatch")
        return TRCorrectionOutputData(
            manifest=copy.deepcopy(manifest),
            records=tuple(copy.deepcopy(validated_records)),
            input_sha256=input_sha,
            output_sha256=output_sha,
        )


__all__ = [
    "ASR_AUDIT_FIELDS",
    "ASR_HALLUCINATION_FIELDS",
    "DEFAULT_BATCH_SIZE",
    "EDITABLE_CORRECTION_FIELDS",
    "FORMAT_VERSION",
    "IMMUTABLE_UTTERANCE_FIELDS",
    "MAX_ASR_HALLUCINATION_AUDIO_BYTES",
    "MAX_ASR_HALLUCINATION_AUDIO_FILES",
    "MAX_ASR_HALLUCINATION_AUDIO_TOTAL_BYTES",
    "MAX_AUTOMATIC_ASR_HALLUCINATION_AUDIO_FILES",
    "MAX_EXPLICIT_ASR_HALLUCINATION_AUDIO_FILES",
    "OUTPUT_RECORD_FIELDS",
    "REVIEW_DISPOSITIONS",
    "SPEECH_HOLE_FIELDS",
    "TRCorrectionError",
    "TRCorrectionOutputData",
    "TRCorrectionPackData",
    "compute_input_sha256",
    "compute_output_sha256",
    "create_tr_correction_output",
    "create_tr_correction_pack",
    "read_tr_correction_pack",
    "validate_input_utterances",
    "validate_asr_hallucination_records",
    "validate_speech_holes",
    "validate_tr_correction_output",
    "validate_tr_correction_pack",
    "validate_tr_correction_records",
]
