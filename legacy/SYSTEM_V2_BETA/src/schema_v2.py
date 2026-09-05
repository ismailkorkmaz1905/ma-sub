"""Aligned-Turkish schema construction for the correction-first V2 flow."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .id_translation import validate_aligned_turkish_schema
from .schema import generate_block_uid


class SchemaV2Error(ValueError):
    """Raised when aligned segmentation cannot be locked for ID translation."""


_ALIGNMENT_COUNT_FIELDS = (
    "unaligned_word_count",
    "synthetic_timing_count",
    "negative_word_gap_count",
    "missing_alignment_score_count",
    "low_alignment_score_count",
    "overlong_alignment_word_count",
    "outward_drift_violation_count",
    "low_score_edited_token_count",
    "unreviewed_deleted_token_count",
)


def _alignment_report_parts(
    alignment_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str | None]:
    """Return counters and digest from a flat report or full aligner output."""

    nested = alignment_report.get("report")
    if nested is None:
        counters = alignment_report
    else:
        if not isinstance(nested, Mapping):
            raise SchemaV2Error("alignment_report.report must be a mapping")
        counters = nested
        for field in _ALIGNMENT_COUNT_FIELDS:
            if field in alignment_report and alignment_report[field] != nested.get(field):
                raise SchemaV2Error(
                    f"alignment_report has conflicting {field} values"
                )

    alignment_sha = alignment_report.get("alignment_sha256")
    nested_sha = counters.get("alignment_sha256")
    if nested is not None and nested_sha is not None and nested_sha != alignment_sha:
        raise SchemaV2Error("alignment_report has conflicting alignment_sha256 values")
    if not isinstance(alignment_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", alignment_sha
    ):
        raise SchemaV2Error("alignment_report is missing alignment_sha256")
    audio_sha = alignment_report.get("audio_sha256")
    if nested is not None and audio_sha is None:
        raise SchemaV2Error("full alignment_report is missing audio_sha256")
    if audio_sha is not None and (
        not isinstance(audio_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", audio_sha) is None
    ):
        raise SchemaV2Error("alignment_report has invalid audio_sha256")
    return counters, alignment_sha, audio_sha


def _digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaV2Error(f"Value is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_zero_alignment_count(
    counters: Mapping[str, Any], field: str, failure: str
) -> None:
    value = counters.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise SchemaV2Error(failure)


def build_aligned_turkish_schema(
    segmented_blocks: Sequence[Mapping[str, Any]],
    *,
    episode: int,
    alignment_report: Mapping[str, Any],
    speech_coverage_report: Mapping[str, Any],
    schema_version: str = "2.0",
) -> dict[str, Any]:
    """Lock readable cues only after corrected Turkish forced alignment.

    The independent alignment and VAD coverage digests are part of the schema
    identity.  Re-running either stage with different evidence therefore
    invalidates all downstream Indonesian translations.
    """

    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise SchemaV2Error("episode must be a positive integer")
    if not isinstance(schema_version, str) or schema_version.split(".", 1)[0] != "2":
        raise SchemaV2Error("schema_version must be a V2 version")
    if not isinstance(alignment_report, Mapping):
        raise SchemaV2Error("alignment_report must be a mapping")
    if not isinstance(speech_coverage_report, Mapping):
        raise SchemaV2Error("speech_coverage_report must be a mapping")
    alignment_counts, alignment_sha, audio_sha = _alignment_report_parts(
        alignment_report
    )
    for field, failure in (
        ("unaligned_word_count", "alignment_report contains unaligned words"),
        ("synthetic_timing_count", "alignment_report contains synthetic timing"),
        ("negative_word_gap_count", "alignment_report contains negative word gaps"),
        (
            "missing_alignment_score_count",
            "alignment_report contains missing alignment scores",
        ),
        (
            "low_alignment_score_count",
            "alignment_report contains low alignment scores",
        ),
        (
            "overlong_alignment_word_count",
            "alignment_report contains overlong aligned words",
        ),
        (
            "outward_drift_violation_count",
            "alignment_report contains excessive outward drift",
        ),
        (
            "low_score_edited_token_count",
            "alignment_report contains low-score edited tokens",
        ),
        (
            "unreviewed_deleted_token_count",
            "alignment_report contains unreviewed deleted tokens",
        ),
    ):
        _require_zero_alignment_count(alignment_counts, field, failure)
    if speech_coverage_report.get("unresolved_speech_region_count") != 0:
        raise SchemaV2Error("speech coverage still contains unresolved regions")

    coverage_sha = _digest(speech_coverage_report)

    blocks: list[dict[str, Any]] = []
    previous_end = -1
    for position, source in enumerate(segmented_blocks, start=1):
        if not isinstance(source, Mapping):
            raise SchemaV2Error(f"segmented block {position} is not an object")
        try:
            start_ms = int(source["start_ms"])
            end_ms = int(source["end_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaV2Error(f"segmented block {position} has invalid timing") from exc
        tr_text = str(source.get("primary_text", source.get("tr_text", ""))).strip()
        if start_ms < 0 or end_ms <= start_ms or start_ms < previous_end:
            raise SchemaV2Error(f"segmented block {position} has unsafe timing")
        if not tr_text:
            raise SchemaV2Error(f"segmented block {position} has empty Turkish text")
        vad_info = source.get("vad_info")
        if not isinstance(vad_info, Mapping):
            raise SchemaV2Error(f"segmented block {position} has no alignment evidence")
        provenance = vad_info.get("alignment_provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise SchemaV2Error(
                f"segmented block {position} has no alignment_provenance"
            )
        provenance_copy = copy.deepcopy(dict(provenance))
        if provenance_copy.get("alignment_sha256") != alignment_sha:
            raise SchemaV2Error(
                f"segmented block {position} alignment digest mismatch"
            )
        if audio_sha is not None and provenance_copy.get("audio_sha256") != audio_sha:
            raise SchemaV2Error(
                f"segmented block {position} audio digest mismatch"
            )
        uid = generate_block_uid(
            episode,
            position,
            start_ms,
            end_ms,
            tr_text,
            schema_version,
        )
        blocks.append(
            {
                "block_uid": uid,
                "episode": episode,
                "block_index": position,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "tr_text": tr_text,
                "alignment_provenance": provenance_copy,
                "context_before": str(source.get("context_before", "")),
                "context_after": str(source.get("context_after", "")),
                "risk_flags": list(source.get("risk_flags", [])),
            }
        )
        previous_end = end_ms

    if not blocks:
        raise SchemaV2Error("segmented_blocks cannot be empty")
    draft = {
        "schema_version": schema_version,
        "episode": episode,
        "block_count": len(blocks),
        "alignment_sha256": alignment_sha,
        "speech_coverage_sha256": coverage_sha,
        "blocks": blocks,
    }
    if audio_sha is not None:
        draft["audio_sha256"] = audio_sha
    try:
        return validate_aligned_turkish_schema(draft)
    except Exception as exc:
        raise SchemaV2Error(f"Aligned Turkish schema validation failed: {exc}") from exc


__all__ = ["SchemaV2Error", "build_aligned_turkish_schema"]
