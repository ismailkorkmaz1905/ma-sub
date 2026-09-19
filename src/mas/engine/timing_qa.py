"""Hard acoustic/readability gates for the subtitle timeline.

V1 proved only that the final SRT retained the timestamps stored in its own
schema.  This module deliberately requires two independent upstream artifacts:
the forced-alignment report and the full-audio VAD coverage report.  A timeline
cannot pass merely because it agrees with itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any

from .speaker import overlap_is_unsafe, speaker_id


class TimingQAV2Error(RuntimeError):
    """Raised when a timeline is not safe to publish."""


@dataclass(frozen=True)
class TimingQAV2Config:
    minimum_duration_ms: int = 700
    maximum_duration_ms: int = 7000
    maximum_cps: float = 20.0
    duplicate_short_threshold_ms: int = 1200
    alignment_timing_source: str = "whisperx_ctc_forced_alignment"

    def __post_init__(self) -> None:
        if self.minimum_duration_ms < 1:
            raise ValueError("minimum_duration_ms must be positive")
        if self.maximum_duration_ms < self.minimum_duration_ms:
            raise ValueError("maximum_duration_ms cannot be below minimum_duration_ms")
        if self.maximum_cps <= 0:
            raise ValueError("maximum_cps must be positive")
        if self.duplicate_short_threshold_ms < self.minimum_duration_ms:
            raise ValueError(
                "duplicate_short_threshold_ms cannot be below minimum_duration_ms"
            )
        if not self.alignment_timing_source.strip():
            raise ValueError("alignment_timing_source cannot be empty")


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w\u00e7\u011f\u0131\u00f6\u015f\u00fc]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _visible_char_count(value: Any) -> int:
    return len(re.sub(r"\s+", "", str(value or "")))


def _require_nonnegative_count(report: Mapping[str, Any], field: str) -> int:
    value = report.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TimingQAV2Error(f"{field} must be a non-negative integer")
    return value


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
    "review_alignment_score_count",
)


def _alignment_report_parts(
    alignment_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    """Accept either the flat QA report or full forced-aligner output."""

    nested = alignment_report.get("report")
    if nested is None:
        counters = alignment_report
    else:
        if not isinstance(nested, Mapping):
            raise TimingQAV2Error("alignment_report.report must be a mapping")
        counters = nested
        for field in _ALIGNMENT_COUNT_FIELDS:
            if field in alignment_report and alignment_report[field] != nested.get(field):
                raise TimingQAV2Error(
                    f"alignment_report has conflicting {field} values"
                )

    alignment_sha = alignment_report.get("alignment_sha256")
    nested_sha = counters.get("alignment_sha256")
    if nested is not None and nested_sha is not None and nested_sha != alignment_sha:
        raise TimingQAV2Error(
            "alignment_report has conflicting alignment_sha256 values"
        )
    if not isinstance(alignment_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", alignment_sha
    ):
        raise TimingQAV2Error("alignment_report has no valid alignment_sha256")
    return counters, alignment_sha


def _acoustic_timing_issues(blocks, source_words):
    if not isinstance(source_words, list) or any(
        not isinstance(word, Mapping) for word in source_words
    ):
        raise TimingQAV2Error("alignment_report.words must be a list of acoustic words")
    lanes = {}
    for word in source_words:
        try:
            lane = speaker_id(word, "alignment word")
            start, end = word["start_ms"], word["end_ms"]
            if (
                isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or start < 0 or end <= start
            ):
                raise ValueError("invalid acoustic word interval")
            tokens = _normalise_text(word.get("text", word.get("word", ""))).split()
        except (KeyError, TypeError, ValueError) as exc:
            raise TimingQAV2Error(f"invalid alignment word: {exc}") from exc
        lanes.setdefault(lane, {"tokens": [], "blocks": []})["tokens"].extend(
            (token, word) for token in tokens
        )
    for block in blocks:
        try:
            lane = speaker_id(block, "block")
        except ValueError:
            continue  # The main loop reports malformed speaker identity.
        lanes.setdefault(lane, {"tokens": [], "blocks": []})["blocks"].append(block)
    issues = []
    for lane in lanes.values():
        expected_tokens = [token for token, _ in lane["tokens"]]
        actual_tokens = [
            token
            for block in lane["blocks"]
            for token in _normalise_text(
                block.get("primary_text", block.get("tr_text", ""))
            ).split()
        ]
        if actual_tokens != expected_tokens:
            issues.append({
                "code": "acoustic_text_mismatch",
                "block_uids": [str(block.get("block_uid", "")) for block in lane["blocks"]],
                "message": "cue text does not preserve complete ordered acoustic words",
            })
            continue
        cursor = 0
        for block in lane["blocks"]:
            count = len(_normalise_text(
                block.get("primary_text", block.get("tr_text", ""))
            ).split())
            owned_words = [word for _, word in lane["tokens"][cursor:cursor + count]]
            cursor += count
            try:
                start, end = int(block["start_ms"]), int(block["end_ms"])
            except (KeyError, TypeError, ValueError):
                continue  # The main loop reports malformed cue timing.
            if not owned_words:
                continue
            first_start = min(word["start_ms"] for word in owned_words)
            last_end = max(word["end_ms"] for word in owned_words)
            if start != first_start or end < last_end:
                issues.append({
                    "code": "cue_acoustic_boundary_mismatch",
                    "block_index": block.get("block_index"),
                    "block_uid": str(block.get("block_uid", "")),
                    "actual_start_ms": start,
                    "actual_end_ms": end,
                    "first_word_start_ms": first_start,
                    "last_word_end_ms": last_end,
                    "utterance_uids": list(dict.fromkeys(
                        word["utterance_uid"] for word in owned_words if word.get("utterance_uid")
                    )),
                })
    return issues


def run_timing_qa_v2(
    blocks: Sequence[Mapping[str, Any]],
    *,
    speech_coverage_report: Mapping[str, Any],
    alignment_report: Mapping[str, Any],
    id_text_by_uid: Mapping[str, str] | None = None,
    config: TimingQAV2Config | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready hard-gate report for an aligned final timeline.

    ``blocks`` must be built from corrected Turkish words after forced
    alignment.  ``speech_coverage_report`` must come from the independent VAD
    inventory, while ``alignment_report`` must describe the CTC alignment run.
    Indonesian text is optional before translation and mandatory at publish.
    """

    settings = config or TimingQAV2Config()
    if not isinstance(speech_coverage_report, Mapping):
        raise TimingQAV2Error("speech_coverage_report must be a mapping")
    if not isinstance(alignment_report, Mapping):
        raise TimingQAV2Error("alignment_report must be a mapping")

    unresolved_speech_region_count = _require_nonnegative_count(
        speech_coverage_report, "unresolved_speech_region_count"
    )
    alignment_counts, alignment_sha256 = _alignment_report_parts(alignment_report)
    unaligned_word_count = _require_nonnegative_count(
        alignment_counts, "unaligned_word_count"
    )
    synthetic_timing_count = _require_nonnegative_count(
        alignment_counts, "synthetic_timing_count"
    )
    negative_word_gap_count = _require_nonnegative_count(
        alignment_counts, "negative_word_gap_count"
    )
    missing_alignment_score_count = _require_nonnegative_count(
        alignment_counts, "missing_alignment_score_count"
    )
    low_alignment_score_count = _require_nonnegative_count(
        alignment_counts, "low_alignment_score_count"
    )
    overlong_alignment_word_count = _require_nonnegative_count(
        alignment_counts, "overlong_alignment_word_count"
    )
    outward_drift_violation_count = _require_nonnegative_count(
        alignment_counts, "outward_drift_violation_count"
    )
    low_score_edited_token_count = _require_nonnegative_count(
        alignment_counts, "low_score_edited_token_count"
    )
    unreviewed_deleted_token_count = _require_nonnegative_count(
        alignment_counts, "unreviewed_deleted_token_count"
    )
    review_alignment_score_count = _require_nonnegative_count(
        alignment_counts, "review_alignment_score_count"
    )

    issues: list[dict[str, Any]] = []
    short_cue_count = 0
    high_cps_tr_count = 0
    high_cps_id_count = 0
    overlap_count = 0
    invalid_timing_count = 0
    alignment_provenance_mismatch_count = 0
    adjacent_short_duplicate_count = 0
    missing_id_count = 0
    previous: Mapping[str, Any] | None = None
    active_blocks: list[tuple[int, str | None]] = []

    for expected_index, block in enumerate(blocks, start=1):
        try:
            block_index = int(block["block_index"])
            start_ms = int(block["start_ms"])
            end_ms = int(block["end_ms"])
            uid = str(block["block_uid"])
        except (KeyError, TypeError, ValueError) as exc:
            invalid_timing_count += 1
            issues.append(
                {
                    "code": "malformed_block",
                    "block_index": expected_index,
                    "message": str(exc),
                }
            )
            previous = block
            continue
        if block_index != expected_index or start_ms < 0 or end_ms <= start_ms or not uid:
            invalid_timing_count += 1
            issues.append(
                {
                    "code": "invalid_timing_or_identity",
                    "block_index": expected_index,
                    "block_uid": uid,
                }
            )
        try:
            current_speaker_id = speaker_id(block, f"block {expected_index}")
        except ValueError as exc:
            current_speaker_id = None
            invalid_timing_count += 1
            issues.append(
                {
                    "code": "invalid_speaker_id",
                    "block_index": expected_index,
                    "block_uid": uid,
                    "message": str(exc),
                }
            )
        active_blocks = [item for item in active_blocks if item[0] > start_ms]
        if any(
            overlap_is_unsafe(prior_speaker, current_speaker_id)
            for _, prior_speaker in active_blocks
        ):
            overlap_count += 1
            issues.append(
                {
                    "code": "subtitle_overlap",
                    "block_index": expected_index,
                    "block_uid": uid,
                    "actual": start_ms,
                    "expected": "no same/unknown-speaker overlap",
                }
            )
        active_blocks.append((end_ms, current_speaker_id))

        duration_ms = end_ms - start_ms
        if duration_ms > settings.maximum_duration_ms:
            invalid_timing_count += 1
            issues.append(
                {
                    "code": "overlong_cue",
                    "block_index": expected_index,
                    "block_uid": uid,
                    "actual": duration_ms,
                    "expected": f"<= {settings.maximum_duration_ms} ms",
                }
            )
        if duration_ms < settings.minimum_duration_ms:
            short_cue_count += 1
            issues.append(
                {
                    "code": "short_cue",
                    "block_index": expected_index,
                    "block_uid": uid,
                    "actual": duration_ms,
                    "expected": f">= {settings.minimum_duration_ms}",
                }
            )

        tr_text = str(block.get("primary_text", block.get("tr_text", ""))).strip()
        tr_cps = _visible_char_count(tr_text) / max(duration_ms / 1000.0, 0.001)
        if tr_cps > settings.maximum_cps:
            high_cps_tr_count += 1
            issues.append(
                {
                    "code": "high_cps_tr",
                    "block_index": expected_index,
                    "block_uid": uid,
                    "actual": round(tr_cps, 2),
                    "expected": f"<= {settings.maximum_cps}",
                }
            )

        if id_text_by_uid is not None:
            id_text = str(id_text_by_uid.get(uid, "")).strip()
            if not id_text:
                missing_id_count += 1
                issues.append(
                    {
                        "code": "missing_id",
                        "block_index": expected_index,
                        "block_uid": uid,
                    }
                )
            else:
                id_cps = _visible_char_count(id_text) / max(
                    duration_ms / 1000.0, 0.001
                )
                if id_cps > settings.maximum_cps:
                    high_cps_id_count += 1
                    issues.append(
                        {
                            "code": "high_cps_id",
                            "block_index": expected_index,
                            "block_uid": uid,
                            "actual": round(id_cps, 2),
                            "expected": f"<= {settings.maximum_cps}",
                        }
                    )

        vad_info = block.get("vad_info")
        if not isinstance(vad_info, Mapping):
            # Locked V2 ID schemas expose the same object directly.
            direct = block.get("alignment_provenance")
            vad_info = {"alignment_provenance": direct} if isinstance(direct, Mapping) else None
        timing_sources = []
        block_alignment_sha = ""
        if isinstance(vad_info, Mapping):
            provenance = vad_info.get("alignment_provenance", vad_info)
            if not isinstance(provenance, Mapping):
                provenance = {}
            source_value = provenance.get(
                "word_timing_sources",
                provenance.get("timing_sources", provenance.get("timing_source")),
            )
            if isinstance(source_value, str):
                timing_sources = [source_value]
            elif isinstance(source_value, Sequence):
                timing_sources = [str(item) for item in source_value]
            block_alignment_sha = str(provenance.get("alignment_sha256", ""))
        if (
            not timing_sources
            or any(item != settings.alignment_timing_source for item in timing_sources)
            or block_alignment_sha != alignment_sha256
        ):
            alignment_provenance_mismatch_count += 1
            issues.append(
                {
                    "code": "alignment_provenance_mismatch",
                    "block_index": expected_index,
                    "block_uid": uid,
                }
            )

        if previous is not None:
            previous_text = _normalise_text(
                previous.get("primary_text", previous.get("tr_text", ""))
            )
            try:
                previous_speaker_id = speaker_id(previous, "previous block")
            except ValueError:
                previous_speaker_id = None
            if (
                previous_text
                and previous_text == _normalise_text(tr_text)
                and overlap_is_unsafe(previous_speaker_id, current_speaker_id)
                and (
                    duration_ms <= settings.duplicate_short_threshold_ms
                    or int(previous.get("end_ms", 0))
                    - int(previous.get("start_ms", 0))
                    <= settings.duplicate_short_threshold_ms
                )
            ):
                adjacent_short_duplicate_count += 1
                issues.append(
                    {
                        "code": "adjacent_short_duplicate",
                        "block_index": expected_index,
                        "block_uid": uid,
                    }
                )
        previous = block

    word_ownership_checked = "words" in alignment_report or "report" in alignment_report
    if word_ownership_checked:
        acoustic_issues = _acoustic_timing_issues(blocks, alignment_report.get("words"))
        invalid_timing_count += len(acoustic_issues)
        issues.extend(acoustic_issues)

    counts = {
        "unresolved_speech_region_count": unresolved_speech_region_count,
        "unaligned_word_count": unaligned_word_count,
        "synthetic_timing_count": synthetic_timing_count,
        "negative_word_gap_count": negative_word_gap_count,
        "missing_alignment_score_count": missing_alignment_score_count,
        "low_alignment_score_count": low_alignment_score_count,
        "overlong_alignment_word_count": overlong_alignment_word_count,
        "outward_drift_violation_count": outward_drift_violation_count,
        "low_score_edited_token_count": low_score_edited_token_count,
        "unreviewed_deleted_token_count": unreviewed_deleted_token_count,
        "short_cue_count": short_cue_count,
        "high_cps_tr_count": high_cps_tr_count,
        "high_cps_id_count": high_cps_id_count,
        "overlap_count": overlap_count,
        "invalid_timing_count": invalid_timing_count,
        "alignment_provenance_mismatch_count": alignment_provenance_mismatch_count,
        "adjacent_short_duplicate_count": adjacent_short_duplicate_count,
        "missing_id_count": missing_id_count,
    }
    passed = all(value == 0 for value in counts.values())
    return {
        "report_version": 2,
        "passed": passed,
        "block_count": len(blocks),
        "alignment_sha256": alignment_sha256,
        "word_ownership_checked": word_ownership_checked,
        "config": asdict(settings),
        # This is an explicit pilot-review warning.  Scores below the hard
        # 0.30 floor are rejected upstream; scores in [0.30, 0.55) are kept
        # visible for operator spot-checking without pretending they are a
        # mechanically resolvable publication failure.
        "review_alignment_score_count": review_alignment_score_count,
        **counts,
        "issues": issues,
    }


def assert_timing_qa_v2(report: Mapping[str, Any]) -> None:
    """Raise unless the complete V2 timing report passes every hard gate."""

    if report.get("passed") is True:
        return
    fields = (
        "unresolved_speech_region_count",
        "unaligned_word_count",
        "synthetic_timing_count",
        "negative_word_gap_count",
        "missing_alignment_score_count",
        "low_alignment_score_count",
        "overlong_alignment_word_count",
        "outward_drift_violation_count",
        "low_score_edited_token_count",
        "unreviewed_deleted_token_count",
        "short_cue_count",
        "high_cps_tr_count",
        "high_cps_id_count",
        "overlap_count",
        "invalid_timing_count",
        "alignment_provenance_mismatch_count",
        "adjacent_short_duplicate_count",
        "missing_id_count",
    )
    failures = [f"{field}={report.get(field)!r}" for field in fields if report.get(field) != 0]
    if not failures:
        failures = ["passed is not true"]
    raise TimingQAV2Error("V2 timing QA failed; required zero: " + ", ".join(failures))


__all__ = [
    "TimingQAV2Config",
    "TimingQAV2Error",
    "assert_timing_qa_v2",
    "run_timing_qa_v2",
]
