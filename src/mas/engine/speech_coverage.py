"""Independent speech/VAD coverage analysis for subtitle preparation.

The ASR word list cannot prove that all audible speech was transcribed: a
missed utterance simply creates no word record.  This module compares an
independently produced VAD timeline with the timed-word timeline and reports
speech spans which still need transcription or an explicit human
``non_dialogue`` review.

The implementation is deliberately model- and audio-independent.  It accepts
JSON-like records, validates their timing strictly, and returns JSON-ready
primitives only.  It never guesses around malformed, overlapping, or
out-of-order source records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any


class SpeechCoverageError(ValueError):
    """Raised when coverage inputs or settings are unsafe or malformed."""


@dataclass(frozen=True)
class SpeechCoverageConfig:
    """Thresholds used by :func:`analyze_speech_coverage`.

    Short gaps between adjacent VAD records are joined before analysis.
    Timed words are padded, then nearby word coverage is joined so normal
    inter-word pauses do not look like missing dialogue.  A speech region is
    flagged when it contains a remaining gap at least ``min_hole_ms`` long or
    its total effective coverage falls below ``min_region_coverage_ratio``.
    """

    vad_merge_gap_ms: int = 150
    word_padding_ms: int = 120
    word_merge_gap_ms: int = 300
    min_hole_ms: int = 750
    min_region_coverage_ratio: float = 0.70
    rescue_padding_ms: int = 500
    rescue_merge_gap_ms: int = 250
    # A timed word must not begin/end deep in non-speech merely because some
    # other portion touches VAD. This applies to each contiguous outside run.
    max_word_outside_speech_ms: int = 120

    def __post_init__(self) -> None:
        for name in (
            "vad_merge_gap_ms",
            "word_padding_ms",
            "word_merge_gap_ms",
            "rescue_padding_ms",
            "rescue_merge_gap_ms",
            "max_word_outside_speech_ms",
        ):
            _require_nonnegative_int(getattr(self, name), f"config.{name}")
        _require_positive_int(self.min_hole_ms, "config.min_hole_ms")
        ratio = self.min_region_coverage_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise SpeechCoverageError(
                "config.min_region_coverage_ratio must be a finite number"
            )
        if not math.isfinite(float(ratio)) or not 0.0 <= float(ratio) <= 1.0:
            raise SpeechCoverageError(
                "config.min_region_coverage_ratio must be between 0 and 1"
            )


@dataclass(frozen=True)
class _Interval:
    start_ms: int
    end_ms: int
    source_record_indices: tuple[int, ...]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpeechCoverageError(f"{path} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpeechCoverageError(f"{path} must be a positive integer")
    return value


# Canonical V2 beta publication policy. Raw diagnostic runs may deliberately
# use other thresholds, but their artifacts are not eligible for final output.
#
# The first full-length pilot showed that the earlier 40/40/150 ms policy
# classified ordinary within-utterance pauses as missing speech.  These values
# preserve independent VAD coverage while requiring a materially long gap
# before launching an expensive rescue pass.
V2_BETA_COVERAGE_POLICY_LABEL = "v2-beta-full-length-dialogue-v1"
V2_BETA_SPEECH_COVERAGE_CONFIG = SpeechCoverageConfig(
    vad_merge_gap_ms=150,
    word_padding_ms=120,
    word_merge_gap_ms=300,
    min_hole_ms=750,
    min_region_coverage_ratio=0.70,
    rescue_padding_ms=500,
    rescue_merge_gap_ms=250,
    max_word_outside_speech_ms=120,
)


def require_v2_beta_speech_coverage_config(
    config: SpeechCoverageConfig,
) -> SpeechCoverageConfig:
    """Return ``config`` only when it exactly matches the beta publish policy."""

    if not isinstance(config, SpeechCoverageConfig):
        raise SpeechCoverageError("config must be a SpeechCoverageConfig")
    if config != V2_BETA_SPEECH_COVERAGE_CONFIG:
        raise SpeechCoverageError(
            "speech coverage config does not match the canonical V2 beta "
            "publication policy"
        )
    return config


def _records_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SpeechCoverageError(f"{label} must be a sequence of records")
    return value


def _validated_intervals(records: Any, label: str) -> list[_Interval]:
    """Return strict, ordered, non-overlapping intervals.

    Touching intervals are valid.  Overlap is not repaired because doing so can
    hide timestamp corruption in either the independent VAD or ASR evidence.
    """

    values = _records_sequence(records, label)
    intervals: list[_Interval] = []
    previous: _Interval | None = None
    for position, record in enumerate(values, start=1):
        if not isinstance(record, Mapping):
            raise SpeechCoverageError(f"{label}[{position}] must be an object")
        try:
            start_ms = _require_nonnegative_int(
                record["start_ms"], f"{label}[{position}].start_ms"
            )
            end_ms = _require_nonnegative_int(
                record["end_ms"], f"{label}[{position}].end_ms"
            )
        except KeyError as exc:
            raise SpeechCoverageError(
                f"{label}[{position}] is missing {exc.args[0]}"
            ) from exc
        if end_ms <= start_ms:
            raise SpeechCoverageError(
                f"{label}[{position}] end_ms must be greater than start_ms"
            )
        current = _Interval(start_ms, end_ms, (position,))
        if previous is not None:
            if current.start_ms < previous.start_ms:
                raise SpeechCoverageError(
                    f"{label}[{position}] is out of chronological order"
                )
            if current.start_ms < previous.end_ms:
                raise SpeechCoverageError(
                    f"{label}[{position}] overlaps {label}[{position - 1}]"
                )
        intervals.append(current)
        previous = current
    return intervals


def _merge_intervals(intervals: Sequence[_Interval], max_gap_ms: int) -> list[_Interval]:
    if not intervals:
        return []
    merged: list[_Interval] = [intervals[0]]
    for interval in intervals[1:]:
        previous = merged[-1]
        if interval.start_ms <= previous.end_ms + max_gap_ms:
            merged[-1] = _Interval(
                previous.start_ms,
                max(previous.end_ms, interval.end_ms),
                previous.source_record_indices + interval.source_record_indices,
            )
        else:
            merged.append(interval)
    return merged


def _effective_word_intervals(
    words: Sequence[_Interval], config: SpeechCoverageConfig
) -> list[_Interval]:
    padded = [
        _Interval(
            max(0, interval.start_ms - config.word_padding_ms),
            interval.end_ms + config.word_padding_ms,
            interval.source_record_indices,
        )
        for interval in words
    ]
    return _merge_intervals(padded, config.word_merge_gap_ms)


def _intersection(region: _Interval, intervals: Sequence[_Interval]) -> list[_Interval]:
    result: list[_Interval] = []
    for interval in intervals:
        if interval.end_ms <= region.start_ms:
            continue
        if interval.start_ms >= region.end_ms:
            break
        start_ms = max(region.start_ms, interval.start_ms)
        end_ms = min(region.end_ms, interval.end_ms)
        if end_ms > start_ms:
            result.append(
                _Interval(start_ms, end_ms, interval.source_record_indices)
            )
    # Intervals are already disjoint, but clipping can make touching boundaries.
    return _merge_intervals(result, 0)


def _subtract(region: _Interval, covered: Sequence[_Interval]) -> list[_Interval]:
    gaps: list[_Interval] = []
    cursor = region.start_ms
    for interval in covered:
        if interval.start_ms > cursor:
            gaps.append(_Interval(cursor, interval.start_ms, ()))
        cursor = max(cursor, interval.end_ms)
    if cursor < region.end_ms:
        gaps.append(_Interval(cursor, region.end_ms, ()))
    return gaps


def _interval_dict(interval: _Interval) -> dict[str, Any]:
    return {
        "start_ms": interval.start_ms,
        "end_ms": interval.end_ms,
        "duration_ms": interval.duration_ms,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 6)


def _normalize_reviews(
    reviewed_non_dialogue: Any,
    speech_regions: Sequence[_Interval],
) -> list[dict[str, Any]]:
    records = _records_sequence(reviewed_non_dialogue, "reviewed_non_dialogue")
    intervals = _validated_intervals(records, "reviewed_non_dialogue")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, (record, interval) in enumerate(zip(records, intervals), start=1):
        assert isinstance(record, Mapping)  # Established by _validated_intervals.
        classification = record.get("classification")
        review_status = record.get("review_status")
        reason = record.get("reason")
        if classification != "non_dialogue":
            raise SpeechCoverageError(
                f"reviewed_non_dialogue[{position}].classification must be "
                "'non_dialogue'"
            )
        if review_status != "reviewed":
            raise SpeechCoverageError(
                f"reviewed_non_dialogue[{position}].review_status must be 'reviewed'"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise SpeechCoverageError(
                f"reviewed_non_dialogue[{position}].reason must be non-empty"
            )
        raw_review_id = record.get("review_id", f"review-{position:04d}")
        if not isinstance(raw_review_id, str) or not raw_review_id.strip():
            raise SpeechCoverageError(
                f"reviewed_non_dialogue[{position}].review_id must be non-empty"
            )
        review_id = raw_review_id.strip()
        if review_id in seen_ids:
            raise SpeechCoverageError(f"duplicate reviewed_non_dialogue review_id: {review_id}")
        seen_ids.add(review_id)

        containing_regions = [
            index
            for index, region in enumerate(speech_regions, start=1)
            if region.start_ms <= interval.start_ms
            and interval.end_ms <= region.end_ms
        ]
        if len(containing_regions) != 1:
            raise SpeechCoverageError(
                f"reviewed_non_dialogue[{position}] must be fully contained in "
                "one merged VAD speech region"
            )
        normalized.append(
            {
                "review_id": review_id,
                "speech_region_index": containing_regions[0],
                "start_ms": interval.start_ms,
                "end_ms": interval.end_ms,
                "duration_ms": interval.duration_ms,
                "classification": "non_dialogue",
                "review_status": "reviewed",
                "reason": reason.strip(),
                "matched_issue_ids": [],
                "partially_overlapping_issue_ids": [],
            }
        )
    return normalized


def _normalize_dialogue_reviews(
    reviewed_dialogue: Any,
) -> tuple[list[dict[str, Any]], list[_Interval]]:
    """Validate exact human-reviewed speech intervals.

    These records extend—not replace—the independent VAD inventory. They are
    accepted only from the higher-level hash-bound audio-review route; this
    pure layer deliberately requires explicit classification/status/reason and
    never infers dialogue from text alone.
    """

    records = _records_sequence(reviewed_dialogue, "reviewed_dialogue")
    intervals = _validated_intervals(records, "reviewed_dialogue")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, (record, interval) in enumerate(zip(records, intervals), start=1):
        assert isinstance(record, Mapping)
        if record.get("classification") != "dialogue":
            raise SpeechCoverageError(
                f"reviewed_dialogue[{position}].classification must be 'dialogue'"
            )
        if record.get("review_status") != "reviewed":
            raise SpeechCoverageError(
                f"reviewed_dialogue[{position}].review_status must be 'reviewed'"
            )
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SpeechCoverageError(
                f"reviewed_dialogue[{position}].reason must be non-empty"
            )
        review_id = record.get("review_id")
        if not isinstance(review_id, str) or not review_id.strip():
            raise SpeechCoverageError(
                f"reviewed_dialogue[{position}].review_id must be non-empty"
            )
        review_id = review_id.strip()
        if review_id in seen_ids:
            raise SpeechCoverageError(f"duplicate reviewed_dialogue review_id: {review_id}")
        seen_ids.add(review_id)
        normalized.append(
            {
                "review_id": review_id,
                "start_ms": interval.start_ms,
                "end_ms": interval.end_ms,
                "duration_ms": interval.duration_ms,
                "classification": "dialogue",
                "review_status": "reviewed",
                "reason": reason.strip(),
            }
        )
    return normalized, intervals


def _outside_runs(interval: _Interval, speech_regions: Sequence[_Interval]) -> list[_Interval]:
    """Return the exact portions of ``interval`` outside the speech union."""

    inside = _intersection(interval, speech_regions)
    return _subtract(interval, inside)


def build_rescue_spans(
    coverage_issues: Sequence[Mapping[str, Any]],
    *,
    padding_ms: int = 500,
    merge_gap_ms: int = 250,
) -> list[dict[str, Any]]:
    """Build padded, ordered, non-overlapping spans for rescue ASR passes.

    Only issues classified as ``unresolved_speech`` are included.  Inputs must
    themselves be ordered and non-overlapping; padding may create overlap, in
    which case the resulting rescue spans are safely merged.
    """

    _require_nonnegative_int(padding_ms, "padding_ms")
    _require_nonnegative_int(merge_gap_ms, "merge_gap_ms")
    values = _records_sequence(coverage_issues, "coverage_issues")
    unresolved_records: list[Mapping[str, Any]] = []
    for position, issue in enumerate(values, start=1):
        if not isinstance(issue, Mapping):
            raise SpeechCoverageError(f"coverage_issues[{position}] must be an object")
        if issue.get("classification") == "unresolved_speech":
            unresolved_records.append(issue)
    intervals = _validated_intervals(unresolved_records, "unresolved_coverage_issues")

    padded: list[dict[str, Any]] = []
    for position, (record, interval) in enumerate(
        zip(unresolved_records, intervals), start=1
    ):
        issue_id = record.get("issue_id", f"issue-{position:04d}")
        if not isinstance(issue_id, str) or not issue_id.strip():
            raise SpeechCoverageError(
                f"unresolved_coverage_issues[{position}].issue_id must be non-empty"
            )
        speech_region_index = record.get("speech_region_index")
        if (
            isinstance(speech_region_index, bool)
            or not isinstance(speech_region_index, int)
            or speech_region_index <= 0
        ):
            raise SpeechCoverageError(
                f"unresolved_coverage_issues[{position}].speech_region_index must "
                "be a positive integer"
            )
        reasons_value = record.get("reasons", [])
        if not isinstance(reasons_value, Sequence) or isinstance(
            reasons_value, (str, bytes, bytearray)
        ):
            raise SpeechCoverageError(
                f"unresolved_coverage_issues[{position}].reasons must be a sequence"
            )
        reasons = []
        for reason in reasons_value:
            if not isinstance(reason, str) or not reason.strip():
                raise SpeechCoverageError(
                    f"unresolved_coverage_issues[{position}].reasons contains "
                    "an invalid value"
                )
            if reason.strip() not in reasons:
                reasons.append(reason.strip())
        padded.append(
            {
                "start_ms": max(0, interval.start_ms - padding_ms),
                "end_ms": interval.end_ms + padding_ms,
                "issue_ids": [issue_id.strip()],
                "speech_region_indices": [speech_region_index],
                "reasons": reasons,
            }
        )

    merged: list[dict[str, Any]] = []
    for span in padded:
        if merged and span["start_ms"] <= merged[-1]["end_ms"] + merge_gap_ms:
            previous = merged[-1]
            previous["end_ms"] = max(previous["end_ms"], span["end_ms"])
            previous["issue_ids"] = sorted(
                set(previous["issue_ids"] + span["issue_ids"])
            )
            previous["speech_region_indices"] = sorted(
                set(
                    previous["speech_region_indices"]
                    + span["speech_region_indices"]
                )
            )
            previous["reasons"] = sorted(set(previous["reasons"] + span["reasons"]))
        else:
            merged.append(dict(span))
    for index, span in enumerate(merged, start=1):
        span["rescue_span_index"] = index
        span["duration_ms"] = span["end_ms"] - span["start_ms"]
    return merged


def analyze_speech_coverage(
    vad_regions: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
    *,
    config: SpeechCoverageConfig | None = None,
    reviewed_non_dialogue: Sequence[Mapping[str, Any]] = (),
    reviewed_dialogue: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compare independent VAD speech against timed-word coverage.

    ``reviewed_non_dialogue`` entries are intentionally strict audit records.
    Each needs ``start_ms``, ``end_ms``, ``classification='non_dialogue'``,
    ``review_status='reviewed'``, and a non-empty ``reason``.  A review resolves
    an issue only when its start and end exactly match that issue and it stays
    inside the same merged VAD speech region.  A broad review can therefore
    never blanket-clear a smaller missing-dialogue interval.

    ``reviewed_dialogue`` is a separate, exact human-reviewed speech inventory.
    It is unioned with independent VAD for containment and coverage, allowing a
    hash-bound upstream audio review to restore real dialogue missed by VAD.
    It never clears an independent-VAD hole by itself: the reviewed interval
    still needs aligned word coverage.
    """

    settings = config or SpeechCoverageConfig()
    if not isinstance(settings, SpeechCoverageConfig):
        raise SpeechCoverageError("config must be a SpeechCoverageConfig")

    raw_vad = _validated_intervals(vad_regions, "vad_regions")
    raw_words = _validated_intervals(words, "words")
    independent_speech_regions = _merge_intervals(
        raw_vad, settings.vad_merge_gap_ms
    )
    dialogue_reviews, dialogue_intervals = _normalize_dialogue_reviews(
        reviewed_dialogue
    )
    speech_regions = _merge_intervals(
        sorted(
            independent_speech_regions + dialogue_intervals,
            key=lambda item: (item.start_ms, item.end_ms),
        ),
        settings.vad_merge_gap_ms,
    )
    effective_words = _effective_word_intervals(raw_words, settings)
    reviews = _normalize_reviews(reviewed_non_dialogue, speech_regions)
    for non_dialogue_review in reviews:
        for dialogue_review in dialogue_reviews:
            if (
                non_dialogue_review["start_ms"] < dialogue_review["end_ms"]
                and dialogue_review["start_ms"] < non_dialogue_review["end_ms"]
            ):
                raise SpeechCoverageError(
                    "reviewed_non_dialogue and reviewed_dialogue records overlap"
                )
    wholly_outside_words = [
        word
        for word in raw_words
        if not any(
            word.start_ms < region.end_ms and region.start_ms < word.end_ms
            for region in speech_regions
        )
    ]
    word_outside_audits: list[dict[str, Any]] = []
    for word in raw_words:
        outside = _outside_runs(word, speech_regions)
        outside_ms = sum(item.duration_ms for item in outside)
        maximum_outside_ms = max(
            (item.duration_ms for item in outside), default=0
        )
        if maximum_outside_ms > settings.max_word_outside_speech_ms:
            word_outside_audits.append(
                {
                    "start_ms": word.start_ms,
                    "end_ms": word.end_ms,
                    "duration_ms": word.duration_ms,
                    "inside_speech_ms": word.duration_ms - outside_ms,
                    "outside_speech_ms": outside_ms,
                    "maximum_contiguous_outside_speech_ms": maximum_outside_ms,
                    "outside_intervals": [
                        _interval_dict(item) for item in outside
                    ],
                    "source_word_record_indices": list(
                        word.source_record_indices
                    ),
                }
            )

    region_reports: list[dict[str, Any]] = []
    issue_drafts: list[dict[str, Any]] = []
    total_speech_ms = 0
    total_covered_ms = 0
    total_raw_covered_ms = 0

    for region_index, region in enumerate(speech_regions, start=1):
        # Padding absorbs timestamp jitter only for words which already have
        # genuine temporal overlap with this independent speech region.  A word
        # wholly in preceding/following silence must not become speech evidence
        # merely because its padding reaches across the VAD boundary.
        region_words = [
            word
            for word in raw_words
            if word.start_ms < region.end_ms and region.start_ms < word.end_ms
        ]
        region_effective_words = _effective_word_intervals(
            region_words,
            settings,
        )
        covered = _intersection(region, region_effective_words)
        raw_covered = _intersection(region, raw_words)
        uncovered = _subtract(region, covered)
        covered_ms = sum(interval.duration_ms for interval in covered)
        raw_covered_ms = sum(interval.duration_ms for interval in raw_covered)
        uncovered_ms = region.duration_ms - covered_ms
        coverage_ratio = _ratio(covered_ms, region.duration_ms)
        significant_holes = [
            interval
            for interval in uncovered
            if interval.duration_ms >= settings.min_hole_ms
        ]
        low_coverage = coverage_ratio < float(
            settings.min_region_coverage_ratio
        )

        if low_coverage:
            # Never turn the whole low-coverage VAD region into a blank
            # correction record: doing so would duplicate the dialogue already
            # represented by ``covered``.  Region-level failure remains in the
            # metrics, while each issue is restricted to genuinely uncovered
            # time.  Short gaps are included here because their aggregate is
            # what caused the region-level coverage threshold to fail.
            for hole in uncovered:
                reasons = ["region_coverage_below_threshold"]
                if hole.duration_ms >= settings.min_hole_ms:
                    reasons.append("speech_hole_above_min_duration")
                issue_drafts.append(
                    {
                        "speech_region_index": region_index,
                        "start_ms": hole.start_ms,
                        "end_ms": hole.end_ms,
                        "duration_ms": hole.duration_ms,
                        "issue_kind": "low_coverage_speech_hole",
                        "reasons": reasons,
                        "region_coverage_ratio": coverage_ratio,
                        "uncovered_duration_ms": hole.duration_ms,
                        "largest_uncovered_span_ms": hole.duration_ms,
                    }
                )
        else:
            for hole in significant_holes:
                issue_drafts.append(
                    {
                        "speech_region_index": region_index,
                        "start_ms": hole.start_ms,
                        "end_ms": hole.end_ms,
                        "duration_ms": hole.duration_ms,
                        "issue_kind": "speech_hole",
                        "reasons": ["speech_hole_above_min_duration"],
                        "region_coverage_ratio": coverage_ratio,
                        "uncovered_duration_ms": hole.duration_ms,
                        "largest_uncovered_span_ms": hole.duration_ms,
                    }
                )

        total_speech_ms += region.duration_ms
        total_covered_ms += covered_ms
        total_raw_covered_ms += raw_covered_ms
        region_reports.append(
            {
                "speech_region_index": region_index,
                "start_ms": region.start_ms,
                "end_ms": region.end_ms,
                "duration_ms": region.duration_ms,
                "source_vad_record_indices": [
                    index
                    for index, source_region in enumerate(raw_vad, start=1)
                    if source_region.start_ms < region.end_ms
                    and region.start_ms < source_region.end_ms
                ],
                "source_reviewed_dialogue_ids": [
                    review["review_id"]
                    for review in dialogue_reviews
                    if review["start_ms"] < region.end_ms
                    and region.start_ms < review["end_ms"]
                ],
                "covered_ms": covered_ms,
                "uncovered_ms": uncovered_ms,
                "coverage_ratio": coverage_ratio,
                "raw_word_covered_ms": raw_covered_ms,
                "raw_word_coverage_ratio": _ratio(
                    raw_covered_ms, region.duration_ms
                ),
                "covered_intervals": [_interval_dict(item) for item in covered],
                "uncovered_intervals": [
                    _interval_dict(item) for item in uncovered
                ],
                "significant_hole_count": len(significant_holes),
                "coverage_below_threshold": low_coverage,
                "classification": "covered_speech",
                "issue_ids": [],
            }
        )

    issues: list[dict[str, Any]] = []
    for index, draft in enumerate(issue_drafts, start=1):
        issue = dict(draft)
        issue_id = f"speech-coverage-{index:04d}"
        issue["issue_id"] = issue_id
        covering_review: dict[str, Any] | None = None
        for review in reviews:
            same_region = (
                review["speech_region_index"] == issue["speech_region_index"]
            )
            exact_interval = (
                review["start_ms"] == issue["start_ms"]
                and issue["end_ms"] == review["end_ms"]
            )
            overlap = (
                review["start_ms"] < issue["end_ms"]
                and issue["start_ms"] < review["end_ms"]
            )
            if same_region and exact_interval:
                covering_review = review
                break
            if same_region and overlap:
                review["partially_overlapping_issue_ids"].append(issue_id)
        if covering_review is not None:
            issue["classification"] = "reviewed_non_dialogue"
            issue["review_id"] = covering_review["review_id"]
            issue["review_reason"] = covering_review["reason"]
            covering_review["matched_issue_ids"].append(issue_id)
        else:
            issue["classification"] = "unresolved_speech"
            issue["review_id"] = None
            issue["review_reason"] = None
        issues.append(issue)

    issue_ids_by_region: dict[int, list[str]] = {}
    classifications_by_region: dict[int, list[str]] = {}
    for issue in issues:
        region_index = issue["speech_region_index"]
        issue_ids_by_region.setdefault(region_index, []).append(issue["issue_id"])
        classifications_by_region.setdefault(region_index, []).append(
            issue["classification"]
        )
    for region in region_reports:
        region_index = region["speech_region_index"]
        classifications = classifications_by_region.get(region_index, [])
        region["issue_ids"] = issue_ids_by_region.get(region_index, [])
        if "unresolved_speech" in classifications:
            region["classification"] = "unresolved_speech"
        elif "reviewed_non_dialogue" in classifications:
            region["classification"] = "reviewed_non_dialogue"

    unresolved_region_indices = sorted(
        {
            issue["speech_region_index"]
            for issue in issues
            if issue["classification"] == "unresolved_speech"
        }
    )
    reviewed_region_indices = sorted(
        {
            issue["speech_region_index"]
            for issue in issues
            if issue["classification"] == "reviewed_non_dialogue"
        }
    )
    rescue_spans = build_rescue_spans(
        issues,
        padding_ms=settings.rescue_padding_ms,
        merge_gap_ms=settings.rescue_merge_gap_ms,
    )

    effective_word_total_ms = sum(item.duration_ms for item in effective_words)
    effective_word_inside_speech_ms = total_covered_ms
    unresolved_issue_count = sum(
        issue["classification"] == "unresolved_speech" for issue in issues
    )
    reviewed_issue_count = sum(
        issue["classification"] == "reviewed_non_dialogue" for issue in issues
    )
    unresolved_region_count = len(unresolved_region_indices)
    metrics = {
        "input_vad_region_count": len(raw_vad),
        "merged_speech_region_count": len(speech_regions),
        "input_word_interval_count": len(raw_words),
        "effective_word_interval_count": len(effective_words),
        "word_interval_wholly_outside_speech_count": len(wholly_outside_words),
        "word_interval_wholly_outside_speech_ms": sum(
            word.duration_ms for word in wholly_outside_words
        ),
        "word_interval_outside_speech_violation_count": len(
            word_outside_audits
        ),
        "word_interval_outside_speech_violation_ms": sum(
            item["outside_speech_ms"] for item in word_outside_audits
        ),
        "reviewed_dialogue_record_count": len(dialogue_reviews),
        "reviewed_dialogue_duration_ms": sum(
            item["duration_ms"] for item in dialogue_reviews
        ),
        "speech_duration_ms": total_speech_ms,
        "covered_speech_ms": total_covered_ms,
        "uncovered_speech_ms": total_speech_ms - total_covered_ms,
        "speech_coverage_ratio": _ratio(total_covered_ms, total_speech_ms),
        "raw_word_covered_speech_ms": total_raw_covered_ms,
        "raw_word_speech_coverage_ratio": _ratio(
            total_raw_covered_ms, total_speech_ms
        ),
        "effective_word_coverage_outside_speech_ms": max(
            0, effective_word_total_ms - effective_word_inside_speech_ms
        ),
        "coverage_issue_count": len(issues),
        "unresolved_coverage_issue_count": unresolved_issue_count,
        "reviewed_non_dialogue_issue_count": reviewed_issue_count,
        "unresolved_speech_region_count": unresolved_region_count,
        "reviewed_non_dialogue_region_count": len(reviewed_region_indices),
        "rescue_span_count": len(rescue_spans),
        "unused_non_dialogue_review_count": sum(
            not review["matched_issue_ids"] for review in reviews
        ),
    }
    return {
        "report_version": "1.0",
        "status": (
            "PASS"
            if unresolved_region_count == 0 and not word_outside_audits
            else "FAIL"
        ),
        "config": asdict(settings),
        "unresolved_speech_region_count": unresolved_region_count,
        "speech_coverage_ratio": metrics["speech_coverage_ratio"],
        "metrics": metrics,
        "speech_regions": region_reports,
        "word_intervals_wholly_outside_speech": [
            _interval_dict(word) for word in wholly_outside_words
        ],
        "word_intervals_exceeding_outside_speech_tolerance": (
            word_outside_audits
        ),
        "coverage_issues": issues,
        "rescue_spans": rescue_spans,
        "reviewed_non_dialogue_records": reviews,
        "reviewed_dialogue_records": dialogue_reviews,
    }


__all__ = [
    "SpeechCoverageConfig",
    "SpeechCoverageError",
    "V2_BETA_COVERAGE_POLICY_LABEL",
    "V2_BETA_SPEECH_COVERAGE_CONFIG",
    "analyze_speech_coverage",
    "build_rescue_spans",
    "require_v2_beta_speech_coverage_config",
]
