"""Deterministic pre-translation subtitle segmentation from timed words.

This module uses no visual scene information and does not call any language or
translation service.  Hard speech/speaker boundaries are established first;
dynamic programming then chooses readable phrase boundaries without dropping or
reordering words.  ``schema.py`` remains the sole authority that fills stable
block UIDs and computes the episode-level schema hash.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .download import (
    atomic_write_json,
    load_valid_stage_marker,
    sha256_file,
    sha256_json,
    utc_now_iso,
    write_stage_marker,
)

LOGGER = logging.getLogger(__name__)
SEGMENTATION_FORMAT_VERSION = "1.2"


class SegmentationError(RuntimeError):
    """Raised when word timing cannot become a safe immutable block structure."""


@dataclass(frozen=True)
class SegmentationConfig:
    """Readable two-line subtitle and timing defaults."""

    schema_version: str = "1.0"
    target_chars_per_line: int = 42
    maximum_lines: int = 2
    preferred_cps: float = 17.0
    preferred_min_duration_ms: int = 1200
    minimum_duration_ms: int = 700
    maximum_duration_ms: int = 7000
    start_lead_ms: int = 0
    end_padding_ms: int = 220
    next_speech_guard_ms: int = 80
    internal_gap_review_ms: int = 1000
    likely_speaker_gap_ms: int = 650
    hard_silence_ms: int = 1000
    sentence_gap_ms: int = 180
    weak_phrase_gap_ms: int = 350
    max_words_per_block: int = 24
    low_confidence_probability: float = 0.55
    verification_similarity_warning: float = 0.50
    youtube_similarity_warning: float = 0.28
    context_blocks: int = 2

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version cannot be empty")
        if self.maximum_lines != 2:
            raise ValueError("maximum_lines must remain 2")
        if self.target_chars_per_line < 20:
            raise ValueError("target_chars_per_line is implausibly small")
        if not 0 <= self.start_lead_ms <= 80:
            raise ValueError("start_lead_ms must be between 0 and 80 ms")
        if not 180 <= self.end_padding_ms <= 240:
            raise ValueError("end_padding_ms must remain between 180 and 240 ms")
        if self.internal_gap_review_ms != 1000:
            raise ValueError("internal_gap_review_ms must remain 1000 ms")
        if self.minimum_duration_ms <= 0 or self.maximum_duration_ms <= self.minimum_duration_ms:
            raise ValueError("subtitle duration bounds are invalid")
        if self.hard_silence_ms > self.internal_gap_review_ms:
            raise ValueError("hard_silence_ms cannot exceed the internal-gap review threshold")
        if self.context_blocks < 0:
            raise ValueError("context_blocks cannot be negative")

    @property
    def preferred_total_chars(self) -> int:
        return self.target_chars_per_line * self.maximum_lines


@dataclass(frozen=True)
class SegmentationReport:
    """Structural report generated before schema locking."""

    valid: bool
    block_count: int
    overlap_count: int
    invalid_timing_count: int
    unresolved_internal_gap_count: int
    mixed_speaker_block_count: int
    empty_text_count: int
    nonsequential_index_count: int
    source_text_mismatch_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise SegmentationError("; ".join(self.errors[:20]))


@dataclass(frozen=True)
class SegmentationResult:
    """Persistent segmentation artifact ready for ``build_episode_schema``."""

    blocks: list[dict[str, Any]]
    output_path: Path
    marker_path: Path
    report: SegmentationReport
    resumed: bool = False


@dataclass(frozen=True)
class _Boundary:
    before_word_offset: int
    reasons: tuple[str, ...]
    speaker_risk: bool


def _normalise_compare(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\wçğıöşü]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _word_text(word: Mapping[str, Any]) -> str:
    return str(word.get("text", word.get("word", "")))


def _join_words(words: Sequence[Mapping[str, Any]]) -> str:
    """Reconstruct Whisper tokens while also supporting synthetic test tokens."""

    output = ""
    for word in words:
        raw = _word_text(word)
        if not raw:
            continue
        if not output:
            output = raw.strip()
        elif raw[0].isspace():
            output += raw
        elif re.match(r"^[,.;:!?%\)\]\}…]+", raw):
            output += raw
        elif output.endswith(("(", "[", "{", "'", '"', "-", "–", "—", "/")):
            output += raw
        else:
            output += " " + raw
    output = re.sub(r"[ \t\r\f\v]+", " ", output)
    output = re.sub(r"\s+([,.;:!?%\)\]\}…])", r"\1", output)
    output = re.sub(r"([\(\[\{])\s+", r"\1", output)
    return output.strip()


def _prepare_words(source_words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not source_words:
        raise SegmentationError("Transcription contains no word timestamps")
    words: list[dict[str, Any]] = []
    previous_start = -1
    for offset, source in enumerate(source_words):
        try:
            start = int(source["start_ms"])
            end = int(source["end_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmentationError(f"Malformed word timing at offset {offset}: {source}") from exc
        text = _word_text(source)
        if start < 0 or end <= start:
            raise SegmentationError(f"Invalid word interval at offset {offset}: {start}-{end}")
        if start < previous_start:
            raise SegmentationError(f"Word timings are not ordered at offset {offset}")
        if not text.strip():
            raise SegmentationError(f"Empty word text at offset {offset}")
        probability = source.get("probability")
        prepared = {
            "source_offset": offset,
            "word_index": int(source.get("word_index", offset + 1)),
            "segment_id": source.get("segment_id"),
            "start_ms": start,
            "end_ms": end,
            "text": text,
            "probability": float(probability) if probability is not None else None,
        }
        timing_source = str(source.get("timing_source", "")).strip()
        if timing_source:
            prepared["timing_source"] = timing_source
        words.append(prepared)
        previous_start = start
    return words


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?…][\"')\]]?$", text.strip()))


def _ends_clause(text: str) -> bool:
    return bool(re.search(r"[,;:][\"')\]]?$", text.strip()))


def _starts_dialogue_turn(text: str) -> bool:
    return bool(re.match(r"^\s*[-–—]\s*\S", text))


def _contains_new_turn(text: str) -> bool:
    return bool(re.search(r"\n\s*[-–—]\s*\S", text))


def _hard_boundaries(words: Sequence[Mapping[str, Any]], config: SegmentationConfig) -> list[_Boundary]:
    boundaries: list[_Boundary] = []
    for offset in range(1, len(words)):
        previous = words[offset - 1]
        current = words[offset]
        gap = int(current["start_ms"]) - int(previous["end_ms"])
        reasons: list[str] = []
        speaker_risk = False
        previous_segment_id = previous.get("segment_id")
        current_segment_id = current.get("segment_id")
        whisper_segment_changed = (
            previous_segment_id is not None
            and current_segment_id is not None
            and previous_segment_id != current_segment_id
        )
        # A faster-whisper segment transition is an authoritative grouping
        # boundary even when the acoustic gap is short.  It is not, by itself,
        # evidence that the speaker changed, so keep it out of the speaker-risk
        # counters unless an explicit dialogue cue or a long turn-like gap also
        # supports that conclusion.
        if whisper_segment_changed:
            reasons.append("whisper_segment_boundary")
        if gap >= config.hard_silence_ms:
            reasons.append("silence_gt_1000ms")
        if _starts_dialogue_turn(_word_text(current)) or _contains_new_turn(_word_text(current)):
            reasons.append("explicit_dialogue_turn")
            speaker_risk = True
        if "\n" in _word_text(previous) and _starts_dialogue_turn(_word_text(current)):
            reasons.append("newline_dialogue_turn")
            speaker_risk = True
        if gap >= config.likely_speaker_gap_ms and (
            _ends_sentence(_word_text(previous))
            or _starts_dialogue_turn(_word_text(current))
            or whisper_segment_changed
        ):
            reasons.append("likely_speaker_turn_gap")
            speaker_risk = True
        if reasons:
            boundaries.append(
                _Boundary(offset, tuple(sorted(set(reasons))), speaker_risk=speaker_risk)
            )
    return boundaries


def _line_balance_cost(text: str, target: int) -> float:
    length = len(text)
    if length <= target:
        return 0.0
    spaces = [index for index, character in enumerate(text) if character == " "]
    if not spaces:
        return float((length - target) ** 2)
    split = min(spaces, key=lambda position: abs(position - length / 2))
    first_length = split
    second_length = length - split - 1
    overflow = max(0, first_length - target) + max(0, second_length - target)
    imbalance = abs(first_length - second_length) / max(1, length)
    return overflow * 8.0 + imbalance * 2.0


def _candidate_cost(
    words: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    config: SegmentationConfig,
) -> float | None:
    candidate = words[start:end]
    if not candidate:
        return None
    word_count = len(candidate)
    if word_count > config.max_words_per_block:
        return None
    text = _join_words(candidate)
    character_count = len(text)
    speech_duration_ms = int(candidate[-1]["end_ms"]) - int(candidate[0]["start_ms"])
    if word_count > 1 and speech_duration_ms > config.maximum_duration_ms:
        return None
    # 42 characters per line is a target, not a destructive hard truncation.
    # Permit a modest overflow so one long Turkish word is never altered.
    if word_count > 1 and character_count > config.preferred_total_chars + 18:
        return None

    expected_ms = max(
        config.preferred_min_duration_ms,
        min(config.maximum_duration_ms - 500, round(character_count / config.preferred_cps * 1000)),
    )
    duration_cost = ((speech_duration_ms - expected_ms) / 1000.0) ** 2
    density_cost = ((character_count - config.preferred_total_chars * 0.70) / 35.0) ** 2
    cost = duration_cost + density_cost + _line_balance_cost(text, config.target_chars_per_line)

    if word_count == 1:
        cost += 8.0
        if _ends_sentence(text) or character_count >= 8:
            cost -= 3.0
    elif word_count == 2 and character_count < 12:
        cost += 3.0

    cps = character_count / max(speech_duration_ms / 1000.0, 0.25)
    if cps > 20:
        cost += ((cps - 20) / 3.0) ** 2

    last_text = _word_text(candidate[-1])
    if _ends_sentence(last_text):
        cost -= 5.0
    elif _ends_clause(last_text):
        cost -= 2.0
    if end < len(words):
        next_gap = int(words[end]["start_ms"]) - int(candidate[-1]["end_ms"])
        cost -= min(max(next_gap, 0), 800) / 180.0
        next_text = _word_text(words[end]).lstrip()
        if next_gap < 120 and not (_ends_sentence(last_text) or _ends_clause(last_text)):
            # A lowercase continuation is an especially poor phrase boundary.
            if next_text[:1].islower():
                cost += 5.0
            else:
                cost += 2.0
    return cost


def _dynamic_groups(words: Sequence[Mapping[str, Any]], config: SegmentationConfig) -> list[list[dict[str, Any]]]:
    count = len(words)
    costs = [math.inf] * (count + 1)
    previous = [-1] * (count + 1)
    costs[0] = 0.0
    for end in range(1, count + 1):
        lower = max(0, end - config.max_words_per_block)
        for start in range(end - 1, lower - 1, -1):
            if not math.isfinite(costs[start]):
                continue
            candidate_cost = _candidate_cost(words, start, end, config)
            if candidate_cost is None:
                continue
            total = costs[start] + candidate_cost
            if total < costs[end]:
                costs[end] = total
                previous[end] = start
    if previous[count] < 0:
        # A pathological single word can exceed every readability target; keep
        # it intact and flag it rather than deleting or changing dialogue.
        return [[dict(word)] for word in words]
    groups: list[list[dict[str, Any]]] = []
    cursor = count
    while cursor > 0:
        start = previous[cursor]
        if start < 0:
            raise SegmentationError("Dynamic segmentation reconstruction failed")
        groups.append([dict(word) for word in words[start:cursor]])
        cursor = start
    groups.reverse()

    # A decoded word with estimated timing can share or overlap the boundary
    # of its real neighbour.  Readability scoring may otherwise split those
    # words into separate cues even though no positive, non-overlapping subtitle
    # intervals can represent that split.  Coalesce only such temporally
    # inseparable groups inside the current hard-boundary chunk; ordinary
    # touching real-word boundaries remain eligible segmentation points.
    coalesced: list[list[dict[str, Any]]] = []
    for group in groups:
        if not coalesced:
            coalesced.append(group)
            continue
        previous_word = coalesced[-1][-1]
        next_word = group[0]
        previous_end = int(previous_word["end_ms"])
        next_start = int(next_word["start_ms"])
        estimated_boundary = bool(previous_word.get("timing_source")) or bool(
            next_word.get("timing_source")
        )
        if next_start < previous_end or (
            next_start == previous_end and estimated_boundary
        ):
            coalesced[-1].extend(group)
        else:
            coalesced.append(group)
    return coalesced


def _group_words(
    words: Sequence[Mapping[str, Any]], config: SegmentationConfig
) -> tuple[list[list[dict[str, Any]]], dict[int, _Boundary]]:
    boundaries = _hard_boundaries(words, config)
    boundary_map = {boundary.before_word_offset: boundary for boundary in boundaries}
    chunks: list[Sequence[Mapping[str, Any]]] = []
    start = 0
    for boundary in boundaries:
        chunks.append(words[start : boundary.before_word_offset])
        start = boundary.before_word_offset
    chunks.append(words[start:])
    groups: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if chunk:
            groups.extend(_dynamic_groups(chunk, config))
    return groups, boundary_map


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _evidence_text(
    start_ms: int,
    end_ms: int,
    records: Sequence[Mapping[str, Any]],
    *,
    text_field: str = "text",
) -> str:
    pieces: list[str] = []
    for record in records:
        try:
            overlap = _overlap_ms(
                start_ms,
                end_ms,
                int(record["start_ms"]),
                int(record["end_ms"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        text = str(record.get(text_field, "")).strip()
        if overlap > 0 and text and (not pieces or text != pieces[-1]):
            pieces.append(text)
    compact: list[str] = []
    for piece in pieces:
        if compact and piece.startswith(compact[-1]):
            compact[-1] = piece
        elif not compact or piece not in compact[-1]:
            compact.append(piece)
    return " ".join(compact).strip()


def _verification_evidence(
    start_ms: int,
    end_ms: int,
    verification_results: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    flags: list[str] = []
    for result in verification_results:
        requested_start = int(result.get("requested_span_start_ms", result.get("start_ms", -1)))
        requested_end = int(result.get("requested_span_end_ms", result.get("end_ms", -1)))
        overlap = _overlap_ms(start_ms, end_ms, requested_start, requested_end)
        if overlap <= 0:
            continue
        if result.get("status") == "failed":
            flags.append("targeted_verification_failed")
            continue
        words = [word for word in result.get("words") or [] if isinstance(word, Mapping)]
        text = _evidence_text(start_ms, end_ms, words) if words else str(result.get("text", "")).strip()
        if text:
            candidates.append((overlap, text, result))
    if not candidates:
        return "", sorted(set(flags))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    flags.append("targeted_verification_available")
    return candidates[0][1], sorted(set(flags))


def _vad_info(
    group: Sequence[Mapping[str, Any]],
    vad_regions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    start_ms = int(group[0]["start_ms"])
    speech_end_ms = int(group[-1]["end_ms"])
    speech_duration = max(1, speech_end_ms - start_ms)
    region_indices: list[int] = []
    overlap_total = 0
    sources: set[str] = set()
    intervals: list[tuple[int, int]] = []
    for region in vad_regions:
        try:
            region_start = int(region["start_ms"])
            region_end = int(region["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        overlap_start = max(start_ms, region_start)
        overlap_end = min(speech_end_ms, region_end)
        if overlap_end > overlap_start:
            intervals.append((overlap_start, overlap_end))
            region_indices.append(int(region.get("vad_region_index", len(region_indices) + 1)))
            sources.add(str(region.get("source", "unknown")))
    merged_intervals: list[tuple[int, int]] = []
    for interval_start, interval_end in sorted(intervals):
        if merged_intervals and interval_start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (
                merged_intervals[-1][0],
                max(merged_intervals[-1][1], interval_end),
            )
        else:
            merged_intervals.append((interval_start, interval_end))
    overlap_total = sum(end - start for start, end in merged_intervals)
    gaps = [
        int(next_word["start_ms"]) - int(word["end_ms"])
        for word, next_word in zip(group, group[1:])
    ]
    probabilities = [
        float(word["probability"])
        for word in group
        if word.get("probability") is not None
    ]
    return {
        "speech_region_indices": sorted(set(region_indices)),
        "sources": sorted(sources),
        "speech_coverage_ratio": min(1.0, overlap_total / speech_duration),
        "internal_gaps_ms": gaps,
        "max_internal_gap_ms": max(gaps, default=0),
        "word_probability_min": min(probabilities) if probabilities else None,
        "word_probability_mean": fmean(probabilities) if probabilities else None,
        "first_word_start_ms": start_ms,
        "last_word_end_ms": speech_end_ms,
    }


def _risk_flags(
    group: Sequence[Mapping[str, Any]],
    primary_text: str,
    verification_text: str,
    youtube_text: str,
    vad_info: Mapping[str, Any],
    config: SegmentationConfig,
) -> set[str]:
    flags: set[str] = set()
    probabilities = [
        float(word["probability"])
        for word in group
        if word.get("probability") is not None
    ]
    if probabilities and min(probabilities) < config.low_confidence_probability:
        flags.add("low_word_confidence")
    if int(vad_info.get("max_internal_gap_ms", 0)) > config.internal_gap_review_ms:
        flags.add("internal_gap_gt_1000ms")
    if not vad_info.get("speech_region_indices"):
        flags.add("vad_boundary_uncertain")
    if len(group) == 1:
        flags.add("one_word_block")
    speech_duration = max(1, int(group[-1]["end_ms"]) - int(group[0]["start_ms"]))
    if len(primary_text) / (speech_duration / 1000.0) > 20:
        flags.add("high_source_cps")
    if len(primary_text) > config.preferred_total_chars:
        flags.add("source_text_over_two_line_target")
    if _starts_dialogue_turn(primary_text) or _contains_new_turn(primary_text):
        flags.add("possible_speaker_change")
    if re.search(r"\[(?:müzik|music|şarkı|alkış)", primary_text, flags=re.IGNORECASE):
        flags.add("music_speech_uncertainty")
    if verification_text:
        similarity = SequenceMatcher(
            None, _normalise_compare(primary_text), _normalise_compare(verification_text)
        ).ratio()
        if similarity < config.verification_similarity_warning:
            flags.add("verification_disagreement")
    if youtube_text:
        similarity = SequenceMatcher(
            None, _normalise_compare(primary_text), _normalise_compare(youtube_text)
        ).ratio()
        if similarity < config.youtube_similarity_warning:
            flags.add("youtube_caption_conflict")
    return flags


def _suspicious_flags(
    start_ms: int,
    end_ms: int,
    suspicious_spans: Sequence[Mapping[str, Any]],
) -> set[str]:
    flags: set[str] = set()
    reason_mapping = {
        "low_word_confidence": "low_word_confidence",
        "very_low_word_confidence": "very_low_word_confidence",
        "low_segment_logprob": "low_segment_logprob",
        "high_no_speech_probability": "high_no_speech_probability",
        "possible_speaker_change": "possible_speaker_change",
        "music_speech_uncertainty": "music_speech_uncertainty",
        "large_internal_gap": "internal_gap_review",
        "name_or_religious_expression": "sensitive_term_verification",
        "youtube_caption_conflict": "youtube_caption_conflict",
        "strange_turkish": "strange_turkish_asr",
    }
    for span in suspicious_spans:
        if _overlap_ms(start_ms, end_ms, int(span["start_ms"]), int(span["end_ms"])) <= 0:
            continue
        for reason in span.get("reasons") or []:
            if reason in reason_mapping:
                flags.add(reason_mapping[reason])
    return flags


def _boundary_risk_for_group(
    group: Sequence[Mapping[str, Any]], boundary_map: Mapping[int, _Boundary]
) -> tuple[set[str], dict[str, Any]]:
    flags: set[str] = set()
    details: dict[str, Any] = {}
    first_offset = int(group[0]["source_offset"])
    after_offset = int(group[-1]["source_offset"]) + 1
    before = boundary_map.get(first_offset)
    after = boundary_map.get(after_offset)
    if before is not None:
        details["boundary_before"] = list(before.reasons)
        if before.speaker_risk:
            flags.update(("possible_speaker_change", "speaker_change_boundary_before"))
    if after is not None:
        details["boundary_after"] = list(after.reasons)
        if after.speaker_risk:
            flags.update(("possible_speaker_change", "speaker_change_boundary_after"))
    return flags, details


def _assign_timings(
    raw_blocks: list[dict[str, Any]], config: SegmentationConfig
) -> None:
    for index, block in enumerate(raw_blocks):
        first_word_start = int(block["_first_word_start_ms"])
        last_word_end = int(block["_last_word_end_ms"])
        # ``start_lead_ms`` is retained in configuration for compatibility with
        # the series policy (an 80 ms upper bound). The stricter absolute rule
        # wins here: actual subtitles start on the first reliable word.
        start = first_word_start
        desired_end = last_word_end + config.end_padding_ms
        next_start = (
            int(raw_blocks[index + 1]["_first_word_start_ms"])
            if index + 1 < len(raw_blocks)
            else None
        )
        if next_start is not None:
            guarded_end = next_start - config.next_speech_guard_ms
            if guarded_end >= last_word_end:
                desired_end = min(desired_end, guarded_end)
            elif next_start > last_word_end:
                desired_end = last_word_end
                block["risk_flags"].append("tight_next_speech_boundary")
            else:
                # Acoustic word intervals themselves overlap. Preserve the last
                # syllable, then start the following subtitle after it. This is
                # the only way to maintain non-overlap without ending early.
                desired_end = last_word_end
                block["risk_flags"].append("overlapping_word_timing_boundary")
        minimum_end = start + config.minimum_duration_ms
        if desired_end < minimum_end:
            extend_to = minimum_end
            if next_start is not None:
                extend_to = min(extend_to, next_start - config.next_speech_guard_ms)
            if extend_to >= last_word_end:
                desired_end = max(desired_end, extend_to)
            if desired_end - start < config.minimum_duration_ms:
                block["risk_flags"].append("short_duration_due_tight_boundary")
        block["start_ms"] = start
        block["end_ms"] = max(last_word_end, desired_end)

    for index in range(len(raw_blocks) - 1):
        current = raw_blocks[index]
        following = raw_blocks[index + 1]
        if int(current["end_ms"]) >= int(following["start_ms"]):
            current_last_word_end = int(current["_last_word_end_ms"])
            if current_last_word_end < int(following["start_ms"]):
                current["end_ms"] = int(following["start_ms"]) - 1
            else:
                following["start_ms"] = int(current["end_ms"]) + 1
                current["risk_flags"].append("overlapping_word_timing_boundary")
                following["risk_flags"].append("late_start_due_overlapping_word_timing")


def _context_text(blocks: Sequence[Mapping[str, Any]], index: int, count: int, *, before: bool) -> str:
    if count <= 0:
        return ""
    if before:
        selected = blocks[max(0, index - count) : index]
    else:
        selected = blocks[index + 1 : index + 1 + count]
    return " / ".join(str(block.get("primary_text", "")) for block in selected).strip()


def build_blocks(
    transcription: Mapping[str, Any],
    episode: int,
    *,
    config: SegmentationConfig | None = None,
    youtube_captions: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the complete immutable-field block records, with UID left blank.

    ``schema.build_episode_schema`` deterministically fills ``block_uid`` and
    computes ``schema_sha256`` after this function returns.
    """

    settings = config or SegmentationConfig()
    if not isinstance(episode, int) or isinstance(episode, bool) or episode <= 0:
        raise ValueError("episode must be a positive integer")
    words = _prepare_words(transcription.get("words") or [])
    groups, boundary_map = _group_words(words, settings)
    captions = list(
        youtube_captions
        if youtube_captions is not None
        else transcription.get("youtube_captions") or []
    )
    vad_regions = [
        region for region in transcription.get("vad_regions") or [] if isinstance(region, Mapping)
    ]
    verification_results = [
        result
        for result in transcription.get("verification_results") or []
        if isinstance(result, Mapping)
    ]
    suspicious_spans = [
        span
        for span in transcription.get("suspicious_spans") or []
        if isinstance(span, Mapping)
    ]

    raw_blocks: list[dict[str, Any]] = []
    for group in groups:
        first_start = int(group[0]["start_ms"])
        last_end = int(group[-1]["end_ms"])
        primary_text = _join_words(group)
        verification_text, verification_flags = _verification_evidence(
            first_start, last_end, verification_results
        )
        youtube_text = _evidence_text(first_start, last_end, captions)
        vad_info = _vad_info(group, vad_regions)
        risk_flags = _risk_flags(
            group,
            primary_text,
            verification_text,
            youtube_text,
            vad_info,
            settings,
        )
        risk_flags.update(verification_flags)
        risk_flags.update(_suspicious_flags(first_start, last_end, suspicious_spans))
        boundary_flags, boundary_details = _boundary_risk_for_group(group, boundary_map)
        risk_flags.update(boundary_flags)
        vad_info.update(boundary_details)
        raw_blocks.append(
            {
                "block_uid": "",
                "episode": episode,
                "block_index": len(raw_blocks) + 1,
                "start_ms": first_start,
                "end_ms": last_end,
                "timing_text": primary_text,
                "primary_text": primary_text,
                "verification_text": verification_text,
                "youtube_text": youtube_text,
                "context_before": "",
                "context_after": "",
                "vad_info": vad_info,
                "risk_flags": sorted(risk_flags),
                "_first_word_start_ms": first_start,
                "_last_word_end_ms": last_end,
            }
        )
    _assign_timings(raw_blocks, settings)
    for index, block in enumerate(raw_blocks):
        block["context_before"] = _context_text(
            raw_blocks, index, settings.context_blocks, before=True
        )
        block["context_after"] = _context_text(
            raw_blocks, index, settings.context_blocks, before=False
        )
        block["risk_flags"] = sorted(set(block["risk_flags"]))
        block.pop("_first_word_start_ms", None)
        block.pop("_last_word_end_ms", None)

    report = validate_segmentation(raw_blocks, words, config=settings)
    report.raise_if_invalid()
    return raw_blocks


def _mixed_speaker_risk(text: str) -> bool:
    turn_markers = len(re.findall(r"(?:^|\n)\s*[-–—]\s*\S", text))
    return turn_markers > 1


def validate_segmentation(
    blocks: Sequence[Mapping[str, Any]],
    source_words: Sequence[Mapping[str, Any]] | None = None,
    *,
    config: SegmentationConfig | None = None,
) -> SegmentationReport:
    """Hard-check order, timing, text coverage, overlap, gaps, and turn mixing."""

    settings = config or SegmentationConfig()
    errors: list[str] = []
    warnings: list[str] = []
    overlap_count = 0
    invalid_timing_count = 0
    internal_gap_count = 0
    mixed_speaker_count = 0
    empty_text_count = 0
    nonsequential_count = 0
    previous_end = -1
    for expected_index, block in enumerate(blocks, start=1):
        prefix = f"block {expected_index}"
        try:
            actual_index = int(block["block_index"])
            start = int(block["start_ms"])
            end = int(block["end_ms"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: malformed index/timing")
            invalid_timing_count += 1
            continue
        if actual_index != expected_index:
            nonsequential_count += 1
            errors.append(f"{prefix}: block_index is {actual_index}")
        if start < 0 or end <= start:
            invalid_timing_count += 1
            errors.append(f"{prefix}: invalid interval {start}-{end}")
        if start <= previous_end:
            overlap_count += 1
            errors.append(f"{prefix}: overlaps prior block ending at {previous_end}")
        previous_end = end
        text = str(block.get("primary_text", "")).strip()
        timing_text = str(block.get("timing_text", "")).strip()
        if not text or not timing_text:
            empty_text_count += 1
            errors.append(f"{prefix}: empty primary/timing text")
        if int(block.get("episode", 0)) <= 0:
            errors.append(f"{prefix}: invalid episode")
        vad_info = block.get("vad_info") or {}
        max_gap = int(vad_info.get("max_internal_gap_ms", 0)) if isinstance(vad_info, Mapping) else 0
        if max_gap > settings.internal_gap_review_ms:
            internal_gap_count += 1
            errors.append(f"{prefix}: unresolved internal gap {max_gap} ms")
        if _mixed_speaker_risk(text):
            mixed_speaker_count += 1
            errors.append(f"{prefix}: contains multiple explicit dialogue turns")
        if end - start > settings.maximum_duration_ms:
            warnings.append(f"{prefix}: duration {end - start} ms exceeds preferred maximum")
        if len(text) > settings.preferred_total_chars:
            warnings.append(
                f"{prefix}: source length {len(text)} exceeds {settings.preferred_total_chars} characters"
            )

    source_text_mismatch_count = 0
    if source_words is not None:
        try:
            prepared = _prepare_words(source_words)
            expected_text = _normalise_compare(_join_words(prepared))
            actual_text = _normalise_compare(
                " ".join(str(block.get("timing_text", "")) for block in blocks)
            )
            if expected_text != actual_text:
                source_text_mismatch_count = 1
                errors.append("segmented timing_text does not preserve the complete ordered word text")
        except SegmentationError as exc:
            source_text_mismatch_count = 1
            errors.append(f"source word validation failed: {exc}")

    return SegmentationReport(
        valid=not errors,
        block_count=len(blocks),
        overlap_count=overlap_count,
        invalid_timing_count=invalid_timing_count,
        unresolved_internal_gap_count=internal_gap_count,
        mixed_speaker_block_count=mixed_speaker_count,
        empty_text_count=empty_text_count,
        nonsequential_index_count=nonsequential_count,
        source_text_mismatch_count=source_text_mismatch_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _load_transcription(value: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        data = dict(value)
        return data, sha256_json(data)
    path = Path(value)
    if not path.is_file():
        raise SegmentationError(f"Transcription JSON does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SegmentationError(f"Cannot read transcription JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SegmentationError("Transcription JSON root must be an object")
    return data, sha256_file(path)


def _report_from_dict(value: Mapping[str, Any]) -> SegmentationReport:
    return SegmentationReport(
        valid=bool(value["valid"]),
        block_count=int(value["block_count"]),
        overlap_count=int(value["overlap_count"]),
        invalid_timing_count=int(value["invalid_timing_count"]),
        unresolved_internal_gap_count=int(value["unresolved_internal_gap_count"]),
        mixed_speaker_block_count=int(value["mixed_speaker_block_count"]),
        empty_text_count=int(value["empty_text_count"]),
        nonsequential_index_count=int(value["nonsequential_index_count"]),
        source_text_mismatch_count=int(value["source_text_mismatch_count"]),
        errors=tuple(value.get("errors", [])),
        warnings=tuple(value.get("warnings", [])),
    )


def _resume_segmentation(
    marker: Mapping[str, Any], marker_path: Path, source_words: Sequence[Mapping[str, Any]], config: SegmentationConfig
) -> SegmentationResult | None:
    try:
        output_path = Path(marker["outputs"]["segmentation"]["path"])
        with output_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        blocks = data["blocks"]
        if not isinstance(blocks, list):
            return None
        report = validate_segmentation(blocks, source_words, config=config)
        if not report.valid:
            return None
        saved_report = _report_from_dict(data["report"])
        if not saved_report.valid or saved_report.block_count != len(blocks):
            return None
        return SegmentationResult(
            blocks=blocks,
            output_path=output_path,
            marker_path=marker_path,
            report=report,
            resumed=True,
        )
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def segment_transcription(
    transcription: Mapping[str, Any] | str | Path,
    episode: int,
    prepare_dir: str | Path,
    *,
    config: SegmentationConfig | None = None,
    youtube_captions: Sequence[Mapping[str, Any]] | None = None,
    force: bool = False,
) -> SegmentationResult:
    """Build/resume an episode-local segmentation artifact and verified marker."""

    settings = config or SegmentationConfig()
    data, transcription_hash = _load_transcription(transcription)
    destination = Path(prepare_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "segmentation.json"
    marker_path = destination / "segmentation.done.json"
    captions_override_hash = (
        sha256_json(list(youtube_captions)) if youtube_captions is not None else None
    )
    input_descriptor = {
        "transcription_sha256": transcription_hash,
        "episode": episode,
        "segmentation_config": asdict(settings),
        "captions_override_sha256": captions_override_hash,
        "format_version": SEGMENTATION_FORMAT_VERSION,
    }
    input_hash = sha256_json(input_descriptor)
    source_words = data.get("words") or []

    if not force:
        marker = load_valid_stage_marker(
            marker_path,
            stage="segmentation",
            input_sha256=input_hash,
            required_output_keys=("segmentation",),
            allowed_root=destination,
        )
        if marker is not None:
            result = _resume_segmentation(marker, marker_path, source_words, settings)
            if result is not None:
                LOGGER.info("Segmentation resumed from verified marker: %s", marker_path)
                return result

    blocks = build_blocks(
        data,
        episode,
        config=settings,
        youtube_captions=youtube_captions,
    )
    report = validate_segmentation(blocks, source_words, config=settings)
    report.raise_if_invalid()
    artifact = {
        "format_version": SEGMENTATION_FORMAT_VERSION,
        "schema_version": settings.schema_version,
        "created_at": utc_now_iso(),
        "episode": episode,
        "transcription_sha256": transcription_hash,
        "segmentation_config": asdict(settings),
        "block_count": len(blocks),
        "blocks": blocks,
        "report": asdict(report),
    }
    atomic_write_json(output_path, artifact)
    output_hash = sha256_file(output_path)
    write_stage_marker(
        marker_path,
        stage="segmentation",
        input_sha256=input_hash,
        outputs={"segmentation": output_path},
        details={
            "transcription_sha256": transcription_hash,
            "segmentation_config_sha256": sha256_json(asdict(settings)),
            "schema_version": settings.schema_version,
            "block_count": len(blocks),
            "segmentation_sha256": output_hash,
            "warning_count": len(report.warnings),
        },
    )
    return SegmentationResult(
        blocks=blocks,
        output_path=output_path,
        marker_path=marker_path,
        report=report,
        resumed=False,
    )


def validate_segmentation_marker(
    marker_path: str | Path,
    *,
    transcription_sha256: str,
    episode: int,
    config: SegmentationConfig | None = None,
    captions_override_sha256: str | None = None,
) -> bool:
    """Validate marker identity and artifact hash (semantic recheck needs source words)."""

    settings = config or SegmentationConfig()
    marker_file = Path(marker_path)
    input_hash = sha256_json(
        {
            "transcription_sha256": transcription_sha256,
            "episode": episode,
            "segmentation_config": asdict(settings),
            "captions_override_sha256": captions_override_sha256,
            "format_version": SEGMENTATION_FORMAT_VERSION,
        }
    )
    return (
        load_valid_stage_marker(
            marker_file,
            stage="segmentation",
            input_sha256=input_hash,
            required_output_keys=("segmentation",),
            allowed_root=marker_file.parent,
        )
        is not None
    )


__all__ = [
    "SegmentationConfig",
    "SegmentationError",
    "SegmentationReport",
    "SegmentationResult",
    "build_blocks",
    "segment_transcription",
    "validate_segmentation",
    "validate_segmentation_marker",
]
