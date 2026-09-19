from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .download import atomic_write_bytes, atomic_write_json, sha256_file, sha256_json
from .id_translation import validate_aligned_turkish_schema
from .schema import SchemaError, atomic_write_jsonl, read_jsonl as read_schema_jsonl
from .subtitle_metadata import is_subtitle_credit, normalized_text, repetition_warnings


WORD_TIMELINE_FORMAT = "mas-semantic-word-timeline-1"
TIMING_SOURCE = "semantic_word_span_v1"
ALIGNMENT_POLICY = "semantic-block-v1"
STRICT_ALIGNMENT_POLICY = "strict-ctc-v1"


class SemanticAlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticAlignmentConfig:
    hard_gap_ms: int = 1000
    sentence_gap_ms: int = 180
    context_words: int = 12
    maximum_window_words: int = 80
    maximum_window_ms: int = 30000
    start_lead_ms: int = 0
    end_padding_ms: int = 220
    next_speech_guard_ms: int = 80
    maximum_cps: float = 20.0
    target_chars_per_line: int = 42
    maximum_chars_per_line: int = 84

    def __post_init__(self):
        if self.hard_gap_ms != 1000:
            raise ValueError("hard_gap_ms must remain 1000 ms")
        if not 0 <= self.start_lead_ms <= 80:
            raise ValueError("start_lead_ms must be between 0 and 80 ms")
        if not 180 <= self.end_padding_ms <= 240:
            raise ValueError("end_padding_ms must be between 180 and 240 ms")
        if self.next_speech_guard_ms != 80:
            raise ValueError("next_speech_guard_ms must remain 80 ms")
        if self.context_words < 0 or self.maximum_window_words < 1:
            raise ValueError("semantic window limits are invalid")
        if self.maximum_window_ms < self.hard_gap_ms or self.maximum_cps <= 0:
            raise ValueError("semantic timing limits are invalid")


def _strict_clone(value):
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise SemanticAlignmentError(f"Semantic evidence is not strict JSON: {exc}") from exc


def write_jsonl(path, records):
    records = list(records)
    if not records:
        return atomic_write_bytes(path, b"\n")
    try:
        return atomic_write_jsonl(path, records)
    except SchemaError as exc:
        raise SemanticAlignmentError(f"Invalid semantic JSONL: {exc}") from exc


def read_jsonl(path):
    try:
        return read_schema_jsonl(path)
    except (SchemaError, UnicodeError) as exc:
        raise SemanticAlignmentError(f"Invalid semantic JSONL: {exc}") from exc


def _require_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SemanticAlignmentError(f"{label} must be a lowercase SHA-256 digest")
    return value


def semantic_producer_identity():
    paths = [Path(__file__), Path(__file__).with_name("semantic_alignment_handoff.py")]
    files = {
        path.name: sha256_file(path)
        for path in paths
        if path.is_file()
    }
    return {
        "format": "mas-semantic-producer-1",
        "files": files,
        "sha256": sha256_json(files),
    }


def _word_text(value):
    return str(value.get("text", value.get("word", ""))).strip()


def _finite_int(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SemanticAlignmentError(f"{label} must be an integer >= {minimum}")
    return value


def _known_metadata_reason(text):
    if is_subtitle_credit(text):
        return "known_subtitle_credit_metadata"
    lexical = normalized_text(text)
    if lexical in {
        "izlediğiniz için teşekkür ederim",
        "izlediginiz icin tesekkur ederim",
    }:
        return "known_non_dialogue_metadata"
    return None


def _vad_for_word(word, vad_regions):
    matches = []
    for fallback, region in enumerate(vad_regions, start=1):
        start = region.get("start_ms")
        end = region.get("end_ms")
        if (
            type(start) is int
            and type(end) is int
            and word["start_ms"] < end
            and start < word["end_ms"]
        ):
            index = region.get("vad_region_index", fallback)
            if isinstance(index, bool) or not isinstance(index, int):
                index = fallback
            matches.append((index, start, end))
    if not matches:
        return None, None, None
    matches.sort()
    return matches[0]


def build_word_timeline(
    raw_asr,
    output_dir,
    *,
    source_sha256,
    timing_source_sha256,
    producer_identity=None,
):
    source_sha256 = _require_sha(source_sha256, "source_sha256")
    timing_source_sha256 = _require_sha(timing_source_sha256, "timing_source_sha256")
    if not isinstance(raw_asr, Mapping):
        raise SemanticAlignmentError("raw_asr must be an object")
    source_words = raw_asr.get("words")
    segments = raw_asr.get("segments")
    vad_regions = raw_asr.get("vad_regions", [])
    if not isinstance(source_words, list) or not source_words:
        raise SemanticAlignmentError("Timing source contains no word timestamps")
    if not isinstance(segments, list) or not isinstance(vad_regions, list):
        raise SemanticAlignmentError("Timing source segment/VAD inventory is malformed")

    segment_by_id = {}
    for position, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping):
            raise SemanticAlignmentError(f"Segment {position} is not an object")
        uid = str(segment.get("segment_id", f"segment-{position:06d}"))
        if uid in segment_by_id:
            raise SemanticAlignmentError(f"Duplicate source segment identity: {uid}")
        segment_by_id[uid] = segment

    base_words = []
    previous_start = -1
    for position, source in enumerate(source_words, start=1):
        if not isinstance(source, Mapping):
            raise SemanticAlignmentError(f"Timing word {position} is not an object")
        start = _finite_int(source.get("start_ms"), f"word {position} start_ms")
        end = _finite_int(source.get("end_ms"), f"word {position} end_ms", minimum=1)
        if end <= start or start < previous_start:
            raise SemanticAlignmentError(f"Timing words are invalid or reordered at {position}")
        text = _word_text(source)
        if not text:
            raise SemanticAlignmentError(f"Timing word {position} has empty text")
        source_segment_uid = str(source.get("segment_id", ""))
        segment = segment_by_id.get(source_segment_uid)
        reason = _known_metadata_reason(segment.get("text", "")) if segment else None
        probability = source.get("probability", source.get("confidence"))
        if probability is not None:
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
                raise SemanticAlignmentError(f"Timing word {position} confidence is invalid")
            probability = float(probability)
        vad_index, vad_start, vad_end = _vad_for_word(
            {"start_ms": start, "end_ms": end}, vad_regions
        )
        item = {
            "sequence_index": position,
            "source_segment_uid": source_segment_uid,
            "text": text,
            "normalized": normalized_text(text),
            "start_ms": start,
            "end_ms": end,
            "confidence": probability,
            "speaker_id": source.get("speaker_id"),
            "vad_region_index": vad_index,
            "vad_start_ms": vad_start,
            "vad_end_ms": vad_end,
            "scene_id": source.get("scene_id"),
            "eligible_speech": reason is None,
            "quarantine_reason": reason,
        }
        if item["speaker_id"] is not None and (
            not isinstance(item["speaker_id"], str) or not item["speaker_id"].strip()
        ):
            raise SemanticAlignmentError(f"Timing word {position} speaker_id is invalid")
        base_words.append(item)
        previous_start = start

    timeline_identity = {
        "format": WORD_TIMELINE_FORMAT,
        "source_sha256": source_sha256,
        "timing_source_sha256": timing_source_sha256,
        "words": base_words,
    }
    identity_sha = sha256_json(timeline_identity)
    words = []
    for item in base_words:
        word = copy.deepcopy(item)
        word["word_id"] = f"tw-{word['sequence_index']:08d}-{identity_sha[:10]}"
        words.append(word)

    word_ids_by_segment = {}
    for word in words:
        word_ids_by_segment.setdefault(word["source_segment_uid"], []).append(word["word_id"])
    quarantine = []
    for position, segment in enumerate(segments, start=1):
        uid = str(segment.get("segment_id", f"segment-{position:06d}"))
        reason = _known_metadata_reason(segment.get("text", ""))
        if reason is None:
            continue
        quarantine.append(
            {
                "source_uid": uid,
                "reason": reason,
                "text": str(segment.get("text", "")).strip(),
                "evidence": "exact known metadata detector",
                "start_ms": segment.get("start_ms"),
                "end_ms": segment.get("end_ms"),
                "word_ids": word_ids_by_segment.get(uid, []),
            }
        )

    output = Path(output_dir)
    timeline_path = output / "word_timeline.jsonl"
    quarantine_path = output / "quarantine.json"
    manifest_path = output / "word_timeline.manifest.json"
    write_jsonl(timeline_path, words)
    atomic_write_json(
        quarantine_path,
        {
            "format": "mas-semantic-quarantine-1",
            "record_count": len(quarantine),
            "word_count": sum(len(item["word_ids"]) for item in quarantine),
            "records": quarantine,
            "repetition_warnings": repetition_warnings(
                [
                    {
                        "uid": str(segment.get("segment_id", index)),
                        "text": segment.get("text", ""),
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                    }
                    for index, segment in enumerate(segments, start=1)
                ]
            ),
        },
    )
    producer = _strict_clone(producer_identity or semantic_producer_identity())
    manifest = {
        "format": WORD_TIMELINE_FORMAT,
        "format_version": 1,
        "source_sha256": source_sha256,
        "timing_source_sha256": timing_source_sha256,
        "word_count": len(words),
        "eligible_word_count": sum(word["eligible_speech"] for word in words),
        "quarantined_word_count": sum(not word["eligible_speech"] for word in words),
        "first_start_ms": words[0]["start_ms"],
        "last_end_ms": words[-1]["end_ms"],
        "timeline_sha256": sha256_file(timeline_path),
        "quarantine_sha256": sha256_file(quarantine_path),
        "identity_sha256": identity_sha,
        "producer_identity": producer,
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "words": words,
        "quarantine": quarantine,
        "manifest": manifest,
        "timeline_path": timeline_path,
        "quarantine_path": quarantine_path,
        "manifest_path": manifest_path,
    }


def load_word_timeline(output_dir):
    root = Path(output_dir)
    timeline_path = root / "word_timeline.jsonl"
    manifest_path = root / "word_timeline.manifest.json"
    quarantine_path = root / "quarantine.json"
    if not all(path.is_file() for path in (timeline_path, manifest_path, quarantine_path)):
        raise SemanticAlignmentError("Semantic word timeline artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    words = read_jsonl(timeline_path)
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != WORD_TIMELINE_FORMAT
        or manifest.get("word_count") != len(words)
        or manifest.get("timeline_sha256") != sha256_file(timeline_path)
        or manifest.get("quarantine_sha256") != sha256_file(quarantine_path)
        or quarantine.get("format") != "mas-semantic-quarantine-1"
    ):
        raise SemanticAlignmentError("Semantic word timeline binding is invalid")
    expected_ids = [
        f"tw-{index:08d}-{manifest['identity_sha256'][:10]}"
        for index in range(1, len(words) + 1)
    ]
    if [word.get("word_id") for word in words] != expected_ids:
        raise SemanticAlignmentError("Semantic word IDs changed or are reordered")
    return {"words": words, "quarantine": quarantine["records"], "manifest": manifest}


def _ends_sentence(text):
    return re.search(r"[.!?…][\"')\]]?$", str(text).strip()) is not None


def _hard_boundary(previous, current, config):
    if current["sequence_index"] != previous["sequence_index"] + 1:
        return True
    if current["start_ms"] - previous["end_ms"] > config.hard_gap_ms:
        return True
    prior_speaker = previous.get("speaker_id")
    current_speaker = current.get("speaker_id")
    if prior_speaker is not None and current_speaker is not None and prior_speaker != current_speaker:
        return True
    prior_vad = previous.get("vad_region_index")
    current_vad = current.get("vad_region_index")
    if prior_vad is not None and current_vad is not None and prior_vad != current_vad:
        return True
    prior_scene = previous.get("scene_id")
    current_scene = current.get("scene_id")
    return prior_scene is not None and current_scene is not None and prior_scene != current_scene


def _candidate_record(record, source, index):
    if not isinstance(record, Mapping):
        return None
    text = record.get("tr_corrected", record.get("asr_text", record.get("text", "")))
    start = record.get("coarse_start_ms", record.get("start_ms"))
    end = record.get("coarse_end_ms", record.get("end_ms"))
    if not isinstance(text, str) or not text.strip() or type(start) is not int or type(end) is not int or end <= start:
        return None
    uid = record.get("utterance_uid", record.get("segment_id", record.get("caption_index", index)))
    return {
        "source": source,
        "source_uid": str(uid),
        "start_ms": start,
        "end_ms": end,
        "text": text.strip(),
    }


def evidence_sources(raw_asr, *, corrected_records=()):
    if not isinstance(raw_asr, Mapping):
        raise SemanticAlignmentError("raw_asr must be an object")
    sources = {"primary": [], "verification": [], "timing": [], "youtube": []}
    primary = list(corrected_records) or list(raw_asr.get("correction_utterances", []))
    for index, record in enumerate(primary, start=1):
        candidate = _candidate_record(record, "primary", index)
        if candidate and not _known_metadata_reason(candidate["text"]):
            sources["primary"].append(candidate)
    for index, record in enumerate(raw_asr.get("verification_results", []), start=1):
        candidate = _candidate_record(record, "verification", index)
        if candidate and not _known_metadata_reason(candidate["text"]):
            sources["verification"].append(candidate)
    for index, record in enumerate(raw_asr.get("segments", []), start=1):
        candidate = _candidate_record(record, "timing", index)
        if candidate and not _known_metadata_reason(candidate["text"]):
            sources["timing"].append(candidate)
    for index, record in enumerate(raw_asr.get("youtube_captions", []), start=1):
        candidate = _candidate_record(record, "youtube", index)
        if candidate and not _known_metadata_reason(candidate["text"]):
            sources["youtube"].append(candidate)
    return sources


def _window_candidates(sources, start_ms, end_ms):
    result = {}
    for name, candidates in sources.items():
        result[name + "_candidates"] = [
            copy.deepcopy(candidate)
            for candidate in candidates
            if candidate["start_ms"] < end_ms and start_ms < candidate["end_ms"]
        ]
    return result


def build_semantic_windows(timeline, sources, *, config=None):
    settings = config or SemanticAlignmentConfig()
    words = timeline.get("words") if isinstance(timeline, Mapping) else None
    if not isinstance(words, list) or not words:
        raise SemanticAlignmentError("Semantic timeline contains no words")
    eligible = [word for word in words if word.get("eligible_speech") is True]
    if not eligible:
        raise SemanticAlignmentError("Semantic timeline contains no eligible speech words")
    chunks = []
    current = []
    for word in eligible:
        boundary = False
        if current:
            previous = current[-1]
            boundary = _hard_boundary(previous, word, settings)
            boundary = boundary or len(current) >= settings.maximum_window_words
            boundary = boundary or word["end_ms"] - current[0]["start_ms"] > settings.maximum_window_ms
            boundary = boundary or (
                len(current) >= 8
                and _ends_sentence(previous["text"])
                and word["start_ms"] - previous["end_ms"] >= settings.sentence_gap_ms
            )
        if boundary:
            chunks.append(current)
            current = []
        current.append(word)
    if current:
        chunks.append(current)

    eligible_positions = {word["word_id"]: index for index, word in enumerate(eligible)}
    windows = []
    for number, owned in enumerate(chunks, start=1):
        first_position = eligible_positions[owned[0]["word_id"]]
        last_position = eligible_positions[owned[-1]["word_id"]]
        previous_context = eligible[max(0, first_position - settings.context_words):first_position]
        next_context = eligible[last_position + 1:last_position + 1 + settings.context_words]
        window_id = f"sw-{number:06d}"
        item = {
            "window_id": window_id,
            "owned_word_ids": [word["word_id"] for word in owned],
            "timing_words": [copy.deepcopy(word) for word in owned],
            **_window_candidates(sources, owned[0]["start_ms"], owned[-1]["end_ms"]),
            "vad": {
                "region_ids": list(dict.fromkeys(
                    word["vad_region_index"] for word in owned if word.get("vad_region_index") is not None
                )),
                "start_ms": min(
                    [word["vad_start_ms"] for word in owned if word.get("vad_start_ms") is not None]
                    or [owned[0]["start_ms"]]
                ),
                "end_ms": max(
                    [word["vad_end_ms"] for word in owned if word.get("vad_end_ms") is not None]
                    or [owned[-1]["end_ms"]]
                ),
            },
            "speaker_hints": list(dict.fromkeys(
                word["speaker_id"] for word in owned if word.get("speaker_id") is not None
            )),
            "scene_hints": list(dict.fromkeys(
                word["scene_id"] for word in owned if word.get("scene_id") is not None
            )),
            "previous_context": {
                "read_only_context_words": [copy.deepcopy(word) for word in previous_context]
            },
            "next_context": {
                "read_only_context_words": [copy.deepcopy(word) for word in next_context]
            },
        }
        windows.append(item)
    ownership = [word_id for window in windows for word_id in window["owned_word_ids"]]
    expected = [word["word_id"] for word in eligible]
    if ownership != expected or len(ownership) != len(set(ownership)):
        raise SemanticAlignmentError("Semantic window ownership is not exact and disjoint")
    return windows


def _tokens(text):
    return normalized_text(text).split()


def _candidate_matches(tokens, needle):
    if not needle or len(needle) > len(tokens):
        return []
    return [
        index
        for index in range(0, len(tokens) - len(needle) + 1)
        if tokens[index:index + len(needle)] == needle
    ]


def _span_safe(words, first, last, config):
    span = words[first:last + 1]
    if any(
        current["start_ms"] - previous["end_ms"] > config.hard_gap_ms
        for previous, current in zip(span, span[1:])
    ):
        return False
    speakers = {word.get("speaker_id") for word in span if word.get("speaker_id") is not None}
    scenes = {word.get("scene_id") for word in span if word.get("scene_id") is not None}
    return len(speakers) <= 1 and len(scenes) <= 1


def deterministic_exact_results(windows, *, config=None):
    settings = config or SemanticAlignmentConfig()
    results = []
    unresolved = []
    source_priority = {"primary": 0, "verification": 1, "youtube": 2, "timing": 3}
    for window in windows:
        words = window["timing_words"]
        word_tokens = [word["normalized"] for word in words]
        spans = {}
        for field in (
            "primary_candidates",
            "verification_candidates",
            "youtube_candidates",
            "timing_candidates",
        ):
            for candidate in window[field]:
                candidate_tokens = _tokens(candidate["text"])
                matches = _candidate_matches(word_tokens, candidate_tokens)
                if len(matches) != 1:
                    continue
                first = matches[0]
                last = first + len(candidate_tokens) - 1
                if not _span_safe(words, first, last, settings):
                    continue
                key = (first, last)
                value = {
                    "first": first,
                    "last": last,
                    "text": candidate["text"],
                    "source": candidate["source"],
                    "source_uid": candidate["source_uid"],
                }
                prior = spans.get(key)
                if prior is None or source_priority[value["source"]] < source_priority[prior["source"]]:
                    spans[key] = value
        by_first = {}
        for span in spans.values():
            by_first.setdefault(span["first"], []).append(span)
        paths = {0: []}
        for offset in range(len(words)):
            if offset not in paths:
                continue
            choices = sorted(
                by_first.get(offset, []),
                key=lambda item: (
                    -(item["last"] - item["first"]),
                    source_priority[item["source"]],
                    item["source_uid"],
                ),
            )
            for choice in choices:
                end = choice["last"] + 1
                candidate_path = paths[offset] + [choice]
                prior = paths.get(end)
                if prior is None or len(candidate_path) < len(prior):
                    paths[end] = candidate_path
        path = paths.get(len(words))
        if not path:
            unresolved.append(copy.deepcopy(window))
            continue
        blocks = [
            {
                "first_word_id": words[item["first"]]["word_id"],
                "last_word_id": words[item["last"]]["word_id"],
                "final_tr": item["text"],
                "confidence": "deterministic",
                "review_required": False,
                "note": "",
            }
            for item in path
        ]
        result = {
            "window_id": window["window_id"],
            "resolution": "deterministic_exact",
            "status": "AUTO_PASS",
            "blocks": blocks,
            "evidence_sha256": sha256_json(
                {
                    "owned_word_ids": window["owned_word_ids"],
                    "blocks": blocks,
                }
            ),
        }
        if len(blocks) == 1:
            result.update(
                first_word_id=blocks[0]["first_word_id"],
                last_word_id=blocks[0]["last_word_id"],
                final_tr=blocks[0]["final_tr"],
            )
        results.append(result)
    return results, unresolved


def _validate_resolution(window, resolution, config):
    expected_window = window["window_id"]
    if resolution.get("window_id") != expected_window:
        raise SemanticAlignmentError(f"Semantic return window identity changed: {expected_window}")
    blocks = resolution.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise SemanticAlignmentError(f"Semantic window {expected_window} has no blocks")
    ids = window["owned_word_ids"]
    index = {word_id: offset for offset, word_id in enumerate(ids)}
    words = window["timing_words"]
    previous_last = -1
    normalized = []
    for number, block in enumerate(blocks, start=1):
        if not isinstance(block, Mapping):
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} is not an object")
        first_id = block.get("first_word_id")
        last_id = block.get("last_word_id")
        if first_id not in index or last_id not in index:
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} uses an unowned word ID")
        first = index[first_id]
        last = index[last_id]
        if first > last or first <= previous_last:
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} is reordered or overlaps")
        if not _span_safe(words, first, last, config):
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} crosses a hard boundary")
        text = block.get("final_tr")
        if not isinstance(text, str) or not text.strip():
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} has empty Turkish text")
        if _known_metadata_reason(text):
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} contains metadata")
        review_required = block.get("review_required", False)
        confidence = block.get("confidence", "")
        note = block.get("note", "")
        if type(review_required) is not bool or not isinstance(confidence, str) or not isinstance(note, str):
            raise SemanticAlignmentError(f"Semantic block {expected_window}/{number} review fields are invalid")
        normalized.append(
            {
                "first_word_id": first_id,
                "last_word_id": last_id,
                "final_tr": text.strip(),
                "confidence": confidence,
                "review_required": review_required,
                "note": note,
            }
        )
        previous_last = last
    return normalized


def validate_coarse_fallback_approvals(
    approvals,
    windows,
    *,
    source_sha256,
    word_timeline_sha256,
):
    if approvals in (None, []):
        return {}
    if not isinstance(approvals, list):
        raise SemanticAlignmentError("Semantic coarse fallback approvals must be a list")
    by_window = {window["window_id"]: window for window in windows}
    accepted = {}
    required = {
        "window_id",
        "source_sha256",
        "word_timeline_sha256",
        "candidate_timing",
        "reason",
        "approved",
        "approval_sha256",
        "approval_auth_tag",
    }
    for approval in approvals:
        if not isinstance(approval, dict) or set(approval) != required:
            raise SemanticAlignmentError("Semantic coarse fallback approval fields are invalid")
        body = {
            key: value
            for key, value in approval.items()
            if key not in {"approval_sha256", "approval_auth_tag"}
        }
        from .raw_asr import RawASRV2Config, _raw_asr_auth_key, _raw_asr_auth_tag
        key = _raw_asr_auth_key(RawASRV2Config())
        expected_auth_tag = _raw_asr_auth_tag(
            key, "semantic-coarse-fallback-approval", body
        )
        if (
            approval["approval_sha256"] != sha256_json(body)
            or not isinstance(approval["approval_auth_tag"], str)
            or not hmac.compare_digest(approval["approval_auth_tag"], expected_auth_tag)
        ):
            raise SemanticAlignmentError("Semantic coarse fallback approval signature changed")
        window_id = approval["window_id"]
        if (
            window_id not in by_window
            or window_id in accepted
            or approval["source_sha256"] != source_sha256
            or approval["word_timeline_sha256"] != word_timeline_sha256
            or approval["approved"] is not True
            or not isinstance(approval["reason"], str)
            or not approval["reason"].strip()
        ):
            raise SemanticAlignmentError("Semantic coarse fallback approval binding is invalid")
        timing = approval["candidate_timing"]
        fields = {"first_word_id", "last_word_id", "start_ms", "end_ms", "final_tr"}
        if not isinstance(timing, dict) or set(timing) != fields:
            raise SemanticAlignmentError("Semantic coarse fallback candidate is invalid")
        blocks = _validate_resolution(
            by_window[window_id],
            {"window_id": window_id, "blocks": [{
                "first_word_id": timing["first_word_id"],
                "last_word_id": timing["last_word_id"],
                "final_tr": timing["final_tr"],
                "confidence": "approved_coarse",
                "review_required": False,
                "note": approval["reason"],
            }]},
            SemanticAlignmentConfig(),
        )
        if (
            type(timing["start_ms"]) is not int
            or type(timing["end_ms"]) is not int
            or timing["start_ms"] < 0
            or timing["end_ms"] <= timing["start_ms"]
        ):
            raise SemanticAlignmentError("Semantic coarse fallback interval is invalid")
        accepted[window_id] = {
            "window_id": window_id,
            "resolution": "approved_coarse_component_fallback",
            "timing_source": "approved_coarse_component_fallback_v1",
            "candidate_timing": copy.deepcopy(timing),
            "blocks": blocks,
            "approval_sha256": approval["approval_sha256"],
        }
    return accepted


def sign_coarse_fallback_approval(approval):
    if not isinstance(approval, Mapping):
        raise SemanticAlignmentError("Semantic coarse fallback approval must be an object")
    body = _strict_clone(dict(approval))
    if {"approval_sha256", "approval_auth_tag"} & set(body):
        raise SemanticAlignmentError("Unsigned approval body contains signature fields")
    from .raw_asr import RawASRV2Config, _raw_asr_auth_key, _raw_asr_auth_tag
    key = _raw_asr_auth_key(RawASRV2Config())
    return {
        **body,
        "approval_sha256": sha256_json(body),
        "approval_auth_tag": _raw_asr_auth_tag(
            key, "semantic-coarse-fallback-approval", body
        ),
    }


def _line_metrics(text, config):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else line + " " + word
        if len(candidate) <= config.target_chars_per_line or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def finalize_semantic_blocks(
    *,
    episode,
    source_sha256,
    timeline,
    windows,
    deterministic_results,
    semantic_results,
    output_dir,
    config=None,
    coarse_approvals=None,
    allow_approved_coarse_release=False,
):
    settings = config or SemanticAlignmentConfig()
    source_sha256 = _require_sha(source_sha256, "source_sha256")
    timeline_words = timeline.get("words")
    timeline_manifest = timeline.get("manifest")
    if not isinstance(timeline_words, list) or not isinstance(timeline_manifest, Mapping):
        raise SemanticAlignmentError("Semantic timeline is incomplete")
    window_ids = [window["window_id"] for window in windows]
    if len(window_ids) != len(set(window_ids)):
        raise SemanticAlignmentError("Semantic windows contain duplicate identities")
    resolutions = {}
    for result in deterministic_results:
        window_id = result.get("window_id")
        if window_id in resolutions:
            raise SemanticAlignmentError("Duplicate deterministic semantic window")
        resolutions[window_id] = copy.deepcopy(result)
    for result in semantic_results:
        window_id = result.get("window_id")
        if window_id in resolutions:
            raise SemanticAlignmentError("Duplicate semantic return window")
        resolutions[window_id] = copy.deepcopy(result)
    fallback = validate_coarse_fallback_approvals(
        coarse_approvals,
        windows,
        source_sha256=source_sha256,
        word_timeline_sha256=timeline_manifest["timeline_sha256"],
    )
    for window_id, result in fallback.items():
        if window_id in resolutions:
            blocks = resolutions[window_id].get("blocks", [])
            if not blocks or not any(block.get("review_required") is True for block in blocks):
                raise SemanticAlignmentError("Approved coarse fallback duplicates a resolved window")
        resolutions[window_id] = result
    missing = [window_id for window_id in window_ids if window_id not in resolutions]
    if missing:
        review_path = Path(output_dir) / "semantic_alignment_review_required.json"
        atomic_write_json(
            review_path,
            {
                "format": "mas-semantic-review-required-1",
                "source_sha256": source_sha256,
                "word_timeline_sha256": timeline_manifest["timeline_sha256"],
                "unresolved_window_ids": missing,
            },
        )
        raise SemanticAlignmentError(
            "Semantic alignment has unresolved windows: " + ", ".join(missing[:20])
        )

    word_by_id = {word["word_id"]: word for word in timeline_words}
    owner = {}
    raw_blocks = []
    review_required_count = 0
    duplicate_word_ownership_count = 0
    cross_window_word_ownership_count = 0
    internal_gap_count = 0
    mixed_speaker_count = 0
    cross_scene_count = 0
    fallback_count = 0
    for window in windows:
        result = resolutions[window["window_id"]]
        blocks = _validate_resolution(window, result, settings)
        window_index = {word_id: index for index, word_id in enumerate(window["owned_word_ids"])}
        for block in blocks:
            first = window_index[block["first_word_id"]]
            last = window_index[block["last_word_id"]]
            span_ids = window["owned_word_ids"][first:last + 1]
            span_words = [word_by_id[word_id] for word_id in span_ids]
            for word_id in span_ids:
                if word_id in owner:
                    duplicate_word_ownership_count += 1
                    if owner[word_id] != window["window_id"]:
                        cross_window_word_ownership_count += 1
                owner[word_id] = window["window_id"]
            if any(
                current["start_ms"] - prior["end_ms"] > settings.hard_gap_ms
                for prior, current in zip(span_words, span_words[1:])
            ):
                internal_gap_count += 1
            speakers = {word.get("speaker_id") for word in span_words if word.get("speaker_id") is not None}
            scenes = {word.get("scene_id") for word in span_words if word.get("scene_id") is not None}
            if len(speakers) > 1:
                mixed_speaker_count += 1
            if len(scenes) > 1:
                cross_scene_count += 1
            review_required_count += block["review_required"] is True
            timing_source = result.get("timing_source", TIMING_SOURCE)
            fallback_count += timing_source == "approved_coarse_component_fallback_v1"
            raw_blocks.append(
                {
                    **block,
                    "semantic_window_id": window["window_id"],
                    "resolution": result.get("resolution", "gpt_semantic"),
                    "timing_source": timing_source,
                    "fallback_timing": result.get("candidate_timing"),
                    "speaker_id": next(iter(speakers)) if len(speakers) == 1 else None,
                    "scene_id": next(iter(scenes)) if len(scenes) == 1 else None,
                    "span_words": span_words,
                }
            )

    eligible_ids = [word["word_id"] for word in timeline_words if word.get("eligible_speech") is True]
    assigned_ids = [word_id for word_id in eligible_ids if word_id in owner]
    unknown_ids = [word_id for word_id in owner if word_id not in word_by_id]
    unassigned_ids = [word_id for word_id in eligible_ids if word_id not in owner]
    order_violations = sum(
        raw_blocks[index]["span_words"][0]["sequence_index"]
        <= raw_blocks[index - 1]["span_words"][-1]["sequence_index"]
        for index in range(1, len(raw_blocks))
    )

    final_blocks = []
    overlap_count = 0
    early_start_count = 0
    early_end_count = 0
    empty_text_count = 0
    more_than_two_lines_count = 0
    line_over_84_count = 0
    long_line_count = 0
    cps_failure_count = 0
    for index, block in enumerate(raw_blocks):
        first_word = block["span_words"][0]
        last_word = block["span_words"][-1]
        fallback_timing = block.get("fallback_timing")
        if fallback_timing is not None:
            start = fallback_timing["start_ms"]
            end = fallback_timing["end_ms"]
        else:
            start = first_word["start_ms"] - settings.start_lead_ms
            if first_word.get("vad_start_ms") is not None:
                start = max(start, first_word["vad_start_ms"])
            start = max(0, start)
            end = last_word["end_ms"] + settings.end_padding_ms
            if last_word.get("vad_end_ms") is not None:
                end = min(end, max(last_word["end_ms"], last_word["vad_end_ms"]))
        if index + 1 < len(raw_blocks):
            next_start = raw_blocks[index + 1]["span_words"][0]["start_ms"]
            guarded = next_start - settings.next_speech_guard_ms
            if guarded >= last_word["end_ms"]:
                end = min(end, guarded)
        if end < last_word["end_ms"]:
            early_end_count += 1
            end = last_word["end_ms"]
        if start > first_word["start_ms"]:
            early_start_count += 1
        if final_blocks and start < final_blocks[-1]["end_ms"]:
            overlap_count += 1
        text = block["final_tr"].strip()
        empty_text_count += not bool(text)
        lines = _line_metrics(text, settings)
        more_than_two_lines_count += len(lines) > 2
        line_over_84_count += any(len(line) > settings.maximum_chars_per_line for line in lines)
        long_line_count += any(len(line) > settings.target_chars_per_line for line in lines)
        duration = max(1, end - start)
        cps_failure_count += len(re.sub(r"\s+", "", text)) / (duration / 1000.0) > settings.maximum_cps
        identity = {
            "episode": episode,
            "source_sha256": source_sha256,
            "first_word_id": block["first_word_id"],
            "last_word_id": block["last_word_id"],
            "sequence_index": index + 1,
        }
        block_uid = f"MA{episode:02d}-BLK-{sha256_json(identity)[:16]}"
        final = {
            "block_uid": block_uid,
            "block_index": index + 1,
            "first_word_id": block["first_word_id"],
            "last_word_id": block["last_word_id"],
            "start_ms": start,
            "end_ms": end,
            "final_tr": text,
            "semantic_window_id": block["semantic_window_id"],
            "resolution": block["resolution"],
            "timing_source": block["timing_source"],
            "confidence": block["confidence"],
            "review_required": block["review_required"],
            "note": block["note"],
        }
        if block["speaker_id"] is not None:
            final["speaker_id"] = block["speaker_id"]
        if block["scene_id"] is not None:
            final["scene_id"] = block["scene_id"]
        final_blocks.append(final)

    report = {
        "format": "mas-semantic-release-qa-1",
        "alignment_policy": ALIGNMENT_POLICY,
        "timing_source": TIMING_SOURCE,
        "semantic_window_count": len(windows),
        "semantic_return_manifest_valid": True,
        "word_timeline_binding_valid": True,
        "eligible_word_count": len(eligible_ids),
        "assigned_word_count": len(assigned_ids),
        "quarantined_word_count": sum(not word.get("eligible_speech") for word in timeline_words),
        "unassigned_word_count": len(unassigned_ids),
        "unassigned_eligible_speech_word_count": len(unassigned_ids),
        "duplicate_owned_word_count": duplicate_word_ownership_count,
        "duplicate_word_ownership_count": duplicate_word_ownership_count,
        "cross_window_word_ownership_count": cross_window_word_ownership_count,
        "unknown_word_id_count": len(unknown_ids),
        "word_order_violation_count": order_violations,
        "internal_gap_over_1000ms_count": internal_gap_count,
        "mixed_known_speaker_block_count": mixed_speaker_count,
        "mixed_speaker_block_count": mixed_speaker_count,
        "cross_scene_block_count": cross_scene_count,
        "cross_scene_word_count": cross_scene_count,
        "orphan_boundary_word_count": len(unassigned_ids),
        "subtitle_spans_scene_change_count": cross_scene_count,
        "overlap_count": overlap_count,
        "early_start_count": early_start_count,
        "early_end_count": early_end_count,
        "empty_text_count": empty_text_count,
        "more_than_two_lines_count": more_than_two_lines_count,
        "line_over_84_count": line_over_84_count,
        "long_line_count": long_line_count,
        "cps_failure_count": cps_failure_count,
        "gpt_supplied_timestamp_field_count": 0,
        "semantic_review_required_count": review_required_count,
        "approved_coarse_fallback_count": fallback_count,
        "approved_coarse_release_allowed": allow_approved_coarse_release,
        "strict_ctc_pass": False,
        "semantic_alignment_pass": fallback_count == 0,
        "unassigned_word_ids": unassigned_ids,
    }
    zero_fields = (
        "unassigned_eligible_speech_word_count",
        "duplicate_word_ownership_count",
        "cross_window_word_ownership_count",
        "unknown_word_id_count",
        "word_order_violation_count",
        "internal_gap_over_1000ms_count",
        "mixed_known_speaker_block_count",
        "cross_scene_block_count",
        "overlap_count",
        "early_start_count",
        "early_end_count",
        "empty_text_count",
        "more_than_two_lines_count",
        "line_over_84_count",
        "cps_failure_count",
        "gpt_supplied_timestamp_field_count",
        "semantic_review_required_count",
    )
    report["word_ownership_complete"] = (
        report["eligible_word_count"] == report["assigned_word_count"]
        and report["eligible_word_count"] + report["quarantined_word_count"] == len(timeline_words)
    )
    report["release_eligible"] = (
        bool(windows)
        and report["word_ownership_complete"]
        and all(report[field] == 0 for field in zero_fields)
        and (fallback_count == 0 or allow_approved_coarse_release)
    )

    output = Path(output_dir)
    blocks_path = output / "final_blocks.jsonl"
    report_path = output / "semantic_release_qa.json"
    write_jsonl(blocks_path, final_blocks)
    atomic_write_json(report_path, report)
    review_path = output / "semantic_alignment_review_required.json"
    if review_required_count:
        atomic_write_json(
            review_path,
            {
                "format": "mas-semantic-review-required-1",
                "source_sha256": source_sha256,
                "word_timeline_sha256": timeline_manifest["timeline_sha256"],
                "unresolved_window_ids": [
                    result["window_id"]
                    for result in semantic_results
                    if any(block.get("review_required") is True for block in result.get("blocks", []))
                ],
            },
        )
    else:
        review_path.unlink(missing_ok=True)
    return {
        "blocks": final_blocks,
        "report": report,
        "blocks_path": blocks_path,
        "report_path": report_path,
    }


def build_semantic_translation_schema(
    *,
    episode,
    final_blocks,
    source_sha256,
    word_timeline_sha256,
    production_policy,
):
    blocks = []
    for block in final_blocks:
        provenance = {
            "alignment_policy": ALIGNMENT_POLICY,
            "timing_source": block["timing_source"],
            "strict_alignment_pass": False,
            "strict_ctc_pass": False,
            "semantic_alignment_pass": block["timing_source"] == TIMING_SOURCE,
            "source_sha256": source_sha256,
            "word_timeline_sha256": word_timeline_sha256,
            "first_word_id": block["first_word_id"],
            "last_word_id": block["last_word_id"],
            "semantic_window_id": block["semantic_window_id"],
        }
        item = {
            "block_uid": block["block_uid"],
            "block_index": block["block_index"],
            "start_ms": block["start_ms"],
            "end_ms": block["end_ms"],
            "tr_text": block["final_tr"],
            "alignment_provenance": provenance,
        }
        if "speaker_id" in block:
            item["speaker_id"] = block["speaker_id"]
        if "scene_id" in block:
            item["scene_id"] = block["scene_id"]
        blocks.append(item)
    schema = {
        "schema_version": "2.0",
        "episode": episode,
        "block_count": len(blocks),
        "blocks": blocks,
        "alignment_policy": ALIGNMENT_POLICY,
        "timing_source": TIMING_SOURCE,
        "strict_ctc_pass": False,
        "semantic_alignment_pass": all(
            block["timing_source"] == TIMING_SOURCE for block in final_blocks
        ),
        "production_policy": _strict_clone(production_policy),
    }
    return validate_aligned_turkish_schema(schema)


def semantic_run_contract(episode, delivery_scope):
    if delivery_scope not in {"whole-episode", "first-hour"}:
        raise SemanticAlignmentError("Unsupported delivery_scope")
    return {
        "format": "mas-run-contract-1",
        "episode": episode,
        "delivery_scope": delivery_scope,
        "alignment_policy": ALIGNMENT_POLICY if episode >= 15 else STRICT_ALIGNMENT_POLICY,
    }


def prepare_semantic_delivery_scope(
    episode_root,
    report,
    *,
    delivery_scope,
    finalization_sha256,
):
    from .srt import assert_srt_roundtrip, parse_srt, write_srt

    if delivery_scope not in {"whole-episode", "first-hour"}:
        raise SemanticAlignmentError("Unsupported semantic delivery scope")
    root = Path(episode_root)
    final_blocks_path = root / report["input_files"]["final_blocks"]["relative_path"]
    id_srt_path = root / report["outputs"]["id_srt"]["relative_path"]
    tr_srt_path = root / report["outputs"]["tr_srt"]["relative_path"]
    final_blocks = read_jsonl(final_blocks_path)
    id_entries = parse_srt(id_srt_path)
    tr_entries = parse_srt(tr_srt_path)
    expected = [(block["block_index"], block["start_ms"], block["end_ms"]) for block in final_blocks]
    if (
        not final_blocks
        or [(entry.index, entry.start_ms, entry.end_ms) for entry in id_entries] != expected
        or [(entry.index, entry.start_ms, entry.end_ms) for entry in tr_entries] != expected
    ):
        raise SemanticAlignmentError("Semantic delivery scope differs from canonical final blocks")

    selected_count = len(final_blocks)
    duration_limit_ms = None
    selected_id_path = id_srt_path
    selected_tr_path = tr_srt_path
    if delivery_scope == "first-hour":
        selected_count = sum(block["start_ms"] < 3_600_000 for block in final_blocks)
        if selected_count < 1:
            raise SemanticAlignmentError("First-hour semantic delivery has no canonical blocks")
        duration_limit_ms = max(3_600_000, final_blocks[selected_count - 1]["end_ms"])
        subtitle_dir = id_srt_path.parent
        selected_id_path = subtitle_dir / f"Muhtemel Ask {report['episode']}.Bolum-id.first-hour.srt"
        selected_tr_path = subtitle_dir / f"Muhtemel Ask {report['episode']}.Bolum-tr.first-hour.srt"
        write_srt(selected_id_path, id_entries[:selected_count])
        write_srt(selected_tr_path, tr_entries[:selected_count])
        assert_srt_roundtrip(selected_id_path, id_entries[:selected_count])
        assert_srt_roundtrip(selected_tr_path, tr_entries[:selected_count])

    selection = {
        "format": "mas-semantic-delivery-scope-1",
        "episode": report["episode"],
        "delivery_scope": delivery_scope,
        "alignment_policy": ALIGNMENT_POLICY,
        "finalization_sha256": _require_sha(finalization_sha256, "finalization_sha256"),
        "canonical_final_blocks_sha256": sha256_file(final_blocks_path),
        "canonical_block_count": len(final_blocks),
        "selected_block_count": selected_count,
        "selected_block_uids": [block["block_uid"] for block in final_blocks[:selected_count]],
        "duration_limit_ms": duration_limit_ms,
        "id_srt_relative_path": selected_id_path.relative_to(root).as_posix(),
        "id_srt_sha256": sha256_file(selected_id_path),
        "tr_srt_relative_path": selected_tr_path.relative_to(root).as_posix(),
        "tr_srt_sha256": sha256_file(selected_tr_path),
    }
    selection_path = (
        root / "work" / "delivery-scopes" / delivery_scope / "semantic-delivery-scope.json"
    )
    atomic_write_json(selection_path, selection)
    return {**selection, "path": selection_path}


def finalize_semantic_episode(
    *,
    episode_root,
    episode,
    source_video,
    raw_asr_path,
    word_timeline_path,
    word_timeline_manifest_path,
    quarantine_path,
    final_blocks_path,
    semantic_qa_path,
    aligned_schema_path,
    id_translation_pack,
    id_translation_zip,
    coarse_fallback_approvals_path=None,
    series_config,
    names_config,
    religious_config,
):
    from .episode_archive import file_record
    from .finalize import (
        _configuration_records,
        _entries,
        _episode_identity,
        _episode_records,
        _evidence_snapshot,
        _publish_transaction,
        _require_mux_pass,
        _source_record,
        _validate_source_chain,
        _validate_audio_chain,
        _validate_finalization_root,
    )
    from .id_translation import (
        load_and_validate_id_translation_zip,
        validate_id_translation_pack,
    )
    from .mux import mux_softsubs
    from .srt import assert_srt_roundtrip, write_srt
    from .subtitle_qa import assert_final_qa, run_subtitle_qa
    from .translation_workspace import validate_id_workspace_output
    import yaml

    root = _validate_finalization_root(episode_root, episode)
    episode_name = _episode_identity(episode)
    source, source_record = _source_record(root, source_video, episode_name)
    source_metadata, download_marker, _ = _validate_source_chain(root, source, source_record)
    audio, audio_metadata, audio_marker, _ = _validate_audio_chain(
        root, source, source_record["sha256"]
    )
    paths = {
        "raw_asr": Path(raw_asr_path),
        "word_timeline": Path(word_timeline_path),
        "word_timeline_manifest": Path(word_timeline_manifest_path),
        "quarantine": Path(quarantine_path),
        "final_blocks": Path(final_blocks_path),
        "semantic_qa": Path(semantic_qa_path),
        "aligned_schema": Path(aligned_schema_path),
        "id_translation_pack": Path(id_translation_pack),
        "id_translation_zip": Path(id_translation_zip),
    }
    if coarse_fallback_approvals_path is not None:
        paths["coarse_fallback_approvals"] = Path(coarse_fallback_approvals_path)
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise SemanticAlignmentError(f"Semantic finalization input is unsafe: {label}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise SemanticAlignmentError(f"Semantic finalization input escaped episode root: {label}")
    raw_data = json.loads(paths["raw_asr"].read_text(encoding="utf-8"))
    if raw_data.get("episode") != episode or raw_data.get("audio_sha256") != sha256_file(audio):
        raise SemanticAlignmentError("Semantic raw ASR audio/episode binding changed")
    timeline_manifest = json.loads(paths["word_timeline_manifest"].read_text(encoding="utf-8"))
    if (
        timeline_manifest.get("format") != WORD_TIMELINE_FORMAT
        or timeline_manifest.get("source_sha256") != source_record["sha256"]
        or timeline_manifest.get("timing_source_sha256") != sha256_file(paths["raw_asr"])
        or timeline_manifest.get("timeline_sha256") != sha256_file(paths["word_timeline"])
        or timeline_manifest.get("quarantine_sha256") != sha256_file(paths["quarantine"])
    ):
        raise SemanticAlignmentError("Semantic word timeline finalization binding changed")
    final_blocks = read_jsonl(paths["final_blocks"])
    semantic_qa = json.loads(paths["semantic_qa"].read_text(encoding="utf-8"))
    if (
        semantic_qa.get("format") != "mas-semantic-release-qa-1"
        or semantic_qa.get("release_eligible") is not True
        or semantic_qa.get("strict_ctc_pass") is not False
        or semantic_qa.get("word_ownership_complete") is not True
        or (
            semantic_qa.get("semantic_alignment_pass") is not True
            and not (
                semantic_qa.get("approved_coarse_fallback_count", 0) > 0
                and semantic_qa.get("approved_coarse_release_allowed") is True
            )
        )
    ):
        raise SemanticAlignmentError("Semantic release QA has not passed")
    schema = validate_aligned_turkish_schema(
        json.loads(paths["aligned_schema"].read_text(encoding="utf-8"))
    )
    if (
        schema.get("alignment_policy") != ALIGNMENT_POLICY
        or schema.get("timing_source") != TIMING_SOURCE
        or schema.get("strict_ctc_pass") is not False
        or schema.get("semantic_alignment_pass") is not semantic_qa.get("semantic_alignment_pass")
        or schema["block_count"] != len(final_blocks)
    ):
        raise SemanticAlignmentError("Semantic translation schema identity is invalid")
    for schema_block, final_block in zip(schema["blocks"], final_blocks):
        if (
            schema_block["block_uid"] != final_block.get("block_uid")
            or schema_block["block_index"] != final_block.get("block_index")
            or schema_block["start_ms"] != final_block.get("start_ms")
            or schema_block["end_ms"] != final_block.get("end_ms")
            or schema_block["tr_text"] != final_block.get("final_tr")
            or schema_block["alignment_provenance"].get("first_word_id") != final_block.get("first_word_id")
            or schema_block["alignment_provenance"].get("last_word_id") != final_block.get("last_word_id")
        ):
            raise SemanticAlignmentError("Final semantic block lock differs from translation schema")

    manifest = validate_id_translation_pack(paths["id_translation_pack"], expected_schema=schema)
    validation = load_and_validate_id_translation_zip(
        schema, paths["id_translation_zip"], input_manifest=manifest
    )
    if "production_policy" in schema:
        validate_id_workspace_output(paths["id_translation_pack"], paths["id_translation_zip"])
    records = validation.ordered_records(schema)
    if any(record.get("review_required") is True for record in records):
        raise SemanticAlignmentError("Indonesian translation still requires review")
    id_by_uid = {record["block_uid"]: record["id_final"] for record in records}
    tr_by_uid = {block["block_uid"]: block["tr_text"] for block in schema["blocks"]}

    config_paths = {
        "series_config": Path(series_config),
        "names_config": Path(names_config),
        "religious_config": Path(religious_config),
    }
    configs = []
    for path in config_paths.values():
        with path.open(encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    series, names, religious = configs
    subtitle = series["subtitle"]
    qa_blocks = [
        {**copy.deepcopy(block), "schema_sha256": schema["schema_sha256"], "primary_text": block["tr_text"]}
        for block in schema["blocks"]
    ]
    qa_records = [
        {**copy.deepcopy(record), "tr_final": block["tr_text"]}
        for block, record in zip(schema["blocks"], records)
    ]
    subtitle_qa = run_subtitle_qa(
        qa_blocks,
        qa_records,
        names_config=names,
        religious_config=religious,
        preferred_max_cps=subtitle["preferred_max_cps"],
        line_limit=subtitle["qa_max_chars_per_line"],
    )
    assert_final_qa(subtitle_qa)
    id_cps_failures = 0
    for block in schema["blocks"]:
        duration = (block["end_ms"] - block["start_ms"]) / 1000.0
        id_cps_failures += len(re.sub(r"\s+", "", id_by_uid[block["block_uid"]])) / duration > 20.0
    if id_cps_failures:
        raise SemanticAlignmentError(f"Indonesian CPS QA failed: {id_cps_failures}")

    tr_entries = _entries(
        schema["blocks"], tr_by_uid,
        target_chars_per_line=subtitle["target_chars_per_line"],
        hard_line_limit=subtitle["qa_max_chars_per_line"],
    )
    id_entries = _entries(
        schema["blocks"], id_by_uid,
        target_chars_per_line=subtitle["target_chars_per_line"],
        hard_line_limit=subtitle["qa_max_chars_per_line"],
    )
    if [
        (entry.index, entry.start_ms, entry.end_ms) for entry in tr_entries
    ] != [
        (entry.index, entry.start_ms, entry.end_ms) for entry in id_entries
    ]:
        raise SemanticAlignmentError("Semantic TR/ID timing identity changed")

    evidence_paths = {
        "source_video": source,
        "download_metadata": source_metadata,
        "download_marker": download_marker,
        "audio": audio,
        "audio_metadata": audio_metadata,
        "audio_marker": audio_marker,
        **paths,
    }
    workspace_receipt = Path(str(paths["id_translation_zip"]) + ".workspace.json")
    if "production_policy" in schema:
        evidence_paths["id_workspace_receipt"] = workspace_receipt
    before = _evidence_snapshot({**evidence_paths, **config_paths})
    input_records = _episode_records(root, evidence_paths)
    configuration_records = _configuration_records(Path(__file__).resolve().parents[3], config_paths)

    final_dir = root / "final"
    subtitle_dir = final_dir / "subtitles"
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "mkv": final_dir / f"{episode_name}.mkv",
        "id_srt": subtitle_dir / f"{episode_name}-id.srt",
        "tr_srt": subtitle_dir / f"{episode_name}-tr.srt",
    }
    report_target = final_dir / f"{episode_name}_SEMANTIC_FINALIZATION_REPORT.json"
    run_id = uuid.uuid4().hex
    staged = {
        key: path.with_name(f".{path.stem}.{run_id}.staged{path.suffix}")
        for key, path in targets.items()
    }
    try:
        write_srt(staged["tr_srt"], tr_entries)
        write_srt(staged["id_srt"], id_entries)
        assert_srt_roundtrip(staged["tr_srt"], tr_entries)
        assert_srt_roundtrip(staged["id_srt"], id_entries)
        mux_report = dict(
            mux_softsubs(
                source,
                staged["id_srt"],
                staged["tr_srt"],
                staged["mkv"],
                verify_stream_hashes=True,
            )
        )
        _require_mux_pass(mux_report, schema["block_count"])
        output_records = {
            key: {
                "relative_path": targets[key].relative_to(root).as_posix(),
                "size_bytes": staged[key].stat().st_size,
                "sha256": sha256_file(staged[key]),
            }
            for key in targets
        }
        report = {
            "report_version": 1,
            "status": "PASS",
            "mode": ALIGNMENT_POLICY,
            "episode": episode,
            "episode_name": episode_name,
            "alignment_policy": ALIGNMENT_POLICY,
            "timing_source": TIMING_SOURCE,
            "strict_ctc_pass": False,
            "semantic_alignment_pass": semantic_qa["semantic_alignment_pass"],
            "release_eligible": True,
            "approved_coarse_fallback_count": semantic_qa["approved_coarse_fallback_count"],
            "schema_version": schema["schema_version"],
            "schema_sha256": schema["schema_sha256"],
            "word_timeline_sha256": timeline_manifest["timeline_sha256"],
            "block_count": schema["block_count"],
            "input_files": input_records,
            "configuration_files": configuration_records,
            "semantic_release_qa": semantic_qa,
            "semantic_subtitle_qa": subtitle_qa,
            "id_translation_validation": {
                "passed": validation.ok,
                "expected_block_count": validation.expected_block_count,
                "output_block_count": validation.output_block_count,
                "review_required_count": 0,
            },
            "timing_identity": {
                "word_timeline_authority": True,
                "gpt_timestamp_authority": False,
                "tr_id_identical": True,
                "srt_roundtrip_exact": True,
            },
            "mkv_verification": {
                "verified": True,
                "video_audio_stream_copy": True,
                "subtitle_order": ["ind", "tur"],
                "indonesian_default": True,
                "turkish_default": False,
                "roundtrip": copy.deepcopy(mux_report["roundtrip"]),
                "stream_hashes": copy.deepcopy(mux_report["stream_hashes"]),
                "output_path": output_records["mkv"]["relative_path"],
                "output_size_bytes": output_records["mkv"]["size_bytes"],
            },
            "outputs": output_records,
        }
        _publish_transaction(
            staged,
            targets,
            report_target=report_target,
            report=report,
            episode_root=root,
            expected_output_records=output_records,
            evidence_paths={**evidence_paths, **config_paths},
            expected_evidence_snapshot=before,
            run_id=run_id,
        )
        if {key: file_record(path, root) for key, path in targets.items()} != output_records:
            raise SemanticAlignmentError("Published semantic artifacts changed")
        return report
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


__all__ = [
    "ALIGNMENT_POLICY",
    "SemanticAlignmentConfig",
    "SemanticAlignmentError",
    "TIMING_SOURCE",
    "WORD_TIMELINE_FORMAT",
    "build_semantic_translation_schema",
    "build_semantic_windows",
    "build_word_timeline",
    "deterministic_exact_results",
    "evidence_sources",
    "finalize_semantic_blocks",
    "finalize_semantic_episode",
    "load_word_timeline",
    "prepare_semantic_delivery_scope",
    "read_jsonl",
    "semantic_producer_identity",
    "semantic_run_contract",
    "sign_coarse_fallback_approval",
    "validate_coarse_fallback_approvals",
    "write_jsonl",
]
