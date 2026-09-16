from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath

from ..reliability import IntegrityError, digest
from .speech_coverage import V2_BETA_SPEECH_COVERAGE_CONFIG


PART_PLAN_FORMAT = "mas-episode-parts-1"
PART_AUDIO_FORMAT = "mas-derived-audio-part-1"
SAMPLE_RATE = 16000
TARGET_DURATION_MS = 3600000
MIN_BOUNDARY_GAP_MS = 1000
DELIVERY_BOUNDARY_POLICY = "delivery-bounded-v1"
MAX_BOUNDARY_DELAY_MS = 90000


class PartScopeError(IntegrityError):
    pass


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PartScopeError(f"{label} must be an integer >= {minimum}")
    return value


def validate_file_record(record):
    if not isinstance(record, dict):
        raise PartScopeError("artifact record must be an object")
    value = record.get("relative_path")
    if (not isinstance(value, str) or not value or "\\" in value
            or ":" in value or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise PartScopeError("artifact path must remain relative to the episode root")
    if not isinstance(record.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", record["sha256"]
    ):
        raise PartScopeError("artifact SHA-256 is invalid")
    _integer(record.get("size_bytes"), "artifact size_bytes", 1)
    return record


def _inventory(plan):
    _integer(plan.get("episode"), "episode", 1)
    validate_file_record(plan.get("source"))
    audio = validate_file_record(plan.get("audio"))
    if audio.get("sample_rate_hz") != SAMPLE_RATE or audio.get("channels") != 1:
        raise PartScopeError("part audio must be mono 16000 Hz")
    samples = _integer(audio.get("sample_count"), "audio sample_count", 1)
    duration_ms = (samples + 15) // 16
    vad = plan.get("vad")
    if (not isinstance(vad, dict) or vad.get("independent_vad") is not True
            or vad.get("audio_sha256") != audio["sha256"]
            or vad.get("sample_count") != samples
            or not isinstance(vad.get("config"), dict) or not vad["config"]
            or not isinstance(vad.get("model"), dict) or not vad["model"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(vad.get("producer_sha256", "")))):
        raise PartScopeError("part plan requires bound independent VAD evidence")
    regions = vad.get("regions")
    if not isinstance(regions, list) or (not regions and plan.get("boundary_policy") != DELIVERY_BOUNDARY_POLICY):
        raise PartScopeError("part plan requires complete nonempty VAD inventory")
    captions = plan.get("captions")
    if captions is not None:
        validate_file_record(captions)
    caption_records = captions.get("records") if captions is not None else []
    if not isinstance(caption_records, list):
        raise PartScopeError("caption records must be a list")
    intervals = []
    for records, index_key in ((regions, "vad_region_index"), (caption_records, "caption_index")):
        previous_start = -1
        for position, item in enumerate(records, 1):
            if not isinstance(item, dict) or item.get(index_key) != position:
                raise PartScopeError(f"{index_key} inventory must be complete and ordered")
            _integer(item[index_key], index_key, 1)
            start = _integer(item.get("start_ms"), f"{index_key} start_ms")
            end = _integer(item.get("end_ms"), f"{index_key} end_ms", 1)
            if start < previous_start or end <= start or end > duration_ms:
                raise PartScopeError(f"{index_key} has invalid source bounds")
            if index_key == "vad_region_index" and item.get("source") != "silero_vad":
                raise PartScopeError("boundary VAD cannot use ASR-derived speech")
            if index_key == "caption_index" and not str(item.get("text", "")).strip():
                raise PartScopeError("caption text cannot be empty")
            previous_start = start
            intervals.append((start, end))
    return samples, regions, caption_records, intervals


def _planned_parts(plan):
    samples, regions, captions, intervals = _inventory(plan)
    if plan.get("target_duration_ms") != TARGET_DURATION_MS:
        raise PartScopeError("part target must remain 3600000 ms")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [(left[1], right[0]) for left, right in zip(merged, merged[1:])
            if right[0] - left[1] >= MIN_BOUNDARY_GAP_MS]
    cuts = [0]
    forced_cuts = set()
    while samples - cuts[-1] > TARGET_DURATION_MS * 16:
        target_ms = cuts[-1] // 16 + TARGET_DURATION_MS
        right_guard_ms = V2_BETA_SPEECH_COVERAGE_CONFIG.max_word_outside_speech_ms
        candidates = [right - right_guard_ms for left, right in gaps
                      if right - right_guard_ms >= target_ms]
        if plan.get("boundary_policy") == DELIVERY_BOUNDARY_POLICY and (
                not candidates or min(candidates) > target_ms + MAX_BOUNDARY_DELAY_MS):
            cut = target_ms * 16
            forced_cuts.add(cut)
        else:
            if not candidates:
                if len(cuts) > 1:
                    break
                raise PartScopeError(
                    f"no proven speech/caption gap at or after {target_ms} ms; "
                    "refusing to cut dialogue or silently make one full-episode part"
                )
            cut = min(candidates) * 16
        if cut >= samples:
            raise PartScopeError("part boundary reaches the source end")
        cuts.append(cut)
    cuts.append(samples)
    parts = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:]), 1):
        if index > 64:
            raise PartScopeError("episode part count exceeds hard maximum 64")
        start_ms, end_ms = start // 16, (end + 15) // 16
        parts.append({
            "part_id": f"part-{index:03d}", "part_index": index,
            "start_ms": start_ms, "end_ms": end_ms,
            "start_sample": start, "end_sample": end,
            **({"boundary_warning": "bounded_cut_without_proven_silence"}
               if start in forced_cuts or end in forced_cuts else {}),
            "vad_region_indices": [item["vad_region_index"] for item in regions
                                   if (item["end_ms"] > start_ms and item["start_ms"] < end_ms
                                       if plan.get("boundary_policy") == DELIVERY_BOUNDARY_POLICY else
                                       start_ms <= item["start_ms"] and item["end_ms"] <= end_ms)],
            "caption_indices": [item["caption_index"] for item in captions
                                if (item["end_ms"] > start_ms and item["start_ms"] < end_ms
                                    if plan.get("boundary_policy") == DELIVERY_BOUNDARY_POLICY else
                                    start_ms <= item["start_ms"] and item["end_ms"] <= end_ms)],
        })
    return parts


def build_part_plan(*, episode, source, audio, vad, captions=None, boundary_policy=None):
    plan = copy.deepcopy({
        "format": PART_PLAN_FORMAT, "episode": episode, "source": source,
        "audio": audio, "vad": vad, "captions": captions,
        "target_duration_ms": TARGET_DURATION_MS,
    })
    if boundary_policy is not None:
        if boundary_policy != DELIVERY_BOUNDARY_POLICY or type(episode) is not int or episode < 14:
            raise PartScopeError("unsupported delivery boundary policy")
        plan['boundary_policy'] = boundary_policy
    plan["parts"] = _planned_parts(plan)
    return plan


def validate_part_plan(plan):
    required = {"format", "episode", "source", "audio", "vad", "captions", "target_duration_ms", "parts"}
    if isinstance(plan, dict) and 'boundary_policy' in plan:
        if (plan['boundary_policy'] != DELIVERY_BOUNDARY_POLICY
                or type(plan.get('episode')) is not int or plan['episode'] < 14):
            raise PartScopeError("unsupported delivery boundary policy")
        required.add('boundary_policy')
    if not isinstance(plan, dict) or set(plan) != required or plan.get("format") != PART_PLAN_FORMAT:
        raise PartScopeError("invalid part plan contract")
    if not isinstance(plan.get("parts"), list) or not 1 <= len(plan["parts"]) <= 64:
        raise PartScopeError("part plan requires between 1 and 64 parts")
    for part in plan["parts"]:
        if not isinstance(part, dict):
            raise PartScopeError("part record must be an object")
        for key in ("part_index", "start_ms", "end_ms", "start_sample", "end_sample"):
            _integer(part.get(key), key)
    if plan.get("parts") != _planned_parts(plan):
        raise PartScopeError("part plan omits, overlaps, or changes an exact source/evidence range")
    return copy.deepcopy(plan)


def get_part(plan, part_id):
    validate_part_plan(plan)
    for part in plan["parts"]:
        if part["part_id"] == part_id:
            return copy.deepcopy(part)
    raise PartScopeError(f"unknown part_id: {part_id!r}")


def project_part_vad(plan, part_id):
    part = get_part(plan, part_id)
    return [{**copy.deepcopy(region),
             "start_ms": max(region["start_ms"], part["start_ms"]) - part["start_ms"],
             "end_ms": min(region["end_ms"], part["end_ms"]) - part["start_ms"]}
            for region in plan["vad"]["regions"]
            if region["vad_region_index"] in part["vad_region_indices"]]


def validate_part_lineage(plan, part_id, lineage):
    part = get_part(plan, part_id)
    if not isinstance(lineage, dict):
        raise PartScopeError("derived audio lineage must be an object")
    for key in ("episode", "start_sample", "end_sample", "start_ms", "end_ms", "sample_rate_hz", "sample_count"):
        _integer(lineage.get(key), key)
    expected = {
        "format": PART_AUDIO_FORMAT, "episode": plan["episode"], "part_id": part_id,
        "plan_sha256": digest(plan), "parent_source_sha256": plan["source"]["sha256"],
        "parent_audio_sha256": plan["audio"]["sha256"],
        "start_sample": part["start_sample"], "end_sample": part["end_sample"],
        "start_ms": part["start_ms"], "end_ms": part["end_ms"],
        "sample_rate_hz": SAMPLE_RATE,
        "sample_count": part["end_sample"] - part["start_sample"],
        "parent_vad_sha256": digest(project_part_vad(plan, part_id)),
    }
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise PartScopeError("derived audio lineage differs from its exact parent part")
    audio = validate_file_record(lineage.get("audio"))
    if audio["relative_path"] != f"parts/{part_id}/prepare/audio.flac":
        raise PartScopeError("derived audio path belongs to another part")
    if lineage.get("captions") is not None:
        caption = validate_file_record(lineage["captions"])
        if caption["relative_path"] != f"parts/{part_id}/prepare/captions.vtt":
            raise PartScopeError("derived captions path belongs to another part")
    if (plan["captions"] is None) != (lineage.get("captions") is None):
        raise PartScopeError("derived caption inventory is missing")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(lineage.get("producer_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(lineage.get("pcm_sha256", "")))
            or not isinstance(lineage.get("ffmpeg_identity"), dict)
            or not lineage["ffmpeg_identity"]):
        raise PartScopeError("derived audio has no producer evidence")
    return copy.deepcopy(lineage)
