import copy
import math
import re
import textwrap
from pathlib import Path

from ..engine.download import atomic_write_bytes
from ..reliability import IntegrityError, atomic_json, digest
from .srt import render


FORMAT = "mas-cue-pilot-1"


def _milliseconds(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise IntegrityError("timestamp must be a finite number of seconds")
    if not math.isfinite(value) or value < 0:
        raise IntegrityError("timestamp must be finite and nonnegative")
    return math.ceil(value * 1000)


def _interval(record, duration_ms):
    start, end = _milliseconds(record["start"]), _milliseconds(record["end"])
    if end <= start or end > duration_ms:
        raise IntegrityError("invalid or out-of-clip interval")
    return start, end


def _speaker(start, end, turns):
    coverage = {}
    for turn in turns:
        overlap = max(0, min(end, turn["end_ms"]) - max(start, turn["start_ms"]))
        if overlap:
            coverage.setdefault(turn["speaker"], []).append(
                (max(start, turn["start_ms"]), min(end, turn["end_ms"])))
    if len(coverage) != 1:
        return None
    speaker, intervals = next(iter(coverage.items()))
    total, cursor = 0, start
    for left, right in sorted(intervals):
        total += max(0, right - max(left, cursor))
        cursor = max(cursor, right)
    return speaker if total >= 0.8 * (end - start) else None


def speaker_groups(segments, speaker_turns, duration_ms):
    turns = []
    for item in speaker_turns:
        start, end = _interval(item, duration_ms)
        label = item.get("speaker")
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise IntegrityError("invalid speaker label")
        turns.append({"start_ms": start, "end_ms": end, "speaker": label})
    groups, issues = [], []
    for segment_index, segment in enumerate(segments):
        _interval(segment, duration_ms)
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise IntegrityError("empty transcript segment")
        words = segment.get("words")
        if not words:
            words = [{"word": text, "start": segment["start"], "end": segment["end"]}]
            issues.append({"kind": "segment_timing_only", "segment_index": segment_index})
        joined = "".join(word["word"] for word in words)
        if " ".join(joined.split()) != " ".join(text.split()):
            raise IntegrityError("word text differs from segment text; do not drop dialogue")
        current = None
        previous_start = -1
        for word_index, word in enumerate(words):
            if not isinstance(word.get("word"), str) or not word["word"].strip():
                raise IntegrityError("empty word")
            start, end = _interval(word, duration_ms)
            if start < previous_start:
                raise IntegrityError("words are not ordered within segment")
            previous_start = start
            speaker = _speaker(start, end, turns)
            if speaker is None:
                issues.append({"kind": "speaker_uncertain", "segment_index": segment_index,
                               "word_index": word_index})
            if current is None or current["speaker"] != speaker:
                current = {"speaker": speaker, "segment_index": segment_index, "words": []}
                groups.append(current)
            current["words"].append(copy.deepcopy(word))
    return groups, issues


def _wrap(text):
    return textwrap.wrap(" ".join(text.split()), width=42,
                         break_long_words=False, break_on_hyphens=False)


def build_pilot(evidence, *, regroup=None):
    body = evidence.get("data")
    if not isinstance(body, dict) or evidence.get("sha256") != digest(body):
        raise IntegrityError("pilot evidence checksum mismatch")
    if body.get("format") != "mas-acoustic-pilot-1":
        raise IntegrityError("unsupported pilot evidence format")
    for key in ("audio_sha256", "source_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(body.get(key, ""))):
            raise IntegrityError(f"invalid {key}")
    for key in ("episode", "duration_ms", "offset_ms"):
        value = body.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "offset_ms" else 1):
            raise IntegrityError(f"invalid {key}")
    if body["duration_ms"] > 120_000:
        raise IntegrityError("pilot clip exceeds 120 seconds")
    groups, issues = speaker_groups(body["segments"], body["speaker_turns"], body["duration_ms"])
    cues = []
    for group in groups:
        pieces = regroup(copy.deepcopy(group["words"])) if regroup else [group["words"]]
        flattened = [word for piece in pieces for word in piece]
        identity = lambda word: (word["word"], word["start"], word["end"])
        if [identity(word) for word in flattened] != [identity(word) for word in group["words"]]:
            raise IntegrityError("regrouping changed words or their timing")
        for piece in pieces:
            if not piece:
                raise IntegrityError("empty regrouped cue")
            start = min(_milliseconds(word["start"]) for word in piece) + body["offset_ms"]
            end = max(_milliseconds(word["end"]) for word in piece) + body["offset_ms"]
            text = "".join(word["word"] for word in piece).strip()
            cue = {"cue_id": f"cue-{len(cues) + 1:05d}", "start_ms": start, "end_ms": end,
                   "text": text, "speaker": group["speaker"], "segment_index": group["segment_index"]}
            cues.append(cue)
            if end - start > 7000 or end - start < 700:
                issues.append({"kind": "cue_duration_review", "cue_id": cue["cue_id"]})
            if len(text) / ((end - start) / 1000) > 20:
                issues.append({"kind": "reading_speed_review", "cue_id": cue["cue_id"]})
    events = {}
    for cue in cues:
        events.setdefault(cue["start_ms"], []).append((True, cue))
        events.setdefault(cue["end_ms"], []).append((False, cue))
    active, rendered = {}, []
    boundaries = sorted(events)
    for index, start in enumerate(boundaries[:-1]):
        for entering, cue in events[start]:
            if entering:
                active[cue["cue_id"]] = cue
            else:
                active.pop(cue["cue_id"], None)
        if not active:
            continue
        present = sorted(active.values(), key=lambda item: item["cue_id"])
        if len(present) > 1:
            speakers = [cue["speaker"] for cue in present]
            if None in speakers or len(set(speakers)) != len(speakers):
                issues.append({"kind": "unresolved_overlap", "start_ms": start})
            lines = ["-" + " ".join(cue["text"].split()) for cue in present]
        else:
            lines = _wrap(present[0]["text"])
        if len(lines) > 2 or any(len(line) > 42 for line in lines):
            issues.append({"kind": "line_length_review", "start_ms": start})
        rendered.append({"start_ms": start, "end_ms": boundaries[index + 1],
                         "text": "\n".join(lines),
                         "source_cue_ids": [cue["cue_id"] for cue in present]})
    result = {"format": FORMAT, "mode": "pilot", "episode": body["episode"],
              "timing_source": "stable_ts_estimated", "evidence_sha256": evidence["sha256"],
              "source_sha256": body["source_sha256"], "audio_sha256": body["audio_sha256"],
              "offset_ms": body["offset_ms"], "cues": cues, "rendered": rendered,
              "issues": issues, "status": "REVIEW_REQUIRED",
              "acoustic_acceptance": "NOT_VERIFIED", "strict_delivery": False}
    return {"data": result, "sha256": digest(result)}


def write_pilot(output_dir, result):
    output_dir = Path(output_dir)
    if result.get("sha256") != digest(result.get("data")):
        raise IntegrityError("pilot result checksum mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pilot.json", "draft.srt"):
        if (output_dir / name).exists():
            raise IntegrityError("pilot output exists; preserve it and choose a new directory")
    atomic_write_bytes(output_dir / "draft.srt", render(result["data"]["rendered"], "text").encode("utf-8"))
    atomic_json(output_dir / "pilot.json", result)
