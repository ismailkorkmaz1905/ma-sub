from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from ..reliability import digest


FORMAT = "mas-speaker-evidence-v1"
VERSION = 1
POLICY = {
    "active_probability": 0.5,
    "minimum_dominant_frames": 5,
    "minimum_dominant_fraction": 0.8,
    "minimum_dominant_mean_probability": 0.8,
    "minimum_peak_probability": 0.9,
}


class SpeakerEvidenceError(ValueError):
    pass


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpeakerEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpeakerEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _pilot_data(
    wrapped: Mapping[str, Any],
    *,
    episode: int,
    clip_id: str,
) -> dict[str, Any]:
    if not isinstance(wrapped, Mapping) or set(wrapped) != {"data", "sha256"}:
        raise SpeakerEvidenceError(f"{clip_id} pilot wrapper is invalid")
    data = wrapped.get("data")
    if not isinstance(data, Mapping) or wrapped.get("sha256") != digest(data):
        raise SpeakerEvidenceError(f"{clip_id} pilot checksum mismatch")
    if (
        data.get("format") != "mas-native-diarization-pilot-1"
        or data.get("status") != "REVIEW_REQUIRED"
        or data.get("production_acceptance") is not False
        or data.get("episode") != episode
        or data.get("speaker_columns") != 4
        or data.get("sample_rate") != 16000
        or not math.isclose(data.get("frame_seconds", -1), 0.08, abs_tol=1e-6)
    ):
        raise SpeakerEvidenceError(f"{clip_id} pilot contract mismatch")
    _sha256(data.get("input_sha256"), f"{clip_id} input_sha256")
    _sha256(data.get("model_sha256"), f"{clip_id} model_sha256")
    probabilities = data.get("probabilities")
    if (
        not isinstance(probabilities, list)
        or not probabilities
        or data.get("frame_count") != len(probabilities)
    ):
        raise SpeakerEvidenceError(f"{clip_id} probabilities are invalid")
    for row in probabilities:
        if (
            not isinstance(row, list)
            or len(row) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in row
            )
        ):
            raise SpeakerEvidenceError(f"{clip_id} probabilities are invalid")
    return copy.deepcopy(dict(data))


def _assignment(
    record: Mapping[str, Any],
    clip: Mapping[str, Any],
) -> dict[str, Any] | None:
    start_ms = _integer(record.get("coarse_start_ms"), "coarse_start_ms")
    end_ms = _integer(record.get("coarse_end_ms"), "coarse_end_ms", minimum=1)
    if end_ms <= start_ms:
        raise SpeakerEvidenceError("correction input bounds are invalid")
    clip_start = int(clip["start_ms"])
    pilot = clip["pilot"]["data"]
    frame_ms = float(pilot["frame_seconds"]) * 1000
    active = []
    for index, row in enumerate(pilot["probabilities"]):
        center_ms = clip_start + (index + 0.5) * frame_ms
        if start_ms <= center_ms < end_ms and max(row) >= POLICY["active_probability"]:
            active.append(row)
    if not active:
        return None
    counts = Counter(max(range(4), key=row.__getitem__) for row in active)
    dominant_column, dominant_frames = counts.most_common(1)[0]
    dominant_fraction = dominant_frames / len(active)
    dominant_values = [row[dominant_column] for row in active]
    dominant_mean = sum(dominant_values) / len(dominant_values)
    peak = max(dominant_values)
    if (
        dominant_frames < POLICY["minimum_dominant_frames"]
        or dominant_fraction < POLICY["minimum_dominant_fraction"]
        or dominant_mean < POLICY["minimum_dominant_mean_probability"]
        or peak < POLICY["minimum_peak_probability"]
    ):
        return None
    uid = record.get("utterance_uid")
    if not isinstance(uid, str) or not uid:
        raise SpeakerEvidenceError("correction input utterance_uid is invalid")
    clip_id = str(clip["clip_id"])
    input_sha = str(pilot["input_sha256"])
    return {
        "utterance_uid": uid,
        "coarse_start_ms": start_ms,
        "coarse_end_ms": end_ms,
        "clip_id": clip_id,
        "pilot_sha256": clip["pilot"]["sha256"],
        "speaker_column": dominant_column + 1,
        "speaker_id": (
            f"nemo-sortformer-{clip_id}-{input_sha[:12]}-column-{dominant_column + 1}"
        ),
        "active_frames": len(active),
        "dominant_frames": dominant_frames,
        "dominant_fraction": round(dominant_fraction, 6),
        "dominant_mean_probability": round(dominant_mean, 6),
        "peak_probability": round(peak, 6),
    }


def _build_assignments(
    input_utterances: Sequence[Mapping[str, Any]],
    clips: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assignments = []
    seen_uids: set[str] = set()
    for record in input_utterances:
        uid = record.get("utterance_uid")
        if not isinstance(uid, str) or not uid or uid in seen_uids:
            raise SpeakerEvidenceError("correction input utterance_uid is invalid or duplicated")
        seen_uids.add(uid)
        start_ms = _integer(record.get("coarse_start_ms"), f"{uid} coarse_start_ms")
        end_ms = _integer(record.get("coarse_end_ms"), f"{uid} coarse_end_ms", minimum=1)
        containing = [
            clip
            for clip in clips
            if int(clip["start_ms"]) <= start_ms and end_ms <= int(clip["end_ms"])
        ]
        if len(containing) > 1:
            raise SpeakerEvidenceError(f"{uid} is contained by multiple evidence clips")
        if containing:
            assignment = _assignment(record, containing[0])
            if assignment is not None:
                assignments.append(assignment)
    return sorted(assignments, key=lambda item: item["utterance_uid"])


def build_speaker_evidence(
    *,
    episode: int,
    audio_sha256: str,
    input_utterances: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    pilot_wrappers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _integer(episode, "episode", minimum=1)
    _sha256(audio_sha256, "audio_sha256")
    if len(intervals) != len(pilot_wrappers) or not intervals:
        raise SpeakerEvidenceError("evidence intervals and pilots must be non-empty and equal")
    clips = []
    previous_end = -1
    model_sha256 = None
    runtime = None
    for index, (interval, wrapped) in enumerate(zip(intervals, pilot_wrappers), 1):
        start_ms = _integer(interval.get("start"), f"clip {index} start_ms")
        end_ms = _integer(interval.get("end"), f"clip {index} end_ms", minimum=1)
        if end_ms <= start_ms or start_ms < previous_end or end_ms - start_ms > 120000:
            raise SpeakerEvidenceError(f"clip {index} bounds are invalid")
        previous_end = end_ms
        clip_id = f"clip-{index:03d}"
        pilot = _pilot_data(wrapped, episode=episode, clip_id=clip_id)
        expected_identity = f"ep{episode}-residual-{index:03d}-{start_ms}-{end_ms}"
        if pilot.get("clip_identity") != expected_identity:
            raise SpeakerEvidenceError(f"{clip_id} identity mismatch")
        expected_frames = math.ceil((end_ms - start_ms) / 80)
        if abs(int(pilot["frame_count"]) - expected_frames) > 1:
            raise SpeakerEvidenceError(f"{clip_id} duration does not match its frame count")
        if model_sha256 is None:
            model_sha256 = pilot["model_sha256"]
            runtime = pilot.get("runtime")
        elif pilot["model_sha256"] != model_sha256 or pilot.get("runtime") != runtime:
            raise SpeakerEvidenceError("evidence clips used different model or runtime identities")
        clips.append(
            {
                "clip_id": clip_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "pilot": copy.deepcopy(dict(wrapped)),
            }
        )
    assignments = _build_assignments(input_utterances, clips)
    data = {
        "format": FORMAT,
        "version": VERSION,
        "episode": episode,
        "audio_sha256": audio_sha256,
        "model_sha256": model_sha256,
        "runtime_sha256": digest(runtime),
        "policy": dict(POLICY),
        "clips": clips,
        "assignments": assignments,
    }
    return {"data": data, "sha256": digest(data)}


def validate_speaker_evidence(
    evidence: Mapping[str, Any],
    *,
    episode: int,
    audio_sha256: str,
    input_utterances: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not isinstance(evidence, Mapping) or set(evidence) != {"data", "sha256"}:
        raise SpeakerEvidenceError("speaker evidence wrapper is invalid")
    data = evidence.get("data")
    if not isinstance(data, Mapping) or evidence.get("sha256") != digest(data):
        raise SpeakerEvidenceError("speaker evidence checksum mismatch")
    if (
        data.get("format") != FORMAT
        or data.get("version") != VERSION
        or data.get("episode") != episode
        or data.get("audio_sha256") != audio_sha256
        or data.get("policy") != POLICY
    ):
        raise SpeakerEvidenceError("speaker evidence identity or policy mismatch")
    rebuilt = build_speaker_evidence(
        episode=episode,
        audio_sha256=audio_sha256,
        input_utterances=input_utterances,
        intervals=[
            {"start": clip.get("start_ms"), "end": clip.get("end_ms")}
            for clip in data.get("clips", [])
            if isinstance(clip, Mapping)
        ],
        pilot_wrappers=[
            clip.get("pilot")
            for clip in data.get("clips", [])
            if isinstance(clip, Mapping)
        ],
    )
    if rebuilt != evidence:
        raise SpeakerEvidenceError("speaker evidence is not reproducible from pilot frames")
    return {
        assignment["utterance_uid"]: assignment["speaker_id"]
        for assignment in data["assignments"]
    }


__all__ = [
    "FORMAT",
    "POLICY",
    "SpeakerEvidenceError",
    "build_speaker_evidence",
    "validate_speaker_evidence",
]
