import copy

import pytest

from mas.engine.speaker_evidence import (
    POLICY,
    SpeakerEvidenceError,
    build_speaker_evidence,
    validate_speaker_evidence,
)
from mas.reliability import digest


def _pilot(probabilities):
    data = {
        "format": "mas-native-diarization-pilot-1",
        "status": "REVIEW_REQUIRED",
        "production_acceptance": False,
        "episode": 15,
        "clip_identity": "ep15-residual-001-0-3000",
        "input_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "runtime": [{"name": "runtime.dll", "bytes": 1, "sha256": "c" * 64}],
        "sample_rate": 16000,
        "frame_seconds": 0.08,
        "frame_count": len(probabilities),
        "speaker_columns": 4,
        "probabilities": probabilities,
    }
    return {"data": data, "sha256": digest(data)}


def _fixture():
    records = [
        {"utterance_uid": "utt-a", "coarse_start_ms": 400, "coarse_end_ms": 960},
        {"utterance_uid": "utt-b", "coarse_start_ms": 1200, "coarse_end_ms": 1760},
    ]
    probabilities = [[0.0, 0.0, 0.0, 0.0] for _ in range(38)]
    for index in range(5, 12):
        probabilities[index] = [0.98, 0.01, 0.0, 0.0]
    for index in range(15, 22):
        probabilities[index] = [0.01, 0.97, 0.0, 0.0]
    evidence = build_speaker_evidence(
        episode=15,
        audio_sha256="d" * 64,
        input_utterances=records,
        intervals=[{"start": 0, "end": 3000}],
        pilot_wrappers=[_pilot(probabilities)],
    )
    return records, evidence


def test_strict_pilot_frames_create_reproducible_acoustic_lanes():
    records, evidence = _fixture()

    assignments = validate_speaker_evidence(
        evidence,
        episode=15,
        audio_sha256="d" * 64,
        input_utterances=records,
    )

    assert evidence["data"]["policy"] == POLICY
    assert assignments["utt-a"].endswith("column-1")
    assert assignments["utt-b"].endswith("column-2")


def test_assignment_cannot_be_forged_even_with_recomputed_wrapper_hash():
    records, evidence = _fixture()
    forged = copy.deepcopy(evidence)
    forged["data"]["assignments"][0]["speaker_id"] = "invented-speaker"
    forged["sha256"] = digest(forged["data"])

    with pytest.raises(SpeakerEvidenceError, match="not reproducible"):
        validate_speaker_evidence(
            forged,
            episode=15,
            audio_sha256="d" * 64,
            input_utterances=records,
        )


def test_ambiguous_or_short_activity_does_not_create_a_speaker_id():
    records = [{"utterance_uid": "utt-a", "coarse_start_ms": 400, "coarse_end_ms": 960}]
    probabilities = [[0.0, 0.0, 0.0, 0.0] for _ in range(38)]
    for index in range(5, 9):
        probabilities[index] = [0.99, 0.01, 0.0, 0.0]
    evidence = build_speaker_evidence(
        episode=15,
        audio_sha256="d" * 64,
        input_utterances=records,
        intervals=[{"start": 0, "end": 3000}],
        pilot_wrappers=[_pilot(probabilities)],
    )

    assert evidence["data"]["assignments"] == []
