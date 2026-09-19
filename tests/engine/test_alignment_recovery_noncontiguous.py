"""Closure authorization must bind target UIDs, model text and audio window.

Synthetic fixtures only.  The model/hardware boundary is faked; the resolver,
the authorization wrapper, the closure planner, the raw journal and the
persisted call budget are the real production code.  Nothing here is acoustic
evidence for any episode.
"""

from __future__ import annotations

import copy
import json

import pytest

import mas.engine.forced_align as fa
from mas.engine.alignment_recovery import (
    RecoveryCallBudget, RecoveryPlanError, build_recovery_plan,
    request_matches_plan,
)
from mas.reliability import digest


def _result(words):
    out = copy.deepcopy(words)
    for word in out:
        word.setdefault("score", 0.90)
    return {"segments": [{"text": "unused", "words": out}], "word_segments": out}


# utt-alpha and utt-charlie overlap acoustically; utt-bravo sits between them in
# source order but its word lands in a clean gap, so the word-overlap component
# is {alpha, charlie} and is NOT contiguous in source order.
_WORDS = {
    "Alpha": [("Alpha", 2.500, 3.000)],
    "Bravo": [("Bravo", 3.600, 4.000)],
    "Charlie": [("Charlie", 2.700, 3.100)],
    "Alpha Charlie": [("Alpha", 2.500, 2.900), ("Charlie", 3.000, 3.400)],
}


class _NonContiguousOverlapWhisperX:
    """Only the model boundary is faked."""

    __version__ = "3.8.6"

    def __init__(self):
        self.align_calls: list[str] = []

    def load_audio(self, path):
        return object()

    def load_align_model(self, *, language_code, device, model_name):
        return object(), {"language": language_code, "type": "huggingface"}

    def align(self, transcript, model, metadata, audio, device,
              interpolate_method="nearest", return_char_alignments=False,
              print_progress=False):
        text = transcript[0]["text"]
        self.align_calls.append(text)
        return _result([{"word": word, "start": start, "end": end}
                        for word, start, end in _WORDS[text]])


def _coarse():
    return [
        {"utterance_uid": "utt-alpha", "start_ms": 1000, "end_ms": 3500,
         "coarse_start_ms": 1000, "coarse_end_ms": 3000,
         "text": "Alpha", "asr_text": "Alpha",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
        {"utterance_uid": "utt-bravo", "start_ms": 2000, "end_ms": 4500,
         "coarse_start_ms": 2000, "coarse_end_ms": 4200,
         "text": "Bravo", "asr_text": "Bravo",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
        {"utterance_uid": "utt-charlie", "start_ms": 2600, "end_ms": 3700,
         "coarse_start_ms": 2600, "coarse_end_ms": 3200,
         "text": "Charlie", "asr_text": "Charlie",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
    ]


_TARGETS = ["utt-alpha", "utt-charlie"]
_IDENTITY_KEYS = ("audio_sha256", "model_state_sha256",
                  "raw_alignment_binding", "source_sha256")


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.flac"
    path.write_bytes(b"synthetic-audio-placeholder")
    return path


@pytest.fixture(autouse=True)
def _stub_model_state(monkeypatch):
    # Hardware boundary only: real torch weights are unavailable offline.
    monkeypatch.setattr(fa, "_model_state_sha256", lambda model, metadata: "a" * 64)


def _plan_scope(checkpoint, source, *, calls=8):
    identity = json.loads(
        (checkpoint / "resume-identity.json").read_text(encoding="utf-8"))["data"]
    scope = {key: identity[key] for key in ("stage", *_IDENTITY_KEYS)}
    scope["target_uids"] = list(_TARGETS)
    scope["context_uids"] = fa._alignment_resume_context_uids(source, _TARGETS)
    scope["recovery_plan"] = build_recovery_plan(
        source, _TARGETS, max_new_ctc_calls=calls)
    return scope


def _forget_one_raw_result(checkpoint, spelled_words):
    """Drop exactly one retained raw CTC result; keep every other checkpoint."""
    dropped = surviving = None
    for path in sorted(checkpoint.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("data", {}).get("result")
        if not isinstance(result, dict):
            continue
        words = [word["word"] for word in result.get("word_segments", [])]
        if words == spelled_words:
            dropped = path
        elif surviving is None:
            surviving = record
    assert dropped is not None and surviving is not None
    dropped.unlink()
    return surviving


def _retain_conflict_receipt(checkpoint, source, surviving, targets):
    targeted = [item for item in source
                if str(item["utterance_uid"]) in set(targets)]
    details = {
        "status": "BLOCKED", "reason": "recovery_time_budget",
        "component_uids": list(targets),
        "source_sha256": digest(targeted),
        "raw_results": {surviving["data"]["uid"]:
                        digest(surviving["data"]["result"])},
        "total_recovery_seconds": 900.1,
        "limits": {"total_recovery_seconds": 900},
    }
    fa.atomic_json(checkpoint / "components/latest-conflict-failure.json",
                   {"data": details, "sha256": digest(details)})


# --------------------------------------------------------------------------- B1

def test_resolver_really_asks_for_a_non_contiguous_component(audio, tmp_path):
    module = _NonContiguousOverlapWhisperX()
    fa.align_corrected_segments(audio, _coarse(), whisperx_module=module,
                                checkpoint_dir=tmp_path / "units")
    assert "Alpha Charlie" in module.align_calls


def test_closure_authorizes_the_non_contiguous_joint_request(audio, tmp_path):
    checkpoint = tmp_path / "units"
    fa.align_corrected_segments(audio, _coarse(),
                                whisperx_module=_NonContiguousOverlapWhisperX(),
                                checkpoint_dir=checkpoint)
    source = fa.validate_coarse_segments(_coarse())
    scope = _plan_scope(checkpoint, source)
    # The closure is already maximal: every UID of the region is authorized.
    assert scope["recovery_plan"]["components"] == [
        ["utt-alpha", "utt-bravo", "utt-charlie"]]
    joint = [{"start": 1.0, "end": 3.7, "text": "Alpha Charlie"}]
    assert request_matches_plan(joint, _TARGETS, source, scope["recovery_plan"])
    assert fa._alignment_request_matches_scope(
        joint, _TARGETS, source, scope,
        {key: scope[key] for key in _IDENTITY_KEYS})


def test_warm_resume_recomputes_only_the_forgotten_joint_call(
        audio, tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_RAW_ASR_AUTH_KEY", "2" * 64)
    checkpoint = tmp_path / "units"
    cold_module = _NonContiguousOverlapWhisperX()
    cold = fa.align_corrected_segments(audio, _coarse(),
                                       whisperx_module=cold_module,
                                       checkpoint_dir=checkpoint)
    assert len(cold_module.align_calls) == 4
    source = fa.validate_coarse_segments(_coarse())
    scope = _plan_scope(checkpoint, source)
    surviving = _forget_one_raw_result(checkpoint, ["Alpha", "Charlie"])
    _retain_conflict_receipt(checkpoint, source, surviving, _TARGETS)

    warm_module = _NonContiguousOverlapWhisperX()
    warm = fa.align_corrected_segments(audio, _coarse(),
                                       whisperx_module=warm_module,
                                       checkpoint_dir=checkpoint,
                                       resume_scope=scope)
    assert warm_module.align_calls == ["Alpha Charlie"]
    assert warm == cold

    # Cache hits do not consume the persisted new-call allowance.
    budget = RecoveryCallBudget(checkpoint, dict(scope))
    saved = json.loads(budget.path.read_text(encoding="utf-8"))
    assert len(saved["data"]["attempts"]) == 1

    # A second resume with everything cached spends no further allowance and
    # issues no model call at all.
    again = _NonContiguousOverlapWhisperX()
    assert fa.align_corrected_segments(
        audio, _coarse(), whisperx_module=again,
        checkpoint_dir=checkpoint, resume_scope=scope) == cold
    assert again.align_calls == []
    assert len(json.loads(budget.path.read_text(encoding="utf-8"))
               ["data"]["attempts"]) == 1


def test_interrupted_call_and_tampered_budget_receipt(
        audio, tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_RAW_ASR_AUTH_KEY", "2" * 64)
    checkpoint = tmp_path / "units"
    fa.align_corrected_segments(audio, _coarse(),
                                whisperx_module=_NonContiguousOverlapWhisperX(),
                                checkpoint_dir=checkpoint)
    source = fa.validate_coarse_segments(_coarse())
    scope = _plan_scope(checkpoint, source, calls=1)
    budget = RecoveryCallBudget(checkpoint, dict(scope))
    budget.reserve("f" * 64)
    with pytest.raises(RecoveryPlanError, match="allowance exhausted"):
        RecoveryCallBudget(checkpoint, dict(scope)).reserve("e" * 64)
    saved = json.loads(budget.path.read_text(encoding="utf-8"))
    saved["data"]["attempts"] = []
    fa.atomic_json(budget.path, saved)
    with pytest.raises(RecoveryPlanError, match="authentication"):
        RecoveryCallBudget(checkpoint, dict(scope))


# --------------------------------------------------------------------------- B2

def test_closure_rejects_uid_text_and_window_mismatches(audio, tmp_path):
    fa.align_corrected_segments(audio, _coarse(),
                                whisperx_module=_NonContiguousOverlapWhisperX(),
                                checkpoint_dir=tmp_path / "units")
    source = fa.validate_coarse_segments(_coarse())
    plan = build_recovery_plan(source, _TARGETS, max_new_ctc_calls=8)
    rejected = [
        # one cue's text carried on another cue's audio window
        ([{"start": 2.6, "end": 3.7, "text": "Alpha"}], ["utt-alpha"]),
        # the requested UID is absent from the run that produced the text
        ([{"start": 2.0, "end": 4.5, "text": "Bravo"}], ["utt-alpha"]),
        ([{"start": 1.0, "end": 4.5, "text": "Alpha Bravo"}], ["utt-charlie"]),
        # wrong order, invented text, unauthorized component
        ([{"start": 1.0, "end": 3.7, "text": "Charlie Alpha"}], _TARGETS),
        ([{"start": 1.0, "end": 3.7, "text": "Alpha Delta"}], _TARGETS),
        ([{"start": 1.0, "end": 3.7, "text": "Uydurma"}], ["utt-alpha"]),
    ]
    for transcript, uids in rejected:
        assert not request_matches_plan(transcript, uids, source, plan), (
            transcript, uids)


def test_changed_source_or_scope_cannot_reuse_the_old_closure(audio, tmp_path):
    checkpoint = tmp_path / "units"
    fa.align_corrected_segments(audio, _coarse(),
                                whisperx_module=_NonContiguousOverlapWhisperX(),
                                checkpoint_dir=checkpoint)
    source = fa.validate_coarse_segments(_coarse())
    scope = _plan_scope(checkpoint, source)
    identity = {key: scope[key] for key in _IDENTITY_KEYS}
    changed = copy.deepcopy(source)
    changed[1]["text"] = "Değişti"
    with pytest.raises((fa.ForcedAlignmentError, RecoveryPlanError)):
        fa._alignment_resume_groups(scope, changed, identity)
    with pytest.raises(fa.ForcedAlignmentError, match="mix dynamic"):
        fa._alignment_resume_groups({**scope, "discovery_component_limit": 2},
                                    source, identity)


# ------------------------------------------- B2 regression: repeated dialogue

def _repeated_source():
    return fa.validate_coarse_segments([
        {"utterance_uid": uid, "text": text, "asr_text": text,
         "start_ms": start_ms, "end_ms": end_ms,
         "deletion_audio_reviewed": False, "speaker_id": "speaker-one"}
        for uid, text, start_ms, end_ms in [
            ("a", "Evet", 0, 900),
            ("b", "Tamam", 1000, 1900),
            ("c", "Evet", 2000, 2900),
            ("d", "Tamam", 3000, 3900),
        ]
    ])


def test_later_matching_context_after_identical_earlier_text():
    source = _repeated_source()
    plan = build_recovery_plan(source, ["d"], max_new_ctc_calls=3)
    assert plan["components"] == [["a", "b", "c", "d"]]
    # c+d is legitimate contiguous context for target d; a+b is identical text
    # but the wrong UID/window. Reject that candidate, not the entire request.
    assert request_matches_plan(
        [{"start": 2.0, "end": 3.9, "text": "Evet Tamam"}], ["d"], source, plan)


def test_repeated_text_does_not_relax_the_uid_text_window_binding():
    source = _repeated_source()
    plan = build_recovery_plan(source, ["d"], max_new_ctc_calls=3)
    # right text, right UID, but the window of the earlier identical run
    assert not request_matches_plan(
        [{"start": 0.0, "end": 1.9, "text": "Evet Tamam"}], ["d"], source, plan)
    # right text and window, but a UID that is not in that run
    assert not request_matches_plan(
        [{"start": 2.0, "end": 3.9, "text": "Evet Tamam"}], ["a"], source, plan)
    # the earlier run is still admissible for its own target
    assert request_matches_plan(
        [{"start": 0.0, "end": 1.9, "text": "Evet Tamam"}], ["a"], source, plan)
    # a single repeated cue still binds to its own window
    assert request_matches_plan(
        [{"start": 3.0, "end": 3.9, "text": "Tamam"}], ["d"], source, plan)
    assert not request_matches_plan(
        [{"start": 1.0, "end": 1.9, "text": "Tamam"}], ["d"], source, plan)
