"""Budget accounting merges components; request authorization identity must not.

``_resolve_alignment_overlaps`` wraps ``align`` to charge a conflict budget.
That wrapper also rewrote ``_mas_component_uids`` with the *merged* budget UID
set, so a single-utterance re-alignment was presented to the authorization layer
as if it targeted the whole merged component.  Synthetic fixtures; only the
model/hardware boundary is faked.
"""

from __future__ import annotations

import copy
import json

import pytest

import mas.engine.forced_align as fa
from mas.engine.alignment_recovery import build_recovery_plan
from mas.reliability import digest


def _result(words):
    out = [{"word": word, "start": start, "end": end, "score": 0.90}
           for word, start, end in words]
    return {"segments": [{"text": "unused", "words": copy.deepcopy(out)}],
            "word_segments": out}


# The joint call does not separate the two cues, so the resolver falls back to
# per-utterance padded candidates while the conflict budget already holds both
# UIDs.  Keyed by (text, start, end) so padded windows are distinguishable.
_SCRIPT = {
    ("Bravo", 0.6, 3.5): [("Bravo", 2.000, 2.700)],
    ("Alpha", 1.1, 3.0): [("Alpha", 1.500, 2.200)],
    ("Bravo Alpha", 0.6, 3.5): [("Bravo", 2.100, 2.700), ("Alpha", 1.500, 2.200)],
    # only the unpadded retry separates the two cues
    ("Alpha", 1.2, 2.95): [("Alpha", 1.500, 1.900)],
}


class _PaddedFallbackWhisperX:
    __version__ = "3.8.6"

    def __init__(self):
        self.align_calls: list[tuple] = []

    def load_audio(self, path):
        return object()

    def load_align_model(self, *, language_code, device, model_name):
        return object(), {"language": language_code, "type": "huggingface"}

    def align(self, transcript, model, metadata, audio, device,
              interpolate_method="nearest", return_char_alignments=False,
              print_progress=False):
        request = transcript[0]
        key = (request["text"], round(request["start"], 3), round(request["end"], 3))
        self.align_calls.append(key)
        if key not in _SCRIPT:
            raise RuntimeError(f"unscripted alignment request {key!r}")
        return _result(_SCRIPT[key])


def _coarse():
    return [
        {"utterance_uid": "utt-bravo", "start_ms": 600, "end_ms": 3500,
         "coarse_start_ms": 600, "coarse_end_ms": 3500,
         "text": "Bravo", "asr_text": "Bravo",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
        {"utterance_uid": "utt-alpha", "start_ms": 1100, "end_ms": 3000,
         "coarse_start_ms": 1200, "coarse_end_ms": 2950,
         "text": "Alpha", "asr_text": "Alpha",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
    ]


_TARGETS = ["utt-bravo", "utt-alpha"]


@pytest.fixture(autouse=True)
def _stub_model_state(monkeypatch):
    monkeypatch.setattr(fa, "_model_state_sha256", lambda model, metadata: "a" * 64)


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.flac"
    path.write_bytes(b"synthetic-audio-placeholder")
    return path


def _record_authorization(monkeypatch):
    """Observe the real authorization helper without replacing its logic."""
    seen = []
    real = fa._alignment_request_matches_scope

    def recorder(transcript, request_uids, source, scope, identity):
        verdict = real(transcript, request_uids, source, scope, identity)
        seen.append({
            "uids": tuple(request_uids) if request_uids is not None else None,
            "text": transcript[0]["text"],
            "start": transcript[0]["start"],
            "end": transcript[0]["end"],
            "authorized": verdict,
        })
        return verdict

    monkeypatch.setattr(fa, "_alignment_request_matches_scope", recorder)
    return seen


def _resume_scope(checkpoint, source):
    identity = json.loads(
        (checkpoint / "resume-identity.json").read_text(encoding="utf-8"))["data"]
    scope = {key: identity[key] for key in (
        "stage", "audio_sha256", "model_state_sha256",
        "raw_alignment_binding", "source_sha256")}
    scope["target_uids"] = list(_TARGETS)
    scope["context_uids"] = fa._alignment_resume_context_uids(source, _TARGETS)
    scope["recovery_plan"] = build_recovery_plan(
        source, _TARGETS, max_new_ctc_calls=4)
    return scope


def _forget_padded_alpha(checkpoint, source):
    dropped = surviving = None
    for path in sorted(checkpoint.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("data", {}).get("result")
        if not isinstance(result, dict):
            continue
        words = [(word["word"], word["start"], word["end"])
                 for word in result.get("word_segments", [])]
        if words == [("Alpha", 1.500, 1.900)]:
            dropped = path
        elif surviving is None:
            surviving = record
    assert dropped is not None and surviving is not None
    dropped.unlink()
    targeted = [item for item in source
                if str(item["utterance_uid"]) in set(_TARGETS)]
    details = {
        "status": "BLOCKED", "reason": "recovery_time_budget",
        "component_uids": list(_TARGETS),
        "source_sha256": digest(targeted),
        "raw_results": {surviving["data"]["uid"]:
                        digest(surviving["data"]["result"])},
        "total_recovery_seconds": 900.1,
        "limits": {"total_recovery_seconds": 900},
    }
    fa.atomic_json(checkpoint / "components/latest-conflict-failure.json",
                   {"data": details, "sha256": digest(details)})


def test_single_utterance_retry_keeps_its_own_request_identity(
        audio, tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_RAW_ASR_AUTH_KEY", "2" * 64)
    checkpoint = tmp_path / "units"
    cold_module = _PaddedFallbackWhisperX()
    cold = fa.align_corrected_segments(audio, _coarse(),
                                       whisperx_module=cold_module,
                                       checkpoint_dir=checkpoint)
    # The padded single-utterance fallback really is issued from inside the
    # conflict budget for the merged component.
    assert ("Alpha", 1.2, 2.95) in cold_module.align_calls
    assert len(cold_module.align_calls) == 4

    source = fa.validate_coarse_segments(_coarse())
    scope = _resume_scope(checkpoint, source)
    _forget_padded_alpha(checkpoint, source)

    seen = _record_authorization(monkeypatch)
    warm_module = _PaddedFallbackWhisperX()
    warm = fa.align_corrected_segments(audio, _coarse(),
                                       whisperx_module=warm_module,
                                       checkpoint_dir=checkpoint,
                                       resume_scope=scope)

    assert warm_module.align_calls == [("Alpha", 1.2, 2.95)]
    assert warm == cold

    padded = [entry for entry in seen if entry["text"] == "Alpha"]
    assert padded, seen
    # The caller asked to replace utt-alpha only; the acoustic window is padded
    # but no other cue's result may be rewritten by this call.
    assert padded[0]["uids"] == ("utt-alpha",), seen
    assert padded[0]["authorized"] is True


def test_conflict_call_budget_still_accounts_for_the_merged_component(
        audio, tmp_path, monkeypatch):
    # Budget accounting is unchanged: the whole conflict component shares one
    # allowance, and exhausting it still blocks with the merged UID list.
    monkeypatch.setattr(fa, "MAX_CONFLICT_ALIGNMENT_CALLS", 1)
    with pytest.raises(fa.AlignmentConflictBlocked) as blocked:
        fa.align_corrected_segments(audio, _coarse(),
                                    whisperx_module=_PaddedFallbackWhisperX(),
                                    checkpoint_dir=tmp_path / "budget")
    details = blocked.value.details
    assert details["reason"] == "conflict_alignment_budget"
    assert details["component_uids"] == _TARGETS
