"""Standalone punctuation records must not break joint/context word slicing.

``_validate_word_record`` keeps a punctuation-only token in the segment text
instead of letting it manufacture an interval, so the independent path tolerates
extra raw records.  The joint and independent-context recovery paths compared a
raw record count against a lexical surface count, so one comma discarded the
whole recovery candidate.  Synthetic fixtures; model boundary faked only.
"""

from __future__ import annotations

import copy

import pytest

import mas.engine.forced_align as fa


def _raw(words):
    out = []
    for entry in words:
        record = {"word": entry[0]}
        if len(entry) > 1:
            record.update(start=entry[1], end=entry[2],
                          score=entry[3] if len(entry) > 3 else 0.90)
        out.append(record)
    return {"segments": [{"text": "unused", "words": copy.deepcopy(out)}],
            "word_segments": out}


class _ScriptedWhisperX:
    """Only the model/hardware boundary is faked."""

    __version__ = "3.8.6"

    def __init__(self, script):
        self.script = script
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
        if text not in self.script:
            raise RuntimeError(f"unscripted alignment request {text!r}")
        return _raw(self.script[text])


@pytest.fixture(autouse=True)
def _stub_model_state(monkeypatch):
    monkeypatch.setattr(fa, "_model_state_sha256", lambda model, metadata: "a" * 64)


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.flac"
    path.write_bytes(b"synthetic-audio-placeholder")
    return path


# ---------------------------------------------- joint overlap recovery (a)

def _overlap_coarse():
    return [
        {"utterance_uid": "utt-alpha", "start_ms": 500, "end_ms": 3400,
         "coarse_start_ms": 1200, "coarse_end_ms": 2600,
         "text": "Alpha, dur!", "asr_text": "Alpha, dur!",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
        {"utterance_uid": "utt-bravo", "start_ms": 600, "end_ms": 3500,
         "coarse_start_ms": 1400, "coarse_end_ms": 2800,
         "text": "Bravo mu?", "asr_text": "Bravo mu?",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
    ]


_JOINT_TEXT = "Alpha, dur! Bravo mu?"
_INDEPENDENT = {
    "Alpha, dur!": [("Alpha,", 1.300, 1.700), ("dur!", 1.800, 2.300)],
    "Bravo mu?": [("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750)],
}
# leading, medial and trailing punctuation-only records around four real words
_JOINT_WITH_PUNCTUATION = [
    (",",), ("Alpha,", 1.300, 1.700), ("dur!", 1.800, 2.100), ("-",),
    ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750), (".",),
]
_JOINT_CLEAN = [
    ("Alpha,", 1.300, 1.700), ("dur!", 1.800, 2.100),
    ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750),
]


def test_joint_component_survives_standalone_punctuation_records(audio, tmp_path):
    punctuated = fa.align_corrected_segments(
        audio, _overlap_coarse(),
        whisperx_module=_ScriptedWhisperX(
            {**_INDEPENDENT, _JOINT_TEXT: _JOINT_WITH_PUNCTUATION}),
        checkpoint_dir=tmp_path / "punctuated")
    clean = fa.align_corrected_segments(
        audio, _overlap_coarse(),
        whisperx_module=_ScriptedWhisperX(
            {**_INDEPENDENT, _JOINT_TEXT: _JOINT_CLEAN}),
        checkpoint_dir=tmp_path / "clean")

    # Same intervals either way; punctuation only changes the raw record count.
    assert ([(word["utterance_uid"], word["text"], word["start_ms"], word["end_ms"])
             for word in punctuated["words"]]
            == [(word["utterance_uid"], word["text"], word["start_ms"], word["end_ms"])
                for word in clean["words"]])
    # Source punctuation is preserved in the emitted word surfaces.
    assert [word["text"] for word in punctuated["words"]] == [
        "Alpha,", "dur!", "Bravo", "mu?"]
    assert punctuated["provenance"]["overlap_resolution"]["final_overlap_count"] == 0


@pytest.mark.parametrize("joint,reason", [
    # a real word is missing
    ([("Alpha,", 1.300, 1.700), ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750)],
     "missing word"),
    # an extra real word appears
    ([("Alpha,", 1.300, 1.700), ("dur!", 1.800, 2.100), ("hemen", 2.110, 2.150),
      ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750)], "extra word"),
    # malformed record: no text at all
    ([(",",), ("Alpha,", 1.300, 1.700), ("dur!", 1.800, 2.100), ("",),
      ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750)], "empty record"),
    # invalid interval on a real word
    ([("Alpha,", 1.300, 1.700), ("dur!", 2.100, 1.800),
      ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750)], "backwards interval"),
    # out-of-range score on a real word
    ([("Alpha,", 1.300, 1.700), ("dur!", 1.800, 2.100, 1.4),
      ("Bravo", 2.200, 2.600), ("mu?", 2.650, 2.750)], "invalid score"),
])
def test_broken_joint_results_are_still_refused(audio, tmp_path, joint, reason):
    module = _ScriptedWhisperX({**_INDEPENDENT, _JOINT_TEXT: joint})
    with pytest.raises(fa.ForcedAlignmentError):
        fa.align_corrected_segments(audio, _overlap_coarse(),
                                    whisperx_module=module,
                                    checkpoint_dir=tmp_path / reason.replace(" ", "-"))


# ------------------------------------ independent-context recovery path (b)

def _context_coarse():
    return [
        {"utterance_uid": "utt-one", "start_ms": 500, "end_ms": 2000,
         "coarse_start_ms": 600, "coarse_end_ms": 1900,
         "text": "Merhaba.", "asr_text": "Merhaba.",
         "deletion_audio_reviewed": False, "speaker_id": "speaker-a"},
        {"utterance_uid": "utt-two", "start_ms": 2100, "end_ms": 3600,
         "coarse_start_ms": 2200, "coarse_end_ms": 3500,
         "text": "Nasılsın?", "asr_text": "Nasılsın?",
         "deletion_audio_reviewed": False, "speaker_a": None,
         "speaker_id": "speaker-b"},
    ]


_CONTEXT_JOINT_TEXT = "Merhaba. Nasilsin?"
_CONTEXT_SCRIPT = {
    "Merhaba.": [("Merhaba.", 0.700, 1.200)],
    # below DEFAULT_MIN_WORD_SCORE, no adjacent support -> recoverable failure
    "Nasilsin?": [("Nasılsın?", 2.300, 2.900, 0.10)],
}
_CONTEXT_WITH_PUNCTUATION = [
    (",",), ("Merhaba.", 0.700, 1.200), ("-",),
    ("Nasılsın?", 2.300, 2.900), (".",),
]
_CONTEXT_CLEAN = [("Merhaba.", 0.700, 1.200), ("Nasılsın?", 2.300, 2.900)]


def test_independent_context_recovery_survives_punctuation_records(audio, tmp_path):
    punctuated = fa.align_corrected_segments(
        audio, _context_coarse(),
        whisperx_module=_ScriptedWhisperX(
            {**_CONTEXT_SCRIPT, _CONTEXT_JOINT_TEXT: _CONTEXT_WITH_PUNCTUATION}),
        checkpoint_dir=tmp_path / "ctx-punctuated")
    clean = fa.align_corrected_segments(
        audio, _context_coarse(),
        whisperx_module=_ScriptedWhisperX(
            {**_CONTEXT_SCRIPT, _CONTEXT_JOINT_TEXT: _CONTEXT_CLEAN}),
        checkpoint_dir=tmp_path / "ctx-clean")

    assert [(word["utterance_uid"], word["text"], word["start_ms"], word["end_ms"])
            for word in punctuated["words"]] == [
        ("utt-one", "Merhaba.", 700, 1200),
        ("utt-two", "Nasılsın?", 2300, 2900)]
    assert ([word["start_ms"] for word in punctuated["words"]]
            == [word["start_ms"] for word in clean["words"]])
    # Punctuation-only raw records are counted, not turned into intervals.
    assert (punctuated["report"]["raw_punctuation_only_word_count"]
            > clean["report"]["raw_punctuation_only_word_count"])


def test_independent_context_recovery_refuses_a_wrong_word_count(audio, tmp_path):
    module = _ScriptedWhisperX({
        **_CONTEXT_SCRIPT,
        _CONTEXT_JOINT_TEXT: [("Merhaba.", 0.700, 1.200),
                              ("Nasılsın?", 2.300, 2.900),
                              ("fazla", 2.950, 3.100)],
    })
    with pytest.raises(fa.ForcedAlignmentError, match="independent alignment failed"):
        fa.align_corrected_segments(audio, _context_coarse(),
                                    whisperx_module=module,
                                    checkpoint_dir=tmp_path / "ctx-extra")
