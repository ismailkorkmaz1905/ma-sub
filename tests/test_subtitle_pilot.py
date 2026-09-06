import copy
import json
import wave

import pytest

from mas import cli
from mas.reliability import IntegrityError, digest, file_digest
from mas.subtitle.pilot import build_pilot, write_pilot
from mas.subtitle.pilot_worker import cached_stage, model_identity, verify_source_clip


def evidence(segments=None, turns=None, offset=0):
    if segments is None:
        segments = [{"start": 1.0, "end": 2.0, "text": "Merhaba dünya.", "words": [
            {"word": "Merhaba", "start": 1.0, "end": 1.4},
            {"word": " dünya.", "start": 1.5, "end": 2.0}]}]
    if turns is None:
        turns = [{"start": 1.0, "end": 2.0, "speaker": "A"}]
    body = {"format": "mas-acoustic-pilot-1", "episode": 11,
            "audio_sha256": "a" * 64, "source_sha256": "b" * 64,
            "offset_ms": offset, "duration_ms": 10_000, "segments": segments,
            "speaker_turns": turns}
    return {"data": body, "sha256": digest(body)}


def test_cue_starts_at_speech_and_never_cuts_its_end():
    source = evidence(offset=4290000)
    before = copy.deepcopy(source)
    result = build_pilot(source)["data"]
    assert result["cues"][0]["start_ms"] == 4291000
    assert result["cues"][0]["end_ms"] == 4292000
    assert source == before
    assert result["strict_delivery"] is False
    assert result["acoustic_acceptance"] == "NOT_VERIFIED"


def test_speaker_change_splits_one_decoder_segment():
    turns = [{"start": 1, "end": 1.4, "speaker": "A"},
             {"start": 1.5, "end": 2, "speaker": "B"}]
    cues = build_pilot(evidence(turns=turns))["data"]["cues"]
    assert [cue["speaker"] for cue in cues] == ["A", "B"]
    assert [cue["text"] for cue in cues] == ["Merhaba", "dünya."]


def test_real_overlap_is_not_hidden_by_exclusive_speaker_assignment():
    turns = [{"start": 1, "end": 2, "speaker": "A"},
             {"start": 1.2, "end": 1.3, "speaker": "B"}]
    result = build_pilot(evidence(turns=turns))["data"]
    assert result["cues"][0]["speaker"] is None
    assert any(issue["kind"] == "speaker_uncertain" for issue in result["issues"])


def test_weak_speaker_coverage_is_unknown():
    turns = [{"start": 1.0, "end": 1.1, "speaker": "A"}]
    result = build_pilot(evidence(turns=turns))["data"]
    assert all(cue["speaker"] is None for cue in result["cues"])


def test_uncertain_word_does_not_create_an_artificial_cue_boundary():
    source = evidence(turns=[{"start": 1.5, "end": 2, "speaker": "A"}])
    result = build_pilot(source)["data"]
    assert len(result["cues"]) == 1
    assert result["cues"][0]["speaker"] is None
    assert result["cues"][0]["text"] == source["data"]["segments"][0]["text"]
    assert (result["cues"][0]["start_ms"], result["cues"][0]["end_ms"]) == (1000, 2000)
    assert any(issue["kind"] == "speaker_uncertain" for issue in result["issues"])


def test_uncertain_bridge_never_merges_two_known_speakers():
    segments = [{"start": 1, "end": 2, "text": "one two three", "words": [
        {"word": "one", "start": 1, "end": 1.3},
        {"word": " two", "start": 1.4, "end": 1.6},
        {"word": " three", "start": 1.7, "end": 2}]}]
    turns = [{"start": 1, "end": 1.3, "speaker": "A"},
             {"start": 1.7, "end": 2, "speaker": "B"}]
    cues = build_pilot(evidence(segments=segments, turns=turns))["data"]["cues"]
    assert [cue["text"] for cue in cues] == ["one two", "three"]
    assert [cue["speaker"] for cue in cues] == [None, "B"]
    assert [(cue["start_ms"], cue["end_ms"]) for cue in cues] == [(1000, 1600), (1700, 2000)]


def test_dual_dialogue_preserves_each_cues_full_display_interval():
    segments = [
        {"start": 1, "end": 3, "text": "Birinci."},
        {"start": 2, "end": 4, "text": "İkinci."}]
    result = build_pilot(evidence(segments=segments, turns=[]))["data"]
    assert [(r["start_ms"], r["end_ms"]) for r in result["rendered"]] == [
        (1000, 2000), (2000, 3000), (3000, 4000)]
    assert result["rendered"][1]["text"] == "-Birinci.\n-İkinci."
    for cue in result["cues"]:
        pieces = [r for r in result["rendered"] if cue["cue_id"] in r["source_cue_ids"]]
        assert pieces[0]["start_ms"] == cue["start_ms"]
        assert pieces[-1]["end_ms"] == cue["end_ms"]
        assert sum(r["end_ms"] - r["start_ms"] for r in pieces) == cue["end_ms"] - cue["start_ms"]
    assert any(r["kind"] == "unresolved_overlap" for r in result["issues"])


def test_identical_text_is_not_automatically_deleted():
    segments = [{"start": 1, "end": 2, "text": "Evet."},
                {"start": 1.5, "end": 2.5, "text": "Evet."}]
    result = build_pilot(evidence(segments=segments, turns=[]))["data"]
    assert len(result["cues"]) == 2
    assert "-Evet.\n-Evet." in [r["text"] for r in result["rendered"]]


def test_readability_does_not_shorten_speech_or_discard_draft(tmp_path):
    segments = [{"start": 1, "end": 9, "text": "uzun " * 25}]
    result = build_pilot(evidence(segments=segments, turns=[]))
    assert result["data"]["cues"][0]["end_ms"] == 9000
    assert any(r["kind"] == "cue_duration_review" for r in result["data"]["issues"])
    write_pilot(tmp_path, result)
    assert (tmp_path / "draft.srt").is_file()
    with pytest.raises(IntegrityError, match="output exists"):
        write_pilot(tmp_path, result)


@pytest.mark.parametrize("mutation", ["hash", "source", "negative_offset", "bool_episode", "long_clip", "word_text", "nan"])
def test_corrupt_evidence_cannot_be_published(mutation):
    item = evidence()
    body = item["data"]
    if mutation == "hash":
        item["sha256"] = "c" * 64
    else:
        if mutation == "source": body["source_sha256"] = "bad"
        if mutation == "negative_offset": body["offset_ms"] = -1
        if mutation == "bool_episode": body["episode"] = True
        if mutation == "long_clip": body["duration_ms"] = 120001
        if mutation == "word_text": body["segments"][0]["words"][0]["word"] = "wrong"
        if mutation == "nan": body["segments"][0]["start"] = float("nan")
        if mutation != "nan": item["sha256"] = digest(body)
    with pytest.raises((IntegrityError, ValueError)):
        build_pilot(item)


@pytest.mark.parametrize("change", ["drop", "text", "timing", "reorder"])
def test_regrouper_may_split_but_cannot_change_evidence(change):
    def bad_regroup(words):
        if change == "drop": words.pop()
        if change == "text": words[0]["word"] = "changed"
        if change == "timing": words[0]["end"] = 1.1
        if change == "reorder": words.reverse()
        return [words]
    with pytest.raises(IntegrityError, match="regrouping changed"):
        build_pilot(evidence(), regroup=bad_regroup)


def test_pure_regrouping_preserves_timing():
    result = build_pilot(evidence(), regroup=lambda words: [[word] for word in words])["data"]
    assert [(cue["start_ms"], cue["end_ms"]) for cue in result["cues"]] == [(1000, 1400), (1500, 2000)]


def test_cached_inference_survives_later_failure_and_rejects_changed_identity(tmp_path):
    path = tmp_path / "stage.json"
    binding = {"audio": "a", "model": "m"}
    assert cached_stage(path, binding, lambda: {"words": ["hello"]}) == {"words": ["hello"]}
    assert cached_stage(path, binding, lambda: pytest.fail("repeated inference")) == {"words": ["hello"]}
    with pytest.raises(IntegrityError, match="binding mismatch"):
        cached_stage(path, {**binding, "model": "new"}, lambda: None)


def test_model_identity_changes_with_weights(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"first")
    first = model_identity(tmp_path)
    weights.write_bytes(b"second")
    assert first != model_identity(tmp_path)


def test_sample_provenance_requires_actual_source_bytes_and_offset(tmp_path):
    source, clip = tmp_path / "source.wav", tmp_path / "clip.wav"
    for path, samples in ((source, b"\x01\x02\x03\x04"), (clip, b"\x02\x03")):
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 1, 1000, 0, "NONE", "not compressed"))
            stream.writeframes(samples)
    verify_source_clip(clip, source, file_digest(source), 1)
    with pytest.raises(IntegrityError, match="source offset"):
        verify_source_clip(clip, source, file_digest(source), 0)
    with pytest.raises(IntegrityError, match="SHA mismatch"):
        verify_source_clip(clip, source, "a" * 64, 1)


def test_cli_replay_never_starts_runpod_or_sends_error_mail(tmp_path, monkeypatch):
    from mas.subtitle import pilot_command
    monkeypatch.setattr(pilot_command, "episode_dir", lambda _: tmp_path)
    monkeypatch.setattr(cli, "run_remote_episode", lambda *a: pytest.fail("paid run"))
    monkeypatch.setattr(cli, "notify", lambda *a: pytest.fail("unsolicited email"))
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(evidence()), encoding="utf-8")
    assert cli.main(["subtitle-pilot", "11", "--evidence", str(source)]) == 0
    assert list((tmp_path / "work/subtitle-pilot").glob("*/draft.srt"))
    assert cli.main(["subtitle-pilot", "12", "--evidence", str(source)]) == 1


def test_clip_without_word_timing_is_preserved_but_not_accepted():
    result = build_pilot(evidence(segments=[{"start": 1, "end": 2, "text": "Merhaba."}]))["data"]
    assert result["cues"][0]["text"] == "Merhaba."
    assert any(r["kind"] == "segment_timing_only" for r in result["issues"])
    assert result["status"] == "REVIEW_REQUIRED"


def test_full_episode_draft_does_not_relax_pilot_limit_or_claim_strict():
    from mas.subtitle.pilot import build_episode_draft
    item = evidence()
    item['data'].update(episode=12, duration_ms=8_352_921)
    item['sha256'] = digest(item['data'])
    with pytest.raises(IntegrityError, match='exceeds 120 seconds'):
        build_pilot(item)
    item['data'].update(format='mas-acoustic-draft-1', timing_source='faster_whisper_estimated')
    item['sha256'] = digest(item['data'])
    result = build_episode_draft(item)['data']
    assert result['mode'] == 'draft'
    assert result['timing_source'] == 'faster_whisper_estimated'
    assert result['strict_delivery'] is False
    assert result['acoustic_acceptance'] == 'NOT_VERIFIED'
    item['data']['timing_source'] = 'mixed_ctc_faster_whisper_draft'
    item['sha256'] = digest(item['data'])
    assert build_episode_draft(item)['data']['timing_source'] == 'mixed_ctc_faster_whisper_draft'
    item['data'].update(format='mas-acoustic-pilot-1', duration_ms=10_000)
    item['sha256'] = digest(item['data'])
    with pytest.raises(IntegrityError, match='unsupported draft timing'):
        build_pilot(item)
