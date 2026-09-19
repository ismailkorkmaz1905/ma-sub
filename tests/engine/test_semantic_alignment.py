import copy
import hashlib
import json
import zipfile
from dataclasses import asdict

import pytest

from mas.engine.download import sha256_file, sha256_json
from mas.engine.id_translation import (
    _jsonl_bytes,
    _pretty_json_bytes,
    _write_zip_atomic,
    build_production_translation_policy,
)
from mas.engine.semantic_alignment import (
    SemanticAlignmentConfig,
    SemanticAlignmentError,
    build_semantic_translation_schema,
    build_semantic_windows,
    build_word_timeline,
    deterministic_exact_results,
    evidence_sources,
    finalize_semantic_blocks,
    load_word_timeline,
    prepare_semantic_delivery_scope,
    read_jsonl,
    semantic_run_contract,
    sign_coarse_fallback_approval,
    write_jsonl,
)
from mas.engine.srt import SubtitleEntry, write_srt
from mas.engine.semantic_alignment_handoff import (
    RETURN_FORMAT,
    create_semantic_alignment_pack,
    validate_semantic_alignment_return,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def raw_for(tokens, *, gap_after=None, speakers=None, scenes=None, repeated_segments=False):
    gap_after = set(gap_after or ())
    words = []
    segments = []
    start = 1000
    for index, token in enumerate(tokens, start=1):
        segment_id = f"segment-{index if repeated_segments else 1}"
        word = {
            "word_index": index,
            "segment_id": segment_id,
            "start_ms": start,
            "end_ms": start + 200,
            "text": token,
            "probability": 0.95,
        }
        if speakers:
            word["speaker_id"] = speakers[index - 1]
        if scenes:
            word["scene_id"] = scenes[index - 1]
        words.append(word)
        start += 1400 if index in gap_after else 300
    if repeated_segments:
        for index, word in enumerate(words, start=1):
            segments.append(
                {
                    "segment_id": word["segment_id"],
                    "start_ms": word["start_ms"],
                    "end_ms": word["end_ms"],
                    "text": word["text"],
                }
            )
    else:
        segments.append(
            {
                "segment_id": "segment-1",
                "start_ms": words[0]["start_ms"],
                "end_ms": words[-1]["end_ms"],
                "text": " ".join(tokens),
            }
        )
    return {
        "words": words,
        "segments": segments,
        "vad_regions": [
            {
                "vad_region_index": 1,
                "start_ms": 900,
                "end_ms": words[-1]["end_ms"] + 500,
                "source": "silero_vad",
            }
        ],
        "correction_utterances": [
            {
                "utterance_uid": "candidate-1",
                "coarse_start_ms": words[0]["start_ms"],
                "coarse_end_ms": words[-1]["end_ms"],
                "asr_text": " ".join(tokens),
            }
        ],
        "youtube_captions": [],
    }


def timeline(tmp_path, raw):
    return build_word_timeline(
        raw,
        tmp_path / "semantic",
        source_sha256=SHA_A,
        timing_source_sha256=SHA_B,
        producer_identity={"format": "test", "sha256": "c" * 64},
    )


def prepare(tmp_path, raw):
    built = timeline(tmp_path, raw)
    windows = build_semantic_windows(built, evidence_sources(raw))
    auto, unresolved = deterministic_exact_results(windows)
    return built, windows, auto, unresolved


def test_empty_semantic_jsonl_remains_a_nonempty_checkpoint(tmp_path):
    path = write_jsonl(tmp_path / "empty.jsonl", [])
    assert path.read_bytes() == b"\n"
    assert read_jsonl(path) == []


def pack(tmp_path, windows, built, auto=()):
    destination = tmp_path / "semantic.zip"
    manifest = create_semantic_alignment_pack(
        windows,
        destination,
        episode=15,
        source_sha256=SHA_A,
        word_timeline_sha256=built["manifest"]["timeline_sha256"],
        all_windows_sha256=sha256_json(windows),
        deterministic_results_sha256=sha256_json(list(auto)),
        producer_sha256="d" * 64,
        config_sha256=sha256_json(asdict(SemanticAlignmentConfig())),
    )
    return destination, manifest


def returned_zip(tmp_path, input_manifest, records, *, mutate_manifest=None):
    batches = []
    payloads = {}
    offset = 0
    for number, source_batch in enumerate(input_manifest["batches"], start=1):
        count = source_batch["window_count"]
        batch = records[offset:offset + count]
        offset += count
        name = f"returned-batch-{number:03d}.jsonl"
        payload = _jsonl_bytes(batch)
        payloads[name] = payload
        batches.append({"return_file": name, "window_count": len(batch),
                        "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "format": RETURN_FORMAT,
        "input_identity_sha256": input_manifest["input_identity_sha256"],
        "input_manifest_sha256": hashlib.sha256(
            json.dumps(input_manifest, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_sha256": input_manifest["source_sha256"],
        "word_timeline_sha256": input_manifest["word_timeline_sha256"],
        "producer_sha256": input_manifest["producer_sha256"],
        "window_count": input_manifest["window_count"],
        "window_ids": input_manifest["window_ids"],
        "batch_count": input_manifest["batch_count"],
        "batches": batches,
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    output = tmp_path / "semantic-return.zip"
    _write_zip_atomic(output, {"manifest.json": _pretty_json_bytes(manifest), **payloads})
    return output


def gpt_records(windows, texts=None):
    texts = texts or [" ".join(word["text"] for word in window["timing_words"]) for window in windows]
    return [
        {
            "window_id": window["window_id"],
            "blocks": [{
                "first_word_id": window["owned_word_ids"][0],
                "last_word_id": window["owned_word_ids"][-1],
                "final_tr": text,
                "confidence": "high",
                "review_required": False,
                "note": "",
            }],
        }
        for window, text in zip(windows, texts)
    ]


def test_exact_word_span_auto_pass_and_deterministic_timeline(tmp_path):
    raw = raw_for(["Defne", "Allah", "aşkına", "ne", "yapıyorsun?"])
    raw["correction_utterances"][0]["asr_text"] = "Defne, Allah aşkına ne yapıyorsun?"
    first = timeline(tmp_path / "first", raw)
    second = timeline(tmp_path / "second", raw)
    assert first["manifest"]["timeline_sha256"] == second["manifest"]["timeline_sha256"]
    assert (first["timeline_path"].read_bytes() == second["timeline_path"].read_bytes())
    windows = build_semantic_windows(first, evidence_sources(raw))
    auto, unresolved = deterministic_exact_results(windows)
    assert not unresolved
    assert auto[0]["status"] == "AUTO_PASS"
    assert auto[0]["first_word_id"] == first["words"][0]["word_id"]
    assert auto[0]["last_word_id"] == first["words"][-1]["word_id"]


def test_asr_spelling_correction_is_accepted_only_by_semantic_return(tmp_path):
    raw = raw_for(["napıyosun"])
    built, windows, auto, unresolved = prepare(tmp_path, raw)
    assert auto
    windows[0]["primary_candidates"][0]["text"] = "ne yapıyorsun?"
    windows[0]["timing_candidates"] = []
    windows[0]["verification_candidates"] = []
    windows[0]["youtube_candidates"] = []
    auto, unresolved = deterministic_exact_results(windows)
    assert not auto and len(unresolved) == 1
    input_pack, manifest = pack(tmp_path, unresolved, built)
    output = returned_zip(tmp_path, manifest, gpt_records(unresolved, ["Ne yapıyorsun?"]))
    validated = validate_semantic_alignment_return(input_pack, output)
    assert validated["results"][0]["blocks"][0]["final_tr"] == "Ne yapıyorsun?"


def test_repeated_identical_text_is_bound_to_window_word_ids(tmp_path):
    raw = raw_for(["Evet."] * 40, gap_after=range(1, 40), repeated_segments=True)
    built = timeline(tmp_path, raw)
    windows = build_semantic_windows(built, evidence_sources(raw))
    assert len(windows) == 40
    input_pack, manifest = pack(tmp_path, windows, built)
    records = gpt_records(windows)
    records[20]["blocks"][0]["first_word_id"] = windows[0]["owned_word_ids"][0]
    records[20]["blocks"][0]["last_word_id"] = windows[0]["owned_word_ids"][0]
    output = returned_zip(tmp_path, manifest, records)
    with pytest.raises(SemanticAlignmentError, match="unowned word ID"):
        validate_semantic_alignment_return(input_pack, output)


def test_split_and_merge_are_word_id_based(tmp_path):
    raw = raw_for(["Bir", "cümle", "devam", "ediyor"])
    built = timeline(tmp_path, raw)
    windows = build_semantic_windows(built, evidence_sources(raw))
    input_pack, manifest = pack(tmp_path, windows, built)
    window = windows[0]
    split = [{"window_id": window["window_id"], "blocks": [
        {"first_word_id": window["owned_word_ids"][0], "last_word_id": window["owned_word_ids"][1],
         "final_tr": "Bir cümle"},
        {"first_word_id": window["owned_word_ids"][2], "last_word_id": window["owned_word_ids"][3],
         "final_tr": "devam ediyor."},
    ]}]
    result = validate_semantic_alignment_return(input_pack, returned_zip(tmp_path, manifest, split))
    assert len(result["results"][0]["blocks"]) == 2
    merged = gpt_records(windows, ["Bir cümle devam ediyor."])
    result = validate_semantic_alignment_return(input_pack, returned_zip(tmp_path, manifest, merged))
    assert len(result["results"][0]["blocks"]) == 1


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (raw_for(["Merhaba", "Selam"], speakers=["A", "B"]), "hard boundary"),
        (raw_for(["Merhaba", "Selam"], scenes=["one", "two"]), "hard boundary"),
        (raw_for(["Merhaba", "Selam"], gap_after=[1]), "hard boundary"),
    ],
)
def test_speaker_scene_and_internal_silence_cannot_be_one_block(tmp_path, raw, match):
    built = timeline(tmp_path, raw)
    windows = build_semantic_windows(built, evidence_sources(raw))
    assert len(windows) == 2
    first = windows[0]
    forged = copy.deepcopy(first)
    forged["owned_word_ids"].append(windows[1]["owned_word_ids"][0])
    forged["timing_words"].append(windows[1]["timing_words"][0])
    forged["next_context"]["read_only_context_words"] = []
    input_pack, manifest = pack(tmp_path, [forged], built)
    output = returned_zip(tmp_path, manifest, gpt_records([forged]))
    with pytest.raises(SemanticAlignmentError, match=match):
        validate_semantic_alignment_return(input_pack, output)


def test_duplicate_ownership_and_missing_speech_fail_release(tmp_path):
    raw = raw_for(["Bir", "iki", "üç"])
    built, windows, _, _ = prepare(tmp_path, raw)
    blocks = [{"window_id": windows[0]["window_id"], "resolution": "gpt_semantic", "blocks": [
        {"first_word_id": windows[0]["owned_word_ids"][0], "last_word_id": windows[0]["owned_word_ids"][1],
         "final_tr": "Bir iki", "confidence": "high", "review_required": False, "note": ""},
    ]}]
    with pytest.raises(SemanticAlignmentError, match="unresolved windows"):
        finalize_semantic_blocks(
            episode=15, source_sha256=SHA_A, timeline=built, windows=windows,
            deterministic_results=[], semantic_results=[], output_dir=tmp_path / "output")
    finalized = finalize_semantic_blocks(
        episode=15, source_sha256=SHA_A, timeline=built, windows=windows,
        deterministic_results=[], semantic_results=blocks, output_dir=tmp_path / "partial")
    assert finalized["report"]["unassigned_eligible_speech_word_count"] == 1
    assert finalized["report"]["release_eligible"] is False


def test_context_theft_unknown_id_reorder_and_timestamp_injection_fail(tmp_path):
    raw = raw_for(["Bir", "iki"], gap_after=[1], repeated_segments=True)
    built = timeline(tmp_path, raw)
    windows = build_semantic_windows(built, evidence_sources(raw))
    input_pack, manifest = pack(tmp_path, windows, built)
    cases = []
    context = copy.deepcopy(gpt_records(windows))
    context[1]["blocks"][0]["first_word_id"] = windows[0]["owned_word_ids"][0]
    context[1]["blocks"][0]["last_word_id"] = windows[0]["owned_word_ids"][0]
    cases.append((context, "unowned word ID"))
    unknown = copy.deepcopy(gpt_records(windows))
    unknown[0]["blocks"][0]["first_word_id"] = "tw-unknown"
    cases.append((unknown, "unowned word ID"))
    reordered = list(reversed(gpt_records(windows)))
    cases.append((reordered, "reordered windows"))
    timestamp = gpt_records(windows)
    timestamp[0]["blocks"][0]["start_ms"] = 123
    cases.append((timestamp, "forbidden timestamp"))
    for number, (records, match) in enumerate(cases):
        output = returned_zip(tmp_path / str(number), manifest, records)
        with pytest.raises(SemanticAlignmentError, match=match):
            validate_semantic_alignment_return(input_pack, output)


def test_stale_pack_is_rejected(tmp_path):
    raw = raw_for(["Merhaba"])
    built = timeline(tmp_path, raw)
    windows = build_semantic_windows(built, evidence_sources(raw))
    input_pack, manifest = pack(tmp_path, windows, built)
    output = returned_zip(
        tmp_path,
        manifest,
        gpt_records(windows),
        mutate_manifest=lambda value: value.update(word_timeline_sha256="e" * 64),
    )
    with pytest.raises(SemanticAlignmentError, match="stale or invalid"):
        validate_semantic_alignment_return(input_pack, output)


def test_106_metadata_records_never_reach_semantic_pack(tmp_path):
    tokens = []
    raw = {"words": [], "segments": [], "vad_regions": [], "correction_utterances": [],
           "youtube_captions": []}
    start = 1000
    for index in range(106):
        segment_id = f"credit-{index:03d}"
        raw["segments"].append({"segment_id": segment_id, "start_ms": start,
                                "end_ms": start + 500, "text": "Altyazı M.K."})
        for token in ("Altyazı", "M.K."):
            raw["words"].append({"word_index": len(raw["words"]) + 1, "segment_id": segment_id,
                                 "start_ms": start, "end_ms": start + 100, "text": token})
            start += 120
        start += 1200
    raw["segments"].append({"segment_id": "dialogue", "start_ms": start,
                            "end_ms": start + 300, "text": "Merhaba"})
    raw["words"].append({"word_index": len(raw["words"]) + 1, "segment_id": "dialogue",
                         "start_ms": start, "end_ms": start + 300, "text": "Merhaba"})
    raw["vad_regions"] = [{"vad_region_index": 1, "start_ms": 0, "end_ms": start + 1000}]
    raw["correction_utterances"] = [{"utterance_uid": "credit-candidate", "coarse_start_ms": 0,
                                      "coarse_end_ms": start, "asr_text": "Altyazı M.K."},
                                     {"utterance_uid": "dialogue", "coarse_start_ms": start,
                                      "coarse_end_ms": start + 300, "asr_text": "Merhaba"}]
    built = timeline(tmp_path, raw)
    assert len(built["quarantine"]) == 106
    windows = build_semantic_windows(built, evidence_sources(raw))
    assert sum(len(window["owned_word_ids"]) for window in windows) == 1
    input_pack, _ = pack(tmp_path, windows, built)
    with zipfile.ZipFile(input_pack) as archive:
        payload = b"\n".join(archive.read(name) for name in archive.namelist() if name.startswith("batch-"))
    assert b"Altyaz" not in payload and b"Takarir" not in payload


def test_final_timestamps_ownership_and_translation_schema_are_locked(tmp_path):
    raw = raw_for(["Defne", "geldi."])
    built, windows, auto, unresolved = prepare(tmp_path, raw)
    assert auto and not unresolved
    finalized = finalize_semantic_blocks(
        episode=15, source_sha256=SHA_A, timeline=built, windows=windows,
        deterministic_results=auto, semantic_results=[], output_dir=tmp_path / "output")
    assert finalized["report"]["release_eligible"] is True
    assert finalized["report"]["assigned_word_count"] == finalized["report"]["eligible_word_count"]
    assert finalized["blocks"][0]["start_ms"] == built["words"][0]["start_ms"]
    assert finalized["blocks"][0]["end_ms"] == built["words"][-1]["end_ms"] + 220
    from mas.pipeline import _load_configs

    _, series, names, religious = _load_configs()
    policy = build_production_translation_policy(series, names, religious)
    schema = build_semantic_translation_schema(
        episode=15, final_blocks=finalized["blocks"], source_sha256=SHA_A,
        word_timeline_sha256=built["manifest"]["timeline_sha256"], production_policy=policy)
    assert schema["blocks"][0]["block_uid"] == finalized["blocks"][0]["block_uid"]
    assert schema["blocks"][0]["start_ms"] == finalized["blocks"][0]["start_ms"]
    assert schema["blocks"][0]["alignment_provenance"]["strict_ctc_pass"] is False


def test_review_required_window_needs_bound_explicit_fallback_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_RAW_ASR_AUTH_KEY", "9" * 64)
    raw = raw_for(["Merhaba"])
    built, windows, _, _ = prepare(tmp_path, raw)
    window = windows[0]
    review = [{
        "window_id": window["window_id"],
        "resolution": "gpt_semantic",
        "blocks": [{
            "first_word_id": window["owned_word_ids"][0],
            "last_word_id": window["owned_word_ids"][0],
            "final_tr": "Merhaba",
            "confidence": "low",
            "review_required": True,
            "note": "evidence is ambiguous",
        }],
    }]
    body = {
        "window_id": window["window_id"],
        "source_sha256": SHA_A,
        "word_timeline_sha256": built["manifest"]["timeline_sha256"],
        "candidate_timing": {
            "first_word_id": window["owned_word_ids"][0],
            "last_word_id": window["owned_word_ids"][0],
            "start_ms": 1000,
            "end_ms": 1500,
            "final_tr": "Merhaba",
        },
        "reason": "Operator approved only this ambiguous component.",
        "approved": True,
    }
    approval = sign_coarse_fallback_approval(body)
    forged = copy.deepcopy(approval)
    forged["approval_auth_tag"] = "0" * 64
    with pytest.raises(SemanticAlignmentError, match="signature changed"):
        finalize_semantic_blocks(
            episode=15, source_sha256=SHA_A, timeline=built, windows=windows,
            deterministic_results=[], semantic_results=review, output_dir=tmp_path / "forged",
            coarse_approvals=[forged], allow_approved_coarse_release=True)
    blocked = finalize_semantic_blocks(
        episode=15, source_sha256=SHA_A, timeline=built, windows=windows,
        deterministic_results=[], semantic_results=review, output_dir=tmp_path / "blocked")
    assert blocked["report"]["release_eligible"] is False
    assert (tmp_path / "blocked/semantic_alignment_review_required.json").is_file()
    released = finalize_semantic_blocks(
        episode=15, source_sha256=SHA_A, timeline=built, windows=windows,
        deterministic_results=[], semantic_results=review, output_dir=tmp_path / "released",
        coarse_approvals=[approval], allow_approved_coarse_release=True)
    assert released["report"]["release_eligible"] is True
    assert released["report"]["semantic_alignment_pass"] is False
    assert released["report"]["approved_coarse_fallback_count"] == 1
    assert released["blocks"][0]["timing_source"] == "approved_coarse_component_fallback_v1"
    assert (released["blocks"][0]["start_ms"], released["blocks"][0]["end_ms"]) == (1000, 1500)


def test_partial_and_whole_share_semantic_policy_and_ctc_is_not_called():
    whole = semantic_run_contract(15, "whole-episode")
    partial = semantic_run_contract(15, "first-hour")
    assert whole["alignment_policy"] == partial["alignment_policy"] == "semantic-block-v1"
    assert whole["delivery_scope"] != partial["delivery_scope"]
    assert semantic_run_contract(1, "whole-episode")["alignment_policy"] == "semantic-block-v1"


def test_partial_and_whole_select_from_same_canonical_blocks(tmp_path):
    root = tmp_path / "episode"
    blocks_path = root / "work/semantic_alignment/final_blocks.jsonl"
    blocks_path.parent.mkdir(parents=True)
    blocks = [
        {"block_uid": "one", "block_index": 1, "start_ms": 3_599_000, "end_ms": 3_600_100},
        {"block_uid": "two", "block_index": 2, "start_ms": 3_601_000, "end_ms": 3_602_000},
    ]
    blocks_path.write_text("".join(json.dumps(block) + "\n" for block in blocks), encoding="utf-8")
    subtitle_dir = root / "output/subtitles"
    entries = [
        SubtitleEntry(1, 3_599_000, 3_600_100, "Bir"),
        SubtitleEntry(2, 3_601_000, 3_602_000, "İki"),
    ]
    write_srt(subtitle_dir / "episode-id.srt", entries)
    write_srt(subtitle_dir / "episode-tr.srt", entries)
    report = {
        "episode": 15,
        "input_files": {"final_blocks": {"relative_path": "work/semantic_alignment/final_blocks.jsonl"}},
        "outputs": {
            "id_srt": {"relative_path": "output/subtitles/episode-id.srt"},
            "tr_srt": {"relative_path": "output/subtitles/episode-tr.srt"},
        },
    }
    whole = prepare_semantic_delivery_scope(
        root, report, delivery_scope="whole-episode", finalization_sha256=SHA_A)
    partial = prepare_semantic_delivery_scope(
        root, report, delivery_scope="first-hour", finalization_sha256=SHA_A)
    assert whole["canonical_final_blocks_sha256"] == partial["canonical_final_blocks_sha256"]
    assert whole["selected_block_uids"] == ["one", "two"]
    assert partial["selected_block_uids"] == ["one"]
    assert partial["duration_limit_ms"] == 3_600_100


def test_resume_timeline_loads_only_exact_bound_artifacts(tmp_path):
    raw = raw_for(["Merhaba"])
    built = timeline(tmp_path, raw)
    loaded = load_word_timeline(tmp_path / "semantic")
    assert loaded["manifest"] == built["manifest"]
    loaded["words"][0]["text"] = "changed"
    built["timeline_path"].write_text(json.dumps(loaded["words"][0]) + "\n", encoding="utf-8")
    with pytest.raises(SemanticAlignmentError, match="binding"):
        load_word_timeline(tmp_path / "semantic")
