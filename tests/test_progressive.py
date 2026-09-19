"""Progressive delivery tests."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas import delivery_first, pipeline, progressive, partial_delivery
from mas.delivery import NEXT_PART, READY_FOR_PARTIAL_ENCODE, WAIT_PART_RETURN
from mas.engine import part_audio, partial_finalize
from mas.engine.part_scope import build_part_plan, project_part_vad
from mas.reliability import atomic_json, digest, file_digest as sha256_file, read_json


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery_first, "enabled", lambda _episode: False)
    source = {"relative_path": "source/movie.mp4", "size_bytes": 10, "sha256": "1" * 64}
    audio = {"relative_path": "work/audio.flac", "size_bytes": 10, "sha256": "2" * 64,
             "sample_rate_hz": 16000, "channels": 1, "sample_count": 7200000 * 16}
    regions = [{"vad_region_index": index, "start_ms": start, "end_ms": end, "source": "silero_vad"}
               for index, (start, end) in enumerate([(1000, 2500), (3600000, 3601000),
                                                    (3603000, 3605000), (7197000, 7199000)], 1)]
    plan = build_part_plan(episode=15, source=source, audio=audio,
        vad={"independent_vad": True, "audio_sha256": audio["sha256"], "sample_count": audio["sample_count"],
             "config": {"policy": "canonical"}, "model": {"name": "silero"},
             "producer_sha256": "3" * 64, "regions": regions})
    atomic_json(tmp_path / "work/part-plan.json", {"data": plan, "sha256": digest(plan)})
    calls, published = [], set()
    monkeypatch.setattr(part_audio, "prepare_episode_parts", lambda *args, **kwargs: copy.deepcopy(plan))
    monkeypatch.setattr(partial_delivery, "validate_worker_published_part",
                        lambda root, episode, part_id, **kwargs: {"authority": "controller"} if part_id in published else None)

    def put(path, content=b"fixture"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def extract(root, episode, part_id, **kwargs):
        calls.append(("extract", part_id))
        child = tmp_path / "parts" / part_id
        path = put(child / "work/audio.flac")
        part = next(part for part in plan["parts"] if part["part_id"] == part_id)
        lineage = {"audio": {"sha256": sha256_file(path)}, "part_id": part_id,
                   "sample_count": part["end_sample"] - part["start_sample"]}
        marker = child / "work/audio-part.done.json"
        atomic_json(marker, lineage)
        return {"audio_path": path, "captions_path": None, "lineage_path": marker,
                "lineage": lineage, "parent_vad_regions": project_part_vad(plan, part_id)}

    monkeypatch.setattr(part_audio, "extract_part_audio", extract)
    monkeypatch.setattr(pipeline, "_stage", lambda path, state, name, action: action())
    monkeypatch.setattr(pipeline, "_pending_extra_audio_review_uids", lambda *args: ())

    def transcribe(path, directory, **kwargs):
        part_id = Path(path).parents[1].name
        calls.append(("asr", part_id))
        assert kwargs["config"].allow_cpu_fallback is False
        raw = {"audio_sha256": sha256_file(path), "independent_vad": True,
               "model": {"settings": {"allow_cpu_fallback": False}}, "correction_utterances": [],
               "speech_hole_records": [], "asr_hallucination_records": [], "vad_regions": []}
        atomic_json(Path(directory) / "raw_asr_v2.json", raw)
        atomic_json(Path(directory) / "raw_asr_v2.done.json", {"fixture": "raw marker"})
        return raw

    monkeypatch.setattr(pipeline, "transcribe_raw_audio", transcribe)

    def tr_pack(_records, _holes, path, **kwargs):
        put(path, b"immutable TR pack")
        return {"input_sha256": "4" * 64}

    monkeypatch.setattr(pipeline, "create_tr_correction_pack", tr_pack)
    monkeypatch.setattr(pipeline, "validate_tr_correction_output",
                        lambda *args: SimpleNamespace(records=[], output_sha256="5" * 64))

    def review(pack, tr_text, tr_final, report_path, recovery_path, **kwargs):
        calls.append(("review", Path(pack).parents[1].name))
        assert kwargs["config"].device == "cuda" and kwargs["config"].allow_cpu_fallback is False
        put(tr_final, b"acoustically corrected TR")
        atomic_json(report_path, {"review_count": 0})
        return {"review_count": 0}

    monkeypatch.setattr(pipeline, "resolve_tr_audio_reviews", review)
    monkeypatch.setattr(pipeline, "read_tr_correction_pack",
                        lambda *args: SimpleNamespace(utterances=[], speech_holes=[], asr_hallucination_records=[],
                                                     manifest={"input_sha256": "4" * 64}))
    monkeypatch.setattr(pipeline, "correction_records_to_alignment_inputs", lambda *args, **kwargs:
                        SimpleNamespace(alignment_inputs=[]))

    def align(path, audio, inputs, regions, sha, **kwargs):
        calls.append(("align", Path(path).parents[1].name))
        atomic_json(path, {"aligned": "immutable child"})
        atomic_json(Path(path).with_suffix(".done.json"), {"fixture": "alignment marker"})
        return {"aligned": "immutable child"}, False

    monkeypatch.setattr(pipeline, "_aligned_checkpoint", align)
    policy = {"glossary": {"canonical_names": [], "religious_terms": []}}
    monkeypatch.setattr(pipeline, "build_production_translation_policy", lambda *args: policy)

    def strict(raw, correction, aligned, **kwargs):
        part_id = kwargs["part_lineage"]["part_id"]
        calls.append(("strict", part_id))
        assert kwargs["parent_vad_regions"] == project_part_vad(plan, part_id)
        assert kwargs["production_policy"] == policy
        return SimpleNamespace(schema={"schema_sha256": digest(part_id), "production_policy": policy,
                                       "blocks": [{"block_uid": "frozen-uid"}]},
                               preparation=SimpleNamespace(window_audit=[]), speech_coverage_report={"status": "PASS"},
                               timing_qa_report={"status": "PASS"})

    monkeypatch.setattr(pipeline, "build_strict_artifacts", strict)

    def id_pack(artifacts, path, **kwargs):
        put(path, b"immutable ID pack")
        return {"schema_sha256": artifacts.schema["schema_sha256"]}

    monkeypatch.setattr(pipeline, "create_id_translation_pack", id_pack)

    def workspace(pack, directory):
        path = Path(directory) / "frozen" / "workspace.json"
        atomic_json(path, {"fixture": "workspace"})
        for index in range(1, 4):
            atomic_json(path.parent / f"worker-{index:02d}/input.json", {"fixture": index})
        return path

    monkeypatch.setattr(pipeline, "prepare_id_translation_workspaces", workspace)
    monkeypatch.setattr(pipeline, "validate_id_workspace_output", lambda *args: calls.append(("workspace_qa", "part-001")))
    monkeypatch.setattr(pipeline, "load_and_validate_id_translation_zip",
                        lambda *args, **kwargs: SimpleNamespace(output_block_count=1))
    monkeypatch.setattr(pipeline, "_validate_id_quality", lambda *args: calls.append(("meaning_timing_qa", "part-001")))

    def finalize(root, episode, part_id, **kwargs):
        calls.append(("export", part_id))
        export = {"mode": "strict-partial-subtitles", "part_id": part_id}
        atomic_json(tmp_path / "parts" / part_id / "work/partial-export.json", export)
        atomic_json(tmp_path / "work/partial-export.json", export)
        return {"status": "PASS", "full_episode_complete": False}

    monkeypatch.setattr(partial_finalize, "finalize_partial_episode", finalize)
    monkeypatch.setattr(partial_finalize, "validate_partial_export", lambda root, episode, part_id, **kwargs:
                        (read_json(tmp_path / "parts" / part_id / "work/partial-export.json"), {"full_episode_complete": False}))
    options = {"root": tmp_path, "episode": 15, "source_video": tmp_path / "source/movie.mp4",
               "audio_path": tmp_path / "work/audio.flac", "total_timeout": 30,
               "config_dir": tmp_path / "config", "series": {"whisper_model": "large-v3", "batch_size": 250},
               "names": {"canonical_names": []}, "religious": {"terms": []}}
    return SimpleNamespace(root=tmp_path, options=options, calls=calls, published=published, put=put, plan=plan)


def add_return(worker, kind, part_id="part-001"):
    suffix = "_TR_TEXT_CORRECTED.zip" if kind == "tr" else "_ID_TRANSLATED.zip"
    worker.put(worker.root / "parts" / part_id / "handoff" / ("Muhtemel Ask 15.Bolum" + suffix))


def test_first_part_reaches_export_before_any_tail_asr(worker):
    assert progressive.run_progressive_worker(**worker.options) == WAIT_PART_RETURN
    tr_handoff = progressive.validate_partial_handoff(worker.root, 15)
    assert tr_handoff["kind"] == "tr" and tr_handoff["part_id"] == "part-001"
    add_return(worker, "tr")
    assert progressive.run_progressive_worker(**worker.options) == WAIT_PART_RETURN
    id_handoff = progressive.validate_partial_handoff(worker.root, 15)
    assert id_handoff["kind"] == "id"
    assert sum("worker-" in item["relative_path"] for item in id_handoff["files"]) == 3
    schema_path = worker.root / "parts/part-001/work/aligned_tr_schema_v2.json"
    frozen_schema_sha = sha256_file(schema_path)
    assert read_json(schema_path)["blocks"] == [{"block_uid": "frozen-uid"}]
    add_return(worker, "id")
    assert progressive.run_progressive_worker(**worker.options) == READY_FOR_PARTIAL_ENCODE
    assert sha256_file(schema_path) == frozen_schema_sha
    assert not (worker.root / "work/raw_asr_v2.json").exists()
    assert worker.calls[-3:] == [("workspace_qa", "part-001"), ("meaning_timing_qa", "part-001"), ("export", "part-001")]
    assert not any(part == "part-002" for _, part in worker.calls)
    assert read_json(worker.root / "parts/part-001/work/partial-handoff-tr.json")["data"] == tr_handoff
    worker.published.add("part-001")
    assert progressive.run_progressive_worker(**worker.options) == WAIT_PART_RETURN
    assert worker.calls.index(("export", "part-001")) < worker.calls.index(("asr", "part-002"))


def test_export_checkpoint_reused_without_gpu_stages(worker):
    add_return(worker, "tr")
    add_return(worker, "id")
    assert progressive.run_progressive_worker(**worker.options) == READY_FOR_PARTIAL_ENCODE
    count = len(worker.calls)
    atomic_json(worker.root / "work/partial-export.json", {"stale_alias": True})
    assert progressive.run_progressive_worker(**worker.options) == READY_FOR_PARTIAL_ENCODE
    assert len(worker.calls) == count
    assert read_json(worker.root / "work/partial-export.json")["part_id"] == "part-001"


def test_all_controller_acknowledgements_request_external_completion_not_full_pass(worker):
    worker.published.update(part["part_id"] for part in worker.plan["parts"])
    assert progressive.run_progressive_worker(**worker.options) == NEXT_PART
    assert not worker.calls
    assert not (worker.root / "output/drive_readback_receipt.json").exists()


def test_handoff_validation_requires_no_parent_media(worker):
    assert not worker.options["source_video"].exists() and not worker.options["audio_path"].exists()
    progressive.run_progressive_worker(**worker.options)
    data = progressive.validate_partial_handoff(worker.root, 15)
    assert data["plan_sha256"] == sha256_file(worker.root / "work/part-plan.json")


def test_changed_pack_or_foreign_return_blocks_handoff(worker):
    progressive.run_progressive_worker(**worker.options)
    handoff = progressive.validate_partial_handoff(worker.root, 15)
    worker.put(worker.root / handoff["pack"]["relative_path"], b"changed")
    with pytest.raises(RuntimeError, match="pack or return ownership"):
        progressive.validate_partial_handoff(worker.root, 15)


def test_cross_part_return_is_rejected_even_when_all_manifest_hashes_recomputed(worker):
    progressive.run_progressive_worker(**worker.options)
    data = progressive.validate_partial_handoff(worker.root, 15)
    data["expected_return"] = data["expected_return"].replace("part-001", "part-002")
    for path in (worker.root / "work/partial-handoff.json", worker.root / "parts/part-001/work/partial-handoff.json",
                 worker.root / "parts/part-001/work/partial-handoff-tr.json"):
        progressive._bound_json(path, data)
    with pytest.raises(RuntimeError, match="pack or return ownership"):
        progressive.validate_partial_handoff(worker.root, 15)


def test_stage_budget_failure_does_not_start_next_stage(worker, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(progressive.time, "monotonic", lambda: now[0])
    previous = pipeline.transcribe_raw_audio

    def slow(*args, **kwargs):
        result = previous(*args, **kwargs)
        now[0] += 31
        return result

    monkeypatch.setattr(pipeline, "transcribe_raw_audio", slow)
    with pytest.raises(RuntimeError, match="episode budget expired"):
        progressive.run_progressive_worker(**worker.options)
    assert worker.calls == [("extract", "part-001"), ("asr", "part-001")]
    assert not (worker.root / "parts/part-001/handoff/Muhtemel Ask 15.Bolum_TR_CORRECTION_PACK.zip").exists()


def test_rejected_quality_never_finalizes_or_starts_tail(worker, monkeypatch):
    add_return(worker, "tr")
    add_return(worker, "id")

    def reject(*args):
        raise RuntimeError("meaning or cue ownership rejected")

    monkeypatch.setattr(pipeline, "_validate_id_quality", reject)
    with pytest.raises(RuntimeError, match="meaning or cue ownership"):
        progressive.run_progressive_worker(**worker.options)
    assert not any(name == "export" or part == "part-002" for name, part in worker.calls)


def test_part_retry_requires_retained_context_before_any_preparation(worker, monkeypatch):
    from mas import retry_authorization

    monkeypatch.setattr(part_audio, "prepare_episode_parts", lambda *args, **kwargs: pytest.fail("no preparation"))
    def reject(*args):
        raise RuntimeError("frozen predecessor changed")
    monkeypatch.setattr(retry_authorization, "validate_part_retry_context", reject)
    with pytest.raises(RuntimeError, match="frozen predecessor changed"):
        progressive.run_progressive_worker(**worker.options, resume_scope={"part_id": "part-001"})
    assert not worker.calls


def test_part_retry_cannot_select_another_unpublished_part(worker, monkeypatch):
    from mas import retry_authorization

    monkeypatch.setattr(retry_authorization, "validate_part_retry_context", lambda *args: {})
    with pytest.raises(RuntimeError, match="earliest unpublished part"):
        progressive.run_progressive_worker(**worker.options, resume_scope={"part_id": "part-002"})
    assert not worker.calls


def test_scoped_retry_preserves_predecessors_and_passes_only_core_alignment_scope(worker, monkeypatch):
    from mas import retry_authorization

    add_return(worker, "tr")
    assert progressive.run_progressive_worker(**worker.options) == WAIT_PART_RETURN
    child = worker.root / "parts/part-001"
    predecessor_paths = [child / relative for relative in (
        "work/raw_asr_v2.json", "work/audio_review_v2.json",
        "handoff/Muhtemel Ask 15.Bolum_TR_CORRECTION_PACK.zip",
        "handoff/Muhtemel Ask 15.Bolum_TR_CORRECTED.zip")]
    frozen = {path: path.read_bytes() for path in predecessor_paths}
    core = {"stage": "forced_alignment", "target_uids": ["b"], "context_uids": ["a", "b", "c"],
            **{key: "a" * 64 for key in ("audio_sha256", "model_state_sha256", "raw_alignment_binding", "source_sha256")}}
    scope = {**core, "part_id": "part-001", "part_plan_sha256": "b" * 64, "part_audio_lineage_sha256": "c" * 64}
    monkeypatch.setattr(retry_authorization, "validate_part_retry_context", lambda root, episode, value: value == scope)
    monkeypatch.setattr(pipeline, "create_tr_correction_pack", lambda *args, **kwargs: pytest.fail("no TR regeneration"))
    checks = []
    def raw(*args, **kwargs):
        assert kwargs["require_resume"] is True
        checks.append("raw")
        return {**read_json(predecessor_paths[0]), "resumed": True}
    def review(*args, **kwargs):
        assert kwargs["require_resume"] is True
        checks.append("review")
        return read_json(predecessor_paths[1])
    align = pipeline._aligned_checkpoint
    def resume_align(*args, **kwargs):
        assert kwargs["resume_scope"] == core
        checks.append("alignment")
        return align(*args, **kwargs)
    monkeypatch.setattr(pipeline, "transcribe_raw_audio", raw)
    monkeypatch.setattr(pipeline, "resolve_tr_audio_reviews", review)
    monkeypatch.setattr(pipeline, "_aligned_checkpoint", resume_align)
    assert progressive.run_progressive_worker(**worker.options, resume_scope=scope) == WAIT_PART_RETURN
    assert checks == ["raw", "review", "alignment"]
    assert {path: path.read_bytes() for path in predecessor_paths} == frozen


def test_failed_child_stage_pointer_binds_latest_saved_state(worker, monkeypatch):
    add_return(worker, "tr")
    def stage(path, state, name, action):
        progressive.set_stage(path, state, name, "running")
        try:
            result = action()
        except Exception:
            progressive.set_stage(path, state, name, "failed")
            raise
        progressive.set_stage(path, state, name, "pass")
        return result
    def fail(*args, **kwargs):
        raise RuntimeError("bounded alignment conflict")
    monkeypatch.setattr(pipeline, "_stage", stage)
    monkeypatch.setattr(pipeline, "_aligned_checkpoint", fail)
    with pytest.raises(RuntimeError, match="bounded alignment conflict"):
        progressive.run_progressive_worker(**worker.options)
    envelope = read_json(worker.root / "work/current-part.json")
    data = envelope["data"]
    assert envelope["sha256"] == digest(data)
    assert data["stage"] == "forced_alignment" and data["part_id"] == "part-001"
    assert data["part_plan_sha256"] == sha256_file(worker.root / "work/part-plan.json")
    assert data["part_audio_lineage_sha256"] == sha256_file(worker.root / "parts/part-001/work/audio-part.done.json")
    assert data["state"]["relative_path"] == "parts/part-001/work/state.json"
    assert data["state"]["sha256"] == sha256_file(worker.root / data["state"]["relative_path"])


def test_ack_export_and_finalize_share_decreasing_deadline(worker, monkeypatch):
    now, observed = [100.0], []
    monkeypatch.setattr(progressive.time, "monotonic", lambda: now[0])
    def wrap(label, action):
        def timed(*args, **kwargs):
            observed.append((label, kwargs["total_timeout"]))
            now[0] += 1
            return action(*args, **kwargs)
        return timed
    monkeypatch.setattr(partial_delivery, "validate_worker_published_part",
                        wrap("ack", partial_delivery.validate_worker_published_part))
    monkeypatch.setattr(partial_finalize, "finalize_partial_episode",
                        wrap("finalize", partial_finalize.finalize_partial_episode))
    monkeypatch.setattr(partial_finalize, "validate_partial_export",
                        wrap("export", partial_finalize.validate_partial_export))
    add_return(worker, "tr")
    add_return(worker, "id")
    assert progressive.run_progressive_worker(**worker.options) == READY_FOR_PARTIAL_ENCODE
    assert observed == [("ack", 30), ("finalize", 29), ("export", 28), ("export", 27)]
    observed.clear()
    assert progressive.run_progressive_worker(**worker.options) == READY_FOR_PARTIAL_ENCODE
    assert observed == [("ack", 30), ("export", 29)]
    observed.clear()
    worker.published.update(part["part_id"] for part in worker.plan["parts"])
    assert progressive.run_progressive_worker(**worker.options) == NEXT_PART
    assert observed == [("ack", 30), ("ack", 29)]
