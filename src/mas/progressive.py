import os
import time
from pathlib import Path

from .engine.episode_archive import file_record
from .engine.part_scope import get_part, validate_file_record, validate_part_plan
from .reliability import atomic_json, digest, file_digest as sha256_file, read_json
from .state import load, save, set_stage


def _bound_json(path, data, *, immutable=False):
    envelope = {"data": data, "sha256": digest(data)}
    if Path(path).is_symlink():
        raise RuntimeError("unsafe partial manifest path")
    if immutable and Path(path).exists():
        if read_json(path) != envelope:
            raise RuntimeError("frozen partial handoff identity changed; preserve the existing evidence")
    else:
        atomic_json(path, envelope)


def _plan_metadata(root, episode):
    path = root / "work" / "part-plan.json"
    record = file_record(path, root)
    envelope = read_json(path)
    plan = envelope.get("data")
    if not isinstance(plan, dict) or envelope.get("sha256") != digest(plan):
        raise RuntimeError("partial plan metadata binding changed")
    plan = validate_part_plan(plan)
    if plan["episode"] != episode:
        raise RuntimeError("partial plan belongs to another episode")
    return plan, record


def write_partial_handoff(root, episode, part_id, kind, pack, expected_return, files):
    root = Path(root).resolve()
    if kind not in {"tr", "id"}:
        raise RuntimeError("unsupported partial handoff kind")
    plan, plan_record = _plan_metadata(root, episode)
    get_part(plan, part_id)
    child = root / "parts" / part_id
    name = f"Muhtemel Ask {episode}.Bolum"
    pack_name = name + ("_TR_CORRECTION_PACK.zip" if kind == "tr" else "_ID_TRANSLATION_PACK.zip")
    return_name = name + ("_TR_TEXT_CORRECTED.zip" if kind == "tr" else "_ID_TRANSLATED.zip")
    if (Path(pack).resolve() != (child / "handoff" / pack_name).resolve()
            or Path(expected_return).resolve() != (child / "handoff" / return_name).resolve()):
        raise RuntimeError("partial handoff pack or return path changed")
    paths = {Path(path) for path in files}
    paths.update({root / plan_record["relative_path"], Path(pack)})
    records = [file_record(path, root) for path in sorted(paths, key=str)]
    prefix = f"parts/{part_id}/"
    if any(record["relative_path"] != "work/part-plan.json"
           and not record["relative_path"].startswith(prefix) for record in records):
        raise RuntimeError("partial handoff includes files outside its owned scope")
    data = {"format": "mas-partial-handoff-1", "episode": episode, "part_id": part_id, "kind": kind,
            "plan_sha256": plan_record["sha256"], "part_plan": plan_record,
            "pack": file_record(pack, root), "files": records,
            "expected_return": (child / "handoff" / return_name).relative_to(root).as_posix()}
    _bound_json(child / "work" / f"partial-handoff-{kind}.json", data, immutable=True)
    _bound_json(child / "work" / "partial-handoff.json", data)
    _bound_json(root / "work" / "partial-handoff.json", data)
    return data


def validate_partial_handoff(root, episode, part_id=None):
    root = Path(root).resolve()
    plan, plan_record = _plan_metadata(root, episode)
    if part_id is not None:
        get_part(plan, part_id)
    path = (root / "parts" / part_id / "work" / "partial-handoff.json" if part_id is not None
            else root / "work" / "partial-handoff.json")
    file_record(path, root)
    envelope = read_json(path)
    data = envelope.get("data")
    if (not isinstance(data, dict) or envelope.get("sha256") != digest(data)
            or data.get("format") != "mas-partial-handoff-1" or data.get("episode") != episode
            or data.get("kind") not in {"tr", "id"}
            or part_id is not None and data.get("part_id") != part_id):
        raise RuntimeError("partial handoff envelope is invalid")
    part_id = data["part_id"]
    get_part(plan, part_id)
    child = root / "parts" / part_id
    kind = data["kind"]
    for manifest in (child / "work" / "partial-handoff.json", child / "work" / f"partial-handoff-{kind}.json"):
        file_record(manifest, root)
        if read_json(manifest) != envelope:
            raise RuntimeError("partial handoff alias differs from frozen phase evidence")
    if data.get("part_plan") != plan_record or data.get("plan_sha256") != plan_record["sha256"]:
        raise RuntimeError("partial handoff is bound to a different plan")
    name = f"Muhtemel Ask {episode}.Bolum"
    pack_name = name + ("_TR_CORRECTION_PACK.zip" if kind == "tr" else "_ID_TRANSLATION_PACK.zip")
    return_name = name + ("_TR_TEXT_CORRECTED.zip" if kind == "tr" else "_ID_TRANSLATED.zip")
    expected_pack = file_record(child / "handoff" / pack_name, root)
    if (data.get("pack") != expected_pack or data.get("expected_return") !=
            f"parts/{part_id}/handoff/{return_name}"):
        raise RuntimeError("partial handoff pack or return ownership changed")
    records = data.get("files")
    if (not isinstance(records, list) or any(not isinstance(item, dict) for item in records)
            or len({item.get("relative_path") for item in records}) != len(records)
            or expected_pack not in records or plan_record not in records):
        raise RuntimeError("partial handoff file inventory is invalid")
    for record in records:
        validate_file_record(record)
        relative = record.get("relative_path", "")
        if (relative != "work/part-plan.json" and not relative.startswith(f"parts/{part_id}/")
                or file_record(root / relative, root) != record):
            raise RuntimeError("partial handoff file binding changed")
    return data


def run_progressive_worker(root, episode, source_video, audio_path, captions_path=None, *, total_timeout,
                           config_dir, series, names, religious, resume_scope=None):
    from .delivery_first import enabled, run_worker
    if enabled(episode) and resume_scope is None:
        return run_worker(root, episode, source_video, audio_path, captions_path,
                          total_timeout=total_timeout, config_dir=config_dir,
                          series=series, names=names, religious=religious)
    from . import pipeline
    from .delivery import NEXT_PART, READY_FOR_PARTIAL_ENCODE, WAIT_PART_RETURN
    from .engine.part_audio import prepare_episode_parts, extract_part_audio
    from .engine.partial_finalize import finalize_partial_episode, validate_partial_export
    from .partial_delivery import validate_worker_published_part

    root = Path(root).resolve()
    deadline = time.monotonic() + total_timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise RuntimeError("progressive episode budget expired")
        return value

    if resume_scope is not None:
        from .retry_authorization import alignment_scope, validate_part_retry_context
        validate_part_retry_context(root, episode, resume_scope)
    plan = prepare_episode_parts(root, episode, source_video, audio_path, captions_path,
                                 total_timeout=remaining())
    part = None
    for candidate in plan["parts"]:
        if validate_worker_published_part(root, episode, candidate["part_id"], total_timeout=remaining()) is None:
            part = candidate
            break
    if resume_scope is not None and (part is None or part["part_id"] != resume_scope["part_id"]):
        raise RuntimeError("authorized retry part is not the earliest unpublished part")
    if part is None:
        remaining()
        return NEXT_PART
    part_id = part["part_id"]
    child = root / "parts" / part_id
    if (root / "parts").is_symlink() or child.is_symlink():
        raise RuntimeError("partial workspace cannot follow a directory symlink")
    dirs = {name: child / name for name in ("work", "handoff", "output")}
    for directory in dirs.values():
        if directory.is_symlink():
            raise RuntimeError("partial workspace cannot follow a directory symlink")
        directory.mkdir(parents=True, exist_ok=True)
    if (child / "work" / "partial-export.json").is_file():
        export, _ = validate_partial_export(root, episode, part_id, total_timeout=remaining())
        atomic_json(root / "work" / "partial-export.json", export)
        remaining()
        return READY_FOR_PARTIAL_ENCODE
    state_path = dirs["work"] / "state.json"
    state = load(state_path, episode)
    plan_sha = sha256_file(root / "work" / "part-plan.json")
    if state.get("part_id", part_id) != part_id or state.get("part_plan_sha256", plan_sha) != plan_sha:
        raise RuntimeError("partial worker checkpoint scope changed")
    state.update(part_id=part_id, part_plan_sha256=plan_sha,
                 publication_authority="external_controller_verified_delivery_ack")
    save(state_path, state)
    print(f"[PART] {part_id}: processing exact source range {part['start_ms']}..{part['end_ms']} ms", flush=True)

    def current_stage(name):
        data = {"format": "mas-current-part-1", "episode": episode, "part_id": part_id,
                "remote_job_token": os.getenv('MAS_REMOTE_JOB_TOKEN'),
                "part_plan_sha256": plan_sha, "part_audio_lineage_sha256": state.get("part_audio_lineage_sha256"),
                "stage": name, "state": file_record(state_path, root)}
        _bound_json(root / "work/current-part.json", data)

    def stage(name, action):
        remaining()
        def run_action():
            current_stage(name)
            return action()
        try:
            result = pipeline._stage(state_path, state, name, run_action)
            remaining()
            return result
        finally:
            current_stage(name)

    current_stage("part_audio")
    audio = extract_part_audio(root, episode, part_id, total_timeout=remaining())
    state["part_audio_lineage_sha256"] = sha256_file(audio["lineage_path"])
    save(state_path, state)
    name = f"Muhtemel Ask {episode}.Bolum"
    tr_pack = dirs["handoff"] / f"{name}_TR_CORRECTION_PACK.zip"
    tr_text = dirs["handoff"] / f"{name}_TR_TEXT_CORRECTED.zip"
    tr_final = dirs["handoff"] / f"{name}_TR_CORRECTED.zip"
    id_pack = dirs["handoff"] / f"{name}_ID_TRANSLATION_PACK.zip"
    id_output = dirs["handoff"] / f"{name}_ID_TRANSLATED.zip"
    review_path = dirs["work"] / "audio_review_v2.json"
    alignment_path = dirs["work"] / "forced_alignment_v2.json"
    schema_path = dirs["work"] / "aligned_tr_schema_v2.json"
    holder = {}

    def transcribe():
        raw = pipeline.transcribe_raw_audio(
            audio["audio_path"], dirs["work"], episode=episode,
            config=pipeline.RawASRConfig(model_name=series["whisper_model"], allow_cpu_fallback=False,
                extra_audio_review_uids=pipeline._pending_extra_audio_review_uids(tr_pack, tr_text)),
            captions_path=audio["captions_path"], canonical_names=tuple(names["canonical_names"]),
            religious_terms=tuple(item["source"] for item in religious["terms"]),
            **({"require_resume": True} if resume_scope is not None else {}))
        if (raw.get("independent_vad") is not True or raw.get("audio_sha256") != audio["lineage"]["audio"]["sha256"]
                or raw.get("model", {}).get("settings", {}).get("allow_cpu_fallback") is not False):
            raise RuntimeError("partial raw ASR lacks exact child audio, independent VAD or CUDA-only policy")
        holder["raw"] = raw
        return {"resumed": raw.get("resumed", False)}

    stage("raw_asr", transcribe)
    raw = holder["raw"]

    def make_tr_pack():
        if resume_scope is not None:
            manifest = pipeline.read_tr_correction_pack(tr_pack).manifest
            return {"input_sha256": manifest["input_sha256"], "resumed": True}
        manifest = pipeline.create_tr_correction_pack(
            raw["correction_utterances"], raw["speech_hole_records"], tr_pack, episode=episode,
            batch_size=250, speech_hole_audio_root=dirs["work"],
            asr_hallucination_records=raw["asr_hallucination_records"], asr_hallucination_audio_root=dirs["work"],
            rebind_text_output_path=tr_text)
        return {"input_sha256": manifest["input_sha256"]}

    stage("tr_pack", make_tr_pack)
    context_files = [audio["lineage_path"], dirs["work"] / "raw_asr_v2.json",
                     dirs["work"] / "raw_asr_v2.done.json"]
    if not tr_text.is_file():
        write_partial_handoff(root, episode, part_id, "tr", tr_pack, tr_text, context_files)
        set_stage(state_path, state, "tr_return", "blocked", expected=str(tr_text))
        current_stage("tr_return")
        return WAIT_PART_RETURN
    stage("tr_return", lambda: {"records": len(pipeline.validate_tr_correction_output(tr_pack, tr_text).records)})
    overrides_path = dirs["work"] / "audio_review_overrides.json"
    speaker_path = dirs["work"] / "speaker_evidence_v1.json"
    overrides = read_json(overrides_path) if overrides_path.is_file() else None
    speaker = read_json(speaker_path) if speaker_path.is_file() else None

    def review():
        holder["review"] = pipeline.resolve_tr_audio_reviews(
            tr_pack, tr_text, tr_final, review_path, dirs["work"] / "audio_review_v2.recovery.json",
            config=pipeline.AudioReviewConfig(model_name=series["whisper_model"], device="cuda", allow_cpu_fallback=False),
            manual_overrides=overrides, **({"require_resume": True} if resume_scope is not None else {}))
        holder["correction"] = pipeline.validate_tr_correction_output(tr_pack, tr_final)
        return {"review_count": holder["review"]["review_count"]}

    stage("audio_review", review)
    correction = holder["correction"]

    def align():
        pack = pipeline.read_tr_correction_pack(tr_pack)
        inputs = pipeline.correction_records_to_alignment_inputs(
            pack.utterances, correction.records, speech_hole_records=pack.speech_holes,
            asr_hallucination_records=pack.asr_hallucination_records, speaker_evidence=speaker,
            episode=episode, audio_sha256=raw["audio_sha256"])
        holder["aligned"], resumed = pipeline._aligned_checkpoint(
            alignment_path, audio["audio_path"], inputs.alignment_inputs, raw["vad_regions"], correction.output_sha256,
            **({"resume_scope": alignment_scope(resume_scope)} if resume_scope is not None else {}))
        return {"resumed": resumed, "device": "cuda"}

    stage("forced_alignment", align)

    def make_id_pack():
        artifacts = pipeline.build_strict_artifacts(
            raw, correction.records, holder["aligned"], episode=episode, acoustic_audio_review=holder["review"],
            speaker_evidence=speaker, production_policy=pipeline.build_production_translation_policy(series, names, religious),
            parent_vad_regions=audio["parent_vad_regions"], part_lineage=audio["lineage"])
        atomic_json(schema_path, artifacts.schema)
        atomic_json(dirs["work"] / "alignment_window_audit_v2.json", list(artifacts.preparation.window_audit))
        atomic_json(dirs["work"] / "final_speech_coverage_v2.json", artifacts.speech_coverage_report)
        atomic_json(dirs["work"] / "pre_id_timing_qa_v2.json", artifacts.timing_qa_report)
        holder["manifest"] = pipeline.create_id_translation_pack(
            artifacts, id_pack, batch_size=series["batch_size"],
            glossary=artifacts.schema["production_policy"]["glossary"])
        holder["workspace"] = pipeline.prepare_id_translation_workspaces(id_pack, dirs["handoff"] / "id-workers")
        holder["artifacts"] = artifacts
        return {"schema_sha256": artifacts.schema["schema_sha256"]}

    stage("id_pack", make_id_pack)
    if not id_output.is_file():
        context_files.extend([tr_pack, tr_text, tr_final, review_path, alignment_path,
            alignment_path.with_suffix(".done.json"), schema_path,
            dirs["work"] / "final_speech_coverage_v2.json", dirs["work"] / "pre_id_timing_qa_v2.json"])
        workspace_path = Path(holder["workspace"])
        context_files.append(workspace_path)
        context_files.extend(workspace_path.parent / f"worker-{index:02d}" / "input.json" for index in range(1, 4))
        write_partial_handoff(root, episode, part_id, "id", id_pack, id_output, context_files)
        set_stage(state_path, state, "id_return", "blocked", expected=str(id_output))
        current_stage("id_return")
        return WAIT_PART_RETURN

    def validate_id():
        pipeline.validate_id_workspace_output(id_pack, id_output)
        result = pipeline.load_and_validate_id_translation_zip(holder["artifacts"].schema, id_output,
                                                              input_manifest=holder["manifest"])
        pipeline._validate_id_quality(holder["artifacts"], result, series, names, religious)
        return {"records": result.output_block_count}

    stage("id_return", validate_id)
    def finalize():
        finalize_partial_episode(root, episode, part_id, config_dir=config_dir, total_timeout=remaining())
        validate_partial_export(root, episode, part_id, total_timeout=remaining())
        return {"part_id": part_id, "mode": "strict-partial-subtitles"}

    stage("strict_partial_finalize", finalize)
    validate_partial_export(root, episode, part_id, total_timeout=remaining())
    remaining()
    return READY_FOR_PARTIAL_ENCODE
