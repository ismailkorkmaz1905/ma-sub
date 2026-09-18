import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from mas import retry_authorization as retry
from mas.reliability import BudgetExceeded, atomic_json, digest


def _case(tmp_path, monkeypatch, *, outcome="pass", bound=True):
    repo, root = tmp_path / "repo", tmp_path / "episode"
    for relative in ("src/mas/fix.py", "tests/test_retained.py", "requirements.lock", "config/runtime_policy.json"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# local fixture\n", encoding="utf-8")
    monkeypatch.setattr(retry, "ROOT", repo)
    start = datetime.now(timezone.utc).isoformat()
    ledger = {"episode": 14, "started_at": start, "limit_seconds": 21600}
    atomic_json(root / "work/controller_budget.json", {"data": ledger, "sha256": digest(ledger)})
    request = {"episode": 14, "commit": "a" * 40, "input_sha256": "a" * 64}
    atomic_json(root / "work/remote-job-request.json", {"data": request, "sha256": digest(request)})
    failure = {"request_sha256": digest(request), "evidence_input_sha256": "e" * 64}
    atomic_json(root / "work/remote-failure-invariant.json", {"data": failure, "sha256": digest(failure)})
    for relative in ("source/download.done.json", "prepare/audio.done.json", "prepare/raw_asr_v2.done.json",
                     "prepare/audio_review_v2.json"):
        atomic_json(root / relative, {"test_checkpoint": True})
    for relative in (
        "prepare/raw_asr_v2.json",
        "translation_input/Muhtemel Ask 14.Bolum_TR_CORRECTION_PACK.zip",
        "translation_output/Muhtemel Ask 14.Bolum_TR_TEXT_CORRECTED.zip",
        "translation_output/Muhtemel Ask 14.Bolum_TR_CORRECTED.zip",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained checkpoint")
    identity = {name: value * 64 for name, value in (
        ("audio_sha256", "a"), ("model_state_sha256", "b"),
        ("raw_alignment_binding", "c"), ("source_sha256", "d"))}
    source = [{"utterance_uid": uid, "start_ms": index * 1000, "end_ms": index * 1000 + 900,
               "coarse_start_ms": index * 1000, "coarse_end_ms": index * 1000 + 900, "text": uid}
              for index, uid in enumerate(("a", "b", "c"), start=1)]
    identity["source_sha256"] = digest(source)
    scope = dict(identity, stage="forced_alignment", target_uids=["b"], context_uids=["a", "b", "c"])
    identity.update(source=source, source_uids=["a", "b", "c"], stage="forced_alignment")
    identity_path = "prepare/forced_alignment_units/resume-identity.json"
    atomic_json(root / identity_path, {"data": identity, "sha256": digest(identity)})
    record = retry._record(root, identity_path)
    component_path = "prepare/forced_alignment_units/components/latest-conflict-failure.json"
    component = {"status": "BLOCKED", "component_uids": ["b"], "source_sha256": digest([source[1]])}
    atomic_json(root / component_path, {"data": component, "sha256": digest(component)})
    def execute(command, **kwargs):
        assert command[:4] == [retry.sys.executable, "-m", "pytest", "-q"]
        assert command[-1] == "tests/test_retained.py::test_regression"
        report = ET.Element("testsuite")
        case = ET.SubElement(report, "testcase", name="test_regression")
        if outcome != "pass":
            ET.SubElement(case, outcome)
        if bound:
            properties = ET.SubElement(case, "properties")
            for name in ("failure", "scope", "diagnostics"):
                key = f"mas_retry_{name}_sha256"
                ET.SubElement(properties, "property", name=key, value=kwargs["env"][key.upper()])
        ET.ElementTree(report).write(command[4].split("=", 1)[1])
        return SimpleNamespace(returncode=1 if outcome in {"failure", "error"} else 0)
    monkeypatch.setattr(retry.subprocess, "run", execute)
    arguments = dict(fixture_nodeids=["tests/test_retained.py::test_regression"],
                     diagnostic_records=[record, retry._record(root, component_path)], resume_scope=scope)
    return repo, root, arguments


def test_code_fix_authorization_requires_actual_bound_fixture_pass(tmp_path, monkeypatch):
    _, root, arguments = _case(tmp_path, monkeypatch)
    permit = retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)
    assert retry.validate_code_fix_resume(root, 14, "b" * 40, permit) == arguments["resume_scope"]
    saved = json.loads(permit.read_text())
    assert saved["data"]["fixture_passed_count"] == 1
    predecessor_paths = {
        item["relative_path"] for item in saved["data"]["predecessors"]
    }
    assert "prepare/raw_asr_v2.json" in predecessor_paths
    assert (
        "translation_output/Muhtemel Ask 14.Bolum_TR_CORRECTED.zip"
        in predecessor_paths
    )


def test_source_identity_normalizes_platform_line_endings(tmp_path):
    windows = tmp_path / "windows"
    linux = tmp_path / "linux"
    relative = "tests/test_retained.py"
    for root, content in ((windows, b"first\r\nsecond\r\n"),
                          (linux, b"first\nsecond\n")):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(content)

    assert retry._source_record(windows, relative) == retry._source_record(
        linux, relative
    )


def test_code_fix_permit_binds_bounded_discovered_component_limit(
        tmp_path, monkeypatch):
    _, root, arguments = _case(tmp_path, monkeypatch)
    arguments["resume_scope"]["discovery_component_limit"] = 3

    permit = retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)

    scope = retry.validate_code_fix_resume(root, 14, "b" * 40, permit)
    assert scope["discovery_component_limit"] == 3


def test_code_fix_permit_binds_signed_continuation_and_recovery_limits(
        tmp_path, monkeypatch):
    _, root, arguments = _case(tmp_path, monkeypatch)
    arguments["resume_scope"].update(
        continuation_component_limit=256,
        recovery_seconds_limit=10800,
    )

    permit = retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)

    scope = retry.validate_code_fix_resume(root, 14, "b" * 40, permit)
    assert scope["continuation_component_limit"] == 256
    assert scope["recovery_seconds_limit"] == 10800


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_failed_or_skipped_fixture_never_authorizes(tmp_path, monkeypatch, outcome):
    _, root, arguments = _case(tmp_path, monkeypatch, outcome=outcome)
    with pytest.raises(retry.RetryAuthorizationError, match="pass|skipped"):
        retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)
    assert not (root / "review/code-fix-resume.json").exists()


def test_unrelated_passing_test_without_failure_binding_never_authorizes(tmp_path, monkeypatch):
    _, root, arguments = _case(tmp_path, monkeypatch, bound=False)
    with pytest.raises(retry.RetryAuthorizationError, match="attest"):
        retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)


@pytest.mark.parametrize("changed", ["code", "fixture", "diagnostic", "commit", "deadline"])
def test_permit_cannot_rebind_changed_evidence_or_restart_wall_clock(tmp_path, monkeypatch, changed):
    repo, root, arguments = _case(tmp_path, monkeypatch)
    retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)
    commit = "b" * 40
    if changed == "code":
        (repo / "src/mas/fix.py").write_text("changed", encoding="utf-8")
    elif changed == "fixture":
        (repo / "tests/test_retained.py").write_text("changed", encoding="utf-8")
    elif changed == "diagnostic":
        (root / arguments["diagnostic_records"][0]["relative_path"]).write_text("changed", encoding="utf-8")
    elif changed == "commit":
        commit = "c" * 40
    else:
        path = root / "work/controller_budget.json"
        body = retry._read_bound(path)
        body["started_at"] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
        atomic_json(path, {"data": body, "sha256": digest(body)})
    with pytest.raises((retry.RetryAuthorizationError, BudgetExceeded)):
        retry.validate_code_fix_resume(root, 14, commit)


def test_scope_cannot_claim_unobserved_model_or_exclude_target(tmp_path, monkeypatch):
    _, root, arguments = _case(tmp_path, monkeypatch)
    arguments["resume_scope"]["model_state_sha256"] = "f" * 64
    with pytest.raises(retry.RetryAuthorizationError, match="alignment identity"):
        retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)
    arguments["resume_scope"]["model_state_sha256"] = "b" * 64
    arguments["resume_scope"]["context_uids"] = ["a"]
    with pytest.raises(retry.RetryAuthorizationError, match="bounded context"):
        retry.authorize_code_fix_retry(root, 14, "b" * 40, **arguments)


def _part_case(tmp_path, monkeypatch):
    from mas.engine import raw_asr as raw_asr_module
    from mas.engine.part_scope import build_part_plan, project_part_vad
    from mas.engine.raw_asr import RawASRV2Config
    from mas.engine.speech_coverage import SpeechCoverageConfig
    from mas.engine.tr_correction import create_tr_correction_pack, create_tr_correction_output, validate_tr_correction_output
    from mas.engine.audio_review import resolve_tr_audio_reviews_v2
    from mas.engine.workflow import correction_records_to_alignment_inputs
    from mas.engine.forced_align import validate_coarse_segments
    from mas.engine.download import write_stage_marker

    repo, root, arguments = _case(tmp_path, monkeypatch)
    part_id = "part-001"
    child = root / "parts" / part_id
    audio = {"relative_path": "prepare/audio.flac", "sha256": "f" * 64, "size_bytes": 100,
             "sample_rate_hz": 16000, "channels": 1, "sample_count": 10000 * 16}
    regions = [{"vad_region_index": index, "start_ms": index * 1000, "end_ms": index * 1000 + 900,
                "source": "silero_vad"} for index in range(1, 4)]
    plan = build_part_plan(episode=14, source={"relative_path": "source/movie.mp4", "sha256": "e" * 64, "size_bytes": 100},
        audio=audio, vad={"independent_vad": True, "audio_sha256": "f" * 64, "sample_count": audio["sample_count"],
                         "config": {"policy": "canonical"}, "model": {"engine": "silero"},
                         "producer_sha256": "1" * 64, "regions": regions})
    for name, data in (("part-plan.json", plan), ("part-vad.json", plan["vad"])):
        atomic_json(root / "work" / name, {"data": data, "sha256": digest(data)})
    part = plan["parts"][0]
    lineage = {"format": "mas-derived-audio-part-1", "episode": 14, "part_id": part_id,
               "plan_sha256": digest(plan), "parent_source_sha256": "e" * 64, "parent_audio_sha256": "f" * 64,
               **{key: part[key] for key in ("start_ms", "end_ms", "start_sample", "end_sample")},
               "sample_rate_hz": 16000, "sample_count": audio["sample_count"],
               "parent_vad_sha256": digest(project_part_vad(plan, part_id)), "captions": None,
               "audio": {"relative_path": f"parts/{part_id}/prepare/audio.flac", "sha256": "a" * 64, "size_bytes": 80},
               "producer_sha256": "2" * 64, "ffmpeg_identity": {"version": "fixture"}, "pcm_sha256": "3" * 64}
    atomic_json(child / "prepare/audio-part.done.json", {"data": lineage, "sha256": digest(lineage)})
    utterances = [{"utterance_uid": uid, "utterance_index": index,
                  "coarse_start_ms": index * 1000, "coarse_end_ms": index * 1000 + 900,
                  "asr_text": uid, "youtube_text": "", "context_before": "", "context_after": "", "risk_flags": [],
                  "asr_audit": {"avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.0, "temperature": 0.0}}
                 for index, uid in enumerate(("a", "b", "c"), 1)]
    segments = [
        {
            "segment_id": f"main-{index}",
            "start_ms": item["coarse_start_ms"],
            "end_ms": item["coarse_end_ms"],
            "text": item["asr_text"],
            "source": "main",
            "word_timing_complete": True,
            "words": [
                {
                    "start_ms": item["coarse_start_ms"],
                    "end_ms": item["coarse_end_ms"],
                    "text": f" {item['asr_text']}",
                }
            ],
        }
        for index, item in enumerate(utterances, 1)
    ]
    words = [
        {**word, "segment_id": segment["segment_id"]}
        for segment in segments
        for word in segment["words"]
    ]
    settings = RawASRV2Config(allow_cpu_fallback=False)
    initial = raw_asr_module._analyze_raw_speech_coverage(
        regions,
        segments,
        words,
        config=settings.speech_coverage_config(),
    )
    batches, budget = raw_asr_module._plan_rescue_batches(
        initial, regions, segments, words, settings
    )
    raw = {"format_version": "2.0", "status": "completed", "input_sha256": "4" * 64, "episode": 14,
           "audio_sha256": "a" * 64, "language": "tr", "synthetic_timing_count": 0, "vad_fallback_reason": None,
           "independent_vad": True, "model": {"settings": asdict(settings)},
           "hallucination_review_utterance_uids": [], "vad_regions": regions, "correction_utterances": utterances,
           "segments": segments, "words": words, "required_acoustic_review_regions": [],
           "initial_speech_coverage": initial, "rescue_batches": batches, "rescue_budget_audit": budget,
           "speech_hole_records": [], "asr_hallucination_records": [],
           "speech_coverage": {"config": asdict(SpeechCoverageConfig())}}
    raw_path = child / "prepare/raw_asr_v2.json"
    atomic_json(raw_path, raw)
    write_stage_marker(child / "prepare/raw_asr_v2.done.json", stage="raw_asr_v2", input_sha256=raw["input_sha256"],
                       outputs={"raw_asr_v2": raw_path}, details={"audio_sha256": "a" * 64})
    name = "Muhtemel Ask 14.Bolum"
    pack = child / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip"
    text = child / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
    final = child / "translation_output" / f"{name}_TR_CORRECTED.zip"
    create_tr_correction_pack(utterances, [], pack, episode=14)
    corrected = [{**record, "tr_corrected": record["asr_text"], "review_required": False, "non_dialogue": False,
                  "note": "", "audio_reviewed": False, "review_disposition": "not_applicable"} for record in utterances]
    create_tr_correction_output(pack, corrected, text)
    resolve_tr_audio_reviews_v2(pack, text, final, child / "prepare/audio_review_v2.json",
                                child / "prepare/audio_review_v2.recovery.json", progress=None)
    bundle = correction_records_to_alignment_inputs(utterances, validate_tr_correction_output(pack, final).records)
    source = validate_coarse_segments(bundle.alignment_inputs)
    identity = retry._read_bound(root / arguments["diagnostic_records"][0]["relative_path"])
    identity.update(source=source, source_sha256=digest(source))
    prefix = f"parts/{part_id}/prepare/forced_alignment_units/"
    identity_path = prefix + "resume-identity.json"
    component_path = prefix + "components/latest-conflict-failure.json"
    component = {"status": "BLOCKED", "component_uids": ["b"], "source_sha256": digest([source[1]])}
    for path, data in ((identity_path, identity), (component_path, component)):
        atomic_json(root / path, {"data": data, "sha256": digest(data)})
    scope = {**arguments["resume_scope"], "source_sha256": digest(source), "part_id": part_id,
             "part_plan_sha256": retry.sha256_file(root / "work/part-plan.json"),
             "part_audio_lineage_sha256": retry.sha256_file(child / "prepare/audio-part.done.json")}
    failure = retry._read_bound(root / "work/remote-failure-invariant.json")
    failure.update({key: scope[key] for key in retry._PART_SCOPE_FIELDS}, part_stage="forced_alignment")
    atomic_json(root / "work/remote-failure-invariant.json", {"data": failure, "sha256": digest(failure)})
    arguments.update(resume_scope=scope, diagnostic_records=[retry._record(root, identity_path), retry._record(root, component_path)])
    return repo, root, arguments


def test_part_retry_validates_frozen_child_predecessors_without_parent_media(tmp_path, monkeypatch):
    _, root, args = _part_case(tmp_path, monkeypatch)
    permit = retry.authorize_code_fix_retry(root, 14, "b" * 40, **args)
    assert retry.validate_code_fix_resume(root, 14, "b" * 40, permit) == args["resume_scope"]
    assert "part_id" not in retry.alignment_scope(args["resume_scope"])
    paths = {item["relative_path"] for item in retry._read_bound(permit)["predecessors"]}
    assert "prepare/raw_asr_v2.done.json" not in paths
    assert "parts/part-001/prepare/raw_asr_v2.json" in paths
    assert "parts/part-001/prepare/audio-part.done.json" in paths
    assert not (root / "prepare/audio.flac").exists()


@pytest.mark.parametrize("change", ["part", "stage", "lineage", "raw", "tr", "diagnostic"])
def test_part_retry_rejects_cross_scope_or_changed_predecessor(tmp_path, monkeypatch, change):
    _, root, args = _part_case(tmp_path, monkeypatch)
    if change in {"part", "stage"}:
        failure_path = root / "work/remote-failure-invariant.json"
        failure = retry._read_bound(failure_path)
        failure["part_id" if change == "part" else "part_stage"] = "part-002" if change == "part" else "id_return"
        atomic_json(failure_path, {"data": failure, "sha256": digest(failure)})
    elif change == "diagnostic":
        args["diagnostic_records"][0] = retry._record(root, "prepare/forced_alignment_units/resume-identity.json")
    else:
        relative = {"lineage": "prepare/audio-part.done.json", "raw": "prepare/raw_asr_v2.json",
                    "tr": "translation_output/Muhtemel Ask 14.Bolum_TR_CORRECTED.zip"}[change]
        (root / "parts/part-001" / relative).write_bytes(b"changed evidence")
    with pytest.raises((retry.RetryAuthorizationError, ValueError, RuntimeError)):
        retry.authorize_code_fix_retry(root, 14, "b" * 40, **args)
    assert not (root / "review/code-fix-resume.json").exists()


def test_same_failed_revision_cannot_receive_code_fix_permit(tmp_path, monkeypatch):
    _, root, args = _case(tmp_path, monkeypatch)
    with pytest.raises(retry.RetryAuthorizationError, match="unchanged failed revision"):
        retry.authorize_code_fix_retry(root, 14, "a" * 40, **args)


@pytest.mark.parametrize("relative,new", [
    ("prepare/raw_asr_v2.json", False),
    ("prepare/audio_review_v2.json", False),
    ("review/audio_review_overrides.json", True),
    ("review/speaker_evidence_v1.json", True),
])
def test_part_permit_rejects_changed_or_new_child_evidence(tmp_path, monkeypatch, relative, new):
    _, root, args = _part_case(tmp_path, monkeypatch)
    permit = retry.authorize_code_fix_retry(root, 14, "b" * 40, **args)
    path = root / "parts/part-001" / relative
    assert path.exists() is not new
    atomic_json(path, {"changed": True})
    with pytest.raises(retry.RetryAuthorizationError, match="checkpoint|changed"):
        retry.validate_code_fix_resume(root, 14, "b" * 40, permit)


def _closure_case(tmp_path, monkeypatch):
    from mas.engine.forced_align import validate_coarse_segments
    _, root, arguments = _case(tmp_path, monkeypatch)
    identity_path = root / arguments['diagnostic_records'][0]['relative_path']
    failure_path = root / arguments['diagnostic_records'][1]['relative_path']
    identity = retry._read_bound(identity_path)
    source = validate_coarse_segments([
        {**item, 'asr_text': item['text'], 'deletion_audio_reviewed': False}
        for item in identity['source']])
    identity.update(source=source, source_sha256=digest(source))
    atomic_json(identity_path, {'data': identity, 'sha256': digest(identity)})
    failure = retry._read_bound(failure_path)
    failure['source_sha256'] = digest([source[1]])
    atomic_json(failure_path, {'data': failure, 'sha256': digest(failure)})
    arguments['resume_scope']['source_sha256'] = digest(source)
    arguments['diagnostic_records'] = [retry._record(root, record['relative_path'])
                                        for record in arguments['diagnostic_records']]
    return root, arguments, source


def test_offline_closure_proposal_is_not_a_gpu_permit(tmp_path, monkeypatch):
    root, arguments, source = _closure_case(tmp_path, monkeypatch)
    path = retry.propose_alignment_recovery(root, 14, max_new_ctc_calls=4)
    body = retry._read_bound(path)
    assert body['status'] == 'PROPOSAL_NOT_AUTHORIZED'
    assert body['closure_cue_count'] == 3
    assert body['scope']['recovery_plan']['components'] == [['a', 'b', 'c']]
    assert not (root / 'review/code-fix-resume.json').exists()
    arguments['resume_scope'] = body['scope']
    permit = retry.authorize_code_fix_retry(root, 14, 'b' * 40, **arguments)
    assert retry.validate_code_fix_resume(root, 14, 'b' * 40, permit) == body['scope']
    with pytest.raises(retry.RetryAuthorizationError, match='proposal episode'):
        retry.propose_alignment_recovery(root, 15, max_new_ctc_calls=4)


def test_planning_cli_never_starts_provider_or_sends_mail_on_missing_evidence(tmp_path, monkeypatch, capsys):
    from mas import cli
    monkeypatch.setattr(cli, 'episode_dir', lambda episode: tmp_path)
    monkeypatch.setattr(cli, 'run_remote_episode', lambda *args: pytest.fail('unexpected paid run'))
    monkeypatch.setattr(cli, 'enqueue_notification', lambda *args, **kwargs: pytest.fail('unexpected mail'))
    assert cli.main(['plan-alignment-recovery', '14', '--max-new-ctc-calls', '64']) == 1
    captured = capsys.readouterr()
    assert 'OFFLINE PROPOSAL ONLY' in captured.err
    assert 'SAFE RETRY:' not in captured.err
