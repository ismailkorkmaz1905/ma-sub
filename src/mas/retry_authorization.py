import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import ROOT
from .delivery import safe_relative
from .hashing import sha256_file
from .reliability import RunBudget, atomic_json, digest


class RetryAuthorizationError(RuntimeError):
    pass


_ALIGNMENT_SCOPE_FIELDS = {"stage", "target_uids", "context_uids", "audio_sha256", "model_state_sha256",
                           "raw_alignment_binding", "source_sha256"}
_ALIGNMENT_SCOPE_OPTIONAL_FIELDS = {"discovery_component_limit"}
_PART_SCOPE_FIELDS = {"part_id", "part_plan_sha256", "part_audio_lineage_sha256"}


def alignment_scope(scope):
    allowed = (
        _ALIGNMENT_SCOPE_FIELDS,
        _ALIGNMENT_SCOPE_FIELDS | _ALIGNMENT_SCOPE_OPTIONAL_FIELDS,
        _ALIGNMENT_SCOPE_FIELDS | _PART_SCOPE_FIELDS,
        _ALIGNMENT_SCOPE_FIELDS | _ALIGNMENT_SCOPE_OPTIONAL_FIELDS | _PART_SCOPE_FIELDS,
    )
    if not isinstance(scope, dict) or set(scope) not in allowed:
        raise RetryAuthorizationError("retry scope fields are invalid")
    if "discovery_component_limit" in scope and (
        type(scope["discovery_component_limit"]) is not int
        or not 1 <= scope["discovery_component_limit"] <= 3
    ):
        raise RetryAuthorizationError("retry discovery component limit is invalid")
    if "part_id" in scope and not re.fullmatch(r"part-[0-9]{3}", str(scope["part_id"])):
        raise RetryAuthorizationError("retry part identity is invalid")
    return {
        key: scope[key]
        for key in _ALIGNMENT_SCOPE_FIELDS | _ALIGNMENT_SCOPE_OPTIONAL_FIELDS
        if key in scope
    }


def retry_diagnostic_layout(episode, part_id=None):
    if part_id is not None and not re.fullmatch(r"part-[0-9]{3}", str(part_id)):
        raise RetryAuthorizationError("retry part identity is invalid")
    prefix = f"parts/{part_id}/" if part_id is not None else ""
    name = f"Muhtemel Ask {episode}.Bolum"
    files = ["source/download.done.json", "prepare/audio.done.json",
             prefix + "prepare/raw_asr_v2.done.json",
             prefix + "prepare/raw_asr_v2.json",
             prefix + "prepare/audio_review_v2.json",
             prefix + f"translation_input/{name}_TR_CORRECTION_PACK.zip",
             prefix + f"translation_output/{name}_TR_TEXT_CORRECTED.zip",
             prefix + f"translation_output/{name}_TR_CORRECTED.zip"]
    if part_id is not None:
        files.extend(["work/part-plan.json", "work/part-vad.json", "work/current-part.json",
                      prefix + "work/state.json", prefix + "prepare/audio-part.done.json",
                      prefix + "review/audio_review_overrides.json", prefix + "review/speaker_evidence_v1.json"])
    return {"files": tuple(files), "prefixes": (prefix + "prepare/forced_alignment_units/",)}


def retry_predecessor_paths(episode, scope, local_root=None):
    alignment_scope(scope)
    part_id = scope.get("part_id")
    files = retry_diagnostic_layout(episode, part_id)["files"]
    if part_id is None:
        return files
    optional = {f"parts/{part_id}/review/{name}" for name in ("audio_review_overrides.json", "speaker_evidence_v1.json")}
    return tuple(path for path in files if path not in {"work/current-part.json", f"parts/{part_id}/work/state.json"}
                 and (path not in optional or local_root is not None and safe_relative(local_root, path).is_file()))


def validate_part_retry_context(local_root, episode, scope):
    alignment_scope(scope)
    if "part_id" not in scope:
        raise RetryAuthorizationError("first-hour retry requires an exact authorized part")
    from .engine.part_audio import load_part_plan
    from .engine.part_scope import validate_part_lineage
    from .engine.tr_correction import read_tr_correction_pack, validate_tr_correction_output
    from .engine.audio_review import validate_audio_review_v2_report
    from .engine.workflow import _validate_raw_vad_inventory, correction_records_to_alignment_inputs
    from .engine.forced_align import validate_coarse_segments

    root = Path(local_root)
    part_id = scope["part_id"]
    plan = load_part_plan(root, episode, verify_files=False)
    lineage_path = safe_relative(root, f"parts/{part_id}/prepare/audio-part.done.json")
    lineage = validate_part_lineage(plan, part_id, _read_bound(lineage_path))
    if (sha256_file(root / "work/part-plan.json") != scope["part_plan_sha256"]
            or sha256_file(lineage_path) != scope["part_audio_lineage_sha256"]
            or _read_bound(root / "work/part-vad.json") != plan["vad"]
            or lineage["audio"]["sha256"] != scope["audio_sha256"]):
        raise RetryAuthorizationError("retry part plan or child audio lineage changed")
    child = root / "parts" / part_id
    raw_path = child / "prepare/raw_asr_v2.json"
    raw = _validate_raw_vad_inventory(json.loads(raw_path.read_text(encoding="utf-8")), episode=episode)
    marker = json.loads((child / "prepare/raw_asr_v2.done.json").read_text(encoding="utf-8"))
    output = marker.get("outputs", {}).get("raw_asr_v2", {})
    if (raw["audio_sha256"] != scope["audio_sha256"]
            or raw["model"]["settings"].get("allow_cpu_fallback") is not False
            or marker.get("stage") != "raw_asr_v2" or marker.get("input_sha256") != raw["input_sha256"]
            or output.get("sha256") != sha256_file(raw_path) or output.get("size_bytes") != raw_path.stat().st_size
            or marker.get("details", {}).get("audio_sha256") != scope["audio_sha256"]):
        raise RetryAuthorizationError("retry raw-ASR predecessor is not the exact child checkpoint")
    name = f"Muhtemel Ask {episode}.Bolum"
    pack_path = child / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip"
    text_path = child / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
    final_path = child / "translation_output" / f"{name}_TR_CORRECTED.zip"
    pack = read_tr_correction_pack(pack_path)
    if (pack.manifest["episode"] != episode or list(pack.utterances) != raw["correction_utterances"]
            or list(pack.speech_holes) != raw["speech_hole_records"]
            or list(pack.asr_hallucination_records) != raw["asr_hallucination_records"]):
        raise RetryAuthorizationError("retry Turkish pack is not bound to frozen child raw ASR")
    validate_audio_review_v2_report(pack_path, text_path, final_path, child / "prepare/audio_review_v2.json")
    final = validate_tr_correction_output(pack_path, final_path)
    speaker_path = child / "review/speaker_evidence_v1.json"
    speaker = json.loads(speaker_path.read_text(encoding="utf-8")) if speaker_path.is_file() else None
    bundle = correction_records_to_alignment_inputs(pack.utterances, final.records,
        speech_hole_records=pack.speech_holes, asr_hallucination_records=pack.asr_hallucination_records,
        speaker_evidence=speaker, episode=episode, audio_sha256=scope["audio_sha256"])
    if digest(validate_coarse_segments(bundle.alignment_inputs)) != scope["source_sha256"]:
        raise RetryAuthorizationError("retry alignment source differs from frozen child Turkish review")
    return lineage


def _part_failure_scope(scope, failure):
    if "part_id" in scope:
        if (failure.get("part_stage") != "forced_alignment"
                or any(failure.get(key) != scope[key] for key in _PART_SCOPE_FIELDS)):
            raise RetryAuthorizationError("retry part is not the currently retained forced-alignment failure")
    elif any(key in failure for key in _PART_SCOPE_FIELDS):
        raise RetryAuthorizationError("partial failure cannot authorize whole-episode retry")


def _read_bound(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    data = saved.get("data")
    if not isinstance(data, dict) or saved.get("sha256") != digest(data):
        raise RetryAuthorizationError(f"retry evidence integrity mismatch: {Path(path).name}")
    return data


def _record(root, relative):
    path = safe_relative(root, relative)
    if not path.is_file() or path.is_symlink():
        raise RetryAuthorizationError("retry evidence must be a regular retained file")
    return {"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _source_record(root, relative):
    path = safe_relative(root, relative)
    if not path.is_file() or path.is_symlink():
        raise RetryAuthorizationError("retry source must be a regular retained file")
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return {
        "relative_path": relative,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _code_identity():
    paths = sorted((ROOT / "src/mas").rglob("*.py"))
    paths += sorted((ROOT / "config/production").glob("*.yaml"))
    paths += [ROOT / "requirements.lock", ROOT / "config/runtime_policy.json"]
    return {
        path.relative_to(ROOT).as_posix(): _source_record(
            ROOT, path.relative_to(ROOT).as_posix()
        )["sha256"]
        for path in paths
    }


def _budget(root, episode):
    data = _read_bound(root / "work/controller_budget.json")
    if data.get("episode") != episode or not 0 < data.get("limit_seconds", 0) <= 21600:
        raise RetryAuthorizationError("retry episode budget identity or hard limit mismatch")
    budget = RunBudget(data["started_at"], limit_seconds=data["limit_seconds"])
    budget.check()
    return data["started_at"], budget


def _scope(scope, identity, component_failure):
    core = alignment_scope(scope)
    keys = _ALIGNMENT_SCOPE_FIELDS
    if scope["stage"] != "forced_alignment":
        raise RetryAuthorizationError("retry scope must identify only forced-alignment components")
    for name in ("target_uids", "context_uids"):
        values = scope[name]
        if (not isinstance(values, list) or not values or len(values) != len(set(values))
                or any(not isinstance(uid, str) or not uid for uid in values)):
            raise RetryAuthorizationError("retry component UIDs must be nonempty and unique")
    if not set(scope["target_uids"]).issubset(scope["context_uids"]):
        raise RetryAuthorizationError("retry targets must belong to the bounded context")
    for name in keys - {"stage", "target_uids", "context_uids"}:
        if not re.fullmatch(r"[0-9a-f]{64}", scope[name] or "") or scope[name] != identity.get(name):
            raise RetryAuthorizationError("retry scope does not match retained alignment identity")
    if (component_failure.get("status") != "BLOCKED"
            or scope["target_uids"] != component_failure.get("component_uids")):
        raise RetryAuthorizationError("retry targets must be the exact retained failed component")
    source = identity.get("source", [])
    targeted = [item for item in source if item.get("utterance_uid") in scope["target_uids"]]
    if digest(source) != scope["source_sha256"] or digest(targeted) != component_failure.get("source_sha256"):
        raise RetryAuthorizationError("retained failed component source binding mismatch")
    from .engine.forced_align import _alignment_resume_groups
    _alignment_resume_groups(core, source, {name: identity[name] for name in
                                            ("audio_sha256", "model_state_sha256", "raw_alignment_binding", "source_sha256")})


def authorize_code_fix_retry(local_root, episode, commit, *, fixture_nodeids, diagnostic_records, resume_scope):
    local_root = Path(local_root)
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise RetryAuthorizationError("retry commit must be a full Git SHA")
    started_at, budget = _budget(local_root, episode)
    failure_path = local_root / "work/remote-failure-invariant.json"
    failure = _read_bound(failure_path)
    request = _read_bound(local_root / "work/remote-job-request.json")
    if request.get("episode") != episode or failure.get("request_sha256") != digest(request):
        raise RetryAuthorizationError("retry failure is not bound to the episode request")
    if not failure.get("evidence_input_sha256"):
        raise RetryAuthorizationError("legacy failure has no validated input binding")
    alignment_scope(resume_scope)
    _part_failure_scope(resume_scope, failure)
    if request.get("commit") == commit:
        raise RetryAuthorizationError("unchanged failed revision cannot authorize another GPU retry")
    predecessors = [_record(local_root, relative) for relative in retry_predecessor_paths(episode, resume_scope, local_root)]
    if "part_id" in resume_scope:
        validate_part_retry_context(local_root, episode, resume_scope)
    retained = []
    for record in diagnostic_records:
        actual = _record(local_root, record["relative_path"])
        if actual != record:
            raise RetryAuthorizationError("retained failure fixture changed")
        retained.append(actual)
    identities = [record for record in retained if Path(record["relative_path"]).name == "resume-identity.json"]
    components = [record for record in retained if Path(record["relative_path"]).name == "latest-conflict-failure.json"]
    if len(identities) != 1 or len(components) != 1:
        raise RetryAuthorizationError("one retained alignment identity and failed component are required")
    prefix = retry_diagnostic_layout(episode, resume_scope.get("part_id"))["prefixes"][0]
    if (identities[0]["relative_path"] != prefix + "resume-identity.json"
            or components[0]["relative_path"] != prefix + "components/latest-conflict-failure.json"
            or any(not record["relative_path"].startswith(prefix) for record in retained)):
        raise RetryAuthorizationError("retry diagnostic namespace differs from the authorized part")
    identity = _read_bound(safe_relative(local_root, identities[0]["relative_path"]))
    _scope(resume_scope, identity, _read_bound(safe_relative(local_root, components[0]["relative_path"])))
    if not fixture_nodeids or len(fixture_nodeids) > 20:
        raise RetryAuthorizationError("between 1 and 20 focused fixture tests are required")
    fixture_files = {}
    for nodeid in fixture_nodeids:
        relative, separator, test_name = nodeid.partition("::")
        if not relative.startswith("tests/") or not relative.endswith(".py") or not separator or not test_name:
            raise RetryAuthorizationError("fixture must be an explicit repository test node ID")
        fixture_files[relative] = _source_record(ROOT, relative)
    code = _code_identity()
    run_key = digest({"failure": digest(failure), "code": code, "tests": fixture_files, "scope": resume_scope})
    junit_relative = f"work/retry-fixtures/{run_key}/pytest.xml"
    junit = safe_relative(local_root, junit_relative)
    junit.parent.mkdir(parents=True, exist_ok=True)
    bindings = {"mas_retry_failure_sha256": digest(failure), "mas_retry_scope_sha256": digest(resume_scope),
                "mas_retry_diagnostics_sha256": digest(retained)}
    environment = dict(
        os.environ,
        MAS_RETRY_DIAGNOSTICS_JSON=json.dumps(retained),
        MAS_RETRY_EPISODE_ROOT=str(local_root.resolve()),
        MAS_RETRY_SCOPE_JSON=json.dumps(resume_scope),
    )
    environment.update({name.upper(): value for name, value in bindings.items()})
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}", *fixture_nodeids],
                            cwd=ROOT, capture_output=True, timeout=min(300, budget.check()), check=False,
                            env=environment)
    budget.check()
    if result.returncode != 0 or not junit.is_file():
        raise RetryAuthorizationError("local failure fixture did not pass; no GPU retry authorized")
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if not cases or any(child.tag in {"failure", "error", "skipped"} for case in cases for child in case):
        raise RetryAuthorizationError("local fixture must have passing, non-skipped test cases")
    if not any({item.get("name"): item.get("value") for item in case.findall("properties/property")}.items()
               >= bindings.items() for case in cases):
        raise RetryAuthorizationError("fixture PASS must attest the exact retained failure, diagnostics and resume scope")
    if code != _code_identity() or any(_source_record(ROOT, name) != value for name, value in fixture_files.items()):
        raise RetryAuthorizationError("code or fixture changed during qualification")
    permit = {"format": "mas-code-fix-resume-1", "episode": episode, "commit": commit,
              "started_at": started_at, "failure_sha256": digest(failure),
              "evidence_input_sha256": failure["evidence_input_sha256"], "scope": resume_scope,
              "code": code, "diagnostics": retained, "predecessors": predecessors,
              "fixture_files": fixture_files,
              "fixture_nodeids": fixture_nodeids, "fixture_result": _record(local_root, junit_relative),
              "fixture_passed_count": len(cases)}
    target = local_root / "review/code-fix-resume.json"
    atomic_json(target, {"data": permit, "sha256": digest(permit)})
    validate_code_fix_resume(local_root, episode, commit, target)
    return target


def validate_code_fix_resume(local_root, episode, commit, permit_path=None):
    local_root = Path(local_root)
    permit = _read_bound(permit_path or local_root / "review/code-fix-resume.json")
    started_at, _ = _budget(local_root, episode)
    failure = _read_bound(local_root / "work/remote-failure-invariant.json")
    if (permit.get("format") != "mas-code-fix-resume-1" or permit.get("episode") != episode
            or permit.get("commit") != commit or permit.get("started_at") != started_at
            or permit.get("failure_sha256") != digest(failure)
            or permit.get("evidence_input_sha256") != failure.get("evidence_input_sha256")
            or permit.get("code") != _code_identity()):
        raise RetryAuthorizationError("code-fix retry binding changed")
    alignment_scope(permit["scope"])
    _part_failure_scope(permit["scope"], failure)
    expected_predecessors = set(retry_predecessor_paths(episode, permit["scope"], local_root))
    if {record["relative_path"] for record in permit.get("predecessors", [])} != expected_predecessors:
        raise RetryAuthorizationError("retry requires every prealignment checkpoint")
    for record in permit["diagnostics"] + permit["predecessors"] + [permit["fixture_result"]]:
        if _record(local_root, record["relative_path"]) != record:
            raise RetryAuthorizationError("retry diagnostic or fixture result changed")
    for relative, record in permit["fixture_files"].items():
        if _source_record(ROOT, relative) != record:
            raise RetryAuthorizationError("retry fixture implementation changed")
    identities = [record for record in permit["diagnostics"] if Path(record["relative_path"]).name == "resume-identity.json"]
    components = [record for record in permit["diagnostics"] if Path(record["relative_path"]).name == "latest-conflict-failure.json"]
    if len(identities) != 1 or len(components) != 1:
        raise RetryAuthorizationError("retained alignment identity or failed component is missing")
    prefix = retry_diagnostic_layout(episode, permit["scope"].get("part_id"))["prefixes"][0]
    if (identities[0]["relative_path"] != prefix + "resume-identity.json"
            or components[0]["relative_path"] != prefix + "components/latest-conflict-failure.json"
            or any(not record["relative_path"].startswith(prefix) for record in permit["diagnostics"])):
        raise RetryAuthorizationError("retry diagnostic namespace differs from the authorized part")
    if "part_id" in permit["scope"]:
        validate_part_retry_context(local_root, episode, permit["scope"])
    _scope(permit["scope"], _read_bound(safe_relative(local_root, identities[0]["relative_path"])),
           _read_bound(safe_relative(local_root, components[0]["relative_path"])))
    return permit["scope"]
