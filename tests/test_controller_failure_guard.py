import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mas import runpod_controller as controller
from mas.reliability import BudgetExceeded, atomic_json, digest


def _failed_job(root, *, attempt=0, legacy=False):
    source, commit = "https://example.invalid/episode", "a" * 40
    base, evidence = controller._remote_input_binding(root, source, commit, 13)
    request = {"episode": 13, "commit": commit, "input_sha256": base,
               "base_input_sha256": base, "attempt": attempt}
    if not legacy:
        request["evidence_input_sha256"] = evidence
    identity = {"episode": 13, "commit": commit, "input_sha256": base,
                "token": hashlib.sha256(f"13\n{commit}\n{base}\n".encode()).hexdigest()}
    atomic_json(root / "work/remote-job-request.json", {"data": request, "sha256": digest(request)})
    atomic_json(root / "work/remote-job-status.json", {"status": "EXITED", "exit_code": 1, "identity": identity})
    return source, commit


@pytest.mark.parametrize("change", ["none", "commit", "options", "legacy"])
def test_unchanged_failure_blocks_before_provider_or_paid_acquisition(tmp_path, monkeypatch, change):
    source, commit = _failed_job(tmp_path, legacy=change == "legacy")
    if change == "commit":
        commit = "b" * 40
    if change == "options":
        monkeypatch.setenv("MAS_MP4_TARGET_GB", "4")
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "episode_dir", lambda _: tmp_path)
    monkeypatch.setattr(controller, "_required_environment", lambda: {})
    monkeypatch.setattr(controller, "_local_preflight", lambda _: (commit, tmp_path))
    monkeypatch.setattr(controller, "_validate_local_tr_return", lambda *_: None)
    monkeypatch.setattr(controller, "preflight_local_id_return", lambda *_: None)
    monkeypatch.setattr(controller, "_prepare_official_source", lambda *_: source)
    monkeypatch.setattr(controller, "CapacityProvider", lambda *_: pytest.fail("paid acquisition path reached"))
    with pytest.raises(controller.RunPodControllerError, match="BLOCKED: unchanged failed evidence"):
        controller.run_remote_episode(13)


def test_changed_validated_evidence_allows_only_bounded_attempt(tmp_path):
    source, commit = _failed_job(tmp_path)
    returned = tmp_path / "translation_output/Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip"
    returned.parent.mkdir()
    returned.write_bytes(b"different validated input")
    controller._guard_failed_remote_job(tmp_path, 13, source, commit)
    base, _ = controller._remote_input_binding(tmp_path, source, commit, 13)
    identity, attempt = controller._remote_attempt_identity(
        base, tmp_path / "work/remote-job-request.json", tmp_path / "work/remote-job-status.json")
    assert attempt == 1
    assert identity == digest({"base_input_sha256": base, "attempt": 1})


def test_audio_review_reset_is_validated_changed_evidence(tmp_path, monkeypatch):
    name = "Muhtemel Ask 13.Bolum"
    text = tmp_path / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
    text.parent.mkdir(parents=True)
    text.write_bytes(b"validated text")
    body = {
        "format": "mas-audio-review-reset-1",
        "episode": 13,
        "correction_input_sha256": "a" * 64,
        "provisional_output_sha256": "b" * 64,
        "reason": "preserve stale review evidence",
        "artifacts": [
            {"relative_path": relative, "size_bytes": 1, "sha256": "c" * 64}
            for relative in (
                "prepare/audio_review_v2.json",
                "prepare/audio_review_v2.recovery.json",
                f"translation_output/{name}_TR_CORRECTED.zip",
            )
        ],
    }
    marker = tmp_path / "review/audio_review_reset.json"
    atomic_json(marker, {"data": body, "sha256": digest(body)})
    monkeypatch.setattr(
        controller,
        "validate_tr_correction_output",
        lambda *_: SimpleNamespace(input_sha256="a" * 64, output_sha256="b" * 64),
    )

    _, with_reset = controller._remote_input_binding(
        tmp_path, "https://example.invalid/episode", "a" * 40, 13
    )
    marker.unlink()
    _, without_reset = controller._remote_input_binding(
        tmp_path, "https://example.invalid/episode", "a" * 40, 13
    )

    assert with_reset != without_reset


def test_changed_commit_allows_bounded_retry_before_pipeline_checkpoint(tmp_path):
    source, prior_commit = _failed_job(tmp_path)
    request = json.loads((tmp_path / "work/remote-job-request.json").read_text())["data"]
    identity = {"episode": 13, "commit": prior_commit, "input_sha256": request["input_sha256"],
                "token": hashlib.sha256(
                    f"13\n{prior_commit}\n{request['input_sha256']}\n".encode()
                ).hexdigest()}
    checkpoint = {"identity": identity, "files": [{"relative_path": "source/source.url"}],
                  "unstable": []}
    atomic_json(tmp_path / "work/remote-checkpoint-manifest.json", checkpoint)
    controller._record_failed_remote_evidence(1, tmp_path, source, prior_commit)

    controller._guard_failed_remote_job(tmp_path, 13, source, "b" * 40)

    authorization = json.loads(
        (tmp_path / "work/remote-retry-authorization.json").read_text()
    )["data"]
    assert authorization["pre_pipeline_code_fix"] is True


def test_changed_commit_allows_bounded_raw_asr_postprocess_retry(tmp_path):
    source, prior_commit = _failed_job(tmp_path)
    request = json.loads((tmp_path / "work/remote-job-request.json").read_text())["data"]
    identity = {
        "episode": 13,
        "commit": prior_commit,
        "input_sha256": request["input_sha256"],
        "token": hashlib.sha256(
            f"13\n{prior_commit}\n{request['input_sha256']}\n".encode()
        ).hexdigest(),
    }
    status_path = tmp_path / "work/remote-job-status.json"
    status = json.loads(status_path.read_text())
    status["progress"] = {"stage": "raw_asr:rescue-42"}
    atomic_json(status_path, status)
    paths = [
        "prepare/audio.done.json",
        "prepare/primary_asr/result.json",
        "prepare/raw_asr_v2.recovery.json",
        "source/Muhtemel Ask 13.Bolum.mkv",
        "source/download.done.json",
        "source/source.url",
        "work/state.json",
    ]
    checkpoint = {
        "identity": identity,
        "files": [
            {"relative_path": path, "sha256": "a" * 64, "size_bytes": 1}
            for path in paths
        ],
        "unstable": [],
    }
    atomic_json(tmp_path / "work/remote-checkpoint-manifest.json", checkpoint)
    controller._record_failed_remote_evidence(1, tmp_path, source, prior_commit)

    controller._guard_failed_remote_job(tmp_path, 13, source, "b" * 40)

    authorization = json.loads(
        (tmp_path / "work/remote-retry-authorization.json").read_text()
    )["data"]
    assert authorization["checkpointed_raw_asr_code_fix"] is True


def test_changed_evidence_retry_limit_cannot_reset_with_commit(tmp_path):
    source, _ = _failed_job(tmp_path, attempt=2)
    returned = tmp_path / "translation_output/Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip"
    returned.parent.mkdir()
    returned.write_bytes(b"changed")
    with pytest.raises(controller.RunPodControllerError, match="retry budget exhausted"):
        controller._guard_failed_remote_job(tmp_path, 13, source, "b" * 40)


def test_changed_evidence_retry_limit_accepts_one_operator_extension(
        tmp_path, monkeypatch):
    source, commit = _failed_job(tmp_path, attempt=2)
    returned = tmp_path / "translation_output/Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip"
    returned.parent.mkdir()
    returned.write_bytes(b"changed")
    monkeypatch.setenv("MAS_CHANGED_EVIDENCE_RETRY_EXTENSION_APPROVED", "1")
    monkeypatch.setenv(
        "MAS_CHANGED_EVIDENCE_RETRY_EXTENSION_REASON",
        "operator approved completion after retained alignment evidence",
    )

    controller._guard_failed_remote_job(tmp_path, 13, source, commit)
    base, _ = controller._remote_input_binding(tmp_path, source, commit, 13)
    identity, attempt = controller._remote_attempt_identity(
        base,
        tmp_path / "work/remote-job-request.json",
        tmp_path / "work/remote-job-status.json",
    )

    assert attempt == 3
    assert identity == digest({"base_input_sha256": base, "attempt": 3})
    ledger = json.loads(
        (tmp_path / "work/remote-retry-extension.json").read_text(encoding="utf-8")
    )
    assert ledger["sha256"] == digest(ledger["data"])
    assert ledger["data"]["previous_limit"] == 2
    assert ledger["data"]["extended_limit"] == 5


def test_changed_evidence_retry_extension_requires_reason(tmp_path, monkeypatch):
    source, commit = _failed_job(tmp_path, attempt=2)
    returned = tmp_path / "translation_output/Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip"
    returned.parent.mkdir()
    returned.write_bytes(b"changed")
    monkeypatch.setenv("MAS_CHANGED_EVIDENCE_RETRY_EXTENSION_APPROVED", "1")

    with pytest.raises(controller.RunPodControllerError, match="REASON"):
        controller._guard_failed_remote_job(tmp_path, 13, source, commit)


def test_retry_authorization_rejects_changed_failure_evidence(tmp_path):
    source, commit = _failed_job(tmp_path)
    returned = tmp_path / "translation_output/Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip"
    returned.parent.mkdir()
    returned.write_bytes(b"changed validated input")
    controller._guard_failed_remote_job(tmp_path, 13, source, commit)
    path = tmp_path / "work/remote-failure-invariant.json"
    saved = json.loads(path.read_text())
    saved["data"]["diagnostics"] = {"forged": "a" * 64}
    atomic_json(path, {"data": saved["data"], "sha256": digest(saved["data"])})
    base, _ = controller._remote_input_binding(tmp_path, source, commit, 13)
    with pytest.raises(controller.RunPodControllerError, match="failure evidence binding"):
        controller._remote_attempt_identity(base, tmp_path / "work/remote-job-request.json",
                                            tmp_path / "work/remote-job-status.json")


def test_pipeline_updated_return_does_not_unlock_same_failed_run(tmp_path):
    source, commit = _failed_job(tmp_path)
    returned = tmp_path / "translation_output/Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip"
    returned.parent.mkdir()
    returned.write_bytes(b"audio review changed this before alignment failed")
    controller._record_failed_remote_evidence(1, tmp_path, source, commit)
    with pytest.raises(controller.RunPodControllerError, match="unchanged failed evidence"):
        controller._guard_failed_remote_job(tmp_path, 13, source, commit)


def test_unrelated_zip_cannot_unlock_failed_evidence(tmp_path):
    source, commit = _failed_job(tmp_path)
    unrelated = tmp_path / "translation_output/unvalidated.zip"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"irrelevant")
    with pytest.raises(controller.RunPodControllerError, match="unchanged failed evidence"):
        controller._guard_failed_remote_job(tmp_path, 13, source, commit)


@pytest.mark.parametrize("image", ["", "registry/image:latest", "registry/image@sha256:abc", "registry/image@sha256:" + "a" * 63])
def test_runtime_image_requires_exact_digest(monkeypatch, image):
    monkeypatch.setenv("MAS_RUNPOD_IMAGE", image)
    with pytest.raises(controller.RunPodControllerError, match="repository@sha256"):
        controller._configured_runtime_image()


def test_runtime_image_accepts_configured_not_invented_digest(monkeypatch):
    image = "registry/image@sha256:" + "a" * 64
    monkeypatch.setenv("MAS_RUNPOD_IMAGE", image)
    assert controller._configured_runtime_image() == image


@pytest.mark.parametrize("value", ["bad value", "bad/value", "bad.value", "bad:value"])
def test_registry_auth_id_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("MAS_RUNPOD_REGISTRY_AUTH_ID", value)
    with pytest.raises(controller.RunPodControllerError, match="REGISTRY_AUTH_ID"):
        controller._configured_registry_auth_id()


def test_registry_auth_id_is_optional_and_validated(monkeypatch):
    monkeypatch.delenv("MAS_RUNPOD_REGISTRY_AUTH_ID", raising=False)
    assert controller._configured_registry_auth_id() is None
    monkeypatch.setenv("MAS_RUNPOD_REGISTRY_AUTH_ID", "auth_123-ABC")
    assert controller._configured_registry_auth_id() == "auth_123-ABC"


def test_historical_excluded_wait_is_retained_but_not_added_to_wall_time(tmp_path, monkeypatch):
    monkeypatch.delenv("MAS_EPISODE_BUDGET_SECONDS", raising=False)
    start = datetime(2026, 9, 14, tzinfo=timezone.utc)
    data = {"episode": 13, "started_at": start.isoformat(), "limit_seconds": 14400,
            "excluded_wait_seconds": 90000}
    atomic_json(tmp_path / "work/controller_budget.json", {"data": data, "sha256": digest(data)})
    with pytest.raises(BudgetExceeded):
        controller._episode_budget(tmp_path, 13, now=start + timedelta(seconds=21600))
    saved = json.loads((tmp_path / "work/controller_budget.json").read_text())
    assert saved["data"]["excluded_wait_seconds"] == 90000
    assert saved["data"]["started_at"] == start.isoformat()


def test_budget_cannot_exceed_six_hour_hard_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_EPISODE_BUDGET_SECONDS", "21601")
    with pytest.raises(controller.RunPodControllerError, match="at most 21600"):
        controller._episode_budget(tmp_path, 13)
