import json
from datetime import datetime, timedelta, timezone

import pytest

from mas import runpod_controller
from mas.reliability import BudgetExceeded, digest


def _release(path):
    body = {"state_sha256": "a" * 64,
            "owned_pods": [{"pod_id": "owned-ep13-pod", "status": "ABSENT",
                            "error_type": None}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": body, "sha256": digest(body)}), encoding="utf-8")


def _handoff(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"verified handoff")


def test_verified_handoff_wait_does_not_consume_active_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_EPISODE_BUDGET_SECONDS", "400")
    started = datetime(2026, 9, 7, tzinfo=timezone.utc)
    handoff = tmp_path / "translation_input/return.zip"
    release = tmp_path / "work/capacity/capacity-state.json"
    _handoff(handoff)
    _release(release)

    budget = runpod_controller._episode_budget(tmp_path, 13, now=started)
    assert budget.remaining(started + timedelta(seconds=100)) == pytest.approx(300)
    runpod_controller._pause_episode_budget(
        tmp_path, 13, reason=20, evidence_path=handoff,
        shutdown_path=release)
    saved_path = tmp_path / "work/controller_budget.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    saved["data"]["wait"]["paused_at"] = (started + timedelta(seconds=100)).isoformat()
    saved["sha256"] = digest(saved["data"])
    saved_path.write_text(json.dumps(saved), encoding="utf-8")

    resumed = runpod_controller._episode_budget(
        tmp_path, 13, now=started + timedelta(days=1))
    assert resumed.remaining(started + timedelta(days=1, seconds=50)) == pytest.approx(250)
    ledger = json.loads(saved_path.read_text(encoding="utf-8"))["data"]
    assert ledger["started_at"] == started.isoformat()
    assert ledger["excluded_wait_seconds"] == pytest.approx(24 * 60 * 60 - 100)


def test_unclosed_active_window_charges_crash_and_offline_time(tmp_path, monkeypatch):
    monkeypatch.setenv("MAS_EPISODE_BUDGET_SECONDS", "400")
    started = datetime(2026, 9, 7, tzinfo=timezone.utc)
    runpod_controller._episode_budget(tmp_path, 13, now=started)

    with pytest.raises(BudgetExceeded):
        runpod_controller._episode_budget(tmp_path, 13, now=started + timedelta(seconds=401))


@pytest.mark.parametrize("reason", [0, 1, 22, 99])
def test_budget_pause_rejects_non_handoff_reason(tmp_path, reason):
    started = datetime(2026, 9, 7, tzinfo=timezone.utc)
    handoff = tmp_path / "translation_input/return.zip"
    release = tmp_path / "work/capacity/capacity-state.json"
    _handoff(handoff)
    _release(release)
    runpod_controller._episode_budget(tmp_path, 13, now=started)

    with pytest.raises(runpod_controller.RunPodControllerError, match="handoff|wait"):
        runpod_controller._pause_episode_budget(
            tmp_path, 13, reason=reason, evidence_path=handoff,
            shutdown_path=release)


def test_budget_pause_requires_verified_release_and_evidence(tmp_path):
    started = datetime(2026, 9, 7, tzinfo=timezone.utc)
    handoff = tmp_path / "translation_input/return.zip"
    release = tmp_path / "work/capacity/capacity-state.json"
    runpod_controller._episode_budget(tmp_path, 13, now=started)

    with pytest.raises(runpod_controller.RunPodControllerError):
        runpod_controller._pause_episode_budget(
            tmp_path, 13, reason=20, evidence_path=handoff,
            shutdown_path=release)
    _handoff(handoff)
    release.parent.mkdir(parents=True, exist_ok=True)
    release.write_text(json.dumps({"data": {"status": "RUNNING"}}), encoding="utf-8")
    with pytest.raises(runpod_controller.RunPodControllerError):
        runpod_controller._pause_episode_budget(
            tmp_path, 13, reason=20, evidence_path=handoff,
            shutdown_path=release)


def test_identical_pause_is_idempotent_but_changed_evidence_fails(tmp_path):
    started = datetime(2026, 9, 7, tzinfo=timezone.utc)
    handoff = tmp_path / "translation_input/return.zip"
    release = tmp_path / "work/capacity/capacity-state.json"
    _handoff(handoff)
    _release(release)
    runpod_controller._episode_budget(tmp_path, 13, now=started)
    arguments = dict(reason=21, evidence_path=handoff, shutdown_path=release)
    runpod_controller._pause_episode_budget(tmp_path, 13, **arguments)
    first = (tmp_path / "work/controller_budget.json").read_bytes()
    runpod_controller._pause_episode_budget(tmp_path, 13, **arguments)
    assert (tmp_path / "work/controller_budget.json").read_bytes() == first

    handoff.write_bytes(b"changed")
    with pytest.raises(runpod_controller.RunPodControllerError, match="evidence"):
        runpod_controller._pause_episode_budget(tmp_path, 13, **arguments)


def test_budget_ledger_tampering_is_rejected(tmp_path):
    started = datetime(2026, 9, 7, tzinfo=timezone.utc)
    runpod_controller._episode_budget(tmp_path, 13, now=started)
    path = tmp_path / "work/controller_budget.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["data"]["excluded_wait_seconds"] = 1
    path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(runpod_controller.RunPodControllerError, match="integrity"):
        runpod_controller._episode_budget(tmp_path, 13, now=started + timedelta(seconds=1))
