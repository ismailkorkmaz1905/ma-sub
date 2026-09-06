import json
import os
import subprocess
import sys
import time

import pytest

from mas.reliability import IntegrityError, atomic_json, digest
from mas.subtitle.pilot_guardian import GuardianProcess, guard_lease


class Provider:
    pod_id = "pod-safe"
    def __init__(self): self.actions = []
    def pod(self, timeout=10): return {"id": self.pod_id, "desiredStatus": "EXITED",
                                      "costPerHr": 0.5, "networkVolumeId": "volume"}
    def stop(self, timeout=15): self.actions.append("stop")
    def start(self, timeout): self.actions.append("start")
    def wait_exited(self, timeout): self.actions.append("wait"); return self.pod()
    def account(self, timeout=10): return {"clientBalance": 2.0}


def state(path, *, parent_pid, stop_at):
    body = {"format": "mas-pilot-guardian-state-1", "pod_id": "pod-safe",
            "parent_pid": parent_pid, "stop_at_epoch": stop_at,
            "shutdown_timeout_seconds": 2, "poll_seconds": 0.01}
    body.update(minimum_balance_usd=1.1, maximum_pod_rate_usd_per_hour=0.5)
    atomic_json(path, {"data": body, "sha256": digest(body)})


@pytest.mark.parametrize("trigger", ["parent_exit", "lease_deadline"])
def test_guardian_subprocess_stops_on_parent_death_or_deadline(tmp_path, trigger):
    state_path = tmp_path / "guardian-state.json"
    parent_pid = 99999999 if trigger == "parent_exit" else os.getpid()
    stop_at = time.time() + 60 if trigger == "parent_exit" else time.time() - 1
    state(state_path, parent_pid=parent_pid, stop_at=stop_at)
    script = """
import sys
from mas.subtitle.pilot_guardian import guard_lease
class P:
 pod_id='pod-safe'
 def pod(self,timeout=10): return {'id':self.pod_id,'desiredStatus':'EXITED','costPerHr':0.5,'networkVolumeId':'volume','env':['secret']}
 def stop(self,timeout=15): pass
 def wait_exited(self,timeout): return self.pod()
 def account(self,timeout=10): return {'clientBalance':2.0}
guard_lease(sys.argv[1],P())
"""
    completed = subprocess.run([sys.executable, "-c", script, str(state_path)], timeout=5)
    assert completed.returncode == 0
    receipt = json.loads((tmp_path / "guardian-receipt.json").read_text())["data"]
    assert receipt["trigger"] == trigger
    assert receipt["pod"]["desiredStatus"] == "EXITED"
    assert "env" not in receipt["pod"]


def test_disarm_is_ignored_until_external_exit(tmp_path):
    path = tmp_path / "guardian-state.json"
    state(path, parent_pid=os.getpid(), stop_at=time.time() - 1)
    (tmp_path / "guardian-disarm.json").write_text("{}")
    provider = Provider()
    calls = iter([
        {"id": "pod-safe", "desiredStatus": "EXITED", "costPerHr": 0.5},
        {"id": "pod-safe", "desiredStatus": "RUNNING", "costPerHr": 0.5},
        {"id": "pod-safe", "desiredStatus": "RUNNING", "costPerHr": 0.5},
    ])
    provider.pod = lambda timeout=10: next(calls)
    provider.wait_exited = lambda timeout: {"id": "pod-safe", "desiredStatus": "EXITED"}
    receipt = guard_lease(path, provider)
    assert receipt["trigger"] == "lease_deadline"
    assert provider.actions == ["stop"]


def test_network_failure_after_ready_triggers_stop(tmp_path):
    path = tmp_path / "guardian-state.json"
    state(path, parent_pid=os.getpid(), stop_at=time.time() + 60)
    provider = Provider()
    saved = json.loads(path.read_text())["data"]
    request = {"format": "mas-pilot-guardian-start-1", "state_sha256": digest(saved),
               "pod_id": "pod-safe"}
    atomic_json(tmp_path / "guardian-start.json", {"data": request, "sha256": digest(request)})
    calls = 0
    def account(timeout=10):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TimeoutError("network down")
        return {"clientBalance": 2.0}
    provider.account = account
    receipt = guard_lease(path, provider)
    assert receipt["trigger"] == "guardian_error"
    assert provider.actions == ["stop", "wait"]
    assert receipt["error_details"]["operation"] == "pre_start_revalidation"


def test_balance_change_after_ready_prevents_start(tmp_path):
    path = tmp_path / "guardian-state.json"
    state(path, parent_pid=os.getpid(), stop_at=time.time() + 60)
    saved = json.loads(path.read_text())["data"]
    request = {"format": "mas-pilot-guardian-start-1", "state_sha256": digest(saved),
               "pod_id": "pod-safe"}
    atomic_json(tmp_path / "guardian-start.json", {"data": request, "sha256": digest(request)})
    provider = Provider()
    balances = iter([2.0, 1.05, 1.05])
    provider.account = lambda timeout=10: {"clientBalance": next(balances)}
    receipt = guard_lease(path, provider)
    assert provider.actions == ["stop", "wait"]
    assert receipt["error_details"]["operation"] == "pre_start_revalidation"


def test_arm_waits_for_ready_and_overrides_stale_child_pod_env(tmp_path, monkeypatch):
    import mas.subtitle.pilot_guardian as module
    captured = {}
    class Process:
        def __init__(self, command, **kwargs):
            captured.update(command=command, env=kwargs["env"])
            saved = json.loads((tmp_path / "guardian-state.json").read_text())["data"]
            ready = {"format": "mas-pilot-guardian-ready-1", "state_sha256": digest(saved),
                     "pod_id": "pod-safe", "pod": {"id": "pod-safe", "desiredStatus": "EXITED"}}
            atomic_json(tmp_path / "guardian-ready.json", {"data": ready, "sha256": digest(ready)})
        def poll(self): return None
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setenv("RUNPOD_POD_ID", "stale-pod")
    guardian = GuardianProcess(tmp_path, "pod-safe", time.time() + 60, 10, 1.1, 0.5)
    guardian.arm("secret-key", timeout=1)
    assert captured["env"]["RUNPOD_POD_ID"] == "pod-safe"
    assert captured["env"]["RUNPOD_API_KEY"] == "secret-key"
    assert "secret-key" not in captured["command"]


def test_arm_rejects_stale_ready_or_dead_guardian(tmp_path, monkeypatch):
    import mas.subtitle.pilot_guardian as module
    stale = {"format": "mas-pilot-guardian-ready-1", "state_sha256": "old",
             "pod_id": "pod-safe"}
    atomic_json(tmp_path / "guardian-ready.json", {"data": stale, "sha256": digest(stale)})
    class Process:
        def __init__(self, command, **kwargs): pass
        def poll(self): return 1
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    guardian = GuardianProcess(tmp_path, "pod-safe", time.time() + 60, 10, 1.1, 0.5)
    with pytest.raises(IntegrityError, match="ready receipt mismatch"):
        guardian.arm("secret-key", timeout=1)


def test_guardian_owns_start_and_confirms_it(tmp_path):
    path = tmp_path / "guardian-state.json"
    state(path, parent_pid=os.getpid(), stop_at=time.time() + 60)
    saved = json.loads(path.read_text())["data"]
    request = {"format": "mas-pilot-guardian-start-1", "state_sha256": digest(saved),
               "pod_id": "pod-safe"}
    atomic_json(tmp_path / "guardian-start.json", {"data": request, "sha256": digest(request)})
    atomic_json(tmp_path / "guardian-disarm.json", {"requested": True})
    provider = Provider()
    guard_lease(path, provider)
    started = json.loads((tmp_path / "guardian-started.json").read_text())["data"]
    assert started["state_sha256"] == digest(saved)
    assert provider.actions[0] == "start"


def test_disarm_rejects_receipt_for_other_state_or_pod(tmp_path):
    guardian = GuardianProcess(tmp_path, "pod-safe", time.time() + 60, 10, 1.1, 0.5)
    guardian.state_sha256 = "expected-state"
    class Process:
        def wait(self, timeout): return 0
    guardian.process = Process()
    receipt = {"format": "mas-pilot-guardian-receipt-1", "state_sha256": "other-state",
               "pod_id": "pod-safe", "pod": {"id": "pod-safe", "desiredStatus": "EXITED"}}
    atomic_json(tmp_path / "guardian-receipt.json", {"data": receipt, "sha256": digest(receipt)})
    with pytest.raises(IntegrityError, match="does not confirm EXITED"):
        guardian.disarm_after_exit(1)


def test_invalid_budget_state_fails_before_provider_call(tmp_path):
    path = tmp_path / "guardian-state.json"
    state(path, parent_pid=os.getpid(), stop_at=time.time() + 60)
    saved = json.loads(path.read_text())
    saved["data"]["minimum_balance_usd"] = -1
    saved["sha256"] = digest(saved["data"])
    path.write_text(json.dumps(saved))
    with pytest.raises(Exception, match="minimum_balance_usd"):
        guard_lease(path, Provider())


def test_nonfinite_live_balance_fails_readiness(tmp_path):
    path = tmp_path / "guardian-state.json"
    state(path, parent_pid=os.getpid(), stop_at=time.time() + 60)
    provider = Provider()
    provider.account = lambda timeout=10: {"clientBalance": float("nan")}
    with pytest.raises(Exception, match="balance is invalid"):
        guard_lease(path, provider)
