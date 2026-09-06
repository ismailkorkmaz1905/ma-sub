from pathlib import Path
import sys

import pytest

from mas.reliability import IntegrityError
from mas.subtitle.pilot_runner import (LeasePlan, ProgressGuard, lease_preflight,
                                       deadline_file_digest, run_bounded_pilot,
                                       run_progress_process)


def account(balance=3.75):
    return {"clientBalance": balance, "currentSpendPerHr": 0.005,
            "isAutoPayEnabled": False}


def test_provider_identifies_client_without_exposing_key(monkeypatch):
    import io
    import urllib.request
    from mas.subtitle.pilot_runner import RunPodPilotProvider

    def respond(request, timeout):
        assert request.get_header("User-agent") == "ma-sub-pilot/1.0"
        assert request.get_header("Accept") == "application/json"
        assert timeout == 7
        return io.BytesIO(b'{"data":{"myself":{"clientBalance":3.7}}}')

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    assert RunPodPilotProvider("pod", "private-test-key").account(7)["clientBalance"] == 3.7


def test_provider_accepts_empty_successful_start_response(monkeypatch):
    import io
    import urllib.request
    from mas.subtitle.pilot_runner import RunPodPilotProvider
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b""))
    assert RunPodPilotProvider("pod", "private-test-key").start(7) == {}
    with pytest.raises(IntegrityError, match="JSONDecodeError"):
        RunPodPilotProvider("pod", "private-test-key").pod(7)


def test_provider_http_failure_preserves_safe_reason_only(monkeypatch):
    import io
    import urllib.request
    import urllib.error
    from mas.subtitle.pilot_runner import RunPodPilotProvider, PilotProviderError

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('https://example/?key=secret', 500, 'secret', {},
                                     io.BytesIO(b'not enough free GPUs; secret'))

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(PilotProviderError) as raised:
        RunPodPilotProvider("pod", "private-test-key").start(7)
    assert raised.value.safe_details == {"http_status": 500, "category": "capacity", "method": "POST"}
    assert "secret" not in str(raised.value)


def pod(status="EXITED"):
    return {"costPerHr": 0.74, "desiredStatus": status}


def test_lease_preserves_reserve_and_all_time_phases():
    result = lease_preflight(account(), pod(), LeasePlan())
    assert result["required_seconds"] == 550
    assert result["minimum_final_balance_usd"] == 1.10
    assert result["required_usd"] == pytest.approx(0.1138194444)


def test_lease_rejects_active_pod_and_insufficient_balance():
    with pytest.raises(IntegrityError, match="EXITED"):
        lease_preflight(account(), pod("RUNNING"), LeasePlan())
    with pytest.raises(IntegrityError, match="insufficient"):
        lease_preflight(account(1.11), pod(), LeasePlan())


def test_progress_requires_measured_change():
    now = [0]
    guard = ProgressGuard(10, clock=lambda: now[0])
    now[0] = 9
    guard.update(completed_units=0, artifact_bytes=0)
    now[0] = 11
    with pytest.raises(TimeoutError, match="no measured"):
        guard.check()


class Provider:
    def __init__(self):
        self.actions = []

    def account(self): return account()
    def pod(self): return pod()
    def start(self, timeout): self.actions.append("start")
    def wait_running(self, timeout): return {"desiredStatus": "RUNNING"}
    def stop(self, timeout=15): self.actions.append("stop")
    def wait_exited(self, timeout): return {"desiredStatus": "EXITED"}


def test_failure_after_start_always_stops_and_notifies(tmp_path):
    provider, notices = Provider(), []
    with pytest.raises(RuntimeError, match="exited with code 3"):
        run_bounded_pilot(provider, [sys.executable, "-c", "raise SystemExit(3)"],
                          tmp_path / "progress.json", [], tmp_path / "run", LeasePlan(),
                          lambda *args: notices.append(args))
    assert provider.actions == ["start", "stop"]
    assert any("FAIL" in event[1] for event in notices)
    assert any("shutdown PASS" in event[1] for event in notices)


def test_ambiguous_start_failure_still_requests_stop(tmp_path):
    provider = Provider()
    def failed_start(timeout):
        provider.actions.append("start")
        raise TimeoutError("ambiguous start response")
    provider.start = failed_start
    with pytest.raises(TimeoutError, match="ambiguous"):
        run_bounded_pilot(provider, [sys.executable, "-c", "pass"], tmp_path / "progress.json",
                          [], tmp_path / "run", LeasePlan(),
                          lambda *args: None)
    assert provider.actions == ["start", "stop"]


def test_notification_failure_cannot_prevent_shutdown(tmp_path):
    provider = Provider()
    def notify(*args):
        raise RuntimeError("mail unavailable")
    with pytest.raises(RuntimeError, match="exited with code 3"):
        run_bounded_pilot(provider, [sys.executable, "-c", "raise SystemExit(3)"],
                          tmp_path / "progress.json", [], tmp_path / "run", LeasePlan(), notify)
    assert provider.actions == ["start", "stop"]


def test_blocked_notification_and_hung_worker_are_bounded_and_stopped(tmp_path):
    import time
    provider = Provider()
    def notify(*args):
        time.sleep(10)
    plan = LeasePlan(requested_work_seconds=1, idle_timeout_seconds=1,
                     notification_timeout_seconds=0.01)
    with pytest.raises(TimeoutError):
        run_bounded_pilot(provider, [sys.executable, "-c", "import time; time.sleep(60)"],
                          tmp_path / "progress.json", [], tmp_path / "run", plan, notify)
    assert provider.actions == ["start", "stop"]


def test_billing_query_failure_prevents_start(tmp_path):
    provider = Provider()
    provider.account = lambda: (_ for _ in ()).throw(TimeoutError("billing unavailable"))
    with pytest.raises(TimeoutError, match="billing unavailable"):
        run_bounded_pilot(provider, [sys.executable, "-c", "pass"], tmp_path / "progress.json",
                          [], tmp_path / "run", LeasePlan(), lambda *args: None)
    assert provider.actions == []


def test_success_hashes_artifacts_and_stops(tmp_path):
    provider, artifact = Provider(), tmp_path / "artifact.json"
    artifact.write_text("result", encoding="utf-8")
    progress = tmp_path / "progress.json"
    command = [sys.executable, "-c",
               f"from pathlib import Path; Path(r'{progress}').write_text('" +
               "{\"completed_units\":1,\"artifact_bytes\":6}')"]
    result = run_bounded_pilot(provider, command, progress, [artifact], tmp_path / "run",
                               LeasePlan(), lambda *args: None)
    assert provider.actions == ["start", "stop"]
    assert result["artifacts"][0]["bytes"] == 6


def test_artifact_hash_checks_deadline_during_read(tmp_path):
    artifact = tmp_path / "large.bin"
    artifact.write_bytes(b"x" * (1024 * 1024 + 1))
    ticks = iter([0, 2])
    with pytest.raises(TimeoutError, match="artifact hashing"):
        deadline_file_digest(artifact, 1, clock=lambda: next(ticks))


def test_shutdown_failure_wins_over_worker_failure(tmp_path):
    provider = Provider()
    provider.wait_exited = lambda timeout: (_ for _ in ()).throw(TimeoutError("shutdown unverified"))
    with pytest.raises(TimeoutError, match="shutdown unverified"):
        run_bounded_pilot(provider, [sys.executable, "-c", "raise SystemExit(3)"],
                          tmp_path / "progress.json", [], tmp_path / "run", LeasePlan(),
                          lambda *args: None)
    events = (tmp_path / "run" / "events.json").read_text(encoding="utf-8")
    assert '"stage": "shutdown"' in events
    assert '"status": "FAIL"' in events


def test_timeout_kills_descendant_process(tmp_path):
    marker = tmp_path / "descendant-lived"
    child = f"import time; from pathlib import Path; time.sleep(1); Path(r'{marker}').write_text('bad')"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(60)"
    with pytest.raises(TimeoutError):
        run_progress_process([sys.executable, "-c", parent, child], tmp_path / "progress.json",
                             timeout_seconds=0.2, idle_timeout_seconds=5,
                             log_path=tmp_path / "worker.log", poll_seconds=0.02)
    import time
    time.sleep(1.1)
    assert not marker.exists()
