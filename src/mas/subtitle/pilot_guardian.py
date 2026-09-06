import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ..reliability import IntegrityError, atomic_json, digest


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:
                return False
            if error == 5:
                return True
            raise OSError(error, "cannot inspect pilot controller process")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(), "cannot inspect pilot controller state")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _safe_pod(value):
    return {key: value.get(key) for key in
            ("id", "desiredStatus", "costPerHr", "networkVolumeId")}


def _provider_amount(value, label):
    try:
        amount = float(value)
    except (TypeError, ValueError):
        raise IntegrityError(f"pilot guardian {label} is invalid") from None
    if not math.isfinite(amount) or amount < 0:
        raise IntegrityError(f"pilot guardian {label} is invalid")
    return amount


def _validated_state(state_path, provider):
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    body = state.get("data")
    required = {"format", "pod_id", "parent_pid", "stop_at_epoch",
                "shutdown_timeout_seconds", "minimum_balance_usd",
                "maximum_pod_rate_usd_per_hour", "poll_seconds"}
    if (not isinstance(body, dict) or set(body) != required
            or state.get("sha256") != digest(body)
            or body.get("format") != "mas-pilot-guardian-state-1"):
        raise IntegrityError("pilot guardian state checksum or fields mismatch")
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", str(body.get("pod_id") or ""))
            or provider.pod_id != body["pod_id"]):
        raise IntegrityError("pilot guardian Pod identity mismatch")
    if isinstance(body["parent_pid"], bool) or not isinstance(body["parent_pid"], int) or body["parent_pid"] <= 0:
        raise IntegrityError("pilot guardian parent PID is invalid")
    for key in ("stop_at_epoch", "shutdown_timeout_seconds", "minimum_balance_usd",
                "maximum_pod_rate_usd_per_hour", "poll_seconds"):
        value = body[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise IntegrityError(f"pilot guardian {key} is invalid")
    return body


def guard_lease(state_path, provider, *, clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
    state_path = Path(state_path)
    body = _validated_state(state_path, provider)
    receipt_path = state_path.with_name("guardian-receipt.json")
    ready_path = state_path.with_name("guardian-ready.json")
    start_path = state_path.with_name("guardian-start.json")
    started_path = state_path.with_name("guardian-started.json")
    disarm_path = state_path.with_name("guardian-disarm.json")
    trigger = None
    account, current_pod = provider.account(10), provider.pod(10)
    if (current_pod.get("id") != body["pod_id"]
            or current_pod.get("desiredStatus") != "EXITED"
            or _provider_amount(current_pod.get("costPerHr"), "Pod rate") > body["maximum_pod_rate_usd_per_hour"]
            or _provider_amount(account.get("clientBalance"), "balance") <= body["minimum_balance_usd"]):
        raise IntegrityError("pilot guardian readiness check failed")
    state_sha256 = digest(body)
    ready = {"format": "mas-pilot-guardian-ready-1", "state_sha256": state_sha256,
             "pod_id": body["pod_id"], "pod": _safe_pod(current_pod),
             "balance_usd": _provider_amount(account.get("clientBalance"), "balance"),
             "observed_epoch": clock()}
    atomic_json(ready_path, {"data": ready, "sha256": digest(ready)})
    next_cost_check = 0.0
    try:
        while not start_path.exists():
            if not _pid_alive(body["parent_pid"]):
                trigger = "parent_exit"
                break
            if clock() >= body["stop_at_epoch"]:
                trigger = "lease_deadline"
                break
            sleep(min(body["poll_seconds"], max(0.01, body["stop_at_epoch"] - clock())))
        if trigger is None:
            request = json.loads(start_path.read_text(encoding="utf-8"))
            request_body = request.get("data")
            if (not isinstance(request_body, dict)
                    or request.get("sha256") != digest(request_body)
                    or request_body != {"format": "mas-pilot-guardian-start-1",
                                        "state_sha256": state_sha256,
                                        "pod_id": body["pod_id"]}):
                raise IntegrityError("pilot guardian start request mismatch")
            if not _pid_alive(body["parent_pid"]) or clock() >= body["stop_at_epoch"]:
                raise IntegrityError("pilot guardian lease expired before start")
            provider.start(min(15, max(0.001, body["stop_at_epoch"] - clock())))
            started_pod = provider.pod(10)
            started = {"format": "mas-pilot-guardian-started-1", "state_sha256": state_sha256,
                       "pod_id": body["pod_id"], "pod": _safe_pod(started_pod),
                       "observed_epoch": clock()}
            atomic_json(started_path, {"data": started, "sha256": digest(started)})
        while trigger is None:
            if clock() >= next_cost_check:
                account, current_pod = provider.account(10), provider.pod(10)
                if (_provider_amount(account.get("clientBalance"), "balance") <= body["minimum_balance_usd"]
                        or _provider_amount(current_pod.get("costPerHr"), "Pod rate") > body["maximum_pod_rate_usd_per_hour"]):
                    trigger = "cost_guard"
                    break
                next_cost_check = clock() + 15
            if disarm_path.exists():
                pod = provider.pod(10)
                if pod.get("desiredStatus") == "EXITED":
                    trigger = "externally_confirmed_exit"
                    break
            if not _pid_alive(body["parent_pid"]):
                trigger = "parent_exit"
                break
            if clock() >= body["stop_at_epoch"]:
                trigger = "lease_deadline"
                break
            sleep(min(body["poll_seconds"], max(0.01, body["stop_at_epoch"] - clock())))
    except Exception:
        trigger = "guardian_error"
    shutdown_error = None
    if trigger != "externally_confirmed_exit":
        shutdown_deadline = monotonic() + body["shutdown_timeout_seconds"]
        try:
            provider.stop(min(15, max(0.001, shutdown_deadline - monotonic())))
            pod = provider.wait_exited(max(0.001, shutdown_deadline - monotonic()))
        except Exception as exc:
            shutdown_error = f"{type(exc).__name__}: guardian shutdown failed"
            pod = {"id": body["pod_id"], "desiredStatus": "UNVERIFIED"}
    if pod.get("desiredStatus") != "EXITED":
        shutdown_error = shutdown_error or "guardian did not externally confirm EXITED"
    try:
        balance = _provider_amount(provider.account(10).get("clientBalance"), "balance")
    except Exception:
        balance = None
    receipt = {"format": "mas-pilot-guardian-receipt-1", "trigger": trigger,
               "state_sha256": state_sha256, "pod_id": body["pod_id"],
               "pod": _safe_pod(pod),
               "balance_usd": balance, "shutdown_error": shutdown_error,
               "observed_epoch": clock()}
    atomic_json(receipt_path, {"data": receipt, "sha256": digest(receipt)})
    if shutdown_error:
        raise IntegrityError(shutdown_error)
    return receipt


class GuardianProcess:
    def __init__(self, output_dir, pod_id, stop_at_epoch, shutdown_timeout_seconds,
                 minimum_balance_usd, maximum_pod_rate_usd_per_hour):
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "guardian-state.json"
        self.disarm_path = self.output_dir / "guardian-disarm.json"
        self.ready_path = self.output_dir / "guardian-ready.json"
        self.start_path = self.output_dir / "guardian-start.json"
        self.started_path = self.output_dir / "guardian-started.json"
        self.pod_id = pod_id
        self.stop_at_epoch = stop_at_epoch
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.minimum_balance_usd = minimum_balance_usd
        self.maximum_pod_rate_usd_per_hour = maximum_pod_rate_usd_per_hour
        self.process = None

    def arm(self, api_key, timeout=20):
        body = {"format": "mas-pilot-guardian-state-1", "pod_id": self.pod_id,
                "parent_pid": os.getpid(), "stop_at_epoch": self.stop_at_epoch,
                "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
                "minimum_balance_usd": self.minimum_balance_usd,
                "maximum_pod_rate_usd_per_hour": self.maximum_pod_rate_usd_per_hour,
                "poll_seconds": 1}
        atomic_json(self.state_path, {"data": body, "sha256": digest(body)})
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        child_env = os.environ.copy()
        child_env["RUNPOD_POD_ID"] = self.pod_id
        child_env["RUNPOD_API_KEY"] = api_key
        self.process = subprocess.Popen(
            [sys.executable, "-m", "mas.subtitle.pilot_guardian", str(self.state_path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"), creationflags=flags, env=child_env)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self.ready_path.exists():
                    ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
                    data = ready.get("data")
                    state_sha256 = digest(body)
                    if (not isinstance(data, dict) or ready.get("sha256") != digest(data)
                            or data.get("state_sha256") != state_sha256
                            or data.get("pod_id") != self.pod_id
                            or data.get("pod", {}).get("id") != self.pod_id
                            or self.process.poll() is not None):
                        raise IntegrityError("pilot guardian ready receipt mismatch")
                    self.state_sha256 = state_sha256
                    return data
                if self.process.poll() is not None:
                    raise RuntimeError("pilot guardian failed readiness handshake")
                time.sleep(0.05)
            raise TimeoutError("pilot guardian readiness handshake timed out")
        except Exception:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=2)
            raise

    def start(self, timeout):
        if self.process is None or self.process.poll() is not None:
            raise IntegrityError("pilot guardian is not alive for start")
        body = {"format": "mas-pilot-guardian-start-1", "state_sha256": self.state_sha256,
                "pod_id": self.pod_id}
        atomic_json(self.start_path, {"data": body, "sha256": digest(body)})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.started_path.exists():
                wrapped = json.loads(self.started_path.read_text(encoding="utf-8"))
                data = wrapped.get("data")
                if (not isinstance(data, dict) or wrapped.get("sha256") != digest(data)
                        or data.get("state_sha256") != self.state_sha256
                        or data.get("pod_id") != self.pod_id
                        or data.get("pod", {}).get("id") != self.pod_id
                        or self.process.poll() is not None):
                    raise IntegrityError("pilot guardian started receipt mismatch")
                return data
            if self.process.poll() is not None:
                raise RuntimeError("pilot guardian exited before start confirmation")
            time.sleep(0.05)
        raise TimeoutError("pilot guardian start confirmation timed out")

    def disarm_after_exit(self, timeout):
        atomic_json(self.disarm_path, {"requested": True})
        self.process.wait(timeout=timeout)
        receipt = json.loads((self.output_dir / "guardian-receipt.json").read_text(encoding="utf-8"))
        body = receipt.get("data")
        if not isinstance(body, dict) or receipt.get("sha256") != digest(body):
            raise IntegrityError("pilot guardian receipt checksum mismatch")
        if (body.get("state_sha256") != self.state_sha256
                or body.get("pod_id") != self.pod_id
                or body.get("pod", {}).get("id") != self.pod_id
                or body.get("pod", {}).get("desiredStatus") != "EXITED"):
            raise IntegrityError("pilot guardian receipt does not confirm EXITED")
        return body


def main(argv=None):
    from .pilot_runner import RunPodPilotProvider
    state_path = Path((argv or sys.argv[1:])[0])
    state = json.loads(state_path.read_text(encoding="utf-8"))["data"]
    api_key = os.getenv("RUNPOD_API_KEY")
    env_pod = os.getenv("RUNPOD_POD_ID")
    try:
        if not api_key or env_pod != state.get("pod_id"):
            raise IntegrityError("guardian credentials or Pod identity are unavailable")
        guard_lease(state_path, RunPodPilotProvider(env_pod, api_key))
    except Exception as exc:
        error = {"format": "mas-pilot-guardian-error-1",
                 "error_type": type(exc).__name__,
                 "error": "pilot guardian failed; inspect safe receipts",
                 "observed_epoch": time.time()}
        atomic_json(state_path.with_name("guardian-error.json"),
                    {"data": error, "sha256": digest(error)})
        raise


if __name__ == "__main__":
    main()
