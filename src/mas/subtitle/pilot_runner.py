import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..reliability import IntegrityError, atomic_json, digest, file_digest


class RunPodPilotProvider:
    def __init__(self, pod_id, api_key):
        self.pod_id, self.api_key = pod_id, api_key

    def _json(self, request, timeout):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (OSError, ValueError) as exc:
            raise IntegrityError("RunPod provider request failed") from None

    def account(self, timeout=15):
        query = json.dumps({"query": "query { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}).encode()
        url = "https://api.runpod.io/graphql?api_key=" + urllib.parse.quote(self.api_key, safe="")
        result = self._json(urllib.request.Request(url, data=query,
                            headers={"Content-Type": "application/json"}), timeout)
        if result.get("errors") or not isinstance(result.get("data", {}).get("myself"), dict):
            raise IntegrityError("RunPod billing query failed")
        return result["data"]["myself"]

    def _pod_request(self, method, suffix="", timeout=15):
        request = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{self.pod_id}{suffix}", method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"})
        return self._json(request, timeout)

    def pod(self, timeout=15):
        return self._pod_request("GET", timeout=timeout)

    def start(self, timeout):
        return self._pod_request("POST", "/start", min(15, timeout))

    def stop(self, timeout=15):
        return self._pod_request("POST", "/stop", timeout)

    def _wait(self, status, timeout):
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"RunPod did not reach {status}")
            value = self.pod(min(15, left))
            if value.get("desiredStatus") == status:
                return value
            time.sleep(min(2, left))

    def wait_running(self, timeout):
        return self._wait("RUNNING", timeout)

    def wait_exited(self, timeout):
        return self._wait("EXITED", timeout)


@dataclass(frozen=True)
class LeasePlan:
    reserve_usd: float = 1.0
    billing_delay_usd: float = 0.10
    shutdown_reserve_seconds: int = 120
    startup_timeout_seconds: int = 180
    idle_timeout_seconds: int = 60
    requested_work_seconds: int = 180
    artifact_reserve_seconds: int = 30
    notification_timeout_seconds: float = 5

    def __post_init__(self):
        for value in (self.reserve_usd, self.billing_delay_usd):
            if not math.isfinite(value) or value < 0:
                raise ValueError("USD reserves must be finite and nonnegative")
        for value in (self.shutdown_reserve_seconds, self.startup_timeout_seconds,
                      self.idle_timeout_seconds, self.requested_work_seconds,
                      self.artifact_reserve_seconds):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("lease time limits must be positive integers")
        if not math.isfinite(self.notification_timeout_seconds) or self.notification_timeout_seconds <= 0:
            raise ValueError("notification timeout must be finite and positive")


def lease_preflight(account, pod, plan):
    balance = float(account["clientBalance"])
    account_rate = float(account["currentSpendPerHr"])
    pod_rate = float(pod["costPerHr"])
    if not all(math.isfinite(value) and value >= 0
               for value in (balance, account_rate, pod_rate)) or pod_rate <= 0:
        raise IntegrityError("invalid RunPod balance or rate")
    if account.get("isAutoPayEnabled") is not False:
        raise IntegrityError("pilot requires auto-pay disabled")
    if pod.get("desiredStatus") != "EXITED":
        raise IntegrityError("pilot Pod must be EXITED before start")
    combined_rate = account_rate + pod_rate
    available = balance - plan.reserve_usd - plan.billing_delay_usd
    required_seconds = (plan.startup_timeout_seconds + plan.requested_work_seconds
                        + plan.artifact_reserve_seconds + plan.shutdown_reserve_seconds
                        + math.ceil(8 * plan.notification_timeout_seconds))
    required_usd = combined_rate * required_seconds / 3600
    if available < required_usd:
        raise IntegrityError("insufficient balance above reserve for bounded pilot lease")
    maximum_seconds = math.floor(available * 3600 / combined_rate)
    return {"balance_usd": balance, "account_rate_usd_per_hour": account_rate,
            "pod_rate_usd_per_hour": pod_rate, "combined_rate_usd_per_hour": combined_rate,
            "required_seconds": required_seconds, "required_usd": required_usd,
            "maximum_seconds_above_reserves": maximum_seconds,
            "minimum_final_balance_usd": plan.reserve_usd + plan.billing_delay_usd}


def convert_source_flac(source_flac, output_wav, provenance_path, *, ffmpeg="ffmpeg"):
    source_flac, output_wav = Path(source_flac), Path(output_wav)
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    before = file_digest(source_flac)
    if output_wav.exists() or Path(provenance_path).exists():
        raise IntegrityError("conversion output exists; preserve it and choose a new path")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-nostdin", "-v", "error", "-i", str(source_flac),
               "-map_metadata", "-1", "-acodec", "pcm_s16le", "-ar", "16000",
               "-ac", "1", str(output_wav)]
    subprocess.run(command, check=True, timeout=180)
    if file_digest(source_flac) != before:
        raise IntegrityError("source FLAC changed during conversion")
    version = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True,
                             text=True, timeout=15).stdout.splitlines()[0]
    body = {"format": "mas-pilot-source-conversion-1", "source_path": str(source_flac.resolve()),
            "started_at": started_at, "elapsed_seconds": time.monotonic() - started_monotonic,
            "source_sha256": before, "source_bytes": source_flac.stat().st_size,
            "output_path": str(output_wav.resolve()), "output_sha256": file_digest(output_wav),
            "output_bytes": output_wav.stat().st_size, "ffmpeg_version": version,
            "parameters": command[1:-1]}
    atomic_json(provenance_path, {"data": body, "sha256": digest(body)})
    return body


class ProgressGuard:
    def __init__(self, idle_timeout_seconds, clock=time.monotonic):
        self.idle_timeout_seconds = idle_timeout_seconds
        self.clock = clock
        self.last_progress_at = clock()
        self.completed_units = 0
        self.artifact_bytes = 0

    def update(self, *, completed_units, artifact_bytes):
        if completed_units < self.completed_units or artifact_bytes < self.artifact_bytes:
            raise IntegrityError("pilot progress counters moved backwards")
        if completed_units > self.completed_units or artifact_bytes > self.artifact_bytes:
            self.last_progress_at = self.clock()
            self.completed_units = completed_units
            self.artifact_bytes = artifact_bytes

    def check(self):
        if self.clock() - self.last_progress_at > self.idle_timeout_seconds:
            raise TimeoutError("pilot made no measured unit or artifact progress")


def _notify_bounded(notify, args, timeout_seconds=5):
    result = []
    def send():
        try:
            result.append(notify(*args))
        except Exception as exc:
            result.append({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    return result[0] if result else {"status": "timeout"}


def deadline_file_digest(path, deadline, clock=time.monotonic):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if clock() >= deadline:
                raise TimeoutError("artifact hashing exceeded its lease reserve")
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return result.hexdigest()
            result.update(chunk)


def _terminate_process_tree(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5, check=False)


def run_progress_process(command, progress_path, *, timeout_seconds, idle_timeout_seconds,
                         log_path, poll_seconds=0.1):
    started = time.monotonic()
    guard = ProgressGuard(idle_timeout_seconds)
    progress_path, log_path = Path(progress_path), Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            list(command), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=(os.name == "posix"),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
        try:
            while process.poll() is None:
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    guard.update(completed_units=int(progress["completed_units"]),
                                 artifact_bytes=int(progress["artifact_bytes"]))
                guard.check()
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("pilot process exceeded its work deadline")
                time.sleep(poll_seconds)
            if process.returncode:
                raise RuntimeError(f"pilot process exited with code {process.returncode}")
        finally:
            if process.poll() is None:
                _terminate_process_tree(process)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)


def run_bounded_pilot(provider, command, progress_path, artifact_paths, output_dir, plan, notify):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    events = []

    def record(stage, status, details):
        event = {"stage": stage, "status": status, "details": details,
                 "observed_monotonic": time.monotonic()}
        events.append(event)
        notification = _notify_bounded(
            notify, (11, f"pilot {stage} {status}", json.dumps(details, sort_keys=True)),
            plan.notification_timeout_seconds)
        event["notification"] = notification
        atomic_json(output_dir / "events.json", events)

    started = False
    result = None
    guardian = None
    stage = "preflight"
    try:
        record("preflight", "START", {})
        account, pod = provider.account(), provider.pod()
        lease = lease_preflight(account, pod, plan)
        record("preflight", "PASS", lease)
        stage = "startup"
        record("startup", "START", {"timeout_seconds": plan.startup_timeout_seconds})
        account, pod = provider.account(), provider.pod()
        lease = lease_preflight(account, pod, plan)
        if isinstance(provider, RunPodPilotProvider):
            from .pilot_guardian import GuardianProcess
            guardian = GuardianProcess(
                output_dir, provider.pod_id,
                time.time() + plan.startup_timeout_seconds + plan.requested_work_seconds
                + plan.artifact_reserve_seconds + math.ceil(8 * plan.notification_timeout_seconds),
                plan.shutdown_reserve_seconds, lease["minimum_final_balance_usd"],
                lease["pod_rate_usd_per_hour"])
            startup_deadline = time.monotonic() + plan.startup_timeout_seconds
            guardian.arm(provider.api_key, timeout=max(0.001, startup_deadline - time.monotonic()))
        started = True
        if guardian is not None:
            guardian.start(max(0.001, startup_deadline - time.monotonic()))
        else:
            startup_deadline = time.monotonic() + plan.startup_timeout_seconds
            provider.start(max(0.001, startup_deadline - time.monotonic()))
        running = provider.wait_running(max(0.001, startup_deadline - time.monotonic()))
        if running.get("desiredStatus") != "RUNNING":
            raise IntegrityError("external observer did not confirm RUNNING")
        record("startup", "PASS", {"pod_status": running["desiredStatus"]})
        stage = "inference"
        record("inference", "START", {"timeout_seconds": plan.requested_work_seconds})
        run_progress_process(command, progress_path, timeout_seconds=plan.requested_work_seconds,
                             idle_timeout_seconds=plan.idle_timeout_seconds,
                             log_path=output_dir / "worker.log")
        artifacts = []
        artifact_deadline = time.monotonic() + plan.artifact_reserve_seconds
        for path in artifact_paths:
            path = Path(path)
            if time.monotonic() >= artifact_deadline:
                raise TimeoutError("artifact retrieval exceeded its lease reserve")
            artifacts.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                              "sha256": deadline_file_digest(path, artifact_deadline)})
        record("inference", "PASS", {"artifacts": artifacts})
        result = artifacts
        return {"lease": lease, "artifacts": artifacts}
    except Exception as exc:
        record(stage, "FAIL", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if started:
            try:
                try:
                    record("shutdown", "START", {"timeout_seconds": plan.shutdown_reserve_seconds})
                finally:
                    shutdown_deadline = time.monotonic() + plan.shutdown_reserve_seconds
                    provider.stop(min(15, plan.shutdown_reserve_seconds))
                stopped = provider.wait_exited(max(0.001, shutdown_deadline - time.monotonic()))
                if stopped.get("desiredStatus") != "EXITED":
                    raise IntegrityError("external observer did not confirm EXITED")
                final_account, final_pod = provider.account(), provider.pod()
                if final_pod.get("desiredStatus") != "EXITED":
                    raise IntegrityError("final provider snapshot is not EXITED")
                record("shutdown", "PASS", {"pod_status": "EXITED",
                                              "balance_usd": float(final_account["clientBalance"])})
                safe_pod = {key: final_pod.get(key) for key in
                            ("id", "desiredStatus", "costPerHr", "networkVolumeId")}
                receipt = {"pod": safe_pod, "account": final_account,
                           "events_sha256": file_digest(output_dir / "events.json")}
                atomic_json(output_dir / "shutdown-receipt.json",
                            {"data": receipt, "sha256": digest(receipt)})
                if guardian is not None:
                    guardian.disarm_after_exit(max(0.001, shutdown_deadline - time.monotonic()))
            except Exception as exc:
                record("shutdown", "FAIL", {"error_type": type(exc).__name__, "error": str(exc)})
                raise
