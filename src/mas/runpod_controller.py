import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

from .config import ROOT, episode_dir
from .engine.tr_correction import validate_tr_correction_output
from .id_return_preflight import preflight_local_id_return
from .engine.download import _validated_cookie_file, validate_download
from .remote import RemoteVerificationError, _run_watchdog
from .reliability import BudgetExceeded, RunBudget, atomic_json, digest
from .delivery import READY_FOR_DELIVERY, WAIT_MP4_SAMPLE, publish_local_delivery, safe_relative, validate_delivery
from .hashing import sha256_file
from .source_discovery import discover_episode_metadata
from .runpod_capacity import (CapacityLease, CapacityPlan, CapacityProvider, CapacityReadinessError,
                             load_storage_quote)


class RunPodControllerError(RuntimeError):
    pass


def _episode_budget(local_root, episode, *, now=None):
    now = now or datetime.now(timezone.utc)
    budget_path = Path(local_root) / "work" / "controller_budget.json"
    starts = []
    excluded_wait = 0.0
    if budget_path.exists():
        saved = json.loads(budget_path.read_text(encoding="utf-8"))
        body = saved.get("data")
        if not isinstance(body, dict) or saved.get("sha256") != digest(body) or body.get("episode") != episode:
            raise RunPodControllerError("episode budget checkpoint integrity mismatch")
        starts.append(datetime.fromisoformat(body["started_at"]))
        excluded_wait = body.get("excluded_wait_seconds", 0.0)
        if not isinstance(excluded_wait, (int, float)) or not math.isfinite(excluded_wait) or excluded_wait < 0:
            raise RunPodControllerError("invalid excluded wait duration")
        wait = body.get("wait")
        if wait:
            for key in ("evidence", "shutdown"):
                if sha256_file(safe_relative(local_root, wait[key + "_path"])) != wait[key + "_sha256"]:
                    raise RunPodControllerError("verified wait evidence changed")
            paused = datetime.fromisoformat(wait["paused_at"])
            elapsed = (now - paused).total_seconds()
            if elapsed < 0:
                raise RunPodControllerError("wait clock moved backwards")
            excluded_wait += elapsed
    preflight_path = Path(local_root) / "work" / "controller-preflight-only.json"
    excluded = {}
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("sha256") != digest(preflight.get("data")):
            raise RunPodControllerError("preflight-only log evidence integrity mismatch")
        excluded = preflight["data"]
    for log_path in (Path(local_root) / "logs").glob("run-*.log"):
        with log_path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        if log_path.name in excluded:
            if digest(first_line) != excluded[log_path.name]:
                raise RunPodControllerError("preflight-only log identity changed")
            continue
        try:
            event = json.loads(first_line)
            if event.get("event") == "run_started" and event.get("episode") == episode:
                starts.append(datetime.fromisoformat(event["timestamp"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise RunPodControllerError(f"cannot establish episode budget from {log_path.name}") from exc
    if any(start.tzinfo is None for start in starts):
        raise RunPodControllerError("episode budget timestamps must include timezone")
    started_at = min(starts) if starts else now
    try:
        limit = float(os.getenv("MAS_EPISODE_BUDGET_SECONDS", "14400"))
        budget = RunBudget((started_at + timedelta(seconds=excluded_wait)).isoformat(), limit_seconds=limit)
    except ValueError as exc:
        raise RunPodControllerError("MAS_EPISODE_BUDGET_SECONDS must be finite and positive") from exc
    body = {"episode": episode, "started_at": started_at.isoformat(),
            "limit_seconds": budget.limit_seconds, "excluded_wait_seconds": excluded_wait}
    atomic_json(budget_path, {"data": body, "sha256": digest(body)})
    budget.check(now)
    return budget


def _pause_episode_budget(local_root, episode, reason, evidence_path, shutdown_path):
    if reason not in (20, 21, WAIT_MP4_SAMPLE):
        raise RunPodControllerError("only a verified human handoff may pause the budget")
    local_root = Path(local_root)
    if not Path(evidence_path).is_file() or not Path(shutdown_path).is_file():
        raise RunPodControllerError("verified handoff/release evidence is missing")
    shutdown = json.loads(Path(shutdown_path).read_text(encoding="utf-8"))
    data = shutdown.get("data")
    if (not isinstance(data, dict) or shutdown.get("sha256") != digest(data) or not data.get("owned_pods")
            or any(item.get("status") != "ABSENT" for item in data["owned_pods"])):
        raise RunPodControllerError("budget wait requires verified external Pod absence")
    budget_path = local_root / "work" / "controller_budget.json"
    saved = json.loads(budget_path.read_text(encoding="utf-8"))
    body = saved["data"]
    if saved.get("sha256") != digest(body) or body.get("episode") != episode:
        raise RunPodControllerError("episode budget checkpoint integrity mismatch")
    wait = {"reason": reason, "paused_at": datetime.now(timezone.utc).isoformat()}
    for key, path in (("evidence", evidence_path), ("shutdown", shutdown_path)):
        relative = Path(path).relative_to(local_root).as_posix()
        wait[key + "_path"] = relative
        wait[key + "_sha256"] = sha256_file(safe_relative(local_root, relative))
    if body.get("wait"):
        prior = dict(body["wait"])
        prior["paused_at"] = wait["paused_at"]
        if prior != wait:
            raise RunPodControllerError("paused handoff evidence changed")
        return
    body["wait"] = wait
    atomic_json(budget_path, {"data": body, "sha256": digest(body)})


def _validate_local_tr_return(local_root, name):
    pack = Path(local_root) / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip"
    returned = Path(local_root) / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
    if returned.is_file():
        if not pack.is_file():
            raise RunPodControllerError(
                f"local Turkish correction pack is missing: {pack}"
            )
        try:
            validate_tr_correction_output(pack, returned)
        except Exception as exc:
            raise RunPodControllerError(
                "local Turkish correction return does not match the current pack"
            ) from exc


class RunPodClient:
    def __init__(self, pod_id, api_key, *, timeout=15, attempts=3, sleep=time.sleep, budget=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id or ""):
            raise RunPodControllerError("RUNPOD_POD_ID is missing or invalid")
        if not api_key or "\r" in api_key or "\n" in api_key:
            raise RunPodControllerError("RUNPOD_API_KEY is missing or invalid")
        self.pod_id = pod_id
        self.api_key = api_key
        self.timeout = timeout
        self.attempts = attempts
        self.sleep = sleep
        self.budget = budget

    def _request_url(self, method, url, *, payload=None, attempts=None):
        data = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "ma-sub-pilot/1.0",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers=headers,
        )
        last_error = None
        request_attempts = self.attempts if attempts is None else attempts
        for attempt in range(request_attempts):
            request_timeout = min(self.timeout, self.budget.check()) if self.budget else self.timeout
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    payload = response.read()
                    if response.status < 200 or response.status >= 300:
                        raise RunPodControllerError(f"RunPod returned HTTP {response.status}")
                    try:
                        result = json.loads(payload) if payload else {}
                    except (ValueError, UnicodeError):
                        raise RunPodControllerError("RunPod returned invalid JSON") from None
                    if not isinstance(result, dict):
                        raise RunPodControllerError("RunPod returned a non-object response")
                    return result
            except urllib.error.HTTPError as exc:
                detail = None
                try:
                    payload = json.loads(exc.read(4096))
                    if isinstance(payload, dict):
                        for key in ("error", "message", "detail"):
                            if isinstance(payload.get(key), str):
                                detail = payload[key].strip()[:500]
                                break
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
                message = f"HTTP {exc.code}"
                if detail:
                    message += f": {detail}"
                last_error = RunPodControllerError(message)
                if attempt + 1 < request_attempts:
                    self.sleep(min(2 ** attempt, self.budget.check()) if self.budget else 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, RunPodControllerError) as exc:
                last_error = exc
                if attempt + 1 < request_attempts:
                    self.sleep(min(2 ** attempt, self.budget.check()) if self.budget else 2 ** attempt)
        raise RunPodControllerError(
            f"RunPod request failed after {request_attempts} attempts: {last_error}"
        )

    def request(self, method, suffix=""):
        return self._request_url(
            method,
            f"https://rest.runpod.io/v1/pods/{self.pod_id}{suffix}",
        )

    def get(self):
        return self.request("GET")

    def start(self):
        return self.request("POST", "/start")

    def stop(self):
        return self.request("POST", "/stop")

    def terminate(self):
        return self._request_url(
            "DELETE",
            f"https://rest.runpod.io/v1/pods/{self.pod_id}",
            attempts=1,
        )

    def get_network_volume(self, volume_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id or ""):
            raise RunPodControllerError("RunPod network volume ID is invalid")
        return self._request_url(
            "GET",
            f"https://rest.runpod.io/v1/networkvolumes/{volume_id}",
        )

    def create_pod(self, payload):
        return self._request_url(
            "POST",
            "https://rest.runpod.io/v1/pods",
            payload=payload,
            attempts=1,
        )

    def wait(self, predicate, description, *, timeout, poll=5):
        started = time.monotonic()
        last = None
        while time.monotonic() - started < timeout:
            last = self.get()
            if predicate(last):
                return last
            elapsed = time.monotonic() - started
            status = last.get("desiredStatus", "UNKNOWN")
            print(f"[RUNPOD] waiting for {description}: status={status}; elapsed={elapsed:.1f}s")
            remaining = max(0, timeout - (time.monotonic() - started))
            if self.budget:
                remaining = min(remaining, self.budget.check())
            self.sleep(min(poll, remaining))
        status = (last or {}).get("desiredStatus", "UNKNOWN")
        raise RunPodControllerError(f"RunPod {description} timed out after {timeout}s; status={status}")


def _required_environment():
    names = (
        "RUNPOD_POD_ID",
        "RUNPOD_API_KEY",
        "MAS_RUNPOD_SSH_KEY",
        "MAS_GMAIL_ADDRESS",
        "MAS_GMAIL_APP_PASSWORD",
        "MAS_NOTIFY_TO",
        "MAS_DRIVE_STRICT_REMOTE",
    )
    values = {}
    missing = []
    for name in names:
        value = os.getenv(name)
        if not value:
            missing.append(name)
        elif "\r" in value or "\n" in value:
            raise RunPodControllerError(f"{name} must not contain line breaks")
        else:
            values[name] = value
    if missing:
        raise RunPodControllerError("missing environment: " + ", ".join(missing))
    cookie = os.getenv("MAS_YTDLP_COOKIES")
    if cookie:
        if "\r" in cookie or "\n" in cookie:
            raise RunPodControllerError("MAS_YTDLP_COOKIES must not contain line breaks")
        values["MAS_YTDLP_COOKIES"] = cookie
    return values


def _rclone_config():
    configured = os.getenv("MAS_RCLONE_CONFIG") or os.getenv("RCLONE_CONFIG")
    path = Path(configured) if configured else Path(os.getenv("APPDATA", "")) / "rclone" / "rclone.conf"
    if not path.is_file() or path.stat().st_size == 0:
        raise RunPodControllerError(f"rclone config is missing: {path}")
    return path.resolve()


def _local_preflight(values):
    for executable in ("git", "ssh", "scp"):
        if not shutil.which(executable):
            raise RunPodControllerError(f"required executable is missing: {executable}")
    key = Path(values["MAS_RUNPOD_SSH_KEY"]).resolve()
    if not key.is_file() or key.stat().st_size == 0:
        raise RunPodControllerError(f"MAS_RUNPOD_SSH_KEY file is missing or empty: {key}")
    cookie = values.get("MAS_YTDLP_COOKIES")
    if cookie:
        path = Path(cookie).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise RunPodControllerError(f"MAS_YTDLP_COOKIES file is missing or empty: {path}")
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise RunPodControllerError("git status failed")
    if result.stdout.strip():
        raise RunPodControllerError("repository must be clean before a RunPod episode run")
    _validated_cookie_file(cookie)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RunPodControllerError("could not resolve a full Git commit SHA")
    return commit, _rclone_config()


def _ssh_endpoint(pod):
    host = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") or mappings.get(22)
    if not host or not port:
        return None
    return str(host), str(port)


def _startup_mode(pod):
    status = pod.get("desiredStatus", "UNKNOWN")
    if status == "EXITED":
        return "start"
    if status == "RUNNING" and os.getenv("MAS_RUNPOD_ADOPT_RUNNING") == "1":
        return "adopt"
    raise RunPodControllerError(f"pod must be EXITED before an automatic run; status={status}")


def _is_capacity_error(exc):
    return "not enough free gpus" in str(exc).lower()


def _persist_windows_user_pod_id(pod_id):
    if os.name != "nt":
        raise RunPodControllerError(
            "automatic Pod migration requires Windows user-environment persistence"
        )
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        winreg.SetValueEx(key, "RUNPOD_POD_ID", 0, winreg.REG_SZ, pod_id)
    os.environ["RUNPOD_POD_ID"] = pod_id


def _configured_gpu_type_ids():
    pool = os.getenv("MAS_RUNPOD_GPU_TYPE_IDS")
    values = (
        [value.strip() for value in pool.split("|")]
        if pool
        else [str(os.getenv("MAS_RUNPOD_GPU_TYPE_ID") or "").strip()]
    )
    if (
        not 1 <= len(values) <= 8
        or any(not value or not re.fullmatch(r"[A-Za-z0-9 ._-]+", value) for value in values)
        or len(set(values)) != len(values)
    ):
        raise RunPodControllerError(
            "MAS_RUNPOD_GPU_TYPE_IDS must contain 1-8 unique GPU IDs separated by |"
        )
    return values


def _migrate_capacity_bound_pod(client, pod):
    volume_id = os.getenv("MAS_RUNPOD_NETWORK_VOLUME_ID")
    data_center_id = os.getenv("MAS_RUNPOD_DATA_CENTER_ID")
    gpu_type_ids = _configured_gpu_type_ids()
    try:
        max_cost = float(os.getenv("MAS_RUNPOD_MAX_COST_PER_HR", "0.75"))
    except ValueError:
        raise RunPodControllerError("maximum Pod hourly cost must be finite and positive") from None
    if not math.isfinite(max_cost) or max_cost <= 0:
        raise RunPodControllerError("maximum Pod hourly cost must be finite and positive")
    if not volume_id or not data_center_id:
        raise RunPodControllerError(
            "automatic Pod migration requires MAS_RUNPOD_NETWORK_VOLUME_ID, "
            "MAS_RUNPOD_DATA_CENTER_ID and a configured GPU type"
        )
    if pod.get("desiredStatus") != "EXITED":
        raise RunPodControllerError("refusing migration because the old Pod is not EXITED")
    if pod.get("networkVolumeId") != volume_id or pod.get("volumeInGb") not in (0, None):
        raise RunPodControllerError(
            "refusing migration because the persistent volume boundary changed"
        )
    volume = client.get_network_volume(volume_id)
    if volume.get("id") != volume_id or volume.get("dataCenterId") != data_center_id:
        raise RunPodControllerError(
            "refusing migration because the network volume identity changed"
        )
    payload = {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": int(pod.get("containerDiskInGb") or 30),
        "dataCenterIds": [data_center_id],
        "dataCenterPriority": "availability",
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "env": dict(pod.get("env") or {}),
        "gpuCount": 1,
        "gpuTypeIds": gpu_type_ids,
        "gpuTypePriority": "availability",
        "imageName": pod.get("imageName"),
        "interruptible": False,
        "name": pod.get("name") or "muhtemel-ask-production",
        "networkVolumeId": volume_id,
        "ports": list(pod.get("ports") or ["8888/http", "22/tcp"]),
        "supportPublicIp": True,
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
    }
    if not payload["imageName"]:
        raise RunPodControllerError("refusing migration because the Pod image is missing")
    old_id = client.pod_id
    client.terminate()
    try:
        created = client.create_pod(payload)
    except RunPodControllerError as exc:
        raise RunPodControllerError(
            f"old Pod {old_id} was terminated but its network volume is retained; "
            f"replacement creation failed: {exc}"
        ) from exc
    new_id = created.get("id")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", new_id or ""):
        raise RunPodControllerError("replacement Pod response failed identity validation")
    replacement = RunPodClient(
        new_id,
        client.api_key,
        timeout=client.timeout,
        attempts=client.attempts,
        sleep=client.sleep,
        budget=getattr(client, "budget", None),
    )
    try:
        if created.get("networkVolumeId") != volume_id:
            raise RunPodControllerError("replacement Pod response failed volume identity validation")
        cost = float(created.get("costPerHr") or created.get("adjustedCostPerHr") or 0)
        if not math.isfinite(cost) or cost <= 0 or cost > max_cost:
            raise RunPodControllerError(
                f"replacement Pod hourly cost {cost} exceeds allowed {max_cost}"
            )
        _persist_windows_user_pod_id(new_id)
    except Exception:
        replacement.budget = None
        replacement.terminate()
        raise
    print(
        f"[RUNPOD] migrated capacity-bound pod {old_id} -> {new_id}; "
        f"network volume preserved; hourly cost={cost:.2f}"
    )
    return replacement


def _stream(chunk, target):
    target.write(chunk.decode("utf-8", "replace"))
    target.flush()


def _network(
    command,
    *,
    idle_timeout=180,
    total_timeout=1800,
    capture=False,
    progress_probe=None,
):
    try:
        return _run_watchdog(
            command,
            idle_timeout=idle_timeout,
            total_timeout=total_timeout,
            stdout_handler=None if capture else lambda chunk: _stream(chunk, sys.stdout),
            stderr_handler=None if capture else lambda chunk: _stream(chunk, sys.stderr),
            progress_probe=progress_probe,
        )
    except RemoteVerificationError as exc:
        raise RunPodControllerError(str(exc)) from exc


def _network_retry(
    command, *, attempts=3, idle_timeout=60, total_timeout=1800, capture=False, budget=None,
    deadline=None, progress_probe=None,
):
    operation_deadline = time.monotonic() + total_timeout
    deadline = min(deadline, operation_deadline) if deadline is not None else operation_deadline
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if budget:
            remaining = min(remaining, budget.check())
        if remaining <= 0:
            raise RunPodControllerError("network command total retry budget expired")
        try:
            return _network(
                command,
                idle_timeout=idle_timeout,
                total_timeout=remaining,
                capture=capture,
                progress_probe=progress_probe,
            )
        except RunPodControllerError:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), max(0, deadline - time.monotonic()))
            if budget:
                delay = min(delay, budget.check())
            print(f"[RUNPOD] network command retry {attempt + 1}/{attempts} in {delay}s")
            time.sleep(delay)


def _remote_file_signature(ssh, remote_path, *, budget=None):
    quoted = shlex.quote(remote_path)
    output = _network_retry(
        ssh
        + [
            "set -euo pipefail; "
            f"stat -c %s -- {quoted}; "
            f"sha256sum -- {quoted}"
        ],
        idle_timeout=60,
        total_timeout=300,
        capture=True,
        budget=budget,
    )
    match = re.fullmatch(
        rb"\s*(\d+)\s*\r?\n([0-9a-f]{64})\s+[^\r\n]+\s*",
        output,
    )
    if not match:
        raise RunPodControllerError("remote file byte/SHA-256 readback is invalid")
    return int(match.group(1)), match.group(2).decode("ascii")


def _remote_growth_probe(ssh, remote_path, *, budget=None):
    last_size = -1
    quoted = shlex.quote(remote_path)

    def probe():
        nonlocal last_size
        try:
            output = _network_retry(
                ssh + [f"stat -c %s -- {quoted}"],
                attempts=1,
                idle_timeout=15,
                total_timeout=30,
                capture=True,
                budget=budget,
            ).strip()
            size = int(output)
        except (RunPodControllerError, ValueError):
            return False
        advanced = size > last_size
        last_size = size
        return advanced

    return probe


def _local_growth_probe(path):
    path = Path(path)
    last_size = -1

    def probe():
        nonlocal last_size
        try:
            size = path.stat().st_size
        except OSError:
            return False
        advanced = size > last_size
        last_size = size
        return advanced

    return probe


def _upload_episode_file_verified(
    source,
    remote_path,
    *,
    ssh,
    scp,
    host,
    budget=None,
    immutable=False,
    transfer_timeout=300,
    monitor_remote_growth=False,
):
    source = Path(source)
    expected_size = source.stat().st_size
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if budget:
                budget.check()
            digest.update(chunk)
    expected_sha256 = digest.hexdigest()
    if immutable:
        quoted = shlex.quote(remote_path)
        existence = _network_retry(ssh + [f"if test -L {quoted}; then echo UNSAFE; "
                                   f"elif test -f {quoted}; then echo PRESENT; "
                                   f"elif test -e {quoted}; then echo UNSAFE; else echo MISSING; fi"],
                                   capture=True, total_timeout=30, budget=budget).strip()
        if existence == b"PRESENT":
            if _remote_file_signature(ssh, remote_path, budget=budget) != (expected_size, expected_sha256):
                raise RunPodControllerError("immutable remote source identity differs; preserve it")
            return {"bytes": expected_size, "sha256": expected_sha256}
        if existence != b"MISSING":
            raise RunPodControllerError("immutable remote source path is unsafe")
    partial = (
        "/workspace/.mas-upload/audio-review-overrides-"
        f"{expected_sha256[:16]}.partial"
    )
    remote_parent = str(Path(remote_path).parent).replace("\\", "/")
    _network_retry(
        ssh + ["install -d -m 700 -- " + shlex.quote(remote_parent)],
        idle_timeout=60,
        total_timeout=300,
        budget=budget,
    )
    transfer_options = {
        "idle_timeout": 60,
        "total_timeout": transfer_timeout,
        "budget": budget,
    }
    if monitor_remote_growth:
        transfer_options.update(
            attempts=1,
            progress_probe=_remote_growth_probe(ssh, partial, budget=budget),
        )
    _network_retry(
        scp + [str(source), f"root@{host}:{partial}"],
        **transfer_options,
    )
    partial_size, partial_sha256 = _remote_file_signature(ssh, partial, budget=budget)
    if (partial_size, partial_sha256) != (expected_size, expected_sha256):
        raise RunPodControllerError(
            "uploaded episode file byte/SHA-256 readback mismatch"
        )
    _network_retry(
        ssh
        + [
            "set -euo pipefail; "
            f"chmod 600 -- {shlex.quote(partial)}; "
            f"mv -f -- {shlex.quote(partial)} {shlex.quote(remote_path)}"
        ],
        idle_timeout=60,
        total_timeout=300,
        budget=budget,
    )
    final_size, final_sha256 = _remote_file_signature(ssh, remote_path, budget=budget)
    if (final_size, final_sha256) != (expected_size, expected_sha256):
        raise RunPodControllerError(
            "installed episode file byte/SHA-256 readback mismatch"
        )
    return {"bytes": expected_size, "sha256": expected_sha256}


def _upload_verified_local_source(
    local_root,
    remote_root,
    source_url,
    *,
    ssh,
    scp,
    host,
    temporary,
    budget=None,
):
    source_dir = Path(local_root) / "source"
    marker_path = source_dir / "download.done.json"
    if not marker_path.is_file():
        return None
    if not validate_download(marker_path, url=source_url):
        raise RunPodControllerError("local source checkpoint is invalid; refusing remote seed")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) - {"video", "metadata", "captions"}:
        raise RunPodControllerError("local source checkpoint has unexpected outputs")

    local_source_root = source_dir.resolve()
    remote_outputs = {}
    receipts = {}
    for key, record in outputs.items():
        if not isinstance(record, dict):
            raise RunPodControllerError("local source checkpoint output is invalid")
        source = Path(str(record.get("path", "")))
        if source.is_symlink() or source.resolve().parent != local_source_root:
            raise RunPodControllerError("local source checkpoint path is outside the source directory")
        remote_path = f"{remote_root}/source/{source.name}"
        receipts[key] = _upload_episode_file_verified(
            source,
            remote_path,
            ssh=ssh,
            scp=scp,
            host=host,
            budget=budget,
            immutable=True,
            transfer_timeout=1800,
            monitor_remote_growth=True,
        )
        remote_record = dict(record)
        remote_record["path"] = remote_path
        remote_outputs[key] = remote_record

    remote_marker = dict(marker)
    remote_marker["outputs"] = remote_outputs
    seed_marker = Path(temporary) / "download.done.json"
    seed_marker.write_text(
        json.dumps(remote_marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipts["marker"] = _upload_episode_file_verified(
        seed_marker,
        f"{remote_root}/source/download.done.json",
        ssh=ssh,
        scp=scp,
        host=host,
        budget=budget,
        immutable=True,
    )
    return receipts


def _upload_audio_review_overrides(local_root, remote_root, *, ssh, scp, host, budget=None):
    source = Path(local_root) / "review" / "audio_review_overrides.json"
    if not source.is_file():
        return None
    return _upload_episode_file_verified(
        source,
        f"{remote_root}/review/audio_review_overrides.json",
        ssh=ssh,
        scp=scp,
        host=host,
        budget=budget,
    )


def _upload_speaker_evidence(local_root, remote_root, *, ssh, scp, host, budget=None):
    source = Path(local_root) / "review" / "speaker_evidence_v1.json"
    if not source.is_file():
        return None
    return _upload_episode_file_verified(
        source,
        f"{remote_root}/review/speaker_evidence_v1.json",
        ssh=ssh,
        scp=scp,
        host=host,
        budget=budget,
    )


def _ssh_args(key, host, port):
    return [
        "ssh",
        "-n",
        "-T",
        "-i",
        str(key),
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=2",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        f"root@{host}",
    ]


def _scp_args(key, host, port):
    return [
        "scp",
        "-i",
        str(key),
        "-P",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
    ]


def _wait_for_ssh(key, host, port, *, timeout=300):
    started = time.monotonic()
    command = _ssh_args(key, host, port) + ["true"]
    attempt = 0
    while time.monotonic() - started < timeout:
        attempt += 1
        try:
            result = subprocess.run(command, capture_output=True,
                                    timeout=min(20, timeout - (time.monotonic() - started)), check=False)
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            return
        if result is not None and b"Permission denied" in result.stderr:
            raise RunPodControllerError(
                "SSH authentication rejected; verify the Pod PUBLIC_KEY and SSH_PUBLIC_KEY"
            )
        elapsed = time.monotonic() - started
        print(f"[RUNPOD] waiting for SSH: attempt={attempt}; elapsed={elapsed:.1f}s")
        time.sleep(min(5, max(0, timeout - (time.monotonic() - started))))
    raise RunPodControllerError(f"SSH readiness timed out after {timeout}s")


def _write_runtime_env(path, values, commit):
    remote_values = {
        "RUNPOD_POD_ID": values["RUNPOD_POD_ID"],
        "RUNPOD_API_KEY": values["RUNPOD_API_KEY"],
        "MAS_GMAIL_ADDRESS": values["MAS_GMAIL_ADDRESS"],
        "MAS_GMAIL_APP_PASSWORD": values["MAS_GMAIL_APP_PASSWORD"],
        "MAS_NOTIFY_TO": values["MAS_NOTIFY_TO"],
        "MAS_DRIVE_STRICT_REMOTE": values["MAS_DRIVE_STRICT_REMOTE"],
        "RCLONE_CONFIG": "/workspace/.mas-secrets/rclone.conf",
        "MAS_RCLONE_CONFIG": "/workspace/.mas-secrets/rclone.conf",
        "MAS_GIT_COMMIT": commit,
        "MAS_VENV_DIR": "/workspace/ma-sub/.venv",
        "UV_CACHE_DIR": "/workspace/.cache/uv",
        "UV_HTTP_TIMEOUT": "120",
        "UV_HTTP_RETRIES": "3",
        "UV_CONCURRENT_DOWNLOADS": "4",
        "HF_HOME": "/workspace/.cache/huggingface",
        "TORCH_HOME": "/workspace/.cache/torch",
        "MAS_EXTERNAL_RUNPOD_CONTROLLER": "1",
    }
    for name in ("MAS_NETWORK_VOLUME_QUOTA_BYTES", "MAS_EPISODE"):
        if name in values:
            remote_values[name] = values[name]
    if values.get("MAS_YTDLP_COOKIES"):
        remote_values["MAS_YTDLP_COOKIES"] = "/workspace/.mas-secrets/youtube-cookies.txt"
    for name in ("MAS_MAX_RUNTIME_SECONDS", "MAS_IDLE_TIMEOUT_SECONDS", "MAS_MP4_TARGET_GB",
                 "MAS_MP4_ENCODER", "MAS_MP4_ENCODER_OPTIONS"):
        if os.getenv(name):
            remote_values[name] = os.environ[name]
    lines = [f"export {name}={shlex.quote(value)}" for name, value in remote_values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


@contextmanager
def _controller_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunPodControllerError("another production controller owns the local lock") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _PaidBudget:
    def __init__(self, episode_budget, lease):
        self.episode_budget, self.lease = episode_budget, lease

    def check(self):
        remaining = min(self.episode_budget.check(), self.lease.remaining_work_seconds())
        if remaining <= 0:
            raise BudgetExceeded("paid work allowance expired; preserving shutdown reserve")
        return remaining


def _prepare_official_source(local_root, episode, source_url, cookie):
    source_dir = local_root / "source"
    url_path = source_dir / "source.url"
    metadata_path = source_dir / "official-source.json"
    existing = url_path.read_text(encoding="utf-8").strip() if url_path.is_file() else None
    if source_url and existing and source_url != existing:
        raise RunPodControllerError("source URL cannot change after episode initialization")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        from .source_discovery import CHANNEL_VIDEOS_URL, is_exact_episode_title
        if (metadata.get("episode") != episode or metadata.get("url") != existing
                or metadata.get("channel_url") != CHANNEL_VIDEOS_URL
                or not is_exact_episode_title(metadata.get("title"), episode)):
            raise RunPodControllerError("saved official source identity mismatch")
        return existing
    metadata = discover_episode_metadata(episode, cookies_file=cookie)
    if (source_url or existing) and metadata["url"] != (source_url or existing):
        raise RunPodControllerError("requested source is not the exact official episode")
    from .engine.download import atomic_write_bytes
    source_dir.mkdir(parents=True, exist_ok=True)
    if not existing:
        atomic_write_bytes(url_path, (metadata["url"] + "\n").encode("utf-8"))
    atomic_json(metadata_path, metadata)
    return metadata["url"]


def run_remote_episode(episode, source_url=None):
    if type(episode) is not int or episode <= 0:
        raise RunPodControllerError("episode must be a positive integer")
    with _controller_lock(ROOT / "var" / "production-controller.lock"):
        values = _required_environment()
        commit, rclone_config = _local_preflight(values)
        local_root = episode_dir(episode)
        _validate_local_tr_return(local_root, f"Muhtemel Ask {episode}.Bolum")
        preflight_local_id_return(local_root, episode, ROOT / "config" / "production")
        last_status = local_root / "work" / "remote-job-status.json"
        if last_status.is_file():
            previous_job = json.loads(last_status.read_text(encoding="utf-8"))
            expected_returns = {
                20: local_root / "translation_output" / f"Muhtemel Ask {episode}.Bolum_TR_TEXT_CORRECTED.zip",
                21: local_root / "translation_output" / f"Muhtemel Ask {episode}.Bolum_ID_TRANSLATED.zip",
                WAIT_MP4_SAMPLE: local_root / "review" / "mp4-sample-approval.json",
            }
            code = previous_job.get("exit_code") if previous_job.get("status") == "EXITED" else None
            budget_path = local_root / "work" / "controller_budget.json"
            if code in expected_returns and not expected_returns[code].is_file() and budget_path.is_file():
                saved = json.loads(budget_path.read_text(encoding="utf-8"))
                if saved.get("sha256") != digest(saved.get("data")):
                    raise RunPodControllerError("episode budget checkpoint integrity mismatch")
                wait = saved["data"].get("wait")
                if wait and wait.get("reason") == code:
                    for key in ("evidence", "shutdown"):
                        if sha256_file(safe_relative(local_root, wait[key + "_path"])) != wait[key + "_sha256"]:
                            raise RunPodControllerError("verified wait evidence changed")
                    print(f"[WAIT] local return required: {expected_returns[code]}; no GPU acquired")
                    return code
        released_path = local_root / "work" / "gpu-released-for-delivery.json"
        if released_path.is_file():
            released = json.loads(released_path.read_text(encoding="utf-8"))
            if (released.get("status") != "ABSENT" or released.get("delivery_sha256") !=
                    sha256_file(local_root / "final" / "burned_mp4_delivery.json")):
                raise RunPodControllerError("transfer-only shutdown/delivery binding changed")
            # A transfer-only retry must never reacquire paid compute.
            return publish_local_delivery(local_root, episode, values["MAS_DRIVE_STRICT_REMOTE"])
        try:
            source_url = _prepare_official_source(
                local_root, episode, source_url, values.get("MAS_YTDLP_COOKIES")
            )
        except Exception:
            # Source preflight cannot spend compute; retain that distinction without resetting a run.
            latest = local_root / "logs" / "LATEST"
            if latest.is_file() and not (local_root / "work" / "controller_budget.json").exists():
                log_name = latest.read_text(encoding="utf-8").strip()
                if Path(log_name).name == log_name:
                    with (local_root / "logs" / log_name).open(encoding="utf-8") as handle:
                        first_line = handle.readline()
                    path = local_root / "work" / "controller-preflight-only.json"
                    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"data": {}}
                    if "sha256" in previous and previous["sha256"] != digest(previous["data"]):
                        raise RunPodControllerError("preflight-only evidence changed")
                    previous["data"][log_name] = digest(first_line)
                    atomic_json(path, {"data": previous["data"], "sha256": digest(previous["data"])})
            raise
        lease_reference = local_root / "work" / "controller-lease.json"
        resume_lease = False
        audit = local_root / "work" / "capacity" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}")
        if lease_reference.is_file():
            saved_reference = json.loads(lease_reference.read_text(encoding="utf-8"))
            previous = saved_reference["data"]
            if saved_reference.get("sha256") != digest(previous):
                raise RunPodControllerError("controller lease reference checksum mismatch")
            previous_audit = safe_relative(local_root, previous["audit"])
            state_path = previous_audit / "capacity-state.json"
            if state_path.is_file():
                saved_state = json.loads(state_path.read_text(encoding="utf-8"))
                if saved_state.get("sha256") != digest(saved_state.get("data")):
                    raise RunPodControllerError("capacity ownership journal changed")
                if saved_state["data"].get("status") not in {"RELEASED", "NO_CAPACITY"}:
                    if previous.get("commit") != commit or previous.get("source_url") != source_url:
                        raise RunPodControllerError("active job requires its original commit and source; no second job started")
                    audit, resume_lease = previous_audit, True
        budget_error = None
        try:
            episode_budget = _episode_budget(local_root, episode)
        except BudgetExceeded as exc:
            if not resume_lease:
                raise
            budget_error = exc
            episode_budget = None
        quote_path = os.getenv("MAS_RUNPOD_STORAGE_QUOTE")
        if not quote_path:
            raise RunPodControllerError("MAS_RUNPOD_STORAGE_QUOTE must name a fresh verified storage-price JSON before compute")
        storage_quote = load_storage_quote(Path(quote_path))
        provider = CapacityProvider(values["RUNPOD_POD_ID"], values["RUNPOD_API_KEY"])
        protected = provider.get_pod(values["RUNPOD_POD_ID"], 15)
        volume = provider.get_volume("xgogcmey5o", 15)
        if type(volume.get("size")) is not int or volume["size"] <= 0:
            raise RunPodControllerError("provider did not report a valid network volume allocation")
        public_key = Path(values["MAS_RUNPOD_SSH_KEY"] + ".pub").read_text(encoding="utf-8").strip()
        payload = {"cloudType": "SECURE", "gpuCount": 1, "volumeInGb": 0,
                   "containerDiskInGb": max(30, int(protected.get("containerDiskInGb") or 30)),
                   "imageName": protected.get("imageName"), "dockerArgs": "",
                   "ports": "22/tcp", "volumeMountPath": "/workspace",
                   "env": [{"key": "PUBLIC_KEY", "value": public_key}],
                   "supportPublicIp": True, "startSsh": True}
        plan = CapacityPlan(episode=episode,
                            maximum_rate_usd_per_hour=float(os.getenv("MAS_RUNPOD_MAX_COST_PER_HR", "0.75")),
                            total_seconds=int(episode_budget.check()) if episode_budget else 14400,
                            storage_quote=storage_quote)

        def ready(pod, remaining):
            remaining = min(90, remaining)
            readiness = time.monotonic() + remaining
            candidate = RunPodClient(pod["id"], values["RUNPOD_API_KEY"], attempts=1)
            try:
                current = candidate.wait(lambda value: value.get("desiredStatus") == "RUNNING"
                                         and _ssh_endpoint(value), "startup", timeout=remaining)
                host, port = _ssh_endpoint(current)
                left = readiness - time.monotonic()
                if left <= 0:
                    raise CapacityReadinessError("readiness deadline expired before SSH")
                _wait_for_ssh(Path(values["MAS_RUNPOD_SSH_KEY"]), host, port, timeout=left)
            except RunPodControllerError as exc:
                message = str(exc).lower()
                if any(word in message for word in ("authentication", "permission denied", "publickey", "http 401", "http 403")):
                    raise
                if any(word in message for word in ("timed out", "deadline", "connection refused", "unreachable")):
                    raise CapacityReadinessError("bounded Pod/SSH readiness failed") from exc
                raise
            pod.update(current)

        if not resume_lease:
            reference = {"commit": commit, "source_url": source_url, "audit": audit.relative_to(local_root).as_posix()}
            atomic_json(lease_reference, {"data": reference, "sha256": digest(reference)})
        published_from_scratch = False
        with CapacityLease(provider, payload, audit, plan,
                           ready=None if budget_error else ready, resume=resume_lease) as lease:
            if budget_error:
                raise budget_error
            runtime_values = dict(values, RUNPOD_POD_ID=lease.pod["id"],
                                  MAS_NETWORK_VOLUME_QUOTA_BYTES=str(volume["size"] * 1_000_000_000),
                                  MAS_EPISODE=str(episode))
            result = _run_remote_session(episode, source_url, values=runtime_values,
                                         commit=commit, rclone_config=rclone_config,
                                         budget=_PaidBudget(episode_budget, lease), pod=lease.pod,
                                         resume_only=resume_lease)
            delivery_path = local_root / "final" / "burned_mp4_delivery.json"
            if result == READY_FOR_DELIVERY and delivery_path.is_file():
                transferred = json.loads(delivery_path.read_text(encoding="utf-8"))
                if transferred["outputs"]["mp4"].get("storage_path"):
                    # The operator requires Drive verification before deleting ephemeral MP4 storage.
                    publish_local_delivery(local_root, episode, values["MAS_DRIVE_STRICT_REMOTE"],
                                           total_timeout=min(3600, lease.remaining_work_seconds()))
                    published_from_scratch = True
        if result == READY_FOR_DELIVERY:
            validate_delivery(local_root, episode)
            atomic_json(local_root / "work" / "gpu-released-for-delivery.json",
                        {"pod_id": lease.pod["id"], "status": "ABSENT", "capacity_audit": str(audit),
                         "delivery_sha256": sha256_file(local_root / "final" / "burned_mp4_delivery.json")})
            return 0 if published_from_scratch else publish_local_delivery(
                local_root, episode, values["MAS_DRIVE_STRICT_REMOTE"])
        if result in (20, 21, WAIT_MP4_SAMPLE):
            evidence = (local_root / "work" / "sample-export.json" if result == WAIT_MP4_SAMPLE else
                        local_root / "translation_input" / (f"Muhtemel Ask {episode}.Bolum_" +
                            ("TR_CORRECTION_PACK.zip" if result == 20 else "ID_TRANSLATION_PACK.zip")))
            _pause_episode_budget(local_root, episode, result, evidence, audit / "capacity-shutdown.json")
        return result


def _download_record(record, local_root, remote_root, scp, host, budget, *, checkpoint=False):
    relative = record["relative_path"]
    destination = safe_relative(local_root, relative)
    expected = record.get("sha256")
    size = record.get("size_bytes", record.get("bytes"))
    if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or type(size) is not int or size < 0:
        raise RunPodControllerError("invalid transfer manifest signature")
    local_episode_file = safe_relative(local_root, relative)
    if checkpoint and record.get("immutable_source") and local_episode_file.is_file():
        if local_episode_file.is_symlink():
            raise RunPodControllerError("local immutable source path is unsafe")
        if local_episode_file.stat().st_size != size or sha256_file(local_episode_file) != expected:
            raise RunPodControllerError("local immutable source identity differs; preserve it")
        return local_episode_file
    if checkpoint:
        destination = local_root / "work" / "remote-checkpoints" / expected / Path(relative).name
    if destination.is_file():
        if destination.stat().st_size == size and sha256_file(destination) == expected:
            return destination
        raise RunPodControllerError("existing local evidence differs; preserve it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_path = record.get("snapshot_path") or record.get("storage_path") or f"{remote_root}/{relative}"
    match = re.search(r"/Muhtemel Ask ([1-9][0-9]*)\.Bolum$", remote_root)
    external = (f"/tmp/mas-ep{match[1]}-output/{Path(relative).name}" if match else None)
    if (not remote_path.startswith(remote_root + "/") and remote_path != external) or "/../" in remote_path:
        raise RunPodControllerError("remote transfer path escapes episode")
    partial = destination.with_name(destination.name + ".partial")
    transfer_options = {
        "idle_timeout": 60,
        "total_timeout": min(1800, budget.check()),
        "budget": budget,
    }
    if size >= 64 * 1024 * 1024:
        transfer_options.update(
            attempts=1,
            progress_probe=_local_growth_probe(partial),
        )
    _network_retry(
        scp + [f"root@{host}:{remote_path}", str(partial)],
        **transfer_options,
    )
    if partial.stat().st_size != size or sha256_file(partial) != expected:
        raise RunPodControllerError("downloaded checkpoint/artifact byte/SHA-256 mismatch")
    os.replace(partial, destination)
    return destination


def _monitor_remote_job(ssh, scp, host, episode, commit, local_root, source_url, runtime_seconds, budget,
                        *, resume_only=False):
    release = f"/workspace/ma-sub/releases/{commit}"
    remote_root = f"/workspace/ma-sub/EPISODES/Muhtemel Ask {episode}.Bolum"
    prefix = ("source /workspace/.mas-secrets/runtime.env; "
              f"export MAS_MAX_RUNTIME_SECONDS={runtime_seconds}; cd {release}; "
              "exec env PYTHONPATH=src /workspace/ma-sub/.venv/bin/python -m mas.remote_job ")
    resume_files = sorted((local_root / "translation_output").glob("*.zip"))
    resume_files += [local_root / "review" / "audio_review_overrides.json",
                     local_root / "review" / "speaker_evidence_v1.json",
                     local_root / "review" / "mp4-sample-approval.json"]
    inputs = {path.relative_to(local_root).as_posix(): sha256_file(path)
              for path in resume_files if path.is_file()}
    input_sha = digest({"source_url": source_url, "commit": commit, "files": inputs,
                        "encoder": os.getenv("MAS_MP4_ENCODER", "h264_nvenc"),
                        "encoder_options": os.getenv("MAS_MP4_ENCODER_OPTIONS"),
                        "target_gb": os.getenv("MAS_MP4_TARGET_GB", "3")})
    request_path = local_root / "work" / "remote-job-request.json"
    if resume_only:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if (request.get("sha256") != digest(request.get("data"))
                or request["data"].get("commit") != commit):
            raise RunPodControllerError("cannot resume an unbound remote job request")
        input_sha = request["data"]["input_sha256"]
    else:
        request = {"commit": commit, "episode": episode, "input_sha256": input_sha}
        atomic_json(request_path, {"data": request, "sha256": digest(request)})
    common = f" --root {release} --episode {episode} --commit {commit} --input-sha256 {input_sha}"
    start = prefix + "start" + common + " --recover-lost --source-url " + shlex.quote(source_url)
    if not resume_only:
        try:
            _network_retry(ssh + [start], capture=True, attempts=3, idle_timeout=30,
                           total_timeout=min(90, budget.check()), budget=budget)
        except RunPodControllerError:
            print("[RUNPOD] start response lost; checking existing job without restarting", flush=True)
    offset = 0
    downloaded = {}
    while True:
        output = _network_retry(ssh + [prefix + "status" + common], capture=True,
                                attempts=3, idle_timeout=30, total_timeout=60, budget=budget)
        status = json.loads(output)
        expected_identity = {"episode": episode, "commit": commit, "input_sha256": input_sha,
                             "token": hashlib.sha256(
                                 f"{episode}\n{commit}\n{input_sha}\n".encode()).hexdigest()}
        if status.get("identity") != expected_identity:
            raise RunPodControllerError("remote job status identity mismatch")
        atomic_json(local_root / "work" / "remote-job-status.json", status)
        print(f"[RUNPOD] remote job status={status.get('status', 'UNKNOWN')}", flush=True)
        log_output = _network_retry(ssh + [prefix + "logs" + common + f" --offset {offset} --max-bytes 65536"],
                                    capture=True, attempts=2, total_timeout=45, budget=budget)
        log = json.loads(log_output)
        text = log.get("text", "")
        if text:
            print(text, end="", flush=True)
        offset = log["next_offset"]
        checkpoint_output = _network_retry(ssh + [prefix + "checkpoints" + common], capture=True,
                                            attempts=2, total_timeout=120, budget=budget)
        checkpoints = json.loads(checkpoint_output)
        if checkpoints.get("identity") != expected_identity:
            raise RunPodControllerError("remote checkpoint identity mismatch")
        for record in checkpoints.get("files", []):
            record_key = (record["relative_path"], record["sha256"], record["size_bytes"])
            cached = downloaded.get(record_key)
            if cached and cached[0].is_file():
                current = cached[0].stat()
                if (current.st_size, current.st_mtime_ns) == cached[1]:
                    continue
            snapshot = _download_record(record, local_root, remote_root, scp, host, budget, checkpoint=True)
            current = snapshot.stat()
            downloaded[record_key] = (snapshot, (current.st_size, current.st_mtime_ns))
            if record["relative_path"] == "work/state.json":
                state_path = local_root / "work" / "state.json"
                if state_path.is_file():
                    retained = local_root / "work" / "remote-checkpoints" / sha256_file(state_path) / "state.json"
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    if not retained.exists():
                        shutil.copyfile(state_path, retained)
                from .engine.download import atomic_write_bytes
                atomic_write_bytes(state_path, snapshot.read_bytes())
        atomic_json(local_root / "work" / "remote-checkpoint-manifest.json", checkpoints)
        if status.get("status") in {"COMPLETED", "EXITED", "FAILED"} and type(status.get("exit_code")) is int:
            if not log.get("eof", True):
                continue
            return status["exit_code"]
        if status.get("status") not in {"RUNNING", "STARTING"}:
            raise RunPodControllerError("remote job is absent/stale; refusing automatic relaunch")
        time.sleep(min(15, budget.check()))


def _collect_remote_results(exit_code, episode, local_root, remote_root, ssh, scp, host, budget, temporary):
    name = f"Muhtemel Ask {episode}.Bolum"
    retrieval_deadline = time.monotonic() + 120
    if exit_code in (READY_FOR_DELIVERY, WAIT_MP4_SAMPLE):
        export_name = "delivery-export.json" if exit_code == READY_FOR_DELIVERY else "sample-export.json"
        export_path = temporary / export_name
        _network_retry(scp + [f"root@{host}:{remote_root}/work/{export_name}", str(export_path)],
                       budget=budget, total_timeout=60)
        manifest = json.loads(export_path.read_text(encoding="utf-8"))
        expected_mode = "strict" if exit_code == READY_FOR_DELIVERY else "review"
        if manifest.get("episode") != episode or manifest.get("mode") != expected_mode:
            raise RunPodControllerError("delivery export identity mismatch")
        for record in manifest["files"]:
            _download_record(record, local_root, remote_root, scp, host, budget)
        atomic_json(local_root / "work" / export_name, manifest)

    if exit_code not in (0, 20, 21, READY_FOR_DELIVERY, WAIT_MP4_SAMPLE):
        diagnostics = (
            (
                f"{remote_root}/translation_input/{name}_TR_CORRECTION_PACK.zip",
                local_root / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip",
            ),
            (
                f"{remote_root}/prepare/audio_review_v2.json",
                local_root / "prepare" / "audio_review_v2.json",
            ),
            (
                f"{remote_root}/prepare/audio_review_v2.recovery.json",
                local_root / "prepare" / "audio_review_v2.recovery.json",
            ),
            (
                f"{remote_root}/translation_output/{name}_TR_TEXT_CORRECTED.zip",
                local_root
                / "translation_output"
                / f"{name}_TR_TEXT_CORRECTED.zip",
            ),
            (
                f"{remote_root}/translation_output/{name}_TR_CORRECTED.zip",
                local_root
                / "translation_output"
                / f"{name}_TR_CORRECTED.zip",
            ),
        )
        for position, (remote_file, local_file) in enumerate(
            diagnostics, start=1
        ):
            temporary_file = temporary / f"diagnostic-{position}"
            try:
                _network_retry(
                    scp
                    + [
                        f"root@{host}:{remote_file}",
                        str(temporary_file),
                    ],
                    attempts=2,
                    deadline=retrieval_deadline,
                )
            except RunPodControllerError as exc:
                print(
                    f"[RUNPOD] diagnostic download warning: {exc}",
                    file=sys.stderr,
                )
                if time.monotonic() >= retrieval_deadline:
                    break
                continue
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if local_file.exists():
                local_file = local_root / "work" / "remote-checkpoints" / sha256_file(temporary_file) / local_file.name
                local_file.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_file, local_file)
            print(f"[RUNPOD] diagnostic downloaded {local_file}")

    handoff = None
    if exit_code == 20:
        handoff = f"{name}_TR_CORRECTION_PACK.zip"
    elif exit_code == 21:
        handoff = f"{name}_ID_TRANSLATION_PACK.zip"
    if handoff:
        local_pack = local_root / "translation_input" / handoff
        local_pack.parent.mkdir(parents=True, exist_ok=True)
        remote_pack = f"{remote_root}/translation_input/{handoff}"
        size, signature = _remote_file_signature(ssh, remote_pack, budget=budget)
        _download_record({"relative_path": f"translation_input/{handoff}",
                          "size_bytes": size, "sha256": signature},
                         local_root, remote_root, scp, host, budget)
        print(f"[HANDOFF] downloaded {local_pack}")
    if exit_code not in (0, 20, 21, READY_FOR_DELIVERY, WAIT_MP4_SAMPLE):
        raise RunPodControllerError(f"remote pipeline failed with exit code {exit_code}")


def _run_remote_session(episode, source_url, *, values, commit, rclone_config, budget, pod, resume_only=False):
    name = f"Muhtemel Ask {episode}.Bolum"
    local_root = episode_dir(episode)
    key = Path(values["MAS_RUNPOD_SSH_KEY"]).resolve()
    cookie = Path(values["MAS_YTDLP_COOKIES"]).resolve() if values.get("MAS_YTDLP_COOKIES") else None
    client = RunPodClient(values["RUNPOD_POD_ID"], values["RUNPOD_API_KEY"])
    client.budget = budget
    started_at = time.monotonic()
    endpoint = None
    try:
        host, port = _ssh_endpoint(pod)
        endpoint = (host, port)
        _wait_for_ssh(key, host, port, timeout=min(300, budget.check()))
        budget.check()
        print(f"[RUNPOD] ready after {time.monotonic() - started_at:.1f}s")

        if resume_only:
            ssh, scp = _ssh_args(key, host, port), _scp_args(key, host, port)
            exit_code = _monitor_remote_job(ssh, scp, host, episode, commit, local_root, source_url,
                                            int(budget.check()), budget, resume_only=True)
            with tempfile.TemporaryDirectory(prefix="ma-sub-resume-") as folder:
                _collect_remote_results(exit_code, episode, local_root,
                    f"/workspace/ma-sub/EPISODES/{name}", ssh, scp, host, budget, Path(folder))
            return exit_code

        with tempfile.TemporaryDirectory(prefix="ma-sub-runpod-") as temporary:
            temporary = Path(temporary)
            archive = temporary / "release.tar.gz"
            runtime_env = temporary / "runtime.env"
            subprocess.run(
                ["git", "archive", "--format=tar.gz", f"--output={archive}", commit],
                cwd=ROOT,
                timeout=min(60, budget.check()),
                check=True,
            )
            _write_runtime_env(runtime_env, values, commit)
            ssh = _ssh_args(key, host, port)
            scp = _scp_args(key, host, port)
            _network_retry(ssh + ["install -d -m 700 /workspace/.mas-secrets /workspace/.mas-upload"], budget=budget)
            _network_retry(scp + [str(archive), f"root@{host}:/workspace/.mas-upload/release.tar.gz"], budget=budget)
            _network_retry(scp + [str(runtime_env), f"root@{host}:/workspace/.mas-secrets/runtime.env"], budget=budget)
            if cookie is not None:
                _network_retry(
                    scp + [str(cookie), f"root@{host}:/workspace/.mas-secrets/youtube-cookies.txt"],
                    budget=budget,
                )
            _network_retry(scp + [str(rclone_config), f"root@{host}:/workspace/.mas-secrets/rclone.conf"], budget=budget)

            deploy_command = (
                "set -euo pipefail; "
                "chmod 600 /workspace/.mas-secrets/*; "
                "install -d /workspace/ma-sub/EPISODES /workspace/ma-sub/releases; "
                f"release=/workspace/ma-sub/releases/{commit}; "
                "install -d \"$release\"; "
                "tar -xzf /workspace/.mas-upload/release.tar.gz -C \"$release\"; "
                "test -e \"$release/EPISODES\" || ln -s /workspace/ma-sub/EPISODES \"$release/EPISODES\"; "
                "source /workspace/.mas-secrets/runtime.env; "
                "cd \"$release\"; ./runpod/bootstrap.sh"
            )
            _network_retry(
                ssh + [deploy_command],
                attempts=1,
                idle_timeout=600,
                total_timeout=min(3600, budget.check()),
                budget=budget,
            )

            remote_root = f"/workspace/ma-sub/EPISODES/{name}"
            for filename in ("source.url", "official-source.json"):
                _upload_episode_file_verified(local_root / "source" / filename,
                    f"{remote_root}/source/{filename}", ssh=ssh, scp=scp, host=host, budget=budget, immutable=True)
            source_receipts = _upload_verified_local_source(
                local_root,
                remote_root,
                source_url,
                ssh=ssh,
                scp=scp,
                host=host,
                temporary=temporary,
                budget=budget,
            )
            if source_receipts is not None:
                print(
                    "[RUNPOD] local source seed verified: "
                    f"video_bytes={source_receipts['video']['bytes']} "
                    f"video_sha256={source_receipts['video']['sha256']}"
                )
            for filename in (f"{name}_TR_TEXT_CORRECTED.zip", f"{name}_ID_TRANSLATED.zip"):
                local_return = local_root / "translation_output" / filename
                if local_return.is_file():
                    destination = f"{remote_root}/translation_output/{filename}"
                    receipt = _upload_episode_file_verified(
                        local_return, destination, ssh=ssh, scp=scp, host=host,
                        budget=budget,
                    )
                    print(f"[RUNPOD] return upload verified: {filename}; "
                          f"bytes={receipt['bytes']} sha256={receipt['sha256']}")

            override_receipt = _upload_audio_review_overrides(
                local_root,
                remote_root,
                ssh=ssh,
                scp=scp,
                host=host,
                budget=budget,
            )
            speaker_receipt = _upload_speaker_evidence(
                local_root,
                remote_root,
                ssh=ssh,
                scp=scp,
                host=host,
                budget=budget,
            )
            approval = local_root / "review" / "mp4-sample-approval.json"
            if approval.is_file():
                _upload_episode_file_verified(approval, f"{remote_root}/review/{approval.name}",
                                               ssh=ssh, scp=scp, host=host, budget=budget)
            if override_receipt is not None:
                print(
                    "[RUNPOD] audio review overrides upload verified: "
                    f"bytes={override_receipt['bytes']} "
                    f"sha256={override_receipt['sha256']}"
                )
            if speaker_receipt is not None:
                print(
                    "[RUNPOD] speaker evidence upload verified: "
                    f"bytes={speaker_receipt['bytes']} "
                    f"sha256={speaker_receipt['sha256']}"
                )

            arguments = [str(episode)]
            if source_url:
                arguments.extend(["--source-url", source_url])
            quoted_arguments = " ".join(shlex.quote(value) for value in arguments)
            runtime_seconds = int(min(
                float(os.getenv("MAS_MAX_RUNTIME_SECONDS", "14400")), budget.check()
            ))
            if runtime_seconds < 1:
                raise BudgetExceeded("less than one second remains in episode runtime budget")
            exit_code = _monitor_remote_job(ssh, scp, host, episode, commit, local_root,
                                            source_url, runtime_seconds, budget)
            _collect_remote_results(exit_code, episode, local_root, remote_root, ssh, scp, host, budget, temporary)
    finally:
        if endpoint:
            client.budget = None
            shutdown_started = time.monotonic()
            print(f"[RUNPOD] stopping pod {values['RUNPOD_POD_ID']}")
            if endpoint:
                try:
                    _network(
                        _ssh_args(key, *endpoint)
                        + ["rm -f -- /workspace/.mas-secrets/runtime.env /workspace/.mas-secrets/youtube-cookies.txt /workspace/.mas-secrets/rclone.conf"],
                        idle_timeout=30,
                        total_timeout=60,
                    )
                except RunPodControllerError as exc:
                    print(f"[RUNPOD] secret cleanup warning: {exc}", file=sys.stderr)
            # The capacity lease owns termination and external absence verification.
    print(f"[RUNPOD] total elapsed {time.monotonic() - started_at:.1f}s")
    return exit_code
