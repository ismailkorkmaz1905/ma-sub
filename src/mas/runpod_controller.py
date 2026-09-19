import hashlib
import hmac
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
from .remote import RemoteVerificationError, _run_watchdog, drive_preflight
from .reliability import BudgetExceeded, RunBudget, atomic_json, digest, file_digest as sha256_file
from .delivery import (READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, WAIT_MP4_SAMPLE, WAIT_PART_RETURN,
                       ALIGNMENT_RECOVERY_COMPLETE, READY_FOR_PARTIAL_ENCODE, NEXT_PART,
                       SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED,
                       publish_local_delivery, safe_relative, validate_delivery)
from .source_discovery import discover_episode_metadata
from .runpod_capacity import (DEFAULT_GPU_TYPE_IDS, CapacityLease, CapacityPlan,
                             CapacityProvider, CapacityReadinessError, load_storage_quote)


class RunPodControllerError(RuntimeError):
    pass


class RunPodCleanupRequired(RunPodControllerError):
    pass


@contextmanager
def _cleanup_on_controller_failure(local_root, episode):
    try:
        yield
    except BaseException as original:
        try:
            reference_path = Path(local_root) / "work/controller-lease.json"
            if reference_path.exists():
                from .partial_delivery import _read_bound
                reference = _read_bound(reference_path)
                audit = safe_relative(local_root, reference["audit"])
                if not audit.is_relative_to((Path(local_root) / "work/capacity").resolve()):
                    raise RunPodControllerError("cleanup audit escaped capacity scope")
                state_path = audit / "capacity-state.json"
                state = _read_bound(state_path)
                if state.get("episode") != episode:
                    raise RunPodControllerError("cleanup lease belongs to another episode")
                if state.get("status") not in {"RELEASED", "NO_CAPACITY"}:
                    key = os.getenv("RUNPOD_API_KEY", "")
                    if not key or "\r" in key or "\n" in key:
                        raise RunPodControllerError("RUNPOD_API_KEY unavailable for external cleanup")
                    provider = CapacityProvider("781ct55zv4gkle", key)
                    CapacityLease.cleanup_existing(provider, audit, episode)
                    print("[RUNPOD] failure recovery verified owned Pod absence", flush=True)
        except Exception as cleanup:
            raise RunPodCleanupRequired(
                "EXTERNAL CLEANUP REQUIRED: owned Pod absence is unverified "
                f"({type(cleanup).__name__}); inspect the retained ownership journal before retry"
            ) from original
        raise


def _derive_raw_asr_auth_key(api_key):
    if not isinstance(api_key, str) or not api_key or "\r" in api_key or "\n" in api_key:
        raise RunPodControllerError("RUNPOD_API_KEY is missing or invalid")
    return hmac.new(
        api_key.encode("utf-8"),
        b"ma-sub/raw-asr-auth-key/v1",
        hashlib.sha256,
    ).hexdigest()


def _runtime_policy():
    return json.loads((Path(__file__).resolve().parents[2] / "config/runtime_policy.json").read_text(encoding="utf-8"))


def _configured_runtime_image():
    image = os.getenv("MAS_RUNPOD_IMAGE", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", image):
        raise RunPodControllerError("MAS_RUNPOD_IMAGE must identify a qualified image by repository@sha256 digest")
    return image


def _configured_registry_auth_id():
    value = os.getenv("MAS_RUNPOD_REGISTRY_AUTH_ID", "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise RunPodControllerError("MAS_RUNPOD_REGISTRY_AUTH_ID is invalid")
    return value


def _local_encoder_preflight(local_root, episode, budget):
    _production_priority()
    execution = os.getenv("MAS_DELIVERY_EXECUTION_PLAN", "local-qsv-v1")
    if execution == "remote-nvenc-v1":
        return
    if execution != "local-qsv-v1":
        raise RunPodControllerError("unsupported MAS_DELIVERY_EXECUTION_PLAN")
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RunPodControllerError(f"local QSV delivery requires {executable} before compute")
    from .engine.burned_mp4 import _qsv_hardware
    hardware = _qsv_hardware(timeout_seconds=min(55, budget.check()))
    budget.check()
    body = {"format": "mas-controller-local-encoder-preflight-1", "episode": episode,
            "execution": execution, "hardware": hardware,
            "scope": "SYNTHETIC_ENCODER_ONLY", "perceptual_acceptance": "NOT_ASSERTED"}
    atomic_json(Path(local_root) / "work/controller-local-encoder-preflight.json",
                {"data": body, "sha256": digest(body)})


def _episode_budget(local_root, episode, *, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        extension_limit = int(os.getenv("MAS_EPISODE_MAX_OPERATOR_EXTENSIONS", "3"))
    except ValueError as exc:
        raise RunPodControllerError("MAS_EPISODE_MAX_OPERATOR_EXTENSIONS must be an integer") from exc
    if not 3 <= extension_limit <= 12:
        raise RunPodControllerError("MAS_EPISODE_MAX_OPERATOR_EXTENSIONS must be between 3 and 12")
    budget_path = Path(local_root) / "work" / "controller_budget.json"
    excluded_wait = 0.0
    prior_limit = None
    operator_extensions = []
    started_at = now
    if budget_path.exists():
        saved = json.loads(budget_path.read_text(encoding="utf-8"))
        body = saved.get("data")
        if not isinstance(body, dict) or saved.get("sha256") != digest(body) or body.get("episode") != episode:
            raise RunPodControllerError("episode budget checkpoint integrity mismatch")
        started_at = datetime.fromisoformat(body["started_at"])
        prior_limit = body.get('limit_seconds')
        if (type(prior_limit) not in (int, float) or not math.isfinite(prior_limit)
                or not 0 < prior_limit <= 21600):
            raise RunPodControllerError('invalid original episode budget limit')
        excluded_wait = body.get("excluded_wait_seconds", 0.0)
        if not isinstance(excluded_wait, (int, float)) or not math.isfinite(excluded_wait) or excluded_wait < 0:
            raise RunPodControllerError("invalid excluded wait duration")
        operator_extensions = body.get("operator_extensions", [])
        if not isinstance(operator_extensions, list) or len(operator_extensions) > extension_limit:
            raise RunPodControllerError("invalid operator budget extension ledger")
        for extension in operator_extensions:
            if (not isinstance(extension, dict)
                    or set(extension) != {"authorized_at", "previous_started_at", "reason"}
                    or not isinstance(extension.get("reason"), str)
                    or not extension["reason"].strip()
                    or len(extension["reason"]) > 500):
                raise RunPodControllerError("invalid operator budget extension ledger")
            authorized_at = datetime.fromisoformat(extension["authorized_at"])
            previous_started_at = datetime.fromisoformat(extension["previous_started_at"])
            if (authorized_at.tzinfo is None or previous_started_at.tzinfo is None
                    or previous_started_at > authorized_at):
                raise RunPodControllerError("invalid operator budget extension timestamps")
        if operator_extensions and body["started_at"] != operator_extensions[-1]["authorized_at"]:
            raise RunPodControllerError("operator budget extension identity mismatch")
        wait = body.get("wait")
        if wait:
            for key in ("evidence", "shutdown"):
                if sha256_file(safe_relative(local_root, wait[key + "_path"])) != wait[key + "_sha256"]:
                    raise RunPodControllerError("verified wait evidence changed")
            paused = datetime.fromisoformat(wait["paused_at"])
            elapsed = (now - paused).total_seconds()
            if elapsed < 0:
                raise RunPodControllerError("wait clock moved backwards")
    else:
        configured_start = os.getenv("MAS_RUN_STARTED_AT")
        if configured_start:
            try:
                started_at = datetime.fromisoformat(configured_start)
            except ValueError as exc:
                raise RunPodControllerError("MAS_RUN_STARTED_AT is invalid") from exc
            if started_at.tzinfo is None or started_at > now:
                raise RunPodControllerError("MAS_RUN_STARTED_AT must be a past timezone-aware timestamp")
    if started_at.tzinfo is None:
        raise RunPodControllerError("episode budget timestamp must include timezone")
    try:
        policy = _runtime_policy()
        limit = float(os.getenv("MAS_EPISODE_BUDGET_SECONDS", str(policy["max_wall_seconds"])))
        if not math.isfinite(limit) or not 0 < limit <= policy["max_wall_seconds"]:
            raise ValueError("episode limit exceeds the hard wall budget")
        if prior_limit is not None:
            limit = min(limit, prior_limit)
        budget = RunBudget(started_at.isoformat(), limit_seconds=limit)
    except ValueError as exc:
        raise RunPodControllerError("MAS_EPISODE_BUDGET_SECONDS must be positive and at most 21600 seconds") from exc
    extension_approved = os.getenv("MAS_EPISODE_BUDGET_EXTENSION_APPROVED") == "1"
    early_extension = os.getenv("MAS_EPISODE_BUDGET_EXTENSION_EARLY") == "1"
    if extension_approved and (budget.remaining(now) <= 0 or early_extension):
        reason = os.getenv("MAS_EPISODE_BUDGET_EXTENSION_REASON", "").strip()
        if len(operator_extensions) >= extension_limit:
            label = "three" if extension_limit == 3 else str(extension_limit)
            raise RunPodControllerError(f"the {label} operator budget extensions are already consumed")
        if not reason or len(reason) > 500:
            raise RunPodControllerError(
                "MAS_EPISODE_BUDGET_EXTENSION_REASON must record the operator approval"
            )
        operator_extensions = [*operator_extensions, {
            "authorized_at": now.isoformat(),
            "previous_started_at": started_at.isoformat(),
            "reason": reason,
        }]
        started_at = now
        budget = RunBudget(started_at.isoformat(), limit_seconds=limit)
    body = {"episode": episode, "started_at": started_at.isoformat(),
            "limit_seconds": budget.limit_seconds, "excluded_wait_seconds": excluded_wait,
            "clock_policy": "all-wall-time-v2", "target_seconds": policy["target_wall_seconds"],
            "operator_extensions": operator_extensions}
    atomic_json(budget_path, {"data": body, "sha256": digest(body)})
    budget.check(now)
    return budget


def _pause_episode_budget(local_root, episode, reason, evidence_path, shutdown_path):
    if reason not in (20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED, WAIT_MP4_SAMPLE, WAIT_PART_RETURN):
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
    pack = Path(local_root) / "handoff" / f"{name}_TR_CORRECTION_PACK.zip"
    returned = Path(local_root) / "handoff" / f"{name}_TR_TEXT_CORRECTED.zip"
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


def _validate_local_semantic_return(local_root, name):
    pack = Path(local_root) / "handoff" / f"{name}_SEMANTIC_ALIGNMENT_PACK.zip"
    returned = Path(local_root) / "handoff" / f"{name}_SEMANTIC_ALIGNMENT_RETURN.zip"
    if not returned.is_file():
        return
    if not pack.is_file():
        raise RunPodControllerError(f"local semantic alignment pack is missing: {pack}")
    try:
        from .engine.semantic_alignment_handoff import validate_semantic_alignment_return
        validate_semantic_alignment_return(pack, returned)
    except Exception as exc:
        raise RunPodControllerError(
            "local semantic alignment return does not match the current pack"
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
    reuse_verified=False,
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
    if immutable or reuse_verified:
        quoted = shlex.quote(remote_path)
        existence = _network_retry(ssh + [f"if test -L {quoted}; then echo UNSAFE; "
                                   f"elif test -f {quoted}; then echo PRESENT; "
                                   f"elif test -e {quoted}; then echo UNSAFE; else echo MISSING; fi"],
                                   capture=True, total_timeout=30, budget=budget).strip()
        if existence == b"PRESENT":
            observed = _remote_file_signature(ssh, remote_path, budget=budget)
            if observed == (expected_size, expected_sha256):
                return {"bytes": expected_size, "sha256": expected_sha256}
            if immutable:
                raise RunPodControllerError("immutable remote source identity differs; preserve it")
        elif existence != b"MISSING":
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
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RunPodControllerError("local source checkpoint is invalid; refusing remote seed") from None
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) - {"video", "metadata", "captions"}:
        raise RunPodControllerError("local source checkpoint has unexpected outputs")

    local_source_root = source_dir.resolve()
    local_outputs = {}
    names = set()
    portable_incomplete = False
    for key, record in outputs.items():
        if not isinstance(record, dict):
            raise RunPodControllerError("local source checkpoint output is invalid")
        recorded_path = str(record.get("path", ""))
        name = Path(recorded_path).name
        if not name or name in names:
            raise RunPodControllerError("local source checkpoint output name is invalid")
        names.add(name)
        recorded = Path(recorded_path)
        portable_remote = recorded_path.replace("\\", "/").rsplit("/", 1)[0] == remote_root + "/source"
        if recorded.resolve().parent == local_source_root:
            source = recorded
        elif portable_remote:
            source = source_dir / name
            if not source.is_file():
                portable_incomplete = True
        else:
            raise RunPodControllerError("local source checkpoint path is outside the source directory")
        local_record = dict(record)
        local_record["path"] = str(source.resolve())
        local_outputs[key] = local_record
    if portable_incomplete:
        return None
    local_marker = dict(marker)
    local_marker["outputs"] = local_outputs
    seed_marker = Path(temporary) / "download.done.json"
    seed_marker.write_text(json.dumps(local_marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not validate_download(seed_marker, url=source_url, allowed_root=source_dir):
        raise RunPodControllerError("local source checkpoint is invalid; refusing remote seed")

    remote_outputs = {}
    receipts = {}
    for key, record in local_outputs.items():
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
    source = Path(local_root) / "work" / "audio_review_overrides.json"
    if not source.is_file():
        return None
    return _upload_episode_file_verified(
        source,
        f"{remote_root}/work/audio_review_overrides.json",
        ssh=ssh,
        scp=scp,
        host=host,
        budget=budget,
    )


def _upload_audio_review_reset(local_root, remote_root, *, ssh, scp, host, budget=None):
    source = Path(local_root) / "work" / "audio_review_reset.json"
    if not source.is_file():
        return None
    return _upload_episode_file_verified(
        source,
        f"{remote_root}/work/audio_review_reset.json",
        ssh=ssh,
        scp=scp,
        host=host,
        budget=budget,
    )


def _upload_speaker_evidence(local_root, remote_root, *, ssh, scp, host, budget=None):
    source = Path(local_root) / "work" / "speaker_evidence_v1.json"
    if not source.is_file():
        return None
    return _upload_episode_file_verified(
        source,
        f"{remote_root}/work/speaker_evidence_v1.json",
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
        "MAS_RAW_ASR_AUTH_KEY": _derive_raw_asr_auth_key(values["RUNPOD_API_KEY"]),
        "MAS_GIT_COMMIT": commit,
        "MAS_VENV_DIR": "/opt/venv",
        "MAS_RUNTIME_MODE": "immutable",
        "UV_CACHE_DIR": "/workspace/.cache/uv",
        "UV_HTTP_TIMEOUT": "120",
        "UV_HTTP_RETRIES": "3",
        "UV_CONCURRENT_DOWNLOADS": "4",
        "HF_HOME": "/workspace/.cache/huggingface",
        "TORCH_HOME": "/workspace/.cache/torch",
        "MAS_EXTERNAL_RUNPOD_CONTROLLER": "1",
        "MAS_DELIVERY_EXECUTION_PLAN": os.getenv("MAS_DELIVERY_EXECUTION_PLAN", "local-qsv-v1"),
        "MAS_PRODUCTION_PRIORITY": _production_priority(),
    }
    for name in ("MAS_NETWORK_VOLUME_QUOTA_BYTES", "MAS_EPISODE", "MAS_CODE_FIX_RESUME"):
        if name in values:
            remote_values[name] = values[name]
    if values.get("MAS_YTDLP_COOKIES"):
        remote_values["MAS_YTDLP_COOKIES"] = "/workspace/.mas-secrets/youtube-cookies.txt"
    for name in ("MAS_MAX_RUNTIME_SECONDS", "MAS_IDLE_TIMEOUT_SECONDS", "MAS_MP4_TARGET_GB",
                 "MAS_MP4_ENCODER", "MAS_MP4_ENCODER_OPTIONS"):
        if os.getenv(name):
            remote_values[name] = os.environ[name]
    if os.getenv("MAS_ALLOW_C0E3_VENV_ADOPTION") == "1":
        remote_values["MAS_ALLOW_C0E3_VENV_ADOPTION"] = "1"
    lines = [f"export {name}={shlex.quote(value)}" for name, value in remote_values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.chmod(path, 0o600)


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


def _capacity_release_evidence(local_root, episode, pod_id, state_record, shutdown_record):
    if (not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id)
            or not isinstance(state_record, dict) or not isinstance(shutdown_record, dict)
            or not isinstance(state_record.get("relative_path"), str)
            or not isinstance(shutdown_record.get("relative_path"), str)
            or not isinstance(state_record.get("sha256"), str)
            or not isinstance(shutdown_record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", state_record["sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", shutdown_record["sha256"])):
        raise RunPodControllerError("delivery release capacity evidence is invalid")
    state_path = safe_relative(local_root, state_record["relative_path"])
    shutdown_path = safe_relative(local_root, shutdown_record["relative_path"])
    capacity_root = (Path(local_root) / "work" / "capacity").resolve()
    if (state_path.name != "capacity-state.json" or shutdown_path.name != "capacity-shutdown.json"
            or state_path.parent != shutdown_path.parent
            or not state_path.parent.resolve().is_relative_to(capacity_root)):
        raise RunPodControllerError("delivery release capacity evidence path is invalid")
    if (sha256_file(state_path) != state_record.get("sha256")
            or sha256_file(shutdown_path) != shutdown_record.get("sha256")):
        raise RunPodControllerError("delivery release capacity evidence changed")
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    state = saved_state.get("data")
    saved_shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
    shutdown = saved_shutdown.get("data")
    if (not isinstance(state, dict) or saved_state.get("sha256") != digest(state)
            or state.get("format") != "mas-capacity-lease-state-1"
            or state.get("episode") != episode or state.get("status") != "RELEASED"):
        raise RunPodControllerError("delivery release capacity state is invalid")
    owned_ids = state.get("owned_pod_ids")
    results = state.get("shutdown")
    if (not isinstance(shutdown, dict) or saved_shutdown.get("sha256") != digest(shutdown)
            or shutdown.get("state_sha256") != digest(state)
            or shutdown.get("owned_pods") != results
            or not isinstance(owned_ids, list) or pod_id not in owned_ids
            or not isinstance(results, list) or not results
            or any(not isinstance(item, dict) for item in results)
            or any(not isinstance(item, str)
                   or not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in owned_ids)
            or any(not isinstance(item.get("pod_id"), str)
                   or not re.fullmatch(r"[A-Za-z0-9_-]+", item["pod_id"])
                   for item in results)
            or len(owned_ids) != len(set(owned_ids))
            or len(results) != len({item.get("pod_id") for item in results})
            or {item.get("pod_id") for item in results} != set(owned_ids)
            or any(item.get("status") != "ABSENT" for item in results)):
        raise RunPodControllerError("delivery release shutdown evidence is invalid")


def _write_delivery_release(local_root, episode, pod_id, audit):
    state_path = Path(audit) / "capacity-state.json"
    shutdown_path = Path(audit) / "capacity-shutdown.json"
    state_record = {
        "relative_path": state_path.relative_to(local_root).as_posix(),
        "sha256": sha256_file(state_path),
    }
    shutdown_record = {
        "relative_path": shutdown_path.relative_to(local_root).as_posix(),
        "sha256": sha256_file(shutdown_path),
    }
    _capacity_release_evidence(local_root, episode, pod_id, state_record, shutdown_record)
    release = {
        "format": "mas-gpu-released-for-delivery-1",
        "episode": episode,
        "pod_id": pod_id,
        "status": "ABSENT",
        "delivery_sha256": sha256_file(local_root / "output" / "burned_mp4_delivery.json"),
        "capacity_state": state_record,
        "capacity_shutdown": shutdown_record,
    }
    atomic_json(local_root / "work" / "gpu-released-for-delivery.json",
                {"data": release, "sha256": digest(release)})


def _validate_delivery_release(local_root, episode, released_path):
    saved = json.loads(Path(released_path).read_text(encoding="utf-8"))
    release = saved.get("data")
    if (not isinstance(release, dict) or saved.get("sha256") != digest(release)
            or release.get("format") != "mas-gpu-released-for-delivery-1"
            or release.get("episode") != episode or release.get("status") != "ABSENT"
            or release.get("delivery_sha256") !=
                sha256_file(local_root / "output" / "burned_mp4_delivery.json")):
        raise RunPodControllerError("transfer-only shutdown/delivery binding changed")
    _capacity_release_evidence(local_root, episode, release.get("pod_id"),
                               release.get("capacity_state"), release.get("capacity_shutdown"))


def _write_encode_release(local_root, episode, pod_id, audit):
    from .local_encode import validate_subtitle_export
    validate_subtitle_export(local_root, episode)
    records = {name: {"relative_path": (Path(audit) / filename).relative_to(local_root).as_posix(),
                      "sha256": sha256_file(Path(audit) / filename)} for name, filename in
               (("capacity_state", "capacity-state.json"), ("capacity_shutdown", "capacity-shutdown.json"))}
    _capacity_release_evidence(local_root, episode, pod_id, records["capacity_state"], records["capacity_shutdown"])
    release = {"format": "mas-gpu-released-for-encode-1", "episode": episode, "pod_id": pod_id,
               "status": "ABSENT", "plan_sha256": sha256_file(local_root / "work/local-encode-plan.json"), **records}
    atomic_json(local_root / "work/gpu-released-for-encode.json", {"data": release, "sha256": digest(release)})


def _finish_local_encode(local_root, episode, budget):
    from .local_encode import complete_local_encode
    remote = os.getenv("MAS_DRIVE_STRICT_REMOTE", "")
    if not remote or "\r" in remote or "\n" in remote:
        raise RunPodControllerError("valid MAS_DRIVE_STRICT_REMOTE is required for local delivery")
    complete_local_encode(local_root, episode, total_timeout=budget.check())
    release = json.loads((local_root / "work/gpu-released-for-encode.json").read_text(encoding="utf-8"))["data"]
    audit = safe_relative(local_root, release["capacity_state"]["relative_path"]).parent
    validate_delivery(local_root, episode)
    _write_delivery_release(local_root, episode, release["pod_id"], audit)
    return publish_local_delivery(local_root, episode, remote, total_timeout=budget.check())


@contextmanager
def _record_release_after_capacity(local_root, episode, audit):
    try:
        yield
    finally:
        try:
            state_path = Path(audit) / "capacity-state.json"
            shutdown_path = Path(audit) / "capacity-shutdown.json"
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            owned = saved["data"].get("owned_pod_ids", [])
            if owned:
                _capacity_release_evidence(local_root, episode, owned[-1],
                    {"relative_path": state_path.relative_to(local_root).as_posix(), "sha256": sha256_file(state_path)},
                    {"relative_path": shutdown_path.relative_to(local_root).as_posix(), "sha256": sha256_file(shutdown_path)})
        except Exception as exc:
            print(f"[CAPACITY] release evidence deferred: {type(exc).__name__}", file=sys.stderr)


def _production_priority():
    execution = os.getenv('MAS_DELIVERY_EXECUTION_PLAN', 'local-qsv-v1')
    priority = os.getenv('MAS_PRODUCTION_PRIORITY',
                         'first-hour-v1' if execution == 'local-qsv-v1' else 'whole-episode-v1')
    if (priority not in {'first-hour-v1', 'whole-episode-v1'}
            or (priority == 'first-hour-v1' and execution != 'local-qsv-v1')):
        raise RunPodControllerError('unsupported production priority and execution combination')
    return priority


def _resume_partial_delivery(local_root, episode):
    if _production_priority() != 'first-hour-v1':
        return None
    from .partial_delivery import (part_directory, validate_published_part, validate_local_tail,
                                  complete_local_part, complete_parts, _read_bound)
    plan_path = local_root / 'work/part-plan.json'
    if not plan_path.is_file():
        return None
    from .engine.part_audio import load_part_plan
    plan = load_part_plan(local_root, episode, verify_files=False)
    lease_path = local_root / 'work/controller-lease.json'
    if lease_path.is_file():
        lease = _read_bound(lease_path)
        state_path = safe_relative(local_root, lease['audit']) / 'capacity-state.json'
        if state_path.is_file() and _read_bound(state_path).get('status') not in {'RELEASED', 'NO_CAPACITY'}:
            return None  # The existing lease cleanup owns an active or expired paid worker.
    saved_budget = _read_bound(local_root / 'work/controller_budget.json')
    if (saved_budget.get('episode') != episode or type(saved_budget.get('limit_seconds')) not in (int, float)
            or not 0 < saved_budget['limit_seconds'] <= 21600):
        raise RunPodControllerError('partial episode budget identity changed')
    budget = RunBudget(saved_budget['started_at'], limit_seconds=saved_budget['limit_seconds'])
    delivery_late = False
    try:
        budget.check()
    except BudgetExceeded:
        from .delivery_first import EXPORT_MODE, _clock
        first = local_root / 'parts' / plan['parts'][0]['part_id'] / 'work/partial-export.json'
        if not first.is_file() or json.loads(first.read_text(encoding='utf-8')).get('mode') != EXPORT_MODE:
            raise
        # This allowance is only used in this local function. It never authorizes a Pod.
        remaining = _clock(3600)
        class LocalDeliveryBudget:
            def check(self):
                seconds = remaining()
                if seconds <= 0:
                    raise BudgetExceeded('Local delivery retry allowance expired; artifacts retained')
                return min(3600, seconds)
        budget = LocalDeliveryBudget()
        delivery_late = True
        atomic_json(local_root / 'work/delivery-deadline-missed.json', {
            'episode': episode, 'original_started_at': saved_budget['started_at'],
            'original_limit_seconds': saved_budget['limit_seconds'],
            'status': 'SLO_MISSED', 'local_retry_seconds': 3600, 'gpu_authorized': False})
    for part in plan['parts']:
        part_id = part['part_id']
        folder = part_directory(local_root, part_id)
        if validate_published_part(local_root, episode, part_id, total_timeout=budget.check()) is not None:
            continue
        if validate_local_tail(local_root, episode, part_id, total_timeout=budget.check()) is not None:
            continue
        if (folder / 'work/gpu-released-for-encode.json').is_file():
            return complete_local_part(local_root, episode, part_id, os.getenv('MAS_DRIVE_STRICT_REMOTE', ''),
                                       total_timeout=budget.check())
        handoff_path = local_root / 'work/partial-handoff.json'
        if not delivery_late and handoff_path.is_file():
            from .progressive import validate_partial_handoff
            handoff = validate_partial_handoff(local_root, episode)
            if handoff['part_id'] == part_id and not safe_relative(local_root, handoff['expected_return']).is_file():
                saved = saved_budget
                wait = saved.get('wait', {})
                if wait.get('reason') == WAIT_PART_RETURN:
                    for key in ('evidence', 'shutdown'):
                        if sha256_file(safe_relative(local_root, wait[key + '_path'])) != wait[key + '_sha256']:
                            raise RunPodControllerError('partial handoff release evidence changed')
                    if wait['evidence_sha256'] != sha256_file(handoff_path):
                        raise RunPodControllerError('partial handoff wait belongs to a different part')
                    RunBudget(saved['started_at'], limit_seconds=saved['limit_seconds']).check()
                    return WAIT_PART_RETURN
                released = _released_partial_audit(local_root, episode, WAIT_PART_RETURN)
                if released is not None:
                    _, audit = released
                    _pause_episode_budget(local_root, episode, WAIT_PART_RETURN, handoff_path,
                                          audit / 'capacity-shutdown.json')
                    return WAIT_PART_RETURN
        export_path = folder / 'work/partial-export.json'
        if export_path.is_file():
            released = _released_partial_audit(local_root, episode, READY_FOR_PARTIAL_ENCODE)
            if released is not None:
                from .partial_delivery import write_part_release
                pod_id, audit = released
                write_part_release(local_root, episode, part_id, pod_id, audit, total_timeout=budget.check())
                return complete_local_part(local_root, episode, part_id, os.getenv('MAS_DRIVE_STRICT_REMOTE', ''),
                                           total_timeout=budget.check())
        return None
    if complete_parts(local_root, episode, total_timeout=budget.check()) is None:
        raise RunPodControllerError('complete-parts inventory changed')
    print('[DELIVERY] planned outputs have verified Drive byte/SHA readback')
    return 0


def _released_partial_audit(local_root, episode, exit_code):
    from .partial_delivery import _read_bound
    reference_path = local_root / 'work/controller-lease.json'
    status_path = local_root / 'work/remote-job-status.json'
    if not reference_path.is_file() or not status_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('status') != 'EXITED' or status.get('exit_code') != exit_code:
        return None
    reference = _read_bound(reference_path)
    request = _read_bound(local_root / 'work/remote-job-request.json')
    from .remote_job import _identity
    if (request.get('episode') != episode or request.get('commit') != reference.get('commit')
            or status.get('identity') != _identity(episode, request['commit'], request['input_sha256'])):
        raise RunPodControllerError('partial release job identity changed')
    audit = safe_relative(local_root, reference['audit'])
    state_path, shutdown_path = audit / 'capacity-state.json', audit / 'capacity-shutdown.json'
    if not state_path.is_file() or not shutdown_path.is_file():
        return None
    state = _read_bound(state_path)
    if state.get('status') != 'RELEASED':
        return None
    owned = state.get('owned_pod_ids', [])
    if not owned:
        raise RunPodControllerError('partial release has no owned Pod identity')
    _capacity_release_evidence(local_root, episode, owned[-1],
        {'relative_path': state_path.relative_to(local_root).as_posix(), 'sha256': sha256_file(state_path)},
        {'relative_path': shutdown_path.relative_to(local_root).as_posix(), 'sha256': sha256_file(shutdown_path)})
    return owned[-1], audit


def _preflight_part_return(local_root, episode):
    if _production_priority() != 'first-hour-v1':
        return
    if not (local_root / 'work/partial-handoff.json').is_file():
        return
    from .progressive import validate_partial_handoff
    handoff = validate_partial_handoff(local_root, episode)
    output = safe_relative(local_root, handoff['expected_return'])
    if not output.is_file():
        return
    pack = safe_relative(local_root, handoff['pack']['relative_path'])
    if handoff['kind'] == 'tr':
        from .engine.tr_correction import validate_tr_correction_output
        validate_tr_correction_output(pack, output)
    else:
        from .engine.translation_workspace import validate_id_workspace_output
        validate_id_workspace_output(pack, output)


def _part_resume_files(local_root, episode):
    if _production_priority() != 'first-hour-v1':
        return []
    if not (local_root / 'work/part-plan.json').is_file():
        return []
    from .engine.part_audio import load_part_plan
    from .partial_delivery import part_directory, write_worker_delivery_ack, write_worker_tail_ack
    plan = load_part_plan(local_root, episode, verify_files=False)
    paths = []
    for part in plan['parts']:
        folder = part_directory(local_root, part['part_id'])
        for suffix in ('TR_TEXT_CORRECTED.zip', 'ID_TRANSLATED.zip', 'ID_TRANSLATED.zip.workspace.json'):
            path = folder / 'handoff' / f'Muhtemel Ask {episode}.Bolum_{suffix}'
            if path.is_file():
                paths.append(path)
        for filename in ('audio_review_overrides.json', 'speaker_evidence_v1.json'):
            path = folder / 'work' / filename
            if path.is_file():
                paths.append(path)
        if (folder / 'output/drive_readback_receipt.json').is_file():
            paths.append(write_worker_delivery_ack(local_root, episode, part['part_id']))
        elif (folder / 'output/local-tail-ready.json').is_file():
            paths.append(write_worker_tail_ack(local_root, episode, part['part_id']))
    return paths


def run_remote_episode(episode, source_url=None, *, alignment_recovery=False):
    if alignment_recovery:
        return _run_remote_episode_once(episode, source_url, alignment_recovery=True)
    for _ in range(65):
        result = _run_remote_episode_once(episode, source_url)
        if result != NEXT_PART:
            return result
    raise RunPodControllerError('progressive episode continuation count exceeded')


def _run_remote_episode_once(episode, source_url=None, *, alignment_recovery=False):
    if type(episode) is not int or episode <= 0:
        raise RunPodControllerError("episode must be a positive integer")
    if alignment_recovery and _production_priority() != "whole-episode-v1":
        raise RunPodControllerError("alignment recovery requires whole-episode-v1 to match its retained scope")
    with _controller_lock(ROOT / "var" / "production-controller.lock"), \
            _cleanup_on_controller_failure(episode_dir(episode), episode):
        local_root = episode_dir(episode)
        if not alignment_recovery:
            partial_result = _resume_partial_delivery(local_root, episode)
            if partial_result is not None:
                return partial_result
        released_path = local_root / "work" / "gpu-released-for-delivery.json"
        if released_path.is_file() and not alignment_recovery:
            _validate_delivery_release(local_root, episode, released_path)
            remote = os.getenv("MAS_DRIVE_STRICT_REMOTE")
            if not remote:
                raise RunPodControllerError(
                    "missing environment: MAS_DRIVE_STRICT_REMOTE"
                )
            if "\r" in remote or "\n" in remote:
                raise RunPodControllerError(
                    "MAS_DRIVE_STRICT_REMOTE must not contain line breaks"
                )
            budget = _episode_budget(local_root, episode)
            return publish_local_delivery(local_root, episode, remote, total_timeout=budget.check())
        if (local_root / "work/gpu-released-for-encode.json").is_file() and not alignment_recovery:
            return _finish_local_encode(local_root, episode, _episode_budget(local_root, episode))
        values = _required_environment()
        commit, rclone_config = _local_preflight(values)
        _validate_local_tr_return(local_root, f"Muhtemel Ask {episode}.Bolum")
        _validate_local_semantic_return(local_root, f"Muhtemel Ask {episode}.Bolum")
        if not alignment_recovery:
            preflight_local_id_return(local_root, episode, ROOT / "config" / "production")
            _preflight_part_return(local_root, episode)
        last_status = local_root / "work" / "remote-job-status.json"
        if last_status.is_file() and not alignment_recovery:
            previous_job = json.loads(last_status.read_text(encoding="utf-8"))
            expected_returns = {
                20: local_root / "handoff" / f"Muhtemel Ask {episode}.Bolum_TR_TEXT_CORRECTED.zip",
                21: local_root / "handoff" / f"Muhtemel Ask {episode}.Bolum_ID_TRANSLATED.zip",
                SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED:
                    local_root / "handoff" / f"Muhtemel Ask {episode}.Bolum_SEMANTIC_ALIGNMENT_RETURN.zip",
                WAIT_MP4_SAMPLE: local_root / "work" / "mp4-sample-approval.json",
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
            state_path = local_root / "work" / "state.json"
            semantic_local_resume = False
            if state_path.is_file():
                local_state = json.loads(state_path.read_text(encoding="utf-8"))
                semantic_local_resume = (
                    local_state.get("run_contract", {}).get("alignment_policy")
                    == "semantic-block-v1"
                )
            if (
                code in (SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED, 21)
                and semantic_local_resume
                and expected_returns[code].is_file()
                and budget_path.is_file()
            ):
                saved = json.loads(budget_path.read_text(encoding="utf-8"))
                if saved.get("sha256") != digest(saved.get("data")):
                    raise RunPodControllerError("episode budget checkpoint integrity mismatch")
                wait = saved["data"].get("wait")
                if not isinstance(wait, dict) or wait.get("reason") != code:
                    raise RunPodControllerError("semantic local resume lacks verified GPU release evidence")
                for key in ("evidence", "shutdown"):
                    if sha256_file(safe_relative(local_root, wait[key + "_path"])) != wait[key + "_sha256"]:
                        raise RunPodControllerError("verified semantic wait evidence changed")
                previous_external = os.environ.get("MAS_EXTERNAL_RUNPOD_CONTROLLER")
                os.environ["MAS_EXTERNAL_RUNPOD_CONTROLLER"] = "1"
                try:
                    from .pipeline import run as run_pipeline
                    local_result = run_pipeline(episode, source_url)
                finally:
                    if previous_external is None:
                        os.environ.pop("MAS_EXTERNAL_RUNPOD_CONTROLLER", None)
                    else:
                        os.environ["MAS_EXTERNAL_RUNPOD_CONTROLLER"] = previous_external
                if local_result == READY_FOR_LOCAL_ENCODE:
                    shutdown_path = safe_relative(local_root, wait["shutdown_path"])
                    shutdown = json.loads(shutdown_path.read_text(encoding="utf-8"))
                    shutdown_data = shutdown.get("data")
                    pods = shutdown_data.get("owned_pods") if isinstance(shutdown_data, dict) else None
                    if not isinstance(pods, list) or len(pods) != 1:
                        raise RunPodControllerError("semantic GPU release Pod identity is invalid")
                    pod_id = pods[0].get("pod_id")
                    _write_encode_release(local_root, episode, pod_id, shutdown_path.parent)
                    return _finish_local_encode(local_root, episode, _episode_budget(local_root, episode))
                if local_result == READY_FOR_DELIVERY:
                    raise RunPodControllerError(
                        "semantic local resume requires local-qsv-v1; remote encoding needs an explicit encoder job"
                    )
                return local_result
        if alignment_recovery:
            retained_source = local_root / "source" / "source.url"
            if not retained_source.is_file():
                raise RunPodControllerError("alignment recovery requires the retained source URL")
            retained_url = retained_source.read_text(encoding="utf-8").strip()
            if not retained_url or (source_url and source_url != retained_url):
                raise RunPodControllerError("alignment recovery source identity mismatch")
            source_url = retained_url
        else:
            source_url = _prepare_official_source(
                local_root, episode, source_url, values.get("MAS_YTDLP_COOKIES")
            )
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
        if not resume_lease:
            _guard_failed_remote_job(local_root, episode, source_url, commit)
        quote_path = os.getenv("MAS_RUNPOD_STORAGE_QUOTE")
        if not quote_path:
            raise RunPodControllerError("MAS_RUNPOD_STORAGE_QUOTE must name a fresh verified storage-price JSON before compute")
        storage_quote = load_storage_quote(Path(quote_path))
        runtime_error = None
        try:
            runtime_image = _configured_runtime_image()
            registry_auth_id = _configured_registry_auth_id()
        except RunPodControllerError as exc:
            if not resume_lease:
                raise
            runtime_image, registry_auth_id, runtime_error = None, None, exc
        if not resume_lease and not alignment_recovery:
            _local_encoder_preflight(local_root, episode, episode_budget)
            drive_readiness = drive_preflight(values["MAS_DRIVE_STRICT_REMOTE"], required_bytes=0,
                                              config_path=rclone_config,
                                              total_timeout=min(60, episode_budget.check()))
            atomic_json(local_root / "work" / "controller-drive-preflight.json", drive_readiness)
        provider = CapacityProvider(values["RUNPOD_POD_ID"], values["RUNPOD_API_KEY"])
        protected = provider.get_pod(values["RUNPOD_POD_ID"], 15)
        volume = provider.get_volume("xgogcmey5o", 15)
        if type(volume.get("size")) is not int or volume["size"] <= 0:
            raise RunPodControllerError("provider did not report a valid network volume allocation")
        public_key = Path(values["MAS_RUNPOD_SSH_KEY"] + ".pub").read_text(encoding="utf-8").strip()
        payload = {"cloudType": "SECURE", "gpuCount": 1, "volumeInGb": 0,
                   "containerDiskInGb": max(30, int(protected.get("containerDiskInGb") or 30)),
                   "imageName": runtime_image or protected.get("imageName"), "dockerArgs": "",
                   "ports": "22/tcp", "volumeMountPath": "/workspace",
                   "env": [{"key": "PUBLIC_KEY", "value": public_key}],
                   "supportPublicIp": True, "startSsh": True}
        if registry_auth_id:
            payload["containerRegistryAuthId"] = registry_auth_id
        plan = CapacityPlan(episode=episode,
                            maximum_rate_usd_per_hour=float(os.getenv("MAS_RUNPOD_MAX_COST_PER_HR", "0.75")),
                            gpu_type_ids=(
                                _configured_gpu_type_ids()
                                if os.getenv("MAS_RUNPOD_GPU_TYPE_IDS") or os.getenv("MAS_RUNPOD_GPU_TYPE_ID")
                                else DEFAULT_GPU_TYPE_IDS
                            ),
                            total_seconds=int(episode_budget.check()) if episode_budget else 14400,
                            startup_seconds=900 if not alignment_recovery else 600,
                            storage_quote=storage_quote, resume=resume_lease)

        def ready(pod, remaining):
            remaining = min(900, remaining)
            readiness = time.monotonic() + remaining
            candidate = RunPodClient(pod["id"], values["RUNPOD_API_KEY"], attempts=1)
            try:
                current = candidate.wait(lambda value: value.get("desiredStatus") == "RUNNING"
                                         and _ssh_endpoint(value), "startup", timeout=remaining)
                if current.get("imageName") != runtime_image:
                    raise RunPodControllerError("provider imageName does not match the configured immutable image")
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
        with _record_release_after_capacity(local_root, episode, audit), CapacityLease(provider, payload, audit, plan,
                           ready=None if budget_error or runtime_error else ready, resume=resume_lease) as lease:
            if budget_error:
                raise budget_error
            if runtime_error:
                raise runtime_error
            runtime_values = dict(values, RUNPOD_POD_ID=lease.pod["id"],
                                  MAS_NETWORK_VOLUME_QUOTA_BYTES=str(volume["size"] * 1_000_000_000),
                                  MAS_EPISODE=str(episode))
            result = _run_remote_session(episode, source_url, values=runtime_values,
                                         commit=commit,
                                         budget=_PaidBudget(episode_budget, lease), pod=lease.pod,
                                         resume_only=resume_lease,
                                         alignment_recovery=alignment_recovery)
        if result == READY_FOR_PARTIAL_ENCODE:
            from .partial_delivery import _read_bound, write_part_release, complete_local_part
            part_id = json.loads((local_root / 'work/partial-export.json').read_text(encoding='utf-8'))['part_id']
            write_part_release(local_root, episode, part_id, lease.pod['id'], audit,
                               total_timeout=episode_budget.check())
            return complete_local_part(local_root, episode, part_id, values['MAS_DRIVE_STRICT_REMOTE'],
                                       total_timeout=episode_budget.check())
        if result == WAIT_PART_RETURN:
            _pause_episode_budget(local_root, episode, result, local_root / 'work/partial-handoff.json',
                                  audit / 'capacity-shutdown.json')
        if result == READY_FOR_LOCAL_ENCODE:
            _write_encode_release(local_root, episode, lease.pod["id"], audit)
            return _finish_local_encode(local_root, episode, episode_budget)
        if result == READY_FOR_DELIVERY:
            validate_delivery(local_root, episode)
            _write_delivery_release(local_root, episode, lease.pod["id"], audit)
            return publish_local_delivery(local_root, episode, values["MAS_DRIVE_STRICT_REMOTE"],
                                          total_timeout=episode_budget.check())
        if result == ALIGNMENT_RECOVERY_COMPLETE and alignment_recovery:
            return result
        if result in (20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED, WAIT_MP4_SAMPLE):
            evidence = (local_root / "work" / "sample-export.json" if result == WAIT_MP4_SAMPLE else
                        local_root / "handoff" / (f"Muhtemel Ask {episode}.Bolum_" +
                            ("TR_CORRECTION_PACK.zip" if result == 20 else
                             "SEMANTIC_ALIGNMENT_PACK.zip" if result == SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED
                             else "ID_TRANSLATION_PACK.zip")))
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
        destination = safe_relative(local_root, f"work/remote-checkpoints/{expected}/{Path(relative).name}")
    remote_path = record.get("snapshot_path") or record.get("storage_path") or f"{remote_root}/{relative}"
    match = re.search(r"/Muhtemel Ask ([1-9][0-9]*)\.Bolum$", remote_root)
    external = (f"/tmp/mas-ep{match[1]}-output/{Path(relative).name}" if match else None)
    if (not remote_path.startswith(remote_root + "/") and remote_path != external) or "/../" in remote_path:
        raise RunPodControllerError("remote transfer path escapes episode")
    if checkpoint and local_episode_file.is_file():
        if local_episode_file.stat().st_size == size and sha256_file(local_episode_file) == expected:
            return local_episode_file
    if destination.is_file():
        if destination.stat().st_size == size and sha256_file(destination) == expected:
            return destination
        if remote_path != external:
            raise RunPodControllerError("existing local evidence differs; preserve it")
        retained_sha = sha256_file(destination)
        retained = safe_relative(
            local_root,
            f"work/remote-checkpoints/{retained_sha}/{destination.name}",
        )
        retained.parent.mkdir(parents=True, exist_ok=True)
        if retained.exists():
            if (retained.is_symlink() or not retained.is_file()
                    or retained.stat().st_size != destination.stat().st_size
                    or sha256_file(retained) != retained_sha):
                raise RunPodControllerError("retained local evidence differs; preserve it")
        else:
            retained_partial = safe_relative(
                local_root,
                f"work/remote-checkpoints/{retained_sha}/{destination.name}.partial",
            )
            shutil.copyfile(destination, retained_partial)
            if (retained_partial.stat().st_size != destination.stat().st_size
                    or sha256_file(retained_partial) != retained_sha):
                raise RunPodControllerError("retained local evidence byte/SHA-256 mismatch")
            os.replace(retained_partial, retained)
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def _record_result_collection_failure(local_root, exit_code, record=None):
    work = Path(local_root) / "work"
    request_path = work / "remote-job-request.json"
    status_path = work / "remote-job-status.json"
    if not request_path.is_file() or not status_path.is_file():
        raise RunPodControllerError("remote collection failure evidence is incomplete")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    data = request.get("data")
    if not isinstance(data, dict) or request.get("sha256") != digest(data):
        raise RunPodControllerError("remote job request integrity mismatch")
    if status.get("status") != "EXITED" or status.get("exit_code") != exit_code:
        raise RunPodControllerError("remote collection failure status mismatch")
    attempt = data.get("attempt", 0)
    if type(attempt) is not int or attempt < 0:
        raise RunPodControllerError("remote job attempt evidence is invalid")
    if (not isinstance(record, dict)
            or not isinstance(record.get("relative_path"), str)
            or "storage_path" not in record
            or (record["storage_path"] is not None
                and not isinstance(record["storage_path"], str))):
        raise RunPodControllerError("remote collection failure path evidence is incomplete")
    failure = {
        "format": "mas-remote-result-collection-failure-1",
        "episode": data.get("episode"),
        "base_input_sha256": data.get("base_input_sha256", data.get("input_sha256")),
        "input_sha256": data.get("input_sha256"),
        "attempt": attempt,
        "exit_code": exit_code,
        "request_sha256": request["sha256"],
        "relative_path": record["relative_path"],
        "storage_path": record["storage_path"],
    }
    atomic_json(work / "remote-result-collection-failure.json",
                {"data": failure, "sha256": digest(failure)})


def _collection_failure_paths_match(failure, data, status_path):
    relative = failure.get("relative_path")
    storage = failure.get("storage_path")
    if ("relative_path" not in failure or "storage_path" not in failure
            or not isinstance(relative, str) or not relative
            or (storage is not None and not isinstance(storage, str))):
        return False
    episode = data.get("episode")
    export_name = {READY_FOR_LOCAL_ENCODE: 'subtitle-export.json',
                   READY_FOR_PARTIAL_ENCODE: 'partial-export.json'}.get(failure.get('exit_code'), 'delivery-export.json')
    export_relative = f"work/{export_name}"
    export_storage = (
        f"/workspace/ma-sub/EPISODES/Muhtemel Ask {episode}.Bolum/"
        f"work/{export_name}"
    )
    if relative == export_relative:
        return storage == export_storage
    try:
        safe_relative(status_path.parent.parent, relative)
    except ValueError:
        return False
    external = f"/tmp/mas-ep{episode}-output/{Path(relative).name}"
    if storage is not None and storage != external:
        return False
    manifest_path = status_path.with_name(export_name)
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    expected_mode = {READY_FOR_LOCAL_ENCODE: 'strict-subtitles',
                     READY_FOR_PARTIAL_ENCODE: 'strict-partial-subtitles'}.get(failure.get('exit_code'), 'strict')
    from .delivery_first import EXPORT_MODE, _tag
    delivery_mode = failure.get('exit_code') == READY_FOR_PARTIAL_ENCODE and manifest.get('mode') == EXPORT_MODE
    if delivery_mode:
        body = dict(manifest)
        tag = body.pop('auth_tag', None)
        if (type(episode) is not int or episode < 1 or not isinstance(tag, str)
                or not hmac.compare_digest(tag, _tag(body, 'export'))):
            return False
    if (manifest.get("episode") != episode or (not delivery_mode and manifest.get("mode") != expected_mode)
            or not isinstance(files, list)):
        return False
    return any(
        isinstance(record, dict)
        and record.get("relative_path") == relative
        and record.get("storage_path") == storage
        for record in files
    )


def _remote_attempt_identity(base_input_sha, request_path, status_path):
    if not request_path.is_file() and not status_path.is_file():
        return base_input_sha, 0
    if not request_path.is_file() or not status_path.is_file():
        raise RunPodControllerError("remote job retry evidence is incomplete")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    data = request.get("data")
    if not isinstance(data, dict) or request.get("sha256") != digest(data):
        raise RunPodControllerError("remote job request integrity mismatch")
    exit_code = status.get("exit_code") if status.get("status") in {"EXITED", "FAILED"} else None
    failure_path = status_path.with_name("remote-result-collection-failure.json")
    if exit_code in (READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, READY_FOR_PARTIAL_ENCODE):
        prior_base = data.get("base_input_sha256", data.get("input_sha256"))
        if prior_base != base_input_sha:
            return base_input_sha, 0
        attempt = data.get("attempt", 0)
        if type(attempt) is not int or attempt < 0:
            raise RunPodControllerError("remote job attempt evidence is invalid")
        expected_status_identity = {
            "episode": data.get("episode"),
            "commit": data.get("commit"),
            "input_sha256": data.get("input_sha256"),
            "token": hashlib.sha256(
                (f"{data.get('episode')}\n{data.get('commit')}\n"
                 f"{data.get('input_sha256')}\n").encode()
            ).hexdigest(),
        }
        if status.get("identity") != expected_status_identity:
            raise RunPodControllerError("remote result collection status identity mismatch")
        if not failure_path.is_file():
            raise RunPodControllerError("remote result collection failure evidence is missing")
        saved = json.loads(failure_path.read_text(encoding="utf-8"))
        failure = saved.get("data")
        if not isinstance(failure, dict) or saved.get("sha256") != digest(failure):
            raise RunPodControllerError("remote result collection failure integrity mismatch")
        expected = {
            "format": "mas-remote-result-collection-failure-1",
            "episode": data.get("episode"),
            "base_input_sha256": prior_base,
            "input_sha256": data.get("input_sha256"),
            "attempt": attempt,
            "exit_code": exit_code,
            "request_sha256": request["sha256"],
        }
        if any(failure.get(key) != value for key, value in expected.items()):
            raise RunPodControllerError("remote result collection failure binding mismatch")
        if not _collection_failure_paths_match(failure, data, status_path):
            raise RunPodControllerError("remote result collection failure path mismatch")
        attempt += 1
        if attempt > _runtime_policy()["max_changed_evidence_retries"]:
            raise RunPodControllerError("BLOCKED: collection retry budget exhausted")
        return digest({"base_input_sha256": base_input_sha, "attempt": attempt}), attempt
    if exit_code in (None, 0, 20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED,
                     READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, WAIT_MP4_SAMPLE,
                     WAIT_PART_RETURN, READY_FOR_PARTIAL_ENCODE):
        return base_input_sha, 0
    ticket_path = status_path.with_name("remote-retry-authorization.json")
    if not ticket_path.is_file():
        raise RunPodControllerError("BLOCKED: unchanged failed job requires changed validated evidence before retry")
    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    authorization = ticket.get("data")
    if (not isinstance(authorization, dict) or ticket.get("sha256") != digest(authorization)
            or authorization.get("request_sha256") != request["sha256"]
            or authorization.get("base_input_sha256") != base_input_sha):
        raise RunPodControllerError("remote retry authorization integrity or binding mismatch")
    failure = json.loads(status_path.with_name("remote-failure-invariant.json").read_text(encoding="utf-8"))
    if (failure.get("sha256") != digest(failure.get("data"))
            or failure["sha256"] != authorization.get("failure_sha256")
            or failure["data"].get("request_sha256") != request["sha256"]
            or failure["data"].get("status_sha256") != digest(status)):
        raise RunPodControllerError("remote retry failure evidence binding mismatch")
    attempt = data.get("attempt", 0)
    if type(attempt) is not int or attempt < 0:
        raise RunPodControllerError("remote job attempt evidence is invalid")
    attempt += 1
    retry_limit = _changed_evidence_retry_limit(status_path.parent, data.get("episode"))
    if authorization.get("qualified_code_fix") is True:
        retry_limit = max(retry_limit, attempt)
    if attempt > retry_limit:
        raise RunPodControllerError("BLOCKED: changed-evidence retry budget exhausted")
    return digest({"base_input_sha256": base_input_sha, "attempt": attempt}), attempt


def _remote_input_binding(local_root, source_url, commit, episode):
    resume_files = [local_root / "handoff" / f"Muhtemel Ask {episode}.Bolum_{suffix}.zip"
                    for suffix in ("TR_TEXT_CORRECTED", "SEMANTIC_ALIGNMENT_RETURN", "ID_TRANSLATED")]
    resume_files += [Path(str(resume_files[-1]) + ".workspace.json")]
    resume_files += _part_resume_files(local_root, episode)
    unvalidated_review_files = {"audio_review_overrides.json", "speaker_evidence_v1.json"}
    validated_returns = {
        path.relative_to(local_root).as_posix(): sha256_file(path)
        for path in resume_files
        if path.is_file() and path.name not in unvalidated_review_files
    }
    reset_sha256 = _validated_audio_review_reset_hash(local_root, episode)
    if reset_sha256 is not None:
        validated_returns["work/audio_review_reset.json"] = reset_sha256
    resume_files += [local_root / "work" / "audio_review_overrides.json",
                    local_root / "work" / "audio_review_reset.json",
                    local_root / "work" / "speaker_evidence_v1.json",
                    local_root / "work" / "mp4-sample-approval.json",
                    local_root / "work" / "code-fix-resume.json"]
    inputs = {path.relative_to(local_root).as_posix(): sha256_file(path)
              for path in resume_files if path.is_file()}
    evidence = digest({"source_url": source_url, "files": validated_returns})
    base = digest({"source_url": source_url, "commit": commit, "files": inputs,
                   "production_priority": _production_priority(),
                   "execution_plan": os.getenv('MAS_DELIVERY_EXECUTION_PLAN', 'local-qsv-v1'),
                   "encoder": os.getenv("MAS_MP4_ENCODER", "h264_nvenc"),
                   "encoder_options": os.getenv("MAS_MP4_ENCODER_OPTIONS"),
                   "target_gb": os.getenv("MAS_MP4_TARGET_GB", "3")})
    return base, evidence


def _validated_audio_review_reset_hash(local_root, episode):
    local_root = Path(local_root)
    path = local_root / "work" / "audio_review_reset.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    body = saved.get("data")
    name = f"Muhtemel Ask {episode}.Bolum"
    expected_paths = {
        "work/audio_review_v2.json",
        "work/audio_review_v2.recovery.json",
        f"handoff/{name}_TR_CORRECTED.zip",
    }
    artifacts = body.get("artifacts") if isinstance(body, dict) else None
    if (
        not isinstance(body, dict)
        or saved.get("sha256") != digest(body)
        or body.get("format") != "mas-audio-review-reset-1"
        or body.get("episode") != episode
        or not isinstance(body.get("reason"), str)
        or not body["reason"].strip()
        or not isinstance(artifacts, list)
        or {item.get("relative_path") for item in artifacts if isinstance(item, dict)}
        != expected_paths
        or any(
            set(item) != {"relative_path", "size_bytes", "sha256"}
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"]))
            for item in artifacts
        )
    ):
        raise RunPodControllerError("audio-review reset evidence integrity mismatch")
    pack = local_root / "handoff" / f"{name}_TR_CORRECTION_PACK.zip"
    provisional = local_root / "handoff" / f"{name}_TR_TEXT_CORRECTED.zip"
    validated = validate_tr_correction_output(pack, provisional)
    if (
        body.get("correction_input_sha256") != validated.input_sha256
        or body.get("provisional_output_sha256") != validated.output_sha256
    ):
        raise RunPodControllerError("audio-review reset evidence input binding mismatch")
    return sha256_file(path)


def _changed_evidence_retry_limit(work, episode, *, now=None):
    policy_limit = _runtime_policy()["max_changed_evidence_retries"]
    path = Path(work) / "remote-retry-extension.json"
    extension_sha256 = None
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        body = saved.get("data")
        if (
            not isinstance(body, dict)
            or saved.get("sha256") != digest(body)
            or body.get("format") != "mas-operator-retry-extension-1"
            or body.get("episode") != episode
            or body.get("previous_limit") != policy_limit
            or body.get("extended_limit") != policy_limit + 3
            or not isinstance(body.get("reason"), str)
            or not body["reason"].strip()
            or len(body["reason"]) > 500
        ):
            raise RunPodControllerError("operator retry extension integrity mismatch")
        authorized_at = datetime.fromisoformat(body["authorized_at"])
        if authorized_at.tzinfo is None:
            raise RunPodControllerError("operator retry extension timestamp is invalid")
        extension_sha256 = saved["sha256"]
        extended_limit = body["extended_limit"]
    elif os.getenv("MAS_CHANGED_EVIDENCE_RETRY_EXTENSION_APPROVED") == "1":
        reason = os.getenv("MAS_CHANGED_EVIDENCE_RETRY_EXTENSION_REASON", "").strip()
        if not reason or len(reason) > 500:
            raise RunPodControllerError(
                "MAS_CHANGED_EVIDENCE_RETRY_EXTENSION_REASON must record the operator approval"
            )
        now = now or datetime.now(timezone.utc)
        body = {
            "format": "mas-operator-retry-extension-1",
            "episode": episode,
            "authorized_at": now.isoformat(),
            "previous_limit": policy_limit,
            "extended_limit": policy_limit + 3,
            "reason": reason,
        }
        extension_sha256 = digest(body)
        atomic_json(path, {"data": body, "sha256": extension_sha256})
        extended_limit = body["extended_limit"]
    else:
        return policy_limit

    continuation_path = Path(work) / "remote-retry-continuation.json"
    if continuation_path.is_file():
        saved = json.loads(continuation_path.read_text(encoding="utf-8"))
        body = saved.get("data")
        if (
            not isinstance(body, dict)
            or saved.get("sha256") != digest(body)
            or body.get("format") != "mas-operator-retry-continuation-1"
            or body.get("episode") != episode
            or body.get("previous_limit") != extended_limit
            or body.get("extended_limit") != extended_limit + 1
            or body.get("prior_extension_sha256") != extension_sha256
            or not isinstance(body.get("reason"), str)
            or not body["reason"].strip()
            or len(body["reason"]) > 500
        ):
            raise RunPodControllerError("operator retry continuation integrity mismatch")
        authorized_at = datetime.fromisoformat(body["authorized_at"])
        if authorized_at.tzinfo is None:
            raise RunPodControllerError("operator retry continuation timestamp is invalid")
        return body["extended_limit"]
    if os.getenv("MAS_CHANGED_EVIDENCE_RETRY_CONTINUATION_APPROVED") != "1":
        return extended_limit
    reason = os.getenv("MAS_CHANGED_EVIDENCE_RETRY_CONTINUATION_REASON", "").strip()
    if not reason or len(reason) > 500:
        raise RunPodControllerError(
            "MAS_CHANGED_EVIDENCE_RETRY_CONTINUATION_REASON must record the operator approval"
        )
    now = now or datetime.now(timezone.utc)
    body = {
        "format": "mas-operator-retry-continuation-1",
        "episode": episode,
        "authorized_at": now.isoformat(),
        "previous_limit": extended_limit,
        "extended_limit": extended_limit + 1,
        "prior_extension_sha256": extension_sha256,
        "reason": reason,
    }
    atomic_json(continuation_path, {"data": body, "sha256": digest(body)})
    return body["extended_limit"]


def _guard_failed_remote_job(local_root, episode, source_url, commit):
    work = local_root / "work"
    request_path, status_path = work / "remote-job-request.json", work / "remote-job-status.json"
    if not status_path.is_file():
        if request_path.is_file():
            raise RunPodControllerError("remote job retry evidence is incomplete")
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    code = status.get("exit_code") if status.get("status") in {"EXITED", "FAILED"} else None
    if code in (READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, READY_FOR_PARTIAL_ENCODE):
        base, _ = _remote_input_binding(local_root, source_url, commit, episode)
        _remote_attempt_identity(base, request_path, status_path)
        return
    if code in (None, 0, 20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED,
                WAIT_MP4_SAMPLE, WAIT_PART_RETURN):
        return
    if not request_path.is_file():
        raise RunPodControllerError("remote job retry evidence is incomplete")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    data = request.get("data")
    if not isinstance(data, dict) or request.get("sha256") != digest(data):
        raise RunPodControllerError("remote job request integrity mismatch")
    identity = {"episode": data.get("episode"), "commit": data.get("commit"),
                "input_sha256": data.get("input_sha256"),
                "token": hashlib.sha256(f"{data.get('episode')}\n{data.get('commit')}\n{data.get('input_sha256')}\n".encode()).hexdigest()}
    if data.get("episode") != episode or status.get("identity") != identity:
        raise RunPodControllerError("remote failure status identity mismatch")
    base, evidence = _remote_input_binding(local_root, source_url, commit, episode)
    diagnostic_paths = ("state.json", "remote-checkpoint-manifest.json")
    diagnostics = {name: sha256_file(work / name) for name in diagnostic_paths if (work / name).is_file()}
    failure = {"format": "mas-failed-job-invariant-1", "request_sha256": request["sha256"],
               "status_sha256": digest(status), "exit_code": code,
               "evidence_input_sha256": data.get("evidence_input_sha256"),
               "diagnostics": diagnostics, **_partial_failure_identity(local_root, episode)}
    failure_path = work / "remote-failure-invariant.json"
    if failure_path.is_file():
        saved_failure = json.loads(failure_path.read_text(encoding="utf-8"))
        prior_failure = saved_failure.get("data")
        if not isinstance(prior_failure, dict) or saved_failure.get("sha256") != digest(prior_failure):
            raise RunPodControllerError("remote failure invariant integrity mismatch")
        if prior_failure.get("request_sha256") == request["sha256"]:
            failure = prior_failure
    else:
        atomic_json(failure_path, {"data": failure, "sha256": digest(failure)})
    code_fix = local_root / "work/code-fix-resume.json"
    qualified_code_fix = False
    if code_fix.is_file():
        from .retry_authorization import validate_code_fix_resume
        validate_code_fix_resume(local_root, episode, commit, code_fix)
        if failure.get("evidence_input_sha256") != evidence:
            raise RunPodControllerError("code-fix retry cannot also change validated text inputs")
        qualified_code_fix = True
    pre_pipeline_code_fix = False
    checkpointed_raw_asr_code_fix = False
    checkpoint_path = work / "remote-checkpoint-manifest.json"
    checkpoint_sha = failure.get("diagnostics", {}).get("remote-checkpoint-manifest.json")
    if data.get("commit") != commit and checkpoint_sha and checkpoint_path.is_file() \
            and sha256_file(checkpoint_path) == checkpoint_sha:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        files = checkpoint.get("files")
        relative_paths = (
            {item.get("relative_path") for item in files}
            if isinstance(files, list) and all(isinstance(item, dict) for item in files)
            else set()
        )
        pre_pipeline_code_fix = (
            checkpoint.get("identity") == identity
            and isinstance(files, list)
            and all(isinstance(item, dict) for item in files)
            and relative_paths <= {"source/source.url"}
            and len(files) <= 1
            and checkpoint.get("unstable") == []
        )
        raw_asr_progress = status.get("progress", {}).get("stage")
        checkpointed_raw_asr_code_fix = (
            checkpoint.get("identity") == identity
            and isinstance(raw_asr_progress, str)
            and raw_asr_progress.startswith("raw_asr:")
            and "work/raw_asr_v2.recovery.json" in relative_paths
            and "work/raw_asr_v2.done.json" not in relative_paths
            and "work/audio.done.json" in relative_paths
            and "source/download.done.json" in relative_paths
            and "source/source.url" in relative_paths
            and any(
                isinstance(path, str) and path.startswith("work/primary_asr/")
                for path in relative_paths
            )
            and all(
                isinstance(path, str)
                and (
                    path in {
                        "work/audio.done.json",
                        "work/raw_asr_v2.recovery.json",
                        "source/download.done.json",
                        "source/source.url",
                        "work/state.json",
                    }
                    or path.startswith("work/primary_asr/")
                    or path.startswith("source/Muhtemel Ask ")
                )
                and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
                and type(item.get("size_bytes")) is int
                and item["size_bytes"] >= 0
                for item, path in (
                    (item, item.get("relative_path")) for item in files
                )
            )
            and checkpoint.get("unstable") == []
        )
    if not qualified_code_fix and not pre_pipeline_code_fix \
            and not checkpointed_raw_asr_code_fix \
            and (not failure.get("evidence_input_sha256") or failure["evidence_input_sha256"] == evidence):
        raise RunPodControllerError("BLOCKED: unchanged failed evidence; code or option changes alone do not authorize another GPU run")
    attempt = data.get("attempt", 0)
    retry_limit = _changed_evidence_retry_limit(work, episode)
    if qualified_code_fix:
        retry_limit = max(retry_limit, attempt + 1)
    if type(attempt) is not int or not 0 <= attempt < retry_limit:
        raise RunPodControllerError("BLOCKED: changed-evidence retry budget exhausted")
    authorization = {"request_sha256": request["sha256"], "base_input_sha256": base,
                     "evidence_input_sha256": evidence, "failure_sha256": digest(failure),
                     "qualified_code_fix": qualified_code_fix,
                     "pre_pipeline_code_fix": pre_pipeline_code_fix,
                     "checkpointed_raw_asr_code_fix": checkpointed_raw_asr_code_fix}
    atomic_json(work / "remote-retry-authorization.json", {"data": authorization, "sha256": digest(authorization)})


def _record_failed_remote_evidence(exit_code, local_root, source_url, commit):
    if exit_code in (0, 20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED,
                     READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, WAIT_MP4_SAMPLE,
                     WAIT_PART_RETURN, READY_FOR_PARTIAL_ENCODE, ALIGNMENT_RECOVERY_COMPLETE):
        return
    work = local_root / "work"
    request = json.loads((work / "remote-job-request.json").read_text(encoding="utf-8"))
    if request.get("sha256") != digest(request.get("data")):
        raise RunPodControllerError("remote job request integrity mismatch")
    status = json.loads((work / "remote-job-status.json").read_text(encoding="utf-8"))
    _, evidence = _remote_input_binding(local_root, source_url, commit, request["data"]["episode"])
    failure = {"format": "mas-failed-job-invariant-1", "request_sha256": request["sha256"],
               "status_sha256": digest(status), "exit_code": exit_code,
               "evidence_input_sha256": evidence,
               "diagnostics": {name: sha256_file(work / name) for name in
                               ("state.json", "remote-checkpoint-manifest.json") if (work / name).is_file()},
               **_partial_failure_identity(local_root, request['data']['episode'])}
    atomic_json(work / "remote-failure-invariant.json", {"data": failure, "sha256": digest(failure)})


def _partial_failure_identity(local_root, episode):
    if _production_priority() != 'first-hour-v1':
        return {}
    from .partial_delivery import _read_bound
    from .delivery import verified_record
    path = local_root / 'work/current-part.json'
    if not path.is_file():
        return {}
    current = _read_bound(path)
    status_path = local_root / 'work/remote-job-status.json'
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if current.get('remote_job_token') != status.get('identity', {}).get('token'):
            return {}  # A failure before child work must not adopt a previous job's alignment failure.
    part_id = current.get('part_id')
    if (current.get('format') != 'mas-current-part-1' or current.get('episode') != episode
            or not re.fullmatch(r'part-[0-9]{3}', str(part_id))):
        raise RunPodControllerError('failed partial stage identity changed')
    identity = {'part_id': part_id, 'part_stage': current.get('stage')}
    if identity['part_stage'] != 'forced_alignment':
        return identity
    state_record = current.get('state', {})
    if state_record.get('relative_path') != f'parts/{part_id}/work/state.json':
        raise RunPodControllerError('failed partial state escaped its part')
    state = json.loads(verified_record(local_root, state_record).read_text(encoding='utf-8'))
    from .engine.part_audio import load_part_plan
    plan = load_part_plan(local_root, episode, verify_files=False)
    if part_id not in {item['part_id'] for item in plan['parts']}:
        raise RunPodControllerError('failed part not present in original plan')
    hashes = {'part_plan_sha256': sha256_file(local_root / 'work/part-plan.json'),
              'part_audio_lineage_sha256': sha256_file(safe_relative(local_root,
                  f'parts/{part_id}/work/audio-part.done.json'))}
    if (state.get('episode') != episode or state.get('part_id') != part_id
            or any(current.get(key) != value or state.get(key) != value for key, value in hashes.items())):
        raise RunPodControllerError('failed part plan/audio evidence changed')
    return {**identity, **hashes}


def _monitor_remote_job(ssh, scp, host, episode, commit, local_root, source_url, runtime_seconds, budget,
                        *, resume_only=False, alignment_recovery=False):
    release = f"/workspace/ma-sub/releases/{commit}"
    remote_root = f"/workspace/ma-sub/EPISODES/Muhtemel Ask {episode}.Bolum"
    prefix = ("source /workspace/.mas-secrets/runtime.env; "
              f"export MAS_MAX_RUNTIME_SECONDS={runtime_seconds}; cd {release}; "
              'exec env PYTHONPATH=src "$MAS_VENV_DIR/bin/python" -m mas.remote_job ')
    base_input_sha, evidence_input_sha = _remote_input_binding(local_root, source_url, commit, episode)
    input_sha = base_input_sha
    request_path = local_root / "work" / "remote-job-request.json"
    if resume_only:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if (request.get("sha256") != digest(request.get("data"))
                or request["data"].get("commit") != commit):
            raise RunPodControllerError("cannot resume an unbound remote job request")
        input_sha = request["data"]["input_sha256"]
    else:
        input_sha, attempt = _remote_attempt_identity(
            base_input_sha,
            request_path,
            local_root / "work" / "remote-job-status.json",
        )
        request = {"commit": commit, "episode": episode, "input_sha256": input_sha,
                   "base_input_sha256": base_input_sha, "attempt": attempt,
                   "evidence_input_sha256": evidence_input_sha}
        atomic_json(request_path, {"data": request, "sha256": digest(request)})
    common = f" --root {release} --episode {episode} --commit {commit} --input-sha256 {input_sha}"
    start = prefix + "start" + common + " --recover-lost --source-url " + shlex.quote(source_url)
    if alignment_recovery:
        start += " --alignment-recovery"
    if not resume_only:
        try:
            _network_retry(ssh + [start], capture=True, attempts=3, idle_timeout=30,
                           total_timeout=min(90, budget.check()), budget=budget)
        except RunPodControllerError:
            print("[RUNPOD] start response lost; checking existing job without restarting", flush=True)
    offset = 0
    downloaded = {}
    last_status = None
    while True:
        poll_idle_timeout = 180 if alignment_recovery else 60
        poll_total_timeout = 240 if alignment_recovery else 120
        output = _network_retry(
            ssh + [prefix + "poll" + common + f" --offset {offset} --max-bytes 65536 --deadline-seconds 60"],
            capture=True, attempts=3, idle_timeout=poll_idle_timeout,
            total_timeout=poll_total_timeout, budget=budget)
        poll = json.loads(output)
        status = poll["status"]
        expected_identity = {"episode": episode, "commit": commit, "input_sha256": input_sha,
                             "token": hashlib.sha256(
                                 f"{episode}\n{commit}\n{input_sha}\n".encode()).hexdigest()}
        if status.get("identity") != expected_identity:
            raise RunPodControllerError("remote job status identity mismatch")
        atomic_json(local_root / "work" / "remote-job-status.json", status)
        status_name = status.get("status", "UNKNOWN")
        if status_name != last_status:
            print(f"[RUNPOD] remote job status={status_name}", flush=True)
            last_status = status_name
        log = poll["log"]
        text = log.get("text", "")
        if text:
            print(text, end="", flush=True)
        offset = log["next_offset"]
        checkpoints = poll["checkpoints"]
        if checkpoints.get("identity") != expected_identity:
            raise RunPodControllerError("remote checkpoint identity mismatch")
        if not alignment_recovery:
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
                if (record["relative_path"] in {'work/state.json', 'work/current-part.json'} or re.fullmatch(
                        r'parts/part-[0-9]{3}/work/state\.json', record['relative_path'])):
                    state_path = safe_relative(local_root, record["relative_path"])
                    if state_path.is_file():
                        retained = local_root / "work" / "remote-checkpoints" / sha256_file(state_path) / state_path.name
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


def _collect_diagnostics(episode, local_root, remote_root, ssh, scp, host, retrieval_deadline):
    request = json.loads((local_root / "work/remote-job-request.json").read_text(encoding="utf-8"))
    data = request.get("data")
    if not isinstance(data, dict) or request.get("sha256") != digest(data):
        raise RunPodControllerError("diagnostic request integrity mismatch")
    from .remote_job import _identity
    expected_identity = _identity(episode, data["commit"], data["input_sha256"])
    release = f"/workspace/ma-sub/releases/{data['commit']}"
    command = ("source /workspace/.mas-secrets/runtime.env; "
               f"cd {release}; exec env PYTHONPATH=src \"$MAS_VENV_DIR/bin/python\" -m mas.remote_job "
               f"diagnostics --root {release} --episode {episode} --commit {data['commit']} "
               f"--input-sha256 {data['input_sha256']} --deadline-seconds 60")
    class DiagnosticBudget:
        def check(self):
            remaining = retrieval_deadline - time.monotonic()
            if remaining <= 0:
                raise RunPodControllerError("diagnostic shared grace expired")
            return remaining
    grace = DiagnosticBudget()
    response = _network_retry(ssh + [command], capture=True, attempts=2,
                              total_timeout=min(60, grace.check()), deadline=retrieval_deadline)
    manifest = json.loads(response)
    if manifest.get("identity") != expected_identity or manifest.get("kind") != "diagnostics":
        raise RunPodControllerError("diagnostic manifest identity mismatch")
    name = f"Muhtemel Ask {episode}.Bolum"
    allowed = {f"handoff/{name}_TR_CORRECTION_PACK.zip", "work/audio_review_v2.json",
               "work/forced_alignment_units/resume-identity.json",
               "work/forced_alignment_units/components/latest-conflict-failure.json",
               "work/forced_alignment_units/components/latest-resume-scope-violation.json",
               "source/download.done.json", "work/audio.done.json", "work/raw_asr_v2.done.json",
               "work/audio_review_v2.recovery.json", f"handoff/{name}_TR_TEXT_CORRECTED.zip",
               f"handoff/{name}_TR_CORRECTED.zip"}
    if manifest.get('part_id') is not None:
        from .retry_authorization import retry_diagnostic_layout
        layout = retry_diagnostic_layout(episode, manifest['part_id'])
        allowed = set(layout['files'])
        allowed.update(prefix + relative for prefix in layout['prefixes'] for relative in (
            'resume-identity.json', 'components/latest-conflict-failure.json',
            'components/latest-resume-scope-violation.json'))
        allowed.add(f"parts/{manifest['part_id']}/work/audio_review_v2.recovery.json")
    records = manifest.get("files")
    if (not isinstance(records, list) or len(records) > len(allowed)
            or any(not isinstance(record, dict) or record.get("relative_path") not in allowed for record in records)
            or len({record["relative_path"] for record in records}) != len(records)):
        raise RunPodControllerError("diagnostic manifest path set is invalid")
    atomic_json(local_root / "work/remote-diagnostic-manifest.json", manifest)
    for missing in manifest.get("missing", []) + manifest.get("unstable", []):
        if missing not in allowed:
            raise RunPodControllerError("diagnostic manifest missing path is invalid")
        print(f"[RUNPOD] diagnostic unavailable: {missing}", file=sys.stderr)
    for record in records:
        try:
            retained = _download_record(record, local_root, remote_root, scp, host, grace, checkpoint=True)
            canonical = safe_relative(local_root, record["relative_path"])
            mutable = (record['relative_path'] == 'work/current-part.json'
                       or re.fullmatch(r'parts/part-[0-9]{3}/work/state\.json', record['relative_path'])
                       or '/forced_alignment_units/' in record['relative_path']
                       or record['relative_path'].endswith('.recovery.json'))
            if not canonical.exists() or mutable:
                from .engine.download import atomic_write_bytes
                if canonical.is_file() and sha256_file(canonical) != record['sha256']:
                    previous = local_root / 'work/remote-checkpoints' / sha256_file(canonical) / canonical.name
                    atomic_write_bytes(previous, canonical.read_bytes())
                atomic_write_bytes(canonical, retained.read_bytes())
        except RunPodControllerError as exc:
            print(f"[RUNPOD] diagnostic download warning: {exc}", file=sys.stderr)
            if time.monotonic() >= retrieval_deadline:
                break


def _collect_partial_results(exit_code, episode, local_root, remote_root, ssh, scp, host, budget, temporary):
    from .partial_delivery import _read_bound, part_directory
    filename = 'partial-handoff.json' if exit_code == WAIT_PART_RETURN else 'partial-export.json'
    staged = temporary / filename
    record = {'relative_path': 'work/' + filename, 'storage_path': remote_root + '/work/' + filename}
    try:
        _network_retry(scp + [f'root@{host}:{remote_root}/work/{filename}', str(staged)],
                       budget=budget, total_timeout=min(60, budget.check()))
        manifest = (_read_bound(staged) if exit_code == WAIT_PART_RETURN else
                    json.loads(staged.read_text(encoding='utf-8')))
        if manifest.get('episode') != episode:
            raise RunPodControllerError('partial transfer episode mismatch')
        part_id = manifest.get('part_id')
        folder = part_directory(local_root, part_id)
        if exit_code == WAIT_PART_RETURN:
            if manifest.get('format') != 'mas-partial-handoff-1' or manifest.get('kind') not in {'tr', 'id'}:
                raise RunPodControllerError('invalid partial handoff transfer')
        elif manifest.get('mode') not in {'strict-partial-subtitles', 'delivery-first-subtitles'}:
            raise RunPodControllerError('invalid partial subtitle export')
        records = manifest.get('files', [])
        relatives = [item['relative_path'] for item in records]
        if not records or len(records) > 512 or len(relatives) != len(set(relatives)):
            raise RunPodControllerError('partial transfer manifest is empty, duplicated or unbounded')
        current = local_root / 'work' / filename
        if current.is_file() and sha256_file(current) != sha256_file(staged):
            retained = local_root / 'work/part-transfer-history' / (sha256_file(current) + '.json')
            from .engine.download import atomic_write_bytes
            atomic_write_bytes(retained, current.read_bytes())
        from .engine.download import atomic_write_bytes
        atomic_write_bytes(current, staged.read_bytes())
        for item in records:
            relative = item['relative_path']
            if not (relative.startswith(f'parts/{part_id}/') or relative.startswith('source/')
                    or relative.startswith('work/') or relative in {'work/part-plan.json', 'work/part-vad.json'}):
                raise RunPodControllerError('partial transfer escaped its part and parent evidence')
            if relative == f'parts/{part_id}/work/{filename}':
                continue
            record = {'relative_path': relative, 'storage_path': item.get('storage_path')}
            _download_record(item, local_root, remote_root, scp, host, budget)
        if exit_code == READY_FOR_PARTIAL_ENCODE:
            from .engine.part_audio import load_part_plan
            captions = load_part_plan(local_root, episode, verify_files=False).get('captions')
            if captions is not None and captions['relative_path'] not in relatives:
                if not captions['relative_path'].startswith('source/'):
                    raise RunPodControllerError('partial caption dependency escaped source evidence')
                _download_record(captions, local_root, remote_root, scp, host, budget)
            lineage_path = folder / 'work/audio-part.done.json'
            if lineage_path.is_file():
                lineage = _read_bound(lineage_path)
                derived_captions = lineage.get('captions')
                if derived_captions is not None and derived_captions['relative_path'] not in relatives:
                    if not derived_captions['relative_path'].startswith(f'parts/{part_id}/work/'):
                        raise RunPodControllerError('partial derived caption dependency escaped its part')
                    _download_record(derived_captions, local_root, remote_root, scp, host, budget)
        alias = folder / 'work' / filename
        if alias.is_file() and sha256_file(alias) != sha256_file(current):
            from .engine.download import atomic_write_bytes
            retained = local_root / 'work/part-transfer-history' / (sha256_file(alias) + '.json')
            atomic_write_bytes(retained, alias.read_bytes())
        atomic_write_bytes(alias, staged.read_bytes())
        if exit_code == WAIT_PART_RETURN:
            frozen = folder / 'work' / ('partial-handoff-' + manifest['kind'] + '.json')
            if frozen.exists() and frozen.read_bytes() != staged.read_bytes():
                raise RunPodControllerError('frozen partial handoff changed during transfer')
            atomic_write_bytes(frozen, staged.read_bytes())
            from .progressive import validate_partial_handoff
            validate_partial_handoff(local_root, episode, part_id)
            if manifest['kind'] == 'id':
                from .engine.translation_workspace import prepare_id_translation_workspaces
                prepare_id_translation_workspaces(safe_relative(local_root, manifest['pack']['relative_path']),
                                                  folder / 'handoff/id-workers')
        else:
            from .engine.partial_finalize import validate_partial_export
            validate_partial_export(local_root, episode, part_id, total_timeout=budget.check())
    except BaseException:
        if exit_code == READY_FOR_PARTIAL_ENCODE:
            _record_result_collection_failure(local_root, exit_code, record)
        raise


def _collect_remote_results(exit_code, episode, local_root, remote_root, ssh, scp, host, budget, temporary):
    name = f"Muhtemel Ask {episode}.Bolum"
    retrieval_deadline = time.monotonic() + 120
    if exit_code in (WAIT_PART_RETURN, READY_FOR_PARTIAL_ENCODE):
        _collect_partial_results(exit_code, episode, local_root, remote_root, ssh, scp, host, budget, temporary)
        return
    if exit_code == NEXT_PART:
        raise RunPodControllerError('worker has no unpublished part but local completion evidence is incomplete; '
                                    'refusing a no-progress GPU continuation')
    if exit_code == ALIGNMENT_RECOVERY_COMPLETE:
        for relative in ("work/forced_alignment_v2.json", "work/forced_alignment_v2.done.json", "work/state.json"):
            size, signature = _remote_file_signature(ssh, f"{remote_root}/{relative}", budget=budget)
            _download_record({"relative_path": relative, "size_bytes": size, "sha256": signature},
                             local_root, remote_root, scp, host, budget)
        return
    if exit_code in (READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, WAIT_MP4_SAMPLE):
        export_name = {READY_FOR_DELIVERY: "delivery-export.json", READY_FOR_LOCAL_ENCODE: "subtitle-export.json",
                       WAIT_MP4_SAMPLE: "sample-export.json"}[exit_code]
        export_path = temporary / export_name
        current_record = {
            "relative_path": f"work/{export_name}",
            "storage_path": f"{remote_root}/work/{export_name}",
        }
        try:
            _network_retry(scp + [f"root@{host}:{remote_root}/work/{export_name}", str(export_path)],
                           budget=budget, total_timeout=60)
            manifest = json.loads(export_path.read_text(encoding="utf-8"))
            expected_mode = {READY_FOR_DELIVERY: "strict", READY_FOR_LOCAL_ENCODE: "strict-subtitles",
                             WAIT_MP4_SAMPLE: "review"}[exit_code]
            if manifest.get("episode") != episode or manifest.get("mode") != expected_mode:
                raise RunPodControllerError("delivery export identity mismatch")
            atomic_json(local_root / "work" / export_name, manifest)
            for record in manifest["files"]:
                current_record = {
                    "relative_path": record.get("relative_path"),
                    "storage_path": record.get("storage_path"),
                }
                _download_record(record, local_root, remote_root, scp, host, budget)
        except BaseException:
            if exit_code in (READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE):
                _record_result_collection_failure(local_root, exit_code, current_record)
            raise

    if exit_code not in (0, 20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED,
                         READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, WAIT_MP4_SAMPLE,
                         ALIGNMENT_RECOVERY_COMPLETE):
        try:
            _collect_diagnostics(episode, local_root, remote_root, ssh, scp, host, retrieval_deadline)
        except RunPodControllerError as exc:
            print(f"[RUNPOD] diagnostic collection warning: {exc}", file=sys.stderr)

    handoff = None
    if exit_code == 20:
        handoff = f"{name}_TR_CORRECTION_PACK.zip"
    elif exit_code == 21:
        handoff = f"{name}_ID_TRANSLATION_PACK.zip"
    elif exit_code == SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED:
        handoff = f"{name}_SEMANTIC_ALIGNMENT_PACK.zip"
    if handoff:
        local_pack = local_root / "handoff" / handoff
        local_pack.parent.mkdir(parents=True, exist_ok=True)
        remote_pack = f"{remote_root}/handoff/{handoff}"
        size, signature = _remote_file_signature(ssh, remote_pack, budget=budget)
        _download_record({"relative_path": f"handoff/{handoff}",
                          "size_bytes": size, "sha256": signature},
                         local_root, remote_root, scp, host, budget)
        if exit_code == 21:
            import zipfile
            with zipfile.ZipFile(local_pack) as archive:
                schema = json.loads(archive.read("schema.json"))
            if "production_policy" in schema:
                from .engine.translation_workspace import prepare_id_translation_workspaces
                prepare_id_translation_workspaces(local_pack, local_root / "handoff/id-workers")
        print(f"[HANDOFF] downloaded {local_pack}")
    if exit_code not in (0, 20, 21, SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED,
                         READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE, WAIT_MP4_SAMPLE):
        raise RunPodControllerError(f"remote pipeline failed with exit code {exit_code}")


def _run_remote_session(episode, source_url, *, values, commit, budget, pod, resume_only=False,
                        alignment_recovery=False):
    name = f"Muhtemel Ask {episode}.Bolum"
    local_root = episode_dir(episode)
    code_fix_path = local_root / "work/code-fix-resume.json"
    if code_fix_path.is_file():
        values = dict(values, MAS_CODE_FIX_RESUME=f"/workspace/ma-sub/EPISODES/{name}/work/code-fix-resume.json")
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
                                            int(budget.check()), budget, resume_only=True,
                                            alignment_recovery=alignment_recovery)
            with tempfile.TemporaryDirectory(prefix="ma-sub-resume-") as folder:
                try:
                    _collect_remote_results(exit_code, episode, local_root,
                        f"/workspace/ma-sub/EPISODES/{name}", ssh, scp, host, budget, Path(folder))
                finally:
                    _record_failed_remote_evidence(exit_code, local_root, source_url, commit)
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
            _network_retry(ssh + [
                "install -d -m 700 /workspace/.mas-secrets /workspace/.mas-upload; "
                "rm -f -- /workspace/.mas-secrets/rclone.conf; "
                "install -m 600 /dev/null /workspace/.mas-secrets/runtime.env"
            ], budget=budget)
            _network_retry(scp + [str(archive), f"root@{host}:/workspace/.mas-upload/release.tar.gz"], budget=budget)
            _network_retry(scp + [str(runtime_env), f"root@{host}:/workspace/.mas-secrets/runtime.env"], budget=budget)
            if cookie is not None:
                _network_retry(
                    scp + [str(cookie), f"root@{host}:/workspace/.mas-secrets/youtube-cookies.txt"],
                    budget=budget,
                )
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
            _upload_episode_file_verified(local_root / 'work/controller_budget.json',
                f'{remote_root}/work/controller_budget.json', ssh=ssh, scp=scp, host=host,
                budget=budget, reuse_verified=True)
            for part_file in _part_resume_files(local_root, episode):
                relative = part_file.relative_to(local_root).as_posix()
                _upload_episode_file_verified(part_file, f'{remote_root}/{relative}',
                    ssh=ssh, scp=scp, host=host, budget=budget, reuse_verified=True)
            override_receipt = reset_receipt = speaker_receipt = None
            if not alignment_recovery:
                for filename in ("source.url", "official-source.json"):
                    _upload_episode_file_verified(local_root / "source" / filename,
                        f"{remote_root}/source/{filename}", ssh=ssh, scp=scp, host=host, budget=budget, immutable=True)
                source_receipts = _upload_verified_local_source(
                    local_root, remote_root, source_url, ssh=ssh, scp=scp, host=host,
                    temporary=temporary, budget=budget)
                if source_receipts is not None:
                    print("[RUNPOD] local source seed verified: "
                          f"video_bytes={source_receipts['video']['bytes']} "
                          f"video_sha256={source_receipts['video']['sha256']}")
                for filename in (
                    f"{name}_TR_TEXT_CORRECTED.zip",
                    f"{name}_SEMANTIC_ALIGNMENT_RETURN.zip",
                    f"{name}_ID_TRANSLATED.zip",
                ):
                    local_return = local_root / "handoff" / filename
                    if local_return.is_file():
                        destination = f"{remote_root}/handoff/{filename}"
                        receipt = _upload_episode_file_verified(local_return, destination, ssh=ssh, scp=scp, host=host,
                                                                budget=budget, reuse_verified=True)
                        print(f"[RUNPOD] return upload verified: {filename}; "
                              f"bytes={receipt['bytes']} sha256={receipt['sha256']}")
                        workspace_receipt = Path(str(local_return) + ".workspace.json")
                        if workspace_receipt.is_file():
                            _upload_episode_file_verified(workspace_receipt, destination + ".workspace.json",
                                                         ssh=ssh, scp=scp, host=host, budget=budget, reuse_verified=True)
                override_receipt = _upload_audio_review_overrides(local_root, remote_root, ssh=ssh, scp=scp,
                                                                   host=host, budget=budget)
                reset_receipt = _upload_audio_review_reset(local_root, remote_root, ssh=ssh, scp=scp,
                                                            host=host, budget=budget)
                speaker_receipt = _upload_speaker_evidence(local_root, remote_root, ssh=ssh, scp=scp,
                                                            host=host, budget=budget)
                approval = local_root / "work" / "mp4-sample-approval.json"
                if approval.is_file():
                    _upload_episode_file_verified(approval, f"{remote_root}/work/{approval.name}",
                                                   ssh=ssh, scp=scp, host=host, budget=budget)
            if code_fix_path.is_file():
                from .retry_authorization import validate_code_fix_resume
                validate_code_fix_resume(local_root, episode, commit, code_fix_path)
                permit = json.loads(code_fix_path.read_text(encoding="utf-8"))["data"]
                relatives = {record["relative_path"] for record in permit["diagnostics"]}
                relatives.update(record["relative_path"] for record in permit["predecessors"])
                relatives.update({permit["fixture_result"]["relative_path"], "work/controller_budget.json",
                                  "work/remote-failure-invariant.json", "work/code-fix-resume.json"})
                for relative in sorted(relatives):
                    _upload_episode_file_verified(safe_relative(local_root, relative), f"{remote_root}/{relative}",
                                                 ssh=ssh, scp=scp, host=host, budget=budget, reuse_verified=True)
            if override_receipt is not None:
                print(
                    "[RUNPOD] audio review overrides upload verified: "
                    f"bytes={override_receipt['bytes']} "
                    f"sha256={override_receipt['sha256']}"
                )
            if reset_receipt is not None:
                print(
                    "[RUNPOD] audio review reset upload verified: "
                    f"bytes={reset_receipt['bytes']} "
                    f"sha256={reset_receipt['sha256']}"
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
            try:
                exit_code = _monitor_remote_job(ssh, scp, host, episode, commit, local_root,
                                                source_url, runtime_seconds, budget,
                                                alignment_recovery=alignment_recovery)
            except BaseException:
                _record_failed_remote_evidence(1, local_root, source_url, commit)
                raise
            try:
                _collect_remote_results(exit_code, episode, local_root, remote_root, ssh, scp, host, budget, temporary)
            finally:
                _record_failed_remote_evidence(exit_code, local_root, source_url, commit)
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
