import hashlib
import json
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

from .config import ROOT, episode_dir
from .engine.tr_correction import validate_tr_correction_output
from .remote import RemoteVerificationError, _run_watchdog


class RunPodControllerError(RuntimeError):
    pass


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
    def __init__(self, pod_id, api_key, *, timeout=15, attempts=3, sleep=time.sleep):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id or ""):
            raise RunPodControllerError("RUNPOD_POD_ID is missing or invalid")
        if not api_key or "\r" in api_key or "\n" in api_key:
            raise RunPodControllerError("RUNPOD_API_KEY is missing or invalid")
        self.pod_id = pod_id
        self.api_key = api_key
        self.timeout = timeout
        self.attempts = attempts
        self.sleep = sleep

    def _request_url(self, method, url, *, payload=None, attempts=None):
        data = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
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
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                    if response.status < 200 or response.status >= 300:
                        raise RunPodControllerError(f"RunPod returned HTTP {response.status}")
                    return json.loads(payload) if payload else {}
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
                    self.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError, RunPodControllerError) as exc:
                last_error = exc
                if attempt + 1 < request_attempts:
                    self.sleep(2 ** attempt)
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
            self.sleep(poll)
        status = (last or {}).get("desiredStatus", "UNKNOWN")
        raise RunPodControllerError(f"RunPod {description} timed out after {timeout}s; status={status}")


def _required_environment():
    names = (
        "RUNPOD_POD_ID",
        "RUNPOD_API_KEY",
        "MAS_RUNPOD_SSH_KEY",
        "MAS_YTDLP_COOKIES",
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
    for name in ("MAS_RUNPOD_SSH_KEY", "MAS_YTDLP_COOKIES"):
        path = Path(values[name]).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise RunPodControllerError(f"{name} file is missing or empty: {path}")
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
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", new_id or "")
        or created.get("networkVolumeId") != volume_id
    ):
        raise RunPodControllerError("replacement Pod response failed identity validation")
    max_cost = float(os.getenv("MAS_RUNPOD_MAX_COST_PER_HR", "0.75"))
    cost = float(created.get("costPerHr") or created.get("adjustedCostPerHr") or 0)
    replacement = RunPodClient(
        new_id,
        client.api_key,
        timeout=client.timeout,
        attempts=client.attempts,
        sleep=client.sleep,
    )
    if cost <= 0 or cost > max_cost:
        replacement.terminate()
        raise RunPodControllerError(
            f"replacement Pod hourly cost {cost} exceeds allowed {max_cost}"
        )
    try:
        _persist_windows_user_pod_id(new_id)
    except Exception:
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


def _network(command, *, idle_timeout=180, total_timeout=1800, capture=False):
    try:
        return _run_watchdog(
            command,
            idle_timeout=idle_timeout,
            total_timeout=total_timeout,
            stdout_handler=None if capture else lambda chunk: _stream(chunk, sys.stdout),
            stderr_handler=None if capture else lambda chunk: _stream(chunk, sys.stderr),
        )
    except RemoteVerificationError as exc:
        raise RunPodControllerError(str(exc)) from exc


def _network_retry(
    command, *, attempts=3, idle_timeout=60, total_timeout=1800, capture=False
):
    for attempt in range(1, attempts + 1):
        try:
            return _network(
                command,
                idle_timeout=idle_timeout,
                total_timeout=total_timeout,
                capture=capture,
            )
        except RunPodControllerError:
            if attempt == attempts:
                raise
            delay = 2 ** (attempt - 1)
            print(f"[RUNPOD] network command retry {attempt + 1}/{attempts} in {delay}s")
            time.sleep(delay)


def _remote_file_signature(ssh, remote_path):
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
    )
    match = re.fullmatch(
        rb"\s*(\d+)\s*\r?\n([0-9a-f]{64})\s+[^\r\n]+\s*",
        output,
    )
    if not match:
        raise RunPodControllerError("remote file byte/SHA-256 readback is invalid")
    return int(match.group(1)), match.group(2).decode("ascii")


def _upload_episode_file_verified(source, remote_path, *, ssh, scp, host):
    source = Path(source)
    expected_size = source.stat().st_size
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    expected_sha256 = digest.hexdigest()
    partial = (
        "/workspace/.mas-upload/audio-review-overrides-"
        f"{expected_sha256[:16]}.partial"
    )
    remote_parent = str(Path(remote_path).parent).replace("\\", "/")
    _network_retry(
        ssh + ["install -d -m 700 -- " + shlex.quote(remote_parent)],
        idle_timeout=60,
        total_timeout=300,
    )
    _network_retry(
        scp + [str(source), f"root@{host}:{partial}"],
        idle_timeout=60,
        total_timeout=300,
    )
    partial_size, partial_sha256 = _remote_file_signature(ssh, partial)
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
    )
    final_size, final_sha256 = _remote_file_signature(ssh, remote_path)
    if (final_size, final_sha256) != (expected_size, expected_sha256):
        raise RunPodControllerError(
            "installed episode file byte/SHA-256 readback mismatch"
        )
    return {"bytes": expected_size, "sha256": expected_sha256}


def _upload_audio_review_overrides(local_root, remote_root, *, ssh, scp, host):
    source = Path(local_root) / "review" / "audio_review_overrides.json"
    if not source.is_file():
        return None
    return _upload_episode_file_verified(
        source,
        f"{remote_root}/review/audio_review_overrides.json",
        ssh=ssh,
        scp=scp,
        host=host,
    )


def _ssh_args(key, host, port):
    return [
        "ssh",
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
            result = subprocess.run(command, capture_output=True, timeout=20, check=False)
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
        time.sleep(5)
    raise RunPodControllerError(f"SSH readiness timed out after {timeout}s")


def _write_runtime_env(path, values, commit):
    remote_values = {
        "RUNPOD_POD_ID": values["RUNPOD_POD_ID"],
        "RUNPOD_API_KEY": values["RUNPOD_API_KEY"],
        "MAS_GMAIL_ADDRESS": values["MAS_GMAIL_ADDRESS"],
        "MAS_GMAIL_APP_PASSWORD": values["MAS_GMAIL_APP_PASSWORD"],
        "MAS_NOTIFY_TO": values["MAS_NOTIFY_TO"],
        "MAS_DRIVE_STRICT_REMOTE": values["MAS_DRIVE_STRICT_REMOTE"],
        "MAS_YTDLP_COOKIES": "/workspace/.mas-secrets/youtube-cookies.txt",
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
    for name in ("MAS_MAX_RUNTIME_SECONDS", "MAS_IDLE_TIMEOUT_SECONDS"):
        if os.getenv(name):
            remote_values[name] = os.environ[name]
    lines = [f"export {name}={shlex.quote(value)}" for name, value in remote_values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_remote_episode(episode, source_url=None):
    if episode <= 0:
        raise RunPodControllerError("episode must be a positive integer")
    values = _required_environment()
    commit, rclone_config = _local_preflight(values)
    key = Path(values["MAS_RUNPOD_SSH_KEY"]).resolve()
    cookie = Path(values["MAS_YTDLP_COOKIES"]).resolve()
    client = RunPodClient(values["RUNPOD_POD_ID"], values["RUNPOD_API_KEY"])
    initial = client.get()
    startup_mode = _startup_mode(initial)

    started_at = time.monotonic()
    controller_started_pod = False
    endpoint = None
    try:
        controller_started_pod = True
        replacement_count = 0
        max_replacements = 2
        startup_timeout = int(os.getenv("MAS_RUNPOD_STARTUP_TIMEOUT_SECONDS", "180"))
        if not 60 <= startup_timeout <= 600:
            raise RunPodControllerError(
                "MAS_RUNPOD_STARTUP_TIMEOUT_SECONDS must be within [60, 600]"
            )
        if startup_mode == "start":
            print(f"[RUNPOD] starting pod {values['RUNPOD_POD_ID']}")
            try:
                client.start()
            except RunPodControllerError as exc:
                if (
                    os.getenv("MAS_RUNPOD_AUTO_MIGRATE") != "1"
                    or not _is_capacity_error(exc)
                ):
                    raise
                controller_started_pod = False
                client = _migrate_capacity_bound_pod(client, initial)
                values["RUNPOD_POD_ID"] = client.pod_id
                controller_started_pod = True
                replacement_count += 1
        else:
            print(f"[RUNPOD] adopting newly deployed pod {values['RUNPOD_POD_ID']}")
        while True:
            try:
                pod = client.wait(
                    lambda item: item.get("desiredStatus") == "RUNNING"
                    and _ssh_endpoint(item),
                    "startup",
                    timeout=startup_timeout,
                )
                break
            except RunPodControllerError as exc:
                if (
                    os.getenv("MAS_RUNPOD_AUTO_MIGRATE") != "1"
                    or "startup timed out" not in str(exc)
                    or replacement_count >= max_replacements
                ):
                    raise
                current = client.get()
                if current.get("desiredStatus") == "RUNNING":
                    client.stop()
                    current = client.wait(
                        lambda item: item.get("desiredStatus") == "EXITED",
                        "stalled-host shutdown",
                        timeout=300,
                    )
                if current.get("desiredStatus") != "EXITED":
                    raise
                client = _migrate_capacity_bound_pod(client, current)
                values["RUNPOD_POD_ID"] = client.pod_id
                replacement_count += 1
        host, port = _ssh_endpoint(pod)
        endpoint = (host, port)
        _wait_for_ssh(key, host, port)
        print(f"[RUNPOD] ready after {time.monotonic() - started_at:.1f}s")

        with tempfile.TemporaryDirectory(prefix="ma-sub-runpod-") as temporary:
            temporary = Path(temporary)
            archive = temporary / "release.tar.gz"
            runtime_env = temporary / "runtime.env"
            subprocess.run(
                ["git", "archive", "--format=tar.gz", f"--output={archive}", commit],
                cwd=ROOT,
                timeout=60,
                check=True,
            )
            _write_runtime_env(runtime_env, values, commit)
            ssh = _ssh_args(key, host, port)
            scp = _scp_args(key, host, port)
            _network_retry(ssh + ["install -d -m 700 /workspace/.mas-secrets /workspace/.mas-upload"])
            _network_retry(scp + [str(archive), f"root@{host}:/workspace/.mas-upload/release.tar.gz"])
            _network_retry(scp + [str(runtime_env), f"root@{host}:/workspace/.mas-secrets/runtime.env"])
            _network_retry(scp + [str(cookie), f"root@{host}:/workspace/.mas-secrets/youtube-cookies.txt"])
            _network_retry(scp + [str(rclone_config), f"root@{host}:/workspace/.mas-secrets/rclone.conf"])

            deploy_command = (
                "set -euo pipefail; "
                "chmod 600 /workspace/.mas-secrets/*; "
                "install -d /workspace/ma-sub/EPISODES /workspace/ma-sub/releases; "
                f"release=/workspace/ma-sub/releases/{commit}; "
                "rm -rf -- \"$release\"; install -d \"$release\"; "
                "tar -xzf /workspace/.mas-upload/release.tar.gz -C \"$release\"; "
                "ln -s /workspace/ma-sub/EPISODES \"$release/EPISODES\"; "
                f"find /workspace/ma-sub/releases -mindepth 1 -maxdepth 1 -type d ! -name {commit} -exec rm -rf -- {{}} +; "
                "source /workspace/.mas-secrets/runtime.env; "
                "cd \"$release\"; ./runpod/bootstrap.sh"
            )
            _network_retry(
                ssh + [deploy_command],
                idle_timeout=600,
                total_timeout=3600,
            )

            name = f"Muhtemel Ask {episode}.Bolum"
            local_root = episode_dir(episode)
            remote_root = f"/workspace/ma-sub/EPISODES/{name}"
            _validate_local_tr_return(local_root, name)
            for filename in (f"{name}_TR_TEXT_CORRECTED.zip", f"{name}_ID_TRANSLATED.zip"):
                local_return = local_root / "translation_output" / filename
                if local_return.is_file():
                    _network_retry(scp + [str(local_return), f"root@{host}:/workspace/.mas-upload/return.zip"])
                    destination = f"{remote_root}/translation_output/{filename}"
                    _network_retry(
                        ssh
                        + [
                            "install -D -m 600 /workspace/.mas-upload/return.zip "
                            + shlex.quote(destination)
                        ]
                    )

            override_receipt = _upload_audio_review_overrides(
                local_root,
                remote_root,
                ssh=ssh,
                scp=scp,
                host=host,
            )
            if override_receipt is not None:
                print(
                    "[RUNPOD] audio review overrides upload verified: "
                    f"bytes={override_receipt['bytes']} "
                    f"sha256={override_receipt['sha256']}"
                )

            arguments = [str(episode)]
            if source_url:
                arguments.extend(["--source-url", source_url])
            quoted_arguments = " ".join(shlex.quote(value) for value in arguments)
            run_command = (
                "set -uo pipefail; source /workspace/.mas-secrets/runtime.env; "
                f"cd /workspace/ma-sub/releases/{commit}; "
                f"./runpod/run-episode.sh {quoted_arguments}; rc=$?; "
                "printf '%s\\n' \"$rc\" > /workspace/.mas-upload/exit-code; exit 0"
            )
            _network(ssh + [run_command], idle_timeout=1800, total_timeout=18000)
            exit_file = temporary / "exit-code"
            _network_retry(scp + [f"root@{host}:/workspace/.mas-upload/exit-code", str(exit_file)])
            try:
                exit_code = int(exit_file.read_text(encoding="ascii").strip())
            except (OSError, UnicodeError, ValueError) as exc:
                raise RunPodControllerError("remote pipeline exit code is invalid") from exc

            if exit_code not in (0, 20, 21):
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
                        )
                    except RunPodControllerError as exc:
                        print(
                            f"[RUNPOD] diagnostic download warning: {exc}",
                            file=sys.stderr,
                        )
                        continue
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
                _network_retry(ssh + ["cp -- " + shlex.quote(remote_pack) + " /workspace/.mas-upload/handoff.zip"])
                _network_retry(scp + [f"root@{host}:/workspace/.mas-upload/handoff.zip", str(local_pack)])
                print(f"[HANDOFF] downloaded {local_pack}")
            if exit_code not in (0, 20, 21):
                raise RunPodControllerError(f"remote pipeline failed with exit code {exit_code}")
    finally:
        if controller_started_pod:
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
            current = client.get()
            if current.get("desiredStatus") != "EXITED":
                client.stop()
                client.wait(lambda item: item.get("desiredStatus") == "EXITED", "shutdown", timeout=300)
            print(f"[RUNPOD] shutdown verified after {time.monotonic() - shutdown_started:.1f}s")
    print(f"[RUNPOD] total elapsed {time.monotonic() - started_at:.1f}s")
    return exit_code
