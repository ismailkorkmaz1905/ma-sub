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
from .remote import RemoteVerificationError, _run_watchdog


class RunPodControllerError(RuntimeError):
    pass


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

    def request(self, method, suffix=""):
        request = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{self.pod_id}{suffix}",
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        last_error = None
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                    if response.status < 200 or response.status >= 300:
                        raise RunPodControllerError(f"RunPod returned HTTP {response.status}")
                    return json.loads(payload) if payload else {}
            except (urllib.error.URLError, TimeoutError, RunPodControllerError) as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    self.sleep(2 ** attempt)
        raise RunPodControllerError(f"RunPod request failed after {self.attempts} attempts: {last_error}")

    def get(self):
        return self.request("GET")

    def start(self):
        return self.request("POST", "/start")

    def stop(self):
        return self.request("POST", "/stop")

    def wait(self, predicate, description, *, timeout, poll=5):
        started = time.monotonic()
        last = None
        while time.monotonic() - started < timeout:
            last = self.get()
            if predicate(last):
                return last
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


def _stream(chunk, target):
    target.write(chunk.decode("utf-8", "replace"))
    target.flush()


def _network(command, *, idle_timeout=180, total_timeout=1800):
    try:
        return _run_watchdog(
            command,
            idle_timeout=idle_timeout,
            total_timeout=total_timeout,
            stdout_handler=lambda chunk: _stream(chunk, sys.stdout),
            stderr_handler=lambda chunk: _stream(chunk, sys.stderr),
        )
    except RemoteVerificationError as exc:
        raise RunPodControllerError(str(exc)) from exc


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
    while time.monotonic() - started < timeout:
        try:
            result = subprocess.run(command, capture_output=True, timeout=20, check=False)
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            return
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
        "MAS_EXTERNAL_RUNPOD_CONTROLLER": "1",
    }
    for name in ("MAS_MAX_RUNTIME_SECONDS", "MAS_IDLE_TIMEOUT_SECONDS"):
        if os.getenv(name):
            remote_values[name] = os.environ[name]
    lines = [f"export {name}={shlex.quote(value)}" for name, value in remote_values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_remote_episode(episode, source_url=None):
    if episode <= 0:
        raise RunPodControllerError("episode must be a positive integer")
    values = _required_environment()
    commit, rclone_config = _local_preflight(values)
    key = Path(values["MAS_RUNPOD_SSH_KEY"]).resolve()
    cookie = Path(values["MAS_YTDLP_COOKIES"]).resolve()
    client = RunPodClient(values["RUNPOD_POD_ID"], values["RUNPOD_API_KEY"])
    initial = client.get()
    if initial.get("desiredStatus") != "EXITED":
        raise RunPodControllerError(
            f"pod must be EXITED before an automatic run; status={initial.get('desiredStatus', 'UNKNOWN')}"
        )

    started_at = time.monotonic()
    controller_started_pod = False
    endpoint = None
    try:
        print(f"[RUNPOD] starting pod {values['RUNPOD_POD_ID']}")
        controller_started_pod = True
        client.start()
        pod = client.wait(
            lambda item: item.get("desiredStatus") == "RUNNING" and _ssh_endpoint(item),
            "startup",
            timeout=600,
        )
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
            _network(ssh + ["install -d -m 700 /workspace/.mas-secrets /workspace/.mas-upload"])
            _network(scp + [str(archive), f"root@{host}:/workspace/.mas-upload/release.tar.gz"])
            _network(scp + [str(runtime_env), f"root@{host}:/workspace/.mas-secrets/runtime.env"])
            _network(scp + [str(cookie), f"root@{host}:/workspace/.mas-secrets/youtube-cookies.txt"])
            _network(scp + [str(rclone_config), f"root@{host}:/workspace/.mas-secrets/rclone.conf"])

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
            _network(ssh + [deploy_command], idle_timeout=600, total_timeout=3600)

            name = f"Muhtemel Ask {episode}.Bolum"
            local_root = episode_dir(episode)
            remote_root = f"/workspace/ma-sub/EPISODES/{name}"
            for filename in (f"{name}_TR_TEXT_CORRECTED.zip", f"{name}_ID_TRANSLATED.zip"):
                local_return = local_root / "translation_output" / filename
                if local_return.is_file():
                    _network(scp + [str(local_return), f"root@{host}:/workspace/.mas-upload/return.zip"])
                    destination = f"{remote_root}/translation_output/{filename}"
                    _network(
                        ssh
                        + [
                            "install -D -m 600 /workspace/.mas-upload/return.zip "
                            + shlex.quote(destination)
                        ]
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
            _network(scp + [f"root@{host}:/workspace/.mas-upload/exit-code", str(exit_file)])
            try:
                exit_code = int(exit_file.read_text(encoding="ascii").strip())
            except (OSError, UnicodeError, ValueError) as exc:
                raise RunPodControllerError("remote pipeline exit code is invalid") from exc

            handoff = None
            if exit_code == 20:
                handoff = f"{name}_TR_CORRECTION_PACK.zip"
            elif exit_code == 21:
                handoff = f"{name}_ID_TRANSLATION_PACK.zip"
            if handoff:
                local_pack = local_root / "translation_input" / handoff
                local_pack.parent.mkdir(parents=True, exist_ok=True)
                remote_pack = f"{remote_root}/translation_input/{handoff}"
                _network(ssh + ["cp -- " + shlex.quote(remote_pack) + " /workspace/.mas-upload/handoff.zip"])
                _network(scp + [f"root@{host}:/workspace/.mas-upload/handoff.zip", str(local_pack)])
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
