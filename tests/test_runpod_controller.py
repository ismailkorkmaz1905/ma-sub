import io
import json
from pathlib import Path

import pytest

from mas import cli
from mas import runpod_controller


class _Response:
    status = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.value).encode()


def test_client_uses_bounded_authenticated_rest_request(monkeypatch):
    captured = {}

    def open_request(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return _Response({"desiredStatus": "EXITED"})

    monkeypatch.setattr(runpod_controller.urllib.request, "urlopen", open_request)
    client = runpod_controller.RunPodClient("pod123", "secret", timeout=7)
    assert client.get()["desiredStatus"] == "EXITED"
    assert captured == {
        "url": "https://rest.runpod.io/v1/pods/pod123",
        "authorization": "Bearer secret",
        "timeout": 7,
    }


def test_client_preserves_safe_runpod_http_error_detail(monkeypatch):
    def open_request(request, timeout):
        raise runpod_controller.urllib.error.HTTPError(
            request.full_url,
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b'{"error":"GPU host is unavailable"}'),
        )

    monkeypatch.setattr(runpod_controller.urllib.request, "urlopen", open_request)
    client = runpod_controller.RunPodClient("pod123", "secret", attempts=1)
    with pytest.raises(runpod_controller.RunPodControllerError, match="GPU host is unavailable"):
        client.start()


def test_client_create_pod_uses_single_json_request(monkeypatch):
    captured = {}

    def open_request(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["content_type"] = request.headers["Content-type"]
        captured["payload"] = json.loads(request.data)
        return _Response({"id": "newpod"})

    monkeypatch.setattr(runpod_controller.urllib.request, "urlopen", open_request)
    client = runpod_controller.RunPodClient("oldpod", "secret")

    assert client.create_pod({"gpuCount": 1}) == {"id": "newpod"}
    assert captured == {
        "url": "https://rest.runpod.io/v1/pods",
        "method": "POST",
        "content_type": "application/json",
        "payload": {"gpuCount": 1},
    }


def test_capacity_migration_preserves_exact_network_volume(monkeypatch):
    for name, value in {
        "MAS_RUNPOD_NETWORK_VOLUME_ID": "volume123",
        "MAS_RUNPOD_DATA_CENTER_ID": "EU-RO-1",
        "MAS_RUNPOD_GPU_TYPE_ID": "NVIDIA GeForce RTX 4090",
        "MAS_RUNPOD_MAX_COST_PER_HR": "0.75",
    }.items():
        monkeypatch.setenv(name, value)
    persisted = []
    monkeypatch.setattr(
        runpod_controller, "_persist_windows_user_pod_id", persisted.append
    )

    class Client:
        pod_id = "oldpod"
        api_key = "secret"
        timeout = 15
        attempts = 3
        sleep = staticmethod(lambda _seconds: None)

        def __init__(self):
            self.terminated = False
            self.payload = None

        def get_network_volume(self, volume_id):
            assert volume_id == "volume123"
            return {"id": volume_id, "dataCenterId": "EU-RO-1", "size": 50}

        def terminate(self):
            self.terminated = True

        def create_pod(self, payload):
            self.payload = payload
            return {
                "id": "newpod",
                "networkVolumeId": "volume123",
                "costPerHr": "0.74",
            }

    client = Client()
    pod = {
        "desiredStatus": "EXITED",
        "networkVolumeId": "volume123",
        "volumeInGb": 0,
        "containerDiskInGb": 30,
        "imageName": "runpod/image",
        "name": "production",
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": "public"},
    }

    replacement = runpod_controller._migrate_capacity_bound_pod(client, pod)

    assert client.terminated is True
    assert client.payload["networkVolumeId"] == "volume123"
    assert client.payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert client.payload["dataCenterIds"] == ["EU-RO-1"]
    assert replacement.pod_id == "newpod"
    assert persisted == ["newpod"]


def test_wait_requires_running_ip_and_ssh_mapping():
    assert runpod_controller._ssh_endpoint({"desiredStatus": "RUNNING"}) is None
    assert runpod_controller._ssh_endpoint(
        {"publicIp": "192.0.2.4", "portMappings": {"22": 10022}}
    ) == ("192.0.2.4", "10022")


def test_safe_network_command_retries_with_backoff(monkeypatch):
    calls = []
    sleeps = []

    def network(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) < 3:
            raise runpod_controller.RunPodControllerError("transient SSH failure")
        return b"ok"

    monkeypatch.setattr(runpod_controller, "_network", network)
    monkeypatch.setattr(runpod_controller.time, "sleep", sleeps.append)

    assert runpod_controller._network_retry(["ssh", "host", "true"]) == b"ok"
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_ssh_has_bounded_liveness_options(tmp_path):
    command = runpod_controller._ssh_args(tmp_path / "key", "192.0.2.4", "10022")
    assert "ConnectionAttempts=2" in command
    assert "ServerAliveInterval=10" in command
    assert "ServerAliveCountMax=3" in command


def test_running_pod_requires_explicit_one_time_adoption(monkeypatch):
    pod = {"desiredStatus": "RUNNING"}
    monkeypatch.delenv("MAS_RUNPOD_ADOPT_RUNNING", raising=False)
    with pytest.raises(runpod_controller.RunPodControllerError, match="must be EXITED"):
        runpod_controller._startup_mode(pod)
    monkeypatch.setenv("MAS_RUNPOD_ADOPT_RUNNING", "1")
    assert runpod_controller._startup_mode(pod) == "adopt"
    assert runpod_controller._startup_mode({"desiredStatus": "EXITED"}) == "start"


def test_ssh_authentication_failure_stops_without_retry(monkeypatch, tmp_path):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return type(
            "Result",
            (),
            {"returncode": 255, "stderr": b"root@host: Permission denied (publickey)."},
        )()

    monkeypatch.setattr(runpod_controller.subprocess, "run", run)
    monkeypatch.setattr(
        runpod_controller.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError("must not retry")),
    )

    with pytest.raises(runpod_controller.RunPodControllerError, match="authentication rejected"):
        runpod_controller._wait_for_ssh(tmp_path / "key", "192.0.2.4", "10022")
    assert len(calls) == 1


def test_runtime_environment_is_shell_quoted_and_does_not_log_secrets(tmp_path):
    values = {
        "RUNPOD_POD_ID": "pod123",
        "RUNPOD_API_KEY": "api secret",
        "MAS_GMAIL_ADDRESS": "from@example.com",
        "MAS_GMAIL_APP_PASSWORD": "mail secret",
        "MAS_NOTIFY_TO": "to@example.com",
        "MAS_DRIVE_STRICT_REMOTE": "gdrive:path with spaces",
    }
    target = tmp_path / "runtime.env"
    runpod_controller._write_runtime_env(target, values, "a" * 40)
    content = target.read_text(encoding="utf-8")
    assert "export RUNPOD_API_KEY='api secret'" in content
    assert "MAS_GIT_COMMIT=" + "a" * 40 in content
    assert "youtube-cookies.txt" in content
    assert "UV_CACHE_DIR=/workspace/.cache/uv" in content
    assert "UV_HTTP_TIMEOUT=120" in content
    assert "UV_HTTP_RETRIES=3" in content
    assert "UV_CONCURRENT_DOWNLOADS=4" in content
    assert "HF_HOME=/workspace/.cache/huggingface" in content
    assert b"\r" not in target.read_bytes()


def test_cli_dispatches_production_run_to_controller(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("mas.runlog.episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run_remote_episode", lambda *args: calls.append(args) or 0)
    monkeypatch.setattr(cli, "run", lambda *args: (_ for _ in ()).throw(AssertionError("local")))
    assert cli.main(["run", "14"]) == 0
    assert calls == [(14, None)]


def test_cli_local_flag_prevents_controller_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("mas.runlog.episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run_remote_episode", lambda *args: (_ for _ in ()).throw(AssertionError("remote")))
    monkeypatch.setattr(cli, "run", lambda *args: calls.append(args) or 20)
    assert cli.main(["run", "14", "--local"]) == 20
    assert calls == [(14, None, False, None)]


def test_local_preflight_rejects_dirty_repository(monkeypatch, tmp_path):
    key = tmp_path / "key"
    cookie = tmp_path / "cookies"
    key.write_text("key", encoding="utf-8")
    cookie.write_text("cookie", encoding="utf-8")
    monkeypatch.setattr(runpod_controller.shutil, "which", lambda value: value)
    monkeypatch.setattr(
        runpod_controller.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": " M file\n"})(),
    )
    with pytest.raises(runpod_controller.RunPodControllerError, match="must be clean"):
        runpod_controller._local_preflight(
            {"MAS_RUNPOD_SSH_KEY": str(key), "MAS_YTDLP_COOKIES": str(cookie)}
        )


def test_start_timeout_still_stops_a_pod_that_became_running(monkeypatch, tmp_path):
    key = tmp_path / "key"
    cookie = tmp_path / "cookie"
    config = tmp_path / "rclone.conf"
    for path in (key, cookie, config):
        path.write_text("value", encoding="utf-8")
    values = {
        "RUNPOD_POD_ID": "pod123",
        "RUNPOD_API_KEY": "api",
        "MAS_RUNPOD_SSH_KEY": str(key),
        "MAS_YTDLP_COOKIES": str(cookie),
        "MAS_GMAIL_ADDRESS": "from@example.com",
        "MAS_GMAIL_APP_PASSWORD": "mail",
        "MAS_NOTIFY_TO": "to@example.com",
        "MAS_DRIVE_STRICT_REMOTE": "gdrive:path",
    }

    class Client:
        def __init__(self, *args):
            self.running = False
            self.stopped = False

        def get(self):
            return {"desiredStatus": "RUNNING" if self.running else "EXITED"}

        def start(self):
            self.running = True
            raise runpod_controller.RunPodControllerError("response lost")

        def stop(self):
            self.running = False
            self.stopped = True

        def wait(self, predicate, description, *, timeout):
            value = self.get()
            assert predicate(value)
            return value

    client = Client()
    monkeypatch.setattr(runpod_controller, "RunPodClient", lambda *args: client)
    monkeypatch.setattr(runpod_controller, "_required_environment", lambda: values)
    monkeypatch.setattr(runpod_controller, "_local_preflight", lambda values: ("a" * 40, config))

    with pytest.raises(runpod_controller.RunPodControllerError, match="response lost"):
        runpod_controller.run_remote_episode(11)
    assert client.stopped is True
