import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mas import cli
from mas import runpod_controller
from mas.reliability import BudgetExceeded


def _remote_identity(request):
    episode = request["episode"]
    commit = request["commit"]
    input_sha = request["input_sha256"]
    return {
        "episode": episode,
        "commit": commit,
        "input_sha256": input_sha,
        "token": hashlib.sha256(f"{episode}\n{commit}\n{input_sha}\n".encode()).hexdigest(),
    }


def test_local_tr_return_must_match_current_pack(tmp_path, monkeypatch):
    name = "Muhtemel Ask 11.Bolum"
    pack = tmp_path / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip"
    returned = tmp_path / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
    pack.parent.mkdir()
    returned.parent.mkdir()
    pack.write_bytes(b"pack")
    returned.write_bytes(b"returned")
    checked = []
    monkeypatch.setattr(
        runpod_controller,
        "validate_tr_correction_output",
        lambda actual_pack, actual_return: checked.append(
            (actual_pack, actual_return)
        ),
    )

    runpod_controller._validate_local_tr_return(tmp_path, name)

    assert checked == [(pack, returned)]


def test_stale_local_tr_return_fails_before_upload(tmp_path, monkeypatch):
    name = "Muhtemel Ask 11.Bolum"
    pack = tmp_path / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip"
    returned = tmp_path / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
    pack.parent.mkdir()
    returned.parent.mkdir()
    pack.write_bytes(b"pack")
    returned.write_bytes(b"stale")
    monkeypatch.setattr(
        runpod_controller,
        "validate_tr_correction_output",
        lambda *_paths: (_ for _ in ()).throw(ValueError("SHA mismatch")),
    )

    with pytest.raises(
        runpod_controller.RunPodControllerError,
        match="does not match the current pack",
    ):
        runpod_controller._validate_local_tr_return(tmp_path, name)


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
        captured["user_agent"] = request.get_header("User-agent")
        captured["timeout"] = timeout
        return _Response({"desiredStatus": "EXITED"})

    monkeypatch.setattr(runpod_controller.urllib.request, "urlopen", open_request)
    client = runpod_controller.RunPodClient("pod123", "secret", timeout=7)
    assert client.get()["desiredStatus"] == "EXITED"
    assert captured == {
        "url": "https://rest.runpod.io/v1/pods/pod123",
        "authorization": "Bearer secret",
        "user_agent": "ma-sub-pilot/1.0",
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
    monkeypatch.delenv("MAS_RUNPOD_GPU_TYPE_IDS", raising=False)
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


def test_capacity_migration_can_select_from_bounded_gpu_pool(monkeypatch):
    monkeypatch.setenv("MAS_RUNPOD_NETWORK_VOLUME_ID", "volume123")
    monkeypatch.setenv("MAS_RUNPOD_DATA_CENTER_ID", "EU-RO-1")
    monkeypatch.setenv(
        "MAS_RUNPOD_GPU_TYPE_IDS",
        "NVIDIA RTX A5000|NVIDIA A40|NVIDIA GeForce RTX 4090",
    )
    monkeypatch.setenv("MAS_RUNPOD_MAX_COST_PER_HR", "0.75")
    monkeypatch.setattr(
        runpod_controller, "_persist_windows_user_pod_id", lambda _pod_id: None
    )

    class Client:
        pod_id = "oldpod"
        api_key = "secret"
        timeout = 15
        attempts = 3
        sleep = staticmethod(lambda _seconds: None)

        def get_network_volume(self, _volume_id):
            return {"id": "volume123", "dataCenterId": "EU-RO-1"}

        def terminate(self):
            pass

        def create_pod(self, payload):
            self.payload = payload
            return {
                "id": "newpod",
                "networkVolumeId": "volume123",
                "costPerHr": 0.44,
            }

    client = Client()
    pod = {
        "desiredStatus": "EXITED",
        "networkVolumeId": "volume123",
        "volumeInGb": 0,
        "imageName": "runpod/image",
        "ports": ["22/tcp"],
    }

    runpod_controller._migrate_capacity_bound_pod(client, pod)

    assert client.payload["gpuTypeIds"] == [
        "NVIDIA RTX A5000",
        "NVIDIA A40",
        "NVIDIA GeForce RTX 4090",
    ]
    assert client.payload["gpuTypePriority"] == "availability"


@pytest.mark.parametrize("maximum", ["NaN", "inf", "-1", "invalid"])
def test_invalid_cost_ceiling_fails_before_provider_mutation(monkeypatch, maximum):
    monkeypatch.setenv("MAS_RUNPOD_GPU_TYPE_ID", "NVIDIA A40")
    monkeypatch.delenv("MAS_RUNPOD_GPU_TYPE_IDS", raising=False)
    monkeypatch.setenv("MAS_RUNPOD_MAX_COST_PER_HR", maximum)
    with pytest.raises(runpod_controller.RunPodControllerError, match="finite and positive"):
        runpod_controller._migrate_capacity_bound_pod(object(), {})


@pytest.mark.parametrize("metadata", [
    {"networkVolumeId": "wrong", "costPerHr": 0.5},
    {"networkVolumeId": "volume123", "costPerHr": "NaN"},
    {"networkVolumeId": "volume123", "costPerHr": "invalid"},
])
def test_replacement_metadata_failure_cleans_up_created_pod(monkeypatch, metadata):
    monkeypatch.setenv("MAS_RUNPOD_NETWORK_VOLUME_ID", "volume123")
    monkeypatch.setenv("MAS_RUNPOD_DATA_CENTER_ID", "EU-RO-1")
    monkeypatch.setenv("MAS_RUNPOD_GPU_TYPE_ID", "NVIDIA A40")
    monkeypatch.setenv("MAS_RUNPOD_MAX_COST_PER_HR", "0.75")
    monkeypatch.delenv("MAS_RUNPOD_GPU_TYPE_IDS", raising=False)
    stopped = []

    class Client:
        pod_id, api_key, timeout, attempts = "old", "secret", 15, 3
        sleep = staticmethod(lambda _: None)
        def get_network_volume(self, _):
            return {"id": "volume123", "dataCenterId": "EU-RO-1"}
        def terminate(self):
            pass
        def create_pod(self, _):
            return {"id": "new", **metadata}

    class Replacement:
        def __init__(self, *args, **kwargs):
            pass
        def terminate(self):
            stopped.append("new")

    monkeypatch.setattr(runpod_controller, "RunPodClient", Replacement)
    with pytest.raises((runpod_controller.RunPodControllerError, ValueError)):
        runpod_controller._migrate_capacity_bound_pod(Client(), {
            "desiredStatus": "EXITED", "networkVolumeId": "volume123",
            "volumeInGb": 0, "imageName": "runpod/image"})
    assert stopped == ["new"]


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


def test_network_retries_share_total_budget(monkeypatch):
    clock = [0.0]
    budgets = []

    def network(*args, **kwargs):
        budgets.append(kwargs["total_timeout"])
        clock[0] += 8
        raise runpod_controller.RunPodControllerError("timeout")

    monkeypatch.setattr(runpod_controller, "_network", network)
    monkeypatch.setattr(runpod_controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runpod_controller.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    with pytest.raises(runpod_controller.RunPodControllerError, match="total retry budget"):
        runpod_controller._network_retry(["ssh", "host"], total_timeout=10)
    assert budgets == [10, 1]


def test_post_run_transfers_share_retrieval_grace(monkeypatch):
    clock = [0.0]
    timeouts = []

    def network(*args, **kwargs):
        timeouts.append(kwargs["total_timeout"])
        clock[0] += min(70, kwargs["total_timeout"])
        return b"ok"

    monkeypatch.setattr(runpod_controller, "_network", network)
    monkeypatch.setattr(runpod_controller.time, "monotonic", lambda: clock[0])
    runpod_controller._network_retry(["scp", "exit-code"], deadline=120)
    runpod_controller._network_retry(["scp", "diagnostic"], deadline=120)
    with pytest.raises(runpod_controller.RunPodControllerError, match="total retry budget"):
        runpod_controller._network_retry(["scp", "next-diagnostic"], deadline=120)
    assert timeouts == [120, 50]


def test_failed_terminal_remote_job_gets_new_attempt_identity(tmp_path):
    request_path = tmp_path / "remote-job-request.json"
    status_path = tmp_path / "remote-job-status.json"
    base = "a" * 64
    request = {"commit": "b" * 40, "episode": 13, "input_sha256": base}
    request_path.write_text(
        json.dumps({"data": request, "sha256": runpod_controller.digest(request)}),
        encoding="utf-8",
    )
    status_path.write_text(json.dumps({"status": "EXITED", "exit_code": 124}), encoding="utf-8")

    input_sha, attempt = runpod_controller._remote_attempt_identity(
        base, request_path, status_path
    )

    assert attempt == 1
    assert input_sha == runpod_controller.digest(
        {"base_input_sha256": base, "attempt": 1}
    )


@pytest.mark.parametrize("exit_code", [0, 20, 21, runpod_controller.WAIT_MP4_SAMPLE])
def test_expected_terminal_remote_job_keeps_input_identity(tmp_path, exit_code):
    request_path = tmp_path / "remote-job-request.json"
    status_path = tmp_path / "remote-job-status.json"
    base = "a" * 64
    request = {"commit": "b" * 40, "episode": 13, "input_sha256": base}
    request_path.write_text(
        json.dumps({"data": request, "sha256": runpod_controller.digest(request)}),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps({"status": "EXITED", "exit_code": exit_code}), encoding="utf-8"
    )

    assert runpod_controller._remote_attempt_identity(
        base, request_path, status_path
    ) == (base, 0)


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("format", "mas-remote-result-collection-failure-0"),
        ("episode", 12),
        ("base_input_sha256", "d" * 64),
        ("input_sha256", "d" * 64),
        ("attempt", 0),
        ("exit_code", 124),
        ("request_sha256", "d" * 64),
        ("relative_path", "work/other.json"),
        ("storage_path", "/tmp/other.json"),
        ("envelope_sha256", "d" * 64),
        ("status_identity", None),
    ],
)
def test_mismatched_collection_failure_marker_fails_closed(tmp_path, field, wrong):
    request_path = tmp_path / "remote-job-request.json"
    status_path = tmp_path / "remote-job-status.json"
    marker_path = tmp_path / "remote-result-collection-failure.json"
    base = "a" * 64
    input_sha = runpod_controller.digest({"base_input_sha256": base, "attempt": 1})
    request = {
        "commit": "b" * 40,
        "episode": 13,
        "input_sha256": input_sha,
        "base_input_sha256": base,
        "attempt": 1,
    }
    request_path.write_text(
        json.dumps({"data": request, "sha256": runpod_controller.digest(request)}),
        encoding="utf-8",
    )
    status_identity = _remote_identity(request)
    if field == "status_identity":
        status_identity["input_sha256"] = "d" * 64
    status_path.write_text(
        json.dumps({"status": "EXITED", "exit_code": runpod_controller.READY_FOR_DELIVERY,
                    "identity": status_identity}),
        encoding="utf-8",
    )
    failure = {
        "format": "mas-remote-result-collection-failure-1",
        "episode": 13,
        "base_input_sha256": base,
        "input_sha256": input_sha,
        "attempt": 1,
        "exit_code": runpod_controller.READY_FOR_DELIVERY,
        "request_sha256": runpod_controller.digest(request),
        "relative_path": "work/delivery-export.json",
        "storage_path": (
            "/workspace/ma-sub/EPISODES/Muhtemel Ask 13.Bolum/"
            "work/delivery-export.json"
        ),
    }
    if field not in {"envelope_sha256", "status_identity"}:
        failure[field] = wrong
    marker_path.write_text(
        json.dumps({"data": failure, "sha256": (
            wrong if field == "envelope_sha256" else runpod_controller.digest(failure)
        )}),
        encoding="utf-8",
    )

    with pytest.raises(runpod_controller.RunPodControllerError, match="mismatch"):
        runpod_controller._remote_attempt_identity(base, request_path, status_path)


def test_missing_collection_failure_marker_fails_closed(tmp_path):
    request_path = tmp_path / "remote-job-request.json"
    status_path = tmp_path / "remote-job-status.json"
    base = "a" * 64
    request = {
        "commit": "b" * 40,
        "episode": 13,
        "input_sha256": base,
        "base_input_sha256": base,
        "attempt": 0,
    }
    request_path.write_text(
        json.dumps({"data": request, "sha256": runpod_controller.digest(request)}),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps({"status": "EXITED", "exit_code": runpod_controller.READY_FOR_DELIVERY,
                    "identity": _remote_identity(request)}),
        encoding="utf-8",
    )

    with pytest.raises(runpod_controller.RunPodControllerError, match="evidence is missing"):
        runpod_controller._remote_attempt_identity(base, request_path, status_path)


def test_delivery_export_failure_records_current_attempt(tmp_path, monkeypatch):
    local_root = tmp_path / "episode"
    work = local_root / "work"
    work.mkdir(parents=True)
    base = "a" * 64
    request = {
        "commit": "b" * 40,
        "episode": 13,
        "input_sha256": base,
        "base_input_sha256": base,
        "attempt": 0,
    }
    request_path = work / "remote-job-request.json"
    status_path = work / "remote-job-status.json"
    request_path.write_text(
        json.dumps({"data": request, "sha256": runpod_controller.digest(request)}),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps({"status": "EXITED", "exit_code": runpod_controller.READY_FOR_DELIVERY,
                    "identity": _remote_identity(request)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runpod_controller,
        "_network_retry",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            runpod_controller.RunPodControllerError("export unavailable")
        ),
    )
    temporary = tmp_path / "transfer"
    temporary.mkdir()

    with pytest.raises(runpod_controller.RunPodControllerError, match="export unavailable"):
        runpod_controller._collect_remote_results(
            runpod_controller.READY_FOR_DELIVERY,
            13,
            local_root,
            "/workspace/ma-sub/EPISODES/Muhtemel Ask 13.Bolum",
            ["ssh"],
            ["scp"],
            "host",
            None,
            temporary,
        )

    marker = json.loads(
        (work / "remote-result-collection-failure.json").read_text(encoding="utf-8")
    )
    assert marker["sha256"] == runpod_controller.digest(marker["data"])
    assert marker["data"]["relative_path"] == "work/delivery-export.json"
    assert marker["data"]["storage_path"] == (
        "/workspace/ma-sub/EPISODES/Muhtemel Ask 13.Bolum/"
        "work/delivery-export.json"
    )
    assert runpod_controller._remote_attempt_identity(
        base, request_path, status_path
    ) == (
        runpod_controller.digest({"base_input_sha256": base, "attempt": 1}),
        1,
    )


@pytest.mark.parametrize("filename", [
    "Muhtemel Ask 13.Bolum.id.bound.mp4",
    "Muhtemel Ask 13.Bolum.id.bound.burn.json",
])
def test_missing_external_delivery_result_advances_remote_identity(
        tmp_path, monkeypatch, filename):
    local_root = tmp_path / "episode"
    work = local_root / "work"
    work.mkdir(parents=True)
    base = "a" * 64
    commit = "b" * 40
    request = {
        "commit": commit,
        "episode": 13,
        "input_sha256": base,
        "base_input_sha256": base,
        "attempt": 0,
    }
    request_path = work / "remote-job-request.json"
    status_path = work / "remote-job-status.json"
    request_path.write_text(
        json.dumps({"data": request, "sha256": runpod_controller.digest(request)}),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps({"status": "EXITED", "exit_code": runpod_controller.READY_FOR_DELIVERY,
                    "identity": _remote_identity(request)}),
        encoding="utf-8",
    )
    record = {
        "relative_path": f"final/{filename}",
        "storage_path": f"/tmp/mas-ep13-output/{filename}",
        "size_bytes": 1,
        "sha256": "c" * 64,
    }
    manifest = {"episode": 13, "mode": "strict", "files": [record]}

    def transfer(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(runpod_controller, "_network_retry", transfer)
    monkeypatch.setattr(
        runpod_controller,
        "_download_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            runpod_controller.RunPodControllerError("No such file")
        ),
    )
    temporary = tmp_path / "transfer"
    temporary.mkdir()

    with pytest.raises(runpod_controller.RunPodControllerError, match="No such file"):
        runpod_controller._collect_remote_results(
            runpod_controller.READY_FOR_DELIVERY,
            13,
            local_root,
            "/workspace/ma-sub/EPISODES/Muhtemel Ask 13.Bolum",
            ["ssh"],
            ["scp"],
            "host",
            None,
            temporary,
        )

    marker = json.loads(
        (work / "remote-result-collection-failure.json").read_text(encoding="utf-8")
    )
    assert marker["sha256"] == runpod_controller.digest(marker["data"])
    assert marker["data"]["relative_path"] == record["relative_path"]
    input_sha, attempt = runpod_controller._remote_attempt_identity(
        base, request_path, status_path
    )
    assert attempt == 1
    assert input_sha == runpod_controller.digest(
        {"base_input_sha256": base, "attempt": 1}
    )
    prior_token = hashlib.sha256(f"13\n{commit}\n{base}\n".encode()).hexdigest()
    next_token = hashlib.sha256(f"13\n{commit}\n{input_sha}\n".encode()).hexdigest()
    assert next_token != prior_token


def test_verified_transfer_cannot_reset_episode_budget(monkeypatch, tmp_path):
    elapsed = [0.0]
    calls = []
    source = tmp_path / "return.zip"
    source.write_bytes(b"returned")

    class Budget:
        def check(self):
            if elapsed[0] >= 10:
                raise BudgetExceeded("episode expired")
            return 10 - elapsed[0]

    def network(command, **kwargs):
        calls.append((command, kwargs["total_timeout"]))
        elapsed[0] += min(4, kwargs["total_timeout"])
        if elapsed[0] >= 10:
            raise runpod_controller.RunPodControllerError("timeout")
        return b""

    monkeypatch.setattr(runpod_controller, "_network", network)
    monkeypatch.setattr(runpod_controller.time, "monotonic", lambda: elapsed[0])
    with pytest.raises(BudgetExceeded, match="episode expired"):
        runpod_controller._upload_episode_file_verified(
            source, "/remote/return.zip", ssh=["ssh"], scp=["scp"], host="host", budget=Budget())
    assert [timeout for _, timeout in calls] == [10, 6, 2]
    assert not any("mv -f" in command[-1] for command, _ in calls)


def test_verified_local_source_uploads_outputs_then_rewritten_marker(monkeypatch, tmp_path):
    local_root = tmp_path / "episode"
    source_dir = local_root / "source"
    source_dir.mkdir(parents=True)
    video = source_dir / "source.mkv"
    metadata = source_dir / "source.metadata.json"
    video.write_bytes(b"video")
    metadata.write_text("{}", encoding="utf-8")
    marker = {
        "stage": "download",
        "outputs": {
            "video": {"path": str(video), "size_bytes": 5, "sha256": "a" * 64},
            "metadata": {"path": str(metadata), "size_bytes": 2, "sha256": "b" * 64},
        },
    }
    (source_dir / "download.done.json").write_text(json.dumps(marker), encoding="utf-8")
    calls = []

    monkeypatch.setattr(runpod_controller, "validate_download", lambda *args, **kwargs: True)

    def upload(source, remote_path, **kwargs):
        calls.append((Path(source), remote_path, kwargs, Path(source).read_bytes()))
        return {"bytes": Path(source).stat().st_size, "sha256": "c" * 64}

    monkeypatch.setattr(runpod_controller, "_upload_episode_file_verified", upload)
    receipts = runpod_controller._upload_verified_local_source(
        local_root,
        "/workspace/episode",
        "https://www.youtube.com/watch?v=episode",
        ssh=["ssh"],
        scp=["scp"],
        host="host",
        temporary=tmp_path,
    )

    assert [call[1] for call in calls] == [
        "/workspace/episode/source/source.mkv",
        "/workspace/episode/source/source.metadata.json",
        "/workspace/episode/source/download.done.json",
    ]
    assert all(call[2]["immutable"] for call in calls)
    assert [call[2].get("transfer_timeout", 300) for call in calls] == [1800, 1800, 300]
    assert [call[2].get("monitor_remote_growth", False) for call in calls] == [True, True, False]
    remote_marker = json.loads(calls[-1][3])
    assert remote_marker["outputs"]["video"]["path"] == "/workspace/episode/source/source.mkv"
    assert remote_marker["outputs"]["metadata"]["path"] == (
        "/workspace/episode/source/source.metadata.json"
    )
    assert receipts["video"]["sha256"] == "c" * 64


def test_verified_local_source_rejects_path_outside_source(monkeypatch, tmp_path):
    local_root = tmp_path / "episode"
    source_dir = local_root / "source"
    source_dir.mkdir(parents=True)
    outside = tmp_path / "outside.mkv"
    outside.write_bytes(b"video")
    (source_dir / "download.done.json").write_text(
        json.dumps({"outputs": {"video": {"path": str(outside)}}}), encoding="utf-8"
    )
    monkeypatch.setattr(runpod_controller, "validate_download", lambda *args, **kwargs: True)

    with pytest.raises(runpod_controller.RunPodControllerError, match="outside"):
        runpod_controller._upload_verified_local_source(
            local_root,
            "/workspace/episode",
            "https://www.youtube.com/watch?v=episode",
            ssh=["ssh"],
            scp=["scp"],
            host="host",
            temporary=tmp_path,
        )


def test_missing_local_source_marker_does_not_touch_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runpod_controller,
        "_upload_episode_file_verified",
        lambda *args, **kwargs: pytest.fail("upload"),
    )

    assert runpod_controller._upload_verified_local_source(
        tmp_path,
        "/workspace/episode",
        "https://www.youtube.com/watch?v=episode",
        ssh=["ssh"],
        scp=["scp"],
        host="host",
        temporary=tmp_path,
    ) is None


def test_remote_growth_probe_reports_only_increasing_size(monkeypatch):
    sizes = iter((b"0\n", b"10\n", b"10\n"))
    monkeypatch.setattr(runpod_controller, "_network_retry", lambda *args, **kwargs: next(sizes))
    probe = runpod_controller._remote_growth_probe(["ssh"], "/remote/partial")

    assert probe() is True
    assert probe() is True
    assert probe() is False


def test_local_growth_probe_reports_only_increasing_size(tmp_path):
    partial = tmp_path / "file.partial"
    probe = runpod_controller._local_growth_probe(partial)

    assert probe() is False
    partial.write_bytes(b"a")
    assert probe() is True
    assert probe() is False
    partial.write_bytes(b"ab")
    assert probe() is True


def test_remote_checkpoint_reuses_matching_local_immutable_source(monkeypatch, tmp_path):
    source = tmp_path / "source" / "episode.mkv"
    source.parent.mkdir()
    source.write_bytes(b"source")
    record = {
        "relative_path": "source/episode.mkv",
        "snapshot_path": "/workspace/episode/source/episode.mkv",
        "size_bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "immutable_source": True,
    }
    monkeypatch.setattr(
        runpod_controller,
        "_network_retry",
        lambda *args, **kwargs: pytest.fail("matching source must not be downloaded"),
    )

    assert runpod_controller._download_record(
        record,
        tmp_path,
        "/workspace/episode",
        ["scp"],
        "host",
        None,
        checkpoint=True,
    ) == source


def test_large_remote_download_has_single_attempt_and_byte_growth_probe(monkeypatch, tmp_path):
    calls = []

    class Budget:
        def check(self):
            return 1800

    def transfer(command, **kwargs):
        calls.append(kwargs)
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_bytes(b"partial")
        return b""

    monkeypatch.setattr(runpod_controller, "_network_retry", transfer)
    record = {
        "relative_path": "final/episode.mp4",
        "size_bytes": 64 * 1024 * 1024,
        "sha256": "a" * 64,
    }

    with pytest.raises(runpod_controller.RunPodControllerError, match="mismatch"):
        runpod_controller._download_record(
            record,
            tmp_path,
            "/workspace/episode",
            ["scp"],
            "host",
            Budget(),
        )

    assert calls[0]["attempts"] == 1
    assert callable(calls[0]["progress_probe"])


def test_provider_request_uses_remaining_episode_time(monkeypatch):
    timeouts = []

    class Budget:
        def check(self):
            return 0.5

    def open_request(request, timeout):
        timeouts.append(timeout)
        return _Response({"desiredStatus": "RUNNING"})

    monkeypatch.setattr(runpod_controller.urllib.request, "urlopen", open_request)
    client = runpod_controller.RunPodClient("pod", "secret", budget=Budget())
    client.get()
    assert timeouts == [0.5]


def test_ssh_probe_is_capped_to_remaining_readiness_time(monkeypatch, tmp_path):
    timeouts = []
    monkeypatch.setattr(runpod_controller.time, "monotonic", lambda: 0)

    def run(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(runpod_controller.subprocess, "run", run)
    runpod_controller._wait_for_ssh(tmp_path / "key", "host", "22", timeout=0.5)
    assert timeouts == [0.5]


def test_invalid_local_return_fails_before_provider_access(monkeypatch, tmp_path):
    monkeypatch.setattr(runpod_controller, "_required_environment", lambda: {})
    monkeypatch.setattr(runpod_controller, "_local_preflight", lambda _: ("a" * 40, tmp_path))
    monkeypatch.setattr(runpod_controller, "_validate_local_tr_return", lambda *_: (_ for _ in ()).throw(
        runpod_controller.RunPodControllerError("stale return")))
    monkeypatch.setattr(runpod_controller, "RunPodClient", lambda *_: pytest.fail("provider access"))
    with pytest.raises(runpod_controller.RunPodControllerError, match="stale return"):
        runpod_controller.run_remote_episode(11)


def test_episode_budget_anchors_existing_logs_across_retries(monkeypatch, tmp_path):
    monkeypatch.delenv("MAS_EPISODE_BUDGET_SECONDS", raising=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    first = datetime.now(timezone.utc) - timedelta(hours=5)
    (logs / "run-original.log").write_text(json.dumps({
        "event": "run_started", "episode": 11, "timestamp": first.isoformat()
    }) + "\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(BudgetExceeded):
            runpod_controller._episode_budget(tmp_path, 11)
    saved = json.loads((tmp_path / "work" / "controller_budget.json").read_text())
    assert saved["data"]["started_at"] == first.isoformat()
    monkeypatch.setenv("MAS_EPISODE_BUDGET_SECONDS", "21600")
    assert runpod_controller._episode_budget(tmp_path, 11).remaining() > 0


def test_episode_budget_checkpoint_tampering_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("MAS_EPISODE_BUDGET_SECONDS", raising=False)
    runpod_controller._episode_budget(tmp_path, 11)
    path = tmp_path / "work" / "controller_budget.json"
    saved = json.loads(path.read_text())
    original_start = saved["data"]["started_at"]
    saved["data"]["started_at"] = (
        datetime.fromisoformat(original_start) + timedelta(seconds=1)
    ).isoformat()
    assert saved["data"]["started_at"] != original_start
    path.write_text(json.dumps(saved))
    with pytest.raises(runpod_controller.RunPodControllerError, match="integrity"):
        runpod_controller._episode_budget(tmp_path, 11)


def test_network_capture_returns_readback_without_streaming(monkeypatch):
    captured = {}

    def watchdog(command, **kwargs):
        captured.update(kwargs)
        return b"readback"

    monkeypatch.setattr(runpod_controller, "_run_watchdog", watchdog)

    assert runpod_controller._network(["ssh", "host"], capture=True) == b"readback"
    assert captured["stdout_handler"] is None
    assert captured["stderr_handler"] is None


def test_audio_review_overrides_upload_requires_partial_and_final_readback(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "Muhtemel Ask 11.Bolum"
    source = local_root / "review" / "audio_review_overrides.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"override":true}\n')
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []

    def network_retry(command, **kwargs):
        calls.append((command, kwargs))
        remote_command = command[-1]
        if command[0] == "ssh" and remote_command.startswith("set -euo pipefail; stat"):
            remote_path = remote_command.split("sha256sum -- ", 1)[1].strip("'")
            return f"18\n{expected_sha256}  {remote_path}\n".encode()
        return b""

    monkeypatch.setattr(runpod_controller, "_network_retry", network_retry)
    receipt = runpod_controller._upload_audio_review_overrides(
        local_root,
        "/workspace/ma-sub/EPISODES/Muhtemel Ask 11.Bolum",
        ssh=["ssh", "root@host"],
        scp=["scp"],
        host="host",
    )

    assert receipt == {"bytes": 18, "sha256": expected_sha256}
    assert len(calls) == 5
    assert calls[1][0][0] == "scp"
    assert calls[1][0][1] == str(source)
    assert calls[1][0][2].startswith(
        "root@host:/workspace/.mas-upload/audio-review-overrides-"
    )
    assert "mv -f --" in calls[3][0][-1]
    assert (
        "'/workspace/ma-sub/EPISODES/Muhtemel Ask 11.Bolum/"
        "review/audio_review_overrides.json'"
    ) in calls[3][0][-1]
    assert all(
        kwargs
        == {
            "idle_timeout": 60,
            "total_timeout": 300,
            "budget": None,
            **({"capture": True} if _command[-1].startswith("set -euo pipefail; stat") else {}),
        }
        for _command, kwargs in calls
    )


def test_audio_review_overrides_upload_fails_before_install_on_readback_mismatch(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "11"
    source = local_root / "review" / "audio_review_overrides.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    calls = []

    def network_retry(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh" and command[-1].startswith("set -euo pipefail; stat"):
            return b"1\n" + b"0" * 64 + b"  /remote/partial\n"
        return b""

    monkeypatch.setattr(runpod_controller, "_network_retry", network_retry)
    with pytest.raises(
        runpod_controller.RunPodControllerError,
        match="uploaded episode file byte/SHA-256 readback mismatch",
    ):
        runpod_controller._upload_audio_review_overrides(
            local_root,
            "/workspace/ma-sub/EPISODES/Muhtemel Ask 11.Bolum",
            ssh=["ssh", "root@host"],
            scp=["scp"],
            host="host",
        )

    assert len(calls) == 3
    assert not any("mv -f --" in command[-1] for command in calls)


def test_audio_review_overrides_upload_rejects_final_readback_mismatch(
    monkeypatch, tmp_path
):
    local_root = tmp_path / "11"
    source = local_root / "review" / "audio_review_overrides.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    signature_reads = 0

    def network_retry(command, **kwargs):
        nonlocal signature_reads
        if command[0] == "ssh" and command[-1].startswith("set -euo pipefail; stat"):
            signature_reads += 1
            sha256 = expected_sha256 if signature_reads == 1 else "0" * 64
            return f"2\n{sha256}  /remote/file\n".encode()
        return b""

    monkeypatch.setattr(runpod_controller, "_network_retry", network_retry)
    with pytest.raises(
        runpod_controller.RunPodControllerError,
        match="installed episode file byte/SHA-256 readback mismatch",
    ):
        runpod_controller._upload_audio_review_overrides(
            local_root,
            "/workspace/ma-sub/EPISODES/Muhtemel Ask 11.Bolum",
            ssh=["ssh", "root@host"],
            scp=["scp"],
            host="host",
        )

    assert signature_reads == 2


def test_missing_audio_review_overrides_does_not_touch_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runpod_controller,
        "_upload_episode_file_verified",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("upload")),
    )

    assert runpod_controller._upload_audio_review_overrides(
        tmp_path,
        "/workspace/ma-sub/EPISODES/Muhtemel Ask 11.Bolum",
        ssh=["ssh", "root@host"],
        scp=["scp"],
        host="host",
    ) is None


def test_speaker_evidence_uses_verified_episode_upload(monkeypatch, tmp_path):
    source = tmp_path / "review" / "speaker_evidence_v1.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        runpod_controller,
        "_upload_episode_file_verified",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"bytes": 2, "sha256": "a" * 64},
    )

    receipt = runpod_controller._upload_speaker_evidence(
        tmp_path,
        "/remote/episode",
        ssh=["ssh"],
        scp=["scp"],
        host="host",
    )

    assert receipt["bytes"] == 2
    assert calls[0][0][0] == source
    assert calls[0][0][1] == "/remote/episode/review/speaker_evidence_v1.json"


def test_ssh_has_bounded_liveness_options(tmp_path):
    command = runpod_controller._ssh_args(tmp_path / "key", "192.0.2.4", "10022")
    assert command[1:3] == ["-n", "-T"]
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


def test_runtime_environment_is_shell_quoted_and_does_not_log_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAS_ALLOW_C0E3_VENV_ADOPTION", raising=False)
    values = {
        "RUNPOD_POD_ID": "pod123",
        "RUNPOD_API_KEY": "api secret",
        "MAS_GMAIL_ADDRESS": "from@example.com",
        "MAS_GMAIL_APP_PASSWORD": "mail secret",
        "MAS_NOTIFY_TO": "to@example.com",
        "MAS_DRIVE_STRICT_REMOTE": "gdrive:path with spaces",
        "MAS_ALLOW_C0E3_VENV_ADOPTION": "1",
    }
    target = tmp_path / "runtime.env"
    runpod_controller._write_runtime_env(target, values, "a" * 40)
    content = target.read_text(encoding="utf-8")
    assert "export RUNPOD_API_KEY='api secret'" in content
    assert "MAS_DRIVE_STRICT_REMOTE" not in content
    assert "RCLONE_CONFIG" not in content
    assert "gdrive:path with spaces" not in content
    assert "MAS_GIT_COMMIT=" + "a" * 40 in content
    assert "MAS_YTDLP_COOKIES" not in content
    assert "UV_CACHE_DIR=/workspace/.cache/uv" in content
    assert "UV_HTTP_TIMEOUT=120" in content
    assert "UV_HTTP_RETRIES=3" in content
    assert "UV_CONCURRENT_DOWNLOADS=4" in content
    assert "HF_HOME=/workspace/.cache/huggingface" in content
    assert "MAS_ALLOW_C0E3_VENV_ADOPTION" not in content
    assert b"\r" not in target.read_bytes()

    values["MAS_YTDLP_COOKIES"] = "C:/private/cookies.txt"
    runpod_controller._write_runtime_env(target, values, "a" * 40)
    content = target.read_text(encoding="utf-8")
    assert "MAS_YTDLP_COOKIES=/workspace/.mas-secrets/youtube-cookies.txt" in content
    assert "C:/private/cookies.txt" not in content

    monkeypatch.setenv("MAS_ALLOW_C0E3_VENV_ADOPTION", "1")
    runpod_controller._write_runtime_env(target, values, "a" * 40)
    assert "export MAS_ALLOW_C0E3_VENV_ADOPTION=1" in target.read_text(
        encoding="utf-8"
    )


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


def test_local_preflight_rejects_malformed_cookie_before_compute(monkeypatch, tmp_path):
    key, cookie = tmp_path / "key", tmp_path / "cookie"
    key.write_text("key")
    cookie.write_text("not a Netscape cookie file")
    monkeypatch.setattr(runpod_controller.shutil, "which", lambda name: name)
    monkeypatch.setattr(runpod_controller.subprocess, "run", lambda *a, **kw:
                        type("Result", (), {"returncode": 0, "stdout": ""})())
    with pytest.raises(RuntimeError, match="Netscape"):
        runpod_controller._local_preflight({"MAS_RUNPOD_SSH_KEY": str(key),
                                            "MAS_YTDLP_COOKIES": str(cookie)})


def test_local_preflight_allows_public_source_without_cookie(monkeypatch, tmp_path):
    key = tmp_path / "key"
    config = tmp_path / "rclone.conf"
    key.write_text("key")
    config.write_text("config")
    monkeypatch.setattr(runpod_controller.shutil, "which", lambda name: name)
    def run(command, **kwargs):
        output = "" if command[1] == "status" else "a" * 40
        return type("Result", (), {"returncode": 0, "stdout": output})()
    monkeypatch.setattr(runpod_controller.subprocess, "run", run)
    monkeypatch.setattr(runpod_controller, "_rclone_config", lambda: config)

    commit, observed_config = runpod_controller._local_preflight(
        {"MAS_RUNPOD_SSH_KEY": str(key)}
    )

    assert commit == "a" * 40
    assert observed_config == config


@pytest.mark.parametrize("failure", [RuntimeError("response lost"), TimeoutError("work expired"),
                                     KeyboardInterrupt()])
def test_session_failure_releases_owned_lease(monkeypatch, tmp_path, failure):
    key = tmp_path / "key"
    cookie = tmp_path / "cookie"
    config = tmp_path / "rclone.conf"
    for path in (key, cookie, config):
        path.write_text("value", encoding="utf-8")
    key.with_suffix(".pub").write_text("ssh-ed25519 test", encoding="utf-8")
    quote_path = tmp_path / "storage-quote.json"
    quote = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
             "source_url": "https://docs.runpod.io/pods/storage/types",
             "network_volume_usd_per_gb_month": 0.07,
             "container_storage_usd_per_gb_month": 0.10}
    quote_path.write_text(json.dumps({"data": quote,
                                      "sha256": runpod_controller.digest(quote)}), encoding="utf-8")
    monkeypatch.setenv("MAS_RUNPOD_STORAGE_QUOTE", str(quote_path))
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

    class Provider:
        def __init__(self, *_):
            pass

        def get_pod(self, *_):
            return {"imageName": "test-image"}

        def get_volume(self, *_):
            return {"size": 50}

    events = []

    class Lease:
        pod = {"id": "owned-new"}

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            events.append("acquired")
            return self

        def __exit__(self, *_):
            events.append("externally_absent")

    monkeypatch.setattr(runpod_controller, "CapacityProvider", Provider)
    monkeypatch.setattr(runpod_controller, "CapacityLease", Lease)
    monkeypatch.setattr(runpod_controller, "ROOT", tmp_path)
    monkeypatch.setattr(runpod_controller, "_required_environment", lambda: values)
    monkeypatch.setattr(runpod_controller, "_local_preflight", lambda values: ("a" * 40, config))
    monkeypatch.setattr(runpod_controller, "_validate_local_tr_return", lambda *_: None)
    monkeypatch.setattr(runpod_controller, "episode_dir", lambda _: tmp_path)
    monkeypatch.setattr(runpod_controller, "_prepare_official_source", lambda *_: "https://example.com")
    monkeypatch.setattr(runpod_controller, "_run_remote_session", lambda *_a, **_k: (_ for _ in ()).throw(failure))
    with pytest.raises(type(failure)):
        runpod_controller.run_remote_episode(11)
    assert events == ["acquired", "externally_absent"]
