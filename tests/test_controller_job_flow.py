import json
import hashlib
import os
import re

import pytest

from mas import runpod_controller as controller
from mas.delivery import READY_FOR_DELIVERY
from mas.reliability import atomic_json, digest


class Budget:
    def check(self):
        return 600


def _remote_response(command):
    text = command[-1]
    identity = None
    if "--input-sha256" in text:
        episode = int(re.search(r"--episode (\d+)", text)[1])
        commit = re.search(r"--commit ([0-9a-f]{40})", text)[1]
        input_sha = re.search(r"--input-sha256 ([0-9a-f]{64})", text)[1]
        token = hashlib.sha256(f"{episode}\n{commit}\n{input_sha}\n".encode()).hexdigest()
        identity = {"episode": episode, "commit": commit,
                    "input_sha256": input_sha, "token": token}
    if " poll " in text:
        return json.dumps({
            "status": {"status": "EXITED", "exit_code": 0, "identity": identity},
            "log": {"text": "", "next_offset": 0, "eof": True},
            "checkpoints": {"files": [], "identity": identity},
        })
    if " logs " in text:
        return json.dumps({"text": "", "next_offset": 0})
    if " checkpoints " in text:
        return json.dumps({"files": [], "identity": identity})
    if " start " in text:
        return "{}"
    raise AssertionError(text)


def test_lost_start_response_retries_idempotent_start_then_only_monitors(monkeypatch, tmp_path):
    real_retry = controller._network_retry
    starts = []
    def lost(command, **kwargs):
        starts.append(command[-1])
        if len(starts) == 1:
            raise controller.RunPodControllerError("response lost")
        return "{}"
    monkeypatch.setattr(controller, "_network", lost)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    def response(command, **kwargs):
        if " start " in command[-1]:
            return real_retry(command, **kwargs)
        return _remote_response(command)
    monkeypatch.setattr(controller, "_network_retry", response)
    result = controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget())
    assert result == 0
    assert len(starts) == 2
    assert " start --root " in starts[0]
    assert " --recover-lost --source-url " in starts[0]
    assert "; exec env PYTHONPATH=src " in starts[0]


def test_alignment_recovery_uses_the_extended_bounded_poll_watchdog(monkeypatch, tmp_path):
    calls = []
    def response(command, **kwargs):
        calls.append((command[-1], kwargs))
        return _remote_response(command)
    monkeypatch.setattr(controller, "_network_retry", response)
    assert controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget(), alignment_recovery=True) == 0
    poll = next(kwargs for command, kwargs in calls if " mas.remote_job poll " in command)
    assert poll["idle_timeout"] == 180
    assert poll["total_timeout"] == 240


def test_alignment_recovery_never_downloads_live_checkpoints(monkeypatch, tmp_path):
    def response(command, **kwargs):
        payload = json.loads(_remote_response(command))
        if " poll " in command[-1]:
            payload["checkpoints"]["files"] = [{
                "relative_path": "work/raw_asr_v2.json",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }]
        return json.dumps(payload)

    monkeypatch.setattr(controller, "_network_retry", response)
    monkeypatch.setattr(controller, "_download_record",
                        lambda *args, **kwargs: pytest.fail("alignment recovery downloaded a live checkpoint"))
    assert controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget(), alignment_recovery=True) == 0


def test_resume_only_never_sends_start(monkeypatch, tmp_path):
    request = {"commit": "a" * 40, "episode": 13, "input_sha256": "b" * 64}
    atomic_json(tmp_path / "work" / "remote-job-request.json",
                {"data": request, "sha256": digest(request)})
    monkeypatch.setattr(controller, "_network",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("start called")))
    calls = []
    def response(command, **kwargs):
        calls.append(command[-1])
        return _remote_response(command)
    monkeypatch.setattr(controller, "_network_retry", response)
    assert controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget(), resume_only=True) == 0
    assert calls
    assert all("mas.remote_job start " not in command for command in calls)
    assert len(calls) == 1
    assert "mas.remote_job poll " in calls[0]
    assert " --max-bytes 65536 --deadline-seconds 60" in calls[0]


def test_stale_remote_identity_status_is_rejected_without_relaunch(monkeypatch, tmp_path):
    request = {"commit": "a" * 40, "episode": 13, "input_sha256": "b" * 64}
    atomic_json(tmp_path / "work" / "remote-job-request.json",
                {"data": request, "sha256": digest(request)})
    calls = []
    def response(command, **kwargs):
        calls.append(command[-1])
        if " poll " in command[-1]:
            payload = json.loads(_remote_response(command))
            payload["status"] = {"status": "RUNNING_OTHER", "exit_code": None}
            return json.dumps(payload)
        return _remote_response(command)
    monkeypatch.setattr(controller, "_network_retry", response)
    with pytest.raises(controller.RunPodControllerError, match="identity mismatch"):
        controller._monitor_remote_job(
            ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
            "https://example.invalid/episode", 500, Budget(), resume_only=True)
    assert all("mas.remote_job start " not in command for command in calls)


def test_poll_rejects_forged_checkpoint_identity(monkeypatch, tmp_path):
    request = {"commit": "a" * 40, "episode": 13, "input_sha256": "b" * 64}
    atomic_json(tmp_path / "work" / "remote-job-request.json",
                {"data": request, "sha256": digest(request)})
    def response(command, **kwargs):
        payload = json.loads(_remote_response(command))
        payload["checkpoints"]["identity"] = {"token": "forged"}
        return json.dumps(payload)
    monkeypatch.setattr(controller, "_network_retry", response)
    with pytest.raises(controller.RunPodControllerError, match="checkpoint identity mismatch"):
        controller._monitor_remote_job(
            ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
            "https://example.invalid/episode", 500, Budget(), resume_only=True)


def test_terminal_poll_drains_bounded_log_before_return(monkeypatch, tmp_path, capsys):
    request = {"commit": "a" * 40, "episode": 13, "input_sha256": "b" * 64}
    atomic_json(tmp_path / "work" / "remote-job-request.json",
                {"data": request, "sha256": digest(request)})
    calls = []
    def response(command, **kwargs):
        calls.append(command[-1])
        payload = json.loads(_remote_response(command))
        offset = int(re.search(r"--offset (\d+)", command[-1])[1])
        payload["log"] = ({"text": "x" * 65536, "next_offset": 65536, "eof": False}
                          if offset == 0 else
                          {"text": "tail", "next_offset": 65540, "eof": True})
        return json.dumps(payload)
    monkeypatch.setattr(controller, "_network_retry", response)

    assert controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget(), resume_only=True) == 0

    assert len(calls) == 2
    assert "--offset 0 --max-bytes 65536" in calls[0]
    assert "--offset 65536 --max-bytes 65536" in calls[1]
    assert capsys.readouterr().out.endswith("tail")


def _patch_local_preflight(monkeypatch, tmp_path):
    monkeypatch.setenv("MAS_RUNPOD_IMAGE", "registry/test@sha256:" + "a" * 64)
    monkeypatch.setattr(controller, "drive_preflight", lambda *args, **kwargs: {"status": "test"})
    monkeypatch.setattr(controller, "_local_encoder_preflight", lambda *args: None)
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    key = tmp_path / "id_ed25519"
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAATEST", encoding="utf-8")
    values = {"RUNPOD_POD_ID": "781ct55zv4gkle", "RUNPOD_API_KEY": "secret",
              "MAS_RUNPOD_SSH_KEY": str(key), "MAS_YTDLP_COOKIES": str(tmp_path / "cookies"),
              "MAS_DRIVE_STRICT_REMOTE": "drive:folder"}
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "episode_dir", lambda episode: episode_root)
    monkeypatch.setattr(controller, "_required_environment", lambda: values)
    monkeypatch.setattr(controller, "_local_preflight", lambda values: ("a" * 40, tmp_path / "rclone"))
    monkeypatch.setattr(controller, "_validate_local_tr_return", lambda *args: None)
    monkeypatch.setattr(controller, "preflight_local_id_return", lambda *args: None)
    return episode_root, values


def test_source_failure_occurs_before_provider_construction(monkeypatch, tmp_path):
    _patch_local_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "_prepare_official_source",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("source absent")))
    monkeypatch.setattr(controller, "CapacityProvider",
                        lambda *args: (_ for _ in ()).throw(AssertionError("provider constructed")))
    with pytest.raises(RuntimeError, match="source absent"):
        controller.run_remote_episode(13)


def test_missing_storage_quote_fails_before_provider_construction(monkeypatch, tmp_path):
    _patch_local_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "_prepare_official_source",
                        lambda *args: "https://example.invalid/episode")
    monkeypatch.setattr(controller, "_episode_budget", lambda *args: Budget())
    monkeypatch.delenv("MAS_RUNPOD_STORAGE_QUOTE", raising=False)
    monkeypatch.setattr(controller, "CapacityProvider",
                        lambda *args: (_ for _ in ()).throw(AssertionError("provider constructed")))
    with pytest.raises(controller.RunPodControllerError, match="MAS_RUNPOD_STORAGE_QUOTE"):
        controller.run_remote_episode(13)


def test_drive_readiness_failure_occurs_before_paid_acquisition(monkeypatch, tmp_path):
    _patch_local_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "_prepare_official_source", lambda *_: "https://example.invalid/episode")
    monkeypatch.setattr(controller, "_episode_budget", lambda *_: Budget())
    monkeypatch.setenv("MAS_RUNPOD_STORAGE_QUOTE", str(tmp_path / "storage.json"))
    monkeypatch.setattr(controller, "load_storage_quote", lambda _: {})
    calls = []
    def preflight(remote, **kwargs):
        calls.append(kwargs)
        raise controller.RemoteVerificationError("private OAuth unavailable")
    monkeypatch.setattr(controller, "drive_preflight", preflight)
    monkeypatch.setattr(controller, "CapacityProvider", lambda *_: pytest.fail("provider acquired before Drive readiness"))
    with pytest.raises(controller.RemoteVerificationError, match="OAuth"):
        controller.run_remote_episode(13)
    assert calls == [{"required_bytes": 0, "config_path": tmp_path / "rclone", "total_timeout": 60}]


def test_local_qsv_failure_occurs_before_provider_or_drive(monkeypatch, tmp_path):
    _patch_local_preflight(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "_prepare_official_source", lambda *_: "https://example.invalid/episode")
    monkeypatch.setattr(controller, "_episode_budget", lambda *_: Budget())
    monkeypatch.setenv("MAS_RUNPOD_STORAGE_QUOTE", str(tmp_path / "storage.json"))
    monkeypatch.setattr(controller, "load_storage_quote", lambda _: {})
    def fail(*args):
        raise ValueError("QSV encoder unavailable")
    monkeypatch.setattr(controller, "_local_encoder_preflight", fail)
    monkeypatch.setattr(controller, "drive_preflight", lambda *a, **k: pytest.fail("Drive queried"))
    monkeypatch.setattr(controller, "CapacityProvider", lambda *_: pytest.fail("provider acquired"))
    with pytest.raises(ValueError, match="QSV encoder unavailable"):
        controller.run_remote_episode(13)


def test_controller_lock_rejects_duplicate_owner(tmp_path):
    lock = tmp_path / "controller.lock"
    with controller._controller_lock(lock):
        with pytest.raises(controller.RunPodControllerError, match="another production controller"):
            with controller._controller_lock(lock):
                pass


def test_capacity_cleanup_finishes_before_publish(monkeypatch, tmp_path):
    episode_root, _ = _patch_local_preflight(monkeypatch, tmp_path)
    events = []
    plan_args = {}
    lease_payload = {}
    monkeypatch.setenv("MAS_DRIVE_STRICT_REMOTE", "drive:folder")
    monkeypatch.setenv("MAS_RUNPOD_REGISTRY_AUTH_ID", "registry-auth-123")
    monkeypatch.setattr(controller, "_prepare_official_source",
                        lambda *args: "https://example.invalid/episode")
    monkeypatch.setattr(controller, "_episode_budget", lambda *args: Budget())
    monkeypatch.setattr(controller, "load_storage_quote", lambda path: {"quote": "test"}, raising=False)
    monkeypatch.setenv("MAS_RUNPOD_STORAGE_QUOTE", str(tmp_path / "storage.json"))
    class Provider:
        def __init__(self, *args): pass
        def get_pod(self, pod_id, timeout):
            return {"id": pod_id, "imageName": "image", "containerDiskInGb": 30}
        def get_volume(self, volume_id, timeout):
            return {"id": volume_id, "size": 50, "dataCenterId": "EU-RO-1"}
    class Lease:
        pod = {"id": "owned"}
        def __init__(self, *args, **kwargs):
            self.audit = args[2]
            lease_payload.update(args[1])
        def __enter__(self):
            events.append("capacity-enter")
            return self
        def __exit__(self, *args):
            self.audit.mkdir(parents=True, exist_ok=True)
            shutdown_result = [{"pod_id": "owned", "status": "ABSENT", "error_type": None}]
            state = {
                "format": "mas-capacity-lease-state-1",
                "episode": 13,
                "owned_pod_ids": ["owned"],
                "shutdown": shutdown_result,
                "status": "RELEASED",
            }
            atomic_json(self.audit / "capacity-state.json",
                        {"data": state, "sha256": digest(state)})
            shutdown = {"state_sha256": digest(state), "owned_pods": shutdown_result}
            atomic_json(self.audit / "capacity-shutdown.json",
                        {"data": shutdown, "sha256": digest(shutdown)})
            events.append("capacity-exit")
        def remaining_work_seconds(self): return 600
    monkeypatch.setattr(controller, "CapacityProvider", Provider)
    monkeypatch.setenv("MAS_RUNPOD_GPU_TYPE_IDS", "NVIDIA L4|NVIDIA RTX PRO 4000 Blackwell")
    monkeypatch.setattr(
        controller,
        "CapacityPlan",
        lambda **kwargs: plan_args.update(kwargs) or object(),
    )
    monkeypatch.setattr(controller, "CapacityLease", Lease)
    monkeypatch.setattr(controller, "_run_remote_session",
                        lambda *args, **kwargs: READY_FOR_DELIVERY)
    monkeypatch.setattr(controller, "validate_delivery", lambda *args: events.append("validate"))
    monkeypatch.setattr(controller, "sha256_file", lambda path: "b" * 64)
    def publish(*args, **kwargs):
        events.append("publish" if "publish" not in events else "publish-retry")
        if events[-1] == "publish":
            raise RuntimeError("Drive unavailable")
        return 0
    monkeypatch.setattr(controller, "publish_local_delivery", publish)
    with pytest.raises(RuntimeError, match="Drive unavailable"):
        controller.run_remote_episode(13)
    assert events == ["capacity-enter", "capacity-exit", "validate", "publish"]
    assert plan_args["gpu_type_ids"] == ["NVIDIA L4", "NVIDIA RTX PRO 4000 Blackwell"]
    assert plan_args["startup_seconds"] == 900
    assert lease_payload["containerRegistryAuthId"] == "registry-auth-123"
    release_path = episode_root / "work" / "gpu-released-for-delivery.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert release["sha256"] == digest(release["data"])
    assert release["data"]["capacity_state"]["relative_path"].endswith("capacity-state.json")
    assert release["data"]["capacity_shutdown"]["relative_path"].endswith("capacity-shutdown.json")

    monkeypatch.setattr(controller, "_required_environment",
                        lambda: pytest.fail("transfer retry required GPU environment"))
    monkeypatch.setattr(controller, "CapacityProvider",
                        lambda *args: pytest.fail("transfer retry acquired GPU"))
    assert controller.run_remote_episode(13) == 0
    assert events == ["capacity-enter", "capacity-exit", "validate", "publish", "publish-retry"]


@pytest.mark.parametrize(("change_mtime", "expected_downloads"), [(False, 1), (True, 2)])
def test_checkpoint_cache_reuses_verified_stat_until_file_changes(
        monkeypatch, tmp_path, change_mtime, expected_downloads):
    statuses = iter([("RUNNING", None), ("EXITED", 0)])
    record = {"relative_path": "source/video.mp4", "sha256": "c" * 64,
              "size_bytes": 1, "snapshot_path": "/workspace/snapshot"}
    def response(command, **kwargs):
        payload = json.loads(_remote_response(command))
        if " poll " in command[-1]:
            payload["status"]["status"], payload["status"]["exit_code"] = next(statuses)
            payload["checkpoints"]["files"] = [record]
        return json.dumps(payload)
    downloads = []
    checkpoint = tmp_path / "cached.mp4"
    def download(*args, **kwargs):
        downloads.append(1)
        checkpoint.write_bytes(b"x")
        return checkpoint
    def between_polls(seconds):
        if change_mtime:
            stat = checkpoint.stat()
            os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    monkeypatch.setattr(controller, "_network", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(controller, "_network_retry", response)
    monkeypatch.setattr(controller, "_download_record", download)
    monkeypatch.setattr(controller.time, "sleep", between_polls)
    assert controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget()) == 0
    assert len(downloads) == expected_downloads


def test_remote_monitor_logs_only_status_transitions(monkeypatch, tmp_path, capsys):
    statuses = iter([("RUNNING", None), ("RUNNING", None), ("EXITED", 0)])

    def response(command, **kwargs):
        payload = json.loads(_remote_response(command))
        if " poll " in command[-1]:
            payload["status"]["status"], payload["status"]["exit_code"] = next(statuses)
        return json.dumps(payload)

    monkeypatch.setattr(controller, "_network_retry", response)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    assert controller._monitor_remote_job(
        ["ssh"], ["scp"], "host", 13, "a" * 40, tmp_path,
        "https://example.invalid/episode", 500, Budget()) == 0

    output = capsys.readouterr().out
    assert output.count("remote job status=RUNNING") == 1
    assert output.count("remote job status=EXITED") == 1
