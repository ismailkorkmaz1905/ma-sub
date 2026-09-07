import json
import os
import sys
import time

import pytest

from mas.remote_job import RemoteJobError, checkpoint_manifest, read_log, start_job, status_job


COMMIT = "a" * 40


def _race_start(root, gate, queue):
    gate.wait()
    try:
        result = start_job(root, 13, COMMIT,
                           [sys.executable, "-c", "import time; time.sleep(.5)"])
        queue.put(("ok", result["started"]))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _wait(root, expected, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = status_job(root, 13, COMMIT)
        if value["status"] == expected:
            return value
        time.sleep(0.02)
    raise AssertionError(status_job(root, 13, COMMIT))


@pytest.mark.skipif(os.name != "posix", reason="remote detached jobs require POSIX")
def test_detached_job_starts_once_and_records_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str((__import__("pathlib").Path(__file__).parents[1] / "src").resolve()))
    command = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(.3)"]
    first = start_job(tmp_path, 13, COMMIT, command)
    assert first["started"] is True
    _wait(tmp_path, "RUNNING")
    second = start_job(tmp_path, 13, COMMIT, command)
    assert second["started"] is False
    assert second["already_running"] is True
    result = _wait(tmp_path, "EXITED")
    assert result["exit_code"] == 0
    assert "started" in __import__("pathlib").Path(first["log_path"]).read_text()


@pytest.mark.skipif(os.name != "posix", reason="remote detached jobs require POSIX")
def test_concurrent_starts_create_one_worker(tmp_path, monkeypatch):
    import multiprocessing
    monkeypatch.setenv("PYTHONPATH", str((__import__("pathlib").Path(__file__).parents[1] / "src").resolve()))
    context = multiprocessing.get_context("fork")
    gate, queue = context.Event(), context.Queue()
    processes = [context.Process(target=_race_start, args=(str(tmp_path), gate, queue)) for _ in range(2)]
    for process in processes:
        process.start()
    gate.set()
    results = [queue.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
    assert sorted(results) == [("ok", False), ("ok", True)]
    _wait(tmp_path, "EXITED")


@pytest.mark.skipif(os.name != "posix", reason="remote detached jobs require POSIX")
def test_termination_kills_stubborn_setsid_descendant(tmp_path):
    from pathlib import Path
    import subprocess
    from mas import remote_job
    child_pid = tmp_path / "child.pid"
    child = ("import os,signal,time; from pathlib import Path; "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)")
    parent = ("import subprocess,sys,time; "
              "subprocess.Popen([sys.executable,'-c',sys.argv[1]],start_new_session=True); "
              "time.sleep(60)")
    worker = subprocess.Popen([sys.executable, "-c", parent, child], start_new_session=True)
    deadline = time.monotonic() + 5
    while not child_pid.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid.is_file()
    pid = int(child_pid.read_text())
    ticks = remote_job._proc_start_ticks(pid)
    remote_job._terminate_worker_tree(worker)
    deadline = time.monotonic() + 2
    while remote_job._pid_matches(pid, start_ticks=ticks) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not remote_job._pid_matches(pid, start_ticks=ticks)


@pytest.mark.skipif(os.name != "posix", reason="remote detached jobs require POSIX")
def test_live_other_identity_is_rejected(tmp_path, monkeypatch):
    from mas import remote_job
    requested = remote_job._identity(13, COMMIT)
    paths = remote_job._paths(tmp_path, 13, requested["token"])
    paths["job"].mkdir(parents=True)
    other = remote_job._identity(13, "b" * 40)
    paths["claim"].write_text(json.dumps({"identity": other, "supervisor": {"pid": 123}}), encoding="utf-8")
    monkeypatch.setattr(remote_job, "_pid_matches", lambda *args, **kwargs: True)
    with pytest.raises(RemoteJobError, match="another remote job"):
        start_job(tmp_path, 13, COMMIT, ["true"])


def test_stale_running_state_is_not_reported_live(tmp_path):
    from mas import remote_job
    identity = remote_job._identity(13, COMMIT)
    paths = remote_job._paths(tmp_path, 13, identity["token"])
    paths["job"].mkdir(parents=True)
    paths["claim"].write_text(json.dumps({"identity": identity, "supervisor_pid": 99999999}), encoding="utf-8")
    paths["state"].write_text(json.dumps({"identity": identity, "status": "RUNNING"}), encoding="utf-8")
    assert status_job(tmp_path, 13, COMMIT)["status"] == "LOST"


@pytest.mark.skipif(os.name != "posix", reason="remote detached jobs require POSIX")
def test_terminal_identity_is_not_restarted(tmp_path, monkeypatch):
    from mas import remote_job
    identity = remote_job._identity(13, COMMIT, "c" * 64)
    paths = remote_job._paths(tmp_path, 13, identity["token"])
    paths["job"].mkdir(parents=True)
    paths["state"].write_text(json.dumps({"identity": identity, "status": "EXITED", "exit_code": 20}),
                              encoding="utf-8")
    monkeypatch.setattr(remote_job.subprocess, "Popen", lambda *a, **kw: pytest.fail("restarted"))
    result = start_job(tmp_path, 13, COMMIT, ["true"], "c" * 64)
    assert result["terminal"] is True
    assert result["exit_code"] == 20


def test_checkpoint_manifest_hashes_only_asr_checkpoint_layout(tmp_path):
    prepare = tmp_path / "EPISODES/Muhtemel Ask 13.Bolum/prepare"
    primary = prepare / "primary_asr"
    primary.mkdir(parents=True)
    (primary / "one.json").write_bytes(b"one")
    (prepare / "raw_asr_v2.recovery.json").write_bytes(b"recovery")
    (prepare / "raw_asr_v2.json").write_bytes(b"final")
    (prepare / "unrelated.json").write_bytes(b"skip")
    result = checkpoint_manifest(tmp_path, 13, COMMIT)
    assert [item["relative_path"] for item in result["files"]] == [
        "prepare/primary_asr/one.json",
        "prepare/raw_asr_v2.json",
        "prepare/raw_asr_v2.recovery.json",
    ]
    assert all(item["size_bytes"] > 0 and len(item["sha256"]) == 64 for item in result["files"])
    assert all((tmp_path / item["snapshot_path"]).read_bytes() for item in result["files"])


def test_checkpoint_manifest_rejects_symlink_outside_episode(tmp_path):
    from pathlib import Path
    prepare = tmp_path / "EPISODES/Muhtemel Ask 13.Bolum/prepare/primary_asr"
    prepare.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        (prepare / "escaped.json").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks unavailable")
    with pytest.raises(RemoteJobError, match="escaped the episode root"):
        checkpoint_manifest(tmp_path, 13, COMMIT)


def test_log_reads_bounded_chunks(tmp_path):
    from mas import remote_job
    identity = remote_job._identity(13, COMMIT)
    path = remote_job._paths(tmp_path, 13, identity["token"])["log"]
    path.parent.mkdir(parents=True)
    path.write_text("abcdef", encoding="utf-8")
    first = read_log(tmp_path, 13, COMMIT, 0, 4)
    second = read_log(tmp_path, 13, COMMIT, first["next_offset"], 4)
    assert (first["text"], first["eof"]) == ("abcd", False)
    assert (second["text"], second["eof"]) == ("ef", True)
    with pytest.raises(RemoteJobError, match="byte limit"):
        read_log(tmp_path, 13, COMMIT, 0, 65537)
