import hashlib
import json

import pytest

from mas import delivery, remote
from mas.reliability import atomic_json


def test_drive_collision_preserves_prior_bytes_before_publication(tmp_path, monkeypatch):
    source = tmp_path / "new.mp4"
    source.write_bytes(b"new movie")
    target = "drive:delivery/Official.mp4"
    files = {target: b"old movie"}
    calls = []
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")

    def run(command, *, stdout_handler=None, **_kwargs):
        operation = command[1]
        calls.append((operation, command[2]))
        if operation == "lsjson":
            return json.dumps([{"Name": "Official.mp4", "ID": "old-id"}]).encode()
        if operation == "cat":
            stdout_handler(files[command[2]])
        if operation == "moveto":
            assert command[3] not in files
            files[command[3]] = files.pop(command[2])
        if operation == "copyto":
            assert any("/.retained/" in key for key in files)
            files[command[3]] = source.read_bytes()
        return b""

    monkeypatch.setattr(remote, "_run_watchdog", run)
    receipt_path = tmp_path / "preserve.json"
    receipt = remote.upload_verified(source, target, preservation_receipt=receipt_path)
    assert files[target] == b"new movie"
    evidence = json.loads(receipt_path.read_text())
    assert evidence["status"] == "PRESERVED"
    assert files[evidence["retained_remote"]] == b"old movie"
    assert evidence["prior"]["sha256"] == hashlib.sha256(b"old movie").hexdigest()
    assert receipt["sha256"] == hashlib.sha256(b"new movie").hexdigest()
    assert calls.index(("cat", evidence["retained_remote"])) < next(
        i for i, value in enumerate(calls) if value[0] == "copyto")


def test_duplicate_drive_name_blocks_without_mutation(tmp_path, monkeypatch):
    source = tmp_path / "new.mp4"
    source.write_bytes(b"new")
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")

    def run(command, **_kwargs):
        assert command[1] in {"mkdir", "lsjson"}
        return json.dumps([{"Name": "Official.mp4"}] * 2).encode()

    monkeypatch.setattr(remote, "_run_watchdog", run)
    with pytest.raises(remote.RemoteVerificationError, match="ambiguous duplicate"):
        remote.upload_verified(source, "drive:delivery/Official.mp4",
                               preservation_receipt=tmp_path / "preserve.json")


@pytest.mark.parametrize("value", ["../source.mp4", "/source.mp4", "C:/source.mp4", "final\\file"])
def test_delivery_paths_cannot_escape_episode(tmp_path, value):
    with pytest.raises(ValueError):
        delivery.safe_relative(tmp_path, value)


def test_local_delivery_rejects_changed_artifact(tmp_path):
    path = tmp_path / "final" / "movie.mp4"
    path.parent.mkdir()
    path.write_bytes(b"changed")
    record = {"relative_path": "final/movie.mp4", "size_bytes": 7,
              "sha256": hashlib.sha256(b"original").hexdigest()}
    with pytest.raises(ValueError, match="byte/SHA"):
        delivery.verified_record(tmp_path, record)
