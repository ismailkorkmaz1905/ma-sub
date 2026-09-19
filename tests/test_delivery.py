import hashlib
import json

import pytest

from mas import delivery, remote
from mas.reliability import atomic_json


@pytest.fixture(autouse=True)
def isolated_rclone_environment(monkeypatch):
    for name in tuple(remote.os.environ):
        if name.startswith("RCLONE_") or name == "MAS_RCLONE_CONFIG":
            monkeypatch.delenv(name)


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
            return json.dumps([{"Name": key.rsplit("/", 1)[-1], "ID": key}
                               for key in files if key.rsplit("/", 1)[0] == command[2]]).encode()
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
    path = tmp_path / "output" / "movie.mp4"
    path.parent.mkdir()
    path.write_bytes(b"changed")
    record = {"relative_path": "output/movie.mp4", "size_bytes": 7,
              "sha256": hashlib.sha256(b"original").hexdigest()}
    with pytest.raises(ValueError, match="byte/SHA"):
        delivery.verified_record(tmp_path, record)


@pytest.fixture
def drive_upload(tmp_path, monkeypatch):
    source = tmp_path / "new.mp4"
    source.write_bytes(b"movie content")
    target = "drive:delivery/Official.mp4"
    files, identities, calls = {}, {}, []
    failures = {"cat": 0, "copyto": 0, "moveto": 0, "moveto_before": 0}
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")

    def run(command, *, stdout_handler=None, **kwargs):
        operation = command[1]
        calls.append(command)
        assert command[command.index("--retries") + 1] == "1"
        if operation == "lsjson":
            return json.dumps([{"Name": key.rsplit("/", 1)[-1], "ID": identities[key]}
                               for key in files if key.rsplit("/", 1)[0] == command[2]]).encode()
        if operation == "copyto":
            assert command[3] not in files
            files[command[3]] = source.read_bytes()
            identities[command[3]] = "upload-object"
        if operation == "moveto":
            if failures["moveto_before"]:
                failures["moveto_before"] -= 1
                raise remote.RemoteVerificationError("connection unavailable before move")
            assert command[3] not in files
            files[command[3]] = files.pop(command[2])
            identities[command[3]] = identities.pop(command[2])
        if operation == "cat":
            if command[2] not in files:
                raise remote.RemoteVerificationError("missing object")
            stdout_handler(files[command[2]])
        if failures.get(operation, 0):
            failures[operation] -= 1
            raise remote.RemoteVerificationError("temporary metadata failure")
        return b""

    monkeypatch.setattr(remote, "_run_watchdog", run)
    return source, target, files, identities, calls, failures


@pytest.mark.parametrize("failed_operation,count", [("cat", 3), ("copyto", 1), ("moveto", 1)])
def test_retry_reuses_uploaded_bytes_after_uncertain_metadata_or_readback(
        drive_upload, tmp_path, failed_operation, count):
    source, target, files, _, calls, failures = drive_upload
    failures[failed_operation] = count
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    result = remote.upload_verified(source, target, preservation_receipt=preserve)
    assert files[target] == source.read_bytes()
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert sum(command[1] == "copyto" for command in calls) == 1
    assert calls[-1][1:3] == ["cat", target]


def test_partial_same_size_wrong_hash_is_preserved_and_never_reuploaded(drive_upload, tmp_path):
    source, target, files, _, calls, failures = drive_upload
    failures["cat"] = 3
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    partial = next(iter(files))
    files[partial] = b"x" * source.stat().st_size
    with pytest.raises(remote.RemoteVerificationError, match="partial Drive byte/SHA"):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    assert target not in files
    assert files[partial] == b"x" * source.stat().st_size
    assert sum(command[1] == "copyto" for command in calls) == 1


def test_partial_object_id_change_blocks_resume(drive_upload, tmp_path):
    source, target, files, identities, _, failures = drive_upload
    failures["cat"] = 3
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    identities[next(iter(files))] = "replacement-object"
    with pytest.raises(remote.RemoteVerificationError, match="partial object identity changed"):
        remote.upload_verified(source, target, preservation_receipt=preserve)


def test_missing_partial_upload_retry_count_is_persistently_bounded(drive_upload, tmp_path, monkeypatch):
    source, target, _, _, calls, _ = drive_upload
    original = remote._run_watchdog
    def run(command, **kwargs):
        if command[1] == "copyto":
            calls.append(command)
            raise remote.RemoteVerificationError("connection unavailable before upload")
        return original(command, **kwargs)
    monkeypatch.setattr(remote, "_run_watchdog", run)
    preserve = tmp_path / "preserve.json"
    for _ in range(2):
        with pytest.raises(remote.RemoteVerificationError, match="connection unavailable"):
            remote.upload_verified(source, target, preservation_receipt=preserve)
    with pytest.raises(remote.RemoteVerificationError, match="retry allowance exhausted"):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    assert sum(command[1] == "copyto" for command in calls) == 2


def test_tampered_upload_checkpoint_blocks_before_network(drive_upload, tmp_path):
    source, target, _, _, calls, failures = drive_upload
    failures["cat"] = 3
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    checkpoint = next(tmp_path.glob("preserve-upload-*.json"))
    saved = json.loads(checkpoint.read_text())
    saved["payload"]["partial"] = "drive:unrelated/file"
    atomic_json(checkpoint, saved)
    calls.clear()
    with pytest.raises(remote.RemoteVerificationError, match="checkpoint binding changed"):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    assert not calls


def test_stream_readback_retry_starts_fresh_hash(monkeypatch):
    attempts = []
    def run(command, *, stdout_handler, **kwargs):
        attempts.append(1)
        stdout_handler(b"first")
        if len(attempts) == 1:
            raise remote.RemoteVerificationError("temporary read error")
        stdout_handler(b"second")
    monkeypatch.setattr(remote, "_run_watchdog", run)
    assert remote._remote_signature(["rclone", "cat", "target"], idle_timeout=1, total_timeout=5) == (
        11, hashlib.sha256(b"firstsecond").hexdigest())
    assert len(attempts) == 2


def _oauth_config(tmp_path, *, private=True, scope="drive"):
    config = tmp_path / "rclone.conf"
    config.write_text("[drive]\ntype = drive\n"
                      + ("client_id = private-client\nclient_secret = secret-fixture\n" if private else "")
                      + f"scope = {scope}\n"
                      + 'token = {"refresh_token": "refresh-fixture"}\n', encoding="utf-8")
    return config


def test_drive_preflight_requires_private_oauth_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")
    monkeypatch.setattr(remote, "_run_watchdog", lambda *args, **kwargs: pytest.fail("network invoked"))
    with pytest.raises(remote.RemoteVerificationError, match="private Drive OAuth"):
        remote.drive_preflight("drive:delivery", required_bytes=10,
                               config_path=_oauth_config(tmp_path, private=False))


@pytest.mark.parametrize("free", [None, True, -1, "100", 4])
def test_drive_preflight_rejects_unknown_or_insufficient_space(tmp_path, monkeypatch, free):
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")
    monkeypatch.setattr(remote, "_run_watchdog", lambda *args, **kwargs: json.dumps({"free": free}).encode())
    with pytest.raises(remote.RemoteVerificationError, match="space"):
        remote.drive_preflight("drive:delivery", required_bytes=5, config_path=_oauth_config(tmp_path))


def test_drive_preflight_binds_identity_without_disclosing_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return b'{"free":100}'
    monkeypatch.setattr(remote, "_run_watchdog", run)
    config = _oauth_config(tmp_path)
    result = remote.drive_preflight("drive:delivery", required_bytes=10, config_path=config)
    assert result["status"] == "PASS" and result["free_bytes"] == 100
    assert len(result["credential_identity_sha256"]) == 64
    assert result["write_access"] == "NOT_PROBED"
    assert "secret-fixture" not in json.dumps(result)
    assert "refresh-fixture" not in json.dumps(result)
    assert calls[0][0][:4] == ["rclone", "about", "drive:", "--json"]
    assert calls[0][0][calls[0][0].index("--config") + 1] == str(config)
    assert calls[0][1]["total_timeout"] <= 60


def test_space_failure_never_moves_retained_final_or_uploads(drive_upload, tmp_path, monkeypatch):
    source, target, files, identities, calls, _ = drive_upload
    files[target], identities[target] = b"prior movie", "prior-id"
    monkeypatch.setattr(remote, "drive_preflight", lambda *args, **kwargs: {"free_bytes": 0})
    with pytest.raises(remote.RemoteVerificationError, match="insufficient Drive space"):
        remote.upload_verified(source, target, preservation_receipt=tmp_path / "preserve.json",
                               require_drive_preflight=True)
    assert files[target] == b"prior movie"
    assert not any(command[1] in {"copyto", "moveto"} for command in calls)


def test_completed_partial_resume_does_not_require_another_file_of_free_space(
        drive_upload, tmp_path, monkeypatch):
    source, target, _, _, calls, failures = drive_upload
    failures["cat"] = 3
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    monkeypatch.setattr(remote, "drive_preflight", lambda *args, **kwargs: {"free_bytes": 0})
    receipt = remote.upload_verified(source, target, preservation_receipt=preserve,
                                     require_drive_preflight=True)
    assert receipt["bytes"] == source.stat().st_size
    assert receipt["preflight"]["free_bytes"] == 0
    assert sum(command[1] == "copyto" for command in calls) == 1


def test_preservation_readback_must_complete_before_retry_can_upload(drive_upload, tmp_path, monkeypatch):
    source, target, files, identities, calls, _ = drive_upload
    files[target], identities[target] = b"old movie", "old-id"
    original = remote._run_watchdog
    failures = [3]
    def run(command, **kwargs):
        result = original(command, **kwargs)
        if command[1] == "cat" and "/.retained/" in command[2] and failures[0]:
            failures[0] -= 1
            raise remote.RemoteVerificationError("retained readback unavailable")
        return result
    monkeypatch.setattr(remote, "_run_watchdog", run)
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError, match="retained readback unavailable"):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    assert not any(command[1] == "copyto" for command in calls)
    retained = next(iter(files))
    assert "/.retained/" in retained
    assert files[retained] == b"old movie"
    remote.upload_verified(source, target, preservation_receipt=preserve)
    assert files[retained] == b"old movie"
    assert files[target] == source.read_bytes()
    assert sum(command[1] == "copyto" for command in calls) == 1


@pytest.mark.parametrize("failure", ["moveto_before", "moveto"])
def test_pending_preservation_reconciles_interrupted_move_without_reupload(
        drive_upload, tmp_path, failure):
    source, target, files, identities, calls, failures = drive_upload
    files[target], identities[target] = b"old movie", "old-id"
    failures[failure] = 1
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)

    receipt = remote.upload_verified(source, target, preservation_receipt=preserve)

    retained = next(key for key in files if "/.retained/" in key)
    assert files[retained] == b"old movie"
    assert identities[retained] == "old-id"
    assert files[target] == source.read_bytes()
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(preserve.read_text())["status"] == "PRESERVED"
    assert sum(command[1] == "copyto" for command in calls) == 1


def test_pending_preservation_rejects_changed_original_before_move_retry(
        drive_upload, tmp_path):
    source, target, files, identities, calls, failures = drive_upload
    files[target], identities[target] = b"old movie", "old-id"
    failures["moveto_before"] = 1
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    files[target] = b"changed old movie"

    with pytest.raises(remote.RemoteVerificationError, match="retained Drive byte/SHA"):
        remote.upload_verified(source, target, preservation_receipt=preserve)

    assert not any(command[1] == "copyto" for command in calls)


def test_pending_preservation_rejects_original_and_archive_both_present(
        drive_upload, tmp_path):
    source, target, files, identities, calls, failures = drive_upload
    files[target], identities[target] = b"old movie", "old-id"
    failures["moveto_before"] = 1
    preserve = tmp_path / "preserve.json"
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=preserve)
    checkpoint = json.loads(next(tmp_path.glob("preserve-upload-*.json")).read_text())
    retained = checkpoint["payload"]["retained_pending"]["retained_remote"]
    files[retained], identities[retained] = b"old movie", "old-id"

    with pytest.raises(remote.RemoteVerificationError, match="ambiguous retained Drive"):
        remote.upload_verified(source, target, preservation_receipt=preserve)

    assert not any(command[1] == "copyto" for command in calls)


def test_preflight_uses_no_secret_values_in_configuration_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")
    config = tmp_path / "rclone.conf"
    config.write_text("this-is-a-secret-without-a-section", encoding="utf-8")
    with pytest.raises(remote.RemoteVerificationError) as error:
        remote.drive_preflight("drive:delivery", required_bytes=0, config_path=config)
    assert "this-is-a-secret" not in str(error.value)


def test_pending_preservation_creates_missing_retention_directory_before_listing(drive_upload, tmp_path, monkeypatch):
    source, target, files, identities, calls, failures = drive_upload
    files[target], identities[target] = b'old movie', 'old-id'
    original = remote._run_watchdog
    directories = set()
    def backend(command, **kwargs):
        if command[1] == 'lsjson' and command[2] not in directories:
            raise remote.RemoteVerificationError('directory not found (rclone exit 3)')
        result = original(command, **kwargs)
        if command[1] == 'mkdir':
            directories.add(command[2])
        if command[1] == 'moveto':
            directories.add(command[3].rsplit('/', 1)[0])
        return result
    monkeypatch.setattr(remote, '_run_watchdog', backend)
    failures['moveto_before'] = 1
    receipt = tmp_path / 'preserve.json'
    with pytest.raises(remote.RemoteVerificationError):
        remote.upload_verified(source, target, preservation_receipt=receipt)
    result = remote.upload_verified(source, target, preservation_receipt=receipt)
    assert result['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    retained = next(k for k in files if '/.retained/' in k)
    assert files[retained] == b'old movie' and identities[retained] == 'old-id'
    assert sum(c[1] == 'copyto' for c in calls) == 1
