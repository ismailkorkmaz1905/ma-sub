import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mas import drive_resumable as resume
from mas import remote


CHUNK = 256 * 1024
TARGET = "drive:delivery/Official.mp4"
PARTIAL = TARGET + ".partial-" + "a" * 32
SESSION = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=secret-session"
CREDENTIALS = {"client_id": "private-client", "client_secret": "secret-client",
               "refresh_token": "secret-refresh", "credential_identity_sha256": "b" * 64}


class FakeDrive:
    def __init__(self, content):
        self.content = content
        self.received = bytearray()
        self.calls = []
        self.fail_after_first_chunk = False
        self.break_process = False
        self.query_status = None
        self.query_range = None
        self.reject_chunks = False
        self.bad_metadata = False
        self.partial = PARTIAL
        self.object_exists = False
        self.created_ids = []

    def __call__(self, method, url, headers, body, timeout):
        assert 0 < timeout <= 30
        self.calls.append((method, url, dict(headers), body))
        if url == "https://oauth2.googleapis.com/token":
            assert method == "POST"
            assert b"grant_type=refresh_token" in body and b"secret-refresh" in body
            return 200, {}, b'{"access_token":"secret-access","token_type":"Bearer"}'
        assert headers["Authorization"] == "Bearer secret-access"
        if method == "GET" and "/generateIds?" in url:
            return 200, {}, b'{"ids":["object-id"]}'
        if method == "GET":
            assert "/files/object-id?" in url
            if not self.object_exists:
                return 404, {}, b""
            return 200, {}, json.dumps({"id": "object-id", "name": self.partial.rsplit("/", 1)[-1],
                "size": str(len(self.received)), "parents": ["folder-id"]}).encode()
        if method == "POST":
            assert json.loads(body) == {"id": "object-id", "name": self.partial.rsplit("/", 1)[-1], "parents": ["folder-id"]}
            if self.object_exists:
                return 409, {}, b""
            self.object_exists = True
            self.created_ids.append("object-id")
            return 200, {"location": SESSION}, b""
        if method == "PATCH":
            assert self.object_exists and "/files/object-id?" in url and json.loads(body) == {}
            self.received.clear()
            self.query_status = None
            return 200, {"location": SESSION}, b""
        assert url == SESSION
        if body:
            if self.reject_chunks:
                raise OSError("unsafe session/secret text from network")
            start = int(headers["Content-Range"].split()[1].split("-")[0])
            assert start == len(self.received)
            assert body == self.content[start:start + len(body)]
            self.received.extend(body)
            if self.break_process:
                self.break_process = False
                raise KeyboardInterrupt("simulated process death after remote acknowledgement")
            if self.fail_after_first_chunk:
                self.fail_after_first_chunk = False
                raise OSError("unsafe session/secret text from network")
        else:
            assert headers["Content-Range"] == f"bytes */{len(self.content)}"
            assert headers["Content-Length"] == "0"
            if self.query_status is not None:
                return self.query_status, {}, b""
            if self.query_range is not None:
                return 308, {"range": self.query_range}, b""
        if len(self.received) == len(self.content):
            metadata = {"id": "object-id", "name": self.partial.rsplit("/", 1)[-1],
                        "size": str(len(self.content)), "parents": ["folder-id"]}
            if self.bad_metadata:
                metadata["name"] = "different.mp4"
            return 200, {}, json.dumps(metadata).encode()
        return 308, {"range": f"bytes=0-{len(self.received)-1}"} if self.received else {}, b""


@pytest.fixture
def setup_upload(tmp_path):
    content = b"a" * CHUNK + b"b" * CHUNK + b"last"
    path = tmp_path / "movie.mp4"
    path.write_bytes(content)
    drive = FakeDrive(content)
    options = {"source": path, "remote": TARGET, "partial": PARTIAL, "parent_id": "folder-id",
               "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
               "credentials": CREDENTIALS, "session_dir": tmp_path / "private", "chunk_size": CHUNK,
               "transport": drive}
    return path, drive, options


def data_calls(drive):
    return [call for call in drive.calls if call[0] == "PUT" and call[3]]


def test_ambiguous_chunk_queries_acknowledged_offset_without_retransmitting(setup_upload):
    _, drive, options = setup_upload
    drive.fail_after_first_chunk = True
    progress = []
    result = resume.upload(**options, progress=lambda *event: progress.append(event))
    assert bytes(drive.received) == drive.content
    assert [call[2]["Content-Range"] for call in data_calls(drive)] == [
        f"bytes 0-{CHUNK - 1}/{len(drive.content)}",
        f"bytes {CHUNK}-{2 * CHUNK - 1}/{len(drive.content)}",
        f"bytes {2 * CHUNK}-{len(drive.content) - 1}/{len(drive.content)}"]
    first_data = drive.calls.index(data_calls(drive)[0])
    assert drive.calls[first_data + 1][3] == b""
    assert result["object_id"] == "object-id"
    assert progress[-1] == ("upload", len(drive.content))
    assert "secret" not in json.dumps(result)


def test_process_death_resumes_saved_session_and_server_offset(setup_upload):
    _, drive, options = setup_upload
    drive.break_process = True
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**options)
    with resume.SessionStore(PARTIAL, options["session_dir"]) as store:
        state = store.load()
        assert state["acked"] == 0 and state["sent_through"] == CHUNK
        assert state["session_uri"] == SESSION
        if os.name == "nt":
            assert SESSION not in store.path.read_text()
    split = len(drive.calls)
    result = resume.upload(**options)
    assert result["status"] == "UPLOAD_COMPLETE"
    assert drive.calls[split + 1][2]["Content-Range"] == f"bytes */{len(drive.content)}"
    assert sum(call[0] == "POST" and call[1] != "https://oauth2.googleapis.com/token" for call in drive.calls) == 1
    assert len(data_calls(drive)) == 3


def test_changed_same_size_source_blocked_before_network(setup_upload):
    path, drive, options = setup_upload
    drive.break_process = True
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**options)
    count = len(drive.calls)
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(resume.ResumableUploadError, match="byte/SHA-256 binding changed"):
        resume.upload(**options)
    assert len(drive.calls) == count
    options["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(resume.ResumableUploadError, match="session source/destination/OAuth binding changed"):
        resume.upload(**options)
    assert len(drive.calls) == count


def test_forbidden_session_never_restarts(setup_upload):
    status = 403
    _, drive, options = setup_upload
    drive.break_process = True
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**options)
    drive.query_status = status
    for _ in range(2):
        with pytest.raises(resume.ResumableUploadError):
            resume.upload(**options)
    assert sum(call[0] == "POST" and call[1] != "https://oauth2.googleapis.com/token" for call in drive.calls) == 1
    assert len(data_calls(drive)) == 1


@pytest.mark.parametrize("value", ["bytes=2-7", "bytes=0-99999999", "garbage"])
def test_invalid_acknowledgement_never_sends_more(setup_upload, value):
    _, drive, options = setup_upload
    drive.query_range = value
    with pytest.raises(resume.ResumableUploadError, match="invalid byte range"):
        resume.upload(**options)
    assert not data_calls(drive)


def test_acknowledgement_cannot_claim_unsent_bytes(setup_upload):
    _, drive, options = setup_upload
    drive.query_range = "bytes=0-42"
    with pytest.raises(resume.ResumableUploadError, match="exceeds sent bytes"):
        resume.upload(**options)
    assert not data_calls(drive)


def test_repeated_failure_is_bounded_without_new_session(setup_upload):
    _, drive, options = setup_upload
    drive.reject_chunks = True
    with pytest.raises(resume.ResumableUploadError, match="allowance exhausted") as error:
        resume.upload(**options)
    assert len(data_calls(drive)) == 4
    assert "secret" not in str(error.value)


def test_ambiguous_initiation_retries_same_reserved_object(setup_upload):
    _, drive, options = setup_upload
    first = [True]
    def lost(method, url, *args):
        result = drive(method, url, *args)
        if method == "POST" and "upload/drive" in url and first[0]:
            first[0] = False
            raise OSError("lost create response")
        return result
    result = resume.upload(**(options | {"transport": lost}))
    assert result['object_id'] == 'object-id'
    assert drive.created_ids == ['object-id']
    assert any(call[0] == 'PATCH' for call in drive.calls)
    assert bytes(drive.received) == drive.content


def test_legacy_ambiguous_initiation_without_reserved_id_does_not_create(setup_upload):
    _, drive, options = setup_upload
    drive.break_process = True
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**options)
    with resume.SessionStore(PARTIAL, options['session_dir']) as store:
        state = store.load()
        state.pop('reserved_id')
        state.pop('session_uri')
        state['status'] = 'INITIATING'
        store.save(state)
    count = len(drive.calls)
    with pytest.raises(resume.ResumableUploadError, match='initiation is ambiguous'):
        resume.upload(**options)
    assert len(drive.calls) == count


def test_expired_session_restarts_only_the_owned_temporary_object(setup_upload):
    _, drive, options = setup_upload
    drive.break_process = True
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**options)
    drive.query_status = 404
    result = resume.upload(**options, allow_session_create=False)
    assert result['object_id'] == 'object-id'
    assert drive.created_ids == ['object-id']
    assert bytes(drive.received) == drive.content
    assert sum(call[0] == 'PATCH' for call in drive.calls) == 1


def test_expired_session_does_not_overwrite_an_object_with_changed_ownership(setup_upload):
    _, drive, options = setup_upload
    drive.break_process = True
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**options)
    drive.query_status = 404
    def foreign(method, url, *args):
        code, headers, data = drive(method, url, *args)
        if method == 'GET' and '/files/object-id?' in url:
            metadata = json.loads(data)
            metadata['parents'] = ['foreign-folder']
            data = json.dumps(metadata).encode()
        return code, headers, data
    with pytest.raises(resume.ResumableUploadError, match='ownership changed'):
        resume.upload(**(options | {'transport': foreign}), allow_session_create=False)
    assert not any(call[0] == 'PATCH' for call in drive.calls)


def test_lost_final_response_then_expired_session_does_not_send_bytes_again(setup_upload):
    _, drive, options = setup_upload
    def lost_final(method, url, headers, body, timeout):
        result = drive(method, url, headers, body, timeout)
        if method == 'PUT' and body and len(drive.received) == len(drive.content):
            raise KeyboardInterrupt('final bytes received, reply lost')
        return result
    with pytest.raises(KeyboardInterrupt):
        resume.upload(**(options | {'transport': lost_final}))
    before = len(data_calls(drive))
    drive.query_status = 404
    assert resume.upload(**options)['status'] == 'UPLOAD_COMPLETE'
    assert len(data_calls(drive)) == before
    assert drive.created_ids == ['object-id']
    assert not any(call[0] == 'PATCH' for call in drive.calls)


def test_lost_create_response_before_remote_creation_is_bounded_and_resumable(setup_upload):
    _, drive, options = setup_upload
    attempts = []
    def broken(method, url, *args):
        if method == 'POST' and 'upload/drive' in url:
            attempts.append(1)
            raise OSError('request never reached server')
        return drive(method, url, *args)
    with pytest.raises(resume.ResumableUploadError, match='initiation retry allowance'):
        resume.upload(**(options | {'transport': broken}))
    assert len(attempts) == 3 and not drive.created_ids
    assert resume.upload(**options, allow_session_create=False)['object_id'] == 'object-id'
    assert drive.created_ids == ['object-id']


def test_completion_metadata_must_match_exact_partial(setup_upload):
    _, drive, options = setup_upload
    drive.bad_metadata = True
    with pytest.raises(resume.ResumableUploadError, match="completion does not match"):
        resume.upload(**options)


@pytest.mark.parametrize("url", ["https://evil.example/upload_id=x",
    "https://www.googleapis.com@evil.example/upload/drive/v3/files?upload_id=x",
    "http://www.googleapis.com/upload/drive/v3/files?upload_id=x",
    "https://www.googleapis.com/other?upload_id=x"])
def test_untrusted_session_location_rejected(url):
    with pytest.raises(resume.ResumableUploadError, match="unsafe upload session"):
        resume._session_url(url)


def test_sessions_never_persist_in_repository():
    with pytest.raises(resume.ResumableUploadError, match="outside Git"):
        resume.session_directory(Path(__file__).resolve().parents[1] / "sessions")


def test_session_single_writer_lock(setup_upload):
    _, _, options = setup_upload
    with resume.SessionStore(PARTIAL, options["session_dir"]):
        with pytest.raises(resume.ResumableUploadError, match="another process"):
            with resume.SessionStore(PARTIAL, options["session_dir"]):
                pytest.fail("concurrent writer obtained lock")


def test_progress_ignores_chatter_and_repeated_acknowledgement():
    observe = remote._UploadProgress()
    assert observe("stderr", b'{"phase":"hash","bytes":100}\n')
    assert observe("stderr", b'{"phase":"upload","bytes":1}\n')
    assert not observe("stderr", b'{"phase":"upload","bytes":1}\n')
    assert not observe("stderr", b'{"message":"heartbeat"}\n')
    assert not observe("stderr", b'{"phase":"upload","bytes":true}\n')


def test_network_worker_watchdog_is_real_process_bounded():
    with pytest.raises(remote.RemoteVerificationError, match="watchdog expired"):
        remote._run_watchdog([sys.executable, "-c", "import time; time.sleep(5)"],
                             idle_timeout=0.15, total_timeout=0.3,
                             progress_observer=remote._UploadProgress())


def test_missing_session_never_creates_replacement(setup_upload):
    _, drive, options = setup_upload
    with pytest.raises(resume.ResumableUploadError, match="saved upload session is missing"):
        resume.upload(**options, allow_session_create=False)
    assert not drive.calls


def test_deadline_applies_after_slow_response(setup_upload, monkeypatch):
    _, drive, options = setup_upload
    now = [100.0]
    monkeypatch.setattr(resume.time, "monotonic", lambda: now[0])

    def delayed(*args):
        result = drive(*args)
        now[0] += 11
        return result

    with pytest.raises(resume.ResumableUploadError, match="deadline or no-progress"):
        resume.upload(**(options | {"transport": delayed}), total_timeout=10)
    assert len(drive.calls) == 1


def test_regressing_acknowledgement_is_not_retransferred(setup_upload):
    _, drive, options = setup_upload

    def crash_second_chunk(method, url, headers, body, timeout):
        if method == "PUT" and body and len(drive.received) == CHUNK:
            raise KeyboardInterrupt()
        return drive(method, url, headers, body, timeout)

    with pytest.raises(KeyboardInterrupt):
        resume.upload(**(options | {"transport": crash_second_chunk}))
    drive.query_range = "bytes=0-42"
    with pytest.raises(resume.ResumableUploadError, match="acknowledgement regressed"):
        resume.upload(**options)
    assert len(data_calls(drive)) == 1


@pytest.mark.parametrize("corrupt_final", [False, True])
def test_production_preserves_old_file_and_requires_full_final_hash(setup_upload, tmp_path, monkeypatch, corrupt_final):
    source, drive, options = setup_upload
    files = {TARGET: b"prior movie"}
    identities = {TARGET: "prior-id"}
    calls = []
    monkeypatch.setattr(remote.shutil, "which", lambda _: "rclone")
    monkeypatch.setattr(remote, "drive_preflight", lambda *args, **kwargs: {
        "free_bytes": len(drive.content) * 2, "credential_identity_sha256": "b" * 64})

    def run(command, *, stdout_handler=None, **_kwargs):
        operation = command[1]
        calls.append((operation, command[2]))
        if operation == "lsjson":
            return json.dumps([{"Name": key.rsplit("/", 1)[-1], "ID": identities[key]}
                               for key in files if key.rsplit("/", 1)[0] == command[2]]).encode()
        if operation == "cat":
            stdout_handler(files[command[2]])
        elif operation == "moveto":
            assert command[3] not in files
            files[command[3]] = files.pop(command[2])
            identities[command[3]] = identities.pop(command[2])
            if command[3] == TARGET and corrupt_final:
                files[TARGET] = b"x" * len(drive.content)
        else:
            assert operation == "mkdir"
        return b""

    def native(path, target, partial, size, sha, preflight, **kwargs):
        retained = next(key for key in files if "/.retained/" in key)
        assert ("cat", retained) in calls
        assert files[retained] == b"prior movie"
        calls.append(("native", partial))
        drive.partial = partial
        result = resume.upload(**(options | {"partial": partial}),
                               allow_session_create=kwargs["allow_session_create"])
        files[partial] = bytes(drive.received)
        identities[partial] = result["object_id"]
        return result

    monkeypatch.setattr(remote, "_run_watchdog", run)
    monkeypatch.setattr(remote, "_upload_resumable", native)
    preserve = tmp_path / "preserve.json"
    if corrupt_final:
        with pytest.raises(remote.RemoteVerificationError, match="final Drive byte/SHA-256 readback mismatch"):
            remote.upload_verified(source, TARGET, preservation_receipt=preserve, require_drive_preflight=True)
    else:
        result = remote.upload_verified(source, TARGET, preservation_receipt=preserve, require_drive_preflight=True)
        assert result["sha256"] == hashlib.sha256(drive.content).hexdigest()
        remote.upload_verified(source, TARGET, preservation_receipt=preserve, require_drive_preflight=True)
    assert len([call for call in calls if call[0] == "native"]) == 1
    evidence = json.loads(preserve.read_text())
    retained = next(key for key in files if "/.retained/" in key)
    assert files[retained] == b"prior movie"
    assert calls.index(("cat", retained)) < next(i for i, call in enumerate(calls) if call[0] == "native")
    assert calls[-1] == ("cat", TARGET)
    for receipt in tmp_path.glob("*.json"):
        assert "secret" not in receipt.read_text()


def test_production_adapter_uses_watchdog_and_exact_folder_identity(tmp_path, monkeypatch):
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"movie")
    sha = hashlib.sha256(b"movie").hexdigest()
    credentials = CREDENTIALS | {"config_path": "private-config"}
    monkeypatch.setattr(remote, "_drive_credentials", lambda *args, **kwargs: credentials)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["total_timeout"] <= 30
        if command[0] == "rclone":
            assert command[1:4] == ["lsjson", "drive:delivery", "--stat"]
            return b'{"ID":"folder-id","IsDir":true}'
        assert command[:3] == [sys.executable, "-m", "mas.drive_resumable"]
        assert isinstance(kwargs["progress_observer"], remote._UploadProgress)
        assert "--allow-session-create" not in command
        assert not any("secret" in part for part in command)
        binding = {"source": str(source.resolve()), "remote": TARGET, "partial": PARTIAL,
                   "parent_id": "folder-id", "bytes": 5, "sha256": sha,
                   "credential_identity_sha256": "b" * 64}
        return json.dumps({"status": "UPLOAD_COMPLETE", "bytes": 5, "sha256": sha,
                           "object_id": "object-id", "session_binding_sha256": remote.digest(binding)}).encode()

    monkeypatch.setattr(remote, "_run_watchdog", run)
    result = remote._upload_resumable(source, TARGET, PARTIAL, 5, sha,
        {"credential_identity_sha256": "b" * 64}, idle_timeout=10, total_timeout=30,
        allow_session_create=False)
    assert result["object_id"] == "object-id"
    assert len(commands) == 2


def test_hard_process_exit_releases_lock_and_preserves_session(setup_upload):
    source, drive, options = setup_upload
    binding = {"source": str(source.resolve()), "remote": TARGET, "partial": PARTIAL,
               "parent_id": "folder-id", "bytes": len(drive.content),
               "sha256": options["sha256"], "credential_identity_sha256": "b" * 64}
    state = {"binding": binding, "status": "ACTIVE", "acked": 0,
             "sent_through": CHUNK, "session_uri": SESSION}
    code = ("import json, os, sys; from mas.drive_resumable import SessionStore; "
            "store = SessionStore(sys.argv[1], sys.argv[2]); store.__enter__(); "
            "store.save(json.loads(sys.argv[3])); os._exit(17)")
    result = subprocess.run([sys.executable, "-c", code, PARTIAL,
                             str(options["session_dir"]), json.dumps(state)],
                            capture_output=True, timeout=10)
    assert result.returncode == 17
    drive.received.extend(drive.content[:CHUNK])
    result = resume.upload(**options, allow_session_create=False)
    assert result["status"] == "UPLOAD_COMPLETE"
    assert len(data_calls(drive)) == 2
    assert all(call[0] != "POST" or call[1] == "https://oauth2.googleapis.com/token" for call in drive.calls)


def test_cli_failure_never_prints_session_or_credentials(setup_upload, monkeypatch, capsys):
    source, _, options = setup_upload
    monkeypatch.setattr(remote, "_drive_credentials", lambda *args, **kwargs: CREDENTIALS)

    def fail(*args, **kwargs):
        raise OSError("request failed " + SESSION + " secret-refresh secret-access")

    monkeypatch.setattr(resume, "upload", fail)
    monkeypatch.setattr(sys, "argv", ["worker", "--source", str(source), "--remote", TARGET,
        "--partial", PARTIAL, "--parent-id", "folder-id", "--sha256", options["sha256"],
        "--identity", "b" * 64, "--config", "private-config", "--size", str(options["size"]),
        "--total-timeout", "10", "--idle-timeout", "5"])
    assert resume.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "secret" not in output.err and "googleapis" not in output.err


def test_oauth_failure_leaves_resumable_prepared_intent_not_missing_session(setup_upload):
    _, drive, options = setup_upload
    def unavailable(*args):
        raise OSError('temporary OAuth outage before file creation')
    with pytest.raises(resume.ResumableUploadError, match='OAuth'):
        resume.upload(**(options | {'transport': unavailable}))
    with resume.SessionStore(PARTIAL, options['session_dir']) as store:
        assert store.load()['status'] == 'PREPARED'
    result = resume.upload(**options, allow_session_create=False)
    assert result['status'] == 'UPLOAD_COMPLETE'
    assert sum(c[0] == 'POST' and c[1] != 'https://oauth2.googleapis.com/token' for c in drive.calls) == 1


def test_prepared_session_survives_controller_flag_before_worker_starts(setup_upload):
    source, drive, options = setup_upload
    binding = {'source': str(source.resolve()), 'remote': TARGET, 'partial': PARTIAL,
               'parent_id': 'folder-id', 'bytes': options['size'], 'sha256': options['sha256'],
               'credential_identity_sha256': CREDENTIALS['credential_identity_sha256']}
    resume.prepare_session(PARTIAL, binding, allow_create=True, session_dir=options['session_dir'])
    assert drive.calls == []
    resume.prepare_session(PARTIAL, binding, allow_create=False, session_dir=options['session_dir'])
    assert resume.upload(**options, allow_session_create=False)['status'] == 'UPLOAD_COMPLETE'
