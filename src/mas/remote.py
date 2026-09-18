import configparser
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .reliability import atomic_json, digest, read_json


class RemoteVerificationError(RuntimeError):
    pass


def _run_watchdog(
    command,
    *,
    idle_timeout=120,
    total_timeout=3600,
    stdout_handler=None,
    stderr_handler=None,
    progress_observer=None,
    progress_probe=None,
    progress_probe_interval=10,
):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    events = queue.Queue()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}

    def read(name, stream):
        read_chunk = getattr(stream, "read1", stream.read)
        while True:
            chunk = read_chunk(65536)
            if not chunk:
                break
            events.put((name, chunk))
        events.put((name, None))

    threads = [threading.Thread(target=read, args=(name, stream), daemon=True)
               for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))]
    for thread in threads:
        thread.start()
    started = last_progress = last_probe = time.monotonic()
    closed = set()
    try:
        while len(closed) < 2 or process.poll() is None:
            now = time.monotonic()
            if progress_probe is not None and now - last_probe >= progress_probe_interval:
                if progress_probe():
                    last_progress = time.monotonic()
                last_probe = time.monotonic()
                now = last_probe
            if now - last_progress > idle_timeout or now - started > total_timeout:
                raise RemoteVerificationError("network progress watchdog expired")
            wait_timeout = max(
                0.01,
                min(1, idle_timeout - (now - last_progress), total_timeout - (now - started)),
            )
            try:
                name, chunk = events.get(timeout=wait_timeout)
            except queue.Empty:
                continue
            if chunk is None:
                closed.add(name)
            else:
                handler = stdout_handler if name == "stdout" else stderr_handler
                if handler is not None:
                    handler(chunk)
                else:
                    buffers[name].extend(chunk)
                    if name == "stderr":
                        del buffers[name][:-65536]
                if progress_observer is None or progress_observer(name, chunk):
                    last_progress = time.monotonic()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    code = process.wait()
    if code:
        message = buffers["stderr"].decode("utf-8", "replace").strip()
        raise RemoteVerificationError(f"network command failed ({code}): {message}")
    return bytes(buffers["stdout"])


def _file_signature(path, *, deadline=None):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise RemoteVerificationError("upload transaction deadline expired while hashing source")
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _remote_signature(command, *, idle_timeout, total_timeout):
    deadline = time.monotonic() + total_timeout
    for attempt in range(3):
        checksum = hashlib.sha256()
        size = 0

        def consume(chunk):
            nonlocal size
            checksum.update(chunk)
            size += len(chunk)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteVerificationError("remote readback deadline expired")
        try:
            _run_watchdog(
                command,
                idle_timeout=idle_timeout,
                total_timeout=remaining,
                stdout_handler=consume,
                progress_observer=lambda name, chunk: name == "stdout" and bool(chunk),
            )
        except RemoteVerificationError:
            if attempt == 2:
                raise
        else:
            return size, checksum.hexdigest()


def _rclone_config_flags():
    configured = os.getenv("MAS_RCLONE_CONFIG") or os.getenv("RCLONE_CONFIG")
    return ["--config", configured] if configured else []


def _drive_credentials(remote, *, config_path=None, total_timeout=60):
    remote_name = remote.split(":", 1)[0]
    if (":" not in remote or not remote_name or any(char in remote_name for char in "/\\,")):
        raise RemoteVerificationError("Drive preflight requires a named rclone remote")
    if not shutil.which("rclone"):
        raise RemoteVerificationError("rclone is required for Drive preflight")
    configured = config_path or os.getenv("MAS_RCLONE_CONFIG") or os.getenv("RCLONE_CONFIG")
    if configured is None:
        response = _run_watchdog(["rclone", "config", "file"],
                                 idle_timeout=min(15, total_timeout), total_timeout=total_timeout)
        configured = response.decode("utf-8").strip().splitlines()[-1]
    config = configparser.ConfigParser(interpolation=None)
    try:
        with Path(configured).open(encoding="utf-8-sig") as handle:
            config.read_file(handle)
    except (OSError, UnicodeError, configparser.Error):
        raise RemoteVerificationError("Drive OAuth configuration cannot be inspected safely") from None
    section = dict(config[remote_name]) if remote_name in config else {}

    def option(name):
        return os.getenv(f"RCLONE_CONFIG_{remote_name.upper()}_{name.upper()}",
                         os.getenv(f"RCLONE_DRIVE_{name.upper()}", section.get(name, ""))).strip()

    if option("type") != "drive":
        raise RemoteVerificationError("Drive preflight requires a direct Drive remote")
    if not option("client_id") or not option("client_secret"):
        raise RemoteVerificationError("private Drive OAuth client is required; shared client is blocked")
    if option("service_account_file") or option("service_account_credentials"):
        raise RemoteVerificationError("Drive preflight requires the configured private OAuth identity")
    if option("scope") not in ("", "drive", "https://www.googleapis.com/auth/drive"):
        raise RemoteVerificationError("Drive OAuth scope cannot preserve existing delivery objects")
    try:
        token = json.loads(option("token"))
        refresh_token = token.get("refresh_token")
    except (ValueError, AttributeError):
        refresh_token = None
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RemoteVerificationError("Drive OAuth refresh identity is missing")
    if option("token_url") or option("auth_url") or option("client_credentials").lower() == "true":
        raise RemoteVerificationError("custom Drive OAuth endpoints or flows are not supported")
    identity = {"client_id": option("client_id"), "refresh_token": refresh_token,
                "root_folder_id": option("root_folder_id"), "team_drive": option("team_drive")}
    return {**identity, "client_secret": option("client_secret"), "remote_name": remote_name,
            "config_path": str(configured), "credential_identity_sha256": digest(identity)}


def drive_preflight(remote, *, required_bytes, config_path=None, total_timeout=60):
    if type(required_bytes) is not int or required_bytes < 0:
        raise RemoteVerificationError("Drive required byte count is invalid")
    started = time.monotonic()
    credentials = _drive_credentials(remote, config_path=config_path, total_timeout=total_timeout)
    remote_name = credentials["remote_name"]
    remaining = total_timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise RemoteVerificationError("Drive preflight deadline expired")
    quota = json.loads(_run_watchdog(
        ["rclone", "about", remote_name + ":", "--json", "--config", credentials["config_path"],
         "--contimeout", "15s", "--timeout", "30s", "--retries", "1", "--low-level-retries", "2"],
        idle_timeout=min(30, remaining), total_timeout=remaining))
    free = quota.get("free") if isinstance(quota, dict) else None
    if type(free) is not int or free < 0:
        raise RemoteVerificationError("Drive available space is UNKNOWN")
    if free < required_bytes:
        raise RemoteVerificationError("insufficient Drive space; retained objects will not be deleted")
    return {"status": "PASS", "checked_at": datetime.now(timezone.utc).isoformat(),
            "remote_name": remote_name, "required_bytes": required_bytes, "free_bytes": free,
            "client_id_sha256": digest(credentials["client_id"]),
            "credential_identity_sha256": credentials["credential_identity_sha256"],
            "write_access": "NOT_PROBED"}


class _RcloneProgress:
    def __init__(self):
        self.pending = b""
        self.bytes = 0
        self.completed = 0

    def __call__(self, name, chunk):
        if name != "stderr":
            return False
        self.pending += chunk
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()[-65536:]
        advanced = False
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            stats = event.get("stats") if isinstance(event, dict) else None
            if not isinstance(stats, dict):
                continue
            transferred = stats.get("bytes", 0)
            completed = stats.get("transfers", 0)
            if any(type(value) is not int or value < 0 for value in (transferred, completed)):
                continue
            if transferred > self.bytes or completed > self.completed:
                advanced = True
                self.bytes = max(self.bytes, transferred)
                self.completed = max(self.completed, completed)
        return advanced


class _UploadProgress:
    def __init__(self):
        self.pending = b""
        self.watermarks = {}

    def __call__(self, name, chunk):
        if name != "stderr":
            return False
        self.pending += chunk
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()[-65536:]
        advanced = False
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("phase") not in ("hash", "upload"):
                continue
            count = event.get("bytes")
            phase = event["phase"]
            if type(count) is int and count > self.watermarks.get(phase, 0):
                self.watermarks[phase] = count
                advanced = True
        return advanced


def _upload_resumable(source, remote, partial, expected_size, expected_sha, preflight,
                      *, idle_timeout, total_timeout, allow_session_create, on_prepared=None):
    deadline = time.monotonic() + total_timeout
    credentials = _drive_credentials(remote, total_timeout=min(60, total_timeout))
    if credentials["credential_identity_sha256"] != preflight["credential_identity_sha256"]:
        raise RemoteVerificationError("Drive OAuth identity changed after preflight")
    parent = remote.rsplit("/", 1)[0]
    if "/" in parent:
        container, folder_name = parent.rsplit("/", 1)
    else:
        remote_name, folder_name = parent.split(":", 1)
        container = remote_name + ":"
    folders = json.loads(_run_watchdog(
        ["rclone", "lsjson", container, "--dirs-only", "--max-depth", "1",
         "--config", credentials["config_path"],
         "--contimeout", "15s", "--timeout", "30s", "--retries", "1", "--low-level-retries", "2"],
        idle_timeout=min(30, idle_timeout), total_timeout=max(0.01, deadline - time.monotonic())))
    matches = ([item for item in folders
                if isinstance(item, dict) and item.get("Name") == folder_name]
               if isinstance(folders, list) else [])
    if (len(matches) != 1 or matches[0].get("IsDir") is not True
            or not isinstance(matches[0].get("ID"), str) or not matches[0]["ID"]):
        raise RemoteVerificationError("Drive destination folder identity is unavailable")
    folder = matches[0]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RemoteVerificationError("Drive session upload deadline expired")
    binding = {"source": str(Path(source).resolve()), "remote": remote, "partial": partial,
               "parent_id": folder["ID"], "bytes": expected_size, "sha256": expected_sha,
               "credential_identity_sha256": credentials["credential_identity_sha256"]}
    if on_prepared is not None:
        from .drive_resumable import prepare_session
        prepare_session(partial, binding, allow_create=allow_session_create)
        on_prepared()
    result = json.loads(_run_watchdog(
        [sys.executable, "-m", "mas.drive_resumable", "--source", str(Path(source).resolve()),
         "--remote", remote, "--partial", partial, "--parent-id", folder["ID"],
         "--size", str(expected_size), "--sha256", expected_sha,
         "--identity", credentials["credential_identity_sha256"],
         "--config", credentials["config_path"], "--total-timeout", str(remaining),
         "--idle-timeout", str(idle_timeout), *(["--allow-session-create"] if allow_session_create else [])],
        idle_timeout=idle_timeout, total_timeout=remaining, progress_observer=_UploadProgress()))
    binding = {"source": str(Path(source).resolve()), "remote": remote, "partial": partial,
               "parent_id": folder["ID"], "bytes": expected_size, "sha256": expected_sha,
               "credential_identity_sha256": credentials["credential_identity_sha256"]}
    if (not isinstance(result, dict)
            or set(result) != {"status", "bytes", "sha256", "object_id", "session_binding_sha256"}
            or result.get("status") != "UPLOAD_COMPLETE"
            or type(result.get("bytes")) is not int or result["bytes"] != expected_size
            or result.get("sha256") != expected_sha
            or not isinstance(result.get("object_id"), str) or not result["object_id"]
            or result.get("session_binding_sha256") != digest(binding)):
        raise RemoteVerificationError("Drive session completion evidence is invalid")
    return result


def upload_verified(source, remote, *, idle_timeout=120, total_timeout=3600,
                    preservation_receipt=None, require_drive_preflight=False):
    deadline = time.monotonic() + total_timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise RemoteVerificationError("upload transaction deadline expired")
        return value

    path = Path(source)
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise RemoteVerificationError(f"unsafe upload source: {path}")
    if not remote or "emergency" in remote.casefold():
        raise RemoteVerificationError("MAS_DRIVE_STRICT_REMOTE must name a strict destination")
    if require_drive_preflight and preservation_receipt is None:
        raise RemoteVerificationError("production Drive upload requires persistent preservation evidence")
    if not shutil.which("rclone"):
        raise RemoteVerificationError("rclone is required for Drive upload verification")
    expected_size, expected_sha = _file_signature(path, deadline=deadline)
    preflight = (drive_preflight(remote, required_bytes=0, total_timeout=min(60, remaining()))
                 if require_drive_preflight else None)

    def receipt():
        result = {"bytes": expected_size, "sha256": expected_sha, "remote": remote}
        if preflight is not None:
            result["preflight"] = preflight
        return result

    partial = remote + f".partial-{uuid.uuid4().hex}"
    common = ["--contimeout", "30s", "--timeout", "2m", "--retries", "1",
              "--low-level-retries", "3", "--stats", "10s", "--use-json-log",
              "--stats-log-level", "NOTICE", *_rclone_config_flags()]
    transaction = None
    transaction_path = None
    listing = []

    def save_transaction():
        if transaction_path is not None:
            atomic_json(transaction_path, {"payload": transaction, "sha256": digest(transaction)})

    def inventory(parent):
        result = json.loads(_run_watchdog(
            ["rclone", "lsjson", parent, "--files-only", "--max-depth", "1", *common],
            idle_timeout=idle_timeout, total_timeout=remaining()))
        if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
            raise RemoteVerificationError("invalid Drive inventory")
        return result

    def unique_object(items, filename):
        matches = [item for item in items if item.get("Name") == filename]
        if len(matches) > 1:
            raise RemoteVerificationError("ambiguous duplicate Drive filename; preserve all objects")
        return matches[0] if matches else None

    def signature(target):
        return _remote_signature(["rclone", "cat", target, *common],
                                 idle_timeout=idle_timeout, total_timeout=remaining())

    if preservation_receipt is not None:
        parent, filename = remote.rsplit("/", 1)
        binding = {"remote": remote, "bytes": expected_size, "sha256": expected_sha}
        evidence_path = Path(preservation_receipt)
        transaction_path = evidence_path.with_name(
            evidence_path.stem + "-upload-" + digest(binding) + ".json")
        if transaction_path.exists():
            envelope = read_json(transaction_path)
            transaction = envelope.get("payload")
            if (not isinstance(transaction, dict) or envelope.get("sha256") != digest(transaction)
                    or transaction.get("binding") != binding):
                raise RemoteVerificationError("Drive upload checkpoint binding changed")
            partial = transaction.get("partial", "")
            suffix = partial.removeprefix(remote + ".partial-")
            if (not partial.startswith(remote + ".partial-") or len(suffix) != 32
                    or any(char not in "0123456789abcdef" for char in suffix)):
                raise RemoteVerificationError("unsafe Drive upload checkpoint path")
            if type(transaction.get("upload_attempts")) is not int or transaction["upload_attempts"] < 0:
                raise RemoteVerificationError("invalid Drive upload checkpoint attempt count")
        else:
            transaction = {"binding": binding, "partial": partial, "upload_attempts": 0,
                           "status": "INVENTORY_PENDING"}
            save_transaction()
        if preflight is not None:
            transaction["preflight"] = preflight
            save_transaction()
        _run_watchdog(["rclone", "mkdir", parent, *common],
                      idle_timeout=idle_timeout, total_timeout=remaining())
        listing = inventory(parent)
        existing = unique_object(listing, filename)
        prior = None
        if existing:
            old_size, old_sha = signature(remote)
            prior = {"remote": remote, "bytes": old_size, "sha256": old_sha,
                     "object_id": existing.get("ID")}
            if not isinstance(prior["object_id"], str) or not prior["object_id"]:
                raise RemoteVerificationError("existing Drive object identity is unavailable")
        retained_pending = transaction.get("retained_pending")
        preservation_reconciled = False
        if retained_pending:
            if not isinstance(retained_pending, dict) or not isinstance(retained_pending.get("prior"), dict):
                raise RemoteVerificationError("invalid retained Drive checkpoint evidence")
            retained = retained_pending.get("retained_remote")
            if not isinstance(retained, str):
                raise RemoteVerificationError("invalid retained Drive checkpoint evidence")
            saved_prior = retained_pending["prior"]
            if (saved_prior.get("remote") != remote
                    or type(saved_prior.get("bytes")) is not int or saved_prior["bytes"] <= 0
                    or not isinstance(saved_prior.get("sha256"), str)
                    or len(saved_prior["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in saved_prior["sha256"])
                    or not isinstance(saved_prior.get("object_id"), str)
                    or not saved_prior["object_id"]):
                raise RemoteVerificationError("invalid retained Drive checkpoint evidence")
            retained_prefix = parent + "/.retained/" + saved_prior["sha256"] + "-"
            retained_suffix = retained.removeprefix(retained_prefix).removesuffix("/" + filename)
            if (not retained.startswith(retained_prefix) or not retained.endswith("/" + filename)
                    or len(retained_suffix) != 32
                    or any(char not in "0123456789abcdef" for char in retained_suffix)):
                raise RemoteVerificationError("unsafe retained Drive checkpoint path")
            if (not isinstance(retained_pending.get("attempt_path"), str)
                    or not isinstance(retained_pending.get("evidence"), dict)):
                raise RemoteVerificationError("invalid retained Drive checkpoint evidence")
            saved_attempt = Path(retained_pending["attempt_path"])
            attempt_suffix = saved_attempt.stem.removeprefix(evidence_path.stem + "-")
            if (saved_attempt.parent.resolve() != evidence_path.parent.resolve()
                    or saved_attempt.suffix != ".json" or len(attempt_suffix) != 32
                    or any(char not in "0123456789abcdef" for char in attempt_suffix)):
                raise RemoteVerificationError("unsafe Drive preservation receipt checkpoint path")
            retained_parent = retained.rsplit("/", 1)[0]
            _run_watchdog(["rclone", "mkdir", retained_parent, *common],
                          idle_timeout=idle_timeout, total_timeout=remaining())
            retained_listing = inventory(retained_parent)
            retained_object = unique_object(retained_listing, filename)
            if existing is not None and retained_object is not None:
                raise RemoteVerificationError("ambiguous retained Drive preservation state")
            if existing is None and retained_object is None:
                raise RemoteVerificationError("retained Drive preservation object is missing")
            current_object = existing if existing is not None else retained_object
            if current_object.get("ID") != saved_prior["object_id"]:
                raise RemoteVerificationError("retained Drive object identity changed")
            current_remote = remote if existing is not None else retained
            preserved = signature(current_remote)
            if preserved != (saved_prior["bytes"], saved_prior["sha256"]):
                raise RemoteVerificationError("retained Drive byte/SHA-256 readback mismatch")
            if existing is not None:
                _run_watchdog(["rclone", "moveto", remote, retained, "--immutable", *common],
                              idle_timeout=idle_timeout, total_timeout=remaining())
                retained_object = unique_object(inventory(retained.rsplit("/", 1)[0]), filename)
                if retained_object is None or retained_object.get("ID") != saved_prior["object_id"]:
                    raise RemoteVerificationError("retained Drive object identity changed")
                preserved = signature(retained)
                if preserved != (saved_prior["bytes"], saved_prior["sha256"]):
                    raise RemoteVerificationError("retained Drive byte/SHA-256 readback mismatch")
            retained_pending["status"] = "PRESERVED"
            atomic_json(retained_pending["attempt_path"], retained_pending["evidence"] | {"status": "PRESERVED"})
            atomic_json(evidence_path, retained_pending["evidence"] | {"status": "PRESERVED"})
            transaction.pop("retained_pending")
            save_transaction()
            prior = None
            preservation_reconciled = True
        if not preservation_reconciled:
            evidence = {"destination": remote, "prior": prior, "status": "INVENTORIED"}
            # Each attempt keeps its own inventory, including failures after a remote move.
            attempt_path = evidence_path.with_name(evidence_path.stem + "-" + uuid.uuid4().hex + ".json")
            atomic_json(attempt_path, evidence)
            atomic_json(evidence_path, evidence)
        if prior and (old_size, old_sha) == (expected_size, expected_sha):
            transaction["status"] = "VERIFIED_FINAL"
            save_transaction()
            return receipt()
        partial_object = unique_object(listing, partial.rsplit("/", 1)[-1])
        if preflight is not None and partial_object is None:
            if preflight["free_bytes"] < expected_size:
                raise RemoteVerificationError("insufficient Drive space; retained objects will not be deleted")
            preflight["required_bytes"] = expected_size
            save_transaction()
        if prior:
            retained = f"{parent}/.retained/{old_sha}-{uuid.uuid4().hex}/{filename}"
            evidence.update(status="PRESERVATION_PLANNED", retained_remote=retained)
            atomic_json(attempt_path, evidence)
            atomic_json(evidence_path, evidence)
            transaction["retained_pending"] = {"retained_remote": retained, "prior": prior,
                "attempt_path": str(attempt_path), "evidence": evidence}
            save_transaction()
            _run_watchdog(["rclone", "moveto", remote, retained, "--immutable", *common],
                          idle_timeout=idle_timeout, total_timeout=remaining())
            preserved = signature(retained)
            if preserved != (old_size, old_sha):
                raise RemoteVerificationError("retained Drive byte/SHA-256 readback mismatch")
            evidence["status"] = "PRESERVED"
            atomic_json(attempt_path, evidence)
            atomic_json(evidence_path, evidence)
            transaction.pop("retained_pending")
            save_transaction()
    partial_object = unique_object(listing, partial.rsplit("/", 1)[-1])
    if partial_object is not None and transaction is not None:
        object_id = partial_object.get("ID")
        expected_object_id = transaction.get("partial_object_id",
                            transaction.get("resumable", {}).get("object_id", object_id))
        if not object_id or expected_object_id != object_id:
            raise RemoteVerificationError("Drive partial object identity changed")
        transaction["partial_object_id"] = object_id
        save_transaction()
    else:
        if transaction is not None and transaction["upload_attempts"] >= 2 and not require_drive_preflight:
            raise RemoteVerificationError("Drive upload retry allowance exhausted; evidence preserved")
        if preflight is not None and preflight["free_bytes"] < expected_size:
            raise RemoteVerificationError("insufficient Drive space; retained objects will not be deleted")
        if transaction is not None:
            if not require_drive_preflight or transaction["upload_attempts"] == 0:
                transaction["upload_attempts"] += 1
            transaction["status"] = "UPLOAD_STARTED"
            save_transaction()
        if require_drive_preflight:
            allow_session_create = not transaction.get("native_session_started", False)
            def prepared():
                transaction["native_session_started"] = True
                save_transaction()
            completion = _upload_resumable(path, remote, partial, expected_size, expected_sha, preflight,
                                          idle_timeout=idle_timeout, total_timeout=remaining(),
                                          allow_session_create=allow_session_create, on_prepared=prepared)
            transaction["native_session_started"] = True
            transaction["resumable"] = completion
            save_transaction()
        else:
            _run_watchdog(["rclone", "copyto", str(path), partial, "--immutable", *common],
                          idle_timeout=idle_timeout, total_timeout=remaining(),
                          progress_observer=_RcloneProgress())
        if transaction is not None:
            partial_object = unique_object(inventory(parent), partial.rsplit("/", 1)[-1])
            if partial_object is None or not partial_object.get("ID"):
                raise RemoteVerificationError("uploaded Drive partial object identity is missing")
            if require_drive_preflight and partial_object["ID"] != completion["object_id"]:
                raise RemoteVerificationError("Drive session and partial object identities differ")
            transaction["partial_object_id"] = partial_object["ID"]
            transaction["status"] = "READBACK_PENDING"
            save_transaction()
    partial_size, partial_sha = signature(partial)
    if partial_size != expected_size or partial_sha != expected_sha:
        raise RemoteVerificationError("partial Drive byte/SHA-256 readback mismatch")
    _run_watchdog(["rclone", "moveto", partial, remote, "--immutable", *common],
                  idle_timeout=idle_timeout, total_timeout=remaining(),
                  progress_observer=_RcloneProgress())
    final_size, final_sha = signature(remote)
    if final_size != expected_size or final_sha != expected_sha:
        raise RemoteVerificationError("final Drive byte/SHA-256 readback mismatch")
    if transaction is not None:
        transaction["status"] = "VERIFIED_FINAL"
        save_transaction()
    return receipt()
