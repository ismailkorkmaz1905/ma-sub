import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path


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
    digest = hashlib.sha256()
    size = 0

    def consume(chunk):
        nonlocal size
        digest.update(chunk)
        size += len(chunk)

    _run_watchdog(
        command,
        idle_timeout=idle_timeout,
        total_timeout=total_timeout,
        stdout_handler=consume,
        progress_observer=lambda name, chunk: name == "stdout" and bool(chunk),
    )
    return size, digest.hexdigest()


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


def upload_verified(source, remote, *, idle_timeout=120, total_timeout=3600,
                    preservation_receipt=None):
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
    if not shutil.which("rclone"):
        raise RemoteVerificationError("rclone is required for Drive upload verification")
    expected_size, expected_sha = _file_signature(path, deadline=deadline)
    partial = remote + f".partial-{uuid.uuid4().hex}"
    common = ["--contimeout", "30s", "--timeout", "2m", "--retries", "3",
              "--low-level-retries", "3", "--stats", "10s", "--use-json-log",
              "--stats-log-level", "NOTICE"]
    if preservation_receipt is not None:
        from .reliability import atomic_json
        parent, filename = remote.rsplit("/", 1)
        _run_watchdog(["rclone", "mkdir", parent, *common],
                      idle_timeout=idle_timeout, total_timeout=remaining())
        listing = json.loads(_run_watchdog(
            ["rclone", "lsjson", parent, "--files-only", "--max-depth", "1", *common],
            idle_timeout=idle_timeout, total_timeout=remaining()))
        matches = [item for item in listing if item.get("Name") == filename]
        if len(matches) > 1:
            raise RemoteVerificationError("ambiguous duplicate Drive filename; preserve all objects")
        prior = None
        if matches:
            old_size, old_sha = _remote_signature(
                ["rclone", "cat", remote, *common], idle_timeout=idle_timeout,
                total_timeout=remaining())
            prior = {"remote": remote, "bytes": old_size, "sha256": old_sha,
                     "object_id": matches[0].get("ID")}
        evidence = {"destination": remote, "prior": prior, "status": "INVENTORIED"}
        # Each attempt keeps its own inventory, including failures after a remote move.
        evidence_path = Path(preservation_receipt)
        attempt_path = evidence_path.with_name(evidence_path.stem + "-" + uuid.uuid4().hex + ".json")
        atomic_json(attempt_path, evidence)
        atomic_json(evidence_path, evidence)
        if prior and (old_size, old_sha) == (expected_size, expected_sha):
            return {"bytes": expected_size, "sha256": expected_sha, "remote": remote}
        if prior:
            retained = f"{parent}/.retained/{old_sha}-{uuid.uuid4().hex}/{filename}"
            evidence.update(status="PRESERVATION_PLANNED", retained_remote=retained)
            atomic_json(attempt_path, evidence)
            atomic_json(evidence_path, evidence)
            _run_watchdog(["rclone", "moveto", remote, retained, "--immutable", *common],
                          idle_timeout=idle_timeout, total_timeout=remaining())
            preserved = _remote_signature(["rclone", "cat", retained, *common],
                                          idle_timeout=idle_timeout, total_timeout=remaining())
            if preserved != (old_size, old_sha):
                raise RemoteVerificationError("retained Drive byte/SHA-256 readback mismatch")
            evidence["status"] = "PRESERVED"
            atomic_json(attempt_path, evidence)
            atomic_json(evidence_path, evidence)
    _run_watchdog(["rclone", "copyto", str(path), partial, *common],
                  idle_timeout=idle_timeout, total_timeout=remaining(),
                  progress_observer=_RcloneProgress())
    partial_size, partial_sha = _remote_signature(
        ["rclone", "cat", partial, "--contimeout", "30s",
         "--timeout", "2m", "--retries", "3", "--low-level-retries", "3"],
        idle_timeout=idle_timeout,
        total_timeout=remaining(),
    )
    if partial_size != expected_size or partial_sha != expected_sha:
        raise RemoteVerificationError("partial Drive byte/SHA-256 readback mismatch")
    _run_watchdog(["rclone", "moveto", partial, remote, "--immutable", *common],
                  idle_timeout=idle_timeout, total_timeout=remaining(),
                  progress_observer=_RcloneProgress())
    final_size, final_sha = _remote_signature(
        ["rclone", "cat", remote, "--contimeout", "30s",
         "--timeout", "2m", "--retries", "3", "--low-level-retries", "3"],
        idle_timeout=idle_timeout,
        total_timeout=remaining(),
    )
    if final_size != expected_size or final_sha != expected_sha:
        raise RemoteVerificationError("final Drive byte/SHA-256 readback mismatch")
    return {"bytes": expected_size, "sha256": expected_sha, "remote": remote}
