import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

from .config import episode_dir


class _Tee:
    def __init__(self, stream, log, lock):
        self.stream = stream
        self.log = log
        self.lock = lock
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass

    def write(self, value):
        with self.lock:
            self.stream.write(value)
            self.log.write(value)
            self.log.flush()
        return len(value)

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_commit():
    configured = os.getenv("MAS_GIT_COMMIT")
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _safe_argv(argv):
    values = []
    redact_next = False
    for value in argv:
        if redact_next:
            values.append("<provided>")
            redact_next = False
        elif value == "--source-url":
            values.append(value)
            redact_next = True
        elif value.startswith("--source-url="):
            values.append("--source-url=<provided>")
        else:
            values.append(value)
    return values


class RunLog:
    def __init__(self, episode, argv):
        self.episode = episode
        self.argv = argv
        self.started = time.monotonic()
        self.started_at = _utc_now()
        self.directory = episode_dir(episode) / ".mas"
        self.path = self.directory / "run.log"
        self.file = None
        self.stdout = None
        self.stderr = None
        self.finished = False
        self.previous_started_at = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", buffering=1)
        self.previous_started_at = os.environ.get("MAS_RUN_STARTED_AT")
        os.environ["MAS_RUN_STARTED_AT"] = self.started_at
        lock = threading.RLock()
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self.stdout, self.file, lock)
        sys.stderr = _Tee(self.stderr, self.file, lock)
        self.event(
            "run_started",
            argv=_safe_argv(self.argv),
            git_commit=_git_commit(),
            configured={
                "youtube_cookies": bool(os.getenv("MAS_YTDLP_COOKIES")),
                "drive": bool(os.getenv("MAS_DRIVE_STRICT_REMOTE")),
                "runpod": bool(os.getenv("RUNPOD_POD_ID") and os.getenv("RUNPOD_API_KEY")),
            },
        )
        print(f"[LOG] {self.path}")
        return self

    def event(self, event, **details):
        record = {"timestamp": _utc_now(), "event": event, "episode": self.episode, **details}
        self.file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def record_exception(self):
        self.event("run_exception")
        traceback.print_exc(file=self.file)
        self.file.flush()

    def finish(self, exit_code):
        self.event(
            "run_finished",
            exit_code=exit_code,
            elapsed_seconds=round(time.monotonic() - self.started, 3),
        )
        self.finished = True

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None and not self.finished:
                self.event("run_exception")
                traceback.print_exception(exc_type, exc, tb, file=self.file)
                self.finish(130 if issubclass(exc_type, KeyboardInterrupt) else 1)
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.file.close()
        finally:
            if self.previous_started_at is None:
                os.environ.pop("MAS_RUN_STARTED_AT", None)
            else:
                os.environ["MAS_RUN_STARTED_AT"] = self.previous_started_at
