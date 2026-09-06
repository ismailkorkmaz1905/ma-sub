"""Measured terminal progress and one atomically updated status object."""
from __future__ import annotations

import sys
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from .reliability import atomic_json


def mark_work_progress(stage, *, completed=None):
    path = os.getenv("MAS_PROGRESS_FILE")
    if path:
        atomic_json(Path(path), {"stage": stage, "completed": completed,
                                "timestamp": datetime.now(timezone.utc).isoformat()})


class Progress:
    def __init__(self, *, episode: int, run_id: str, stage: str,
                 status_path: Path, total: int | None = None,
                 stream=None, interval_seconds: float = 0.5, initial_processed: int = 0):
        if total is not None and (type(total) is not int or total <= 0):
            raise ValueError('total must be a positive integer or None')
        if type(initial_processed) is not int or initial_processed < 0 or (total is not None and initial_processed > total):
            raise ValueError('invalid restored progress count')
        if interval_seconds <= 0:
            raise ValueError('interval_seconds must be positive')
        self.episode, self.run_id, self.stage = episode, run_id, stage
        self.status_path, self.total = Path(status_path), total
        self.stream = stream if stream is not None else sys.stderr
        self.interval_seconds = interval_seconds
        self.started = self.last_progress = time.monotonic()
        self.processed = self.initial_processed = initial_processed
        self.state, self.last_uid, self.last_checkpoint_at = 'RUNNING', None, None
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._render_lock = threading.RLock()
        self._thread = None
        self._last_print = float('-inf')
        self.last_error = None

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(0, time.monotonic() - self.started)
            newly_processed = self.processed - self.initial_processed
            eta = ((self.total - self.processed) * elapsed / newly_processed
                   if self.total and newly_processed and self.state == 'RUNNING' else None)
            return dict(episode=self.episode, run_id=self.run_id, stage=self.stage,
                        status=self.state, processed=self.processed, total=self.total,
                        percent=round(100 * self.processed / self.total, 2) if self.total else None,
                        elapsed_sec=round(elapsed, 2), eta_sec=round(eta, 2) if eta is not None else None,
                        eta_is_estimate=True, last_uid=self.last_uid,
                        last_checkpoint_at=self.last_checkpoint_at,
                        seconds_since_progress=round(time.monotonic() - self.last_progress, 2))

    def advance(self, processed: int, *, uid: str | None = None,
                checkpoint_saved: bool = False) -> None:
        with self._lock:
            if type(processed) is not int or processed < self.processed:
                raise ValueError('processed must be a monotonic integer')
            if self.total is not None and processed > self.total:
                raise ValueError('processed cannot exceed total')
            if processed > self.processed:
                self.last_progress = time.monotonic()
            self.processed, self.last_uid = processed, uid
            if checkpoint_saved:
                self.last_checkpoint_at = datetime.now(timezone.utc).isoformat()
        self.refresh()

    def set_state(self, state: str) -> None:
        if state not in {'PENDING', 'RUNNING', 'PASS', 'FAIL', 'BLOCKED'}:
            raise ValueError('invalid stage state')
        with self._lock:
            if state == 'PASS' and self.total is not None and self.processed != self.total:
                raise ValueError('cannot PASS an incomplete stage')
            self.state = state
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> None:
        with self._render_lock:
            self._refresh(force=force)

    def _refresh(self, *, force: bool = False) -> None:
        data = self.snapshot()
        atomic_json(self.status_path, data)
        tty = bool(getattr(self.stream, 'isatty', lambda: False)())
        now = time.monotonic()
        if not force and not tty and now - self._last_print < 10:
            return
        self._last_print = now
        percent = data['percent']
        if percent is None:
            marker = int(data['elapsed_sec'] * 2) % 20
            bar = '-' * marker + '>' + '-' * (19 - marker)
            count = 'toplam henüz bilinmiyor'
        else:
            filled = min(20, int(percent / 5))
            bar = '=' * filled + '-' * (20 - filled)
            count = f"{percent:.1f}% | {data['processed']}/{data['total']}"
        eta_label = f"{data['eta_sec']:.0f} sn" if data['eta_sec'] is not None else 'henüz bilinmiyor'
        message = (f"{self.stage} [{bar}] {count} | {data['status']} | "
                   f"geçen {data['elapsed_sec']:.0f} sn | tahmini kalan {eta_label} | "
                   f"son ilerleme {data['seconds_since_progress']:.0f} sn önce")
        self.stream.write(('\r' if tty else '') + message + ('\x1b[K' if tty else '\n'))
        self.stream.flush()

    def _loop(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.refresh()
            except OSError as exc:
                # Observability failure must be surfaced to the owning stage.
                self.last_error = exc
                self._stop.set()

    def __enter__(self):
        self.refresh(force=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if exc is not None:
            with self._lock:
                self.state = 'FAIL'
        self.refresh(force=True)
        if getattr(self.stream, 'isatty', lambda: False)():
            self.stream.write('\n')
        if self.last_error is not None and exc is None:
            raise self.last_error
        return False
