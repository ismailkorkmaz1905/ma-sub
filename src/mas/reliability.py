"""EP12 operational guards. No model calls, text edits, or silent fallbacks.

The 4-hour deadline measures wall time, including external handoffs. It is a
budget, not a completion guarantee. A deadline must never turn failed QA into
PASS. Long commands run in their own process group so timeout kills children.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

POLICY_VERSION = 'ep12-1'


class IntegrityError(ValueError):
    """Non-retryable input, output, schema, or checkpoint failure."""


class BudgetExceeded(TimeoutError):
    """Checkpoint remains valid; operator must explicitly extend the budget."""


class OperationFailed(RuntimeError):
    def __init__(self, stage: str, cause: str, *, retryable: bool = False):
        super().__init__(f'{stage}: {cause}')
        self.stage, self.retryable = stage, retryable


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')


def digest(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before touching the destination. Reject NaN/Infinity.
    encoded = (json.dumps(data, ensure_ascii=False, indent=2,
                          sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
        if os.name == 'posix':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass(frozen=True)
class RunBudget:
    started_at: str
    limit_seconds: float = 4 * 60 * 60

    def __post_init__(self) -> None:
        _positive(self.limit_seconds, 'limit_seconds')
        start = datetime.fromisoformat(self.started_at)
        if start.tzinfo is None:
            raise ValueError('started_at must include a timezone')

    def remaining(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError('now must include a timezone')
        elapsed = (now - datetime.fromisoformat(self.started_at)).total_seconds()
        if elapsed < -5:
            raise IntegrityError('clock precedes the saved run start')
        return max(0.0, self.limit_seconds - max(0.0, elapsed))

    def check(self, now: datetime | None = None) -> float:
        left = self.remaining(now)
        if left <= 0:
            raise BudgetExceeded('4-hour wall-time budget exhausted; checkpoint preserved')
        return left


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    operation_timeout_seconds: float = 120
    total_timeout_seconds: float = 300
    backoff_seconds: float = 2

    def __post_init__(self) -> None:
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or not 1 <= self.attempts <= 5:
            raise ValueError('attempts must be an integer from 1 to 5')
        for name in ('operation_timeout_seconds', 'total_timeout_seconds', 'backoff_seconds'):
            _positive(getattr(self, name), name)


def bounded_retry(operation: Callable[[float], object], *, policy: RetryPolicy,
                  remaining_seconds: float, clock=time.monotonic, sleep=time.sleep):
    """Retry only explicitly classified transient failures, never validation.

    operation receives the permitted timeout and MUST enforce it (run_command
    does). Auth/schema/hash errors must be non-retryable. No nested retry loops.
    """
    _positive(remaining_seconds, 'remaining_seconds')
    deadline = clock() + min(policy.total_timeout_seconds, remaining_seconds)
    for attempt in range(policy.attempts):
        left = deadline - clock()
        if left <= 0:
            raise BudgetExceeded('operation retry budget exhausted')
        try:
            return operation(min(policy.operation_timeout_seconds, left))
        except OperationFailed as exc:
            if not exc.retryable or attempt + 1 == policy.attempts:
                raise
            delay = min(policy.backoff_seconds * 2 ** attempt, max(0, deadline - clock()))
            if delay:
                sleep(delay)
    raise AssertionError('unreachable')


def run_command(command: Sequence[str], *, stage: str, log_path: Path,
                timeout_seconds: float, heartbeat: Callable[[], None] | None = None,
                stdout_path: Path | None = None, cwd: Path | None = None) -> None:
    """Bound a complete process tree; stream output to disk, not RAM.

    An alive process is not proof of progress. Heartbeat refreshes only elapsed
    time. Percent/ETA must come from measured bytes, audio time, or finished UIDs.
    """
    _positive(timeout_seconds, 'timeout_seconds')
    if not command or isinstance(command, (str, bytes)):
        raise ValueError('command must be an argv sequence; shell strings are forbidden')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with log_path.open('ab') as log:
        log.write((json.dumps({'event':'started', 'stage':stage, 'timeout_seconds':timeout_seconds})+'\n').encode())
        log.flush()
        output = Path(stdout_path).open('wb') if stdout_path else None
        try:
            process = subprocess.Popen(list(command), stdin=subprocess.DEVNULL,
                                       stdout=output or log, stderr=log, cwd=cwd,
                                       start_new_session=(os.name == 'posix'))
            try:
                while process.poll() is None:
                    if heartbeat:
                        heartbeat()
                    left = deadline - time.monotonic()
                    if left <= 0:
                        log.write(b'{"event":"timeout"}\n')
                        log.flush()
                        raise OperationFailed(stage, 'operation timed out', retryable=True)
                    time.sleep(min(0.1, left))
                if process.returncode:
                    # Unknown process failures are not assumed transient.
                    raise OperationFailed(stage, f'exit code {process.returncode}; see {log_path.name}')
            finally:
                if process.poll() is None:
                    if os.name == 'posix':
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        if os.name == 'posix':
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait(timeout=2)
        finally:
            if output:
                output.close()


def artifact(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise IntegrityError(f'artifact is absent or is a symlink: {path.name}')
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': file_digest(path)}


def make_checkpoint(*, stage: str, run_id: str, inputs: Mapping, config: Mapping,
                    outputs: Sequence[Path], completed_uids: Sequence[str],
                    git_commit: str, code_sha256: str, container_digest: str | None,
                    started_at: str, finished_at: str, provider: str, gpu: str | None,
                    model_versions: Mapping) -> dict:
    if not outputs:
        raise IntegrityError('a completed stage must have at least one output')
    if len(completed_uids) != len(set(completed_uids)):
        raise IntegrityError('completed UID list contains duplicates')
    body = dict(stage=stage, run_id=run_id, status='PASS', inputs=dict(inputs),
                config=dict(config), outputs=[artifact(p) for p in outputs],
                completed_uids=list(completed_uids), record_count=len(completed_uids),
                git_commit=git_commit, code_sha256=code_sha256,
                container_digest=container_digest, started_at=started_at,
                finished_at=finished_at, provider=provider, gpu=gpu,
                model_versions=dict(model_versions), policy_version=POLICY_VERSION)
    return {'data': body, 'sha256': digest(body)}


def checkpoint_reusable(checkpoint: Mapping, *, stage: str, inputs: Mapping,
                        config: Mapping, code_sha256: str) -> bool:
    body = checkpoint.get('data')
    if not isinstance(body, dict) or checkpoint.get('sha256') != digest(body):
        raise IntegrityError('checkpoint checksum mismatch')
    if (body.get('status') != 'PASS' or body.get('stage') != stage
            or body.get('inputs') != dict(inputs) or body.get('config') != dict(config)
            or body.get('code_sha256') != code_sha256
            or body.get('policy_version') != POLICY_VERSION):
        return False
    outputs = body.get('outputs')
    if not isinstance(outputs, list) or not outputs:
        return False
    for item in outputs:
        path = Path(item['path'])
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != item['bytes']
                or file_digest(path) != item['sha256']):
            return False
    return True


class UnitJournal:
    """Per-UID atomic results; crash loses at most the unit currently running.

    The binding includes input/config/code hashes. Different input gets a new
    cache directory; old work is not deleted or silently rebound.
    """
    def __init__(self, root: Path, binding: Mapping):
        self.binding = digest(dict(binding))
        self.root = Path(root) / self.binding
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, uid: str) -> Path:
        if not isinstance(uid, str) or not uid.strip():
            raise IntegrityError('UID must be a nonempty string')
        return self.root / (hashlib.sha256(uid.encode('utf-8')).hexdigest() + '.json')

    def read(self, uid: str):
        p = self._path(uid)
        if not p.exists():
            return None
        try:
            saved = json.loads(p.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise IntegrityError(f'corrupt checkpoint for UID {uid}') from exc
        body = saved.get('data')
        if not isinstance(body, dict) or saved.get('sha256') != digest(body):
            raise IntegrityError(f'checkpoint checksum mismatch for UID {uid}')
        if body.get('uid') != uid or body.get('binding') != self.binding:
            raise IntegrityError(f'checkpoint identity mismatch for UID {uid}')
        return body['result']

    def write(self, uid: str, result: object) -> None:
        if result is None:
            raise IntegrityError('unit result cannot be None; use an explicit decision object')
        body = {'uid': uid, 'binding': self.binding, 'result': result}
        atomic_json(self._path(uid), {'data': body, 'sha256': digest(body)})
