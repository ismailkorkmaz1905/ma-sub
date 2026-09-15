import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if os.name == "posix":
    import fcntl
else:
    fcntl = None


class RemoteJobError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _identity(episode, commit, input_sha256=None):
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise RemoteJobError("episode must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise RemoteJobError("commit must be a full lowercase Git SHA")
    input_sha256 = input_sha256 or "0" * 64
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise RemoteJobError("input SHA-256 must be lowercase hexadecimal")
    token = hashlib.sha256(f"{episode}\n{commit}\n{input_sha256}\n".encode()).hexdigest()
    return {"episode": episode, "commit": commit, "input_sha256": input_sha256, "token": token}


def _paths(root, episode, token=None):
    episode_root = (Path(root).resolve() / "EPISODES" / f"Muhtemel Ask {episode}.Bolum").resolve()
    jobs_root = episode_root / "work" / "remote-jobs"
    job_root = jobs_root / token if token else jobs_root
    return {
        "episode": episode_root,
        "job": job_root,
        "claim": episode_root / "work" / "remote-job.claim.json",
        "lock": episode_root / "work" / "remote-job.lock",
        "request": job_root / "request.json",
        "state": job_root / "state.json",
        "log": episode_root / "logs" / (f"remote-job-{token}.log" if token else "remote-job.log"),
    }


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _proc_start_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def _pid_matches(pid, token=None, start_ticks=None):
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid < 1
            or (token is not None and (not isinstance(token, str)
                or not re.fullmatch(r"[0-9a-f]{64}", token)))):
        return False
    current_ticks = _proc_start_ticks(pid)
    if current_ticks is None or (start_ticks is not None and current_ticks != start_ticks):
        return False
    command_line = Path(f"/proc/{pid}/cmdline")
    if token is not None:
        try:
            return token.encode() in command_line.read_bytes().split(b"\0")
        except OSError:
            return False
    return True


def _claim_live(claim):
    if not isinstance(claim, dict):
        return False
    worker = claim.get("worker")
    if isinstance(worker, dict) and _pid_matches(worker.get("pid"), start_ticks=worker.get("start_ticks")):
        return True
    supervisor = claim.get("supervisor")
    identity = claim.get("identity")
    token = identity.get("token") if isinstance(identity, dict) else None
    if isinstance(supervisor, dict) and _pid_matches(
            supervisor.get("pid"), token, supervisor.get("start_ticks")):
        return True
    starter = claim.get("starter")
    return isinstance(starter, dict) and _pid_matches(
        starter.get("pid"), start_ticks=starter.get("start_ticks"))


def _descendant_process_groups(parent_pid):
    parents = {}
    groups = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tail = (entry / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
            pid = int(entry.name)
            parents[pid] = int(tail[1])
            groups[pid] = int(tail[2])
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    descendants = set()
    frontier = {parent_pid}
    while frontier:
        children = {pid for pid, parent in parents.items() if parent in frontier and pid not in descendants}
        descendants.update(children)
        frontier = children
    records = {}
    for pid in descendants:
        group = groups.get(pid, 0)
        ticks = _proc_start_ticks(pid)
        if group > 1 and ticks is not None:
            records.setdefault(group, []).append((pid, ticks))
    return records


def _terminate_worker_tree(worker):
    groups = _descendant_process_groups(worker.pid)
    worker_ticks = _proc_start_ticks(worker.pid)
    if worker_ticks is not None:
        groups.setdefault(worker.pid, []).append((worker.pid, worker_ticks))
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        worker.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    for group, members in groups.items():
        live = False
        for pid, ticks in members:
            try:
                live = _pid_matches(pid, start_ticks=ticks) and os.getpgid(pid) == group
            except ProcessLookupError:
                live = False
            if live:
                break
        if live:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if worker.poll() is None:
        worker.wait(timeout=10)


def status_job(root, episode, commit, input_sha256=None):
    identity = _identity(episode, commit, input_sha256)
    paths = _paths(root, episode, identity["token"])
    claim = _read_json(paths["claim"])
    state = _read_json(paths["state"])
    claim_identity = claim.get("identity") if claim else None
    claim_live = _claim_live(claim)
    live = claim_live and claim_identity == identity
    if live:
        status = "RUNNING" if claim.get("worker") or claim.get("supervisor") else "STARTING"
    elif claim_live:
        status = "RUNNING_OTHER"
    elif state and state.get("identity") == identity and state.get("status") == "EXITED":
        status = "EXITED"
    elif state and state.get("identity") == identity and state.get("status") in {"STARTING", "RUNNING"}:
        status = "LOST"
    else:
        status = "ABSENT"
    return {
        "identity": identity,
        "status": status,
        "live": live,
        "conflicting_identity": claim_identity if claim_live and not live else None,
        "supervisor_pid": (claim.get("supervisor") or {}).get("pid") if claim else None,
        "worker_pid": state.get("worker_pid") if state else None,
        "exit_code": state.get("exit_code") if status == "EXITED" else None,
        "state_path": str(paths["state"]),
        "log_path": str(paths["log"]),
        "progress": _read_json(paths["episode"] / "logs" / "useful-progress.json"),
        "started_at": state.get("started_at") if state else None,
        "deadline_at": state.get("deadline_at") if state else None,
        "ended_at": state.get("ended_at") if state else None,
        "failure": state.get("failure") if state else None,
    }


def start_job(root, episode, commit, command, input_sha256=None, *, recover_lost=False):
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise RemoteJobError("remote detached jobs require Linux /proc")
    identity = _identity(episode, commit, input_sha256)
    paths = _paths(root, episode, identity["token"])
    paths["job"].mkdir(parents=True, exist_ok=True)
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = _read_json(paths["claim"])
        terminal = _read_json(paths["state"])
        if terminal and terminal.get("identity") == identity and terminal.get("status") == "EXITED":
            return {**status_job(root, episode, commit, input_sha256), "started": False,
                    "already_running": False, "terminal": True}
        if (terminal and terminal.get("identity") == identity
                and terminal.get("status") in {"STARTING", "RUNNING"}
                and not _claim_live(existing)):
            if not recover_lost:
                raise RemoteJobError("prior remote job ownership was lost; reconcile before restart")
            recovery_path = paths["job"] / "recoveries.json"
            recovery = _read_json(recovery_path) or {"identity": identity, "records": []}
            if (recovery.get("identity") != identity or not isinstance(recovery.get("records"), list)):
                raise RemoteJobError("remote job recovery evidence is invalid")
            recovery["records"].append({
                "recovered_at": _now(),
                "state": terminal,
                "claim": existing,
            })
            _atomic_json(recovery_path, recovery)
        if _claim_live(existing):
            if existing.get("identity") != identity:
                raise RemoteJobError("another remote job is already running for this episode")
            return {**status_job(root, episode, commit, input_sha256), "started": False,
                    "already_running": True, "terminal": False}
        request = {"identity": identity, "command": list(command), "created_at": _now()}
        if not request["command"] or not all(isinstance(item, str) and item for item in request["command"]):
            raise RemoteJobError("remote job command is empty or invalid")
        starter = {"pid": os.getpid(), "start_ticks": _proc_start_ticks(os.getpid())}
        _atomic_json(paths["claim"], {"identity": identity, "starter": starter, "claimed_at": _now()})
        _atomic_json(paths["request"], request)
        _atomic_json(paths["state"], {"identity": identity, "status": "STARTING",
                                      "started_at": _now()})
        supervisor = subprocess.Popen(
            [sys.executable, "-m", "mas.remote_job", "_supervise", "--root", str(Path(root).resolve()),
             "--episode", str(episode), "--commit", commit, "--token", identity["token"]],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True)
        _atomic_json(paths["claim"], {"identity": identity, "starter": starter,
                    "supervisor": {"pid": supervisor.pid, "start_ticks": _proc_start_ticks(supervisor.pid)},
                    "claimed_at": _now()})
    return {**status_job(root, episode, commit, input_sha256), "started": True,
            "already_running": False, "terminal": False}


def supervise(root, episode, commit, token):
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise RemoteJobError("remote detached jobs require Linux /proc")
    paths = _paths(root, episode, token)
    with paths["lock"].open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = _read_json(paths["request"])
        identity = request.get("identity") if request else None
        if not isinstance(identity, dict) or identity != _identity(
                episode, commit, identity.get("input_sha256")) or token != identity["token"]:
            raise RemoteJobError("supervisor identity token mismatch")
        claim = _read_json(paths["claim"])
        supervisor = claim.get("supervisor") if claim else None
        if (not request or request.get("identity") != identity or not claim
                or claim.get("identity") != identity or not isinstance(supervisor, dict)
                or supervisor.get("pid") != os.getpid()):
            raise RemoteJobError("remote job request or claim identity mismatch")
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    runtime_seconds = int(os.getenv("MAS_MAX_RUNTIME_SECONDS", "14400"))
    if runtime_seconds < 1:
        raise RemoteJobError("MAS_MAX_RUNTIME_SECONDS must be positive")
    deadline_at = (datetime.now(timezone.utc) + timedelta(seconds=runtime_seconds)).isoformat()
    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    worker = None
    exit_code = 70
    failure = None
    try:
        with paths["log"].open("ab", buffering=0) as output:
            worker = subprocess.Popen(request["command"], cwd=Path(root).resolve(), stdin=subprocess.DEVNULL,
                                      stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                                      env={**os.environ, 'MAS_REMOTE_JOB_TOKEN': identity['token']})
            worker_record = {"pid": worker.pid, "start_ticks": _proc_start_ticks(worker.pid)}
            _atomic_json(paths["claim"], {"identity": identity,
                        "supervisor": {"pid": os.getpid(), "start_ticks": _proc_start_ticks(os.getpid())},
                        "worker": worker_record, "claimed_at": claim.get("claimed_at")})
            _atomic_json(paths["state"], {"identity": identity, "status": "RUNNING",
                                        "supervisor_pid": os.getpid(), "worker_pid": worker.pid,
                                        "started_at": started_at, "deadline_at": deadline_at})
            exit_code = worker.wait(timeout=runtime_seconds + 60)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.TimeoutExpired):
            exit_code = 124
        if worker and worker.poll() is None:
            _terminate_worker_tree(worker)
    _atomic_json(paths["state"], {"identity": identity, "status": "EXITED",
                                  "supervisor_pid": os.getpid(),
                                  "worker_pid": worker.pid if worker else None,
                                  "started_at": started_at, "deadline_at": deadline_at,
                                  "ended_at": _now(), "exit_code": exit_code, "failure": failure})
    current = _read_json(paths["claim"])
    if current and current.get("identity") == identity:
        paths["claim"].unlink(missing_ok=True)
    return exit_code


def checkpoint_manifest(root, episode, commit, input_sha256=None, deadline_seconds=60, *, diagnostics=False):
    if not isinstance(deadline_seconds, (int, float)) or not 0 < deadline_seconds <= 60:
        raise RemoteJobError("snapshot deadline must be within (0, 60] seconds")
    identity = _identity(episode, commit, input_sha256)
    paths = _paths(root, episode, identity["token"])
    prepare = paths["episode"] / "prepare"
    episode_root = paths["episode"]
    candidates = list((prepare / "primary_asr").glob("*.json"))
    candidates += [prepare / name for name in (
        "raw_asr_v2.recovery.json", "raw_asr_v2.json", "raw_asr_v2.done.json")]
    candidates += [episode_root / "source" / "source.url",
                   episode_root / "source" / "download.done.json",
                   prepare / "audio.done.json", prepare / "audio_review_v2.json",
                   episode_root / "translation_output" / f"Muhtemel Ask {episode}.Bolum_ID_TRANSLATED.zip.workspace.json",
                   episode_root / "work" / "state.json"]
    candidates += list((episode_root / "work" / "encoder-qualification").glob("*/technical-qualification.json"))
    outbox = sorted((episode_root / "work/notification-outbox").glob("*.json"))
    if len(outbox) > 256:
        raise RemoteJobError("notification snapshot count exceeds the bounded manifest")
    candidates += outbox
    part_id = None
    current_path = episode_root / 'work/current-part.json'
    if current_path.is_file():
        from .reliability import digest as evidence_digest
        current = _read_json(current_path)
        data = current.get('data') if isinstance(current, dict) else None
        if (not isinstance(data, dict) or current.get('sha256') != evidence_digest(data)
                or data.get('episode') != episode
                or not re.fullmatch(r'part-[0-9]{3}', str(data.get('part_id', '')))):
            raise RemoteJobError('current part snapshot identity is invalid')
        part_id = data['part_id']
        child = episode_root / 'parts' / part_id
        candidates += [current_path, episode_root / 'work/part-plan.json', episode_root / 'work/part-vad.json',
                       child / 'work/state.json', child / 'prepare/audio-part.done.json',
                       child / 'prepare/raw_asr_v2.recovery.json', child / 'prepare/raw_asr_v2.json',
                       child / 'prepare/raw_asr_v2.done.json', child / 'prepare/audio_review_v2.json']
        candidates += list((child / 'prepare/primary_asr').glob('*.json'))
    if diagnostics:
        name = f"Muhtemel Ask {episode}.Bolum"
        candidates = [episode_root / relative for relative in (
            f"translation_input/{name}_TR_CORRECTION_PACK.zip",
            "prepare/audio_review_v2.json", "prepare/audio_review_v2.recovery.json",
            "prepare/forced_alignment_units/resume-identity.json",
            "prepare/forced_alignment_units/components/latest-conflict-failure.json",
            "source/download.done.json", "prepare/audio.done.json", "prepare/raw_asr_v2.done.json",
            f"translation_output/{name}_TR_TEXT_CORRECTED.zip",
            f"translation_output/{name}_TR_CORRECTED.zip",
        )]
        if part_id is not None:
            from .retry_authorization import retry_diagnostic_layout
            layout = retry_diagnostic_layout(episode, part_id)
            candidates = [episode_root / relative for relative in layout['files']]
            candidates += [episode_root / prefix / relative for prefix in layout['prefixes'] for relative in (
                'resume-identity.json', 'components/latest-conflict-failure.json')]
            candidates.append(episode_root / f'parts/{part_id}/prepare/audio_review_v2.recovery.json')
    missing = [path.relative_to(episode_root).as_posix() for path in candidates if not path.is_file()]
    files = []
    unstable = []
    snapshot_root = paths["job"] / "snapshots"
    deadline = time.monotonic() + deadline_seconds
    resolved_candidates = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        path = candidate.resolve()
        try:
            path.relative_to(episode_root)
        except ValueError:
            raise RemoteJobError("checkpoint path escaped the episode root") from None
        resolved_candidates.add(path)
    for path in sorted(resolved_candidates):
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        paths["job"].mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=paths["job"])
        try:
            with path.open("rb") as stream, os.fdopen(descriptor, "wb") as snapshot:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("checkpoint snapshot deadline exceeded")
                    size += len(block)
                    digest.update(block)
                    snapshot.write(block)
                snapshot.flush()
                os.fsync(snapshot.fileno())
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                unstable.append(str(path.relative_to(episode_root)).replace("\\", "/"))
                continue
            sha256 = digest.hexdigest()
            snapshot_root.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_root / (sha256 + path.suffix)
            if snapshot_path.exists():
                os.unlink(temporary)
            else:
                os.replace(temporary, snapshot_path)
            files.append({"relative_path": str(path.relative_to(episode_root)).replace("\\", "/"),
                          "snapshot_path": str(snapshot_path),
                          "size_bytes": size, "sha256": sha256})
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    marker = _read_json(episode_root / "source" / "download.done.json")
    video_record = (marker.get("outputs") or {}).get("video") if marker else None
    if not diagnostics and isinstance(video_record, dict):
        try:
            video = Path(video_record["path"]).resolve()
            video.relative_to((episode_root / "source").resolve())
            size = video.stat().st_size
        except (KeyError, OSError, ValueError, TypeError):
            pass
        else:
            sha256 = video_record.get("sha256")
            if size == video_record.get("size_bytes") and re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
                files.append({"relative_path": str(video.relative_to(episode_root)).replace("\\", "/"),
                              "snapshot_path": str(video), "size_bytes": size,
                              "sha256": sha256, "immutable_source": True})
    files.sort(key=lambda item: item["relative_path"])
    return {"identity": identity, "created_at": _now(), "files": files, "unstable": unstable,
            **({'part_id': part_id} if part_id is not None else {}),
            **({"missing": missing, "kind": "diagnostics"} if diagnostics else {})}


def read_log(root, episode, commit, offset=0, max_bytes=65536, input_sha256=None):
    identity = _identity(episode, commit, input_sha256)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise RemoteJobError("log offset must be a nonnegative integer")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 65536:
        raise RemoteJobError("log byte limit must be within [1, 65536]")
    path = _paths(root, episode, identity["token"])["log"]
    if not path.is_file():
        return {"offset": offset, "next_offset": offset, "text": "", "eof": True}
    size = path.stat().st_size
    position = min(offset, size)
    with path.open("rb") as stream:
        stream.seek(position)
        payload = stream.read(max_bytes)
    return {"offset": position, "next_offset": position + len(payload),
            "text": payload.decode("utf-8", "replace"),
            "eof": position + len(payload) >= size}


def poll_job(root, episode, commit, offset=0, max_bytes=65536, input_sha256=None,
             deadline_seconds=60):
    return {
        "status": status_job(root, episode, commit, input_sha256),
        "log": read_log(root, episode, commit, offset, max_bytes, input_sha256),
        "checkpoints": checkpoint_manifest(
            root, episode, commit, input_sha256, deadline_seconds),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m mas.remote_job")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("start", "status", "checkpoints", "diagnostics", "logs", "poll", "_supervise"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--episode", required=True, type=int)
        command.add_argument("--commit", required=True)
        command.add_argument("--input-sha256")
    commands.choices["start"].add_argument("--source-url")
    commands.choices["start"].add_argument("--recover-lost", action="store_true")
    commands.choices["logs"].add_argument("--offset", type=int, default=0)
    commands.choices["logs"].add_argument("--max-bytes", type=int, default=65536)
    commands.choices["checkpoints"].add_argument("--deadline-seconds", type=int, default=60)
    commands.choices["diagnostics"].add_argument("--deadline-seconds", type=int, default=60)
    commands.choices["poll"].add_argument("--offset", type=int, default=0)
    commands.choices["poll"].add_argument("--max-bytes", type=int, default=65536)
    commands.choices["poll"].add_argument("--deadline-seconds", type=int, default=60)
    commands.choices["_supervise"].add_argument("--token", required=True)
    args = parser.parse_args(argv)
    if args.action == "start":
        command = ["bash", str(Path(args.root).resolve() / "runpod" / "run-episode.sh"), str(args.episode)]
        if args.source_url:
            command += ["--source-url", args.source_url]
        result = start_job(args.root, args.episode, args.commit, command, args.input_sha256,
                           recover_lost=args.recover_lost)
    elif args.action == "status":
        result = status_job(args.root, args.episode, args.commit, args.input_sha256)
    elif args.action in {"checkpoints", "diagnostics"}:
        result = checkpoint_manifest(args.root, args.episode, args.commit, args.input_sha256,
                                     args.deadline_seconds, diagnostics=args.action == "diagnostics")
    elif args.action == "logs":
        result = read_log(args.root, args.episode, args.commit, args.offset, args.max_bytes,
                          args.input_sha256)
    elif args.action == "poll":
        result = poll_job(args.root, args.episode, args.commit, args.offset, args.max_bytes,
                          args.input_sha256, args.deadline_seconds)
    else:
        return supervise(args.root, args.episode, args.commit, args.token)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
