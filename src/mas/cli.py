import argparse
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .audio_review_ui import run_audio_review_ui
from .config import episode_dir
from .engine.download import DownloadError, _validated_cookie_file
from .notify import enqueue_notification, send_email
from .pipeline import run, status
from .runlog import RunLog
from .runpod_controller import RunPodCleanupRequired, drain_cli_notifications, run_remote_episode


def _runpod_preflight(*, worker_only=False):
    failures = []

    required = [
        "MAS_GMAIL_ADDRESS",
        "MAS_GMAIL_APP_PASSWORD",
        "RUNPOD_POD_ID",
        "RUNPOD_API_KEY",
    ]
    if not worker_only:
        required.append("MAS_DRIVE_STRICT_REMOTE")
    for name in required:
        if not os.getenv(name):
            failures.append(f"{name}: MISSING")

    cookie_value = os.getenv("MAS_YTDLP_COOKIES")
    try:
        cookie_path = _validated_cookie_file(cookie_value)
        records = []
        if cookie_path:
            records = [
                line
                for line in Path(cookie_path).read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
        if cookie_path is None:
            print("youtube_cookies: NOT_SET")
        elif not records or not any("youtube.com" in line.split("\t", 1)[0] for line in records):
            failures.append("youtube_cookies: no YouTube cookie records")
        else:
            print("youtube_cookies: OK")
    except (DownloadError, OSError, UnicodeError) as exc:
        failures.append(f"youtube_cookies: {exc}")

    try:
        import torch

        if not torch.cuda.is_available():
            failures.append("cuda: UNAVAILABLE")
        else:
            print("cuda: OK")
    except Exception:
        failures.append("cuda: torch unavailable")

    remote = os.getenv("MAS_DRIVE_STRICT_REMOTE", "") if not worker_only else ""
    if not worker_only and remote and ":" not in remote:
        failures.append("rclone_remote: invalid remote path")
    elif not worker_only and remote and shutil.which("rclone"):
        remote_name = remote.split(":", 1)[0] + ":"
        try:
            configured = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            names = configured.stdout.splitlines() if configured.returncode == 0 else []
            if remote_name not in names:
                failures.append("rclone_remote: not configured")
            else:
                failure = "unreachable"
                for attempt in range(1, 4):
                    print(f"[PREFLIGHT] Drive check attempt {attempt}/3", flush=True)
                    try:
                        reachable = subprocess.run(
                            ["rclone", "lsd", remote],
                            capture_output=True,
                            timeout=60,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        failure = "timeout"
                    else:
                        if reachable.returncode == 0:
                            print("rclone_remote: OK")
                            break
                        failure = "unreachable"
                    if attempt < 3:
                        time.sleep(2 ** (attempt - 1))
                else:
                    failures.append(f"rclone_remote: {failure}")
        except subprocess.TimeoutExpired:
            failures.append("rclone_remote: listremotes timeout")

    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


def doctor_controller():
    from . import runpod_controller as controller
    from .engine.burned_mp4 import _qsv_hardware
    phase = "local_configuration"
    try:
        for executable in ("git", "ssh", "scp", "ffmpeg", "ffprobe", "rclone"):
            if not shutil.which(executable):
                raise RuntimeError("controller tool missing: " + executable)
        values = controller._required_environment()
        commit, config = controller._local_preflight(values)
        public_key = Path(values["MAS_RUNPOD_SSH_KEY"] + ".pub")
        if not public_key.is_file() or not public_key.read_text(encoding="utf-8").strip():
            raise RuntimeError("controller SSH public key is missing or empty")
        controller._configured_runtime_image()
        quote_path = os.getenv("MAS_RUNPOD_STORAGE_QUOTE")
        if not quote_path:
            raise RuntimeError("MAS_RUNPOD_STORAGE_QUOTE is missing")
        controller.load_storage_quote(Path(quote_path))
        controller._production_priority()
        execution = os.getenv("MAS_DELIVERY_EXECUTION_PLAN", "local-qsv-v1")
        if execution == "local-qsv-v1":
            _qsv_hardware(timeout_seconds=55)
        elif execution != "remote-nvenc-v1":
            raise RuntimeError("unsupported MAS_DELIVERY_EXECUTION_PLAN")
        phase = "drive_readiness"
        readiness = controller.drive_preflight(values["MAS_DRIVE_STRICT_REMOTE"],
            required_bytes=0, config_path=config, total_timeout=60)
        print(f"controller_preflight: PASS; commit={commit}; execution={execution}")
        print(f"local_disk_free_bytes: {shutil.disk_usage(controller.ROOT).free}")
        print(f"drive_free_bytes: {readiness['free_bytes']}")
        print("scope: read-only readiness and synthetic encoder; no Pod acquired")
        print("not_verified: image startup, real GPU inference, episode sizing, subtitle quality, delivery speed")
        return 0
    except Exception as exc:
        # OAuth/provider failures can contain credentials. Only our local
        # validation errors are printable; never render arbitrary network errors.
        message = (str(exc) if phase == "local_configuration"
                   and isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__)
        print("controller_preflight: FAIL at " + phase + ": " + message, file=sys.stderr)
        return 1


def doctor(strict_runpod=False, runpod_worker=False, controller=False):
    print("mas doctor")
    if controller:
        return doctor_controller()
    missing = []
    for executable in ("python", "ffmpeg", "ffprobe", "git", "rclone"):
        available = bool(shutil.which(executable))
        print(f"{executable}: {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(executable)
    if strict_runpod or runpod_worker:
        required_tools = ("python", "ffmpeg", "ffprobe", "git")
        if not runpod_worker:
            required_tools += ("rclone",)
        missing_tools = [
            executable
            for executable in required_tools
            if not shutil.which(executable)
        ]
        if missing_tools:
            print("required_tools: MISSING " + ", ".join(missing_tools), file=sys.stderr)
            return 1
        return _runpod_preflight(worker_only=runpod_worker)
    try:
        import torch
        print(f"torch: {torch.__version__}; cuda={torch.cuda.is_available()}")
    except Exception:
        print("torch: MISSING")
        missing.append("torch")
    print("scope: tool inventory only; not production readiness")
    return 1 if missing else 0


def test():
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


def notify_test():
    result = send_email(None, "bildirim testi", "MAS e-posta bildirimi çalışıyor.")
    if result.get("status") != "sent":
        raise RuntimeError("MAS_GMAIL_ADDRESS and MAS_GMAIL_APP_PASSWORD are required")
    print(f"Test email sent to {result['recipient']}")
    return 0


def _should_notify_run_failure(exc):
    if getattr(exc, "_mas_notification_sent", False):
        return False
    message = str(exc).lower()
    if any(
        text in message
        for text in (
            "not enough free gpus",
            "no free gpu",
            "capacity-bound",
            "remote pipeline failed with exit code",
        )
    ):
        return False
    return True


def clean(episode, destroy=False):
    target = episode_dir(episode) / "work"
    print(("DELETE " if destroy else "DRY-RUN ") + str(target))
    if destroy:
        shutil.rmtree(target, ignore_errors=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mas")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("episode", type=int)
    run_parser.add_argument("--source-url")
    run_parser.add_argument("--fixture", action="store_true")
    run_parser.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    run_parser.add_argument("--stop-after", type=int, choices=(1, 2, 3), help=argparse.SUPPRESS)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("episode", type=int)
    doctor_parser = commands.add_parser("doctor")
    doctor_modes = doctor_parser.add_mutually_exclusive_group()
    doctor_modes.add_argument("--strict-runpod", action="store_true")
    doctor_modes.add_argument("--strict-runpod-worker", action="store_true")
    doctor_modes.add_argument("--controller", action="store_true")
    commands.add_parser("test")
    commands.add_parser("notify-test")
    pilot_parser = commands.add_parser("subtitle-pilot")
    pilot_parser.add_argument("episode", type=int)
    pilot_parser.add_argument("--pack")
    pilot_parser.add_argument("--uid", action="append", default=[])
    pilot_parser.add_argument("--audio")
    pilot_parser.add_argument("--evidence")
    pilot_parser.add_argument("--model-dir")
    pilot_parser.add_argument("--diarization-model-dir")
    pilot_parser.add_argument("--source-sha256")
    pilot_parser.add_argument("--source-audio")
    pilot_parser.add_argument("--offset-ms", type=int, default=0)
    review_audio_parser = commands.add_parser("review-audio")
    review_audio_parser.add_argument("episode", type=int)
    emergency_parser = commands.add_parser("emergency-segment")
    emergency_parser.add_argument("episode", type=int)
    emergency_actions = emergency_parser.add_subparsers(dest="action", required=True)
    emergency_prepare = emergency_actions.add_parser("prepare")
    emergency_prepare.add_argument("--corrected-zip")
    emergency_finalize = emergency_actions.add_parser("finalize")
    emergency_finalize.add_argument("--translations", required=True)
    clean_parser = commands.add_parser("clean")
    clean_parser.add_argument("episode", type=int)
    clean_parser.add_argument("--destroy", action="store_true")
    args = parser.parse_args(argv)
    context = RunLog(args.episode, ["mas", *(argv or sys.argv[1:])]) if args.command == "run" else nullcontext()
    with context as run_log:
        try:
            if args.command == "run":
                if args.fixture or args.local or args.stop_after is not None:
                    result = run(args.episode, args.source_url, args.fixture, args.stop_after)
                else:
                    result = run_remote_episode(args.episode, args.source_url)
            elif args.command == "status":
                result = status(args.episode)
            elif args.command == "doctor":
                if args.controller:
                    result = doctor(controller=True)
                elif args.strict_runpod_worker:
                    result = doctor(runpod_worker=True)
                elif args.strict_runpod:
                    result = doctor(strict_runpod=True)
                else:
                    result = doctor()
            elif args.command == "test":
                result = test()
            elif args.command == "notify-test":
                result = notify_test()
            elif args.command == "subtitle-pilot":
                from .subtitle.pilot_command import run_pilot_command
                result = run_pilot_command(args)
            elif args.command == "review-audio":
                result = run_audio_review_ui(args.episode)
            elif args.command == "emergency-segment":
                from .emergency_segment import run_emergency_segment_command
                result = run_emergency_segment_command(args)
            else:
                result = clean(args.episode, args.destroy)
            if run_log:
                run_log.finish(result)
            return result
        except Exception as exc:
            if run_log:
                run_log.record_exception()
            if args.command != "subtitle-pilot" and getattr(args, "episode", None) and _should_notify_run_failure(exc):
                details = f"Sonuç: çalıştırma tamamlanamadı.\nHata: {type(exc).__name__}: {exc}"
                if run_log:
                    details += f"\nSon log: {run_log.directory / 'LATEST'}"
                if isinstance(exc, RunPodCleanupRequired):
                    details += "\nSonraki adım: owned Pod yokluğunu dışarıdan doğrulayın; doğrulamadan yeniden başlatmayın."
                else:
                    details += f"\nSonraki adım: logu inceleyip ./mas run {args.episode} komutuyla güvenli devam edin."
                enqueue_notification(args.episode, "çalıştırma başarısız", details,
                                     root=run_log.directory.parent if run_log else episode_dir(args.episode),
                                     kind="terminal")
                if (args.command == "run" and not args.fixture and not args.local
                        and args.stop_after is None and not os.getenv("MAS_REMOTE_JOB_TOKEN")):
                    drain_cli_notifications(args.episode)
            print(f"FAILED STAGE: {args.command.upper()}\nCAUSE: {exc}\nCHECKPOINT PRESERVED: yes", file=sys.stderr)
            if isinstance(exc, RunPodCleanupRequired):
                print("DO NOT RELAUNCH: externally verify owned Pod absence first", file=sys.stderr)
            elif args.command == "subtitle-pilot":
                print("PILOT ONLY: no automatic full-run retry; inspect preserved evidence", file=sys.stderr)
            elif getattr(args, "episode", None):
                print(f"SAFE RETRY:\n./mas run {args.episode}", file=sys.stderr)
            if run_log:
                run_log.finish(1)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
