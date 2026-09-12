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
from .notify import notify, send_email
from .pipeline import run, status
from .runlog import RunLog
from .runpod_controller import run_remote_episode


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


def doctor(strict_runpod=False, runpod_worker=False):
    print("mas doctor")
    for executable in ("python", "ffmpeg", "ffprobe", "git", "rclone"):
        print(f"{executable}: {'OK' if shutil.which(executable) else 'MISSING'}")
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
    return 0


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
    doctor_parser.add_argument("--strict-runpod", action="store_true")
    doctor_parser.add_argument("--strict-runpod-worker", action="store_true")
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
                if args.strict_runpod_worker:
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
                    details += f"\nLog: {run_log.path}"
                details += f"\nSonraki adım: logu inceleyip ./mas run {args.episode} komutuyla güvenli devam edin."
                notify(args.episode, "çalıştırma başarısız", details)
            print(f"FAILED STAGE: {args.command.upper()}\nCAUSE: {exc}\nCHECKPOINT PRESERVED: yes", file=sys.stderr)
            if args.command == "subtitle-pilot":
                print("PILOT ONLY: no automatic full-run retry; inspect preserved evidence", file=sys.stderr)
            elif getattr(args, "episode", None):
                print(f"SAFE RETRY:\n./mas run {args.episode}", file=sys.stderr)
            if run_log:
                run_log.finish(1)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
