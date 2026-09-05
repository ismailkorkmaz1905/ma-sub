import argparse
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

from .config import episode_dir
from .engine.download import DownloadError, _validated_cookie_file
from .notify import notify, send_email
from .pipeline import run, status
from .runlog import RunLog
from .runpod_controller import run_remote_episode


def _runpod_preflight():
    failures = []

    for name in (
        "MAS_GMAIL_ADDRESS",
        "MAS_GMAIL_APP_PASSWORD",
        "MAS_YTDLP_COOKIES",
        "MAS_DRIVE_STRICT_REMOTE",
        "RUNPOD_POD_ID",
        "RUNPOD_API_KEY",
    ):
        if not os.getenv(name):
            failures.append(f"{name}: MISSING")

    try:
        cookie_path = _validated_cookie_file(os.getenv("MAS_YTDLP_COOKIES"))
        records = []
        if cookie_path:
            records = [
                line
                for line in Path(cookie_path).read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
        if not records or not any("youtube.com" in line.split("\t", 1)[0] for line in records):
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

    remote = os.getenv("MAS_DRIVE_STRICT_REMOTE", "")
    if remote and ":" not in remote:
        failures.append("rclone_remote: invalid remote path")
    elif remote and shutil.which("rclone"):
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
                reachable = subprocess.run(
                    ["rclone", "lsd", remote],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if reachable.returncode != 0:
                    failures.append("rclone_remote: unreachable")
                else:
                    print("rclone_remote: OK")
        except subprocess.TimeoutExpired:
            failures.append("rclone_remote: timeout")

    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


def doctor(strict_runpod=False):
    print("mas doctor")
    for executable in ("python", "ffmpeg", "ffprobe", "git", "rclone"):
        print(f"{executable}: {'OK' if shutil.which(executable) else 'MISSING'}")
    if strict_runpod:
        missing_tools = [
            executable
            for executable in ("python", "ffmpeg", "ffprobe", "git", "rclone")
            if not shutil.which(executable)
        ]
        if missing_tools:
            print("required_tools: MISSING " + ", ".join(missing_tools), file=sys.stderr)
            return 1
        return _runpod_preflight()
    try:
        import torch
        print(f"torch: {torch.__version__}; cuda={torch.cuda.is_available()}")
    except Exception:
        print("torch: MISSING")
    return 0


def test():
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


def notify_test():
    result = send_email(None, "bildirim testi", "MAS e-posta bildirimi calisiyor.")
    if result.get("status") != "sent":
        raise RuntimeError("MAS_GMAIL_ADDRESS and MAS_GMAIL_APP_PASSWORD are required")
    print(f"Test email sent to {result['recipient']}")
    return 0


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
    commands.add_parser("test")
    commands.add_parser("notify-test")
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
                result = doctor(True) if args.strict_runpod else doctor()
            elif args.command == "test":
                result = test()
            elif args.command == "notify-test":
                result = notify_test()
            else:
                result = clean(args.episode, args.destroy)
            if run_log:
                run_log.finish(result)
            return result
        except Exception as exc:
            if run_log:
                run_log.record_exception()
            if getattr(args, "episode", None):
                details = f"{type(exc).__name__}: {exc}"
                if run_log:
                    details += f"\nLog: {run_log.path}"
                notify(args.episode, "pipeline basarisiz", details)
            print(f"FAILED STAGE: {args.command.upper()}\nCAUSE: {exc}\nCHECKPOINT PRESERVED: yes", file=sys.stderr)
            if getattr(args, "episode", None):
                print(f"SAFE RETRY:\n./mas run {args.episode}", file=sys.stderr)
            if run_log:
                run_log.finish(1)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
