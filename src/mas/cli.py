import argparse
import shutil
import subprocess
import sys

from .config import episode_dir
from .notify import notify, send_email
from .pipeline import run, status


def doctor():
    print("mas doctor")
    for executable in ("python", "ffmpeg", "ffprobe", "git", "rclone"):
        print(f"{executable}: {'OK' if shutil.which(executable) else 'MISSING'}")
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
    run_parser.add_argument("--stop-after", type=int, choices=(1, 2, 3), help=argparse.SUPPRESS)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("episode", type=int)
    commands.add_parser("doctor")
    commands.add_parser("test")
    commands.add_parser("notify-test")
    clean_parser = commands.add_parser("clean")
    clean_parser.add_argument("episode", type=int)
    clean_parser.add_argument("--destroy", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args.episode, args.source_url, args.fixture, args.stop_after)
        if args.command == "status":
            return status(args.episode)
        if args.command == "doctor":
            return doctor()
        if args.command == "test":
            return test()
        if args.command == "notify-test":
            return notify_test()
        return clean(args.episode, args.destroy)
    except Exception as exc:
        if getattr(args, "episode", None):
            notify(args.episode, "pipeline basarisiz", f"{type(exc).__name__}: {exc}")
        print(f"FAILED STAGE: {args.command.upper()}\nCAUSE: {exc}\nCHECKPOINT PRESERVED: yes", file=sys.stderr)
        if getattr(args, "episode", None):
            print(f"SAFE RETRY:\n./mas run {args.episode}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
