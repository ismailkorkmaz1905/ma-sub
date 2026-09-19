import argparse
import importlib.util
import json
import shutil
import subprocess
import sys

from .pipeline import run, status, status_summary


def doctor():
    failed = False
    for executable in ("ffmpeg", "ffprobe"):
        available = bool(shutil.which(executable))
        print(f"{executable}: {'OK' if available else 'MISSING'}")
        failed = failed or not available
    for module in ("yaml", "yt_dlp", "faster_whisper", "ctranslate2"):
        available = importlib.util.find_spec(module) is not None
        print(f"{module}: {'OK' if available else 'MISSING'}")
        failed = failed or not available
    if importlib.util.find_spec("ctranslate2"):
        import ctranslate2

        gpu_count = ctranslate2.get_cuda_device_count()
        print(f"cuda_gpus: {gpu_count}")
    else:
        gpu_count = 0
    if shutil.which("rclone"):
        print("rclone: OK")
    else:
        print("rclone: OPTIONAL (only needed for Drive upload)")
    return 1 if failed or gpu_count < 1 else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mas")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="start or resume an episode")
    run_parser.add_argument("episode", type=int)
    source = run_parser.add_mutually_exclusive_group()
    source.add_argument("--source-url")
    source.add_argument("--source")
    run_parser.add_argument("--subtitles-only", action="store_true")

    status_parser = commands.add_parser("status", help="show the current step")
    status_parser.add_argument("episode", type=int)
    status_parser.add_argument("--json", action="store_true")

    commands.add_parser("doctor", help="check ffmpeg and GPU ASR readiness")
    commands.add_parser("test", help="run local tests")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(
                args.episode,
                source_url=args.source_url,
                source_file=args.source,
                subtitles_only=args.subtitles_only,
            )
        if args.command == "status":
            if args.json:
                print(json.dumps(status(args.episode), ensure_ascii=False, indent=2))
                return 0
            return status_summary(args.episode)
        if args.command == "doctor":
            return doctor()
        return subprocess.call([sys.executable, "-m", "pytest", "-q"])
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
