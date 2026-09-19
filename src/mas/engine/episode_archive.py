import os
import stat
from pathlib import Path

from .download import sha256_file


VIDEO_SUFFIXES = frozenset(
    {".avi", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".ts", ".webm"}
)


def file_record(path: str | os.PathLike[str], episode_root: str | os.PathLike[str]) -> dict:
    root = Path(episode_root).resolve(strict=True)
    value = Path(path)
    try:
        mode = value.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"Artifact is missing: {value}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"Artifact is not a regular file: {value}")
    resolved = value.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"Artifact is outside the episode folder: {value}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"Artifact is empty: {value}")
    return {
        "relative_path": resolved.relative_to(root).as_posix(),
        "size_bytes": size,
        "sha256": sha256_file(resolved),
    }
