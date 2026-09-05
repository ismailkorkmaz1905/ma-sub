"""Verified post-finalization MKV archival and exact-plan cleanup.

The destructive half of this module is intentionally narrow: it acts on one
exact episode directory, never follows links, unlinks only files captured in a
fresh plan, and removes only empty intermediate directories.  A receipt kept
outside the episode makes a partially or fully cleaned archive independently
verifiable after FINALIZE's report has gone.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from collections.abc import Callable
from typing import Any, Mapping

from .download import atomic_write_json, sha256_file, sha256_json, utc_now_iso
from .mux import compute_av_stream_hashes, mux_softsubs, verify_mkv_roundtrip


RECEIPT_VERSION = 1
PLAN_VERSION = 1
VIDEO_SUFFIXES = frozenset(
    {".avi", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".ts", ".webm"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArchiveError(RuntimeError):
    """Raised before an unsafe, unverifiable, or ambiguous archive action."""


def _episode_identity(episode: int) -> tuple[int, str]:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise ArchiveError("EPISODE must be a positive integer")
    return episode, f"Muhtemel Ask {episode}.Bolum"


def expected_cleanup_confirmation(episode: int) -> str:
    """Return the exact, episode-bound cleanup confirmation phrase."""

    episode, _ = _episode_identity(episode)
    return f"DELETE EPISODE {episode} INTERMEDIATES"


def validate_episode_root(episode_root: str | os.PathLike[str], episode: int) -> Path:
    """Require the exact direct child of the project's ``EPISODES`` folder."""

    _, episode_name = _episode_identity(episode)
    root = Path(episode_root)
    if root.name != episode_name:
        raise ArchiveError(
            f"Episode folder must be named exactly {episode_name!r}; got {root.name!r}"
        )
    if root.parent.name != "EPISODES" or root.parent.parent.name != "Muhtemel_Ask_Subtitles":
        raise ArchiveError(
            "Episode folder must be a direct child of Muhtemel_Ask_Subtitles/EPISODES"
        )
    for label, path in (
        ("project folder", root.parent.parent),
        ("EPISODES folder", root.parent),
        ("episode folder", root),
    ):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise ArchiveError(f"{label} does not exist: {path}") from exc
        if stat.S_ISLNK(mode):
            raise ArchiveError(f"Refusing symlinked {label}: {path}")
        if not stat.S_ISDIR(mode):
            raise ArchiveError(f"{label} is not a directory: {path}")
    resolved_parent = root.parent.resolve(strict=True)
    resolved = root.resolve(strict=True)
    if resolved.parent != resolved_parent:
        raise ArchiveError("Resolved episode folder is not a direct child of EPISODES")
    return resolved


def _relative_path(root: Path, raw: Any, label: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ArchiveError(f"{label} has an invalid relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArchiveError(f"{label} escapes or is not a normalized relative path: {raw!r}")
    relative = pure.as_posix()
    candidate = root.joinpath(*pure.parts)
    # Parent resolution catches symlinked ancestors even when the leaf is absent.
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArchiveError(f"{label} parent does not exist: {relative}") from exc
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ArchiveError(f"{label} escapes the episode folder: {relative}")
    return relative, candidate


def _record_shape(record: Any, label: str) -> tuple[str, int, str]:
    if not isinstance(record, Mapping):
        raise ArchiveError(f"{label} record is missing or invalid")
    relative = record.get("relative_path")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if not isinstance(relative, str):
        raise ArchiveError(f"{label} relative_path is missing")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ArchiveError(f"{label} size_bytes must be a positive integer")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ArchiveError(f"{label} sha256 is invalid")
    return relative, size, digest


def verify_recorded_file(
    episode_root: str | os.PathLike[str],
    record: Mapping[str, Any],
    label: str,
    *,
    expected_relative_path: str | None = None,
) -> Path:
    """Verify a recorded file's safe path, type, size and SHA-256."""

    root = Path(episode_root).resolve(strict=True)
    raw_relative, expected_size, expected_digest = _record_shape(record, label)
    relative, path = _relative_path(root, raw_relative, label)
    if expected_relative_path is not None and relative != expected_relative_path:
        raise ArchiveError(
            f"{label} path mismatch: expected {expected_relative_path!r}, got {relative!r}"
        )
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ArchiveError(f"{label} is missing: {relative}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ArchiveError(f"{label} is not a regular non-symlink file: {relative}")
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ArchiveError(f"{label} resolves outside the episode folder: {relative}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArchiveError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise ArchiveError(
            f"{label} SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )
    return path


def file_record(path: str | os.PathLike[str], episode_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a receipt-compatible record for one regular episode file."""

    root = Path(episode_root).resolve(strict=True)
    value = Path(path)
    try:
        mode = value.lstat().st_mode
    except FileNotFoundError as exc:
        raise ArchiveError(f"Archive artifact is missing: {value}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ArchiveError(f"Archive artifact is not a regular file: {value}")
    resolved = value.resolve(strict=True)
    if root not in resolved.parents:
        raise ArchiveError(f"Archive artifact is outside the episode folder: {value}")
    if resolved.stat().st_size <= 0:
        raise ArchiveError(f"Archive artifact is empty: {value}")
    return {
        "relative_path": resolved.relative_to(root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ArchiveError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ArchiveError(f"{label} is not a regular non-symlink file: {path}")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ArchiveError(f"{label} is unexpectedly large")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"{label} root must be a JSON object")
    return value


def _receipt_path(
    receipt_path: str | os.PathLike[str], root: Path, episode_name: str
) -> Path:
    value = Path(receipt_path)
    expected = (
        root.parent.parent
        / "ARCHIVE_REPORTS"
        / f"{episode_name}_ARCHIVE_RECEIPT.json"
    )
    if Path(os.path.abspath(value)) != expected:
        raise ArchiveError(
            f"Receipt path must be exactly {expected}"
        )
    if value.is_symlink():
        raise ArchiveError(f"Refusing symlinked archive receipt: {value}")
    archive_dir = expected.parent
    if archive_dir.exists() or archive_dir.is_symlink():
        mode = archive_dir.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ArchiveError(f"Refusing unsafe ARCHIVE_REPORTS path: {archive_dir}")
        if archive_dir.resolve(strict=True).parent != root.parent.parent:
            raise ArchiveError("ARCHIVE_REPORTS resolves outside the project folder")
    return expected


def _report_inputs(root: Path, episode: int, report: Mapping[str, Any]) -> dict[str, Any]:
    _, episode_name = _episode_identity(episode)
    if report.get("report_version") != 1:
        raise ArchiveError("FINALIZATION report version mismatch")
    if report.get("status") != "PASS":
        raise ArchiveError("FINALIZATION report is not PASS")
    if report.get("episode") != episode or report.get("episode_name") != episode_name:
        raise ArchiveError("FINALIZATION report episode identity mismatch")
    inputs = report.get("input_files")
    outputs = report.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ArchiveError("FINALIZATION report file records are missing")

    source_record = inputs.get("source_video")
    source = verify_recorded_file(root, source_record, "Source video")
    if source.suffix.casefold() not in VIDEO_SUFFIXES:
        raise ArchiveError(f"Recorded source does not use a supported video suffix: {source.name}")
    expected = {
        "id_srt": f"final/{episode_name}.id-final.srt",
        "tr_srt": f"final/{episode_name}.tr-final.srt",
        "id_infuse_sidecar": f"source/{source.stem}-id.srt",
        "tr_infuse_sidecar": f"source/{source.stem}-tr.srt",
    }
    paths: dict[str, Path] = {}
    records: dict[str, Mapping[str, Any]] = {}
    for key, expected_relative in expected.items():
        record = outputs.get(key)
        path = verify_recorded_file(
            root, record, key, expected_relative_path=expected_relative
        )
        paths[key] = path
        records[key] = record
    if records["id_srt"].get("sha256") != records["id_infuse_sidecar"].get("sha256"):
        raise ArchiveError("Indonesian canonical SRT and Infuse sidecar differ")
    if records["tr_srt"].get("sha256") != records["tr_infuse_sidecar"].get("sha256"):
        raise ArchiveError("Turkish canonical SRT and Infuse sidecar differ")

    return {
        "source": "finalization_report",
        "episode": episode,
        "episode_name": episode_name,
        "episode_root": root,
        "report_path": root / "final" / f"{episode_name}_FINALIZATION_REPORT.json",
        "receipt": None,
        "source_video": source,
        "source_video_record": dict(source_record),
        "id_srt": paths["id_srt"],
        "tr_srt": paths["tr_srt"],
        "id_infuse_sidecar": paths["id_infuse_sidecar"],
        "tr_infuse_sidecar": paths["tr_infuse_sidecar"],
        "srt_records": {key: dict(value) for key, value in records.items()},
        "mkv": root / "final" / f"{episode_name} - Endonezce + Turkce.mkv",
        "source_av_stream_hashes": None,
    }


def _receipt_inputs(root: Path, episode: int, receipt: Mapping[str, Any]) -> dict[str, Any]:
    _, episode_name = _episode_identity(episode)
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        raise ArchiveError("Archive receipt version mismatch")
    if receipt.get("status") not in {"READY", "PASS"}:
        raise ArchiveError("Archive receipt status is neither READY nor PASS")
    if receipt.get("episode") != episode or receipt.get("episode_name") != episode_name:
        raise ArchiveError("Archive receipt episode identity mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArchiveError("Archive receipt artifacts are missing")
    expected = {
        "mkv": f"final/{episode_name} - Endonezce + Turkce.mkv",
        "id_srt": f"final/{episode_name}.id-final.srt",
        "tr_srt": f"final/{episode_name}.tr-final.srt",
    }
    verified: dict[str, Path] = {}
    for key, expected_relative in expected.items():
        verified[key] = verify_recorded_file(
            root, artifacts.get(key), key, expected_relative_path=expected_relative
        )

    source_record = artifacts.get("source_video")
    source_relative, _, _ = _record_shape(source_record, "Source video")
    _, source = _relative_path(root, source_relative, "Source video")
    if source.is_symlink():
        raise ArchiveError("Receipt source video path is a symlink")
    if source.exists():
        source = verify_recorded_file(root, source_record, "Source video")
    else:
        policy = receipt.get("cleanup")
        if not isinstance(policy, Mapping) or policy.get("keep_source_video") is not False:
            raise ArchiveError("Archive receipt requires a source video which is missing")

    sidecar_expected = {
        "id_infuse_sidecar": f"source/{source.stem}-id.srt",
        "tr_infuse_sidecar": f"source/{source.stem}-tr.srt",
    }
    for key, expected_relative in sidecar_expected.items():
        verified[key] = verify_recorded_file(
            root, artifacts.get(key), key, expected_relative_path=expected_relative
        )
    if artifacts["id_srt"].get("sha256") != artifacts["id_infuse_sidecar"].get("sha256"):
        raise ArchiveError("Receipt Indonesian canonical SRT and sidecar differ")
    if artifacts["tr_srt"].get("sha256") != artifacts["tr_infuse_sidecar"].get("sha256"):
        raise ArchiveError("Receipt Turkish canonical SRT and sidecar differ")
    source_hashes = receipt.get("source_av_stream_hashes")
    if not isinstance(source_hashes, list) or not source_hashes:
        raise ArchiveError("Archive receipt has no source A/V stream hashes")

    return {
        "source": "archive_receipt",
        "episode": episode,
        "episode_name": episode_name,
        "episode_root": root,
        "report_path": root / "final" / f"{episode_name}_FINALIZATION_REPORT.json",
        "receipt": dict(receipt),
        "source_video": source,
        "source_video_record": dict(source_record),
        "id_srt": verified["id_srt"],
        "tr_srt": verified["tr_srt"],
        "id_infuse_sidecar": verified["id_infuse_sidecar"],
        "tr_infuse_sidecar": verified["tr_infuse_sidecar"],
        "srt_records": {
            key: dict(artifacts[key])
            for key in ("id_srt", "tr_srt", "id_infuse_sidecar", "tr_infuse_sidecar")
        },
        "mkv": verified["mkv"],
        "source_av_stream_hashes": source_hashes,
    }


def load_verified_archive_inputs(
    *,
    episode_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    episode: int,
) -> dict[str, Any]:
    """Load trusted FINALIZE inputs, or a prior external archive receipt."""

    root = validate_episode_root(episode_root, episode)
    _, episode_name = _episode_identity(episode)
    receipt_file = _receipt_path(receipt_path, root, episode_name)
    report_path = root / "final" / f"{episode_name}_FINALIZATION_REPORT.json"
    if report_path.exists() or report_path.is_symlink():
        report = _read_json(report_path, "FINALIZATION report")
        return _report_inputs(root, episode, report)
    receipt = _read_json(receipt_file, "Archive receipt")
    return _receipt_inputs(root, episode, receipt)


def _require_mux_pass(report: Mapping[str, Any]) -> None:
    if not (
        report.get("verified") is True
        and report.get("video_audio_stream_copy") is True
        and report.get("subtitle_order") == ["ind", "tur"]
        and report.get("indonesian_default") is True
        and report.get("turkish_default") is False
        and report.get("roundtrip", {}).get("exact") is True
        and report.get("stream_hashes", {}).get("checked") is True
        and report.get("stream_hashes", {}).get("match") is True
    ):
        raise ArchiveError("MKV verification report is incomplete or failed")


def ensure_verified_mkv(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Create an atomic stream-copy MKV or independently revalidate it."""

    source = Path(inputs["source_video"])
    mkv = Path(inputs["mkv"])
    id_srt = Path(inputs["id_srt"])
    tr_srt = Path(inputs["tr_srt"])
    if mkv.is_symlink():
        raise ArchiveError(f"Canonical MKV path is a symlink: {mkv}")
    if mkv.exists():
        if not mkv.is_file():
            raise ArchiveError(f"Canonical MKV is not a regular file: {mkv}")
        roundtrip = verify_mkv_roundtrip(mkv, id_srt, tr_srt)
        output_hashes = compute_av_stream_hashes(mkv)
        if source.is_file():
            verify_recorded_file(
                inputs["episode_root"], inputs["source_video_record"], "Source video"
            )
            source_hashes = compute_av_stream_hashes(source)
        else:
            source_hashes = inputs.get("source_av_stream_hashes")
        if not isinstance(source_hashes, list) or output_hashes != source_hashes:
            raise ArchiveError("MKV compressed A/V stream hashes differ from the source")
        report = {
            "verified": True,
            "output_path": str(mkv),
            "output_size_bytes": mkv.stat().st_size,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": roundtrip,
            "stream_hashes": {
                "checked": True,
                "match": True,
                "source": source_hashes,
                "output": output_hashes,
            },
            "resumed": True,
        }
    else:
        if not source.is_file():
            raise ArchiveError("Cannot build a missing MKV after the source video was removed")
        report = mux_softsubs(
            source, id_srt, tr_srt, mkv, verify_stream_hashes=True
        )
        report = dict(report)
        report["resumed"] = False
    _require_mux_pass(report)
    return report


def _scan_tree(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    directories: list[str] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name.casefold())
        except OSError as exc:
            raise ArchiveError(f"Cannot scan episode directory: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                details = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArchiveError(f"Cannot inspect episode entry: {relative}") from exc
            mode = details.st_mode
            if stat.S_ISLNK(mode):
                raise ArchiveError(f"Refusing descendant symlink: {relative}")
            if stat.S_ISDIR(mode):
                directories.append(relative)
                visit(path)
            elif stat.S_ISREG(mode):
                files.append(
                    {
                        "relative_path": relative,
                        "size_bytes": details.st_size,
                        "mtime_ns": details.st_mtime_ns,
                    }
                )
            else:
                raise ArchiveError(f"Refusing special filesystem entry: {relative}")

    visit(root)
    files.sort(key=lambda item: item["relative_path"])
    directories.sort()
    return {"files": files, "directories": directories}


def plan_media_only_cleanup(
    inputs: Mapping[str, Any], *, keep_source_video: bool
) -> dict[str, Any]:
    """Snapshot the full tree and return the exact non-archive deletion set."""

    if not isinstance(keep_source_video, bool):
        raise ArchiveError("keep_source_video must be True or False")
    root = validate_episode_root(inputs["episode_root"], inputs["episode"])
    snapshot = _scan_tree(root)
    source_relative = inputs["source_video_record"]["relative_path"]
    mkv_relative = Path(inputs["mkv"]).resolve().relative_to(root).as_posix()
    delete: list[dict[str, Any]] = []
    protected: list[str] = []
    extra_videos: list[str] = []
    for record in snapshot["files"]:
        relative = record["relative_path"]
        suffix = PurePosixPath(relative).suffix.casefold()
        if relative == mkv_relative or suffix == ".srt":
            protected.append(relative)
        elif relative == source_relative:
            (protected if keep_source_video else delete).append(
                relative if keep_source_video else dict(record)
            )
        elif suffix in VIDEO_SUFFIXES:
            protected.append(relative)
            extra_videos.append(relative)
        else:
            delete.append(dict(record))
    delete.sort(key=lambda item: item["relative_path"])
    protected.sort()
    extra_videos.sort()
    fingerprint_payload = {
        "episode": inputs["episode"],
        "keep_source_video": keep_source_video,
        "files": snapshot["files"],
        "directories": snapshot["directories"],
        "delete": delete,
        "protected": protected,
    }
    return {
        "plan_version": PLAN_VERSION,
        "episode": inputs["episode"],
        "keep_source_video": keep_source_video,
        "tree_fingerprint": sha256_json(fingerprint_payload),
        "tree": snapshot,
        "delete": delete,
        "protected": protected,
        "unverified_extra_videos": extra_videos,
        "delete_file_count": len(delete),
        "delete_size_bytes": sum(item["size_bytes"] for item in delete),
    }


def _artifact_records(inputs: Mapping[str, Any], mkv_report: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(inputs["episode_root"])
    return {
        "source_video": dict(inputs["source_video_record"]),
        "mkv": file_record(inputs["mkv"], root),
        "id_srt": file_record(inputs["id_srt"], root),
        "tr_srt": file_record(inputs["tr_srt"], root),
        "id_infuse_sidecar": file_record(inputs["id_infuse_sidecar"], root),
        "tr_infuse_sidecar": file_record(inputs["tr_infuse_sidecar"], root),
    }


def _ready_receipt(
    inputs: Mapping[str, Any],
    mkv_report: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    prior = inputs.get("receipt")
    prior_cleanup = prior.get("cleanup") if isinstance(prior, Mapping) else None
    same_policy = (
        isinstance(prior_cleanup, Mapping)
        and prior_cleanup.get("keep_source_video") == plan["keep_source_video"]
    )
    if same_policy:
        original_plan = prior_cleanup.get("original_plan", prior_cleanup.get("plan"))
    else:
        original_plan = plan
    if not isinstance(original_plan, Mapping):
        raise ArchiveError("Prior archive receipt has no valid original cleanup plan")
    return {
        "receipt_version": RECEIPT_VERSION,
        "status": "READY",
        "created_at": utc_now_iso(),
        "episode": inputs["episode"],
        "episode_name": inputs["episode_name"],
        "artifacts": _artifact_records(inputs, mkv_report),
        "source_av_stream_hashes": mkv_report["stream_hashes"]["source"],
        "mkv_verification": {
            "verified": True,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": dict(mkv_report["roundtrip"]),
            "stream_hashes_checked": True,
            "stream_hashes_match": True,
        },
        "cleanup": {
            "keep_source_video": plan["keep_source_video"],
            "original_plan": dict(original_plan),
            "plan": dict(plan),
        },
    }


def write_archive_receipt(path: str | os.PathLike[str], receipt: Mapping[str, Any]) -> Path:
    """Atomically write and read back a JSON archive receipt."""

    target = Path(path)
    if target.is_symlink():
        raise ArchiveError(f"Refusing symlinked archive receipt: {target}")
    atomic_write_json(target, dict(receipt))
    readback = _read_json(target, "Archive receipt")
    if readback != dict(receipt):
        raise ArchiveError("Archive receipt JSON readback mismatch")
    return target


def _verify_ready_artifacts(
    root: Path, receipt: Mapping[str, Any], inputs: Mapping[str, Any]
) -> None:
    artifacts = receipt["artifacts"]
    for key in ("mkv", "id_srt", "tr_srt", "id_infuse_sidecar", "tr_infuse_sidecar"):
        verify_recorded_file(root, artifacts[key], key)
    source = Path(inputs["source_video"])
    cleanup = receipt.get("cleanup")
    if not isinstance(cleanup, Mapping) or not isinstance(
        cleanup.get("keep_source_video"), bool
    ):
        raise ArchiveError("Receipt cleanup source-video policy is invalid")
    if source.is_symlink():
        raise ArchiveError("Source video became a symlink")
    if cleanup["keep_source_video"] is True and not source.is_file():
        raise ArchiveError("Source video disappeared despite KEEP_SOURCE_VIDEO=True")
    if source.exists():
        verify_recorded_file(root, artifacts["source_video"], "Source video")
    roundtrip = verify_mkv_roundtrip(
        inputs["mkv"], inputs["id_srt"], inputs["tr_srt"]
    )
    if roundtrip.get("exact") is not True:
        raise ArchiveError("Fresh MKV subtitle round-trip verification failed")
    output_hashes = compute_av_stream_hashes(inputs["mkv"])
    if output_hashes != receipt.get("source_av_stream_hashes"):
        raise ArchiveError("Fresh MKV A/V hashes differ from the receipt's source hashes")
    if source.exists() and compute_av_stream_hashes(source) != output_hashes:
        raise ArchiveError("Fresh source and MKV A/V stream hashes differ")


def _assert_partial_tree_matches_receipt(
    current: Mapping[str, Any],
    prior_plan: Mapping[str, Any],
    *,
    allowed_new_delete: frozenset[str] = frozenset(),
) -> None:
    prior_files = {
        item["relative_path"]: item for item in prior_plan.get("tree", {}).get("files", [])
        if isinstance(item, Mapping) and isinstance(item.get("relative_path"), str)
    }
    for current_file in current.get("tree", {}).get("files", []):
        prior = prior_files.get(current_file["relative_path"])
        if prior != current_file:
            raise ArchiveError(
                "Episode tree changed after FINALIZATION report removal; refusing ambiguous cleanup: "
                + current_file["relative_path"]
            )
    prior_delete = {
        item["relative_path"] for item in prior_plan.get("delete", [])
        if isinstance(item, Mapping) and isinstance(item.get("relative_path"), str)
    }
    current_delete = {item["relative_path"] for item in current.get("delete", [])}
    if not current_delete.issubset(prior_delete | set(allowed_new_delete)):
        raise ArchiveError("Cleanup candidates differ from the external READY receipt")


def execute_media_only_cleanup(
    inputs: Mapping[str, Any],
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    confirmation: str | None,
) -> dict[str, Any]:
    """Re-scan, freshly verify, unlink only the exact plan, then post-check."""

    expected = expected_cleanup_confirmation(inputs["episode"])
    if confirmation != expected:
        raise ArchiveError(f"Cleanup confirmation must match exactly: {expected}")
    root = validate_episode_root(inputs["episode_root"], inputs["episode"])
    current = plan_media_only_cleanup(
        inputs, keep_source_video=bool(plan["keep_source_video"])
    )
    if dict(current) != dict(plan):
        raise ArchiveError("Episode tree changed after cleanup preview; nothing was deleted")
    # From this point onward only the freshly generated canonical plan is used.
    plan = current
    _verify_ready_artifacts(root, receipt, inputs)

    source_relative = inputs["source_video_record"]["relative_path"]
    report_path = Path(inputs["report_path"])
    report_relative = (
        report_path.resolve(strict=False).relative_to(root).as_posix()
    )
    ordinary = []
    report_item = []
    source_item = []
    for item in plan["delete"]:
        relative = item["relative_path"]
        if relative == source_relative:
            source_item.append(item)
        elif relative == report_relative:
            report_item.append(item)
        else:
            ordinary.append(item)
    ordered = ordinary + report_item + source_item
    deleted: list[dict[str, Any]] = []
    for item in ordered:
        relative, path = _relative_path(root, item["relative_path"], "Cleanup candidate")
        try:
            details = path.lstat()
        except FileNotFoundError as exc:
            raise ArchiveError(f"Cleanup candidate disappeared before deletion: {relative}") from exc
        actual = {
            "relative_path": relative,
            "size_bytes": details.st_size,
            "mtime_ns": details.st_mtime_ns,
        }
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise ArchiveError(f"Cleanup candidate changed type: {relative}")
        if actual != item:
            raise ArchiveError(f"Cleanup candidate changed after preview: {relative}")
        path.unlink()
        deleted.append(dict(item))

    removed_directories: list[str] = []
    for relative in sorted(
        plan.get("tree", {}).get("directories", []),
        key=lambda value: (value.count("/"), value),
        reverse=True,
    ):
        if relative in {"source", "final"}:
            continue
        _, directory = _relative_path(root, relative, "Cleanup directory")
        if not directory.exists():
            continue
        try:
            directory.rmdir()
        except OSError as exc:
            if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                continue
            raise ArchiveError(f"Cannot remove empty directory: {relative}") from exc
        removed_directories.append(relative)

    post_plan = plan_media_only_cleanup(
        inputs, keep_source_video=bool(plan["keep_source_video"])
    )
    if post_plan["delete_file_count"] != 0:
        raise ArchiveError("Unexpected non-video/non-SRT files remain after cleanup")
    _verify_ready_artifacts(root, receipt, inputs)
    return {
        "deleted_file_count": len(deleted),
        "deleted_size_bytes": sum(item["size_bytes"] for item in deleted),
        "deleted_relative_paths": [item["relative_path"] for item in deleted],
        "removed_empty_directories": removed_directories,
        "remaining_relative_paths": post_plan["protected"],
    }


def archive_episode(
    *,
    episode_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    episode: int,
    delete_intermediates: bool = False,
    keep_source_video: bool = False,
    confirmation: str | None | Callable[[str], str] = None,
) -> dict[str, Any]:
    """Create/verify the MKV, write a READY receipt, and optionally clean up."""

    if not isinstance(delete_intermediates, bool):
        raise ArchiveError("delete_intermediates must be True or False")
    if not isinstance(keep_source_video, bool):
        raise ArchiveError("keep_source_video must be True or False")
    root = validate_episode_root(episode_root, episode)
    _, episode_name = _episode_identity(episode)
    receipt_file = _receipt_path(receipt_path, root, episode_name)
    inputs = load_verified_archive_inputs(
        episode_root=root, receipt_path=receipt_file, episode=episode
    )
    prior_receipt = inputs.get("receipt")
    if (
        isinstance(prior_receipt, Mapping)
        and prior_receipt.get("status") == "PASS"
        and prior_receipt.get("cleanup", {}).get("keep_source_video") is False
        and keep_source_video is True
        and not Path(inputs["source_video"]).exists()
    ):
        raise ArchiveError("The source video was already removed and cannot be restored")

    mkv_report = ensure_verified_mkv(inputs)
    if Path(inputs["source_video"]).exists():
        verify_recorded_file(root, inputs["source_video_record"], "Source video")
    plan = plan_media_only_cleanup(inputs, keep_source_video=keep_source_video)
    print(
        "MKV verification: PASS "
        f"(stream-copy, exact ID/TR round-trip, A/V hashes match; {mkv_report['output_size_bytes']:,} bytes)"
    )
    print(
        f"Cleanup preview: {plan['delete_file_count']:,} files, "
        f"{plan['delete_size_bytes']:,} bytes"
    )
    for item in plan["delete"]:
        print(f"  DELETE {item['relative_path']} ({item['size_bytes']:,} bytes)")
    for item in plan["unverified_extra_videos"]:
        print(f"  KEEP unverified extra video: {item}")

    if isinstance(prior_receipt, Mapping) and inputs["source"] == "archive_receipt":
        prior_cleanup = prior_receipt.get("cleanup", {})
        prior_plan = prior_cleanup.get("original_plan", prior_cleanup.get("plan"))
        if not isinstance(prior_plan, Mapping):
            raise ArchiveError("Archive receipt has no cleanup plan")
        prior_kept_source = (
            prior_receipt.get("cleanup", {}).get("keep_source_video") is True
        )
        allowed_new_delete = (
            frozenset({inputs["source_video_record"]["relative_path"]})
            if prior_kept_source and keep_source_video is False
            else frozenset()
        )
        _assert_partial_tree_matches_receipt(
            plan, prior_plan, allowed_new_delete=allowed_new_delete
        )
        if (
            prior_receipt.get("status") == "PASS"
            and plan["delete_file_count"] == 0
            and prior_receipt.get("cleanup", {}).get("keep_source_video") == keep_source_video
        ):
            result = dict(prior_receipt)
            result["no_op"] = True
            print("Verified archive already complete; nothing was deleted.")
            return result

    ready = _ready_receipt(inputs, mkv_report, plan)
    write_archive_receipt(receipt_file, ready)
    if not delete_intermediates:
        result = dict(ready)
        result["cleanup_preview"] = {
            "delete_file_count": plan["delete_file_count"],
            "delete_size_bytes": plan["delete_size_bytes"],
            "delete_relative_paths": [item["relative_path"] for item in plan["delete"]],
        }
        print("Dry run: no episode files were deleted.")
        return result

    resolved_confirmation = (
        confirmation(expected_cleanup_confirmation(episode))
        if callable(confirmation)
        else confirmation
    )
    cleanup_result = execute_media_only_cleanup(
        inputs, ready, plan, confirmation=resolved_confirmation
    )
    completed = dict(ready)
    completed["status"] = "PASS"
    completed["completed_at"] = utc_now_iso()
    completed["source_removed"] = not Path(inputs["source_video"]).exists()
    completed["cleanup"] = {
        "keep_source_video": keep_source_video,
        "original_plan": ready["cleanup"]["original_plan"],
        "plan": plan,
        "result": cleanup_result,
        "cumulative": {
            "deleted_file_count": ready["cleanup"]["original_plan"]["delete_file_count"],
            "deleted_size_bytes": ready["cleanup"]["original_plan"]["delete_size_bytes"],
            "deleted_relative_paths": [
                item["relative_path"]
                for item in ready["cleanup"]["original_plan"]["delete"]
            ],
        },
    }
    write_archive_receipt(receipt_file, completed)
    print(
        f"Cleanup PASS: removed {cleanup_result['deleted_file_count']:,} files / "
        f"{cleanup_result['deleted_size_bytes']:,} bytes"
    )
    return completed


__all__ = [
    "ArchiveError",
    "VIDEO_SUFFIXES",
    "archive_episode",
    "ensure_verified_mkv",
    "execute_media_only_cleanup",
    "expected_cleanup_confirmation",
    "file_record",
    "load_verified_archive_inputs",
    "plan_media_only_cleanup",
    "validate_episode_root",
    "verify_recorded_file",
    "write_archive_receipt",
]

