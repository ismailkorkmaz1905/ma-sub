"""Resumable, episode-local YouTube acquisition for Google Colab.

The module deliberately keeps YouTube handling separate from ffmpeg probing.  A
download marker is accepted only when every recorded output still has the same
size and SHA-256 digest.  yt-dlp writes into a hidden temporary directory and a
validated media file is atomically moved into its final location.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
MARKER_VERSION = 1
DOWNLOAD_VALIDATION_VERSION = 2
_CHUNK_SIZE = 8 * 1024 * 1024
_WORKSPACE_NAME = ".yt-dlp-work"
_WORKSPACE_REQUEST_NAME = "request.json"
_PUBLISH_JOURNAL_NAME = ".download.publish.json"
_MEDIA_SUFFIXES = {
    ".3gp",
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ogg",
    ".ogv",
    ".ts",
    ".webm",
}


class DownloadError(RuntimeError):
    """Raised when a source cannot be downloaded and validated."""


class YouTubeAuthenticationError(DownloadError):
    """Raised when YouTube requires fresh browser authentication."""


class MarkerError(RuntimeError):
    """Raised when a stage marker cannot be written safely."""


@dataclass(frozen=True)
class DownloadResult:
    """Artifacts produced by :func:`download_source`."""

    video_path: Path
    metadata_path: Path
    captions_path: Path | None
    marker_path: Path
    metadata: dict[str, Any]
    resumed: bool = False


def utc_now_iso() -> str:
    """Return a stable UTC timestamp suitable for manifests."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Hash *path* without loading it into memory."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for fingerprints and atomic files."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON data."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write bytes beside the target, fsync, then replace it atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: str | Path, value: Any, *, pretty: bool = True) -> Path:
    """Atomically write UTF-8 JSON and verify that it can be read back."""

    if pretty:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    else:
        payload = canonical_json_bytes(value) + b"\n"
    target = atomic_write_bytes(path, payload)
    try:
        with target.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        target.unlink(missing_ok=True)
        raise MarkerError(f"Atomic JSON read-back failed for {target}: {exc}") from exc
    return target


def _output_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MarkerError(f"Stage output does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise MarkerError(f"Stage output is empty: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "sha256": sha256_file(path),
    }


def write_stage_marker(
    marker_path: str | Path,
    *,
    stage: str,
    input_sha256: str,
    outputs: Mapping[str, str | Path],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a marker only after hashing every completed output."""

    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise MarkerError("input_sha256 must be a lowercase SHA-256 hex digest")
    marker = {
        "marker_version": MARKER_VERSION,
        "stage": stage,
        "completed_at": utc_now_iso(),
        "input_sha256": input_sha256,
        "outputs": {
            name: _output_record(Path(output_path))
            for name, output_path in sorted(outputs.items())
        },
        "details": dict(details or {}),
    }
    atomic_write_json(marker_path, marker)
    return marker


def load_valid_stage_marker(
    marker_path: str | Path,
    *,
    stage: str,
    input_sha256: str,
    required_output_keys: Iterable[str] = (),
    optional_output_keys: Iterable[str] = (),
    allowed_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return a marker only if its identity and all file hashes still match.

    Any malformed, stale, missing, moved, or partially written output causes a
    cache miss.  It is intentionally safe to call this on an interrupted run.
    """

    marker_file = Path(marker_path)
    if not marker_file.is_file():
        return None
    try:
        with marker_file.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("marker_version") != MARKER_VERSION:
            return None
        if marker.get("stage") != stage or marker.get("input_sha256") != input_sha256:
            return None
        outputs = marker.get("outputs")
        if not isinstance(outputs, dict):
            return None
        required_keys = set(required_output_keys)
        optional_keys = set(optional_output_keys) - required_keys
        if not required_keys.issubset(outputs):
            return None

        root = Path(allowed_root).resolve() if allowed_root is not None else None
        invalid_optional: list[str] = []
        for name, output in list(outputs.items()):
            if not isinstance(output, dict):
                if name in optional_keys:
                    outputs.pop(name, None)
                    invalid_optional.append(name)
                    continue
                return None
            output_path = Path(str(output.get("path", ""))).resolve()
            if root is not None and output_path != root and root not in output_path.parents:
                valid = False
            elif not output_path.is_file():
                valid = False
            else:
                try:
                    actual_size = output_path.stat().st_size
                    expected_hash = output.get("sha256")
                    valid = (
                        actual_size > 0
                        and actual_size == output.get("size_bytes")
                        and isinstance(expected_hash, str)
                        and sha256_file(output_path) == expected_hash
                    )
                except OSError:
                    valid = False
            if not valid:
                if name in optional_keys:
                    outputs.pop(name, None)
                    invalid_optional.append(name)
                    continue
                return None
        if invalid_optional:
            marker["_invalid_optional_outputs"] = sorted(invalid_optional)
        return marker
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _import_yt_dlp() -> Any:
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise DownloadError(
            "yt-dlp is not installed. Run the PREPARE dependency cell first."
        ) from exc
    return yt_dlp


def _validated_cookie_file(cookies_file: str | Path | None) -> Path | None:
    """Return a readable Netscape cookie file without exposing its contents."""

    if cookies_file is None:
        return None
    candidate = Path(cookies_file).expanduser()
    try:
        cookie_path = candidate.resolve(strict=True)
    except OSError:
        raise DownloadError("Cookie file was not found") from None
    if not cookie_path.is_file():
        raise DownloadError("Cookie path is not a file")
    try:
        with cookie_path.open("rb") as handle:
            first_line = handle.readline(512)
    except OSError:
        raise DownloadError("Cookie file could not be read") from None
    first_line = first_line.removeprefix(b"\xef\xbb\xbf").rstrip(b"\r\n")
    if first_line not in {b"# HTTP Cookie File", b"# Netscape HTTP Cookie File"}:
        raise DownloadError(
            "cookies_file must be a Netscape/Mozilla cookies.txt file whose first "
            "line is '# Netscape HTTP Cookie File'"
        )
    return cookie_path


def _is_youtube_bot_auth_error(error: BaseException) -> bool:
    message = str(error).casefold()
    return "sign in to confirm" in message and "not a bot" in message


def _safe_info(info: Mapping[str, Any], original_url: str) -> dict[str, Any]:
    """Keep useful source metadata without embedding yt-dlp's huge format list."""

    requested_downloads: list[dict[str, Any]] = []
    for item in info.get("requested_downloads") or []:
        if isinstance(item, Mapping):
            requested_downloads.append(
                {
                    key: item.get(key)
                    for key in (
                        "format_id",
                        "ext",
                        "protocol",
                        "vcodec",
                        "acodec",
                        "width",
                        "height",
                        "fps",
                        "tbr",
                        "filesize",
                        "filesize_approx",
                    )
                    if item.get(key) is not None
                }
            )
    return {
        "original_url": original_url,
        "webpage_url": info.get("webpage_url") or original_url,
        "extractor": info.get("extractor"),
        "extractor_key": info.get("extractor_key"),
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "upload_date": info.get("upload_date"),
        "duration_seconds_reported": info.get("duration"),
        "format_id": info.get("format_id"),
        "format": info.get("format"),
        "ext": info.get("ext"),
        "vcodec_reported": info.get("vcodec"),
        "acodec_reported": info.get("acodec"),
        "requested_downloads": requested_downloads,
    }


def _find_downloaded_media(directory: Path) -> Path:
    ignored_suffixes = {
        ".part",
        ".ytdl",
        ".json",
        ".vtt",
        ".srt",
        ".ass",
        ".lrc",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() not in ignored_suffixes
        and not path.name.endswith(".temp")
        and path.stat().st_size > 0
    ]
    if not candidates:
        raise DownloadError(f"yt-dlp completed but no media file was found in {directory}")
    return max(candidates, key=lambda path: path.stat().st_size)


def _select_caption_language(info: Mapping[str, Any], language: str) -> tuple[str, str] | None:
    """Prefer manual Turkish captions, then automatic Turkish captions."""

    requested = language.casefold()
    for source_name in ("subtitles", "automatic_captions"):
        tracks = info.get(source_name) or {}
        if not isinstance(tracks, Mapping):
            continue
        keys = [str(key) for key, value in tracks.items() if value]
        exact = [key for key in keys if key.casefold() == requested]
        prefixed = [key for key in keys if key.casefold().startswith(requested + "-")]
        if exact or prefixed:
            return (exact or sorted(prefixed, key=len))[0], source_name
    return None


def _normalise_caption_file(path: Path) -> None:
    """Ensure a downloaded caption is valid, non-empty UTF-8 text."""

    raw = path.read_bytes()
    if not raw:
        raise DownloadError(f"Downloaded caption is empty: {path}")
    text = raw.decode("utf-8-sig")
    if "-->" not in text:
        raise DownloadError(f"Downloaded caption has no timed cues: {path}")
    # Reject HTML error bodies while retaining legal VTT markup.
    if re.search(r"<html(?:\s|>)", text[:1000], flags=re.IGNORECASE):
        raise DownloadError(f"Downloaded caption looks like an HTML error page: {path}")
    atomic_write_bytes(path, text.encode("utf-8"))


def retrieve_turkish_captions(
    url: str,
    source_dir: str | Path,
    *,
    language: str = "tr",
    info: Mapping[str, Any] | None = None,
    retries: int = 3,
    socket_timeout: int = 30,
    cookies_file: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Best-effort caption retrieval; failures never invalidate the video.

    Returns ``(path, details)``.  ``path`` is ``None`` when no usable Turkish
    track exists or YouTube temporarily rejects the caption request.
    """

    yt_dlp = _import_yt_dlp()
    destination = Path(source_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        cookie_path = _validated_cookie_file(cookies_file)
        if info is None:
            info_options: dict[str, Any] = {
                "quiet": True,
                "no_warnings": False,
                "noplaylist": True,
                "retries": retries,
                "socket_timeout": socket_timeout,
            }
            if cookie_path is not None:
                info_options["cookiefile"] = str(cookie_path)
            with yt_dlp.YoutubeDL(info_options) as ydl:
                info = ydl.extract_info(url, download=False)
        selection = _select_caption_language(info or {}, language)
        if selection is None:
            return None, {"available": False, "reason": "no_turkish_caption_track"}
        track_language, source_name = selection
        with tempfile.TemporaryDirectory(prefix=".captions-", dir=destination) as tmp_name:
            tmp_dir = Path(tmp_name)
            options = {
                "outtmpl": str(tmp_dir / "captions.%(ext)s"),
                "skip_download": True,
                "noplaylist": True,
                "writesubtitles": source_name == "subtitles",
                "writeautomaticsub": source_name == "automatic_captions",
                "subtitleslangs": [track_language],
                "subtitlesformat": "vtt/best",
                "retries": retries,
                "fragment_retries": retries,
                "extractor_retries": retries,
                "socket_timeout": socket_timeout,
                "quiet": True,
                "no_warnings": False,
            }
            if cookie_path is not None:
                options["cookiefile"] = str(cookie_path)
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.extract_info(url, download=True)
            candidates = sorted(tmp_dir.glob("captions*.vtt"))
            if not candidates:
                candidates = sorted(
                    path for path in tmp_dir.iterdir() if path.is_file() and "caption" in path.name
                )
            if not candidates:
                return None, {
                    "available": True,
                    "downloaded": False,
                    "language": track_language,
                    "source": source_name,
                    "reason": "caption_artifact_missing",
                }
            downloaded = candidates[0]
            _normalise_caption_file(downloaded)
            final_path = destination / "youtube.tr.vtt"
            os.replace(downloaded, final_path)
            return final_path, {
                "available": True,
                "downloaded": True,
                "language": track_language,
                "source": "manual" if source_name == "subtitles" else "automatic",
                "sha256": sha256_file(final_path),
            }
    except Exception as exc:  # captions are explicitly optional
        if cookies_file is None:
            LOGGER.warning("Turkish YouTube captions could not be retrieved: %s", exc)
        else:
            LOGGER.warning(
                "Authenticated Turkish caption retrieval failed (%s)",
                type(exc).__name__,
            )
        return None, {
            "available": None,
            "downloaded": False,
            "reason": "caption_retrieval_failed",
            "error_type": type(exc).__name__,
        }


def _metadata_from_marker(marker: Mapping[str, Any]) -> DownloadResult | None:
    try:
        outputs = marker["outputs"]
        video_path = Path(outputs["video"]["path"])
        metadata_path = Path(outputs["metadata"]["path"])
        captions_path = (
            Path(outputs["captions"]["path"]) if "captions" in outputs else None
        )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return DownloadResult(
            video_path=video_path,
            metadata_path=metadata_path,
            captions_path=captions_path,
            marker_path=Path(str(marker.get("_marker_path", "download.done.json"))),
            metadata=metadata,
            resumed=True,
        )
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


def _marker_has_read_validation(marker: Mapping[str, Any]) -> bool:
    details = marker.get("details")
    if not isinstance(details, Mapping):
        return False
    validation = details.get("source_validation")
    return (
        isinstance(validation, Mapping)
        and validation.get("validation_version") == DOWNLOAD_VALIDATION_VERSION
        and validation.get("read_through_eof") is True
    )


def _require_direct_child(path: Path, parent: Path, *, label: str) -> Path:
    """Reject transaction paths that escape their episode-local directory."""

    resolved_parent = parent.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_parent:
        raise DownloadError(f"{label} is not a direct child of {resolved_parent}: {path}")
    return path


def _safe_remove_workspace(workspace: Path, destination: Path) -> None:
    """Remove only the workflow's exact, non-symlink workspace directory."""

    _require_direct_child(workspace, destination, label="yt-dlp workspace")
    if workspace.name != _WORKSPACE_NAME:
        raise DownloadError(f"Refusing to remove unexpected workspace: {workspace}")
    if workspace.is_symlink():
        raise DownloadError(f"Refusing to use symlink workspace: {workspace}")
    if workspace.exists():
        if not workspace.is_dir():
            raise DownloadError(f"yt-dlp workspace is not a directory: {workspace}")
        shutil.rmtree(workspace)


def _prepare_workspace(
    destination: Path,
    *,
    input_sha256: str,
) -> Path:
    """Return a stable workspace whose partial files belong to this request."""

    workspace = destination / _WORKSPACE_NAME
    request_path = workspace / _WORKSPACE_REQUEST_NAME
    if workspace.is_symlink():
        raise DownloadError(f"Refusing to use symlink workspace: {workspace}")
    if workspace.exists() and not workspace.is_dir():
        raise DownloadError(f"yt-dlp workspace is not a directory: {workspace}")
    if workspace.is_dir():
        try:
            with request_path.open("r", encoding="utf-8") as handle:
                request = json.load(handle)
            same_request = (
                isinstance(request, dict)
                and request.get("input_sha256") == input_sha256
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            same_request = False
        if not same_request:
            _safe_remove_workspace(workspace, destination)

    workspace.mkdir(parents=False, exist_ok=True)
    atomic_write_json(
        request_path,
        {
            "workspace_version": 1,
            "input_sha256": input_sha256,
            "created_or_resumed_at": utc_now_iso(),
        },
    )
    return workspace


def _validate_output_stem(output_stem: str) -> str:
    """Return a safe direct-child basename for the published source video."""

    if not isinstance(output_stem, str) or not output_stem:
        raise ValueError("output_stem must be a non-empty string")
    if output_stem != output_stem.strip():
        raise ValueError("output_stem must not have leading or trailing whitespace")
    if output_stem in {".", ".."} or output_stem.startswith("."):
        raise ValueError("output_stem must be a visible filename stem")
    if any(character in output_stem for character in ("/", "\\", "\0")):
        raise ValueError("output_stem must not contain path separators or NUL")
    if any(ord(character) < 32 for character in output_stem):
        raise ValueError("output_stem must not contain control characters")
    if Path(output_stem).suffix.lower() in _MEDIA_SUFFIXES:
        raise ValueError("output_stem must not include a media-file extension")
    return output_stem


def _managed_source_artifacts(
    destination: Path,
    *,
    output_stem: str = "source",
) -> set[Path]:
    """Return exact workflow-owned source artifacts eligible for replacement.

    ``source`` remains managed for compatibility with workspaces created before
    episode-labelled video names were introduced.  No other basename is swept.
    """

    expected_stem = _validate_output_stem(output_stem)
    managed_stems = {"source", expected_stem}

    managed = {
        destination / "source.metadata.json",
        destination / "source.media.json",
        destination / "youtube.tr.vtt",
    }
    for stem in managed_stems:
        managed.add(destination / f"{stem}-id.srt")
        managed.add(destination / f"{stem}-tr.srt")
    for child in destination.iterdir():
        if (
            child.is_file()
            and child.stem in managed_stems
            and child.suffix.lower() in _MEDIA_SUFFIXES
        ):
            managed.add(child)
    return managed


def _load_publish_journal(journal_path: Path, destination: Path) -> dict[str, Any]:
    try:
        with journal_path.open("r", encoding="utf-8") as handle:
            journal = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Cannot recover publish journal {journal_path}: {exc}") from exc
    if not isinstance(journal, dict) or journal.get("journal_version") != 1:
        raise DownloadError(f"Unsupported publish journal: {journal_path}")
    workspace = Path(str(journal.get("workspace", "")))
    _require_direct_child(workspace, destination, label="publish workspace")
    if workspace.name != _WORKSPACE_NAME or workspace.is_symlink():
        raise DownloadError(f"Unsafe publish workspace in journal: {workspace}")
    records = journal.get("records")
    if not isinstance(records, list):
        raise DownloadError(f"Publish journal has no record list: {journal_path}")
    for record in records:
        if not isinstance(record, dict):
            raise DownloadError(f"Malformed publish record in {journal_path}")
        _require_direct_child(
            Path(str(record.get("target", ""))),
            destination,
            label="publish target",
        )
        backup = Path(str(record.get("backup", ""))).resolve()
        resolved_workspace = workspace.resolve()
        if resolved_workspace not in backup.parents:
            raise DownloadError(f"Publish backup escapes workspace: {backup}")
        staged_value = record.get("staged")
        if staged_value:
            staged = Path(str(staged_value)).resolve()
            if resolved_workspace not in staged.parents:
                raise DownloadError(f"Staged artifact escapes workspace: {staged}")
    return journal


def _rollback_publish(journal: Mapping[str, Any]) -> None:
    """Restore the prior committed source set and retain new staged work."""

    records = journal["records"]
    # Restore the marker last so it never claims files while they are in motion.
    ordered = sorted(
        records,
        key=lambda item: Path(str(item["target"])).name == "download.done.json",
    )
    for record in ordered:
        target = Path(str(record["target"]))
        backup = Path(str(record["backup"]))
        staged_value = record.get("staged")
        staged = Path(str(staged_value)) if staged_value else None
        existed = record.get("existed") is True

        prior_was_moved = backup.is_file()
        new_target_exists = target.is_file() and (prior_was_moved or not existed)
        if new_target_exists:
            if staged is not None and not staged.exists():
                staged.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, staged)
            else:
                target.unlink()
        if prior_was_moved:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, target)


def _finish_publish_cleanup(journal_path: Path, journal: Mapping[str, Any]) -> None:
    for record in journal["records"]:
        backup = Path(str(record["backup"]))
        if backup.is_file():
            backup.unlink()
    journal_path.unlink(missing_ok=True)


def _recover_interrupted_publish(destination: Path) -> None:
    """Commit or roll back an interrupted multi-file source publication."""

    journal_path = destination / _PUBLISH_JOURNAL_NAME
    if not journal_path.is_file():
        return
    journal = _load_publish_journal(journal_path, destination)
    publish_id = str(journal.get("publish_id", ""))
    input_hash = str(journal.get("input_sha256", ""))
    marker = load_valid_stage_marker(
        destination / "download.done.json",
        stage="download",
        input_sha256=input_hash,
        required_output_keys=("video", "metadata"),
        allowed_root=destination,
    )
    committed = (
        marker is not None
        and isinstance(marker.get("details"), dict)
        and marker["details"].get("publish_id") == publish_id
    )
    if committed:
        _finish_publish_cleanup(journal_path, journal)
        return
    _rollback_publish(journal)
    _finish_publish_cleanup(journal_path, journal)


def _publish_outputs(
    destination: Path,
    workspace: Path,
    *,
    input_sha256: str,
    staged_outputs: Mapping[str, tuple[Path, Path]],
    marker_outputs: Mapping[str, Path],
    managed_targets: Iterable[Path],
    marker_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a verified source set with journaled rollback and marker-last commit."""

    journal_path = destination / _PUBLISH_JOURNAL_NAME
    if journal_path.exists():
        raise DownloadError(f"Unrecovered publish journal already exists: {journal_path}")
    publish_id = uuid.uuid4().hex
    backup_dir = workspace / "publish-backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=False)

    staged_by_target = {
        target.resolve(): staged for staged, target in staged_outputs.values()
    }
    targets = {Path(path) for path in managed_targets}
    targets.update(target for _, target in staged_outputs.values())
    targets.add(destination / "download.done.json")
    records: list[dict[str, Any]] = []
    for index, target in enumerate(sorted(targets, key=lambda path: path.name)):
        _require_direct_child(target, destination, label="managed publish target")
        if target.exists() and not target.is_file():
            raise DownloadError(f"Managed publish target is not a file: {target}")
        staged = staged_by_target.get(target.resolve())
        if staged is not None:
            resolved_staged = staged.resolve()
            if workspace.resolve() not in resolved_staged.parents:
                raise DownloadError(f"Staged output escapes workspace: {staged}")
            if not staged.is_file() or staged.stat().st_size <= 0:
                raise DownloadError(f"Staged output is missing or empty: {staged}")
        records.append(
            {
                "target": str(target.resolve()),
                "backup": str((backup_dir / f"{index:03d}-{target.name}").resolve()),
                "staged": str(staged.resolve()) if staged is not None else None,
                "existed": target.is_file(),
            }
        )

    journal = {
        "journal_version": 1,
        "publish_id": publish_id,
        "input_sha256": input_sha256,
        "workspace": str(workspace.resolve()),
        "created_at": utc_now_iso(),
        "records": records,
    }
    atomic_write_json(journal_path, journal)
    try:
        for record in records:
            target = Path(record["target"])
            backup = Path(record["backup"])
            if record["existed"] and target.is_file():
                os.replace(target, backup)
        for staged, target in staged_outputs.values():
            os.replace(staged, target)
        details = dict(marker_details)
        details["publish_id"] = publish_id
        marker = write_stage_marker(
            destination / "download.done.json",
            stage="download",
            input_sha256=input_sha256,
            outputs=marker_outputs,
            details=details,
        )
    except BaseException:
        _rollback_publish(journal)
        _finish_publish_cleanup(journal_path, journal)
        raise

    try:
        _finish_publish_cleanup(journal_path, journal)
    except OSError as exc:
        # The committed marker contains the publish ID.  A later invocation can
        # safely finish deleting only these journaled backups.
        LOGGER.warning("Publish committed but cleanup will be retried: %s", exc)
    return marker


def _caption_retry_on_resume(
    result: DownloadResult,
    *,
    url: str,
    destination: Path,
    language: str,
    retries: int,
    socket_timeout: int,
    input_sha256: str,
    repair_invalid_caption_output: bool,
    cookies_file: str | Path | None,
) -> DownloadResult:
    """Retry optional captions without redownloading a marker-verified video."""

    if result.captions_path is not None:
        return result
    final_caption = destination / "youtube.tr.vtt"
    # A caption not covered by the verified marker is an interrupted or stale
    # optional artifact.  Removing this exact workflow-owned filename prevents
    # it from being consumed accidentally while the independent retry runs.
    final_caption.unlink(missing_ok=True)
    workspace = _prepare_workspace(destination, input_sha256=input_sha256)
    caption_workspace = workspace / "caption-retry"
    caption_workspace.mkdir(parents=True, exist_ok=True)
    staged_caption = caption_workspace / "youtube.tr.vtt"
    staged_caption.unlink(missing_ok=True)
    captions_path, caption_details = retrieve_turkish_captions(
        url,
        caption_workspace,
        language=language,
        info=None,
        retries=max(1, min(retries, 3)),
        socket_timeout=socket_timeout,
        cookies_file=cookies_file,
    )
    if captions_path is None and not repair_invalid_caption_output:
        try:
            _safe_remove_workspace(workspace, destination)
        except OSError as exc:
            LOGGER.warning("Could not clean caption retry workspace: %s", exc)
        return result
    metadata = dict(result.metadata)
    metadata["captions"] = caption_details
    staged_metadata = workspace / "source.metadata.json.staged"
    atomic_write_json(staged_metadata, metadata)
    final_metadata = destination / "source.metadata.json"
    marker_outputs: dict[str, Path] = {
        "video": result.video_path,
        "metadata": final_metadata,
    }
    staged_outputs: dict[str, tuple[Path, Path]] = {
        "metadata": (staged_metadata, final_metadata),
    }
    if captions_path is not None:
        marker_outputs["captions"] = final_caption
        staged_outputs["captions"] = (captions_path, final_caption)
    _publish_outputs(
        destination,
        workspace,
        input_sha256=input_sha256,
        staged_outputs=staged_outputs,
        marker_outputs=marker_outputs,
        managed_targets=(final_metadata, final_caption),
        marker_details={
            "original_url": url,
            "source_sha256": sha256_file(result.video_path),
            "output_stem": result.video_path.stem,
            "captions_optional": True,
            "caption_status": caption_details,
            "caption_retry_without_video_download": True,
            "source_validation": result.metadata.get("source_validation"),
        },
    )
    if not (destination / _PUBLISH_JOURNAL_NAME).exists():
        try:
            _safe_remove_workspace(workspace, destination)
        except OSError as exc:
            LOGGER.warning("Could not clean completed caption workspace: %s", exc)
    return DownloadResult(
        video_path=result.video_path,
        metadata_path=final_metadata,
        captions_path=final_caption if captions_path is not None else None,
        marker_path=destination / "download.done.json",
        metadata=metadata,
        resumed=True,
    )


def download_source(
    url: str,
    source_dir: str | Path,
    *,
    language: str = "tr",
    format_selector: str = "bestvideo[protocol=https]+bestaudio[protocol=https]/best[protocol=https]/best",
    merge_output_format: str = "mkv",
    retries: int = 5,
    attempts: int = 3,
    socket_timeout: int = 30,
    concurrent_fragments: int = 4,
    force: bool = False,
    output_stem: str = "source",
    cookies_file: str | Path | None = None,
) -> DownloadResult:
    """Download one source video and optional Turkish captions, resumably.

    The caller is responsible for ensuring that downloading the supplied URL is
    permitted.  Playlists are always disabled to preserve episode isolation.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("A non-empty YouTube/source URL is required")
    if attempts < 1 or retries < 0:
        raise ValueError("attempts must be >= 1 and retries must be >= 0")
    published_stem = _validate_output_stem(output_stem)

    destination = Path(source_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_publish(destination)
    marker_path = destination / "download.done.json"
    metadata_path = destination / "source.metadata.json"
    # The requested basename is deliberately not download identity.  A valid
    # marker from an older ``source.ext`` workspace may therefore resume
    # without a costly redownload; new acquisitions use ``published_stem``.
    input_descriptor = {
        "url": url.strip(),
        "language": language,
        "format_selector": format_selector,
        "merge_output_format": merge_output_format,
        "playlist": False,
    }
    input_hash = sha256_json(input_descriptor)

    if not force:
        marker = load_valid_stage_marker(
            marker_path,
            stage="download",
            input_sha256=input_hash,
            required_output_keys=("video", "metadata"),
            optional_output_keys=("captions",),
            allowed_root=destination,
        )
        if marker is not None and _marker_has_read_validation(marker):
            marker["_marker_path"] = str(marker_path)
            result = _metadata_from_marker(marker)
            if result is not None:
                result.metadata["source_validation"] = dict(
                    marker["details"]["source_validation"]
                )
                try:
                    from .media import verify_media_readable

                    verify_media_readable(result.video_path, require_video_audio=True)
                except Exception as exc:
                    LOGGER.warning(
                        "Verified-marker source failed current EOF validation; "
                        "the source will be reacquired: %s",
                        exc,
                    )
                else:
                    LOGGER.info(
                        "Download stage resumed from verified marker: %s", marker_path
                    )
                    return _caption_retry_on_resume(
                        result,
                        url=url.strip(),
                        destination=destination,
                        language=language,
                        retries=retries,
                        socket_timeout=socket_timeout,
                        input_sha256=input_hash,
                        repair_invalid_caption_output=(
                            "captions"
                            in marker.get("_invalid_optional_outputs", [])
                        ),
                        cookies_file=cookies_file,
                    )

    cookie_path = _validated_cookie_file(cookies_file)
    yt_dlp = _import_yt_dlp()
    workspace = _prepare_workspace(destination, input_sha256=input_hash)
    info: Mapping[str, Any] | None = None
    try:
        options = {
            "format": format_selector,
            "outtmpl": str(workspace / "source.%(ext)s"),
            "merge_output_format": merge_output_format,
            "noplaylist": True,
            "continuedl": True,
            "nopart": False,
            "overwrites": bool(force),
            "retries": retries,
            "fragment_retries": retries,
            "skip_unavailable_fragments": False,
            "extractor_retries": retries,
            "file_access_retries": retries,
            "socket_timeout": socket_timeout,
            "concurrent_fragment_downloads": concurrent_fragments,
            "js_runtimes": {
                "deno": {"path": shutil.which("deno") or "/usr/local/bin/deno"}
            },
            "quiet": False,
            "no_warnings": False,
        }
        if cookie_path is not None:
            options["cookiefile"] = str(cookie_path)
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                LOGGER.info("Downloading source (attempt %d/%d)", attempt, attempts)
                with yt_dlp.YoutubeDL(options) as ydl:
                    extracted = ydl.extract_info(url.strip(), download=True)
                    if extracted is None:
                        raise DownloadError("yt-dlp returned no metadata")
                    if extracted.get("_type") == "playlist":
                        entries = [entry for entry in extracted.get("entries") or [] if entry]
                        if len(entries) != 1:
                            raise DownloadError("Playlist input is not allowed")
                        extracted = entries[0]
                    info = extracted
                break
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                last_error = exc
                if _is_youtube_bot_auth_error(exc):
                    if cookie_path is None:
                        raise YouTubeAuthenticationError(
                            "YouTube rejected this Colab runtime as automated traffic. "
                            "Upload a fresh Netscape-format cookies.txt file; no source "
                            "files were published."
                        ) from exc
                    raise YouTubeAuthenticationError(
                        "YouTube rejected the supplied browser cookies. Re-export fresh "
                        "YouTube cookies from a private/incognito session and retry; no "
                        "source files were published."
                    ) from None
                if attempt == attempts:
                    break
                delay = min(2 ** (attempt - 1), 15)
                if cookie_path is None:
                    LOGGER.warning(
                        "yt-dlp attempt %d failed (%s); retrying in %d seconds",
                        attempt,
                        exc,
                        delay,
                    )
                else:
                    LOGGER.warning(
                        "Authenticated yt-dlp attempt %d failed (%s); retrying in "
                        "%d seconds",
                        attempt,
                        type(exc).__name__,
                        delay,
                    )
                time.sleep(delay)
        if info is None:
            if cookie_path is not None:
                error_type = type(last_error).__name__ if last_error is not None else "Error"
                raise DownloadError(
                    "Authenticated yt-dlp download failed after "
                    f"{attempts} attempts ({error_type}); no source files were published."
                )
            raise DownloadError(f"yt-dlp failed after {attempts} attempts: {last_error}")

        downloaded = _find_downloaded_media(workspace)
        if downloaded.stat().st_size <= 1024:
            raise DownloadError(f"Downloaded source is implausibly small: {downloaded}")
        from .media import probe_media, verify_media_readable

        try:
            technical_metadata = probe_media(downloaded, include_hash=False)
            source_validation = verify_media_readable(
                downloaded, require_video_audio=True
            )
        except Exception:
            # A completed-looking file that fails the EOF pass must not be
            # skipped forever by yt-dlp's no-overwrite resume behavior.
            downloaded.unlink(missing_ok=True)
            raise

        source_validation["validation_version"] = DOWNLOAD_VALIDATION_VERSION
        source_hash = sha256_file(downloaded)
        final_video = destination / f"{published_stem}{downloaded.suffix.lower()}"
        caption_workspace = workspace / "captions"
        caption_workspace.mkdir(parents=True, exist_ok=True)
        (caption_workspace / "youtube.tr.vtt").unlink(missing_ok=True)
        captions_path, caption_details = retrieve_turkish_captions(
            url.strip(),
            caption_workspace,
            language=language,
            info=info,
            retries=max(1, min(retries, 3)),
            socket_timeout=socket_timeout,
            cookies_file=cookie_path,
        )
        metadata = _safe_info(info, url.strip())
        technical_metadata["path"] = str(final_video.resolve())
        technical_metadata["file_name"] = final_video.name
        metadata.update(technical_metadata)
        metadata.update(
            {
                "downloaded_at": utc_now_iso(),
                "source_file": final_video.name,
                "file_size_bytes": downloaded.stat().st_size,
                "sha256": source_hash,
                "captions": caption_details,
                "source_validation": source_validation,
            }
        )
        staged_metadata = workspace / "source.metadata.json.staged"
        atomic_write_json(staged_metadata, metadata)

        outputs: dict[str, Path] = {"video": final_video, "metadata": metadata_path}
        staged_outputs: dict[str, tuple[Path, Path]] = {
            "video": (downloaded, final_video),
            "metadata": (staged_metadata, metadata_path),
        }
        if captions_path is not None:
            final_caption = destination / "youtube.tr.vtt"
            outputs["captions"] = final_caption
            staged_outputs["captions"] = (captions_path, final_caption)
        _publish_outputs(
            destination,
            workspace,
            input_sha256=input_hash,
            staged_outputs=staged_outputs,
            marker_outputs=outputs,
            managed_targets=_managed_source_artifacts(
                destination, output_stem=published_stem
            ),
            marker_details={
                "original_url": url.strip(),
                "source_sha256": source_hash,
                "output_stem": published_stem,
                "captions_optional": True,
                "caption_status": caption_details,
                "source_validation": source_validation,
            },
        )
        if not (destination / _PUBLISH_JOURNAL_NAME).exists():
            try:
                _safe_remove_workspace(workspace, destination)
            except OSError as exc:
                LOGGER.warning("Could not clean completed yt-dlp workspace: %s", exc)
        return DownloadResult(
            video_path=final_video,
            metadata_path=metadata_path,
            captions_path=outputs.get("captions"),
            marker_path=marker_path,
            metadata=metadata,
            resumed=False,
        )
    except DownloadError:
        raise
    except Exception as exc:
        if cookie_path is not None:
            raise DownloadError(
                "Authenticated source processing failed "
                f"({type(exc).__name__}); no source files were published."
            ) from None
        raise DownloadError(f"Source download failed: {exc}") from exc


def validate_download(
    marker_path: str | Path,
    *,
    url: str,
    language: str = "tr",
    format_selector: str = "bestvideo[protocol=https]+bestaudio[protocol=https]/best[protocol=https]/best",
    merge_output_format: str = "mkv",
) -> bool:
    """Validate a download marker against the requested URL and settings."""

    marker_file = Path(marker_path)
    input_hash = sha256_json(
        {
            "url": url.strip(),
            "language": language,
            "format_selector": format_selector,
            "merge_output_format": merge_output_format,
            "playlist": False,
        }
    )
    marker = load_valid_stage_marker(
        marker_file,
        stage="download",
        input_sha256=input_hash,
        required_output_keys=("video", "metadata"),
        optional_output_keys=("captions",),
        allowed_root=marker_file.parent,
    )
    if marker is None or not _marker_has_read_validation(marker):
        return False
    try:
        from .media import verify_media_readable

        verify_media_readable(marker["outputs"]["video"]["path"])
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def strip_vtt_markup(text: str) -> str:
    """Small public helper shared by transcription/segmentation code."""

    text = re.sub(r"<\d\d:\d\d(?::\d\d)?\.\d{3}>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


__all__ = [
    "DownloadError",
    "DownloadResult",
    "MarkerError",
    "YouTubeAuthenticationError",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "download_source",
    "load_valid_stage_marker",
    "retrieve_turkish_captions",
    "sha256_file",
    "sha256_json",
    "strip_vtt_markup",
    "utc_now_iso",
    "validate_download",
    "write_stage_marker",
]
