"""ffprobe metadata and resumable lossless audio extraction."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .download import (
    atomic_write_json,
    load_valid_stage_marker,
    sha256_file,
    sha256_json,
    utc_now_iso,
    write_stage_marker,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DURATION_TOLERANCE_MS = 250
AUDIO_ALIGNMENT_VERSION = 2


class MediaError(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot produce a valid media artifact."""


@dataclass(frozen=True)
class AudioExtractionResult:
    """Verified result of the audio extraction stage."""

    audio_path: Path
    metadata_path: Path
    marker_path: Path
    metadata: dict[str, Any]
    resumed: bool = False


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MediaError(
            f"{name} is not available in this Colab runtime. Run the PREPARE setup cell."
        )
    return executable


def _run(
    command: Sequence[str],
    *,
    description: str,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{description} timed out after {timeout_seconds} seconds") from exc
    except OSError as exc:
        raise MediaError(f"{description} could not start: {exc}") from exc
    if process.returncode != 0:
        diagnostic = (process.stderr or process.stdout).strip()[-4000:]
        raise MediaError(
            f"{description} failed with exit code {process.returncode}: {diagnostic}"
        )
    return process


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_media(path: str | Path, *, include_hash: bool = False) -> dict[str, Any]:
    """Return normalized source metadata backed by ffprobe stream data."""

    media_path = Path(path)
    if not media_path.is_file() or media_path.stat().st_size <= 0:
        raise MediaError(f"Media file is missing or empty: {media_path}")
    ffprobe = _require_executable("ffprobe")
    process = _run(
        (
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(media_path),
        ),
        description=f"ffprobe of {media_path.name}",
        timeout_seconds=300,
    )
    try:
        raw = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {media_path}: {exc}") from exc
    raw_streams = raw.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise MediaError(f"ffprobe found no streams in {media_path}")

    streams: list[dict[str, Any]] = []
    for stream in raw_streams:
        if not isinstance(stream, Mapping):
            continue
        disposition = stream.get("disposition") or {}
        tags = stream.get("tags") or {}
        streams.append(
            {
                "index": _int_or_none(stream.get("index")),
                "codec_type": stream.get("codec_type"),
                "codec_name": stream.get("codec_name"),
                "codec_long_name": stream.get("codec_long_name"),
                "profile": stream.get("profile"),
                "time_base": stream.get("time_base"),
                "start_time_seconds": _float_or_none(stream.get("start_time")),
                "duration_seconds": _float_or_none(stream.get("duration")),
                "bit_rate": _int_or_none(stream.get("bit_rate")),
                "sample_rate_hz": _int_or_none(stream.get("sample_rate")),
                "channels": _int_or_none(stream.get("channels")),
                "channel_layout": stream.get("channel_layout"),
                "width_px": _int_or_none(stream.get("width")),
                "height_px": _int_or_none(stream.get("height")),
                "average_frame_rate": stream.get("avg_frame_rate"),
                "language": tags.get("language") if isinstance(tags, Mapping) else None,
                "default": (
                    bool(disposition.get("default"))
                    if isinstance(disposition, Mapping)
                    else False
                ),
            }
        )
    format_data = raw.get("format") or {}
    duration = _float_or_none(format_data.get("duration"))
    if duration is None:
        stream_durations = [
            stream["duration_seconds"]
            for stream in streams
            if stream.get("duration_seconds") is not None
        ]
        duration = max(stream_durations, default=None)
    if duration is None or duration <= 0:
        raise MediaError(f"ffprobe found no positive duration in {media_path}")

    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    result: dict[str, Any] = {
        "path": str(media_path.resolve()),
        "file_name": media_path.name,
        "file_size_bytes": media_path.stat().st_size,
        "duration_seconds": duration,
        "duration_ms": round(duration * 1000),
        "container": format_data.get("format_name"),
        "container_long_name": format_data.get("format_long_name"),
        "format_start_time_seconds": _float_or_none(format_data.get("start_time")),
        "format_bit_rate": _int_or_none(format_data.get("bit_rate")),
        "video_codec": video_streams[0].get("codec_name") if video_streams else None,
        "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
        "stream_start_times_seconds": {
            str(stream["index"]): stream["start_time_seconds"] for stream in streams
        },
        "stream_time_bases": {str(stream["index"]): stream["time_base"] for stream in streams},
        "streams": streams,
    }
    if include_hash:
        result["sha256"] = sha256_file(media_path)
    return result


def verify_media_readable(
    path: str | Path,
    *,
    require_video_audio: bool = True,
) -> dict[str, Any]:
    """Read the selected A/V streams through EOF and fail on demux errors.

    ffprobe can successfully read the header of a truncated fast-start MP4.  A
    stream-copy pass is much cheaper than decoding a two-hour video while still
    forcing ffmpeg to demux every packet through EOF.  ``-xerror`` makes corrupt
    packets fatal instead of leaving a final-looking, incomplete source file.
    """

    media_path = Path(path)
    metadata = probe_media(media_path, include_hash=False)
    stream_types = {
        stream.get("codec_type") for stream in metadata.get("streams", [])
    }
    if require_video_audio and not {"video", "audio"}.issubset(stream_types):
        raise MediaError(f"Media must contain both video and audio streams: {media_path}")

    maps: list[str] = []
    if "video" in stream_types:
        maps.extend(("-map", "0:v:0"))
    if "audio" in stream_types:
        maps.extend(("-map", "0:a:0"))
    if not maps:
        raise MediaError(f"Media has no readable video/audio stream: {media_path}")

    ffmpeg = _require_executable("ffmpeg")
    _run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(media_path),
            *maps,
            "-c",
            "copy",
            "-f",
            "null",
            "-",
        ),
        description=f"full read-through validation of {media_path.name}",
        timeout_seconds=None,
    )
    return {
        "read_through_eof": True,
        "checked_streams": sorted(str(value) for value in stream_types if value),
        "checked_at": utc_now_iso(),
    }


def _verify_audio_decodable(path: Path) -> None:
    """Decode a FLAC through EOF so a valid header cannot mask corruption."""

    ffmpeg = _require_executable("ffmpeg")
    _run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        ),
        description=f"full audio decode validation of {path.name}",
        timeout_seconds=None,
    )


def _source_audio_timeline(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the selected audio stream on the source playback timeline."""

    streams = metadata.get("streams")
    if not isinstance(streams, list):
        raise MediaError("Source probe metadata has no stream list")
    video_streams = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise MediaError("Source probe metadata has no audio stream")

    audio_start = _float_or_none(audio_streams[0].get("start_time_seconds"))
    video_start = (
        _float_or_none(video_streams[0].get("start_time_seconds"))
        if video_streams
        else None
    )
    format_start = _float_or_none(metadata.get("format_start_time_seconds"))
    known_starts = [
        value for value in (video_start, audio_start) if value is not None
    ]
    playback_origin = format_start
    if playback_origin is None:
        playback_origin = min(known_starts, default=0.0)
    if audio_start is None:
        # Most containers omit both stream and format starts when their timeline
        # begins at zero.  Falling back to the playback origin preserves that
        # common case without inventing a shift.
        audio_start = playback_origin

    offset_ms = round((audio_start - playback_origin) * 1000)
    return {
        "alignment_version": AUDIO_ALIGNMENT_VERSION,
        "playback_origin_seconds": playback_origin,
        "video_start_seconds": video_start,
        "audio_start_seconds": audio_start,
        "audio_offset_ms": offset_ms,
        "target_duration_ms": int(metadata["duration_ms"]),
    }


def collect_source_metadata(
    source_path: str | Path,
    *,
    original_url: str | None = None,
    download_metadata: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Combine yt-dlp identity metadata with authoritative ffprobe values."""

    merged: dict[str, Any] = {}
    if isinstance(download_metadata, Mapping):
        merged.update(download_metadata)
    elif download_metadata is not None:
        metadata_path = Path(download_metadata)
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise TypeError("metadata root is not an object")
            merged.update(loaded)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise MediaError(f"Cannot read download metadata {metadata_path}: {exc}") from exc
    if original_url is not None:
        merged["original_url"] = original_url
    merged.update(probe_media(source_path, include_hash=True))
    merged["probed_at"] = utc_now_iso()
    return merged


def save_source_metadata(
    source_path: str | Path,
    output_path: str | Path,
    *,
    original_url: str | None = None,
    download_metadata: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Collect and atomically save complete source metadata."""

    metadata = collect_source_metadata(
        source_path,
        original_url=original_url,
        download_metadata=download_metadata,
    )
    atomic_write_json(output_path, metadata)
    return metadata


def _validate_audio_metadata(path: Path, metadata: Mapping[str, Any]) -> bool:
    try:
        if metadata.get("audio_codec") != "flac":
            return False
        streams = metadata.get("streams")
        audio_streams = [
            stream
            for stream in streams
            if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
        ]
        if len(audio_streams) != 1:
            return False
        stream = audio_streams[0]
        if stream.get("channels") != 1 or stream.get("sample_rate_hz") != 16000:
            return False
        return path.is_file() and path.stat().st_size > 0 and metadata.get("duration_ms", 0) > 0
    except (TypeError, ValueError):
        return False


def validate_audio(
    audio_path: str | Path,
    *,
    expected_duration_ms: int | None = None,
    duration_tolerance_ms: int = DEFAULT_DURATION_TOLERANCE_MS,
) -> dict[str, Any]:
    """Probe an extracted FLAC and reject truncated or wrong-format output."""

    path = Path(audio_path)
    metadata = probe_media(path, include_hash=True)
    if not _validate_audio_metadata(path, metadata):
        raise MediaError(f"Extracted audio is not mono 16 kHz FLAC: {path}")
    _verify_audio_decodable(path)
    if expected_duration_ms is not None:
        delta = abs(int(metadata["duration_ms"]) - int(expected_duration_ms))
        if delta > duration_tolerance_ms:
            raise MediaError(
                "Extracted audio duration differs from source by "
                f"{delta} ms (allowed {duration_tolerance_ms} ms)"
            )
    return metadata


def _resume_audio(
    marker: Mapping[str, Any], marker_path: Path
) -> AudioExtractionResult | None:
    try:
        audio_path = Path(marker["outputs"]["audio"]["path"])
        metadata_path = Path(marker["outputs"]["metadata"]["path"])
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict) or not _validate_audio_metadata(audio_path, metadata):
            return None
        timeline = metadata.get("source_timeline")
        if not isinstance(timeline, Mapping):
            return None
        if timeline.get("alignment_version") != AUDIO_ALIGNMENT_VERSION:
            return None
        target_duration_ms = timeline.get("target_duration_ms")
        if not isinstance(target_duration_ms, int) or target_duration_ms <= 0:
            return None
        if abs(int(metadata.get("duration_ms", 0)) - target_duration_ms) > DEFAULT_DURATION_TOLERANCE_MS:
            return None
        if not isinstance(timeline.get("audio_offset_ms"), int):
            return None
        return AudioExtractionResult(
            audio_path=audio_path,
            metadata_path=metadata_path,
            marker_path=marker_path,
            metadata=metadata,
            resumed=True,
        )
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def extract_audio(
    source_path: str | Path,
    prepare_dir: str | Path,
    *,
    source_sha256: str | None = None,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    compression_level: int = 8,
    force: bool = False,
) -> AudioExtractionResult:
    """Extract mono 16 kHz lossless audio with an independently verified marker."""

    if sample_rate_hz != 16000 or channels != 1:
        raise ValueError("The production ASR path requires mono 16 kHz audio")
    if not 0 <= compression_level <= 12:
        raise ValueError("FLAC compression_level must be between 0 and 12")
    source = Path(source_path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise MediaError(f"Source media is missing or empty: {source}")

    destination = Path(prepare_dir)
    destination.mkdir(parents=True, exist_ok=True)
    audio_path = destination / "audio.flac"
    metadata_path = destination / "audio.metadata.json"
    marker_path = destination / "audio.done.json"
    computed_source_hash = sha256_file(source)
    if source_sha256 is not None:
        supplied_source_hash = str(source_sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", supplied_source_hash):
            raise ValueError("source_sha256 must be a SHA-256 hex digest")
        if supplied_source_hash != computed_source_hash:
            raise MediaError(
                "Supplied source_sha256 does not match the current source media"
            )
    actual_source_hash = computed_source_hash
    settings = {
        "source_sha256": actual_source_hash,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "codec": "flac",
        "compression_level": compression_level,
        "audio_stream": "0:a:0",
        "timeline_alignment_version": AUDIO_ALIGNMENT_VERSION,
    }
    input_hash = sha256_json(settings)

    if not force:
        marker = load_valid_stage_marker(
            marker_path,
            stage="audio",
            input_sha256=input_hash,
            required_output_keys=("audio", "metadata"),
            allowed_root=destination,
        )
        if marker is not None:
            result = _resume_audio(marker, marker_path)
            if result is not None:
                LOGGER.info("Audio stage resumed from verified marker: %s", marker_path)
                return result

    source_metadata = probe_media(source, include_hash=False)
    if not any(stream.get("codec_type") == "audio" for stream in source_metadata["streams"]):
        raise MediaError(f"Source has no audio stream: {source}")

    timeline = _source_audio_timeline(source_metadata)
    target_duration_ms = int(timeline["target_duration_ms"])
    if target_duration_ms <= 0:
        raise MediaError("Source playback duration must be positive")
    audio_offset_ms = int(timeline["audio_offset_ms"])
    filters: list[str] = []
    if audio_offset_ms < 0:
        filters.append(f"atrim=start={-audio_offset_ms / 1000:.6f}")
    # Filters otherwise inherit the source packet PTS.  Normalize first, then
    # deliberately recreate only the offset that belongs on the playback/SRT
    # timeline.
    filters.append("asetpts=PTS-STARTPTS")
    if audio_offset_ms > 0:
        filters.append(f"adelay={audio_offset_ms}:all=1")
    # Bound apad with -t below.  The resulting FLAC uses the same zero-based
    # playback timeline as external SRT files, including any leading A/V offset.
    filters.append("apad")

    ffmpeg = _require_executable("ffmpeg")
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=".audio.", suffix=".tmp.flac", dir=destination
    )
    os.close(file_descriptor)
    temporary_audio = Path(temp_name)
    temporary_audio.unlink(missing_ok=True)  # ffmpeg must create it itself
    try:
        _run(
            (
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                ",".join(filters),
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate_hz),
                "-c:a",
                "flac",
                "-compression_level",
                str(compression_level),
                "-t",
                f"{target_duration_ms / 1000:.3f}",
                str(temporary_audio),
            ),
            description="ffmpeg audio extraction",
            timeout_seconds=None,
        )
        metadata = validate_audio(
            temporary_audio,
            expected_duration_ms=target_duration_ms,
        )
        os.replace(temporary_audio, audio_path)
        # The digest is unchanged by rename; update path and independently hash
        # the final artifact so the metadata itself is self-contained.
        metadata["path"] = str(audio_path.resolve())
        metadata["file_name"] = audio_path.name
        metadata["sha256"] = sha256_file(audio_path)
        metadata.update(
            {
                "source_path": str(source.resolve()),
                "source_sha256": actual_source_hash,
                "extraction_settings": settings,
                "source_timeline": timeline,
                "extracted_at": utc_now_iso(),
            }
        )
        atomic_write_json(metadata_path, metadata)
        write_stage_marker(
            marker_path,
            stage="audio",
            input_sha256=input_hash,
            outputs={"audio": audio_path, "metadata": metadata_path},
            details={
                "source_sha256": actual_source_hash,
                "audio_sha256": metadata["sha256"],
                "duration_ms": metadata["duration_ms"],
                "source_timeline": timeline,
                "settings": settings,
            },
        )
        return AudioExtractionResult(
            audio_path=audio_path,
            metadata_path=metadata_path,
            marker_path=marker_path,
            metadata=metadata,
            resumed=False,
        )
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"Audio extraction failed: {exc}") from exc
    finally:
        temporary_audio.unlink(missing_ok=True)


def validate_audio_marker(
    marker_path: str | Path,
    *,
    source_sha256: str,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    compression_level: int = 8,
) -> bool:
    """Check marker identity, hashes, and saved audio probe metadata."""

    marker_file = Path(marker_path)
    settings = {
        "source_sha256": source_sha256,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "codec": "flac",
        "compression_level": compression_level,
        "audio_stream": "0:a:0",
        "timeline_alignment_version": AUDIO_ALIGNMENT_VERSION,
    }
    marker = load_valid_stage_marker(
        marker_file,
        stage="audio",
        input_sha256=sha256_json(settings),
        required_output_keys=("audio", "metadata"),
        allowed_root=marker_file.parent,
    )
    return marker is not None and _resume_audio(marker, marker_file) is not None


__all__ = [
    "AudioExtractionResult",
    "MediaError",
    "collect_source_metadata",
    "extract_audio",
    "probe_media",
    "save_source_metadata",
    "sha256_file",
    "validate_audio",
    "validate_audio_marker",
    "verify_media_readable",
]

