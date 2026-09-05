"""Verified, atomic Matroska soft-subtitle muxing with FFmpeg.

Only the Indonesian and Turkish subtitle streams are added. Source video and
audio are explicitly stream-copied, subtitle tracks are extracted again, and
their SRT content is compared exactly before the final MKV is published.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

try:  # Supports package imports and flat Colab module imports.
    from .srt import SubtitleEntry, parse_srt
except ImportError:  # pragma: no cover - exercised in Colab notebook mode
    from srt import SubtitleEntry, parse_srt


class MuxError(RuntimeError):
    """Raised when muxing or any post-mux verification fails."""


_STREAM_HASH_RE = re.compile(
    r"^\s*(?P<index>\d+)\s*,\s*(?P<type>[va])\s*,\s*"
    r"(?P<algorithm>[A-Za-z0-9_-]+)=(?P<digest>[0-9A-Fa-f]+)\s*$"
)


def _tool(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise MuxError(f"Required executable not found: {binary}")
    return resolved


def _run(
    command: Sequence[str],
    *,
    description: str,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise MuxError(f"{description} timed out after {timeout} seconds") from exc
    except OSError as exc:
        raise MuxError(f"Cannot run {description}: {exc}") from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "no diagnostic output").strip()
        if len(details) > 6_000:
            details = details[-6_000:]
        raise MuxError(
            f"{description} failed with exit code {result.returncode}:\n{details}"
        )
    return result


def _input_file(path: str | os.PathLike[str], label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise MuxError(f"{label} does not exist or is not a file: {value}")
    if value.stat().st_size <= 0:
        raise MuxError(f"{label} is empty: {value}")
    return value


def probe_media(
    path: str | os.PathLike[str], *, ffprobe_bin: str = "ffprobe"
) -> dict[str, Any]:
    """Return FFprobe JSON for a media file."""

    source = _input_file(path, "Media file")
    command = [
        _tool(ffprobe_bin),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    result = _run(command, description=f"ffprobe of {source.name}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MuxError(f"ffprobe returned invalid JSON for {source}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list):
        raise MuxError(f"ffprobe returned no stream list for {source}")
    return payload


def _streams(probe: Mapping[str, Any], codec_type: str) -> list[Mapping[str, Any]]:
    raw = probe.get("streams", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [
        stream
        for stream in raw
        if isinstance(stream, Mapping) and stream.get("codec_type") == codec_type
    ]


def _stream_summary(stream: Mapping[str, Any]) -> dict[str, Any]:
    disposition = stream.get("disposition")
    tags = stream.get("tags")
    return {
        "index": stream.get("index"),
        "codec_type": stream.get("codec_type"),
        "codec_name": stream.get("codec_name"),
        "time_base": stream.get("time_base"),
        "start_time": stream.get("start_time"),
        "duration": stream.get("duration"),
        "language": tags.get("language") if isinstance(tags, Mapping) else None,
        "title": tags.get("title") if isinstance(tags, Mapping) else None,
        "default": disposition.get("default")
        if isinstance(disposition, Mapping)
        else None,
    }


def compute_av_stream_hashes(
    path: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    algorithm: str = "sha256",
) -> list[dict[str, Any]]:
    """Hash compressed video/audio packet payloads after stream-copy mapping."""

    source = _input_file(path, "Media file")
    command = [
        _tool(ffmpeg_bin),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-f",
        "streamhash",
        "-hash",
        algorithm,
        "-",
    ]
    result = _run(command, description=f"audio/video stream hashing of {source.name}")
    hashes: list[dict[str, Any]] = []
    ordinals = {"v": 0, "a": 0}
    unparsed: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        match = _STREAM_HASH_RE.fullmatch(line)
        if not match:
            unparsed.append(line)
            continue
        short_type = match.group("type")
        hashes.append(
            {
                "type": "video" if short_type == "v" else "audio",
                "ordinal": ordinals[short_type],
                "algorithm": match.group("algorithm").lower(),
                "digest": match.group("digest").lower(),
            }
        )
        ordinals[short_type] += 1
    if unparsed:
        raise MuxError(
            "Unexpected FFmpeg streamhash output: " + " | ".join(unparsed[:5])
        )
    if not hashes:
        raise MuxError(f"No video/audio stream hashes were produced for {source}")
    return hashes


def _subtitle_timestamp_compensation(
    source: Path,
    source_probe: Mapping[str, Any],
    *,
    ffprobe_bin: str,
) -> Decimal:
    """Compensate FFmpeg's Matroska shift for negative first AV presentation.

    AAC/Opus encoder-delay packets commonly have a small negative PTS even when
    the container reports start_time=0. Matroska moves every presented stream
    forward to avoid that negative timestamp, which would otherwise move SRT
    cues too. Decode timestamps from reordered H.264/H.265 B-frames can be more
    negative without changing presentation start, so DTS must not drive this
    compensation. A matching negative input offset keeps extracted subtitle
    milliseconds exact.
    """

    minimum = Decimal("0")
    av_streams = _streams(source_probe, "video") + _streams(source_probe, "audio")
    for stream in av_streams:
        stream_index = stream.get("index")
        if not isinstance(stream_index, int):
            continue
        command = [
            _tool(ffprobe_bin),
            "-v",
            "error",
            "-select_streams",
            str(stream_index),
            "-read_intervals",
            "%+#1",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time",
            "-of",
            "json",
            str(source),
        ]
        result = _run(
            command,
            description=f"first-packet timestamp probe for stream {stream_index}",
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MuxError(
                f"ffprobe returned invalid packet JSON for stream {stream_index}"
            ) from exc
        packets = payload.get("packets", ()) if isinstance(payload, Mapping) else ()
        if not isinstance(packets, Sequence) or isinstance(packets, (str, bytes)):
            continue
        for packet in packets[:1]:
            if not isinstance(packet, Mapping):
                continue
            raw = packet.get("pts_time")
            if not isinstance(raw, str):
                continue
            try:
                value = Decimal(raw)
            except InvalidOperation:
                continue
            if value < minimum:
                minimum = value
    return minimum


def _assert_entry_lists_equal(
    actual: Sequence[SubtitleEntry],
    expected: Sequence[SubtitleEntry],
    language: str,
) -> None:
    if len(actual) != len(expected):
        raise MuxError(
            f"Extracted {language} subtitle block count differs: "
            f"expected {len(expected)}, got {len(actual)}"
        )
    for position, (actual_entry, expected_entry) in enumerate(
        zip(actual, expected), start=1
    ):
        if actual_entry.index != expected_entry.index:
            raise MuxError(
                f"Extracted {language} index mismatch at block {position}: "
                f"expected {expected_entry.index}, got {actual_entry.index}"
            )
        if (
            actual_entry.start_ms != expected_entry.start_ms
            or actual_entry.end_ms != expected_entry.end_ms
        ):
            raise MuxError(
                f"Extracted {language} timing mismatch at block {position}: "
                f"expected {expected_entry.start_ms}-{expected_entry.end_ms} ms, "
                f"got {actual_entry.start_ms}-{actual_entry.end_ms} ms"
            )
        if actual_entry.text != expected_entry.text:
            raise MuxError(
                f"Extracted {language} text mismatch at block {position}: "
                f"expected {expected_entry.text!r}, got {actual_entry.text!r}"
            )


def _temporary_output(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=suffix, dir=str(destination.parent)
    )
    os.close(descriptor)
    return Path(name)


def extract_subtitle_tracks(
    mkv_path: str | os.PathLike[str],
    id_output: str | os.PathLike[str],
    tr_output: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[Path, Path]:
    """Atomically extract subtitle ordinal 0 (ID) and 1 (TR) as UTF-8 SRT."""

    source = _input_file(mkv_path, "MKV file")
    id_destination = Path(id_output)
    tr_destination = Path(tr_output)
    if id_destination.resolve() == tr_destination.resolve():
        raise MuxError("Indonesian and Turkish extraction paths must differ")
    id_temp = _temporary_output(id_destination, ".id.tmp.srt")
    tr_temp = _temporary_output(tr_destination, ".tr.tmp.srt")
    try:
        command = [
            _tool(ffmpeg_bin),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:s:0",
            "-c:s",
            "srt",
            str(id_temp),
            "-map",
            "0:s:1",
            "-c:s",
            "srt",
            str(tr_temp),
        ]
        _run(command, description=f"subtitle extraction from {source.name}")
        # UTF-8 decoding and strict SRT structure are validated before publish.
        parse_srt(id_temp)
        parse_srt(tr_temp)
        id_destination.parent.mkdir(parents=True, exist_ok=True)
        tr_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(id_temp, id_destination)
        os.replace(tr_temp, tr_destination)
    except BaseException:
        for temporary in (id_temp, tr_temp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    return id_destination, tr_destination


def extract_softsubs(
    mkv_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[Path, Path]:
    """Convenience wrapper returning ``extracted.id.srt`` and ``extracted.tr.srt``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return extract_subtitle_tracks(
        mkv_path,
        directory / "extracted.id.srt",
        directory / "extracted.tr.srt",
        ffmpeg_bin=ffmpeg_bin,
    )


def _verify_subtitle_metadata(probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    subtitles = _streams(probe, "subtitle")
    if len(subtitles) != 2:
        raise MuxError(
            f"Muxed MKV must contain exactly 2 subtitle streams; got {len(subtitles)}"
        )
    summaries = [_stream_summary(stream) for stream in subtitles]
    id_stream, tr_stream = summaries
    if id_stream.get("codec_name") not in {"subrip", "srt"}:
        raise MuxError(
            f"Indonesian subtitle codec is not SubRip: {id_stream.get('codec_name')!r}"
        )
    if tr_stream.get("codec_name") not in {"subrip", "srt"}:
        raise MuxError(
            f"Turkish subtitle codec is not SubRip: {tr_stream.get('codec_name')!r}"
        )
    if id_stream.get("language") != "ind":
        raise MuxError(
            f"Subtitle track 1 must have language=ind; got {id_stream.get('language')!r}"
        )
    if tr_stream.get("language") != "tur":
        raise MuxError(
            f"Subtitle track 2 must have language=tur; got {tr_stream.get('language')!r}"
        )
    if id_stream.get("default") != 1:
        raise MuxError("Indonesian subtitle track is not the default track")
    if tr_stream.get("default") != 0:
        raise MuxError("Turkish subtitle track must be non-default")
    return summaries


def verify_mkv_roundtrip(
    mkv_path: str | os.PathLike[str],
    id_srt: str | os.PathLike[str],
    tr_srt: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    """Require exact subtitle extraction round-trip and track metadata."""

    mkv = _input_file(mkv_path, "MKV file")
    id_source = _input_file(id_srt, "Indonesian SRT")
    tr_source = _input_file(tr_srt, "Turkish SRT")
    expected_id = parse_srt(id_source)
    expected_tr = parse_srt(tr_source)
    if not expected_id or not expected_tr:
        raise MuxError("Input SRT files must contain at least one subtitle block")
    probe = probe_media(mkv, ffprobe_bin=ffprobe_bin)
    subtitle_summaries = _verify_subtitle_metadata(probe)
    with tempfile.TemporaryDirectory(prefix="subtitle-roundtrip-") as directory:
        extracted_id, extracted_tr = extract_softsubs(
            mkv, directory, ffmpeg_bin=ffmpeg_bin
        )
        actual_id = parse_srt(extracted_id)
        actual_tr = parse_srt(extracted_tr)
        _assert_entry_lists_equal(actual_id, expected_id, "Indonesian")
        _assert_entry_lists_equal(actual_tr, expected_tr, "Turkish")
    return {
        "exact": True,
        "id_block_count": len(expected_id),
        "tr_block_count": len(expected_tr),
        "subtitle_streams": subtitle_summaries,
    }


def _verify_av_streams(
    source_probe: Mapping[str, Any], output_probe: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for codec_type in ("video", "audio"):
        source = _streams(source_probe, codec_type)
        output = _streams(output_probe, codec_type)
        if len(source) != len(output):
            raise MuxError(
                f"{codec_type.title()} stream count changed during mux: "
                f"expected {len(source)}, got {len(output)}"
            )
        source_codecs = [stream.get("codec_name") for stream in source]
        output_codecs = [stream.get("codec_name") for stream in output]
        if source_codecs != output_codecs:
            raise MuxError(
                f"{codec_type.title()} codecs changed during mux: "
                f"expected {source_codecs!r}, got {output_codecs!r}"
            )
        result[f"source_{codec_type}"] = [_stream_summary(stream) for stream in source]
        result[f"output_{codec_type}"] = [_stream_summary(stream) for stream in output]
    if not result["source_video"]:
        raise MuxError("Source contains no video stream")
    return result


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def mux_softsubs(
    source_video: str | os.PathLike[str],
    id_srt: str | os.PathLike[str],
    tr_srt: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    verify_stream_hashes: bool = True,
) -> dict[str, Any]:
    """Mux, fully verify, then atomically publish a soft-subtitle MKV.

    Subtitle order is Indonesian (default) followed by Turkish (non-default).
    The final path is untouched if any FFmpeg, FFprobe, extraction, timing, text,
    metadata, codec, or requested packet-hash check fails.
    """

    source = _input_file(source_video, "Source video")
    id_source = _input_file(id_srt, "Indonesian SRT")
    tr_source = _input_file(tr_srt, "Turkish SRT")
    destination = Path(out_path)
    if destination.suffix.lower() != ".mkv":
        raise MuxError("Soft-subtitle output path must end with .mkv")
    resolved_inputs = {source.resolve(), id_source.resolve(), tr_source.resolve()}
    if destination.resolve() in resolved_inputs:
        raise MuxError("MKV output path must differ from every input path")

    # Reject malformed/non-UTF-8 subtitle input before launching FFmpeg.
    expected_id = parse_srt(id_source)
    expected_tr = parse_srt(tr_source)
    if not expected_id or not expected_tr:
        raise MuxError("Input SRT files must contain at least one subtitle block")

    ffmpeg = _tool(ffmpeg_bin)
    source_probe = probe_media(source, ffprobe_bin=ffprobe_bin)
    if not _streams(source_probe, "video"):
        raise MuxError("Source contains no video stream")
    source_hashes = (
        compute_av_stream_hashes(source, ffmpeg_bin=ffmpeg)
        if verify_stream_hashes
        else None
    )
    subtitle_compensation = _subtitle_timestamp_compensation(
        source,
        source_probe,
        ffprobe_bin=ffprobe_bin,
    )

    temporary = _temporary_output(destination, ".partial.mkv")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
    ]
    for subtitle_source in (id_source, tr_source):
        if subtitle_compensation < 0:
            command.extend(("-itsoffset", format(subtitle_compensation, "f")))
        command.extend(("-sub_charenc", "UTF-8", "-i", str(subtitle_source)))
    command.extend([
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-map",
        "1:0",
        "-map",
        "2:0",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "srt",
        "-metadata:s:s:0",
        "language=ind",
        "-metadata:s:s:0",
        "title=Bahasa Indonesia",
        "-disposition:s:0",
        "default",
        "-metadata:s:s:1",
        "language=tur",
        "-metadata:s:s:1",
        "title=Türkçe",
        "-disposition:s:1",
        "0",
        str(temporary),
    ])
    try:
        _run(command, description=f"soft-subtitle mux to {destination.name}")
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise MuxError("FFmpeg produced no non-empty MKV")

        output_probe = probe_media(temporary, ffprobe_bin=ffprobe_bin)
        av_report = _verify_av_streams(source_probe, output_probe)
        subtitle_summaries = _verify_subtitle_metadata(output_probe)

        # Extract the exact temporary MKV which may be published. This avoids a
        # time-of-check/time-of-use gap around a pre-existing final filename.
        with tempfile.TemporaryDirectory(prefix="mux-roundtrip-") as directory:
            extracted_id, extracted_tr = extract_softsubs(
                temporary, directory, ffmpeg_bin=ffmpeg
            )
            actual_id = parse_srt(extracted_id)
            actual_tr = parse_srt(extracted_tr)
            _assert_entry_lists_equal(actual_id, expected_id, "Indonesian")
            _assert_entry_lists_equal(actual_tr, expected_tr, "Turkish")

        output_hashes = (
            compute_av_stream_hashes(temporary, ffmpeg_bin=ffmpeg)
            if verify_stream_hashes
            else None
        )
        if verify_stream_hashes and output_hashes != source_hashes:
            raise MuxError(
                "Compressed video/audio stream hashes changed during mux; "
                "the MKV will not be published"
            )

        _fsync_file(temporary)
        output_size = temporary.stat().st_size
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        return {
            "verified": True,
            "output_path": str(destination),
            "output_size_bytes": output_size,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": {
                "exact": True,
                "id_block_count": len(expected_id),
                "tr_block_count": len(expected_tr),
            },
            "subtitle_streams": subtitle_summaries,
            "av_streams": av_report,
            "stream_hashes": {
                "checked": verify_stream_hashes,
                "match": True if verify_stream_hashes else None,
                "source": source_hashes,
                "output": output_hashes,
            },
            "subtitle_timestamp_compensation_seconds": format(
                subtitle_compensation, "f"
            ),
            "ffmpeg_command": command[:-1] + [str(destination)],
        }
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "MuxError",
    "compute_av_stream_hashes",
    "extract_softsubs",
    "extract_subtitle_tracks",
    "mux_softsubs",
    "probe_media",
    "verify_mkv_roundtrip",
]
