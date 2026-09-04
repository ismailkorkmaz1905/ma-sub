from __future__ import annotations

FILES: dict[str, str] = {
"src/mas/stages/__init__.py": '''from . import acquire, align, archive, asr, audio_review, captions, finalize, media, prepare_id, prepare_tr, qc, reconcile, vad

__all__ = [
    "acquire",
    "align",
    "archive",
    "asr",
    "audio_review",
    "captions",
    "finalize",
    "media",
    "prepare_id",
    "prepare_tr",
    "qc",
    "reconcile",
    "vad",
]
''',
"src/mas/stages/acquire.py": '''from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..context import EpisodeContext
from ..errors import DependencyError, FatalError
from ..hashing import atomic_write_json, atomic_write_text, sha256_file


def _existing_media(ctx: EpisodeContext) -> list[Path]:
    result: list[Path] = []
    for pattern in ctx.settings.source_globs:
        result.extend(ctx.root.joinpath("source").glob(pattern))
    return sorted({path.resolve() for path in result if path.is_file()}, key=lambda path: str(path))


def _local_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    candidate = Path(value).expanduser()
    return candidate if candidate.exists() else None


def _write_fixture(ctx: EpisodeContext) -> dict[str, Any]:
    fixture_path = ctx.root / "source" / "fixture_records.json"
    if not fixture_path.exists():
        records = [
            {
                "block_uid": "fixture-0001",
                "block_index": 1,
                "start_ms": 1000,
                "end_ms": 2600,
                "tr_text": "Merhaba Defne.",
                "id_text": "Halo Defne.",
                "speaker": "A",
                "words": [
                    {"word": "Merhaba", "start_ms": 1030, "end_ms": 1730, "probability": 0.98},
                    {"word": "Defne.", "start_ms": 1780, "end_ms": 2540, "probability": 0.97},
                ],
                "provenance": [{"source": "fixture_asr"}],
                "flags": [],
                "review_required": False,
            },
            {
                "block_uid": "fixture-0002",
                "block_index": 2,
                "start_ms": 3000,
                "end_ms": 4800,
                "tr_text": "Nasılsın?",
                "id_text": "Apa kabar?",
                "speaker": "B",
                "words": [{"word": "Nasılsın?", "start_ms": 3060, "end_ms": 4700, "probability": 0.96}],
                "provenance": [{"source": "fixture_asr"}],
                "flags": [],
                "review_required": False,
            },
            {
                "block_uid": "fixture-0003",
                "block_index": 3,
                "start_ms": 5100,
                "end_ms": 6900,
                "tr_text": "250 lira borcum kaldı.",
                "id_text": "Utangku tinggal 250 lira.",
                "speaker": "A",
                "words": [
                    {"word": "250", "start_ms": 5150, "end_ms": 5480, "probability": 0.99},
                    {"word": "lira", "start_ms": 5500, "end_ms": 5830, "probability": 0.97},
                    {"word": "borcum", "start_ms": 5860, "end_ms": 6320, "probability": 0.94},
                    {"word": "kaldı.", "start_ms": 6350, "end_ms": 6820, "probability": 0.95},
                ],
                "provenance": [{"source": "fixture_asr"}],
                "flags": [],
                "review_required": False,
            },
        ]
        atomic_write_json(fixture_path, records)
    marker = ctx.root / "source" / "FIXTURE"
    atomic_write_text(marker, "offline deterministic fixture\n")
    manifest = {
        "episode": ctx.episode,
        "fixture": True,
        "source": str(fixture_path.relative_to(ctx.root)),
        "sha256": sha256_file(fixture_path),
        "size": fixture_path.stat().st_size,
        "duration_ms": 8000,
    }
    target = ctx.root / "source" / "source_manifest.json"
    atomic_write_json(target, manifest)
    return {"outputs": [fixture_path, marker, target], "metadata": manifest}


def _download(ctx: EpisodeContext, url: str) -> Path:
    executable = shutil.which("yt-dlp")
    if not executable:
        raise DependencyError("yt-dlp is required to acquire a remote source URL")
    source_dir = ctx.root / "source"
    output_template = str(source_dir / "download.%(ext)s")
    command = [
        executable,
        "--no-playlist",
        "--newline",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "tr,tr-TR,tr.*",
        "--sub-format",
        "vtt",
        "--merge-output-format",
        "mkv",
        "-o",
        output_template,
    ]
    cookies = os.environ.get("MAS_YTDLP_COOKIES")
    if cookies:
        command.extend(["--cookies", cookies])
    command.append(url)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        tail = (result.stderr or result.stdout)[-3000:]
        raise FatalError(f"yt-dlp failed with exit code {result.returncode}: {tail}")
    media = _existing_media(ctx)
    if not media:
        raise FatalError("yt-dlp completed but no source media file was produced")
    return max(media, key=lambda path: path.stat().st_mtime_ns)


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    if ctx.fixture:
        return _write_fixture(ctx)
    source_dir = ctx.root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    selected: Path | None = None
    acquisition = "existing"
    if ctx.source_url:
        local = _local_path(ctx.source_url)
        if local is not None:
            if not local.is_file():
                raise FatalError(f"source path is not a readable file: {local}")
            extension = local.suffix.lower() or ".mkv"
            selected = source_dir / f"source{extension}"
            if local.resolve() != selected.resolve():
                temp = selected.with_suffix(selected.suffix + ".partial")
                shutil.copy2(local, temp)
                os.replace(temp, selected)
            acquisition = "local_copy"
        else:
            selected = _download(ctx, ctx.source_url)
            acquisition = "yt_dlp"
        atomic_write_text(source_dir / "source.url", ctx.source_url.strip() + "\n")
    else:
        media = _existing_media(ctx)
        if media:
            selected = media[0]
    if selected is None or not selected.exists() or selected.stat().st_size == 0:
        raise FatalError(
            "no source media found; pass --source-url or place a supported video in the episode source directory"
        )
    manifest = {
        "episode": ctx.episode,
        "fixture": False,
        "acquisition": acquisition,
        "source_url": ctx.source_url,
        "media_path": str(selected.relative_to(ctx.root)),
        "sha256": sha256_file(selected),
        "size": selected.stat().st_size,
    }
    target = source_dir / "source_manifest.json"
    atomic_write_json(target, manifest)
    outputs = [selected, target]
    source_url_file = source_dir / "source.url"
    if source_url_file.exists():
        outputs.append(source_url_file)
    outputs.extend(path for path in source_dir.glob("*.vtt") if path.is_file())
    outputs.extend(path for path in source_dir.glob("*.srt") if path.is_file())
    outputs.extend(path for path in source_dir.glob("*.json3") if path.is_file())
    return {"outputs": sorted(set(outputs), key=lambda path: str(path)), "metadata": manifest}
''',
"src/mas/stages/media.py": '''from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..context import EpisodeContext
from ..errors import DependencyError, FatalError
from ..hashing import atomic_write_json, sha256_file


def _probe(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if not executable:
        raise DependencyError("ffprobe is required")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise FatalError(f"ffprobe could not read source media: {result.stderr[-2000:]}")
    try:
        payload = json.loads(result.stdout)
        duration = float(payload.get("format", {}).get("duration") or 0)
    except (ValueError, json.JSONDecodeError) as exc:
        raise FatalError(f"invalid ffprobe output for {path}") from exc
    if duration <= 0:
        raise FatalError(f"source media has no usable positive duration: {path}")
    payload["duration_ms"] = round(duration * 1000)
    return payload


def _source_path(ctx: EpisodeContext) -> Path:
    manifest = json.loads((ctx.root / "source" / "source_manifest.json").read_text(encoding="utf-8"))
    path_value = manifest.get("media_path")
    if not path_value:
        raise FatalError("source manifest does not contain media_path")
    path = ctx.root / path_value
    if not path.exists():
        raise FatalError(f"source media disappeared: {path}")
    return path


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    target = ctx.root / "work" / "media" / "media_manifest.json"
    if ctx.fixture:
        source_manifest = json.loads((ctx.root / "source" / "source_manifest.json").read_text(encoding="utf-8"))
        manifest = {
            "episode": ctx.episode,
            "fixture": True,
            "duration_ms": int(source_manifest.get("duration_ms", 8000)),
            "audio_path": None,
            "chunks": [],
        }
        atomic_write_json(target, manifest)
        return {"outputs": [target], "metadata": manifest}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DependencyError("ffmpeg is required")
    source = _source_path(ctx)
    probe = _probe(source)
    media_dir = ctx.root / "work" / "media"
    audio = media_dir / "audio.wav"
    temp_audio = media_dir / "audio.wav.partial"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temp_audio),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode or not temp_audio.exists() or temp_audio.stat().st_size == 0:
        raise FatalError(f"audio extraction failed: {result.stderr[-2500:]}")
    os.replace(temp_audio, audio)
    chunks_dir = media_dir / "chunks"
    temp_chunks = Path(tempfile.mkdtemp(prefix="chunks.", dir=media_dir))
    try:
        segment_pattern = temp_chunks / "chunk_%04d.wav"
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio),
                "-f",
                "segment",
                "-segment_time",
                str(ctx.settings.asr_chunk_seconds),
                "-reset_timestamps",
                "1",
                "-c:a",
                "copy",
                str(segment_pattern),
            ],
            text=True,
            capture_output=True,
        )
        chunk_files = sorted(temp_chunks.glob("chunk_*.wav"))
        if result.returncode or not chunk_files:
            raise FatalError(f"audio chunking failed: {result.stderr[-2500:]}")
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        os.replace(temp_chunks, chunks_dir)
    finally:
        if temp_chunks.exists():
            shutil.rmtree(temp_chunks, ignore_errors=True)
    chunk_rows: list[dict[str, Any]] = []
    total_ms = int(probe["duration_ms"])
    chunk_ms = int(ctx.settings.asr_chunk_seconds * 1000)
    for index, chunk in enumerate(sorted(chunks_dir.glob("chunk_*.wav"))):
        offset = index * chunk_ms
        chunk_rows.append(
            {
                "index": index,
                "path": str(chunk.relative_to(ctx.root)),
                "offset_ms": offset,
                "duration_ms": min(chunk_ms, max(0, total_ms - offset)),
                "sha256": sha256_file(chunk),
                "size": chunk.stat().st_size,
            }
        )
    manifest = {
        "episode": ctx.episode,
        "fixture": False,
        "source_path": str(source.relative_to(ctx.root)),
        "source_sha256": sha256_file(source),
        "duration_ms": total_ms,
        "audio_path": str(audio.relative_to(ctx.root)),
        "audio_sha256": sha256_file(audio),
        "audio_size": audio.stat().st_size,
        "probe": probe,
        "chunks": chunk_rows,
    }
    atomic_write_json(target, manifest)
    outputs = [target, audio] + [ctx.root / row["path"] for row in chunk_rows]
    return {"outputs": outputs, "metadata": {"duration_ms": total_ms, "chunk_count": len(chunk_rows)}}
''',
"src/mas/stages/captions.py": '''from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from ..context import EpisodeContext
from ..hashing import atomic_write_json, sha256_file

_TIME = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})|(?P<m2>\d{1,2}):(?P<s2>\d{2})[,.](?P<ms2>\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


def _ms(value: str) -> int:
    match = _TIME.search(value.strip())
    if not match:
        raise ValueError(value)
    if match.group("h") is not None:
        hours = int(match.group("h"))
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
        millis = int(match.group("ms"))
    else:
        hours = 0
        minutes = int(match.group("m2"))
        seconds = int(match.group("s2"))
        millis = int(match.group("ms2"))
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _clean(text: str) -> str:
    value = html.unescape(_TAG.sub("", text))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_timed_text(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    cues: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if "-->" not in line:
            index += 1
            continue
        left, right = line.split("-->", 1)
        right_time = right.strip().split()[0]
        try:
            start_ms = _ms(left)
            end_ms = _ms(right_time)
        except ValueError:
            index += 1
            continue
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            if not lines[index].strip().isdigit():
                text_lines.append(lines[index])
            index += 1
        text = _clean(" ".join(text_lines))
        if text and end_ms > start_ms:
            cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
        index += 1
    return cues


def _parse_json3(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cues: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        start_ms = int(event.get("tStartMs", 0))
        duration_ms = int(event.get("dDurationMs", 0))
        text = _clean("".join(segment.get("utf8", "") for segment in event.get("segs", [])))
        if text and duration_ms > 0:
            cues.append({"start_ms": start_ms, "end_ms": start_ms + duration_ms, "text": text})
    return cues


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    source_dir = ctx.root / "source"
    candidates = sorted(
        [*source_dir.glob("*.vtt"), *source_dir.glob("*.srt"), *source_dir.glob("*.json3")],
        key=lambda path: ("tr" not in path.name.lower(), str(path)),
    )
    cues: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in candidates:
        try:
            parsed = _parse_json3(path) if path.suffix.lower() == ".json3" else _parse_timed_text(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sources.append(
                {
                    "path": str(path.relative_to(ctx.root)),
                    "sha256": sha256_file(path),
                    "status": "invalid",
                    "error": str(exc),
                }
            )
            continue
        sources.append(
            {
                "path": str(path.relative_to(ctx.root)),
                "sha256": sha256_file(path),
                "status": "parsed",
                "cue_count": len(parsed),
            }
        )
        for ordinal, cue in enumerate(parsed):
            cue.update(
                {
                    "caption_uid": f"caption-{len(cues) + 1:06d}",
                    "source_path": str(path.relative_to(ctx.root)),
                    "source_ordinal": ordinal,
                    "provenance": "youtube_caption",
                }
            )
            cues.append(cue)
    cues.sort(key=lambda row: (row["start_ms"], row["end_ms"], row["caption_uid"]))
    target = ctx.root / "work" / "captions.json"
    payload = {"episode": ctx.episode, "sources": sources, "caption_count": len(cues), "captions": cues}
    atomic_write_json(target, payload)
    return {"outputs": [target], "metadata": {"caption_count": len(cues), "source_count": len(sources)}}
''',
"src/mas/stages/vad.py": '''from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..context import EpisodeContext
from ..hashing import atomic_write_json

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


def _speech_from_silence(duration_ms: int, text: str) -> list[dict[str, int]]:
    events: list[tuple[str, int]] = []
    for line in text.splitlines():
        start = _SILENCE_START.search(line)
        if start:
            events.append(("start", round(float(start.group(1)) * 1000)))
        end = _SILENCE_END.search(line)
        if end:
            events.append(("end", round(float(end.group(1)) * 1000)))
    events.sort(key=lambda item: item[1])
    speech: list[dict[str, int]] = []
    cursor = 0
    for kind, point in events:
        point = min(max(point, 0), duration_ms)
        if kind == "start" and point > cursor:
            speech.append({"start_ms": cursor, "end_ms": point})
        elif kind == "end":
            cursor = max(cursor, point)
    if cursor < duration_ms:
        speech.append({"start_ms": cursor, "end_ms": duration_ms})
    return [row for row in speech if row["end_ms"] > row["start_ms"]]


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    media = json.loads((ctx.root / "work" / "media" / "media_manifest.json").read_text(encoding="utf-8"))
    duration_ms = int(media["duration_ms"])
    status = "ok"
    warnings: list[str] = []
    if ctx.fixture:
        records = json.loads((ctx.root / "source" / "fixture_records.json").read_text(encoding="utf-8"))
        speech = [{"start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"])} for row in records]
    else:
        ffmpeg = shutil.which("ffmpeg")
        audio_value = media.get("audio_path")
        audio = ctx.root / audio_value if audio_value else None
        if not ffmpeg or audio is None or not audio.exists():
            status = "fallback"
            warnings.append("VAD unavailable; treating complete source duration as one speech span")
            speech = [{"start_ms": 0, "end_ms": duration_ms}]
        else:
            filter_value = (
                f"silencedetect=noise={ctx.settings.vad_silence_db}:d={ctx.settings.vad_min_silence_seconds}"
            )
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(audio), "-af", filter_value, "-f", "null", "-"],
                text=True,
                capture_output=True,
            )
            if result.returncode:
                status = "fallback"
                warnings.append(f"silencedetect failed with exit code {result.returncode}")
                speech = [{"start_ms": 0, "end_ms": duration_ms}]
            else:
                speech = _speech_from_silence(duration_ms, result.stderr)
                if not speech:
                    status = "fallback"
                    warnings.append("silencedetect produced no speech spans")
                    speech = [{"start_ms": 0, "end_ms": duration_ms}]
    target = ctx.root / "work" / "vad.json"
    payload = {
        "episode": ctx.episode,
        "status": status,
        "warnings": warnings,
        "duration_ms": duration_ms,
        "speech_span_count": len(speech),
        "speech": speech,
    }
    atomic_write_json(target, payload)
    return {"outputs": [target], "metadata": {"status": status, "speech_span_count": len(speech)}}
''',
"src/mas/stages/asr.py": '''from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from ..context import EpisodeContext
from ..errors import DependencyError, FatalError
from ..hashing import atomic_write_json, sha256_file, sha256_json

_REPEAT = re.compile(r"\b(.{2,30}?)\b(?:\s+\1\b){3,}", re.I)
_TEXT = re.compile(r"[\wÇĞİÖŞÜçğıöşü]", re.UNICODE)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _audit(text: str, no_speech_probability: float, average_log_probability: float) -> tuple[list[str], bool]:
    flags: list[str] = []
    stripped = text.strip()
    if not stripped or not _TEXT.search(stripped):
        flags.append("suspected_asr_hallucination")
    if _REPEAT.search(stripped):
        flags.append("suspected_asr_hallucination")
    if no_speech_probability >= 0.60 and average_log_probability <= -1.0:
        flags.append("suspected_asr_hallucination")
    flags = list(dict.fromkeys(flags))
    return flags, bool(flags)


def _fixture(ctx: EpisodeContext) -> dict[str, Any]:
    source = ctx.root / "source" / "fixture_records.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    for row in records:
        row.setdefault("asr_text", row.get("tr_text", ""))
        row.setdefault(
            "asr_audit",
            {
                "avg_logprob": -0.05,
                "no_speech_prob": 0.01,
                "confidence": 0.97,
                "model": "fixture",
            },
        )
        row.setdefault("provenance", []).append({"source": "asr", "model": "fixture"})
    target = ctx.root / "work" / "asr" / "asr.json"
    payload = {
        "episode": ctx.episode,
        "fixture": True,
        "model": {"name": "fixture", "faster_whisper": "not-used"},
        "record_count": len(records),
        "records": records,
    }
    atomic_write_json(target, payload)
    return {"outputs": [target], "metadata": payload["model"] | {"record_count": len(records)}}


def _chunk_valid(path: Path, input_sha: str, config_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("input_sha256") == input_sha and payload.get("config_sha256") == config_sha


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    if ctx.fixture:
        return _fixture(ctx)
    media_path = ctx.root / "work" / "media" / "media_manifest.json"
    media = json.loads(media_path.read_text(encoding="utf-8"))
    chunks = list(media.get("chunks") or [])
    if not chunks:
        raise FatalError("media stage produced no ASR chunks")
    config_payload = {
        "model": ctx.settings.asr_model,
        "device": ctx.settings.asr_device,
        "compute_type": ctx.settings.asr_compute_type,
        "beam_size": ctx.settings.asr_beam_size,
        "language": ctx.settings.asr_language,
        "vad_filter": ctx.settings.vad_filter,
        "word_timestamps": True,
        "condition_on_previous_text": False,
    }
    config_sha = sha256_json(config_payload)
    asr_dir = ctx.root / "work" / "asr" / "chunks"
    asr_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in chunks
        if not _chunk_valid(asr_dir / f"chunk_{int(row['index']):04d}.json", str(row["sha256"]), config_sha)
    ]
    model: Any | None = None
    if pending:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyError(
                "faster-whisper is required for real ASR; run runpod/bootstrap.sh in the GPU image"
            ) from exc
        try:
            model = WhisperModel(
                ctx.settings.asr_model,
                device=ctx.settings.asr_device,
                compute_type=ctx.settings.asr_compute_type,
            )
        except Exception as exc:
            raise FatalError(
                f"could not load ASR model {ctx.settings.asr_model} on {ctx.settings.asr_device}: {exc}"
            ) from exc
    for chunk in chunks:
        chunk_index = int(chunk["index"])
        chunk_output = asr_dir / f"chunk_{chunk_index:04d}.json"
        if _chunk_valid(chunk_output, str(chunk["sha256"]), config_sha):
            continue
        if model is None:
            raise FatalError("ASR model was not loaded for a pending chunk")
        audio = ctx.root / chunk["path"]
        offset_ms = int(chunk["offset_ms"])
        try:
            segments_iter, info = model.transcribe(
                str(audio),
                language=ctx.settings.asr_language,
                beam_size=ctx.settings.asr_beam_size,
                word_timestamps=True,
                vad_filter=ctx.settings.vad_filter,
                condition_on_previous_text=False,
            )
            segments = list(segments_iter)
        except Exception as exc:
            raise FatalError(f"ASR failed on chunk {chunk_index}: {exc}") from exc
        rows: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            start_ms = offset_ms + round(float(segment.start) * 1000)
            end_ms = offset_ms + round(float(segment.end) * 1000)
            text = str(segment.text or "").strip()
            average_log_probability = float(getattr(segment, "avg_logprob", -99.0))
            no_speech_probability = float(getattr(segment, "no_speech_prob", 0.0))
            flags, review_required = _audit(text, no_speech_probability, average_log_probability)
            words: list[dict[str, Any]] = []
            probabilities: list[float] = []
            for word in getattr(segment, "words", None) or []:
                if word.start is None or word.end is None:
                    continue
                probability = float(getattr(word, "probability", 0.0) or 0.0)
                probabilities.append(probability)
                words.append(
                    {
                        "word": str(word.word),
                        "start_ms": offset_ms + round(float(word.start) * 1000),
                        "end_ms": offset_ms + round(float(word.end) * 1000),
                        "probability": probability,
                    }
                )
            uid_seed = {
                "episode": ctx.episode,
                "chunk_sha256": chunk["sha256"],
                "chunk_index": chunk_index,
                "segment_index": segment_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
            rows.append(
                {
                    "block_uid": f"asr-{sha256_json(uid_seed)[:20]}",
                    "block_index": 0,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "tr_text": text,
                    "id_text": "",
                    "asr_text": text,
                    "words": words,
                    "speaker": None,
                    "provenance": [
                        {
                            "source": "asr",
                            "model": ctx.settings.asr_model,
                            "chunk_index": chunk_index,
                            "segment_index": segment_index,
                        }
                    ],
                    "asr_audit": {
                        "avg_logprob": average_log_probability,
                        "no_speech_prob": no_speech_probability,
                        "confidence": mean(probabilities) if probabilities else None,
                        "compression_ratio": float(getattr(segment, "compression_ratio", 0.0)),
                        "temperature": float(getattr(segment, "temperature", 0.0)),
                    },
                    "flags": flags,
                    "review_required": review_required,
                    "audio_reviewed": False,
                    "review_disposition": None,
                }
            )
        payload = {
            "episode": ctx.episode,
            "chunk_index": chunk_index,
            "input_sha256": chunk["sha256"],
            "config_sha256": config_sha,
            "config": config_payload,
            "detected_language": getattr(info, "language", None),
            "detected_language_probability": getattr(info, "language_probability", None),
            "segments": rows,
        }
        atomic_write_json(chunk_output, payload)
    del model
    all_rows: list[dict[str, Any]] = []
    chunk_outputs: list[Path] = []
    for chunk in chunks:
        path = asr_dir / f"chunk_{int(chunk['index']):04d}.json"
        if not _chunk_valid(path, str(chunk["sha256"]), config_sha):
            raise FatalError(f"ASR chunk checkpoint is invalid after processing: {path.name}")
        chunk_outputs.append(path)
        all_rows.extend(json.loads(path.read_text(encoding="utf-8"))["segments"])
    all_rows.sort(key=lambda row: (int(row["start_ms"]), int(row["end_ms"]), str(row["block_uid"])))
    for index, row in enumerate(all_rows, 1):
        row["block_index"] = index
    model_info = {
        "name": ctx.settings.asr_model,
        "device": ctx.settings.asr_device,
        "compute_type": ctx.settings.asr_compute_type,
        "faster_whisper": _version("faster-whisper"),
        "ctranslate2": _version("ctranslate2"),
    }
    target = ctx.root / "work" / "asr" / "asr.json"
    payload = {
        "episode": ctx.episode,
        "fixture": False,
        "model": model_info,
        "config_sha256": config_sha,
        "record_count": len(all_rows),
        "records": all_rows,
    }
    atomic_write_json(target, payload)
    return {
        "outputs": [target, *chunk_outputs],
        "metadata": model_info | {"record_count": len(all_rows), "chunk_count": len(chunks)},
    }
''',
"src/mas/stages/reconcile.py": '''from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from ..context import EpisodeContext
from ..hashing import atomic_write_json, sha256_json
from ..subtitle.timing import intersects, temporal_iou


def _best_caption(row: dict[str, Any], captions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    score = 0.0
    for caption in captions:
        if not intersects(row, caption, tolerance_ms=250):
            continue
        timing = temporal_iou(row, caption)
        text = SequenceMatcher(
            None, str(row.get("tr_text", "")).lower(), str(caption.get("text", "")).lower()
        ).ratio()
        candidate = 0.75 * timing + 0.25 * text
        if candidate > score:
            best = caption
            score = candidate
    return best, score


def _has_dialogue_overlap(span: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    span_duration = max(1, int(span["end_ms"]) - int(span["start_ms"]))
    covered = 0
    for row in rows:
        start = max(int(span["start_ms"]), int(row["start_ms"]))
        end = min(int(span["end_ms"]), int(row["end_ms"]))
        covered += max(0, end - start)
    return covered / span_duration >= 0.35


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    asr_payload = json.loads((ctx.root / "work" / "asr" / "asr.json").read_text(encoding="utf-8"))
    caption_payload = json.loads((ctx.root / "work" / "captions.json").read_text(encoding="utf-8"))
    vad_payload = json.loads((ctx.root / "work" / "vad.json").read_text(encoding="utf-8"))
    rows = [dict(row) for row in asr_payload.get("records", [])]
    captions = list(caption_payload.get("captions", []))
    matched_caption_uids: set[str] = set()
    review_items: list[dict[str, Any]] = []
    for row in rows:
        caption, score = _best_caption(row, captions)
        if caption is not None and score >= 0.20:
            matched_caption_uids.add(str(caption["caption_uid"]))
            row["youtube_caption_text"] = caption["text"]
            row["caption_match_score"] = round(score, 4)
            row.setdefault("provenance", []).append(
                {
                    "source": "youtube_caption",
                    "caption_uid": caption["caption_uid"],
                    "source_path": caption["source_path"],
                    "match_score": round(score, 4),
                }
            )
            if score < 0.45:
                row.setdefault("flags", []).append("weak_caption_match")
                row["review_required"] = True
        if row.get("review_required"):
            review_items.append(
                {
                    "review_uid": f"review-{row['block_uid']}",
                    "block_uid": row["block_uid"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "reasons": sorted(set(row.get("flags") or [])),
                    "requires_audio": "suspected_asr_hallucination" in (row.get("flags") or []),
                }
            )
    for caption in captions:
        if str(caption["caption_uid"]) in matched_caption_uids:
            continue
        uid = f"caption-{sha256_json({'episode': ctx.episode, 'caption': caption})[:20]}"
        rows.append(
            {
                "block_uid": uid,
                "block_index": 0,
                "start_ms": int(caption["start_ms"]),
                "end_ms": int(caption["end_ms"]),
                "tr_text": str(caption["text"]).strip(),
                "id_text": "",
                "asr_text": "",
                "youtube_caption_text": str(caption["text"]).strip(),
                "words": [],
                "speaker": None,
                "provenance": [
                    {
                        "source": "youtube_caption",
                        "caption_uid": caption["caption_uid"],
                        "source_path": caption["source_path"],
                        "match_score": None,
                    }
                ],
                "asr_audit": None,
                "flags": ["orphan_youtube_caption"],
                "review_required": True,
                "audio_reviewed": False,
                "review_disposition": None,
            }
        )
        review_items.append(
            {
                "review_uid": f"review-{uid}",
                "block_uid": uid,
                "start_ms": caption["start_ms"],
                "end_ms": caption["end_ms"],
                "reasons": ["orphan_youtube_caption"],
                "requires_audio": True,
            }
        )
    rows = [row for row in rows if str(row.get("tr_text", "")).strip()]
    rows.sort(key=lambda row: (int(row["start_ms"]), int(row["end_ms"]), str(row["block_uid"])))
    for index, row in enumerate(rows, 1):
        row["block_index"] = index
        row["flags"] = sorted(set(row.get("flags") or []))
    minimum = round(ctx.settings.unresolved_speech_min_seconds * 1000)
    for ordinal, span in enumerate(vad_payload.get("speech", []), 1):
        if int(span["end_ms"]) - int(span["start_ms"]) < minimum or _has_dialogue_overlap(span, rows):
            continue
        review_uid = f"vad-{sha256_json({'episode': ctx.episode, 'span': span, 'ordinal': ordinal})[:20]}"
        review_items.append(
            {
                "review_uid": review_uid,
                "block_uid": None,
                "start_ms": int(span["start_ms"]),
                "end_ms": int(span["end_ms"]),
                "reasons": ["unresolved_vad_speech"],
                "requires_audio": True,
            }
        )
    records_target = ctx.root / "work" / "records.json"
    review_target = ctx.write_review_items(review_items)
    payload = {
        "episode": ctx.episode,
        "record_count": len(rows),
        "asr_record_count": len(asr_payload.get("records", [])),
        "caption_count": len(captions),
        "orphan_caption_count": sum(1 for item in review_items if "orphan_youtube_caption" in item["reasons"]),
        "unresolved_speech_count": sum(1 for item in review_items if "unresolved_vad_speech" in item["reasons"]),
        "records": rows,
    }
    atomic_write_json(records_target, payload)
    return {
        "outputs": [records_target, review_target],
        "metadata": {
            "record_count": len(rows),
            "review_count": len(review_items),
            "unresolved_speech_count": payload["unresolved_speech_count"],
        },
    }
''',
"src/mas/stages/prepare_tr.py": '''from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import repo_root
from ..context import EpisodeContext
from ..hashing import atomic_write_json, atomic_write_text
from ..packs.build import build_pack


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _prompt(ctx: EpisodeContext) -> str:
    rules = repo_root() / "rules"
    return f"""You are correcting Turkish subtitle text for {ctx.name}.

Return one ZIP named exactly:
{ctx.name}_TR_TEXT_CORRECTED.zip

Edit only Turkish text fields allowed by manifest.json. Preserve every immutable field byte-for-byte in meaning and value, including block_uid, block_index, start_ms, end_ms, schema_version, schema_sha256, hashes, audit fields, review flags, record count, and order. Do not invent audio-dependent dialogue. Records requiring audio review must remain flagged unless audio_reviewed is already true. Do not split, merge, add, remove, or reorder records. Put the edited records.json and the unchanged manifest.json back in the ZIP.

Canonical subtitle specification:
{_read(rules / 'SUBTITLE_SPEC.md')}

Canonical Turkish and Indonesian language specification:
{_read(rules / 'TRANSLATION_SPEC.md')}

Canonical rule inventory and precedence notes:
{_read(rules / 'V2_RULE_INVENTORY.md')}
"""


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads((ctx.root / "work" / "records.json").read_text(encoding="utf-8"))
    records = payload["records"]
    prompt = _prompt(ctx)
    prompt_path = ctx.root / "translation_input" / "TR_PROMPT.txt"
    pack_path = ctx.root / "translation_input" / f"{ctx.name}_TR_CORRECTION_PACK.zip"
    manifest_path = ctx.root / "translation_input" / "TR_MANIFEST.json"
    atomic_write_text(prompt_path, prompt)
    manifest = build_pack(
        pack_path,
        ctx.episode,
        "TR",
        records,
        prompt_text=prompt,
        metadata={"review_required_path": "work/review_required.json"},
        context_blocks=ctx.settings.context_blocks,
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "outputs": [pack_path, prompt_path, manifest_path],
        "metadata": {"record_count": len(records), "waiting_for": f"{ctx.name}_TR_TEXT_CORRECTED.zip"},
    }
''',
"src/mas/stages/audio_review.py": '''from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..context import EpisodeContext
from ..hashing import atomic_write_json, sha256_file


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    review_path = ctx.root / "work" / "review_required.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {"items": []}
    candidates = [item for item in review.get("items", []) if item.get("requires_audio")]
    clips_dir = ctx.root / "work" / "audio_review" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    media_manifest = json.loads(
        (ctx.root / "work" / "media" / "media_manifest.json").read_text(encoding="utf-8")
    )
    audio_value = media_manifest.get("audio_path")
    audio = ctx.root / audio_value if audio_value else None
    ffmpeg = shutil.which("ffmpeg")
    rows: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for item in candidates:
        uid = str(item["review_uid"])
        start_ms = max(0, int(item["start_ms"]) - ctx.settings.review_clip_padding_ms)
        end_ms = int(item["end_ms"]) + ctx.settings.review_clip_padding_ms
        target = clips_dir / f"{uid}.wav"
        status = "not-created"
        if not ctx.fixture and ffmpeg and audio is not None and audio.exists():
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start_ms / 1000:.3f}",
                    "-to",
                    f"{end_ms / 1000:.3f}",
                    "-i",
                    str(audio),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(target),
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
                status = "created"
                outputs.append(target)
            else:
                status = f"ffmpeg-failed-{result.returncode}"
        rows.append(
            {
                **item,
                "clip_path": str(target.relative_to(ctx.root)) if status == "created" else None,
                "clip_sha256": sha256_file(target) if status == "created" else None,
                "clip_start_ms": start_ms,
                "clip_end_ms": end_ms,
                "status": status,
            }
        )
    manifest_path = ctx.root / "work" / "audio_review" / "manifest.json"
    payload = {"episode": ctx.episode, "candidate_count": len(rows), "candidates": rows}
    atomic_write_json(manifest_path, payload)
    outputs.append(manifest_path)
    return {"outputs": outputs, "metadata": {"candidate_count": len(rows)}}
''',
"src/mas/stages/align.py": '''from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from ..context import EpisodeContext
from ..hashing import atomic_write_json


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    corrected_path = ctx.root / "work" / "tr_corrected.json"
    payload = json.loads(corrected_path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload["records"]]
    aligned = 0
    anomalies = 0
    for row in rows:
        words = [word for word in row.get("words") or [] if word.get("start_ms") is not None and word.get("end_ms") is not None]
        original = str(row.get("tr_text_original") or row.get("asr_text") or "")
        corrected = str(row.get("tr_text") or "")
        similarity = SequenceMatcher(None, _normalize(original), _normalize(corrected)).ratio() if original else 0.0
        row["alignment_provenance"] = {
            "backend": "asr_word_timing",
            "text_similarity": round(similarity, 4),
            "word_count": len(words),
        }
        if words and similarity >= 0.45:
            word_start = int(words[0]["start_ms"])
            word_end = int(words[-1]["end_ms"])
            if word_end > word_start:
                row["start_ms"] = max(int(row["start_ms"]), word_start)
                row["end_ms"] = min(int(row["end_ms"]), word_end)
                if int(row["end_ms"]) <= int(row["start_ms"]):
                    row["start_ms"] = int(payload["source_timing"][row["block_uid"]]["start_ms"]) if payload.get("source_timing") else word_start
                    row["end_ms"] = int(payload["source_timing"][row["block_uid"]]["end_ms"]) if payload.get("source_timing") else word_end
                aligned += 1
        elif not ctx.fixture:
            row.setdefault("flags", []).append("alignment_anomaly")
            row["review_required"] = True
            anomalies += 1
        row["flags"] = sorted(set(row.get("flags") or []))
    target = ctx.root / "work" / "aligned.json"
    result = {
        "episode": ctx.episode,
        "backend": "asr_word_timing",
        "record_count": len(rows),
        "aligned_count": aligned,
        "alignment_anomaly_count": anomalies,
        "records": rows,
    }
    atomic_write_json(target, result)
    return {
        "outputs": [target],
        "metadata": {"backend": "asr_word_timing", "aligned_count": aligned, "anomaly_count": anomalies},
    }
''',
"src/mas/stages/prepare_id.py": '''from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import repo_root
from ..context import EpisodeContext
from ..hashing import atomic_write_json, atomic_write_text
from ..packs.build import build_pack


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _prompt(ctx: EpisodeContext) -> str:
    rules = repo_root() / "rules"
    return f"""You are translating the corrected Turkish dialogue of {ctx.name} into natural Indonesian subtitles.

Return one ZIP named exactly:
{ctx.name}_ID_TRANSLATED.zip

Edit only Indonesian text fields allowed by manifest.json. Preserve every immutable field, block_uid, block_index, timing, schema/hash values, record count, and order. Do not split, merge, add, remove, or reorder records. Keep proper names, numbers, dates, quantities, and currencies correct. Preserve relationship dynamics, intimacy, anger, sarcasm, humor, insults, hesitation, interruptions, unfinished speech, and emotional tone. Do not sanitize dialogue, explain jokes, add translator notes, or add speaker labels. Use natural aku/kamu/nggak/udah/aja where the scene permits, and saya/Anda/Pak/Bu in formal contexts. Put the edited records.json and unchanged manifest.json back in the ZIP.

Canonical subtitle specification:
{_read(rules / 'SUBTITLE_SPEC.md')}

Canonical translation specification:
{_read(rules / 'TRANSLATION_SPEC.md')}

Canonical glossary:
{_read(rules / 'GLOSSARY.json')}

Canonical names:
{_read(rules / 'NAMES.json')}

Canonical rule inventory and precedence notes:
{_read(rules / 'V2_RULE_INVENTORY.md')}
"""


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads((ctx.root / "work" / "aligned.json").read_text(encoding="utf-8"))
    records = payload["records"]
    prompt = _prompt(ctx)
    prompt_path = ctx.root / "translation_input" / "ID_PROMPT.txt"
    pack_path = ctx.root / "translation_input" / f"{ctx.name}_ID_TRANSLATION_PACK.zip"
    manifest_path = ctx.root / "translation_input" / "ID_MANIFEST.json"
    atomic_write_text(prompt_path, prompt)
    manifest = build_pack(
        pack_path,
        ctx.episode,
        "ID",
        records,
        prompt_text=prompt,
        metadata={"translation_register": "natural_contextual_indonesian"},
        context_blocks=ctx.settings.context_blocks,
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "outputs": [pack_path, prompt_path, manifest_path],
        "metadata": {"record_count": len(records), "waiting_for": f"{ctx.name}_ID_TRANSLATED.zip"},
    }
''',
"src/mas/stages/finalize.py": '''from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..context import EpisodeContext
from ..errors import FatalError
from ..hashing import atomic_write_json, atomic_write_text, sha256_file
from ..subtitle.segmentation import wrap_text
from ..subtitle.srt import render


def _source(ctx: EpisodeContext) -> Path | None:
    manifest_path = ctx.root / "source" / "source_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("media_path")
    if not value:
        return None
    path = ctx.root / value
    return path if path.exists() else None


def _mux(ctx: EpisodeContext, source: Path, tr_srt: Path, id_srt: Path, target: Path) -> tuple[bool, str | None]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg unavailable for MKV mux"
    temp = target.with_suffix(".mkv.partial")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-i",
        str(tr_srt),
        "-i",
        str(id_srt),
        "-map",
        "0",
        "-map",
        "1:0",
        "-map",
        "2:0",
        "-c",
        "copy",
        "-metadata:s:s:0",
        "language=tur",
        "-metadata:s:s:0",
        "title=Türkçe",
        "-metadata:s:s:1",
        "language=ind",
        "-metadata:s:s:1",
        "title=Bahasa Indonesia",
        str(temp),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode or not temp.exists() or temp.stat().st_size == 0:
        temp.unlink(missing_ok=True)
        return False, (result.stderr or "MKV mux produced no output")[-2000:]
    os.replace(temp, target)
    return True, None


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    translated = json.loads((ctx.root / "work" / "id_translated.json").read_text(encoding="utf-8"))
    rows = [dict(row) for row in translated["records"]]
    if not rows:
        raise FatalError("no translated subtitle records exist")
    render_rows: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["tr_text"] = wrap_text(str(copy.get("tr_text", "")), ctx.settings.max_line_chars, ctx.settings.max_lines)
        copy["id_text"] = wrap_text(str(copy.get("id_text", "")), ctx.settings.max_line_chars, ctx.settings.max_lines)
        render_rows.append(copy)
    subtitles_dir = ctx.root / "final" / "subtitles"
    tr_srt = subtitles_dir / f"{ctx.name}-tr.srt"
    id_srt = subtitles_dir / f"{ctx.name}-id.srt"
    atomic_write_text(tr_srt, render(render_rows, "tr_text"))
    atomic_write_text(id_srt, render(render_rows, "id_text"))
    final_records = ctx.root / "work" / "final_records.json"
    atomic_write_json(final_records, {"episode": ctx.episode, "records": render_rows})
    outputs = [tr_srt, id_srt, final_records]
    warnings: list[str] = []
    muxed = False
    source = _source(ctx)
    mkv = ctx.root / "final" / f"{ctx.name}.mkv"
    if ctx.settings.mux_mkv and source is not None and not ctx.fixture:
        muxed, warning = _mux(ctx, source, tr_srt, id_srt, mkv)
        if muxed:
            outputs.append(mkv)
        elif warning:
            warnings.append(warning)
    manifest_path = ctx.root / "final" / "manifest.json"
    manifest = {
        "episode": ctx.episode,
        "record_count": len(render_rows),
        "turkish_srt": {"path": str(tr_srt.relative_to(ctx.root)), "sha256": sha256_file(tr_srt)},
        "indonesian_srt": {"path": str(id_srt.relative_to(ctx.root)), "sha256": sha256_file(id_srt)},
        "mkv": {"path": str(mkv.relative_to(ctx.root)), "sha256": sha256_file(mkv)} if muxed else None,
        "warnings": warnings,
    }
    atomic_write_json(manifest_path, manifest)
    outputs.append(manifest_path)
    return {
        "outputs": outputs,
        "metadata": {"record_count": len(render_rows), "muxed": muxed, "warning_count": len(warnings)},
    }
''',
"src/mas/stages/qc.py": '''from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .. import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION
from ..config import git_commit
from ..context import EpisodeContext
from ..hashing import atomic_write_json, atomic_write_text, sha256_file
from ..subtitle.validation import qc as validate_subtitles


def _count_flags(records: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in records:
        counter.update(row.get("flags") or [])
    return counter


def run(ctx: EpisodeContext, state: dict[str, Any]) -> dict[str, Any]:
    final_payload = json.loads((ctx.root / "work" / "final_records.json").read_text(encoding="utf-8"))
    records = final_payload["records"]
    media = json.loads((ctx.root / "work" / "media" / "media_manifest.json").read_text(encoding="utf-8"))
    asr_payload = json.loads((ctx.root / "work" / "asr" / "asr.json").read_text(encoding="utf-8"))
    tr_manifest = json.loads((ctx.root / "translation_input" / "TR_MANIFEST.json").read_text(encoding="utf-8"))
    id_manifest = json.loads((ctx.root / "translation_input" / "ID_MANIFEST.json").read_text(encoding="utf-8"))
    issues = validate_subtitles(records, ctx.settings, "id_text")
    flags = _count_flags(records)
    finalize_metadata = state.get("stages", {}).get("finalize", {}).get("metadata", {})
    for warning in finalize_metadata.get("warnings", []):
        issues.append({"kind": "finalize_warning", "detail": warning, "severity": "review"})
    fatal = [issue for issue in issues if issue.get("severity") == "fatal"]
    review = [issue for issue in issues if issue.get("severity") != "fatal"]
    if fatal:
        status = "FAIL"
    elif review or any(row.get("review_required") for row in records):
        status = "PASS_WITH_REVIEW"
    else:
        status = "PASS"
    stage_runtime = {
        name: round(float(value.get("runtime_seconds", 0.0)), 3)
        for name, value in state.get("stages", {}).items()
        if isinstance(value, dict)
    }
    id_srt = ctx.root / "final" / "subtitles" / f"{ctx.name}-id.srt"
    tr_srt = ctx.root / "final" / "subtitles" / f"{ctx.name}-tr.srt"
    counts = Counter(issue["kind"] for issue in issues)
    report = {
        "status": status,
        "episode": ctx.episode,
        "source_duration_ms": int(media.get("duration_ms", 0)),
        "asr_block_count": int(asr_payload.get("record_count", 0)),
        "final_subtitle_block_count": len(records),
        "unresolved_speech_candidates": flags["unresolved_vad_speech"]
        + counts["unresolved_vad_speech"],
        "asr_hallucination_candidates": flags["suspected_asr_hallucination"],
        "orphan_caption_candidates": flags["orphan_youtube_caption"],
        "speaker_boundary_warnings": counts["speaker_boundary"],
        "timing_warnings": counts["invalid_duration"] + counts["non_increasing_timestamp"],
        "overlaps": counts["overlap"],
        "cps_violations": counts["cps"],
        "too_short_blocks": counts["too_short"],
        "too_long_blocks": counts["too_long"],
        "missing_translations": counts["missing_translation"],
        "immutable_field_validation": "PASS",
        "issues": issues,
        "input_output_hashes": {
            "tr_input_sha256": tr_manifest["input_sha256"],
            "tr_schema_sha256": tr_manifest["schema_sha256"],
            "id_input_sha256": id_manifest["input_sha256"],
            "id_schema_sha256": id_manifest["schema_sha256"],
            "tr_srt_sha256": sha256_file(tr_srt),
            "id_srt_sha256": sha256_file(id_srt),
        },
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": git_commit(),
        "rules_version": RULES_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_versions": asr_payload.get("model", {}),
        "stage_runtime_seconds": stage_runtime,
    }
    json_path = ctx.root / "reports" / "qc.json"
    md_path = ctx.root / "reports" / "qc.md"
    atomic_write_json(json_path, report)
    lines = [
        f"# {ctx.name} QC",
        "",
        f"Status: {status}",
        "",
        f"Source duration: {report['source_duration_ms']} ms",
        f"ASR blocks: {report['asr_block_count']}",
        f"Final subtitle blocks: {report['final_subtitle_block_count']}",
        f"Unresolved speech candidates: {report['unresolved_speech_candidates']}",
        f"ASR hallucination candidates: {report['asr_hallucination_candidates']}",
        f"Orphan caption candidates: {report['orphan_caption_candidates']}",
        f"Overlaps: {report['overlaps']}",
        f"CPS violations: {report['cps_violations']}",
        f"Missing translations: {report['missing_translations']}",
        "",
        "## Issues",
        "",
    ]
    if issues:
        for issue in issues:
            lines.append(f"- {issue.get('severity', 'review')}: {issue.get('kind')} - {json.dumps(issue, ensure_ascii=False, sort_keys=True)}")
    else:
        lines.append("- None")
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    review_items = [
        {
            "review_uid": f"qc-{index:06d}",
            "block_uid": issue.get("block_uid"),
            "reasons": [issue["kind"]],
            "details": issue,
        }
        for index, issue in enumerate(review, 1)
    ]
    review_path = ctx.root / "work" / "review_required.json"
    existing = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {"items": []}
    combined = list(existing.get("items", [])) + review_items
    deduped: dict[str, dict[str, Any]] = {}
    for item in combined:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        deduped[key] = item
    ctx.write_review_items(list(deduped.values()))
    return {"outputs": [json_path, md_path, review_path], "metadata": {"status_result": status, "issue_count": len(issues)}}
''',
"src/mas/stages/archive.py": '''from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..context import EpisodeContext
from ..hashing import atomic_write_json, sha256_file


def run(ctx: EpisodeContext, _state: dict[str, Any]) -> dict[str, Any]:
    required = [
        ctx.root / "final" / "subtitles" / f"{ctx.name}-tr.srt",
        ctx.root / "final" / "subtitles" / f"{ctx.name}-id.srt",
        ctx.root / "reports" / "qc.json",
    ]
    rows: list[dict[str, Any]] = []
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"archive validation failed; required output missing: {path}")
        rows.append(
            {"path": str(path.relative_to(ctx.root)), "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    target = ctx.root / "archive" / "validated_manifest.json"
    payload = {
        "episode": ctx.episode,
        "validated": True,
        "files": rows,
        "intermediates_deleted": False,
        "source_deleted": False,
        "translation_outputs_deleted": False,
    }
    atomic_write_json(target, payload)
    return {"outputs": [target], "metadata": {"validated_file_count": len(rows)}}
''',
"src/mas/pipeline.py": '''from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from . import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION
from .config import Settings, git_commit, repo_root
from .context import EpisodeContext
from .errors import InjectedFailure, InvariantError, WaitingForHandoff
from .hashing import hash_files, sha256_file, sha256_json
from .log import stage_log
from .packs.validate import validate_returned
from .state import invalidate_from, load, output_rows, save, stage_is_valid, utc_now
from .stages import acquire, align, archive, asr, audio_review, captions, finalize, media, prepare_id, prepare_tr, qc, reconcile, vad

StageFunction = Callable[[EpisodeContext, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    label: str
    function: StageFunction
    dependencies: tuple[str, ...]
    config_fields: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage("source", "[1/9] SOURCE", acquire.run, (), ("source_globs",)),
    Stage("media", "[2/9] AUDIO", media.run, ("source",), ("asr_chunk_seconds",)),
    Stage("captions", "[2/9] CAPTIONS", captions.run, ("source",)),
    Stage("vad", "[2/9] VAD", vad.run, ("media",), ("vad_silence_db", "vad_min_silence_seconds")),
    Stage(
        "asr",
        "[3/9] ASR",
        asr.run,
        ("media",),
        ("asr_model", "asr_device", "asr_compute_type", "asr_beam_size", "asr_language", "vad_filter"),
    ),
    Stage("reconcile", "[4/9] RECONCILE", reconcile.run, ("asr", "captions", "vad"), ("unresolved_speech_min_seconds",)),
    Stage("tr_pack", "[5/9] TR PACKAGE", prepare_tr.run, ("reconcile",), ("context_blocks",)),
    Stage("audio_review", "[5/9] AUDIO REVIEW", audio_review.run, ("reconcile", "media"), ("review_clip_padding_ms",)),
    Stage("tr_return", "[6/9] TR VALIDATE", lambda context, state: _tr_return(context), ("tr_pack", "audio_review")),
    Stage("align", "[6/9] ALIGNMENT", align.run, ("tr_return",), ("alignment_enabled",)),
    Stage("id_pack", "[7/9] ID PACKAGE", prepare_id.run, ("align",), ("context_blocks",)),
    Stage("id_return", "[8/9] ID VALIDATE", lambda context, state: _id_return(context), ("id_pack",), ("fail_on_number_change",)),
    Stage("finalize", "[8/9] FINALIZE", finalize.run, ("id_return", "source"), ("mux_mkv", "max_line_chars", "max_lines")),
    Stage("qc", "[9/9] QC", qc.run, ("finalize", "asr", "media", "tr_pack", "id_pack"), ("min_duration_ms", "max_duration_ms", "max_cps", "max_line_chars", "max_lines", "max_overlap_ms")),
    Stage("archive", "[9/9] ARCHIVE", archive.run, ("qc",)),
)
ORDER = [stage.name for stage in STAGES]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise InvariantError(f"records file has an invalid shape: {path}")
    return records


def _tr_return(ctx: EpisodeContext) -> dict[str, Any]:
    returned_path = ctx.root / "translation_output" / f"{ctx.name}_TR_TEXT_CORRECTED.zip"
    if not returned_path.exists():
        raise WaitingForHandoff("TR", returned_path.name, gpu_work_complete=True)
    source_records = _load_records(ctx.root / "work" / "records.json")
    manifest = json.loads((ctx.root / "translation_input" / "TR_MANIFEST.json").read_text(encoding="utf-8"))
    corrected = validate_returned(returned_path, manifest, source_records)
    audio_locked = {
        "unresolved_vad_speech",
        "suspected_asr_hallucination",
        "orphan_youtube_caption",
    }
    for source, result in zip(source_records, corrected, strict=True):
        flags = set(source.get("flags") or [])
        if flags & audio_locked and not source.get("audio_reviewed"):
            if str(result.get("tr_text", "")).strip() != str(source.get("tr_text", "")).strip():
                raise InvariantError(
                    f"audio-dependent Turkish text changed without audio_reviewed=true for {source['block_uid']}"
                )
    target = ctx.root / "work" / "tr_corrected.json"
    timing = {
        row["block_uid"]: {"start_ms": row["start_ms"], "end_ms": row["end_ms"]} for row in source_records
    }
    from .hashing import atomic_write_json

    atomic_write_json(
        target,
        {
            "episode": ctx.episode,
            "return_zip_sha256": sha256_file(returned_path),
            "record_count": len(corrected),
            "source_timing": timing,
            "records": corrected,
        },
    )
    return {"outputs": [target], "metadata": {"record_count": len(corrected), "return_zip_sha256": sha256_file(returned_path)}}


def _id_return(ctx: EpisodeContext) -> dict[str, Any]:
    returned_path = ctx.root / "translation_output" / f"{ctx.name}_ID_TRANSLATED.zip"
    if not returned_path.exists():
        raise WaitingForHandoff("ID", returned_path.name, gpu_work_complete=False)
    source_records = _load_records(ctx.root / "work" / "aligned.json")
    manifest = json.loads((ctx.root / "translation_input" / "ID_MANIFEST.json").read_text(encoding="utf-8"))
    translated = validate_returned(
        returned_path,
        manifest,
        source_records,
        fail_on_number_change=ctx.settings.fail_on_number_change,
    )
    target = ctx.root / "work" / "id_translated.json"
    from .hashing import atomic_write_json

    atomic_write_json(
        target,
        {
            "episode": ctx.episode,
            "return_zip_sha256": sha256_file(returned_path),
            "record_count": len(translated),
            "records": translated,
        },
    )
    return {"outputs": [target], "metadata": {"record_count": len(translated), "return_zip_sha256": sha256_file(returned_path)}}


def _stage_code_fingerprint(stage: Stage) -> str:
    root = repo_root()
    files = [Path(stage.function.__code__.co_filename), root / "src" / "mas" / "pipeline.py"]
    common = [
        root / "src" / "mas" / "state.py",
        root / "src" / "mas" / "hashing.py",
    ]
    if stage.name in {"tr_pack", "tr_return", "id_pack", "id_return"}:
        common.extend([root / "src" / "mas" / "packs" / "build.py", root / "src" / "mas" / "packs" / "validate.py"])
    return hash_files([*files, *common], root)


def _stage_config_fingerprint(settings: Settings, stage: Stage) -> str:
    return sha256_json({field: getattr(settings, field) for field in stage.config_fields})


def _external_fingerprint(ctx: EpisodeContext, stage: Stage) -> dict[str, Any]:
    if stage.name == "source":
        rows: list[dict[str, Any]] = []
        source_dir = ctx.root / "source"
        if source_dir.exists():
            for pattern in ctx.settings.source_globs:
                for path in sorted(source_dir.glob(pattern)):
                    if path.is_file():
                        rows.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
        return {"source_url": ctx.source_url, "fixture": ctx.fixture, "existing": rows}
    if stage.name == "tr_return":
        path = ctx.root / "translation_output" / f"{ctx.name}_TR_TEXT_CORRECTED.zip"
        return {"returned_zip": sha256_file(path) if path.exists() else "missing"}
    if stage.name == "id_return":
        path = ctx.root / "translation_output" / f"{ctx.name}_ID_TRANSLATED.zip"
        return {"returned_zip": sha256_file(path) if path.exists() else "missing"}
    return {}


def _input_fingerprint(ctx: EpisodeContext, state: dict[str, Any], stage: Stage) -> str:
    dependencies: dict[str, Any] = {}
    for name in stage.dependencies:
        value = state.get("stages", {}).get(name, {})
        dependencies[name] = {
            "outputs": value.get("outputs", []),
            "input_fingerprint": value.get("input_fingerprint"),
            "code_fingerprint": value.get("code_fingerprint"),
            "config_fingerprint": value.get("config_fingerprint"),
        }
    return sha256_json(
        {
            "episode": ctx.episode,
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "dependencies": dependencies,
            "external": _external_fingerprint(ctx, stage),
        }
    )


def _print_wait(waiting: WaitingForHandoff) -> None:
    print(f"[WAIT] {waiting.kind} {'CORRECTION' if waiting.kind == 'TR' else 'TRANSLATION'}")
    if waiting.gpu_work_complete:
        print("GPU WORK COMPLETE")
        print("Safe to stop RunPod now.")
    print("\nWaiting for:")
    print(waiting.filename)


def run(episode: int, source_url: str | None = None, fixture: bool = False) -> int:
    settings = Settings.load()
    ctx = EpisodeContext(episode=episode, settings=settings, source_url=source_url, fixture=fixture)
    ctx.ensure_dirs()
    state = load(ctx.state_path, episode)
    state.update(
        {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "git_commit": git_commit(),
        }
    )
    save(ctx.state_path, state)
    fail_after = os.environ.get("MAS_FAIL_AFTER_STAGE")
    for stage in STAGES:
        input_fp = _input_fingerprint(ctx, state, stage)
        code_fp = _stage_code_fingerprint(stage)
        config_fp = _stage_config_fingerprint(settings, stage)
        if stage_is_valid(state, stage.name, input_fp, code_fp, config_fp, ctx.root):
            print(f"{stage.label} SKIP checkpoint valid")
            continue
        invalidated = invalidate_from(state, ORDER, stage.name)
        if invalidated:
            save(ctx.state_path, state)
        print(f"{stage.label} START")
        started = monotonic()
        state["current_stage"] = stage.name
        state["status"] = "RUNNING"
        save(ctx.state_path, state)
        try:
            with stage_log(ctx.root / "logs", stage.name) as (log_handle, _):
                result = stage.function(ctx, state)
                log_handle.write(json.dumps({"event": "result", "metadata": result.get("metadata", {})}, ensure_ascii=False) + "\n")
        except WaitingForHandoff as waiting:
            state["status"] = f"WAITING_{waiting.kind}"
            state["waiting_for"] = waiting.filename
            state["current_stage"] = stage.name
            save(ctx.state_path, state)
            _print_wait(waiting)
            return waiting.exit_code
        runtime = monotonic() - started
        outputs = [Path(path) for path in result.get("outputs", [])]
        state["stages"][stage.name] = {
            "status": "complete",
            "completed_at": utc_now(),
            "input_fingerprint": input_fp,
            "code_fingerprint": code_fp,
            "config_fingerprint": config_fp,
            "outputs": output_rows(outputs, ctx.root),
            "metadata": result.get("metadata", {}),
            "runtime_seconds": runtime,
        }
        state["current_stage"] = None
        state["waiting_for"] = None
        save(ctx.state_path, state)
        print(f"{stage.label} COMPLETE {runtime:.2f} s")
        if fail_after == stage.name:
            raise InjectedFailure(f"injected interruption after completed stage {stage.name}")
    state["status"] = state["stages"].get("qc", {}).get("metadata", {}).get("status_result", "COMPLETE")
    state["current_stage"] = None
    state["waiting_for"] = None
    save(ctx.state_path, state)
    print(f"FINAL STATUS: {state['status']}")
    return 2 if state["status"] == "FAIL" else 0


def status(episode: int, as_json: bool = False) -> int:
    settings = Settings.load()
    ctx = EpisodeContext(episode=episode, settings=settings)
    state = load(ctx.state_path, episode)
    if as_json:
        print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"Episode: {episode}")
    print(f"Status: {state.get('status', 'NEW')}")
    print(f"Pipeline: {state.get('pipeline_version')}")
    print(f"Git commit: {state.get('git_commit')}")
    for stage in STAGES:
        value = state.get("stages", {}).get(stage.name, {})
        print(f"{stage.name:14} {value.get('status', 'pending')}")
    if state.get("waiting_for"):
        print(f"Waiting for: {state['waiting_for']}")
    return 0
''',
"src/mas/cli.py": '''from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import Settings, episode_dir, git_commit, repo_root
from .errors import MasError
from .pipeline import run, status


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def doctor(strict_gpu: bool = False) -> int:
    print(f"ma-sub {__version__} ({git_commit()})")
    failures = 0
    for executable, required in (("python", True), ("ffmpeg", True), ("ffprobe", True), ("git", True), ("yt-dlp", False)):
        found = shutil.which(executable)
        status_value = "OK" if found else ("MISSING" if required else "WARN missing")
        print(f"{executable}: {status_value}{' - ' + found if found else ''}")
        if required and not found:
            failures += 1
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        print(f"torch: {torch.__version__}; cuda={cuda}; torch_cuda={torch.version.cuda}")
    except ImportError:
        print("torch: WARN not installed in this environment")
    print(f"faster-whisper: {_package_version('faster-whisper')}")
    print(f"whisperx: {_package_version('whisperx')}")
    try:
        settings = Settings.load()
        print(f"config: OK - model={settings.asr_model}, device={settings.asr_device}")
    except Exception as exc:
        print(f"config: FAIL - {exc}")
        failures += 1
    if strict_gpu and not cuda:
        print("gpu: FAIL - NVIDIA CUDA is required by --strict-gpu")
        failures += 1
    elif not cuda:
        print("gpu: WARN - CPU-only commands and tests work; real ASR requires the RunPod GPU image")
    else:
        print("gpu: OK")
    return 1 if failures else 0


def test(pytest_args: list[str]) -> int:
    command = [sys.executable, "-m", "pytest", *(pytest_args or ["-q"])]
    return subprocess.call(command, cwd=repo_root())


def clean(episode: int, execute: bool = False) -> int:
    root = episode_dir(episode)
    candidates = [
        root / "work" / "media" / "chunks",
        root / "work" / "audio_review" / "clips",
        root / "logs",
    ]
    protected = [root / "source", root / "translation_input", root / "translation_output", root / "reports", root / "final"]
    print("Protected and never removed by clean:")
    for path in protected:
        print(f"  KEEP {path}")
    for path in candidates:
        action = "DELETE" if execute else "DRY-RUN"
        print(f"{action} {path}")
        if execute and path.exists():
            shutil.rmtree(path)
    if not execute:
        print("No files changed. Pass --execute for the listed cache paths only.")
    return 0


def _failure_log(episode: int | None, command: str, exc: BaseException) -> Path:
    root = episode_dir(episode) if episode else repo_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    target = log_dir / f"{stamp}_{command}_failure.log"
    target.write_text(traceback.format_exc(), encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mas", description="Muhtemel Ask resumable subtitle pipeline")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run or resume one episode")
    run_parser.add_argument("episode", type=int)
    run_parser.add_argument("--source-url")
    run_parser.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)
    status_parser = sub.add_parser("status", help="show episode checkpoints")
    status_parser.add_argument("episode", type=int)
    status_parser.add_argument("--json", action="store_true")
    doctor_parser = sub.add_parser("doctor", help="check runtime dependencies")
    doctor_parser.add_argument("--strict-gpu", action="store_true")
    test_parser = sub.add_parser("test", help="run deterministic test suite")
    test_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    clean_parser = sub.add_parser("clean", help="preview or remove regenerable caches")
    clean_parser.add_argument("episode", type=int)
    clean_parser.add_argument("--dry-run", action="store_true", default=True)
    clean_parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args.episode, args.source_url, args.fixture)
        if args.command == "status":
            return status(args.episode, args.json)
        if args.command == "doctor":
            return doctor(args.strict_gpu)
        if args.command == "test":
            return test(args.pytest_args)
        if args.command == "clean":
            return clean(args.episode, args.execute)
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("INTERRUPTED\nCHECKPOINT PRESERVED: yes", file=sys.stderr)
        return 130
    except (MasError, OSError, ValueError, json.JSONDecodeError) as exc:
        episode = getattr(args, "episode", None)
        log_path = _failure_log(episode, args.command, exc)
        stage = args.command.upper()
        if episode:
            state_path = episode_dir(episode) / "work" / "state.json"
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    stage = str(state.get("current_stage") or stage).upper()
                except (OSError, json.JSONDecodeError):
                    pass
        print(f"FAILED STAGE: {stage}", file=sys.stderr)
        print(f"CAUSE: {exc}", file=sys.stderr)
        print("CHECKPOINT PRESERVED: yes", file=sys.stderr)
        print(f"TRACEBACK: {log_path}", file=sys.stderr)
        if episode:
            print("SAFE RETRY:", file=sys.stderr)
            print(f"./mas run {episode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
"mas": '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m mas.cli "$@"
''',
}
