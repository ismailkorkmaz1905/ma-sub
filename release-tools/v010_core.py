from __future__ import annotations

FILES: dict[str, str] = {
"src/mas/__init__.py": '''from __future__ import annotations

__version__ = "0.1.0"
PIPELINE_VERSION = __version__
SCHEMA_VERSION = "2.0"
RULES_VERSION = "0.1.0"
''',
"src/mas/errors.py": '''from __future__ import annotations

from dataclasses import dataclass


class MasError(RuntimeError):
    """Base error for an operator-visible pipeline failure."""


class FatalError(MasError):
    pass


class StateCorruptError(FatalError):
    pass


class InvariantError(FatalError):
    pass


class DependencyError(FatalError):
    pass


class InjectedFailure(MasError):
    pass


@dataclass(slots=True)
class WaitingForHandoff(MasError):
    kind: str
    filename: str
    gpu_work_complete: bool = False

    @property
    def exit_code(self) -> int:
        return 20 if self.kind == "TR" else 21

    def __str__(self) -> str:
        return f"Waiting for {self.kind} handoff: {self.filename}"
''',
"src/mas/hashing.py": '''from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[str | Path], root: str | Path | None = None) -> str:
    root_path = Path(root).resolve() if root else None
    rows: list[dict[str, str | int]] = []
    for raw in sorted((Path(x) for x in paths), key=lambda x: str(x)):
        if not raw.exists() or not raw.is_file():
            rows.append({"path": str(raw), "missing": 1})
            continue
        resolved = raw.resolve()
        name = str(resolved.relative_to(root_path)) if root_path and resolved.is_relative_to(root_path) else str(raw)
        rows.append({"path": name, "sha256": sha256_file(raw), "size": raw.stat().st_size})
    return sha256_json(rows)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
''',
"src/mas/config.py": '''from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION
from .hashing import hash_files, sha256_json


def repo_root() -> Path:
    override = os.environ.get("MAS_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return Path(os.environ.get("MAS_HOME", repo_root())).expanduser().resolve()


def episode_name(episode: int) -> str:
    if episode < 1:
        raise ValueError("episode must be at least 1")
    return f"Muhtemel Ask {episode}.Bolum"


def episode_dir(episode: int) -> Path:
    return workspace_root() / "EPISODES" / episode_name(episode)


def git_commit() -> str:
    env_value = os.environ.get("MAS_GIT_COMMIT")
    if env_value:
        return env_value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass(slots=True)
class Settings:
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = SCHEMA_VERSION
    rules_version: str = RULES_VERSION
    asr_model: str = "large-v3"
    asr_device: str = "cuda"
    asr_compute_type: str = "float16"
    asr_beam_size: int = 5
    asr_chunk_seconds: int = 900
    asr_language: str = "tr"
    vad_filter: bool = True
    vad_silence_db: str = "-35dB"
    vad_min_silence_seconds: float = 0.35
    unresolved_speech_min_seconds: float = 0.55
    min_duration_ms: int = 700
    max_duration_ms: int = 7000
    max_cps: float = 20.0
    max_line_chars: int = 42
    max_lines: int = 2
    max_overlap_ms: int = 0
    context_blocks: int = 3
    review_clip_padding_ms: int = 750
    alignment_enabled: bool = True
    mux_mkv: bool = True
    fail_on_number_change: bool = True
    source_globs: list[str] = field(default_factory=lambda: ["*.mkv", "*.mp4", "*.webm", "*.mov", "*.m4v"])
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        root = repo_root()
        config_path = Path(os.environ.get("MAS_CONFIG", root / "config" / "pipeline.yaml"))
        raw: dict[str, Any] = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config must be an object: {config_path}")
            raw = loaded
        flattened: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    flattened[f"{key}_{child_key}"] = child_value
            else:
                flattened[key] = value
        aliases = {
            "asr_chunk_duration_seconds": "asr_chunk_seconds",
            "subtitle_min_duration_ms": "min_duration_ms",
            "subtitle_max_duration_ms": "max_duration_ms",
            "subtitle_max_cps": "max_cps",
            "subtitle_max_line_chars": "max_line_chars",
            "subtitle_max_lines": "max_lines",
            "subtitle_max_overlap_ms": "max_overlap_ms",
            "vad_silence_db": "vad_silence_db",
            "vad_min_silence_seconds": "vad_min_silence_seconds",
            "handoff_context_blocks": "context_blocks",
            "audio_review_clip_padding_ms": "review_clip_padding_ms",
            "finalize_mux_mkv": "mux_mkv",
        }
        values: dict[str, Any] = {}
        fields = cls.__dataclass_fields__
        for key, value in flattened.items():
            target = aliases.get(key, key)
            if target in fields and target != "raw":
                values[target] = value
        values["raw"] = raw
        return cls(**values)

    def fingerprint(self) -> str:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "raw"}
        return sha256_json(payload)

    def rules_fingerprint(self) -> str:
        root = repo_root()
        paths = list((root / "rules").glob("*")) + [root / "config" / "pipeline.yaml"]
        return hash_files([p for p in paths if p.is_file()], root)
''',
"src/mas/state.py": '''from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION
from .config import git_commit
from .errors import StateCorruptError
from .hashing import atomic_write_json, sha256_file

STATE_FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_state(episode: int) -> dict[str, Any]:
    now = utc_now()
    return {
        "state_format_version": STATE_FORMAT_VERSION,
        "episode": episode,
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "git_commit": git_commit(),
        "created_at": now,
        "updated_at": now,
        "status": "NEW",
        "stages": {},
    }


def load(path: str | Path, episode: int) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return new_state(episode)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateCorruptError(f"corrupted state file {p}: {exc}") from exc
    if not isinstance(data, dict) or data.get("episode") != episode or not isinstance(data.get("stages"), dict):
        raise StateCorruptError(f"state file belongs to another episode or has an invalid shape: {p}")
    if data.get("state_format_version") != STATE_FORMAT_VERSION:
        raise StateCorruptError(
            f"unsupported state format {data.get('state_format_version')}; expected {STATE_FORMAT_VERSION}"
        )
    return data


def save(path: str | Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def output_rows(paths: Iterable[Path], episode_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda x: str(x)):
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"stage output is missing: {path}")
        rows.append(
            {
                "path": str(path.resolve().relative_to(episode_root.resolve())),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return rows


def stage_is_valid(
    state: dict[str, Any],
    name: str,
    input_fingerprint: str,
    code_fingerprint: str,
    config_fingerprint: str,
    episode_root: Path,
) -> bool:
    stage = state.get("stages", {}).get(name)
    if not isinstance(stage, dict) or stage.get("status") != "complete":
        return False
    if stage.get("input_fingerprint") != input_fingerprint:
        return False
    if stage.get("code_fingerprint") != code_fingerprint:
        return False
    if stage.get("config_fingerprint") != config_fingerprint:
        return False
    for row in stage.get("outputs", []):
        path = episode_root / row["path"]
        if not path.exists() or not path.is_file():
            return False
        if path.stat().st_size != row.get("size") or sha256_file(path) != row.get("sha256"):
            return False
    return True


def invalidate_from(state: dict[str, Any], ordered_names: list[str], name: str) -> list[str]:
    try:
        start = ordered_names.index(name)
    except ValueError:
        return []
    removed: list[str] = []
    for stage_name in ordered_names[start:]:
        if stage_name in state.get("stages", {}):
            removed.append(stage_name)
            state["stages"].pop(stage_name, None)
    return removed
''',
"src/mas/context.py": '''from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings, episode_dir, episode_name
from .hashing import atomic_write_json


@dataclass(slots=True)
class EpisodeContext:
    episode: int
    settings: Settings
    source_url: str | None = None
    fixture: bool = False

    @property
    def root(self) -> Path:
        return episode_dir(self.episode)

    @property
    def name(self) -> str:
        return episode_name(self.episode)

    @property
    def state_path(self) -> Path:
        return self.root / "work" / "state.json"

    def ensure_dirs(self) -> None:
        for rel in (
            "source",
            "work",
            "work/media/chunks",
            "work/asr/chunks",
            "work/audio_review/clips",
            "translation_input",
            "translation_output",
            "reports",
            "final/subtitles",
            "logs",
            "archive",
        ):
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    def write_review_items(self, items: list[dict[str, Any]]) -> Path:
        target = self.root / "work" / "review_required.json"
        atomic_write_json(target, {"episode": self.episode, "count": len(items), "items": items})
        return target
''',
"src/mas/log.py": '''from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Iterator, TextIO


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


@contextmanager
def stage_log(log_dir: Path, stage: str) -> Iterator[tuple[TextIO, float]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{stamp()}_{stage}.log"
    started = monotonic()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "stage": stage, "utc": stamp()}) + "\n")
        handle.flush()
        try:
            yield handle, started
        except Exception:
            handle.write(traceback.format_exc())
            handle.flush()
            raise
        else:
            handle.write(json.dumps({"event": "complete", "stage": stage, "seconds": monotonic() - started}) + "\n")
            handle.flush()
''',
"src/mas/subtitle/__init__.py": '''from .models import SubtitleBlock
from .srt import render
from .validation import qc

__all__ = ["SubtitleBlock", "render", "qc"]
''',
"src/mas/subtitle/models.py": '''from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SubtitleBlock:
    block_uid: str
    block_index: int
    start_ms: int
    end_ms: int
    tr_text: str = ""
    id_text: str = ""
    speaker: str | None = None
    provenance: list[dict[str, Any]] = field(default_factory=list)
    review_required: bool = False
    flags: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SubtitleBlock":
        known = {
            "block_uid",
            "block_index",
            "start_ms",
            "end_ms",
            "tr_text",
            "id_text",
            "speaker",
            "provenance",
            "review_required",
            "flags",
        }
        return cls(
            block_uid=str(row["block_uid"]),
            block_index=int(row["block_index"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            tr_text=str(row.get("tr_text") or row.get("corrected_tr") or row.get("text") or ""),
            id_text=str(row.get("id_text") or row.get("translated_text") or ""),
            speaker=row.get("speaker"),
            provenance=list(row.get("provenance") or []),
            review_required=bool(row.get("review_required", False)),
            flags=list(row.get("flags") or []),
            extras={key: value for key, value in row.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        row = dict(self.extras)
        row.update(
            {
                "block_uid": self.block_uid,
                "block_index": self.block_index,
                "start_ms": self.start_ms,
                "end_ms": self.end_ms,
                "tr_text": self.tr_text,
                "id_text": self.id_text,
                "speaker": self.speaker,
                "provenance": self.provenance,
                "review_required": self.review_required,
                "flags": self.flags,
            }
        )
        return row
''',
"src/mas/subtitle/timing.py": '''from __future__ import annotations

from typing import Any


def duration_ms(row: dict[str, Any]) -> int:
    return int(row["end_ms"]) - int(row["start_ms"])


def overlap_ms(left: dict[str, Any], right: dict[str, Any]) -> int:
    return max(0, int(left["end_ms"]) - int(right["start_ms"]))


def temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    start = max(int(left["start_ms"]), int(right["start_ms"]))
    end = min(int(left["end_ms"]), int(right["end_ms"]))
    intersection = max(0, end - start)
    union = max(int(left["end_ms"]), int(right["end_ms"])) - min(
        int(left["start_ms"]), int(right["start_ms"])
    )
    return intersection / union if union else 0.0


def intersects(left: dict[str, Any], right: dict[str, Any], tolerance_ms: int = 0) -> bool:
    return int(left["start_ms"]) < int(right["end_ms"]) + tolerance_ms and int(right["start_ms"]) < int(
        left["end_ms"]
    ) + tolerance_ms
''',
"src/mas/subtitle/speakers.py": '''from __future__ import annotations

import re
from typing import Any

_LABEL = re.compile(r"^\s*(?:[-–—]\s*)?(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9 _-]{1,24}:)\s*")


def strip_visible_speaker_label(text: str) -> str:
    return _LABEL.sub("", text, count=1).strip()


def may_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_speaker = left.get("speaker")
    right_speaker = right.get("speaker")
    if left_speaker and right_speaker and left_speaker != right_speaker:
        return False
    if left.get("speaker_change_after") or right.get("speaker_change_before"):
        return False
    return True
''',
"src/mas/subtitle/segmentation.py": '''from __future__ import annotations

import re


def wrap_text(text: str, max_chars: int = 42, max_lines: int = 2) -> str:
    clean = re.sub(r"[ \t]+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    if len(clean) <= max_chars:
        return clean
    words = clean.split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars or len(lines) >= max_lines - 1:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > max_lines:
        return "\n".join(lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])])
    return "\n".join(lines)
''',
"src/mas/subtitle/srt.py": '''from __future__ import annotations

import re
from typing import Any, Iterable

from ..errors import InvariantError
from .speakers import strip_visible_speaker_label

_METADATA = re.compile(r"\b(?:block_uid|schema_sha256|schema_version|asr_audit|review_required)\b", re.I)


def timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise InvariantError(f"negative subtitle timestamp: {milliseconds}")
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render(records: Iterable[dict[str, Any]], field: str, *, strip_speaker_labels: bool = True) -> str:
    blocks: list[str] = []
    previous_start = -1
    seen: set[str] = set()
    for ordinal, row in enumerate(records, start=1):
        uid = str(row.get("block_uid", ""))
        if not uid or uid in seen:
            raise InvariantError(f"duplicate or missing block_uid at record {ordinal}: {uid!r}")
        seen.add(uid)
        start = int(row["start_ms"])
        end = int(row["end_ms"])
        if start < previous_start or end <= start:
            raise InvariantError(f"invalid timing for {uid}: {start}-{end}")
        previous_start = start
        text = str(row.get(field) or "").strip()
        if strip_speaker_labels:
            text = strip_visible_speaker_label(text)
        if not text:
            raise InvariantError(f"blank {field} for {uid}")
        if _METADATA.search(text):
            raise InvariantError(f"metadata leaked into subtitle text for {uid}")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks.append(f"{ordinal}\n{timestamp(start)} --> {timestamp(end)}\n{text}")
    if not blocks:
        raise InvariantError("cannot render an empty SRT")
    return "\n\n".join(blocks) + "\n"
''',
"src/mas/subtitle/validation.py": '''from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .timing import duration_ms, overlap_ms


def visible_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def qc(records: Iterable[dict[str, Any]], settings: Any | None = None, field: str = "id_text") -> list[dict[str, Any]]:
    min_duration = int(getattr(settings, "min_duration_ms", 700))
    max_duration = int(getattr(settings, "max_duration_ms", 7000))
    max_cps = float(getattr(settings, "max_cps", 20.0))
    max_chars = int(getattr(settings, "max_line_chars", 42))
    max_lines = int(getattr(settings, "max_lines", 2))
    max_overlap = int(getattr(settings, "max_overlap_ms", 0))
    rows = list(records)
    issues: list[dict[str, Any]] = []
    uids = [str(row.get("block_uid", "")) for row in rows]
    for uid, count in Counter(uids).items():
        if not uid or count > 1:
            issues.append({"kind": "duplicate_block", "block_uid": uid, "count": count, "severity": "fatal"})
    previous: dict[str, Any] | None = None
    for row in rows:
        uid = str(row.get("block_uid", ""))
        start = int(row.get("start_ms", 0))
        end = int(row.get("end_ms", 0))
        length = end - start
        text = str(row.get(field) or "").strip()
        if end <= start:
            issues.append({"kind": "invalid_duration", "block_uid": uid, "duration_ms": length, "severity": "fatal"})
        elif length < min_duration:
            issues.append({"kind": "too_short", "block_uid": uid, "duration_ms": length, "severity": "review"})
        elif length > max_duration:
            issues.append({"kind": "too_long", "block_uid": uid, "duration_ms": length, "severity": "review"})
        if not text:
            issues.append({"kind": "missing_translation", "block_uid": uid, "severity": "fatal"})
        elif length > 0:
            cps = visible_chars(text) / (length / 1000)
            if cps > max_cps:
                issues.append({"kind": "cps", "block_uid": uid, "cps": round(cps, 2), "severity": "review"})
        lines = text.splitlines() if text else []
        if len(lines) > max_lines:
            issues.append({"kind": "line_count", "block_uid": uid, "lines": len(lines), "severity": "review"})
        for line_number, line in enumerate(lines, 1):
            if len(line) > max_chars:
                issues.append(
                    {
                        "kind": "line_length",
                        "block_uid": uid,
                        "line": line_number,
                        "characters": len(line),
                        "severity": "review",
                    }
                )
        if previous is not None:
            overlap = overlap_ms(previous, row)
            if overlap > max_overlap:
                issues.append(
                    {
                        "kind": "overlap",
                        "block_uid": uid,
                        "previous_block_uid": previous.get("block_uid"),
                        "overlap_ms": overlap,
                        "severity": "review",
                    }
                )
            if start < int(previous.get("start_ms", 0)):
                issues.append({"kind": "non_increasing_timestamp", "block_uid": uid, "severity": "fatal"})
            left_speaker = previous.get("speaker")
            right_speaker = row.get("speaker")
            if left_speaker and right_speaker and left_speaker != right_speaker and row.get("merged_from"):
                issues.append({"kind": "speaker_boundary", "block_uid": uid, "severity": "review"})
        previous = row
        for flag in row.get("flags") or []:
            if flag in {
                "unresolved_vad_speech",
                "suspected_asr_hallucination",
                "orphan_youtube_caption",
                "uncertain_speaker",
                "alignment_anomaly",
            }:
                issues.append({"kind": flag, "block_uid": uid, "severity": "review"})
    return issues
''',
"src/mas/packs/__init__.py": '''from .build import build_pack
from .validate import validate_returned

__all__ = ["build_pack", "validate_returned"]
''',
"src/mas/packs/hashing.py": '''from __future__ import annotations

from typing import Any

from ..hashing import sha256_json

TR_MUTABLE_FIELDS = {"tr_text", "tr_text_corrected", "corrected_tr", "text_corrected"}
ID_MUTABLE_FIELDS = {"id_text", "translated_text", "translation"}
DERIVED_CONTEXT_FIELDS = {"context_before", "context_after", "handoff_context"}


def mutable_fields(kind: str) -> set[str]:
    normalized = kind.upper()
    if normalized == "TR":
        return set(TR_MUTABLE_FIELDS)
    if normalized == "ID":
        return set(ID_MUTABLE_FIELDS)
    raise ValueError(f"unsupported pack kind: {kind}")


def immutable_projection(record: dict[str, Any], kind: str) -> dict[str, Any]:
    excluded = mutable_fields(kind) | DERIVED_CONTEXT_FIELDS
    return {key: value for key, value in record.items() if key not in excluded}


def immutable_hash(record: dict[str, Any], kind: str) -> str:
    return sha256_json(immutable_projection(record, kind))


def schema_descriptor(records: list[dict[str, Any]], kind: str, schema_version: str) -> dict[str, Any]:
    fields: dict[str, set[str]] = {}
    for row in records:
        for key, value in row.items():
            fields.setdefault(key, set()).add(type(value).__name__)
    return {
        "schema_version": schema_version,
        "kind": kind.upper(),
        "fields": {key: sorted(types) for key, types in sorted(fields.items())},
        "mutable_fields": sorted(mutable_fields(kind)),
        "required_immutable_fields": ["block_uid", "block_index", "start_ms", "end_ms"],
    }
''',
"src/mas/packs/build.py": '''from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .. import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION
from ..config import git_commit
from ..errors import InvariantError
from ..hashing import canonical_json_bytes, sha256_bytes, sha256_json
from .hashing import immutable_hash, schema_descriptor

_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _zip_write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def _contextualize(records: list[dict[str, Any]], radius: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        row = dict(source)
        row["context_before"] = [
            {
                "block_uid": records[pos].get("block_uid"),
                "tr_text": records[pos].get("tr_text", ""),
            }
            for pos in range(max(0, index - radius), index)
        ]
        row["context_after"] = [
            {
                "block_uid": records[pos].get("block_uid"),
                "tr_text": records[pos].get("tr_text", ""),
            }
            for pos in range(index + 1, min(len(records), index + radius + 1))
        ]
        result.append(row)
    return result


def build_pack(
    path: str | Path,
    episode: int,
    kind: str,
    records: list[dict[str, Any]],
    prompt_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    context_blocks: int = 3,
) -> dict[str, Any]:
    normalized = kind.upper()
    if normalized not in {"TR", "ID"}:
        raise ValueError(f"unsupported pack kind: {kind}")
    if not records:
        raise InvariantError("translation pack cannot be empty")
    uids = [str(row.get("block_uid", "")) for row in records]
    indexes = [int(row.get("block_index", -1)) for row in records]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise InvariantError("translation pack has a missing or duplicate block_uid")
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        raise InvariantError("translation pack block_index values are missing, duplicated, or reordered")
    required = {"block_uid", "block_index", "start_ms", "end_ms"}
    for row in records:
        missing = required - row.keys()
        if missing:
            raise InvariantError(f"record {row.get('block_uid')} is missing immutable fields: {sorted(missing)}")
    schema = schema_descriptor(records, normalized, SCHEMA_VERSION)
    schema_sha = sha256_json(schema)
    source_sha = sha256_json(records)
    immutable = [immutable_hash(row, normalized) for row in records]
    manifest: dict[str, Any] = {
        "format": "ma-sub-chatgpt-handoff",
        "format_version": 1,
        "episode": episode,
        "episode_name": f"Muhtemel Ask {episode}.Bolum",
        "kind": normalized,
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": git_commit(),
        "rules_version": RULES_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha,
        "input_sha256": source_sha,
        "record_count": len(records),
        "block_uids": uids,
        "block_indexes": indexes,
        "immutable_record_sha256": immutable,
        "schema": schema,
        "metadata": metadata or {},
    }
    payload = _contextualize(records, context_blocks)
    records_bytes = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    manifest["records_payload_sha256"] = sha256_bytes(records_bytes)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w") as archive:
            _zip_write(archive, "manifest.json", manifest_bytes)
            _zip_write(archive, "records.json", records_bytes)
            _zip_write(archive, "schema.json", canonical_json_bytes(schema) + b"\n")
            _zip_write(
                archive,
                "PROMPT.txt",
                (prompt_text or "Use the generated prompt beside this package. Preserve every immutable field.\n").encode("utf-8"),
            )
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return manifest
''',
"src/mas/packs/validate.py": '''from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from ..errors import InvariantError
from ..hashing import sha256_bytes, sha256_json
from .hashing import immutable_hash, mutable_fields

_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,]\d+)?(?!\w)")
_CURRENCY = re.compile(r"(?:[$€£₺¥₱]|\b(?:TL|TRY|USD|EUR|IDR|PHP|SGD)\b)", re.I)


def _read_json(archive: zipfile.ZipFile, name: str) -> Any:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise InvariantError(f"returned ZIP is missing {name}") from exc
    if info.file_size > _MAX_MEMBER_BYTES:
        raise InvariantError(f"returned ZIP member is too large: {name}")
    return json.loads(archive.read(info).decode("utf-8"))


def _safe_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        pure = Path(info.filename)
        if pure.is_absolute() or ".." in pure.parts:
            raise InvariantError(f"unsafe ZIP member path: {info.filename}")
        if info.file_size > _MAX_MEMBER_BYTES:
            raise InvariantError(f"returned ZIP member is too large: {info.filename}")


def _output_text(row: dict[str, Any], kind: str) -> str:
    candidates = (
        ("tr_text_corrected", "corrected_tr", "text_corrected", "tr_text")
        if kind == "TR"
        else ("id_text", "translated_text", "translation")
    )
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_text(row: dict[str, Any]) -> str:
    for key in ("tr_text", "tr_text_corrected", "corrected_tr", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _validate_numbers(source: str, translated: str, uid: str) -> None:
    if _NUMBER.findall(source) != _NUMBER.findall(translated):
        raise InvariantError(f"numbers changed in translated record {uid}")
    if _CURRENCY.findall(source) != _CURRENCY.findall(translated):
        raise InvariantError(f"currency marker changed in translated record {uid}")


def validate_returned(
    zip_path: str | Path,
    expected_manifest: dict[str, Any],
    source_records: list[dict[str, Any]],
    *,
    fail_on_number_change: bool = True,
) -> list[dict[str, Any]]:
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            _safe_members(archive)
            manifest = _read_json(archive, "manifest.json")
            records = _read_json(archive, "records.json")
            records_info = archive.getinfo("records.json")
            records_bytes = archive.read(records_info)
    except zipfile.BadZipFile as exc:
        raise InvariantError(f"returned package is not a valid ZIP: {path.name}") from exc
    if not isinstance(manifest, dict) or not isinstance(records, list):
        raise InvariantError("returned package manifest/records shape is invalid")
    exact_fields = (
        "format",
        "format_version",
        "episode",
        "episode_name",
        "kind",
        "rules_version",
        "schema_version",
        "schema_sha256",
        "input_sha256",
        "record_count",
        "block_uids",
        "block_indexes",
        "immutable_record_sha256",
    )
    for field in exact_fields:
        if manifest.get(field) != expected_manifest.get(field):
            raise InvariantError(
                f"returned package invariant failed: {field}; expected {expected_manifest.get(field)!r}, got {manifest.get(field)!r}"
            )
    if manifest.get("records_payload_sha256") and manifest.get("records_payload_sha256") != sha256_bytes(records_bytes):
        # ChatGPT is expected to change records, so its own manifest must be updated only when a return_manifest is used.
        # Preserve compatibility with V2 packages by not treating the original payload hash as immutable.
        pass
    if len(records) != len(source_records):
        raise InvariantError(
            f"returned package record count changed: expected {len(source_records)}, got {len(records)}"
        )
    kind = str(expected_manifest["kind"]).upper()
    expected_uids = expected_manifest["block_uids"]
    expected_indexes = expected_manifest["block_indexes"]
    returned_uids = [str(row.get("block_uid", "")) for row in records]
    returned_indexes = [int(row.get("block_index", -1)) for row in records]
    if returned_uids != expected_uids:
        missing = sorted(set(expected_uids) - set(returned_uids))
        added = sorted(set(returned_uids) - set(expected_uids))
        raise InvariantError(f"block_uid order/content changed; missing={missing[:10]}, added={added[:10]}")
    if returned_indexes != expected_indexes:
        raise InvariantError("block_index order/content changed")
    result: list[dict[str, Any]] = []
    mutable = mutable_fields(kind)
    for position, (source, returned) in enumerate(zip(source_records, records, strict=True)):
        uid = expected_uids[position]
        if immutable_hash(returned, kind) != expected_manifest["immutable_record_sha256"][position]:
            source_projection = {key: value for key, value in source.items() if key not in mutable and key not in {"context_before", "context_after", "handoff_context"}}
            returned_projection = {key: value for key, value in returned.items() if key not in mutable and key not in {"context_before", "context_after", "handoff_context"}}
            changed = sorted(
                key
                for key in set(source_projection) | set(returned_projection)
                if source_projection.get(key) != returned_projection.get(key)
            )
            raise InvariantError(f"immutable field changed for {uid}: {changed}")
        text = _output_text(returned, kind)
        if not text:
            raise InvariantError(f"returned package has no {kind} text for {uid}")
        merged = dict(source)
        if kind == "TR":
            merged["tr_text_original"] = _source_text(source)
            merged["tr_text"] = text
            merged["tr_text_corrected"] = text
            merged["text_provenance"] = "chatgpt_tr_correction"
        else:
            merged["id_text"] = text
            merged["text_provenance_id"] = "chatgpt_id_translation"
            if fail_on_number_change:
                _validate_numbers(_source_text(source), text, uid)
        result.append(merged)
    if sha256_json([row.get("block_uid") for row in result]) != sha256_json(expected_uids):
        raise InvariantError("internal return validation order check failed")
    return result
''',
}
