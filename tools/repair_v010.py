from __future__ import annotations

import json
import textwrap
from pathlib import Path


def write(path: str, content: str, executable: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    if executable:
        target.chmod(0o755)


def complete_runtime_exists() -> bool:
    pipeline = Path("src/mas/pipeline.py")
    required = [
        pipeline,
        Path("src/mas/state.py"),
        Path("src/mas/packs/build.py"),
        Path("src/mas/packs/validate.py"),
        Path("rules/TRANSLATION_SPEC.md"),
        Path("rules/GLOSSARY.json"),
        Path("rules/NAMES.json"),
        Path("runpod/preflight.sh"),
    ]
    if not all(path.exists() for path in required):
        return False
    text = pipeline.read_text(encoding="utf-8", errors="ignore")
    return (
        "GPU source acquisition/ASR CLI wiring remains the v0.1.0 RunPod integration blocker" not in text
        and "input_sha256" in text
        and "checkpoint" in text.lower()
    )


if complete_runtime_exists():
    print("Production runtime already complete; repair skipped.")
    raise SystemExit(0)

write(
    "src/mas/version.py",
    """
    PIPELINE_VERSION = "0.1.0"
    SCHEMA_VERSION = "mas.subtitle.v2"
    RULES_VERSION = "v2-beta-2026-09-04"
    """,
)

write(
    "src/mas/config.py",
    """
    from __future__ import annotations

    import os
    from pathlib import Path


    def repo_root() -> Path:
        return Path(__file__).resolve().parents[2]


    def workspace_root() -> Path:
        return Path(os.environ.get("MAS_WORKSPACE", repo_root())).expanduser().resolve()


    def episode_name(episode: int) -> str:
        return f"Muhtemel Ask {episode}.Bolum"


    def episode_dir(episode: int) -> Path:
        return workspace_root() / "EPISODES" / episode_name(episode)


    def rules_dir() -> Path:
        return repo_root() / "rules"
    """,
)

write(
    "src/mas/hashing.py",
    """
    from __future__ import annotations

    import hashlib
    import json
    from pathlib import Path


    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def canonical_json(value: object) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


    def sha256_json(value: object) -> str:
        return hashlib.sha256(canonical_json(value)).hexdigest()
    """,
)

write(
    "src/mas/errors.py",
    """
    class MasError(RuntimeError):
        pass


    class PackageValidationError(MasError):
        pass
    """,
)

write(
    "src/mas/state.py",
    """
    from __future__ import annotations

    import json
    import os
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    from .hashing import sha256_file
    from .version import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION


    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def fresh_state(episode: int) -> dict:
        return {
            "episode": episode,
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "stages": {},
        }


    def load(path: Path, episode: int) -> dict:
        path = Path(path)
        if not path.exists():
            return fresh_state(episode)
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Corrupted state file {path}: {exc}") from exc
        if state.get("episode") != episode:
            raise RuntimeError(
                f"State episode mismatch: expected {episode}, got {state.get('episode')}"
            )
        state.setdefault("stages", {})
        return state


    def save(path: Path, state: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = utc_now()
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


    def checkpoint_valid(
        stage: dict,
        input_hash: str,
        config_hash: str,
        outputs: list[Path],
    ) -> bool:
        if stage.get("status") != "complete":
            return False
        if stage.get("input_sha256") != input_hash:
            return False
        if stage.get("config_sha256") != config_hash:
            return False
        expected = stage.get("outputs", {})
        for output in outputs:
            output = Path(output)
            if not output.exists() or expected.get(str(output)) != sha256_file(output):
                return False
        return True
    """,
)

write(
    "src/mas/packs/build.py",
    """
    from __future__ import annotations

    import json
    import zipfile
    from pathlib import Path

    from ..hashing import sha256_json
    from ..version import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION

    IMMUTABLE_FIELDS = [
        "block_uid",
        "block_index",
        "start_ms",
        "end_ms",
        "asr_audit",
        "audio_reviewed",
        "review_disposition",
        "schema_version",
        "schema_sha256",
    ]


    def schema_descriptor(kind: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "immutable_fields": IMMUTABLE_FIELDS,
            "editable_field": "tr_text" if kind == "TR" else "id_text",
        }


    def build_pack(
        path: Path,
        episode: int,
        kind: str,
        records: list[dict],
        prompt_text: str = "",
    ) -> dict:
        kind = kind.upper()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = schema_descriptor(kind)
        schema_hash = sha256_json(schema)
        packaged = []
        for record in records:
            item = dict(record)
            item["schema_version"] = SCHEMA_VERSION
            item["schema_sha256"] = schema_hash
            packaged.append(item)
        manifest = {
            "episode": episode,
            "kind": kind,
            "pipeline_version": PIPELINE_VERSION,
            "rules_version": RULES_VERSION,
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": schema_hash,
            "input_sha256": sha256_json(packaged),
            "record_count": len(packaged),
            "immutable_fields": IMMUTABLE_FIELDS,
            "records_file": "records.json",
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            archive.writestr(
                "records.json",
                json.dumps(packaged, ensure_ascii=False, indent=2) + "\n",
            )
            archive.writestr("INSTRUCTIONS.md", prompt_text)
        return manifest
    """,
)

write(
    "src/mas/packs/validate.py",
    """
    from __future__ import annotations

    import json
    import zipfile
    from pathlib import Path

    from ..errors import PackageValidationError
    from .build import IMMUTABLE_FIELDS


    def read_package(path: Path) -> tuple[dict, list[dict]]:
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if not {"manifest.json", "records.json"}.issubset(names):
                    raise PackageValidationError(
                        "ZIP must contain manifest.json and records.json"
                    )
                return (
                    json.loads(archive.read("manifest.json")),
                    json.loads(archive.read("records.json")),
                )
        except PackageValidationError:
            raise
        except Exception as exc:
            raise PackageValidationError(f"Invalid ZIP: {exc}") from exc


    def validate_returned(
        path: Path,
        expected_manifest: dict,
        expected_records: list[dict],
    ) -> list[dict]:
        manifest, records = read_package(path)
        for field in (
            "episode",
            "kind",
            "schema_version",
            "schema_sha256",
            "input_sha256",
            "record_count",
        ):
            if manifest.get(field) != expected_manifest.get(field):
                raise PackageValidationError(
                    f"{field} mismatch: expected {expected_manifest.get(field)!r}, "
                    f"got {manifest.get(field)!r}"
                )
        if len(records) != len(expected_records):
            raise PackageValidationError(
                f"record count mismatch: expected {len(expected_records)}, got {len(records)}"
            )
        seen = set()
        for position, (before, after) in enumerate(zip(expected_records, records)):
            if not isinstance(after, dict):
                raise PackageValidationError(f"record {position} is not an object")
            if after.get("block_uid") != before.get("block_uid"):
                raise PackageValidationError(
                    f"block_uid/order changed at record {position}"
                )
            if after.get("block_index") != before.get("block_index"):
                raise PackageValidationError(
                    f"block_index/order changed at record {position}"
                )
            for field in IMMUTABLE_FIELDS:
                if field in before and after.get(field) != before.get(field):
                    raise PackageValidationError(
                        f"immutable field changed at record {position}, "
                        f"uid={before.get('block_uid')}: {field}"
                    )
            uid = after.get("block_uid")
            if uid in seen:
                raise PackageValidationError("duplicate block_uid in returned package")
            seen.add(uid)
        return records
    """,
)

write(
    "src/mas/subtitle/srt.py",
    """
    from __future__ import annotations

    import re


    def timestamp(milliseconds: int) -> str:
        milliseconds = max(0, int(milliseconds))
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        seconds, milliseconds = divmod(milliseconds, 1_000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


    def render(records: list[dict], field: str = "id_text") -> str:
        output = []
        index = 0
        for record in records:
            text = str(record.get(field) or "").replace("\r", "").strip()
            text = re.sub(
                r"(?im)^\s*(speaker|konuşmacı)\s*\w*\s*:\s*",
                "",
                text,
            )
            if not text:
                continue
            index += 1
            output.extend(
                [
                    str(index),
                    f"{timestamp(record['start_ms'])} --> {timestamp(record['end_ms'])}",
                    text,
                    "",
                ]
            )
        return "\n".join(output).rstrip() + "\n"
    """,
)

write(
    "src/mas/subtitle/validation.py",
    """
    from __future__ import annotations

    import re

    DEFAULTS = {
        "min_duration_ms": 500,
        "max_duration_ms": 8_000,
        "max_cps": 22.0,
        "max_line_length": 42,
        "max_lines": 2,
    }


    def qc(records: list[dict], thresholds: dict | None = None) -> list[dict]:
        limits = {**DEFAULTS, **(thresholds or {})}
        issues = []
        previous_end = -1
        seen = set()
        for record in records:
            uid = record.get("block_uid")
            start = int(record.get("start_ms", -1))
            end = int(record.get("end_ms", -1))
            text = str(record.get("id_text") or record.get("tr_text") or "")

            def add(kind: str, **extra: object) -> None:
                issues.append(
                    {
                        "type": kind,
                        "block_uid": uid,
                        "block_index": record.get("block_index"),
                        **extra,
                    }
                )

            if uid in seen:
                add("duplicate_block_uid")
            seen.add(uid)
            if end <= start:
                add("invalid_duration", start_ms=start, end_ms=end)
            duration = end - start
            if duration < limits["min_duration_ms"]:
                add("too_short", duration_ms=duration)
            if duration > limits["max_duration_ms"]:
                add("too_long", duration_ms=duration)
            if start < previous_end:
                add("overlap", previous_end_ms=previous_end, start_ms=start)
            previous_end = max(previous_end, end)
            if not text.strip():
                add("missing_text")
            cps = len(re.sub(r"\s+", "", text)) / (duration / 1000) if duration > 0 else 999
            if cps > limits["max_cps"]:
                add("cps", cps=round(cps, 2))
            lines = text.splitlines() or [""]
            if len(lines) > limits["max_lines"]:
                add("too_many_lines", lines=len(lines))
            if max(map(len, lines)) > limits["max_line_length"]:
                add("line_too_long", length=max(map(len, lines)))
            if re.search(r"(?i)^\s*(speaker|konuşmacı)\b", text):
                add("speaker_label_leak")
        return issues
    """,
)

write(
    "src/mas/pipeline.py",
    r'''
    from __future__ import annotations

    import json
    import os
    import re
    import shutil
    import subprocess
    import time
    import uuid
    from pathlib import Path

    from .config import episode_dir, episode_name, rules_dir
    from .hashing import sha256_file, sha256_json
    from .packs.build import build_pack
    from .packs.validate import validate_returned
    from .state import checkpoint_valid, load, save, utc_now
    from .subtitle.srt import render
    from .subtitle.validation import qc
    from .version import PIPELINE_VERSION, RULES_VERSION, SCHEMA_VERSION

    STAGES = [
        "source",
        "audio",
        "asr",
        "reconcile",
        "tr_pack",
        "tr_return",
        "align",
        "id_pack",
        "id_return",
        "finalize",
        "qc",
    ]


    def ensure_directories(episode: int) -> Path:
        directory = episode_dir(episode)
        for relative in (
            "source",
            "work",
            "translation_input",
            "translation_output",
            "reports",
            "final/subtitles",
            "logs",
        ):
            (directory / relative).mkdir(parents=True, exist_ok=True)
        return directory


    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))


    def config_fingerprint() -> str:
        files = sorted(path for path in rules_dir().rglob("*") if path.is_file())
        return sha256_json(
            [
                PIPELINE_VERSION,
                SCHEMA_VERSION,
                RULES_VERSION,
                [
                    (str(path.relative_to(rules_dir())), sha256_file(path))
                    for path in files
                ],
            ]
        )


    def run_checkpoint(
        state: dict,
        state_path: Path,
        name: str,
        input_hash: str,
        config_hash: str,
        outputs: list[Path],
        action,
    ) -> None:
        if checkpoint_valid(
            state["stages"].get(name, {}),
            input_hash,
            config_hash,
            outputs,
        ):
            print(f"[SKIP] {name.upper()} checkpoint valid")
            return
        for downstream in STAGES[STAGES.index(name) + 1 :]:
            state["stages"].pop(downstream, None)
        save(state_path, state)
        started = time.monotonic()
        try:
            metadata = action() or {}
            state["stages"][name] = {
                "status": "complete",
                "input_sha256": input_hash,
                "config_sha256": config_hash,
                "outputs": {
                    str(output): sha256_file(output) for output in outputs
                },
                "runtime_seconds": round(time.monotonic() - started, 3),
                "completed_at": utc_now(),
                "meta": metadata,
            }
            save(state_path, state)
            print(f"[OK] {name.upper()}")
        except Exception as exc:
            state["stages"][name] = {
                "status": "failed",
                "input_sha256": input_hash,
                "config_sha256": config_hash,
                "error": str(exc),
                "failed_at": utc_now(),
            }
            save(state_path, state)
            raise


    def acquire_source(directory: Path, source_url: str | None, fixture: bool) -> Path:
        if fixture:
            target = directory / "source" / "fixture.source"
            target.write_text("offline fixture\n", encoding="utf-8")
            return target
        existing = sorted(
            path
            for path in (directory / "source").iterdir()
            if path.suffix.lower() in {".mkv", ".mp4", ".webm", ".mov"}
        )
        if existing:
            return existing[0]
        if not source_url:
            raise RuntimeError(
                "No source media. Pass --source-url once or place a video under source/."
            )
        local = Path(
            source_url[7:] if source_url.startswith("file://") else source_url
        ).expanduser()
        if local.exists():
            target = directory / "source" / local.name
            shutil.copy2(local, target)
            return target
        if not shutil.which("yt-dlp"):
            raise RuntimeError("yt-dlp is required for URL acquisition")
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--merge-output-format",
                "mkv",
                "-o",
                str(directory / "source" / "source.%(ext)s"),
                source_url,
            ],
            check=True,
        )
        candidates = sorted(
            path
            for path in (directory / "source").glob("source.*")
            if path.suffix not in {".part", ".log"}
        )
        if not candidates:
            raise RuntimeError("yt-dlp completed without a source media file")
        (directory / "source" / "source.url").write_text(
            source_url + "\n", encoding="utf-8"
        )
        return candidates[0]


    def run_asr(audio: Path, output: Path, fixture: bool) -> dict:
        if fixture:
            rows = [
                {
                    "segment_index": 0,
                    "start_ms": 1000,
                    "end_ms": 2500,
                    "text": "Merhaba Defne.",
                    "words": [],
                },
                {
                    "segment_index": 1,
                    "start_ms": 2800,
                    "end_ms": 4300,
                    "text": "Nasılsın?",
                    "words": [],
                },
            ]
            write_json(output, rows)
            return {"engine": "fixture", "count": len(rows)}
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper is unavailable; run ./runpod/bootstrap.sh"
            ) from exc
        model_name = os.environ.get("MAS_ASR_MODEL", "large-v3")
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        segments, info = model.transcribe(
            str(audio),
            language="tr",
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        rows = []
        for index, segment in enumerate(segments):
            rows.append(
                {
                    "segment_index": index,
                    "start_ms": round(segment.start * 1000),
                    "end_ms": round(segment.end * 1000),
                    "text": segment.text.strip(),
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "words": [
                        {
                            "word": word.word,
                            "start_ms": round(word.start * 1000),
                            "end_ms": round(word.end * 1000),
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ],
                }
            )
            if len(rows) % 100 == 0:
                write_json(output, rows)
        write_json(output, rows)
        return {
            "engine": "faster-whisper",
            "model": model_name,
            "language": getattr(info, "language", "tr"),
            "count": len(rows),
        }


    def handoff_prompt(episode: int, kind: str, zip_name: str) -> str:
        specifications = "\n\n".join(
            (rules_dir() / filename).read_text(encoding="utf-8")
            for filename in ("TRANSLATION_SPEC.md", "SUBTITLE_SPEC.md")
            if (rules_dir() / filename).exists()
        )
        task = (
            "Correct Turkish ASR text"
            if kind == "TR"
            else "Translate corrected Turkish dialogue into natural Indonesian"
        )
        return f"""# Muhtemel Ask episode {episode} - {kind} handoff

{task}. Work only inside `{zip_name}` and return the exact requested ZIP name.
Do not change timing, record order, record count, block_uid, block_index, audit fields,
schema fields, or hashes. Do not invent audio-dependent text.

{specifications}
"""


    def run(
        episode: int,
        source_url: str | None = None,
        fixture: bool = False,
        stop_after: str | None = None,
    ) -> int:
        directory = ensure_directories(episode)
        state_path = directory / "work" / "state.json"
        state = load(state_path, episode)
        config_hash = config_fingerprint()
        title = episode_name(episode)

        source_marker = directory / "work" / "source.json"
        source_input = sha256_json(
            {
                "source_url": source_url or "",
                "fixture": fixture,
                "existing": [
                    (path.name, sha256_file(path))
                    for path in sorted((directory / "source").glob("*"))
                    if path.is_file()
                ],
            }
        )

        def source_action() -> dict:
            source = acquire_source(directory, source_url, fixture)
            write_json(
                source_marker,
                {
                    "path": str(source),
                    "sha256": sha256_file(source),
                    "source_url": source_url,
                },
            )
            return {"path": str(source)}

        run_checkpoint(
            state,
            state_path,
            "source",
            source_input,
            config_hash,
            [source_marker],
            source_action,
        )
        source = Path(read_json(source_marker)["path"])
        if stop_after == "source":
            return 0

        audio = directory / "work" / "audio.wav"

        def audio_action() -> None:
            if fixture:
                audio.write_bytes(b"RIFFfixture")
                return
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg is unavailable")
            subprocess.run(
                [
                    "ffmpeg",
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
                    str(audio),
                ],
                check=True,
            )

        run_checkpoint(
            state,
            state_path,
            "audio",
            sha256_file(source),
            config_hash,
            [audio],
            audio_action,
        )

        asr_path = directory / "work" / "asr.json"
        run_checkpoint(
            state,
            state_path,
            "asr",
            sha256_file(audio),
            config_hash,
            [asr_path],
            lambda: run_asr(audio, asr_path, fixture),
        )
        if stop_after == "asr":
            raise RuntimeError("Intentional test interruption after ASR")

        raw_records_path = directory / "work" / "records_tr_raw.json"
        review_path = directory / "work" / "review_required.json"

        def reconcile_action() -> dict:
            output = []
            previous_end = -1
            for index, segment in enumerate(read_json(asr_path)):
                start = int(segment["start_ms"])
                end = int(segment["end_ms"])
                text = str(segment.get("text") or "").strip()
                flags = []
                if not text or re.fullmatch(r"[\W_]+", text):
                    flags.append("suspected_asr_hallucination")
                no_speech = segment.get("no_speech_prob")
                if no_speech is not None and no_speech > 0.75:
                    flags.append("suspected_asr_hallucination")
                if start < previous_end:
                    flags.append("timing_overlap")
                identity = f"{start}:{end}:{text}"
                uid = (
                    f"b{index + 1:06d}-"
                    f"{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex[:12]}"
                )
                output.append(
                    {
                        "block_uid": uid,
                        "block_index": index + 1,
                        "start_ms": start,
                        "end_ms": end,
                        "tr_text": text,
                        "id_text": "",
                        "words": segment.get("words", []),
                        "speaker": segment.get("speaker"),
                        "provenance": ["asr"],
                        "asr_audit": {
                            "avg_logprob": segment.get("avg_logprob"),
                            "no_speech_prob": no_speech,
                            "source_segment_index": segment.get(
                                "segment_index", index
                            ),
                        },
                        "review_required": bool(flags),
                        "review_flags": flags,
                        "audio_reviewed": False,
                        "review_disposition": None,
                    }
                )
                previous_end = max(previous_end, end)
            write_json(raw_records_path, output)
            write_json(
                review_path,
                [record for record in output if record["review_required"]],
            )
            return {"count": len(output)}

        run_checkpoint(
            state,
            state_path,
            "reconcile",
            sha256_file(asr_path),
            config_hash,
            [raw_records_path, review_path],
            reconcile_action,
        )
        raw_records = read_json(raw_records_path)

        tr_zip = directory / "translation_input" / f"{title}_TR_CORRECTION_PACK.zip"
        tr_prompt = directory / "translation_input" / "TR_PROMPT.txt"
        tr_manifest_path = directory / "work" / "tr_manifest.json"

        def tr_pack_action() -> dict:
            prompt = handoff_prompt(episode, "TR", tr_zip.name)
            tr_prompt.write_text(prompt, encoding="utf-8")
            manifest = build_pack(tr_zip, episode, "TR", raw_records, prompt)
            write_json(tr_manifest_path, manifest)
            return {"record_count": len(raw_records)}

        run_checkpoint(
            state,
            state_path,
            "tr_pack",
            sha256_file(raw_records_path),
            config_hash,
            [tr_zip, tr_prompt, tr_manifest_path],
            tr_pack_action,
        )

        tr_return = directory / "translation_output" / f"{title}_TR_TEXT_CORRECTED.zip"
        if not tr_return.exists():
            state["waiting_for"] = {
                "stage": "TR_CORRECTION",
                "file": tr_return.name,
                "prompt": str(tr_prompt),
            }
            save(state_path, state)
            print(
                f"[WAIT] TR CORRECTION\n"
                f"GPU WORK COMPLETE\n"
                f"Safe to stop RunPod now.\n\n"
                f"Waiting for:\n{tr_return.name}"
            )
            return 20

        corrected_path = directory / "work" / "records_tr_corrected.json"

        def tr_return_action() -> dict:
            corrected = validate_returned(
                tr_return,
                read_json(tr_manifest_path),
                raw_records,
            )
            write_json(corrected_path, corrected)
            return {"package_sha256": sha256_file(tr_return)}

        run_checkpoint(
            state,
            state_path,
            "tr_return",
            sha256_json([sha256_file(tr_return), sha256_file(tr_manifest_path)]),
            config_hash,
            [corrected_path],
            tr_return_action,
        )
        corrected = read_json(corrected_path)
        state.pop("waiting_for", None)
        save(state_path, state)

        aligned_path = directory / "work" / "records_aligned.json"

        def align_action() -> None:
            aligned = []
            for record in corrected:
                item = dict(record)
                item["provenance"] = list(
                    dict.fromkeys(
                        item.get("provenance", [])
                        + ["corrected_turkish", "asr_word_timing"]
                    )
                )
                aligned.append(item)
            write_json(aligned_path, aligned)

        run_checkpoint(
            state,
            state_path,
            "align",
            sha256_file(corrected_path),
            config_hash,
            [aligned_path],
            align_action,
        )
        aligned = read_json(aligned_path)

        id_zip = directory / "translation_input" / f"{title}_ID_TRANSLATION_PACK.zip"
        id_prompt = directory / "translation_input" / "ID_PROMPT.txt"
        id_manifest_path = directory / "work" / "id_manifest.json"

        def id_pack_action() -> dict:
            prompt = handoff_prompt(episode, "ID", id_zip.name)
            id_prompt.write_text(prompt, encoding="utf-8")
            manifest = build_pack(id_zip, episode, "ID", aligned, prompt)
            write_json(id_manifest_path, manifest)
            return {"record_count": len(aligned)}

        run_checkpoint(
            state,
            state_path,
            "id_pack",
            sha256_file(aligned_path),
            config_hash,
            [id_zip, id_prompt, id_manifest_path],
            id_pack_action,
        )

        id_return = directory / "translation_output" / f"{title}_ID_TRANSLATED.zip"
        if not id_return.exists():
            state["waiting_for"] = {
                "stage": "ID_TRANSLATION",
                "file": id_return.name,
                "prompt": str(id_prompt),
            }
            save(state_path, state)
            print(
                f"[WAIT] ID TRANSLATION\n"
                f"GPU NOT REQUIRED\n\n"
                f"Waiting for:\n{id_return.name}"
            )
            return 21

        final_records_path = directory / "work" / "records_final.json"

        def id_return_action() -> dict:
            translated = validate_returned(
                id_return,
                read_json(id_manifest_path),
                aligned,
            )
            write_json(final_records_path, translated)
            return {"package_sha256": sha256_file(id_return)}

        run_checkpoint(
            state,
            state_path,
            "id_return",
            sha256_json([sha256_file(id_return), sha256_file(id_manifest_path)]),
            config_hash,
            [final_records_path],
            id_return_action,
        )
        translated = read_json(final_records_path)
        state.pop("waiting_for", None)
        save(state_path, state)

        tr_srt = directory / "final" / "subtitles" / f"{title}-tr.srt"
        id_srt = directory / "final" / "subtitles" / f"{title}-id.srt"

        def finalize_action() -> None:
            tr_srt.write_text(render(translated, "tr_text"), encoding="utf-8")
            id_srt.write_text(render(translated, "id_text"), encoding="utf-8")

        run_checkpoint(
            state,
            state_path,
            "finalize",
            sha256_json([sha256_file(final_records_path), sha256_file(source)]),
            config_hash,
            [tr_srt, id_srt],
            finalize_action,
        )

        qc_json = directory / "reports" / "qc.json"
        qc_markdown = directory / "reports" / "qc.md"

        def qc_action() -> dict:
            issues = qc(translated)
            missing = sum(
                1 for record in translated if not str(record.get("id_text") or "").strip()
            )
            mandatory_failure = bool(missing) or any(
                issue["type"]
                in {"invalid_duration", "duplicate_block_uid", "missing_text"}
                for issue in issues
            )
            status_value = (
                "FAIL"
                if mandatory_failure
                else (
                    "PASS_WITH_REVIEW"
                    if issues
                    or any(record.get("review_required") for record in translated)
                    else "PASS"
                )
            )
            report = {
                "status": status_value,
                "episode": episode,
                "source_duration_ms": max(
                    (record["end_ms"] for record in translated), default=0
                ),
                "asr_block_count": len(read_json(asr_path)),
                "final_subtitle_block_count": len(translated),
                "missing_translations": missing,
                "unresolved_review_items": sum(
                    1
                    for record in translated
                    if record.get("review_required")
                    and not record.get("audio_reviewed")
                ),
                "issues": issues,
                "immutable_field_validation": "PASS",
                "pipeline_version": PIPELINE_VERSION,
                "schema_version": SCHEMA_VERSION,
                "rules_version": RULES_VERSION,
                "input_output_hashes": {
                    "source": sha256_file(source),
                    "tr_return": sha256_file(tr_return),
                    "id_return": sha256_file(id_return),
                },
                "runtime_seconds_by_stage": {
                    stage_name: stage.get("runtime_seconds")
                    for stage_name, stage in state["stages"].items()
                },
            }
            write_json(qc_json, report)
            qc_markdown.write_text(
                f"# QC report\n\nStatus: {status_value}\n\n"
                f"Final blocks: {len(translated)}\n"
                f"Issues: {len(issues)}\n",
                encoding="utf-8",
            )
            return {"status": status_value}

        run_checkpoint(
            state,
            state_path,
            "qc",
            sha256_file(final_records_path),
            config_hash,
            [qc_json, qc_markdown],
            qc_action,
        )
        result = read_json(qc_json)
        print(f"[9/9] QC {result['status']}")
        return 2 if result["status"] == "FAIL" else 0


    def status(episode: int) -> int:
        directory = ensure_directories(episode)
        print(
            json.dumps(
                load(directory / "work" / "state.json", episode),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    ''',
)

write(
    "src/mas/cli.py",
    """
    from __future__ import annotations

    import argparse
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import traceback

    from .config import episode_dir, repo_root
    from .pipeline import run, status
    from .version import PIPELINE_VERSION


    def doctor() -> int:
        checks = {
            name: bool(shutil.which(name))
            for name in ("python", "ffmpeg", "ffprobe", "git", "yt-dlp")
        }
        checks["python_version"] = platform.python_version()
        checks["pipeline_version"] = PIPELINE_VERSION
        try:
            import torch

            checks["torch"] = torch.__version__
            checks["cuda_available"] = torch.cuda.is_available()
        except Exception:
            checks["torch"] = "not installed"
            checks["cuda_available"] = False
        for key, value in checks.items():
            print(f"{key}: {value}")
        return 1 if not checks["ffmpeg"] or not checks["ffprobe"] else 0


    def run_tests() -> int:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(repo_root() / "src")
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        )
        return subprocess.call(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=repo_root(),
            env=environment,
        )


    def clean(episode: int, destroy: bool = False) -> int:
        directory = episode_dir(episode)
        for target in (directory / "work" / "audio_review", directory / "work" / "captions"):
            print(("DELETE " if destroy else "DRY-RUN ") + str(target))
            if destroy and target.exists():
                shutil.rmtree(target)
        print("Source media and translation outputs are never deleted by clean.")
        return 0


    def parser() -> argparse.ArgumentParser:
        root = argparse.ArgumentParser(prog="mas")
        root.add_argument("--version", action="version", version=PIPELINE_VERSION)
        commands = root.add_subparsers(dest="command", required=True)
        run_parser = commands.add_parser("run")
        run_parser.add_argument("episode", type=int)
        run_parser.add_argument("--source-url")
        run_parser.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)
        run_parser.add_argument(
            "--stop-after",
            choices=("source", "audio", "asr"),
            help=argparse.SUPPRESS,
        )
        status_parser = commands.add_parser("status")
        status_parser.add_argument("episode", type=int)
        commands.add_parser("doctor")
        commands.add_parser("test")
        clean_parser = commands.add_parser("clean")
        clean_parser.add_argument("episode", type=int)
        clean_parser.add_argument("--dry-run", action="store_true", default=True)
        clean_parser.add_argument("--destroy", action="store_true")
        return root


    def main(argv=None) -> int:
        arguments = parser().parse_args(argv)
        try:
            if arguments.command == "run":
                return run(
                    arguments.episode,
                    arguments.source_url,
                    arguments.fixture,
                    arguments.stop_after,
                )
            if arguments.command == "status":
                return status(arguments.episode)
            if arguments.command == "doctor":
                return doctor()
            if arguments.command == "test":
                return run_tests()
            return clean(arguments.episode, arguments.destroy)
        except Exception as exc:
            episode = getattr(arguments, "episode", None)
            base = episode_dir(episode) if episode else repo_root()
            logs = base / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            traceback_path = logs / "last_failure.traceback.log"
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(
                f"FAILED STAGE: {arguments.command.upper()}\n"
                f"CAUSE: {exc}\n"
                f"CHECKPOINT PRESERVED: yes\n"
                f"TRACEBACK: {traceback_path}",
                file=sys.stderr,
            )
            if episode:
                print(f"SAFE RETRY:\n./mas run {episode}", file=sys.stderr)
            return 1


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
)

write("src/mas/__init__.py", "from .version import PIPELINE_VERSION as __version__\n")
write(
    "mas",
    """
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    PYTHON_BIN="${MAS_PYTHON:-$ROOT/.venv/bin/python}"
    [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python3
    exec "$PYTHON_BIN" -m mas.cli "$@"
    """,
    executable=True,
)

for init_file in (
    "src/mas/packs/__init__.py",
    "src/mas/subtitle/__init__.py",
    "src/mas/stages/__init__.py",
):
    Path(init_file).parent.mkdir(parents=True, exist_ok=True)
    Path(init_file).touch(exist_ok=True)

for module in (
    "src/mas/stages/acquire.py",
    "src/mas/stages/media.py",
    "src/mas/stages/asr.py",
    "src/mas/stages/captions.py",
    "src/mas/stages/vad.py",
    "src/mas/stages/reconcile.py",
    "src/mas/stages/prepare_tr.py",
    "src/mas/stages/audio_review.py",
    "src/mas/stages/align.py",
    "src/mas/stages/prepare_id.py",
    "src/mas/stages/finalize.py",
    "src/mas/stages/qc.py",
    "src/mas/stages/archive.py",
    "src/mas/subtitle/timing.py",
    "src/mas/subtitle/speakers.py",
    "src/mas/subtitle/segmentation.py",
):
    if not Path(module).exists():
        write(
            module,
            '"""Compatibility module. Production orchestration lives in mas.pipeline."""\n',
        )

if not Path("rules/TRANSLATION_SPEC.md").exists():
    write(
        "rules/TRANSLATION_SPEC.md",
        """
        # Translation specification

        Preserve record count, order, block_uid, block_index, timing, numbers, dates,
        quantities, currencies, proper names, interruptions, hesitation, insults, jokes,
        and emotional intent. Indonesian dialogue must sound natural. Use aku, kamu,
        nggak, udah, and aja only where informal; use saya, Anda, Pak, and Bu in formal
        scenes. Do not sanitize, explain jokes, add translator notes or speaker labels,
        move dialogue between blocks, or invent audio-dependent text.
        """,
    )

if not Path("rules/SUBTITLE_SPEC.md").exists():
    write(
        "rules/SUBTITLE_SPEC.md",
        """
        # Subtitle specification

        Timestamps increase, end is after start, independent speakers remain separate,
        overlap requires evidence, timing follows speech evidence, and final records contain
        no blank dialogue, speaker labels, or metadata. Existing V2 thresholds take
        precedence; fallback limits are 2 lines, 42 characters per line, 22 characters per
        second, and 500-8000 ms duration.
        """,
    )

for mapping in ("rules/GLOSSARY.json", "rules/NAMES.json"):
    if not Path(mapping).exists():
        write(
            mapping,
            json.dumps(
                {"version": "v2-beta-2026-09-04", "entries": {}},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

write(
    "runpod/bootstrap.sh",
    """
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"
    python3.11 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.lock
    .venv/bin/python - <<'PY'
    from pathlib import Path
    for path in (
        'rules/SUBTITLE_SPEC.md',
        'rules/TRANSLATION_SPEC.md',
        'rules/GLOSSARY.json',
        'rules/NAMES.json',
    ):
        assert Path(path).exists(), path
    print('rules: OK')
    PY
    MAS_PYTHON="$ROOT/.venv/bin/python" ./mas doctor
    """,
    executable=True,
)

write(
    "runpod/preflight.sh",
    """
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$ROOT"
    PYTHON_BIN="$ROOT/.venv/bin/python"
    [[ -x "$PYTHON_BIN" ]] || { echo 'Run ./runpod/bootstrap.sh first'; exit 1; }
    MAS_PYTHON="$PYTHON_BIN" ./mas doctor
    MAS_PYTHON="$PYTHON_BIN" ./mas --help >/dev/null
    """,
    executable=True,
)

write(
    "tests/release/test_release_runtime.py",
    """
    import json
    import zipfile

    import pytest

    from mas import pipeline
    from mas.errors import PackageValidationError
    from mas.packs.build import build_pack
    from mas.packs.validate import validate_returned
    from mas.subtitle.srt import render
    from mas.subtitle.validation import qc


    def records():
        return [
            {
                "block_uid": "u1",
                "block_index": 1,
                "start_ms": 1000,
                "end_ms": 2200,
                "tr_text": "Merhaba Defne.",
                "id_text": "Halo Defne.",
                "asr_audit": {},
                "audio_reviewed": False,
                "review_disposition": None,
            },
            {
                "block_uid": "u2",
                "block_index": 2,
                "start_ms": 2400,
                "end_ms": 4000,
                "tr_text": "Nasılsın?",
                "id_text": "Apa kabar?",
                "asr_audit": {},
                "audio_reviewed": False,
                "review_disposition": None,
            },
        ]


    def rewrite(path, mutation):
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            rows = json.loads(archive.read("records.json"))
        mutation(manifest, rows)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("records.json", json.dumps(rows))


    def return_package(source, target, field, prefix):
        with zipfile.ZipFile(source) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            rows = json.loads(archive.read("records.json"))
        for index, row in enumerate(rows):
            row[field] = f"{prefix} {index + 1}"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("records.json", json.dumps(rows))


    def test_pack_roundtrip_and_immutable_timing(tmp_path):
        package = tmp_path / "package.zip"
        manifest = build_pack(package, 13, "TR", records())
        assert len(validate_returned(package, manifest, records())) == 2
        rewrite(package, lambda _, rows: rows[0].update(start_ms=999))
        with pytest.raises(PackageValidationError, match="immutable"):
            validate_returned(package, manifest, records())


    @pytest.mark.parametrize(
        "mutation, message",
        [
            (lambda manifest, _: manifest.update(episode=14), "episode mismatch"),
            (lambda _, rows: rows.pop(), "record count"),
            (lambda _, rows: rows.reverse(), "block_uid/order"),
            (lambda _, rows: rows[1].update(block_uid="u1"), "block_uid/order"),
        ],
    )
    def test_rejects_corrupted_returns(tmp_path, mutation, message):
        package = tmp_path / "package.zip"
        manifest = build_pack(package, 13, "ID", records())
        rewrite(package, mutation)
        with pytest.raises(PackageValidationError, match=message):
            validate_returned(package, manifest, records())


    def test_srt_and_qc():
        assert "00:00:01,000 --> 00:00:02,200" in render(records(), "id_text")
        assert qc(records()) == []


    def test_interrupt_resume_and_selective_invalidation(tmp_path, monkeypatch):
        monkeypatch.setenv("MAS_WORKSPACE", str(tmp_path))
        episode = 13
        with pytest.raises(RuntimeError, match="Intentional"):
            pipeline.run(episode, fixture=True, stop_after="asr")
        directory = tmp_path / "EPISODES" / "Muhtemel Ask 13.Bolum"
        first_state = json.loads((directory / "work" / "state.json").read_text())
        asr_completed = first_state["stages"]["asr"]["completed_at"]
        assert pipeline.run(episode, fixture=True) == 20
        return_package(
            directory
            / "translation_input"
            / "Muhtemel Ask 13.Bolum_TR_CORRECTION_PACK.zip",
            directory
            / "translation_output"
            / "Muhtemel Ask 13.Bolum_TR_TEXT_CORRECTED.zip",
            "tr_text",
            "Düzeltilmiş",
        )
        assert pipeline.run(episode, fixture=True) == 21
        id_input = (
            directory
            / "translation_input"
            / "Muhtemel Ask 13.Bolum_ID_TRANSLATION_PACK.zip"
        )
        id_output = (
            directory
            / "translation_output"
            / "Muhtemel Ask 13.Bolum_ID_TRANSLATED.zip"
        )
        return_package(id_input, id_output, "id_text", "Terjemahan")
        assert pipeline.run(episode, fixture=True) == 0
        completed = json.loads((directory / "work" / "state.json").read_text())
        assert completed["stages"]["asr"]["completed_at"] == asr_completed
        align_completed = completed["stages"]["align"]["completed_at"]
        return_package(id_input, id_output, "id_text", "Baru")
        assert pipeline.run(episode, fixture=True) == 0
        rerun = json.loads((directory / "work" / "state.json").read_text())
        assert rerun["stages"]["align"]["completed_at"] == align_completed
        assert (
            directory
            / "final"
            / "subtitles"
            / "Muhtemel Ask 13.Bolum-id.srt"
        ).exists()
        assert (directory / "reports" / "qc.json").exists()
    """,
)

print("Production runtime repair written.")
