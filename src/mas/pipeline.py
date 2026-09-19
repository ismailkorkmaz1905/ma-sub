import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EPISODES_ROOT = Path(os.getenv("MAS_EPISODES_ROOT", ROOT / "EPISODES"))
CONFIG_PATH = ROOT / "config" / "config.yaml"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov"}
WAIT_TRANSLATION = 20


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_bytes(path, data.encode("utf-8"))


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return value


def _jsonl_bytes(records):
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def _write_jsonl(path, records):
    _atomic_bytes(path, _jsonl_bytes(records))


def _read_jsonl(path):
    records = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at line {line_number}: {path}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"JSON object expected at line {line_number}: {path}")
        records.append(value)
    return records


def _load_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError("config/config.yaml is invalid")
    return config


def _paths(episode, create=False):
    root = EPISODES_ROOT / f"Muhtemel Ask {episode}.Bolum"
    paths = {
        "root": root,
        "source": root / "source",
        "work": root / "work",
        "handoff": root / "handoff",
        "output": root / "output",
        "state": root / "work" / "state.json",
    }
    if create:
        for key in ("source", "work", "handoff", "output"):
            paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _load_state(path, episode):
    if not Path(path).is_file():
        return {"episode": episode, "stage": "NEW"}
    state = _read_json(path)
    if state.get("episode") != episode:
        raise RuntimeError("state episode does not match the requested episode")
    return state


def _save_state(path, state, stage, **values):
    state.update(values)
    state["stage"] = stage
    _write_json(path, state)


def _run(command, *, cwd=None, timeout=None, capture=False):
    completed = subprocess.run(
        command,
        cwd=cwd,
        timeout=timeout,
        check=False,
        text=True,
        capture_output=capture,
    )
    if completed.returncode:
        detail = ""
        if capture:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command[0]}"
            + (f"\n{detail}" if detail else "")
        )
    return completed


def _probe_video(path):
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
        capture=True,
    )
    payload = json.loads(completed.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"source has no video stream: {path}")


def _source_files(source_dir):
    return sorted(
        path
        for path in Path(source_dir).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _copy_source(source_file, source_dir):
    source = Path(source_file).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"source file is missing or unsafe: {source}")
    destination = Path(source_dir) / ("video" + source.suffix.lower())
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise RuntimeError("source directory already contains a different video")
        return destination
    shutil.copy2(source, destination)
    return destination


def _download_source(url, source_dir):
    output = str(Path(source_dir) / "video.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        output,
    ]
    cookies = os.getenv("MAS_YTDLP_COOKIES")
    if cookies:
        cookie_path = Path(cookies).expanduser().resolve()
        if not cookie_path.is_file():
            raise RuntimeError("MAS_YTDLP_COOKIES does not point to a file")
        command.extend(["--cookies", str(cookie_path)])
    command.append(url)
    _run(command, timeout=int(os.getenv("MAS_DOWNLOAD_TIMEOUT", "7200")))
    candidates = _source_files(source_dir)
    if len(candidates) != 1:
        raise RuntimeError("download must produce exactly one source video")
    return candidates[0]


def _ensure_source(paths, state, source_url=None, source_file=None):
    saved = state.get("source")
    if saved:
        source = paths["root"] / saved.get("path", "")
        try:
            source.resolve().relative_to(paths["source"].resolve())
        except (ValueError, OSError) as exc:
            raise RuntimeError("recorded source path escapes source/") from exc
        if not source.is_file() or source.is_symlink():
            raise RuntimeError("recorded source video is missing or unsafe")
        observed = file_sha256(source)
        if observed != saved.get("sha256"):
            raise RuntimeError("immutable source SHA-256 changed")
        if source_url and saved.get("url") and source_url != saved["url"]:
            raise RuntimeError("episode already has a different source URL")
        return source

    existing = _source_files(paths["source"])
    if len(existing) > 1:
        raise RuntimeError("source/ contains more than one video")
    if existing:
        source = existing[0]
    elif source_file:
        source = _copy_source(source_file, paths["source"])
    elif source_url:
        source = _download_source(source_url, paths["source"])
    else:
        raise RuntimeError("first run needs --source-url URL or --source FILE")

    _probe_video(source)
    source_record = {
        "path": source.relative_to(paths["root"]).as_posix(),
        "sha256": file_sha256(source),
        "size_bytes": source.stat().st_size,
    }
    if source_url:
        source_record["url"] = source_url
    _save_state(paths["state"], state, "SOURCE_READY", source=source_record)
    return source


def _clean_text(words):
    text = " ".join(word["text"].strip() for word in words if word["text"].strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_credit(text):
    normalized = re.sub(r"[^a-zçğıöşü0-9]+", " ", text.casefold()).strip()
    return normalized in {"altyazı m k", "takarir m k"}


def build_blocks(episode, source_sha256, words, subtitle_config):
    max_chars = int(subtitle_config["maximum_block_chars"])
    max_duration = int(subtitle_config["maximum_duration_ms"])
    min_duration = int(subtitle_config["minimum_duration_ms"])
    max_gap = int(subtitle_config["split_gap_ms"])
    end_padding = int(subtitle_config["end_padding_ms"])
    next_guard = int(subtitle_config["next_speech_guard_ms"])

    cleaned = []
    for word in words:
        text = str(word.get("text", "")).strip()
        start = word.get("start_ms")
        end = word.get("end_ms")
        if not text or type(start) is not int or type(end) is not int or start < 0 or end <= start:
            continue
        cleaned.append(
            {
                "text": text,
                "start_ms": start,
                "end_ms": end,
                "break_after": bool(word.get("break_after")),
            }
        )
    if not cleaned:
        raise RuntimeError("ASR returned no timed words")
    if any(left["start_ms"] > right["start_ms"] for left, right in zip(cleaned, cleaned[1:])):
        raise RuntimeError("ASR words are not time ordered")

    groups = []
    current = []

    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in cleaned:
        if current:
            gap = word["start_ms"] - current[-1]["end_ms"]
            candidate = _clean_text([*current, word])
            duration = word["end_ms"] - current[0]["start_ms"]
            if gap > max_gap or duration > max_duration or len(candidate) > max_chars:
                flush()
        current.append(word)
        duration = current[-1]["end_ms"] - current[0]["start_ms"]
        text = _clean_text(current)
        sentence_end = bool(re.search(r"[.!?…][\"')\]]?$", text))
        if sentence_end and duration >= min_duration:
            flush()
        elif word["break_after"] and (
            _is_credit(text) or (duration >= min_duration and len(text) >= 24)
        ):
            flush()
    flush()

    filtered = [group for group in groups if not _is_credit(_clean_text(group))]
    if not filtered:
        raise RuntimeError("all ASR blocks were filtered as subtitle metadata")

    blocks = []
    for index, group in enumerate(filtered, 1):
        start = group[0]["start_ms"]
        raw_end = group[-1]["end_ms"]
        next_start = filtered[index][0]["start_ms"] if index < len(filtered) else None
        end = raw_end + end_padding
        if next_start is not None:
            guarded = next_start - next_guard
            end = min(end, guarded if guarded > raw_end else next_start)
        end = max(start + 1, end)
        identity = f"{episode}|{source_sha256}|{index}|{start}|{end}"
        uid = f"MA{episode:02d}-{index:05d}-{hashlib.sha256(identity.encode()).hexdigest()[:8]}"
        blocks.append(
            {
                "block_uid": uid,
                "start_ms": start,
                "end_ms": end,
                "source_text": _clean_text(group),
            }
        )

    for previous, current_block in zip(blocks, blocks[1:]):
        if previous["end_ms"] > current_block["start_ms"]:
            raise RuntimeError("generated subtitle blocks overlap")
    return blocks


def _transcribe_source(source, config):
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper runtime is not installed") from exc
    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError("CUDA GPU is required for ASR")

    asr = config["asr"]
    model = WhisperModel(
        os.getenv("MAS_WHISPER_MODEL", asr["model"]),
        device="cuda",
        compute_type=asr["compute_type"],
        download_root=os.getenv("MAS_MODEL_DIR") or None,
    )
    segments, _ = model.transcribe(
        str(source),
        language="tr",
        beam_size=int(asr["beam_size"]),
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    words = []
    for segment in segments:
        segment_words = []
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            segment_words.append(
                {
                    "text": word.word,
                    "start_ms": round(word.start * 1000),
                    "end_ms": round(word.end * 1000),
                    "break_after": False,
                }
            )
        if not segment_words and segment.text.strip():
            segment_words.append(
                {
                    "text": segment.text.strip(),
                    "start_ms": round(segment.start * 1000),
                    "end_ms": round(segment.end * 1000),
                    "break_after": False,
                }
            )
        if segment_words:
            segment_words[-1]["break_after"] = True
            words.extend(segment_words)
    return words


def _ensure_blocks(paths, state, source, config):
    blocks_path = paths["work"] / "blocks.jsonl"
    saved_sha = state.get("blocks_sha256")
    if blocks_path.is_file() and saved_sha == file_sha256(blocks_path):
        blocks = _read_jsonl(blocks_path)
        if blocks:
            return blocks
    words = _transcribe_source(source, config)
    blocks = build_blocks(
        state["episode"], state["source"]["sha256"], words, config["subtitle"]
    )
    _write_jsonl(blocks_path, blocks)
    _save_state(
        paths["state"],
        state,
        "ASR_READY",
        blocks_sha256=file_sha256(blocks_path),
        block_count=len(blocks),
    )
    return blocks


def _instructions(config):
    names = ", ".join(config["names"]["canonical"])
    religious = "\n".join(
        f"- {item['source']} -> {' / '.join(item['indonesian'])}"
        for item in config["religious_terms"]
    )
    return f"""# Translation return

Correct the Turkish ASR text and translate it into natural conversational Indonesian.

Return one ZIP containing only `manifest.json` and `translated.jsonl`.

Copy `pack_id` and `block_count` from the input manifest. Each output line must contain
exactly `block_uid`, `tr`, and `id`. Keep every block UID once and in the input order.
Do not add timestamps, split, merge, delete, create, or reorder blocks. Do not use
Markdown fences in JSONL. Both text fields must be non-empty single-line strings.

Canonical names: {names}

Religious wording:
{religious}

Preserve meaning, names, numbers, tone, unfinished speech, and religious expressions.
Do not invent dialogue or subtitle credits. The program rejects altered identity,
ordering, missing blocks, extra fields, forbidden name spellings, and stale returns.
"""


def _zip_bytes(files):
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, files[name])
        handle.seek(0)
        return handle.read()


def _create_pack(path, episode, source_sha256, blocks, config):
    blocks_data = _jsonl_bytes(blocks)
    instructions = _instructions(config).encode("utf-8")
    glossary = {
        "canonical_names": config["names"]["canonical"],
        "forbidden_name_variants": config["names"]["forbidden"],
        "religious_terms": config["religious_terms"],
    }
    glossary_data = json.dumps(
        glossary, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    manifest = {
        "format": "mas-translation-pack-1",
        "episode": episode,
        "source_sha256": source_sha256,
        "blocks_sha256": _sha256_bytes(blocks_data),
        "policy_sha256": _sha256_bytes(instructions + glossary_data),
        "block_count": len(blocks),
    }
    manifest["pack_id"] = _sha256_bytes(_canonical_json(manifest))
    data = _zip_bytes(
        {
            "INSTRUCTIONS.md": instructions,
            "blocks.jsonl": blocks_data,
            "glossary.json": glossary_data,
            "manifest.json": json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True
            ).encode("utf-8")
            + b"\n",
        }
    )
    _atomic_bytes(path, data)
    return manifest


def _read_return_zip(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("translated ZIP contains duplicate file names")
        if set(names) != {"manifest.json", "translated.jsonl"}:
            raise RuntimeError("translated ZIP must contain only manifest.json and translated.jsonl")
        if any(info.is_dir() or info.file_size > 50 * 1024 * 1024 for info in infos):
            raise RuntimeError("translated ZIP contains an invalid member")
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            lines = archive.read("translated.jsonl").decode("utf-8").splitlines()
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("translated ZIP contains invalid UTF-8 or JSON") from exc
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid translated JSONL at line {number}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"translated line {number} is not an object")
        records.append(record)
    return manifest, records


def _validate_translation(return_zip, pack_manifest, blocks, config):
    manifest, translated = _read_return_zip(return_zip)
    expected_manifest = {
        "format": "mas-translated-1",
        "pack_id": pack_manifest["pack_id"],
        "block_count": len(blocks),
    }
    if manifest != expected_manifest:
        raise RuntimeError("translated manifest is stale or invalid")
    if len(translated) != len(blocks):
        raise RuntimeError("translated block count does not match the input")

    forbidden = [value.casefold() for value in config["names"]["forbidden"]]
    checked = []
    for index, (source, record) in enumerate(zip(blocks, translated), 1):
        if set(record) != {"block_uid", "tr", "id"}:
            raise RuntimeError(f"translated block {index} has unexpected fields")
        if record["block_uid"] != source["block_uid"]:
            raise RuntimeError(f"translated block order or identity changed at {index}")
        tr = record["tr"]
        indonesian = record["id"]
        if not isinstance(tr, str) or not isinstance(indonesian, str):
            raise RuntimeError(f"translated block {index} text must be a string")
        tr = re.sub(r"\s+", " ", tr).strip()
        indonesian = re.sub(r"\s+", " ", indonesian).strip()
        if not tr or not indonesian:
            raise RuntimeError(f"translated block {index} has empty text")
        combined = f"{tr}\n{indonesian}".casefold()
        invalid_name = next((value for value in forbidden if value in combined), None)
        if invalid_name:
            raise RuntimeError(f"translated block {index} contains forbidden name: {invalid_name}")
        for item in config["religious_terms"]:
            if item["source"].casefold() in tr.casefold() and not any(
                value.casefold() in indonesian.casefold() for value in item["indonesian"]
            ):
                raise RuntimeError(
                    f"translated block {index} does not preserve {item['source']}"
                )
        checked.append({"block_uid": source["block_uid"], "tr": tr, "id": indonesian})
    return checked


def _format_timestamp(milliseconds):
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _wrap_subtitle(text, width):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return [text]
    words = text.split()
    choices = []
    for split in range(1, len(words)):
        left = " ".join(words[:split])
        right = " ".join(words[split:])
        if len(left) <= width and len(right) <= width:
            choices.append((abs(len(left) - len(right)), left, right))
    if not choices:
        raise RuntimeError(f"subtitle cannot fit into two {width}-character lines: {text}")
    _, left, right = min(choices)
    return [left, right]


def _write_srt(path, blocks, translated, language, width):
    lines = []
    for index, (block, record) in enumerate(zip(blocks, translated), 1):
        text = record[language]
        wrapped = _wrap_subtitle(text, width)
        lines.extend(
            [
                str(index),
                f"{_format_timestamp(block['start_ms'])} --> {_format_timestamp(block['end_ms'])}",
                *wrapped,
                "",
            ]
        )
    _atomic_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _burn_video(source, subtitle, output, episode_root):
    relative_subtitle = subtitle.relative_to(episode_root).as_posix().replace("'", r"\'")
    style = "FontName=Arial,FontSize=20,Outline=2,Shadow=0,Alignment=2,MarginV=32"
    subtitle_filter = f"subtitles=filename='{relative_subtitle}':force_style='{style}'"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-vf",
        subtitle_filter,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        os.getenv("MAS_VIDEO_ENCODER", "libx264"),
        "-preset",
        os.getenv("MAS_VIDEO_PRESET", "medium"),
        "-crf",
        os.getenv("MAS_VIDEO_CRF", "20"),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run(
        command,
        cwd=episode_root,
        timeout=int(os.getenv("MAS_ENCODE_TIMEOUT", "21600")),
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce the burned MP4")
    _probe_video(output)


def _remote_file(remote_root, episode_name, filename):
    separator = "" if remote_root.endswith(("/", ":")) else "/"
    return f"{remote_root}{separator}{episode_name}/{filename}"


def _upload_verified(local_path, episode_root, remote_root):
    remote_path = _remote_file(remote_root, episode_root.name, local_path.name)
    timeout = int(os.getenv("MAS_DRIVE_TIMEOUT", "21600"))
    _run(["rclone", "copyto", str(local_path), remote_path], timeout=timeout)
    with tempfile.TemporaryDirectory(prefix="mas-drive-readback-") as directory:
        readback = Path(directory) / local_path.name
        _run(["rclone", "copyto", remote_path, str(readback)], timeout=timeout)
        if readback.stat().st_size != local_path.stat().st_size:
            raise RuntimeError("Drive byte-count readback mismatch")
        if file_sha256(readback) != file_sha256(local_path):
            raise RuntimeError("Drive SHA-256 readback mismatch")
    return {
        "remote": remote_path,
        "size_bytes": local_path.stat().st_size,
        "sha256": file_sha256(local_path),
        "verified": True,
    }


def run(episode, source_url=None, source_file=None, subtitles_only=False):
    if type(episode) is not int or episode < 1:
        raise RuntimeError("episode must be a positive integer")
    if source_url and source_file:
        raise RuntimeError("use only one of --source-url and --source")

    config = _load_config()
    paths = _paths(episode, create=True)
    state = _load_state(paths["state"], episode)
    source = _ensure_source(paths, state, source_url, source_file)
    blocks = _ensure_blocks(paths, state, source, config)

    name = paths["root"].name
    pack_path = paths["handoff"] / f"{name}_TRANSLATION_PACK.zip"
    return_path = paths["handoff"] / f"{name}_TRANSLATED.zip"
    pack_manifest = _create_pack(
        pack_path, episode, state["source"]["sha256"], blocks, config
    )
    _save_state(
        paths["state"],
        state,
        "WAIT_TRANSLATION",
        pack={
            "path": pack_path.relative_to(paths["root"]).as_posix(),
            "sha256": file_sha256(pack_path),
            "pack_id": pack_manifest["pack_id"],
        },
    )
    if not return_path.is_file():
        print("TRANSLATION_HANDOFF_REQUIRED")
        print(f"Input: {pack_path}")
        print(f"Expected: {return_path}")
        print("The GPU is no longer needed while this ZIP is prepared.")
        return WAIT_TRANSLATION

    translated = _validate_translation(return_path, pack_manifest, blocks, config)
    validated_path = paths["work"] / "translated.jsonl"
    _write_jsonl(validated_path, translated)
    _save_state(
        paths["state"],
        state,
        "TRANSLATION_READY",
        return_sha256=file_sha256(return_path),
        translated_sha256=file_sha256(validated_path),
    )

    previous_video = state.get("outputs", {}).get("mp4", {})
    previous_drive = state.get("drive", {})
    tr_srt = paths["output"] / f"{name}.tr.srt"
    id_srt = paths["output"] / f"{name}.id.srt"
    width = int(config["subtitle"]["characters_per_line"])
    _write_srt(tr_srt, blocks, translated, "tr", width)
    _write_srt(id_srt, blocks, translated, "id", width)
    outputs = {
        "tr_srt": {"path": tr_srt.name, "sha256": file_sha256(tr_srt)},
        "id_srt": {"path": id_srt.name, "sha256": file_sha256(id_srt)},
    }
    _save_state(paths["state"], state, "SUBTITLES_READY", outputs=outputs)

    if subtitles_only:
        _save_state(paths["state"], state, "DONE_SUBTITLES", outputs=outputs)
        print(f"DONE_SUBTITLES\nOutput: {paths['output']}")
        return 0

    mp4 = paths["output"] / f"{name}.id.mp4"
    if not (
        mp4.is_file()
        and previous_video.get("sha256") == file_sha256(mp4)
        and previous_video.get("subtitle_sha256") == file_sha256(id_srt)
        and previous_video.get("source_sha256") == state["source"]["sha256"]
    ):
        _burn_video(source, id_srt, mp4, paths["root"])
    outputs["mp4"] = {
        "path": mp4.name,
        "sha256": file_sha256(mp4),
        "size_bytes": mp4.stat().st_size,
        "subtitle_sha256": file_sha256(id_srt),
        "source_sha256": state["source"]["sha256"],
    }

    remote = os.getenv("MAS_DRIVE_REMOTE", "").strip()
    if remote:
        expected_remote = _remote_file(remote, paths["root"].name, mp4.name)
        if (
            previous_drive.get("verified")
            and previous_drive.get("sha256") == file_sha256(mp4)
            and previous_drive.get("remote") == expected_remote
        ):
            receipt = previous_drive
        else:
            receipt = _upload_verified(mp4, paths["root"], remote)
        receipt_path = paths["output"] / "drive-receipt.json"
        _write_json(receipt_path, receipt)
        _save_state(
            paths["state"], state, "DONE_DRIVE", outputs=outputs, drive=receipt
        )
        print(f"DONE_DRIVE\nOutput: {mp4}\nRemote: {receipt['remote']}")
    else:
        _save_state(paths["state"], state, "DONE_LOCAL", outputs=outputs)
        print(f"DONE_LOCAL\nOutput: {mp4}")
    return 0


def status(episode):
    paths = _paths(episode)
    if not paths["state"].is_file():
        return {"episode": episode, "stage": "NOT_STARTED"}
    return _load_state(paths["state"], episode)


def status_summary(episode):
    current = status(episode)
    print(f"Episode {episode}: {current['stage']}")
    if current["stage"] == "WAIT_TRANSLATION":
        root = _paths(episode)["root"]
        name = root.name
        print(f"Input: {root / 'handoff' / (name + '_TRANSLATION_PACK.zip')}")
        print(f"Expected: {root / 'handoff' / (name + '_TRANSLATED.zip')}")
    elif current["stage"].startswith("DONE"):
        print(f"Output: {_paths(episode)['output']}")
    return 0
