from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import html
import importlib.metadata
import math
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from ..reliability import atomic_json, digest, file_digest, read_json, run_command
from .part_scope import (
    PART_AUDIO_FORMAT, SAMPLE_RATE, PartScopeError, build_part_plan, get_part,
    project_part_vad, validate_file_record, validate_part_lineage, validate_part_plan,
)


_VERIFIED_AUDIO = OrderedDict()


def _verification_key(plan, lineage):
    return digest({
        "parent_audio_sha256": plan["audio"]["sha256"],
        "child_audio_sha256": lineage["audio"]["sha256"],
        "start_sample": lineage["start_sample"], "end_sample": lineage["end_sample"],
        "sample_count": lineage["sample_count"], "sample_rate_hz": lineage["sample_rate_hz"],
        "producer_sha256": lineage["producer_sha256"],
        "ffmpeg_identity": lineage["ffmpeg_identity"],
        "validator_sha256": _producer_sha(), "pcm_sha256": lineage["pcm_sha256"],
    })


def _remember_verified(key):
    _VERIFIED_AUDIO[key] = True
    _VERIFIED_AUDIO.move_to_end(key)
    while len(_VERIFIED_AUDIO) > 64:
        _VERIFIED_AUDIO.popitem(last=False)


def _deadline(total_timeout):
    if (isinstance(total_timeout, bool) or not isinstance(total_timeout, (int, float))
            or not math.isfinite(total_timeout) or total_timeout <= 0):
        raise PartScopeError("total_timeout must be finite and positive")
    return time.monotonic() + total_timeout


def _remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("part producer total deadline exhausted; evidence retained")
    return left


def _sha(path, deadline=None):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if deadline is not None:
                _remaining(deadline)
            result.update(chunk)
    return result.hexdigest()


def _path(root, relative):
    root = Path(root).resolve()
    candidate = root / relative
    if candidate.is_symlink():
        raise PartScopeError("part artifacts cannot be symlinks")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise PartScopeError("part artifact escapes episode root")
    return resolved


def _record(root, path, deadline=None):
    root, path = Path(root).resolve(), Path(path)
    if path.is_symlink() or not path.is_file():
        raise PartScopeError(f"part artifact is absent or symlink: {path.name}")
    path = path.resolve()
    if not path.is_relative_to(root):
        raise PartScopeError("part parent/output must remain inside episode root")
    record = {"relative_path": path.relative_to(root).as_posix(),
              "sha256": _sha(path, deadline), "size_bytes": path.stat().st_size}
    return validate_file_record(record)


def _verify_file(root, record, deadline=None, *, verified=None):
    # Python 3.11 Windows ctime is creation time, not a change counter.
    # Retain full byte reads there rather than trusting restored mtime/size.
    if sys.platform == "win32":
        verified = None
    validate_file_record(record)
    path = _path(root, record["relative_path"])
    expected = {key: record[key] for key in ("relative_path", "sha256", "size_bytes")}
    def stamp():
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    try:
        before = stamp()
    except OSError as exc:
        raise PartScopeError(f"part artifact is absent or inaccessible: {record['relative_path']}") from exc
    key = (str(path), record["sha256"], record["size_bytes"])
    if deadline is not None:
        _remaining(deadline)
    if verified is not None and verified.get(key) == before:
        return path
    if _record(root, path, deadline) != expected or stamp() != before:
        raise PartScopeError(f"part artifact byte identity mismatch: {record['relative_path']}")
    if verified is not None:
        verified[key] = before
    return path


def _read_envelope(path):
    try:
        envelope = read_json(path)
    except (OSError, ValueError) as exc:
        raise PartScopeError(f"invalid part receipt: {Path(path).name}") from exc
    if (not isinstance(envelope, dict) or set(envelope) != {"data", "sha256"}
            or not isinstance(envelope["data"], dict)
            or envelope["sha256"] != digest(envelope["data"])):
        raise PartScopeError(f"part receipt checksum mismatch: {Path(path).name}")
    return envelope["data"]


def _write_envelope(path, data):
    atomic_json(path, {"data": data, "sha256": digest(data)})


def load_part_plan(root, episode, *, verify_files=True):
    root = Path(root).resolve()
    plan = validate_part_plan(_read_envelope(_path(root, "work/part-plan.json")))
    if plan["episode"] != episode:
        raise PartScopeError("part plan belongs to another episode")
    if verify_files:
        for record in (plan["source"], plan["audio"], plan["captions"]):
            if record is not None:
                _verify_file(root, record)
        if _read_envelope(_path(root, "work/part-vad.json")) != plan["vad"]:
            raise PartScopeError("part plan changed the retained parent VAD inventory")
    return plan


def _producer_sha():
    from . import raw_asr, transcribe
    return digest({"part_audio": file_digest(Path(__file__)),
                   "raw_asr": file_digest(Path(raw_asr.__file__)),
                   "transcribe": file_digest(Path(transcribe.__file__))})


def _audit_config():
    from .raw_asr import RawASRV2Config
    settings = RawASRV2Config(allow_cpu_fallback=False)
    config = replace(settings.transcription_config(),
                     vad_speech_pad_ms=settings.audit_vad_speech_pad_ms)
    evidence = {"sample_rate_hz": SAMPLE_RATE,
                "vad_parameters": config.vad_parameters,
                "coverage_config": asdict(settings.speech_coverage_config())}
    return config, evidence


def _vad_model_identity():
    distribution = importlib.metadata.distribution("faster-whisper")
    files = {}
    for entry in distribution.files or ():
        name = str(entry).replace("\\", "/")
        if name.endswith(".onnx") or name in ("faster_whisper/vad.py", "faster_whisper/audio.py"):
            path = Path(distribution.locate_file(entry))
            if not path.is_file():
                raise PartScopeError("independent VAD model file is missing")
            files[name] = file_digest(path)
    if not any(name.endswith(".onnx") for name in files):
        raise PartScopeError("cannot identify installed independent VAD model bytes")
    return {"package": "faster-whisper", "version": distribution.version, "files": files}


def _run(command, root, name, deadline, *, stdout_path=None):
    _remaining(deadline)
    ceiling = 60 if name.endswith("-probe") else 600
    run_command(command, timeout_seconds=min(ceiling, _remaining(deadline)),
                log_path=_path(root, f"work/{name}.log"), stage=name,
                stdout_path=stdout_path)
    _remaining(deadline)


def _probe_audio(path, root, name, deadline):
    executable = shutil.which("ffprobe")
    if not executable:
        raise PartScopeError("ffprobe is required for exact part sample provenance")
    output = _path(root, f"work/{name}.probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    _run([executable, "-v", "error", "-select_streams", "a", "-show_entries",
          "stream=sample_rate,channels,duration_ts,time_base,codec_name", "-of", "json", str(path)],
         root, name + "-probe", deadline, stdout_path=output)
    try:
        streams = read_json(output)["streams"]
        if len(streams) != 1:
            raise ValueError("expected one audio stream")
        stream = streams[0]
        rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        samples = Fraction(stream["duration_ts"]) * Fraction(stream["time_base"]) * rate
        if rate != SAMPLE_RATE or channels != 1 or samples.denominator != 1 or samples <= 0:
            raise ValueError("expected positive integer samples of mono16k audio")
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise PartScopeError("audio probe cannot prove exact mono16k sample count") from exc
    return {"sample_rate_hz": rate, "channels": channels, "sample_count": int(samples)}


def _scan_vad(audio_path, output_path, sample_count):
    from .transcribe import _extract_vad_regions
    audio_sha = file_digest(Path(audio_path))
    model = _vad_model_identity()
    config, config_evidence = _audit_config()
    regions, fallback = _extract_vad_regions(Path(audio_path), config, ())
    if fallback not in (None, "silero_vad_returned_no_regions") or any(region.get("source") != "silero_vad" for region in regions):
        raise PartScopeError("independent parent VAD failed; fallback cannot define part boundaries")
    if file_digest(Path(audio_path)) != audio_sha or _vad_model_identity() != model:
        raise PartScopeError("parent audio or independent VAD model changed during scan")
    _write_envelope(output_path, {
        "independent_vad": True, "audio_sha256": audio_sha, "sample_count": sample_count,
        "config": config_evidence, "model": model, "producer_sha256": _producer_sha(),
        "regions": regions,
    })


def _pcm_sha(path, root, name, deadline, *, start_sample=None, end_sample=None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PartScopeError("ffmpeg is required for source/child PCM readback")
    output = _path(root, f"work/{name}.pcm-sha256.txt")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
               "-map", "0:a:0"]
    if start_sample is not None:
        command += ["-af", f"atrim=start_sample={start_sample}:end_sample={end_sample},asetpts=PTS-STARTPTS"]
    command += ["-c:a", "pcm_s32le", "-f", "hash", "-hash", "sha256", "-"]
    _run(command, root, name, deadline, stdout_path=output)
    value = output.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"SHA256=[0-9a-f]{64}", value):
        raise PartScopeError("decoded PCM hash readback is invalid")
    return value.split("=", 1)[1]


def prepare_episode_parts(root, episode, source_video, audio_path, captions_path=None, *, total_timeout):
    from .transcribe import load_vtt_captions
    root, deadline = Path(root).resolve(), _deadline(total_timeout)
    _path(root, "work").mkdir(parents=True, exist_ok=True)
    source = _record(root, source_video, deadline)
    audio = _record(root, audio_path, deadline)
    captions = _record(root, captions_path, deadline) if captions_path is not None else None
    if captions is not None:
        captions["records"] = load_vtt_captions(captions_path)
    plan_path = _path(root, "work/part-plan.json")
    if plan_path.exists():
        plan = load_part_plan(root, episode, verify_files=False)
        if (plan["source"] != source or plan["captions"] != captions
                or any(plan["audio"].get(key) != value for key, value in audio.items())):
            raise PartScopeError("existing part plan belongs to different immutable parents")
        if _read_envelope(_path(root, "work/part-vad.json")) != plan["vad"]:
            raise PartScopeError("part plan/VAD receipt identity mismatch")
        _remaining(deadline)
        return plan
    audio.update(_probe_audio(audio_path, root, "parent-audio", deadline))
    vad_path = _path(root, "work/part-vad.json")
    _, config = _audit_config()
    model, producer = _vad_model_identity(), _producer_sha()
    binding = {"audio_sha256": audio["sha256"], "sample_count": audio["sample_count"],
               "config": config, "model": model, "producer_sha256": producer}
    if not vad_path.exists():
        failure_path = _path(root, f"work/part-vad-failures/{digest(binding)}.json")
        if failure_path.exists():
            raise PartScopeError("unchanged independent VAD failure is retained; no automatic rerun")
        try:
            _run([sys.executable, "-m", "mas.engine.part_audio", "_vad", str(audio_path),
                  str(vad_path), str(audio["sample_count"])], root, "part-vad", deadline)
        except Exception as exc:
            _write_envelope(failure_path, {"binding": binding, "error": str(exc)})
            raise
    vad = _read_envelope(vad_path)
    if any(vad.get(key) != value for key, value in binding.items()):
        raise PartScopeError("retained independent VAD has stale audio/model/config identity")
    for record in (source, audio, captions):
        if record is not None:
            _verify_file(root, record, deadline)
    from .part_scope import DELIVERY_BOUNDARY_POLICY
    plan = build_part_plan(episode=episode, source=source, audio=audio, vad=vad, captions=captions,
                           boundary_policy=DELIVERY_BOUNDARY_POLICY if episode >= 14 else None)
    _write_envelope(plan_path, plan)
    return plan


def _caption_records(plan, part):
    return [{**item, "caption_index": index,
             "start_ms": max(item["start_ms"], part["start_ms"]) - part["start_ms"],
             "end_ms": min(item["end_ms"], part["end_ms"]) - part["start_ms"]}
            for index, item in enumerate(
                (item for item in plan["captions"]["records"]
                 if item["caption_index"] in part["caption_indices"]), 1)]


def _caption_text(records):
    def timestamp(ms):
        hours, remaining = divmod(ms, 3600000)
        minutes, remaining = divmod(remaining, 60000)
        seconds, millis = divmod(remaining, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"
    return "WEBVTT\n\n" + "".join(
        f"{item['caption_index']}\n{timestamp(item['start_ms'])} --> "
        f"{timestamp(item['end_ms'])}\n{html.escape(item['text'], quote=False)}\n\n" for item in records)


def _result(root, plan, lineage, *, deadline=None, probe=False, verified=None):
    from .transcribe import load_vtt_captions
    part_id = lineage["part_id"]
    validate_part_lineage(plan, part_id, lineage)
    audio_path = _verify_file(root, lineage["audio"], deadline, verified=verified)
    caption_path = None
    if lineage["captions"] is not None:
        caption_path = _verify_file(root, lineage["captions"], deadline, verified=verified)
        if load_vtt_captions(caption_path) != _caption_records(plan, get_part(plan, part_id)):
            raise PartScopeError("child captions omit or alter parent text/timing")
    verification_key = _verification_key(plan, lineage)
    if probe and verification_key not in _VERIFIED_AUDIO:
        metadata = _probe_audio(audio_path, root, part_id, deadline)
        if metadata["sample_count"] != lineage["sample_count"]:
            raise PartScopeError("child audio sample count differs from its retained source range")
        parent_pcm = _pcm_sha(_path(root, plan["audio"]["relative_path"]), root,
                              part_id + "-parent-pcm", deadline,
                              start_sample=lineage["start_sample"], end_sample=lineage["end_sample"])
        child_pcm = _pcm_sha(audio_path, root, part_id + "-child-pcm", deadline)
        if parent_pcm != child_pcm or child_pcm != lineage["pcm_sha256"]:
            raise PartScopeError("child decoded PCM differs from its exact parent sample range")
        _remember_verified(verification_key)
    elif verification_key in _VERIFIED_AUDIO:
        _VERIFIED_AUDIO.move_to_end(verification_key)
    return {"audio_path": audio_path, "captions_path": caption_path,
            "lineage_path": _path(root, f"parts/{part_id}/prepare/audio-part.done.json"),
            "lineage": lineage, "plan_sha256": digest(plan),
            "parent_vad_regions": project_part_vad(plan, part_id)}


def validate_part_audio(root, episode, part_id, *, total_timeout=300, verified=None):
    root, deadline = Path(root).resolve(), _deadline(total_timeout)
    plan = load_part_plan(root, episode, verify_files=False)
    get_part(plan, part_id)
    for record in (plan["source"], plan["audio"], plan["captions"]):
        if record is not None:
            _verify_file(root, record, deadline, verified=verified)
    if _read_envelope(_path(root, "work/part-vad.json")) != plan["vad"]:
        raise PartScopeError("parent VAD receipt differs from part plan")
    lineage = _read_envelope(_path(root, f"parts/{part_id}/prepare/audio-part.done.json"))
    return _result(root, plan, lineage, deadline=deadline, probe=True, verified=verified)


def extract_part_audio(root, episode, part_id, *, total_timeout):
    root, deadline = Path(root).resolve(), _deadline(total_timeout)
    plan = load_part_plan(root, episode, verify_files=False)
    part = get_part(plan, part_id)
    for record in (plan["source"], plan["audio"], plan["captions"]):
        if record is not None:
            _verify_file(root, record, deadline)
    if _read_envelope(_path(root, "work/part-vad.json")) != plan["vad"]:
        raise PartScopeError("parent VAD receipt differs from part plan")
    destination = _path(root, f"parts/{part_id}/prepare")
    destination.mkdir(parents=True, exist_ok=True)
    lineage_path = destination / "audio-part.done.json"
    if lineage_path.exists():
        return _result(root, plan, _read_envelope(lineage_path), deadline=deadline, probe=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PartScopeError("ffmpeg is required for derived part audio")
    version_path = _path(root, f"work/{part_id}-ffmpeg-version.txt")
    _run([ffmpeg, "-version"], root, part_id + "-ffmpeg-version", deadline, stdout_path=version_path)
    ffmpeg_identity = {"executable_sha256": _sha(ffmpeg, deadline),
                       "version": version_path.read_text(encoding="utf-8").splitlines()[0]}
    binding = {"plan_sha256": digest(plan), "part_id": part_id,
               "producer_sha256": _producer_sha(), "ffmpeg_identity": ffmpeg_identity}
    failure_path = _path(root, f"parts/{part_id}/work/audio-failures/{digest(binding)}.json")
    if failure_path.exists():
        raise PartScopeError(f"{part_id}: unchanged derived-audio failure is retained")
    temporary = destination / f"audio.{uuid.uuid4().hex}.partial.flac"
    try:
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
              "-i", str(_path(root, plan["audio"]["relative_path"])), "-map", "0:a:0",
              "-af", f"atrim=start_sample={part['start_sample']}:end_sample={part['end_sample']},asetpts=PTS-STARTPTS",
              "-c:a", "flac", "-compression_level", "8", str(temporary)],
             root, part_id + "-extract", deadline)
        metadata = _probe_audio(temporary, root, part_id, deadline)
        expected_samples = part["end_sample"] - part["start_sample"]
        if metadata["sample_count"] != expected_samples:
            raise PartScopeError(f"{part_id}: derived audio did not retain the exact sample range")
        parent_pcm = _pcm_sha(_path(root, plan["audio"]["relative_path"]), root,
                              part_id + "-parent-pcm", deadline,
                              start_sample=part["start_sample"], end_sample=part["end_sample"])
        child_pcm = _pcm_sha(temporary, root, part_id + "-child-pcm", deadline)
        if parent_pcm != child_pcm:
            raise PartScopeError(f"{part_id}: extraction changed the exact parent audio samples")
        _verify_file(root, plan["audio"], deadline)
        audio_path = destination / "audio.flac"
        if audio_path.exists():
            audio_path.replace(destination / f"audio.{uuid.uuid4().hex}.unverified.flac")
        temporary.replace(audio_path)
        caption_record = None
        if plan["captions"] is not None:
            caption_path = destination / "captions.vtt"
            caption_temp = destination / f"captions.{uuid.uuid4().hex}.tmp"
            caption_temp.write_text(_caption_text(_caption_records(plan, part)), encoding="utf-8")
            if caption_path.exists():
                caption_path.replace(destination / f"captions.{uuid.uuid4().hex}.unverified.vtt")
            caption_temp.replace(caption_path)
            caption_record = _record(root, caption_path, deadline)
        lineage = {
            "format": PART_AUDIO_FORMAT, "episode": episode, "part_id": part_id,
            "plan_sha256": digest(plan), "parent_source_sha256": plan["source"]["sha256"],
            "parent_audio_sha256": plan["audio"]["sha256"],
            "start_sample": part["start_sample"], "end_sample": part["end_sample"],
            "start_ms": part["start_ms"], "end_ms": part["end_ms"],
            "sample_rate_hz": SAMPLE_RATE, "sample_count": expected_samples,
            "audio": _record(root, audio_path, deadline), "captions": caption_record,
            "producer_sha256": binding["producer_sha256"], "ffmpeg_identity": ffmpeg_identity,
            "pcm_sha256": child_pcm,
            "parent_vad_sha256": digest(project_part_vad(plan, part_id)),
        }
        validate_part_lineage(plan, part_id, lineage)
        _write_envelope(lineage_path, lineage)
        _remember_verified(_verification_key(plan, lineage))
        return _result(root, plan, lineage, deadline=deadline)
    except Exception as exc:
        _write_envelope(failure_path, {"binding": binding, "error": str(exc)})
        raise


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "_vad":
        raise SystemExit("internal usage: part_audio _vad AUDIO OUTPUT SAMPLE_COUNT")
    _scan_vad(sys.argv[2], sys.argv[3], int(sys.argv[4]))
