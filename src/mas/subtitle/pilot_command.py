import io
import json
import re
import sys
import wave
import zipfile
from pathlib import Path

from ..config import episode_dir
from ..engine.download import atomic_write_bytes
from ..engine.tr_correction import read_tr_correction_pack
from ..reliability import IntegrityError, atomic_json, digest, file_digest, run_command
from .pilot import build_pilot, write_pilot
from .pilot_worker import model_identity, verify_source_clip


def prepare_samples(episode, pack_path, uids):
    pack = read_tr_correction_pack(pack_path)
    if pack.manifest["episode"] != episode:
        raise IntegrityError("sample pack episode mismatch")
    if not uids or len(uids) > 8 or len(set(uids)) != len(uids):
        raise ValueError("request one to eight distinct sample UIDs")
    candidates = {record.get("utterance_uid", record.get("hole_uid")): record
                  for record in (*pack.speech_holes, *pack.asr_hallucination_records)}
    missing = sorted(set(uids) - set(candidates))
    if missing:
        raise IntegrityError(f"no embedded source audio for requested UIDs: {missing}")
    pack_sha = file_digest(Path(pack_path))
    samples = []
    payloads = []
    with zipfile.ZipFile(pack_path) as archive:
        for index, uid in enumerate(uids, 1):
            item = candidates[uid]
            payload = archive.read(item["audio_member"])
            import hashlib
            if hashlib.sha256(payload).hexdigest() != item["audio_sha256"]:
                raise IntegrityError("embedded sample SHA mismatch")
            with wave.open(io.BytesIO(payload), "rb") as stream:
                duration = round(stream.getnframes() * 1000 / stream.getframerate())
            if not 0 < duration <= 120_000:
                raise IntegrityError("sample exceeds pilot duration limit")
            if abs(duration - (item["clip_end_ms"] - item["clip_start_ms"])) > 2:
                raise IntegrityError("sample WAV duration disagrees with source offsets")
            name = f"sample-{index:02d}.wav"
            samples.append({"uid": uid, "path": name, "audio_sha256": item["audio_sha256"],
                            "offset_ms": item["clip_start_ms"], "duration_ms": duration,
                            "source_pack_sha256": pack_sha, "member": item["audio_member"]})
            payloads.append((name, payload))
    output = episode_dir(episode) / "work" / "subtitle-pilot" / ("samples-" + digest(samples)[:16])
    if output.exists():
        saved = json.loads((output / "samples.json").read_text(encoding="utf-8"))
        if saved != {"episode": episode, "samples": samples}:
            raise IntegrityError("saved sample manifest differs")
        for item in samples:
            if file_digest(output / item["path"]) != item["audio_sha256"]:
                raise IntegrityError("saved sample audio differs")
    else:
        for name, payload in payloads:
            atomic_write_bytes(output / name, payload)
        atomic_json(output / "samples.json", {"episode": episode, "samples": samples})
    print(output / "samples.json")
    return 0


def run_pilot_command(args):
    if args.episode < 1:
        raise ValueError("episode must be positive")
    if args.pack:
        if args.audio or args.evidence:
            raise ValueError("sample extraction cannot be combined with inference or replay")
        return prepare_samples(args.episode, args.pack, args.uid)
    if bool(args.audio) == bool(args.evidence):
        raise ValueError("choose exactly one of --audio or --evidence")
    root = episode_dir(args.episode) / "work" / "subtitle-pilot"
    if args.evidence:
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        result = build_pilot(evidence)
        if result["data"]["episode"] != args.episode:
            raise IntegrityError("replay episode mismatch")
        output = root / ("replay-" + result["sha256"][:16])
        write_pilot(output, result)
    else:
        if not args.model_dir or not args.diarization_model_dir:
            raise ValueError("pre-provisioned local ASR and diarization model directories are required")
        if not re.fullmatch(r"[a-f0-9]{64}", args.source_sha256 or "") or args.offset_ms < 0:
            raise ValueError("source SHA-256 and nonnegative source-relative --offset-ms are required")
        if not args.source_audio:
            raise ValueError("--source-audio WAV is required to verify sample provenance")
        verify_source_clip(args.audio, args.source_audio, args.source_sha256, args.offset_ms)
        with wave.open(args.audio, "rb") as stream:
            if not 0 < stream.getnframes() / stream.getframerate() <= 120:
                raise ValueError("pilot accepts at most 120 seconds of WAV")
        request = {"episode": args.episode, "audio": str(Path(args.audio).resolve()),
                   "audio_sha256": file_digest(Path(args.audio)), "offset_ms": args.offset_ms,
                   "source_sha256": args.source_sha256,
                   "source_audio": str(Path(args.source_audio).resolve()),
                   "model_sha256": model_identity(args.model_dir),
                   "diarization_sha256": model_identity(args.diarization_model_dir),
                   "model_dir": str(Path(args.model_dir).resolve()),
                   "diarization_model_dir": str(Path(args.diarization_model_dir).resolve())}
        output = root / ("inference-" + digest(request)[:16])
        if (output / "pilot.json").exists():
            raise IntegrityError("completed pilot exists; inspect it instead of repeating inference")
        request["output"] = str(output.resolve())
        atomic_json(output / "request.json", request)
        run_command([sys.executable, "-m", "mas.subtitle.pilot_worker", str(output / "request.json")],
                    stage="subtitle-pilot", log_path=output / "worker.log", timeout_seconds=180)
    print(f"PILOT DRAFT: {output}; acoustic acceptance NOT VERIFIED; no strict publication")
    return 0
