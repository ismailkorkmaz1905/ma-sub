import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .config import ROOT, episode_dir
from .hashing import sha256_file, sha256_json
from .notify import notify
from .remote import upload_verified
from .runpod import stop_current_pod
from .source_discovery import discover_episode_source
from .state import load, save, set_stage
from .engine.api import (
    AudioReviewConfig,
    RawASRConfig,
    build_strict_artifacts,
    correction_records_to_alignment_inputs,
    create_id_translation_pack,
    finalize_episode,
    resolve_tr_audio_reviews,
    transcribe_raw_audio,
)
from .engine.download import atomic_write_bytes, atomic_write_json, download_source
from .engine.forced_align import align_corrected_segments, validate_forced_alignment_data
from .engine.id_translation import (
    load_and_validate_id_translation_zip,
    load_default_id_translation_glossary,
)
from .engine.media import extract_audio
from .engine.tr_correction import create_tr_correction_pack, read_tr_correction_pack, validate_tr_correction_output


WAIT_TR = 20
WAIT_ID = 21
STRICT_DRIVE_OUTPUTS = ("mkv", "tr_srt", "id_srt")
STAGE_NOTIFICATION_NAMES = {
    "download": "kaynak dosyası kontrolü",
    "audio": "ses dosyası hazırlığı",
    "raw_asr": "Türkçe konuşma tanıma ve kanıt hazırlığı",
    "tr_pack": "Türkçe düzeltme paketi hazırlığı",
    "tr_return": "Türkçe düzeltme dönüşü doğrulaması",
    "audio_review": "Türkçe ses incelemesi",
    "forced_alignment": "altyazı zaman hizalaması",
    "id_pack": "Endonezce çeviri paketi hazırlığı",
    "id_return": "Endonezce çeviri dönüşü doğrulaması",
    "finalize": "final altyazı ve video üretimi",
    "drive_readback": "Google Drive yükleme ve hash doğrulaması",
}


def _paths(episode):
    root = episode_dir(episode)
    name = f"Muhtemel Ask {episode}.Bolum"
    dirs = {key: root / key for key in ("source", "prepare", "translation_input",
                                        "translation_output", "review", "final",
                                        "emergency", "logs", "work")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return root, name, dirs


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_guard(state, source_path):
    digest = sha256_file(source_path)
    recorded = state.get("source_sha256")
    if recorded and recorded != digest:
        raise RuntimeError("immutable source SHA-256 changed")
    state["source_sha256"] = digest
    return digest


def _guard_existing_source(state, source_dir):
    source_value = state.get("source_path")
    if not source_value:
        return
    source = Path(source_value).resolve()
    source_root = Path(source_dir).resolve()
    if source.parent != source_root or source.is_symlink() or not source.is_file():
        raise RuntimeError("recorded immutable source path is missing or unsafe")
    _source_guard(state, source)
    marker_path = source_root / "download.done.json"
    marker = _json(marker_path) if marker_path.is_file() else None
    video = marker.get("outputs", {}).get("video", {}) if isinstance(marker, dict) else {}
    if (marker or {}).get("stage") != "download" or Path(str(video.get("path", ""))).resolve() != source:
        raise RuntimeError("immutable source marker is missing or changed; refusing reacquisition")
    if video.get("sha256") != state["source_sha256"] or video.get("size_bytes") != source.stat().st_size:
        raise RuntimeError("immutable source marker does not match the recorded source")


def _stage(path, state, name, action):
    started = time.monotonic()
    notification_name = STAGE_NOTIFICATION_NAMES.get(name, name)
    set_stage(path, state, name, "running")
    print(f"[STAGE] {name}: START", flush=True)
    start_details = None
    if name == "download":
        start_details = (
            "Mevcut immutable kaynak varsa yeniden indirilmeyecek; byte ve "
            "SHA-256 bütünlüğü doğrulanacak."
        )
    notify(state["episode"], f"{notification_name} başladı", start_details)
    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(30):
            elapsed = time.monotonic() - started
            print(f"[STAGE] {name}: RUNNING elapsed={elapsed:.1f}s", flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        details = action() or {}
    except Exception as exc:
        heartbeat_stop.set()
        heartbeat_thread.join()
        elapsed = time.monotonic() - started
        set_stage(
            path,
            state,
            name,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=round(elapsed, 3),
        )
        print(f"[STAGE] {name}: FAIL elapsed={elapsed:.1f}s", flush=True)
        notify(
            state["episode"],
            f"{notification_name} başarısız",
            f"{type(exc).__name__}: {exc}",
        )
        raise
    heartbeat_stop.set()
    heartbeat_thread.join()
    elapsed = time.monotonic() - started
    set_stage(
        path,
        state,
        name,
        "pass",
        elapsed_seconds=round(elapsed, 3),
        **details,
    )
    print(f"[STAGE] {name}: PASS elapsed={elapsed:.1f}s", flush=True)
    notify(
        state["episode"],
        f"{notification_name} tamamlandı",
        f"Süre: {elapsed:.1f} saniye",
    )
    return details


def _load_configs():
    config_dir = ROOT / "config" / "production"
    values = []
    for name in ("series.yaml", "names.yaml", "religious_terms.yaml"):
        with (config_dir / name).open(encoding="utf-8") as handle:
            values.append(yaml.safe_load(handle))
    return config_dir, *values


def _validated_source_url(value):
    url = str(value).strip()
    parsed = urlparse(url)
    if (
        not url
        or any(character.isspace() for character in url)
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("source URL is invalid")
    return url


def _pending_extra_audio_review_uids(tr_pack, tr_text):
    pack_path = Path(tr_pack)
    output_path = Path(tr_text)
    if not pack_path.is_file() or not output_path.is_file():
        return ()
    pack = read_tr_correction_pack(pack_path)
    output = validate_tr_correction_output(pack_path, output_path)
    review_uids = {str(item["hole_uid"]) for item in pack.speech_holes}
    review_uids.update(
        str(item["utterance_uid"])
        for item in pack.asr_hallucination_records
    )
    explicit_review_uids = {
        str(item["utterance_uid"])
        for item in pack.asr_hallucination_records
        if "explicit_uid_review_request" in str(item.get("reason", "")).split(", ")
    }
    return tuple(
        str(record["utterance_uid"])
        for record in output.records
        if str(record["utterance_uid"]) in explicit_review_uids
        or (
            record["review_required"] is True
            and str(record["utterance_uid"]) not in review_uids
        )
    )


def _resolve_source_url(url_path, state, episode, requested_url=None):
    existing = None
    invalid_existing = False
    if url_path.exists() or url_path.is_symlink():
        if url_path.is_symlink() or not url_path.is_file():
            raise RuntimeError("source URL path is unsafe")
        try:
            existing = _validated_source_url(url_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, RuntimeError):
            invalid_existing = True

    requested = _validated_source_url(requested_url) if requested_url is not None else None
    if existing is not None:
        if requested is not None and requested != existing:
            raise RuntimeError("source URL cannot change after episode initialization")
        return existing
    if invalid_existing and (state.get("source_path") or state.get("source_sha256")):
        raise RuntimeError("recorded source URL is invalid; refusing source identity change")

    resolved = requested or discover_episode_source(
        episode,
        cookies_file=os.getenv("MAS_YTDLP_COOKIES"),
    )
    resolved = _validated_source_url(resolved)
    atomic_write_bytes(url_path, (resolved + "\n").encode("utf-8"))
    if _validated_source_url(url_path.read_text(encoding="utf-8")) != resolved:
        raise RuntimeError("source URL atomic readback failed")
    return resolved


def run(episode, source_url=None, fixture=False, stop_after=None):
    if fixture:
        return run_fixture(episode, stop_after=stop_after)
    root, name, dirs = _paths(episode)
    state_path = dirs["work"] / "state.json"
    state = load(state_path, episode)
    state["mode"] = "strict"
    save(state_path, state)
    notify(episode, "işleme alındı")
    url_path = dirs["source"] / "source.url"
    url = _resolve_source_url(url_path, state, episode, source_url)
    config_dir, series, names, religious = _load_configs()
    holder = {}
    _guard_existing_source(state, dirs["source"])

    def acquire():
        result = download_source(url, dirs["source"], output_stem=name, retries=3,
                                 attempts=3, socket_timeout=30,
                                 cookies_file=os.getenv("MAS_YTDLP_COOKIES"))
        holder["download"] = result
        digest = _source_guard(state, result.video_path)
        state["source_path"] = str(Path(result.video_path).resolve())
        save(state_path, state)
        return {"source": str(result.video_path), "sha256": digest, "resumed": result.resumed}
    _stage(state_path, state, "download", acquire)
    download = holder["download"]

    def prepare_audio():
        _source_guard(state, download.video_path)
        result = extract_audio(download.video_path, dirs["prepare"], source_sha256=state["source_sha256"])
        holder["audio"] = result
        return {"path": str(result.audio_path), "sha256": sha256_file(result.audio_path), "resumed": result.resumed}
    _stage(state_path, state, "audio", prepare_audio)
    audio = holder["audio"]

    tr_pack = dirs["translation_input"] / f"{name}_TR_CORRECTION_PACK.zip"
    tr_text = dirs["translation_output"] / f"{name}_TR_TEXT_CORRECTED.zip"
    extra_audio_review_uids = _pending_extra_audio_review_uids(tr_pack, tr_text)

    def transcribe():
        _source_guard(state, download.video_path)
        data = transcribe_raw_audio(
            audio.audio_path, dirs["prepare"], episode=episode,
            config=RawASRConfig(
                model_name=series["whisper_model"],
                allow_cpu_fallback=False,
                extra_audio_review_uids=extra_audio_review_uids,
            ),
            captions_path=download.captions_path,
            canonical_names=tuple(names["canonical_names"]),
            religious_terms=tuple(item["source"] for item in religious["terms"]),
        )
        settings = data.get("model", {}).get("settings", {})
        if data.get("independent_vad") is not True or settings.get("allow_cpu_fallback") is not False:
            raise RuntimeError("strict raw ASR requires independent VAD and GPU-only policy")
        holder["raw"] = data
        path = dirs["prepare"] / "raw_asr_v2.json"
        return {"path": str(path), "sha256": sha256_file(path), "resumed": data.get("resumed", False)}
    _stage(state_path, state, "raw_asr", transcribe)
    raw = holder["raw"]

    def make_tr_pack():
        manifest = create_tr_correction_pack(
            raw["correction_utterances"], raw["speech_hole_records"], tr_pack,
            episode=episode, batch_size=250, speech_hole_audio_root=dirs["prepare"],
            asr_hallucination_records=raw["asr_hallucination_records"],
            asr_hallucination_audio_root=dirs["prepare"],
            rebind_text_output_path=dirs["translation_output"] / f"{name}_TR_TEXT_CORRECTED.zip")
        return {"path": str(tr_pack), "sha256": sha256_file(tr_pack), "input_sha256": manifest["input_sha256"]}
    _stage(state_path, state, "tr_pack", make_tr_pack)

    if not tr_text.is_file():
        set_stage(state_path, state, "tr_return", "blocked", expected=str(tr_text))
        notify(episode, "Türkçe düzeltme bekleniyor", str(tr_text))
        print(f"[WAIT] TR CORRECTION\nPack: {tr_pack}\nExpected: {tr_text}\nSafe to stop RunPod.")
        return WAIT_TR
    def validate_tr_return():
        result = validate_tr_correction_output(tr_pack, tr_text)
        return {"sha256": sha256_file(tr_text), "records": len(result.records)}
    _stage(state_path, state, "tr_return", validate_tr_return)

    tr_output = dirs["translation_output"] / f"{name}_TR_CORRECTED.zip"
    review_path = dirs["prepare"] / "audio_review_v2.json"
    overrides_path = dirs["review"] / "audio_review_overrides.json"
    overrides = _json(overrides_path) if overrides_path.is_file() else None
    def review_audio():
        report = resolve_tr_audio_reviews(
            tr_pack, tr_text, tr_output, review_path,
            dirs["prepare"] / "audio_review_v2.recovery.json",
            config=AudioReviewConfig(model_name=series["whisper_model"], device="cuda", allow_cpu_fallback=False),
            manual_overrides=overrides)
        holder["correction"] = validate_tr_correction_output(tr_pack, tr_output)
        holder["review"] = report
        return {"path": str(review_path), "sha256": sha256_file(review_path), "review_count": report["review_count"]}
    _stage(state_path, state, "audio_review", review_audio)
    correction = holder["correction"]
    alignment_path = dirs["prepare"] / "forced_alignment_v2.json"
    def align():
        pack = read_tr_correction_pack(tr_pack)
        bundle = correction_records_to_alignment_inputs(
            pack.utterances,
            correction.records,
            speech_hole_records=pack.speech_holes,
            asr_hallucination_records=pack.asr_hallucination_records,
        )
        if alignment_path.is_file():
            result = _json(alignment_path)
            validate_forced_alignment_data(result)
            if result.get("audio_sha256") != sha256_file(audio.audio_path) or result.get("provenance", {}).get("device") != "cuda":
                raise RuntimeError("cached alignment is stale or was not produced on CUDA")
            resumed = True
        else:
            _source_guard(state, download.video_path)
            result = align_corrected_segments(audio.audio_path, bundle.alignment_inputs, device="cuda")
            atomic_write_json(alignment_path, result)
            resumed = False
        holder["aligned"] = result
        return {"path": str(alignment_path), "sha256": sha256_file(alignment_path), "device": "cuda", "resumed": resumed}
    _stage(state_path, state, "forced_alignment", align)
    aligned = holder["aligned"]

    schema_path = dirs["prepare"] / "aligned_tr_schema_v2.json"
    id_pack = dirs["translation_input"] / f"{name}_ID_TRANSLATION_PACK.zip"
    def make_id_pack():
        artifacts = build_strict_artifacts(
            raw,
            correction.records,
            aligned,
            episode=episode,
            acoustic_audio_review=holder["review"],
        )
        atomic_write_json(schema_path, artifacts.schema)
        atomic_write_json(dirs["prepare"] / "alignment_window_audit_v2.json", list(artifacts.preparation.window_audit))
        atomic_write_json(dirs["prepare"] / "final_speech_coverage_v2.json", artifacts.speech_coverage_report)
        atomic_write_json(dirs["prepare"] / "pre_id_timing_qa_v2.json", artifacts.timing_qa_report)
        manifest = create_id_translation_pack(
            artifacts,
            id_pack,
            batch_size=series["batch_size"],
            glossary=load_default_id_translation_glossary(),
        )
        holder["artifacts"] = artifacts
        holder["id_manifest"] = manifest
        return {
            "path": str(id_pack),
            "sha256": sha256_file(id_pack),
            "schema_sha256": manifest["schema_sha256"],
        }
    _stage(state_path, state, "id_pack", make_id_pack)
    artifacts = holder["artifacts"]
    manifest = holder["id_manifest"]

    id_output = dirs["translation_output"] / f"{name}_ID_TRANSLATED.zip"
    if not id_output.is_file():
        set_stage(state_path, state, "id_return", "blocked", expected=str(id_output))
        notify(episode, "Endonezce çeviri bekleniyor", str(id_output))
        print(f"[WAIT] ID TRANSLATION\nPack: {id_pack}\nExpected: {id_output}\nSafe to stop RunPod.")
        return WAIT_ID
    def validate_id_return():
        result = load_and_validate_id_translation_zip(
            artifacts.schema,
            id_output,
            input_manifest=manifest,
        )
        return {"sha256": sha256_file(id_output), "records": result.output_block_count}
    _stage(state_path, state, "id_return", validate_id_return)

    report_path = dirs["final"] / f"{name}_FINALIZATION_REPORT_V2.json"
    def finalize():
        report = finalize_episode(
            episode_root=root, episode=episode, source_video=download.video_path,
            raw_asr_path=dirs["prepare"] / "raw_asr_v2.json", forced_alignment_path=alignment_path,
            tr_correction_pack=tr_pack, tr_text_correction_output=tr_text,
            tr_correction_output=tr_output, audio_review_path=review_path,
            aligned_schema=schema_path, id_translation_pack=id_pack, id_translation_zip=id_output,
            series_config=config_dir / "series.yaml", names_config=config_dir / "names.yaml",
            religious_config=config_dir / "religious_terms.yaml")
        if report.get("status") != "PASS":
            raise RuntimeError("strict finalization did not PASS")
        holder["final_report"] = report
        return {"report": str(report_path), "sha256": sha256_file(report_path)}
    _stage(state_path, state, "finalize", finalize)
    report = holder["final_report"]

    remote_root = os.getenv("MAS_DRIVE_STRICT_REMOTE")
    if not remote_root:
        set_stage(state_path, state, "drive_readback", "blocked", error="MAS_DRIVE_STRICT_REMOTE is not set")
        raise RuntimeError("strict output is local only; Drive byte/SHA-256 readback is mandatory")
    receipt_path = dirs["final"] / "drive_readback_receipt.json"
    def publish():
        receipts = []
        for key in STRICT_DRIVE_OUTPUTS:
            local = root / report["outputs"][key]["relative_path"]
            receipts.append(upload_verified(local, f"{remote_root.rstrip('/')}/{name}/{local.name}"))
        atomic_write_json(receipt_path, {"status": "PASS", "mode": "strict", "files": receipts})
        return {"receipt": str(receipt_path), "sha256": sha256_file(receipt_path)}
    _stage(state_path, state, "drive_readback", publish)
    notify(episode, "bölüm hazır", str(receipt_path))
    set_stage(state_path, state, "compute_shutdown", "running")
    if os.getenv("MAS_EXTERNAL_RUNPOD_CONTROLLER") == "1":
        shutdown = {"requested": False, "reason": "external_controller", "delegated": True}
    else:
        shutdown = stop_current_pod()
    set_stage(state_path, state, "compute_shutdown", "pass", **shutdown)
    print("STRICT PASS")
    return 0


def run_fixture(episode, stop_after=None):
    _, _, dirs = _paths(episode)
    state_path = dirs["work"] / "fixture-state.json"
    state = load(state_path, episode)
    state["mode"] = "offline-fixture"
    for index, name in enumerate(("fixture_source", "fixture_transform", "fixture_verify"), start=1):
        if state.get("stages", {}).get(name, {}).get("status") == "pass":
            continue
        payload = {"episode": episode, "stage": name, "previous": state.get("fixture_sha256")}
        output = dirs["work"] / f"{name}.json"
        atomic_write_json(output, payload)
        state["fixture_sha256"] = sha256_json(payload)
        set_stage(state_path, state, name, "pass", output=str(output), sha256=sha256_file(output))
        if stop_after == index:
            return 75
    return 0


def status(episode):
    _, _, dirs = _paths(episode)
    print(json.dumps(load(dirs["work"] / "state.json", episode), indent=2, ensure_ascii=False))
    return 0
