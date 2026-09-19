import json
import os
import re
import stat
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .config import ROOT, episode_dir
from .hashing import sha256_file, sha256_json
from .notify import enqueue_notification as notify
from .progress import mark_work_progress
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
from .engine.download import (
    atomic_write_bytes,
    atomic_write_json,
    download_source,
    load_valid_stage_marker,
    validate_download,
    write_stage_marker,
)
from .engine.forced_align import (
    align_corrected_segments,
    correction_deletes_lexical_tokens,
    validate_forced_alignment_data,
)
from .engine.id_translation import (
    build_production_translation_policy,
    load_and_validate_id_translation_zip,
    load_default_id_translation_glossary,
)
from .engine.media import extract_audio
from .engine.translation_workspace import prepare_id_translation_workspaces, validate_id_workspace_output
from .engine.subtitle_qa import run_subtitle_qa, assert_final_qa
from .engine.timing_qa import TimingQAV2Config, run_timing_qa_v2, assert_timing_qa_v2
from .engine.burned_mp4 import (burn_indonesian_mp4, SUBTITLE_STYLE, plan_encoding_settings,
                               inspect_encoding_storage, create_encoding_samples, qualify_encoding)
from .engine.episode_archive import file_record
from .delivery import (ALIGNMENT_RECOVERY_COMPLETE, READY_FOR_DELIVERY, READY_FOR_LOCAL_ENCODE,
                       WAIT_MP4_SAMPLE, write_delivery_export)
from .engine.tr_correction import create_tr_correction_pack, read_tr_correction_pack, validate_tr_correction_output


WAIT_TR = 20
WAIT_ID = 21
STRICT_DRIVE_OUTPUTS = ("mp4",)
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
    "strict_finalize": "strict altyazı ve softsub video üretimi",
    "burn_mp4": "Endonezce altyazılı final video üretimi",
    "drive_readback": "Google Drive yükleme ve hash doğrulaması",
}
STAGE_START_DETAILS = {
    "download": "Kaynak kimliği ile byte ve SHA-256 bütünlüğü kontrol edilecek; gerekirse indirme başlatılacak.",
    "audio": "Değişmez kaynak videodan üretim ses dosyası hazırlanacak.",
    "raw_asr": "GPU üzerinde Türkçe ASR ve inceleme kanıtları hazırlanacak; CPU fallback kullanılmayacak.",
    "tr_pack": "Türkçe düzeltme için hash bağlı handoff paketi hazırlanacak.",
    "tr_return": "Dönen Türkçe düzeltme paketinin şeması, kimliği ve değişmez alanları doğrulanacak.",
    "audio_review": "Bekleyen Türkçe kayıtlar ses kanıtıyla incelenecek.",
    "forced_alignment": "Düzeltilmiş Türkçe metin CUDA üzerinde akustik olarak hizalanacak.",
    "id_pack": "Endonezce çeviri için değişmez Türkçe metne bağlı paket hazırlanacak.",
    "id_return": "Dönen Endonezce çevirinin kimliği, sırası ve değişmez alanları doğrulanacak.",
    "strict_finalize": "Strict altyazılar üretilecek ve tüm zamanlama ve kalite kuralları yeniden doğrulanacak.",
    "burn_mp4": "Doğrulanmış Endonezce altyazı değişmez kaynağa gömülecek ve encoding makbuzu doğrulanacak.",
    "drive_readback": "Final dosyaları geçici adla yüklenecek; byte ve SHA-256 readback doğrulanacak.",
}
def _requires_external_delivery():
    return (
        os.getenv("MAS_EXTERNAL_RUNPOD_CONTROLLER") == "1"
        or bool(os.getenv("RUNPOD_POD_ID", "").strip())
    )


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


def _apply_audio_review_reset(root, episode, name, tr_pack, tr_text):
    marker_path = Path(root) / "review" / "audio_review_reset.json"
    if not marker_path.is_file():
        return None
    saved = _json(marker_path)
    body = saved.get("data")
    marker_sha256 = saved.get("sha256")
    if (
        not isinstance(body, dict)
        or marker_sha256 != sha256_json(body)
        or body.get("format") != "mas-audio-review-reset-1"
        or body.get("episode") != episode
        or not isinstance(body.get("reason"), str)
        or not body["reason"].strip()
    ):
        raise RuntimeError("audio-review reset marker integrity mismatch")
    pack = read_tr_correction_pack(tr_pack)
    provisional = validate_tr_correction_output(tr_pack, tr_text)
    if (
        body.get("correction_input_sha256") != pack.manifest["input_sha256"]
        or body.get("provisional_output_sha256") != provisional.output_sha256
    ):
        raise RuntimeError("audio-review reset marker input binding mismatch")
    expected_paths = {
        "prepare/audio_review_v2.json",
        "prepare/audio_review_v2.recovery.json",
        f"translation_output/{name}_TR_CORRECTED.zip",
    }
    artifacts = body.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or {item.get("relative_path") for item in artifacts if isinstance(item, dict)}
        != expected_paths
    ):
        raise RuntimeError("audio-review reset artifact inventory mismatch")
    archive = Path(root) / "review" / "superseded-audio-review" / marker_sha256
    receipt_path = archive / "receipt.json"
    if receipt_path.is_file():
        receipt = _json(receipt_path)
        receipt_body = receipt.get("data")
        if (
            not isinstance(receipt_body, dict)
            or receipt.get("sha256") != sha256_json(receipt_body)
            or receipt_body.get("reset_marker_sha256") != marker_sha256
            or receipt_body.get("artifacts") != artifacts
        ):
            raise RuntimeError("audio-review reset receipt integrity mismatch")
        return receipt_body
    for item in artifacts:
        if (
            set(item) != {"relative_path", "size_bytes", "sha256"}
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] <= 0
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise RuntimeError("audio-review reset artifact record is invalid")
        source = Path(root) / item["relative_path"]
        destination = archive / item["relative_path"]
        if destination.is_file():
            candidate = destination
        elif source.is_file() and not source.is_symlink():
            if source.stat().st_size != item["size_bytes"] or sha256_file(source) != item["sha256"]:
                raise RuntimeError("audio-review reset source artifact changed")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            candidate = destination
        else:
            raise RuntimeError("audio-review reset source artifact is missing")
        if candidate.stat().st_size != item["size_bytes"] or sha256_file(candidate) != item["sha256"]:
            raise RuntimeError("audio-review reset archive readback mismatch")
    receipt_body = {
        "format": "mas-audio-review-reset-receipt-1",
        "episode": episode,
        "reset_marker_sha256": marker_sha256,
        "artifacts": artifacts,
    }
    atomic_write_json(
        receipt_path,
        {"data": receipt_body, "sha256": sha256_json(receipt_body)},
    )
    return receipt_body


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
    notification_root = path.parent.parent if path.parent.name == 'work' else path.parent
    if notification_root.parent.name == 'parts':
        notification_name = notification_root.name + ': ' + notification_name
        notification_root = notification_root.parent.parent
    set_stage(path, state, name, "running")
    mark_work_progress(name)
    print(f"[STAGE] {name}: START", flush=True)
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
            f"Sonuç: aşama tamamlanamadı.\n"
            f"Hata: {type(exc).__name__}: {exc}\n"
            f"Sonraki adım: hatayı giderip aynı bölüm komutuyla güvenli devam edin.",
            root=notification_root,
            kind="terminal",
        )
        try:
            exc._mas_notification_sent = True
        except (AttributeError, TypeError):
            pass
        raise
    heartbeat_stop.set()
    heartbeat_thread.join()
    elapsed = time.monotonic() - started
    if details.get("sample_review_required"):
        set_stage(path, state, name, "blocked", elapsed_seconds=round(elapsed, 3), **details)
        print(f"[STAGE] {name}: WAIT_SAMPLE_REVIEW", flush=True)
        return details
    set_stage(
        path,
        state,
        name,
        "pass",
        elapsed_seconds=round(elapsed, 3),
        **details,
    )
    mark_work_progress(name, completed=True)
    print(f"[STAGE] {name}: PASS elapsed={elapsed:.1f}s", flush=True)
    if "resumed" in details:
        print(f"[CHECKPOINT] {name}: resumed={str(bool(details['resumed'])).lower()}", flush=True)
    if name in {'audio', 'raw_asr', 'forced_alignment', 'strict_finalize', 'strict_partial_finalize'} and not details.get('resumed'):
        notify(state['episode'], f'{notification_name} tamamlandı',
               json.dumps(details, ensure_ascii=False, sort_keys=True), root=notification_root, kind='milestone')
    return details


def _ensure_forced_alignment_marker(alignment_path, result, input_sha256,
                                    correction_output_sha256):
    marker_path = alignment_path.with_suffix(".done.json")
    if marker_path.exists() or marker_path.is_symlink():
        if marker_path.is_symlink() or not marker_path.is_file():
            raise RuntimeError("forced-alignment stage marker is unsafe; preserve it")
        marker = load_valid_stage_marker(
            marker_path,
            stage="forced_alignment_v2",
            input_sha256=input_sha256,
            required_output_keys=("forced_alignment_v2",),
            allowed_root=alignment_path.parent,
        )
        details = marker.get("details") if marker is not None else None
        if (
            marker is None
            or Path(marker["outputs"]["forced_alignment_v2"]["path"]).resolve(strict=True)
            != alignment_path.resolve(strict=True)
            or not isinstance(details, dict)
            or details.get("alignment_sha256") != result.get("alignment_sha256")
            or details.get("correction_output_sha256") != correction_output_sha256
        ):
            raise RuntimeError("forced-alignment stage marker binding changed; preserve it")
        return
    write_stage_marker(
        marker_path,
        stage="forced_alignment_v2",
        input_sha256=input_sha256,
        outputs={"forced_alignment_v2": alignment_path},
        details={
            "alignment_sha256": result["alignment_sha256"],
            "correction_output_sha256": correction_output_sha256,
        },
    )


def _aligned_checkpoint(alignment_path, audio_path, alignment_inputs, vad_regions,
                        correction_output_sha256, *, resume_scope=None):
    binding = {
        "audio_sha256": sha256_file(audio_path),
        "alignment_inputs": alignment_inputs,
        "vad_regions": vad_regions,
        "device": "cuda",
        "code": {
            name: sha256_file(ROOT / name)
            for name in (
                "src/mas/engine/forced_align.py",
                "src/mas/engine/alignment_recovery.py",
                "src/mas/engine/speaker.py",
                "src/mas/engine/workflow.py",
                "requirements.lock",
            )
        },
    }
    input_sha256 = sha256_json(binding)
    marker_path = alignment_path.with_suffix(".binding.json")
    if alignment_path.is_file():
        marker = _json(marker_path) if marker_path.is_file() else {}
        if (
            marker.get("input_sha256") != input_sha256
            or marker.get("output_sha256") != sha256_file(alignment_path)
        ):
            raise RuntimeError("cached alignment binding is stale or missing; preserve it before realignment")
        result = _json(alignment_path)
        validate_forced_alignment_data(result)
        if (
            result.get("audio_sha256") != binding["audio_sha256"]
            or result.get("provenance", {}).get("device") != "cuda"
        ):
            raise RuntimeError("cached alignment is stale or was not produced on CUDA")
        _ensure_forced_alignment_marker(
            alignment_path, result, input_sha256, correction_output_sha256
        )
        return result, True
    result = align_corrected_segments(
        audio_path, alignment_inputs, device="cuda", vad_regions=vad_regions,
        checkpoint_dir=alignment_path.parent / "forced_alignment_units",
        **({"resume_scope": resume_scope} if resume_scope is not None else {}),
    )
    atomic_write_json(alignment_path, result)
    atomic_write_json(marker_path, {
        "input_sha256": input_sha256,
        "output_sha256": sha256_file(alignment_path),
    })
    _ensure_forced_alignment_marker(
        alignment_path, result, input_sha256, correction_output_sha256
    )
    return result, False


_STRICT_FINALIZE_PRODUCER_FILES = (
    "src/mas/engine/audio_review.py",
    "src/mas/engine/download.py",
    "src/mas/engine/episode_archive.py",
    "src/mas/engine/finalize.py",
    "src/mas/engine/forced_align.py",
    "src/mas/engine/alignment_recovery.py",
    "src/mas/engine/id_translation.py",
    "src/mas/engine/subtitle_metadata.py",
    "src/mas/engine/media.py",
    "src/mas/engine/mux.py",
    "src/mas/engine/srt.py",
    "src/mas/engine/subtitle_qa.py",
    "src/mas/engine/timing_qa.py",
    "src/mas/engine/tr_correction.py",
    "src/mas/engine/workflow.py",
    "requirements.lock",
)


def _strict_finalize_input_sha256(episode, inputs):
    records = {}
    for name, value in sorted(inputs.items()):
        path = Path(value)
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"strict finalization input is missing or unsafe: {name}")
        records[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    producer = {
        name: sha256_file(ROOT / name)
        for name in _STRICT_FINALIZE_PRODUCER_FILES
    }
    return sha256_json({"episode": episode, "inputs": records, "producer": producer})


def _load_strict_finalize_checkpoint(marker_path, report_path, expected_outputs,
                                     episode_root, episode, input_sha256):
    marker_path = Path(marker_path)
    report_path = Path(report_path)
    if not marker_path.exists() and not marker_path.is_symlink():
        return None
    if marker_path.is_symlink() or not marker_path.is_file():
        raise RuntimeError("strict finalization checkpoint marker is unsafe; preserve it")
    marker = load_valid_stage_marker(
        marker_path,
        stage="strict_finalize_v2",
        input_sha256=input_sha256,
        required_output_keys=tuple(expected_outputs),
        allowed_root=Path(episode_root) / "final",
    )
    if marker is None:
        raise RuntimeError("strict finalization checkpoint binding changed; preserve it")
    for name, expected in expected_outputs.items():
        recorded = Path(marker["outputs"][name]["path"])
        if recorded.resolve(strict=True) != Path(expected).resolve(strict=True):
            raise RuntimeError("strict finalization checkpoint output path changed; preserve it")
    try:
        report_details = report_path.lstat()
        expected_report_path = Path(expected_outputs["report"])
        report_is_canonical = (
            report_path.resolve(strict=True)
            == expected_report_path.resolve(strict=True)
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "strict finalization report path is missing or unsafe; preserve it"
        ) from exc
    if (
        stat.S_ISLNK(report_details.st_mode)
        or not stat.S_ISREG(report_details.st_mode)
        or report_details.st_size <= 0
        or not report_is_canonical
    ):
        raise RuntimeError(
            "strict finalization report path is missing or unsafe; preserve it"
        )
    try:
        report = _json(report_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("strict finalization report is unreadable; preserve it") from exc
    if report.get("status") != "PASS" or report.get("episode") != episode:
        raise RuntimeError("strict finalization checkpoint has no matching PASS report")
    actual_outputs = {
        name: file_record(path, episode_root)
        for name, path in expected_outputs.items()
        if name != "report"
    }
    if report.get("outputs") != actual_outputs:
        raise RuntimeError("strict finalization report output binding changed; preserve it")
    return report


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
            (
                record["review_required"] is True
                or correction_deletes_lexical_tokens(
                    str(record["asr_text"]),
                    str(record["tr_corrected"]),
                )
            )
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


def _validate_id_quality(artifacts, validation, series, names, religious):
    schema = artifacts.schema
    policy = build_production_translation_policy(series, names, religious)
    if "production_policy" in schema and schema["production_policy"] != policy:
        raise RuntimeError("production policy changed after translation pack freeze")
    records = validation.ordered_records(schema)
    pending = [record["block_uid"] for record in records if record.get("review_required")]
    if pending:
        raise RuntimeError(f"Indonesian translation still requires review: {pending}")
    timing_report = run_timing_qa_v2(
        schema["blocks"],
        speech_coverage_report=artifacts.speech_coverage_report,
        alignment_report=artifacts.alignment_report,
        id_text_by_uid={record["block_uid"]: record["id_final"] for record in records},
        config=TimingQAV2Config(**policy["timing_qa"]),
    )
    assert_timing_qa_v2(timing_report)
    qa_report = run_subtitle_qa(
        [{**block, "schema_sha256": schema["schema_sha256"], "primary_text": block["tr_text"]}
         for block in schema["blocks"]],
        [{**record, "tr_final": record["tr_text"]} for record in records],
        names_config=names,
        religious_config=religious,
        preferred_max_cps=series["subtitle"]["preferred_max_cps"],
        line_limit=series["subtitle"]["qa_max_chars_per_line"],
    )
    assert_final_qa(qa_report)
    return qa_report


def run(episode, source_url=None, fixture=False, stop_after=None, alignment_recovery=False):
    if fixture:
        return run_fixture(episode, stop_after=stop_after)
    root, name, dirs = _paths(episode)
    state_path = dirs["work"] / "state.json"
    state = load(state_path, episode)
    state["mode"] = "strict"
    save(state_path, state)
    url_path = dirs["source"] / "source.url"
    url = _resolve_source_url(url_path, state, episode, source_url)
    config_dir, series, names, religious = _load_configs()
    resume_scope = None
    if os.getenv("MAS_CODE_FIX_RESUME"):
        from .retry_authorization import validate_code_fix_resume, retry_predecessor_paths
        resume_scope = validate_code_fix_resume(root, episode, os.environ["MAS_GIT_COMMIT"],
                                               permit_path=os.environ["MAS_CODE_FIX_RESUME"])
        priority = os.getenv('MAS_PRODUCTION_PRIORITY', 'whole-episode-v1')
        if (priority == 'first-hour-v1') != ('part_id' in resume_scope):
            raise RuntimeError("Scoped retry authority does not match the production priority")
        for relative in retry_predecessor_paths(episode, resume_scope, root):
            if not (root / relative).is_file():
                raise RuntimeError(f"Scoped alignment retry lacks required checkpoint: {relative}")
    if alignment_recovery:
        if resume_scope is None:
            raise RuntimeError("alignment recovery requires a validated code-fix resume scope")
        required = (
            dirs["source"] / "download.done.json",
            dirs["prepare"] / "audio.done.json",
            dirs["prepare"] / "raw_asr_v2.done.json",
            dirs["translation_input"] / f"{name}_TR_CORRECTION_PACK.zip",
            dirs["translation_output"] / f"{name}_TR_TEXT_CORRECTED.zip",
            dirs["translation_output"] / f"{name}_TR_CORRECTED.zip",
            dirs["prepare"] / "audio_review_v2.json",
        )
        if any(not path.is_file() for path in required):
            raise RuntimeError("alignment recovery requires retained remote checkpoints; no source download is allowed")
        if not validate_download(dirs["source"] / "download.done.json", url=url, allowed_root=dirs["source"]):
            raise RuntimeError("alignment recovery source checkpoint is invalid; no source download is allowed")
    encoder = os.getenv("MAS_MP4_ENCODER", "h264_nvenc")
    target = float(os.getenv("MAS_MP4_TARGET_GB", "3"))
    encoder_options = json.loads(os.environ["MAS_MP4_ENCODER_OPTIONS"]) if os.getenv("MAS_MP4_ENCODER_OPTIONS") else None
    external_delivery = _requires_external_delivery()
    execution_plan = os.getenv("MAS_DELIVERY_EXECUTION_PLAN", "remote-nvenc-v1")
    if execution_plan not in {"remote-nvenc-v1", "local-qsv-v1"}:
        raise RuntimeError("unsupported delivery execution plan")
    id_output = dirs["translation_output"] / f"{name}_ID_TRANSLATED.zip"
    sample_approval = dirs["review"] / "mp4-sample-approval.json"
    if (os.getenv("MAS_EXTERNAL_RUNPOD_CONTROLLER") == "1" and execution_plan == "remote-nvenc-v1"
            and encoder != "h264_nvenc"):
        raise RuntimeError("RunPod MP4 production requires the explicitly supported NVENC encoder")
    holder = {}
    _guard_existing_source(state, dirs["source"])

    def acquire():
        result = download_source(url, dirs["source"], output_stem=name, retries=3,
                                 attempts=3, socket_timeout=30,
                                 cookies_file=os.getenv("MAS_YTDLP_COOKIES"),
                                 expected_source_sha256=state.get("source_sha256"),
                                 freeze_captions=(dirs["prepare"] / "primary_asr").is_dir() or any(
                                     (dirs["prepare"] / filename).is_file()
                                     for filename in ("raw_asr_v2.json", "raw_asr_v2.recovery.json")
                                 ))
        holder["download"] = result
        digest = _source_guard(state, result.video_path)
        state["source_path"] = str(Path(result.video_path).resolve())
        save(state_path, state)
        if os.getenv("MAS_EXTERNAL_RUNPOD_CONTROLLER") == "1" and execution_plan == "remote-nvenc-v1":
            pod_id = os.environ["RUNPOD_POD_ID"]
            if not pod_id.isalnum():
                raise RuntimeError("invalid temporary Pod identity")
            qualification = qualify_encoding(result.video_path, dirs["work"] / "encoder-qualification" / pod_id,
                encoder=encoder, target_size_gb=target, encoder_options=encoder_options)
            measured = json.loads(qualification.read_text(encoding="utf-8"))
            if (
                id_output.is_file()
                and sample_approval.is_file()
                and measured["projected_full_encode_seconds"]
                >= float(os.getenv("MAS_MAX_RUNTIME_SECONDS", "14400"))
            ):
                raise RuntimeError("measured MP4 encoding alone exceeds the paid runtime allowance")
        return {"source": str(result.video_path), "sha256": digest, "resumed": result.resumed}
    _stage(state_path, state, "download", acquire)
    download = holder["download"]
    if stop_after == 1:
        return 75

    def prepare_audio():
        _source_guard(state, download.video_path)
        result = extract_audio(download.video_path, dirs["prepare"], source_sha256=state["source_sha256"])
        holder["audio"] = result
        return {"path": str(result.audio_path), "sha256": sha256_file(result.audio_path), "resumed": result.resumed}
    _stage(state_path, state, "audio", prepare_audio)
    audio = holder["audio"]
    if stop_after == 2:
        return 75

    priority = os.getenv('MAS_PRODUCTION_PRIORITY', 'whole-episode-v1')
    if priority not in {'whole-episode-v1', 'first-hour-v1'}:
        raise RuntimeError('unsupported production priority')
    if priority == 'first-hour-v1':
        if not external_delivery or execution_plan != 'local-qsv-v1':
            raise RuntimeError('first-hour production requires externally released local QSV delivery')
        from .progressive import run_progressive_worker
        from .reliability import RunBudget, digest
        saved_budget = _json(root / 'work/controller_budget.json')
        budget = saved_budget.get('data')
        if (not isinstance(budget, dict) or saved_budget.get('sha256') != digest(budget)
                or budget.get('episode') != episode or not 0 < budget.get('limit_seconds', 0) <= 21600):
            raise RuntimeError('first-hour production requires the original bounded episode clock')
        remaining = RunBudget(budget['started_at'], limit_seconds=budget['limit_seconds']).check()
        return run_progressive_worker(root, episode, download.video_path, audio.audio_path, download.captions_path,
                                      total_timeout=remaining, config_dir=config_dir, series=series,
                                      names=names, religious=religious, resume_scope=resume_scope)

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
            **({"require_resume": True} if resume_scope is not None else {}),
        )
        settings = data.get("model", {}).get("settings", {})
        if data.get("independent_vad") is not True or settings.get("allow_cpu_fallback") is not False:
            raise RuntimeError("strict raw ASR requires independent VAD and GPU-only policy")
        holder["raw"] = data
        path = dirs["prepare"] / "raw_asr_v2.json"
        return {"path": str(path), "sha256": sha256_file(path), "resumed": data.get("resumed", False)}
    _stage(state_path, state, "raw_asr", transcribe)
    raw = holder["raw"]
    if stop_after == 3:
        return 75

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
        notify(
            episode,
            "Türkçe düzeltme bekleniyor",
            f"Sonuç: Türkçe düzeltme paketi hazır.\n"
            f"Beklenen dönüş: {tr_text}\n"
            f"Sonraki adım: düzeltilmiş ZIP'i bu konuma koyup ./mas run {episode} komutunu yeniden çalıştırın.",
            root=root, kind="action",
        )
        print(f"[WAIT] TR CORRECTION\nPack: {tr_pack}\nExpected: {tr_text}\nSafe to stop RunPod.")
        return WAIT_TR
    def validate_tr_return():
        result = validate_tr_correction_output(tr_pack, tr_text)
        return {"sha256": sha256_file(tr_text), "records": len(result.records)}
    _stage(state_path, state, "tr_return", validate_tr_return)

    tr_output = dirs["translation_output"] / f"{name}_TR_CORRECTED.zip"
    review_path = dirs["prepare"] / "audio_review_v2.json"
    _apply_audio_review_reset(root, episode, name, tr_pack, tr_text)
    overrides_path = dirs["review"] / "audio_review_overrides.json"
    speaker_evidence_path = dirs["review"] / "speaker_evidence_v1.json"
    overrides = _json(overrides_path) if overrides_path.is_file() else None
    speaker_evidence = (
        _json(speaker_evidence_path) if speaker_evidence_path.is_file() else None
    )
    def review_audio():
        resumed = tr_output.is_file() and review_path.is_file()
        report = resolve_tr_audio_reviews(
            tr_pack, tr_text, tr_output, review_path,
            dirs["prepare"] / "audio_review_v2.recovery.json",
            config=AudioReviewConfig(model_name=series["whisper_model"], device="cuda", allow_cpu_fallback=False),
            manual_overrides=overrides,
            **({"require_resume": True} if resume_scope is not None else {}))
        holder["correction"] = validate_tr_correction_output(tr_pack, tr_output)
        holder["review"] = report
        return {"path": str(review_path), "sha256": sha256_file(review_path),
                "review_count": report["review_count"], "resumed": resumed}
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
            speaker_evidence=speaker_evidence,
            episode=episode,
            audio_sha256=raw["audio_sha256"],
        )
        _source_guard(state, download.video_path)
        result, resumed = _aligned_checkpoint(
            alignment_path, audio.audio_path, bundle.alignment_inputs, raw["vad_regions"],
            correction.output_sha256,
            **({"resume_scope": resume_scope} if resume_scope is not None else {}),
        )
        holder["aligned"] = result
        return {"path": str(alignment_path), "sha256": sha256_file(alignment_path), "device": "cuda", "resumed": resumed}
    _stage(state_path, state, "forced_alignment", align)
    aligned = holder["aligned"]
    if alignment_recovery:
        return ALIGNMENT_RECOVERY_COMPLETE

    schema_path = dirs["prepare"] / "aligned_tr_schema_v2.json"
    id_pack = dirs["translation_input"] / f"{name}_ID_TRANSLATION_PACK.zip"
    def make_id_pack():
        artifacts = build_strict_artifacts(
            raw,
            correction.records,
            aligned,
            episode=episode,
            acoustic_audio_review=holder["review"],
            speaker_evidence=speaker_evidence,
            production_policy=build_production_translation_policy(series, names, religious),
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
        if 'production_policy' in artifacts.schema:
            prepare_id_translation_workspaces(id_pack, dirs['translation_input'] / 'id-workers')
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

    if not id_output.is_file():
        set_stage(state_path, state, "id_return", "blocked", expected=str(id_output))
        notify(
            episode,
            "Endonezce çeviri bekleniyor",
            f"Sonuç: Endonezce çeviri paketi hazır.\n"
            f"Beklenen dönüş: {id_output}\n"
            f"Sonraki adım: çevrilmiş ZIP'i bu konuma koyup ./mas run {episode} komutunu yeniden çalıştırın.",
            root=root, kind="action",
        )
        print(f"[WAIT] ID TRANSLATION\nPack: {id_pack}\nExpected: {id_output}\nSafe to stop RunPod.")
        return WAIT_ID
    def validate_id_return():
        if 'production_policy' in artifacts.schema:
            validate_id_workspace_output(id_pack, id_output)
        result = load_and_validate_id_translation_zip(
            artifacts.schema,
            id_output,
            input_manifest=manifest,
        )
        _validate_id_quality(artifacts, result, series, names, religious)
        return {"sha256": sha256_file(id_output), "records": result.output_block_count}
    _stage(state_path, state, "id_return", validate_id_return)

    report_path = dirs["final"] / f"{name}_FINALIZATION_REPORT_V2.json"
    strict_marker_path = dirs["final"] / "strict_finalize.done.json"
    strict_outputs = {
        "report": report_path,
        "mkv": dirs["final"] / f"{name}.mkv",
        "id_srt": dirs["final"] / "subtitles" / f"{name}-id.srt",
        "tr_srt": dirs["final"] / "subtitles" / f"{name}-tr.srt",
    }
    strict_inputs = {
        "source_video": download.video_path,
        "download_metadata": dirs["source"] / "source.metadata.json",
        "download_marker": dirs["source"] / "download.done.json",
        "audio": audio.audio_path,
        "audio_metadata": dirs["prepare"] / "audio.metadata.json",
        "audio_marker": dirs["prepare"] / "audio.done.json",
        "raw_asr": dirs["prepare"] / "raw_asr_v2.json",
        "raw_asr_marker": dirs["prepare"] / "raw_asr_v2.done.json",
        "forced_alignment": alignment_path,
        "forced_alignment_marker": dirs["prepare"] / "forced_alignment_v2.done.json",
        "tr_correction_pack": tr_pack,
        "tr_text_correction_output": tr_text,
        "tr_correction_output": tr_output,
        "audio_review": review_path,
        "aligned_schema": schema_path,
        "id_translation_pack": id_pack,
        "id_translation_output": id_output,
        "series_config": config_dir / "series.yaml",
        "names_config": config_dir / "names.yaml",
        "religious_config": config_dir / "religious_terms.yaml",
    }
    if 'production_policy' in artifacts.schema:
        strict_inputs['id_workspace_receipt'] = Path(str(id_output) + '.workspace.json')
    def strict_finalize():
        input_sha256 = _strict_finalize_input_sha256(episode, strict_inputs)
        report = _load_strict_finalize_checkpoint(
            strict_marker_path,
            report_path,
            strict_outputs,
            root,
            episode,
            input_sha256,
        )
        resumed = report is not None
        if report is None:
            report = finalize_episode(
                episode_root=root, episode=episode, source_video=download.video_path,
                raw_asr_path=dirs["prepare"] / "raw_asr_v2.json",
                forced_alignment_path=alignment_path,
                tr_correction_pack=tr_pack, tr_text_correction_output=tr_text,
                tr_correction_output=tr_output, audio_review_path=review_path,
                aligned_schema=schema_path, id_translation_pack=id_pack,
                id_translation_zip=id_output,
                series_config=config_dir / "series.yaml",
                names_config=config_dir / "names.yaml",
                religious_config=config_dir / "religious_terms.yaml",
            )
            if report.get("status") != "PASS":
                raise RuntimeError("strict finalization did not PASS")
            if _strict_finalize_input_sha256(episode, strict_inputs) != input_sha256:
                raise RuntimeError("strict finalization inputs changed before checkpoint")
            write_stage_marker(
                strict_marker_path,
                stage="strict_finalize_v2",
                input_sha256=input_sha256,
                outputs=strict_outputs,
            )
        holder["final_report"] = report
        return {"report": str(report_path), "sha256": sha256_file(report_path),
                "resumed": resumed}
    _stage(state_path, state, "strict_finalize", strict_finalize)
    report = holder["final_report"]

    if external_delivery and execution_plan == "local-qsv-v1":
        from .local_encode import write_subtitle_export
        write_subtitle_export(root, episode, target_size_gb=target)
        set_stage(state_path, state, "burn_mp4", "blocked",
                  reason="verified subtitle collection and external GPU release precede local QSV")
        return READY_FOR_LOCAL_ENCODE

    def burn_mp4():
        id_srt = root / report["outputs"]["id_srt"]["relative_path"]
        settings, _ = plan_encoding_settings(download.video_path, encoder=encoder, target_size_gb=target,
                                             encoder_options=encoder_options)
        identity = sha256_json({"source": sha256_file(download.video_path),
                                "id_srt": sha256_file(id_srt),
                                "style": SUBTITLE_STYLE, "settings": settings["identity_sha256"]})[:12]
        mp4 = dirs["final"] / f"{name}.id.{identity}.mp4"
        if os.getenv("MAS_EXTERNAL_RUNPOD_CONTROLLER") == "1" and not sample_approval.is_file():
            samples = dirs["work"] / "encoding-samples" / identity
            manifest_path = create_encoding_samples(download.video_path, id_srt, samples,
                                                     encoder=encoder, target_size_gb=target,
                                                     encoder_options=encoder_options)
            files = [file_record(path, root) for path in sorted(samples.iterdir()) if path.is_file()]
            atomic_write_json(dirs["work"] / "sample-export.json",
                              {"episode": episode, "mode": "review", "files": files})
            holder["sample_wait"] = True
            return {"report": str(report_path), "samples": str(manifest_path), "sample_review_required": True}
        network = os.getenv("MAS_NETWORK_VOLUME_QUOTA_BYTES")
        storage = {}
        if network:
            storage = inspect_encoding_storage(dirs["final"], network_volume_root="/workspace",
                                               network_volume_quota_bytes=int(network))
            atomic_write_json(dirs["work"] / "encoding-storage.json", storage)
            # Allow room beyond the target; size is never an output acceptance ceiling.
            if storage["network_volume_free_bytes"] < target * 1_000_000_000 * 2:
                mp4 = Path(f"/tmp/mas-ep{episode}-output") / mp4.name
        burn_indonesian_mp4(download.video_path, id_srt, mp4, encoder=encoder, target_size_gb=target,
                            encoder_options=encoder_options,
                            sample_approval_path=sample_approval if sample_approval.is_file() else None,
                            require_sample_approval=os.getenv("MAS_EXTERNAL_RUNPOD_CONTROLLER") == "1",
                            network_volume_root="/workspace" if network and mp4.is_relative_to(root) else None,
                            network_volume_quota_bytes=int(network) if network else None)
        if mp4.is_relative_to(root):
            mp4_record = file_record(mp4, root)
        else:
            mp4_record = {"relative_path": f"final/{mp4.name}", "storage_path": str(mp4),
                          "size_bytes": mp4.stat().st_size, "sha256": sha256_file(mp4)}
        delivery = {"format": "mas-burned-mp4-delivery-1", "mode": "strict",
                    "strict_finalization_sha256": sha256_file(report_path),
                    "encoding_receipt_sha256": sha256_file(mp4.with_suffix('.burn.json')),
                    "outputs": {"mp4": mp4_record}}
        delivery_path = dirs["final"] / "burned_mp4_delivery.json"
        atomic_write_json(delivery_path, delivery)
        holder["delivery"] = delivery
        return {"delivery": str(delivery_path), "delivery_sha256": sha256_file(delivery_path)}
    _stage(state_path, state, "burn_mp4", burn_mp4)

    if holder.get("sample_wait"):
        set_stage(state_path, state, "burn_mp4", "blocked", reason="MP4 samples require review",
                  approval_path=str(dirs["review"] / "mp4-sample-approval.json"))
        notify(episode, "MP4 örnekleri inceleme bekliyor", "Örnekleri inceleyip kaynak/ayar bağlı onay kaydını tamamlayın.")
        return WAIT_MP4_SAMPLE

    if external_delivery:
        write_delivery_export(root, episode)
        set_stage(state_path, state, "drive_readback", "blocked",
                  reason="external controller must publish after verified GPU shutdown")
        return READY_FOR_DELIVERY

    remote_root = os.getenv("MAS_DRIVE_STRICT_REMOTE")
    if not remote_root:
        set_stage(state_path, state, "drive_readback", "blocked", error="MAS_DRIVE_STRICT_REMOTE is not set")
        raise RuntimeError("strict output is local only; Drive byte/SHA-256 readback is mandatory")
    receipt_path = dirs["final"] / "drive_readback_receipt.json"
    def publish():
        receipts = []
        for key in STRICT_DRIVE_OUTPUTS:
            record = holder["delivery"]["outputs"][key]
            local = root / record["relative_path"]
            if local.stat().st_size != record["size_bytes"] or sha256_file(local) != record["sha256"]:
                raise RuntimeError("burned MP4 changed after final verification")
            receipts.append(upload_verified(local, f"{remote_root.rstrip('/')}/{name}/{local.name}"))
        atomic_write_json(receipt_path, {"status": "PASS", "mode": "strict", "files": receipts})
        return {"receipt": str(receipt_path), "sha256": sha256_file(receipt_path)}
    _stage(state_path, state, "drive_readback", publish)
    notify(
        episode,
        "bölüm hazır",
        f"Sonuç: strict final dosyalar Drive'a yüklendi ve byte/SHA-256 readback doğrulandı.\n"
        f"Makbuz: {receipt_path}\n"
        "Sonraki adım: teslimat makbuzunu arşivleyin.",
    )
    set_stage(state_path, state, "compute_shutdown", "running")
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
    root, _, dirs = _paths(episode)
    result = load(dirs["work"] / "state.json", episode)
    if (dirs["work"] / "part-plan.json").is_file():
        from .engine.part_audio import load_part_plan
        from .partial_delivery import _read_bound

        plan = load_part_plan(root, episode, verify_files=False)
        plan_sha = sha256_file(dirs["work"] / "part-plan.json")
        progress = {"mode": "first-hour-v1", "current_part": None, "recorded_only": True,
                    "live_verification": False, "parts": []}
        pointer = None
        try:
            if (dirs["work"] / "current-part.json").is_file():
                pointer = _read_bound(dirs["work"] / "current-part.json")
                part_id = pointer.get("part_id")
                if (pointer.get("format") != "mas-current-part-1" or pointer.get("episode") != episode
                        or part_id not in {part["part_id"] for part in plan["parts"]}
                        or pointer.get("part_plan_sha256") != plan_sha
                        or pointer.get("state") != file_record(root / "parts" / part_id / "work/state.json", root)):
                    raise ValueError("current part metadata binding changed")
                progress["current_part"] = part_id
        except (OSError, ValueError, KeyError, TypeError):
            pointer = None
            progress["current_part_metadata"] = "INVALID_LOCAL_METADATA"
        for part in plan["parts"]:
            part_id = part["part_id"]
            child = root / "parts" / part_id
            row = {"part_id": part_id, "source_start_ms": part["start_ms"], "source_end_ms": part["end_ms"],
                   "stage": None}
            try:
                if (child / "work/state.json").is_file():
                    state = load(child / "work/state.json", episode)
                    if state.get("part_id") != part_id or state.get("part_plan_sha256") != plan_sha:
                        raise ValueError("child state scope changed")
                    stages = state.get("stages", {})
                    if stages:
                        stage = max(stages, key=lambda key: stages[key].get("updated_at", ""))
                        if pointer is not None and pointer["part_id"] == part_id:
                            stage = pointer["stage"]
                        row["stage"] = {"name": stage, **stages[stage]}
            except (OSError, ValueError, KeyError, TypeError):
                row["stage"] = {"metadata_status": "INVALID_LOCAL_METADATA"}
            for label, relative, format_name in (
                ("handoff", "work/partial-handoff.json", "mas-partial-handoff-1"),
                ("delivery", "final/drive_readback_receipt.json", "mas-part-delivery-1"),
            ):
                path = child / relative
                row[label] = {"present": path.is_file(), "stored_status": None}
                if not path.is_file():
                    continue
                try:
                    data = _read_bound(path)
                    if (data.get("format") != format_name or data.get("episode") != episode
                            or data.get("part_id") != part_id or data.get("plan_sha256") != plan_sha):
                        raise ValueError("stored metadata scope changed")
                    row[label].update(stored_status=data.get("status"), kind=data.get("kind"),
                                      metadata_status="STORED_METADATA_BOUND", path=path.relative_to(root).as_posix())
                except (OSError, ValueError, KeyError, TypeError):
                    row[label]["metadata_status"] = "INVALID_LOCAL_METADATA"
            if episode >= 14:
                from .delivery_first import read_signed, quality_summary, _tag, EXPORT_MODE
                import hmac
                export_path = child / "work/partial-export.json"
                if export_path.is_file():
                    try:
                        export = _json(export_path)
                        body = dict(export)
                        tag = body.pop('auth_tag', None)
                        if (body.get('mode') != EXPORT_MODE or body.get('episode') != episode
                                or body.get('part_id') != part_id or body.get('plan_sha256') != plan_sha
                                or not isinstance(tag, str) or not hmac.compare_digest(tag, _tag(body, 'export'))):
                            raise ValueError('delivery metadata authentication failed')
                        from .delivery import verified_record
                        report = _json(verified_record(root, export['report']))
                        row['quality'] = {**quality_summary(report),
                            'metadata_status': 'SIGNED_STORED_REPORT', 'artifact_revalidation': False}
                    except Exception:
                        row['quality'] = {'metadata_status': 'UNVERIFIED_LOCAL_METADATA'}
                tail_path = child / 'final/local-tail-ready.json'
                if tail_path.is_file():
                    try:
                        tail = read_signed(tail_path, 'local-tail-ready')
                        if (tail.get('episode') != episode or tail.get('part_id') != part_id
                                or tail.get('plan_sha256') != plan_sha
                                or tail.get('status') != 'LOCAL_ENCODED_NOT_PUBLISHED'):
                            raise ValueError('tail metadata scope changed')
                        row['local_tail'] = {'stored_status': tail['status'],
                            'artifact_revalidation': False, 'remote_publication': False}
                    except Exception:
                        row['local_tail'] = {'metadata_status': 'UNVERIFIED_LOCAL_METADATA'}
            progress["parts"].append(row)
        if episode >= 14:
            full_path = root / 'final/delivery-first/full-drive-receipt.json'
            progress['full_delivery'] = {'present': full_path.is_file(), 'live_verification': False}
            if full_path.is_file():
                try:
                    from .delivery_first import read_signed
                    full = read_signed(full_path, 'full-delivery')
                    if (full['identity']['episode'] != episode or full['identity']['plan_sha256'] != plan_sha
                            or full['status'] != 'DELIVERED_WITH_WARNINGS'
                            or full['quality_status'] != 'NOT_STRICT'
                            or full['remote']['bytes'] != full['mp4']['size_bytes']
                            or full['remote']['sha256'] != full['mp4']['sha256']):
                        raise ValueError('full delivery metadata scope changed')
                    progress['full_delivery'].update(stored_status=full['status'],
                        quality_status=full['quality_status'], remote=full['remote']['remote'],
                        bytes=full['remote']['bytes'], sha256=full['remote']['sha256'],
                        metadata_status='SIGNED_STORED_RECEIPT', artifact_revalidation=False)
                except Exception:
                    progress['full_delivery']['metadata_status'] = 'UNVERIFIED_LOCAL_METADATA'
        result["progressive"] = progress
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
