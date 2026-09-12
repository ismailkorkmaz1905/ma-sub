import json
from pathlib import Path
import re
import unicodedata

from .config import episode_dir
from .engine.download import atomic_write_bytes, sha256_file
from .engine.srt import SubtitleEntry, render_srt, visible_length, wrap_text
from .engine.tr_correction import validate_tr_correction_output
from .engine.translation_validation import DEFAULT_KNOWN_NAMES, _contains_name_form


FORMAT = "mas-emergency-segment-timing-1"
EPISODE_13_CORRECTED_SHA256 = (
    "7a9cfd922b5c7db13bcd1d1002bcdba7eda5ede08c79cee3db9f5135f819c8e1"
)
WORK_FIELDS = {
    "utterance_uid",
    "original_coarse_start_ms",
    "original_coarse_end_ms",
    "start_ms",
    "end_ms",
    "tr_corrected",
    "id_translation",
}
MINIMUM_DURATION_MS = 700
MAX_CPS = 20.0
METADATA_PREFIXES = ("altyazı",)
RELIGIOUS_RULES = (
    (("allah askina",), "demi allah"),
    (("allah allah", "allahim"), "ya allah"),
    (("ya rabbim",), "ya rabb"),
    (("insallah",), "insyaallah"),
    (("masallah",), "masyaallah"),
    (("tovbe estagfurullah",), "tobat, astagfirullah"),
    (("estagfurullah",), "astagfirullah"),
    (("la havle vela kuvvete illa billah",), "la hawla wala quwwata illa billah"),
)


class EmergencySegmentError(ValueError):
    pass


def _jsonl_bytes(records):
    return (
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        )
    ).encode("utf-8")


def _read_jsonl(path):
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EmergencySegmentError(f"Cannot read JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EmergencySegmentError(f"Blank JSONL line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EmergencySegmentError(f"Invalid JSONL at line {line_number}") from exc
        if not isinstance(record, dict):
            raise EmergencySegmentError(f"JSONL line {line_number} is not an object")
        records.append(record)
    if not records:
        raise EmergencySegmentError("JSONL is empty")
    return records


def _write_preserving(path, payload):
    destination = Path(path)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise EmergencySegmentError(
                f"Existing emergency artifact differs; preserve it: {destination}"
            )
        return destination
    return atomic_write_bytes(destination, payload)


def _artifact(path, root):
    target = Path(path)
    return {
        "path": target.relative_to(root).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _layout(text):
    if not isinstance(text, str) or not text.strip() or text != text.strip():
        raise EmergencySegmentError("Subtitle text must be nonempty without outer whitespace")
    try:
        laid_out = wrap_text(text, target=42, max_lines=2, hard_limit=84)
    except ValueError as exc:
        raise EmergencySegmentError(str(exc)) from exc
    characters = visible_length(laid_out.replace("\n", ""))
    required_ms = max(MINIMUM_DURATION_MS, (characters * 1000 + 19) // 20)
    return laid_out, required_ms, characters


def _fold(text):
    value = str(text).casefold().replace("ı", "i").replace("'", "").replace("’", "")
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )


def _translation_quality_issues(source, target):
    source_folded = _fold(source)
    target_folded = _fold(target)
    issues = []
    source_numbers = re.findall(r"\d+(?:[.,:]\d+)*", source)
    target_numbers = re.findall(r"\d+(?:[.,:]\d+)*", target)
    if source_numbers != target_numbers:
        issues.append(f"numbers changed: {source_numbers} -> {target_numbers}")
    for name in DEFAULT_KNOWN_NAMES:
        if _contains_name_form(source, name) and _fold(name) not in target_folded:
            issues.append(f"canonical name missing: {name}")
    for source_forms, required in RELIGIOUS_RULES:
        if any(form in source_folded for form in source_forms) and required not in target_folded:
            issues.append(f"religious expression must contain: {required}")
    if re.search(r"(?<![a-z])allah(?![a-z])", source_folded) and "allah" not in target_folded:
        issues.append("Allah was not preserved")
    return issues


def _schedule(records):
    scheduled = []
    adjustments = []
    previous_end = 0
    for position, record in enumerate(records, start=1):
        uid = str(record["utterance_uid"])
        original_start = int(record["coarse_start_ms"])
        original_end = int(record["coarse_end_ms"])
        if original_start < 0 or original_end <= original_start:
            raise EmergencySegmentError(f"Invalid coarse timing for {uid}")
        text = str(record["tr_corrected"])
        laid_out, required_ms, _ = _layout(text)
        start_ms = max(original_start, previous_end)
        end_ms = max(original_end, start_ms + required_ms)
        reasons = []
        if start_ms != original_start:
            reasons.append("overlap_shift")
        if end_ms != original_end:
            reasons.append("readability_extension")
        item = {
            "utterance_uid": uid,
            "original_coarse_start_ms": original_start,
            "original_coarse_end_ms": original_end,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "tr_corrected": text,
            "id_translation": "",
        }
        scheduled.append((item, SubtitleEntry(position, start_ms, end_ms, laid_out)))
        if reasons:
            adjustments.append(
                {
                    "utterance_uid": uid,
                    "original_coarse_start_ms": original_start,
                    "original_coarse_end_ms": original_end,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_delta_ms": start_ms - original_start,
                    "end_delta_ms": end_ms - original_end,
                    "reasons": reasons,
                }
            )
        previous_end = end_ms
    return scheduled, adjustments


def _default_corrected(root, episode):
    if episode != 13:
        raise EmergencySegmentError("--corrected-zip is required outside Episode 13")
    return (
        root
        / "work"
        / "remote-checkpoints"
        / EPISODE_13_CORRECTED_SHA256
        / f"Muhtemel Ask {episode}.Bolum_TR_CORRECTED.zip"
    )


def prepare_emergency_segment(episode, corrected_zip=None, *, root=None):
    episode_root = Path(root) if root is not None else episode_dir(episode)
    corrected = (
        Path(corrected_zip) if corrected_zip is not None else _default_corrected(episode_root, episode)
    )
    pack = (
        episode_root
        / "translation_input"
        / f"Muhtemel Ask {episode}.Bolum_TR_CORRECTION_PACK.zip"
    )
    validated = validate_tr_correction_output(pack, corrected)
    corrected_sha256 = sha256_file(corrected)
    if episode == 13 and corrected_sha256 != EPISODE_13_CORRECTED_SHA256:
        raise EmergencySegmentError("Episode 13 corrected ZIP is not the authorized checkpoint")
    dialogue_candidates = [
        record
        for record in validated.records
        if record["non_dialogue"] is False and str(record["tr_corrected"]).strip()
    ]
    quarantined = [
        {
            "utterance_uid": record["utterance_uid"],
            "utterance_index": record["utterance_index"],
            "coarse_start_ms": record["coarse_start_ms"],
            "coarse_end_ms": record["coarse_end_ms"],
            "tr_corrected": record["tr_corrected"],
            "reason": "subtitle-credit metadata",
        }
        for record in dialogue_candidates
        if str(record["tr_corrected"]).casefold().startswith(METADATA_PREFIXES)
    ]
    quarantined_uids = {record["utterance_uid"] for record in quarantined}
    dialogue = [
        record for record in dialogue_candidates if record["utterance_uid"] not in quarantined_uids
    ]
    scheduled, adjustments = _schedule(dialogue)
    work_records = [item for item, _ in scheduled]
    entries = [entry for _, entry in scheduled]

    output = episode_root / "emergency" / "segment-timing"
    work_path = output / "translation_work.jsonl"
    tr_srt_path = output / f"Muhtemel Ask {episode}.Bolum-tr.srt"
    audit_path = output / "timing_adjustments.jsonl"
    quarantine_path = output / "metadata_quarantine.jsonl"
    manifest_path = output / "manifest.json"
    _write_preserving(work_path, _jsonl_bytes(work_records))
    _write_preserving(tr_srt_path, render_srt(entries).encode("utf-8"))
    _write_preserving(audit_path, _jsonl_bytes(adjustments))
    _write_preserving(quarantine_path, _jsonl_bytes(quarantined))
    manifest = {
        "format": FORMAT,
        "mode": "emergency",
        "strict_eligible": False,
        "episode": episode,
        "corrected_zip": _artifact(corrected, episode_root),
        "correction_pack": _artifact(pack, episode_root),
        "correction_input_sha256": validated.input_sha256,
        "correction_output_sha256": validated.output_sha256,
        "timing_policy": {
            "source": "coarse_start_ms/coarse_end_ms",
            "overlap_resolution": "forward-shift-with-readability-extension-v1",
            "minimum_duration_ms": MINIMUM_DURATION_MS,
            "maximum_cps": MAX_CPS,
            "text_merged": False,
        },
        "cue_count": len(work_records),
        "adjustment_count": len(adjustments),
        "quarantine_count": len(quarantined),
        "artifacts": {
            "translation_work": _artifact(work_path, episode_root),
            "turkish_srt": _artifact(tr_srt_path, episode_root),
            "timing_adjustments": _artifact(audit_path, episode_root),
            "metadata_quarantine": _artifact(quarantine_path, episode_root),
        },
    }
    payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _write_preserving(manifest_path, payload)
    return manifest_path


def _load_manifest(episode_root):
    path = episode_root / "emergency" / "segment-timing" / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EmergencySegmentError("Emergency segment manifest is missing or invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != FORMAT
        or manifest.get("mode") != "emergency"
        or manifest.get("strict_eligible") is not False
    ):
        raise EmergencySegmentError("Emergency segment manifest identity is invalid")
    for artifact in (
        manifest["corrected_zip"],
        manifest["correction_pack"],
        *manifest["artifacts"].values(),
    ):
        candidate = (episode_root / artifact["path"]).resolve()
        if not candidate.is_relative_to(episode_root.resolve()):
            raise EmergencySegmentError("Emergency manifest path escapes the episode directory")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != artifact["bytes"]
            or sha256_file(candidate) != artifact["sha256"]
        ):
            raise EmergencySegmentError(f"Bound emergency input changed: {artifact['path']}")
    return manifest


def finalize_emergency_segment(episode, translations, *, root=None):
    episode_root = Path(root) if root is not None else episode_dir(episode)
    manifest = _load_manifest(episode_root)
    if manifest.get("episode") != episode:
        raise EmergencySegmentError("Emergency segment episode mismatch")
    work_path = episode_root / manifest["artifacts"]["translation_work"]["path"]
    expected = _read_jsonl(work_path)
    returned = _read_jsonl(translations)
    if len(returned) != len(expected):
        raise EmergencySegmentError("Indonesian return is not UID-complete")

    entries = []
    timing_adjustments = []
    previous_end = 0
    previous_bound_end = 0
    max_cps = 0.0
    quality_issues = []
    for position, (trusted, actual) in enumerate(zip(expected, returned), start=1):
        if set(actual) != WORK_FIELDS:
            raise EmergencySegmentError(f"Indonesian record {position} has wrong fields")
        for field in WORK_FIELDS - {"id_translation"}:
            if actual.get(field) != trusted.get(field):
                raise EmergencySegmentError(
                    f"Indonesian record {position} changed immutable field {field}"
                )
        text = actual["id_translation"]
        for issue in _translation_quality_issues(trusted["tr_corrected"], text):
            quality_issues.append(f"{trusted['utterance_uid']}: {issue}")
        laid_out, required_ms, characters = _layout(text)
        bound_start_ms = trusted["start_ms"]
        bound_end_ms = trusted["end_ms"]
        if bound_start_ms < previous_bound_end or bound_end_ms <= bound_start_ms:
            raise EmergencySegmentError("Bound emergency timing overlaps or is invalid")
        start_ms = max(bound_start_ms, previous_end)
        end_ms = max(bound_end_ms, start_ms + required_ms)
        if start_ms != bound_start_ms or end_ms != bound_end_ms:
            timing_adjustments.append(
                {
                    "utterance_uid": trusted["utterance_uid"],
                    "bound_start_ms": bound_start_ms,
                    "bound_end_ms": bound_end_ms,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_delta_ms": start_ms - bound_start_ms,
                    "end_delta_ms": end_ms - bound_end_ms,
                }
            )
        duration_ms = end_ms - start_ms
        max_cps = max(max_cps, characters * 1000 / duration_ms)
        entries.append(SubtitleEntry(position, start_ms, end_ms, laid_out))
        previous_end = end_ms
        previous_bound_end = bound_end_ms

    if quality_issues:
        raise EmergencySegmentError(
            "Indonesian translation quality checks failed: " + "; ".join(quality_issues[:20])
        )

    output = episode_root / "emergency" / "segment-timing"
    id_srt_path = output / f"Muhtemel Ask {episode}.Bolum-id.srt"
    id_timing_path = output / "indonesian_timing_adjustments.jsonl"
    _write_preserving(id_srt_path, render_srt(entries).encode("utf-8"))
    _write_preserving(id_timing_path, _jsonl_bytes(timing_adjustments))
    receipt_path = output / "subtitle_checksum_receipt.json"
    receipt = {
        "format": "mas-emergency-segment-subtitle-receipt-1",
        "status": "EMERGENCY_SUBTITLES_READY",
        "mode": "emergency",
        "strict_eligible": False,
        "episode": episode,
        "cue_count": len(entries),
        "timing_adjustment_count": len(timing_adjustments),
        "maximum_observed_cps": round(max_cps, 6),
        "manifest": _artifact(output / "manifest.json", episode_root),
        "translation_return": _artifact(Path(translations).resolve(), episode_root.resolve())
        if Path(translations).resolve().is_relative_to(episode_root.resolve())
        else {
            "path": str(Path(translations).resolve()),
            "bytes": Path(translations).stat().st_size,
            "sha256": sha256_file(translations),
        },
        "outputs": {
            "indonesian_srt": _artifact(id_srt_path, episode_root),
            "indonesian_timing_adjustments": _artifact(id_timing_path, episode_root),
            "turkish_srt": manifest["artifacts"]["turkish_srt"],
            "timing_adjustments": manifest["artifacts"]["timing_adjustments"],
        },
    }
    payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _write_preserving(receipt_path, payload)
    return receipt_path


def run_emergency_segment_command(args):
    if args.action == "prepare":
        result = prepare_emergency_segment(args.episode, args.corrected_zip)
    else:
        result = finalize_emergency_segment(args.episode, args.translations)
    print(result)
    return 0
