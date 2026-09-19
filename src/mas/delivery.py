import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .hashing import sha256_file
from .reliability import atomic_json
from .remote import upload_verified
from .source_discovery import CHANNEL_VIDEOS_URL, is_exact_episode_title
from .state import load, set_stage
from .notify import enqueue_notification as notify


READY_FOR_DELIVERY = 22
WAIT_MP4_SAMPLE = 23
READY_FOR_LOCAL_ENCODE = 24
WAIT_PART_RETURN = 25
READY_FOR_PARTIAL_ENCODE = 26
NEXT_PART = 27
ALIGNMENT_RECOVERY_COMPLETE = 28


def safe_relative(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid artifact relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError("artifact path escapes episode")
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("artifact path escapes episode")
    return path


def verified_record(root, record):
    path = safe_relative(root, record["relative_path"])
    if (not path.is_file() or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]):
        raise ValueError("delivery artifact byte/SHA-256 mismatch")
    return path


def remote_storage_path(root, record, episode):
    canonical = safe_relative(root, record["relative_path"])
    storage = record.get("storage_path")
    if storage is None:
        return canonical
    expected = Path(f"/tmp/mas-ep{episode}-output") / canonical.name
    if storage != str(expected) or expected.is_symlink():
        raise ValueError("unexpected external MP4 storage path")
    return expected


def _sample_records(root, receipt):
    approval = receipt.get("sample_approval")
    if approval is None:
        return []
    records = [{"relative_path": "review/mp4-sample-approval.json", "size_bytes": approval["bytes"],
                "sha256": approval["sha256"]}]
    approval_file = verified_record(root, records[0])
    approved = json.loads(approval_file.read_text(encoding="utf-8"))
    relative = approved["sample_manifest_path"]
    if not relative.startswith("work/encoding-samples/"):
        raise ValueError("sample manifest is outside strict encoder samples")
    manifest_path = safe_relative(root, relative)
    records.append({"relative_path": relative, "size_bytes": manifest_path.stat().st_size,
                    "sha256": approved["sample_manifest_sha256"]})
    verified_record(root, records[-1])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["samples"]) != 3:
        raise ValueError("sample manifest must bind three samples")
    for sample in manifest["samples"]:
        for key in ("output", "subtitle"):
            filename = sample[key + "_file"]
            if PurePosixPath(filename).name != filename:
                raise ValueError("invalid sample filename")
            record = {"relative_path": str(PurePosixPath(relative).parent / filename),
                      "size_bytes": sample[key + "_bytes"], "sha256": sample[key + "_sha256"]}
            verified_record(root, record)
            records.append(record)
    return records


def validate_delivery(root, episode):
    root = Path(root)
    name = f"Muhtemel Ask {episode}.Bolum"
    delivery_path = root / "final" / "burned_mp4_delivery.json"
    delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
    report_path = root / "final" / f"{name}_FINALIZATION_REPORT_V2.json"
    if delivery.get("mode") != "strict" or sha256_file(report_path) != delivery["strict_finalization_sha256"]:
        raise ValueError("strict delivery report binding changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("episode") != episode:
        raise ValueError("strict finalization has not passed for this episode")
    mp4 = verified_record(root, delivery["outputs"]["mp4"])
    receipt_path = mp4.with_suffix(".burn.json")
    if sha256_file(receipt_path) != delivery["encoding_receipt_sha256"]:
        raise ValueError("MP4 encoding receipt binding changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "VERIFIED_ENCODING"
            or receipt.get("output_sha256") != delivery["outputs"]["mp4"]["sha256"]
            or receipt.get("output_bytes") != mp4.stat().st_size):
        raise ValueError("MP4 encoding evidence does not match output")
    for record in report["outputs"].values():
        verified_record(root, record)
    inputs = {key: verified_record(root, record) for key, record in report["input_files"].items()}
    source = inputs["source_video"]
    id_srt = verified_record(root, report["outputs"]["id_srt"])
    if receipt["inputs"] != {"source_sha256": sha256_file(source), "id_srt_sha256": sha256_file(id_srt)}:
        raise ValueError("MP4 source/subtitle binding changed")
    if ("execution_plan_sha256" in delivery or "qualification" in delivery
            or (receipt.get("encoder") == "h264_qsv" and not receipt.get("sample_approval"))):
        from .local_encode import validate_local_encoding_evidence
        validate_local_encoding_evidence(root, episode, delivery, receipt)
    _sample_records(root, receipt)
    return delivery, mp4


def write_delivery_export(root, episode):
    root = Path(root)
    report_path = root / "final" / f"Muhtemel Ask {episode}.Bolum_FINALIZATION_REPORT_V2.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    delivery_path = root / "final" / "burned_mp4_delivery.json"
    delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
    records = list(report["input_files"].values()) + list(report["outputs"].values())
    mp4_record = delivery["outputs"]["mp4"]
    records.append(mp4_record)
    receipt_relative = str(PurePosixPath(mp4_record["relative_path"]).with_suffix(".burn.json"))
    receipt_file = remote_storage_path(root, mp4_record, episode).with_suffix(".burn.json")
    receipt_record = {"relative_path": receipt_relative, "size_bytes": receipt_file.stat().st_size,
                      "sha256": sha256_file(receipt_file)}
    if "storage_path" in mp4_record:
        receipt_record["storage_path"] = str(receipt_file)
    records.append(receipt_record)
    records.extend(_sample_records(root, json.loads(receipt_file.read_text(encoding="utf-8"))))
    extras = [report_path, delivery_path,
              root / "source" / "official-source.json", root / "source" / "source.url"]
    for path in extras:
        records.append({"relative_path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    unique = {record["relative_path"]: record for record in records}
    for record in unique.values():
        actual = remote_storage_path(root, record, episode)
        if actual.stat().st_size != record["size_bytes"] or sha256_file(actual) != record["sha256"]:
            raise ValueError("export file identity changed")
    manifest = {"episode": episode, "mode": "strict", "files": list(unique.values())}
    atomic_json(root / "work" / "delivery-export.json", manifest)
    return manifest


def publish_local_delivery(root, episode, remote_root, *, total_timeout=3600):
    started = time.monotonic()
    deadline = started + total_timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Drive publication wall-time budget exhausted")
        return value

    root = Path(root)
    delivery, mp4 = validate_delivery(root, episode)
    metadata = json.loads((root / "source" / "official-source.json").read_text(encoding="utf-8"))
    source_url = (root / "source" / "source.url").read_text(encoding="utf-8").strip()
    title = metadata.get("title", "")
    if (metadata.get("episode") != episode or metadata.get("url") != source_url
            or metadata.get("channel_url") != CHANNEL_VIDEOS_URL
            or not is_exact_episode_title(title, episode)
            or re.search(r'[\\/:*?"<>|\x00-\x1f]', title)):
        raise ValueError("official episode title/source binding is invalid")
    state_path = root / "work" / "state.json"
    state = load(state_path, episode)
    set_stage(state_path, state, "drive_readback", "running")
    evidence = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING"}
    evidence["start_notification"] = notify(episode, "Drive aktarimi basladi", "MP4 byte/SHA-256 readback yapiliyor.")
    event_path = root / "work" / "drive-publication.json"
    atomic_json(event_path, evidence)
    try:
        receipt = upload_verified(mp4, f"{remote_root.rstrip('/')}/{title}.mp4",
                                  total_timeout=remaining(),
                                  preservation_receipt=root / "final" / "drive-preservation.json",
                                  require_drive_preflight=True)
        # Local inputs must remain unchanged across the complete network transaction.
        validate_delivery(root, episode)
        remaining()
        receipt_path = root / "final" / "drive_readback_receipt.json"
        atomic_json(receipt_path, {"status": "PASS", "mode": "strict", "files": [receipt],
                                  "delivery_sha256": sha256_file(root / "final" / "burned_mp4_delivery.json"),
                                  "perceptual_acceptance": "NOT_ASSERTED"})
        set_stage(state_path, state, "drive_readback", "pass", receipt=str(receipt_path),
                  sha256=sha256_file(receipt_path))
        evidence["status"] = "PASS"
        evidence["result_notification"] = notify(episode, "Drive aktarimi tamamlandi", str(receipt_path),
                                                 root=root, kind="terminal")
    except Exception as exc:
        set_stage(state_path, state, "drive_readback", "failed", error=str(exc))
        evidence.update(status="FAILED", error_type=type(exc).__name__)
        evidence["result_notification"] = notify(episode, "Drive aktarimi basarisiz", type(exc).__name__,
                                                 root=root, kind="terminal")
        raise
    finally:
        evidence.update(ended_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started)
        atomic_json(event_path, evidence)
    return 0
