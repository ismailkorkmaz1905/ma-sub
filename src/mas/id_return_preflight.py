import copy
import json
import zipfile
from pathlib import Path

import yaml

from .engine.id_translation import (
    load_and_validate_id_translation_zip,
    load_default_id_translation_glossary,
    validate_id_translation_pack,
)
from .engine.subtitle_qa import assert_final_qa, run_subtitle_qa
from .engine.timing_qa import TimingQAV2Config, _visible_char_count
from .reliability import IntegrityError, atomic_json, digest, file_digest


def preflight_local_id_return(local_root, episode, config_dir):
    local_root, config_dir = Path(local_root), Path(config_dir)
    name = f"Muhtemel Ask {episode}.Bolum"
    pack = local_root / "handoff" / f"{name}_ID_TRANSLATION_PACK.zip"
    returned = local_root / "handoff" / f"{name}_ID_TRANSLATED.zip"
    if not returned.is_file():
        return None
    report_path = local_root / "work" / "id_return_local_preflight.json"
    before = {"return_sha256": file_digest(returned),
              "pack_sha256": file_digest(pack) if pack.is_file() else None}
    report = {"format": "mas-id-return-local-preflight-1", "episode": episode,
              **before, "status": "FAIL", "issues": []}
    try:
        if not pack.is_file():
            raise IntegrityError("local Indonesian return exists without its input pack")
        manifest = validate_id_translation_pack(
            pack, expected_glossary=load_default_id_translation_glossary())
        with zipfile.ZipFile(pack) as archive:
            schema = json.loads(archive.read("schema.json"))
        if schema.get("episode") != episode or manifest.get("episode") != episode:
            raise IntegrityError("local Indonesian pack episode mismatch")
        workspace_receipt = Path(str(returned) + ".workspace.json")
        if "production_policy" in schema:
            from .engine.translation_workspace import validate_id_workspace_output
            validate_id_workspace_output(pack, returned, receipt_path=workspace_receipt)
            before["workspace_sha256"] = file_digest(workspace_receipt)
            report["workspace_sha256"] = before["workspace_sha256"]
        validation = load_and_validate_id_translation_zip(
            schema, returned, input_manifest=manifest)
        records = validation.ordered_records(schema)
        review_uids = [item["block_uid"] for item in records
                       if item.get("review_required") is True]
        review_count = len(review_uids)
        with (config_dir / "series.yaml").open(encoding="utf-8") as stream:
            series = yaml.safe_load(stream)
        with (config_dir / "names.yaml").open(encoding="utf-8") as stream:
            names = yaml.safe_load(stream)
        with (config_dir / "religious_terms.yaml").open(encoding="utf-8") as stream:
            religious = yaml.safe_load(stream)
        subtitle = series["subtitle"]
        qa_blocks, qa_records = [], []
        for block, record in zip(schema["blocks"], records):
            qa_block = copy.deepcopy(block)
            qa_block["schema_sha256"] = schema["schema_sha256"]
            qa_block["primary_text"] = block["tr_text"]
            qa_blocks.append(qa_block)
            qa_record = copy.deepcopy(record)
            qa_record["schema_sha256"] = schema["schema_sha256"]
            qa_record["tr_final"] = block["tr_text"]
            qa_records.append(qa_record)
        semantic = run_subtitle_qa(
            qa_blocks, qa_records, names_config=names, religious_config=religious,
            preferred_max_cps=subtitle["preferred_max_cps"],
            line_limit=subtitle["qa_max_chars_per_line"])
        timing_config = TimingQAV2Config()
        cps_issues = []
        for block, record in zip(schema["blocks"], records):
            duration_ms = int(block["end_ms"]) - int(block["start_ms"])
            cps = _visible_char_count(record["id_final"]) / max(duration_ms / 1000.0, 0.001)
            if cps > timing_config.maximum_cps:
                cps_issues.append({"code": "high_cps_id", "block_uid": block["block_uid"],
                                   "actual": round(cps, 2),
                                   "expected": f"<= {timing_config.maximum_cps}"})
        report["validation"] = {"expected_block_count": validation.expected_block_count,
                                "output_block_count": validation.output_block_count,
                                "review_required_count": review_count,
                                "review_required_uids": review_uids}
        report["semantic_subtitle_qa"] = semantic
        report["id_timing_qa"] = {"maximum_cps": timing_config.maximum_cps,
                                  "high_cps_id_count": len(cps_issues),
                                  "issues": cps_issues,
                                  "passed": not cps_issues}
        if review_count:
            report["issues"].append(f"review_required_count={review_count}")
        if semantic.get("review_required_count") != 0:
            report["issues"].append("semantic subtitle QA still requires review")
        if cps_issues:
            report["issues"].append(f"high_cps_id_count={len(cps_issues)}")
        try:
            assert_final_qa(semantic)
        except Exception as exc:
            report["issues"].append(str(exc))
        after = {"return_sha256": file_digest(returned), "pack_sha256": file_digest(pack)}
        if "workspace_sha256" in before:
            after["workspace_sha256"] = file_digest(workspace_receipt)
        if after != before:
            raise IntegrityError("Indonesian pack or return changed during local preflight")
        if report["issues"]:
            raise IntegrityError("local Indonesian return requires review or failed subtitle QA")
        report["status"] = "PASS"
    except Exception as exc:
        report["issues"].append(f"{type(exc).__name__}: {exc}")
        unsigned = copy.deepcopy(report)
        report["sha256"] = digest(unsigned)
        atomic_json(report_path, report)
        raise
    unsigned = copy.deepcopy(report)
    report["sha256"] = digest(unsigned)
    atomic_json(report_path, report)
    return report
