from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .download import canonical_json_bytes
from .id_translation import (
    _jsonl_bytes,
    _open_checked_zip,
    _parse_json,
    _parse_jsonl,
    _pretty_json_bytes,
    _write_zip_atomic,
)
from .semantic_alignment import (
    ALIGNMENT_POLICY,
    SemanticAlignmentConfig,
    SemanticAlignmentError,
    _known_metadata_reason,
    _validate_resolution,
)


INPUT_FORMAT = "mas-semantic-alignment-pack-1"
RETURN_FORMAT = "mas-semantic-alignment-return-1"
INPUT_BATCH_PREFIX = "batch-"
RETURN_BATCH_PREFIX = "returned-batch-"

SEMANTIC_ALIGNMENT_INSTRUCTIONS = """# Semantic alignment production contract

You are the linguistic authority for Turkish subtitle text and block boundaries.
The supplied timing words are the physical speech order. Primary, verification,
timing and YouTube VTT candidates are evidence, not absolute truth.

For every window, return exactly one record in the original order. You may split
one window into multiple natural subtitle blocks or merge coarse candidates when
they are one same-speaker sentence. Each block must choose first_word_id and
last_word_id only from that window's owned_word_ids. Context words are read-only.
Blocks must be ordered, contiguous spans and non-overlapping. Never combine known
different speakers or cross a known scene boundary. A block must not cross a
speech gap over 1000 ms.

Never return start_ms, end_ms, timestamp, duration_ms or any other timestamp.
Python derives display timestamps from the immutable timing words. Never invent
spoken dialogue, add subtitle credits/metadata, or create speech over music-only
evidence. Preserve religious language without censoring it. If uncertain, set
review_required=true and explain briefly in note.

Use these canonical names exactly when the evidence refers to them: Defne,
Kadir, Tolga, Levent, Levent Bartıner, Bartıner, Mine, Melis, Özlem, Selim,
Selma, Sultan, Zeynep, Zeyno, Suzi, Leyla, Oğuz, Yavuz, Zeliha, Emindağ.
Emindağ is one word. Bartıner uses this spelling.

Each returned block may contain only first_word_id, last_word_id, final_tr,
confidence, review_required and note. final_tr must be natural corrected Turkish.
Copy the exact input identity fields into the return manifest. Do not omit,
duplicate or reorder windows.
"""


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _batches(records, *, target_count, maximum_bytes):
    if not 150 <= target_count <= 300:
        raise ValueError("target_count must be between 150 and 300")
    batches = []
    current = []
    current_size = 0
    for record in records:
        size = len(canonical_json_bytes(record)) + 1
        if current and (len(current) >= target_count or current_size + size > maximum_bytes):
            batches.append(current)
            current = []
            current_size = 0
        current.append(copy.deepcopy(record))
        current_size += size
    if current:
        batches.append(current)
    return batches


def create_semantic_alignment_pack(
    unresolved_windows,
    destination,
    *,
    episode,
    source_sha256,
    word_timeline_sha256,
    all_windows_sha256,
    deterministic_results_sha256,
    producer_sha256,
    config_sha256,
    target_batch_count=200,
    maximum_batch_bytes=1_500_000,
):
    windows = [copy.deepcopy(window) for window in unresolved_windows]
    if not windows:
        raise SemanticAlignmentError("A semantic handoff pack needs unresolved windows")
    window_ids = [window.get("window_id") for window in windows]
    if (
        any(not isinstance(window_id, str) or not window_id for window_id in window_ids)
        or len(window_ids) != len(set(window_ids))
    ):
        raise SemanticAlignmentError("Semantic handoff window identities are invalid")
    for window in windows:
        owned = set(window.get("owned_word_ids", []))
        context = {
            word.get("word_id")
            for side in ("previous_context", "next_context")
            for word in window.get(side, {}).get("read_only_context_words", [])
        }
        if not owned or len(owned) != len(window["owned_word_ids"]) or owned & context:
            raise SemanticAlignmentError("Semantic handoff ownership/context is invalid")
        candidate_texts = [
            candidate.get("text", "")
            for field in (
                "primary_candidates",
                "verification_candidates",
                "timing_candidates",
                "youtube_candidates",
            )
            for candidate in window.get(field, [])
        ]
        if any(_known_metadata_reason(text) for text in candidate_texts):
            raise SemanticAlignmentError("Known metadata reached the semantic handoff")

    batches = _batches(
        windows,
        target_count=target_batch_count,
        maximum_bytes=maximum_batch_bytes,
    )
    payloads = {
        "SEMANTIC_ALIGNMENT_INSTRUCTIONS.md": SEMANTIC_ALIGNMENT_INSTRUCTIONS.encode("utf-8"),
    }
    descriptors = []
    for number, batch in enumerate(batches, start=1):
        name = f"{INPUT_BATCH_PREFIX}{number:03d}.jsonl"
        payload = _jsonl_bytes(batch)
        payloads[name] = payload
        descriptors.append(
            {
                "input_file": name,
                "return_file": f"{RETURN_BATCH_PREFIX}{number:03d}.jsonl",
                "window_count": len(batch),
                "first_window_id": batch[0]["window_id"],
                "last_window_id": batch[-1]["window_id"],
                "sha256": _sha_bytes(payload),
            }
        )
    identity = {
        "format": INPUT_FORMAT,
        "episode": episode,
        "alignment_policy": ALIGNMENT_POLICY,
        "source_sha256": source_sha256,
        "word_timeline_sha256": word_timeline_sha256,
        "all_windows_sha256": all_windows_sha256,
        "unresolved_windows_sha256": hashlib.sha256(_jsonl_bytes(windows)).hexdigest(),
        "deterministic_results_sha256": deterministic_results_sha256,
        "producer_sha256": producer_sha256,
        "config_sha256": config_sha256,
        "window_count": len(windows),
        "window_ids": window_ids,
        "batch_count": len(batches),
        "batches": descriptors,
        "instructions_sha256": _sha_bytes(payloads["SEMANTIC_ALIGNMENT_INSTRUCTIONS.md"]),
    }
    manifest = {
        **identity,
        "input_identity_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        "file_sha256": {
            name: _sha_bytes(payload)
            for name, payload in sorted(payloads.items())
        },
    }
    ordered = {
        "manifest.json": _pretty_json_bytes(manifest),
        "SEMANTIC_ALIGNMENT_INSTRUCTIONS.md": payloads["SEMANTIC_ALIGNMENT_INSTRUCTIONS.md"],
    }
    for descriptor in descriptors:
        ordered[descriptor["input_file"]] = payloads[descriptor["input_file"]]
    _write_zip_atomic(Path(destination), ordered)
    validate_semantic_alignment_pack(destination)
    return manifest


def validate_semantic_alignment_pack(path):
    path = Path(path)
    with _open_checked_zip(path) as archive:
        names = archive.namelist()
        if "manifest.json" not in names or "SEMANTIC_ALIGNMENT_INSTRUCTIONS.md" not in names:
            raise SemanticAlignmentError("Semantic pack members are incomplete")
        manifest = _parse_json(archive.read("manifest.json"), member="manifest.json")
        required = {
            "format",
            "episode",
            "alignment_policy",
            "source_sha256",
            "word_timeline_sha256",
            "all_windows_sha256",
            "unresolved_windows_sha256",
            "deterministic_results_sha256",
            "producer_sha256",
            "config_sha256",
            "window_count",
            "window_ids",
            "batch_count",
            "batches",
            "instructions_sha256",
            "input_identity_sha256",
            "file_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise SemanticAlignmentError("Semantic pack manifest fields are invalid")
        identity = {
            key: value
            for key, value in manifest.items()
            if key not in {"input_identity_sha256", "file_sha256"}
        }
        if (
            manifest["format"] != INPUT_FORMAT
            or manifest["alignment_policy"] != ALIGNMENT_POLICY
            or manifest["input_identity_sha256"] != hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
            or manifest["batch_count"] != len(manifest["batches"])
            or manifest["window_count"] != len(manifest["window_ids"])
            or len(manifest["window_ids"]) != len(set(manifest["window_ids"]))
        ):
            raise SemanticAlignmentError("Semantic pack identity is invalid")
        expected_names = {"manifest.json", "SEMANTIC_ALIGNMENT_INSTRUCTIONS.md"}
        windows = []
        for number, descriptor in enumerate(manifest["batches"], start=1):
            expected_descriptor_fields = {
                "input_file",
                "return_file",
                "window_count",
                "first_window_id",
                "last_window_id",
                "sha256",
            }
            if not isinstance(descriptor, dict) or set(descriptor) != expected_descriptor_fields:
                raise SemanticAlignmentError("Semantic pack batch descriptor is invalid")
            input_name = f"{INPUT_BATCH_PREFIX}{number:03d}.jsonl"
            return_name = f"{RETURN_BATCH_PREFIX}{number:03d}.jsonl"
            if descriptor["input_file"] != input_name or descriptor["return_file"] != return_name:
                raise SemanticAlignmentError("Semantic pack batch order is invalid")
            expected_names.add(input_name)
            payload = archive.read(input_name)
            if descriptor["sha256"] != _sha_bytes(payload):
                raise SemanticAlignmentError("Semantic pack batch digest changed")
            records = _parse_jsonl(payload, member=input_name)
            if (
                descriptor["window_count"] != len(records)
                or not records
                or descriptor["first_window_id"] != records[0].get("window_id")
                or descriptor["last_window_id"] != records[-1].get("window_id")
            ):
                raise SemanticAlignmentError("Semantic pack batch inventory changed")
            windows.extend(records)
        if set(names) != expected_names:
            raise SemanticAlignmentError("Semantic pack contains unexpected members")
        for name, expected in manifest["file_sha256"].items():
            if name not in expected_names or name == "manifest.json" or _sha_bytes(archive.read(name)) != expected:
                raise SemanticAlignmentError("Semantic pack file digest changed")
        if set(manifest["file_sha256"]) != expected_names - {"manifest.json"}:
            raise SemanticAlignmentError("Semantic pack file digest inventory changed")
        if (
            [record.get("window_id") for record in windows] != manifest["window_ids"]
            or hashlib.sha256(_jsonl_bytes(windows)).hexdigest() != manifest["unresolved_windows_sha256"]
            or manifest["instructions_sha256"] != _sha_bytes(archive.read("SEMANTIC_ALIGNMENT_INSTRUCTIONS.md"))
        ):
            raise SemanticAlignmentError("Semantic pack ordered window binding changed")
    return manifest, windows


def validate_semantic_alignment_return(input_pack, returned_zip, *, config=None):
    settings = config or SemanticAlignmentConfig()
    input_manifest, windows = validate_semantic_alignment_pack(input_pack)
    by_window = {window["window_id"]: window for window in windows}
    with _open_checked_zip(Path(returned_zip)) as archive:
        names = archive.namelist()
        if "manifest.json" not in names:
            raise SemanticAlignmentError("Semantic return manifest is missing")
        manifest = _parse_json(archive.read("manifest.json"), member="manifest.json")
        required = {
            "format",
            "input_identity_sha256",
            "input_manifest_sha256",
            "source_sha256",
            "word_timeline_sha256",
            "producer_sha256",
            "window_count",
            "window_ids",
            "batch_count",
            "batches",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise SemanticAlignmentError("Semantic return manifest fields are invalid")
        if (
            manifest["format"] != RETURN_FORMAT
            or manifest["input_identity_sha256"] != input_manifest["input_identity_sha256"]
            or manifest["input_manifest_sha256"] != hashlib.sha256(canonical_json_bytes(input_manifest)).hexdigest()
            or manifest["source_sha256"] != input_manifest["source_sha256"]
            or manifest["word_timeline_sha256"] != input_manifest["word_timeline_sha256"]
            or manifest["producer_sha256"] != input_manifest["producer_sha256"]
            or manifest["window_count"] != input_manifest["window_count"]
            or manifest["window_ids"] != input_manifest["window_ids"]
            or manifest["batch_count"] != input_manifest["batch_count"]
            or not isinstance(manifest["batches"], list)
            or len(manifest["batches"]) != manifest["batch_count"]
        ):
            raise SemanticAlignmentError("Semantic return binding is stale or invalid")
        expected_names = {"manifest.json"}
        records = []
        for number, descriptor in enumerate(manifest["batches"], start=1):
            fields = {"return_file", "window_count", "sha256"}
            if not isinstance(descriptor, dict) or set(descriptor) != fields:
                raise SemanticAlignmentError("Semantic return batch descriptor is invalid")
            name = f"{RETURN_BATCH_PREFIX}{number:03d}.jsonl"
            if descriptor["return_file"] != name:
                raise SemanticAlignmentError("Semantic return batch order changed")
            expected_names.add(name)
            payload = archive.read(name)
            if descriptor["sha256"] != _sha_bytes(payload):
                raise SemanticAlignmentError("Semantic return batch digest changed")
            batch = _parse_jsonl(payload, member=name)
            if descriptor["window_count"] != len(batch):
                raise SemanticAlignmentError("Semantic return batch count changed")
            records.extend(batch)
        if set(names) != expected_names:
            raise SemanticAlignmentError("Semantic return contains unexpected members")

    returned_ids = [record.get("window_id") if isinstance(record, dict) else None for record in records]
    if returned_ids != input_manifest["window_ids"] or len(returned_ids) != len(set(returned_ids)):
        raise SemanticAlignmentError("Semantic return omitted, duplicated or reordered windows")
    timestamp_fields = {
        "start_ms",
        "end_ms",
        "timestamp",
        "duration_ms",
        "start",
        "end",
    }
    results = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"window_id", "blocks"}:
            raise SemanticAlignmentError("Semantic return record fields are invalid")
        window = by_window[record["window_id"]]
        blocks = record["blocks"]
        if not isinstance(blocks, list) or not blocks:
            raise SemanticAlignmentError("Semantic return window has no blocks")
        normalized = []
        for block in blocks:
            if not isinstance(block, dict):
                raise SemanticAlignmentError("Semantic return block must be an object")
            if timestamp_fields & set(block):
                raise SemanticAlignmentError("GPT supplied a forbidden timestamp field")
            required_block = {"first_word_id", "last_word_id", "final_tr"}
            optional_block = {"confidence", "review_required", "note"}
            if not required_block.issubset(block) or set(block) - required_block - optional_block:
                raise SemanticAlignmentError("Semantic return block fields are invalid")
            normalized.append(
                {
                    "first_word_id": block["first_word_id"],
                    "last_word_id": block["last_word_id"],
                    "final_tr": block["final_tr"],
                    "confidence": block.get("confidence", ""),
                    "review_required": block.get("review_required", False),
                    "note": block.get("note", ""),
                }
            )
        checked = _validate_resolution(
            window,
            {"window_id": record["window_id"], "blocks": normalized},
            settings,
        )
        results.append(
            {
                "window_id": record["window_id"],
                "resolution": "gpt_semantic",
                "blocks": checked,
            }
        )
    return {
        "manifest": manifest,
        "results": results,
        "gpt_supplied_timestamp_field_count": 0,
        "review_required_count": sum(
            block["review_required"]
            for result in results
            for block in result["blocks"]
        ),
    }


__all__ = [
    "INPUT_FORMAT",
    "RETURN_FORMAT",
    "SEMANTIC_ALIGNMENT_INSTRUCTIONS",
    "create_semantic_alignment_pack",
    "validate_semantic_alignment_pack",
    "validate_semantic_alignment_return",
]
