"""Synthetic fixtures shared by the production workflow tests."""

from __future__ import annotations

import copy
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mas.engine.schema import build_episode_schema  # noqa: E402


def raw_blocks(
    count: int,
    *,
    episode: int = 12,
    start_ms: int = 1_000,
    spacing_ms: int = 2_000,
    duration_ms: int = 1_200,
) -> list[dict[str, Any]]:
    """Return deterministic, non-overlapping blocks with unique evidence."""

    blocks: list[dict[str, Any]] = []
    for offset in range(count):
        index = offset + 1
        cue_start = start_ms + offset * spacing_ms
        cue_end = cue_start + duration_ms
        text = f"Bu benzersiz cümle {index} numaralı sahneye aittir."
        blocks.append(
            {
                "episode": episode,
                "block_index": index,
                "start_ms": cue_start,
                "end_ms": cue_end,
                "timing_text": text,
                "primary_text": text,
                "verification_text": text,
                "youtube_text": text,
                "context_before": (
                    "" if index == 1 else f"Önceki benzersiz cümle {index - 1}."
                ),
                "context_after": (
                    "" if index == count else f"Sonraki benzersiz cümle {index + 1}."
                ),
                "vad_info": {
                    "first_word_start_ms": cue_start + 50,
                    "last_word_end_ms": cue_end - 180,
                    "max_internal_gap_ms": 300,
                    "internal_gap_resolved": True,
                },
                "risk_flags": [],
            }
        )
    return blocks


def make_schema(
    count: int = 4,
    *,
    episode: int = 12,
    schema_version: str = "1.0",
) -> dict[str, Any]:
    return build_episode_schema(
        raw_blocks(count, episode=episode),
        episode=episode,
        schema_version=schema_version,
    )


def synthetic_transcription(block_count: int = 10) -> dict[str, Any]:
    """Return timed words/VAD whose 1,600 ms gaps create clear phrase blocks."""

    words: list[dict[str, Any]] = []
    vad_regions: list[dict[str, Any]] = []
    word_index = 0
    for block_offset in range(block_count):
        block_number = block_offset + 1
        phrase_start = 1_000 + block_offset * 4_000
        tokens = (
            "Bu",
            "benzersiz",
            "cümle",
            str(block_number),
            "numaralı",
            "sahneye",
            "aittir",
        )
        for token_offset, token in enumerate(tokens):
            word_index += 1
            token_start = phrase_start + token_offset * 350
            words.append(
                {
                    "word_index": word_index,
                    "segment_id": 1,
                    "start_ms": token_start,
                    "end_ms": token_start + 300,
                    "text": token,
                    "probability": 0.99,
                }
            )
        vad_regions.append(
            {
                "vad_region_index": block_number,
                "start_ms": phrase_start - 50,
                "end_ms": phrase_start + 2_450,
                "source": "synthetic-test",
            }
        )
    return {
        "language": "tr",
        "words": words,
        "vad_regions": vad_regions,
        "verification_results": [],
        "suspicious_spans": [],
    }


def translation_records(
    schema: Mapping[str, Any],
    *,
    include_optional_echoes: bool = False,
) -> list[dict[str, Any]]:
    """Return one exact output record for every schema block."""

    digest = str(schema["schema_sha256"])
    records: list[dict[str, Any]] = []
    for block in schema["blocks"]:
        index = int(block["block_index"])
        record: dict[str, Any] = {
            "block_uid": block["block_uid"],
            "schema_sha256": digest,
            "tr_final": block["primary_text"],
            "id_final": f"Kalimat unik ini milik adegan nomor {index}.",
            "review_required": False,
            "note": "",
        }
        if include_optional_echoes:
            record.update(
                {
                    "episode": block["episode"],
                    "block_index": block["block_index"],
                    "start_ms": block["start_ms"],
                    "end_ms": block["end_ms"],
                }
            )
        records.append(record)
    return records


def records_by_uid(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["block_uid"]): copy.deepcopy(dict(record)) for record in records}


def clone_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy([dict(record) for record in records])


def write_translated_zip(
    path: Path,
    schema: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    batch_sizes: Sequence[int] | None = None,
    report_overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Write the documented Work Ultra output ZIP contract."""

    if batch_sizes is None:
        batch_sizes = (len(records),)
    if sum(batch_sizes) != len(records):
        raise ValueError("batch sizes must consume every record")

    report = {
        "schema_sha256": schema["schema_sha256"],
        "total_input_blocks": schema["block_count"],
        "total_output_blocks": len(records),
        "missing_block_count": 0,
        "duplicate_block_count": 0,
        "review_required_count": sum(
            record.get("review_required") is True for record in records
        ),
    }
    report.update(dict(report_overrides or {}))

    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for batch_number, batch_size in enumerate(batch_sizes, start=1):
            batch = records[cursor : cursor + batch_size]
            cursor += batch_size
            payload = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in batch
            )
            archive.writestr(
                f"translated_batch_{batch_number:03d}.jsonl",
                payload.encode("utf-8"),
            )
        archive.writestr(
            "translation_report.json",
            (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
    return path

