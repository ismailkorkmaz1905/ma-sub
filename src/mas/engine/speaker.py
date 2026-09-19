from __future__ import annotations

from typing import Any, Mapping


def speaker_id(record: Mapping[str, Any], label: str) -> str | None:
    if "speaker_id" not in record:
        return None
    value = record["speaker_id"]
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} speaker_id must be a non-empty trimmed string")
    return value


def overlap_is_unsafe(left: str | None, right: str | None) -> bool:
    return left is None or right is None or left == right
