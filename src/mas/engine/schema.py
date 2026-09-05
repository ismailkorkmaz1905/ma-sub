"""Deterministic immutable subtitle schemas and atomic JSON I/O.

The schema is the trust anchor between PREPARE and FINALIZE.  Translation text
is never part of this module and translations are never associated by a block
index.  A block UID binds an episode, timing, source text and schema version;
the episode digest then binds the complete ordered collection of block records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SCHEMA_VERSION = "1.0"
UID_DIGEST_LENGTH = 12

IMMUTABLE_BLOCK_FIELDS: tuple[str, ...] = (
    "block_uid",
    "episode",
    "block_index",
    "start_ms",
    "end_ms",
    "timing_text",
    "primary_text",
    "verification_text",
    "youtube_text",
    "context_before",
    "context_after",
    "vad_info",
    "risk_flags",
)

TEXT_FIELDS: tuple[str, ...] = (
    "timing_text",
    "primary_text",
    "verification_text",
    "youtube_text",
    "context_before",
    "context_after",
)

_TRANSLATION_ONLY_FIELDS = {
    "tr_final",
    "id_final",
    "review_required",
    "note",
}


class SchemaError(ValueError):
    """Raised when an immutable schema is missing, corrupt or inconsistent."""

    def __init__(self, message: str, *, field: str | None = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


SchemaValidationError = SchemaError


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SchemaError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise SchemaError(f"Non-finite JSON number is forbidden: {value}")


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field} must be an integer", field=field, value=value)
    if minimum is not None and value < minimum:
        raise SchemaError(
            f"{field} must be at least {minimum}", field=field, value=value
        )
    return value


def _require_string(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string", field=field, value=value)
    if nonempty and not value.strip():
        raise SchemaError(f"{field} must not be empty", field=field, value=value)
    return value


def normalize_source_text(text: str) -> str:
    """Return a stable, lossless-enough normalization for UID generation.

    NFC keeps Turkish characters canonical.  All Unicode whitespace, including
    non-breaking spaces and line endings, is collapsed to one ASCII space.
    Case and punctuation remain significant so an actual source correction
    produces a new identity rather than silently reusing an old one.
    """

    _require_string(text, "source_text")
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Value is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def generate_block_uid(
    episode: int,
    block_index: int,
    start_ms: int,
    end_ms: int,
    source_text: str,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> str:
    """Generate a deterministic block UID.

    ``block_index`` is present only in the human-readable prefix.  The digest,
    which is the identity-bearing part, depends on episode, exact timing,
    normalized source text and schema version.  It therefore cannot be reused
    safely after retiming or resegmentation.
    """

    episode = _require_int(episode, "episode", minimum=1)
    block_index = _require_int(block_index, "block_index", minimum=1)
    start_ms = _require_int(start_ms, "start_ms", minimum=0)
    end_ms = _require_int(end_ms, "end_ms", minimum=1)
    if end_ms <= start_ms:
        raise SchemaError(
            "end_ms must be greater than start_ms",
            field="end_ms",
            value=end_ms,
        )
    schema_version = _require_string(
        schema_version, "schema_version", nonempty=True
    )
    source_text = normalize_source_text(source_text)
    if not source_text:
        raise SchemaError(
            "source_text used for block_uid must not be empty",
            field="source_text",
            value=source_text,
        )

    identity = {
        "episode": episode,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source_text": source_text,
        "schema_version": schema_version,
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return f"MA{episode:02d}-{block_index:06d}-{digest[:UID_DIGEST_LENGTH]}"


make_block_uid = generate_block_uid


def _source_text_for_uid(block: Mapping[str, Any]) -> str:
    for field in ("timing_text", "primary_text", "verification_text", "youtube_text"):
        value = block.get(field, "")
        if isinstance(value, str) and normalize_source_text(value):
            return value
    raise SchemaError(
        "A block needs non-empty timing_text or source evidence for UID generation",
        field="timing_text",
    )


def _canonicalize_block(
    raw_block: Mapping[str, Any],
    *,
    episode: int,
    schema_version: str,
    expected_index: int,
    allow_blank_uid: bool,
) -> dict[str, Any]:
    if not isinstance(raw_block, Mapping):
        raise SchemaError(
            f"Block {expected_index} must be a JSON object", value=raw_block
        )

    unexpected_translation_fields = _TRANSLATION_ONLY_FIELDS.intersection(raw_block)
    if unexpected_translation_fields:
        fields = ", ".join(sorted(unexpected_translation_fields))
        raise SchemaError(
            f"Immutable block contains translation-only fields: {fields}"
        )

    unexpected_fields = set(raw_block).difference(IMMUTABLE_BLOCK_FIELDS)
    if unexpected_fields:
        raise SchemaError(
            "Immutable block contains unsupported fields that are not covered by "
            f"the schema contract: {', '.join(sorted(unexpected_fields))}"
        )

    missing = [
        field
        for field in IMMUTABLE_BLOCK_FIELDS
        if field not in raw_block and field != "block_uid"
    ]
    if missing:
        raise SchemaError(
            f"Block {expected_index} is missing immutable fields: {', '.join(missing)}"
        )

    block_episode = raw_block.get("episode")
    block_episode = _require_int(block_episode, "episode", minimum=1)
    if block_episode != episode:
        raise SchemaError(
            f"Block {expected_index} episode mismatch: expected {episode}, "
            f"got {block_episode}",
            field="episode",
            value=block_episode,
        )

    block_index = _require_int(raw_block.get("block_index"), "block_index", minimum=1)
    if block_index != expected_index:
        raise SchemaError(
            f"Block order/index mismatch at position {expected_index}: "
            f"got block_index {block_index}",
            field="block_index",
            value=block_index,
        )

    start_ms = _require_int(raw_block.get("start_ms"), "start_ms", minimum=0)
    end_ms = _require_int(raw_block.get("end_ms"), "end_ms", minimum=1)
    if end_ms <= start_ms:
        raise SchemaError(
            f"Block {block_index} end_ms must be greater than start_ms",
            field="end_ms",
            value=end_ms,
        )

    text_values: dict[str, str] = {}
    for field in TEXT_FIELDS:
        text_values[field] = _require_string(raw_block.get(field), field)

    # vad_info is deliberately shape-agnostic, but it must be deterministic JSON.
    vad_info = copy.deepcopy(raw_block.get("vad_info"))
    canonical_json_bytes(vad_info)

    risk_flags = raw_block.get("risk_flags")
    if not isinstance(risk_flags, list) or any(
        not isinstance(flag, str) for flag in risk_flags
    ):
        raise SchemaError(
            "risk_flags must be a JSON list of strings",
            field="risk_flags",
            value=risk_flags,
        )

    source_text = _source_text_for_uid(raw_block)
    expected_uid = generate_block_uid(
        episode,
        block_index,
        start_ms,
        end_ms,
        source_text,
        schema_version,
    )
    supplied_uid = raw_block.get("block_uid", "")
    if supplied_uid is None:
        supplied_uid = ""
    if not isinstance(supplied_uid, str):
        raise SchemaError(
            "block_uid must be a string", field="block_uid", value=supplied_uid
        )
    if supplied_uid.strip() and supplied_uid != expected_uid:
        raise SchemaError(
            f"Block {block_index} UID mismatch: expected {expected_uid}, "
            f"got {supplied_uid}",
            field="block_uid",
            value=supplied_uid,
        )
    if not supplied_uid and not allow_blank_uid:
        raise SchemaError(
            f"Block {block_index} is missing block_uid",
            field="block_uid",
            value=supplied_uid,
        )

    # Emit only the authoritative immutable shape.  This prevents transient ASR
    # implementation details from accidentally changing the translation contract.
    return {
        "block_uid": expected_uid,
        "episode": episode,
        "block_index": block_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        **text_values,
        "vad_info": vad_info,
        "risk_flags": list(risk_flags),
    }


def compute_schema_sha256(
    blocks: Sequence[Mapping[str, Any]],
    episode: int | None = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> str:
    """Hash the complete ordered immutable block structure.

    Blocks are validated and canonicalized before hashing.  The wrapper metadata
    is also covered, so the same records cannot be relabeled as another episode
    or schema version.
    """

    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise SchemaError("blocks must be an ordered sequence")
    if not blocks:
        raise SchemaError("An episode schema must contain at least one block")

    if episode is None:
        first_episode = blocks[0].get("episode") if isinstance(blocks[0], Mapping) else None
        episode = _require_int(first_episode, "episode", minimum=1)
    else:
        episode = _require_int(episode, "episode", minimum=1)
    schema_version = _require_string(
        schema_version, "schema_version", nonempty=True
    )

    canonical_blocks: list[dict[str, Any]] = []
    previous_start = -1
    previous_end = -1
    seen_uids: set[str] = set()
    for position, block in enumerate(blocks, start=1):
        canonical = _canonicalize_block(
            block,
            episode=episode,
            schema_version=schema_version,
            expected_index=position,
            allow_blank_uid=False,
        )
        if canonical["start_ms"] < previous_start:
            raise SchemaError(
                f"Block {position} starts before the preceding block"
            )
        if canonical["start_ms"] < previous_end:
            raise SchemaError(
                f"Block {position} overlaps the preceding block ending at "
                f"{previous_end} ms"
            )
        previous_start = canonical["start_ms"]
        previous_end = canonical["end_ms"]
        uid = canonical["block_uid"]
        if uid in seen_uids:
            raise SchemaError(f"Duplicate block_uid in schema: {uid}")
        seen_uids.add(uid)
        canonical_blocks.append(canonical)

    digest_input = {
        "schema_version": schema_version,
        "episode": episode,
        "blocks": canonical_blocks,
    }
    return hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()


calculate_schema_sha256 = compute_schema_sha256


def build_episode_schema(
    blocks: Iterable[Mapping[str, Any]],
    episode: int,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Fill deterministic UIDs and build a deterministic episode schema."""

    episode = _require_int(episode, "episode", minimum=1)
    schema_version = _require_string(
        schema_version, "schema_version", nonempty=True
    )
    raw_blocks = list(blocks)
    if not raw_blocks:
        raise SchemaError("An episode schema must contain at least one block")

    canonical_blocks: list[dict[str, Any]] = []
    previous_start = -1
    previous_end = -1
    seen_uids: set[str] = set()
    for position, raw_block in enumerate(raw_blocks, start=1):
        canonical = _canonicalize_block(
            raw_block,
            episode=episode,
            schema_version=schema_version,
            expected_index=position,
            allow_blank_uid=True,
        )
        if canonical["start_ms"] < previous_start:
            raise SchemaError(
                f"Block {position} starts before the preceding block"
            )
        if canonical["start_ms"] < previous_end:
            raise SchemaError(
                f"Block {position} overlaps the preceding block ending at "
                f"{previous_end} ms"
            )
        previous_start = canonical["start_ms"]
        previous_end = canonical["end_ms"]
        uid = canonical["block_uid"]
        if uid in seen_uids:
            raise SchemaError(f"Duplicate block_uid in schema: {uid}")
        seen_uids.add(uid)
        canonical_blocks.append(canonical)

    schema_sha256 = compute_schema_sha256(
        canonical_blocks, episode=episode, schema_version=schema_version
    )
    return {
        "schema_version": schema_version,
        "episode": episode,
        "block_count": len(canonical_blocks),
        "schema_sha256": schema_sha256,
        "blocks": canonical_blocks,
    }


build_schema = build_episode_schema


def validate_episode_schema(
    schema: Mapping[str, Any],
    *,
    expected_episode: int | None = None,
    expected_schema_version: str | None = None,
) -> dict[str, Any]:
    """Validate and return a canonical deep copy of an episode schema."""

    if not isinstance(schema, Mapping):
        raise SchemaError("schema must be a JSON object")
    episode = _require_int(schema.get("episode"), "episode", minimum=1)
    schema_version = _require_string(
        schema.get("schema_version"), "schema_version", nonempty=True
    )
    if expected_episode is not None and episode != expected_episode:
        raise SchemaError(
            f"Wrong episode schema: expected {expected_episode}, got {episode}",
            field="episode",
            value=episode,
        )
    if (
        expected_schema_version is not None
        and schema_version != expected_schema_version
    ):
        raise SchemaError(
            "Wrong schema version: "
            f"expected {expected_schema_version}, got {schema_version}",
            field="schema_version",
            value=schema_version,
        )

    blocks = schema.get("blocks")
    if not isinstance(blocks, list):
        raise SchemaError("schema.blocks must be a JSON array", field="blocks")
    declared_count = _require_int(
        schema.get("block_count"), "block_count", minimum=1
    )
    if declared_count != len(blocks):
        raise SchemaError(
            f"block_count mismatch: declared {declared_count}, actual {len(blocks)}",
            field="block_count",
            value=declared_count,
        )

    actual_sha = compute_schema_sha256(
        blocks, episode=episode, schema_version=schema_version
    )
    declared_sha = _require_string(
        schema.get("schema_sha256"), "schema_sha256", nonempty=True
    )
    if not re.fullmatch(r"[0-9a-f]{64}", declared_sha):
        raise SchemaError(
            "schema_sha256 must be 64 lowercase hexadecimal characters",
            field="schema_sha256",
            value=declared_sha,
        )
    if declared_sha != actual_sha:
        raise SchemaError(
            f"schema_sha256 mismatch: expected {actual_sha}, got {declared_sha}",
            field="schema_sha256",
            value=declared_sha,
        )

    canonical_blocks = [
        _canonicalize_block(
            block,
            episode=episode,
            schema_version=schema_version,
            expected_index=position,
            allow_blank_uid=False,
        )
        for position, block in enumerate(blocks, start=1)
    ]
    # Return only the exact canonical immutable shape, detached from the input.
    return {
        "schema_version": schema_version,
        "episode": episode,
        "block_count": declared_count,
        "schema_sha256": actual_sha,
        "blocks": canonical_blocks,
    }


validate_schema = validate_episode_schema


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    validator: Any,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        validator(temporary_path)
        os.replace(temporary_path, destination)
        return destination
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    """Write UTF-8 JSON through a parsed temporary file and atomic replace."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Value is not valid JSON: {exc}") from exc

    def validate(path: Path) -> None:
        read_json(path)

    return _atomic_write_text(path, payload, validator=validate)


write_json_atomic = atomic_write_json


def atomic_write_jsonl(
    path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]
) -> Path:
    """Write UTF-8 JSONL through a fully parsed temporary file."""

    lines: list[str] = []
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise SchemaError(f"JSONL record {line_number} must be an object")
        try:
            lines.append(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"JSONL record {line_number} is invalid: {exc}"
            ) from exc
    payload = "\n".join(lines) + ("\n" if lines else "")

    def validate(path: Path) -> None:
        read_jsonl(path)

    return _atomic_write_text(path, payload, validator=validate)


write_jsonl_atomic = atomic_write_jsonl


def read_json(path: str | os.PathLike[str]) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
    except UnicodeDecodeError as exc:
        raise SchemaError(f"JSON is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_keys,
                        parse_constant=_reject_nonfinite_json_constant,
                    )
                except json.JSONDecodeError as exc:
                    raise SchemaError(
                        f"Invalid JSONL in {path} at line {line_number}, "
                        f"column {exc.colno}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise SchemaError(
                        f"JSONL record in {path} line {line_number} must be an object"
                    )
                records.append(value)
    except UnicodeDecodeError as exc:
        raise SchemaError(f"JSONL is not valid UTF-8: {path}") from exc
    return records


def save_schema(path: str | os.PathLike[str], schema: Mapping[str, Any]) -> Path:
    canonical = validate_episode_schema(schema)
    return atomic_write_json(path, canonical)


def load_schema(
    path: str | os.PathLike[str],
    *,
    expected_episode: int | None = None,
    expected_schema_version: str | None = None,
) -> dict[str, Any]:
    return validate_episode_schema(
        read_json(path),
        expected_episode=expected_episode,
        expected_schema_version=expected_schema_version,
    )


__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "IMMUTABLE_BLOCK_FIELDS",
    "SchemaError",
    "SchemaValidationError",
    "atomic_write_json",
    "atomic_write_jsonl",
    "build_episode_schema",
    "build_schema",
    "calculate_schema_sha256",
    "canonical_json_bytes",
    "compute_schema_sha256",
    "generate_block_uid",
    "load_schema",
    "make_block_uid",
    "normalize_source_text",
    "read_json",
    "read_jsonl",
    "save_schema",
    "sha256_file",
    "validate_episode_schema",
    "validate_schema",
    "write_json_atomic",
    "write_jsonl_atomic",
]
