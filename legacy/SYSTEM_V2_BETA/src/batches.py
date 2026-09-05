"""Translation batch construction and validated atomic pack creation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

try:  # Package import in notebooks/tests.
    from .schema import (
        IMMUTABLE_BLOCK_FIELDS,
        SchemaError,
        atomic_write_json,
        canonical_json_bytes,
        read_json,
        read_jsonl,
        sha256_file,
        validate_episode_schema,
    )
except ImportError:  # Direct module import after adding src/ to sys.path.
    from schema import (  # type: ignore
        IMMUTABLE_BLOCK_FIELDS,
        SchemaError,
        atomic_write_json,
        canonical_json_bytes,
        read_json,
        read_jsonl,
        sha256_file,
        validate_episode_schema,
    )


DEFAULT_BATCH_SIZE = 400
DEFAULT_MIN_BATCH_SIZE = 350
DEFAULT_MAX_BATCH_SIZE = 500
BATCH_NAME_RE = re.compile(r"^batch_(\d{3})\.jsonl$")


class BatchError(ValueError):
    """Raised when a translation pack or batch contract is invalid."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BatchError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise BatchError(f"Non-finite JSON number is forbidden: {value}")


@dataclass(frozen=True)
class BatchFile:
    name: str
    records: tuple[dict[str, Any], ...]

    @property
    def block_count(self) -> int:
        return len(self.records)


def _balanced_batch_sizes(
    total_blocks: int,
    *,
    target_size: int,
    min_size: int,
    max_size: int,
) -> list[int]:
    if isinstance(total_blocks, bool) or not isinstance(total_blocks, int):
        raise BatchError("total_blocks must be an integer")
    if total_blocks < 1:
        raise BatchError("At least one block is required")
    for value, label in (
        (target_size, "target_size"),
        (min_size, "min_size"),
        (max_size, "max_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BatchError(f"{label} must be a positive integer")
    if not min_size <= target_size <= max_size:
        raise BatchError("Require min_size <= target_size <= max_size")

    if total_blocks <= max_size:
        return [total_blocks]

    minimum_batch_count = (total_blocks + max_size - 1) // max_size
    maximum_batch_count = max(1, total_blocks // min_size)
    preferred_batch_count = max(1, round(total_blocks / target_size))
    if minimum_batch_count <= maximum_batch_count:
        batch_count = min(
            max(preferred_batch_count, minimum_batch_count), maximum_batch_count
        )
    else:
        # Some totals cannot satisfy both approximate limits (for example 600).
        # Favor never creating an oversized upload batch.
        batch_count = minimum_batch_count

    base, remainder = divmod(total_blocks, batch_count)
    return [base + (1 if index < remainder else 0) for index in range(batch_count)]


def build_translation_records(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create ordered, read-only input records from a validated schema."""

    trusted = validate_episode_schema(schema)
    common = {
        "schema_version": trusted["schema_version"],
        "schema_sha256": trusted["schema_sha256"],
    }
    records: list[dict[str, Any]] = []
    for block in trusted["blocks"]:
        record = {field: copy.deepcopy(block[field]) for field in IMMUTABLE_BLOCK_FIELDS}
        record.update(common)
        records.append(record)
    return records


def build_translation_batches(
    schema: Mapping[str, Any],
    *,
    target_size: int = DEFAULT_BATCH_SIZE,
    min_size: int = DEFAULT_MIN_BATCH_SIZE,
    max_size: int = DEFAULT_MAX_BATCH_SIZE,
    batch_size: int | None = None,
) -> list[BatchFile]:
    """Pack all blocks once in global schema order.

    ``batch_size`` is a convenience alias used by the notebooks.  The final
    batch sizes are balanced so a tiny trailing batch is avoided when possible.
    Blocks already contain their immutable ``context_before`` and
    ``context_after`` fields; no context record is duplicated or emitted.
    """

    if batch_size is not None:
        target_size = batch_size
        # A deliberate non-default size (often used in tests) acts as an exact
        # upper bound while preserving normal 350-500 production defaults.
        if batch_size < min_size:
            min_size = batch_size
            max_size = batch_size
        elif batch_size > max_size:
            max_size = batch_size
    records = build_translation_records(schema)
    sizes = _balanced_batch_sizes(
        len(records),
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
    )
    batches: list[BatchFile] = []
    cursor = 0
    for number, size in enumerate(sizes, start=1):
        batch_records = tuple(copy.deepcopy(records[cursor : cursor + size]))
        batches.append(BatchFile(f"batch_{number:03d}.jsonl", batch_records))
        cursor += size
    if cursor != len(records):  # Defensive invariant.
        raise BatchError("Internal batch packing count mismatch")
    return batches


pack_translation_batches = build_translation_batches


def _json_bytes(value: Any, *, indent: int = 2) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - installed by requirements.
        raise BatchError("PyYAML is required to read YAML glossary files") from exc
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def _default_glossary() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    names_path = root / "config" / "names.yaml"
    religious_path = root / "config" / "religious_terms.yaml"
    glossary: dict[str, Any] = {}
    if names_path.exists():
        names = _load_yaml(names_path) or {}
        if not isinstance(names, Mapping):
            raise BatchError(f"Invalid names glossary: {names_path}")
        glossary["canonical_names"] = copy.deepcopy(
            names.get("canonical_names", [])
        )
        glossary["forbidden_name_variants"] = copy.deepcopy(
            names.get("forbidden_variants", {})
        )
        glossary["source_name_variants"] = copy.deepcopy(
            names.get("source_variants", {})
        )
    if religious_path.exists():
        religious = _load_yaml(religious_path) or {}
        if not isinstance(religious, Mapping):
            raise BatchError(f"Invalid religious glossary: {religious_path}")
        glossary["religious_terms"] = copy.deepcopy(religious.get("terms", []))
        glossary["allah_only_semoga_is_invalid"] = bool(
            religious.get("allah_only_semoga_is_invalid", True)
        )
    return glossary


def _resolve_glossary(
    glossary: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> dict[str, Any]:
    if glossary is None:
        return _default_glossary()
    if isinstance(glossary, Mapping):
        canonical_json_bytes(glossary)
        return copy.deepcopy(dict(glossary))
    path = Path(glossary)
    if not path.is_file():
        raise BatchError(f"Glossary file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = _load_yaml(path)
    else:
        value = read_json(path)
    if not isinstance(value, Mapping):
        raise BatchError("glossary must be a JSON/YAML object")
    return copy.deepcopy(dict(value))


def _resolve_instructions(
    instructions_path: str | os.PathLike[str] | None,
) -> bytes:
    path = (
        Path(instructions_path)
        if instructions_path is not None
        else Path(__file__).resolve().parents[1] / "TRANSLATION_INSTRUCTIONS.md"
    )
    if not path.is_file():
        raise BatchError(f"Translation instructions not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchError(f"Translation instructions are not UTF-8: {path}") from exc
    if not text.strip():
        raise BatchError("Translation instructions are empty")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _validate_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name:
        raise BatchError(f"Unsafe ZIP member path: {name}")
    if len(pure.parts) != 1:
        raise BatchError(f"ZIP members must be top-level files: {name}")


def _parse_json_bytes(payload: bytes, *, member: str) -> Any:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchError(f"{member} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise BatchError(
            f"Invalid JSON in {member} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc


def _parse_jsonl_bytes(payload: bytes, *, member: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchError(f"{member} is not valid UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise BatchError(
                f"Invalid JSONL in {member} line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise BatchError(f"{member} line {line_number} must be a JSON object")
        records.append(record)
    return records


def _validate_packed_records(
    schema: Mapping[str, Any], batch_files: Sequence[BatchFile]
) -> None:
    trusted = validate_episode_schema(schema)
    expected_records = build_translation_records(trusted)
    actual_records: list[dict[str, Any]] = []
    previous_number = 0
    for batch in batch_files:
        if not batch.records:
            raise BatchError(f"Input batch contains zero records: {batch.name}")
        match = BATCH_NAME_RE.fullmatch(batch.name)
        if not match:
            raise BatchError(f"Invalid input batch filename: {batch.name}")
        number = int(match.group(1))
        if number != previous_number + 1:
            raise BatchError(
                f"Input batch sequence gap: expected {previous_number + 1:03d}, "
                f"got {number:03d}"
            )
        previous_number = number
        actual_records.extend(batch.records)
    if len(actual_records) != len(expected_records):
        raise BatchError(
            f"Packed block count mismatch: expected {len(expected_records)}, "
            f"got {len(actual_records)}"
        )
    for position, (expected, actual) in enumerate(
        zip(expected_records, actual_records), start=1
    ):
        if actual != expected:
            uid = actual.get("block_uid") if isinstance(actual, Mapping) else None
            raise BatchError(
                f"Packed record mismatch at position {position}, block_uid={uid!r}"
            )


def validate_translation_pack(
    pack_path: str | os.PathLike[str],
    *,
    expected_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate ZIP CRC, exact members, hashes, schema and every input record."""

    path = Path(pack_path)
    if not path.is_file():
        raise BatchError(f"Translation pack not found: {path}")
    try:
        archive = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise BatchError(f"Corrupt translation pack ZIP: {path}") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise BatchError("Translation pack contains duplicate ZIP member names")
        for name in names:
            _validate_member_name(name)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise BatchError(f"Translation pack CRC failure: {bad_member}")
        required_static = {
            "manifest.json",
            "schema.json",
            "glossary.json",
            "TRANSLATION_INSTRUCTIONS.md",
        }
        missing_static = required_static.difference(names)
        if missing_static:
            raise BatchError(
                "Translation pack missing members: " + ", ".join(sorted(missing_static))
            )

        manifest = _parse_json_bytes(
            archive.read("manifest.json"), member="manifest.json"
        )
        if not isinstance(manifest, dict):
            raise BatchError("manifest.json must contain a JSON object")
        packed_schema_raw = _parse_json_bytes(
            archive.read("schema.json"), member="schema.json"
        )
        glossary_document = _parse_json_bytes(
            archive.read("glossary.json"), member="glossary.json"
        )
        if not isinstance(glossary_document, dict):
            raise BatchError("glossary.json must contain a JSON object")
        try:
            instructions_text = archive.read("TRANSLATION_INSTRUCTIONS.md").decode(
                "utf-8-sig"
            )
        except UnicodeDecodeError as exc:
            raise BatchError("TRANSLATION_INSTRUCTIONS.md is not valid UTF-8") from exc
        if not instructions_text.strip():
            raise BatchError("TRANSLATION_INSTRUCTIONS.md is empty")
        try:
            packed_schema = validate_episode_schema(packed_schema_raw)
        except SchemaError as exc:
            raise BatchError(f"Invalid schema.json: {exc}") from exc
        if expected_schema is not None:
            trusted_expected = validate_episode_schema(expected_schema)
            for key in ("episode", "schema_version", "schema_sha256", "block_count"):
                if packed_schema[key] != trusted_expected[key]:
                    raise BatchError(
                        f"Packed schema {key} mismatch: expected "
                        f"{trusted_expected[key]!r}, got {packed_schema[key]!r}"
                    )

        for key in ("episode", "schema_version", "schema_sha256", "block_count"):
            if manifest.get(key) != packed_schema[key]:
                raise BatchError(
                    f"Manifest {key} mismatch: expected {packed_schema[key]!r}, "
                    f"got {manifest.get(key)!r}"
                )
        batch_count = manifest.get("batch_count")
        if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count < 1:
            raise BatchError("Manifest batch_count must be a positive integer")
        expected_batch_names = [
            f"batch_{number:03d}.jsonl" for number in range(1, batch_count + 1)
        ]
        expected_members = required_static.union(expected_batch_names)
        if set(names) != expected_members:
            missing = sorted(expected_members.difference(names))
            extra = sorted(set(names).difference(expected_members))
            raise BatchError(
                f"Translation pack member mismatch; missing={missing}, extra={extra}"
            )

        descriptors = manifest.get("batches")
        if not isinstance(descriptors, list) or len(descriptors) != batch_count:
            raise BatchError("Manifest batches list does not match batch_count")
        descriptor_by_name: dict[str, Mapping[str, Any]] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise BatchError("Every manifest batch descriptor must be an object")
            name = descriptor.get("input_file")
            if not isinstance(name, str) or name in descriptor_by_name:
                raise BatchError("Invalid or duplicate input_file in manifest batches")
            descriptor_by_name[name] = descriptor
        if list(descriptor_by_name) != expected_batch_names:
            raise BatchError("Manifest batch descriptor order/names are invalid")

        batch_files: list[BatchFile] = []
        for name in expected_batch_names:
            payload = archive.read(name)
            descriptor = descriptor_by_name[name]
            actual_hash = _sha256_bytes(payload)
            if descriptor.get("sha256") != actual_hash:
                raise BatchError(f"Batch hash mismatch: {name}")
            records = _parse_jsonl_bytes(payload, member=name)
            declared_count = descriptor.get("block_count")
            if (
                isinstance(declared_count, bool)
                or not isinstance(declared_count, int)
                or declared_count < 1
            ):
                raise BatchError(
                    f"Batch block_count must be a positive integer: {name}"
                )
            if declared_count != len(records):
                raise BatchError(f"Batch block_count mismatch: {name}")
            if not records:
                raise BatchError(f"Input batch contains zero records: {name}")
            if descriptor.get("first_block_uid") != records[0].get("block_uid"):
                raise BatchError(f"Batch first_block_uid mismatch: {name}")
            if descriptor.get("last_block_uid") != records[-1].get("block_uid"):
                raise BatchError(f"Batch last_block_uid mismatch: {name}")
            batch_files.append(BatchFile(name, tuple(records)))

        file_hashes = manifest.get("file_sha256")
        if not isinstance(file_hashes, Mapping):
            raise BatchError("Manifest file_sha256 must be an object")
        for member in ("schema.json", "glossary.json", "TRANSLATION_INSTRUCTIONS.md"):
            if file_hashes.get(member) != _sha256_bytes(archive.read(member)):
                raise BatchError(f"Manifest file hash mismatch: {member}")
        _validate_packed_records(packed_schema, batch_files)
        return copy.deepcopy(manifest)


def _manifest_for_pack(
    schema: Mapping[str, Any],
    batches: Sequence[BatchFile],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    trusted = validate_episode_schema(schema)
    if not batches:
        raise BatchError("Translation pack must contain at least one batch")
    descriptors: list[dict[str, Any]] = []
    for batch in batches:
        if not batch.records:
            raise BatchError(f"Input batch contains zero records: {batch.name}")
        payload = payloads[batch.name]
        descriptors.append(
            {
                "input_file": batch.name,
                "output_file": f"translated_{batch.name}",
                "block_count": len(batch.records),
                "first_block_uid": batch.records[0]["block_uid"],
                "last_block_uid": batch.records[-1]["block_uid"],
                "sha256": _sha256_bytes(payload),
            }
        )
    return {
        "format_version": "1.0",
        "series_name": "Muhtemel Ask",
        "episode": trusted["episode"],
        "schema_version": trusted["schema_version"],
        "schema_sha256": trusted["schema_sha256"],
        "block_count": trusted["block_count"],
        "total_input_blocks": trusted["block_count"],
        "batch_count": len(batches),
        "batches": descriptors,
        "file_sha256": {
            member: _sha256_bytes(payloads[member])
            for member in (
                "schema.json",
                "glossary.json",
                "TRANSLATION_INSTRUCTIONS.md",
            )
        },
    }


def create_translation_pack(
    schema: Mapping[str, Any],
    out_zip: str | os.PathLike[str],
    *,
    glossary: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    instructions_path: str | os.PathLike[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_blocks: int = 2,
    marker_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Create, CRC-check and atomically publish a translation input ZIP.

    A validated ``translation_pack.done.json`` marker is written only after the
    final ZIP has been reopened and checked against the trusted schema.
    """

    trusted = validate_episode_schema(schema)
    if isinstance(context_blocks, bool) or not isinstance(context_blocks, int) or context_blocks < 0:
        raise BatchError("context_blocks must be a non-negative integer")
    # Context is already immutable in each schema block.  Keep the parameter in
    # the public contract/config and record it, but never duplicate context as
    # translatable records.
    batches = build_translation_batches(trusted, batch_size=batch_size)
    glossary_value = _resolve_glossary(glossary)
    instructions = _resolve_instructions(instructions_path)

    payloads: dict[str, bytes] = {
        "schema.json": _json_bytes(trusted),
        "glossary.json": _json_bytes(glossary_value),
        "TRANSLATION_INSTRUCTIONS.md": instructions,
    }
    for batch in batches:
        payloads[batch.name] = _jsonl_bytes(batch.records)
    manifest = _manifest_for_pack(trusted, batches, payloads)
    manifest["batch_context_blocks"] = context_blocks
    payloads["manifest.json"] = _json_bytes(manifest)

    destination = Path(out_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_filename = f"Muhtemel Ask {trusted['episode']}.Bolum_TRANSLATION_PACK.zip"
    if destination.name != expected_filename:
        # The caller may intentionally choose another path during tests, but the
        # manifest makes the canonical user-facing name explicit.
        manifest["canonical_filename"] = expected_filename
        payloads["manifest.json"] = _json_bytes(manifest)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        member_order = [
            "manifest.json",
            "schema.json",
            "glossary.json",
            "TRANSLATION_INSTRUCTIONS.md",
            *[batch.name for batch in batches],
        ]
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in member_order:
                archive.writestr(_zip_info(name), payloads[name], compresslevel=9)
        validate_translation_pack(temporary, expected_schema=trusted)
        os.replace(temporary, destination)
        temporary = None
        validated_manifest = validate_translation_pack(
            destination, expected_schema=trusted
        )
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    marker_destination = (
        Path(marker_path)
        if marker_path is not None
        else destination.parent / "translation_pack.done.json"
    )
    marker = {
        "stage": "translation_pack",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "episode": trusted["episode"],
        "schema_version": trusted["schema_version"],
        "schema_sha256": trusted["schema_sha256"],
        "output_file": destination.name,
        "output_size_bytes": destination.stat().st_size,
        "output_sha256": sha256_file(destination),
        "manifest_sha256": _sha256_bytes(payloads["manifest.json"]),
    }
    atomic_write_json(marker_destination, marker)
    validate_translation_pack_marker(
        marker_destination, destination, expected_schema=trusted
    )
    return validated_manifest


def validate_translation_pack_marker(
    marker_path: str | os.PathLike[str],
    pack_path: str | os.PathLike[str],
    *,
    expected_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    marker = read_json(marker_path)
    if not isinstance(marker, dict):
        raise BatchError("translation_pack.done.json must contain an object")
    path = Path(pack_path)
    if marker.get("stage") != "translation_pack":
        raise BatchError("Wrong stage in translation pack marker")
    if marker.get("output_file") != path.name:
        raise BatchError("Translation pack marker filename mismatch")
    if marker.get("output_size_bytes") != path.stat().st_size:
        raise BatchError("Translation pack marker size mismatch")
    if marker.get("output_sha256") != sha256_file(path):
        raise BatchError("Translation pack marker SHA-256 mismatch")
    manifest = validate_translation_pack(path, expected_schema=expected_schema)
    for key in ("episode", "schema_version", "schema_sha256"):
        if marker.get(key) != manifest.get(key):
            raise BatchError(f"Translation pack marker {key} mismatch")
    with zipfile.ZipFile(path, "r") as archive:
        manifest_sha = _sha256_bytes(archive.read("manifest.json"))
    if marker.get("manifest_sha256") != manifest_sha:
        raise BatchError("Translation pack marker manifest hash mismatch")
    return marker


def read_batch_files(paths: Sequence[str | os.PathLike[str]]) -> list[BatchFile]:
    batches: list[BatchFile] = []
    for path_value in paths:
        path = Path(path_value)
        records = tuple(read_jsonl(path))
        if not records:
            raise BatchError(f"Input batch contains zero records: {path.name}")
        batches.append(BatchFile(path.name, records))
    return batches


__all__ = [
    "BatchError",
    "BatchFile",
    "DEFAULT_BATCH_SIZE",
    "build_translation_batches",
    "build_translation_records",
    "create_translation_pack",
    "pack_translation_batches",
    "read_batch_files",
    "validate_translation_pack",
    "validate_translation_pack_marker",
]
