"""Strict Indonesian-only translation contract for the subtitle pipeline.

The engine deliberately separates Turkish correction/timing from Indonesian
translation. The input to this module is an already corrected and
forced-aligned Turkish schema.  Every translated output record must echo that
immutable record byte-for-byte at the JSON-value level and may add only
``id_final``, ``review_required`` and ``note``.

Legacy protocol identifiers remain accepted so existing artifacts are
reproducible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .speaker import overlap_is_unsafe, speaker_id


ID_PACK_FORMAT_VERSION = "2.0"
DEFAULT_ID_BATCH_SIZE = 400

ID_GLOSSARY_FIELDS = frozenset(
    {
        "canonical_names",
        "forbidden_name_variants",
        "source_name_variants",
        "religious_terms",
        "allah_only_semoga_is_invalid",
    }
)

REQUIRED_ALIGNED_BLOCK_FIELDS = frozenset(
    {
        "block_uid",
        "block_index",
        "start_ms",
        "end_ms",
        "tr_text",
        "alignment_provenance",
    }
)
PACK_RECORD_FIELDS = frozenset({"episode", "schema_version", "schema_sha256"})
ID_OUTPUT_FIELDS = frozenset({"id_final", "review_required", "note"})

INPUT_BATCH_RE = re.compile(r"^batch_(\d{3})\.jsonl$")
OUTPUT_BATCH_RE = re.compile(r"^translated_batch_(\d{3})\.jsonl$")

ID_TRANSLATION_INSTRUCTIONS = """# Indonesian-only translation contract

Translate only `tr_text` into natural Indonesian and write it to `id_final`.

## Indonesian dialogue style

- Write natural, conversational Indonesian for subtitles, not literal or
  word-for-word machine translation.
- Preserve the speaker's relationship, register and personality. `aku`,
  `kamu`, `nggak`, `udah` and `aja` are appropriate in ordinary informal
  dialogue, but never force slang into a formal, professional, older-younger
  or respectful exchange. Use `saya`, `Anda`, `Pak` and `Bu` when the scene
  requires them.
- Preserve romance, anger, sarcasm, comedy, insults, hesitation, unfinished
  speech and emotional intensity. Do not censor, soften, explain or improve
  what the character means.
- Keep the result concise and readable as a subtitle. Do not add speaker
  labels, translator notes, background explanation or dialogue that is not in
  `tr_text`.
- Read nearby ordered records for conversational context and pronoun/register
  consistency, but translate only the current record. Never move words or
  meaning between records.

- Copy every input record exactly and in the same order.
- Never change `block_uid`, `block_index`, `start_ms`, `end_ms`, `tr_text`,
  `alignment_provenance`, `episode`, `schema_version`, `schema_sha256`, or any
  other input field.
- Add only `id_final`, `review_required`, and `note`.
- `id_final` is required and must not be blank.
- `review_required` is optional and, when present, must be boolean.
- `note` is optional and, when present, must be a string.
- Never split, merge, omit, duplicate, insert, or reorder records.
- Read `glossary.json` before translating. Preserve the canonical spelling of
  every proper name; never emit a forbidden name variant.
- Preserve every number and money value in `tr_text`, including every repeated
  occurrence. Never omit, merge, invent, or change a value.
- If `tr_text` contains Allah, `id_final` must retain the literal word Allah;
  `semoga` alone is not a translation of Allah.
- When a source phrase in `religious_terms` occurs, use one of its
  `preferred_indonesian` mappings and preserve Allah wherever required.
"""

_MANIFEST_FIELDS = frozenset(
    {
        "format_version",
        "kind",
        "episode",
        "schema_version",
        "schema_sha256",
        "block_count",
        "batch_count",
        "batches",
        "file_sha256",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "format_version",
        "kind",
        "schema_sha256",
        "total_input_blocks",
        "total_output_blocks",
        "missing_block_count",
        "duplicate_block_count",
        "review_required_count",
    }
)


class IDTranslationError(ValueError):
    """Raised when an ID pack, schema, or translated output is invalid."""

    def __init__(
        self,
        message: str,
        result: "IDTranslationValidationResult | None" = None,
    ) -> None:
        if result is not None and result.issues:
            message = f"{message}: {result.issues[0]}"
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class IDTranslationValidationResult:
    """Validated output plus a read-only UID lookup."""

    schema_sha256: str
    expected_block_count: int
    output_block_count: int
    issues: tuple[str, ...] = ()
    records_by_uid: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def ordered_records(self, schema: Mapping[str, Any]) -> list[dict[str, Any]]:
        if not self.ok:
            raise IDTranslationError(
                "Cannot expose Indonesian translations after failed validation",
                self,
            )
        trusted = validate_aligned_turkish_schema(schema)
        if trusted["schema_sha256"] != self.schema_sha256:
            raise IDTranslationError(
                "Validated Indonesian output belongs to a different schema"
            )
        return [
            copy.deepcopy(dict(self.records_by_uid[block["block_uid"]]))
            for block in trusted["blocks"]
        ]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IDTranslationError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise IDTranslationError(f"Non-finite JSON number is forbidden: {value}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise IDTranslationError(f"Value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [_canonical_json_bytes(dict(record)) for record in records]
    return b"\n".join(lines) + (b"\n" if lines else b"")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IDTranslationError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IDTranslationError(f"{field_name} must be a non-empty string")
    return value


def _json_clone(value: Any) -> Any:
    """Deep-copy through strict JSON so tuples/custom mappings cannot drift."""

    return json.loads(
        _canonical_json_bytes(value).decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except IDTranslationError:
        return False


def _validated_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise IDTranslationError(f"{field_name} must be a JSON array")
    result: list[str] = []
    for position, item in enumerate(value, start=1):
        text = _require_nonempty_string(item, f"{field_name}[{position}]")
        if text in result:
            raise IDTranslationError(f"{field_name} contains duplicate value {text!r}")
        result.append(text)
    return result


def _validated_variant_map(
    value: Any,
    field_name: str,
    canonical_names: set[str],
) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise IDTranslationError(f"{field_name} must be a JSON object")
    result: dict[str, list[str]] = {}
    for raw_name, raw_variants in value.items():
        name = _require_nonempty_string(raw_name, f"{field_name} key")
        if name not in canonical_names:
            raise IDTranslationError(
                f"{field_name} references unknown canonical name {name!r}"
            )
        result[name] = _validated_string_list(
            raw_variants,
            f"{field_name}.{name}",
        )
    return result


def validate_id_translation_glossary(
    glossary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the hash-bound semantic guidance shipped in every ID pack."""

    if not isinstance(glossary, Mapping):
        raise IDTranslationError("ID translation glossary must be a JSON object")
    trusted = _json_clone(dict(glossary))
    if set(trusted) != ID_GLOSSARY_FIELDS:
        missing = sorted(ID_GLOSSARY_FIELDS.difference(trusted))
        extra = sorted(set(trusted).difference(ID_GLOSSARY_FIELDS))
        raise IDTranslationError(
            "ID translation glossary field mismatch; "
            f"missing={missing}, extra={extra}"
        )

    canonical_names = _validated_string_list(
        trusted["canonical_names"],
        "glossary.canonical_names",
    )
    if not canonical_names:
        raise IDTranslationError("glossary.canonical_names must not be empty")
    canonical_set = set(canonical_names)
    forbidden = _validated_variant_map(
        trusted["forbidden_name_variants"],
        "glossary.forbidden_name_variants",
        canonical_set,
    )
    source_variants = _validated_variant_map(
        trusted["source_name_variants"],
        "glossary.source_name_variants",
        canonical_set,
    )

    raw_terms = trusted["religious_terms"]
    if not isinstance(raw_terms, list) or not raw_terms:
        raise IDTranslationError("glossary.religious_terms must be a non-empty array")
    terms: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for position, raw_term in enumerate(raw_terms, start=1):
        if not isinstance(raw_term, dict):
            raise IDTranslationError(
                f"glossary.religious_terms[{position}] must be an object"
            )
        allowed = {"source", "preferred_indonesian", "must_preserve_allah"}
        if not {"source", "preferred_indonesian"}.issubset(raw_term) or not set(
            raw_term
        ).issubset(allowed):
            raise IDTranslationError(
                f"glossary.religious_terms[{position}] has invalid fields"
            )
        source = _require_nonempty_string(
            raw_term["source"],
            f"glossary.religious_terms[{position}].source",
        )
        if source in seen_sources:
            raise IDTranslationError(
                f"glossary.religious_terms contains duplicate source {source!r}"
            )
        seen_sources.add(source)
        preferred = _validated_string_list(
            raw_term["preferred_indonesian"],
            f"glossary.religious_terms[{position}].preferred_indonesian",
        )
        if not preferred:
            raise IDTranslationError(
                f"glossary.religious_terms[{position}] needs a preferred mapping"
            )
        preserve_allah = raw_term.get("must_preserve_allah", False)
        if not isinstance(preserve_allah, bool):
            raise IDTranslationError(
                f"glossary.religious_terms[{position}].must_preserve_allah "
                "must be boolean"
            )
        # Only a standalone Allah token (including forms such as Allah'ım)
        # requires the explicit literal-preservation flag.  Turkish lexical
        # forms such as Maşallah/İnşallah contain the same character sequence
        # but intentionally use their glossary mapping instead.
        contains_literal_allah = re.search(
            r"(?<!\w)allah(?!\w)",
            source,
            flags=re.IGNORECASE,
        ) is not None
        if contains_literal_allah and preserve_allah is not True:
            raise IDTranslationError(
                f"glossary.religious_terms[{position}] containing Allah must "
                "set must_preserve_allah=true"
            )
        term = {
            "source": source,
            "preferred_indonesian": preferred,
        }
        if "must_preserve_allah" in raw_term:
            term["must_preserve_allah"] = preserve_allah
        terms.append(term)

    allah_rule = trusted["allah_only_semoga_is_invalid"]
    if allah_rule is not True:
        raise IDTranslationError(
            "glossary.allah_only_semoga_is_invalid must be true"
        )
    return {
        "canonical_names": canonical_names,
        "forbidden_name_variants": forbidden,
        "source_name_variants": source_variants,
        "religious_terms": terms,
        "allah_only_semoga_is_invalid": True,
    }


def load_default_id_translation_glossary() -> dict[str, Any]:
    """Load and validate the repository's canonical name/religious configs."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pinned runtime dependency.
        raise IDTranslationError(
            "PyYAML is required to load the ID translation glossary"
        ) from exc
    config_root = Path(__file__).resolve().parents[3] / "config" / "production"
    names_path = config_root / "names.yaml"
    religious_path = config_root / "religious_terms.yaml"
    try:
        with names_path.open("r", encoding="utf-8-sig") as handle:
            names = yaml.safe_load(handle)
        with religious_path.open("r", encoding="utf-8-sig") as handle:
            religious = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise IDTranslationError(
            f"Could not load the canonical ID glossary configs: {exc}"
        ) from exc
    if not isinstance(names, Mapping) or not isinstance(religious, Mapping):
        raise IDTranslationError("Canonical ID glossary configs must be objects")
    return validate_id_translation_glossary(
        {
            "canonical_names": copy.deepcopy(names.get("canonical_names", [])),
            "forbidden_name_variants": copy.deepcopy(
                names.get("forbidden_variants", {})
            ),
            "source_name_variants": copy.deepcopy(names.get("source_variants", {})),
            "religious_terms": copy.deepcopy(religious.get("terms", [])),
            "allah_only_semoga_is_invalid": religious.get(
                "allah_only_semoga_is_invalid",
                True,
            ),
        }
    )


def validate_aligned_turkish_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a final Turkish forced-aligned V2 schema.

    Additional block fields are allowed, but they become immutable too.  This
    lets alignment implementations retain richer evidence without weakening
    the ID translation boundary.
    """

    if not isinstance(schema, Mapping):
        raise IDTranslationError("Aligned Turkish schema must be a JSON object")
    trusted = _json_clone(dict(schema))

    schema_version = _require_nonempty_string(
        trusted.get("schema_version"), "schema_version"
    )
    if schema_version.split(".", 1)[0] != "2":
        raise IDTranslationError("ID translation requires a V2 schema_version")
    episode = _require_int(trusted.get("episode"), "episode", minimum=1)

    blocks = trusted.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise IDTranslationError("blocks must be a non-empty JSON array")
    declared_count = trusted.get("block_count", len(blocks))
    declared_count = _require_int(declared_count, "block_count", minimum=1)
    if declared_count != len(blocks):
        raise IDTranslationError(
            f"block_count mismatch: expected {len(blocks)}, got {declared_count}"
        )
    trusted["block_count"] = len(blocks)

    seen_uids: set[str] = set()
    previous_start_ms = -1
    active_blocks: list[tuple[int, str | None]] = []
    for position, block in enumerate(blocks, start=1):
        if not isinstance(block, dict):
            raise IDTranslationError(f"Block {position} must be a JSON object")
        missing = REQUIRED_ALIGNED_BLOCK_FIELDS.difference(block)
        if missing:
            raise IDTranslationError(
                f"Block {position} is missing immutable fields: "
                + ", ".join(sorted(missing))
            )
        forbidden = ID_OUTPUT_FIELDS.intersection(block)
        if forbidden:
            raise IDTranslationError(
                f"Block {position} contains ID-output fields: "
                + ", ".join(sorted(forbidden))
            )
        if "schema_sha256" in block or "schema_version" in block:
            raise IDTranslationError(
                f"Block {position} contains reserved pack identity fields"
            )

        uid = _require_nonempty_string(block.get("block_uid"), "block_uid")
        if uid in seen_uids:
            raise IDTranslationError(f"Duplicate block_uid in schema: {uid}")
        seen_uids.add(uid)

        index = _require_int(block.get("block_index"), "block_index", minimum=1)
        if index != position:
            raise IDTranslationError(
                f"Block order/index mismatch at position {position}: got {index}"
            )
        start_ms = _require_int(block.get("start_ms"), "start_ms", minimum=0)
        end_ms = _require_int(block.get("end_ms"), "end_ms", minimum=1)
        if end_ms <= start_ms:
            raise IDTranslationError(
                f"Block {position} end_ms must be greater than start_ms"
            )
        if start_ms < previous_start_ms:
            raise IDTranslationError(
                f"Block {position} starts before the preceding block"
            )
        previous_start_ms = start_ms
        try:
            current_speaker_id = speaker_id(block, f"Block {position}")
        except ValueError as exc:
            raise IDTranslationError(str(exc)) from exc
        active_blocks = [item for item in active_blocks if item[0] > start_ms]
        if any(
            overlap_is_unsafe(prior_speaker, current_speaker_id)
            for _, prior_speaker in active_blocks
        ):
            raise IDTranslationError(
                f"Block {position} has a same/unknown-speaker overlap"
            )
        active_blocks.append((end_ms, current_speaker_id))

        _require_nonempty_string(block.get("tr_text"), "tr_text")
        provenance = block.get("alignment_provenance")
        if not isinstance(provenance, dict) or not provenance:
            raise IDTranslationError(
                f"Block {position} alignment_provenance must be a non-empty object"
            )
        if "episode" in block:
            block_episode = _require_int(block.get("episode"), "episode", minimum=1)
            if block_episode != episode:
                raise IDTranslationError(
                    f"Block {position} episode mismatch: expected {episode}, "
                    f"got {block_episode}"
                )

    declared_hash = trusted.pop("schema_sha256", None)
    digest = _sha256_bytes(_canonical_json_bytes(trusted))
    if declared_hash is not None:
        _require_nonempty_string(declared_hash, "schema_sha256")
        if declared_hash != digest:
            raise IDTranslationError(
                f"schema_sha256 mismatch: expected {digest}, got {declared_hash}"
            )
    trusted["schema_sha256"] = digest
    return trusted


def build_id_translation_records(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build immutable, ordered ID-translation input records."""

    trusted = validate_aligned_turkish_schema(schema)
    records: list[dict[str, Any]] = []
    for block in trusted["blocks"]:
        record = copy.deepcopy(block)
        record["episode"] = trusted["episode"]
        record["schema_version"] = trusted["schema_version"]
        record["schema_sha256"] = trusted["schema_sha256"]
        records.append(record)
    return records


def _batch_records(
    records: Sequence[Mapping[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    batch_size = _require_int(batch_size, "batch_size", minimum=1)
    return [
        [copy.deepcopy(dict(record)) for record in records[offset : offset + batch_size]]
        for offset in range(0, len(records), batch_size)
    ]


def _manifest_for_pack(
    schema: Mapping[str, Any],
    batches: Sequence[Sequence[Mapping[str, Any]]],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = []
    for number, batch in enumerate(batches, start=1):
        name = f"batch_{number:03d}.jsonl"
        descriptors.append(
            {
                "input_file": name,
                "output_file": f"translated_batch_{number:03d}.jsonl",
                "block_count": len(batch),
                "first_block_uid": batch[0]["block_uid"],
                "last_block_uid": batch[-1]["block_uid"],
                "sha256": _sha256_bytes(payloads[name]),
            }
        )
    return {
        "format_version": ID_PACK_FORMAT_VERSION,
        "kind": "indonesian_translation_input",
        "episode": schema["episode"],
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "block_count": schema["block_count"],
        "batch_count": len(batches),
        "batches": descriptors,
        "file_sha256": {
            name: _sha256_bytes(payload)
            for name, payload in sorted(payloads.items())
        },
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_zip_atomic(destination: Path, payloads: Mapping[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(_zip_info(name), payload)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_id_translation_pack(
    schema: Mapping[str, Any],
    out_zip: str | os.PathLike[str],
    *,
    batch_size: int = DEFAULT_ID_BATCH_SIZE,
    glossary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an atomic, deterministic Indonesian-only input pack."""

    trusted = validate_aligned_turkish_schema(schema)
    trusted_glossary = (
        load_default_id_translation_glossary()
        if glossary is None
        else validate_id_translation_glossary(glossary)
    )
    records = build_id_translation_records(trusted)
    batches = _batch_records(records, batch_size)
    payloads: dict[str, bytes] = {
        "schema.json": _pretty_json_bytes(trusted),
        "glossary.json": _pretty_json_bytes(trusted_glossary),
        "ID_TRANSLATION_INSTRUCTIONS.md": ID_TRANSLATION_INSTRUCTIONS.encode("utf-8"),
    }
    for number, batch in enumerate(batches, start=1):
        payloads[f"batch_{number:03d}.jsonl"] = _jsonl_bytes(batch)
    manifest = _manifest_for_pack(trusted, batches, payloads)

    ordered_payloads = {
        "manifest.json": _pretty_json_bytes(manifest),
        "schema.json": payloads["schema.json"],
        "glossary.json": payloads["glossary.json"],
        "ID_TRANSLATION_INSTRUCTIONS.md": payloads[
            "ID_TRANSLATION_INSTRUCTIONS.md"
        ],
    }
    for number in range(1, len(batches) + 1):
        name = f"batch_{number:03d}.jsonl"
        ordered_payloads[name] = payloads[name]

    destination = Path(out_zip)
    _write_zip_atomic(destination, ordered_payloads)
    validated = validate_id_translation_pack(
        destination,
        expected_schema=trusted,
        expected_glossary=trusted_glossary,
    )
    if validated != manifest:  # Defensive: validator must reproduce it exactly.
        destination.unlink(missing_ok=True)
        raise IDTranslationError("Published ID translation manifest drifted")
    return copy.deepcopy(manifest)


def _validate_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name:
        raise IDTranslationError(f"Unsafe ZIP member path: {name}")
    if len(pure.parts) != 1:
        raise IDTranslationError(f"ZIP members must be top-level files: {name}")


def _open_checked_zip(path: Path) -> zipfile.ZipFile:
    if not path.is_file():
        raise IDTranslationError(f"ZIP not found: {path}")
    try:
        archive = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise IDTranslationError(f"Corrupt ZIP: {path}") from exc
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        archive.close()
        raise IDTranslationError("ZIP contains duplicate member names")
    try:
        for info in infos:
            _validate_member_name(info.filename)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise IDTranslationError(
                    f"ZIP symbolic links are forbidden: {info.filename}"
                )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise IDTranslationError(f"ZIP CRC failure: {bad_member}")
    except Exception:
        archive.close()
        raise
    return archive


def _parse_json(payload: bytes, *, member: str) -> Any:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IDTranslationError(f"{member} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise IDTranslationError(
            f"Invalid JSON in {member} at line {exc.lineno}: {exc.msg}"
        ) from exc


def _parse_jsonl(payload: bytes, *, member: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IDTranslationError(f"{member} is not valid UTF-8") from exc
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
            raise IDTranslationError(
                f"Invalid JSONL in {member} line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise IDTranslationError(
                f"{member} line {line_number} must be a JSON object"
            )
        records.append(record)
    return records


def validate_id_translation_pack(
    pack_path: str | os.PathLike[str],
    *,
    expected_schema: Mapping[str, Any] | None = None,
    expected_glossary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify exact members, hashes, schema identity, and every input record."""

    path = Path(pack_path)
    archive = _open_checked_zip(path)
    with archive:
        names = archive.namelist()
        required = {
            "manifest.json",
            "schema.json",
            "glossary.json",
            "ID_TRANSLATION_INSTRUCTIONS.md",
        }
        if not required.issubset(names):
            raise IDTranslationError(
                "ID translation pack is missing required members"
            )
        manifest = _parse_json(archive.read("manifest.json"), member="manifest.json")
        if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
            raise IDTranslationError("manifest.json has an invalid field set")
        if manifest.get("format_version") != ID_PACK_FORMAT_VERSION:
            raise IDTranslationError("Unsupported ID translation pack version")
        if manifest.get("kind") != "indonesian_translation_input":
            raise IDTranslationError("Invalid ID translation pack kind")

        packed_schema_raw = _parse_json(
            archive.read("schema.json"), member="schema.json"
        )
        packed_schema = validate_aligned_turkish_schema(packed_schema_raw)
        if archive.read("schema.json") != _pretty_json_bytes(packed_schema):
            raise IDTranslationError("schema.json is not deterministic canonical JSON")
        if expected_schema is not None:
            expected = validate_aligned_turkish_schema(expected_schema)
            if not _json_equal(packed_schema, expected):
                raise IDTranslationError("Packed schema differs from expected schema")

        packed_glossary_raw = _parse_json(
            archive.read("glossary.json"), member="glossary.json"
        )
        packed_glossary = validate_id_translation_glossary(packed_glossary_raw)
        if archive.read("glossary.json") != _pretty_json_bytes(packed_glossary):
            raise IDTranslationError(
                "glossary.json is not deterministic canonical JSON"
            )
        if expected_glossary is not None:
            expected = validate_id_translation_glossary(expected_glossary)
            if not _json_equal(packed_glossary, expected):
                raise IDTranslationError(
                    "Packed glossary differs from expected glossary"
                )

        instructions = archive.read("ID_TRANSLATION_INSTRUCTIONS.md")
        if instructions != ID_TRANSLATION_INSTRUCTIONS.encode("utf-8"):
            raise IDTranslationError("ID translation instructions were modified")

        batch_count = _require_int(
            manifest.get("batch_count"), "manifest batch_count", minimum=1
        )
        batch_names = [
            f"batch_{number:03d}.jsonl" for number in range(1, batch_count + 1)
        ]
        expected_names = required.union(batch_names)
        if set(names) != expected_names:
            raise IDTranslationError("ID translation pack member set is invalid")

        descriptors = manifest.get("batches")
        if not isinstance(descriptors, list) or len(descriptors) != batch_count:
            raise IDTranslationError("Manifest batch descriptors are invalid")

        payloads: dict[str, bytes] = {
            "schema.json": archive.read("schema.json"),
            "glossary.json": archive.read("glossary.json"),
            "ID_TRANSLATION_INSTRUCTIONS.md": instructions,
        }
        actual_batches: list[list[dict[str, Any]]] = []
        for number, name in enumerate(batch_names, start=1):
            payload = archive.read(name)
            payloads[name] = payload
            batch = _parse_jsonl(payload, member=name)
            if not batch:
                raise IDTranslationError(f"Input batch contains zero records: {name}")
            actual_batches.append(batch)
            descriptor = descriptors[number - 1]
            if not isinstance(descriptor, dict):
                raise IDTranslationError(f"Invalid batch descriptor: {name}")
            expected_descriptor_keys = {
                "input_file",
                "output_file",
                "block_count",
                "first_block_uid",
                "last_block_uid",
                "sha256",
            }
            if set(descriptor) != expected_descriptor_keys:
                raise IDTranslationError(f"Invalid batch descriptor fields: {name}")
            if descriptor.get("input_file") != name:
                raise IDTranslationError(f"Manifest input_file mismatch: {name}")
            if descriptor.get("output_file") != f"translated_batch_{number:03d}.jsonl":
                raise IDTranslationError(f"Manifest output_file mismatch: {name}")
            if descriptor.get("block_count") != len(batch):
                raise IDTranslationError(f"Manifest block_count mismatch: {name}")
            if descriptor.get("first_block_uid") != batch[0].get("block_uid"):
                raise IDTranslationError(f"Manifest first_block_uid mismatch: {name}")
            if descriptor.get("last_block_uid") != batch[-1].get("block_uid"):
                raise IDTranslationError(f"Manifest last_block_uid mismatch: {name}")
            if descriptor.get("sha256") != _sha256_bytes(payload):
                raise IDTranslationError(f"Manifest batch hash mismatch: {name}")

        declared_hashes = manifest.get("file_sha256")
        actual_hashes = {
            name: _sha256_bytes(payload)
            for name, payload in sorted(payloads.items())
        }
        if not isinstance(declared_hashes, dict) or declared_hashes != actual_hashes:
            raise IDTranslationError("Manifest file_sha256 map is invalid")

        expected_records = build_id_translation_records(packed_schema)
        actual_records = [record for batch in actual_batches for record in batch]
        if not _json_equal(actual_records, expected_records):
            raise IDTranslationError(
                "Packed ID records differ from the immutable aligned schema"
            )

        reproduced = _manifest_for_pack(packed_schema, actual_batches, payloads)
        if not _json_equal(manifest, reproduced):
            raise IDTranslationError("Manifest is not deterministic or self-consistent")
        return copy.deepcopy(manifest)


def validate_id_translation_records(
    schema: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    raise_on_error: bool = True,
) -> IDTranslationValidationResult:
    """Enforce exact immutable echoes and an ID-only translated payload."""

    trusted = validate_aligned_turkish_schema(schema)
    expected_records = build_id_translation_records(trusted)
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise IDTranslationError("Translated records must be a sequence")

    raw_records: list[Any] = list(records)
    issues: list[str] = []
    actual_uids: list[str | None] = []
    for position, record in enumerate(raw_records, start=1):
        if not isinstance(record, Mapping):
            issues.append(f"Record {position} is not a JSON object")
            actual_uids.append(None)
            continue
        uid = record.get("block_uid")
        if not isinstance(uid, str) or not uid.strip():
            issues.append(f"Record {position} has an invalid block_uid")
            actual_uids.append(None)
        else:
            actual_uids.append(uid)

    expected_uids = [record["block_uid"] for record in expected_records]
    expected_set = set(expected_uids)
    actual_counter = Counter(uid for uid in actual_uids if uid is not None)
    duplicates = sorted(uid for uid, count in actual_counter.items() if count > 1)
    missing = [uid for uid in expected_uids if actual_counter[uid] == 0]
    extra = sorted(uid for uid in actual_counter if uid not in expected_set)
    if len(raw_records) != len(expected_records):
        issues.append(
            f"Block count mismatch: expected {len(expected_records)}, "
            f"got {len(raw_records)}"
        )
    if missing:
        issues.append("Missing block_uid values: " + ", ".join(missing))
    if duplicates:
        issues.append("Duplicate block_uid values: " + ", ".join(duplicates))
    if extra:
        issues.append("Extra block_uid values: " + ", ".join(extra))
    if actual_uids != expected_uids:
        issues.append("Translated records are not in exact schema order")

    expected_by_uid = {record["block_uid"]: record for record in expected_records}
    normalized_by_uid: dict[str, dict[str, Any]] = {}
    for position, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, Mapping):
            continue
        record = dict(raw_record)
        uid = record.get("block_uid")
        if not isinstance(uid, str) or uid not in expected_by_uid:
            continue
        expected = expected_by_uid[uid]
        expected_fields = set(expected)
        actual_fields = set(record)
        missing_fields = sorted(expected_fields.difference(actual_fields))
        extra_fields = sorted(actual_fields.difference(expected_fields | ID_OUTPUT_FIELDS))
        if missing_fields:
            issues.append(
                f"Record {position} ({uid}) is missing immutable fields: "
                + ", ".join(missing_fields)
            )
        if extra_fields:
            issues.append(
                f"Record {position} ({uid}) has forbidden fields: "
                + ", ".join(extra_fields)
            )
        for field_name in sorted(expected_fields.intersection(actual_fields)):
            if not _json_equal(record[field_name], expected[field_name]):
                issues.append(
                    f"Record {position} ({uid}) changed immutable field "
                    f"{field_name}"
                )

        id_final = record.get("id_final")
        if not isinstance(id_final, str) or not id_final.strip():
            issues.append(f"Record {position} ({uid}) has empty id_final")
        review_required = record.get("review_required", False)
        if not isinstance(review_required, bool):
            issues.append(
                f"Record {position} ({uid}) review_required must be boolean"
            )
        note = record.get("note", "")
        if not isinstance(note, str):
            issues.append(f"Record {position} ({uid}) note must be a string")

        if uid not in normalized_by_uid and not missing_fields and not extra_fields:
            normalized = copy.deepcopy(expected)
            normalized["id_final"] = id_final
            normalized["review_required"] = review_required
            normalized["note"] = note
            normalized_by_uid[uid] = normalized

    unique_issues = tuple(dict.fromkeys(issues))
    frozen_records: Mapping[str, Mapping[str, Any]]
    if unique_issues:
        frozen_records = MappingProxyType({})
    else:
        frozen_records = MappingProxyType(
            {
                uid: MappingProxyType(copy.deepcopy(record))
                for uid, record in normalized_by_uid.items()
            }
        )
    result = IDTranslationValidationResult(
        schema_sha256=trusted["schema_sha256"],
        expected_block_count=len(expected_records),
        output_block_count=len(raw_records),
        issues=unique_issues,
        records_by_uid=frozen_records,
    )
    if raise_on_error and not result.ok:
        raise IDTranslationError("Indonesian translation validation failed", result)
    return result


def _expected_output_batch_names(
    input_manifest: Mapping[str, Any] | None,
) -> list[str] | None:
    if input_manifest is None:
        return None
    if not isinstance(input_manifest, Mapping):
        raise IDTranslationError("input_manifest must be a JSON object")
    descriptors = input_manifest.get("batches")
    if not isinstance(descriptors, list) or not descriptors:
        raise IDTranslationError("input_manifest has no batch descriptors")
    names: list[str] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise IDTranslationError("input_manifest batch descriptor is invalid")
        name = descriptor.get("output_file")
        if not isinstance(name, str) or OUTPUT_BATCH_RE.fullmatch(name) is None:
            raise IDTranslationError("input_manifest output_file is invalid")
        names.append(name)
    if len(names) != len(set(names)):
        raise IDTranslationError("input_manifest contains duplicate output files")
    return names


def load_and_validate_id_translation_zip(
    schema: Mapping[str, Any],
    translated_zip: str | os.PathLike[str],
    *,
    input_manifest: Mapping[str, Any] | None = None,
) -> IDTranslationValidationResult:
    """Load an ID-only output ZIP and apply every structural identity gate."""

    trusted = validate_aligned_turkish_schema(schema)
    if input_manifest is not None:
        if input_manifest.get("schema_sha256") != trusted["schema_sha256"]:
            raise IDTranslationError("input_manifest belongs to a different schema")
        if input_manifest.get("block_count") != trusted["block_count"]:
            raise IDTranslationError("input_manifest block_count mismatch")

    archive = _open_checked_zip(Path(translated_zip))
    with archive:
        names = archive.namelist()
        expected_batch_names = _expected_output_batch_names(input_manifest)
        actual_batch_names = sorted(
            name for name in names if OUTPUT_BATCH_RE.fullmatch(name)
        )
        if not actual_batch_names:
            raise IDTranslationError("Translated ZIP contains no output batches")
        for number, name in enumerate(actual_batch_names, start=1):
            match = OUTPUT_BATCH_RE.fullmatch(name)
            assert match is not None
            if int(match.group(1)) != number:
                raise IDTranslationError("Translated batch sequence has a gap")
        if expected_batch_names is not None and actual_batch_names != expected_batch_names:
            raise IDTranslationError("Translated batch filenames do not match the pack")
        expected_members = set(actual_batch_names) | {"translation_report.json"}
        if set(names) != expected_members:
            raise IDTranslationError("Translated ZIP has missing or extra members")

        records: list[dict[str, Any]] = []
        for batch_index, name in enumerate(actual_batch_names):
            batch = _parse_jsonl(archive.read(name), member=name)
            if not batch:
                raise IDTranslationError(f"Translated batch is empty: {name}")
            if input_manifest is not None:
                descriptor = input_manifest["batches"][batch_index]
                if descriptor.get("block_count") != len(batch):
                    raise IDTranslationError(
                        f"Translated batch block_count mismatch: {name}"
                    )
            records.extend(batch)

        result = validate_id_translation_records(trusted, records)
        report = _parse_json(
            archive.read("translation_report.json"),
            member="translation_report.json",
        )
        if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
            raise IDTranslationError("translation_report.json has an invalid field set")
        expected_report = {
            "format_version": ID_PACK_FORMAT_VERSION,
            "kind": "indonesian_translation_output",
            "schema_sha256": trusted["schema_sha256"],
            "total_input_blocks": trusted["block_count"],
            "total_output_blocks": len(records),
            "missing_block_count": 0,
            "duplicate_block_count": 0,
            "review_required_count": sum(
                record.get("review_required") is True for record in records
            ),
        }
        if not _json_equal(report, expected_report):
            raise IDTranslationError("translation_report.json values are invalid")
        return result


def create_id_translation_output_zip(
    schema: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    out_zip: str | os.PathLike[str],
    *,
    input_manifest: Mapping[str, Any] | None = None,
    batch_size: int = DEFAULT_ID_BATCH_SIZE,
) -> Path:
    """Create a deterministic output ZIP from already translated records.

    This helper is useful for trusted tooling and tests.  Untrusted Work output
    should go directly through :func:`load_and_validate_id_translation_zip`.
    """

    trusted = validate_aligned_turkish_schema(schema)
    result = validate_id_translation_records(trusted, records)
    ordered = result.ordered_records(trusted)

    expected_names = _expected_output_batch_names(input_manifest)
    if input_manifest is not None:
        if input_manifest.get("schema_sha256") != trusted["schema_sha256"]:
            raise IDTranslationError("input_manifest belongs to a different schema")
        sizes = [descriptor["block_count"] for descriptor in input_manifest["batches"]]
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
            raise IDTranslationError("input_manifest batch size is invalid")
        if sum(sizes) != len(ordered):
            raise IDTranslationError("input_manifest batch sizes do not cover records")
    else:
        batches = _batch_records(ordered, batch_size)
        sizes = [len(batch) for batch in batches]
        expected_names = [
            f"translated_batch_{number:03d}.jsonl"
            for number in range(1, len(sizes) + 1)
        ]
    assert expected_names is not None

    payloads: dict[str, bytes] = {}
    cursor = 0
    for name, size in zip(expected_names, sizes):
        payloads[name] = _jsonl_bytes(ordered[cursor : cursor + size])
        cursor += size
    report = {
        "format_version": ID_PACK_FORMAT_VERSION,
        "kind": "indonesian_translation_output",
        "schema_sha256": trusted["schema_sha256"],
        "total_input_blocks": trusted["block_count"],
        "total_output_blocks": len(ordered),
        "missing_block_count": 0,
        "duplicate_block_count": 0,
        "review_required_count": sum(
            record.get("review_required") is True for record in ordered
        ),
    }
    payloads["translation_report.json"] = _pretty_json_bytes(report)
    destination = Path(out_zip)
    _write_zip_atomic(destination, payloads)
    load_and_validate_id_translation_zip(
        trusted,
        destination,
        input_manifest=input_manifest,
    )
    return destination


# Explicit aliases retain compatibility with the imported protocol names.
create_id_only_translation_pack = create_id_translation_pack
load_and_validate_id_only_translation_zip = load_and_validate_id_translation_zip


__all__ = [
    "DEFAULT_ID_BATCH_SIZE",
    "ID_GLOSSARY_FIELDS",
    "ID_OUTPUT_FIELDS",
    "ID_PACK_FORMAT_VERSION",
    "ID_TRANSLATION_INSTRUCTIONS",
    "IDTranslationError",
    "IDTranslationValidationResult",
    "REQUIRED_ALIGNED_BLOCK_FIELDS",
    "build_id_translation_records",
    "create_id_only_translation_pack",
    "create_id_translation_output_zip",
    "create_id_translation_pack",
    "load_and_validate_id_only_translation_zip",
    "load_and_validate_id_translation_zip",
    "load_default_id_translation_glossary",
    "validate_aligned_turkish_schema",
    "validate_id_translation_glossary",
    "validate_id_translation_pack",
    "validate_id_translation_records",
]
