from __future__ import annotations

import re
from pathlib import Path

pipeline_path = Path("src/mas/pipeline.py")
text = pipeline_path.read_text(encoding="utf-8")

old_source = '''        source_input = sha256_json(
            {
                "source_url": source_url or "",
                "fixture": fixture,
                "existing": [
                    (path.name, sha256_file(path))
                    for path in sorted((directory / "source").glob("*"))
                    if path.is_file()
                ],
            }
        )
'''
new_source = '''        if fixture:
            source_input = sha256_json({"fixture": True})
        elif source_url is None and source_marker.exists():
            previous_source = read_json(source_marker)
            previous_path = Path(previous_source["path"])
            source_input = sha256_json(
                {
                    "source_url": previous_source.get("source_url") or "",
                    "source_sha256": (
                        sha256_file(previous_path) if previous_path.exists() else "missing"
                    ),
                }
            )
        else:
            source_input = sha256_json(
                {
                    "source_url": source_url or "",
                    "existing": [
                        (path.name, sha256_file(path))
                        for path in sorted((directory / "source").glob("*"))
                        if path.is_file()
                    ],
                }
            )
'''
if old_source in text:
    text = text.replace(old_source, new_source)

old_fingerprint = '''    def config_fingerprint() -> str:
        files = sorted(path for path in rules_dir().rglob("*") if path.is_file())
        return sha256_json(
            [
                PIPELINE_VERSION,
                SCHEMA_VERSION,
                RULES_VERSION,
                [
                    (str(path.relative_to(rules_dir())), sha256_file(path))
                    for path in files
                ],
            ]
        )
'''
new_fingerprint = '''    def rules_fingerprint() -> str:
        files = sorted(path for path in rules_dir().rglob("*") if path.is_file())
        return sha256_json(
            [
                RULES_VERSION,
                [
                    (str(path.relative_to(rules_dir())), sha256_file(path))
                    for path in files
                ],
            ]
        )


    def stage_config_hash(stage: str) -> str:
        base = [PIPELINE_VERSION, SCHEMA_VERSION, stage]
        if stage == "asr":
            base.extend(
                [
                    os.environ.get("MAS_ASR_MODEL", "large-v3"),
                    "word_timestamps=true",
                    "vad_filter=true",
                    "beam_size=5",
                ]
            )
        if stage in {
            "reconcile",
            "tr_pack",
            "tr_return",
            "align",
            "id_pack",
            "id_return",
            "finalize",
            "qc",
        }:
            base.append(rules_fingerprint())
        return sha256_json(base)
'''
if old_fingerprint in text:
    text = text.replace(old_fingerprint, new_fingerprint)
    text = text.replace("        config_hash = config_fingerprint()\n", "")
    for stage in (
        "source",
        "audio",
        "asr",
        "reconcile",
        "tr_pack",
        "tr_return",
        "align",
        "id_pack",
        "id_return",
        "finalize",
        "qc",
    ):
        pattern = rf'(\n\s+"{stage}",\n\s+[^\n]+,\n)\s+config_hash,'
        text, _ = re.subn(
            pattern,
            rf'\1            stage_config_hash("{stage}"),',
            text,
            count=1,
        )

old_started = '''        started = time.monotonic()
        try:
            metadata = action() or {}
'''
new_started = '''        started = time.monotonic()
        logs_directory = state_path.parent.parent / "logs"
        logs_directory.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().replace(":", "").replace("+00:00", "Z")
        log_path = logs_directory / f"{stamp}_{name}.log"

        def log(event: str, **payload: object) -> None:
            row = {"timestamp": utc_now(), "stage": name, "event": event, **payload}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\\n")

        log("start", input_sha256=input_hash, config_sha256=config_hash)
        try:
            metadata = action() or {}
'''
if old_started in text:
    text = text.replace(old_started, new_started)
    text = text.replace(
        '''            save(state_path, state)
            print(f"[OK] {name.upper()}")
        except Exception as exc:
''',
        '''            save(state_path, state)
            log("complete", outputs=[str(output) for output in outputs], metadata=metadata)
            print(f"[OK] {name.upper()}")
        except Exception as exc:
''',
        1,
    )
    text = text.replace(
        '''            save(state_path, state)
            raise
''',
        '''            save(state_path, state)
            log("failed", error=str(exc))
            raise
''',
        1,
    )

pipeline_path.write_text(text, encoding="utf-8")

Path("src/mas/packs/validate.py").write_text(
    '''from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ..errors import PackageValidationError


def read_package(path: Path) -> tuple[dict, list[dict]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if not {"manifest.json", "records.json"}.issubset(names):
                raise PackageValidationError(
                    "ZIP must contain manifest.json and records.json"
                )
            manifest = json.loads(archive.read("manifest.json"))
            records = json.loads(archive.read("records.json"))
    except PackageValidationError:
        raise
    except Exception as exc:
        raise PackageValidationError(f"Invalid ZIP: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(records, list):
        raise PackageValidationError("manifest.json must be an object and records.json a list")
    return manifest, records


def validate_returned(
    path: Path,
    expected_manifest: dict,
    expected_records: list[dict],
) -> list[dict]:
    manifest, records = read_package(path)
    for field in (
        "episode",
        "kind",
        "schema_version",
        "schema_sha256",
        "input_sha256",
        "record_count",
    ):
        if manifest.get(field) != expected_manifest.get(field):
            raise PackageValidationError(
                f"{field} mismatch: expected {expected_manifest.get(field)!r}, "
                f"got {manifest.get(field)!r}"
            )
    if len(records) != len(expected_records):
        raise PackageValidationError(
            f"record count mismatch: expected {len(expected_records)}, got {len(records)}"
        )
    editable = "tr_text" if manifest.get("kind") == "TR" else "id_text"
    schema_version = manifest.get("schema_version")
    schema_hash = manifest.get("schema_sha256")
    seen = set()
    for position, (before, after) in enumerate(zip(expected_records, records)):
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise PackageValidationError(f"record {position} is not an object")
        if after.get("block_uid") != before.get("block_uid"):
            raise PackageValidationError(f"block_uid/order changed at record {position}")
        if after.get("block_index") != before.get("block_index"):
            raise PackageValidationError(f"block_index/order changed at record {position}")
        for field, value in before.items():
            if field in {editable, "schema_version", "schema_sha256"}:
                continue
            if after.get(field) != value:
                raise PackageValidationError(
                    f"immutable field changed at record {position}, "
                    f"uid={before.get('block_uid')}: {field}"
                )
        if after.get("schema_version") != schema_version:
            raise PackageValidationError(
                f"immutable field changed at record {position}, "
                f"uid={before.get('block_uid')}: schema_version"
            )
        if after.get("schema_sha256") != schema_hash:
            raise PackageValidationError(
                f"immutable field changed at record {position}, "
                f"uid={before.get('block_uid')}: schema_sha256"
            )
        allowed = set(before) | {editable, "schema_version", "schema_sha256"}
        extras = set(after) - allowed
        if extras:
            raise PackageValidationError(
                f"unexpected fields at record {position}, "
                f"uid={before.get('block_uid')}: {sorted(extras)}"
            )
        uid = after.get("block_uid")
        if uid in seen:
            raise PackageValidationError("duplicate block_uid in returned package")
        seen.add(uid)
    return records
''',
    encoding="utf-8",
)

print("Applied resume, scoped invalidation, structured logging, and strict pack fixes.")
