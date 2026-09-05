"""Strict, atomic V2 finalization from independently verified evidence.

The finalizer rebuilds the publishable schema from raw ASR, Turkish
corrections and the complete forced-alignment artifact. A saved PASS summary
is never accepted as timing evidence. The exact source/audio chain, all stage
markers, both translation packs and the canonical semantic-QA configuration
are hash-checked again immediately before the PASS report is committed.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
import uuid

import yaml

from .audio_review_v2 import (
    AudioReviewV2Error,
    validate_audio_review_v2_report,
)
from .download import atomic_write_json, load_valid_stage_marker, sha256_file, utc_now_iso
from .episode_archive import VIDEO_SUFFIXES, file_record, validate_episode_root
from .forced_align import validate_forced_alignment_data
from .id_translation import (
    IDTranslationValidationResult,
    load_and_validate_id_translation_zip,
    validate_aligned_turkish_schema,
    validate_id_translation_pack,
)
from .media import AUDIO_ALIGNMENT_VERSION, validate_audio_marker
from .mux import mux_softsubs
from .srt import SubtitleEntry, assert_srt_roundtrip, wrap_text, write_srt
from .subtitle_qa import assert_final_qa, run_subtitle_qa
from .timing_qa_v2 import assert_timing_qa_v2, run_timing_qa_v2
from .tr_correction import (
    compute_input_sha256,
    read_tr_correction_pack,
    validate_tr_correction_output,
)
from .v2_pipeline import build_strict_v2_artifacts


FINALIZATION_REPORT_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ALIGNMENT_HARD_ZERO_FIELDS = (
    "unaligned_lexical_token_count",
    "unaligned_word_count",
    "synthetic_timing_count",
    "interpolated_timing_count",
    "overlap_violation_count",
    "negative_word_gap_count",
    "missing_alignment_score_count",
    "low_alignment_score_count",
    "overlong_alignment_word_count",
    "outward_drift_violation_count",
    "low_score_edited_token_count",
    "unreviewed_deleted_token_count",
)
_ALIGNMENT_POLICY_FIELDS = (
    "timing_source",
    "engine",
    "whisperx_version",
    "model_name",
    "model_type",
    "language",
    "device",
    "interpolation",
    "min_word_score",
    "review_word_score",
    "edited_token_min_word_score",
    "max_word_duration_ms",
    "max_outward_drift_ms",
)
_ALIGNMENT_EDIT_SUMMARY_FIELDS = (
    "unchanged_token_count",
    "replaced_token_count",
    "inserted_token_count",
    "edited_token_count",
    "edited_token_ratio",
    "deleted_asr_token_count",
    "low_score_edited_token_count",
    "unreviewed_deleted_token_count",
)


class FinalizationV2Error(RuntimeError):
    """Raised before an incomplete or unverifiable V2 publication."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalizationV2Error(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise FinalizationV2Error(f"Non-finite JSON value is forbidden: {value}")


def _load_json_file(
    path_value: str | os.PathLike[str], label: str
) -> tuple[dict[str, Any], Path]:
    """Load one strict JSON object from a regular, non-symlink file."""

    try:
        path = Path(path_value)
        details = path.lstat()
    except (FileNotFoundError, OSError, TypeError) as exc:
        raise FinalizationV2Error(
            f"{label} is not a regular file: {path_value!r}"
        ) from exc
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_size <= 0
    ):
        raise FinalizationV2Error(f"{label} is not a non-empty regular file: {path}")
    try:
        parsed = json.loads(
            path.read_bytes().decode("utf-8-sig", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationV2Error(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise FinalizationV2Error(f"{label} root must be a JSON object")
    return parsed, path.resolve(strict=True)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise FinalizationV2Error(f"Duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_file(path: Path, label: str) -> dict[str, Any]:
    try:
        details = path.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise FinalizationV2Error(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_size <= 0
    ):
        raise FinalizationV2Error(f"{label} is not a non-empty regular file: {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FinalizationV2Error(f"{label} is unreadable or invalid YAML") from exc
    if not isinstance(value, dict):
        raise FinalizationV2Error(f"{label} root must be a mapping")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalizationV2Error(f"Cannot hash non-JSON evidence: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _digest_bound_report(
    report: Mapping[str, Any],
    *,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    """Return one strict PASS report after verifying its own digest."""

    if not isinstance(report, Mapping):
        raise FinalizationV2Error(f"{label} must be an object")
    trusted = copy.deepcopy(dict(report))
    digest = trusted.get(digest_field)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise FinalizationV2Error(f"{label} has no valid {digest_field}")
    unsigned = copy.deepcopy(trusted)
    unsigned.pop(digest_field, None)
    if _canonical_sha256(unsigned) != digest:
        raise FinalizationV2Error(f"{label} {digest_field} mismatch")
    if trusted.get("status") != "PASS":
        raise FinalizationV2Error(f"{label} is not PASS")
    return trusted


def _raw_policy_report(
    raw_data: Mapping[str, Any],
    *,
    raw_artifact_sha256: str,
    audio_sha256: str,
) -> dict[str, Any]:
    """Persist the exact canonical raw detector/coverage policy."""

    model = raw_data.get("model")
    settings = model.get("settings") if isinstance(model, Mapping) else None
    raw_coverage = raw_data.get("speech_coverage")
    coverage_config = (
        raw_coverage.get("config")
        if isinstance(raw_coverage, Mapping)
        else None
    )
    if not isinstance(settings, Mapping) or not isinstance(
        coverage_config, Mapping
    ):
        raise FinalizationV2Error(
            "Raw ASR does not preserve its model and speech-coverage policy"
        )
    model_settings = copy.deepcopy(dict(settings))
    embedded_coverage_config = copy.deepcopy(dict(coverage_config))
    review_uids = raw_data.get("hallucination_review_utterance_uids")
    if not isinstance(review_uids, list):
        raise FinalizationV2Error(
            "Raw ASR explicit audio-review UID evidence is malformed"
        )
    draft: dict[str, Any] = {
        "report_version": "2.0",
        "status": "PASS",
        "policy_label": model_settings.get("coverage_policy_label"),
        "raw_input_sha256": raw_data.get("input_sha256"),
        "raw_artifact_sha256": raw_artifact_sha256,
        "audio_sha256": audio_sha256,
        "model_settings": model_settings,
        "model_settings_sha256": _canonical_sha256(model_settings),
        "speech_coverage_config": embedded_coverage_config,
        "speech_coverage_config_sha256": _canonical_sha256(
            embedded_coverage_config
        ),
        "explicit_audio_review_uids": copy.deepcopy(review_uids),
    }
    draft["raw_policy_sha256"] = _canonical_sha256(draft)
    return draft


def _alignment_policy_report(
    forced_data: Mapping[str, Any],
    *,
    audio_sha256: str,
) -> dict[str, Any]:
    """Persist the validated acoustic policy and every hard-zero counter."""

    provenance = forced_data.get("provenance")
    counters = forced_data.get("report")
    if not isinstance(provenance, Mapping) or not isinstance(counters, Mapping):
        raise FinalizationV2Error(
            "Forced alignment lacks complete provenance or counters"
        )
    missing_policy = sorted(
        field for field in _ALIGNMENT_POLICY_FIELDS if field not in provenance
    )
    if missing_policy:
        raise FinalizationV2Error(
            "Forced alignment policy is incomplete: "
            f"missing={missing_policy}"
        )
    hard_zero_counters: dict[str, int] = {}
    for field in _ALIGNMENT_HARD_ZERO_FIELDS:
        value = counters.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise FinalizationV2Error(
                f"Forced alignment {field} must be integer zero"
            )
        hard_zero_counters[field] = value
    review_count = counters.get("review_alignment_score_count")
    if (
        isinstance(review_count, bool)
        or not isinstance(review_count, int)
        or review_count < 0
    ):
        raise FinalizationV2Error(
            "Forced alignment review_alignment_score_count is malformed"
        )
    selected_policy = {
        field: copy.deepcopy(provenance[field])
        for field in _ALIGNMENT_POLICY_FIELDS
    }
    full_provenance = copy.deepcopy(dict(provenance))
    draft: dict[str, Any] = {
        "report_version": "2.0",
        "status": "PASS",
        "audio_sha256": audio_sha256,
        "alignment_sha256": forced_data.get("alignment_sha256"),
        "policy": selected_policy,
        "provenance": full_provenance,
        "provenance_sha256": _canonical_sha256(full_provenance),
        "hard_zero_counters": hard_zero_counters,
        "review_alignment_score_count": review_count,
        "minimum_alignment_score": counters.get("minimum_alignment_score"),
        "maximum_alignment_word_duration_ms": counters.get(
            "maximum_alignment_word_duration_ms"
        ),
        "maximum_early_outward_drift_ms": counters.get(
            "maximum_early_outward_drift_ms"
        ),
        "maximum_late_outward_drift_ms": counters.get(
            "maximum_late_outward_drift_ms"
        ),
    }
    draft["alignment_policy_sha256"] = _canonical_sha256(draft)
    return draft


def _alignment_edit_audit_report(
    forced_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist exact per-utterance lexical edits and deletion authorization."""

    segments = forced_data.get("segments")
    counters = forced_data.get("report")
    if not isinstance(segments, list) or not isinstance(counters, Mapping):
        raise FinalizationV2Error(
            "Forced alignment lacks complete lexical edit evidence"
        )
    summary: dict[str, Any] = {}
    for field in _ALIGNMENT_EDIT_SUMMARY_FIELDS:
        if field not in counters:
            raise FinalizationV2Error(
                f"Forced alignment edit summary is missing {field}"
            )
        summary[field] = copy.deepcopy(counters[field])
    if (
        summary["low_score_edited_token_count"] != 0
        or summary["unreviewed_deleted_token_count"] != 0
    ):
        raise FinalizationV2Error(
            "Forced alignment has unsafe edited or deleted lexical tokens"
        )
    segment_audits: list[dict[str, Any]] = []
    for position, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping) or not isinstance(
            segment.get("edit_audit"), Mapping
        ):
            raise FinalizationV2Error(
                f"Forced alignment segment {position} lacks edit_audit"
            )
        segment_audits.append(
            {
                "segment_index": segment.get("segment_index"),
                "utterance_uid": segment.get("utterance_uid"),
                "asr_text": segment.get("asr_text"),
                "corrected_text": segment.get("text"),
                "deletion_audio_reviewed": segment.get(
                    "deletion_audio_reviewed"
                ),
                "edit_audit": copy.deepcopy(dict(segment["edit_audit"])),
            }
        )
    draft: dict[str, Any] = {
        "report_version": "2.0",
        "status": "PASS",
        "alignment_sha256": forced_data.get("alignment_sha256"),
        "segment_count": len(segment_audits),
        "summary": summary,
        "segments": segment_audits,
    }
    draft["alignment_edit_audit_sha256"] = _canonical_sha256(draft)
    return draft


def _episode_identity(episode: int) -> str:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise FinalizationV2Error("episode must be a positive integer")
    return f"Muhtemel Ask {episode}.Bolum"


def _require_workflow_directory(root: Path, name: str) -> Path:
    path = root / name
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise FinalizationV2Error(
            f"Required episode directory is missing: {path}"
        ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise FinalizationV2Error(f"Unsafe episode directory: {path}")
    if path.resolve(strict=True).parent != root.resolve(strict=True):
        raise FinalizationV2Error(f"Episode directory escapes its root: {path}")
    return path.resolve(strict=True)


def _exact_regular_file(
    supplied: str | os.PathLike[str], expected: Path, label: str
) -> Path:
    try:
        path = Path(supplied)
        details = path.lstat()
        resolved = path.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except (FileNotFoundError, OSError, TypeError) as exc:
        raise FinalizationV2Error(
            f"{label} is missing or unsafe: {supplied!r}"
        ) from exc
    if (
        stat.S_ISLNK(details.st_mode)
        or not stat.S_ISREG(details.st_mode)
        or details.st_size <= 0
    ):
        raise FinalizationV2Error(f"{label} is not a non-empty regular file: {path}")
    if resolved != expected_resolved:
        raise FinalizationV2Error(
            f"{label} must use the canonical path {expected}; got {path}"
        )
    return expected_resolved


def _safe_output_directory(path: Path, expected_parent: Path) -> None:
    if not path.exists():
        path.mkdir()
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise FinalizationV2Error(f"Unsafe output directory: {path}")
    if path.resolve(strict=True).parent != expected_parent.resolve(strict=True):
        raise FinalizationV2Error(f"Output directory escapes episode root: {path}")


def _canonical_output_paths(root: Path, episode_name: str) -> dict[str, Path]:
    final = root / "final"
    _safe_output_directory(final, root)
    subtitles = final / "subtitles"
    _safe_output_directory(subtitles, final)
    return {
        "mkv": final / f"{episode_name}.mkv",
        "id_srt": subtitles / f"{episode_name}-id.srt",
        "tr_srt": subtitles / f"{episode_name}-tr.srt",
        "report": final / f"{episode_name}_FINALIZATION_REPORT_V2.json",
    }


def _source_record(
    root: Path,
    source_video: str | os.PathLike[str],
    episode_name: str,
) -> tuple[Path, dict[str, Any]]:
    try:
        record = file_record(source_video, root)
    except Exception as exc:
        raise FinalizationV2Error(
            "Source video is missing, empty, or unsafe"
        ) from exc
    source = root / record["relative_path"]
    if (
        source.parent != root / "source"
        or source.stem != episode_name
        or source.suffix.casefold() not in VIDEO_SUFFIXES
    ):
        raise FinalizationV2Error(
            "Source video must be the episode-named file directly inside source/"
        )
    return source, record


def _validated_stage_marker(
    marker_path: Path,
    *,
    stage: str,
    expected_outputs: Mapping[str, Path],
    optional_output_keys: tuple[str, ...] = (),
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    raw, resolved_marker = _load_json_file(marker_path, f"{stage} stage marker")
    if resolved_marker != marker_path.resolve(strict=True):
        raise FinalizationV2Error(f"{stage} marker path changed during validation")
    input_sha = raw.get("input_sha256")
    if not isinstance(input_sha, str) or _SHA256_RE.fullmatch(input_sha) is None:
        raise FinalizationV2Error(f"{stage} marker has no valid input_sha256")
    if expected_input_sha256 is not None and input_sha != expected_input_sha256:
        raise FinalizationV2Error(f"{stage} marker input identity mismatch")
    allowed_keys = set(expected_outputs).union(optional_output_keys)
    raw_outputs = raw.get("outputs")
    if not isinstance(raw_outputs, dict) or set(raw_outputs).difference(allowed_keys):
        raise FinalizationV2Error(f"{stage} marker has unexpected output keys")
    marker = load_valid_stage_marker(
        marker_path,
        stage=stage,
        input_sha256=input_sha,
        required_output_keys=tuple(expected_outputs),
        optional_output_keys=optional_output_keys,
        allowed_root=marker_path.parent,
    )
    if marker is None:
        raise FinalizationV2Error(f"{stage} marker/output hash validation failed")
    for key, expected_path in expected_outputs.items():
        try:
            recorded = Path(str(marker["outputs"][key]["path"])).resolve(strict=True)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise FinalizationV2Error(
                f"{stage} marker output {key!r} is malformed"
            ) from exc
        if recorded != expected_path.resolve(strict=True):
            raise FinalizationV2Error(
                f"{stage} marker redirects {key!r} away from its canonical file"
            )
    return raw


def _validate_source_chain(
    root: Path,
    source: Path,
    source_record: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    source_dir = root / "source"
    metadata_path = source_dir / "source.metadata.json"
    marker_path = source_dir / "download.done.json"
    metadata, _ = _load_json_file(metadata_path, "Download source metadata")
    marker = _validated_stage_marker(
        marker_path,
        stage="download",
        expected_outputs={"video": source, "metadata": metadata_path},
        optional_output_keys=("captions",),
    )
    source_sha = source_record["sha256"]
    details = marker.get("details")
    if not isinstance(details, Mapping) or details.get("source_sha256") != source_sha:
        raise FinalizationV2Error(
            "Download marker is not bound to current source bytes"
        )
    required_metadata = {
        "sha256": source_sha,
        "file_size_bytes": source_record["size_bytes"],
        "source_file": source.name,
        "file_name": source.name,
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise FinalizationV2Error(
                f"Download source metadata {field} is not bound to current source"
            )
    try:
        metadata_source = Path(str(metadata.get("path", ""))).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise FinalizationV2Error("Download source metadata path is invalid") from exc
    if metadata_source != source.resolve(strict=True):
        raise FinalizationV2Error(
            "Download source metadata redirects the source path"
        )
    return metadata_path, marker_path, marker


def _validate_audio_chain(
    root: Path, source: Path, source_sha: str
) -> tuple[Path, Path, Path, dict[str, Any]]:
    prepare = root / "prepare"
    audio = prepare / "audio.flac"
    metadata_path = prepare / "audio.metadata.json"
    marker_path = prepare / "audio.done.json"
    for path, label in (
        (audio, "Canonical episode audio"),
        (metadata_path, "Audio metadata"),
        (marker_path, "Audio stage marker"),
    ):
        _exact_regular_file(path, path, label)
    if not validate_audio_marker(marker_path, source_sha256=source_sha):
        raise FinalizationV2Error("Audio marker/settings/hash validation failed")
    marker = _validated_stage_marker(
        marker_path,
        stage="audio",
        expected_outputs={"audio": audio, "metadata": metadata_path},
    )
    metadata, _ = _load_json_file(metadata_path, "Audio metadata")
    audio_sha = sha256_file(audio)
    details = marker.get("details")
    if not isinstance(details, Mapping):
        raise FinalizationV2Error("Audio marker details are missing")
    bindings = (
        (details.get("source_sha256"), source_sha, "marker source_sha256"),
        (details.get("audio_sha256"), audio_sha, "marker audio_sha256"),
        (metadata.get("source_sha256"), source_sha, "metadata source_sha256"),
        (metadata.get("sha256"), audio_sha, "metadata audio sha256"),
    )
    for actual, expected, label in bindings:
        if actual != expected:
            raise FinalizationV2Error(f"Audio {label} identity mismatch")
    timeline = metadata.get("source_timeline")
    if (
        not isinstance(timeline, Mapping)
        or timeline.get("alignment_version") != AUDIO_ALIGNMENT_VERSION
    ):
        raise FinalizationV2Error(
            "Audio metadata lacks the canonical source timeline"
        )
    try:
        metadata_audio = Path(str(metadata.get("path", ""))).resolve(strict=True)
        metadata_source = Path(str(metadata.get("source_path", ""))).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise FinalizationV2Error("Audio metadata paths are invalid") from exc
    if (
        metadata_audio != audio.resolve(strict=True)
        or metadata_source != source.resolve(strict=True)
    ):
        raise FinalizationV2Error("Audio metadata path binding mismatch")
    return audio, metadata_path, marker_path, metadata


def _translation_report(
    result: IDTranslationValidationResult,
    ordered_records: list[dict[str, Any]],
) -> dict[str, Any]:
    review_count = sum(
        record.get("review_required") is True for record in ordered_records
    )
    return {
        "passed": result.ok and review_count == 0,
        "schema_sha256": result.schema_sha256,
        "expected_block_count": result.expected_block_count,
        "output_block_count": result.output_block_count,
        "missing_block_count": 0,
        "duplicate_block_count": 0,
        "review_required_count": review_count,
    }


def _entries(
    blocks: list[dict[str, Any]],
    text_by_uid: Mapping[str, str],
    *,
    target_chars_per_line: int,
    hard_line_limit: int,
) -> list[SubtitleEntry]:
    entries: list[SubtitleEntry] = []
    for expected_index, block in enumerate(blocks, start=1):
        uid = str(block["block_uid"])
        text = text_by_uid.get(uid)
        if not isinstance(text, str) or not text.strip():
            raise FinalizationV2Error(f"Missing subtitle text for block {uid}")
        block_index = int(block["block_index"])
        if block_index != expected_index:
            raise FinalizationV2Error(
                f"Non-contiguous block_index: expected {expected_index}, "
                f"got {block_index}"
            )
        entries.append(
            SubtitleEntry(
                block_index,
                int(block["start_ms"]),
                int(block["end_ms"]),
                wrap_text(
                    text,
                    target=target_chars_per_line,
                    max_lines=2,
                    hard_limit=hard_line_limit,
                ),
            )
        )
    return entries


def _require_mux_pass(report: Mapping[str, Any], block_count: int) -> None:
    roundtrip = report.get("roundtrip")
    stream_hashes = report.get("stream_hashes")
    if not isinstance(roundtrip, Mapping) or not isinstance(stream_hashes, Mapping):
        raise FinalizationV2Error("MKV verification report is incomplete")
    source_hashes = stream_hashes.get("source")
    output_hashes = stream_hashes.get("output")
    if not (
        report.get("verified") is True
        and report.get("video_audio_stream_copy") is True
        and report.get("subtitle_order") == ["ind", "tur"]
        and report.get("indonesian_default") is True
        and report.get("turkish_default") is False
        and roundtrip.get("exact") is True
        and roundtrip.get("id_block_count") == block_count
        and roundtrip.get("tr_block_count") == block_count
        and stream_hashes.get("checked") is True
        and stream_hashes.get("match") is True
        and isinstance(source_hashes, list)
        and bool(source_hashes)
        and source_hashes == output_hashes
    ):
        raise FinalizationV2Error(
            "MKV stream-copy, subtitle round-trip, or A/V hash verification failed"
        )


def _episode_records(
    root: Path, paths: Mapping[str, Path]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        try:
            records[key] = file_record(path, root)
        except Exception as exc:
            raise FinalizationV2Error(
                f"Input evidence {key!r} changed or is unsafe"
            ) from exc
    return records


def _configuration_records(
    project_root: Path,
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        try:
            details = path.lstat()
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(
                project_root.resolve(strict=True)
            ).as_posix()
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise FinalizationV2Error(
                f"Configuration evidence {key!r} is unsafe"
            ) from exc
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_size <= 0
        ):
            raise FinalizationV2Error(
                f"Configuration evidence {key!r} is not a regular file"
            )
        records[key] = {
            "relative_path": relative,
            "size_bytes": details.st_size,
            "sha256": sha256_file(resolved),
        }
    return records


def _evidence_snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        try:
            details = path.lstat()
        except (FileNotFoundError, OSError) as exc:
            raise FinalizationV2Error(
                f"Evidence file {key!r} disappeared"
            ) from exc
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_size <= 0
        ):
            raise FinalizationV2Error(f"Evidence file {key!r} became unsafe")
        snapshot[key] = {
            "size_bytes": details.st_size,
            "sha256": sha256_file(path),
        }
    return snapshot


def _publish_transaction(
    staged: Mapping[str, Path],
    targets: Mapping[str, Path],
    *,
    report_target: Path,
    report: dict[str, Any],
    episode_root: Path,
    expected_output_records: Mapping[str, Mapping[str, Any]],
    evidence_paths: Mapping[str, Path],
    expected_evidence_snapshot: Mapping[str, Mapping[str, Any]],
    run_id: str,
) -> None:
    """Replace outputs, rehash every input, then write PASS last."""

    all_targets = {key: targets[key] for key in ("mkv", "id_srt", "tr_srt")}
    all_targets["report"] = report_target
    backups: dict[str, Path] = {}
    published: set[str] = set()
    try:
        for key, target in all_targets.items():
            if target.exists() or target.is_symlink():
                details = target.lstat()
                if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                    raise FinalizationV2Error(
                        f"Refusing to replace non-regular output: {target}"
                    )
                backup = target.with_name(
                    f".{target.name}.{run_id}.superseded"
                )
                os.replace(target, backup)
                backups[key] = backup

        for key in ("id_srt", "tr_srt", "mkv"):
            os.replace(staged[key], targets[key])
            published.add(key)
        actual_records = {
            key: file_record(targets[key], episode_root)
            for key in ("mkv", "id_srt", "tr_srt")
        }
        if actual_records != dict(expected_output_records):
            raise FinalizationV2Error(
                "Published V2 artifact records differ before PASS commit"
            )

        # This is the last expensive operation before the atomic PASS write.
        if _evidence_snapshot(evidence_paths) != dict(expected_evidence_snapshot):
            raise FinalizationV2Error(
                "A finalization evidence file changed before PASS commit"
            )
        atomic_write_json(report_target, report)
        published.add("report")
        try:
            readback = json.loads(report_target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FinalizationV2Error(
                "Finalization report JSON readback failed"
            ) from exc
        if readback != report:
            raise FinalizationV2Error("Finalization report JSON readback mismatch")
    except BaseException:
        for key in reversed(("report", "mkv", "tr_srt", "id_srt")):
            target = all_targets[key]
            if key in published and (target.exists() or target.is_symlink()):
                if target.is_symlink() or not target.is_file():
                    raise FinalizationV2Error(
                        f"Cannot safely roll back unexpected output: {target}"
                    )
                target.unlink()
            backup = backups.get(key)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FinalizationV2Error(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) <= 0
    ):
        raise FinalizationV2Error(f"{label} must be a positive number")
    return float(value)


def finalize_episode_v2(
    *,
    episode_root: str | os.PathLike[str],
    episode: int,
    source_video: str | os.PathLike[str],
    raw_asr_path: str | os.PathLike[str],
    forced_alignment_path: str | os.PathLike[str],
    tr_correction_pack: str | os.PathLike[str],
    tr_text_correction_output: str | os.PathLike[str],
    tr_correction_output: str | os.PathLike[str],
    audio_review_path: str | os.PathLike[str],
    aligned_schema: str | os.PathLike[str],
    id_translation_pack: str | os.PathLike[str],
    id_translation_zip: str | os.PathLike[str],
    series_config: str | os.PathLike[str],
    names_config: str | os.PathLike[str],
    religious_config: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create and atomically publish the only valid V2 episode layout.

    Every evidence/config argument is a path to its canonical workflow file;
    in-memory summaries are intentionally unsupported at this production
    boundary.
    """

    episode_name = _episode_identity(episode)
    root = validate_episode_root(episode_root, episode)
    source_dir = _require_workflow_directory(root, "source")
    prepare_dir = _require_workflow_directory(root, "prepare")
    translation_input_dir = _require_workflow_directory(root, "translation_input")
    translation_output_dir = _require_workflow_directory(root, "translation_output")
    project_root = root.parent.parent

    source, source_file_record = _source_record(root, source_video, episode_name)
    if source.parent != source_dir:
        raise FinalizationV2Error(
            "Source video escaped the canonical source directory"
        )
    source_metadata_path, download_marker_path, _ = _validate_source_chain(
        root, source, source_file_record
    )
    source_sha = source_file_record["sha256"]
    audio_path, audio_metadata_path, audio_marker_path, _ = _validate_audio_chain(
        root, source, source_sha
    )
    audio_sha = sha256_file(audio_path)

    raw_path = _exact_regular_file(
        raw_asr_path,
        prepare_dir / "raw_asr_v2.json",
        "Raw ASR V2 artifact",
    )
    forced_path = _exact_regular_file(
        forced_alignment_path,
        prepare_dir / "forced_alignment_v2.json",
        "Full forced-alignment artifact",
    )
    tr_pack_path = _exact_regular_file(
        tr_correction_pack,
        translation_input_dir / f"{episode_name}_TR_CORRECTION_PACK.zip",
        "Turkish correction input pack",
    )
    tr_text_output_path = _exact_regular_file(
        tr_text_correction_output,
        translation_output_dir / f"{episode_name}_TR_TEXT_CORRECTED.zip",
        "Text-only Turkish correction output",
    )
    tr_output_path = _exact_regular_file(
        tr_correction_output,
        translation_output_dir / f"{episode_name}_TR_CORRECTED.zip",
        "Turkish correction output",
    )
    audio_review_report_path = _exact_regular_file(
        audio_review_path,
        prepare_dir / "audio_review_v2.json",
        "Bounded Colab audio-review report",
    )
    schema_path = _exact_regular_file(
        aligned_schema,
        prepare_dir / "aligned_tr_schema_v2.json",
        "Aligned Turkish V2 schema",
    )
    id_pack_path = _exact_regular_file(
        id_translation_pack,
        translation_input_dir / f"{episode_name}_ID_TRANSLATION_PACK.zip",
        "Indonesian translation input pack",
    )
    id_zip_path = _exact_regular_file(
        id_translation_zip,
        translation_output_dir / f"{episode_name}_ID_TRANSLATED.zip",
        "Indonesian translation output",
    )

    # V2 remains isolated from the production V1 runtime and configuration.
    system_config_dir = project_root / "SYSTEM_V2_BETA" / "config"
    series_path = _exact_regular_file(
        series_config,
        system_config_dir / "series.yaml",
        "Series configuration",
    )
    names_path = _exact_regular_file(
        names_config,
        system_config_dir / "names.yaml",
        "Names configuration",
    )
    religious_path = _exact_regular_file(
        religious_config,
        system_config_dir / "religious_terms.yaml",
        "Religious-terms configuration",
    )
    series_data = _load_yaml_file(series_path, "Series configuration")
    names_data = _load_yaml_file(names_path, "Names configuration")
    religious_data = _load_yaml_file(
        religious_path, "Religious-terms configuration"
    )
    subtitle_config = series_data.get("subtitle")
    if not isinstance(subtitle_config, Mapping):
        raise FinalizationV2Error(
            "Series configuration has no subtitle mapping"
        )
    target_chars_per_line = _positive_int(
        subtitle_config.get("target_chars_per_line"),
        "subtitle.target_chars_per_line",
    )
    hard_line_limit = _positive_int(
        subtitle_config.get("qa_max_chars_per_line"),
        "subtitle.qa_max_chars_per_line",
    )
    preferred_max_cps = _positive_float(
        subtitle_config.get("preferred_max_cps"),
        "subtitle.preferred_max_cps",
    )
    if hard_line_limit < target_chars_per_line:
        raise FinalizationV2Error(
            "Subtitle hard line limit is below the target"
        )

    raw_data, _ = _load_json_file(raw_path, "Raw ASR V2 artifact")
    if (
        raw_data.get("episode") != episode
        or raw_data.get("audio_sha256") != audio_sha
    ):
        raise FinalizationV2Error(
            "Raw ASR belongs to another episode or audio file"
        )
    try:
        raw_audio_path = Path(
            str(raw_data.get("audio_path", ""))
        ).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise FinalizationV2Error("Raw ASR audio_path is invalid") from exc
    if raw_audio_path != audio_path.resolve(strict=True):
        raise FinalizationV2Error(
            "Raw ASR is not bound to the canonical audio path"
        )
    raw_input_sha = raw_data.get("input_sha256")
    if (
        not isinstance(raw_input_sha, str)
        or _SHA256_RE.fullmatch(raw_input_sha) is None
    ):
        raise FinalizationV2Error("Raw ASR has no valid input_sha256")
    raw_marker_path = prepare_dir / "raw_asr_v2.done.json"
    raw_marker = _validated_stage_marker(
        raw_marker_path,
        stage="raw_asr_v2",
        expected_outputs={"raw_asr_v2": raw_path},
        expected_input_sha256=raw_input_sha,
    )
    raw_details = raw_marker.get("details")
    if (
        not isinstance(raw_details, Mapping)
        or raw_details.get("audio_sha256") != audio_sha
    ):
        raise FinalizationV2Error(
            "Raw ASR marker is not bound to canonical audio"
        )

    expected_correction_sha = compute_input_sha256(
        raw_data.get("correction_utterances", []),
        raw_data.get("speech_hole_records", []),
        episode=episode,
        asr_hallucination_records=raw_data.get(
            "asr_hallucination_records", []
        ),
    )
    correction_pack = read_tr_correction_pack(
        tr_pack_path,
        expected_input_sha256=expected_correction_sha,
    )
    if correction_pack.manifest.get("episode") != episode:
        raise FinalizationV2Error(
            "Turkish correction pack episode mismatch"
        )
    correction_output = validate_tr_correction_output(
        tr_pack_path, tr_output_path
    )
    unresolved_tr_review_count = sum(
        record.get("review_required") is True
        for record in correction_output.records
    )
    if unresolved_tr_review_count:
        raise FinalizationV2Error(
            "Turkish correction output still requires review: "
            f"{unresolved_tr_review_count} record(s)"
        )
    try:
        acoustic_audio_review = validate_audio_review_v2_report(
            tr_pack_path,
            tr_text_output_path,
            tr_output_path,
            audio_review_report_path,
        )
    except AudioReviewV2Error as exc:
        raise FinalizationV2Error(
            f"Bounded Colab audio-review evidence is invalid: {exc}"
        ) from exc

    forced_data, _ = _load_json_file(
        forced_path, "Full forced-alignment artifact"
    )
    validate_forced_alignment_data(forced_data)
    if forced_data.get("audio_sha256") != audio_sha:
        raise FinalizationV2Error(
            "Forced alignment belongs to stale or different audio"
        )
    forced_marker_path = prepare_dir / "forced_alignment_v2.done.json"
    forced_marker = _validated_stage_marker(
        forced_marker_path,
        stage="forced_alignment_v2",
        expected_outputs={"forced_alignment_v2": forced_path},
    )
    forced_details = forced_marker.get("details")
    if not isinstance(forced_details, Mapping):
        raise FinalizationV2Error(
            "Forced-alignment marker details are missing"
        )
    if (
        forced_details.get("alignment_sha256")
        != forced_data.get("alignment_sha256")
    ):
        raise FinalizationV2Error(
            "Forced-alignment marker digest mismatch"
        )
    if (
        forced_details.get("correction_output_sha256")
        != correction_output.output_sha256
    ):
        raise FinalizationV2Error(
            "Forced alignment belongs to another correction output"
        )

    # Recompute speech coverage from independent VAD and complete aligned
    # words, resegment, rebuild schema, and rerun the pre-ID timing gates.
    artifacts = build_strict_v2_artifacts(
        raw_data,
        correction_output.records,
        forced_data,
        episode=episode,
        acoustic_audio_review=acoustic_audio_review,
    )
    schema_raw, _ = _load_json_file(
        schema_path, "Aligned Turkish V2 schema"
    )
    trusted_schema = validate_aligned_turkish_schema(schema_raw)
    if trusted_schema != artifacts.schema:
        raise FinalizationV2Error(
            "Persisted aligned schema differs from independently rebuilt V2 schema"
        )
    if trusted_schema.get("audio_sha256") != audio_sha:
        raise FinalizationV2Error(
            "Aligned schema audio identity mismatch"
        )

    raw_policy_v2 = _raw_policy_report(
        raw_data,
        raw_artifact_sha256=sha256_file(raw_path),
        audio_sha256=audio_sha,
    )
    alignment_policy_v2 = _alignment_policy_report(
        forced_data,
        audio_sha256=audio_sha,
    )
    alignment_edit_audit_v2 = _alignment_edit_audit_report(forced_data)
    audio_review_v2 = _digest_bound_report(
        artifacts.audio_review_report,
        digest_field="audio_review_sha256",
        label="V2 audio-review report",
    )
    strict_word_vad_v2 = _digest_bound_report(
        artifacts.strict_word_vad_report,
        digest_field="strict_word_vad_sha256",
        label="V2 strict word/VAD report",
    )
    if (
        audio_review_v2.get("pending_audio_review_count") != 0
        or strict_word_vad_v2.get("unsafe_word_count") != 0
        or strict_word_vad_v2.get(
            "confirmed_dialogue_coverage_fail_count"
        )
        != 0
    ):
        raise FinalizationV2Error(
            "Audio review or strict word/VAD evidence contains unresolved risk"
        )
    if (
        strict_word_vad_v2.get("audio_sha256") != audio_sha
        or strict_word_vad_v2.get("alignment_sha256")
        != forced_data.get("alignment_sha256")
        or strict_word_vad_v2.get("audio_review_sha256")
        != audio_review_v2.get("audio_review_sha256")
    ):
        raise FinalizationV2Error(
            "Strict word/VAD evidence identity chain mismatch"
        )
    if (
        artifacts.speech_coverage_report.get("audio_review_v2")
        != audio_review_v2
        or artifacts.speech_coverage_report.get("strict_word_vad_v2")
        != strict_word_vad_v2
    ):
        raise FinalizationV2Error(
            "Speech coverage does not bind the exact audio-review/VAD audits"
        )
    coverage_sha = _canonical_sha256(artifacts.speech_coverage_report)
    if trusted_schema.get("speech_coverage_sha256") != coverage_sha:
        raise FinalizationV2Error(
            "Aligned schema does not bind the recomputed speech coverage"
        )

    id_manifest = validate_id_translation_pack(
        id_pack_path,
        expected_schema=trusted_schema,
    )
    validation = load_and_validate_id_translation_zip(
        trusted_schema,
        id_zip_path,
        input_manifest=id_manifest,
    )
    ordered_records = validation.ordered_records(trusted_schema)
    id_review_count = sum(
        record.get("review_required") is True
        for record in ordered_records
    )
    if id_review_count:
        raise FinalizationV2Error(
            "Indonesian translation still requires review: "
            f"{id_review_count} block(s)"
        )
    id_by_uid = {
        str(record["block_uid"]): str(record["id_final"])
        for record in ordered_records
    }
    tr_by_uid = {
        str(block["block_uid"]): str(block["tr_text"])
        for block in trusted_schema["blocks"]
    }

    timing_report = run_timing_qa_v2(
        trusted_schema["blocks"],
        speech_coverage_report=artifacts.speech_coverage_report,
        alignment_report=artifacts.alignment_report,
        id_text_by_uid=id_by_uid,
    )
    assert_timing_qa_v2(timing_report)

    qa_blocks: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    for block, record in zip(
        trusted_schema["blocks"], ordered_records
    ):
        qa_block = copy.deepcopy(block)
        qa_block["schema_sha256"] = trusted_schema["schema_sha256"]
        qa_block["primary_text"] = block["tr_text"]
        qa_blocks.append(qa_block)
        qa_record = copy.deepcopy(record)
        qa_record["schema_sha256"] = trusted_schema["schema_sha256"]
        qa_record["tr_final"] = block["tr_text"]
        qa_records.append(qa_record)
    semantic_qa_report = run_subtitle_qa(
        qa_blocks,
        qa_records,
        names_config=names_data,
        religious_config=religious_data,
        preferred_max_cps=preferred_max_cps,
        line_limit=hard_line_limit,
    )
    assert_final_qa(semantic_qa_report)
    if semantic_qa_report.get("review_required_count") != 0:
        raise FinalizationV2Error(
            "Semantic QA still contains review-required blocks"
        )

    tr_entries = _entries(
        trusted_schema["blocks"],
        tr_by_uid,
        target_chars_per_line=target_chars_per_line,
        hard_line_limit=hard_line_limit,
    )
    id_entries = _entries(
        trusted_schema["blocks"],
        id_by_uid,
        target_chars_per_line=target_chars_per_line,
        hard_line_limit=hard_line_limit,
    )
    tr_timing = [
        (entry.index, entry.start_ms, entry.end_ms)
        for entry in tr_entries
    ]
    id_timing = [
        (entry.index, entry.start_ms, entry.end_ms)
        for entry in id_entries
    ]
    if tr_timing != id_timing:
        raise FinalizationV2Error(
            "TR and ID SRT timing identities differ"
        )

    episode_evidence_paths = {
        "source_video": source,
        "download_metadata": source_metadata_path,
        "download_marker": download_marker_path,
        "audio": audio_path,
        "audio_metadata": audio_metadata_path,
        "audio_marker": audio_marker_path,
        "raw_asr_v2": raw_path,
        "raw_asr_v2_marker": raw_marker_path,
        "forced_alignment_v2": forced_path,
        "forced_alignment_v2_marker": forced_marker_path,
        "tr_correction_pack": tr_pack_path,
        "tr_correction_output": tr_output_path,
        "aligned_schema_v2": schema_path,
        "id_translation_pack": id_pack_path,
        "id_translation_zip": id_zip_path,
    }
    config_paths = {
        "series_config": series_path,
        "names_config": names_path,
        "religious_config": religious_path,
    }
    all_evidence_paths = {**episode_evidence_paths, **config_paths}
    runtime_evidence_paths = {
        **all_evidence_paths,
        "tr_text_correction_output": tr_text_output_path,
        "bounded_audio_review_v2": audio_review_report_path,
    }
    initial_runtime_evidence_snapshot = _evidence_snapshot(
        runtime_evidence_paths
    )
    initial_evidence_snapshot = {
        key: copy.deepcopy(initial_runtime_evidence_snapshot[key])
        for key in all_evidence_paths
    }
    input_records = _episode_records(root, episode_evidence_paths)
    if input_records["source_video"] != source_file_record:
        raise FinalizationV2Error(
            "Source video changed during evidence validation"
        )
    configuration_records = _configuration_records(
        project_root, config_paths
    )

    paths = _canonical_output_paths(root, episode_name)
    run_id = uuid.uuid4().hex
    staged = {
        "id_srt": paths["id_srt"].with_name(
            f".{paths['id_srt'].stem}.{run_id}.staged.srt"
        ),
        "tr_srt": paths["tr_srt"].with_name(
            f".{paths['tr_srt'].stem}.{run_id}.staged.srt"
        ),
        "mkv": paths["mkv"].with_name(
            f".{paths['mkv'].stem}.{run_id}.staged.mkv"
        ),
    }
    try:
        write_srt(staged["tr_srt"], tr_entries)
        write_srt(staged["id_srt"], id_entries)
        assert_srt_roundtrip(staged["tr_srt"], tr_entries)
        assert_srt_roundtrip(staged["id_srt"], id_entries)

        mux_report = dict(
            mux_softsubs(
                source,
                staged["id_srt"],
                staged["tr_srt"],
                staged["mkv"],
                verify_stream_hashes=True,
            )
        )
        _require_mux_pass(
            mux_report, trusted_schema["block_count"]
        )

        output_records = {
            key: {
                "relative_path": paths[key].relative_to(root).as_posix(),
                "size_bytes": staged[key].stat().st_size,
                "sha256": sha256_file(staged[key]),
            }
            for key in ("mkv", "id_srt", "tr_srt")
        }
        sanitized_mux_report = {
            "verified": True,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": copy.deepcopy(
                dict(mux_report["roundtrip"])
            ),
            "stream_hashes": copy.deepcopy(
                dict(mux_report["stream_hashes"])
            ),
            "output_path": output_records["mkv"]["relative_path"],
            "output_size_bytes": output_records["mkv"]["size_bytes"],
        }
        report = {
            "report_version": FINALIZATION_REPORT_VERSION,
            "status": "PASS",
            "completed_at": utc_now_iso(),
            "episode": episode,
            "episode_name": episode_name,
            "schema_version": trusted_schema["schema_version"],
            "schema_sha256": trusted_schema["schema_sha256"],
            "audio_sha256": audio_sha,
            "alignment_sha256": forced_data["alignment_sha256"],
            "speech_coverage_sha256": coverage_sha,
            "block_count": trusted_schema["block_count"],
            "input_files": input_records,
            "configuration_files": configuration_records,
            "input_hashes": {
                key: value["sha256"]
                for key, value in sorted(
                    initial_evidence_snapshot.items()
                )
            },
            "evidence_validation": {
                "source_audio_chain_exact": True,
                "full_forced_alignment_revalidated": True,
                "bounded_audio_review_revalidated": True,
                "bounded_audio_review_sha256": acoustic_audio_review[
                    "audio_review_sha256"
                ],
                "speech_coverage_recomputed": True,
                "schema_rebuilt_exact": True,
                "semantic_qa_config_hash_bound": True,
                "evidence_rehashed_immediately_before_pass": True,
            },
            "tr_correction_validation": {
                # ``output_sha256`` binds the exact ZIP bytes, as every other
                # report file digest does.  The manifest's canonical records
                # digest remains explicit instead of overloading that name.
                "output_sha256": initial_evidence_snapshot[
                    "tr_correction_output"
                ]["sha256"],
                "records_sha256": correction_output.output_sha256,
                "record_count": len(correction_output.records),
                "review_required_count": 0,
                "audio_reviewed_count": sum(
                    record.get("audio_reviewed") is True
                    for record in correction_output.records
                ),
                "pending_audio_review_count": sum(
                    record.get("review_disposition")
                    == "pending_audio_review"
                    for record in correction_output.records
                ),
                "confirmed_dialogue_count": sum(
                    record.get("review_disposition")
                    == "confirmed_dialogue"
                    for record in correction_output.records
                ),
                "reviewed_non_dialogue_count": sum(
                    record.get("review_disposition")
                    == "reviewed_non_dialogue"
                    for record in correction_output.records
                ),
                "discarded_asr_hallucination_count": sum(
                    record.get("review_disposition")
                    == "discarded_asr_hallucination"
                    for record in correction_output.records
                ),
            },
            "id_translation_validation": _translation_report(
                validation, ordered_records
            ),
            "raw_policy_v2": raw_policy_v2,
            "alignment_policy_v2": alignment_policy_v2,
            "alignment_edit_audit_v2": alignment_edit_audit_v2,
            "audio_review_v2": audio_review_v2,
            "strict_word_vad_v2": strict_word_vad_v2,
            "speech_coverage_v2": copy.deepcopy(
                artifacts.speech_coverage_report
            ),
            "timing_qa_v2": timing_report,
            "semantic_subtitle_qa": semantic_qa_report,
            "timing_identity": {
                "forced_alignment": True,
                "independent_speech_coverage": True,
                "tr_id_identical": True,
                "srt_roundtrip_exact": True,
            },
            "mkv_verification": sanitized_mux_report,
            "outputs": output_records,
        }
        _publish_transaction(
            staged,
            paths,
            report_target=paths["report"],
            report=report,
            episode_root=root,
            expected_output_records=output_records,
            evidence_paths=runtime_evidence_paths,
            expected_evidence_snapshot=initial_runtime_evidence_snapshot,
            run_id=run_id,
        )

        actual_outputs = {
            key: file_record(paths[key], root)
            for key in ("mkv", "id_srt", "tr_srt")
        }
        if actual_outputs != output_records:
            raise FinalizationV2Error(
                "Published V2 artifact records changed"
            )
        return report
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


__all__ = [
    "FINALIZATION_REPORT_VERSION",
    "FinalizationV2Error",
    "finalize_episode_v2",
]
