"""Verified single-copy episode archive with exact-plan cleanup.

Archive v2 deliberately keeps only three episode files::

    final/Muhtemel Ask X.Bolum.mkv
    final/subtitles/Muhtemel Ask X.Bolum-id.srt
    final/subtitles/Muhtemel Ask X.Bolum-tr.srt

The MKV contains the same Indonesian and Turkish subtitle tracks, so playback
does not depend on sidecar discovery.  The standalone SRT files are retained
once, in a subdirectory, for editing and export.  Destructive cleanup is
possible only after fresh stream-copy and subtitle round-trip verification,
an external atomic READY receipt, an unchanged tree snapshot, and an exact
episode-bound confirmation phrase.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any

from .download import atomic_write_json, sha256_json, utc_now_iso
from .episode_archive import (
    ArchiveError as ArchiveV2Error,
    VIDEO_SUFFIXES,
    file_record,
    validate_episode_root,
    verify_recorded_file,
)
from .mux import compute_av_stream_hashes, mux_softsubs, verify_mkv_roundtrip
from .raw_asr import validate_publishable_raw_asr_v2_policy
from .transcribe import TranscriptionError


RECEIPT_VERSION = 2
PLAN_VERSION = 2
LAYOUT_VERSION = 2

_PIPELINE_DIRECTORIES = frozenset(
    {
        "source",
        "prepare",
        "translation_input",
        "translation_output",
        "review",
        "final",
    }
)
_ALLOWED_SUBDIRECTORIES = frozenset({"final/subtitles"})
_ALLOWED_SUFFIXES = {
    "source": frozenset({".json", ".srt", ".vtt"}) | VIDEO_SUFFIXES,
    "prepare": frozenset({".flac", ".json", ".wav"}),
    "translation_input": frozenset({".json", ".zip"}),
    "translation_output": frozenset({".json", ".zip"}),
    "review": frozenset({".xlsx"}),
    "final": frozenset({".json", ".mkv", ".srt"}),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V2_INPUT_KEYS = frozenset(
    {
        "source_video",
        "download_metadata",
        "download_marker",
        "audio",
        "audio_metadata",
        "audio_marker",
        "raw_asr_v2",
        "raw_asr_v2_marker",
        "forced_alignment_v2",
        "forced_alignment_v2_marker",
        "tr_correction_pack",
        "tr_correction_output",
        "aligned_schema_v2",
        "id_translation_pack",
        "id_translation_zip",
    }
)
_V2_CONFIGURATION_KEYS = frozenset(
    {"series_config", "names_config", "religious_config"}
)
_TIMING_ZERO_FIELDS = (
    "unresolved_speech_region_count",
    "unaligned_word_count",
    "synthetic_timing_count",
    "negative_word_gap_count",
    "missing_alignment_score_count",
    "low_alignment_score_count",
    "overlong_alignment_word_count",
    "outward_drift_violation_count",
    "low_score_edited_token_count",
    "unreviewed_deleted_token_count",
    "short_cue_count",
    "high_cps_tr_count",
    "high_cps_id_count",
    "overlap_count",
    "invalid_timing_count",
    "alignment_provenance_mismatch_count",
    "adjacent_short_duplicate_count",
    "missing_id_count",
)
_SEMANTIC_ZERO_FIELDS = (
    "missing_translation_count",
    "duplicate_translation_count",
    "extra_translation_count",
    "positional_translation_mismatch_count",
    "overlap_count",
    "early_start_count",
    "early_end_count",
    "empty_text_count",
    "more_than_two_lines_count",
    "line_over_84_count",
    "mixed_speaker_block_count",
    "unresolved_internal_gap_count",
    "schema_mismatch_count",
    "uid_mismatch_count",
    "order_mismatch_count",
    "timing_mismatch_count",
    "timing_error_count",
    "special_name_mismatch_count",
    "numeric_mismatch_count",
    "money_mismatch_count",
    "religious_expression_mismatch_count",
    "allah_preservation_failure_count",
    "music_speech_handling_count",
    "utf8_error_count",
    "srt_structure_error_count",
    "anchor_mismatch_count",
    "review_required_count",
)
_FORCED_ALIGNMENT_ZERO_FIELDS = (
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
)
_COVERAGE_CONFIG_FIELDS = frozenset(
    {
        "vad_merge_gap_ms",
        "word_padding_ms",
        "word_merge_gap_ms",
        "min_hole_ms",
        "min_region_coverage_ratio",
        "rescue_padding_ms",
        "rescue_merge_gap_ms",
        "max_word_outside_speech_ms",
    }
)
_RAW_POLICY_KEYS = frozenset(
    {
        "report_version",
        "status",
        "policy_label",
        "raw_input_sha256",
        "raw_artifact_sha256",
        "audio_sha256",
        "model_settings",
        "model_settings_sha256",
        "speech_coverage_config",
        "speech_coverage_config_sha256",
        "explicit_audio_review_uids",
        "raw_policy_sha256",
    }
)
_ALIGNMENT_POLICY_KEYS = frozenset(
    {
        "report_version",
        "status",
        "audio_sha256",
        "alignment_sha256",
        "policy",
        "provenance",
        "provenance_sha256",
        "hard_zero_counters",
        "review_alignment_score_count",
        "minimum_alignment_score",
        "maximum_alignment_word_duration_ms",
        "maximum_early_outward_drift_ms",
        "maximum_late_outward_drift_ms",
        "alignment_policy_sha256",
    }
)
_ALIGNMENT_POLICY_FIELDS = frozenset(
    {
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
    }
)
_ALIGNMENT_REPORT_ZERO_FIELDS = _FORCED_ALIGNMENT_ZERO_FIELDS + (
    "low_score_edited_token_count",
    "unreviewed_deleted_token_count",
)
_ALIGNMENT_EDIT_SUMMARY_FIELDS = frozenset(
    {
        "unchanged_token_count",
        "replaced_token_count",
        "inserted_token_count",
        "edited_token_count",
        "edited_token_ratio",
        "deleted_asr_token_count",
        "low_score_edited_token_count",
        "unreviewed_deleted_token_count",
    }
)
_AUDIO_REVIEW_KEYS = frozenset(
    {
        "report_version",
        "status",
        "reviewed_outcome_count",
        "confirmed_dialogue_count",
        "reviewed_non_dialogue_count",
        "discarded_asr_hallucination_count",
        "pending_audio_review_count",
        "outcomes",
        "audio_review_sha256",
    }
)
_STRICT_WORD_VAD_KEYS = frozenset(
    {
        "report_version",
        "status",
        "policy",
        "raw_asr_input_sha256",
        "audio_sha256",
        "alignment_sha256",
        "audio_review_sha256",
        "aligned_word_count",
        "confirmed_dialogue_exception_word_count",
        "unsafe_word_count",
        "confirmed_dialogue_required_coverage_count",
        "confirmed_dialogue_coverage_fail_count",
        "confirmed_dialogue_exception_words",
        "confirmed_dialogue_required_coverage",
        "unsafe_words",
        "strict_word_vad_sha256",
    }
)


def _episode_identity(episode: int) -> tuple[int, str]:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise ArchiveV2Error("EPISODE must be a positive integer")
    return episode, f"Muhtemel Ask {episode}.Bolum"


def expected_cleanup_confirmation_v2(episode: int) -> str:
    """Return the exact episode-bound confirmation used by archive v2."""

    episode, _ = _episode_identity(episode)
    return f"DELETE EPISODE {episode} INTERMEDIATES"


def canonical_archive_paths(
    episode_root: str | os.PathLike[str], episode: int
) -> dict[str, Path]:
    """Return the only three paths retained inside a completed episode."""

    root = validate_episode_root(episode_root, episode)
    _, episode_name = _episode_identity(episode)
    return {
        "mkv": root / "final" / f"{episode_name}.mkv",
        "id_srt": root / "final" / "subtitles" / f"{episode_name}-id.srt",
        "tr_srt": root / "final" / "subtitles" / f"{episode_name}-tr.srt",
    }


def _canonical_relatives(episode: int) -> dict[str, str]:
    _, episode_name = _episode_identity(episode)
    return {
        "mkv": f"final/{episode_name}.mkv",
        "id_srt": f"final/subtitles/{episode_name}-id.srt",
        "tr_srt": f"final/subtitles/{episode_name}-tr.srt",
    }


def _safe_relative(
    root: Path,
    raw: Any,
    label: str,
    *,
    allow_missing_parent: bool = False,
) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ArchiveV2Error(f"{label} has an invalid relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArchiveV2Error(f"{label} is not a normalized relative path: {raw!r}")
    relative = pure.as_posix()
    candidate = root.joinpath(*pure.parts)
    try:
        parent = candidate.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        if not allow_missing_parent:
            raise ArchiveV2Error(f"{label} parent is missing: {relative}") from exc
        # A completed cleanup legitimately removes the historical source
        # directory.  The normalized PurePosixPath above keeps this lexical
        # candidate inside root even when there is no parent left to resolve.
        return relative, candidate
    if parent != root and root not in parent.parents:
        raise ArchiveV2Error(f"{label} escapes the episode folder: {relative}")
    return relative, candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise ArchiveV2Error(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ArchiveV2Error(f"{label} is not a regular non-symlink file: {path}")
    if details.st_size > 32 * 1024 * 1024:
        raise ArchiveV2Error(f"{label} is unexpectedly large")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveV2Error(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ArchiveV2Error(f"{label} root must be a JSON object")
    return value


def _receipt_path(
    receipt_path: str | os.PathLike[str], root: Path, episode_name: str
) -> Path:
    expected = (
        root.parent.parent
        / "ARCHIVE_REPORTS"
        / f"{episode_name}_ARCHIVE_RECEIPT_V2.json"
    )
    supplied = Path(os.path.abspath(receipt_path))
    if supplied != expected:
        raise ArchiveV2Error(f"Archive v2 receipt path must be exactly {expected}")
    if supplied.is_symlink():
        raise ArchiveV2Error(f"Refusing symlinked archive v2 receipt: {supplied}")
    reports = expected.parent
    if reports.exists() or reports.is_symlink():
        details = reports.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ArchiveV2Error(f"Refusing unsafe ARCHIVE_REPORTS path: {reports}")
        if reports.resolve(strict=True).parent != root.parent.parent:
            raise ArchiveV2Error("ARCHIVE_REPORTS resolves outside the project folder")
    return expected


def _file_snapshot(path: Path, root: Path) -> dict[str, Any]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ArchiveV2Error(f"Refusing non-regular episode file: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": details.st_size,
        "mtime_ns": details.st_mtime_ns,
    }


def _scan_tree(root: Path) -> dict[str, Any]:
    """Scan without following links and enforce the pipeline-owned tree shape."""

    files: list[dict[str, Any]] = []
    directories: list[str] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name.casefold())
        except OSError as exc:
            raise ArchiveV2Error(f"Cannot scan episode directory: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                details = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArchiveV2Error(f"Cannot inspect episode entry: {relative}") from exc
            mode = details.st_mode
            if stat.S_ISLNK(mode):
                raise ArchiveV2Error(f"Refusing descendant symlink: {relative}")
            if stat.S_ISDIR(mode):
                if "/" not in relative:
                    if relative not in _PIPELINE_DIRECTORIES:
                        raise ArchiveV2Error(
                            f"Unexpected top-level episode directory: {relative}"
                        )
                elif relative not in _ALLOWED_SUBDIRECTORIES:
                    raise ArchiveV2Error(f"Unexpected nested episode directory: {relative}")
                directories.append(relative)
                visit(path)
            elif stat.S_ISREG(mode):
                parts = PurePosixPath(relative).parts
                if len(parts) < 2 or parts[0] not in _PIPELINE_DIRECTORIES:
                    raise ArchiveV2Error(f"Unexpected top-level episode file: {relative}")
                suffix = PurePosixPath(relative).suffix.casefold()
                if suffix not in _ALLOWED_SUFFIXES[parts[0]]:
                    raise ArchiveV2Error(
                        f"Episode file is outside the cleanup whitelist: {relative}"
                    )
                files.append(
                    {
                        "relative_path": relative,
                        "size_bytes": details.st_size,
                        "mtime_ns": details.st_mtime_ns,
                    }
                )
            else:
                raise ArchiveV2Error(f"Refusing special filesystem entry: {relative}")

    visit(root)
    files.sort(key=lambda item: item["relative_path"])
    directories.sort()
    return {"files": files, "directories": directories}


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> Path:
    if path.is_symlink():
        raise ArchiveV2Error(f"Refusing symlinked archive v2 receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ArchiveV2Error(f"Refusing unsafe ARCHIVE_REPORTS path: {path.parent}")
    atomic_write_json(path, dict(receipt))
    if _read_json(path, "Archive v2 receipt") != dict(receipt):
        raise ArchiveV2Error("Archive v2 receipt readback mismatch")
    return path


def _verify_report_record(
    root: Path,
    record: Any,
    label: str,
    *,
    expected_relative: str | None = None,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(record, Mapping):
        raise ArchiveV2Error(f"{label} record is missing")
    value = dict(record)
    path = verify_recorded_file(
        root, value, label, expected_relative_path=expected_relative
    )
    return value, path


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArchiveV2Error(f"FINALIZATION v2 {label} must be an object")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str] | set[str], label: str
) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        raise ArchiveV2Error(
            f"FINALIZATION v2 {label} key mismatch; "
            f"missing={sorted(expected_set - actual)}, extra={sorted(actual - expected_set)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArchiveV2Error(f"FINALIZATION v2 {label} must be lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArchiveV2Error(
            f"FINALIZATION v2 {label} must be a non-negative integer"
        )
    return value


def _require_positive_int(value: Any, label: str) -> int:
    result = _require_nonnegative_int(value, label)
    if result == 0:
        raise ArchiveV2Error(f"FINALIZATION v2 {label} must be positive")
    return result


def _require_zero_fields(
    value: Mapping[str, Any], fields: tuple[str, ...], label: str
) -> None:
    for field in fields:
        count = _require_nonnegative_int(value.get(field), f"{label}.{field}")
        if count != 0:
            raise ArchiveV2Error(
                f"FINALIZATION v2 {label}.{field} must be zero; got {count}"
            )


def _require_self_digest(
    value: Mapping[str, Any], digest_field: str, label: str
) -> str:
    digest = _require_sha256(value.get(digest_field), f"{label}.{digest_field}")
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    if sha256_json(unsigned) != digest:
        raise ArchiveV2Error(f"FINALIZATION v2 {label} self-digest mismatch")
    return digest


def _v2_input_relatives(episode_name: str, source_relative: str) -> dict[str, str]:
    return {
        "source_video": source_relative,
        "download_metadata": "source/source.metadata.json",
        "download_marker": "source/download.done.json",
        "audio": "prepare/audio.flac",
        "audio_metadata": "prepare/audio.metadata.json",
        "audio_marker": "prepare/audio.done.json",
        "raw_asr_v2": "prepare/raw_asr_v2.json",
        "raw_asr_v2_marker": "prepare/raw_asr_v2.done.json",
        "forced_alignment_v2": "prepare/forced_alignment_v2.json",
        "forced_alignment_v2_marker": "prepare/forced_alignment_v2.done.json",
        "tr_correction_pack": (
            f"translation_input/{episode_name}_TR_CORRECTION_PACK.zip"
        ),
        "tr_correction_output": (
            f"translation_output/{episode_name}_TR_CORRECTED.zip"
        ),
        "aligned_schema_v2": "prepare/aligned_tr_schema_v2.json",
        "id_translation_pack": (
            f"translation_input/{episode_name}_ID_TRANSLATION_PACK.zip"
        ),
        "id_translation_zip": (
            f"translation_output/{episode_name}_ID_TRANSLATED.zip"
        ),
    }


def _v2_configuration_relatives() -> dict[str, str]:
    return {
        "series_config": "config/production/series.yaml",
        "names_config": "config/production/names.yaml",
        "religious_config": "config/production/religious_terms.yaml",
    }


def _verify_v2_evidence_records(
    root: Path,
    report: Mapping[str, Any],
    *,
    episode_name: str,
    source_relative: str,
) -> dict[str, Path]:
    inputs = _require_mapping(report.get("input_files"), "input_files")
    configs = _require_mapping(
        report.get("configuration_files"), "configuration_files"
    )
    hashes = _require_mapping(report.get("input_hashes"), "input_hashes")
    _require_exact_keys(inputs, _V2_INPUT_KEYS, "input_files")
    _require_exact_keys(configs, _V2_CONFIGURATION_KEYS, "configuration_files")
    _require_exact_keys(
        hashes,
        _V2_INPUT_KEYS | _V2_CONFIGURATION_KEYS,
        "input_hashes",
    )

    paths: dict[str, Path] = {}
    for key, relative in _v2_input_relatives(
        episode_name, source_relative
    ).items():
        record = _require_mapping(inputs.get(key), f"input_files.{key}")
        _require_exact_keys(
            record,
            {"relative_path", "size_bytes", "sha256"},
            f"input_files.{key}",
        )
        _, path = _verify_report_record(
            root,
            record,
            f"V2 input {key}",
            expected_relative=relative,
        )
        digest = _require_sha256(
            hashes.get(key), f"input_hashes.{key}"
        )
        if digest != record.get("sha256"):
            raise ArchiveV2Error(
                f"FINALIZATION v2 input_hashes.{key} does not bind its file record"
            )
        paths[key] = path

    project_root = root.parent.parent
    for key, relative in _v2_configuration_relatives().items():
        record = _require_mapping(configs.get(key), f"configuration_files.{key}")
        _require_exact_keys(
            record,
            {"relative_path", "size_bytes", "sha256"},
            f"configuration_files.{key}",
        )
        path = verify_recorded_file(
            project_root,
            record,
            f"V2 configuration {key}",
            expected_relative_path=relative,
        )
        digest = _require_sha256(
            hashes.get(key), f"input_hashes.{key}"
        )
        if digest != record.get("sha256"):
            raise ArchiveV2Error(
                f"FINALIZATION v2 input_hashes.{key} does not bind its file record"
            )
        paths[key] = path
    return paths


def _validate_v2_policy_summaries(
    report: Mapping[str, Any],
    *,
    raw: Mapping[str, Any],
    forced: Mapping[str, Any],
    forced_marker: Mapping[str, Any],
    coverage: Mapping[str, Any],
    coverage_config: Mapping[str, Any],
    input_records: Mapping[str, Any],
    audio_sha: str,
    alignment_sha: str,
    timing_review_count: int,
    bounded_audio_review_sha: str,
) -> None:
    raw_policy = _require_mapping(report.get("raw_policy_v2"), "raw_policy_v2")
    _require_exact_keys(raw_policy, _RAW_POLICY_KEYS, "raw_policy_v2")
    if raw_policy.get("report_version") != "2.0" or raw_policy.get("status") != "PASS":
        raise ArchiveV2Error("FINALIZATION v2 raw_policy_v2 is not a V2 PASS")
    raw_model = _require_mapping(raw.get("model"), "raw_asr_v2.model")
    raw_settings = _require_mapping(
        raw_model.get("settings"), "raw_asr_v2.model.settings"
    )
    explicit_review_uids = raw.get("hallucination_review_utterance_uids")
    if not isinstance(explicit_review_uids, list) or any(
        not isinstance(uid, str) or not uid for uid in explicit_review_uids
    ) or len(set(explicit_review_uids)) != len(explicit_review_uids):
        raise ArchiveV2Error(
            "FINALIZATION v2 raw explicit audio-review UID evidence is malformed"
        )
    if not (
        raw_policy.get("policy_label") == raw_settings.get("coverage_policy_label")
        and raw_policy.get("raw_input_sha256") == raw.get("input_sha256")
        and raw_policy.get("raw_artifact_sha256")
        == input_records["raw_asr_v2"].get("sha256")
        and raw_policy.get("audio_sha256") == audio_sha
        and raw_policy.get("model_settings") == raw_settings
        and raw_policy.get("speech_coverage_config") == coverage_config
        and raw_policy.get("explicit_audio_review_uids") == explicit_review_uids
    ):
        raise ArchiveV2Error("FINALIZATION v2 raw_policy_v2 evidence binding mismatch")
    if raw_policy.get("model_settings_sha256") != sha256_json(raw_settings):
        raise ArchiveV2Error("FINALIZATION v2 raw model-settings digest mismatch")
    if raw_policy.get("speech_coverage_config_sha256") != sha256_json(
        dict(coverage_config)
    ):
        raise ArchiveV2Error("FINALIZATION v2 raw coverage-policy digest mismatch")
    _require_self_digest(raw_policy, "raw_policy_sha256", "raw_policy_v2")

    alignment_policy = _require_mapping(
        report.get("alignment_policy_v2"), "alignment_policy_v2"
    )
    _require_exact_keys(
        alignment_policy, _ALIGNMENT_POLICY_KEYS, "alignment_policy_v2"
    )
    if (
        alignment_policy.get("report_version") != "2.0"
        or alignment_policy.get("status") != "PASS"
        or alignment_policy.get("audio_sha256") != audio_sha
        or alignment_policy.get("alignment_sha256") != alignment_sha
    ):
        raise ArchiveV2Error("FINALIZATION v2 alignment_policy_v2 identity mismatch")
    provenance = _require_mapping(
        forced.get("provenance"), "forced_alignment_v2.provenance"
    )
    policy = _require_mapping(
        alignment_policy.get("policy"), "alignment_policy_v2.policy"
    )
    _require_exact_keys(policy, _ALIGNMENT_POLICY_FIELDS, "alignment_policy_v2.policy")
    expected_policy = {field: provenance.get(field) for field in _ALIGNMENT_POLICY_FIELDS}
    if policy != expected_policy or alignment_policy.get("provenance") != provenance:
        raise ArchiveV2Error("FINALIZATION v2 alignment policy/provenance mismatch")
    minimum_policy_score = policy.get("min_word_score")
    maximum_policy_duration = policy.get("max_word_duration_ms")
    maximum_policy_drift = policy.get("max_outward_drift_ms")
    if not (
        policy.get("timing_source") == "whisperx_ctc_forced_alignment"
        and policy.get("engine") == "whisperx"
        and policy.get("interpolation") in {"disabled:none", "disabled:ignore"}
        and isinstance(minimum_policy_score, (int, float))
        and not isinstance(minimum_policy_score, bool)
        and 0.30 <= float(minimum_policy_score) <= 1.0
        and policy.get("review_word_score") == 0.55
        and policy.get("edited_token_min_word_score") == 0.55
        and isinstance(maximum_policy_duration, int)
        and not isinstance(maximum_policy_duration, bool)
        and 1 <= maximum_policy_duration <= 2500
        and isinstance(maximum_policy_drift, int)
        and not isinstance(maximum_policy_drift, bool)
        and 0 <= maximum_policy_drift <= 500
    ):
        raise ArchiveV2Error("FINALIZATION v2 alignment hard policy is unsafe")
    if alignment_policy.get("provenance_sha256") != sha256_json(provenance):
        raise ArchiveV2Error("FINALIZATION v2 alignment provenance digest mismatch")
    hard_counters = _require_mapping(
        alignment_policy.get("hard_zero_counters"),
        "alignment_policy_v2.hard_zero_counters",
    )
    _require_exact_keys(
        hard_counters,
        set(_ALIGNMENT_REPORT_ZERO_FIELDS),
        "alignment_policy_v2.hard_zero_counters",
    )
    _require_zero_fields(
        hard_counters,
        _ALIGNMENT_REPORT_ZERO_FIELDS,
        "alignment_policy_v2.hard_zero_counters",
    )
    forced_report = _require_mapping(
        forced.get("report"), "forced_alignment_v2.report"
    )
    if any(forced_report.get(field) != hard_counters[field] for field in hard_counters):
        raise ArchiveV2Error("FINALIZATION v2 alignment hard counters are unbound")
    if alignment_policy.get("review_alignment_score_count") != timing_review_count:
        raise ArchiveV2Error("FINALIZATION v2 alignment warning count mismatch")
    for field in (
        "minimum_alignment_score",
        "maximum_alignment_word_duration_ms",
        "maximum_early_outward_drift_ms",
        "maximum_late_outward_drift_ms",
    ):
        if alignment_policy.get(field) != forced_report.get(field):
            raise ArchiveV2Error(
                f"FINALIZATION v2 alignment_policy_v2.{field} is unbound"
            )
    minimum_score = alignment_policy.get("minimum_alignment_score")
    if (
        isinstance(minimum_score, bool)
        or not isinstance(minimum_score, (int, float))
        or float(minimum_score) < float(minimum_policy_score)
    ):
        raise ArchiveV2Error("FINALIZATION v2 minimum alignment score is unsafe")
    for field, maximum in (
        ("maximum_alignment_word_duration_ms", maximum_policy_duration),
        ("maximum_early_outward_drift_ms", maximum_policy_drift),
        ("maximum_late_outward_drift_ms", maximum_policy_drift),
    ):
        value = alignment_policy.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise ArchiveV2Error(f"FINALIZATION v2 {field} is unsafe")
    _require_self_digest(
        alignment_policy, "alignment_policy_sha256", "alignment_policy_v2"
    )

    edit_audit = _require_mapping(
        report.get("alignment_edit_audit_v2"), "alignment_edit_audit_v2"
    )
    _require_exact_keys(
        edit_audit,
        {
            "report_version",
            "status",
            "alignment_sha256",
            "segment_count",
            "summary",
            "segments",
            "alignment_edit_audit_sha256",
        },
        "alignment_edit_audit_v2",
    )
    forced_segments = forced.get("segments")
    if not isinstance(forced_segments, list) or not forced_segments:
        raise ArchiveV2Error("FINALIZATION v2 forced alignment has no segments")
    expected_segments: list[dict[str, Any]] = []
    for position, segment in enumerate(forced_segments, start=1):
        if not isinstance(segment, Mapping) or not isinstance(
            segment.get("edit_audit"), Mapping
        ):
            raise ArchiveV2Error(
                f"FINALIZATION v2 forced segment {position} edit audit is missing"
            )
        expected_segments.append(
            {
                "segment_index": segment.get("segment_index"),
                "utterance_uid": segment.get("utterance_uid"),
                "asr_text": segment.get("asr_text"),
                "corrected_text": segment.get("text"),
                "deletion_audio_reviewed": segment.get("deletion_audio_reviewed"),
                "edit_audit": dict(segment["edit_audit"]),
            }
        )
    summary = _require_mapping(
        edit_audit.get("summary"), "alignment_edit_audit_v2.summary"
    )
    _require_exact_keys(
        summary, _ALIGNMENT_EDIT_SUMMARY_FIELDS, "alignment_edit_audit_v2.summary"
    )
    if any(summary.get(field) != forced_report.get(field) for field in summary):
        raise ArchiveV2Error("FINALIZATION v2 alignment edit summary is unbound")
    _require_zero_fields(
        summary,
        ("low_score_edited_token_count", "unreviewed_deleted_token_count"),
        "alignment_edit_audit_v2.summary",
    )
    if not (
        edit_audit.get("report_version") == "2.0"
        and edit_audit.get("status") == "PASS"
        and edit_audit.get("alignment_sha256") == alignment_sha
        and edit_audit.get("segment_count") == len(expected_segments)
        and edit_audit.get("segments") == expected_segments
    ):
        raise ArchiveV2Error("FINALIZATION v2 alignment edit audit binding mismatch")
    _require_self_digest(
        edit_audit,
        "alignment_edit_audit_sha256",
        "alignment_edit_audit_v2",
    )

    audio_review = _require_mapping(report.get("audio_review_v2"), "audio_review_v2")
    _require_exact_keys(audio_review, _AUDIO_REVIEW_KEYS, "audio_review_v2")
    outcomes = audio_review.get("outcomes")
    if not isinstance(outcomes, list) or any(not isinstance(item, Mapping) for item in outcomes):
        raise ArchiveV2Error("FINALIZATION v2 audio_review_v2.outcomes is malformed")
    review_counts = {
        "confirmed_dialogue_count": sum(
            item.get("review_disposition") == "confirmed_dialogue" for item in outcomes
        ),
        "reviewed_non_dialogue_count": sum(
            item.get("review_disposition") == "reviewed_non_dialogue" for item in outcomes
        ),
        "discarded_asr_hallucination_count": sum(
            item.get("review_disposition") == "discarded_asr_hallucination"
            for item in outcomes
        ),
    }
    if not (
        audio_review.get("report_version") == "2.0"
        and audio_review.get("status") == "PASS"
        and audio_review.get("reviewed_outcome_count") == len(outcomes)
        and audio_review.get("pending_audio_review_count") == 0
        and all(audio_review.get(field) == count for field, count in review_counts.items())
    ):
        raise ArchiveV2Error("FINALIZATION v2 audio-review counts/pass mismatch")
    if outcomes and any(
        item.get("acoustic_audit_sha256") != bounded_audio_review_sha
        for item in outcomes
    ):
        raise ArchiveV2Error(
            "FINALIZATION v2 audio outcomes do not bind the bounded Colab audit"
        )
    _require_self_digest(audio_review, "audio_review_sha256", "audio_review_v2")

    tr_validation = _require_mapping(
        report.get("tr_correction_validation"), "tr_correction_validation"
    )
    _require_exact_keys(
        tr_validation,
        {
            "output_sha256",
            "records_sha256",
            "record_count",
            "review_required_count",
            "audio_reviewed_count",
            "pending_audio_review_count",
            "confirmed_dialogue_count",
            "reviewed_non_dialogue_count",
            "discarded_asr_hallucination_count",
        },
        "tr_correction_validation",
    )
    records_sha = _require_sha256(
        tr_validation.get("records_sha256"),
        "tr_correction_validation.records_sha256",
    )
    forced_marker_details = _require_mapping(
        forced_marker.get("details"), "forced_alignment_v2_marker.details"
    )
    if forced_marker_details.get("correction_output_sha256") != records_sha:
        raise ArchiveV2Error(
            "FINALIZATION v2 Turkish record digest is not forced-marker bound"
        )
    for field in (
        "audio_reviewed_count",
        "pending_audio_review_count",
        "confirmed_dialogue_count",
        "reviewed_non_dialogue_count",
        "discarded_asr_hallucination_count",
    ):
        _require_nonnegative_int(
            tr_validation.get(field), f"tr_correction_validation.{field}"
        )
    if not (
        tr_validation.get("audio_reviewed_count") == len(outcomes)
        and tr_validation.get("pending_audio_review_count") == 0
        and all(tr_validation.get(field) == count for field, count in review_counts.items())
    ):
        raise ArchiveV2Error("FINALIZATION v2 Turkish/audio-review count mismatch")

    strict_vad = _require_mapping(
        report.get("strict_word_vad_v2"), "strict_word_vad_v2"
    )
    _require_exact_keys(strict_vad, _STRICT_WORD_VAD_KEYS, "strict_word_vad_v2")
    strict_policy = _require_mapping(
        strict_vad.get("policy"), "strict_word_vad_v2.policy"
    )
    max_outside = coverage_config.get("max_word_outside_speech_ms")
    audit_pad = raw_settings.get("audit_vad_speech_pad_ms")
    if (
        isinstance(max_outside, bool)
        or not isinstance(max_outside, int)
        or max_outside < 0
        or isinstance(audit_pad, bool)
        or not isinstance(audit_pad, int)
        or audit_pad < 0
    ):
        raise ArchiveV2Error("FINALIZATION v2 strict word/VAD distance policy is malformed")
    if not (
        strict_vad.get("report_version") == "2.0"
        and strict_vad.get("status") == "PASS"
        and strict_vad.get("raw_asr_input_sha256") == raw.get("input_sha256")
        and strict_vad.get("audio_sha256") == audio_sha
        and strict_vad.get("alignment_sha256") == alignment_sha
        and strict_vad.get("audio_review_sha256")
        == audio_review.get("audio_review_sha256")
        and strict_policy.get("vad_source") == "silero_vad"
        and strict_policy.get("minimum_word_vad_overlap_ratio") == 0.25
        and strict_policy.get("audit_vad_speech_pad_ms") == audit_pad
        and strict_policy.get("maximum_leading_outside_vad_ms") == max_outside
        and strict_policy.get("maximum_trailing_outside_vad_ms") == max_outside
        and strict_policy.get("maximum_contiguous_non_vad_ms") == max_outside
        and strict_policy.get("maximum_effective_distance_from_unpadded_vad_core_ms")
        == audit_pad + max_outside
    ):
        raise ArchiveV2Error("FINALIZATION v2 strict word/VAD policy identity mismatch")
    for count_field in (
        "aligned_word_count",
        "confirmed_dialogue_exception_word_count",
        "unsafe_word_count",
        "confirmed_dialogue_required_coverage_count",
        "confirmed_dialogue_coverage_fail_count",
    ):
        _require_nonnegative_int(strict_vad.get(count_field), f"strict_word_vad_v2.{count_field}")
    if (
        strict_vad.get("aligned_word_count") != len(forced.get("words", []))
        or strict_vad.get("unsafe_word_count") != 0
        or strict_vad.get("confirmed_dialogue_coverage_fail_count") != 0
        or strict_vad.get("confirmed_dialogue_exception_word_count")
        != len(strict_vad.get("confirmed_dialogue_exception_words", []))
        or strict_vad.get("confirmed_dialogue_required_coverage_count")
        != len(strict_vad.get("confirmed_dialogue_required_coverage", []))
        or strict_vad.get("unsafe_words") != []
    ):
        raise ArchiveV2Error("FINALIZATION v2 strict word/VAD hard gate mismatch")
    _require_self_digest(
        strict_vad, "strict_word_vad_sha256", "strict_word_vad_v2"
    )
    if (
        coverage.get("audio_review_v2") != audio_review
        or coverage.get("strict_word_vad_v2") != strict_vad
    ):
        raise ArchiveV2Error("FINALIZATION v2 coverage audit copies are not exact")


def _validate_v2_core_gates(
    root: Path,
    report: Mapping[str, Any],
    *,
    episode: int,
    episode_name: str,
    source_relative: str,
) -> dict[str, Any]:
    """Reject a saved PASS unless every independently produced V2 gate is bound."""

    block_count = _require_positive_int(report.get("block_count"), "block_count")
    schema_version = report.get("schema_version")
    if not isinstance(schema_version, str) or schema_version.split(".", 1)[0] != "2":
        raise ArchiveV2Error("FINALIZATION v2 schema_version is not V2")
    schema_sha = _require_sha256(report.get("schema_sha256"), "schema_sha256")
    audio_sha = _require_sha256(report.get("audio_sha256"), "audio_sha256")
    alignment_sha = _require_sha256(
        report.get("alignment_sha256"), "alignment_sha256"
    )
    coverage_sha = _require_sha256(
        report.get("speech_coverage_sha256"), "speech_coverage_sha256"
    )

    evidence_paths = _verify_v2_evidence_records(
        root,
        report,
        episode_name=episode_name,
        source_relative=source_relative,
    )
    inputs = dict(report["input_files"])
    if inputs["audio"].get("sha256") != audio_sha:
        raise ArchiveV2Error("FINALIZATION v2 audio_sha256 does not bind audio input")

    raw = _read_json(evidence_paths["raw_asr_v2"], "Raw ASR V2 evidence")
    forced = _read_json(
        evidence_paths["forced_alignment_v2"], "Forced-alignment V2 evidence"
    )
    forced_marker = _read_json(
        evidence_paths["forced_alignment_v2_marker"],
        "Forced-alignment V2 marker evidence",
    )
    schema = _read_json(
        evidence_paths["aligned_schema_v2"], "Aligned-schema V2 evidence"
    )
    if raw.get("episode") != episode or raw.get("audio_sha256") != audio_sha:
        raise ArchiveV2Error("FINALIZATION v2 raw ASR episode/audio identity mismatch")
    try:
        validate_publishable_raw_asr_v2_policy(raw)
    except TranscriptionError as exc:
        raise ArchiveV2Error(
            f"FINALIZATION v2 raw ASR policy is not publishable: {exc}"
        ) from exc
    if forced.get("audio_sha256") != audio_sha:
        raise ArchiveV2Error("FINALIZATION v2 forced alignment audio identity mismatch")
    if forced.get("alignment_sha256") != alignment_sha:
        raise ArchiveV2Error("FINALIZATION v2 alignment_sha256 evidence mismatch")
    if (
        schema.get("schema_version") != schema_version
        or schema.get("schema_sha256") != schema_sha
        or schema.get("audio_sha256") != audio_sha
        or schema.get("speech_coverage_sha256") != coverage_sha
        or schema.get("block_count") != block_count
    ):
        raise ArchiveV2Error("FINALIZATION v2 aligned schema identity mismatch")

    evidence = _require_mapping(report.get("evidence_validation"), "evidence_validation")
    evidence_flags = {
        "source_audio_chain_exact",
        "full_forced_alignment_revalidated",
        "bounded_audio_review_revalidated",
        "speech_coverage_recomputed",
        "schema_rebuilt_exact",
        "semantic_qa_config_hash_bound",
        "evidence_rehashed_immediately_before_pass",
    }
    for field in evidence_flags:
        if evidence.get(field) is not True:
            raise ArchiveV2Error(
                f"FINALIZATION v2 evidence_validation.{field} must be true"
            )
    bounded_audio_review_sha = _require_sha256(
        evidence.get("bounded_audio_review_sha256"),
        "evidence_validation.bounded_audio_review_sha256",
    )

    tr_validation = _require_mapping(
        report.get("tr_correction_validation"), "tr_correction_validation"
    )
    if (
        _require_sha256(
            tr_validation.get("output_sha256"),
            "tr_correction_validation.output_sha256",
        )
        != inputs["tr_correction_output"].get("sha256")
    ):
        raise ArchiveV2Error("FINALIZATION v2 Turkish correction hash mismatch")
    _require_positive_int(
        tr_validation.get("record_count"), "tr_correction_validation.record_count"
    )
    _require_zero_fields(
        tr_validation,
        ("review_required_count",),
        "tr_correction_validation",
    )

    id_validation = _require_mapping(
        report.get("id_translation_validation"), "id_translation_validation"
    )
    if id_validation.get("passed") is not True:
        raise ArchiveV2Error("FINALIZATION v2 Indonesian validation did not pass")
    if id_validation.get("schema_sha256") != schema_sha:
        raise ArchiveV2Error("FINALIZATION v2 Indonesian schema identity mismatch")
    for field in ("expected_block_count", "output_block_count"):
        if _require_positive_int(
            id_validation.get(field), f"id_translation_validation.{field}"
        ) != block_count:
            raise ArchiveV2Error(
                f"FINALIZATION v2 id_translation_validation.{field} mismatch"
            )
    _require_zero_fields(
        id_validation,
        ("missing_block_count", "duplicate_block_count", "review_required_count"),
        "id_translation_validation",
    )

    coverage = _require_mapping(report.get("speech_coverage_v2"), "speech_coverage_v2")
    if coverage.get("status") != "PASS":
        raise ArchiveV2Error("FINALIZATION v2 speech coverage did not pass")
    _require_zero_fields(
        coverage,
        ("unresolved_speech_region_count",),
        "speech_coverage_v2",
    )
    metrics = _require_mapping(
        coverage.get("metrics"), "speech_coverage_v2.metrics"
    )
    _require_zero_fields(
        metrics,
        (
            "unresolved_speech_region_count",
            "unresolved_coverage_issue_count",
            "word_interval_wholly_outside_speech_count",
        ),
        "speech_coverage_v2.metrics",
    )
    coverage_config = _require_mapping(
        coverage.get("config"), "speech_coverage_v2.config"
    )
    _require_exact_keys(
        coverage_config, _COVERAGE_CONFIG_FIELDS, "speech_coverage_v2.config"
    )
    raw_coverage = _require_mapping(raw.get("speech_coverage"), "raw_asr_v2.speech_coverage")
    raw_coverage_config = _require_mapping(
        raw_coverage.get("config"), "raw_asr_v2.speech_coverage.config"
    )
    if raw_coverage_config != coverage_config:
        raise ArchiveV2Error("FINALIZATION v2 coverage policy is not raw-evidence bound")
    if sha256_json(coverage) != coverage_sha:
        raise ArchiveV2Error("FINALIZATION v2 speech coverage digest mismatch")

    timing = _require_mapping(report.get("timing_qa_v2"), "timing_qa_v2")
    if (
        timing.get("report_version") != 2
        or timing.get("passed") is not True
        or timing.get("block_count") != block_count
        or timing.get("alignment_sha256") != alignment_sha
    ):
        raise ArchiveV2Error("FINALIZATION v2 timing QA identity/pass mismatch")
    _require_mapping(timing.get("config"), "timing_qa_v2.config")
    _require_zero_fields(timing, _TIMING_ZERO_FIELDS, "timing_qa_v2")
    review_score_count = _require_nonnegative_int(
        timing.get("review_alignment_score_count"),
        "timing_qa_v2.review_alignment_score_count",
    )
    _validate_v2_policy_summaries(
        report,
        raw=raw,
        forced=forced,
        forced_marker=forced_marker,
        coverage=coverage,
        coverage_config=coverage_config,
        input_records=inputs,
        audio_sha=audio_sha,
        alignment_sha=alignment_sha,
        timing_review_count=review_score_count,
        bounded_audio_review_sha=bounded_audio_review_sha,
    )

    semantic = _require_mapping(
        report.get("semantic_subtitle_qa"), "semantic_subtitle_qa"
    )
    if (
        semantic.get("passed") is not True
        or semantic.get("timings_identical") is not True
        or semantic.get("tr_block_count") != block_count
        or semantic.get("id_block_count") != block_count
    ):
        raise ArchiveV2Error("FINALIZATION v2 semantic subtitle QA did not pass")
    _require_zero_fields(semantic, _SEMANTIC_ZERO_FIELDS, "semantic_subtitle_qa")

    identity = _require_mapping(report.get("timing_identity"), "timing_identity")
    for field in (
        "forced_alignment",
        "independent_speech_coverage",
        "tr_id_identical",
        "srt_roundtrip_exact",
    ):
        if identity.get(field) is not True:
            raise ArchiveV2Error(f"FINALIZATION v2 timing_identity.{field} must be true")

    outputs = _require_mapping(report.get("outputs"), "outputs")
    _require_exact_keys(outputs, {"mkv", "id_srt", "tr_srt"}, "outputs")
    for key in ("mkv", "id_srt", "tr_srt"):
        output_record = _require_mapping(outputs.get(key), f"outputs.{key}")
        _require_exact_keys(
            output_record,
            {"relative_path", "size_bytes", "sha256"},
            f"outputs.{key}",
        )
    mux = _require_mapping(report.get("mkv_verification"), "mkv_verification")
    roundtrip = _require_mapping(
        mux.get("roundtrip"), "mkv_verification.roundtrip"
    )
    stream_hashes = _require_mapping(
        mux.get("stream_hashes"), "mkv_verification.stream_hashes"
    )
    source_streams = stream_hashes.get("source")
    output_streams = stream_hashes.get("output")
    expected_mkv_relative = _canonical_relatives(episode)["mkv"]
    if not (
        mux.get("verified") is True
        and mux.get("video_audio_stream_copy") is True
        and mux.get("subtitle_order") == ["ind", "tur"]
        and mux.get("indonesian_default") is True
        and mux.get("turkish_default") is False
        and mux.get("output_path") == expected_mkv_relative
        and mux.get("output_size_bytes") == outputs["mkv"].get("size_bytes")
        and roundtrip.get("exact") is True
        and roundtrip.get("id_block_count") == block_count
        and roundtrip.get("tr_block_count") == block_count
        and stream_hashes.get("checked") is True
        and stream_hashes.get("match") is True
        and isinstance(source_streams, list)
        and bool(source_streams)
        and source_streams == output_streams
    ):
        raise ArchiveV2Error("FINALIZATION v2 MKV verification is incomplete")

    forced_provenance = _require_mapping(
        forced.get("provenance"), "forced_alignment_v2.provenance"
    )
    minimum_score = forced_provenance.get("min_word_score")
    review_score = forced_provenance.get("review_word_score")
    max_duration = forced_provenance.get("max_word_duration_ms")
    max_drift = forced_provenance.get("max_outward_drift_ms")
    if (
        isinstance(minimum_score, bool)
        or not isinstance(minimum_score, (int, float))
        or not 0.30 <= float(minimum_score) <= 1.0
        or review_score != 0.55
        or isinstance(max_duration, bool)
        or not isinstance(max_duration, int)
        or not 1 <= max_duration <= 2500
        or isinstance(max_drift, bool)
        or not isinstance(max_drift, int)
        or not 0 <= max_drift <= 500
    ):
        raise ArchiveV2Error("FINALIZATION v2 forced-alignment hard policy mismatch")
    forced_report = _require_mapping(
        forced.get("report"), "forced_alignment_v2.report"
    )
    _require_zero_fields(
        forced_report,
        _ALIGNMENT_REPORT_ZERO_FIELDS,
        "forced_alignment_v2.report",
    )
    if forced_report.get("alignment_sha256") != alignment_sha:
        raise ArchiveV2Error("FINALIZATION v2 forced-alignment report digest mismatch")
    forced_warning_count = _require_nonnegative_int(
        forced_report.get("review_alignment_score_count"),
        "forced_alignment_v2.report.review_alignment_score_count",
    )
    if forced_warning_count != review_score_count:
        raise ArchiveV2Error("FINALIZATION v2 alignment review-warning count mismatch")

    warnings: list[dict[str, Any]] = []
    if review_score_count:
        warnings.append(
            {
                "code": "alignment_score_pilot_review",
                "count": review_score_count,
                "message": (
                    f"{review_score_count} aligned word(s) scored from 0.30 through "
                    "0.549 and require Episode 10 pilot spot-checking."
                ),
            }
        )
    return {"pilot_warnings": warnings, "evidence_paths": evidence_paths}


def _load_finalization_inputs(root: Path, episode: int) -> dict[str, Any]:
    _, episode_name = _episode_identity(episode)
    v2_report_path = root / "final" / f"{episode_name}_FINALIZATION_REPORT_V2.json"
    v1_report_path = root / "final" / f"{episode_name}_FINALIZATION_REPORT.json"
    # Never let a malformed or symlinked V2 report silently fall back to V1.
    # The distinct names let an Episode 10 V2 pilot coexist with, and roll
    # back to, the already-published V1 finalization record.
    report_path = (
        v2_report_path
        if v2_report_path.exists() or v2_report_path.is_symlink()
        else v1_report_path
    )
    report = _read_json(report_path, "FINALIZATION report")
    report_version = report.get("report_version")
    if (
        isinstance(report_version, bool)
        or not isinstance(report_version, int)
        or report_version not in {1, 2}
        or report.get("status") != "PASS"
    ):
        raise ArchiveV2Error("FINALIZATION report is not a supported PASS report")
    if report_version == 2 and report_path != v2_report_path:
        raise ArchiveV2Error("FINALIZATION v2 report must use the V2-specific filename")
    if report_version == 1 and report_path != v1_report_path:
        raise ArchiveV2Error("FINALIZATION v1 report must use the legacy filename")
    if report.get("episode") != episode or report.get("episode_name") != episode_name:
        raise ArchiveV2Error("FINALIZATION report episode identity mismatch")
    inputs = report.get("input_files")
    outputs = report.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ArchiveV2Error("FINALIZATION report file records are missing")

    source_record, source = _verify_report_record(
        root, inputs.get("source_video"), "Source video"
    )
    source_relative = source.relative_to(root).as_posix()
    if source.parent != root / "source" or source.stem != episode_name:
        raise ArchiveV2Error("Source video is not the exact episode-named source file")
    if source.suffix.casefold() not in VIDEO_SUFFIXES:
        raise ArchiveV2Error("Source video suffix is unsupported")

    v2_validation: dict[str, Any] = {"pilot_warnings": []}
    if report_version == 2:
        v2_validation = _validate_v2_core_gates(
            root,
            report,
            episode=episode,
            episode_name=episode_name,
            source_relative=source_relative,
        )

    if report_version == 1:
        id_relative = f"final/{episode_name}.id-final.srt"
        tr_relative = f"final/{episode_name}.tr-final.srt"
    else:
        canonical = _canonical_relatives(episode)
        id_relative = canonical["id_srt"]
        tr_relative = canonical["tr_srt"]
    id_record, id_srt = _verify_report_record(
        root,
        outputs.get("id_srt"),
        "Indonesian final SRT",
        expected_relative=id_relative,
    )
    tr_record, tr_srt = _verify_report_record(
        root,
        outputs.get("tr_srt"),
        "Turkish final SRT",
        expected_relative=tr_relative,
    )
    legacy: dict[str, dict[str, Any]] = {
        "source_video": source_record,
        "finalization_report": file_record(report_path, root),
    }
    if report_version == 1:
        legacy["id_srt"] = id_record
        legacy["tr_srt"] = tr_record

    if report_version == 1:
        sidecar_specs = {
            "id_infuse_sidecar": f"source/{source.stem}-id.srt",
            "tr_infuse_sidecar": f"source/{source.stem}-tr.srt",
        }
        for key, expected in sidecar_specs.items():
            if key not in outputs:
                continue
            record, _ = _verify_report_record(
                root, outputs.get(key), key, expected_relative=expected
            )
            canonical_record = id_record if key.startswith("id_") else tr_record
            if record.get("sha256") != canonical_record.get("sha256"):
                raise ArchiveV2Error(f"{key} differs from its canonical final SRT")
            legacy[key] = record
    elif any(key in outputs for key in ("id_infuse_sidecar", "tr_infuse_sidecar")):
        raise ArchiveV2Error("FINALIZATION v2 report must not publish duplicate sidecars")

    legacy_mkv: Path | None = None
    if report_version == 2:
        mkv_record, _ = _verify_report_record(
            root,
            outputs.get("mkv"),
            "Canonical final MKV",
            expected_relative=_canonical_relatives(episode)["mkv"],
        )
    elif "mkv" in outputs:
        raw_mkv_record = outputs.get("mkv")
        if not isinstance(raw_mkv_record, Mapping):
            raise ArchiveV2Error("FINALIZATION MKV record is missing")
        mkv_record = dict(raw_mkv_record)
        old_relative = f"final/{episode_name} - Endonezce + Turkce.mkv"
        canonical_relative = f"final/{episode_name}.mkv"
        recorded_relative = mkv_record.get("relative_path")
        if recorded_relative == canonical_relative:
            # A future FINALIZE may already publish the v2 name.
            verify_recorded_file(
                root,
                mkv_record,
                "Canonical final MKV",
                expected_relative_path=canonical_relative,
            )
        elif recorded_relative == old_relative:
            old_path = root / PurePosixPath(old_relative)
            if old_path.exists() or old_path.is_symlink():
                legacy_mkv = verify_recorded_file(
                    root,
                    mkv_record,
                    "Legacy final MKV",
                    expected_relative_path=old_relative,
                )
                legacy["legacy_mkv"] = mkv_record
            else:
                # Recover the narrow crash window after a verified atomic
                # legacy->canonical rename but before the READY receipt write.
                relocated = dict(mkv_record)
                relocated["relative_path"] = canonical_relative
                verify_recorded_file(
                    root,
                    relocated,
                    "Relocated canonical MKV",
                    expected_relative_path=canonical_relative,
                )
        else:
            raise ArchiveV2Error("FINALIZATION MKV path is not a recognized canonical path")

    return {
        "episode": episode,
        "episode_name": episode_name,
        "episode_root": root,
        "source": f"finalization_report_v{report_version}",
        "finalization_report_version": report_version,
        "source_video": source,
        "source_video_record": source_record,
        "id_srt_source": id_srt,
        "tr_srt_source": tr_srt,
        "legacy_mkv": legacy_mkv,
        "legacy_records": legacy,
        "report_path": report_path,
        "receipt": None,
        "source_av_stream_hashes": None,
        "pilot_warnings": list(v2_validation["pilot_warnings"]),
    }


def _validate_plan_shape(plan: Any, episode: int) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ArchiveV2Error("Archive v2 receipt has no cleanup plan")
    value = dict(plan)
    if value.get("plan_version") != PLAN_VERSION or value.get("episode") != episode:
        raise ArchiveV2Error("Archive v2 cleanup plan identity mismatch")
    protected = value.get("protected")
    expected_protected = sorted(_canonical_relatives(episode).values())
    if protected != expected_protected:
        raise ArchiveV2Error("Archive v2 cleanup protected whitelist mismatch")
    tree = value.get("tree")
    delete = value.get("delete")
    if not isinstance(tree, Mapping) or not isinstance(delete, list):
        raise ArchiveV2Error("Archive v2 cleanup plan is malformed")
    tree_files = tree.get("files")
    tree_dirs = tree.get("directories")
    if not isinstance(tree_files, list) or not isinstance(tree_dirs, list):
        raise ArchiveV2Error("Archive v2 cleanup tree is malformed")
    if not all(isinstance(relative, str) for relative in tree_dirs) or tree_dirs != sorted(
        set(tree_dirs)
    ):
        raise ArchiveV2Error("Archive v2 cleanup directory list is invalid")
    file_paths: set[str] = set()
    for record in tree_files:
        if not isinstance(record, Mapping):
            raise ArchiveV2Error("Archive v2 cleanup tree record is malformed")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or relative in file_paths:
            raise ArchiveV2Error("Archive v2 cleanup tree paths are invalid")
        size = record.get("size_bytes")
        mtime = record.get("mtime_ns")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or isinstance(mtime, bool)
            or not isinstance(mtime, int)
            or mtime < 0
        ):
            raise ArchiveV2Error("Archive v2 cleanup tree metadata is invalid")
        file_paths.add(relative)
    delete_paths: set[str] = set()
    for record in delete:
        if not isinstance(record, Mapping) or record not in tree_files:
            raise ArchiveV2Error("Archive v2 delete record is not in its tree snapshot")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or relative in delete_paths:
            raise ArchiveV2Error("Archive v2 delete paths are invalid")
        if relative in expected_protected:
            raise ArchiveV2Error("Archive v2 plan attempts to delete a protected artifact")
        delete_paths.add(relative)
    if file_paths != delete_paths | set(expected_protected):
        raise ArchiveV2Error("Archive v2 plan does not classify every episode file")
    expected_delete_size = sum(
        record.get("size_bytes", -1) for record in delete if isinstance(record, Mapping)
    )
    if (
        value.get("delete_file_count") != len(delete)
        or value.get("delete_size_bytes") != expected_delete_size
    ):
        raise ArchiveV2Error("Archive v2 cleanup totals are inconsistent")
    fingerprint_payload = {
        "episode": episode,
        "files": tree_files,
        "directories": tree_dirs,
        "delete": delete,
        "protected": protected,
    }
    if value.get("tree_fingerprint") != sha256_json(fingerprint_payload):
        raise ArchiveV2Error("Archive v2 cleanup plan fingerprint mismatch")
    return value


def _load_receipt_inputs(root: Path, episode: int, receipt: Mapping[str, Any]) -> dict[str, Any]:
    _, episode_name = _episode_identity(episode)
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        raise ArchiveV2Error("Archive v2 receipt version mismatch")
    if receipt.get("layout_version") != LAYOUT_VERSION:
        raise ArchiveV2Error("Archive v2 layout version mismatch")
    if receipt.get("status") not in {"READY", "PASS"}:
        raise ArchiveV2Error("Archive v2 receipt status is invalid")
    if receipt.get("episode") != episode or receipt.get("episode_name") != episode_name:
        raise ArchiveV2Error("Archive v2 receipt episode identity mismatch")
    cleanup = receipt.get("cleanup")
    if not isinstance(cleanup, Mapping):
        raise ArchiveV2Error("Archive v2 receipt cleanup record is missing")
    plan = _validate_plan_shape(cleanup.get("plan"), episode)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArchiveV2Error("Archive v2 receipt artifacts are missing")
    relative = _canonical_relatives(episode)
    paths = {
        key: verify_recorded_file(
            root, artifacts.get(key), key, expected_relative_path=relative[key]
        )
        for key in ("mkv", "id_srt", "tr_srt")
    }
    source_hashes = receipt.get("source_av_stream_hashes")
    if not isinstance(source_hashes, list) or not source_hashes:
        raise ArchiveV2Error("Archive v2 receipt has no source A/V stream hashes")
    raw_warnings = receipt.get("pilot_warnings", [])
    if not isinstance(raw_warnings, list) or any(
        not isinstance(item, Mapping) for item in raw_warnings
    ):
        raise ArchiveV2Error("Archive v2 receipt pilot warnings are malformed")
    legacy = receipt.get("legacy_inputs")
    if not isinstance(legacy, Mapping) or not isinstance(legacy.get("source_video"), Mapping):
        raise ArchiveV2Error("Archive v2 receipt legacy input records are missing")
    source_record = dict(legacy["source_video"])
    source_relative = source_record.get("relative_path")
    if not isinstance(source_relative, str):
        raise ArchiveV2Error("Archive v2 source record path is missing")
    _, source = _safe_relative(
        root,
        source_relative,
        "Receipt source video",
        allow_missing_parent=True,
    )
    if source.exists() or source.is_symlink():
        verify_recorded_file(root, source_record, "Receipt source video")
    return {
        "episode": episode,
        "episode_name": episode_name,
        "episode_root": root,
        "source": "archive_v2_receipt",
        "source_video": source,
        "source_video_record": source_record,
        "id_srt_source": paths["id_srt"],
        "tr_srt_source": paths["tr_srt"],
        "legacy_mkv": None,
        "legacy_records": {key: dict(value) for key, value in legacy.items()},
        "receipt": dict(receipt),
        "plan": plan,
        "source_av_stream_hashes": list(source_hashes),
        "pilot_warnings": [dict(item) for item in raw_warnings],
        **paths,
    }


def load_archive_v2_inputs(
    *,
    episode_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    episode: int,
) -> dict[str, Any]:
    """Load a prior v2 receipt, or the current verified FINALIZE report."""

    root = validate_episode_root(episode_root, episode)
    _, episode_name = _episode_identity(episode)
    receipt_file = _receipt_path(receipt_path, root, episode_name)
    if receipt_file.exists() or receipt_file.is_symlink():
        return _load_receipt_inputs(
            root, episode, _read_json(receipt_file, "Archive v2 receipt")
        )
    return _load_finalization_inputs(root, episode)


def _ensure_safe_output_directories(root: Path) -> None:
    final = root / "final"
    if not final.exists():
        final.mkdir()
    details = final.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ArchiveV2Error("Refusing unsafe final directory")
    subtitles = final / "subtitles"
    if not subtitles.exists():
        subtitles.mkdir()
    details = subtitles.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ArchiveV2Error("Refusing unsafe final/subtitles directory")
    if final.resolve(strict=True).parent != root:
        raise ArchiveV2Error("final directory resolves outside the episode folder")
    if subtitles.resolve(strict=True).parent != final.resolve(strict=True):
        raise ArchiveV2Error("final/subtitles resolves outside the final directory")


def _atomic_copy_verified(source: Path, destination: Path, root: Path) -> None:
    """Publish one byte-exact copy while refusing an incompatible existing file."""

    source_record = file_record(source, root)
    if destination.is_symlink():
        raise ArchiveV2Error(f"Refusing symlinked canonical subtitle: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ArchiveV2Error(f"Canonical subtitle is not a regular file: {destination}")
        if destination.stat().st_size != source.stat().st_size:
            raise ArchiveV2Error(f"Existing canonical subtitle differs: {destination}")
        # file_record computes the SHA-256 while also rejecting links and escapes.
        destination_record = file_record(destination, root)
        if destination_record["sha256"] != source_record["sha256"]:
            raise ArchiveV2Error(f"Existing canonical subtitle differs: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if file_record(temporary, root)["sha256"] != source_record["sha256"]:
            raise ArchiveV2Error(f"Canonical subtitle copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_mux_pass(report: Mapping[str, Any]) -> None:
    stream_hashes = report.get("stream_hashes")
    if not isinstance(stream_hashes, Mapping):
        raise ArchiveV2Error("MKV verification has no stream hashes")
    if not (
        report.get("verified") is True
        and report.get("video_audio_stream_copy") is True
        and report.get("subtitle_order") == ["ind", "tur"]
        and report.get("indonesian_default") is True
        and report.get("turkish_default") is False
        and report.get("roundtrip", {}).get("exact") is True
        and stream_hashes.get("checked") is True
        and stream_hashes.get("match") is True
        and stream_hashes.get("source") == stream_hashes.get("output")
    ):
        raise ArchiveV2Error("MKV stream-copy or subtitle round-trip verification failed")


def _existing_mkv_report(
    mkv: Path,
    id_srt: Path,
    tr_srt: Path,
    source_hashes: list[dict[str, Any]],
) -> dict[str, Any]:
    roundtrip = verify_mkv_roundtrip(mkv, id_srt, tr_srt)
    output_hashes = compute_av_stream_hashes(mkv)
    report = {
        "verified": True,
        "output_path": str(mkv),
        "output_size_bytes": mkv.stat().st_size,
        "video_audio_stream_copy": True,
        "subtitle_order": ["ind", "tur"],
        "indonesian_default": True,
        "turkish_default": False,
        "roundtrip": roundtrip,
        "stream_hashes": {
            "checked": True,
            "match": output_hashes == source_hashes,
            "source": source_hashes,
            "output": output_hashes,
        },
        "resumed": True,
    }
    _require_mux_pass(report)
    return report


def _allowed_video_paths(inputs: Mapping[str, Any], canonical_mkv: Path) -> set[str]:
    root = Path(inputs["episode_root"])
    allowed = {canonical_mkv.relative_to(root).as_posix()}
    source = Path(inputs["source_video"])
    if source.exists() or source.is_symlink():
        allowed.add(source.relative_to(root).as_posix())
    for record in inputs.get("legacy_records", {}).values():
        if not isinstance(record, Mapping):
            continue
        relative = record.get("relative_path")
        if isinstance(relative, str) and PurePosixPath(relative).suffix.casefold() in VIDEO_SUFFIXES:
            allowed.add(relative)
    prior = inputs.get("plan")
    if isinstance(prior, Mapping):
        for record in prior.get("delete", []):
            relative = record.get("relative_path") if isinstance(record, Mapping) else None
            if isinstance(relative, str) and PurePosixPath(relative).suffix.casefold() in VIDEO_SUFFIXES:
                allowed.add(relative)
    return allowed


def _reject_unexpected_video(
    snapshot: Mapping[str, Any], allowed_video_paths: set[str]
) -> None:
    unexpected = sorted(
        record["relative_path"]
        for record in snapshot["files"]
        if PurePosixPath(record["relative_path"]).suffix.casefold() in VIDEO_SUFFIXES
        and record["relative_path"] not in allowed_video_paths
    )
    if unexpected:
        raise ArchiveV2Error(
            "Unexpected video file(s) block archive cleanup: " + ", ".join(unexpected)
        )


def ensure_verified_archive_v2(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Create/canonicalize and freshly verify the single retained MKV and SRTs."""

    root = validate_episode_root(inputs["episode_root"], inputs["episode"])
    paths = canonical_archive_paths(root, inputs["episode"])
    _ensure_safe_output_directories(root)
    before = _scan_tree(root)
    _reject_unexpected_video(before, _allowed_video_paths(inputs, paths["mkv"]))

    _atomic_copy_verified(Path(inputs["id_srt_source"]), paths["id_srt"], root)
    _atomic_copy_verified(Path(inputs["tr_srt_source"]), paths["tr_srt"], root)

    source = Path(inputs["source_video"])
    source_hashes = inputs.get("source_av_stream_hashes")
    if source.exists():
        verify_recorded_file(root, inputs["source_video_record"], "Source video")
        source_hashes = compute_av_stream_hashes(source)
    if not isinstance(source_hashes, list) or not source_hashes:
        raise ArchiveV2Error("Cannot verify canonical MKV without source A/V hashes")

    target = paths["mkv"]
    if target.is_symlink():
        raise ArchiveV2Error(f"Refusing symlinked canonical MKV: {target}")
    legacy_mkv = inputs.get("legacy_mkv")
    if target.exists():
        if not target.is_file():
            raise ArchiveV2Error(f"Canonical MKV is not a regular file: {target}")
        report = _existing_mkv_report(
            target, paths["id_srt"], paths["tr_srt"], source_hashes
        )
    elif isinstance(legacy_mkv, Path) and legacy_mkv.exists():
        # Validate the old FINALIZE MKV first, then rename it atomically.  This
        # avoids temporarily allocating a third full-size episode video.
        _existing_mkv_report(
            legacy_mkv, paths["id_srt"], paths["tr_srt"], source_hashes
        )
        os.replace(legacy_mkv, target)
        report = _existing_mkv_report(
            target, paths["id_srt"], paths["tr_srt"], source_hashes
        )
        report["canonicalized_legacy_mkv"] = True
    else:
        if not source.is_file():
            raise ArchiveV2Error("Cannot build a missing canonical MKV after source removal")
        report = dict(
            mux_softsubs(
                source,
                paths["id_srt"],
                paths["tr_srt"],
                target,
                verify_stream_hashes=True,
            )
        )
        report["resumed"] = False
    _require_mux_pass(report)
    return {
        "paths": paths,
        "mkv_report": report,
        "source_av_stream_hashes": list(source_hashes),
    }


def plan_archive_v2_cleanup(
    inputs: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact deletion plan, retaining only the three canonical files."""

    root = validate_episode_root(inputs["episode_root"], inputs["episode"])
    paths = verification.get("paths")
    if not isinstance(paths, Mapping):
        raise ArchiveV2Error("Archive v2 verification paths are missing")
    protected = sorted(_canonical_relatives(inputs["episode"]).values())
    for key, expected in _canonical_relatives(inputs["episode"]).items():
        if Path(paths.get(key, "")) != root / PurePosixPath(expected):
            raise ArchiveV2Error(f"Archive v2 canonical {key} path mismatch")
    snapshot = _scan_tree(root)
    _reject_unexpected_video(snapshot, _allowed_video_paths(inputs, Path(paths["mkv"])))
    present = {record["relative_path"] for record in snapshot["files"]}
    if not set(protected).issubset(present):
        raise ArchiveV2Error("One or more canonical archive artifacts are missing")
    delete = [
        dict(record)
        for record in snapshot["files"]
        if record["relative_path"] not in protected
    ]
    fingerprint_payload = {
        "episode": inputs["episode"],
        "files": snapshot["files"],
        "directories": snapshot["directories"],
        "delete": delete,
        "protected": protected,
    }
    return {
        "plan_version": PLAN_VERSION,
        "episode": inputs["episode"],
        "tree_fingerprint": sha256_json(fingerprint_payload),
        "tree": snapshot,
        "delete": delete,
        "protected": protected,
        "delete_file_count": len(delete),
        "delete_size_bytes": sum(item["size_bytes"] for item in delete),
    }


def _verification_receipt(
    inputs: Mapping[str, Any],
    verification: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(inputs["episode_root"])
    paths = verification["paths"]
    report = verification["mkv_report"]
    _require_mux_pass(report)
    return {
        "receipt_version": RECEIPT_VERSION,
        "layout_version": LAYOUT_VERSION,
        "status": "READY",
        "created_at": utc_now_iso(),
        "episode": inputs["episode"],
        "episode_name": inputs["episode_name"],
        "artifacts": {
            key: file_record(paths[key], root) for key in ("mkv", "id_srt", "tr_srt")
        },
        "legacy_inputs": {
            key: dict(value) for key, value in inputs.get("legacy_records", {}).items()
        },
        "source_av_stream_hashes": list(verification["source_av_stream_hashes"]),
        "pilot_warnings": [
            dict(item) for item in inputs.get("pilot_warnings", [])
        ],
        "verification": {
            "verified": True,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": dict(report["roundtrip"]),
            "stream_hashes_checked": True,
            "stream_hashes_match": True,
        },
        "cleanup": {"plan": dict(plan)},
    }


def _verify_canonical_artifacts(
    root: Path, receipt: Mapping[str, Any], *, verify_remaining_legacy: bool
) -> None:
    episode = receipt["episode"]
    artifacts = receipt["artifacts"]
    relative = _canonical_relatives(episode)
    paths = {
        key: verify_recorded_file(
            root, artifacts[key], key, expected_relative_path=relative[key]
        )
        for key in ("mkv", "id_srt", "tr_srt")
    }
    source_hashes = receipt.get("source_av_stream_hashes")
    _existing_mkv_report(paths["mkv"], paths["id_srt"], paths["tr_srt"], source_hashes)
    if not verify_remaining_legacy:
        return
    cleanup = receipt.get("cleanup")
    if not isinstance(cleanup, Mapping):
        raise ArchiveV2Error("Archive v2 receipt cleanup record is missing")
    plan = _validate_plan_shape(cleanup.get("plan"), episode)
    planned_delete = {item["relative_path"] for item in plan["delete"]}
    for key, record in receipt.get("legacy_inputs", {}).items():
        if not isinstance(record, Mapping):
            raise ArchiveV2Error(f"Invalid legacy receipt record: {key}")
        record_relative = record.get("relative_path")
        if not isinstance(record_relative, str):
            raise ArchiveV2Error(f"Legacy receipt record has no path: {key}")
        _, path = _safe_relative(
            root,
            record_relative,
            f"Legacy {key}",
            allow_missing_parent=True,
        )
        if path.exists() or path.is_symlink():
            if record_relative not in planned_delete:
                raise ArchiveV2Error(f"Legacy file is not in cleanup plan: {record_relative}")
            verify_recorded_file(root, record, f"Legacy {key}")


def _remaining_plan(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only a byte-for-byte tree subset caused by prior planned deletes."""

    current = _scan_tree(root)
    prior_files = {item["relative_path"]: item for item in plan["tree"]["files"]}
    prior_delete = {item["relative_path"] for item in plan["delete"]}
    current_paths: set[str] = set()
    for record in current["files"]:
        relative = record["relative_path"]
        current_paths.add(relative)
        if prior_files.get(relative) != record:
            raise ArchiveV2Error(
                "Episode tree changed after archive v2 preview: " + relative
            )
    missing = set(prior_files) - current_paths
    if not missing.issubset(prior_delete):
        raise ArchiveV2Error("A protected archive v2 artifact disappeared")
    prior_directories = set(plan["tree"]["directories"])
    if not set(current["directories"]).issubset(prior_directories):
        raise ArchiveV2Error("Episode directory tree changed after archive v2 preview")
    remaining_delete = [
        dict(prior_files[path]) for path in sorted(current_paths & prior_delete)
    ]
    return {
        **dict(plan),
        "current_tree": current,
        "delete": remaining_delete,
        "delete_file_count": len(remaining_delete),
        "delete_size_bytes": sum(item["size_bytes"] for item in remaining_delete),
    }


def _remove_empty_work_directories(root: Path, directories: list[str]) -> list[str]:
    removed: list[str] = []
    for relative in sorted(directories, key=lambda value: (value.count("/"), value), reverse=True):
        if relative in {"final", "final/subtitles"}:
            continue
        _, path = _safe_relative(root, relative, "Cleanup directory")
        if not path.exists():
            continue
        try:
            path.rmdir()
        except OSError as exc:
            if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                continue
            raise ArchiveV2Error(f"Cannot remove empty cleanup directory: {relative}") from exc
        removed.append(relative)
    return removed


def _assert_completed_layout(root: Path, episode: int) -> None:
    snapshot = _scan_tree(root)
    expected_files = set(_canonical_relatives(episode).values())
    actual_files = {item["relative_path"] for item in snapshot["files"]}
    if actual_files != expected_files:
        raise ArchiveV2Error("Completed archive v2 does not contain exactly three files")
    if set(snapshot["directories"]) != {"final", "final/subtitles"}:
        raise ArchiveV2Error("Completed archive v2 has unexpected directories")


def execute_archive_v2_cleanup(
    inputs: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    confirmation: str | None,
) -> dict[str, Any]:
    """Freshly verify, then unlink only unchanged records from the signed-off plan."""

    expected = expected_cleanup_confirmation_v2(inputs["episode"])
    if confirmation != expected:
        raise ArchiveV2Error(f"Cleanup confirmation must match exactly: {expected}")
    root = validate_episode_root(inputs["episode_root"], inputs["episode"])
    cleanup = receipt.get("cleanup")
    if not isinstance(cleanup, Mapping):
        raise ArchiveV2Error("Archive v2 receipt cleanup record is missing")
    plan = _validate_plan_shape(cleanup.get("plan"), inputs["episode"])
    remaining = _remaining_plan(root, plan)
    _reject_unexpected_video(
        remaining["current_tree"],
        _allowed_video_paths(inputs, Path(inputs["mkv"])),
    )
    _verify_canonical_artifacts(root, receipt, verify_remaining_legacy=True)

    # Work files go first.  Full-size source/legacy videos are intentionally
    # last, after every fresh verification and every mutation check has passed.
    ordered = sorted(
        remaining["delete"],
        key=lambda item: (
            PurePosixPath(item["relative_path"]).suffix.casefold() in VIDEO_SUFFIXES,
            item["relative_path"],
        ),
    )
    deleted: list[dict[str, Any]] = []
    for item in ordered:
        relative, path = _safe_relative(root, item["relative_path"], "Cleanup candidate")
        try:
            current = _file_snapshot(path, root)
        except FileNotFoundError as exc:
            raise ArchiveV2Error(
                f"Cleanup candidate disappeared before deletion: {relative}"
            ) from exc
        if current != item:
            raise ArchiveV2Error(f"Cleanup candidate changed after preview: {relative}")
        path.unlink()
        deleted.append(dict(item))

    removed = _remove_empty_work_directories(
        root, list(remaining["current_tree"]["directories"])
    )
    _assert_completed_layout(root, inputs["episode"])
    _verify_canonical_artifacts(root, receipt, verify_remaining_legacy=False)
    return {
        "deleted_file_count": len(deleted),
        "deleted_size_bytes": sum(item["size_bytes"] for item in deleted),
        "deleted_relative_paths": [item["relative_path"] for item in deleted],
        "removed_empty_directories": removed,
        "remaining_relative_paths": sorted(_canonical_relatives(inputs["episode"]).values()),
    }


def archive_episode_v2(
    *,
    episode_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    episode: int,
    delete_intermediates: bool = False,
    confirmation: str | None | Callable[[str], str] = None,
) -> dict[str, Any]:
    """Build/verify the v2 archive and optionally commit its exact cleanup plan."""

    if not isinstance(delete_intermediates, bool):
        raise ArchiveV2Error("delete_intermediates must be True or False")
    root = validate_episode_root(episode_root, episode)
    _, episode_name = _episode_identity(episode)
    receipt_file = _receipt_path(receipt_path, root, episode_name)
    inputs = load_archive_v2_inputs(
        episode_root=root, receipt_path=receipt_file, episode=episode
    )

    prior = inputs.get("receipt")
    if isinstance(prior, Mapping):
        _verify_canonical_artifacts(root, prior, verify_remaining_legacy=True)
        cleanup = prior.get("cleanup")
        if not isinstance(cleanup, Mapping):
            raise ArchiveV2Error("Archive v2 receipt cleanup record is missing")
        plan = _validate_plan_shape(cleanup.get("plan"), episode)
        remaining = _remaining_plan(root, plan)
        if prior.get("status") == "PASS":
            _assert_completed_layout(root, episode)
            result = dict(prior)
            result["no_op"] = True
            return result
        if remaining["delete_file_count"] == 0:
            _assert_completed_layout(root, episode)
            completed = dict(prior)
            completed["status"] = "PASS"
            completed["completed_at"] = utc_now_iso()
            completed["cleanup"] = {
                "plan": plan,
                "result": {
                    "deleted_file_count": 0,
                    "deleted_size_bytes": 0,
                    "deleted_relative_paths": [],
                    "removed_empty_directories": [],
                    "remaining_relative_paths": sorted(_canonical_relatives(episode).values()),
                },
            }
            _write_receipt(receipt_file, completed)
            completed["resumed_after_completed_cleanup"] = True
            return completed
        ready = dict(prior)
        ready["cleanup_preview"] = {
            "delete_file_count": remaining["delete_file_count"],
            "delete_size_bytes": remaining["delete_size_bytes"],
            "delete_relative_paths": [item["relative_path"] for item in remaining["delete"]],
        }
    else:
        verification = ensure_verified_archive_v2(inputs)
        inputs.update(verification["paths"])
        plan = plan_archive_v2_cleanup(inputs, verification)
        ready = _verification_receipt(inputs, verification, plan)
        _write_receipt(receipt_file, ready)
        ready["cleanup_preview"] = {
            "delete_file_count": plan["delete_file_count"],
            "delete_size_bytes": plan["delete_size_bytes"],
            "delete_relative_paths": [item["relative_path"] for item in plan["delete"]],
        }

    if not delete_intermediates:
        return ready
    resolved_confirmation = (
        confirmation(expected_cleanup_confirmation_v2(episode))
        if callable(confirmation)
        else confirmation
    )
    # Reload the external READY receipt so cleanup never trusts an in-memory
    # object that differs from the atomically published recovery checkpoint.
    persisted = _read_json(receipt_file, "Archive v2 receipt")
    cleanup_result = execute_archive_v2_cleanup(
        inputs, persisted, confirmation=resolved_confirmation
    )
    completed = dict(persisted)
    completed["status"] = "PASS"
    completed["completed_at"] = utc_now_iso()
    completed["cleanup"] = {"plan": plan, "result": cleanup_result}
    _write_receipt(receipt_file, completed)
    return completed


__all__ = [
    "ArchiveV2Error",
    "LAYOUT_VERSION",
    "PLAN_VERSION",
    "RECEIPT_VERSION",
    "archive_episode_v2",
    "canonical_archive_paths",
    "ensure_verified_archive_v2",
    "execute_archive_v2_cleanup",
    "expected_cleanup_confirmation_v2",
    "load_archive_v2_inputs",
    "plan_archive_v2_cleanup",
]
