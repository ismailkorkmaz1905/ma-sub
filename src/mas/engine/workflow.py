"""Pure orchestration helpers for the correction-first subtitle flow.

The heavy ASR and forced-alignment model calls live in their own modules.  The
helpers here only validate and connect their JSON contracts:

1. validated Turkish corrections are routed to alignment or explicit
   non-dialogue review;
2. the corrected dialogue is bound to the returned forced alignment;
3. aligned words are compared with the independent raw-audio VAD inventory;
4. only a fully covered, alignment-backed timeline can become a V2 schema and
   Indonesian-only translation pack.

No helper writes a notebook, touches Drive, or mutates an input object.  The
only file-producing function is the explicit ID-pack wrapper at the end.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .forced_align import validate_coarse_segments, validate_forced_alignment_data
from .id_translation import (
    DEFAULT_ID_BATCH_SIZE,
    create_id_translation_pack,
    validate_aligned_turkish_schema,
)
from .aligned_schema import build_aligned_turkish_schema
from .segmentation import SegmentationConfig, build_blocks
from .raw_asr import RawASRV2Config
from .speech_coverage import (
    SpeechCoverageConfig,
    analyze_speech_coverage,
    require_v2_beta_speech_coverage_config,
)
from .timing_qa import (
    TimingQAV2Config,
    assert_timing_qa_v2,
    run_timing_qa_v2,
)
from .audio_review import (
    AUDIO_REVIEW_V2_FORMAT,
    AUDIO_REVIEW_V2_VERSION,
    MACHINE_NOTE_PREFIX,
    MANUAL_NOTE_PREFIX,
)
from .tr_correction import (
    TRCorrectionError,
    compute_output_sha256,
    validate_asr_hallucination_records,
    validate_input_utterances,
    validate_speech_holes,
    validate_tr_correction_records,
)


DEFAULT_ALIGNMENT_PADDING_MS = 500
MIN_STRICT_WORD_VAD_OVERLAP_RATIO = 0.25
MAX_STRICT_WORD_VAD_EDGE_OUTSIDE_MS = 120

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


class V2PipelineError(RuntimeError):
    """Raised when adjacent V2 stages cannot be safely connected."""


@dataclass(frozen=True)
class AlignmentInputBundle:
    """Correction output routed to the acoustic-alignment and review paths."""

    alignment_inputs: tuple[dict[str, Any], ...]
    reviewed_non_dialogue: tuple[dict[str, Any], ...]
    reviewed_dialogue: tuple[dict[str, Any], ...]
    discarded_asr_hallucinations: tuple[dict[str, Any], ...]
    window_audit: tuple[dict[str, Any], ...]
    correction_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class V2PipelineArtifacts:
    """Pure, validated artifacts ready for the Indonesian translation stage."""

    episode: int
    preparation: AlignmentInputBundle
    audio_review_report: dict[str, Any]
    strict_word_vad_report: dict[str, Any]
    alignment_report: dict[str, Any]
    speech_coverage_report: dict[str, Any]
    segmented_blocks: tuple[dict[str, Any], ...]
    schema: dict[str, Any]
    timing_qa_report: dict[str, Any]


def _strict_json_copy(value: Any, label: str) -> Any:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise V2PipelineError(f"{label} is not strict JSON: {exc}") from exc


def _canonical_sha256(value: Any, label: str) -> str:
    trusted = _strict_json_copy(value, label)
    payload = json.dumps(
        trusted,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_episode(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise V2PipelineError("episode must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V2PipelineError(f"{label} must be a non-negative integer")
    return value


def correction_records_to_alignment_inputs(
    input_utterances: Sequence[Mapping[str, Any]],
    correction_records: Sequence[Mapping[str, Any]],
    *,
    speech_hole_records: Sequence[Mapping[str, Any]] = (),
    asr_hallucination_records: Sequence[Mapping[str, Any]] = (),
    alignment_padding_ms: int = DEFAULT_ALIGNMENT_PADDING_MS,
) -> AlignmentInputBundle:
    """Route exact, validated Turkish corrections to alignment or review.

    Coarse ASR boundaries are only search windows, not final cue boundaries.
    Dialogue windows therefore receive symmetric padding (500 ms by default),
    clamped at zero on the left.  Adjacent padded windows may overlap; the
    forced-alignment validator still rejects overlapping *aligned words*.

    No record may proceed while ``review_required`` is true.  A record marked
    ``non_dialogue`` can clear VAD coverage only when it is the exact immutable
    counterpart of an audio-reviewed raw speech-hole record. A separately
    reviewed ASR hallucination is discarded from alignment but is never sent
    to the VAD-clearing reviewed-non-dialogue path.
    """

    padding_ms = _nonnegative_int(alignment_padding_ms, "alignment_padding_ms")
    try:
        trusted_inputs = validate_input_utterances(input_utterances)
        trusted_holes = validate_speech_holes(speech_hole_records)
        trusted_hallucinations = validate_asr_hallucination_records(
            asr_hallucination_records
        )
    except TRCorrectionError as exc:
        raise V2PipelineError(f"Turkish correction evidence is invalid: {exc}") from exc

    # Preserve precise pipeline-level diagnostics before the correction
    # module applies its own global hole-link validator.  The full immutable
    # correction validator still runs immediately afterward.
    holes_by_uid = {hole["hole_uid"]: hole for hole in trusted_holes}
    hallucinations_by_uid = {
        candidate["utterance_uid"]: candidate
        for candidate in trusted_hallucinations
    }
    inputs_by_uid = {item["utterance_uid"]: item for item in trusted_inputs}
    if isinstance(correction_records, Sequence) and not isinstance(
        correction_records, (str, bytes, bytearray)
    ):
        for raw_record in correction_records:
            if not isinstance(raw_record, Mapping) or raw_record.get("non_dialogue") is not True:
                continue
            uid = raw_record.get("utterance_uid")
            trusted_input = inputs_by_uid.get(uid) if isinstance(uid, str) else None
            hole = holes_by_uid.get(uid) if isinstance(uid, str) else None
            hallucination = (
                hallucinations_by_uid.get(uid) if isinstance(uid, str) else None
            )
            input_flags = (
                set(trusted_input.get("risk_flags", []))
                if trusted_input is not None
                else set()
            )
            is_hole = hole is not None and "unresolved_vad_speech" in input_flags
            is_hallucination = (
                hallucination is not None
                and bool(
                    {
                        "suspected_asr_hallucination",
                        "orphan_youtube_caption",
                    }.intersection(input_flags)
                )
            )
            if not is_hole and not is_hallucination:
                raise V2PipelineError(
                    f"Non-dialogue correction {uid} is not an exact "
                    "audio-reviewed speech-hole or ASR-hallucination record"
                )
            trusted_review = hole if is_hole else hallucination
            assert trusted_review is not None
            if (
                raw_record.get("coarse_start_ms") != trusted_review["start_ms"]
                or raw_record.get("coarse_end_ms") != trusted_review["end_ms"]
            ):
                raise V2PipelineError(
                    f"Non-dialogue correction {uid} changed immutable review bounds"
                )
    try:
        trusted_corrections = validate_tr_correction_records(
            trusted_inputs,
            correction_records,
            speech_holes=trusted_holes,
            asr_hallucination_records=trusted_hallucinations,
        )
    except TRCorrectionError as exc:
        raise V2PipelineError(f"Turkish correction output is invalid: {exc}") from exc
    pending_review_uids = [
        str(record["utterance_uid"])
        for record in trusted_corrections
        if record["review_required"] is True
    ]
    if pending_review_uids:
        raise V2PipelineError(
            "TR correction review gate failed: "
            f"review_required_count={len(pending_review_uids)}, "
            f"utterance_uids={pending_review_uids}"
        )
    alignment_inputs: list[dict[str, Any]] = []
    reviewed_non_dialogue: list[dict[str, Any]] = []
    reviewed_dialogue: list[dict[str, Any]] = []
    discarded_asr_hallucinations: list[dict[str, Any]] = []
    window_audit: list[dict[str, Any]] = []
    for record in trusted_corrections:
        uid = str(record["utterance_uid"])
        coarse_start = int(record["coarse_start_ms"])
        coarse_end = int(record["coarse_end_ms"])
        is_non_dialogue = record["non_dialogue"] is True
        is_hallucination_discard = (
            is_non_dialogue and uid in hallucinations_by_uid
        )
        audit: dict[str, Any] = {
            "utterance_uid": uid,
            "utterance_index": int(record["utterance_index"]),
            "coarse_start_ms": coarse_start,
            "coarse_end_ms": coarse_end,
            "alignment_padding_ms": padding_ms,
            "review_required": bool(record["review_required"]),
            "route": (
                "discarded_asr_hallucination"
                if is_hallucination_discard
                else "reviewed_non_dialogue"
                if is_non_dialogue
                else "alignment"
            ),
            "audio_reviewed": bool(record["audio_reviewed"]),
            "review_disposition": str(record["review_disposition"]),
        }
        if is_non_dialogue:
            if is_hallucination_discard:
                candidate = hallucinations_by_uid[uid]
                discarded_asr_hallucinations.append(
                    {
                        "candidate_uid": candidate["candidate_uid"],
                        "utterance_uid": uid,
                        "start_ms": coarse_start,
                        "end_ms": coarse_end,
                        "disposition": "discarded_asr_hallucination",
                        "reason": str(record["note"]).strip(),
                    }
                )
                audit["alignment_start_ms"] = None
                audit["alignment_end_ms"] = None
                window_audit.append(audit)
                continue
            hole = holes_by_uid.get(uid)
            input_flags = set(record.get("risk_flags", []))
            if hole is None or "unresolved_vad_speech" not in input_flags:
                raise V2PipelineError(
                    f"Non-dialogue correction {uid} is not an exact "
                    "audio-reviewed speech-hole record"
                )
            if (
                int(hole["start_ms"]) != coarse_start
                or int(hole["end_ms"]) != coarse_end
            ):
                raise V2PipelineError(
                    f"Non-dialogue correction {uid} changed speech-hole bounds"
                )
            reviewed_non_dialogue.append(
                {
                    "review_id": f"tr-correction:{uid}",
                    "start_ms": coarse_start,
                    "end_ms": coarse_end,
                    "classification": "non_dialogue",
                    "review_status": "reviewed",
                    "reason": str(record["note"]).strip(),
                }
            )
            audit["alignment_start_ms"] = None
            audit["alignment_end_ms"] = None
        else:
            if uid in hallucinations_by_uid:
                candidate = hallucinations_by_uid[uid]
                reviewed_dialogue.append(
                    {
                        "review_id": f"asr-hallucination:{candidate['candidate_uid']}",
                        "start_ms": coarse_start,
                        "end_ms": coarse_end,
                        "classification": "dialogue",
                        "review_status": "reviewed",
                        "reason": str(record["note"]).strip(),
                    }
                )
            padded_start = max(0, coarse_start - padding_ms)
            padded_end = coarse_end + padding_ms
            alignment_inputs.append(
                {
                    "start_ms": padded_start,
                    "end_ms": padded_end,
                    "coarse_start_ms": coarse_start,
                    "coarse_end_ms": coarse_end,
                    "text": str(record["tr_corrected"]),
                    "asr_text": str(record["asr_text"]),
                    "deletion_audio_reviewed": bool(
                        uid in hallucinations_by_uid
                        and record["non_dialogue"] is False
                        and record["review_required"] is False
                        and record["audio_reviewed"] is True
                        and record["review_disposition"]
                        == "confirmed_dialogue"
                    ),
                    "audio_reviewed": bool(record["audio_reviewed"]),
                    "review_disposition": str(record["review_disposition"]),
                    "utterance_uid": uid,
                }
            )
            audit["alignment_start_ms"] = padded_start
            audit["alignment_end_ms"] = padded_end
        window_audit.append(audit)

    if not alignment_inputs:
        raise V2PipelineError("At least one corrected dialogue needs forced alignment")
    # Reuse the alignment module's exact public input contract.  It explicitly
    # allows padded search-window overlap.
    trusted_alignment_inputs = validate_coarse_segments(alignment_inputs)
    return AlignmentInputBundle(
        alignment_inputs=tuple(copy.deepcopy(trusted_alignment_inputs)),
        reviewed_non_dialogue=tuple(copy.deepcopy(reviewed_non_dialogue)),
        reviewed_dialogue=tuple(copy.deepcopy(reviewed_dialogue)),
        discarded_asr_hallucinations=tuple(
            copy.deepcopy(discarded_asr_hallucinations)
        ),
        window_audit=tuple(copy.deepcopy(window_audit)),
        correction_records=tuple(copy.deepcopy(trusted_corrections)),
    )


def _validate_raw_vad_inventory(
    raw_asr_data: Mapping[str, Any],
    *,
    episode: int | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_asr_data, Mapping):
        raise V2PipelineError("raw_asr_data must be a JSON object")
    trusted = _strict_json_copy(dict(raw_asr_data), "raw_asr_data")
    if trusted.get("format_version") != "2.0":
        raise V2PipelineError("raw_asr_data must use format_version 2.0")
    if trusted.get("status") != "completed":
        raise V2PipelineError("raw_asr_data is not a completed canonical artifact")
    input_sha = trusted.get("input_sha256")
    if (
        not isinstance(input_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", input_sha) is None
    ):
        raise V2PipelineError("raw_asr_data has no valid input_sha256")
    raw_episode = _positive_episode(trusted.get("episode"))
    if episode is not None and raw_episode != episode:
        raise V2PipelineError(
            f"raw_asr_data episode mismatch: expected {episode}, got {raw_episode}"
        )
    if trusted.get("language") not in (None, "tr"):
        raise V2PipelineError("raw_asr_data language must be Turkish")
    if trusted.get("synthetic_timing_count", 0) != 0:
        raise V2PipelineError("raw_asr_data contains synthetic timing")
    if trusted.get("independent_vad") is not True:
        raise V2PipelineError(
            "raw_asr_data is explicitly non-publishable without "
            "independent_vad=true"
        )
    audio_sha = trusted.get("audio_sha256")
    if not isinstance(audio_sha, str) or re.fullmatch(r"[0-9a-f]{64}", audio_sha) is None:
        raise V2PipelineError("raw_asr_data has no valid audio_sha256")

    # A VAD inventory derived from ASR word timestamps is not independent and
    # cannot prove that ASR did not miss speech.  The current V2 runtime uses
    # Silero VAD; any fallback is a hard stop rather than a silent downgrade.
    if trusted.get("vad_fallback_reason") not in (None, ""):
        raise V2PipelineError(
            "raw_asr_data has no independent VAD inventory "
            f"({trusted.get('vad_fallback_reason')})"
        )
    model = trusted.get("model")
    model_settings = model.get("settings") if isinstance(model, Mapping) else None
    if not isinstance(model_settings, Mapping):
        raise V2PipelineError("raw_asr_data model.settings is missing")
    actual_policy = _strict_json_copy(dict(model_settings), "raw ASR policy")
    expected_policy = _strict_json_copy(
        asdict(RawASRV2Config()), "canonical raw ASR policy"
    )
    if actual_policy.get("allow_cpu_fallback") is False:
        expected_policy["allow_cpu_fallback"] = False
    actual_review_uids = actual_policy.pop("extra_audio_review_uids", None)
    expected_policy.pop("extra_audio_review_uids", None)
    if (
        not isinstance(actual_review_uids, list)
        or any(
            not isinstance(uid, str) or not uid.strip()
            for uid in actual_review_uids
        )
        or len(set(actual_review_uids)) != len(actual_review_uids)
    ):
        raise V2PipelineError(
            "raw ASR extra_audio_review_uids must be a unique string list"
        )
    if actual_policy != expected_policy:
        changed = sorted(
            key
            for key in set(actual_policy).union(expected_policy)
            if actual_policy.get(key) != expected_policy.get(key)
        )
        raise V2PipelineError(
            "raw_asr_data was not produced with the canonical V2 beta policy; "
            f"changed_fields={changed}"
        )
    top_review_uids = trusted.get("hallucination_review_utterance_uids")
    if top_review_uids != actual_review_uids:
        raise V2PipelineError(
            "raw ASR explicit audio-review UID evidence mismatch"
        )
    audit_vad_pad = (
        model_settings.get("audit_vad_speech_pad_ms")
        if isinstance(model_settings, Mapping)
        else None
    )
    if (
        isinstance(audit_vad_pad, bool)
        or not isinstance(audit_vad_pad, int)
        or audit_vad_pad < 0
    ):
        raise V2PipelineError(
            "raw_asr_data does not preserve model.settings."
            "audit_vad_speech_pad_ms"
        )
    vad_regions = trusted.get("vad_regions")
    if (
        isinstance(vad_regions, (str, bytes, bytearray))
        or not isinstance(vad_regions, list)
        or not vad_regions
    ):
        raise V2PipelineError("raw_asr_data vad_regions must be a non-empty list")
    for position, region in enumerate(vad_regions, start=1):
        if not isinstance(region, dict):
            raise V2PipelineError(f"vad_regions[{position}] must be an object")
        if region.get("source") != "silero_vad":
            raise V2PipelineError(
                f"vad_regions[{position}] is not independent Silero VAD evidence"
            )
    return trusted


def _assert_alignment_matches_inputs(
    forced_alignment_data: Mapping[str, Any],
    alignment_inputs: Sequence[Mapping[str, Any]],
) -> None:
    validate_forced_alignment_data(forced_alignment_data)
    segments = forced_alignment_data.get("segments")
    assert isinstance(segments, Sequence)  # Proven by the validator above.
    if len(segments) != len(alignment_inputs):
        raise V2PipelineError(
            "Forced alignment segment count does not match corrected dialogue: "
            f"expected {len(alignment_inputs)}, got {len(segments)}"
        )
    for position, (segment, expected) in enumerate(
        zip(segments, alignment_inputs), start=1
    ):
        assert isinstance(segment, Mapping)
        comparisons = (
            ("utterance_uid", "utterance_uid"),
            ("text", "text"),
            ("asr_text", "asr_text"),
            ("deletion_audio_reviewed", "deletion_audio_reviewed"),
            ("audio_reviewed", "audio_reviewed"),
            ("review_disposition", "review_disposition"),
            ("alignment_window_start_ms", "start_ms"),
            ("alignment_window_end_ms", "end_ms"),
            ("coarse_start_ms", "coarse_start_ms"),
            ("coarse_end_ms", "coarse_end_ms"),
        )
        for actual_field, expected_field in comparisons:
            actual_value = segment.get(actual_field)
            expected_value = expected.get(expected_field)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise V2PipelineError(
                    f"Forced alignment record {position} changed {actual_field}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )


def _assert_same_audio(
    trusted_raw: Mapping[str, Any],
    forced_alignment_data: Mapping[str, Any],
) -> str:
    raw_audio_sha = trusted_raw.get("audio_sha256")
    alignment_audio_sha = forced_alignment_data.get("audio_sha256")
    if (
        not isinstance(alignment_audio_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", alignment_audio_sha) is None
    ):
        raise V2PipelineError("forced_alignment_data has no valid audio_sha256")
    if alignment_audio_sha != raw_audio_sha:
        raise V2PipelineError(
            "Forced alignment belongs to stale or different audio: "
            f"raw={raw_audio_sha}, alignment={alignment_audio_sha}"
        )
    return alignment_audio_sha


def _coverage_config_from_raw(
    trusted_raw: Mapping[str, Any],
) -> SpeechCoverageConfig:
    """Rebuild the exact raw-pass coverage policy for the final comparison."""

    raw_coverage = trusted_raw.get("speech_coverage")
    if not isinstance(raw_coverage, Mapping):
        raise V2PipelineError(
            "raw_asr_data does not preserve the speech coverage report"
        )
    raw_config = raw_coverage.get("config")
    if not isinstance(raw_config, Mapping):
        raise V2PipelineError(
            "raw_asr_data does not preserve the exact speech coverage config"
        )
    missing = sorted(_COVERAGE_CONFIG_FIELDS.difference(raw_config))
    extra = sorted(set(raw_config).difference(_COVERAGE_CONFIG_FIELDS))
    if missing or extra:
        raise V2PipelineError(
            "raw speech coverage config field mismatch; "
            f"missing={missing}, extra={extra}"
        )
    try:
        config = SpeechCoverageConfig(
            **{name: raw_config[name] for name in _COVERAGE_CONFIG_FIELDS}
        )
    except (TypeError, ValueError) as exc:
        raise V2PipelineError(
            f"raw speech coverage config is invalid: {exc}"
        ) from exc
    try:
        require_v2_beta_speech_coverage_config(config)
    except ValueError as exc:
        raise V2PipelineError(
            f"raw speech coverage config is not publishable: {exc}"
        ) from exc
    return config


def _audio_review_report(
    preparation: AlignmentInputBundle,
    *,
    speech_holes: Sequence[Mapping[str, Any]],
    asr_hallucinations: Sequence[Mapping[str, Any]],
    acoustic_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every exceptional TR disposition to immutable WAV evidence."""

    trusted_acoustic_audit: dict[str, Any] | None = None
    audit_outcomes_by_uid: dict[str, Mapping[str, Any]] = {}
    machine_or_manual_records = [
        correction
        for correction in preparation.correction_records
        if str(correction.get("note", "")).startswith(
            (MACHINE_NOTE_PREFIX, MANUAL_NOTE_PREFIX)
        )
    ]
    if acoustic_audit is None and machine_or_manual_records:
        raise V2PipelineError(
            "Colab/manual audio decisions require the hash-bound "
            "audio_review_v2.json audit"
        )
    if acoustic_audit is not None:
        trusted_acoustic_audit = _strict_json_copy(
            dict(acoustic_audit), "acoustic_audio_review_v2"
        )
        if (
            trusted_acoustic_audit.get("format") != AUDIO_REVIEW_V2_FORMAT
            or trusted_acoustic_audit.get("format_version")
            != AUDIO_REVIEW_V2_VERSION
            or trusted_acoustic_audit.get("status") != "PASS"
            or trusted_acoustic_audit.get("pending_count") != 0
        ):
            raise V2PipelineError("Acoustic audio-review audit is not a final PASS")
        digest = trusted_acoustic_audit.get("audio_review_sha256")
        expected_digest = _canonical_sha256(
            {
                key: value
                for key, value in trusted_acoustic_audit.items()
                if key != "audio_review_sha256"
            },
            "acoustic_audio_review_v2",
        )
        if digest != expected_digest:
            raise V2PipelineError("Acoustic audio-review audit digest mismatch")
        correction_sha = compute_output_sha256(preparation.correction_records)
        if (
            trusted_acoustic_audit.get("correction_output_sha256")
            != correction_sha
        ):
            raise V2PipelineError(
                "Acoustic audio-review audit belongs to different Turkish corrections"
            )
        raw_audit_outcomes = trusted_acoustic_audit.get("outcomes")
        if not isinstance(raw_audit_outcomes, list) or any(
            not isinstance(item, Mapping) for item in raw_audit_outcomes
        ):
            raise V2PipelineError("Acoustic audio-review outcomes are malformed")
        audit_uids = [str(item.get("utterance_uid", "")) for item in raw_audit_outcomes]
        if len(set(audit_uids)) != len(audit_uids) or any(not uid for uid in audit_uids):
            raise V2PipelineError("Acoustic audio-review outcome UIDs are invalid")
        audit_outcomes_by_uid = {
            uid: item for uid, item in zip(audit_uids, raw_audit_outcomes)
        }

    holes_by_uid = {str(item["hole_uid"]): item for item in speech_holes}
    hallucinations_by_uid = {
        str(item["utterance_uid"]): item for item in asr_hallucinations
    }
    outcomes: list[dict[str, Any]] = []
    for correction in preparation.correction_records:
        disposition = str(correction["review_disposition"])
        if disposition == "not_applicable":
            continue
        if disposition == "pending_audio_review":
            raise V2PipelineError(
                f"Audio review is still pending for {correction['utterance_uid']}"
            )
        uid = str(correction["utterance_uid"])
        if uid in hallucinations_by_uid:
            evidence_kind = "asr_hallucination_candidate"
            evidence = hallucinations_by_uid[uid]
            evidence_uid = str(evidence["candidate_uid"])
        elif uid in holes_by_uid:
            evidence_kind = "speech_hole"
            evidence = holes_by_uid[uid]
            evidence_uid = str(evidence["hole_uid"])
        else:
            raise V2PipelineError(
                f"Audio-reviewed correction {uid} has no immutable WAV evidence"
            )
        if correction["audio_reviewed"] is not True:
            raise V2PipelineError(
                f"Audio-reviewed disposition for {uid} lacks audio_reviewed=true"
            )
        outcome = {
                "utterance_uid": uid,
                "utterance_index": int(correction["utterance_index"]),
                "start_ms": int(evidence["start_ms"]),
                "end_ms": int(evidence["end_ms"]),
                "clip_start_ms": int(evidence["clip_start_ms"]),
                "clip_end_ms": int(evidence["clip_end_ms"]),
                "evidence_kind": evidence_kind,
                "evidence_uid": evidence_uid,
                "audio_member": str(evidence["audio_member"]),
                "audio_sha256": str(evidence["audio_sha256"]),
                "audio_size_bytes": int(evidence["audio_size_bytes"]),
                "audio_reviewed": True,
                "review_disposition": disposition,
                "non_dialogue": bool(correction["non_dialogue"]),
                "review_note": str(correction["note"]).strip(),
            }
        if trusted_acoustic_audit is not None:
            acoustic_outcome = audit_outcomes_by_uid.get(uid)
            if acoustic_outcome is None:
                raise V2PipelineError(
                    f"Acoustic audio-review audit is missing {uid}"
                )
            text_sha = hashlib.sha256(
                str(correction["tr_corrected"]).encode("utf-8")
            ).hexdigest()
            comparisons = {
                "evidence_kind": evidence_kind.replace(
                    "asr_hallucination_candidate", "asr_caption_candidate"
                ),
                "evidence_uid": evidence_uid,
                "audio_member": evidence["audio_member"],
                "audio_sha256": evidence["audio_sha256"],
                "decision": disposition,
                "tr_corrected_sha256": text_sha,
            }
            for field, expected in comparisons.items():
                if acoustic_outcome.get(field) != expected:
                    raise V2PipelineError(
                        f"Acoustic audio-review outcome {uid} changed {field}"
                    )
            outcome["review_source"] = acoustic_outcome.get("source")
            outcome["acoustic_audit_sha256"] = trusted_acoustic_audit[
                "audio_review_sha256"
            ]
        outcomes.append(outcome)

    expected_reviewed_uids = set(holes_by_uid).union(hallucinations_by_uid)
    actual_reviewed_uids = {str(item["utterance_uid"]) for item in outcomes}
    if actual_reviewed_uids != expected_reviewed_uids:
        raise V2PipelineError(
            "Every immutable review WAV must have one final disposition; "
            f"missing={sorted(expected_reviewed_uids - actual_reviewed_uids)}, "
            f"extra={sorted(actual_reviewed_uids - expected_reviewed_uids)}"
        )
    if trusted_acoustic_audit is not None and set(audit_outcomes_by_uid) != expected_reviewed_uids:
        raise V2PipelineError(
            "Acoustic audio-review inventory mismatch; "
            f"missing={sorted(expected_reviewed_uids - set(audit_outcomes_by_uid))}, "
            f"extra={sorted(set(audit_outcomes_by_uid) - expected_reviewed_uids)}"
        )

    counts = {
        disposition: sum(
            item["review_disposition"] == disposition for item in outcomes
        )
        for disposition in (
            "confirmed_dialogue",
            "reviewed_non_dialogue",
            "discarded_asr_hallucination",
        )
    }
    draft: dict[str, Any] = {
        "report_version": "2.0",
        "status": "PASS",
        "reviewed_outcome_count": len(outcomes),
        "confirmed_dialogue_count": counts["confirmed_dialogue"],
        "reviewed_non_dialogue_count": counts["reviewed_non_dialogue"],
        "discarded_asr_hallucination_count": counts[
            "discarded_asr_hallucination"
        ],
        "pending_audio_review_count": 0,
        "outcomes": outcomes,
    }
    draft["audio_review_sha256"] = _canonical_sha256(
        draft, "audio_review_report"
    )
    return draft


def _strict_word_vad_gate(
    trusted_raw: Mapping[str, Any],
    trusted_alignment: Mapping[str, Any],
    audio_review_report: Mapping[str, Any],
    coverage_config: SpeechCoverageConfig,
) -> dict[str, Any]:
    """Reject aligned words supported only by negligible Silero overlap.

    The raw audit Silero intervals are unmerged but already include the exact
    digest-bound audit pad (currently 60 ms).  The sole exception is a word
    wholly inside the exact bounds of the same utterance's immutable,
    WAV-hash-bound ``audio_reviewed=true, confirmed_dialogue`` decision.
    """

    vad_cores = [
        (int(region["start_ms"]), int(region["end_ms"]))
        for region in trusted_raw["vad_regions"]
    ]
    confirmed_by_uid = {
        str(outcome["utterance_uid"]): outcome
        for outcome in audio_review_report["outcomes"]
        if outcome["review_disposition"] == "confirmed_dialogue"
    }
    confirmed_dialogue_coverage: list[dict[str, Any]] = []
    for uid, review in confirmed_by_uid.items():
        target_start = int(review["start_ms"])
        target_end = int(review["end_ms"])
        target_words = [
            word
            for word in trusted_alignment["words"]
            if str(word["utterance_uid"]) == uid
            and int(word["start_ms"]) < target_end
            and target_start < int(word["end_ms"])
        ]
        required_report = analyze_speech_coverage(
            [{"start_ms": target_start, "end_ms": target_end}],
            target_words,
            config=coverage_config,
        )
        confirmed_dialogue_coverage.append(
            {
                "utterance_uid": uid,
                "review_evidence_uid": review["evidence_uid"],
                "review_audio_sha256": review["audio_sha256"],
                "start_ms": target_start,
                "end_ms": target_end,
                "aligned_word_count": len(target_words),
                "status": required_report["status"],
                "unresolved_speech_region_count": required_report[
                    "unresolved_speech_region_count"
                ],
                "speech_coverage_ratio": required_report[
                    "speech_coverage_ratio"
                ],
                "coverage_issues": required_report["coverage_issues"],
            }
        )
    violations: list[dict[str, Any]] = []
    exception_words: list[dict[str, Any]] = []
    words = trusted_alignment["words"]
    for position, word in enumerate(words, start=1):
        start_ms = int(word["start_ms"])
        end_ms = int(word["end_ms"])
        duration_ms = end_ms - start_ms
        intersections = [
            (max(start_ms, vad_start), min(end_ms, vad_end))
            for vad_start, vad_end in vad_cores
            if start_ms < vad_end and vad_start < end_ms
        ]
        overlap_ms = sum(end - start for start, end in intersections)
        overlap_ratio = overlap_ms / duration_ms
        if intersections:
            leading_outside_ms = intersections[0][0] - start_ms
            trailing_outside_ms = end_ms - intersections[-1][1]
        else:
            leading_outside_ms = duration_ms
            trailing_outside_ms = duration_ms
        uncovered_runs: list[int] = []
        cursor = start_ms
        for overlap_start, overlap_end in intersections:
            if overlap_start > cursor:
                uncovered_runs.append(overlap_start - cursor)
            cursor = max(cursor, overlap_end)
        if cursor < end_ms:
            uncovered_runs.append(end_ms - cursor)
        maximum_contiguous_non_vad_ms = max(uncovered_runs, default=0)
        compliant = (
            overlap_ratio >= MIN_STRICT_WORD_VAD_OVERLAP_RATIO
            and maximum_contiguous_non_vad_ms
            <= coverage_config.max_word_outside_speech_ms
        )
        if compliant:
            continue

        uid = str(word["utterance_uid"])
        review = confirmed_by_uid.get(uid)
        exact_review_exception = bool(
            review is not None
            and review["audio_reviewed"] is True
            and int(review["start_ms"]) <= start_ms
            and end_ms <= int(review["end_ms"])
        )
        audit = {
            "word_index": int(word.get("word_index", position)),
            "utterance_uid": uid,
            "text": str(word["text"]),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "strict_vad_overlap_ms": overlap_ms,
            "strict_vad_overlap_ratio": round(overlap_ratio, 6),
            "leading_outside_vad_ms": leading_outside_ms,
            "trailing_outside_vad_ms": trailing_outside_ms,
            "maximum_contiguous_non_vad_ms": maximum_contiguous_non_vad_ms,
        }
        if exact_review_exception:
            exception_words.append(
                {
                    **audit,
                    "review_evidence_uid": review["evidence_uid"],
                    "review_audio_sha256": review["audio_sha256"],
                    "review_start_ms": review["start_ms"],
                    "review_end_ms": review["end_ms"],
                }
            )
        else:
            violations.append(audit)

    confirmed_dialogue_coverage_fail_count = sum(
        item["status"] != "PASS"
        or item["unresolved_speech_region_count"] != 0
        for item in confirmed_dialogue_coverage
    )
    draft: dict[str, Any] = {
        "report_version": "2.0",
        "status": (
            "PASS"
            if not violations and confirmed_dialogue_coverage_fail_count == 0
            else "FAIL"
        ),
        "policy": {
            "vad_source": "silero_vad",
            "vad_intervals": "audit_silero_padded_unmerged",
            "audit_vad_speech_pad_ms": trusted_raw["model"]["settings"][
                "audit_vad_speech_pad_ms"
            ],
            "minimum_word_vad_overlap_ratio": (
                MIN_STRICT_WORD_VAD_OVERLAP_RATIO
            ),
            "maximum_leading_outside_vad_ms": (
                coverage_config.max_word_outside_speech_ms
            ),
            "maximum_trailing_outside_vad_ms": (
                coverage_config.max_word_outside_speech_ms
            ),
            "maximum_contiguous_non_vad_ms": (
                coverage_config.max_word_outside_speech_ms
            ),
            # Silero intervals already carry the canonical 60 ms audit pad;
            # the 120 ms word-edge allowance therefore reaches at most 180 ms
            # beyond the detector's unpadded speech core in this beta policy.
            "maximum_effective_distance_from_unpadded_vad_core_ms": (
                trusted_raw["model"]["settings"]["audit_vad_speech_pad_ms"]
                + coverage_config.max_word_outside_speech_ms
            ),
            "exception": (
                "exact_utterance_uid_and_review_bounds_with_hash_bound_"
                "audio_reviewed_confirmed_dialogue"
            ),
        },
        "raw_asr_input_sha256": trusted_raw["input_sha256"],
        "audio_sha256": trusted_raw["audio_sha256"],
        "alignment_sha256": trusted_alignment["alignment_sha256"],
        "audio_review_sha256": audio_review_report["audio_review_sha256"],
        "aligned_word_count": len(words),
        "confirmed_dialogue_exception_word_count": len(exception_words),
        "unsafe_word_count": len(violations),
        "confirmed_dialogue_required_coverage_count": len(
            confirmed_dialogue_coverage
        ),
        "confirmed_dialogue_coverage_fail_count": (
            confirmed_dialogue_coverage_fail_count
        ),
        "confirmed_dialogue_exception_words": exception_words,
        "confirmed_dialogue_required_coverage": confirmed_dialogue_coverage,
        "unsafe_words": violations,
    }
    draft["strict_word_vad_sha256"] = _canonical_sha256(
        draft, "strict_word_vad_report"
    )
    return draft


def recompute_final_speech_coverage(
    raw_asr_data: Mapping[str, Any],
    forced_alignment_data: Mapping[str, Any],
    reviewed_non_dialogue: Sequence[Mapping[str, Any]],
    *,
    config: SpeechCoverageConfig | None = None,
    expected_alignment_inputs: Sequence[Mapping[str, Any]] | None = None,
    reviewed_dialogue: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compare final forced-aligned words with independent raw-audio VAD."""

    trusted_raw = _validate_raw_vad_inventory(raw_asr_data)
    validate_forced_alignment_data(forced_alignment_data)
    _assert_same_audio(trusted_raw, forced_alignment_data)
    if expected_alignment_inputs is not None:
        _assert_alignment_matches_inputs(
            forced_alignment_data,
            expected_alignment_inputs,
        )
    words = forced_alignment_data.get("words")
    assert isinstance(words, Sequence)  # Proven by forced-alignment validation.
    reviews = _strict_json_copy(
        list(reviewed_non_dialogue),
        "reviewed_non_dialogue",
    )
    dialogue_reviews = _strict_json_copy(
        list(reviewed_dialogue),
        "reviewed_dialogue",
    )
    effective_config = config or _coverage_config_from_raw(trusted_raw)
    if not isinstance(effective_config, SpeechCoverageConfig):
        raise V2PipelineError("config must be SpeechCoverageConfig")
    report = analyze_speech_coverage(
        trusted_raw["vad_regions"],
        words,
        config=effective_config,
        reviewed_non_dialogue=reviews,
        reviewed_dialogue=dialogue_reviews,
    )
    return copy.deepcopy(report)


def _validated_episode_and_inputs(
    raw_asr_data: Mapping[str, Any],
    episode: int | None,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    raw_episode_value = raw_asr_data.get("episode") if isinstance(raw_asr_data, Mapping) else None
    resolved_episode = (
        _positive_episode(raw_episode_value)
        if episode is None
        else _positive_episode(episode)
    )
    trusted_raw = _validate_raw_vad_inventory(
        raw_asr_data,
        episode=resolved_episode,
    )
    raw_inputs = trusted_raw.get("correction_utterances")
    if not isinstance(raw_inputs, list):
        raise V2PipelineError(
            "raw_asr_data is missing correction_utterances"
        )
    trusted_inputs = validate_input_utterances(raw_inputs)
    return resolved_episode, trusted_raw, trusted_inputs


def build_strict_v2_artifacts(
    raw_asr_data: Mapping[str, Any],
    correction_records: Sequence[Mapping[str, Any]],
    forced_alignment_data: Mapping[str, Any],
    *,
    episode: int | None = None,
    alignment_padding_ms: int = DEFAULT_ALIGNMENT_PADDING_MS,
    coverage_config: SpeechCoverageConfig | None = None,
    segmentation_config: SegmentationConfig | None = None,
    timing_qa_config: TimingQAV2Config | None = None,
    acoustic_audio_review: Mapping[str, Any] | None = None,
) -> V2PipelineArtifacts:
    """Build the strict aligned blocks/schema after external forced alignment."""

    resolved_episode, trusted_raw, input_utterances = _validated_episode_and_inputs(
        raw_asr_data,
        episode,
    )
    preparation = correction_records_to_alignment_inputs(
        input_utterances,
        correction_records,
        speech_hole_records=trusted_raw.get("speech_hole_records", []),
        asr_hallucination_records=trusted_raw.get(
            "asr_hallucination_records", []
        ),
        alignment_padding_ms=alignment_padding_ms,
    )
    trusted_alignment = _strict_json_copy(
        dict(forced_alignment_data),
        "forced_alignment_data",
    )
    _assert_alignment_matches_inputs(
        trusted_alignment,
        preparation.alignment_inputs,
    )
    audio_sha = _assert_same_audio(trusted_raw, trusted_alignment)

    # A caller may use ``recompute_final_speech_coverage`` with a custom policy
    # for diagnostics, but the schema/publication gate must use the exact
    # hash-bound policy preserved by raw ASR.  Otherwise a large min-hole or a
    # zero coverage ratio could turn genuinely missing speech into a false PASS.
    raw_coverage_config = _coverage_config_from_raw(trusted_raw)
    if coverage_config is not None and coverage_config != raw_coverage_config:
        raise V2PipelineError(
            "Publication coverage_config must exactly match the raw-ASR "
            "speech coverage policy"
        )

    coverage_report = recompute_final_speech_coverage(
        trusted_raw,
        trusted_alignment,
        preparation.reviewed_non_dialogue,
        config=raw_coverage_config,
        expected_alignment_inputs=preparation.alignment_inputs,
        reviewed_dialogue=preparation.reviewed_dialogue,
    )
    audio_review_report = _audio_review_report(
        preparation,
        speech_holes=validate_speech_holes(
            trusted_raw.get("speech_hole_records", [])
        ),
        asr_hallucinations=validate_asr_hallucination_records(
            trusted_raw.get("asr_hallucination_records", [])
        ),
        acoustic_audit=acoustic_audio_review,
    )
    strict_word_vad_report = _strict_word_vad_gate(
        trusted_raw,
        trusted_alignment,
        audio_review_report,
        raw_coverage_config,
    )
    # Both audits are included before schema construction, so their complete
    # contents are transitively bound by schema.speech_coverage_sha256.
    coverage_report["audio_review_v2"] = copy.deepcopy(audio_review_report)
    coverage_report["strict_word_vad_v2"] = copy.deepcopy(
        strict_word_vad_report
    )
    unresolved = coverage_report.get("unresolved_speech_region_count")
    if (
        unresolved != 0
        or coverage_report.get("status") != "PASS"
        or strict_word_vad_report.get("status") != "PASS"
        or strict_word_vad_report.get("unsafe_word_count") != 0
        or strict_word_vad_report.get(
            "confirmed_dialogue_coverage_fail_count"
        )
        != 0
    ):
        raise V2PipelineError(
            "Final aligned words do not safely match the independent VAD "
            "inventory: "
            f"unresolved_speech_region_count={unresolved!r}, "
            "strict_word_vad_unsafe_word_count="
            f"{strict_word_vad_report.get('unsafe_word_count')!r}, "
            "confirmed_dialogue_coverage_fail_count="
            f"{strict_word_vad_report.get('confirmed_dialogue_coverage_fail_count')!r}"
        )

    settings = segmentation_config or SegmentationConfig(schema_version="2.0")
    if not isinstance(settings, SegmentationConfig):
        raise V2PipelineError("segmentation_config must be SegmentationConfig")
    if str(settings.schema_version).split(".", 1)[0] != "2":
        raise V2PipelineError("segmentation_config must use a V2 schema_version")

    segmentation_vad_regions = copy.deepcopy(trusted_raw["vad_regions"])
    next_vad_index = max(
        (
            int(region.get("vad_region_index", position))
            for position, region in enumerate(segmentation_vad_regions, start=1)
        ),
        default=0,
    )
    for review in audio_review_report["outcomes"]:
        if review["review_disposition"] != "confirmed_dialogue":
            continue
        next_vad_index += 1
        segmentation_vad_regions.append(
            {
                "vad_region_index": next_vad_index,
                "start_ms": int(review["start_ms"]),
                "end_ms": int(review["end_ms"]),
                "source": "audio_reviewed_confirmed_dialogue",
                "utterance_uid": str(review["utterance_uid"]),
                "review_audio_sha256": str(review["audio_sha256"]),
            }
        )

    segmentation_input = {
        "words": copy.deepcopy(trusted_alignment["words"]),
        "vad_regions": segmentation_vad_regions,
        "youtube_captions": copy.deepcopy(
            trusted_raw.get("youtube_captions", [])
        ),
        "verification_results": [],
        "suspicious_spans": [],
    }
    segmented_blocks = build_blocks(
        segmentation_input,
        resolved_episode,
        config=settings,
    )
    forced_provenance = copy.deepcopy(trusted_alignment["provenance"])
    for position, block in enumerate(segmented_blocks, start=1):
        vad_info = block.get("vad_info")
        if not isinstance(vad_info, dict):
            raise V2PipelineError(
                f"Segmented block {position} lost its alignment evidence"
            )
        block_provenance = vad_info.get("alignment_provenance")
        if not isinstance(block_provenance, dict):
            raise V2PipelineError(
                f"Segmented block {position} lost alignment_provenance"
            )
        for name, value in forced_provenance.items():
            block_provenance.setdefault(name, copy.deepcopy(value))
        block_provenance["audio_sha256"] = audio_sha

    # Preserve the complete validated alignment object while mirroring the
    # strict count fields at the top level for schema/timing gate APIs.
    alignment_report = copy.deepcopy(trusted_alignment)
    alignment_report.update(copy.deepcopy(trusted_alignment["report"]))
    alignment_report["alignment_sha256"] = trusted_alignment["alignment_sha256"]
    schema = build_aligned_turkish_schema(
        segmented_blocks,
        episode=resolved_episode,
        alignment_report=alignment_report,
        speech_coverage_report=coverage_report,
        schema_version=settings.schema_version,
    )
    trusted_schema = validate_aligned_turkish_schema(schema)
    timing_report = run_timing_qa_v2(
        trusted_schema["blocks"],
        speech_coverage_report=coverage_report,
        alignment_report=alignment_report,
        config=timing_qa_config,
    )
    assert_timing_qa_v2(timing_report)
    return V2PipelineArtifacts(
        episode=resolved_episode,
        preparation=preparation,
        audio_review_report=copy.deepcopy(audio_review_report),
        strict_word_vad_report=copy.deepcopy(strict_word_vad_report),
        alignment_report=copy.deepcopy(alignment_report),
        speech_coverage_report=copy.deepcopy(coverage_report),
        segmented_blocks=tuple(copy.deepcopy(segmented_blocks)),
        schema=copy.deepcopy(trusted_schema),
        timing_qa_report=copy.deepcopy(timing_report),
    )


def create_v2_id_translation_pack(
    artifacts: V2PipelineArtifacts,
    out_zip: str | Path,
    *,
    batch_size: int = DEFAULT_ID_BATCH_SIZE,
    glossary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the deterministic ID-only pack from validated V2 artifacts."""

    if not isinstance(artifacts, V2PipelineArtifacts):
        raise V2PipelineError("artifacts must be V2PipelineArtifacts")
    if artifacts.timing_qa_report.get("passed") is not True:
        raise V2PipelineError("V2 timing QA has not passed")
    schema = validate_aligned_turkish_schema(artifacts.schema)
    if schema["episode"] != artifacts.episode:
        raise V2PipelineError("V2 artifact episode/schema mismatch")
    return create_id_translation_pack(
        schema,
        out_zip,
        batch_size=batch_size,
        glossary=glossary,
    )


# Discoverable aliases for notebook/integration callers.
prepare_forced_alignment_inputs = correction_records_to_alignment_inputs
combine_alignment_and_raw_coverage = recompute_final_speech_coverage
build_v2_pipeline_artifacts = build_strict_v2_artifacts


__all__ = [
    "AlignmentInputBundle",
    "DEFAULT_ALIGNMENT_PADDING_MS",
    "MAX_STRICT_WORD_VAD_EDGE_OUTSIDE_MS",
    "MIN_STRICT_WORD_VAD_OVERLAP_RATIO",
    "V2PipelineArtifacts",
    "V2PipelineError",
    "build_strict_v2_artifacts",
    "build_v2_pipeline_artifacts",
    "combine_alignment_and_raw_coverage",
    "correction_records_to_alignment_inputs",
    "create_v2_id_translation_pack",
    "prepare_forced_alignment_inputs",
    "recompute_final_speech_coverage",
]
