"""Coarse Turkish ASR for the correction-first workflow.

The timestamps emitted here are evidence for finding speech holes and slicing
the correction workload.  They are never a final subtitle timing authority.
Final word timings are produced later by ``forced_align`` after the Turkish
text has been corrected.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shutil
import wave
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from ..progress import mark_work_progress
from typing import Any, Iterable, Mapping, Sequence

from .download import (
    atomic_write_json,
    load_valid_stage_marker,
    sha256_file,
    sha256_json,
    write_stage_marker,
)
from .speech_coverage import (
    SpeechCoverageConfig,
    V2_BETA_COVERAGE_POLICY_LABEL,
    V2_BETA_SPEECH_COVERAGE_CONFIG,
    analyze_speech_coverage,
    require_v2_beta_speech_coverage_config,
)
from ..reliability import IntegrityError, UnitJournal
from . import primary_checkpoint
from .tr_correction import (
    MAX_ASR_HALLUCINATION_AUDIO_FILES,
    MAX_AUTOMATIC_ASR_HALLUCINATION_AUDIO_FILES,
    MAX_EXPLICIT_ASR_HALLUCINATION_AUDIO_FILES,
    MAX_SPEECH_HOLE_AUDIO_FILES,
    TRCorrectionError,
    validate_asr_hallucination_records,
    validate_input_utterances,
    validate_speech_holes,
)
from .transcribe import (
    TranscriptionConfig,
    TranscriptionError,
    _extract_clip,
    _extract_vad_regions,
    _import_whisper,
    _is_cuda_runtime_error,
    _package_version,
    _prompt_text,
    _select_device,
    load_vtt_captions,
)


RAW_ASR_V2_FORMAT_VERSION = "2.0"
V2_BETA_VAD_MIN_SPEECH_MS = 120
V2_BETA_AUDIT_VAD_SPEECH_PAD_MS = 60
V2_BETA_ORPHAN_CAPTION_EVIDENCE_COVERAGE_FLOOR = 0.50
# The speech-coverage policy's largest accepted between-word blind zone is
# 229 ms.  Caption evidence must not hide a contiguous interval larger than
# that simply because the other half of the caption is covered.
V2_BETA_ORPHAN_CAPTION_MAX_UNEXPLAINED_RUN_MS = 229
V2_BETA_ORPHAN_CAPTION_MIN_ABSOLUTE_EVIDENCE_MS = 120
V2_BETA_ORPHAN_CAPTION_TRAILING_DISPLAY_TOLERANCE_MS = 500
V2_BETA_CORRECTION_MAX_INTERNAL_WORD_GAP_MS = 650
V2_BETA_CORRECTION_REVIEW_WORD_GAP_MS = 5_000
V2_BETA_ORPHAN_CAPTION_MIN_LEXICAL_ASR_OVERLAP_MS = 120
V2_BETA_ORPHAN_CAPTION_MIN_SHARED_TOKENS = 2
V2_BETA_ORPHAN_CAPTION_MIN_CHARACTER_BIGRAM_DICE = 0.50
RAW_ASR_V2_RECOVERY_FORMAT = "raw-asr-v2-recovery-1"
RAW_ASR_V2_RECOVERY_FILENAME = "raw_asr_v2.recovery.json"
RAW_ASR_V2_UNIT_JOURNAL_FORMAT = "raw-asr-v2-unit-journal-1"
RAW_ASR_AUTH_KEY_ENV = "MAS_RAW_ASR_AUTH_KEY"
RAW_ASR_AUTH_DOMAIN = b"ma-sub/raw-asr-producer-auth/v1\0"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawASRV2Config:
    coverage_policy_label: str = V2_BETA_COVERAGE_POLICY_LABEL
    model_name: str = "large-v3"
    language: str = "tr"
    beam_size: int = 5
    best_of: int = 5
    compute_type_gpu: str = "float16"
    compute_type_cpu: str = "int8"
    allow_cpu_fallback: bool = False
    condition_on_previous_text: bool = False
    require_independent_vad: bool = True
    vad_threshold: float = 0.50
    vad_min_speech_ms: int = V2_BETA_VAD_MIN_SPEECH_MS
    vad_min_silence_ms: int = 500
    # faster-whisper's internal ASR filter keeps broader context; independent
    # audit VAD uses the tighter pad below so adjacent silence is not blessed.
    vad_speech_pad_ms: int = 200
    audit_vad_speech_pad_ms: int = V2_BETA_AUDIT_VAD_SPEECH_PAD_MS
    # Full-length dialogue policy. Short within-utterance pauses are normal and
    # must not become hundreds of false speech-hole rescue jobs.
    word_padding_ms: int = 120
    word_merge_gap_ms: int = 300
    max_word_outside_speech_ms: int = 120
    rescue_beam_size: int = 8
    rescue_padding_ms: int = 500
    rescue_min_hole_ms: int = 750
    rescue_min_coverage_ratio: float = 0.70
    # The fixed floor protects short/normal episodes. Long episodes may use a
    # proportional budget only when the primary pass already covers at least
    # 90% of independently detected speech. The hard cap still stops runaway
    # VAD/ASR mismatches before hundreds of targeted model calls are launched.
    rescue_max_spans: int = 96
    rescue_max_span_fraction: float = 0.11
    rescue_adaptive_min_coverage_ratio: float = 0.90
    rescue_hard_max_spans: int = 192
    rescue_mandatory_hard_max_spans: int = 256
    hallucination_max_vad_overlap_ratio: float = 0.20
    hallucination_min_avg_logprob: float = -0.80
    hallucination_max_no_speech_prob: float = 0.50
    hallucination_max_compression_ratio: float = 2.40
    hallucination_temperature_threshold: float = 0.0
    hallucination_text_markers: tuple[str, ...] = (
        "altyazı",
        "müzik",
        "music",
        "subtitle",
        "♪",
        "♫",
    )
    # Optional exact stable UIDs from a prior diagnostic run. Each requested
    # utterance receives the same immutable contextual-WAV review evidence as
    # an automatically detected hallucination candidate.
    extra_audio_review_uids: tuple[str, ...] = ()
    orphan_caption_min_evidence_coverage_ratio: float = (
        V2_BETA_ORPHAN_CAPTION_EVIDENCE_COVERAGE_FLOOR
    )
    orphan_caption_max_unexplained_run_ms: int = (
        V2_BETA_ORPHAN_CAPTION_MAX_UNEXPLAINED_RUN_MS
    )
    orphan_caption_min_absolute_evidence_ms: int = (
        V2_BETA_ORPHAN_CAPTION_MIN_ABSOLUTE_EVIDENCE_MS
    )
    orphan_caption_trailing_display_tolerance_ms: int = (
        V2_BETA_ORPHAN_CAPTION_TRAILING_DISPLAY_TOLERANCE_MS
    )
    correction_max_internal_word_gap_ms: int = (
        V2_BETA_CORRECTION_MAX_INTERNAL_WORD_GAP_MS
    )
    correction_review_word_gap_ms: int = V2_BETA_CORRECTION_REVIEW_WORD_GAP_MS
    orphan_caption_min_lexical_asr_overlap_ms: int = (
        V2_BETA_ORPHAN_CAPTION_MIN_LEXICAL_ASR_OVERLAP_MS
    )
    orphan_caption_min_shared_tokens: int = (
        V2_BETA_ORPHAN_CAPTION_MIN_SHARED_TOKENS
    )
    orphan_caption_min_character_bigram_dice: float = (
        V2_BETA_ORPHAN_CAPTION_MIN_CHARACTER_BIGRAM_DICE
    )
    review_clip_padding_ms: int = 750
    review_clip_min_duration_ms: int = 1_500

    def __post_init__(self) -> None:
        if (
            not isinstance(self.coverage_policy_label, str)
            or not self.coverage_policy_label.strip()
        ):
            raise ValueError("coverage_policy_label must be non-empty")
        if self.language != "tr":
            raise ValueError("V2 raw ASR requires Turkish")
        if not isinstance(self.condition_on_previous_text, bool):
            raise ValueError("condition_on_previous_text must be boolean")
        if not isinstance(self.require_independent_vad, bool):
            raise ValueError("require_independent_vad must be boolean")
        if (
            isinstance(self.audit_vad_speech_pad_ms, bool)
            or not isinstance(self.audit_vad_speech_pad_ms, int)
            or self.audit_vad_speech_pad_ms < 0
        ):
            raise ValueError("audit_vad_speech_pad_ms must be a non-negative integer")
        if self.beam_size < 1 or self.best_of < 1 or self.rescue_beam_size < 1:
            raise ValueError("beam sizes and best_of must be positive")
        if (
            isinstance(self.word_padding_ms, bool)
            or not isinstance(self.word_padding_ms, int)
            or self.word_padding_ms < 0
            or isinstance(self.word_merge_gap_ms, bool)
            or not isinstance(self.word_merge_gap_ms, int)
            or self.word_merge_gap_ms < 0
            or isinstance(self.max_word_outside_speech_ms, bool)
            or not isinstance(self.max_word_outside_speech_ms, int)
            or self.max_word_outside_speech_ms < 0
        ):
            raise ValueError("word coverage gap settings must be non-negative integers")
        if self.rescue_padding_ms < 0 or self.rescue_min_hole_ms < 1:
            raise ValueError("invalid rescue duration settings")
        if not 0 <= self.rescue_min_coverage_ratio <= 1:
            raise ValueError("rescue_min_coverage_ratio must be in [0, 1]")
        if not 0 <= self.hallucination_max_vad_overlap_ratio <= 1:
            raise ValueError("hallucination_max_vad_overlap_ratio must be in [0, 1]")
        for name in (
            "hallucination_min_avg_logprob",
            "hallucination_max_no_speech_prob",
            "hallucination_max_compression_ratio",
            "hallucination_temperature_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if (
            not isinstance(self.hallucination_text_markers, tuple)
            or not self.hallucination_text_markers
            or any(
                not isinstance(marker, str) or not marker.strip()
                for marker in self.hallucination_text_markers
            )
            or len(set(self.hallucination_text_markers))
            != len(self.hallucination_text_markers)
        ):
            raise ValueError(
                "hallucination_text_markers must be a non-empty unique tuple"
            )
        if (
            not isinstance(self.extra_audio_review_uids, tuple)
            or any(
                not isinstance(uid, str) or not uid.strip()
                for uid in self.extra_audio_review_uids
            )
            or len(set(self.extra_audio_review_uids))
            != len(self.extra_audio_review_uids)
        ):
            raise ValueError(
                "extra_audio_review_uids must be a unique tuple of non-empty UIDs"
            )
        if (
            isinstance(self.orphan_caption_min_evidence_coverage_ratio, bool)
            or not isinstance(
                self.orphan_caption_min_evidence_coverage_ratio, (int, float)
            )
            or not math.isfinite(
                float(self.orphan_caption_min_evidence_coverage_ratio)
            )
            or not 0.0
            <= float(self.orphan_caption_min_evidence_coverage_ratio)
            <= 1.0
        ):
            raise ValueError(
                "orphan_caption_min_evidence_coverage_ratio must be in [0, 1]"
            )
        if (
            isinstance(self.orphan_caption_max_unexplained_run_ms, bool)
            or not isinstance(self.orphan_caption_max_unexplained_run_ms, int)
            or self.orphan_caption_max_unexplained_run_ms < 0
        ):
            raise ValueError(
                "orphan_caption_max_unexplained_run_ms must be a "
                "non-negative integer"
            )
        if (
            isinstance(self.orphan_caption_min_absolute_evidence_ms, bool)
            or not isinstance(
                self.orphan_caption_min_absolute_evidence_ms, int
            )
            or self.orphan_caption_min_absolute_evidence_ms < 1
            or isinstance(
                self.orphan_caption_trailing_display_tolerance_ms, bool
            )
            or not isinstance(
                self.orphan_caption_trailing_display_tolerance_ms, int
            )
            or self.orphan_caption_trailing_display_tolerance_ms < 0
        ):
            raise ValueError("orphan caption duration settings are invalid")
        if (
            isinstance(self.correction_max_internal_word_gap_ms, bool)
            or not isinstance(self.correction_max_internal_word_gap_ms, int)
            or self.correction_max_internal_word_gap_ms < 1
            or type(self.correction_review_word_gap_ms) is not int
            or self.correction_review_word_gap_ms < self.correction_max_internal_word_gap_ms
            or isinstance(
                self.orphan_caption_min_lexical_asr_overlap_ms, bool
            )
            or not isinstance(
                self.orphan_caption_min_lexical_asr_overlap_ms, int
            )
            or self.orphan_caption_min_lexical_asr_overlap_ms < 1
            or isinstance(self.orphan_caption_min_shared_tokens, bool)
            or not isinstance(self.orphan_caption_min_shared_tokens, int)
            or self.orphan_caption_min_shared_tokens < 1
        ):
            raise ValueError("correction/caption lexical limits are invalid")
        if (
            isinstance(self.orphan_caption_min_character_bigram_dice, bool)
            or not isinstance(
                self.orphan_caption_min_character_bigram_dice, (int, float)
            )
            or not math.isfinite(
                float(self.orphan_caption_min_character_bigram_dice)
            )
            or not 0.0
            <= float(self.orphan_caption_min_character_bigram_dice)
            <= 1.0
        ):
            raise ValueError(
                "orphan_caption_min_character_bigram_dice must be in [0, 1]"
            )
        if self.rescue_max_spans < 1:
            raise ValueError("rescue_max_spans must be positive")
        if (
            isinstance(self.rescue_max_span_fraction, bool)
            or not isinstance(self.rescue_max_span_fraction, (int, float))
            or not math.isfinite(float(self.rescue_max_span_fraction))
            or not 0.0 < float(self.rescue_max_span_fraction) <= 1.0
        ):
            raise ValueError("rescue_max_span_fraction must be in (0, 1]")
        if (
            isinstance(self.rescue_adaptive_min_coverage_ratio, bool)
            or not isinstance(
                self.rescue_adaptive_min_coverage_ratio, (int, float)
            )
            or not math.isfinite(
                float(self.rescue_adaptive_min_coverage_ratio)
            )
            or not 0.0
            <= float(self.rescue_adaptive_min_coverage_ratio)
            <= 1.0
        ):
            raise ValueError(
                "rescue_adaptive_min_coverage_ratio must be in [0, 1]"
            )
        if (
            isinstance(self.rescue_hard_max_spans, bool)
            or not isinstance(self.rescue_hard_max_spans, int)
            or self.rescue_hard_max_spans < self.rescue_max_spans
        ):
            raise ValueError(
                "rescue_hard_max_spans must be an integer greater than or "
                "equal to rescue_max_spans"
            )
        if (type(self.rescue_mandatory_hard_max_spans) is not int
                or not self.rescue_hard_max_spans <= self.rescue_mandatory_hard_max_spans <= 256):
            raise ValueError("rescue_mandatory_hard_max_spans must be between rescue_hard_max_spans and 256")
        if (
            isinstance(self.review_clip_padding_ms, bool)
            or not isinstance(self.review_clip_padding_ms, int)
            or self.review_clip_padding_ms < 0
            or isinstance(self.review_clip_min_duration_ms, bool)
            or not isinstance(self.review_clip_min_duration_ms, int)
            or self.review_clip_min_duration_ms < 1
        ):
            raise ValueError("review clip settings are invalid")

    def transcription_config(self) -> TranscriptionConfig:
        return TranscriptionConfig(
            model_name=self.model_name,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            compute_type_gpu=self.compute_type_gpu,
            compute_type_cpu=self.compute_type_cpu,
            allow_cpu_fallback=self.allow_cpu_fallback,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_threshold=self.vad_threshold,
            vad_min_speech_ms=self.vad_min_speech_ms,
            vad_min_silence_ms=self.vad_min_silence_ms,
            vad_speech_pad_ms=self.vad_speech_pad_ms,
            targeted_verification=False,
        )

    def speech_coverage_config(self) -> SpeechCoverageConfig:
        """Return the hash-bound policy shared by raw rescue and final QA."""

        return SpeechCoverageConfig(
            word_padding_ms=self.word_padding_ms,
            word_merge_gap_ms=self.word_merge_gap_ms,
            min_hole_ms=self.rescue_min_hole_ms,
            min_region_coverage_ratio=self.rescue_min_coverage_ratio,
            rescue_padding_ms=self.rescue_padding_ms,
            max_word_outside_speech_ms=self.max_word_outside_speech_ms,
        )


def _settings_with_review_requests(
    config: RawASRV2Config | None,
    review_uids: Sequence[str],
) -> tuple[RawASRV2Config, list[str]]:
    settings = config or RawASRV2Config()
    if not isinstance(settings, RawASRV2Config):
        raise TranscriptionError("config must be RawASRV2Config")
    if isinstance(review_uids, (str, bytes, bytearray)) or not isinstance(
        review_uids, Sequence
    ):
        raise TranscriptionError(
            "hallucination_review_utterance_uids must be a sequence"
        )
    requested = list(settings.extra_audio_review_uids)
    for position, uid in enumerate(review_uids, start=1):
        if not isinstance(uid, str) or not uid.strip():
            raise TranscriptionError(
                "hallucination_review_utterance_uids"
                f"[{position}] must be a non-empty string"
            )
        normalized_uid = uid.strip()
        if normalized_uid not in requested:
            requested.append(normalized_uid)
    return (
        replace(settings, extra_audio_review_uids=tuple(requested)),
        requested,
    )


def _raw_asr_input_sha256(
    *,
    episode: int,
    audio_sha256: str,
    caption_sha256: str | None,
    settings: RawASRV2Config,
    canonical_names: Sequence[str],
    religious_terms: Sequence[str],
    requested_hallucination_reviews: Sequence[str],
) -> str:
    return sha256_json(
        {
            "format_version": RAW_ASR_V2_FORMAT_VERSION,
            "episode": episode,
            "audio_sha256": audio_sha256,
            "config": asdict(settings),
            "canonical_names": list(canonical_names),
            "religious_terms": list(religious_terms),
            "captions_sha256": caption_sha256,
            "hallucination_review_utterance_uids": list(
                requested_hallucination_reviews
            ),
        }
    )


def effective_rescue_span_limit(
    config: RawASRV2Config,
    *,
    vad_region_count: int,
    speech_coverage_ratio: float,
) -> int:
    """Return the bounded rescue budget for one primary ASR pass.

    The proportional allowance is available only for a healthy primary pass.
    This lets long dialogue-heavy episodes slightly exceed the fixed floor
    without allowing a broken VAD/ASR pairing to launch hundreds of rescues.
    """

    if not isinstance(config, RawASRV2Config):
        raise TranscriptionError("config must be RawASRV2Config")
    if (
        isinstance(vad_region_count, bool)
        or not isinstance(vad_region_count, int)
        or vad_region_count < 0
    ):
        raise TranscriptionError("vad_region_count must be a non-negative integer")
    if (
        isinstance(speech_coverage_ratio, bool)
        or not isinstance(speech_coverage_ratio, (int, float))
        or not math.isfinite(float(speech_coverage_ratio))
        or not 0.0 <= float(speech_coverage_ratio) <= 1.0
    ):
        raise TranscriptionError("speech_coverage_ratio must be in [0, 1]")

    if float(speech_coverage_ratio) < float(
        config.rescue_adaptive_min_coverage_ratio
    ):
        return config.rescue_max_spans
    proportional_limit = math.ceil(
        vad_region_count * float(config.rescue_max_span_fraction)
    )
    return min(
        config.rescue_hard_max_spans,
        max(config.rescue_max_spans, proportional_limit),
    )


def _bounded_rescue_batches(spans, *, limit, hard_limit, mandatory_hard_limit=None, trusted_intervals=()):
    absolute_limit = hard_limit if mandatory_hard_limit is None else mandatory_hard_limit
    if len(spans) > absolute_limit:
        raise TranscriptionError(
            f"Speech-hole rescue requires {len(spans)} spans, above hard limit {absolute_limit}"
        )
    ordinary_count = sum(not span.get("required_source_records") for span in spans)
    if ordinary_count > hard_limit:
        raise TranscriptionError(f"Ordinary speech-hole rescue requires {ordinary_count} spans, above hard limit {hard_limit}")
    batches = [dict(span, source_rescue_span_indices=[index])
               for index, span in enumerate(spans, start=1)]
    max_duration = max((span["end_ms"] - span["start_ms"] for span in spans), default=0)
    while len(batches) > limit:
        eligible = [
            (right["start_ms"] - left["end_ms"], index)
            for index, (left, right) in enumerate(zip(batches, batches[1:]))
            if right["end_ms"] - left["start_ms"] <= min(
                max_duration, left.get("max_duration_ms", max_duration),
                right.get("max_duration_ms", max_duration))
            and (not left.get("required_source_records") and not right.get("required_source_records")
                 or left.get("required_source_records") == right.get("required_source_records")
                 and left["end_ms"] == right["start_ms"])
            and not any(left["end_ms"] < trusted["end_ms"]
                        and trusted["start_ms"] < right["start_ms"]
                        for trusted in trusted_intervals)
        ]
        if not eligible:
            break
        _, index = min(eligible)
        left, right = batches[index:index + 2]
        merged = dict(left, end_ms=right["end_ms"])
        if "max_duration_ms" in left or "max_duration_ms" in right:
            merged["max_duration_ms"] = min(left.get("max_duration_ms", max_duration),
                                             right.get("max_duration_ms", max_duration))
        for key in ("issue_ids", "speech_region_indices", "reasons", "source_rescue_span_indices"):
            merged[key] = sorted(set(left.get(key, []) + right.get(key, [])))
        sources = left.get("required_source_records", []) + right.get("required_source_records", [])
        if sources:
            merged["required_source_records"] = [
                {"source_id": source_id, "source_sha256": digest}
                for source_id, digest in sorted({(item["source_id"], item["source_sha256"]) for item in sources})
            ]
        batches[index:index + 2] = [merged]
    ordinary_count = sum(not batch.get("required_source_records") for batch in batches)
    if len(batches) > limit and ordinary_count > limit:
        raise TranscriptionError(
            f"Speech-hole rescue cannot fit {limit} batches without exceeding "
            f"original maximum duration {max_duration} ms or crossing trusted dialogue; "
            f"remaining={len(batches)}"
        )
    for index, batch in enumerate(batches, start=1):
        batch["rescue_span_index"] = index
        batch["duration_ms"] = batch["end_ms"] - batch["start_ms"]
    return batches


def _rescue_budget_audit(batches, *, limit, hard_limit, mandatory_hard_limit=None):
    overflow = max(0, len(batches) - limit)
    return {
        "adaptive_target": limit,
        "hard_limit": hard_limit,
        "mandatory_hard_limit": hard_limit if mandatory_hard_limit is None else mandatory_hard_limit,
        "actual_batch_count": len(batches),
        "mandatory_batch_count": sum(bool(batch.get("required_source_records")) for batch in batches),
        "mandatory_overflow_count": overflow,
        "mandatory_overflow_reason": "preserve_required_incomplete_source_evidence" if overflow else None,
    }


def _plan_rescue_batches(initial_coverage, vad_regions, segments, words, settings):
    metrics = initial_coverage["metrics"]
    ratio = metrics.get("independent_vad_speech_coverage_ratio", metrics["speech_coverage_ratio"])
    limit = effective_rescue_span_limit(settings, vad_region_count=len(vad_regions), speech_coverage_ratio=ratio)
    hard_limit = (settings.rescue_hard_max_spans if ratio >= settings.rescue_adaptive_min_coverage_ratio else limit)
    batches = _bounded_rescue_batches(
        initial_coverage["rescue_spans"], limit=limit, hard_limit=hard_limit,
        mandatory_hard_limit=settings.rescue_mandatory_hard_max_spans,
        trusted_intervals=_coverage_intervals_for_segments(segments, words),
    )
    return batches, _rescue_budget_audit(
        batches, limit=limit, hard_limit=hard_limit,
        mandatory_hard_limit=settings.rescue_mandatory_hard_max_spans,
    )


def validate_raw_rescue_plan(raw_asr_data):
    try:
        saved_settings = dict(raw_asr_data["model"]["settings"])
        for key in ("extra_audio_review_uids", "hallucination_text_markers"):
            if key in saved_settings:
                saved_settings[key] = tuple(saved_settings[key])
        settings = RawASRV2Config(**saved_settings)
        if "rescue_mandatory_hard_max_spans" not in saved_settings:
            raise TranscriptionError("rescue policy lacks mandatory absolute ceiling")
        canonical = RawASRV2Config()
        for key in ("rescue_max_spans", "rescue_max_span_fraction", "rescue_adaptive_min_coverage_ratio",
                    "rescue_hard_max_spans", "rescue_mandatory_hard_max_spans"):
            if getattr(settings, key) != getattr(canonical, key):
                raise TranscriptionError(f"rescue policy is not canonical: {key}")
        require_v2_beta_speech_coverage_config(settings.speech_coverage_config())
        segments = [segment for segment in raw_asr_data["segments"]
                    if not str(segment.get("source", "main")).startswith("rescue")]
        vad_regions, words = raw_asr_data["vad_regions"], raw_asr_data["words"]
        expected_initial = _analyze_raw_speech_coverage(
            vad_regions, segments, words, config=settings.speech_coverage_config(),
        )
        expected_batches, expected_budget = _plan_rescue_batches(expected_initial, vad_regions, segments, words, settings)
        if raw_asr_data.get("initial_speech_coverage") != expected_initial:
            raise TranscriptionError("initial speech coverage differs from canonical primary evidence")
        if raw_asr_data.get("rescue_batches") != expected_batches:
            raise TranscriptionError("rescue batches differ from canonical primary evidence")
        if raw_asr_data.get("rescue_budget_audit") != expected_budget:
            raise TranscriptionError("rescue budget audit differs from canonical policy and batch identities")
    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptionError(f"canonical rescue execution proof is invalid: {exc}") from exc


def validate_publishable_raw_asr_v2_policy(
    raw_asr_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact canonical detector/coverage policy for publication.

    A raw run may use custom thresholds for diagnostics, but a self-consistent
    hash does not make a loose detector safe. Publication therefore binds all
    raw policy fields to :class:`RawASRV2Config` defaults. The sole variable
    field is ``extra_audio_review_uids`` because adding exact WAV reviews can
    only increase scrutiny; it cannot relax an automatic gate.
    """

    if not isinstance(raw_asr_data, Mapping):
        raise TranscriptionError("raw_asr_data must be a JSON object")
    model = raw_asr_data.get("model")
    if not isinstance(model, Mapping):
        raise TranscriptionError("raw_asr_data.model must be an object")
    actual_settings = model.get("settings")
    if not isinstance(actual_settings, Mapping):
        raise TranscriptionError("raw_asr_data.model.settings must be an object")
    try:
        normalized_actual = json.loads(
            json.dumps(
                dict(actual_settings),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        normalized_expected = json.loads(
            json.dumps(
                asdict(RawASRV2Config()),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(
            f"raw ASR policy settings are not strict JSON: {exc}"
        ) from exc
    if set(normalized_actual) != set(normalized_expected):
        missing = sorted(set(normalized_expected).difference(normalized_actual))
        extra = sorted(set(normalized_actual).difference(normalized_expected))
        raise TranscriptionError(
            "raw ASR policy field mismatch; "
            f"missing={missing}, extra={extra}"
        )
    variable_field = "extra_audio_review_uids"
    for field, expected in normalized_expected.items():
        if field == variable_field:
            continue
        if normalized_actual[field] != expected:
            raise TranscriptionError(
                f"raw ASR policy field {field} does not match the canonical "
                "V2 beta publication policy"
            )
    extra_review_uids = normalized_actual[variable_field]
    if (
        not isinstance(extra_review_uids, list)
        or any(not isinstance(uid, str) or not uid.strip() for uid in extra_review_uids)
        or len(set(extra_review_uids)) != len(extra_review_uids)
    ):
        raise TranscriptionError(
            "raw ASR policy extra_audio_review_uids is malformed"
        )
    for report_name in ("initial_speech_coverage", "speech_coverage"):
        report = raw_asr_data.get(report_name)
        if not isinstance(report, Mapping) or not isinstance(
            report.get("config"), Mapping
        ):
            raise TranscriptionError(
                f"raw_asr_data.{report_name}.config is missing"
            )
        try:
            coverage = SpeechCoverageConfig(**dict(report["config"]))
            require_v2_beta_speech_coverage_config(coverage)
        except (TypeError, ValueError) as exc:
            raise TranscriptionError(
                f"raw_asr_data.{report_name}.config is not the canonical "
                f"V2 beta policy: {exc}"
            ) from exc
    return dict(normalized_actual)


def _finite_ms(value: Any, label: str, *, offset_ms: int = 0) -> int:
    try:
        number = round(float(value) * 1000) + offset_ms
    except (TypeError, ValueError, OverflowError) as exc:
        raise TranscriptionError(f"Invalid {label}: {value!r}") from exc
    return number


def _optional_finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptionError(f"Invalid {label}: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise TranscriptionError(f"Invalid {label}: {value!r}")
    return number


def consume_coarse_segments(
    iterator: Iterable[Any], *, offset_ms: int = 0, source: str = "main"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read faster-whisper output without fabricating any missing word time."""

    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(iterator, start=1):
        start_ms = _finite_ms(getattr(segment, "start", None), "segment start", offset_ms=offset_ms)
        end_ms = _finite_ms(getattr(segment, "end", None), "segment end", offset_ms=offset_ms)
        text = str(getattr(segment, "text", "")).strip()
        if start_ms < 0 or end_ms <= start_ms:
            raise TranscriptionError(f"Invalid coarse segment interval {start_ms}-{end_ms}")
        local_words: list[dict[str, Any]] = []
        for candidate in getattr(segment, "words", None) or []:
            word_text = str(getattr(candidate, "word", ""))
            start = getattr(candidate, "start", None)
            end = getattr(candidate, "end", None)
            if not word_text.strip() or start is None or end is None:
                continue
            word_start = _finite_ms(start, "word start", offset_ms=offset_ms)
            word_end = _finite_ms(end, "word end", offset_ms=offset_ms)
            if word_start < 0 or word_end <= word_start:
                continue
            probability = getattr(candidate, "probability", None)
            record = {
                "word_index": 0,
                "segment_id": f"{source}-{ordinal}",
                "start_ms": word_start,
                "end_ms": word_end,
                "text": word_text,
                "probability": float(probability) if probability is not None else None,
                "timing_source": "faster_whisper_provisional",
            }
            local_words.append(record)
            words.append(record)
        segments.append(
            {
                "segment_id": f"{source}-{ordinal}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "word_timing_complete": _segment_has_complete_word_inventory(
                    {"text": text, "words": local_words}
                ),
                "words": local_words,
                "source": source,
                "asr_audit": {
                    "avg_logprob": _optional_finite_number(
                        getattr(segment, "avg_logprob", None),
                        "segment avg_logprob",
                    ),
                    "no_speech_prob": _optional_finite_number(
                        getattr(segment, "no_speech_prob", None),
                        "segment no_speech_prob",
                    ),
                    "compression_ratio": _optional_finite_number(
                        getattr(segment, "compression_ratio", None),
                        "segment compression_ratio",
                    ),
                    "temperature": _optional_finite_number(
                        getattr(segment, "temperature", None),
                        "segment temperature",
                    ),
                },
            }
        )
        mark_work_progress("raw_asr:" + source, completed=ordinal)
    words.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    for index, word in enumerate(words, start=1):
        word["word_index"] = index
    return segments, words


def _overlap_ms(first: Mapping[str, Any], second: Mapping[str, Any]) -> int:
    return max(
        0,
        min(int(first["end_ms"]), int(second["end_ms"]))
        - max(int(first["start_ms"]), int(second["start_ms"])),
    )


def merge_rescue_evidence(
    primary_segments: Sequence[Mapping[str, Any]],
    primary_words: Sequence[Mapping[str, Any]],
    rescue_segments: Sequence[Mapping[str, Any]],
    rescue_words: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge only genuinely new rescue evidence; never overwrite primary text."""

    merged_segments = [dict(item) for item in primary_segments]
    incomplete_ids = {
        str(item["segment_id"])
        for item in primary_segments
        if "words" in item and not _segment_has_complete_word_inventory(item)
    }
    for segment in rescue_segments:
        rescue_text = str(segment.get("text", "")).strip()
        if not rescue_text:
            continue
        rescue_normalized = _normalized_lexical_text(rescue_text)
        duplicate = any(
            str(existing.get("segment_id", "")) not in incomplete_ids
            and _overlap_ms(existing, segment) > 0
            and (
                _normalized_lexical_text(existing.get("text", ""))
                == rescue_normalized
                or (
                    _overlap_ms(existing, segment)
                    >= max(
                        1,
                        min(
                            int(existing["end_ms"]) - int(existing["start_ms"]),
                            int(segment["end_ms"]) - int(segment["start_ms"]),
                        )
                        // 2,
                    )
                    and f" {rescue_normalized} "
                    in (
                        " "
                        + _normalized_lexical_text(existing.get("text", ""))
                        + " "
                    )
                )
            )
            for existing in merged_segments
        )
        if not duplicate:
            merged_segments.append(dict(segment))
    merged_segments.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))

    merged_words = [dict(item) for item in primary_words]
    for word in rescue_words:
        duration = int(word["end_ms"]) - int(word["start_ms"])
        duplicate = any(
            str(existing.get("segment_id", "")) not in incomplete_ids
            and _overlap_ms(existing, word) >= max(1, min(duration, int(existing["end_ms"]) - int(existing["start_ms"])) // 2)
            and str(existing.get("text", "")).strip().casefold()
            == str(word.get("text", "")).strip().casefold()
            for existing in merged_words
        )
        if not duplicate:
            merged_words.append(dict(word))
    merged_words.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"]), str(item.get("text", ""))))
    for index, word in enumerate(merged_words, start=1):
        word["word_index"] = index
    return merged_segments, merged_words


def build_coverage_intervals(
    words: Sequence[Mapping[str, Any]], *, merge_touching: bool = True
) -> list[dict[str, int]]:
    """Return a strict interval union for VAD coverage analysis.

    Provisional faster-whisper words may overlap.  That makes them unsuitable
    as canonical subtitle timings, but their union is still valid evidence that
    ASR heard something in that portion of the independent VAD region.  The raw
    records remain untouched and are never forwarded to final segmentation.
    """

    intervals: list[tuple[int, int]] = []
    for position, word in enumerate(words, start=1):
        try:
            start_ms = int(word["start_ms"])
            end_ms = int(word["end_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(
                f"Malformed provisional word interval at position {position}"
            ) from exc
        if start_ms < 0 or end_ms <= start_ms:
            raise TranscriptionError(
                f"Invalid provisional word interval at position {position}: "
                f"{start_ms}-{end_ms}"
            )
        intervals.append((start_ms, end_ms))
    intervals.sort()
    merged: list[list[int]] = []
    for start_ms, end_ms in intervals:
        if merged and start_ms < merged[-1][1] + (1 if merge_touching else 0):
            merged[-1][1] = max(merged[-1][1], end_ms)
        else:
            merged.append([start_ms, end_ms])
    return [
        {"start_ms": start_ms, "end_ms": end_ms}
        for start_ms, end_ms in merged
    ]


def _normalized_lexical_text(text: Any) -> str:
    """Return punctuation-insensitive Unicode text for evidence comparison."""

    return " ".join(
        re.findall(r"[^\W_]+", str(text).casefold(), flags=re.UNICODE)
    )


def _is_known_subtitle_hallucination(text: Any) -> bool:
    return _normalized_lexical_text(text) in {"altyazı m k", "altyazı k m"}


def _segment_has_complete_word_inventory(segment: Mapping[str, Any]) -> bool:
    words = segment.get("words")
    if not isinstance(words, list) or not words:
        return False
    if any(
        not isinstance(word, Mapping)
        or not str(word.get("text", "")).strip()
        or type(word.get("start_ms")) is not int
        or type(word.get("end_ms")) is not int
        or word["start_ms"] < 0
        or word["end_ms"] <= word["start_ms"]
        for word in words
    ):
        return False
    ordered = sorted(words, key=lambda word: (word["start_ms"], word["end_ms"]))
    return bool(str(segment.get("text", "")).strip()) and (
        _normalized_lexical_text("".join(str(word["text"]) for word in ordered))
        == _normalized_lexical_text(segment["text"])
    )


def _coverage_intervals_for_segments(segments, words):
    # Keep the complete raw inventory for audit; incomplete owners cannot
    # conceal VAD holes just because they emitted some plausible word times.
    build_coverage_intervals(words)
    complete_words = {
        (str(segment["segment_id"]), word["start_ms"], word["end_ms"], word["text"])
        for segment in segments
        if _segment_has_complete_word_inventory(segment)
        for word in segment["words"]
    }
    return build_coverage_intervals([
        word for word in words
        if (str(word.get("segment_id", "")), word["start_ms"], word["end_ms"], word["text"])
        in complete_words
    ])


def _excluded_incomplete_segments(segments, *, episode):
    excluded = []
    for segment in segments:
        if _segment_has_complete_word_inventory(segment):
            continue
        records = build_correction_utterances([segment], episode=episode)
        excluded.append({
            "segment_id": str(segment["segment_id"]),
            "segment_sha256": sha256_json(segment),
            "utterance_uid": records[0]["utterance_uid"] if records else None,
            "start_ms": segment["start_ms"], "end_ms": segment["end_ms"],
            "disposition": "excluded_incomplete_asr_hypothesis",
            "reason": "incomplete_provisional_word_timing",
        })
    return excluded


def _analyze_raw_speech_coverage(vad_regions, segments, words, *, config):
    return analyze_speech_coverage(
        vad_regions, _coverage_intervals_for_segments(segments, words), config=config,
        required_uncovered_intervals=required_speech_coverage_regions(segments, vad_regions),
        include_required_outside_vad=True,
    )


def required_acoustic_review_regions(segments, vad_regions):
    regions = []
    for source in required_speech_coverage_regions(segments, vad_regions):
        cursor = source["start_ms"]
        outside = []
        for vad in vad_regions:
            if vad["end_ms"] <= cursor:
                continue
            if vad["start_ms"] >= source["end_ms"]:
                break
            if vad["start_ms"] > cursor:
                outside.append((cursor, vad["start_ms"]))
            cursor = max(cursor, vad["end_ms"])
        if cursor < source["end_ms"]:
            outside.append((cursor, source["end_ms"]))
        for start, end in outside:
            for start_ms in range(start, end, 10_000):
                regions.append(dict(source, start_ms=start_ms, end_ms=min(end, start_ms + 10_000)))
    return sorted(regions, key=lambda item: (item["start_ms"], item["end_ms"], item["source_id"]))


def required_speech_coverage_regions(segments, vad_regions):
    regions = []
    for segment in segments:
        if _segment_has_complete_word_inventory(segment):
            continue
        bounds = [{"start_ms": segment["start_ms"], "end_ms": segment["end_ms"]}]
        bounds.extend({"start_ms": word["start_ms"], "end_ms": word["end_ms"]}
                      for word in segment.get("words", [])
                      if isinstance(word, Mapping) and str(word.get("text", "")).strip()
                      and type(word.get("start_ms")) is int and type(word.get("end_ms")) is int
                      and word["start_ms"] >= 0 and word["end_ms"] > word["start_ms"])
        source_sha = sha256_json(segment)
        regions.extend(dict(bound, source_id=str(segment["segment_id"]), source_sha256=source_sha)
                       for bound in build_coverage_intervals(bounds, merge_touching=True))
    return sorted(regions, key=lambda item: (item["start_ms"], item["end_ms"], item["source_id"]))


def _split_segment_for_correction(
    segment: Mapping[str, Any], *, max_internal_word_gap_ms: int,
    review_word_gap_ms: int = V2_BETA_CORRECTION_REVIEW_WORD_GAP_MS,
) -> list[dict[str, Any]]:
    """Split one coarse ASR segment at provisional timed-word gaps.

    faster-whisper may place words tens of seconds apart in one segment after
    VAD time restoration. Treating the segment bounds as one utterance makes a
    stray early word look attached to much later real dialogue. Complete word
    timing lets us split that record without inventing any timestamp. If the
    timed words do not reproduce the segment text, retain the original record
    and mark its provisional timing incomplete instead of dropping untimed
    text.
    """

    copied = dict(segment)
    if "words" not in segment:
        return [copied]
    raw_words = segment.get("words")
    if (
        isinstance(raw_words, (str, bytes, bytearray))
        or not isinstance(raw_words, Sequence)
        or not raw_words
    ):
        copied["word_timing_complete"] = False
        return [copied]
    words: list[dict[str, Any]] = []
    for position, raw_word in enumerate(raw_words, start=1):
        if not isinstance(raw_word, Mapping):
            raise TranscriptionError(
                f"Coarse segment word {position} must be an object"
            )
        word_text = str(raw_word.get("text", ""))
        start_ms = raw_word.get("start_ms")
        end_ms = raw_word.get("end_ms")
        if (
            not word_text.strip()
            or isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or start_ms < 0
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
        ):
            copied["word_timing_complete"] = False
            return [copied]
        words.append(dict(raw_word))
    words.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    leading_gap = words[0]["start_ms"] - int(segment["start_ms"])
    trailing_gap = int(segment["end_ms"]) - words[-1]["end_ms"]
    if (leading_gap >= review_word_gap_ms or trailing_gap >= review_word_gap_ms
            or any(right["start_ms"] - left["end_ms"] >= review_word_gap_ms
                   for left, right in zip(words, words[1:]))):
        copied["provisional_word_gap_requires_audio_review"] = True
    reconstructed = "".join(str(word["text"]) for word in words).strip()
    if _normalized_lexical_text(reconstructed) != _normalized_lexical_text(
        segment.get("text", "")
    ):
        copied["word_timing_complete"] = False
        return [copied]

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if (
            current
            and int(word["start_ms"]) - int(current[-1]["end_ms"])
            > max_internal_word_gap_ms
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)

    split: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        derived = dict(segment)
        derived["segment_id"] = (
            f"{str(segment.get('segment_id', 'segment'))}-part-{group_index}"
        )
        derived["start_ms"] = int(group[0]["start_ms"])
        derived["end_ms"] = int(group[-1]["end_ms"])
        derived["text"] = "".join(
            str(word["text"]) for word in group
        ).strip()
        derived["words"] = [dict(word) for word in group]
        derived["word_timing_complete"] = True
        if ((group_index == 1 and leading_gap >= review_word_gap_ms)
                or (group_index == len(groups) and trailing_gap >= review_word_gap_ms)
                or (group_index > 1
             and group[0]["start_ms"] - groups[group_index - 2][-1]["end_ms"] >= review_word_gap_ms)
                or (group_index < len(groups)
                    and groups[group_index][0]["start_ms"] - group[-1]["end_ms"] >= review_word_gap_ms)):
            derived["provisional_word_gap_requires_audio_review"] = True
        split.append(derived)
    return split


def build_correction_utterances(
    segments: Sequence[Mapping[str, Any]],
    *,
    episode: int,
    youtube_captions: Sequence[Mapping[str, Any]] = (),
    context_count: int = 2,
    max_internal_word_gap_ms: int = (
        V2_BETA_CORRECTION_MAX_INTERNAL_WORD_GAP_MS
    ),
    review_word_gap_ms: int = V2_BETA_CORRECTION_REVIEW_WORD_GAP_MS,
) -> list[dict[str, Any]]:
    """Create stable coarse utterance records for Turkish correction."""

    if context_count < 0:
        raise ValueError("context_count cannot be negative")
    if (
        isinstance(max_internal_word_gap_ms, bool)
        or not isinstance(max_internal_word_gap_ms, int)
        or max_internal_word_gap_ms < 1
    ):
        raise ValueError("max_internal_word_gap_ms must be a positive integer")
    if type(review_word_gap_ms) is not int or review_word_gap_ms < max_internal_word_gap_ms:
        raise ValueError("review_word_gap_ms must be an integer >= max_internal_word_gap_ms")
    materialized: list[dict[str, Any]] = []
    for segment in segments:
        if not str(segment.get("text", "")).strip():
            continue
        materialized.extend(
            _split_segment_for_correction(
                segment,
                max_internal_word_gap_ms=max_internal_word_gap_ms,
                review_word_gap_ms=review_word_gap_ms,
            )
        )
    materialized.sort(
        key=lambda item: (
            int(item["start_ms"]),
            int(item["end_ms"]),
            str(item.get("segment_id", "")),
        )
    )
    records: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(materialized):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        identity = json.dumps(
            {"episode": episode, "start_ms": start_ms, "end_ms": end_ms, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        youtube_pieces: list[str] = []
        for caption in youtube_captions:
            try:
                overlap = min(end_ms, int(caption["end_ms"])) - max(
                    start_ms, int(caption["start_ms"])
                )
            except (KeyError, TypeError, ValueError):
                continue
            caption_text = str(caption.get("text", "")).strip()
            if overlap > 0 and caption_text and (
                not youtube_pieces or youtube_pieces[-1] != caption_text
            ):
                youtube_pieces.append(caption_text)
        before = " / ".join(
            str(item.get("text", "")).strip()
            for item in materialized[max(0, segment_index - context_count) : segment_index]
            if str(item.get("text", "")).strip()
        )
        after = " / ".join(
            str(item.get("text", "")).strip()
            for item in materialized[
                segment_index + 1 : segment_index + 1 + context_count
            ]
            if str(item.get("text", "")).strip()
        )
        risk_flags: list[str] = []
        if not bool(segment.get("word_timing_complete")):
            risk_flags.append("incomplete_provisional_word_timing")
        if segment.get("provisional_word_gap_requires_audio_review") is True:
            risk_flags.append("provisional_word_gap_requires_audio_review")
        if str(segment.get("source", "main")).startswith("rescue"):
            risk_flags.append("speech_hole_rescue_asr")
        records.append(
            {
                "utterance_uid": f"MA{episode:02d}-TR-{hashlib.sha256(identity).hexdigest()[:16]}",
                "utterance_index": len(records) + 1,
                "coarse_start_ms": start_ms,
                "coarse_end_ms": end_ms,
                "asr_text": text,
                "youtube_text": " ".join(youtube_pieces),
                "context_before": before,
                "context_after": after,
                "risk_flags": risk_flags,
                "asr_audit": dict(
                    segment.get(
                        "asr_audit",
                        {
                            "avg_logprob": None,
                            "no_speech_prob": None,
                            "compression_ratio": None,
                            "temperature": None,
                        },
                    )
                ),
            }
        )
    # Validate at the production boundary so a later pack creation cannot
    # discover a shape or ordering mismatch after the expensive ASR run.
    return validate_input_utterances(records) if records else []


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as audio:
            frame_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or frame_rate != 16_000
                or frame_count <= 0
            ):
                raise TranscriptionError(
                    "Review WAV must be non-empty mono 16 kHz PCM16"
                )
    except (OSError, EOFError, wave.Error) as exc:
        raise TranscriptionError(f"Could not inspect review WAV: {exc}") from exc
    return round(frame_count * 1000 / frame_rate)


def _extract_review_clip(
    source_audio: Path,
    clip_path: Path,
    *,
    target_start_ms: int,
    target_end_ms: int,
    padding_ms: int,
    minimum_duration_ms: int,
) -> tuple[int, int]:
    """Extract audible context while keeping decision target bounds separate."""

    target_duration = target_end_ms - target_start_ms
    desired_duration = max(
        minimum_duration_ms,
        target_duration + (2 * padding_ms),
    )
    clip_start_ms = max(0, target_start_ms - padding_ms)
    requested_end_ms = max(
        target_end_ms + padding_ms,
        clip_start_ms + desired_duration,
    )
    _extract_clip(source_audio, clip_path, clip_start_ms, requested_end_ms)
    actual_duration = _wav_duration_ms(clip_path)
    clip_end_ms = clip_start_ms + actual_duration

    # Near EOF, ffmpeg may return less trailing context than requested. Shift
    # the clip earlier and retry once so a short target remains judgeable.
    if actual_duration < desired_duration and clip_start_ms > 0:
        clip_start_ms = max(0, clip_end_ms - desired_duration)
        _extract_clip(source_audio, clip_path, clip_start_ms, requested_end_ms)
        actual_duration = _wav_duration_ms(clip_path)
        clip_end_ms = clip_start_ms + actual_duration
    if clip_start_ms > target_start_ms or clip_end_ms < target_end_ms:
        raise TranscriptionError(
            "Review WAV does not contain the complete immutable target interval"
        )
    return clip_start_ms, clip_end_ms


def _speech_hole_metadata(issue, *, episode, utterances):
    start_ms, end_ms = int(issue["start_ms"]), int(issue["end_ms"])
    prior = [item for item in utterances if int(item.get("coarse_end_ms", -1)) <= start_ms]
    following = [item for item in utterances if int(item.get("coarse_start_ms", 1 << 62)) >= end_ms]
    identity = json.dumps(
        {"episode": episode, "start_ms": start_ms, "end_ms": end_ms, "issue_id": issue.get("issue_id")},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    hole_uid = f"MA{episode:02d}-HOLE-{hashlib.sha256(identity).hexdigest()[:16]}"
    return {
        "hole_uid": hole_uid, "start_ms": start_ms, "end_ms": end_ms,
        "reason": ", ".join(str(item) for item in issue.get("reasons", [])) or "unresolved_speech",
        "context_before": str(prior[-1].get("asr_text", "")) if prior else "",
        "context_after": str(following[0].get("asr_text", "")) if following else "",
        "risk_flags": ["unresolved_vad_speech", "manual_audio_review_required"],
        "audio_member": f"speech_hole_audio/{hole_uid}.wav",
    }


def build_speech_hole_records(
    coverage_report: Mapping[str, Any],
    *,
    episode: int,
    audio_path: str | Path,
    audio_output_root: str | Path,
    utterances: Sequence[Mapping[str, Any]] = (),
    config: RawASRV2Config | None = None,
    existing_records: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Extract final unresolved VAD regions as auditable WAV-backed records."""

    settings = config or RawASRV2Config()
    if not isinstance(settings, RawASRV2Config):
        raise TranscriptionError("config must be RawASRV2Config")
    issues = coverage_report.get("coverage_issues", [])
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes, bytearray)):
        raise TranscriptionError("coverage_issues must be a sequence")
    unresolved_issues = [
        issue
        for issue in issues
        if isinstance(issue, Mapping)
        and issue.get("classification") == "unresolved_speech"
    ]
    if len(unresolved_issues) > MAX_SPEECH_HOLE_AUDIO_FILES:
        raise TranscriptionError(
            "Final unresolved speech-hole count exceeds the bounded WAV limit"
        )
    source_audio = Path(audio_path)
    if not source_audio.is_file() or source_audio.stat().st_size <= 0:
        raise TranscriptionError("Speech-hole source audio is missing or empty")
    output_root = Path(audio_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clip_root = output_root / "speech_hole_audio"
    clip_root.mkdir(parents=True, exist_ok=True)
    holes: list[dict[str, Any]] = []
    reusable_by_uid = {item["hole_uid"]: item for item in validate_speech_holes(existing_records)}
    for issue in unresolved_issues:
        canonical = _speech_hole_metadata(issue, episode=episode, utterances=utterances)
        clip_path = output_root / canonical["audio_member"]
        existing = reusable_by_uid.get(canonical["hole_uid"])
        if existing is not None and all(existing[key] == value for key, value in canonical.items()):
            _validate_persisted_review_audio(existing, prepare_dir=output_root, label="recovery speech-hole")
            holes.append(dict(existing, hole_index=len(holes) + 1))
            continue
        if clip_path.exists():
            retained = clip_root / "retained" / sha256_file(clip_path) / clip_path.name
            retained.parent.mkdir(parents=True, exist_ok=True)
            if not retained.exists():
                shutil.copy2(clip_path, retained)
            if sha256_file(retained) != sha256_file(clip_path):
                raise TranscriptionError("Could not preserve superseded speech-hole WAV")
        clip_start_ms, clip_end_ms = _extract_review_clip(
            source_audio,
            clip_path,
            target_start_ms=canonical["start_ms"],
            target_end_ms=canonical["end_ms"],
            padding_ms=settings.review_clip_padding_ms,
            minimum_duration_ms=settings.review_clip_min_duration_ms,
        )
        clip_size = clip_path.stat().st_size
        holes.append(
            {
                **canonical,
                "hole_index": len(holes) + 1,
                "clip_start_ms": clip_start_ms,
                "clip_end_ms": clip_end_ms,
                "audio_sha256": sha256_file(clip_path),
                "audio_size_bytes": clip_size,
            }
        )
    return validate_speech_holes(holes)


def _require_asr_hallucination_candidate_budget(
    *,
    automatic_count: int,
    explicit_only_count: int,
    total_count: int,
    reviewable_count: int,
    candidate_reason_counts: Mapping[str, int],
) -> None:
    """Keep detector runaways separate from bounded explicit review repair."""

    if automatic_count > MAX_AUTOMATIC_ASR_HALLUCINATION_AUDIO_FILES:
        raise TranscriptionError(
            "Automatically suspected ASR hallucination count exceeds the "
            "bounded WAV limit; "
            f"automatic_candidates={automatic_count}, "
            f"automatic_limit={MAX_AUTOMATIC_ASR_HALLUCINATION_AUDIO_FILES}"
        )
    if explicit_only_count > MAX_EXPLICIT_ASR_HALLUCINATION_AUDIO_FILES:
        raise TranscriptionError(
            "Explicit ASR/caption review request count exceeds the bounded WAV "
            "limit; "
            f"explicit_only_candidates={explicit_only_count}, "
            f"explicit_limit={MAX_EXPLICIT_ASR_HALLUCINATION_AUDIO_FILES}"
        )
    if total_count > MAX_ASR_HALLUCINATION_AUDIO_FILES:
        reason_summary = ",".join(
            f"{reason}={count}"
            for reason, count in sorted(candidate_reason_counts.items())
        )
        raise TranscriptionError(
            "Suspected ASR hallucination count exceeds the bounded WAV limit; "
            f"candidates={total_count}, "
            f"limit={MAX_ASR_HALLUCINATION_AUDIO_FILES}, "
            f"reviewable_utterances={reviewable_count}, "
            f"reasons={reason_summary or 'none'}"
        )


def build_asr_hallucination_records(
    utterances: Sequence[Mapping[str, Any]],
    vad_regions: Sequence[Mapping[str, Any]],
    *,
    episode: int,
    audio_path: str | Path,
    audio_output_root: str | Path,
    config: RawASRV2Config | None = None,
    review_request_uids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flag evidence-bound ordinary ASR utterances for mandatory audio review.

    Automatic candidates have zero/low unpadded independent-VAD overlap or a
    suspicious faster-whisper confidence/music signal.  Callers may also name
    exact stable utterance UIDs from an earlier diagnostic run.  These are
    *candidates*, not automatic deletions: every one receives an exact,
    hash-bound WAV. A later discard may remove only this linked ASR record and
    must never clear a missing independent-VAD speech issue.
    """

    settings = config or RawASRV2Config()
    if not isinstance(settings, RawASRV2Config):
        raise TranscriptionError("config must be RawASRV2Config")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise TranscriptionError("episode must be a positive integer")
    trusted_utterances = validate_input_utterances(utterances)
    incomplete = [item["utterance_uid"] for item in trusted_utterances
                  if "incomplete_provisional_word_timing" in item["risk_flags"]]
    if incomplete:
        raise TranscriptionError(
            f"Incomplete ASR hypotheses must be excluded before review WAV extraction: {incomplete}"
        )
    if isinstance(vad_regions, (str, bytes, bytearray)) or not isinstance(
        vad_regions, Sequence
    ):
        raise TranscriptionError("vad_regions must be a sequence")
    trusted_vad: list[tuple[int, int]] = []
    previous_end = -1
    for position, raw in enumerate(vad_regions, start=1):
        if not isinstance(raw, Mapping):
            raise TranscriptionError(f"VAD region {position} must be an object")
        start_ms = raw.get("start_ms")
        end_ms = raw.get("end_ms")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or start_ms < 0
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
        ):
            raise TranscriptionError(
                f"Invalid VAD region interval at position {position}"
            )
        if start_ms < previous_end:
            raise TranscriptionError(
                f"VAD region {position} overlaps or precedes its predecessor"
            )
        if str(raw.get("source", "")).strip() != "silero_vad":
            raise TranscriptionError(
                "ASR hallucination review requires independent Silero VAD"
            )
        trusted_vad.append((start_ms, end_ms))
        previous_end = end_ms
    if not trusted_vad:
        raise TranscriptionError(
            "ASR hallucination review requires a non-empty independent VAD inventory"
        )

    if isinstance(review_request_uids, (str, bytes, bytearray)) or not isinstance(
        review_request_uids, Sequence
    ):
        raise TranscriptionError("review_request_uids must be a sequence")
    requested: list[str] = []
    for position, uid in enumerate(review_request_uids, start=1):
        if not isinstance(uid, str) or not uid.strip():
            raise TranscriptionError(
                f"review_request_uids[{position}] must be a non-empty string"
            )
        normalized_uid = uid.strip()
        if normalized_uid in requested:
            raise TranscriptionError(
                f"Duplicate hallucination review request UID: {normalized_uid}"
            )
        requested.append(normalized_uid)
    reviewable_by_uid = {
        str(utterance["utterance_uid"]): utterance
        for utterance in trusted_utterances
        if (
            str(utterance["asr_text"]).strip()
            or (
                "orphan_youtube_caption" in utterance["risk_flags"]
                and str(utterance["youtube_text"]).strip()
            )
        )
        and "unresolved_vad_speech" not in utterance["risk_flags"]
    }
    unknown_requests = sorted(set(requested).difference(reviewable_by_uid))
    if unknown_requests:
        raise TranscriptionError(
            "Hallucination review requests do not match ordinary ASR utterances: "
            f"{unknown_requests}"
        )

    source_audio = Path(audio_path)
    if not source_audio.is_file() or source_audio.stat().st_size <= 0:
        raise TranscriptionError(
            "ASR hallucination-review source audio is missing or empty"
        )
    candidate_reasons: dict[str, list[str]] = {}
    candidate_reason_counts: dict[str, int] = {}
    automatic_candidate_uids: set[str] = set()
    for utterance_uid, utterance in reviewable_by_uid.items():
        start_ms = int(utterance["coarse_start_ms"])
        end_ms = int(utterance["coarse_end_ms"])
        overlap_ms = sum(
            max(0, min(end_ms, vad_end) - max(start_ms, vad_start))
            for vad_start, vad_end in trusted_vad
        )
        overlap_ratio = overlap_ms / (end_ms - start_ms)
        structural_reasons: list[str] = []
        confidence_reasons: list[str] = []
        is_orphan_caption = "orphan_youtube_caption" in utterance["risk_flags"]
        if "provisional_word_gap_requires_audio_review" in utterance["risk_flags"]:
            structural_reasons.append("provisional_word_gap_requires_audio_review")
        if "incomplete_provisional_word_timing" in utterance["risk_flags"]:
            structural_reasons.append("incomplete_provisional_word_timing")
        if is_orphan_caption:
            structural_reasons.append(
                "orphan_youtube_caption_without_asr_or_vad_overlap"
            )
        elif overlap_ms == 0:
            structural_reasons.append("zero_unpadded_independent_vad_overlap")
        elif overlap_ratio <= settings.hallucination_max_vad_overlap_ratio:
            structural_reasons.append("low_unpadded_independent_vad_overlap")
        audit = utterance["asr_audit"]
        avg_logprob = audit["avg_logprob"]
        no_speech_prob = audit["no_speech_prob"]
        compression_ratio = audit["compression_ratio"]
        temperature = audit["temperature"]
        if (
            avg_logprob is not None
            and float(avg_logprob) <= settings.hallucination_min_avg_logprob
        ):
            confidence_reasons.append("low_asr_average_log_probability")
        if (
            no_speech_prob is not None
            and float(no_speech_prob)
            >= settings.hallucination_max_no_speech_prob
        ):
            confidence_reasons.append("high_asr_no_speech_probability")
        if (
            compression_ratio is not None
            and float(compression_ratio)
            >= settings.hallucination_max_compression_ratio
        ):
            confidence_reasons.append("high_asr_compression_ratio")
        if (
            temperature is not None
            and float(temperature) > settings.hallucination_temperature_threshold
        ):
            confidence_reasons.append("nonzero_asr_sampling_temperature")
        asr_text = str(utterance["asr_text"])
        normalized_text = asr_text.casefold()
        has_text_marker = any(
            marker.casefold() in normalized_text
            for marker in settings.hallucination_text_markers
        )
        known_subtitle_hallucination = _is_known_subtitle_hallucination(asr_text)
        if has_text_marker:
            confidence_reasons.append("music_or_subtitle_text_marker")
        if known_subtitle_hallucination:
            confidence_reasons.append("known_subtitle_hallucination_signature")
        explicitly_requested = utterance_uid in requested
        # One weak confidence signal is common in real dialogue and must not
        # create a mandatory WAV by itself. Automatic review requires either
        # independent timing evidence, a high-compression repetition signal,
        # or at least two independent weak signals. This keeps the detector
        # sensitive to speech-in-silence hallucinations without turning every
        # low-confidence but audible line into a false candidate.
        automatic_candidate = (
            known_subtitle_hallucination
            or bool(structural_reasons)
            or "high_asr_compression_ratio" in confidence_reasons
            or len(confidence_reasons) >= 2
        )
        reasons = structural_reasons + confidence_reasons
        if explicitly_requested:
            reasons.append("explicit_uid_review_request")
        if automatic_candidate or explicitly_requested:
            candidate_reasons[utterance_uid] = reasons
            if automatic_candidate:
                automatic_candidate_uids.add(utterance_uid)
            for reason in reasons:
                candidate_reason_counts[reason] = (
                    candidate_reason_counts.get(reason, 0) + 1
                )
    candidates = [
        utterance
        for utterance in trusted_utterances
        if str(utterance["utterance_uid"]) in candidate_reasons
    ]
    explicit_only_uids = set(candidate_reasons).difference(automatic_candidate_uids)
    _require_asr_hallucination_candidate_budget(
        automatic_count=len(automatic_candidate_uids),
        explicit_only_count=len(explicit_only_uids),
        total_count=len(candidates),
        reviewable_count=len(reviewable_by_uid),
        candidate_reason_counts=candidate_reason_counts,
    )

    output_root = Path(audio_output_root)
    clip_root = output_root / "asr_hallucination_audio"
    if candidates:
        clip_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for candidate_index, utterance in enumerate(candidates, start=1):
        utterance_uid = str(utterance["utterance_uid"])
        start_ms = int(utterance["coarse_start_ms"])
        end_ms = int(utterance["coarse_end_ms"])
        identity = json.dumps(
            {
                "episode": episode,
                "utterance_uid": utterance_uid,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "asr_text": utterance["asr_text"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        candidate_uid = (
            f"MA{episode:02d}-ASR-HALL-{hashlib.sha256(identity).hexdigest()[:16]}"
        )
        audio_member = f"asr_hallucination_audio/{candidate_uid}.wav"
        clip_path = output_root / audio_member
        clip_start_ms, clip_end_ms = _extract_review_clip(
            source_audio,
            clip_path,
            target_start_ms=start_ms,
            target_end_ms=end_ms,
            padding_ms=settings.review_clip_padding_ms,
            minimum_duration_ms=settings.review_clip_min_duration_ms,
        )
        risk_flags = list(utterance["risk_flags"])
        candidate_flags = ["manual_audio_review_required"]
        if "orphan_youtube_caption" not in risk_flags:
            candidate_flags.insert(0, "suspected_asr_hallucination")
        for flag in candidate_flags:
            if flag not in risk_flags:
                risk_flags.append(flag)
        records.append(
            {
                "candidate_uid": candidate_uid,
                "candidate_index": candidate_index,
                "utterance_uid": utterance_uid,
                "utterance_index": int(utterance["utterance_index"]),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "clip_start_ms": clip_start_ms,
                "clip_end_ms": clip_end_ms,
                "reason": ", ".join(candidate_reasons[utterance_uid]),
                "asr_text": str(utterance["asr_text"]),
                "youtube_text": str(utterance["youtube_text"]),
                "context_before": str(utterance["context_before"]),
                "context_after": str(utterance["context_after"]),
                "risk_flags": risk_flags,
                "asr_audit": dict(utterance["asr_audit"]),
                "audio_member": audio_member,
                "audio_sha256": sha256_file(clip_path),
                "audio_size_bytes": clip_path.stat().st_size,
            }
        )

    updated_utterances: list[dict[str, Any]] = []
    records_by_utterance = {
        str(record["utterance_uid"]): record for record in records
    }
    for utterance in trusted_utterances:
        updated = dict(utterance)
        record = records_by_utterance.get(str(utterance["utterance_uid"]))
        if record is not None:
            updated["risk_flags"] = list(record["risk_flags"])
        updated_utterances.append(updated)
    updated_utterances = validate_input_utterances(updated_utterances)
    trusted_records = validate_asr_hallucination_records(records)
    return updated_utterances, trusted_records


def include_speech_holes_for_correction(
    utterances: Sequence[Mapping[str, Any]],
    holes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Put unresolved VAD spans in the editable correction stream.

    These records contain no invented transcript.  The Turkish pass must either
    supply what is actually spoken or explicitly mark the interval
    ``non_dialogue`` with a reason.  The immutable read-only hole list remains
    in the pack as the audit trail.
    """

    trusted_utterances = (
        validate_input_utterances(utterances) if utterances else []
    )
    trusted_holes = validate_speech_holes(holes)
    combined = [dict(item) for item in trusted_utterances]
    for hole in trusted_holes:
        combined.append(
            {
                "utterance_uid": str(hole["hole_uid"]),
                "utterance_index": 0,
                "coarse_start_ms": int(hole["start_ms"]),
                "coarse_end_ms": int(hole["end_ms"]),
                "asr_text": "",
                "youtube_text": "",
                "context_before": str(hole.get("context_before", "")),
                "context_after": str(hole.get("context_after", "")),
                "risk_flags": list(hole.get("risk_flags", [])),
                "asr_audit": {
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "compression_ratio": None,
                    "temperature": None,
                },
            }
        )
    combined.sort(
        key=lambda item: (
            int(item["coarse_start_ms"]),
            int(item["coarse_end_ms"]),
            str(item["utterance_uid"]),
        )
    )
    for index, item in enumerate(combined, start=1):
        item["utterance_index"] = index
    # Blank evidence is accepted only for the explicit unresolved-VAD records
    # above. No dialogue text is manufactured here.
    return validate_input_utterances(combined) if combined else []


def _character_bigram_dice(first: str, second: str) -> float:
    """Return multiset Dice similarity over normalized character bigrams."""

    normalized_first = _normalized_lexical_text(first)
    normalized_second = _normalized_lexical_text(second)
    if not normalized_first or not normalized_second:
        return 0.0

    def bigrams(value: str) -> Counter[str]:
        if len(value) < 2:
            return Counter([value])
        return Counter(
            value[index : index + 2] for index in range(len(value) - 1)
        )

    first_bigrams = bigrams(normalized_first)
    second_bigrams = bigrams(normalized_second)
    intersection = sum(
        min(count, second_bigrams.get(item, 0))
        for item, count in first_bigrams.items()
    )
    denominator = sum(first_bigrams.values()) + sum(second_bigrams.values())
    return (2.0 * intersection / denominator) if denominator else 0.0


def _caption_has_matching_asr_evidence(
    caption_text: str,
    overlapping_asr: Sequence[Mapping[str, Any]],
    *,
    caption_start_ms: int,
    caption_end_ms: int,
    min_overlap_ms: int,
    min_shared_tokens: int,
    min_character_bigram_dice: float,
) -> bool:
    """Recognize a caption already represented by overlapping ASR text.

    This is content de-duplication, not a timing waiver. At least 120 ms of
    exact ASR/caption overlap is required before lexical similarity can explain
    a rolling-caption boundary. A one-millisecond touch therefore remains a
    mandatory audio-review candidate.
    """

    clipped: list[tuple[int, int]] = []
    pieces: list[str] = []
    for utterance in overlapping_asr:
        start_ms = max(caption_start_ms, int(utterance["coarse_start_ms"]))
        end_ms = min(caption_end_ms, int(utterance["coarse_end_ms"]))
        if end_ms <= start_ms:
            continue
        clipped.append((start_ms, end_ms))
        text = str(utterance.get("asr_text", "")).strip()
        if text:
            pieces.append(text)
    clipped.sort()
    merged: list[list[int]] = []
    for start_ms, end_ms in clipped:
        if merged and start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ms)
        else:
            merged.append([start_ms, end_ms])
    overlap_ms = sum(end_ms - start_ms for start_ms, end_ms in merged)
    if overlap_ms < min_overlap_ms or not pieces:
        return False

    caption_tokens = set(_normalized_lexical_text(caption_text).split())
    asr_text = " ".join(pieces)
    asr_tokens = set(_normalized_lexical_text(asr_text).split())
    shared_token_count = len(caption_tokens.intersection(asr_tokens))
    return (
        shared_token_count >= min_shared_tokens
        or _character_bigram_dice(caption_text, asr_text)
        >= float(min_character_bigram_dice)
    )


def include_orphan_youtube_captions_for_correction(
    utterances: Sequence[Mapping[str, Any]],
    youtube_captions: Sequence[Mapping[str, Any]],
    vad_regions: Sequence[Mapping[str, Any]],
    *,
    episode: int,
    min_evidence_coverage_ratio: float = (
        V2_BETA_ORPHAN_CAPTION_EVIDENCE_COVERAGE_FLOOR
    ),
    max_unexplained_run_ms: int = (
        V2_BETA_ORPHAN_CAPTION_MAX_UNEXPLAINED_RUN_MS
    ),
    min_absolute_evidence_ms: int = (
        V2_BETA_ORPHAN_CAPTION_MIN_ABSOLUTE_EVIDENCE_MS
    ),
    trailing_display_tolerance_ms: int = (
        V2_BETA_ORPHAN_CAPTION_TRAILING_DISPLAY_TOLERANCE_MS
    ),
    min_lexical_asr_overlap_ms: int = (
        V2_BETA_ORPHAN_CAPTION_MIN_LEXICAL_ASR_OVERLAP_MS
    ),
    min_shared_tokens: int = V2_BETA_ORPHAN_CAPTION_MIN_SHARED_TOKENS,
    min_character_bigram_dice: float = (
        V2_BETA_ORPHAN_CAPTION_MIN_CHARACTER_BIGRAM_DICE
    ),
) -> list[dict[str, Any]]:
    """Add captions not substantially explained by ASR/VAD for audio review.

    A one-millisecond boundary touch is not evidence that a whole caption was
    heard or transcribed. The caption is considered explained only when the
    union of raw ASR utterance intervals and independent audit-VAD intervals
    covers at least half of its exact target interval with no unexplained run
    above the strict bound. YouTube rolling captions may legitimately retain a
    completed phrase for 500 ms; after rolling-cue normalization, at least
    120 ms of real evidence plus bounded leading/internal gaps also explains
    that display tail. A caption is also already represented when at least 120
    ms of overlapping ASR carries matching words or strong character-level
    text evidence. Everything else becomes a mandatory WAV-backed review
    candidate.
    """

    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise TranscriptionError("episode must be a positive integer")
    if (
        isinstance(min_evidence_coverage_ratio, bool)
        or not isinstance(min_evidence_coverage_ratio, (int, float))
        or not math.isfinite(float(min_evidence_coverage_ratio))
        or not 0.0 <= float(min_evidence_coverage_ratio) <= 1.0
    ):
        raise TranscriptionError(
            "min_evidence_coverage_ratio must be a finite ratio in [0, 1]"
        )
    if (
        isinstance(max_unexplained_run_ms, bool)
        or not isinstance(max_unexplained_run_ms, int)
        or max_unexplained_run_ms < 0
    ):
        raise TranscriptionError(
            "max_unexplained_run_ms must be a non-negative integer"
        )
    if (
        isinstance(min_absolute_evidence_ms, bool)
        or not isinstance(min_absolute_evidence_ms, int)
        or min_absolute_evidence_ms < 1
        or isinstance(trailing_display_tolerance_ms, bool)
        or not isinstance(trailing_display_tolerance_ms, int)
        or trailing_display_tolerance_ms < 0
    ):
        raise TranscriptionError("orphan caption duration settings are invalid")
    if (
        isinstance(min_lexical_asr_overlap_ms, bool)
        or not isinstance(min_lexical_asr_overlap_ms, int)
        or min_lexical_asr_overlap_ms < 1
        or isinstance(min_shared_tokens, bool)
        or not isinstance(min_shared_tokens, int)
        or min_shared_tokens < 1
        or isinstance(min_character_bigram_dice, bool)
        or not isinstance(min_character_bigram_dice, (int, float))
        or not math.isfinite(float(min_character_bigram_dice))
        or not 0.0 <= float(min_character_bigram_dice) <= 1.0
    ):
        raise TranscriptionError("orphan caption lexical settings are invalid")
    trusted_utterances = validate_input_utterances(utterances)
    if isinstance(youtube_captions, (str, bytes, bytearray)) or not isinstance(
        youtube_captions, Sequence
    ):
        raise TranscriptionError("youtube_captions must be a sequence")
    if isinstance(vad_regions, (str, bytes, bytearray)) or not isinstance(
        vad_regions, Sequence
    ):
        raise TranscriptionError("vad_regions must be a sequence")
    vad_intervals: list[tuple[int, int]] = []
    for position, region in enumerate(vad_regions, start=1):
        if not isinstance(region, Mapping):
            raise TranscriptionError(f"VAD region {position} must be an object")
        start_ms = region.get("start_ms")
        end_ms = region.get("end_ms")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or start_ms < 0
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
            or str(region.get("source", "")) != "silero_vad"
        ):
            raise TranscriptionError(
                f"Invalid independent VAD region at position {position}"
            )
        vad_intervals.append((start_ms, end_ms))
    asr_utterances = [
        item for item in trusted_utterances if str(item["asr_text"]).strip()
    ]
    asr_intervals = [
        (int(item["coarse_start_ms"]), int(item["coarse_end_ms"]))
        for item in asr_utterances
    ]
    evidence_intervals = sorted(asr_intervals + vad_intervals)
    merged_evidence: list[tuple[int, int]] = []
    for start_ms, end_ms in evidence_intervals:
        if merged_evidence and start_ms <= merged_evidence[-1][1]:
            merged_evidence[-1] = (
                merged_evidence[-1][0],
                max(merged_evidence[-1][1], end_ms),
            )
        else:
            merged_evidence.append((start_ms, end_ms))
    combined = [dict(item) for item in trusted_utterances]
    previous_caption_end = -1
    for position, caption in enumerate(youtube_captions, start=1):
        if not isinstance(caption, Mapping):
            raise TranscriptionError(f"YouTube caption {position} must be an object")
        start_ms = caption.get("start_ms")
        end_ms = caption.get("end_ms")
        text = caption.get("text")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or start_ms < 0
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
            or not isinstance(text, str)
        ):
            raise TranscriptionError(f"Invalid YouTube caption at position {position}")
        if start_ms < previous_caption_end:
            raise TranscriptionError(
                "YouTube captions must be ordered and non-overlapping"
            )
        previous_caption_end = end_ms
        normalized_text = text.strip()
        if not normalized_text:
            continue
        evidence_coverage_ms = sum(
            max(0, min(end_ms, evidence_end) - max(start_ms, evidence_start))
            for evidence_start, evidence_end in merged_evidence
            if start_ms < evidence_end and evidence_start < end_ms
        )
        cursor_ms = start_ms
        longest_leading_or_internal_run_ms = 0
        for evidence_start, evidence_end in merged_evidence:
            clipped_start_ms = max(start_ms, evidence_start)
            clipped_end_ms = min(end_ms, evidence_end)
            if clipped_end_ms <= clipped_start_ms:
                continue
            if clipped_start_ms > cursor_ms:
                longest_leading_or_internal_run_ms = max(
                    longest_leading_or_internal_run_ms,
                    clipped_start_ms - cursor_ms,
                )
            cursor_ms = max(cursor_ms, clipped_end_ms)
        trailing_unexplained_ms = end_ms - cursor_ms
        longest_unexplained_run_ms = max(
            longest_leading_or_internal_run_ms, trailing_unexplained_ms
        )
        coverage_ratio = evidence_coverage_ms / (end_ms - start_ms)
        strict_explanation = (
            coverage_ratio >= float(min_evidence_coverage_ratio)
            and longest_unexplained_run_ms <= max_unexplained_run_ms
        )
        rolling_display_explanation = (
            evidence_coverage_ms >= min_absolute_evidence_ms
            and longest_leading_or_internal_run_ms <= max_unexplained_run_ms
            and trailing_unexplained_ms <= trailing_display_tolerance_ms
        )
        overlapping_asr = [
            item
            for item in asr_utterances
            if start_ms < int(item["coarse_end_ms"])
            and int(item["coarse_start_ms"]) < end_ms
        ]
        lexical_asr_explanation = _caption_has_matching_asr_evidence(
            normalized_text,
            overlapping_asr,
            caption_start_ms=start_ms,
            caption_end_ms=end_ms,
            min_overlap_ms=min_lexical_asr_overlap_ms,
            min_shared_tokens=min_shared_tokens,
            min_character_bigram_dice=float(min_character_bigram_dice),
        )
        if (
            strict_explanation
            or rolling_display_explanation
            or lexical_asr_explanation
        ):
            continue
        identity = json.dumps(
            {
                "episode": episode,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "youtube_text": normalized_text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        combined.append(
            {
                "utterance_uid": (
                    f"MA{episode:02d}-YT-ORPHAN-"
                    f"{hashlib.sha256(identity).hexdigest()[:16]}"
                ),
                "utterance_index": 0,
                "coarse_start_ms": start_ms,
                "coarse_end_ms": end_ms,
                "asr_text": "",
                "youtube_text": normalized_text,
                "context_before": "",
                "context_after": "",
                "risk_flags": [
                    "orphan_youtube_caption",
                    "manual_audio_review_required",
                ],
                "asr_audit": {
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "compression_ratio": None,
                    "temperature": None,
                },
            }
        )
    combined.sort(
        key=lambda item: (
            int(item["coarse_start_ms"]),
            int(item["coarse_end_ms"]),
            str(item["utterance_uid"]),
        )
    )
    for index, item in enumerate(combined, start=1):
        item["utterance_index"] = index
    return validate_input_utterances(combined)


def _validate_persisted_review_audio(
    record: Mapping[str, Any],
    *,
    prepare_dir: str | Path,
    label: str,
) -> None:
    member = str(record["audio_member"])
    member_parts = PurePosixPath(member).parts
    root = Path(prepare_dir).resolve()
    current = Path(prepare_dir)
    for part in member_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} audio uses a symlink: {member}")
    candidate = Path(prepare_dir).joinpath(*member_parts)
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if (
        not resolved.is_file()
        or resolved.stat().st_size != record["audio_size_bytes"]
    ):
        raise ValueError(f"{label} audio size mismatch: {member}")
    if sha256_file(resolved) != record["audio_sha256"]:
        raise ValueError(f"{label} audio SHA-256 mismatch: {member}")
    expected_duration = int(record["clip_end_ms"]) - int(
        record["clip_start_ms"]
    )
    actual_duration = _wav_duration_ms(resolved)
    if abs(actual_duration - expected_duration) > 100:
        raise ValueError(
            f"{label} audio duration mismatch: expected {expected_duration} ms, "
            f"got {actual_duration} ms"
        )


def _validate_persisted_raw_asr_v2(
    data: Any,
    *,
    input_sha256: str,
    audio_sha256: str,
    episode: int,
    prepare_dir: str | Path,
) -> list[str]:
    """Validate the identity and minimum structure of a resumable artifact."""

    if not isinstance(data, Mapping):
        return ["raw ASR output is not an object"]
    errors: list[str] = []
    if data.get("format_version") != RAW_ASR_V2_FORMAT_VERSION:
        errors.append("unsupported format_version")
    if data.get("status") != "completed":
        errors.append("raw ASR output is not completed")
    if data.get("input_sha256") != input_sha256:
        errors.append("input_sha256 mismatch")
    if data.get("audio_sha256") != audio_sha256:
        errors.append("audio_sha256 mismatch")
    if data.get("episode") != episode:
        errors.append("episode mismatch")
    if data.get("language") != "tr":
        errors.append("raw ASR language is not Turkish")
    if data.get("synthetic_timing_count") != 0:
        errors.append("raw ASR contains synthetic timing")
    if not isinstance(data.get("independent_vad"), bool):
        errors.append("independent_vad is not boolean")
    for key in (
        "segments",
        "words",
        "vad_regions",
        "rescue_failures",
        "youtube_captions",
        "correction_utterances",
        "speech_hole_records",
        "asr_hallucination_records",
    ):
        if not isinstance(data.get(key), list):
            errors.append(f"{key} is not a list")
    for key in ("model", "initial_speech_coverage", "speech_coverage"):
        if not isinstance(data.get(key), Mapping):
            errors.append(f"{key} is not an object")
    if all(isinstance(data.get(key), list) for key in ("segments", "words", "vad_regions")):
        try:
            coverage = data.get("speech_coverage", {})
            recomputed = _analyze_raw_speech_coverage(
                data["vad_regions"], data["segments"], data["words"],
                config=SpeechCoverageConfig(**dict(coverage["config"])),
            )
            if recomputed != coverage:
                errors.append("speech coverage does not match complete lexical word inventory")
            expected_holes = sorted(
                (issue["start_ms"], issue["end_ms"])
                for issue in recomputed["coverage_issues"]
                if issue["classification"] == "unresolved_speech"
            )
            actual_holes = sorted(
                (hole["start_ms"], hole["end_ms"])
                for hole in data.get("speech_hole_records", [])
            )
            if expected_holes != actual_holes:
                errors.append("speech-hole inventory does not match complete lexical word coverage")
            validate_raw_rescue_plan(data)
        except (KeyError, TypeError, ValueError, TranscriptionError) as exc:
            errors.append(f"complete lexical word coverage validation failed: {exc}")
    vad_value = data.get("vad_regions")
    if isinstance(vad_value, list):
        derived_independent_vad = (
            not bool(str(data.get("vad_fallback_reason") or "").strip())
            and bool(vad_value)
            and all(
                isinstance(region, Mapping)
                and str(region.get("source", "")).strip() == "silero_vad"
                for region in vad_value
            )
        )
        if data.get("independent_vad") is not derived_independent_vad:
            errors.append("independent_vad does not match VAD provenance")
    holes_value = data.get("speech_hole_records")
    if isinstance(holes_value, list):
        try:
            trusted_holes = validate_speech_holes(holes_value)
            for hole in trusted_holes:
                _validate_persisted_review_audio(
                    hole,
                    prepare_dir=prepare_dir,
                    label="speech-hole",
                )
        except (OSError, RuntimeError, TRCorrectionError, ValueError) as exc:
            errors.append(f"speech-hole audio validation failed: {exc}")
    candidates_value = data.get("asr_hallucination_records")
    utterances_value = data.get("correction_utterances")
    if isinstance(candidates_value, list) and isinstance(utterances_value, list):
        try:
            trusted_candidates = validate_asr_hallucination_records(
                candidates_value
            )
            trusted_utterances = validate_input_utterances(utterances_value)
            persisted_by_uid = {item["utterance_uid"]: item for item in trusted_utterances}
            settings = data["model"]["settings"]
            excluded = _excluded_incomplete_segments(data["segments"], episode=episode)
            if data.get("excluded_incomplete_segments", []) != excluded:
                raise ValueError("excluded incomplete segment audit does not match raw ownership")
            if data.get("required_acoustic_review_regions") != required_acoustic_review_regions(data["segments"], data["vad_regions"]):
                raise ValueError("required acoustic review regions do not match raw ownership")
            if any(item["utterance_uid"] in persisted_by_uid for item in excluded):
                raise ValueError("excluded incomplete ASR hypothesis remains in correction input")
            expected_utterances = build_correction_utterances(
                [segment for segment in data["segments"] if _segment_has_complete_word_inventory(segment)],
                episode=episode,
                max_internal_word_gap_ms=settings["correction_max_internal_word_gap_ms"],
                review_word_gap_ms=settings["correction_review_word_gap_ms"],
            )
            holes_by_uid = {hole["hole_uid"]: hole for hole in data["speech_hole_records"]}
            for issue in data["speech_coverage"]["coverage_issues"]:
                if issue["classification"] != "unresolved_speech":
                    continue
                canonical = _speech_hole_metadata(issue, episode=episode, utterances=expected_utterances)
                actual = holes_by_uid.get(canonical["hole_uid"])
                if actual is None or any(actual[key] != value for key, value in canonical.items()):
                    raise ValueError("speech-hole canonical identity/reason/context mismatch")
            for expected in expected_utterances:
                persisted = persisted_by_uid.get(expected["utterance_uid"])
                if persisted is None or any(
                    persisted[key] != expected[key]
                    for key in ("coarse_start_ms", "coarse_end_ms", "asr_text")
                ):
                    raise ValueError("raw segment and correction utterance ownership mismatch")
                required_flags = set(expected["risk_flags"])
                if data.get("independent_vad") is True and required_flags.intersection({
                    "incomplete_provisional_word_timing",
                    "provisional_word_gap_requires_audio_review",
                }):
                    required_flags.update({"suspected_asr_hallucination", "manual_audio_review_required"})
                if not required_flags.issubset(persisted["risk_flags"]):
                    raise ValueError("raw segment correction is missing mandatory acoustic risk flags")
            flagged = {
                str(item["utterance_uid"]): item
                for item in trusted_utterances
                if (
                    "suspected_asr_hallucination" in item["risk_flags"]
                    or "orphan_youtube_caption" in item["risk_flags"]
                )
            }
            candidates_by_utterance = {
                str(item["utterance_uid"]): item
                for item in trusted_candidates
            }
            if data.get("independent_vad") is True:
                base_utterances: list[dict[str, Any]] = []
                for utterance in trusted_utterances:
                    if "orphan_youtube_caption" in utterance["risk_flags"]:
                        continue
                    copied = dict(utterance)
                    copied["utterance_index"] = len(base_utterances) + 1
                    base_utterances.append(copied)
                model = data.get("model")
                settings = model.get("settings") if isinstance(model, Mapping) else None
                if not isinstance(settings, Mapping):
                    raise ValueError("raw ASR model settings are missing")
                recomputed_utterances = include_orphan_youtube_captions_for_correction(
                    base_utterances,
                    data.get("youtube_captions", []),
                    data.get("vad_regions", []),
                    episode=episode,
                    min_evidence_coverage_ratio=settings.get(
                        "orphan_caption_min_evidence_coverage_ratio"
                    ),
                    max_unexplained_run_ms=settings.get(
                        "orphan_caption_max_unexplained_run_ms"
                    ),
                    min_absolute_evidence_ms=settings.get(
                        "orphan_caption_min_absolute_evidence_ms"
                    ),
                    trailing_display_tolerance_ms=settings.get(
                        "orphan_caption_trailing_display_tolerance_ms"
                    ),
                    min_lexical_asr_overlap_ms=settings.get(
                        "orphan_caption_min_lexical_asr_overlap_ms"
                    ),
                    min_shared_tokens=settings.get(
                        "orphan_caption_min_shared_tokens"
                    ),
                    min_character_bigram_dice=settings.get(
                        "orphan_caption_min_character_bigram_dice"
                    ),
                )
                recomputed_orphans = {
                    str(item["utterance_uid"]): item
                    for item in recomputed_utterances
                    if "orphan_youtube_caption" in item["risk_flags"]
                }
                persisted_orphans = {
                    str(item["utterance_uid"]): item
                    for item in trusted_utterances
                    if "orphan_youtube_caption" in item["risk_flags"]
                }
                if recomputed_orphans != persisted_orphans:
                    raise ValueError(
                        "persisted orphan YouTube-caption candidates do not "
                        "match canonical ASR/VAD interval evidence"
                    )
            if set(flagged) != set(candidates_by_utterance):
                raise ValueError(
                    "flagged utterances and ASR hallucination records do not match"
                )
            comparisons = {
                "utterance_index": "utterance_index",
                "coarse_start_ms": "start_ms",
                "coarse_end_ms": "end_ms",
                "asr_text": "asr_text",
                "youtube_text": "youtube_text",
                "context_before": "context_before",
                "context_after": "context_after",
                "risk_flags": "risk_flags",
                "asr_audit": "asr_audit",
            }
            for uid, candidate_record in candidates_by_utterance.items():
                utterance = flagged[uid]
                for utterance_field, candidate_field in comparisons.items():
                    if utterance[utterance_field] != candidate_record[candidate_field]:
                        raise ValueError(
                            f"ASR hallucination {uid} link mismatch in "
                            f"{utterance_field}"
                        )
                _validate_persisted_review_audio(
                    candidate_record,
                    prepare_dir=prepare_dir,
                    label="ASR hallucination",
                )
            if data.get("independent_vad") is not True and trusted_candidates:
                raise ValueError(
                    "non-independent debug artifact cannot carry hallucination "
                    "review candidates"
                )
        except (KeyError, TypeError, OSError, RuntimeError, TRCorrectionError, ValueError) as exc:
            errors.append(f"ASR hallucination validation failed: {exc}")
    return errors


def validate_persisted_raw_asr_v2(
    data: Mapping[str, Any],
    *,
    expected_input_sha256: str,
    audio_sha256: str,
    episode: int,
    prepare_dir: str | Path,
    require_independent_vad: bool = True,
) -> dict[str, Any]:
    """Validate and defensively copy one persisted raw-ASR V2 artifact.

    Production callers should keep ``require_independent_vad=True``.  The
    optional false value exists only so diagnostic tooling can inspect an
    explicitly unsafe debug artifact; it does not make that artifact
    publishable.
    """

    if (
        not isinstance(expected_input_sha256, str)
        or len(expected_input_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_input_sha256)
    ):
        raise TranscriptionError(
            "expected_input_sha256 must be a lowercase SHA-256 digest"
        )
    if (
        not isinstance(audio_sha256, str)
        or len(audio_sha256) != 64
        or any(character not in "0123456789abcdef" for character in audio_sha256)
    ):
        raise TranscriptionError("audio_sha256 must be a lowercase SHA-256 digest")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise TranscriptionError("episode must be a positive integer")
    if not isinstance(require_independent_vad, bool):
        raise TranscriptionError("require_independent_vad must be boolean")
    try:
        payload = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        trusted = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(f"Raw ASR V2 artifact is not strict JSON: {exc}") from exc
    if require_independent_vad:
        auth_key = _raw_asr_auth_key(RawASRV2Config())
        if auth_key is None:
            raise TranscriptionError(
                "Production raw ASR artifact authentication key is unavailable"
            )
        _verify_raw_asr_artifact_auth(trusted, auth_key)
    errors = _validate_persisted_raw_asr_v2(
        trusted,
        input_sha256=expected_input_sha256,
        audio_sha256=audio_sha256,
        episode=episode,
        prepare_dir=prepare_dir,
    )
    if require_independent_vad and trusted.get("independent_vad") is not True:
        errors.append("independent VAD is required for production")
    if require_independent_vad:
        try:
            validate_publishable_raw_asr_v2_policy(trusted)
        except TranscriptionError as exc:
            errors.append(str(exc))
    if errors:
        raise TranscriptionError(
            "Persisted raw ASR V2 validation failed: " + "; ".join(errors[:20])
        )
    return trusted


def load_valid_raw_asr_v2(
    prepare_dir: str | Path,
    *,
    audio_path: str | Path,
    episode: int,
    expected_input_sha256: str | None = None,
    require_independent_vad: bool = True,
) -> dict[str, Any] | None:
    """Load the canonical output only when marker, audio, and identity agree.

    Invalid, stale, redirected, or interrupted artifacts return ``None``.  Bad
    caller arguments still raise :class:`TranscriptionError` so configuration
    mistakes are not mistaken for an ordinary cache miss.
    """

    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise TranscriptionError("episode must be a positive integer")
    if not isinstance(require_independent_vad, bool):
        raise TranscriptionError("require_independent_vad must be boolean")
    source = Path(audio_path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise TranscriptionError(f"Audio input is missing or empty: {source}")
    destination = Path(prepare_dir)
    output_path = destination / "raw_asr_v2.json"
    marker_path = destination / "raw_asr_v2.done.json"
    if not marker_path.is_file():
        return None
    try:
        marker_envelope = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_input_sha = marker_envelope.get("input_sha256")
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return None
    if (
        not isinstance(marker_input_sha, str)
        or len(marker_input_sha) != 64
        or any(character not in "0123456789abcdef" for character in marker_input_sha)
    ):
        return None
    if expected_input_sha256 is not None:
        if (
            not isinstance(expected_input_sha256, str)
            or len(expected_input_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_input_sha256
            )
        ):
            raise TranscriptionError(
                "expected_input_sha256 must be a lowercase SHA-256 digest"
            )
        if marker_input_sha != expected_input_sha256:
            return None
    marker = load_valid_stage_marker(
        marker_path,
        stage="raw_asr_v2",
        input_sha256=marker_input_sha,
        required_output_keys=("raw_asr_v2",),
        allowed_root=destination,
    )
    if marker is None:
        return None
    try:
        output_record = marker["outputs"]["raw_asr_v2"]
        cached_path = Path(str(output_record["path"])).resolve()
        if cached_path != output_path.resolve():
            return None
        data = json.loads(cached_path.read_text(encoding="utf-8"))
        return validate_persisted_raw_asr_v2(
            data,
            expected_input_sha256=marker_input_sha,
            audio_sha256=sha256_file(source),
            episode=episode,
            prepare_dir=destination,
            require_independent_vad=require_independent_vad,
        )
    except (
        KeyError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        TranscriptionError,
    ):
        return None


def _unit_result_sha256(result: Any) -> str:
    return sha256_json(
        json.loads(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    )


def _raw_asr_auth_key(settings: RawASRV2Config) -> bytes | None:
    value = os.getenv(RAW_ASR_AUTH_KEY_ENV)
    if not value:
        if settings.allow_cpu_fallback or not settings.require_independent_vad:
            return None
        raise TranscriptionError(
            f"{RAW_ASR_AUTH_KEY_ENV} is required for production raw ASR"
        )
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TranscriptionError(
            f"{RAW_ASR_AUTH_KEY_ENV} must be a strong 256-bit lowercase hex key"
        )
    return bytes.fromhex(value)


def _raw_asr_auth_tag(key: bytes, purpose: str, body: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(body),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        key,
        RAW_ASR_AUTH_DOMAIN + purpose.encode("ascii") + b"\0" + encoded,
        hashlib.sha256,
    ).hexdigest()


def _sign_raw_asr_artifact(data: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    signed = dict(data)
    signed.pop("raw_asr_auth_tag", None)
    signed["raw_asr_auth_tag"] = _raw_asr_auth_tag(
        key, "completed-artifact", signed
    )
    return signed


def _verify_raw_asr_artifact_auth(data: Mapping[str, Any], key: bytes) -> None:
    tag = data.get("raw_asr_auth_tag")
    body = dict(data)
    body.pop("raw_asr_auth_tag", None)
    expected = _raw_asr_auth_tag(key, "completed-artifact", body)
    if not isinstance(tag, str) or not hmac.compare_digest(tag, expected):
        raise TranscriptionError(
            "Completed raw ASR artifact authentication failed for the current key"
        )


def _signed_unit_result(
    key: bytes,
    *,
    purpose: str,
    binding: Mapping[str, Any],
    uid: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = json.loads(
        json.dumps(dict(result), ensure_ascii=False, allow_nan=False)
    )
    body = {
        "binding_sha256": sha256_json(dict(binding)),
        "uid": uid,
        "result_sha256": _unit_result_sha256(evidence),
    }
    return {
        "evidence": evidence,
        "auth_tag": _raw_asr_auth_tag(key, purpose, body),
    }


def _verify_signed_unit_result(
    saved: Any,
    key: bytes,
    *,
    purpose: str,
    binding: Mapping[str, Any],
    uid: str,
) -> dict[str, Any]:
    if not isinstance(saved, Mapping) or not isinstance(saved.get("evidence"), Mapping):
        raise TranscriptionError(f"{purpose} producer receipt is unsigned")
    evidence = dict(saved["evidence"])
    expected = _signed_unit_result(
        key,
        purpose=purpose,
        binding=binding,
        uid=uid,
        result=evidence,
    )["auth_tag"]
    tag = saved.get("auth_tag")
    if not isinstance(tag, str) or not hmac.compare_digest(tag, expected):
        raise TranscriptionError(f"{purpose} producer receipt authentication failed")
    return evidence


def _primary_auth_record_path(destination: Path, identity: Mapping[str, Any]) -> Path:
    return destination / "raw_asr_auth" / "primary" / f"{sha256_json(identity)}.json"


def _primary_receipt_record(
    destination: Path,
    checkpoint: Path,
    identity: Mapping[str, Any],
    key: bytes,
    *,
    create: bool,
) -> dict[str, Any] | None:
    receipt_sha256 = sha256_file(checkpoint)
    auth_body = {
        "identity_sha256": sha256_json(identity),
        "receipt_sha256": receipt_sha256,
    }
    auth_tag = _raw_asr_auth_tag(key, "primary", auth_body)
    auth_path = _primary_auth_record_path(destination, identity)
    if create:
        atomic_write_json(auth_path, {**auth_body, "auth_tag": auth_tag})
    elif not auth_path.is_file():
        return None
    try:
        saved = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TranscriptionError("Primary ASR authentication record is invalid") from exc
    if saved != {**auth_body, "auth_tag": auth_tag}:
        raise TranscriptionError("Primary ASR receipt authentication failed")
    return {
        "relative_path": checkpoint.relative_to(destination).as_posix(),
        "sha256": receipt_sha256,
        "auth_tag": auth_tag,
    }


def _primary_asr_options(
    settings: RawASRV2Config,
    prompt: str | None,
) -> dict[str, Any]:
    return {
        "language": settings.language,
        "task": "transcribe",
        "beam_size": settings.beam_size,
        "best_of": settings.best_of,
        "temperature": 0.0,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "initial_prompt": prompt,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": settings.transcription_config().vad_parameters,
    }


def _raw_asr_producer_identity() -> dict[str, Any]:
    return primary_checkpoint.producer_identity(
        (
            transcribe_raw_audio_v2,
            _primary_asr_options,
            consume_coarse_segments,
            _finite_ms,
            _optional_finite_number,
            _extract_vad_regions,
        )
    )


def _canonical_primary_identity(
    *,
    episode: int,
    audio_sha256: str,
    settings: RawASRV2Config,
    prompt: str | None,
    model_identity: Mapping[str, Any],
    producer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if settings.allow_cpu_fallback:
        raise TranscriptionError(
            "Recovery requires the canonical GPU-only raw ASR policy"
        )
    return {
        "format": "mas-primary-asr-1",
        "episode": episode,
        "audio_sha256": audio_sha256,
        "model": dict(model_identity),
        "producer": dict(producer_identity),
        "options": _primary_asr_options(settings, prompt),
        "device": "cuda",
        "compute_type": settings.compute_type_gpu,
        "cpu_threads": max(1, os.cpu_count() or 1),
        "num_workers": 1,
    }


def _vad_journal_binding(
    *,
    audio_sha256: str,
    primary_receipt_sha256: str,
    settings: RawASRV2Config,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": RAW_ASR_V2_UNIT_JOURNAL_FORMAT,
        "kind": "independent_vad",
        "audio_sha256": audio_sha256,
        "primary_receipt_sha256": primary_receipt_sha256,
        "producer": dict(producer),
        "settings": {
            "threshold": settings.vad_threshold,
            "min_speech_duration_ms": settings.vad_min_speech_ms,
            "min_silence_duration_ms": settings.vad_min_silence_ms,
            "speech_pad_ms": settings.audit_vad_speech_pad_ms,
            "require_independent_vad": settings.require_independent_vad,
        },
    }


def _rescue_journal_binding(
    *,
    audio_sha256: str,
    primary_receipt_sha256: str,
    model_identity: Mapping[str, Any],
    producer: Mapping[str, Any],
    settings: RawASRV2Config,
    prompt: str | None,
    rescue_batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format": RAW_ASR_V2_UNIT_JOURNAL_FORMAT,
        "kind": "rescue_asr",
        "audio_sha256": audio_sha256,
        "primary_receipt_sha256": primary_receipt_sha256,
        "model": dict(model_identity),
        "producer": dict(producer),
        "settings": {
            "language": settings.language,
            "beam_size": settings.rescue_beam_size,
            "best_of": max(settings.best_of, settings.rescue_beam_size),
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "initial_prompt": prompt,
            "word_timestamps": True,
            "vad_filter": False,
        },
        "canonical_rescue_plan_sha256": sha256_json(list(rescue_batches)),
    }


def _rescue_batch_uid(index: int, span: Mapping[str, Any]) -> str:
    return f"rescue-{index:03d}-{sha256_json(dict(span))}"


def _load_primary_receipt_for_recovery(
    destination: Path,
    record: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    auth_key: bytes,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    relative_path = record.get("relative_path")
    expected_sha256 = record.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise TranscriptionError("Recovery primary producer receipt is malformed")
    receipt_path = (destination / relative_path).resolve()
    primary_root = (destination / "primary_asr").resolve()
    expected_path = primary_root / f"{sha256_json(expected_identity)}.json"
    if (
        receipt_path != expected_path
        or receipt_path.parent != primary_root
        or receipt_path.is_symlink()
    ):
        raise TranscriptionError("Recovery primary producer receipt path is unsafe")
    if not receipt_path.is_file() or sha256_file(receipt_path) != expected_sha256:
        raise TranscriptionError("Recovery primary producer receipt hash mismatch")
    auth_body = {
        "identity_sha256": sha256_json(expected_identity),
        "receipt_sha256": expected_sha256,
    }
    expected_auth_tag = _raw_asr_auth_tag(auth_key, "primary", auth_body)
    if not isinstance(record.get("auth_tag"), str) or not hmac.compare_digest(
        record["auth_tag"], expected_auth_tag
    ):
        raise TranscriptionError("Recovery primary producer receipt authentication failed")
    auth_path = _primary_auth_record_path(destination, expected_identity)
    try:
        saved_auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TranscriptionError(
            "Recovery primary authentication record is missing or invalid"
        ) from exc
    if saved_auth != {**auth_body, "auth_tag": expected_auth_tag}:
        raise TranscriptionError("Recovery primary authentication record mismatch")
    try:
        envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
        body = envelope["data"]
        identity = body["identity"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TranscriptionError("Recovery primary producer receipt is invalid") from exc
    if (
        not isinstance(body, Mapping)
        or not isinstance(identity, Mapping)
        or identity != dict(expected_identity)
    ):
        raise TranscriptionError("Recovery primary producer receipt identity mismatch")
    loaded = primary_checkpoint.load_primary(receipt_path, expected_identity)
    if loaded is None:
        raise TranscriptionError("Recovery primary producer receipt is stale")
    return loaded, identity


def _validate_recovery_producer_receipts(
    destination: Path,
    checkpoint: Mapping[str, Any],
    *,
    audio_sha256: str,
    episode: int,
    settings: RawASRV2Config,
    canonical_names: Sequence[str],
    religious_terms: Sequence[str],
    segments: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
    vad_regions: Sequence[Mapping[str, Any]],
    rescue_failures: Sequence[Mapping[str, Any]],
    expected_primary_identity: Mapping[str, Any],
    auth_key: bytes,
) -> None:
    receipts = checkpoint.get("producer_receipts")
    if not isinstance(receipts, Mapping):
        raise TranscriptionError(
            "Recovery checkpoint lacks immutable producer receipts"
        )
    primary_record = receipts.get("primary")
    if not isinstance(primary_record, Mapping):
        raise TranscriptionError("Recovery primary producer receipt is missing")
    primary, identity = _load_primary_receipt_for_recovery(
        destination,
        primary_record,
        expected_identity=expected_primary_identity,
        auth_key=auth_key,
    )
    producer = expected_primary_identity.get("producer")
    model_identity = expected_primary_identity.get("model")
    if not isinstance(producer, Mapping) or not isinstance(model_identity, Mapping):
        raise TranscriptionError("Recovery primary producer identity is incomplete")

    primary_receipt_sha256 = str(primary_record["sha256"])
    vad_binding = _vad_journal_binding(
        audio_sha256=audio_sha256,
        primary_receipt_sha256=primary_receipt_sha256,
        settings=settings,
        producer=producer,
    )
    try:
        saved_vad_result = UnitJournal(
            destination / "raw_asr_units" / "vad", vad_binding
        ).read("independent-vad")
    except (IntegrityError, KeyError, TypeError, ValueError) as exc:
        raise TranscriptionError("Recovery VAD producer receipt is invalid") from exc
    vad_result = _verify_signed_unit_result(
        saved_vad_result,
        auth_key,
        purpose="vad",
        binding=vad_binding,
        uid="independent-vad",
    )
    if (
        receipts.get("vad_result_sha256") != _unit_result_sha256(saved_vad_result)
        or receipts.get("vad_auth_tag") != saved_vad_result.get("auth_tag")
        or vad_result.get("vad_regions") != list(vad_regions)
        or vad_result.get("vad_fallback_reason") != checkpoint.get("vad_fallback_reason")
        or vad_result.get("independent_vad") != checkpoint.get("independent_vad")
    ):
        raise TranscriptionError("Recovery VAD evidence differs from its producer receipt")

    batches = checkpoint.get("rescue_batches")
    if not isinstance(batches, list):
        raise TranscriptionError("Recovery canonical rescue plan is missing")
    rescue_binding = _rescue_journal_binding(
        audio_sha256=audio_sha256,
        primary_receipt_sha256=primary_receipt_sha256,
        model_identity=model_identity,
        producer=producer,
        settings=settings,
        prompt=_prompt_text(canonical_names, religious_terms),
        rescue_batches=batches,
    )
    saved_rescue_hashes = receipts.get("rescue_result_sha256")
    saved_rescue_tags = receipts.get("rescue_auth_tags")
    if not isinstance(saved_rescue_hashes, Mapping) or not isinstance(
        saved_rescue_tags, Mapping
    ):
        raise TranscriptionError("Recovery rescue producer receipts are missing")
    journal = UnitJournal(destination / "raw_asr_units" / "rescue", rescue_binding)
    receipt_segments: list[dict[str, Any]] = []
    receipt_words: list[dict[str, Any]] = []
    receipt_failures: list[dict[str, Any]] = []
    expected_uids: list[str] = []
    for index, span in enumerate(batches, start=1):
        if not isinstance(span, Mapping):
            raise TranscriptionError("Recovery rescue batch identity is malformed")
        uid = _rescue_batch_uid(index, span)
        expected_uids.append(uid)
        try:
            saved_result = journal.read(uid)
        except (IntegrityError, KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(
                f"Recovery rescue producer receipt is invalid: {uid}"
            ) from exc
        result = _verify_signed_unit_result(
            saved_result,
            auth_key,
            purpose="rescue",
            binding=rescue_binding,
            uid=uid,
        )
        if (
            result.get("target") != dict(span)
            or saved_rescue_hashes.get(uid) != _unit_result_sha256(saved_result)
            or saved_rescue_tags.get(uid) != saved_result.get("auth_tag")
        ):
            raise TranscriptionError(
                f"Recovery rescue evidence differs from producer receipt: {uid}"
            )
        if result.get("status") == "completed":
            found_segments = result.get("segments")
            found_words = result.get("words")
            if not isinstance(found_segments, list) or not isinstance(found_words, list):
                raise TranscriptionError(f"Recovery rescue result is malformed: {uid}")
            receipt_segments.extend(found_segments)
            receipt_words.extend(found_words)
        elif result.get("status") == "manual_review_required":
            failure = result.get("failure")
            if not isinstance(failure, Mapping):
                raise TranscriptionError(f"Recovery rescue failure is malformed: {uid}")
            receipt_failures.append(dict(failure))
        else:
            raise TranscriptionError(f"Recovery rescue status is invalid: {uid}")
    if set(saved_rescue_hashes) != set(expected_uids) or set(
        saved_rescue_tags
    ) != set(expected_uids):
        raise TranscriptionError("Recovery rescue producer receipt inventory mismatch")
    receipt_segments, receipt_words = merge_rescue_evidence(
        primary["segments"], primary["words"], receipt_segments, receipt_words
    )
    if (
        receipt_segments != list(segments)
        or receipt_words != list(words)
        or receipt_failures != list(rescue_failures)
    ):
        raise TranscriptionError(
            "Recovery ASR evidence differs from immutable producer receipts"
        )


def _write_raw_asr_v2_recovery_checkpoint(
    destination: Path,
    *,
    episode: int,
    source_path: Path,
    audio_sha256: str,
    caption_file: Path | None,
    caption_sha256: str | None,
    settings: RawASRV2Config,
    canonical_names: Sequence[str],
    religious_terms: Sequence[str],
    requested_hallucination_reviews: Sequence[str],
    main_pass_device: str,
    main_pass_compute_type: str,
    final_runtime_device: str,
    final_runtime_compute_type: str,
    runtime_fallback_reason: str | None,
    language: str,
    segments: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
    vad_regions: Sequence[Mapping[str, Any]],
    vad_fallback_reason: str | None,
    independent_vad: bool,
    initial_speech_coverage: Mapping[str, Any],
    speech_coverage: Mapping[str, Any],
    rescue_failures: Sequence[Mapping[str, Any]],
    rescue_span_count: int,
    rescue_batches: Sequence[Mapping[str, Any]],
    rescue_budget_audit: Mapping[str, Any],
    youtube_captions: Sequence[Mapping[str, Any]],
    speech_hole_records: Sequence[Mapping[str, Any]],
    producer_receipts: Mapping[str, Any],
) -> Path:
    """Persist expensive ASR/VAD evidence before bounded review gates run."""

    checkpoint_path = destination / RAW_ASR_V2_RECOVERY_FILENAME
    atomic_write_json(
        checkpoint_path,
        {
            "format": RAW_ASR_V2_RECOVERY_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "audio_path": str(source_path.resolve()),
            "audio_sha256": audio_sha256,
            "captions_path": (
                str(caption_file.resolve()) if caption_file is not None else None
            ),
            "caption_sha256": caption_sha256,
            "canonical_names": list(canonical_names),
            "religious_terms": list(religious_terms),
            "hallucination_review_utterance_uids": list(
                requested_hallucination_reviews
            ),
            "model": {
                "main_pass_device": main_pass_device,
                "main_pass_compute_type": main_pass_compute_type,
                "final_runtime_device": final_runtime_device,
                "final_runtime_compute_type": final_runtime_compute_type,
                "runtime_fallback_reason": runtime_fallback_reason,
                "language": language,
                "settings": asdict(settings),
            },
            "segments": list(segments),
            "words": list(words),
            "vad_regions": list(vad_regions),
            "vad_fallback_reason": vad_fallback_reason,
            "independent_vad": independent_vad,
            "initial_speech_coverage": dict(initial_speech_coverage),
            "speech_coverage": dict(speech_coverage),
            "rescue_failures": list(rescue_failures),
            "rescue_span_count": rescue_span_count,
            "rescue_batches": list(rescue_batches),
            "rescue_budget_audit": dict(rescue_budget_audit),
            "youtube_captions": list(youtube_captions),
            "speech_hole_records": list(speech_hole_records),
            "producer_receipts": dict(producer_receipts),
        },
    )
    return checkpoint_path


def recover_raw_asr_v2_from_checkpoint(
    audio_path: str | Path,
    prepare_dir: str | Path,
    *,
    episode: int,
    config: RawASRV2Config | None = None,
    canonical_names: Sequence[str] = (),
    religious_terms: Sequence[str] = (),
    captions_path: str | Path | None = None,
    hallucination_review_utterance_uids: Sequence[str] = (),
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Finish post-processing from hash-bound ASR/VAD evidence without a GPU."""

    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise TranscriptionError("episode must be a positive integer")
    settings, requested_hallucination_reviews = _settings_with_review_requests(
        config,
        hallucination_review_utterance_uids,
    )
    auth_key = _raw_asr_auth_key(settings)
    if auth_key is None:
        raise TranscriptionError(
            "Raw ASR recovery requires authenticated production evidence"
        )
    source_path = Path(audio_path)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise TranscriptionError(f"Audio input is missing or empty: {source_path}")
    destination = Path(prepare_dir)
    destination.mkdir(parents=True, exist_ok=True)
    recovery_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else destination / RAW_ASR_V2_RECOVERY_FILENAME
    )
    if not recovery_path.is_file() or recovery_path.stat().st_size <= 0:
        raise TranscriptionError(
            f"Raw ASR recovery checkpoint is missing or empty: {recovery_path}"
        )
    try:
        checkpoint = json.loads(recovery_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TranscriptionError(
            f"Could not read raw ASR recovery checkpoint: {exc}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise TranscriptionError("Raw ASR recovery checkpoint must be an object")
    if checkpoint.get("format") != RAW_ASR_V2_RECOVERY_FORMAT:
        raise TranscriptionError("Unsupported raw ASR recovery checkpoint format")
    if checkpoint.get("episode") != episode:
        raise TranscriptionError("Recovery checkpoint episode mismatch")

    audio_sha = sha256_file(source_path)
    if checkpoint.get("audio_sha256") != audio_sha:
        raise TranscriptionError("Recovery checkpoint audio SHA-256 mismatch")
    caption_file = Path(captions_path) if captions_path is not None else None
    if caption_file is not None and not caption_file.is_file():
        raise TranscriptionError(f"Caption input is missing: {caption_file}")
    caption_sha = sha256_file(caption_file) if caption_file is not None else None
    if checkpoint.get("caption_sha256") != caption_sha:
        raise TranscriptionError("Recovery checkpoint caption SHA-256 mismatch")
    if list(checkpoint.get("canonical_names", [])) != list(canonical_names):
        raise TranscriptionError("Recovery checkpoint canonical-name list mismatch")
    if list(checkpoint.get("religious_terms", [])) != list(religious_terms):
        raise TranscriptionError("Recovery checkpoint religious-term list mismatch")
    checkpoint_review_uids = checkpoint.get(
        "hallucination_review_utterance_uids", []
    )
    if (
        not isinstance(checkpoint_review_uids, list)
        or any(
            not isinstance(uid, str) or not uid.strip()
            for uid in checkpoint_review_uids
        )
        or len(set(checkpoint_review_uids)) != len(checkpoint_review_uids)
    ):
        raise TranscriptionError("Recovery checkpoint review-UID list is malformed")
    # Review requests affect only bounded WAV extraction and candidate metadata;
    # they never alter the expensive ASR/VAD evidence stored in this checkpoint.
    # A later run may therefore add exact UIDs without retranscribing the episode.
    # Removing a request is rejected so recovery cannot silently reduce scrutiny.
    if not set(checkpoint_review_uids).issubset(requested_hallucination_reviews):
        raise TranscriptionError(
            "Recovery checkpoint review-UIDs cannot be removed"
        )

    saved_model = checkpoint.get("model")
    if not isinstance(saved_model, Mapping) or not isinstance(
        saved_model.get("settings"), Mapping
    ):
        raise TranscriptionError("Recovery checkpoint model settings are missing")
    try:
        saved_settings = json.loads(
            json.dumps(saved_model["settings"], sort_keys=True, allow_nan=False)
        )
        current_settings = json.loads(
            json.dumps(asdict(settings), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(
            f"Recovery checkpoint settings are not strict JSON: {exc}"
        ) from exc
    postprocess_only_fields = {
        "correction_max_internal_word_gap_ms",
        "correction_review_word_gap_ms",
        "extra_audio_review_uids",
        "orphan_caption_min_lexical_asr_overlap_ms",
        "orphan_caption_min_shared_tokens",
        "orphan_caption_min_character_bigram_dice",
    }
    required_saved_fields = set(current_settings).difference(
        postprocess_only_fields
    )
    missing_saved_fields = sorted(
        required_saved_fields.difference(saved_settings)
    )
    if missing_saved_fields:
        raise TranscriptionError(
            "Recovery checkpoint settings are incomplete; missing="
            f"{missing_saved_fields}"
        )
    mismatched_settings = sorted(
        key
        for key, value in saved_settings.items()
        if key not in postprocess_only_fields
        and (key not in current_settings or current_settings[key] != value)
    )
    if mismatched_settings:
        raise TranscriptionError(
            "Recovery checkpoint ASR settings mismatch; fields="
            f"{mismatched_settings}"
        )
    if checkpoint.get("independent_vad") is not True or bool(
        str(checkpoint.get("vad_fallback_reason") or "").strip()
    ):
        raise TranscriptionError(
            "Recovery checkpoint lacks independent Silero VAD evidence"
        )

    def checkpoint_list(name: str) -> list[Any]:
        value = checkpoint.get(name)
        if not isinstance(value, list):
            raise TranscriptionError(
                f"Recovery checkpoint {name} must be a list"
            )
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

    segments = checkpoint_list("segments")
    words = checkpoint_list("words")
    vad_regions = checkpoint_list("vad_regions")
    rescue_failures = checkpoint_list("rescue_failures")
    build_coverage_intervals(words)
    if not vad_regions or any(
        not isinstance(region, Mapping)
        or str(region.get("source", "")) != "silero_vad"
        for region in vad_regions
    ):
        raise TranscriptionError(
            "Recovery checkpoint VAD inventory is missing or not independent"
        )
    initial_coverage = checkpoint.get("initial_speech_coverage")
    final_coverage = checkpoint.get("speech_coverage")
    if not isinstance(initial_coverage, Mapping) or not isinstance(
        final_coverage, Mapping
    ):
        raise TranscriptionError("Recovery checkpoint coverage reports are missing")
    initial_coverage = json.loads(
        json.dumps(initial_coverage, ensure_ascii=False, allow_nan=False)
    )
    final_coverage = json.loads(
        json.dumps(final_coverage, ensure_ascii=False, allow_nan=False)
    )
    validate_raw_rescue_plan(checkpoint)
    expected_primary_identity = _canonical_primary_identity(
        episode=episode,
        audio_sha256=audio_sha,
        settings=settings,
        prompt=_prompt_text(canonical_names, religious_terms),
        model_identity=primary_checkpoint.model_identity(
            primary_checkpoint.resolve_model(settings.model_name)
        ),
        producer_identity=_raw_asr_producer_identity(),
    )
    if (
        saved_model.get("main_pass_device") != expected_primary_identity["device"]
        or saved_model.get("main_pass_compute_type")
        != expected_primary_identity["compute_type"]
        or saved_model.get("final_runtime_device") != "cuda"
        or saved_model.get("final_runtime_compute_type") != settings.compute_type_gpu
        or saved_model.get("runtime_fallback_reason") not in (None, "")
        or saved_model.get("language") != settings.language
    ):
        raise TranscriptionError(
            "Recovery runtime does not match the canonical GPU-only producer identity"
        )
    _validate_recovery_producer_receipts(
        destination,
        checkpoint,
        audio_sha256=audio_sha,
        episode=episode,
        settings=settings,
        canonical_names=canonical_names,
        religious_terms=religious_terms,
        segments=segments,
        words=words,
        vad_regions=vad_regions,
        rescue_failures=rescue_failures,
        expected_primary_identity=expected_primary_identity,
        auth_key=auth_key,
    )
    for coverage in (initial_coverage, final_coverage):
        try:
            require_v2_beta_speech_coverage_config(
                SpeechCoverageConfig(**dict(coverage["config"]))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(
                f"Recovery checkpoint coverage policy is invalid: {exc}"
            ) from exc

    speech_hole_records = validate_speech_holes(
        checkpoint_list("speech_hole_records")
    )
    for hole in speech_hole_records:
        try:
            _validate_persisted_review_audio(
                hole,
                prepare_dir=destination,
                label="recovery speech-hole",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise TranscriptionError(
                f"Recovery checkpoint speech-hole audio failed validation: {exc}"
            ) from exc
    unresolved_intervals = sorted(
        (
            int(issue["start_ms"]),
            int(issue["end_ms"]),
        )
        for issue in final_coverage.get("coverage_issues", [])
        if isinstance(issue, Mapping)
        and issue.get("classification") == "unresolved_speech"
    )
    persisted_hole_intervals = sorted(
        (int(hole["start_ms"]), int(hole["end_ms"]))
        for hole in speech_hole_records
    )
    if unresolved_intervals != persisted_hole_intervals:
        raise TranscriptionError(
            "Recovery checkpoint speech-hole records do not match final coverage"
        )

    youtube_captions = (
        load_vtt_captions(caption_file) if caption_file is not None else []
    )
    correction_utterances = build_correction_utterances(
        [segment for segment in segments if _segment_has_complete_word_inventory(segment)],
        episode=episode,
        youtube_captions=youtube_captions,
        max_internal_word_gap_ms=settings.correction_max_internal_word_gap_ms,
        review_word_gap_ms=settings.correction_review_word_gap_ms,
    )
    # Recovery reuses expensive inference, not a legacy coverage conclusion
    # that may have credited words from an incomplete/repetitive segment.
    primary_segments = [
        segment for segment in segments
        if not str(segment.get("source", "main")).startswith("rescue")
    ]
    initial_coverage = _analyze_raw_speech_coverage(
        vad_regions, primary_segments, words,
        config=settings.speech_coverage_config(),
    )
    final_coverage = _analyze_raw_speech_coverage(
        vad_regions, segments, words,
        config=settings.speech_coverage_config(),
    )
    speech_hole_records = build_speech_hole_records(
        final_coverage,
        episode=episode,
        audio_path=source_path,
        audio_output_root=destination,
        utterances=correction_utterances,
        config=settings,
        existing_records=speech_hole_records,
    )
    correction_utterances = include_speech_holes_for_correction(
        correction_utterances,
        speech_hole_records,
    )
    correction_utterances = include_orphan_youtube_captions_for_correction(
        correction_utterances,
        youtube_captions,
        vad_regions,
        episode=episode,
        min_evidence_coverage_ratio=(
            settings.orphan_caption_min_evidence_coverage_ratio
        ),
        max_unexplained_run_ms=(
            settings.orphan_caption_max_unexplained_run_ms
        ),
        min_absolute_evidence_ms=(
            settings.orphan_caption_min_absolute_evidence_ms
        ),
        trailing_display_tolerance_ms=(
            settings.orphan_caption_trailing_display_tolerance_ms
        ),
        min_lexical_asr_overlap_ms=(
            settings.orphan_caption_min_lexical_asr_overlap_ms
        ),
        min_shared_tokens=settings.orphan_caption_min_shared_tokens,
        min_character_bigram_dice=(
            settings.orphan_caption_min_character_bigram_dice
        ),
    )
    excluded_incomplete_segments = _excluded_incomplete_segments(segments, episode=episode)
    excluded_uids = {item["utterance_uid"] for item in excluded_incomplete_segments}
    correction_utterances, asr_hallucination_records = (
        build_asr_hallucination_records(
            correction_utterances,
            vad_regions,
            episode=episode,
            audio_path=source_path,
            audio_output_root=destination,
            config=settings,
            review_request_uids=[uid for uid in requested_hallucination_reviews if uid not in excluded_uids],
        )
    )

    if sha256_file(source_path) != audio_sha:
        raise TranscriptionError(
            "Audio input changed during raw ASR recovery; refusing to commit"
        )
    input_sha = _raw_asr_input_sha256(
        episode=episode,
        audio_sha256=audio_sha,
        caption_sha256=caption_sha,
        settings=settings,
        canonical_names=canonical_names,
        religious_terms=religious_terms,
        requested_hallucination_reviews=requested_hallucination_reviews,
    )
    main_pass_device = str(
        saved_model.get("main_pass_device", saved_model.get("device", "unknown"))
    )
    main_pass_compute_type = str(
        saved_model.get(
            "main_pass_compute_type",
            saved_model.get("compute_type", "unknown"),
        )
    )
    final_runtime_device = str(
        saved_model.get("final_runtime_device", main_pass_device)
    )
    final_runtime_compute_type = str(
        saved_model.get("final_runtime_compute_type", main_pass_compute_type)
    )
    data = {
        "format_version": RAW_ASR_V2_FORMAT_VERSION,
        "status": "completed",
        "input_sha256": input_sha,
        "episode": episode,
        "audio_path": str(source_path.resolve()),
        "audio_sha256": audio_sha,
        "caption_source_sha256": caption_sha,
        "hallucination_review_utterance_uids": requested_hallucination_reviews,
        "model": {
            "name": settings.model_name,
            "device": main_pass_device,
            "compute_type": main_pass_compute_type,
            "final_runtime_device": final_runtime_device,
            "final_runtime_compute_type": final_runtime_compute_type,
            "runtime_fallback_reason": saved_model.get(
                "runtime_fallback_reason"
            ),
            "faster_whisper_version": _package_version("faster-whisper"),
            "settings": asdict(settings),
        },
        "language": str(saved_model.get("language", settings.language)),
        "segments": segments,
        "excluded_incomplete_segments": excluded_incomplete_segments,
        "required_acoustic_review_regions": required_acoustic_review_regions(segments, vad_regions),
        "words": words,
        "vad_regions": vad_regions,
        "vad_fallback_reason": checkpoint.get("vad_fallback_reason"),
        "independent_vad": True,
        "initial_speech_coverage": initial_coverage,
        "speech_coverage": final_coverage,
        "rescue_failures": rescue_failures,
        "rescue_batches": checkpoint.get("rescue_batches", []),
        "rescue_budget_audit": checkpoint.get("rescue_budget_audit"),
        "youtube_captions": youtube_captions,
        "correction_utterances": correction_utterances,
        "speech_hole_records": speech_hole_records,
        "asr_hallucination_records": asr_hallucination_records,
        "synthetic_timing_count": 0,
        "resumed": True,
        "recovery_checkpoint_sha256": sha256_file(recovery_path),
    }
    data = _sign_raw_asr_artifact(data, auth_key)
    validation_errors = _validate_persisted_raw_asr_v2(
        data,
        input_sha256=input_sha,
        audio_sha256=audio_sha,
        episode=episode,
        prepare_dir=destination,
    )
    if validation_errors:
        raise TranscriptionError(
            "Recovered raw ASR V2 validation failed: "
            + "; ".join(validation_errors[:20])
        )
    output_path = destination / "raw_asr_v2.json"
    marker_path = destination / "raw_asr_v2.done.json"
    atomic_write_json(output_path, data)
    rescue_span_count = checkpoint.get("rescue_span_count")
    if isinstance(rescue_span_count, bool) or not isinstance(
        rescue_span_count, int
    ):
        rescue_span_count = len(initial_coverage.get("rescue_spans", []))
    write_stage_marker(
        marker_path,
        stage="raw_asr_v2",
        input_sha256=input_sha,
        outputs={"raw_asr_v2": output_path},
        details={
            "audio_sha256": audio_sha,
            "utterance_count": len(correction_utterances),
            "asr_hallucination_candidate_count": len(
                asr_hallucination_records
            ),
            "vad_region_count": len(vad_regions),
            "rescue_span_count": rescue_span_count,
            "unresolved_speech_region_count": final_coverage.get(
                "unresolved_speech_region_count"
            ),
            "main_pass_device": main_pass_device,
            "final_runtime_device": final_runtime_device,
            "runtime_fallback_reason": saved_model.get(
                "runtime_fallback_reason"
            ),
            "independent_vad": True,
            "recovered_from_checkpoint": True,
        },
    )
    LOGGER.info(
        "Recovered raw ASR V2 without model inference: utterances=%s, "
        "hallucination_candidates=%s",
        len(correction_utterances),
        len(asr_hallucination_records),
    )
    return data


def transcribe_raw_audio_v2(
    audio_path: str | Path,
    prepare_dir: str | Path,
    *,
    episode: int,
    config: RawASRV2Config | None = None,
    canonical_names: Sequence[str] = (),
    religious_terms: Sequence[str] = (),
    captions_path: str | Path | None = None,
    hallucination_review_utterance_uids: Sequence[str] = (),
    force: bool = False,
    require_resume: bool = False,
) -> dict[str, Any]:
    """Run raw ASR, rescue VAD holes, and persist a correction-first artifact."""

    if force and require_resume:
        raise TranscriptionError("Scoped alignment retry cannot force raw-ASR inference")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise TranscriptionError("episode must be a positive integer")
    settings, requested_hallucination_reviews = _settings_with_review_requests(
        config,
        hallucination_review_utterance_uids,
    )
    auth_key = _raw_asr_auth_key(settings)
    source_path = Path(audio_path)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise TranscriptionError(f"Audio input is missing or empty: {source_path}")
    destination = Path(prepare_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "raw_asr_v2.json"
    marker_path = destination / "raw_asr_v2.done.json"
    audio_sha = sha256_file(source_path)
    caption_file = Path(captions_path) if captions_path is not None else None
    caption_sha = sha256_file(caption_file) if caption_file and caption_file.is_file() else None
    input_sha = _raw_asr_input_sha256(
        episode=episode,
        audio_sha256=audio_sha,
        caption_sha256=caption_sha,
        settings=settings,
        canonical_names=canonical_names,
        religious_terms=religious_terms,
        requested_hallucination_reviews=requested_hallucination_reviews,
    )
    if not force:
        cached = load_valid_raw_asr_v2(
            destination,
            audio_path=source_path,
            episode=episode,
            expected_input_sha256=input_sha,
            require_independent_vad=settings.require_independent_vad,
        )
        if cached is not None:
            cached["resumed"] = True
            return cached
        if auth_key is not None and output_path.is_file() and marker_path.is_file():
            try:
                completed = json.loads(output_path.read_text(encoding="utf-8"))
                if not isinstance(completed, Mapping):
                    raise TranscriptionError("Completed raw ASR artifact is malformed")
                _verify_raw_asr_artifact_auth(completed, auth_key)
            except (OSError, UnicodeError, json.JSONDecodeError, TranscriptionError) as exc:
                raise TranscriptionError(
                    "Completed raw ASR cache is not authenticated by the current key; "
                    "an explicit force rerun is required"
                ) from exc
        if require_resume:
            raise TranscriptionError("Scoped alignment retry requires a validated raw-ASR checkpoint")
        recovery_path = destination / RAW_ASR_V2_RECOVERY_FILENAME
        if recovery_path.is_file():
            return recover_raw_asr_v2_from_checkpoint(
                source_path,
                destination,
                episode=episode,
                config=settings,
                canonical_names=canonical_names,
                religious_terms=religious_terms,
                captions_path=caption_file,
                checkpoint_path=recovery_path,
            )

    # Fail before loading a multi-gigabyte model when an explicitly requested
    # caption file is absent or malformed.
    youtube_captions = (
        load_vtt_captions(caption_file) if caption_file is not None else []
    )
    transcription_settings = settings.transcription_config()
    audit_vad_settings = replace(
        transcription_settings,
        vad_speech_pad_ms=settings.audit_vad_speech_pad_ms,
    )
    WhisperModel, ctranslate2 = _import_whisper()
    device, compute_type = _select_device(transcription_settings, ctranslate2)
    cpu_threads = max(1, os.cpu_count() or 1)
    runtime_fallback_reason: str | None = None
    model: Any | None = None
    model_path = primary_checkpoint.resolve_model(settings.model_name)
    primary_model_identity = primary_checkpoint.model_identity(model_path)
    primary_receipt_record: dict[str, Any] | None = None

    def load_model(target_device: str, target_compute_type: str) -> Any:
        return WhisperModel(
            str(model_path),
            device=target_device,
            compute_type=target_compute_type,
            cpu_threads=cpu_threads,
            num_workers=1,
        )

    def run_main_pass(runtime_model: Any) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        nonlocal primary_receipt_record
        options = _primary_asr_options(settings, prompt)
        identity = {"format": "mas-primary-asr-1", "episode": episode,
                    "audio_sha256": audio_sha, "model": primary_model_identity,
                    "producer": primary_producer, "options": options,
                    "device": device, "compute_type": compute_type,
                    "cpu_threads": cpu_threads, "num_workers": 1}
        checkpoint = destination / "primary_asr" / (sha256_json(identity) + ".json")
        cached = None if force else primary_checkpoint.load_primary(checkpoint, identity)
        if cached is not None:
            if auth_key is not None:
                primary_receipt_record = _primary_receipt_record(
                    destination, checkpoint, identity, auth_key, create=False
                )
                if primary_receipt_record is None:
                    cached = None
            else:
                primary_receipt_record = {
                    "relative_path": checkpoint.relative_to(destination).as_posix(),
                    "sha256": sha256_file(checkpoint),
                }
        if cached is not None:
            LOGGER.info("Reusing hash-bound primary ASR before VAD/rescue")
            mark_work_progress("raw_asr:primary_checkpoint", completed=True)
            return SimpleNamespace(language=cached["language"]), cached["segments"], cached["words"]
        iterator, pass_info = runtime_model.transcribe(str(source_path), **options)
        pass_segments, pass_words = consume_coarse_segments(
            iterator, source="main"
        )
        if sha256_file(source_path) != audio_sha:
            raise TranscriptionError("Audio input changed during raw ASR primary inference")
        if primary_checkpoint.model_identity(model_path) != primary_model_identity:
            raise TranscriptionError("primary ASR model changed during inference")
        primary_checkpoint.save_primary(
            checkpoint, identity, str(getattr(pass_info, "language", settings.language)),
            pass_segments, pass_words)
        if auth_key is not None:
            primary_receipt_record = _primary_receipt_record(
                destination, checkpoint, identity, auth_key, create=True
            )
        else:
            primary_receipt_record = {
                "relative_path": checkpoint.relative_to(destination).as_posix(),
                "sha256": sha256_file(checkpoint),
            }
        return pass_info, pass_segments, pass_words

    def run_rescue_pass(
        runtime_model: Any,
        clip_path: Path,
        *,
        start_ms: int,
        rescue_index: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        iterator, _ = runtime_model.transcribe(
            str(clip_path),
            language=settings.language,
            task="transcribe",
            beam_size=settings.rescue_beam_size,
            best_of=max(settings.best_of, settings.rescue_beam_size),
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            word_timestamps=True,
            vad_filter=False,
        )
        return consume_coarse_segments(
            iterator,
            offset_ms=start_ms,
            source=f"rescue-{rescue_index}",
        )

    primary_producer = _raw_asr_producer_identity()
    try:
        try:
            model = load_model(device, compute_type)
        except Exception as exc:
            if (
                device == "cuda"
                and settings.allow_cpu_fallback
                and _is_cuda_runtime_error(exc)
            ):
                LOGGER.warning(
                    "CUDA raw-ASR model initialization failed (%s). Retrying "
                    "once on CPU/%s.",
                    exc,
                    settings.compute_type_cpu,
                )
                runtime_fallback_reason = (
                    f"CUDA initialization: {type(exc).__name__}: {exc}"
                )
                device = "cpu"
                compute_type = settings.compute_type_cpu
                try:
                    model = load_model(device, compute_type)
                except Exception as cpu_exc:
                    raise TranscriptionError(
                        "Raw ASR CUDA initialization failed and the one CPU/int8 "
                        f"fallback also failed: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise TranscriptionError(
                    f"Could not load raw ASR model {settings.model_name} on "
                    f"{device}/{compute_type}: {exc}"
                ) from exc

        prompt = _prompt_text(canonical_names, religious_terms)
        try:
            info, primary_segments, primary_words = run_main_pass(model)
        except Exception as exc:
            if (
                device == "cuda"
                and settings.allow_cpu_fallback
                and _is_cuda_runtime_error(exc)
            ):
                LOGGER.warning(
                    "CUDA failed during raw ASR inference (%s). Retrying the "
                    "full failed pass once on CPU/%s.",
                    exc,
                    settings.compute_type_cpu,
                )
                runtime_fallback_reason = (
                    f"CUDA inference: {type(exc).__name__}: {exc}"
                )
                model = None
                gc.collect()
                device = "cpu"
                compute_type = settings.compute_type_cpu
                try:
                    model = load_model(device, compute_type)
                    info, primary_segments, primary_words = run_main_pass(model)
                except Exception as cpu_exc:
                    raise TranscriptionError(
                        "The CUDA raw ASR attempt failed and the one CPU/int8 "
                        f"fallback also failed: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise TranscriptionError(
                    f"Full Turkish raw ASR pass failed: {exc}"
                ) from exc

        main_pass_device = device
        main_pass_compute_type = compute_type
        if primary_receipt_record is None:
            raise TranscriptionError("Primary ASR producer receipt was not committed")
        vad_binding = _vad_journal_binding(
            audio_sha256=audio_sha,
            primary_receipt_sha256=primary_receipt_record["sha256"],
            settings=settings,
            producer=primary_producer,
        )
        vad_journal = UnitJournal(
            destination / "raw_asr_units" / "vad", vad_binding
        )
        try:
            saved_vad_result = None if force else vad_journal.read("independent-vad")
        except IntegrityError as exc:
            raise TranscriptionError("Independent VAD producer receipt is invalid") from exc
        if saved_vad_result is None:
            vad_regions, vad_fallback_reason = _extract_vad_regions(
                source_path, audit_vad_settings, primary_words
            )
            independent_vad = (
                not bool(str(vad_fallback_reason or "").strip())
                and bool(vad_regions)
                and all(
                    isinstance(region, Mapping)
                    and str(region.get("source", "")).strip() == "silero_vad"
                    for region in vad_regions
                )
            )
            vad_result = {
                "vad_regions": vad_regions,
                "vad_fallback_reason": vad_fallback_reason,
                "independent_vad": independent_vad,
            }
            if sha256_file(source_path) != audio_sha:
                raise TranscriptionError("Audio input changed during independent VAD")
            saved_vad_result = (
                _signed_unit_result(
                    auth_key,
                    purpose="vad",
                    binding=vad_binding,
                    uid="independent-vad",
                    result=vad_result,
                )
                if auth_key is not None
                else vad_result
            )
            vad_journal.write("independent-vad", saved_vad_result)
        else:
            vad_result = (
                _verify_signed_unit_result(
                    saved_vad_result,
                    auth_key,
                    purpose="vad",
                    binding=vad_binding,
                    uid="independent-vad",
                )
                if auth_key is not None
                else saved_vad_result
            )
            if not isinstance(vad_result, Mapping):
                raise TranscriptionError("Independent VAD producer receipt is malformed")
            vad_regions = vad_result.get("vad_regions")
            vad_fallback_reason = vad_result.get("vad_fallback_reason")
            independent_vad = vad_result.get("independent_vad")
            if not isinstance(vad_regions, list) or not isinstance(independent_vad, bool):
                raise TranscriptionError("Independent VAD producer receipt is malformed")
        if settings.require_independent_vad and not independent_vad:
            raise TranscriptionError(
                "V2 requires independent audio VAD; word-timing-derived or "
                "fallback speech regions cannot prove missing-dialogue coverage"
            )
        if not independent_vad:
            LOGGER.warning(
                "Unsafe debug override accepted non-independent VAD. This raw "
                "artifact must not pass final timing/publication QA."
            )
        if not vad_regions and not primary_words:
            raise TranscriptionError(
                "Raw ASR and independent VAD both returned no timed speech "
                "evidence; speech coverage cannot be proven"
            )
        coverage_config = settings.speech_coverage_config()
        initial_coverage = _analyze_raw_speech_coverage(
            vad_regions, primary_segments, primary_words,
            config=coverage_config,
        )
        rescue_spans = list(initial_coverage.get("rescue_spans", []))
        coverage_metrics = initial_coverage.get("metrics", {})
        primary_speech_coverage_ratio = float(
            coverage_metrics.get("independent_vad_speech_coverage_ratio",
                                 coverage_metrics.get("speech_coverage_ratio", 0.0))
        )
        try:
            rescue_spans, rescue_budget_audit = _plan_rescue_batches(
                initial_coverage, vad_regions, primary_segments, primary_words, settings,
            )
        except TranscriptionError as exc:
            rescue_total_ms = sum(
                int(span["end_ms"]) - int(span["start_ms"])
                for span in rescue_spans
            )
            rescue_max_ms = max(
                (
                    int(span["end_ms"]) - int(span["start_ms"])
                    for span in rescue_spans
                ),
                default=0,
            )
            raise TranscriptionError(
                f"{exc}; inspect the "
                "VAD/ASR inputs; "
                f"primary_segments={len(primary_segments)}, "
                f"primary_words={len(primary_words)}, "
                f"vad_regions={len(vad_regions)}, "
                f"speech_coverage_ratio="
                f"{primary_speech_coverage_ratio:.4f}, "
                f"rescue_total_seconds={rescue_total_ms / 1000.0:.3f}, "
                f"longest_rescue_seconds={rescue_max_ms / 1000.0:.3f}"
            ) from exc

        rescue_segments: list[dict[str, Any]] = []
        rescue_words: list[dict[str, Any]] = []
        rescue_failures: list[dict[str, Any]] = []
        rescue_result_hashes: dict[str, str] = {}
        rescue_auth_tags: dict[str, str] = {}
        rescue_binding = _rescue_journal_binding(
            audio_sha256=audio_sha,
            primary_receipt_sha256=primary_receipt_record["sha256"],
            model_identity=primary_model_identity,
            producer=primary_producer,
            settings=settings,
            prompt=prompt,
            rescue_batches=rescue_spans,
        )
        rescue_journal = UnitJournal(
            destination / "raw_asr_units" / "rescue",
            rescue_binding,
        )
        clips_dir = destination / "speech_holes"
        clips_dir.mkdir(parents=True, exist_ok=True)
        for index, span in enumerate(rescue_spans, start=1):
            start_ms = int(span["start_ms"])
            end_ms = int(span["end_ms"])
            batch_uid = _rescue_batch_uid(index, span)
            try:
                saved_rescue_result = (
                    None if force else rescue_journal.read(batch_uid)
                )
            except IntegrityError as exc:
                raise TranscriptionError(
                    f"Rescue producer receipt is invalid: {batch_uid}"
                ) from exc
            if saved_rescue_result is not None:
                rescue_result = (
                    _verify_signed_unit_result(
                        saved_rescue_result,
                        auth_key,
                        purpose="rescue",
                        binding=rescue_binding,
                        uid=batch_uid,
                    )
                    if auth_key is not None
                    else saved_rescue_result
                )
                if not isinstance(rescue_result, Mapping) or rescue_result.get(
                    "target"
                ) != dict(span):
                    raise TranscriptionError(
                        f"Rescue producer receipt identity mismatch: {batch_uid}"
                    )
                if rescue_result.get("status") == "completed":
                    found_segments = rescue_result.get("segments")
                    found_words = rescue_result.get("words")
                    if not isinstance(found_segments, list) or not isinstance(
                        found_words, list
                    ):
                        raise TranscriptionError(
                            f"Rescue producer receipt is malformed: {batch_uid}"
                        )
                    rescue_segments.extend(found_segments)
                    rescue_words.extend(found_words)
                elif rescue_result.get("status") == "manual_review_required":
                    failure = rescue_result.get("failure")
                    if not isinstance(failure, Mapping):
                        raise TranscriptionError(
                            f"Rescue producer receipt is malformed: {batch_uid}"
                        )
                    rescue_failures.append(dict(failure))
                else:
                    raise TranscriptionError(
                        f"Rescue producer receipt status is invalid: {batch_uid}"
                    )
                rescue_result_hashes[batch_uid] = _unit_result_sha256(
                    saved_rescue_result
                )
                if auth_key is not None:
                    rescue_auth_tags[batch_uid] = saved_rescue_result["auth_tag"]
                continue
            clip_path = (
                clips_dir
                / f"speech_hole_{index:03d}_{start_ms}_{end_ms}.wav"
            )
            rescue_result: dict[str, Any]
            try:
                _extract_clip(source_path, clip_path, start_ms, end_ms)
                found_segments, found_words = run_rescue_pass(
                    model,
                    clip_path,
                    start_ms=start_ms,
                    rescue_index=index,
                )
            except Exception as exc:
                if (
                    device == "cuda"
                    and settings.allow_cpu_fallback
                    and _is_cuda_runtime_error(exc)
                ):
                    LOGGER.warning(
                        "CUDA failed during speech-hole rescue (%s). Switching "
                        "remaining rescue work to CPU/%s.",
                        exc,
                        settings.compute_type_cpu,
                    )
                    runtime_fallback_reason = (
                        f"CUDA rescue inference: {type(exc).__name__}: {exc}"
                    )
                    model = None
                    gc.collect()
                    device = "cpu"
                    compute_type = settings.compute_type_cpu
                    try:
                        model = load_model(device, compute_type)
                    except Exception as cpu_init_exc:
                        raise TranscriptionError(
                            "CUDA speech-hole rescue failed and the CPU/int8 "
                            f"fallback model could not be loaded: {cpu_init_exc}"
                        ) from cpu_init_exc
                    try:
                        found_segments, found_words = run_rescue_pass(
                            model,
                            clip_path,
                            start_ms=start_ms,
                            rescue_index=index,
                        )
                        rescue_result = {
                            "target": dict(span),
                            "status": "completed",
                            "segments": found_segments,
                            "words": found_words,
                            "runtime_device": device,
                            "runtime_compute_type": compute_type,
                        }
                    except Exception as cpu_exc:
                        failure = {
                                "start_ms": start_ms,
                                "end_ms": end_ms,
                                "error_type": type(exc).__name__,
                                "fallback_attempted": True,
                                "fallback_error_type": type(cpu_exc).__name__,
                                "status": "manual_review_required",
                            }
                        rescue_result = {
                            "target": dict(span),
                            "status": "manual_review_required",
                            "failure": failure,
                            "runtime_device": device,
                            "runtime_compute_type": compute_type,
                        }
                else:
                    failure = {
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "error_type": type(exc).__name__,
                            "fallback_attempted": False,
                            "status": "manual_review_required",
                        }
                    rescue_result = {
                        "target": dict(span),
                        "status": "manual_review_required",
                        "failure": failure,
                        "runtime_device": device,
                        "runtime_compute_type": compute_type,
                    }
            else:
                rescue_result = {
                    "target": dict(span),
                    "status": "completed",
                    "segments": found_segments,
                    "words": found_words,
                    "runtime_device": device,
                    "runtime_compute_type": compute_type,
                }
            if sha256_file(source_path) != audio_sha:
                raise TranscriptionError("Audio input changed during rescue ASR")
            saved_rescue_result = (
                _signed_unit_result(
                    auth_key,
                    purpose="rescue",
                    binding=rescue_binding,
                    uid=batch_uid,
                    result=rescue_result,
                )
                if auth_key is not None
                else rescue_result
            )
            rescue_journal.write(batch_uid, saved_rescue_result)
            rescue_result_hashes[batch_uid] = _unit_result_sha256(
                saved_rescue_result
            )
            if auth_key is not None:
                rescue_auth_tags[batch_uid] = saved_rescue_result["auth_tag"]
            if rescue_result["status"] == "completed":
                rescue_segments.extend(rescue_result["segments"])
                rescue_words.extend(rescue_result["words"])
            else:
                rescue_failures.append(rescue_result["failure"])

        segments, words = merge_rescue_evidence(
            primary_segments, primary_words, rescue_segments, rescue_words
        )
        final_coverage = _analyze_raw_speech_coverage(
            vad_regions, segments, words,
            config=coverage_config,
        )
        correction_utterances = build_correction_utterances(
            [segment for segment in segments if _segment_has_complete_word_inventory(segment)],
            episode=episode,
            youtube_captions=youtube_captions,
            max_internal_word_gap_ms=(
                settings.correction_max_internal_word_gap_ms
            ),
            review_word_gap_ms=settings.correction_review_word_gap_ms,
        )
        speech_hole_records = build_speech_hole_records(
            final_coverage,
            episode=episode,
            audio_path=source_path,
            audio_output_root=destination,
            utterances=correction_utterances,
            config=settings,
        )
        _write_raw_asr_v2_recovery_checkpoint(
            destination,
            episode=episode,
            source_path=source_path,
            audio_sha256=audio_sha,
            caption_file=caption_file,
            caption_sha256=caption_sha,
            settings=settings,
            canonical_names=canonical_names,
            religious_terms=religious_terms,
            requested_hallucination_reviews=(
                requested_hallucination_reviews
            ),
            main_pass_device=main_pass_device,
            main_pass_compute_type=main_pass_compute_type,
            final_runtime_device=device,
            final_runtime_compute_type=compute_type,
            runtime_fallback_reason=runtime_fallback_reason,
            language=str(getattr(info, "language", settings.language)),
            segments=segments,
            words=words,
            vad_regions=vad_regions,
            vad_fallback_reason=vad_fallback_reason,
            independent_vad=independent_vad,
            initial_speech_coverage=initial_coverage,
            speech_coverage=final_coverage,
            rescue_failures=rescue_failures,
            rescue_span_count=len(rescue_spans),
            rescue_batches=rescue_spans,
            rescue_budget_audit=rescue_budget_audit,
            youtube_captions=youtube_captions,
            speech_hole_records=speech_hole_records,
            producer_receipts={
                "primary": primary_receipt_record,
                "vad_result_sha256": _unit_result_sha256(saved_vad_result),
                "vad_auth_tag": (
                    saved_vad_result.get("auth_tag")
                    if auth_key is not None
                    else None
                ),
                "rescue_result_sha256": rescue_result_hashes,
                "rescue_auth_tags": rescue_auth_tags,
            },
        )
        correction_utterances = include_speech_holes_for_correction(
            correction_utterances, speech_hole_records
        )
        excluded_incomplete_segments = _excluded_incomplete_segments(segments, episode=episode)
        excluded_uids = {item["utterance_uid"] for item in excluded_incomplete_segments}
        if independent_vad:
            correction_utterances = include_orphan_youtube_captions_for_correction(
                correction_utterances,
                youtube_captions,
                vad_regions,
                episode=episode,
                min_evidence_coverage_ratio=(
                    settings.orphan_caption_min_evidence_coverage_ratio
                ),
                max_unexplained_run_ms=(
                    settings.orphan_caption_max_unexplained_run_ms
                ),
                min_absolute_evidence_ms=(
                    settings.orphan_caption_min_absolute_evidence_ms
                ),
                trailing_display_tolerance_ms=(
                    settings.orphan_caption_trailing_display_tolerance_ms
                ),
                min_lexical_asr_overlap_ms=(
                    settings.orphan_caption_min_lexical_asr_overlap_ms
                ),
                min_shared_tokens=settings.orphan_caption_min_shared_tokens,
                min_character_bigram_dice=(
                    settings.orphan_caption_min_character_bigram_dice
                ),
            )
            (
                correction_utterances,
                asr_hallucination_records,
            ) = build_asr_hallucination_records(
                correction_utterances,
                vad_regions,
                episode=episode,
                audio_path=source_path,
                audio_output_root=destination,
                config=settings,
                review_request_uids=[uid for uid in requested_hallucination_reviews if uid not in excluded_uids],
            )
        else:
            if requested_hallucination_reviews:
                raise TranscriptionError(
                    "Explicit ASR hallucination review requires independent VAD"
                )
            # Unsafe debug artifacts remain inspectable but non-publishable.
            # Candidate inference is deliberately disabled because word-derived
            # fallback regions cannot establish ASR-vs-audio independence.
            asr_hallucination_records = []
        # The ASR iterator is lazy and may run for hours.  Re-hash the source at
        # the commit boundary so bytes replaced during inference can never be
        # blessed by an output/marker carrying the old input identity.
        commit_audio_sha = sha256_file(source_path)
        if commit_audio_sha != audio_sha:
            raise TranscriptionError(
                "Audio input changed during raw ASR; refusing to write a "
                "completed artifact or marker"
            )
        data = {
            "format_version": RAW_ASR_V2_FORMAT_VERSION,
            "status": "completed",
            "input_sha256": input_sha,
            "episode": episode,
            "audio_path": str(source_path.resolve()),
            "audio_sha256": audio_sha,
            "caption_source_sha256": caption_sha,
            "hallucination_review_utterance_uids": (
                requested_hallucination_reviews
            ),
            "model": {
                "name": settings.model_name,
                "device": main_pass_device,
                "compute_type": main_pass_compute_type,
                "final_runtime_device": device,
                "final_runtime_compute_type": compute_type,
                "runtime_fallback_reason": runtime_fallback_reason,
                "faster_whisper_version": _package_version("faster-whisper"),
                "settings": asdict(settings),
            },
            "language": str(getattr(info, "language", settings.language)),
            "segments": segments,
            "excluded_incomplete_segments": excluded_incomplete_segments,
            "required_acoustic_review_regions": required_acoustic_review_regions(segments, vad_regions),
            "words": words,
            "vad_regions": vad_regions,
            "vad_fallback_reason": vad_fallback_reason,
            "independent_vad": independent_vad,
            "initial_speech_coverage": initial_coverage,
            "speech_coverage": final_coverage,
            "rescue_failures": rescue_failures,
            "rescue_batches": rescue_spans,
            "rescue_budget_audit": rescue_budget_audit,
            "youtube_captions": youtube_captions,
            "correction_utterances": correction_utterances,
            "speech_hole_records": speech_hole_records,
            "asr_hallucination_records": asr_hallucination_records,
            "synthetic_timing_count": 0,
            "resumed": False,
        }
        if auth_key is not None:
            data = _sign_raw_asr_artifact(data, auth_key)
        validation_errors = _validate_persisted_raw_asr_v2(
            data,
            input_sha256=input_sha,
            audio_sha256=audio_sha,
            episode=episode,
            prepare_dir=destination,
        )
        if validation_errors:
            raise TranscriptionError(
                "Raw ASR V2 validation failed: "
                + "; ".join(validation_errors[:20])
            )
        atomic_write_json(output_path, data)
        write_stage_marker(
            marker_path,
            stage="raw_asr_v2",
            input_sha256=input_sha,
            outputs={"raw_asr_v2": output_path},
            details={
                "audio_sha256": audio_sha,
                "utterance_count": len(data["correction_utterances"]),
                "asr_hallucination_candidate_count": len(
                    asr_hallucination_records
                ),
                "vad_region_count": len(vad_regions),
                "rescue_span_count": len(rescue_spans),
                "unresolved_speech_region_count": final_coverage.get(
                    "unresolved_speech_region_count"
                ),
                "main_pass_device": main_pass_device,
                "final_runtime_device": device,
                "runtime_fallback_reason": runtime_fallback_reason,
                "independent_vad": independent_vad,
            },
        )
        return data
    finally:
        model = None
        gc.collect()


__all__ = [
    "RAW_ASR_V2_FORMAT_VERSION",
    "RawASRV2Config",
    "build_correction_utterances",
    "build_coverage_intervals",
    "build_asr_hallucination_records",
    "build_speech_hole_records",
    "effective_rescue_span_limit",
    "include_orphan_youtube_captions_for_correction",
    "include_speech_holes_for_correction",
    "load_valid_raw_asr_v2",
    "recover_raw_asr_v2_from_checkpoint",
    "consume_coarse_segments",
    "merge_rescue_evidence",
    "transcribe_raw_audio_v2",
    "validate_publishable_raw_asr_v2_policy",
    "validate_persisted_raw_asr_v2",
    "V2_BETA_AUDIT_VAD_SPEECH_PAD_MS",
    "V2_BETA_CORRECTION_MAX_INTERNAL_WORD_GAP_MS",
    "V2_BETA_ORPHAN_CAPTION_EVIDENCE_COVERAGE_FLOOR",
    "V2_BETA_ORPHAN_CAPTION_MIN_CHARACTER_BIGRAM_DICE",
    "V2_BETA_ORPHAN_CAPTION_MIN_LEXICAL_ASR_OVERLAP_MS",
    "V2_BETA_ORPHAN_CAPTION_MIN_SHARED_TOKENS",
    "V2_BETA_VAD_MIN_SPEECH_MS",
]
