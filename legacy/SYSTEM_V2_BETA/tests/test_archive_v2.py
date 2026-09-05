from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.archive_v2 import (
    ArchiveV2Error,
    archive_episode_v2,
    canonical_archive_paths,
    expected_cleanup_confirmation_v2,
)
from src.download import sha256_json
from src.episode_archive import file_record
from src.raw_asr_v2 import RawASRV2Config


class ArchiveV2Tests(unittest.TestCase):
    episode = 11
    episode_name = "Muhtemel Ask 11.Bolum"

    def _workspace(self, temporary: str) -> tuple[Path, Path]:
        project = Path(temporary) / "Muhtemel_Ask_Subtitles"
        root = project / "EPISODES" / self.episode_name
        for name in (
            "source",
            "prepare",
            "translation_input",
            "translation_output",
            "review",
            "final",
        ):
            (root / name).mkdir(parents=True, exist_ok=True)
        receipt = (
            project
            / "ARCHIVE_REPORTS"
            / f"{self.episode_name}_ARCHIVE_RECEIPT_V2.json"
        )
        return root, receipt

    @staticmethod
    def _srt(text: str) -> bytes:
        return f"1\n00:00:00,100 --> 00:00:00,800\n{text}\n".encode("utf-8")

    def _populate(self, root: Path, *, create_legacy_mkv: bool = True) -> dict[str, Path]:
        source = root / "source" / f"{self.episode_name}.mp4"
        id_srt = root / "final" / f"{self.episode_name}.id-final.srt"
        tr_srt = root / "final" / f"{self.episode_name}.tr-final.srt"
        id_sidecar = root / "source" / f"{self.episode_name}-id.srt"
        tr_sidecar = root / "source" / f"{self.episode_name}-tr.srt"
        legacy_mkv = root / "final" / f"{self.episode_name} - Endonezce + Turkce.mkv"
        source.write_bytes(b"source-video")
        id_srt.write_bytes(self._srt("Halo."))
        tr_srt.write_bytes(self._srt("Merhaba."))
        id_sidecar.write_bytes(id_srt.read_bytes())
        tr_sidecar.write_bytes(tr_srt.read_bytes())
        if create_legacy_mkv:
            legacy_mkv.write_bytes(b"verified-legacy-mkv")
        (root / "source" / "download.done.json").write_text("{}", encoding="utf-8")
        (root / "source" / "source.metadata.json").write_text("{}", encoding="utf-8")
        (root / "prepare" / "audio.flac").write_bytes(b"audio")
        (root / "prepare" / "schema.json").write_text("{}", encoding="utf-8")
        (root / "translation_input" / "pack.zip").write_bytes(b"pack")
        (root / "translation_output" / "translated.zip").write_bytes(b"translated")
        (root / "review" / "review.xlsx").write_bytes(b"review")
        report_path = root / "final" / f"{self.episode_name}_FINALIZATION_REPORT.json"
        outputs = {
            "id_srt": file_record(id_srt, root),
            "tr_srt": file_record(tr_srt, root),
            "id_infuse_sidecar": file_record(id_sidecar, root),
            "tr_infuse_sidecar": file_record(tr_sidecar, root),
        }
        if create_legacy_mkv:
            outputs["mkv"] = file_record(legacy_mkv, root)
        report = {
            "report_version": 1,
            "status": "PASS",
            "episode": self.episode,
            "episode_name": self.episode_name,
            "input_files": {"source_video": file_record(source, root)},
            "outputs": outputs,
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {
            "source": source,
            "id_srt": id_srt,
            "tr_srt": tr_srt,
            "id_sidecar": id_sidecar,
            "tr_sidecar": tr_sidecar,
            "legacy_mkv": legacy_mkv,
            "report": report_path,
        }

    def _populate_v2(self, root: Path) -> dict[str, Path]:
        source = root / "source" / f"{self.episode_name}.mp4"
        canonical = canonical_archive_paths(root, self.episode)
        canonical["id_srt"].parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"source-video")
        canonical["id_srt"].write_bytes(self._srt("Halo."))
        canonical["tr_srt"].write_bytes(self._srt("Merhaba."))
        canonical["mkv"].write_bytes(b"verified-canonical-mkv")
        evidence = {
            "source_video": source,
            "download_metadata": root / "source" / "source.metadata.json",
            "download_marker": root / "source" / "download.done.json",
            "audio": root / "prepare" / "audio.flac",
            "audio_metadata": root / "prepare" / "audio.metadata.json",
            "audio_marker": root / "prepare" / "audio.done.json",
            "raw_asr_v2": root / "prepare" / "raw_asr_v2.json",
            "raw_asr_v2_marker": root / "prepare" / "raw_asr_v2.done.json",
            "forced_alignment_v2": root / "prepare" / "forced_alignment_v2.json",
            "forced_alignment_v2_marker": (
                root / "prepare" / "forced_alignment_v2.done.json"
            ),
            "tr_correction_pack": (
                root
                / "translation_input"
                / f"{self.episode_name}_TR_CORRECTION_PACK.zip"
            ),
            "tr_correction_output": (
                root
                / "translation_output"
                / f"{self.episode_name}_TR_CORRECTED.zip"
            ),
            "aligned_schema_v2": root / "prepare" / "aligned_tr_schema_v2.json",
            "id_translation_pack": (
                root
                / "translation_input"
                / f"{self.episode_name}_ID_TRANSLATION_PACK.zip"
            ),
            "id_translation_zip": (
                root
                / "translation_output"
                / f"{self.episode_name}_ID_TRANSLATED.zip"
            ),
        }
        for key in (
            "download_metadata",
            "download_marker",
            "audio_metadata",
            "audio_marker",
            "raw_asr_v2_marker",
            "forced_alignment_v2_marker",
        ):
            evidence[key].write_text('{"fixture":true}', encoding="utf-8")
        evidence["audio"].write_bytes(b"audio")
        audio_sha = file_record(evidence["audio"], root)["sha256"]
        alignment_sha = "b" * 64
        schema_sha = "c" * 64
        tr_records_sha = "e" * 64
        raw_settings_object = RawASRV2Config()
        coverage_config = asdict(raw_settings_object.speech_coverage_config())
        raw_input_sha = "d" * 64
        model_settings = json.loads(json.dumps(asdict(raw_settings_object)))
        evidence["raw_asr_v2"].write_text(
            json.dumps(
                {
                    "episode": self.episode,
                    "audio_sha256": audio_sha,
                    "input_sha256": raw_input_sha,
                    "model": {"settings": model_settings},
                    "hallucination_review_utterance_uids": [],
                    "initial_speech_coverage": {"config": coverage_config},
                    "speech_coverage": {"config": coverage_config},
                }
            ),
            encoding="utf-8",
        )
        forced_zero_fields = {
            key: 0
            for key in (
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
        }
        edit_summary = {
            "unchanged_token_count": 1,
            "replaced_token_count": 0,
            "inserted_token_count": 0,
            "edited_token_count": 0,
            "edited_token_ratio": 0.0,
            "deleted_asr_token_count": 0,
            "low_score_edited_token_count": 0,
            "unreviewed_deleted_token_count": 0,
        }
        edit_detail = {
            "unchanged_token_count": 1,
            "replaced_token_count": 0,
            "inserted_token_count": 0,
            "edited_token_count": 0,
            "edited_token_ratio": 0.0,
            "deleted_asr_token_count": 0,
            "low_score_edited_token_count": 0,
            "unreviewed_deleted_token_count": 0,
        }
        forced_provenance = {
            "timing_source": "whisperx_ctc_forced_alignment",
            "engine": "whisperx",
            "whisperx_version": "fixture",
            "model_name": "fixture",
            "model_type": "fixture",
            "language": "tr",
            "device": "cpu",
            "interpolation": "disabled:none",
            "min_word_score": 0.30,
            "review_word_score": 0.55,
            "edited_token_min_word_score": 0.55,
            "max_word_duration_ms": 2500,
            "max_outward_drift_ms": 500,
        }
        forced_segment = {
            "segment_index": 1,
            "utterance_uid": "U11-000001",
            "asr_text": "Merhaba",
            "text": "Merhaba",
            "deletion_audio_reviewed": False,
            "edit_audit": edit_detail,
        }
        forced_word = {
            "word_index": 1,
            "utterance_uid": "U11-000001",
            "text": "Merhaba",
        }
        evidence["forced_alignment_v2"].write_text(
            json.dumps(
                {
                    "audio_sha256": audio_sha,
                    "alignment_sha256": alignment_sha,
                    "provenance": forced_provenance,
                    "segments": [forced_segment],
                    "words": [forced_word],
                    "report": {
                        **forced_zero_fields,
                        **edit_summary,
                        "review_alignment_score_count": 0,
                        "minimum_alignment_score": 0.90,
                        "maximum_alignment_word_duration_ms": 700,
                        "maximum_early_outward_drift_ms": 0,
                        "maximum_late_outward_drift_ms": 0,
                        "alignment_sha256": alignment_sha,
                    },
                }
            ),
            encoding="utf-8",
        )
        evidence["forced_alignment_v2_marker"].write_text(
            json.dumps(
                {
                    "details": {
                        "alignment_sha256": alignment_sha,
                        "correction_output_sha256": tr_records_sha,
                    }
                }
            ),
            encoding="utf-8",
        )
        evidence["aligned_schema_v2"].write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "schema_sha256": schema_sha,
                    "audio_sha256": audio_sha,
                    "block_count": 1,
                }
            ),
            encoding="utf-8",
        )
        for key in (
            "tr_correction_pack",
            "tr_correction_output",
            "id_translation_pack",
            "id_translation_zip",
        ):
            evidence[key].write_bytes(key.encode("utf-8"))

        project = root.parent.parent
        config_root = project / "SYSTEM_V2_BETA" / "config"
        config_root.mkdir(parents=True, exist_ok=True)
        configurations = {
            "series_config": config_root / "series.yaml",
            "names_config": config_root / "names.yaml",
            "religious_config": config_root / "religious_terms.yaml",
        }
        for key, path in configurations.items():
            path.write_text(f"fixture: {key}\n", encoding="utf-8")
        (root / "review" / "review.xlsx").write_bytes(b"review")
        report_path = (
            root / "final" / f"{self.episode_name}_FINALIZATION_REPORT_V2.json"
        )
        input_records = {
            key: file_record(path, root) for key, path in evidence.items()
        }
        configuration_records = {
            key: file_record(path, project) for key, path in configurations.items()
        }
        input_hashes = {
            key: record["sha256"]
            for key, record in {**input_records, **configuration_records}.items()
        }
        timing_zeros = {
            key: 0
            for key in (
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
        }
        semantic_zeros = {
            key: 0
            for key in (
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
        }
        coverage = {
            "report_version": "1.0",
            "status": "PASS",
            "config": coverage_config,
            "unresolved_speech_region_count": 0,
            "speech_coverage_ratio": 1.0,
            "metrics": {
                "unresolved_speech_region_count": 0,
                "unresolved_coverage_issue_count": 0,
                "word_interval_wholly_outside_speech_count": 0,
            },
        }
        raw_policy = {
            "report_version": "2.0",
            "status": "PASS",
            "policy_label": model_settings["coverage_policy_label"],
            "raw_input_sha256": raw_input_sha,
            "raw_artifact_sha256": input_records["raw_asr_v2"]["sha256"],
            "audio_sha256": audio_sha,
            "model_settings": model_settings,
            "model_settings_sha256": sha256_json(model_settings),
            "speech_coverage_config": coverage_config,
            "speech_coverage_config_sha256": sha256_json(coverage_config),
            "explicit_audio_review_uids": [],
        }
        raw_policy["raw_policy_sha256"] = sha256_json(raw_policy)
        alignment_policy = {
            "report_version": "2.0",
            "status": "PASS",
            "audio_sha256": audio_sha,
            "alignment_sha256": alignment_sha,
            "policy": forced_provenance,
            "provenance": forced_provenance,
            "provenance_sha256": sha256_json(forced_provenance),
            "hard_zero_counters": forced_zero_fields,
            "review_alignment_score_count": 0,
            "minimum_alignment_score": 0.90,
            "maximum_alignment_word_duration_ms": 700,
            "maximum_early_outward_drift_ms": 0,
            "maximum_late_outward_drift_ms": 0,
        }
        alignment_policy["alignment_policy_sha256"] = sha256_json(
            alignment_policy
        )
        alignment_edit_audit = {
            "report_version": "2.0",
            "status": "PASS",
            "alignment_sha256": alignment_sha,
            "segment_count": 1,
            "summary": edit_summary,
            "segments": [
                {
                    "segment_index": 1,
                    "utterance_uid": "U11-000001",
                    "asr_text": "Merhaba",
                    "corrected_text": "Merhaba",
                    "deletion_audio_reviewed": False,
                    "edit_audit": edit_detail,
                }
            ],
        }
        alignment_edit_audit["alignment_edit_audit_sha256"] = sha256_json(
            alignment_edit_audit
        )
        audio_review = {
            "report_version": "2.0",
            "status": "PASS",
            "reviewed_outcome_count": 0,
            "confirmed_dialogue_count": 0,
            "reviewed_non_dialogue_count": 0,
            "discarded_asr_hallucination_count": 0,
            "pending_audio_review_count": 0,
            "outcomes": [],
        }
        audio_review["audio_review_sha256"] = sha256_json(audio_review)
        strict_word_vad = {
            "report_version": "2.0",
            "status": "PASS",
            "policy": {
                "vad_source": "silero_vad",
                "vad_intervals": "audit_silero_padded_unmerged",
                "audit_vad_speech_pad_ms": 60,
                "minimum_word_vad_overlap_ratio": 0.25,
                "maximum_leading_outside_vad_ms": 120,
                "maximum_trailing_outside_vad_ms": 120,
                "maximum_contiguous_non_vad_ms": 120,
                "maximum_effective_distance_from_unpadded_vad_core_ms": 180,
                "exception": (
                    "exact_utterance_uid_and_review_bounds_with_hash_bound_"
                    "audio_reviewed_confirmed_dialogue"
                ),
            },
            "raw_asr_input_sha256": raw_input_sha,
            "audio_sha256": audio_sha,
            "alignment_sha256": alignment_sha,
            "audio_review_sha256": audio_review["audio_review_sha256"],
            "aligned_word_count": 1,
            "confirmed_dialogue_exception_word_count": 0,
            "unsafe_word_count": 0,
            "confirmed_dialogue_required_coverage_count": 0,
            "confirmed_dialogue_coverage_fail_count": 0,
            "confirmed_dialogue_exception_words": [],
            "confirmed_dialogue_required_coverage": [],
            "unsafe_words": [],
        }
        strict_word_vad["strict_word_vad_sha256"] = sha256_json(strict_word_vad)
        coverage["audio_review_v2"] = audio_review
        coverage["strict_word_vad_v2"] = strict_word_vad
        coverage_sha = sha256_json(coverage)
        evidence["aligned_schema_v2"].write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "schema_sha256": schema_sha,
                    "audio_sha256": audio_sha,
                    "speech_coverage_sha256": coverage_sha,
                    "block_count": 1,
                }
            ),
            encoding="utf-8",
        )
        input_records["aligned_schema_v2"] = file_record(
            evidence["aligned_schema_v2"], root
        )
        input_hashes["aligned_schema_v2"] = input_records["aligned_schema_v2"][
            "sha256"
        ]
        streams = self._stream_hashes()
        report = {
            "report_version": 2,
            "status": "PASS",
            "episode": self.episode,
            "episode_name": self.episode_name,
            "schema_version": "2.0",
            "schema_sha256": schema_sha,
            "audio_sha256": audio_sha,
            "alignment_sha256": alignment_sha,
            "speech_coverage_sha256": coverage_sha,
            "block_count": 1,
            "input_files": input_records,
            "configuration_files": configuration_records,
            "input_hashes": input_hashes,
            "evidence_validation": {
                "source_audio_chain_exact": True,
                "full_forced_alignment_revalidated": True,
                "bounded_audio_review_revalidated": True,
                "bounded_audio_review_sha256": "a" * 64,
                "speech_coverage_recomputed": True,
                "schema_rebuilt_exact": True,
                "semantic_qa_config_hash_bound": True,
                "evidence_rehashed_immediately_before_pass": True,
            },
            "tr_correction_validation": {
                "output_sha256": input_records["tr_correction_output"]["sha256"],
                "records_sha256": tr_records_sha,
                "record_count": 1,
                "review_required_count": 0,
                "audio_reviewed_count": 0,
                "pending_audio_review_count": 0,
                "confirmed_dialogue_count": 0,
                "reviewed_non_dialogue_count": 0,
                "discarded_asr_hallucination_count": 0,
            },
            "id_translation_validation": {
                "passed": True,
                "schema_sha256": schema_sha,
                "expected_block_count": 1,
                "output_block_count": 1,
                "missing_block_count": 0,
                "duplicate_block_count": 0,
                "review_required_count": 0,
            },
            "raw_policy_v2": raw_policy,
            "alignment_policy_v2": alignment_policy,
            "alignment_edit_audit_v2": alignment_edit_audit,
            "audio_review_v2": audio_review,
            "strict_word_vad_v2": strict_word_vad,
            "speech_coverage_v2": coverage,
            "timing_qa_v2": {
                "report_version": 2,
                "passed": True,
                "block_count": 1,
                "alignment_sha256": alignment_sha,
                "config": {},
                "review_alignment_score_count": 0,
                **timing_zeros,
            },
            "semantic_subtitle_qa": {
                "passed": True,
                "tr_block_count": 1,
                "id_block_count": 1,
                "timings_identical": True,
                **semantic_zeros,
            },
            "timing_identity": {
                "forced_alignment": True,
                "independent_speech_coverage": True,
                "tr_id_identical": True,
                "srt_roundtrip_exact": True,
            },
            "outputs": {
                "mkv": file_record(canonical["mkv"], root),
                "id_srt": file_record(canonical["id_srt"], root),
                "tr_srt": file_record(canonical["tr_srt"], root),
            },
            "mkv_verification": {
                "verified": True,
                "video_audio_stream_copy": True,
                "subtitle_order": ["ind", "tur"],
                "indonesian_default": True,
                "turkish_default": False,
                "roundtrip": {
                    "exact": True,
                    "id_block_count": 1,
                    "tr_block_count": 1,
                },
                "stream_hashes": {
                    "checked": True,
                    "match": True,
                    "source": streams,
                    "output": streams,
                },
                "output_path": canonical["mkv"].relative_to(root).as_posix(),
                "output_size_bytes": canonical["mkv"].stat().st_size,
            },
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {
            "source": source,
            "report": report_path,
            **canonical,
        }

    @staticmethod
    def _stream_hashes() -> list[dict[str, object]]:
        return [
            {
                "type": "video",
                "ordinal": 0,
                "algorithm": "sha256",
                "digest": "a" * 64,
            }
        ]

    @classmethod
    def _mux_report(cls, output: Path) -> dict[str, object]:
        hashes = cls._stream_hashes()
        return {
            "verified": True,
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": {"exact": True, "id_block_count": 1, "tr_block_count": 1},
            "stream_hashes": {
                "checked": True,
                "match": True,
                "source": hashes,
                "output": hashes,
            },
        }

    def _fake_mux(self, source: Path, id_srt: Path, tr_srt: Path, output: Path, **_: object):
        output.write_bytes(b"new-stream-copy-mkv")
        return self._mux_report(output)

    def _media_patches(self):
        return (
            patch("src.archive_v2.compute_av_stream_hashes", return_value=self._stream_hashes()),
            patch(
                "src.archive_v2.verify_mkv_roundtrip",
                return_value={"exact": True, "id_block_count": 1, "tr_block_count": 1},
            ),
            patch("src.archive_v2.mux_softsubs", side_effect=self._fake_mux),
        )

    def test_confirmation_and_canonical_paths_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self._workspace(temporary)
            paths = canonical_archive_paths(root, self.episode)
            self.assertEqual(paths["mkv"], root.resolve() / "final" / f"{self.episode_name}.mkv")
            self.assertEqual(
                paths["id_srt"],
                root.resolve() / "final" / "subtitles" / f"{self.episode_name}-id.srt",
            )
        self.assertEqual(
            expected_cleanup_confirmation_v2(self.episode),
            "DELETE EPISODE 11 INTERMEDIATES",
        )

    def test_full_cleanup_keeps_exactly_one_mkv_and_two_srts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch as mocked_mux:
                result = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    confirmation=expected_cleanup_confirmation_v2(self.episode),
                )
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(mocked_mux.called, "verified legacy MKV should be renamed, not remuxed")
            canonical = canonical_archive_paths(root, self.episode)
            remaining = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                remaining,
                sorted(path.relative_to(root).as_posix() for path in canonical.values()),
            )
            self.assertFalse(paths["source"].exists())
            self.assertFalse(paths["id_sidecar"].exists())
            self.assertFalse(paths["tr_sidecar"].exists())
            self.assertTrue(receipt.is_file())
            persisted = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(persisted["receipt_version"], 2)
            self.assertEqual(persisted["status"], "PASS")

    def test_v2_report_uses_canonical_outputs_without_copy_or_remux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate_v2(root)
            original = {
                key: value.read_bytes() for key, value in paths.items() if key in {"mkv", "id_srt", "tr_srt"}
            }
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch as mocked_mux:
                ready = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertEqual(ready["status"], "READY")
            self.assertFalse(mocked_mux.called)
            self.assertEqual(
                set(ready["legacy_inputs"]),
                {"source_video", "finalization_report"},
            )
            self.assertEqual(
                original,
                {key: paths[key].read_bytes() for key in original},
            )
            self.assertTrue(paths["source"].is_file())
            self.assertTrue(paths["report"].is_file())

    def test_v2_specific_report_is_preferred_without_overwriting_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate_v2(root)
            v1_report = root / "final" / f"{self.episode_name}_FINALIZATION_REPORT.json"
            v1_report.write_text(
                json.dumps(
                    {
                        "report_version": 1,
                        "status": "FAIL",
                        "episode": self.episode,
                        "episode_name": self.episode_name,
                    }
                ),
                encoding="utf-8",
            )
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                ready = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertEqual(ready["status"], "READY")
            self.assertTrue(paths["report"].is_file())
            self.assertTrue(v1_report.is_file())

    def test_invalid_v2_report_never_falls_back_to_valid_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            v1_paths = self._populate(root)
            v2_report = (
                root / "final" / f"{self.episode_name}_FINALIZATION_REPORT_V2.json"
            )
            v2_report.write_text('{"report_version":2,"status":"FAIL"}', encoding="utf-8")
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                with self.assertRaisesRegex(
                    ArchiveV2Error, "not a supported PASS report"
                ):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                    )
            self.assertTrue(v1_paths["report"].is_file())
            self.assertTrue(v2_report.is_file())
            self.assertFalse(receipt.exists())

    def test_report_version_must_match_its_filename(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root, receipt = self._workspace(temporary)
                if version == 1:
                    paths = self._populate_v2(root)
                    report_path = paths["report"]
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report["report_version"] = 1
                    expected = "v1 report must use the legacy filename"
                else:
                    paths = self._populate_v2(root)
                    v2_path = paths["report"]
                    report = json.loads(v2_path.read_text(encoding="utf-8"))
                    v2_path.unlink()
                    report_path = (
                        root / "final" / f"{self.episode_name}_FINALIZATION_REPORT.json"
                    )
                    expected = "v2 report must use the V2-specific filename"
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ArchiveV2Error, expected):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                    )

    def test_v2_report_cleanup_retains_exactly_canonical_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate_v2(root)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch as mocked_mux:
                result = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    confirmation=expected_cleanup_confirmation_v2(self.episode),
                )
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(mocked_mux.called)
            self.assertFalse(paths["source"].exists())
            self.assertFalse(paths["report"].exists())
            self.assertEqual(
                sorted(
                    path.relative_to(root).as_posix()
                    for path in root.rglob("*")
                    if path.is_file()
                ),
                sorted(
                    path.relative_to(root).as_posix()
                    for key, path in paths.items()
                    if key in {"mkv", "id_srt", "tr_srt"}
                ),
            )

    def test_v2_report_rejects_legacy_paths_sidecars_and_missing_mkv(self) -> None:
        for mutation in ("legacy-srt", "sidecar-key", "missing-mkv"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, receipt = self._workspace(temporary)
                paths = self._populate_v2(root)
                report = json.loads(paths["report"].read_text(encoding="utf-8"))
                if mutation == "legacy-srt":
                    legacy = root / "final" / f"{self.episode_name}.id-final.srt"
                    legacy.write_bytes(paths["id_srt"].read_bytes())
                    report["outputs"]["id_srt"] = file_record(legacy, root)
                    expected = "path mismatch"
                elif mutation == "sidecar-key":
                    sidecar = root / "source" / f"{self.episode_name}-id.srt"
                    sidecar.write_bytes(paths["id_srt"].read_bytes())
                    report["outputs"]["id_infuse_sidecar"] = file_record(sidecar, root)
                    expected = "outputs key mismatch"
                else:
                    del report["outputs"]["mkv"]
                    expected = "outputs key mismatch"
                paths["report"].write_text(json.dumps(report), encoding="utf-8")
                compute_patch, roundtrip_patch, mux_patch = self._media_patches()
                with compute_patch, roundtrip_patch, mux_patch:
                    with self.assertRaisesRegex(ArchiveV2Error, expected):
                        archive_episode_v2(
                            episode_root=root,
                            receipt_path=receipt,
                            episode=self.episode,
                        )
                self.assertTrue(paths["source"].is_file())
                self.assertFalse(receipt.exists())

    def test_v2_report_hard_gates_cannot_be_missing_nonzero_or_unbound(self) -> None:
        mutations = (
            "missing-timing-counter",
            "nonzero-timing-counter",
            "semantic-review",
            "coverage-digest",
            "input-hash",
            "evidence-flag",
            "id-review",
            "mux-stream-copy",
            "unsafe-alignment-policy",
            "missing-raw-policy",
            "raw-policy-digest",
            "alignment-hard-counter",
            "edit-audit-digest",
            "audio-review-pending",
            "strict-word-vad-unsafe",
            "coverage-audit-copy",
            "tr-records-digest",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, receipt = self._workspace(temporary)
                paths = self._populate_v2(root)
                report = json.loads(paths["report"].read_text(encoding="utf-8"))
                if mutation == "missing-timing-counter":
                    del report["timing_qa_v2"]["missing_id_count"]
                    expected = "missing_id_count"
                elif mutation == "nonzero-timing-counter":
                    report["timing_qa_v2"]["overlap_count"] = 1
                    expected = "overlap_count must be zero"
                elif mutation == "semantic-review":
                    report["semantic_subtitle_qa"]["review_required_count"] = 1
                    expected = "review_required_count must be zero"
                elif mutation == "coverage-digest":
                    report["speech_coverage_sha256"] = "f" * 64
                    expected = "aligned schema identity mismatch"
                elif mutation == "input-hash":
                    report["input_hashes"]["audio"] = "f" * 64
                    expected = "does not bind its file record"
                elif mutation == "evidence-flag":
                    report["evidence_validation"]["schema_rebuilt_exact"] = False
                    expected = "schema_rebuilt_exact must be true"
                elif mutation == "id-review":
                    report["id_translation_validation"]["review_required_count"] = 1
                    expected = "review_required_count must be zero"
                elif mutation == "mux-stream-copy":
                    report["mkv_verification"]["video_audio_stream_copy"] = False
                    expected = "MKV verification is incomplete"
                elif mutation == "unsafe-alignment-policy":
                    forced_path = root / "prepare" / "forced_alignment_v2.json"
                    forced = json.loads(forced_path.read_text(encoding="utf-8"))
                    forced["provenance"]["min_word_score"] = 0.20
                    forced_path.write_text(json.dumps(forced), encoding="utf-8")
                    record = file_record(forced_path, root)
                    report["input_files"]["forced_alignment_v2"] = record
                    report["input_hashes"]["forced_alignment_v2"] = record["sha256"]
                    expected = "alignment policy/provenance mismatch"
                elif mutation == "missing-raw-policy":
                    del report["raw_policy_v2"]
                    expected = "raw_policy_v2 must be an object"
                elif mutation == "raw-policy-digest":
                    report["raw_policy_v2"]["raw_policy_sha256"] = "f" * 64
                    expected = "raw_policy_v2 self-digest mismatch"
                elif mutation == "alignment-hard-counter":
                    report["alignment_policy_v2"]["hard_zero_counters"][
                        "unaligned_word_count"
                    ] = 1
                    report["alignment_policy_v2"].pop("alignment_policy_sha256")
                    report["alignment_policy_v2"]["alignment_policy_sha256"] = (
                        sha256_json(report["alignment_policy_v2"])
                    )
                    expected = "unaligned_word_count must be zero"
                elif mutation == "edit-audit-digest":
                    report["alignment_edit_audit_v2"][
                        "alignment_edit_audit_sha256"
                    ] = "f" * 64
                    expected = "alignment_edit_audit_v2 self-digest mismatch"
                elif mutation == "audio-review-pending":
                    report["audio_review_v2"]["pending_audio_review_count"] = 1
                    report["audio_review_v2"].pop("audio_review_sha256")
                    report["audio_review_v2"]["audio_review_sha256"] = sha256_json(
                        report["audio_review_v2"]
                    )
                    expected = "audio-review counts/pass mismatch"
                elif mutation == "strict-word-vad-unsafe":
                    report["strict_word_vad_v2"]["unsafe_word_count"] = 1
                    report["strict_word_vad_v2"].pop("strict_word_vad_sha256")
                    report["strict_word_vad_v2"]["strict_word_vad_sha256"] = (
                        sha256_json(report["strict_word_vad_v2"])
                    )
                    expected = "strict word/VAD hard gate mismatch"
                elif mutation == "coverage-audit-copy":
                    report["speech_coverage_v2"]["audio_review_v2"] = {}
                    report["speech_coverage_sha256"] = sha256_json(
                        report["speech_coverage_v2"]
                    )
                    schema_path = root / "prepare" / "aligned_tr_schema_v2.json"
                    schema = json.loads(schema_path.read_text(encoding="utf-8"))
                    schema["speech_coverage_sha256"] = report[
                        "speech_coverage_sha256"
                    ]
                    schema_path.write_text(json.dumps(schema), encoding="utf-8")
                    record = file_record(schema_path, root)
                    report["input_files"]["aligned_schema_v2"] = record
                    report["input_hashes"]["aligned_schema_v2"] = record["sha256"]
                    expected = "coverage audit copies are not exact"
                else:
                    report["tr_correction_validation"]["records_sha256"] = "f" * 64
                    expected = "not forced-marker bound"
                paths["report"].write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ArchiveV2Error, expected):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                    )
                self.assertFalse(receipt.exists())

    def test_alignment_review_score_warning_is_allowed_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate_v2(root)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            forced_path = root / "prepare" / "forced_alignment_v2.json"
            forced = json.loads(forced_path.read_text(encoding="utf-8"))
            forced["report"]["review_alignment_score_count"] = 2
            forced_path.write_text(json.dumps(forced), encoding="utf-8")
            record = file_record(forced_path, root)
            report["input_files"]["forced_alignment_v2"] = record
            report["input_hashes"]["forced_alignment_v2"] = record["sha256"]
            report["timing_qa_v2"]["review_alignment_score_count"] = 2
            report["alignment_policy_v2"]["review_alignment_score_count"] = 2
            report["alignment_policy_v2"].pop("alignment_policy_sha256")
            report["alignment_policy_v2"]["alignment_policy_sha256"] = sha256_json(
                report["alignment_policy_v2"]
            )
            paths["report"].write_text(json.dumps(report), encoding="utf-8")

            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                ready = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertEqual(ready["status"], "READY")
            self.assertEqual(
                ready["pilot_warnings"],
                [
                    {
                        "code": "alignment_score_pilot_review",
                        "count": 2,
                        "message": (
                            "2 aligned word(s) scored from 0.30 through 0.549 "
                            "and require Episode 10 pilot spot-checking."
                        ),
                    }
                ],
            )
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["pilot_warnings"],
                ready["pilot_warnings"],
            )

    def test_without_legacy_mkv_uses_verified_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            self._populate(root, create_legacy_mkv=False)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch as mocked_mux:
                result = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertEqual(result["status"], "READY")
            self.assertEqual(mocked_mux.call_count, 1)
            self.assertTrue(canonical_archive_paths(root, self.episode)["mkv"].is_file())
            self.assertTrue((root / "prepare" / "audio.flac").is_file())

    def test_resume_after_verified_legacy_rename_before_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            canonical = canonical_archive_paths(root, self.episode)["mkv"]
            paths["legacy_mkv"].replace(canonical)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch as mocked_mux:
                result = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertEqual(result["status"], "READY")
            self.assertFalse(mocked_mux.called)
            self.assertTrue(canonical.is_file())
            self.assertTrue(receipt.is_file())

    def test_verification_failure_never_removes_source_work_or_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with (
                patch("src.archive_v2.compute_av_stream_hashes", return_value=self._stream_hashes()),
                patch("src.archive_v2.verify_mkv_roundtrip", return_value={"exact": False}),
            ):
                with self.assertRaisesRegex(ArchiveV2Error, "verification failed"):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=True,
                        confirmation=expected_cleanup_confirmation_v2(self.episode),
                    )
            self.assertTrue(paths["source"].is_file())
            self.assertTrue(paths["id_sidecar"].is_file())
            self.assertTrue((root / "prepare" / "audio.flac").is_file())
            self.assertFalse(receipt.exists())

    def test_unexpected_video_is_a_hard_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            extra = root / "source" / "bonus.mp4"
            extra.write_bytes(b"unexpected")
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                with self.assertRaisesRegex(ArchiveV2Error, "Unexpected video"):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=True,
                        confirmation=expected_cleanup_confirmation_v2(self.episode),
                    )
            self.assertTrue(extra.is_file())
            self.assertTrue(paths["source"].is_file())
            self.assertFalse(receipt.exists())

    def test_whitelist_and_symlink_defenses_stop_before_cleanup(self) -> None:
        for mutation in ("unknown-suffix", "symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, receipt = self._workspace(temporary)
                paths = self._populate(root)
                if mutation == "unknown-suffix":
                    (root / "prepare" / "notes.txt").write_text("mine", encoding="utf-8")
                    expected = "outside the cleanup whitelist"
                else:
                    outside = Path(temporary) / "outside.json"
                    outside.write_text("{}", encoding="utf-8")
                    (root / "prepare" / "linked.json").symlink_to(outside)
                    expected = "descendant symlink"
                compute_patch, roundtrip_patch, mux_patch = self._media_patches()
                with compute_patch, roundtrip_patch, mux_patch:
                    with self.assertRaisesRegex(ArchiveV2Error, expected):
                        archive_episode_v2(
                            episode_root=root,
                            receipt_path=receipt,
                            episode=self.episode,
                        )
                self.assertTrue(paths["source"].is_file())
                self.assertFalse(receipt.exists())

    def test_wrong_confirmation_and_tree_mutation_delete_nothing(self) -> None:
        for mutation in ("wrong", "tree-change"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, receipt = self._workspace(temporary)
                paths = self._populate(root)
                if mutation == "wrong":
                    confirmation: object = "DELETE SOMETHING ELSE"
                else:
                    def confirmation(expected: str) -> str:
                        (root / "prepare" / "appeared.json").write_text("{}", encoding="utf-8")
                        return expected

                compute_patch, roundtrip_patch, mux_patch = self._media_patches()
                with compute_patch, roundtrip_patch, mux_patch:
                    with self.assertRaises(ArchiveV2Error):
                        archive_episode_v2(
                            episode_root=root,
                            receipt_path=receipt,
                            episode=self.episode,
                            delete_intermediates=True,
                            confirmation=confirmation,  # type: ignore[arg-type]
                        )
                self.assertTrue(paths["source"].is_file())
                self.assertTrue(paths["id_sidecar"].is_file())
                self.assertTrue((root / "prepare" / "audio.flac").is_file())
                self.assertEqual(
                    json.loads(receipt.read_text(encoding="utf-8"))["status"], "READY"
                )

    def test_ready_receipt_resumes_a_partial_exact_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                ready = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertEqual(ready["status"], "READY")
            # Simulate an interruption after two records from the exact plan were removed.
            paths["report"].unlink()
            (root / "prepare" / "schema.json").unlink()
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                result = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    confirmation=expected_cleanup_confirmation_v2(self.episode),
                )
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(paths["source"].exists())
            self.assertEqual(
                len([path for path in root.rglob("*") if path.is_file()]), 3
            )

    def test_pass_receipt_is_idempotent_and_rejects_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            self._populate(root)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    confirmation=expected_cleanup_confirmation_v2(self.episode),
                )
                rerun = archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            self.assertTrue(rerun["no_op"])

            (root / "prepare").mkdir()
            (root / "prepare" / "new.json").write_text("{}", encoding="utf-8")
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                with self.assertRaisesRegex(ArchiveV2Error, "tree changed"):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                    )

    def test_tampered_canonical_file_blocks_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                archive_episode_v2(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                )
            canonical_archive_paths(root, self.episode)["tr_srt"].write_bytes(b"tampered")
            compute_patch, roundtrip_patch, mux_patch = self._media_patches()
            with compute_patch, roundtrip_patch, mux_patch:
                with self.assertRaisesRegex(ArchiveV2Error, "mismatch"):
                    archive_episode_v2(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=True,
                        confirmation=expected_cleanup_confirmation_v2(self.episode),
                    )
            self.assertTrue(paths["source"].is_file())
            self.assertTrue((root / "prepare" / "audio.flac").is_file())


if __name__ == "__main__":
    unittest.main()
