from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mas.engine.audio_review import resolve_tr_audio_reviews_v2
from mas.engine.archive import _load_finalization_inputs
from mas.engine.download import atomic_write_json, sha256_file, sha256_json, write_stage_marker
from mas.engine.finalize import FinalizationV2Error, finalize_episode_v2
from mas.engine.forced_align import _alignment_sha256, align_corrected_segments
from mas.engine.id_translation import (
    build_id_translation_records,
    create_id_translation_output_zip,
)
from mas.engine.media import AUDIO_ALIGNMENT_VERSION
from mas.engine import raw_asr as raw_asr_module
from mas.engine.raw_asr import RawASRV2Config
from mas.engine.srt import parse_srt
from mas.engine.subtitle_qa import SubtitleQAError
from mas.engine.timing_qa import TimingQAV2Error
from mas.engine.tr_correction import (
    create_tr_correction_output,
    create_tr_correction_pack,
    validate_tr_correction_output,
)
from mas.engine.workflow import (
    build_strict_v2_artifacts,
    correction_records_to_alignment_inputs,
    create_v2_id_translation_pack,
)


AUDIO_BYTES = b"canonical-test-audio"


class _FakeWhisperX:
    __version__ = "3.8.6"

    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)

    def load_audio(self, path: str) -> object:
        return object()

    def load_align_model(
        self, *, language_code: str, device: str, model_name: str
    ) -> tuple[object, dict[str, str]]:
        return object(), {"language": language_code, "type": "huggingface"}

    def align(
        self,
        transcript: list[dict],
        model: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
    ) -> dict:
        return self.results.pop(0)


def _alignment_result(words: list[dict]) -> dict:
    return {
        "segments": [{"text": "unused", "words": words}],
        "word_segments": words,
    }


class FinalizeV2Tests(unittest.TestCase):
    episode = 12
    episode_name = "Muhtemel Ask 12.Bolum"

    @staticmethod
    def _json(path: Path, value: dict) -> None:
        atomic_write_json(path, value)

    @staticmethod
    def _utterances() -> list[dict]:
        return [
            {
                "utterance_uid": "utt-1",
                "utterance_index": 1,
                "coarse_start_ms": 1_000,
                "coarse_end_ms": 2_500,
                "asr_text": "Defne 12 geldi.",
                "youtube_text": "Defne 12 geldi.",
                "context_before": "",
                "context_after": "Bugün nasılsın?",
                "risk_flags": [],
                "asr_audit": {
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.0,
                    "temperature": 0.0,
                },
            },
            {
                "utterance_uid": "utt-2",
                "utterance_index": 2,
                "coarse_start_ms": 4_500,
                "coarse_end_ms": 6_000,
                "asr_text": "Bugun nasilsin?",
                "youtube_text": "Bugün nasılsın?",
                "context_before": "Defne 12 geldi.",
                "context_after": "",
                "risk_flags": [],
                "asr_audit": {
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.0,
                    "temperature": 0.0,
                },
            },
        ]

    @classmethod
    def _corrections(cls) -> list[dict]:
        records = copy.deepcopy(cls._utterances())
        records[0].update(
            {
                "tr_corrected": "Defne 12 geldi.",
                "non_dialogue": False,
                "review_required": False,
                "audio_reviewed": False,
                "review_disposition": "not_applicable",
                "note": "",
            }
        )
        records[1].update(
            {
                "tr_corrected": "Bugün nasılsın?",
                "non_dialogue": False,
                "review_required": False,
                "audio_reviewed": False,
                "review_disposition": "not_applicable",
                "note": "",
            }
        )
        return records

    def _workspace(self, temporary: str, project_name="Muhtemel_Ask_Subtitles") -> tuple[Path, dict[str, Path]]:
        project = Path(temporary) / project_name
        checkout = patch("mas.engine.finalize.ROOT", project)
        checkout.start()
        self.addCleanup(checkout.stop)
        root = project / "EPISODES" / self.episode_name
        for name in (
            "source",
            "prepare",
            "translation_input",
            "translation_output",
        ):
            (root / name).mkdir(parents=True, exist_ok=True)
        config = project / "config" / "production"
        config.mkdir(parents=True)
        paths = {
            "source": root / "source" / f"{self.episode_name}.mp4",
            "download_metadata": root / "source" / "source.metadata.json",
            "download_marker": root / "source" / "download.done.json",
            "audio": root / "prepare" / "audio.flac",
            "audio_metadata": root / "prepare" / "audio.metadata.json",
            "audio_marker": root / "prepare" / "audio.done.json",
            "raw": root / "prepare" / "raw_asr_v2.json",
            "raw_marker": root / "prepare" / "raw_asr_v2.done.json",
            "forced": root / "prepare" / "forced_alignment_v2.json",
            "forced_marker": root / "prepare" / "forced_alignment_v2.done.json",
            "schema": root / "prepare" / "aligned_tr_schema_v2.json",
            "tr_pack": (
                root
                / "translation_input"
                / f"{self.episode_name}_TR_CORRECTION_PACK.zip"
            ),
            "tr_text_output": (
                root
                / "translation_output"
                / f"{self.episode_name}_TR_TEXT_CORRECTED.zip"
            ),
            "tr_output": (
                root
                / "translation_output"
                / f"{self.episode_name}_TR_CORRECTED.zip"
            ),
            "audio_review": root / "prepare" / "audio_review_v2.json",
            "audio_review_recovery": (
                root / "prepare" / "audio_review_v2.recovery.json"
            ),
            "id_pack": (
                root
                / "translation_input"
                / f"{self.episode_name}_ID_TRANSLATION_PACK.zip"
            ),
            "translated": (
                root
                / "translation_output"
                / f"{self.episode_name}_ID_TRANSLATED.zip"
            ),
            "series_config": config / "series.yaml",
            "names_config": config / "names.yaml",
            "religious_config": config / "religious_terms.yaml",
        }
        paths["source"].write_bytes(b"synthetic-source-video")
        paths["audio"].write_bytes(AUDIO_BYTES)
        paths["series_config"].write_text(
            "subtitle:\n"
            "  target_chars_per_line: 42\n"
            "  qa_max_chars_per_line: 84\n"
            "  preferred_max_cps: 20.0\n",
            encoding="utf-8",
        )
        paths["names_config"].write_text(
            "canonical_names:\n  - Defne\n",
            encoding="utf-8",
        )
        paths["religious_config"].write_text(
            "terms: []\nallah_only_semoga_is_invalid: true\n",
            encoding="utf-8",
        )
        return root, paths

    def _source_audio_chain(self, paths: dict[str, Path]) -> tuple[str, str]:
        source_sha = sha256_file(paths["source"])
        source_metadata = {
            "path": str(paths["source"].resolve()),
            "file_name": paths["source"].name,
            "source_file": paths["source"].name,
            "file_size_bytes": paths["source"].stat().st_size,
            "sha256": source_sha,
        }
        self._json(paths["download_metadata"], source_metadata)
        write_stage_marker(
            paths["download_marker"],
            stage="download",
            input_sha256="d" * 64,
            outputs={
                "video": paths["source"],
                "metadata": paths["download_metadata"],
            },
            details={"source_sha256": source_sha},
        )

        audio_sha = sha256_file(paths["audio"])
        timeline = {
            "alignment_version": AUDIO_ALIGNMENT_VERSION,
            "target_duration_ms": 7_000,
            "audio_offset_ms": 0,
        }
        settings = {
            "source_sha256": source_sha,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "codec": "flac",
            "compression_level": 8,
            "audio_stream": "0:a:0",
            "timeline_alignment_version": AUDIO_ALIGNMENT_VERSION,
        }
        audio_metadata = {
            "path": str(paths["audio"].resolve()),
            "file_name": paths["audio"].name,
            "sha256": audio_sha,
            "source_path": str(paths["source"].resolve()),
            "source_sha256": source_sha,
            "audio_codec": "flac",
            "duration_ms": 7_000,
            "streams": [
                {
                    "codec_type": "audio",
                    "channels": 1,
                    "sample_rate_hz": 16_000,
                }
            ],
            "source_timeline": timeline,
            "extraction_settings": settings,
        }
        self._json(paths["audio_metadata"], audio_metadata)
        write_stage_marker(
            paths["audio_marker"],
            stage="audio",
            input_sha256=sha256_json(settings),
            outputs={
                "audio": paths["audio"],
                "metadata": paths["audio_metadata"],
            },
            details={
                "source_sha256": source_sha,
                "audio_sha256": audio_sha,
                "duration_ms": 7_000,
                "source_timeline": timeline,
                "settings": settings,
            },
        )
        return source_sha, audio_sha

    def _raw(
        self,
        paths: dict[str, Path],
        audio_sha: str,
        *,
        utterances: list[dict] | None = None,
        vad_regions: list[dict] | None = None,
    ) -> dict:
        coverage_config = asdict(RawASRV2Config().speech_coverage_config())
        trusted_utterances = copy.deepcopy(
            utterances if utterances is not None else self._utterances()
        )
        segments = [
            {
                "segment_id": f"main-{index}",
                "start_ms": item["coarse_start_ms"],
                "end_ms": item["coarse_end_ms"],
                "text": item["asr_text"],
                "source": "main",
                "word_timing_complete": True,
                "words": [
                    {
                        "start_ms": item["coarse_start_ms"],
                        "end_ms": item["coarse_end_ms"],
                        "text": f" {item['asr_text']}",
                    }
                ],
            }
            for index, item in enumerate(trusted_utterances, start=1)
        ]
        trusted_vad = vad_regions or [
            {
                "vad_region_index": 1,
                "start_ms": 1_000,
                "end_ms": 2_500,
                "source": "silero_vad",
            },
            {
                "vad_region_index": 2,
                "start_ms": 4_500,
                "end_ms": 6_000,
                "source": "silero_vad",
            },
        ]
        words = [
            {**copy.deepcopy(word), "segment_id": segment["segment_id"]}
            for segment in segments
            for word in segment["words"]
        ]
        settings = RawASRV2Config()
        initial = raw_asr_module._analyze_raw_speech_coverage(
            trusted_vad,
            segments,
            words,
            config=settings.speech_coverage_config(),
        )
        batches, budget = raw_asr_module._plan_rescue_batches(
            initial, trusted_vad, segments, words, settings
        )
        return {
            "format_version": "2.0",
            "status": "completed",
            "input_sha256": "e" * 64,
            "episode": self.episode,
            "audio_path": str(paths["audio"].resolve()),
            "audio_sha256": audio_sha,
            "language": "tr",
            "segments": segments,
            "words": words,
            "required_acoustic_review_regions": (
                raw_asr_module.required_acoustic_review_regions(
                    segments, trusted_vad
                )
            ),
            "synthetic_timing_count": 0,
            "vad_fallback_reason": None,
            "independent_vad": True,
            "model": {"settings": asdict(RawASRV2Config())},
            "hallucination_review_utterance_uids": [],
            "vad_regions": trusted_vad,
            "initial_speech_coverage": initial,
            "rescue_batches": batches,
            "rescue_budget_audit": budget,
            "speech_coverage": {"config": coverage_config},
            "youtube_captions": [],
            "correction_utterances": trusted_utterances,
            "speech_hole_records": [],
            "asr_hallucination_records": [],
        }

    @staticmethod
    def _fake_whisperx() -> _FakeWhisperX:
        return _FakeWhisperX(
            [
                _alignment_result(
                    [
                        {"word": "Defne", "start": 1.08, "end": 1.35, "score": 0.96},
                        {"word": "12", "start": 1.42, "end": 1.55, "score": 0.97},
                        {"word": "geldi.", "start": 1.62, "end": 2.35, "score": 0.95},
                    ]
                ),
                _alignment_result(
                    [
                        {"word": "Bugün", "start": 4.58, "end": 5.05, "score": 0.94},
                        {"word": "nasılsın?", "start": 5.12, "end": 5.85, "score": 0.95},
                    ]
                ),
            ]
        )

    def _replace_id_output(
        self,
        paths: dict[str, Path],
        schema: dict,
        id_texts: list[str],
        *,
        review_required: bool = False,
    ) -> None:
        records = build_id_translation_records(schema)
        for record, id_text in zip(records, id_texts):
            record["id_final"] = id_text
            record["review_required"] = review_required
            record["note"] = "manual check" if review_required else ""
        manifest = create_v2_id_translation_pack(
            build_strict_v2_artifacts(
                self._loaded(paths["raw"]),
                validate_tr_correction_output(
                    paths["tr_pack"], paths["tr_output"]
                ).records,
                self._loaded(paths["forced"]),
                episode=self.episode,
            ),
            paths["id_pack"],
        )
        create_id_translation_output_zip(
            schema,
            records,
            paths["translated"],
            input_manifest=manifest,
        )

    @staticmethod
    def _loaded(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _inputs(
        self,
        root: Path,
        paths: dict[str, Path],
        *,
        id_texts: list[str] | None = None,
        utterances: list[dict] | None = None,
        corrections: list[dict] | None = None,
        vad_regions: list[dict] | None = None,
        whisperx_module: _FakeWhisperX | None = None,
    ) -> dict:
        _, audio_sha = self._source_audio_chain(paths)
        raw = self._raw(
            paths,
            audio_sha,
            utterances=utterances,
            vad_regions=vad_regions,
        )
        self._json(paths["raw"], raw)
        write_stage_marker(
            paths["raw_marker"],
            stage="raw_asr_v2",
            input_sha256=raw["input_sha256"],
            outputs={"raw_asr_v2": paths["raw"]},
            details={"audio_sha256": audio_sha, "independent_vad": True},
        )

        create_tr_correction_pack(
            raw["correction_utterances"],
            raw["speech_hole_records"],
            paths["tr_pack"],
            episode=self.episode,
            asr_hallucination_records=raw["asr_hallucination_records"],
        )
        create_tr_correction_output(
            paths["tr_pack"],
            corrections if corrections is not None else self._corrections(),
            paths["tr_text_output"],
        )
        resolve_tr_audio_reviews_v2(
            paths["tr_pack"],
            paths["tr_text_output"],
            paths["tr_output"],
            paths["audio_review"],
            paths["audio_review_recovery"],
            progress=None,
        )
        correction_output = validate_tr_correction_output(
            paths["tr_pack"], paths["tr_output"]
        )
        preparation = correction_records_to_alignment_inputs(
            raw["correction_utterances"],
            correction_output.records,
            asr_hallucination_records=raw["asr_hallucination_records"],
        )
        forced = align_corrected_segments(
            paths["audio"],
            preparation.alignment_inputs,
            whisperx_module=(
                whisperx_module
                if whisperx_module is not None
                else self._fake_whisperx()
            ),
            device="cpu",
        )
        self._json(paths["forced"], forced)
        write_stage_marker(
            paths["forced_marker"],
            stage="forced_alignment_v2",
            input_sha256="f" * 64,
            outputs={"forced_alignment_v2": paths["forced"]},
            details={
                "alignment_sha256": forced["alignment_sha256"],
                "correction_output_sha256": correction_output.output_sha256,
            },
        )
        artifacts = build_strict_v2_artifacts(
            raw,
            correction_output.records,
            forced,
            episode=self.episode,
        )
        self._json(paths["schema"], artifacts.schema)
        manifest = create_v2_id_translation_pack(
            artifacts, paths["id_pack"]
        )
        records = build_id_translation_records(artifacts.schema)
        translations = id_texts or ["Defne 12 datang.", "Apa kabar?"]
        for record, id_text in zip(records, translations):
            record["id_final"] = id_text
            record["review_required"] = False
            record["note"] = ""
        create_id_translation_output_zip(
            artifacts.schema,
            records,
            paths["translated"],
            input_manifest=manifest,
        )
        return artifacts.schema

    def _kwargs(self, root: Path, paths: dict[str, Path]) -> dict:
        return {
            "episode_root": root,
            "episode": self.episode,
            "source_video": paths["source"],
            "raw_asr_path": paths["raw"],
            "forced_alignment_path": paths["forced"],
            "tr_correction_pack": paths["tr_pack"],
            "tr_text_correction_output": paths["tr_text_output"],
            "tr_correction_output": paths["tr_output"],
            "audio_review_path": paths["audio_review"],
            "aligned_schema": paths["schema"],
            "id_translation_pack": paths["id_pack"],
            "id_translation_zip": paths["translated"],
            "series_config": paths["series_config"],
            "names_config": paths["names_config"],
            "religious_config": paths["religious_config"],
        }

    def _rewrite_forced_alignment(
        self,
        paths: dict[str, Path],
        mutate: object,
    ) -> None:
        forced = self._loaded(paths["forced"])
        mutate(forced)
        digest = _alignment_sha256(forced)
        forced["alignment_sha256"] = digest
        forced["report"]["alignment_sha256"] = digest
        for word in forced["words"]:
            word["alignment_sha256"] = digest
        for segment in forced["segments"]:
            for word in segment["words"]:
                word["alignment_sha256"] = digest
        self._json(paths["forced"], forced)
        write_stage_marker(
            paths["forced_marker"],
            stage="forced_alignment_v2",
            input_sha256="f" * 64,
            outputs={"forced_alignment_v2": paths["forced"]},
            details={
                "alignment_sha256": digest,
                "correction_output_sha256": validate_tr_correction_output(
                    paths["tr_pack"], paths["tr_output"]
                ).output_sha256,
            },
        )

    @staticmethod
    def _mux_report(output: Path, block_count: int = 2) -> dict:
        hashes = [
            {
                "type": "video",
                "ordinal": 0,
                "algorithm": "sha256",
                "digest": "b" * 64,
            }
        ]
        return {
            "verified": True,
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": {
                "exact": True,
                "id_block_count": block_count,
                "tr_block_count": block_count,
            },
            "stream_hashes": {
                "checked": True,
                "match": True,
                "source": hashes,
                "output": hashes,
            },
        }

    def _fake_mux(
        self,
        source: Path,
        id_srt: Path,
        tr_srt: Path,
        output: Path,
        **kwargs: object,
    ) -> dict:
        self.assertEqual(source.name, f"{self.episode_name}.mp4")
        self.assertEqual(
            [entry.text for entry in parse_srt(id_srt)],
            ["Defne 12 datang.", "Apa kabar?"],
        )
        self.assertEqual(
            [entry.text for entry in parse_srt(tr_srt)],
            ["Defne 12 geldi.", "Bugün nasılsın?"],
        )
        self.assertIs(kwargs.get("verify_stream_hashes"), True)
        output.write_bytes(b"verified-stream-copy-mkv")
        return self._mux_report(output)

    def test_pass_rebuilds_full_evidence_and_preserves_legacy_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            schema = self._inputs(root, paths)
            final = root / "final"
            final.mkdir()
            legacy = {
                "report": final / f"{self.episode_name}_FINALIZATION_REPORT.json",
                "id": final / f"{self.episode_name}.id-final.srt",
                "tr": final / f"{self.episode_name}.tr-final.srt",
                "mkv": final / f"{self.episode_name} - Endonezce + Turkce.mkv",
            }
            legacy_bytes = {
                "report": b'{"legacy":true}',
                "id": b"legacy-id",
                "tr": b"legacy-tr",
                "mkv": b"legacy-mkv",
            }
            for key, path in legacy.items():
                path.write_bytes(legacy_bytes[key])

            with patch(
                "mas.engine.finalize.mux_softsubs", side_effect=self._fake_mux
            ) as mocked:
                report = finalize_episode_v2(**self._kwargs(root, paths))

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(report["report_version"], 2)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["schema_sha256"], schema["schema_sha256"])
            self.assertTrue(report["timing_qa_v2"]["passed"])
            self.assertTrue(report["timing_qa_v2"]["word_ownership_checked"])
            self.assertTrue(report["semantic_subtitle_qa"]["passed"])
            self.assertEqual(
                report["id_translation_validation"]["review_required_count"], 0
            )
            self.assertTrue(
                report["evidence_validation"]["speech_coverage_recomputed"]
            )
            self.assertEqual(report["raw_policy_v2"]["status"], "PASS")
            self.assertEqual(
                report["raw_policy_v2"]["raw_artifact_sha256"],
                report["input_files"]["raw_asr_v2"]["sha256"],
            )
            self.assertEqual(
                report["alignment_policy_v2"]["hard_zero_counters"],
                {
                    "unaligned_lexical_token_count": 0,
                    "unaligned_word_count": 0,
                    "synthetic_timing_count": 0,
                    "interpolated_timing_count": 0,
                    "overlap_violation_count": 0,
                    "negative_word_gap_count": 0,
                    "missing_alignment_score_count": 0,
                    "low_alignment_score_count": 0,
                    "overlong_alignment_word_count": 0,
                    "outward_drift_violation_count": 0,
                    "low_score_edited_token_count": 0,
                    "unreviewed_deleted_token_count": 0,
                },
            )
            self.assertEqual(
                report["alignment_edit_audit_v2"]["summary"][
                    "unreviewed_deleted_token_count"
                ],
                0,
            )
            self.assertEqual(
                report["audio_review_v2"],
                report["speech_coverage_v2"]["audio_review_v2"],
            )
            self.assertEqual(
                report["strict_word_vad_v2"],
                report["speech_coverage_v2"]["strict_word_vad_v2"],
            )
            self.assertEqual(set(report["outputs"]), {"mkv", "id_srt", "tr_srt"})
            report_path = (
                final / f"{self.episode_name}_FINALIZATION_REPORT_V2.json"
            )
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")), report
            )
            archive_input = _load_finalization_inputs(root, self.episode)
            self.assertEqual(
                archive_input["finalization_report_version"], 2
            )
            self.assertEqual(
                archive_input["source"], "finalization_report_v2"
            )
            for key, path in legacy.items():
                self.assertEqual(path.read_bytes(), legacy_bytes[key])

    def test_weakened_alignment_policy_or_hard_counter_cannot_reach_mux(
        self,
    ) -> None:
        mutations = {
            "minimum word score": lambda value: value["provenance"].__setitem__(
                "min_word_score", 0.29
            ),
            "maximum word duration": lambda value: value[
                "provenance"
            ].__setitem__("max_word_duration_ms", 2_501),
            "maximum outward drift": lambda value: value[
                "provenance"
            ].__setitem__("max_outward_drift_ms", 501),
            "edited-token score": lambda value: value[
                "provenance"
            ].__setitem__("edited_token_min_word_score", 0.54),
            "low-score edited counter": lambda value: value[
                "report"
            ].__setitem__("low_score_edited_token_count", 1),
            "unreviewed deletion counter": lambda value: value[
                "report"
            ].__setitem__("unreviewed_deleted_token_count", 1),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root, paths = self._workspace(temporary)
                self._inputs(root, paths)
                self._rewrite_forced_alignment(paths, mutate)
                with patch("mas.engine.finalize.mux_softsubs") as mocked:
                    with self.assertRaises(Exception):
                        finalize_episode_v2(**self._kwargs(root, paths))
                mocked.assert_not_called()

    def test_999ms_decoder_segment_gap_is_split_in_final_srts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            utterances = self._utterances()
            utterances[1]["coarse_start_ms"] = 3_300
            utterances[1]["coarse_end_ms"] = 4_700
            corrections = copy.deepcopy(utterances)
            for record, corrected in zip(
                corrections,
                ("Defne 12 geldi.", "Bugün nasılsın?"),
            ):
                record.update(
                    {
                        "tr_corrected": corrected,
                        "non_dialogue": False,
                        "review_required": False,
                        "audio_reviewed": False,
                        "review_disposition": "not_applicable",
                        "note": "",
                    }
                )
            whisperx = _FakeWhisperX(
                [
                    _alignment_result(
                        [
                            {
                                "word": "Defne",
                                "start": 1.08,
                                "end": 1.35,
                                "score": 0.96,
                            },
                            {
                                "word": "12",
                                "start": 1.42,
                                "end": 1.55,
                                "score": 0.97,
                            },
                            {
                                "word": "geldi.",
                                "start": 1.62,
                                "end": 2.35,
                                "score": 0.95,
                            },
                        ]
                    ),
                    _alignment_result(
                        [
                            {
                                "word": "Bugün",
                                "start": 3.349,
                                "end": 3.82,
                                "score": 0.94,
                            },
                            {
                                "word": "nasılsın?",
                                "start": 3.89,
                                "end": 4.62,
                                "score": 0.95,
                            },
                        ]
                    ),
                ]
            )
            self._inputs(
                root,
                paths,
                utterances=utterances,
                corrections=corrections,
                vad_regions=[
                    {
                        "vad_region_index": 1,
                        "start_ms": 1_000,
                        "end_ms": 2_500,
                        "source": "silero_vad",
                    },
                    {
                        "vad_region_index": 2,
                        "start_ms": 3_300,
                        "end_ms": 4_700,
                        "source": "silero_vad",
                    },
                ],
                whisperx_module=whisperx,
            )
            with patch(
                "mas.engine.finalize.mux_softsubs", side_effect=self._fake_mux
            ):
                finalize_episode_v2(**self._kwargs(root, paths))

            tr_entries = parse_srt(
                root
                / "final"
                / "subtitles"
                / f"{self.episode_name}-tr.srt"
            )
            self.assertEqual(
                [entry.text for entry in tr_entries],
                ["Defne 12 geldi.", "Bugün nasılsın?"],
            )
            self.assertEqual(tr_entries[1].start_ms, 3_349)
            self.assertLess(tr_entries[0].end_ms, tr_entries[1].start_ms)

    def test_150ms_evet_deletion_requires_audio_review_before_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)
            raw = self._loaded(paths["raw"])
            first = raw["correction_utterances"][0]
            first["coarse_end_ms"] = first["coarse_start_ms"] + 150
            first["asr_text"] = "evet Defne 12 geldi."
            self._json(paths["raw"], raw)
            write_stage_marker(
                paths["raw_marker"],
                stage="raw_asr_v2",
                input_sha256=raw["input_sha256"],
                outputs={"raw_asr_v2": paths["raw"]},
                details={
                    "audio_sha256": raw["audio_sha256"],
                    "independent_vad": True,
                },
            )
            create_tr_correction_pack(
                raw["correction_utterances"],
                raw["speech_hole_records"],
                paths["tr_pack"],
                episode=self.episode,
                asr_hallucination_records=raw[
                    "asr_hallucination_records"
                ],
            )
            corrections = copy.deepcopy(raw["correction_utterances"])
            for position, record in enumerate(corrections):
                record.update(
                    {
                        "tr_corrected": (
                            "Defne 12 geldi."
                            if position == 0
                            else "Bugün nasılsın?"
                        ),
                        "non_dialogue": False,
                        "review_required": position == 0,
                        "audio_reviewed": False,
                        "review_disposition": (
                            "pending_audio_review"
                            if position == 0
                            else "not_applicable"
                        ),
                        "note": (
                            "150 ms evet tokeni için exact WAV incelemesi gerekli."
                            if position == 0
                            else ""
                        ),
                    }
                )
            create_tr_correction_output(
                paths["tr_pack"], corrections, paths["tr_output"]
            )

            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaisesRegex(
                    FinalizationV2Error, "still requires review"
                ):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_flat_fabricated_alignment_cannot_reach_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)
            flat = {
                "alignment_sha256": "a" * 64,
                "unaligned_word_count": 0,
                "synthetic_timing_count": 0,
                "negative_word_gap_count": 0,
                "missing_alignment_score_count": 0,
                "low_alignment_score_count": 0,
            }
            self._json(paths["forced"], flat)
            write_stage_marker(
                paths["forced_marker"],
                stage="forced_alignment_v2",
                input_sha256="f" * 64,
                outputs={"forced_alignment_v2": paths["forced"]},
                details={
                    "alignment_sha256": flat["alignment_sha256"],
                    "correction_output_sha256": validate_tr_correction_output(
                        paths["tr_pack"], paths["tr_output"]
                    ).output_sha256,
                },
            )
            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaises(Exception):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_tampered_bounded_audio_review_cannot_reach_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)
            audit = self._loaded(paths["audio_review"])
            audit["provisional_output_sha256"] = "0" * 64
            self._json(paths["audio_review"], audit)
            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaisesRegex(
                    FinalizationV2Error,
                    "Bounded Colab audio-review evidence is invalid",
                ):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_stale_audio_full_alignment_cannot_reach_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)
            stale_audio = root / "prepare" / "stale.flac"
            stale_audio.write_bytes(b"stale-other-episode-audio")
            corrections = validate_tr_correction_output(
                paths["tr_pack"], paths["tr_output"]
            )
            raw = self._loaded(paths["raw"])
            bundle = correction_records_to_alignment_inputs(
                raw["correction_utterances"],
                corrections.records,
                asr_hallucination_records=raw["asr_hallucination_records"],
            )
            stale = align_corrected_segments(
                stale_audio,
                bundle.alignment_inputs,
                whisperx_module=self._fake_whisperx(),
                device="cpu",
            )
            self._json(paths["forced"], stale)
            write_stage_marker(
                paths["forced_marker"],
                stage="forced_alignment_v2",
                input_sha256="f" * 64,
                outputs={"forced_alignment_v2": paths["forced"]},
                details={
                    "alignment_sha256": stale["alignment_sha256"],
                    "correction_output_sha256": corrections.output_sha256,
                },
            )
            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaisesRegex(
                    FinalizationV2Error, "stale or different audio"
                ):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_semantic_anchor_mismatch_stops_before_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            schema = self._inputs(root, paths)
            self._replace_id_output(
                paths, schema, ["Defne datang.", "Apa kabar?"]
            )
            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaises(SubtitleQAError):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_high_indonesian_cps_stops_before_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(
                root,
                paths,
                id_texts=[
                    "Defne 12 datang dengan kalimat yang sangat amat panjang sekali.",
                    "Apa kabar?",
                ],
            )
            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaises(TimingQAV2Error):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_id_review_required_stops_before_mux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            schema = self._inputs(root, paths)
            self._replace_id_output(
                paths,
                schema,
                ["Defne 12 datang.", "Apa kabar?"],
                review_required=True,
            )
            with patch("mas.engine.finalize.mux_softsubs") as mocked:
                with self.assertRaisesRegex(
                    FinalizationV2Error, "still requires review"
                ):
                    finalize_episode_v2(**self._kwargs(root, paths))
            mocked.assert_not_called()

    def test_incomplete_mux_verification_publishes_no_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)

            def bad_mux(*args: object, **kwargs: object) -> dict:
                output = Path(args[3])
                output.write_bytes(b"untrusted-mkv")
                report = self._mux_report(output)
                report["stream_hashes"]["match"] = False
                return report

            with patch("mas.engine.finalize.mux_softsubs", side_effect=bad_mux):
                with self.assertRaisesRegex(
                    FinalizationV2Error, "verification failed"
                ):
                    finalize_episode_v2(**self._kwargs(root, paths))
            final = root / "final"
            self.assertFalse((final / f"{self.episode_name}.mkv").exists())
            self.assertFalse(
                (
                    final
                    / f"{self.episode_name}_FINALIZATION_REPORT_V2.json"
                ).exists()
            )

    def test_post_mux_evidence_mutation_rolls_back_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)

            def mutating_mux(*args: object, **kwargs: object) -> dict:
                output = Path(args[3])
                output.write_bytes(b"verified-stream-copy-mkv")
                paths["names_config"].write_text(
                    "canonical_names: []\n", encoding="utf-8"
                )
                return self._mux_report(output)

            with patch(
                "mas.engine.finalize.mux_softsubs", side_effect=mutating_mux
            ):
                with self.assertRaisesRegex(
                    FinalizationV2Error, "changed before PASS"
                ):
                    finalize_episode_v2(**self._kwargs(root, paths))
            self.assertFalse(
                (root / "final" / f"{self.episode_name}.mkv").exists()
            )
            self.assertFalse(
                (
                    root
                    / "final"
                    / f"{self.episode_name}_FINALIZATION_REPORT_V2.json"
                ).exists()
            )

    def test_report_commit_failure_restores_all_prior_v2_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, paths = self._workspace(temporary)
            self._inputs(root, paths)
            final = root / "final"
            subtitles = final / "subtitles"
            subtitles.mkdir(parents=True)
            prior_paths = {
                "mkv": final / f"{self.episode_name}.mkv",
                "id_srt": subtitles / f"{self.episode_name}-id.srt",
                "tr_srt": subtitles / f"{self.episode_name}-tr.srt",
                "report": (
                    final
                    / f"{self.episode_name}_FINALIZATION_REPORT_V2.json"
                ),
            }
            prior_bytes = {
                "mkv": b"prior-mkv",
                "id_srt": b"prior-id",
                "tr_srt": b"prior-tr",
                "report": b'{"report_version":2}',
            }
            for key, path in prior_paths.items():
                path.write_bytes(prior_bytes[key])

            with (
                patch(
                    "mas.engine.finalize.mux_softsubs",
                    side_effect=self._fake_mux,
                ),
                patch(
                    "mas.engine.finalize.atomic_write_json",
                    side_effect=OSError("disk full"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    finalize_episode_v2(**self._kwargs(root, paths))

            for key, path in prior_paths.items():
                self.assertEqual(path.read_bytes(), prior_bytes[key])
            self.assertFalse(any(root.rglob("*.superseded")))


if __name__ == "__main__":
    unittest.main()
