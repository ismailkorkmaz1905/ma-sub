from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "02_ALIGN_PREPARE_ID.ipynb"


class AlignPrepareIDNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.source = "\n".join(cls.code_cells)

    def test_all_code_cells_compile(self) -> None:
        for index, source in enumerate(self.code_cells):
            compile(source, f"02_ALIGN_PREPARE_ID.ipynb:code-cell-{index}", "exec")

    def test_mounts_drive_and_uses_only_v2_runtime_requirements(self) -> None:
        self.assertIn('drive.mount("/content/drive"', self.source)
        self.assertIn(
            'Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA")',
            self.source,
        )
        self.assertNotIn(
            'Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM")',
            self.source,
        )
        self.assertIn('SYSTEM_ROOT / "requirements-v2-colab.txt"', self.source)
        self.assertNotIn('SYSTEM_ROOT / "requirements-colab.txt"', self.source)
        self.assertIn('module_name.startswith("src.")', self.source)
        self.assertIn("importlib.invalidate_caches()", self.source)

    def test_exact_episode_paths_are_used_without_glob_fallbacks(self) -> None:
        self.assertIn(
            'f"{EPISODE_NAME}_TR_CORRECTION_PACK.zip"', self.source
        )
        self.assertIn('f"{EPISODE_NAME}_TR_CORRECTED.zip"', self.source)
        self.assertIn('f"{EPISODE_NAME}_TR_TEXT_CORRECTED.zip"', self.source)
        self.assertIn('"aligned_tr_schema_v2.json"', self.source)
        self.assertIn('f"{EPISODE_NAME}_ID_TRANSLATION_PACK.zip"', self.source)
        self.assertNotIn("glob(", self.source)
        self.assertNotIn("rglob(", self.source)

    def test_text_only_output_is_acoustically_resolved_before_alignment(self) -> None:
        provisional_validation = self.source.index(
            "text_correction_output = validate_tr_correction_output("
        )
        audio_review = self.source.index("resolve_tr_audio_reviews_v2(")
        final_validation = self.source.index(
            "\ncorrection_output = validate_tr_correction_output("
        )
        routing = self.source.index("correction_records_to_alignment_inputs(")
        self.assertLess(provisional_validation, audio_review)
        self.assertLess(audio_review, final_validation)
        self.assertLess(final_validation, routing)
        self.assertIn("AUDIO_REVIEW_RECOVERY_PATH", self.source)
        self.assertIn("manual_overrides=MANUAL_AUDIO_REVIEW", self.source)
        self.assertIn("pending_audio_review", self.source)
        self.assertIn("Audio(data=archive.read", self.source)
        self.assertIn("validate_audio_review_v2_report(", self.source)
        self.assertIn("acoustic_audio_review=audio_review_data", self.source)
        self.assertIn(
            '"audio_review_sha256": audio_review_data["audio_review_sha256"]',
            self.source,
        )
        self.assertIn("AUDIO_REVIEW_REPORT_PATH", self.source)
        snapshot = self.source.index("input_file_snapshot = {")
        routing = self.source.index("correction_records_to_alignment_inputs(")
        self.assertIn("AUDIO_REVIEW_REPORT_PATH", self.source[snapshot:routing])
        self.assertIn('"audio_review_v2": AUDIO_REVIEW_REPORT_PATH', self.source)

    def test_exact_correction_pair_is_validated_before_alignment(self) -> None:
        pack_read = self.source.index("read_tr_correction_pack(")
        output_validation = self.source.index("validate_tr_correction_output(")
        input_derivation = self.source.index("correction_records_to_alignment_inputs(")
        alignment = self.source.index("align_corrected_segments(")
        self.assertLess(pack_read, output_validation)
        self.assertLess(output_validation, input_derivation)
        self.assertLess(input_derivation, alignment)
        self.assertIn("expected_input_sha256=expected_correction_input_sha", self.source)
        self.assertIn(
            'asr_hallucination_records=raw_asr_data.get("asr_hallucination_records", [])',
            self.source,
        )

    def test_raw_asr_uses_public_artifact_and_marker_loader(self) -> None:
        self.assertIn("from src.raw_asr_v2 import load_valid_raw_asr_v2", self.source)
        self.assertIn("raw_asr_data = load_valid_raw_asr_v2(", self.source)
        self.assertIn("require_independent_vad=True", self.source)
        self.assertNotIn("with RAW_ASR_PATH.open", self.source)

    def test_alignment_resume_identity_binds_audio_text_model_and_code(self) -> None:
        self.assertIn('"audio_sha256": audio_sha256', self.source)
        self.assertIn(
            '"correction_output_sha256": correction_output.output_sha256',
            self.source,
        )
        self.assertIn('"alignment_inputs": alignment_bundle.alignment_inputs', self.source)
        self.assertIn('"model_name": ALIGNMENT_MODEL', self.source)
        self.assertIn('"min_word_score": MIN_WORD_SCORE', self.source)
        self.assertIn('"device": alignment_device', self.source)
        self.assertIn('"code_sha256": alignment_code_sha256', self.source)
        self.assertIn("load_valid_stage_marker(", self.source)
        self.assertIn("validate_forced_alignment_data(", self.source)
        self.assertIn(
            'data.get("audio_sha256") != audio_sha256', self.source
        )
        self.assertIn(
            "persisted_inputs != alignment_bundle.alignment_inputs", self.source
        )
        self.assertIn(
            "atomic_write_json(FORCED_ALIGNMENT_PATH, forced_alignment)", self.source
        )

    def test_alignment_uses_hole_bound_routing_and_acoustic_score_gate(self) -> None:
        self.assertIn(
            "speech_hole_records=correction_pack.speech_holes", self.source
        )
        self.assertIn("MIN_WORD_SCORE = 0.30", self.source)
        self.assertIn("MAX_WORD_DURATION_MS = 2500", self.source)
        self.assertIn("MAX_OUTWARD_DRIFT_MS = 500", self.source)
        self.assertIn("min_word_score=MIN_WORD_SCORE", self.source)
        self.assertIn("max_word_duration_ms=MAX_WORD_DURATION_MS", self.source)
        self.assertIn("max_outward_drift_ms=MAX_OUTWARD_DRIFT_MS", self.source)
        self.assertIn('segment["coarse_start_ms"]', self.source)
        self.assertIn('segment["coarse_end_ms"]', self.source)
        self.assertIn('segment["asr_text"]', self.source)
        self.assertIn('segment["deletion_audio_reviewed"]', self.source)

    def test_long_run_inputs_are_rehashed_immediately_before_publish(self) -> None:
        snapshot = self.source.index("input_file_snapshot = {")
        alignment = self.source.index("align_corrected_segments(")
        rehash = self.source.index("current_input_snapshot = {")
        publish = self.source.index("create_v2_id_translation_pack(")
        self.assertLess(snapshot, alignment)
        self.assertLess(alignment, rehash)
        self.assertLess(rehash, publish)
        for name in (
            "RAW_ASR_PATH",
            "AUDIO_PATH",
            "TR_CORRECTION_PACK_PATH",
            "TR_CORRECTION_OUTPUT_PATH",
        ):
            self.assertIn(name, self.source[snapshot:alignment])

    def test_independent_vad_is_a_hard_publication_gate(self) -> None:
        self.assertIn(
            'raw_asr_data.get("independent_vad") is not True', self.source
        )

    def test_full_v2_gate_order_precedes_exact_id_pack_publish(self) -> None:
        self.assertNotIn("recompute_final_speech_coverage(", self.source)
        artifacts = self.source.index("artifacts = build_strict_v2_artifacts(")
        coverage = self.source.index(
            "final_speech_coverage = require_authoritative_coverage(artifacts)"
        )
        qa = self.source.index("assert_timing_qa_v2(")
        publish = self.source.index("create_v2_id_translation_pack(")
        validate_pack = self.source.index("validate_id_translation_pack(")
        self.assertLess(artifacts, qa)
        self.assertLess(artifacts, coverage)
        self.assertLess(coverage, qa)
        self.assertLess(qa, publish)
        self.assertLess(publish, validate_pack)
        self.assertIn("unresolved_speech_region_count", self.source)

    def test_authoritative_coverage_gate_executes_on_enriched_report(self) -> None:
        tree = ast.parse(self.source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "require_authoritative_coverage"
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[function], type_ignores=[])
        )
        namespace: dict[str, object] = {}
        exec(compile(module, "02-coverage-gate", "exec"), namespace)
        gate = namespace["require_authoritative_coverage"]
        report = {
            "status": "PASS",
            "unresolved_speech_region_count": 0,
            "audio_review_v2": {
                "status": "PASS",
                "pending_audio_review_count": 0,
            },
            "strict_word_vad_v2": {
                "status": "PASS",
                "unsafe_word_count": 0,
                "confirmed_dialogue_coverage_fail_count": 0,
            },
        }
        self.assertIs(gate(SimpleNamespace(speech_coverage_report=report)), report)
        failed = json.loads(json.dumps(report))
        failed["strict_word_vad_v2"]["unsafe_word_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "Authoritative"):
            gate(SimpleNamespace(speech_coverage_report=failed))

    def test_id_pack_is_bound_to_explicit_canonical_glossary(self) -> None:
        self.assertIn("load_default_id_translation_glossary()", self.source)
        self.assertIn('SYSTEM_ROOT / "config/names.yaml"', self.source)
        self.assertIn('SYSTEM_ROOT / "config/religious_terms.yaml"', self.source)
        self.assertIn('"id_glossary_sha256": sha256_json(id_glossary)', self.source)
        self.assertIn("glossary=id_glossary", self.source)
        self.assertIn("expected_glossary=id_glossary", self.source)

    def test_device_is_explicit_and_legacy_pipeline_is_absent(self) -> None:
        self.assertIn('ALIGN_DEVICE = "auto"', self.source)
        self.assertIn('ALIGN_DEVICE not in {"auto", "cuda", "cpu"}', self.source)
        self.assertIn("torch.cuda.is_available()", self.source)
        forbidden = (
            "from src.schema import",
            "from src.batches import",
            "from src.translation_validation import",
            "build_episode_schema",
            "segment_transcription",
            "create_translation_pack",
            "load_and_validate_translated_zip",
        )
        for legacy_name in forbidden:
            with self.subTest(legacy_name=legacy_name):
                self.assertNotIn(legacy_name, self.source)


if __name__ == "__main__":
    unittest.main()
