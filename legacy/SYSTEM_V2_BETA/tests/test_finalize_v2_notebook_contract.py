from __future__ import annotations

import json
from pathlib import Path
import unittest


class FinalizeV2NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "03_FINALIZE_V2.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.code = "\n".join(cls.code_cells)

    def test_every_code_cell_compiles(self) -> None:
        for index, source in enumerate(self.code_cells):
            compile(source, f"03_FINALIZE_V2.ipynb:cell-{index}", "exec")

    def test_uses_only_isolated_pinned_v2_runtime(self) -> None:
        self.assertIn(
            'Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA")',
            self.code,
        )
        self.assertNotIn(
            'Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM")',
            self.code,
        )
        self.assertIn('SYSTEM_ROOT / "requirements-v2-colab.txt"', self.code)
        self.assertNotIn('SYSTEM_ROOT / "requirements-colab.txt"', self.code)
        self.assertIn('module_name.startswith("src.")', self.code)
        self.assertIn("importlib.invalidate_caches()", self.code)

    def test_calls_only_path_bound_v2_finalizer_with_exact_names(self) -> None:
        self.assertEqual(self.code.count("finalize_episode_v2("), 1)
        for value in (
            '"raw_asr_v2.json"',
            '"forced_alignment_v2.json"',
            '"aligned_tr_schema_v2.json"',
            'f"{EPISODE_NAME}_TR_CORRECTION_PACK.zip"',
            'f"{EPISODE_NAME}_TR_TEXT_CORRECTED.zip"',
            'f"{EPISODE_NAME}_TR_CORRECTED.zip"',
            '"audio_review_v2.json"',
            'f"{EPISODE_NAME}_ID_TRANSLATION_PACK.zip"',
            'f"{EPISODE_NAME}_ID_TRANSLATED.zip"',
            '"config/series.yaml"',
            '"config/names.yaml"',
            '"config/religious_terms.yaml"',
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.code)
        self.assertIn(
            "tr_text_correction_output=TR_TEXT_OUTPUT_PATH", self.code
        )
        self.assertIn("audio_review_path=AUDIO_REVIEW_PATH", self.code)
        self.assertNotIn("glob(", self.code)
        self.assertNotIn("rglob(", self.code)

    def test_download_marker_and_loaded_module_are_verified(self) -> None:
        self.assertIn("load_valid_stage_marker(", self.code)
        self.assertIn('required_output_keys=("video", "metadata")', self.code)
        self.assertIn("SYSTEM_ROOT.resolve() not in module_path.parents", self.code)

    def test_v2_report_and_single_copy_outputs_are_exact(self) -> None:
        self.assertIn("_FINALIZATION_REPORT_V2.json", self.code)
        self.assertNotIn('f"{EPISODE_NAME}_FINALIZATION_REPORT.json"', self.code)
        self.assertIn('f"final/{EPISODE_NAME}.mkv"', self.code)
        self.assertIn('f"final/subtitles/{EPISODE_NAME}-id.srt"', self.code)
        self.assertIn('f"final/subtitles/{EPISODE_NAME}-tr.srt"', self.code)
        self.assertIn('print("FINALIZATION V2 PASS")', self.code)
        self.assertIn("PILOT WARNING ONLY", self.code)
        self.assertIn("review_alignment_score_count", self.code)
        self.assertIn("confirmed_dialogue_count", self.code)
        self.assertIn("reviewed_non_dialogue_count", self.code)
        self.assertIn("discarded_asr_hallucination_count", self.code)
        self.assertIn("pending_audio_review_count", self.code)


if __name__ == "__main__":
    unittest.main()
