from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREPARE = PROJECT_ROOT / "01_PREPARE.ipynb"
PREPARE_TR = PROJECT_ROOT / "01_PREPARE_TR.ipynb"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _code_sources(notebook: dict) -> list[str]:
    return [
        _source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class PrepareTRNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = json.loads(LEGACY_PREPARE.read_text(encoding="utf-8"))
        cls.notebook = json.loads(PREPARE_TR.read_text(encoding="utf-8"))
        cls.code_cells = _code_sources(cls.notebook)
        cls.code = "\n\n".join(cls.code_cells)
        cls.markdown = "\n\n".join(
            _source(cell)
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

    def test_notebook_is_valid_and_every_code_cell_compiles(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(
            self.notebook["metadata"]["colab"]["name"],
            "01_PREPARE_TR.ipynb",
        )
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        for index, source in enumerate(self.code_cells):
            compile(source, f"01_PREPARE_TR.ipynb:cell{index}", "exec")

    def test_proven_mount_cookie_and_download_flows_are_reused(self) -> None:
        legacy_code = _code_sources(self.legacy)
        legacy_setup = next(source for source in legacy_code if "drive.mount(" in source)
        expected_setup = legacy_setup.replace(
            "from pathlib import Path\nimport os",
            "from pathlib import Path\nimport importlib\nimport os",
        ).replace(
            'SYSTEM_ROOT = Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM")',
            'SYSTEM_ROOT = Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA")',
        ).replace(
            'f"System files were not found at {SYSTEM_ROOT}. Follow 00_README_FIRST.md."',
            'f"V2 beta system files were not found at {SYSTEM_ROOT}. Follow V2_BETA_README.md."',
        ).replace(
            'if str(SYSTEM_ROOT) not in sys.path:\n'
            '    sys.path.insert(0, str(SYSTEM_ROOT))\n'
            'print("Runtime ready.")',
            'if str(SYSTEM_ROOT) not in sys.path:\n'
            '    sys.path.insert(0, str(SYSTEM_ROOT))\n'
            'for module_name in tuple(sys.modules):\n'
            '    if module_name == "src" or module_name.startswith("src."):\n'
            '        del sys.modules[module_name]\n'
            'importlib.invalidate_caches()\n'
            'print("V2 beta runtime ready.")',
        )
        actual_setup = next(source for source in self.code_cells if "drive.mount(" in source)
        self.assertEqual(actual_setup, expected_setup)
        self.assertIn("requirements-colab.txt", actual_setup)
        self.assertNotIn("requirements-v2-colab.txt", actual_setup)
        self.assertIn('module_name.startswith("src.")', actual_setup)
        self.assertIn("importlib.invalidate_caches()", actual_setup)
        self.assertIn("SYSTEM_V2_BETA", actual_setup)

        legacy_cookie = next(
            source for source in legacy_code if "def upload_youtube_cookies():" in source
        )
        actual_cookie = next(
            source for source in self.code_cells if "def upload_youtube_cookies():" in source
        )
        self.assertEqual(actual_cookie, legacy_cookie)

        legacy_download = next(
            source for source in legacy_code if "download = download_source(" in source
        )
        actual_download = next(
            source for source in self.code_cells if "download = download_source(" in source
        )
        self.assertEqual(actual_download, legacy_download)

    def test_v2_prepare_stops_at_tr_correction_pack(self) -> None:
        self.assertIn(
            "from src.raw_asr_v2 import RawASRV2Config, transcribe_raw_audio_v2",
            self.code,
        )
        self.assertIn("raw_asr = transcribe_raw_audio_v2(", self.code)
        self.assertIn("EXTRA_AUDIO_REVIEW_UIDS = []", self.code)
        self.assertIn(
            "extra_audio_review_uids=tuple(EXTRA_AUDIO_REVIEW_UIDS)",
            self.code,
        )
        self.assertIn('raw_asr.get("independent_vad") is not True', self.code)
        self.assertIn("create_tr_correction_pack,", self.code)
        self.assertIn("validate_tr_correction_output,", self.code)
        self.assertIn("manifest = create_tr_correction_pack(", self.code)
        self.assertIn('speech_hole_audio_root=DIRS["prepare"]', self.code)
        self.assertIn(
            'asr_hallucination_audio_root=DIRS["prepare"]', self.code
        )
        self.assertIn(
            "asr_hallucination_records=asr_hallucination_records", self.code
        )
        self.assertIn("rebind_text_output_path=text_output_path", self.code)
        self.assertIn("Pro does not need to repeat the Turkish correction", self.code)
        self.assertIn(
            'DIRS["translation_input"]\n    / f"{EPISODE_NAME}_TR_CORRECTION_PACK.zip"',
            self.code,
        )

        forbidden = (
            "transcribe_audio(",
            "segment_transcription(",
            "build_episode_schema(",
            "save_schema(",
            "create_translation_pack(",
            "from src.segment",
            "from src.schema",
            "from src.batches",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.code)

    def test_unresolved_speech_is_counted_and_reserved_for_colab_review(self) -> None:
        self.assertIn(
            'unresolved_speech_candidate_count = len(speech_hole_records)',
            self.code,
        )
        self.assertIn("blank_audio_review_count", self.code)
        self.assertIn("Every unresolved speech candidate", self.code)
        self.assertIn("bounded, resumable Colab audio audit", self.markdown)
        self.assertIn("hash-bound contextual WAV evidence", self.markdown)
        self.assertIn("must not open or transcribe the bundled WAVs", self.code)
        self.assertIn("Keep every flagged audio decision pending", self.code)
        self.assertIn(
            'hallucination_review_count = len(asr_hallucination_records)',
            self.code,
        )
        self.assertIn("orphan_caption_count", self.code)

    def test_pack_publish_path_and_counts_are_hard_checked(self) -> None:
        self.assertIn(
            'expected_name = f"{EPISODE_NAME}_TR_CORRECTION_PACK.zip"',
            self.code,
        )
        self.assertIn("not pack_path.is_file()", self.code)
        self.assertIn(
            'manifest["utterance_count"] != len(correction_utterances)',
            self.code,
        )
        self.assertIn(
            'manifest["speech_hole_count"] != unresolved_speech_candidate_count',
            self.code,
        )
        self.assertIn(
            'manifest["asr_hallucination_count"] != hallucination_review_count',
            self.code,
        )


if __name__ == "__main__":
    unittest.main()
