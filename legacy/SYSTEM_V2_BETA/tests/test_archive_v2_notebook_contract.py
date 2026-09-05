from __future__ import annotations

import json
from pathlib import Path
import unittest


class ArchiveV2NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook_path = Path(__file__).resolve().parents[1] / "04_ARCHIVE_V2.ipynb"
        cls.notebook = json.loads(cls.notebook_path.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

    def test_all_code_cells_compile(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(
                    "".join(cell.get("source", [])),
                    f"04_ARCHIVE_V2.ipynb:cell{index}",
                    "exec",
                )

    def test_uses_only_archive_v2_entrypoint_for_cleanup(self) -> None:
        self.assertIn(
            'SYSTEM_ROOT = Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA")',
            self.code,
        )
        self.assertNotIn(
            'SYSTEM_ROOT = Path("/content/drive/MyDrive/Muhtemel_Ask_Subtitles/SYSTEM")',
            self.code,
        )
        self.assertIn("Upload the current SYSTEM_V2_BETA release", self.code)
        self.assertIn("import src.archive_v2 as archive_v2", self.code)
        self.assertIn("archive_episode_v2 = archive_v2.archive_episode_v2", self.code)
        self.assertEqual(self.code.count("archive_episode_v2("), 1)
        self.assertNotIn("src.episode_archive", self.code)
        self.assertNotIn("KEEP_SOURCE_VIDEO", self.code)
        for forbidden in (".unlink(", "rmtree(", "os.remove(", "send2trash", "Trash"):
            self.assertNotIn(forbidden, self.code)

    def test_cleanup_is_dry_run_by_default_and_exactly_confirmed(self) -> None:
        self.assertIn("DELETE_INTERMEDIATES = False", self.code)
        self.assertIn(
            "EXPECTED_CONFIRMATION = expected_cleanup_confirmation_v2(EPISODE)",
            self.code,
        )
        self.assertIn("if expected != EXPECTED_CONFIRMATION:", self.code)
        self.assertIn(
            "confirmation=request_cleanup_confirmation if DELETE_INTERMEDIATES else None",
            self.code,
        )
        self.assertEqual(self.code.count("input("), 1)
        self.assertIn("Dry-run mode: source/work duplicates will remain", self.code)

    def test_receipt_and_retained_layout_are_exact_v2_paths(self) -> None:
        self.assertIn("_ARCHIVE_RECEIPT_V2.json", self.code)
        self.assertIn("RETAINED_PATHS = canonical_archive_paths", self.code)
        self.assertIn('(\"mkv\", \"id_srt\", \"tr_srt\")', self.code)
        self.assertIn("_actual != _expected", self.code)
        self.assertIn("exactly three canonical files remain", self.code)
        self.assertIn("final/Muhtemel Ask X.Bolum.mkv", self.markdown)
        self.assertIn("final/subtitles/Muhtemel Ask X.Bolum-id.srt", self.markdown)
        self.assertIn("final/subtitles/Muhtemel Ask X.Bolum-tr.srt", self.markdown)
        self.assertIn('_pilot_warnings = archive_result.get("pilot_warnings", [])', self.code)
        self.assertIn("PILOT REVIEW WARNINGS", self.code)

    def test_explains_v1_v2_migration_and_temporary_duplication(self) -> None:
        normalized = self.markdown.casefold()
        self.assertIn("version-1 report", normalized)
        self.assertIn("version-2 report", normalized)
        self.assertIn("*_finalization_report.json", normalized)
        self.assertIn("*_finalization_report_v2.json", normalized)
        self.assertIn("preserve v1 rollback", normalized)
        self.assertIn("system_v2_beta", normalized)
        self.assertIn("03_finalize_v2.ipynb", normalized)
        self.assertIn("source video and a final mkv may both exist before cleanup", normalized)
        self.assertIn("temporary duplication preserves resumability", normalized)
        self.assertIn("removed only after every verification passes", normalized)
        self.assertIn("infuse does not need duplicate sidecars", normalized)

    def test_v1_notebooks_do_not_load_beta_archive_code(self) -> None:
        project_root = self.notebook_path.parent
        for notebook_name in (
            "01_PREPARE.ipynb",
            "02_FINALIZE.ipynb",
            "03_ARCHIVE_CLEANUP.ipynb",
        ):
            with self.subTest(notebook=notebook_name):
                notebook = json.loads(
                    (project_root / notebook_name).read_text(encoding="utf-8")
                )
                code = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook["cells"]
                    if cell.get("cell_type") == "code"
                )
                self.assertNotIn("SYSTEM_V2_BETA", code)
                self.assertNotIn("src.archive_v2", code)
                self.assertNotIn("archive_episode_v2", code)


if __name__ == "__main__":
    unittest.main()
