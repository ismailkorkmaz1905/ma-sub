from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARE_NOTEBOOK = PROJECT_ROOT / "01_PREPARE.ipynb"
FINALIZE_NOTEBOOK = PROJECT_ROOT / "02_FINALIZE.ipynb"


class FinalizeNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepare_notebook = json.loads(
            PREPARE_NOTEBOOK.read_text(encoding="utf-8")
        )
        cls.notebook = json.loads(FINALIZE_NOTEBOOK.read_text(encoding="utf-8"))
        cls.finalize_source = "".join(cls.notebook["cells"][12]["source"])

    def test_all_code_cells_compile(self) -> None:
        for notebook_name, notebook in (
            ("01_PREPARE.ipynb", self.prepare_notebook),
            ("02_FINALIZE.ipynb", self.notebook),
        ):
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "code":
                    compile(
                        "".join(cell.get("source", [])),
                        f"{notebook_name}:cell{index}",
                        "exec",
                    )

    def test_name_variant_maps_stay_separate_across_notebooks(self) -> None:
        prepare_source = "".join(
            "".join(cell.get("source", []))
            for cell in self.prepare_notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        finalize_source = "".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn(
            '"source_name_variants": names_config.get("source_variants", {})',
            prepare_source,
        )
        self.assertIn(
            '"allah_only_semoga_is_invalid": religious_config.get(',
            prepare_source,
        )
        self.assertIn(
            'source_name_variants=names_config.get("source_variants", {})',
            finalize_source,
        )
        self.assertIn(
            'forbidden_name_variants=names_config.get("forbidden_variants", {})',
            finalize_source,
        )

    def _prepare_cookie_sources(self) -> tuple[str, str]:
        code_cells = [
            "".join(cell.get("source", []))
            for cell in self.prepare_notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cookie_helper = next(
            source for source in code_cells if "def upload_youtube_cookies():" in source
        )
        download_cell = next(
            source for source in code_cells if "download = download_source(" in source
        )
        return cookie_helper, download_cell

    def _execute_cookie_helpers(self, root: Path) -> dict[str, object]:
        cookie_helper, _ = self._prepare_cookie_sources()
        runtime_root = root / "runtime"
        runtime_root.mkdir()
        namespace: dict[str, object] = {
            "Path": Path,
            "os": os,
            "shutil": shutil,
            "series_config": {"drive_root": str(root / "drive")},
        }
        exec(cookie_helper, namespace)
        namespace["YOUTUBE_COOKIE_RUNTIME_ROOT"] = runtime_root
        return namespace

    def _execute_download_flow(
        self,
        root: Path,
        *,
        download_side_effect: list[object],
        load_saved: Mock,
        upload: Mock,
        save: Mock,
        discard: Mock,
    ) -> Mock:
        _, download_cell = self._prepare_cookie_sources()
        source_dir = root / "source"
        prepare_dir = root / "prepare"
        source_dir.mkdir()
        prepare_dir.mkdir()
        namespace = {
            "SOURCE_URL": "https://www.youtube.com/watch?v=synthetic",
            "DIRS": {"source": source_dir, "prepare": prepare_dir},
            "FORCE_REBUILD": False,
            "EPISODE_NAME": "Muhtemel Ask 10.Bolum",
            "load_saved_youtube_cookies": load_saved,
            "upload_youtube_cookies": upload,
            "save_youtube_cookies_to_drive": save,
            "discard_youtube_cookies": discard,
        }
        with (
            patch("src.download.download_source", side_effect=download_side_effect) as mocked,
            patch("src.media.save_source_metadata", return_value={"sha256": "source-sha"}),
            patch("src.media.extract_audio", return_value=SimpleNamespace(resumed=False)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exec(download_cell, namespace)
        return mocked

    @staticmethod
    def _youtube_cookie_payload(secret: str = "cookie-secret") -> bytes:
        return (
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\t"
            f"{secret}\n"
        ).encode("utf-8")

    def test_prepare_uses_drive_backed_cookie_fallback(self) -> None:
        cookie_helper, download_cell = self._prepare_cookie_sources()

        self.assertIn("output_stem=EPISODE_NAME", download_cell)
        self.assertIn("YouTubeAuthenticationError", download_cell)
        self.assertIn("except YouTubeAuthenticationError:", download_cell)
        self.assertIn("cookies_file=youtube_cookie_file", download_cell)
        self.assertNotIn("cookies_file=YOUTUBE_COOKIE_DRIVE_PATH", download_cell)
        self.assertIn("load_saved_youtube_cookies()", download_cell)
        self.assertIn("save_youtube_cookies_to_drive(youtube_cookie_file)", download_cell)
        self.assertIn("finally:", download_cell)
        self.assertIn("discard_youtube_cookies(youtube_cookie_file)", download_cell)
        fresh_branch = download_cell[download_cell.index("if download is None:") :]
        stored_branch = download_cell[: download_cell.index("if download is None:")]
        self.assertNotIn("save_youtube_cookies_to_drive(", stored_branch)
        fresh_download = fresh_branch.index("download = download_source(")
        fresh_save = fresh_branch.index("save_youtube_cookies_to_drive(")
        self.assertLess(fresh_download, fresh_save)
        self.assertIn("files.upload()", cookie_helper)
        self.assertIn('YOUTUBE_COOKIE_RUNTIME_ROOT = Path("/content")', cookie_helper)
        self.assertIn('Path(series_config["drive_root"]) / "PRIVATE"', cookie_helper)
        self.assertIn("stage_youtube_cookies(YOUTUBE_COOKIE_DRIVE_PATH.read_bytes())", cookie_helper)
        self.assertIn("os.replace(temporary_path, YOUTUBE_COOKIE_DRIVE_PATH)", cookie_helper)
        self.assertIn("0o600", cookie_helper)
        self.assertIn("with warnings.catch_warnings():", cookie_helper)
        self.assertIn('raise ValueError("The cookies.txt file is invalid") from None', cookie_helper)
        self.assertIn("shutil.rmtree(auth_dir, ignore_errors=True)", cookie_helper)
        self.assertNotIn("SYSTEM_ROOT", cookie_helper)
        self.assertNotIn("EPISODE_ROOT", cookie_helper)

    def test_cookie_helpers_copy_saved_drive_cookie_to_private_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = self._execute_cookie_helpers(root)
            drive_path = Path(namespace["YOUTUBE_COOKIE_DRIVE_PATH"])
            drive_path.parent.mkdir(parents=True)
            payload = self._youtube_cookie_payload()
            drive_path.write_bytes(payload)

            with contextlib.redirect_stdout(io.StringIO()):
                runtime_path = namespace["load_saved_youtube_cookies"]()
            self.assertIsNotNone(runtime_path)
            runtime_path = Path(runtime_path)
            self.assertNotEqual(runtime_path, drive_path)
            self.assertEqual(runtime_path.read_bytes(), payload)
            self.assertEqual(runtime_path.parent.parent, root / "runtime")
            self.assertEqual(stat.S_IMODE(runtime_path.stat().st_mode), 0o600)

            with contextlib.redirect_stdout(io.StringIO()):
                saved = namespace["save_youtube_cookies_to_drive"](runtime_path)
            self.assertTrue(saved)
            namespace["discard_youtube_cookies"](runtime_path)
            self.assertFalse(runtime_path.parent.exists())
            self.assertEqual(drive_path.read_bytes(), payload)

    def test_cookie_helpers_reject_non_youtube_saved_file_without_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = self._execute_cookie_helpers(root)
            drive_path = Path(namespace["YOUTUBE_COOKIE_DRIVE_PATH"])
            drive_path.parent.mkdir(parents=True)
            secret = "must-not-appear-in-output"
            payload = self._youtube_cookie_payload(secret).replace(
                b".youtube.com", b".google.com"
            )
            drive_path.write_bytes(payload)

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                runtime_path = namespace["load_saved_youtube_cookies"]()
            self.assertIsNone(runtime_path)
            self.assertNotIn(secret, captured.getvalue())
            self.assertNotIn(str(drive_path), captured.getvalue())
            self.assertEqual(list((root / "runtime").iterdir()), [])

    def test_cookie_save_failure_redacts_path_and_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            namespace = self._execute_cookie_helpers(root)
            secret = "drive-save-secret"
            runtime_path = namespace["stage_youtube_cookies"](
                self._youtube_cookie_payload(secret)
            )
            invalid_target = root / "drive" / "PRIVATE" / "directory-target"
            invalid_target.mkdir(parents=True)
            namespace["YOUTUBE_COOKIE_DRIVE_PATH"] = invalid_target

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                saved = namespace["save_youtube_cookies_to_drive"](runtime_path)
            namespace["discard_youtube_cookies"](runtime_path)
            self.assertFalse(saved)
            self.assertNotIn(secret, captured.getvalue())
            self.assertNotIn(str(invalid_target), captured.getvalue())

    def test_anonymous_success_never_reads_or_writes_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = SimpleNamespace(
                video_path=root / "source.mkv", metadata={}, resumed=False
            )
            load_saved = Mock(side_effect=AssertionError("saved cookie was read"))
            upload = Mock(side_effect=AssertionError("upload was opened"))
            save = Mock(side_effect=AssertionError("cookie was saved"))
            discard = Mock(side_effect=AssertionError("cookie was discarded"))

            mocked = self._execute_download_flow(
                root,
                download_side_effect=[result],
                load_saved=load_saved,
                upload=upload,
                save=save,
                discard=discard,
            )
            self.assertEqual(mocked.call_count, 1)
            self.assertNotIn("cookies_file", mocked.call_args.kwargs)
            load_saved.assert_not_called()
            upload.assert_not_called()
            save.assert_not_called()
            discard.assert_not_called()

    def test_stale_saved_cookie_falls_back_once_and_persists_fresh_cookie(self) -> None:
        from src.download import YouTubeAuthenticationError

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stored = root / "runtime" / "stored" / "cookies.txt"
            fresh = root / "runtime" / "fresh" / "cookies.txt"
            result = SimpleNamespace(
                video_path=root / "source.mkv", metadata={}, resumed=False
            )
            load_saved = Mock(return_value=stored)
            upload = Mock(return_value=fresh)
            save = Mock(return_value=True)
            discard = Mock()
            auth_error = YouTubeAuthenticationError("browser verification required")

            mocked = self._execute_download_flow(
                root,
                download_side_effect=[auth_error, auth_error, result],
                load_saved=load_saved,
                upload=upload,
                save=save,
                discard=discard,
            )
            self.assertEqual(mocked.call_count, 3)
            self.assertNotIn("cookies_file", mocked.call_args_list[0].kwargs)
            self.assertEqual(mocked.call_args_list[1].kwargs["cookies_file"], stored)
            self.assertEqual(mocked.call_args_list[2].kwargs["cookies_file"], fresh)
            load_saved.assert_called_once_with()
            upload.assert_called_once_with()
            save.assert_called_once_with(fresh)
            self.assertEqual(discard.call_args_list[0].args, (stored,))
            self.assertEqual(discard.call_args_list[1].args, (fresh,))

    def test_rejected_fresh_cookie_is_not_persisted_or_retried(self) -> None:
        from src.download import YouTubeAuthenticationError

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stored = root / "runtime" / "stored" / "cookies.txt"
            fresh = root / "runtime" / "fresh" / "cookies.txt"
            load_saved = Mock(return_value=stored)
            upload = Mock(return_value=fresh)
            save = Mock(return_value=True)
            discard = Mock()
            auth_error = YouTubeAuthenticationError("browser verification required")

            with self.assertRaises(YouTubeAuthenticationError):
                self._execute_download_flow(
                    root,
                    download_side_effect=[auth_error, auth_error, auth_error],
                    load_saved=load_saved,
                    upload=upload,
                    save=save,
                    discard=discard,
                )
            upload.assert_called_once_with()
            save.assert_not_called()
            self.assertEqual(discard.call_args_list[0].args, (stored,))
            self.assertEqual(discard.call_args_list[1].args, (fresh,))

    def test_input_hash_guard_covers_resume_publish_and_commit(self) -> None:
        source = self.finalize_source
        self.assertIn("def assert_current_input_hashes(context):", source)
        self.assertIn(
            'assert_current_input_hashes("while accepting a prior finalization")',
            source,
        )
        publish_guard = source.index(
            'assert_current_input_hashes("before publishing final outputs")'
        )
        invalidate = source.index("os.replace(finalization_report_path, prior_report_backup)")
        first_publish = source.index("atomic_publish(tr_stage, tr_final_path)")
        self.assertLess(publish_guard, invalidate)
        self.assertLess(publish_guard, first_publish)
        commit_guard = source.index(
            'assert_current_input_hashes("before writing the PASS commit marker")'
        )
        report_publish = source.index(
            "atomic_write_json(finalization_report_path, finalization_report)"
        )
        self.assertLess(commit_guard, report_publish)
        self.assertIn('"input_files": input_file_records', source)

    def test_non_mkv_run_cannot_accept_or_leave_a_canonical_mkv(self) -> None:
        source = self.finalize_source
        self.assertIn("elif mkv_final_path.exists():", source)
        self.assertIn("if not CREATE_MKV and mkv_final_path.exists():", source)
        self.assertIn("os.replace(mkv_final_path, superseded_mkv_path)", source)
        self.assertIn("superseded_mkv_path.unlink(missing_ok=True)", source)


if __name__ == "__main__":
    unittest.main()

