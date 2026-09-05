from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.episode_archive import (
    ArchiveError,
    archive_episode,
    execute_media_only_cleanup,
    expected_cleanup_confirmation,
    file_record,
    load_verified_archive_inputs,
    plan_media_only_cleanup,
    validate_episode_root,
)


FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class EpisodeArchiveTests(unittest.TestCase):
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
        receipt = project / "ARCHIVE_REPORTS" / f"{self.episode_name}_ARCHIVE_RECEIPT.json"
        return root, receipt

    @staticmethod
    def _srt(text: str) -> bytes:
        return f"1\n00:00:00,100 --> 00:00:00,800\n{text}\n".encode("utf-8")

    def _populate(self, root: Path) -> dict[str, Path]:
        source = root / "source" / f"{self.episode_name}.mp4"
        id_srt = root / "final" / f"{self.episode_name}.id-final.srt"
        tr_srt = root / "final" / f"{self.episode_name}.tr-final.srt"
        id_sidecar = root / "source" / f"{self.episode_name}-id.srt"
        tr_sidecar = root / "source" / f"{self.episode_name}-tr.srt"
        source.write_bytes(b"synthetic-source-video")
        id_srt.write_bytes(self._srt("Halo."))
        tr_srt.write_bytes(self._srt("Merhaba."))
        id_sidecar.write_bytes(id_srt.read_bytes())
        tr_sidecar.write_bytes(tr_srt.read_bytes())
        (root / "prepare" / "audio.flac").write_bytes(b"audio-intermediate")
        (root / "prepare" / "schema.json").write_text("{}", encoding="utf-8")
        (root / "translation_input" / "pack.zip").write_bytes(b"pack")
        (root / "translation_output" / "translated.zip").write_bytes(b"translated")
        (root / "review" / "review.xlsx").write_bytes(b"review")
        report_path = root / "final" / f"{self.episode_name}_FINALIZATION_REPORT.json"
        report = {
            "report_version": 1,
            "status": "PASS",
            "episode": self.episode,
            "episode_name": self.episode_name,
            "input_files": {"source_video": file_record(source, root)},
            "outputs": {
                "id_srt": file_record(id_srt, root),
                "tr_srt": file_record(tr_srt, root),
                "id_infuse_sidecar": file_record(id_sidecar, root),
                "tr_infuse_sidecar": file_record(tr_sidecar, root),
            },
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {
            "source": source,
            "id_srt": id_srt,
            "tr_srt": tr_srt,
            "id_sidecar": id_sidecar,
            "tr_sidecar": tr_sidecar,
            "report": report_path,
            "mkv": root / "final" / f"{self.episode_name} - Endonezce + Turkce.mkv",
        }

    @staticmethod
    def _fake_mux_report(mkv: Path) -> dict[str, object]:
        return {
            "verified": True,
            "output_path": str(mkv),
            "output_size_bytes": mkv.stat().st_size,
            "video_audio_stream_copy": True,
            "subtitle_order": ["ind", "tur"],
            "indonesian_default": True,
            "turkish_default": False,
            "roundtrip": {"exact": True, "id_block_count": 1, "tr_block_count": 1},
            "stream_hashes": {
                "checked": True,
                "match": True,
                "source": [
                    {
                        "type": "video",
                        "ordinal": 0,
                        "algorithm": "sha256",
                        "digest": "a" * 64,
                    }
                ],
                "output": [
                    {
                        "type": "video",
                        "ordinal": 0,
                        "algorithm": "sha256",
                        "digest": "a" * 64,
                    }
                ],
            },
            "resumed": False,
        }

    def _fake_ensure(self, inputs: dict[str, object]) -> dict[str, object]:
        mkv = Path(inputs["mkv"])
        if not mkv.exists():
            mkv.write_bytes(b"verified-stream-copy-mkv")
        return self._fake_mux_report(mkv)

    def test_exact_episode_root_and_confirmation_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self._workspace(temporary)
            self.assertEqual(validate_episode_root(root, self.episode), root.resolve())
            wrong = root.with_name("Muhtemel Ask 12.Bolum")
            wrong.mkdir()
            with self.assertRaisesRegex(ArchiveError, "named exactly"):
                validate_episode_root(wrong, self.episode)
        self.assertEqual(
            expected_cleanup_confirmation(11), "DELETE EPISODE 11 INTERMEDIATES"
        )
        with self.assertRaises(ArchiveError):
            expected_cleanup_confirmation(True)

    def test_episode_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "Muhtemel_Ask_Subtitles"
            episodes = project / "EPISODES"
            episodes.mkdir(parents=True)
            actual = project / "actual"
            actual.mkdir()
            root = episodes / self.episode_name
            root.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ArchiveError, "symlinked episode folder"):
                validate_episode_root(root, self.episode)

    def test_project_ancestor_and_dangling_receipt_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            actual = temporary_root / "actual-project"
            root = actual / "EPISODES" / self.episode_name
            root.mkdir(parents=True)
            linked_project = temporary_root / "Muhtemel_Ask_Subtitles"
            linked_project.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ArchiveError, "symlinked project folder"):
                validate_episode_root(
                    linked_project / "EPISODES" / self.episode_name, self.episode
                )

        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            self._populate(root)
            receipt.parent.mkdir(parents=True)
            receipt.symlink_to(receipt.parent / "missing-receipt.json")
            with self.assertRaisesRegex(ArchiveError, "symlinked archive receipt"):
                load_verified_archive_inputs(
                    episode_root=root, receipt_path=receipt, episode=self.episode
                )

    def test_descendant_symlink_and_special_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            self._populate(root)
            outside = Path(temporary) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (root / "prepare" / "linked.txt").symlink_to(outside)
            inputs = load_verified_archive_inputs(
                episode_root=root, receipt_path=receipt, episode=self.episode
            )
            inputs["mkv"].write_bytes(b"mkv")
            with self.assertRaisesRegex(ArchiveError, "descendant symlink"):
                plan_media_only_cleanup(inputs, keep_source_video=False)
            (root / "prepare" / "linked.txt").unlink()
            fifo = root / "prepare" / "named-pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ArchiveError, "special filesystem entry"):
                plan_media_only_cleanup(inputs, keep_source_video=False)

    def test_pass_report_identity_size_and_hash_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            inputs = load_verified_archive_inputs(
                episode_root=root, receipt_path=receipt, episode=self.episode
            )
            self.assertEqual(inputs["source"], "finalization_report")
            paths["source"].write_bytes(b"tampered-same-ish")
            with self.assertRaisesRegex(ArchiveError, "mismatch"):
                load_verified_archive_inputs(
                    episode_root=root, receipt_path=receipt, episode=self.episode
                )

    def test_invalid_present_report_never_falls_back_to_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure):
                with contextlib.redirect_stdout(io.StringIO()):
                    archive_episode(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=False,
                    )
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["status"] = "FAIL"
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ArchiveError, "not PASS"):
                load_verified_archive_inputs(
                    episode_root=root, receipt_path=receipt, episode=self.episode
                )

    def test_plan_preserves_archive_media_and_deletes_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            paths["mkv"].write_bytes(b"mkv")
            extra_video = root / "source" / "bonus.webm"
            extra_video.write_bytes(b"bonus")
            staged_video = root / "final" / ".episode.run.staged.mkv"
            staged_video.write_bytes(b"staged")
            inputs = load_verified_archive_inputs(
                episode_root=root, receipt_path=receipt, episode=self.episode
            )
            plan = plan_media_only_cleanup(inputs, keep_source_video=False)
            deleted = {item["relative_path"] for item in plan["delete"]}
            protected = set(plan["protected"])
            self.assertIn(paths["source"].relative_to(root).as_posix(), deleted)
            self.assertIn(staged_video.relative_to(root).as_posix(), protected)
            self.assertIn(paths["mkv"].relative_to(root).as_posix(), protected)
            self.assertIn(extra_video.relative_to(root).as_posix(), protected)
            self.assertTrue(all(path.suffix == ".srt" for path in (
                paths["id_srt"], paths["tr_srt"], paths["id_sidecar"], paths["tr_sidecar"]
            )))
            self.assertEqual(
                plan["unverified_extra_videos"],
                ["final/.episode.run.staged.mkv", "source/bonus.webm"],
            )

    def test_dry_run_deletes_nothing_and_writes_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
            with patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = archive_episode(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=False,
                    )
            after_without_new_mkv = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path != paths["mkv"]
            )
            self.assertEqual(after_without_new_mkv, before)
            self.assertEqual(result["status"], "READY")
            self.assertGreater(result["cleanup_preview"]["delete_file_count"], 0)
            self.assertTrue(receipt.is_file())

    def test_wrong_confirmation_and_tree_change_delete_nothing(self) -> None:
        for mutation in ("wrong", "whitespace", "tree-change"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, receipt = self._workspace(temporary)
                paths = self._populate(root)
                candidate = root / "prepare" / "audio.flac"

                if mutation == "wrong":
                    confirmation = "DELETE SOMETHING ELSE"
                elif mutation == "whitespace":
                    confirmation = expected_cleanup_confirmation(self.episode) + " "
                else:
                    def confirmation(expected: str) -> str:
                        (root / "prepare" / "appeared-after-preview.json").write_text(
                            "{}", encoding="utf-8"
                        )
                        return expected

                with patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(ArchiveError):
                            archive_episode(
                                episode_root=root,
                                receipt_path=receipt,
                                episode=self.episode,
                                delete_intermediates=True,
                                confirmation=confirmation,
                            )
                self.assertTrue(candidate.is_file())
                self.assertTrue(paths["source"].is_file())
                self.assertTrue(paths["report"].is_file())

    def test_mutated_plan_cannot_add_a_protected_mkv_to_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure):
                with contextlib.redirect_stdout(io.StringIO()):
                    ready = archive_episode(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=False,
                    )
            inputs = load_verified_archive_inputs(
                episode_root=root, receipt_path=receipt, episode=self.episode
            )
            malicious = copy.deepcopy(ready["cleanup"]["plan"])
            mkv_relative = paths["mkv"].relative_to(root).as_posix()
            mkv_tree_record = next(
                item
                for item in malicious["tree"]["files"]
                if item["relative_path"] == mkv_relative
            )
            malicious["delete"].append(dict(mkv_tree_record))
            malicious["delete_file_count"] += 1
            malicious["delete_size_bytes"] += mkv_tree_record["size_bytes"]
            with self.assertRaisesRegex(ArchiveError, "changed after cleanup preview"):
                execute_media_only_cleanup(
                    inputs,
                    ready,
                    malicious,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            self.assertTrue(paths["mkv"].is_file())
            self.assertTrue(paths["source"].is_file())
            self.assertTrue(paths["id_srt"].is_file())

    def test_successful_cleanup_keeps_only_mkv_and_srts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with (
                patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure),
                patch("src.episode_archive._verify_ready_artifacts", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    keep_source_video=False,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            remaining = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                remaining,
                sorted(
                    [
                        paths["mkv"].relative_to(root).as_posix(),
                        paths["id_srt"].relative_to(root).as_posix(),
                        paths["tr_srt"].relative_to(root).as_posix(),
                        paths["id_sidecar"].relative_to(root).as_posix(),
                        paths["tr_sidecar"].relative_to(root).as_posix(),
                    ]
                ),
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["source_removed"])
            self.assertGreater(result["cleanup"]["result"]["deleted_size_bytes"], 0)
            self.assertTrue(receipt.is_file())

    def test_keep_source_video_policy_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with (
                patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure),
                patch("src.episode_archive._verify_ready_artifacts", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    keep_source_video=True,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            self.assertTrue(paths["source"].is_file())
            self.assertFalse(result["source_removed"])
            self.assertTrue(paths["mkv"].is_file())

    def test_keep_source_video_disappearance_blocks_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)

            def remove_source_after_preview(expected: str) -> str:
                paths["source"].unlink()
                return expected

            with patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ArchiveError, "tree changed"):
                        archive_episode(
                            episode_root=root,
                            receipt_path=receipt,
                            episode=self.episode,
                            delete_intermediates=True,
                            keep_source_video=True,
                            confirmation=remove_source_after_preview,
                        )
            self.assertTrue((root / "prepare" / "audio.flac").is_file())
            self.assertTrue(paths["report"].is_file())

    def test_completed_keep_source_archive_can_later_remove_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with (
                patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure),
                patch("src.episode_archive._verify_ready_artifacts", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                first = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    keep_source_video=True,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            self.assertEqual(first["status"], "PASS")
            self.assertTrue(paths["source"].is_file())
            with (
                patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure),
                patch("src.episode_archive._verify_ready_artifacts", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                second = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    keep_source_video=False,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            self.assertEqual(second["status"], "PASS")
            self.assertFalse(paths["source"].exists())

    def test_ready_receipt_recovers_a_partial_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            with patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure):
                with contextlib.redirect_stdout(io.StringIO()):
                    archive_episode(
                        episode_root=root,
                        receipt_path=receipt,
                        episode=self.episode,
                        delete_intermediates=False,
                        keep_source_video=False,
                    )
            # Simulate a stopped cleanup after the report and source are gone.
            paths["report"].unlink()
            paths["source"].unlink()
            (root / "prepare" / "schema.json").unlink()
            with (
                patch("src.episode_archive.ensure_verified_mkv", side_effect=self._fake_ensure),
                patch("src.episode_archive._verify_ready_artifacts", return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    keep_source_video=False,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            self.assertEqual(result["status"], "PASS")
            self.assertFalse((root / "prepare" / "audio.flac").exists())

    def test_notebook_contract_compiles_and_prompts_after_archive_preview(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "03_ARCHIVE_CLEANUP.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(
                    "".join(cell.get("source", [])),
                    f"03_ARCHIVE_CLEANUP.ipynb:cell{index}",
                    "exec",
                )
        self.assertIn("DELETE_INTERMEDIATES = False", code)
        self.assertIn("KEEP_SOURCE_VIDEO = False", code)
        self.assertIn("confirmation=request_cleanup_confirmation", code)
        self.assertIn("del sys.modules[_module_name]", code)
        self.assertIn("Loaded archive module from an unexpected path", code)
        self.assertIn("archive_episode(", code)
        self.assertNotIn("rmtree", code)
        self.assertNotIn("Trash", code.split("archive_episode(", 1)[0])


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg and ffprobe are required")
class EpisodeArchiveFFmpegIntegrationTests(unittest.TestCase):
    episode = EpisodeArchiveTests.episode
    episode_name = EpisodeArchiveTests.episode_name
    _workspace = EpisodeArchiveTests._workspace
    _srt = staticmethod(EpisodeArchiveTests._srt)
    _populate = EpisodeArchiveTests._populate

    def _make_media(self, path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=160x90:rate=25",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-t",
                "1",
                "-c:v",
                "mpeg4",
                "-q:v",
                "8",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_real_stream_copy_archive_and_media_only_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt = self._workspace(temporary)
            paths = self._populate(root)
            self._make_media(paths["source"])
            # Rebuild the report because the real source bytes replaced the fixture.
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["input_files"]["source_video"] = file_record(paths["source"], root)
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=True,
                    keep_source_video=False,
                    confirmation=expected_cleanup_confirmation(self.episode),
                )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(paths["mkv"].is_file())
            self.assertFalse(paths["source"].exists())
            self.assertTrue(
                all(
                    path.suffix.casefold() in {".mkv", ".srt"}
                    for path in root.rglob("*")
                    if path.is_file()
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rerun = archive_episode(
                    episode_root=root,
                    receipt_path=receipt,
                    episode=self.episode,
                    delete_intermediates=False,
                    keep_source_video=False,
                )
            self.assertTrue(rerun["no_op"])


if __name__ == "__main__":
    unittest.main()

