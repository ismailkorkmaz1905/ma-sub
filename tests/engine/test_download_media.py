from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import types
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

from mas.engine.download import (
    DOWNLOAD_VALIDATION_VERSION,
    DownloadError,
    YouTubeAuthenticationError,
    atomic_write_json,
    download_source,
    sha256_file,
    sha256_json,
    write_stage_marker,
)
from mas.engine.media import (
    AUDIO_ALIGNMENT_VERSION,
    MediaError,
    extract_audio,
    validate_audio,
    validate_audio_marker,
    verify_media_readable,
)


FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *command],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_source(path: Path, *, duration: int = 3, audio_offset: int = 0) -> None:
    if audio_offset:
        audio_duration = duration - audio_offset
        inputs = [
            "-f",
            "lavfi",
            "-t",
            str(duration),
            "-i",
            "color=black:size=160x90:rate=25",
            "-itsoffset",
            str(audio_offset),
            "-f",
            "lavfi",
            "-t",
            str(audio_duration),
            "-i",
            "sine=frequency=440:sample_rate=48000",
        ]
    else:
        inputs = [
            "-f",
            "lavfi",
            "-i",
            "color=black:size=160x90:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            str(duration),
        ]
    _run_ffmpeg(
        [
            *inputs,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "mpeg4",
            "-q:v",
            "8",
            "-c:a",
            "aac",
            *( ["-movflags", "+faststart"] if path.suffix == ".mp4" else [] ),
            str(path),
        ]
    )


def _pcm_rms(path: Path, *, start_seconds: float, duration_seconds: float) -> float:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    samples = array("h")
    samples.frombytes(process.stdout)
    if not samples:
        raise AssertionError("ffmpeg decoded no PCM samples")
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class _FakeYtDlpState:
    def __init__(self, fixture: Path) -> None:
        self.fixture = fixture
        self.caption_available = False
        self.fail_first_media = False
        self.media_download_calls = 0
        self.caption_download_calls = 0
        self.saw_partial = False
        self.caption_error: str | None = None
        self.media_error: str | None = None
        self.ydl_options: list[dict[str, object]] = []

    def info(self) -> dict[str, object]:
        tracks = {"tr": [{"ext": "vtt"}]} if self.caption_available else {}
        return {
            "id": "synthetic-video",
            "title": "Synthetic episode",
            "duration": 3.0,
            "ext": "mkv",
            "webpage_url": "https://www.youtube.com/watch?v=synthetic",
            "subtitles": tracks,
            "automatic_captions": {},
        }


def _fake_yt_dlp_module(state: _FakeYtDlpState) -> object:
    class YoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options
            state.ydl_options.append(dict(options))

        def __enter__(self) -> "YoutubeDL":
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            if not download:
                return state.info()
            outtmpl = str(self.options.get("outtmpl", ""))
            if self.options.get("skip_download"):
                state.caption_download_calls += 1
                if state.caption_error is not None:
                    raise RuntimeError(state.caption_error)
                if state.caption_available:
                    caption = Path(outtmpl.replace("%(ext)s", "tr.vtt"))
                    caption.parent.mkdir(parents=True, exist_ok=True)
                    caption.write_text(
                        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nMerhaba.\n",
                        encoding="utf-8",
                    )
                return state.info()

            state.media_download_calls += 1
            if state.media_error is not None:
                raise RuntimeError(state.media_error)
            target = Path(outtmpl.replace("%(ext)s", "mkv"))
            partial = target.with_name(target.name + ".part")
            state.saw_partial = state.saw_partial or partial.is_file()
            if state.fail_first_media and state.media_download_calls == 1:
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(b"resumable-partial")
                raise RuntimeError("synthetic interrupted transfer")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(state.fixture, target)
            partial.unlink(missing_ok=True)
            return state.info()

    return types.SimpleNamespace(YoutubeDL=YoutubeDL)


def _write_youtube_cookie(path: Path, *, secret: str = "test-secret") -> None:
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSAPISID\t{secret}\n",
        encoding="utf-8",
    )


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg and ffprobe are required")
class MediaIntegrityTests(unittest.TestCase):
    def test_audio_preserves_nonzero_source_playback_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "delayed.mp4"
            _make_source(source, duration=5, audio_offset=2)

            result = extract_audio(source, root / "prepare")
            timeline = result.metadata["source_timeline"]

            self.assertEqual(timeline["alignment_version"], AUDIO_ALIGNMENT_VERSION)
            self.assertGreaterEqual(timeline["audio_offset_ms"], 1_900)
            self.assertLessEqual(timeline["audio_offset_ms"], 2_050)
            self.assertLessEqual(abs(result.metadata["duration_ms"] - 5_000), 250)
            self.assertLess(
                _pcm_rms(result.audio_path, start_seconds=0.1, duration_seconds=1.5),
                5.0,
            )
            self.assertGreater(
                _pcm_rms(result.audio_path, start_seconds=2.2, duration_seconds=0.5),
                100.0,
            )
            self.assertTrue(extract_audio(source, root / "prepare").resumed)

    def test_truncated_faststart_source_fails_eof_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = root / "full.mp4"
            truncated = root / "truncated.mp4"
            _make_source(full, duration=8)
            payload = full.read_bytes()
            truncated.write_bytes(payload[: len(payload) // 2])

            with self.assertRaises(MediaError):
                verify_media_readable(truncated)

    def test_audio_duration_tolerance_rejects_one_second_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "short.flac"
            _run_ffmpeg(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=16000",
                    "-t",
                    "2",
                    "-ac",
                    "1",
                    "-c:a",
                    "flac",
                    str(audio),
                ]
            )
            with self.assertRaises(MediaError):
                validate_audio(audio, expected_duration_ms=3_000)

    def test_supplied_source_digest_must_match_current_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mkv"
            _make_source(source)
            with self.assertRaisesRegex(MediaError, "does not match"):
                extract_audio(source, root / "prepare", source_sha256="a" * 64)

    def test_audio_marker_without_alignment_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mkv"
            prepare = root / "prepare"
            _make_source(source)
            result = extract_audio(source, prepare)
            marker = json.loads(result.marker_path.read_text(encoding="utf-8"))
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            metadata.pop("source_timeline")
            atomic_write_json(result.metadata_path, metadata)
            write_stage_marker(
                result.marker_path,
                stage="audio",
                input_sha256=marker["input_sha256"],
                outputs={"audio": result.audio_path, "metadata": result.metadata_path},
            )

            self.assertFalse(
                validate_audio_marker(
                    result.marker_path,
                    source_sha256=sha256_file(source),
                )
            )


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg and ffprobe are required")
class DownloadResumeTests(unittest.TestCase):
    url = "https://www.youtube.com/watch?v=synthetic"

    def test_cookie_reaches_media_and_caption_without_persisting_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            cookie = root / "private-youtube-cookie.txt"
            secret = "never-persist-this-cookie-value"
            _make_source(fixture)
            _write_youtube_cookie(cookie, secret=secret)
            state = _FakeYtDlpState(fixture)
            state.caption_available = True

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                result = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    cookies_file=cookie,
                )

            self.assertEqual(len(state.ydl_options), 2)
            self.assertTrue(
                all(options.get("remote_components") == {"ejs:github"} for options in state.ydl_options)
            )
            self.assertTrue(all(options.get("js_runtimes") for options in state.ydl_options))
            self.assertTrue(
                all(options.get("cookiefile") == str(cookie.resolve()) for options in state.ydl_options)
            )
            persisted = result.metadata_path.read_text(encoding="utf-8")
            persisted += result.marker_path.read_text(encoding="utf-8")
            self.assertNotIn(str(cookie), persisted)
            self.assertNotIn(secret, persisted)

    def test_cookie_reaches_resumed_caption_probe_and_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            cookie = root / "youtube-cookies.txt"
            _make_source(fixture)
            _write_youtube_cookie(cookie)
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                first = download_source(self.url, source_dir, attempts=1)
                self.assertIsNone(first.captions_path)
                state.caption_available = True
                option_count = len(state.ydl_options)
                second = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    cookies_file=cookie,
                )

            retry_options = state.ydl_options[option_count:]
            self.assertTrue(second.resumed)
            self.assertEqual(state.media_download_calls, 1)
            self.assertEqual(len(retry_options), 2)
            self.assertTrue(
                all(options.get("remote_components") == {"ejs:github"} for options in retry_options)
            )
            self.assertTrue(
                all(options.get("cookiefile") == str(cookie.resolve()) for options in retry_options)
            )

    def test_cookie_path_does_not_change_resume_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            first_cookie = root / "first.txt"
            second_cookie = root / "second.txt"
            _make_source(fixture)
            _write_youtube_cookie(first_cookie, secret="first-secret")
            _write_youtube_cookie(second_cookie, secret="second-secret")
            state = _FakeYtDlpState(fixture)
            state.caption_available = True

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                first = download_source(
                    self.url, source_dir, attempts=1, cookies_file=first_cookie
                )
                second = download_source(
                    self.url, source_dir, attempts=1, cookies_file=second_cookie
                )

            self.assertFalse(first.resumed)
            self.assertTrue(second.resumed)
            self.assertEqual(state.media_download_calls, 1)

    def test_missing_cookie_is_not_required_for_verified_marker_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            cookie = root / "temporary-cookie.txt"
            _make_source(fixture)
            _write_youtube_cookie(cookie)
            state = _FakeYtDlpState(fixture)
            state.caption_available = True

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                download_source(self.url, source_dir, attempts=1, cookies_file=cookie)
                cookie.unlink()
                resumed = download_source(
                    self.url, source_dir, attempts=1, cookies_file=cookie
                )

            self.assertTrue(resumed.resumed)
            self.assertEqual(state.media_download_calls, 1)

    def test_youtube_bot_auth_error_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)
            state.media_error = (
                "Sign in to confirm you're not a bot. Use --cookies-from-browser "
                "or --cookies for the authentication."
            )

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                with self.assertRaisesRegex(
                    YouTubeAuthenticationError, "fresh Netscape-format cookies.txt"
                ):
                    download_source(self.url, source_dir, attempts=3)

            self.assertEqual(state.media_download_calls, 1)

    def test_invalid_cookie_file_fails_before_yt_dlp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            malformed = root / "cookies.json"
            malformed.write_text('{"cookies": []}', encoding="utf-8")

            with self.assertRaisesRegex(DownloadError, "Netscape/Mozilla"):
                download_source(self.url, source_dir, attempts=1, cookies_file=malformed)
            with self.assertRaisesRegex(DownloadError, "was not found"):
                download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    cookies_file=root / "missing.txt",
                )

    def test_authenticated_caption_failure_redacts_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            cookie = root / "private-cookie.txt"
            secret = "caption-cookie-secret-must-not-persist"
            _make_source(fixture)
            _write_youtube_cookie(cookie, secret=secret)
            state = _FakeYtDlpState(fixture)
            state.caption_available = True
            state.caption_error = f"synthetic failure {cookie.resolve()} {secret}"

            with (
                patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)),
                self.assertLogs("mas.engine.download", level="WARNING") as captured,
            ):
                result = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    cookies_file=cookie,
                )

            persisted = result.metadata_path.read_text(encoding="utf-8")
            persisted += result.marker_path.read_text(encoding="utf-8")
            logged = "\n".join(captured.output)
            self.assertNotIn(str(cookie.resolve()), persisted)
            self.assertNotIn(secret, persisted)
            self.assertNotIn(str(cookie.resolve()), logged)
            self.assertNotIn(secret, logged)
            self.assertIn('"error_type": "RuntimeError"', persisted)

    def test_authenticated_media_failure_redacts_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            cookie = root / "private-cookie.txt"
            secret = "media-cookie-secret-must-not-leak"
            _make_source(fixture)
            _write_youtube_cookie(cookie, secret=secret)
            state = _FakeYtDlpState(fixture)
            state.media_error = f"synthetic failure {cookie.resolve()} {secret}"

            with (
                patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)),
                self.assertLogs("mas.engine.download", level="WARNING") as captured,
            ):
                with self.assertRaises(DownloadError) as raised:
                    download_source(
                        self.url,
                        source_dir,
                        attempts=2,
                        cookies_file=cookie,
                    )

            request_path = source_dir / ".yt-dlp-work" / "request.json"
            persisted = request_path.read_text(encoding="utf-8")
            logged = "\n".join(captured.output)
            visible_error = str(raised.exception)
            self.assertNotIn(str(cookie.resolve()), persisted)
            self.assertNotIn(secret, persisted)
            self.assertNotIn(str(cookie.resolve()), logged)
            self.assertNotIn(secret, logged)
            self.assertNotIn(str(cookie.resolve()), visible_error)
            self.assertNotIn(secret, visible_error)

    def test_episode_named_source_resumes_with_marker_verified_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            episode_stem = "Muhtemel Ask 12.Bolum"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                first = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    output_stem=episode_stem,
                )
                second = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    output_stem=episode_stem,
                )

            expected_video = source_dir / f"{episode_stem}.mkv"
            self.assertEqual(first.video_path, expected_video)
            self.assertEqual(second.video_path, expected_video)
            self.assertTrue(second.resumed)
            self.assertEqual(state.media_download_calls, 1)
            self.assertEqual(first.metadata["source_file"], expected_video.name)
            marker = json.loads(second.marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["outputs"]["video"]["path"], str(expected_video.resolve()))
            self.assertEqual(marker["details"]["output_stem"], episode_stem)

    def test_legacy_source_marker_still_resumes_when_episode_stem_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                legacy = download_source(self.url, source_dir, attempts=1)
                resumed = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    output_stem="Muhtemel Ask 11.Bolum",
                )

            self.assertEqual(legacy.video_path.name, "source.mkv")
            self.assertEqual(resumed.video_path, legacy.video_path)
            self.assertTrue(resumed.resumed)
            self.assertEqual(state.media_download_calls, 1)

    def test_output_stem_rejects_paths_and_media_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "source"
            for unsafe in ("../episode", "nested/episode", "episode.mkv", ".hidden"):
                with self.subTest(output_stem=unsafe):
                    with self.assertRaises(ValueError):
                        download_source(
                            self.url,
                            source_dir,
                            attempts=1,
                            output_stem=unsafe,
                        )

    def test_stable_workspace_reuses_partial_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)
            state.fail_first_media = True

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                with self.assertRaises(DownloadError):
                    download_source(self.url, source_dir, attempts=1)
                self.assertTrue((source_dir / ".yt-dlp-work/source.mkv.part").is_file())
                result = download_source(self.url, source_dir, attempts=1)

            self.assertTrue(state.saw_partial)
            self.assertTrue(result.video_path.is_file())
            self.assertFalse((source_dir / ".yt-dlp-work").exists())

    def test_missing_caption_retries_without_redownloading_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                first = download_source(self.url, source_dir, attempts=1)
                self.assertIsNone(first.captions_path)
                state.caption_available = True
                second = download_source(self.url, source_dir, attempts=1)

            self.assertTrue(second.resumed)
            self.assertEqual(state.media_download_calls, 1)
            self.assertTrue((source_dir / "youtube.tr.vtt").is_file())
            marker = json.loads(second.marker_path.read_text(encoding="utf-8"))
            self.assertIn("captions", marker["outputs"])

    def test_corrupt_optional_caption_is_dropped_when_retry_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)
            state.caption_available = True

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                first = download_source(self.url, source_dir, attempts=1)
                self.assertIsNotNone(first.captions_path)
                (source_dir / "youtube.tr.vtt").write_text("corrupt", encoding="utf-8")
                state.caption_available = False
                second = download_source(self.url, source_dir, attempts=1)

            self.assertTrue(second.resumed)
            self.assertIsNone(second.captions_path)
            self.assertEqual(state.media_download_calls, 1)
            self.assertFalse((source_dir / "youtube.tr.vtt").exists())
            marker = json.loads(second.marker_path.read_text(encoding="utf-8"))
            self.assertNotIn("captions", marker["outputs"])

    def test_publish_removes_only_stale_workflow_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            source_dir.mkdir()
            _make_source(fixture)
            (source_dir / "source.mp4").write_bytes(b"old-source")
            (source_dir / "youtube.tr.vtt").write_text("old caption", encoding="utf-8")
            (source_dir / "source-id.srt").write_text("old sidecar", encoding="utf-8")
            (source_dir / "source-tr.srt").write_text("old sidecar", encoding="utf-8")
            (source_dir / "source.media.json").write_text("{}", encoding="utf-8")
            (source_dir / "keep-me.txt").write_text("user file", encoding="utf-8")
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                result = download_source(self.url, source_dir, attempts=1)

            self.assertEqual(result.video_path.name, "source.mkv")
            self.assertFalse((source_dir / "source.mp4").exists())
            self.assertFalse((source_dir / "youtube.tr.vtt").exists())
            self.assertFalse((source_dir / "source-id.srt").exists())
            self.assertFalse((source_dir / "source-tr.srt").exists())
            self.assertFalse((source_dir / "source.media.json").exists())
            self.assertEqual((source_dir / "keep-me.txt").read_text(), "user file")

    def test_episode_named_publish_retires_only_legacy_and_matching_stem_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            source_dir.mkdir()
            episode_stem = "Muhtemel Ask 12.Bolum"
            _make_source(fixture)
            (source_dir / "source.mp4").write_bytes(b"legacy source")
            (source_dir / "source-id.srt").write_text("legacy", encoding="utf-8")
            (source_dir / f"{episode_stem}.mp4").write_bytes(b"stale episode source")
            (source_dir / f"{episode_stem}-id.srt").write_text("stale", encoding="utf-8")
            (source_dir / f"{episode_stem}-tr.srt").write_text("stale", encoding="utf-8")
            unrelated = source_dir / "Another Episode.mkv"
            unrelated.write_bytes(b"user-owned")
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                result = download_source(
                    self.url,
                    source_dir,
                    attempts=1,
                    output_stem=episode_stem,
                )

            self.assertEqual(result.video_path.name, f"{episode_stem}.mkv")
            self.assertFalse((source_dir / "source.mp4").exists())
            self.assertFalse((source_dir / "source-id.srt").exists())
            self.assertFalse((source_dir / f"{episode_stem}.mp4").exists())
            self.assertFalse((source_dir / f"{episode_stem}-id.srt").exists())
            self.assertFalse((source_dir / f"{episode_stem}-tr.srt").exists())
            self.assertEqual(unrelated.read_bytes(), b"user-owned")

    def test_interrupted_marker_commit_leaves_no_uncommitted_final_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            _make_source(fixture)
            state = _FakeYtDlpState(fixture)

            with (
                patch(
                    "mas.engine.download._import_yt_dlp",
                    return_value=_fake_yt_dlp_module(state),
                ),
                patch("mas.engine.download.write_stage_marker", side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    download_source(self.url, source_dir, attempts=1)

            self.assertFalse((source_dir / "source.mkv").exists())
            self.assertFalse((source_dir / "source.metadata.json").exists())
            self.assertFalse((source_dir / "download.done.json").exists())
            self.assertTrue((source_dir / ".yt-dlp-work/source.mkv").is_file())

    def test_legacy_marker_without_eof_evidence_does_not_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            source_dir = root / "source"
            source_dir.mkdir()
            _make_source(fixture)
            old_video = source_dir / "source.mkv"
            shutil.copyfile(fixture, old_video)
            old_metadata = source_dir / "source.metadata.json"
            atomic_write_json(old_metadata, {"legacy": True})
            input_hash = sha256_json(
                {
                    "url": self.url,
                    "language": "tr",
                    "format_selector": "bestvideo[protocol=https]+bestaudio[protocol=https]/best[protocol=https]/best",
                    "merge_output_format": "mkv",
                    "playlist": False,
                }
            )
            write_stage_marker(
                source_dir / "download.done.json",
                stage="download",
                input_sha256=input_hash,
                outputs={"video": old_video, "metadata": old_metadata},
                details={"source_sha256": sha256_file(old_video)},
            )
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                result = download_source(self.url, source_dir, attempts=1)

            self.assertFalse(result.resumed)
            self.assertEqual(state.media_download_calls, 1)
            marker = json.loads(result.marker_path.read_text(encoding="utf-8"))
            self.assertTrue(marker["details"]["source_validation"]["read_through_eof"])

    def test_hash_valid_marker_cannot_resume_a_truncated_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.mkv"
            complete_mp4 = root / "complete.mp4"
            source_dir = root / "source"
            source_dir.mkdir()
            _make_source(fixture)
            _make_source(complete_mp4, duration=8)
            payload = complete_mp4.read_bytes()
            old_video = source_dir / "source.mp4"
            old_video.write_bytes(payload[: len(payload) // 2])
            old_metadata = source_dir / "source.metadata.json"
            atomic_write_json(old_metadata, {"synthetic": "truncated"})
            input_hash = sha256_json(
                {
                    "url": self.url,
                    "language": "tr",
                    "format_selector": "bestvideo[protocol=https]+bestaudio[protocol=https]/best[protocol=https]/best",
                    "merge_output_format": "mkv",
                    "playlist": False,
                }
            )
            write_stage_marker(
                source_dir / "download.done.json",
                stage="download",
                input_sha256=input_hash,
                outputs={"video": old_video, "metadata": old_metadata},
                details={
                    "source_sha256": sha256_file(old_video),
                    "source_validation": {
                        "validation_version": DOWNLOAD_VALIDATION_VERSION,
                        "read_through_eof": True,
                    },
                },
            )
            state = _FakeYtDlpState(fixture)

            with patch("mas.engine.download._import_yt_dlp", return_value=_fake_yt_dlp_module(state)):
                result = download_source(self.url, source_dir, attempts=1)

            self.assertFalse(result.resumed)
            self.assertEqual(state.media_download_calls, 1)
            self.assertEqual(result.video_path.name, "source.mkv")
            self.assertFalse(old_video.exists())


if __name__ == "__main__":
    unittest.main()
