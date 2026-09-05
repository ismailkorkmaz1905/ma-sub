from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.mux import extract_softsubs, mux_softsubs, verify_mkv_roundtrip
from src.srt import SubtitleEntry, parse_srt, write_srt


FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg and ffprobe are required")
class MKVRoundTripTests(unittest.TestCase):
    def _make_source(self, path: Path) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x90:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000",
            "-t",
            "3",
            "-c:v",
            "mpeg4",
            "-q:v",
            "8",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    def _make_h264_aac_source(self, path: Path) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=320x180:rate=25",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-bf",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    def _first_packet(self, path: Path, stream_index: int) -> dict[str, str]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                str(stream_index),
                "-read_intervals",
                "%+#1",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,dts_time",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)["packets"][0]

    def test_24_mkv_subtitle_extraction_round_trip_and_stream_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mkv"
            id_srt = root / "Muhtemel Ask 12.Bolum.id-final.srt"
            tr_srt = root / "Muhtemel Ask 12.Bolum.tr-final.srt"
            output = root / "Muhtemel Ask 12.Bolum - Endonezce + Turkce.mkv"
            self._make_source(source)

            id_entries = [
                SubtitleEntry(1, 100, 950, "Özlem baik-baik saja."),
                SubtitleEntry(2, 1_200, 2_100, "Ya Allah, Defne datang."),
            ]
            tr_entries = [
                SubtitleEntry(1, 100, 950, "Özlem iyi."),
                SubtitleEntry(2, 1_200, 2_100, "Allah'ım, Defne geldi."),
            ]
            write_srt(id_srt, id_entries)
            write_srt(tr_srt, tr_entries)

            report = mux_softsubs(
                source,
                id_srt,
                tr_srt,
                output,
                verify_stream_hashes=True,
            )
            independent = verify_mkv_roundtrip(output, id_srt, tr_srt)
            extracted_id, extracted_tr = extract_softsubs(output, root / "extracted")

            self.assertTrue(output.is_file())
            self.assertTrue(report["verified"])
            self.assertTrue(report["video_audio_stream_copy"])
            self.assertEqual(report["subtitle_order"], ["ind", "tur"])
            self.assertTrue(report["indonesian_default"])
            self.assertFalse(report["turkish_default"])
            self.assertTrue(report["stream_hashes"]["checked"])
            self.assertTrue(report["stream_hashes"]["match"])
            self.assertEqual(
                report["stream_hashes"]["source"],
                report["stream_hashes"]["output"],
            )
            self.assertTrue(independent["exact"])
            self.assertEqual(parse_srt(extracted_id), id_entries)
            self.assertEqual(parse_srt(extracted_tr), tr_entries)

    def test_h264_decode_reordering_does_not_shift_subtitle_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            id_srt = root / "episode.id-final.srt"
            tr_srt = root / "episode.tr-final.srt"
            output = root / "episode.mkv"
            self._make_h264_aac_source(source)

            video_packet = self._first_packet(source, 0)
            audio_packet = self._first_packet(source, 1)
            self.assertLess(
                float(video_packet["dts_time"]), float(video_packet["pts_time"])
            )
            self.assertLess(float(audio_packet["pts_time"]), 0.0)
            self.assertLess(
                float(video_packet["dts_time"]), float(audio_packet["pts_time"])
            )

            id_entries = [SubtitleEntry(1, 1_000, 2_500, "Defne sudah datang.")]
            tr_entries = [SubtitleEntry(1, 1_000, 2_500, "Defne geldi.")]
            write_srt(id_srt, id_entries)
            write_srt(tr_srt, tr_entries)

            report = mux_softsubs(
                source,
                id_srt,
                tr_srt,
                output,
                verify_stream_hashes=True,
            )
            extracted_id, extracted_tr = extract_softsubs(output, root / "extracted")

            self.assertTrue(report["verified"])
            self.assertEqual(report["subtitle_order"], ["ind", "tur"])
            self.assertTrue(report["indonesian_default"])
            self.assertFalse(report["turkish_default"])
            self.assertTrue(report["stream_hashes"]["match"])
            self.assertEqual(parse_srt(extracted_id), id_entries)
            self.assertEqual(parse_srt(extracted_tr), tr_entries)


if __name__ == "__main__":
    unittest.main()

