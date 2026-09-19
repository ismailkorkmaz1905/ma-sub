import pytest
from types import SimpleNamespace

from mas.engine import download
from mas.engine.download import DownloadError, _DownloadProgressWatchdog


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_download_watchdog_accepts_measured_byte_progress():
    clock = Clock()
    watchdog = _DownloadProgressWatchdog(30, 900, clock=clock)
    for downloaded in (1, 2, 3):
        clock.value += 29
        watchdog({"status": "downloading", "downloaded_bytes": downloaded})
    watchdog.check()


def test_download_watchdog_rejects_no_progress():
    clock = Clock()
    watchdog = _DownloadProgressWatchdog(30, 900, clock=clock)
    watchdog({"status": "downloading", "downloaded_bytes": 1})
    clock.value = 31
    with pytest.raises(DownloadError, match="no-progress"):
        watchdog({"status": "downloading", "downloaded_bytes": 1})


def test_download_watchdog_does_not_count_estimate_changes_as_progress():
    clock = Clock()
    watchdog = _DownloadProgressWatchdog(30, 900, clock=clock)
    watchdog({"status": "downloading", "downloaded_bytes": 1, "total_bytes_estimate": 10})
    clock.value = 31
    with pytest.raises(DownloadError, match="no-progress"):
        watchdog({"status": "downloading", "downloaded_bytes": 1, "total_bytes_estimate": 20})


def test_download_watchdog_rejects_total_timeout_even_with_progress():
    clock = Clock()
    watchdog = _DownloadProgressWatchdog(30, 90, clock=clock)
    for downloaded in (1, 2, 3):
        clock.value += 29
        watchdog({"status": "downloading", "downloaded_bytes": downloaded})
    clock.value = 91
    with pytest.raises(DownloadError, match="total timeout"):
        watchdog({"status": "downloading", "downloaded_bytes": 4})


@pytest.mark.parametrize("failure", ["eof", "hash", "marker", "force"])
def test_locked_source_never_reacquired_or_replaced(tmp_path, monkeypatch, failure):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"locked source bytes")
    before = source.read_bytes()
    expected = download.sha256_file(source)
    marker = {"details": {"source_validation": {}}}
    result = SimpleNamespace(video_path=source, metadata={})
    monkeypatch.setattr(download, "load_valid_stage_marker", lambda *a, **k: None if failure == "marker" else marker)
    monkeypatch.setattr(download, "_marker_has_read_validation", lambda *a: True)
    monkeypatch.setattr(download, "_metadata_from_marker", lambda *a: result)
    monkeypatch.setattr(download, "_import_yt_dlp", lambda: pytest.fail("reacquisition attempted"))
    def eof(*args, **kwargs):
        if failure == "eof":
            raise RuntimeError("transient media check failure")
    monkeypatch.setattr("mas.engine.media.verify_media_readable", eof)
    with pytest.raises(DownloadError, match="immutable"):
        download.download_source("https://example.test/video", tmp_path,
                                 expected_source_sha256="a" * 64 if failure == "hash" else expected,
                                 force=failure == "force")
    assert source.read_bytes() == before


def test_frozen_caption_snapshot_skips_optional_network_enrichment(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"locked source bytes")
    result = SimpleNamespace(video_path=source, metadata={}, captions_path=None)
    marker = {"details": {"source_validation": {}}}
    monkeypatch.setattr(download, "load_valid_stage_marker", lambda *a, **k: marker)
    monkeypatch.setattr(download, "_marker_has_read_validation", lambda *a: True)
    monkeypatch.setattr(download, "_metadata_from_marker", lambda *a: result)
    monkeypatch.setattr("mas.engine.media.verify_media_readable", lambda *a, **k: {})
    monkeypatch.setattr(download, "_caption_retry_on_resume", lambda *a, **k: pytest.fail("snapshot enriched"))
    assert download.download_source("https://example.test/video", tmp_path,
                                    expected_source_sha256=download.sha256_file(source),
                                    freeze_captions=True) is result
