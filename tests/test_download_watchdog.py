import pytest

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
