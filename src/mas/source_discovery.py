import json
import re
import sys
import unicodedata
from urllib.parse import parse_qs, urlparse

from .engine.download import DownloadError, _validated_cookie_file
from .remote import RemoteVerificationError, _run_watchdog


CHANNEL_VIDEOS_URL = "https://www.youtube.com/@muhtemelaskdizi/videos"


class SourceDiscoveryError(RuntimeError):
    pass


def _title_tokens(title):
    text = unicodedata.normalize("NFKD", str(title).casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.translate(str.maketrans({"ı": "i", "ş": "s", "ğ": "g"}))
    return re.findall(r"[a-z0-9]+", text)


def is_exact_episode_title(title, episode):
    return _title_tokens(title) == ["muhtemel", "ask", str(episode), "bolum"]


def _watch_url(entry):
    video_id = str(entry.get("id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        webpage_url = str(entry.get("webpage_url") or entry.get("url") or "")
        parsed = urlparse(webpage_url)
        if parsed.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            raise SourceDiscoveryError("matching episode has no valid YouTube video ID")
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise SourceDiscoveryError("matching episode has no valid YouTube video ID")
    return f"https://www.youtube.com/watch?v={video_id}"


def _list_channel_videos(cookies_file, *, idle_timeout, total_timeout):
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--skip-download",
        "--socket-timeout",
        "10",
        "--retries",
        "3",
        "--extractor-retries",
        "3",
        "--dump-json",
    ]
    if cookies_file is not None:
        command.extend(("--cookies", str(cookies_file)))
    command.append(CHANNEL_VIDEOS_URL)
    try:
        output = _run_watchdog(
            command,
            idle_timeout=idle_timeout,
            total_timeout=total_timeout,
        )
    except RemoteVerificationError as exc:
        raise SourceDiscoveryError(f"YouTube source discovery failed: {exc}") from exc

    entries = []
    for line in output.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceDiscoveryError("yt-dlp returned invalid discovery metadata") from exc
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def discover_episode_source(
    episode,
    *,
    cookies_file=None,
    idle_timeout=30,
    total_timeout=300,
):
    if not isinstance(episode, int) or isinstance(episode, bool) or episode < 1:
        raise ValueError("episode must be a positive integer")
    try:
        cookie_path = _validated_cookie_file(cookies_file)
    except DownloadError as exc:
        raise SourceDiscoveryError(str(exc)) from exc
    matches = [
        entry
        for entry in _list_channel_videos(
            cookie_path,
            idle_timeout=idle_timeout,
            total_timeout=total_timeout,
        )
        if is_exact_episode_title(entry.get("title"), episode)
    ]
    if not matches:
        raise SourceDiscoveryError(
            f"exact full episode {episode} was not found on the official channel"
        )
    urls = {_watch_url(entry) for entry in matches}
    if len(urls) != 1:
        raise SourceDiscoveryError(
            f"multiple exact full episode {episode} sources were found on the official channel"
        )
    return urls.pop()
