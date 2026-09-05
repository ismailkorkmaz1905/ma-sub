import json

import pytest

from mas import source_discovery


def _listing(*entries):
    return b"".join(json.dumps(entry).encode("utf-8") + b"\n" for entry in entries)


@pytest.mark.parametrize(
    "title",
    (
        "Muhtemel Aşk 13. Bölüm",
        "MUHTEMEL ASK 13.BOLUM",
        "Muhtemel  Ask - 13 . Bolum",
    ),
)
def test_exact_episode_title_accepts_turkish_and_spacing_variants(title):
    assert source_discovery.is_exact_episode_title(title, 13)


@pytest.mark.parametrize(
    "title",
    (
        "Muhtemel Aşk 13. Bölüm Fragman",
        "Muhtemel Aşk 13. Bölüm Ön İzleme",
        "Muhtemel Aşk 13. Bölüm Özeti",
        "Muhtemel Aşk 13. Bölümden Klip",
        "Muhtemel Aşk 12. Bölüm",
    ),
)
def test_exact_episode_title_rejects_non_episode_or_extra_words(title):
    assert not source_discovery.is_exact_episode_title(title, 13)


def test_discovery_returns_canonical_watch_url_and_bounded_command(monkeypatch):
    calls = []

    def run(command, *, idle_timeout, total_timeout):
        calls.append((command, idle_timeout, total_timeout))
        return _listing(
            {"id": "clip0000001", "title": "Muhtemel Aşk 13. Bölüm Fragman"},
            {"id": "episode0013", "title": "Muhtemel Aşk 13. Bölüm"},
        )

    monkeypatch.setattr(source_discovery, "_run_watchdog", run)

    assert source_discovery.discover_episode_source(13) == (
        "https://www.youtube.com/watch?v=episode0013"
    )
    command, idle_timeout, total_timeout = calls[0]
    assert command[:3] == [source_discovery.sys.executable, "-m", "yt_dlp"]
    assert "--flat-playlist" in command
    assert "--playlist-end" not in command
    assert command[command.index("--socket-timeout") + 1] == "10"
    assert command[command.index("--retries") + 1] == "3"
    assert command[command.index("--extractor-retries") + 1] == "3"
    assert command[-1] == source_discovery.CHANNEL_VIDEOS_URL
    assert (idle_timeout, total_timeout) == (30, 300)


def test_discovery_passes_validated_cookie_path_without_reading_contents(monkeypatch, tmp_path):
    cookie = tmp_path / "cookies.txt"
    cookie.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    def run(command, *, idle_timeout, total_timeout):
        assert command[command.index("--cookies") + 1] == str(cookie.resolve())
        return _listing({"id": "episode0013", "title": "Muhtemel Ask 13. Bolum"})

    monkeypatch.setattr(source_discovery, "_run_watchdog", run)
    assert source_discovery.discover_episode_source(13, cookies_file=cookie).endswith(
        "episode0013"
    )


def test_discovery_rejects_ambiguous_exact_matches(monkeypatch):
    monkeypatch.setattr(
        source_discovery,
        "_run_watchdog",
        lambda *args, **kwargs: _listing(
            {"id": "episode0013", "title": "Muhtemel Aşk 13. Bölüm"},
            {"id": "episode1013", "title": "Muhtemel Ask 13. Bolum"},
        ),
    )
    with pytest.raises(source_discovery.SourceDiscoveryError, match="multiple exact"):
        source_discovery.discover_episode_source(13)


def test_discovery_rejects_not_found(monkeypatch):
    monkeypatch.setattr(
        source_discovery,
        "_run_watchdog",
        lambda *args, **kwargs: _listing(
            {"id": "clip0000001", "title": "Muhtemel Aşk 13. Bölüm Fragman"},
            {"id": "episode0012", "title": "Muhtemel Aşk 12. Bölüm"},
        ),
    )
    with pytest.raises(source_discovery.SourceDiscoveryError, match="was not found"):
        source_discovery.discover_episode_source(13)


def test_pipeline_discovers_once_and_persists_immutable_source_url(tmp_path, monkeypatch):
    source_url = "https://www.youtube.com/watch?v=episode0013"
    monkeypatch.setattr(source_discovery, "_run_watchdog", lambda *args, **kwargs: b"")
    monkeypatch.setattr("mas.pipeline.episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr("mas.pipeline.discover_episode_source", lambda *args, **kwargs: source_url)
    monkeypatch.setattr("mas.pipeline._load_configs", lambda: (_ for _ in ()).throw(RuntimeError("stop")))

    with pytest.raises(RuntimeError, match="stop"):
        source_discovery_path = tmp_path / "13" / "source" / "source.url"
        from mas.pipeline import run

        run(13)

    assert source_discovery_path.read_text(encoding="utf-8") == source_url + "\n"

    monkeypatch.setattr(
        "mas.pipeline.discover_episode_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not rediscover")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        run(13)


def test_pipeline_recovers_incomplete_source_url_before_source_initialization(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    url_path = source_dir / "source.url"
    url_path.write_bytes(b"")
    resolved = "https://www.youtube.com/watch?v=episode0013"
    monkeypatch.setattr("mas.pipeline.discover_episode_source", lambda *args, **kwargs: resolved)

    from mas.pipeline import _resolve_source_url

    assert _resolve_source_url(url_path, {}, 13) == resolved
    assert url_path.read_bytes() == (resolved + "\n").encode("utf-8")


def test_pipeline_rejects_invalid_source_url_after_source_initialization(tmp_path):
    url_path = tmp_path / "source.url"
    url_path.write_bytes(b"")

    from mas.pipeline import _resolve_source_url

    with pytest.raises(RuntimeError, match="refusing source identity change"):
        _resolve_source_url(url_path, {"source_sha256": "0" * 64}, 13)


def test_pipeline_never_rewrites_valid_initialized_source_url(tmp_path, monkeypatch):
    url_path = tmp_path / "source.url"
    original = "https://www.youtube.com/watch?v=episode0013"
    url_path.write_text(original + "\n", encoding="utf-8")
    before = url_path.stat().st_mtime_ns
    monkeypatch.setattr(
        "mas.pipeline.discover_episode_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not discover")),
    )

    from mas.pipeline import _resolve_source_url

    assert _resolve_source_url(url_path, {}, 13, original) == original
    assert url_path.stat().st_mtime_ns == before


@pytest.mark.parametrize("url", ("", "not-a-url", "file:///episode.mkv", "https://user:pass@example.com/x"))
def test_pipeline_rejects_invalid_source_urls(tmp_path, url):
    from mas.pipeline import _resolve_source_url

    with pytest.raises(RuntimeError, match="source URL is invalid"):
        _resolve_source_url(tmp_path / "source.url", {}, 13, url)
