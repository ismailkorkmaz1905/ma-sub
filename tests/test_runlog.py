import io
import json
import os
from pathlib import Path
import threading

import pytest

from mas import cli, runlog


def _log(root):
    return root / "15" / ".mas" / "run.log"


def test_tee_writes_turkish_as_utf8():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    log = io.StringIO()
    tee = runlog._Tee(stream, log, threading.RLock())

    tee.write("Türkçe hizalama: Ağabeyim.\n")
    tee.flush()

    assert stream.encoding == "utf-8"
    assert raw.getvalue().decode("utf-8").replace("\r\n", "\n") == "Türkçe hizalama: Ağabeyim.\n"
    assert log.getvalue() == "Türkçe hizalama: Ağabeyim.\n"


def test_run_uses_one_redacted_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: print("pipeline output") or 20)
    monkeypatch.setenv("MAS_YTDLP_COOKIES", "secret-cookie-path")

    assert cli.main(["run", "15", "--source-url", "https://example.invalid/private", "--local"]) == 20

    content = _log(tmp_path).read_text(encoding="utf-8")
    records = [json.loads(line) for line in content.splitlines() if line.startswith("{")]
    assert records[0]["event"] == "run_started"
    assert records[0]["argv"][4] == "<provided>"
    assert records[0]["configured"]["youtube_cookies"] is True
    assert records[-1]["event"] == "run_finished"
    assert records[-1]["exit_code"] == 20
    assert "pipeline output" in content
    assert "secret-cookie-path" not in content
    assert "https://example.invalid/private" not in content


def test_failed_run_records_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: (_ for _ in ()).throw(ValueError("broken stage")))

    assert cli.main(["run", "15", "--local"]) == 1

    content = _log(tmp_path).read_text(encoding="utf-8")
    assert '"event": "run_exception"' in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError: broken stage" in content
    assert "FAILED STAGE: RUN" in content
    assert '"exit_code": 1' in content


def test_repeated_run_reuses_one_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: 0)

    assert cli.main(["run", "15", "--local"]) == 0
    first = _log(tmp_path)
    assert cli.main(["run", "15", "--local"]) == 0
    assert list((tmp_path / "15").rglob("*.log")) == [first]


def test_source_url_equals_form_is_redacted():
    assert runlog._safe_argv(["run", "15", "--source-url=https://example.invalid/private"]) == [
        "run", "15", "--source-url=<provided>"
    ]


def test_non_run_command_does_not_create_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "doctor", lambda: 0)
    assert cli.main(["doctor"]) == 0
    assert not list(Path(tmp_path).rglob("*.log"))


def test_help_keeps_operator_surface_small(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    output = capsys.readouterr().out
    assert stopped.value.code == 0
    assert "{run,status,doctor,test}" in output
    assert "subtitle-pilot" not in output


def test_remote_worker_does_not_open_a_second_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setenv("MAS_REMOTE_JOB_TOKEN", "worker-job-token")
    monkeypatch.setattr(cli, "run", lambda *args: 0)

    assert cli.main(["run", "15", "--local"]) == 0
    assert not list(Path(tmp_path).rglob("*.log"))


def test_run_start_exists_only_during_run(tmp_path, monkeypatch):
    seen = []
    monkeypatch.delenv("MAS_RUN_STARTED_AT", raising=False)
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: seen.append(os.environ["MAS_RUN_STARTED_AT"]) or 0)

    assert cli.main(["run", "15", "--local"]) == 0
    assert seen[0].endswith("+00:00")
    assert "MAS_RUN_STARTED_AT" not in os.environ
