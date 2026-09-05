import json
from pathlib import Path

from mas import cli
from mas import runlog


def _latest(root):
    logs = root / "13" / "logs"
    name = (logs / "LATEST").read_text(encoding="utf-8").strip()
    return logs / name


def test_run_log_captures_output_metadata_and_redacts_source_url(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: print("pipeline output") or 20)
    monkeypatch.setenv("MAS_GMAIL_ADDRESS", "sender@gmail.com")
    monkeypatch.setenv("MAS_GMAIL_APP_PASSWORD", "secret-app-password")
    monkeypatch.setenv("MAS_YTDLP_COOKIES", "secret-cookie-path")

    assert cli.main(["run", "13", "--source-url", "https://example.invalid/private", "--local"]) == 20

    content = _latest(tmp_path).read_text(encoding="utf-8")
    records = [json.loads(line) for line in content.splitlines() if line.startswith("{")]
    assert records[0]["event"] == "run_started"
    assert records[0]["argv"][4] == "<provided>"
    assert records[0]["configured"]["gmail"] is True
    assert records[0]["configured"]["youtube_cookies"] is True
    assert records[-1]["event"] == "run_finished"
    assert records[-1]["exit_code"] == 20
    assert "pipeline output" in content
    assert "secret-app-password" not in content
    assert "secret-cookie-path" not in content
    assert "https://example.invalid/private" not in content


def test_failed_run_records_traceback_and_operator_summary(tmp_path, monkeypatch):
    notifications = []
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: (_ for _ in ()).throw(ValueError("broken stage")))
    monkeypatch.setattr(cli, "notify", lambda *args: notifications.append(args))

    assert cli.main(["run", "13", "--local"]) == 1

    content = _latest(tmp_path).read_text(encoding="utf-8")
    assert '"event": "run_exception"' in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError: broken stage" in content
    assert "FAILED STAGE: RUN" in content
    assert "SAFE RETRY:" in content
    assert '"exit_code": 1' in content
    assert notifications[0][:2] == (13, "çalıştırma başarısız")
    assert "Sonuç: çalıştırma tamamlanamadı." in notifications[0][2]
    assert "ValueError: broken stage" in notifications[0][2]
    assert str(_latest(tmp_path)) in notifications[0][2]
    assert "Sonraki adım:" in notifications[0][2]


def test_transient_runpod_capacity_failure_does_not_send_email(tmp_path, monkeypatch):
    notifications = []
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(
        cli,
        "run",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("There are not enough free GPUs on the host machine")
        ),
    )
    monkeypatch.setattr(cli, "notify", lambda *args: notifications.append(args))

    assert cli.main(["run", "13", "--local"]) == 1
    assert notifications == []


def test_stage_failure_marker_suppresses_duplicate_top_level_email(tmp_path, monkeypatch):
    notifications = []
    error = ValueError("broken stage")
    error._mas_notification_sent = True
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: (_ for _ in ()).throw(error))
    monkeypatch.setattr(cli, "notify", lambda *args: notifications.append(args))

    assert cli.main(["run", "13", "--local"]) == 1
    assert notifications == []


def test_remote_stage_failure_does_not_send_duplicate_generic_email(tmp_path, monkeypatch):
    notifications = []
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(
        cli,
        "run",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("remote pipeline failed with exit code 1")
        ),
    )
    monkeypatch.setattr(cli, "notify", lambda *args: notifications.append(args))

    assert cli.main(["run", "13", "--local"]) == 1
    assert notifications == []


def test_source_url_equals_form_is_redacted():
    assert runlog._safe_argv(["run", "13", "--source-url=https://example.invalid/private"]) == [
        "run",
        "13",
        "--source-url=<provided>",
    ]


def test_non_run_command_does_not_create_episode_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "doctor", lambda: 0)
    assert cli.main(["doctor"]) == 0
    assert not list(Path(tmp_path).rglob("*.log"))
