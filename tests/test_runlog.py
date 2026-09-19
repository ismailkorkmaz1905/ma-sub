import json
import io
import os
from pathlib import Path
import threading

import pytest

from mas import cli
from mas import runlog, notify


def test_tee_reconfigures_narrow_stream_for_turkish_output():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    log = io.StringIO()
    tee = runlog._Tee(stream, log, threading.RLock())

    tee.write("Türkçe hizalama: Ağabeyim.\n")
    tee.flush()

    assert stream.encoding == "utf-8"
    assert raw.getvalue().decode("utf-8").replace("\r\n", "\n") == (
        "Türkçe hizalama: Ağabeyim.\n"
    )
    assert log.getvalue() == "Türkçe hizalama: Ağabeyim.\n"


def _latest(root):
    return root / "13" / ".mas" / "run.log"


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
    monkeypatch.setattr(cli, "enqueue_notification", lambda *args, **kwargs: notifications.append((args, kwargs)))

    assert cli.main(["run", "13", "--local"]) == 1

    content = _latest(tmp_path).read_text(encoding="utf-8")
    assert '"event": "run_exception"' in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError: broken stage" in content
    assert "FAILED STAGE: RUN" in content
    assert "SAFE RETRY:" in content
    assert '"exit_code": 1' in content
    args, kwargs = notifications[0]
    assert args[:2] == (13, "çalıştırma başarısız")
    assert "Sonuç: çalıştırma tamamlanamadı." in args[2]
    assert "ValueError: broken stage" in args[2]
    assert str(_latest(tmp_path)) in args[2]
    assert "Sonraki adım:" in args[2]
    assert kwargs == {"root": tmp_path / "13", "kind": "terminal"}


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
    monkeypatch.setattr(cli, "enqueue_notification", lambda *args, **kwargs: notifications.append(args))

    assert cli.main(["run", "13", "--local"]) == 1
    assert notifications == []


def test_stage_failure_marker_suppresses_duplicate_top_level_email(tmp_path, monkeypatch):
    notifications = []
    error = ValueError("broken stage")
    error._mas_notification_sent = True
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: (_ for _ in ()).throw(error))
    monkeypatch.setattr(cli, "enqueue_notification", lambda *args, **kwargs: notifications.append(args))

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
    monkeypatch.setattr(cli, "enqueue_notification", lambda *args, **kwargs: notifications.append(args))

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


def test_help_keeps_the_operator_surface_small(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])

    output = capsys.readouterr().out
    assert stopped.value.code == 0
    assert "{run,status,doctor,test}" in output
    assert "subtitle-pilot" not in output
    assert "plan-alignment-recovery" not in output


def test_remote_worker_uses_only_the_supervisor_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setenv("MAS_REMOTE_JOB_TOKEN", "worker-job-token")
    monkeypatch.setattr(cli, "run", lambda *args: 0)

    assert cli.main(["run", "13", "--local"]) == 0
    assert not list(Path(tmp_path).rglob("*.log"))


def test_repeated_failed_run_reuses_one_log_and_preserves_terminal_submission(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: (_ for _ in ()).throw(ValueError("broken stage")))
    monkeypatch.setattr(notify, "send_email", lambda *args, **kwargs: sent.append(kwargs["message_id"]) or {"status": "sent"})

    assert cli.main(["run", "13", "--local"]) == 1
    first_log = _latest(tmp_path)
    outbox = tmp_path / "13/work/notification-outbox"
    record = next(outbox.glob("*.json"))
    assert json.loads(record.read_text(encoding="utf-8"))["data"]["status"] == "queued"
    assert sent == []
    assert notify.drain_outbox(tmp_path / "13")[0]["status"] == "sent"
    first_receipt = record.read_bytes()

    assert cli.main(["run", "13", "--local"]) == 1
    assert _latest(tmp_path) == first_log
    assert list((tmp_path / "13").rglob("*.log")) == [first_log]
    assert record.read_bytes() == first_receipt
    assert not (outbox / "history").exists()
    assert notify.drain_outbox(tmp_path / "13") == []
    assert len(sent) == 1


def test_run_start_is_available_only_inside_run(tmp_path, monkeypatch):
    seen = []
    monkeypatch.delenv("MAS_RUN_STARTED_AT", raising=False)
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(cli, "run", lambda *args: seen.append(os.environ["MAS_RUN_STARTED_AT"]) or 0)

    assert cli.main(["run", "13", "--local"]) == 0
    assert seen and seen[0].endswith("+00:00")
    assert "MAS_RUN_STARTED_AT" not in os.environ


def test_notify_test_remains_explicit_direct_email(monkeypatch):
    sent = []
    queued = []
    monkeypatch.setattr(cli, "send_email", lambda *args: sent.append(args) or {"status": "sent", "recipient": "test@example.invalid"})
    monkeypatch.setattr(cli, "enqueue_notification", lambda *args, **kwargs: queued.append(args))
    assert cli.main(["notify-test"]) == 0
    assert len(sent) == 1
    assert sent[0][:2] == (None, "bildirim testi")
    assert queued == []


@pytest.mark.parametrize("flags,pod_id,worker_token,expected", [
    ([], "", "", ["queued", "safe-drain"]),
    ([], "protected-pod", "", ["queued", "safe-drain"]),
    (["--local"], "", "", ["queued"]),
    (["--fixture"], "", "", ["queued"]),
    (["--stop-after", "1"], "", "", ["queued"]),
    ([], "worker-pod", "worker-job-token", ["queued"]),
])
def test_only_local_controller_failure_requests_safe_post_enqueue_drain(tmp_path, monkeypatch, flags, pod_id, worker_token, expected):
    events = []
    monkeypatch.setattr(runlog, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setenv("RUNPOD_POD_ID", pod_id)
    monkeypatch.setenv("MAS_REMOTE_JOB_TOKEN", worker_token)
    fail = lambda *args: (_ for _ in ()).throw(ValueError("broken stage"))
    monkeypatch.setattr(cli, "run", fail)
    monkeypatch.setattr(cli, "run_remote_episode", fail)
    monkeypatch.setattr(cli, "enqueue_notification", lambda *args, **kwargs: events.append("queued"))
    monkeypatch.setattr(cli, "drain_cli_notifications", lambda episode: events.append("safe-drain"))
    assert cli.main(["run", "13", *flags]) == 1
    assert events == expected
