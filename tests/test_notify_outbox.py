import json

from mas import notify


def test_progress_spool_never_waits_for_smtp(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "send_email", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SMTP")))
    result = notify.enqueue_notification(14, "alignment started", "local event", root=tmp_path)
    assert result["status"] == "recorded"
    assert notify.drain_outbox(tmp_path) == []


def test_terminal_spool_drains_once_and_keeps_submission_identity(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send_email", lambda *a, **k: sent.append(k["message_id"]) or {"status": "sent"})
    notify.enqueue_notification(14, "ready", "receipt", root=tmp_path, kind="terminal")
    assert notify.drain_outbox(tmp_path)[0]["status"] == "sent"
    notify.enqueue_notification(14, "ready", "receipt", root=tmp_path, kind="terminal")
    assert notify.drain_outbox(tmp_path) == []
    assert len(sent) == 1


def test_uncertain_submission_is_preserved_without_duplicate_retry(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("submission result lost")
    monkeypatch.setattr(notify, "send_email", fail)
    notify.enqueue_notification(14, "failed", "reason", root=tmp_path, kind="terminal")
    assert notify.drain_outbox(tmp_path)[0]["status"] == "uncertain"
    assert notify.drain_outbox(tmp_path) == []


def test_spool_preserves_changed_transition_and_redacts_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value")
    for details in ("first secret-value", "second secret-value"):
        notify.enqueue_notification(14, "blocked", details, root=tmp_path, kind="action")
    records = list((tmp_path / "work/notification-outbox").rglob("*.json"))
    assert len(records) == 2
    assert all("secret-value" not in path.read_text() for path in records)


def test_expired_drain_preserves_terminal_event(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, "send_email", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SMTP")))
    notify.enqueue_notification(14, "ready", "receipt", root=tmp_path, kind="terminal")
    assert notify.drain_outbox(tmp_path, total_timeout=0) == []
    path = next((tmp_path / "work/notification-outbox").glob("*.json"))
    assert json.loads(path.read_text())["data"]["status"] == "queued"


def test_disabled_mail_does_not_consume_delivery_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(notify, 'send_email', lambda *a, **k: {'status': 'disabled'})
    notify.enqueue_notification(14, 'ready', root=tmp_path, kind='terminal')
    for _ in range(4):
        assert notify.drain_outbox(tmp_path)[0]['status'] == 'disabled'
    record = next((tmp_path / 'work/notification-outbox').glob('*.json'))
    assert json.loads(record.read_text())['data']['attempts'] == 0


def test_outbox_disables_internal_smtp_retry_and_ignores_corrupt_record(tmp_path, monkeypatch):
    def send(*args, **kwargs):
        assert kwargs['max_attempts'] == 1
        return {'status': 'sent'}
    monkeypatch.setattr(notify, 'send_email', send)
    notify.enqueue_notification(14, 'ready', root=tmp_path, kind='terminal')
    (tmp_path / 'work/notification-outbox/000.json').write_text('{')
    assert [r['status'] for r in notify.drain_outbox(tmp_path)] == ['invalid', 'sent']
