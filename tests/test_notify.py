import smtplib

from mas import cli
from mas import notify


class FakeSMTP:
    sent = None

    def __init__(self, host, port, timeout):
        assert (host, port, timeout) == ("smtp.gmail.com", 465, 20)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def login(self, sender, password):
        assert (sender, password) == ("sender@gmail.com", "app-password")

    def send_message(self, message):
        FakeSMTP.sent = message


def test_email_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("MAS_GMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("MAS_GMAIL_APP_PASSWORD", raising=False)
    assert notify.send_email(13, "basladi") == {"status": "disabled"}


def test_email_sent_to_self_by_default(monkeypatch):
    monkeypatch.setenv("MAS_GMAIL_ADDRESS", "sender@gmail.com")
    monkeypatch.setenv("MAS_GMAIL_APP_PASSWORD", "app-password")
    monkeypatch.delenv("MAS_NOTIFY_TO", raising=False)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)

    result = notify.send_email(13, "bölüm hazır", "Sonuç: teslimat doğrulandı.\nMakbuz: receipt.json")

    assert result == {"status": "sent", "recipient": "sender@gmail.com"}
    assert FakeSMTP.sent["To"] == "sender@gmail.com"
    assert "Muhtemel Aşk 13. Bölüm" in FakeSMTP.sent["Subject"]
    assert "Bölüm: 13" in FakeSMTP.sent.get_content()
    assert FakeSMTP.sent.get_content_charset() == "utf-8"
    assert b"=?utf-8?" in FakeSMTP.sent.as_bytes().lower()
    assert "teslimat doğrulandı" in FakeSMTP.sent.get_content()
    assert "receipt.json" in FakeSMTP.sent.get_content()


def test_notification_failure_does_not_fail_pipeline(monkeypatch):
    monkeypatch.setattr(notify, "send_email", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    assert notify.notify(13, "basladi")["status"] == "failed"


def test_notify_test_command_fails_without_credentials(monkeypatch, capsys):
    monkeypatch.setattr(cli, "send_email", lambda *args: {"status": "disabled"})

    assert cli.main(["notify-test"]) == 1
    assert "MAS_GMAIL_ADDRESS and MAS_GMAIL_APP_PASSWORD are required" in capsys.readouterr().err


def test_notify_test_command_sends_real_test_message(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "send_email",
        lambda *args: calls.append(args) or {"status": "sent", "recipient": "sender@gmail.com"},
    )

    assert cli.main(["notify-test"]) == 0
    assert calls == [(None, "bildirim testi", "MAS e-posta bildirimi çalışıyor.")]
    assert "Test email sent to sender@gmail.com" in capsys.readouterr().out
