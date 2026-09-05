import os
import smtplib
import time
from email.header import Header
from email.message import EmailMessage


def enabled():
    return bool(os.getenv("MAS_GMAIL_ADDRESS") and os.getenv("MAS_GMAIL_APP_PASSWORD"))


def send_email(episode, event, details=None):
    if not enabled():
        return {"status": "disabled"}

    sender = os.environ["MAS_GMAIL_ADDRESS"]
    recipient = os.getenv("MAS_NOTIFY_TO", sender)
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    subject = f"Muhtemel Aşk {episode}. Bölüm" if episode is not None else "Muhtemel Aşk"
    message["Subject"] = Header(f"{subject} - {event}", "utf-8").encode()
    message.set_content(
        f"Bölüm: {episode if episode is not None else '-'}\n"
        f"Durum: {event}\n"
        f"Detay:\n{details or '-'}\n",
        charset="utf-8",
        cte="quoted-printable",
    )

    last_error = None
    for attempt in range(1, 4):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
                smtp.login(sender, os.environ["MAS_GMAIL_APP_PASSWORD"])
                smtp.send_message(message)
            return {"status": "sent", "recipient": recipient}
        except (OSError, smtplib.SMTPException) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"email notification failed after 3 attempts: {last_error}")


def notify(episode, event, details=None):
    try:
        return send_email(episode, event, details)
    except Exception as exc:
        print(f"[EMAIL WARNING] {exc}")
        return {"status": "failed", "error": str(exc)}
