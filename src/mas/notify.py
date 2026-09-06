import os
import json
import smtplib
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid


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
    message["Message-ID"] = make_msgid()
    message["Date"] = formatdate(localtime=False, usegmt=True)
    subject = f"Muhtemel Aşk {episode}. Bölüm" if episode is not None else "Muhtemel Aşk"
    message["Subject"] = f"{subject} - {event}"
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
                refused = smtp.send_message(message)
                if refused:
                    raise smtplib.SMTPRecipientsRefused(refused)
            return {"status": "sent", "recipient": recipient,
                    "message_id": message["Message-ID"]}
        except (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused,
                smtplib.SMTPSenderRefused):
            raise RuntimeError("email authentication or recipient rejected; not retried") from None
        except (OSError, smtplib.SMTPException) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"email notification failed after 3 attempts: {last_error}")


def notify(episode, event, details=None):
    try:
        result = send_email(episode, event, details)
    except Exception as exc:
        error = str(exc)
        for key in ("MAS_GMAIL_APP_PASSWORD", "RUNPOD_API_KEY"):
            value = os.getenv(key)
            if value:
                error = error.replace(value, "<redacted>")
        result = {"status": "failed", "error": error}
    print("[EMAIL] " + json.dumps({"episode": episode, "event": event, **result},
                                 ensure_ascii=False), flush=True)
    return result
