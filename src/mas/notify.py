import os
import json
import smtplib
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from datetime import datetime, timezone
from pathlib import Path

from .config import episode_dir
from .reliability import atomic_json, digest, read_json


def enabled():
    return bool(os.getenv("MAS_GMAIL_ADDRESS") and os.getenv("MAS_GMAIL_APP_PASSWORD"))


def send_email(episode, event, details=None, *, message_id=None, total_timeout=None, max_attempts=3):
    if not enabled():
        return {"status": "disabled"}

    sender = os.environ["MAS_GMAIL_ADDRESS"]
    recipient = os.getenv("MAS_NOTIFY_TO", sender)
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Message-ID"] = message_id or make_msgid()
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

    deadline = time.monotonic() + total_timeout if total_timeout is not None else None

    def timeout():
        remaining = 20 if deadline is None else min(20, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("notification drain deadline expired")
        return remaining

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=timeout()) as smtp:
                if getattr(smtp, "sock", None):
                    smtp.sock.settimeout(timeout())
                smtp.login(sender, os.environ["MAS_GMAIL_APP_PASSWORD"])
                if getattr(smtp, "sock", None):
                    smtp.sock.settimeout(timeout())
                refused = smtp.send_message(message)
                if refused:
                    raise smtplib.SMTPRecipientsRefused(refused)
            return {"status": "sent", "recipient": recipient,
                    "message_id": message["Message-ID"]}
        except (smtplib.SMTPAuthenticationError, smtplib.SMTPRecipientsRefused,
                smtplib.SMTPSenderRefused):
            raise RuntimeError("email authentication or recipient rejected; not retried") from None
        except (OSError, smtplib.SMTPException) as exc:
            if isinstance(exc, smtplib.SMTPResponseException) and 500 <= exc.smtp_code < 600:
                return {"status": "blocked", "smtp_code": exc.smtp_code,
                        "error": "permanent SMTP rejection; not retried"}
            last_error = exc
            if attempt < max_attempts:
                if deadline is not None and deadline - time.monotonic() <= attempt:
                    break
                time.sleep(attempt)
    raise RuntimeError(f"email notification failed after {max_attempts} attempts: {last_error}")


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


def enqueue_notification(episode, event, details=None, *, root=None, kind="progress"):
    try:
        return _enqueue_notification(episode, event, details, root=root, kind=kind)
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__}


def _enqueue_notification(episode, event, details=None, *, root=None, kind="progress"):
    if kind not in {"progress", "milestone", "action", "terminal"}:
        raise ValueError("invalid notification kind")
    root = Path(root) if root is not None else episode_dir(episode)
    key = digest({"episode": episode, "event": event, "kind": kind})
    path = root / "work" / "notification-outbox" / (key + ".json")
    clean_details = str(details or "")
    for name in ("MAS_GMAIL_APP_PASSWORD", "RUNPOD_API_KEY"):
        value = os.getenv(name)
        if value:
            clean_details = clean_details.replace(value, "<redacted>")
    prior = None
    if path.is_file():
        envelope = read_json(path)
        prior = envelope.get("data")
        if not isinstance(prior, dict) or envelope.get("sha256") != digest(prior):
            raise ValueError("notification outbox integrity mismatch")
        if prior["details"] == clean_details:
            return {"status": prior["status"], "event_id": key}
    body = {"episode": episode, "event": event, "kind": kind, "details": clean_details,
            "status": "recorded" if kind == "progress" else "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message_id": make_msgid(), "attempts": 0}
    if prior is not None:
        retained = path.parent / "history" / (digest(prior) + ".json")
        atomic_json(retained, {"data": prior, "sha256": digest(prior)})
    atomic_json(path, {"data": body, "sha256": digest(body)})
    return {"status": body["status"], "event_id": key}


def drain_outbox(root, *, total_timeout=60, max_messages=3):
    deadline = time.monotonic() + min(60, max(0, total_timeout))
    results = []
    submitted = 0
    for path in sorted((Path(root) / "work" / "notification-outbox").glob("*.json")):
        if submitted >= max_messages or time.monotonic() >= deadline:
            break
        try:
            envelope = read_json(path)
        except (OSError, ValueError):
            results.append({"status": "invalid", "event_id": path.stem})
            continue
        body = envelope.get("data")
        if not isinstance(body, dict) or envelope.get("sha256") != digest(body):
            results.append({"status": "invalid", "event_id": path.stem})
            continue
        if body["status"] != "queued" or body["attempts"] >= 3:
            continue
        body.update(status="sending", attempts=body["attempts"] + 1)
        atomic_json(path, {"data": body, "sha256": digest(body)})
        submitted += 1
        try:
            result = send_email(body["episode"], body["event"], body["details"],
                                message_id=body["message_id"], total_timeout=deadline-time.monotonic(),
                                max_attempts=1)
        except Exception:
            result = {"status": "uncertain"}
        body["status"] = "queued" if result["status"] == "disabled" else result["status"]
        if result["status"] == "disabled":
            body["attempts"] -= 1
        body["result"] = result
        atomic_json(path, {"data": body, "sha256": digest(body)})
        results.append({"event_id": path.stem, **result})
    return results
