import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from mas import cli


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_ENV = {
    "MAS_GMAIL_ADDRESS": "sender@example.com",
    "MAS_GMAIL_APP_PASSWORD": "app-password-secret",
    "MAS_DRIVE_STRICT_REMOTE": "gdrive:MyDrive/Muhtemel_Ask_Subtitles",
    "RUNPOD_POD_ID": "pod-id",
    "RUNPOD_API_KEY": "runpod-key-secret",
}


def _configure(monkeypatch, tmp_path):
    cookie = tmp_path / "youtube-cookies.txt"
    cookie.write_text(
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tsecret\n",
        encoding="utf-8",
    )
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MAS_YTDLP_COOKIES", str(cookie))
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )


def test_strict_runpod_doctor_passes_without_printing_secrets(tmp_path, monkeypatch, capsys):
    _configure(monkeypatch, tmp_path)

    def fake_run(command, **kwargs):
        if command[1] == "listremotes":
            return subprocess.CompletedProcess(command, 0, stdout="gdrive:\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.doctor(strict_runpod=True) == 0
    output = capsys.readouterr()
    combined = output.out + output.err
    assert "app-password-secret" not in combined
    assert "runpod-key-secret" not in combined
    assert "youtube_cookies: OK" in combined
    assert "cuda: OK" in combined
    assert "rclone_remote: OK" in combined


def test_strict_runpod_doctor_fails_before_remote_check(tmp_path, monkeypatch, capsys):
    _configure(monkeypatch, tmp_path)
    monkeypatch.delenv("RUNPOD_API_KEY")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "listremotes":
            return subprocess.CompletedProcess(command, 0, stdout="other:\n", stderr="")
        raise AssertionError("remote reachability must not be attempted")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.doctor(strict_runpod=True) == 1
    output = capsys.readouterr()
    assert "RUNPOD_API_KEY: MISSING" in output.err
    assert "cuda: UNAVAILABLE" in output.err
    assert "rclone_remote: not configured" in output.err
    assert calls == [["rclone", "listremotes"]]


def test_bootstrap_reuses_persistent_environment_and_installs_dependencies():
    script = (ROOT / "runpod" / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"' in script
    assert 'if [[ ! -x "$VENV/bin/python" ]]; then' in script
    assert (
        'uv pip install --python "$VENV/bin/python" '
        '--index-strategy unsafe-best-match --requirements requirements.lock'
    ) in script
    assert "uv pip sync" not in script
