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


def test_strict_runpod_doctor_allows_public_source_without_cookies(tmp_path, monkeypatch, capsys):
    _configure(monkeypatch, tmp_path)
    monkeypatch.delenv("MAS_YTDLP_COOKIES")

    def fake_run(command, **kwargs):
        if command[1] == "listremotes":
            return subprocess.CompletedProcess(command, 0, stdout="gdrive:\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.doctor(strict_runpod=True) == 0
    assert "youtube_cookies: NOT_SET" in capsys.readouterr().out


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


def test_strict_runpod_doctor_retries_drive_timeouts(tmp_path, monkeypatch, capsys):
    _configure(monkeypatch, tmp_path)
    attempts = []
    sleeps = []

    def fake_run(command, **kwargs):
        if command[1] == "listremotes":
            return subprocess.CompletedProcess(command, 0, stdout="gdrive:\n", stderr="")
        attempts.append(kwargs["timeout"])
        if len(attempts) < 3:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert cli.doctor(strict_runpod=True) == 0
    assert attempts == [60, 60, 60]
    assert sleeps == [1, 2]
    assert "rclone_remote: OK" in capsys.readouterr().out


def test_bootstrap_reuses_persistent_environment_and_installs_dependencies():
    script = (ROOT / "runpod" / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"' in script
    assert 'MAS_BIN_DIR="${MAS_BIN_DIR:-/workspace/.local/bin}"' in script
    assert "denoland/deno/releases/download/v2.9.5" in script
    assert "sha256sum --check --status" in script
    assert '"$MAS_BIN_DIR/rclone"' in script
    assert 'export UV_INSTALL_DIR="$MAS_BIN_DIR"' in script
    assert "timeout 300 apt-get -o Acquire::Retries=3 update" in script
    assert "timeout 600 apt-get -o Acquire::Retries=3 install" in script
    assert 'RUNTIME_MARKER="$VENV/.mas-runtime-abi.json"' in script
    assert '"format": "mas-runtime-abi-marker-1"' in script
    for binding in (
        '"requirements_sha256"',
        'sys.implementation.name',
        'platform.python_version()',
        'sys.implementation.cache_tag',
        'sysconfig.get_config_var("SOABI")',
        'platform.machine()',
        '"uv_version"',
        'platform.freedesktop_os_release()',
        'os.confstr("CS_GNU_LIBC_VERSION")',
        'torch.__version__',
        'torch.version.cuda',
        'torch.backends.cudnn.version()',
        '"_GLIBCXX_USE_CXX11_ABI"',
        '"ffmpeg"',
        '"ffprobe"',
    ):
        assert binding in script
    assert 'wrapped = {"data": data, "sha256": hashlib.sha256(canonical).hexdigest()}' in script
    assert 'ffmpeg_path="$(readlink -f -- "$(command -v ffmpeg)")"' in script
    assert 'FFMPEG_SHA256="$(sha256sum "$ffmpeg_path"' in script
    assert 'cmp -s -- "$RUNTIME_MARKER" "$observed_marker"' in script
    assert 'uv venv --python 3.11 --relocatable "$candidate"' in script
    assert 'mv -- "$candidate" "$VENV"' in script
    assert 'remove_rebuild_tree "$backup"' in script
    assert 'uv pip install --python "$candidate/bin/python"' in script
    assert '--index-strategy unsafe-best-match --requirements requirements.lock' in script
    assert script.index('write_runtime_marker "$candidate/bin/python"') < script.index(
        'mv -- "$VENV" "$backup"'
    )
    assert "uv pip sync" not in script
    assert 'PATH="$VENV/bin:$PATH" ./mas doctor --strict-runpod' in script

    runner = (ROOT / "runpod" / "run-episode.sh").read_text(encoding="utf-8")
    assert 'export PATH="${MAS_BIN_DIR:-/workspace/.local/bin}:$PATH"' in runner
