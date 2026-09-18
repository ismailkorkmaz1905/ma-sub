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


def test_runpod_worker_doctor_does_not_require_or_contact_drive(tmp_path, monkeypatch, capsys):
    _configure(monkeypatch, tmp_path)
    monkeypatch.delenv("MAS_DRIVE_STRICT_REMOTE")
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: None if name == "rclone" else name,
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RunPod worker must not contact Drive")
        ),
    )

    assert cli.doctor(runpod_worker=True) == 0
    output = capsys.readouterr()
    assert "MAS_DRIVE_STRICT_REMOTE: MISSING" not in output.err
    assert "rclone_remote" not in output.out + output.err


def test_bootstrap_reuses_persistent_environment_and_installs_dependencies():
    script = (ROOT / "runpod" / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"' in script
    assert 'MAS_BIN_DIR="${MAS_BIN_DIR:-/workspace/.local/bin}"' in script
    assert "denoland/deno/releases/download/v2.9.5" in script
    assert "sha256sum --check --status" in script
    assert 'downloads.rclone.org' not in script
    assert 'export UV_INSTALL_DIR="$MAS_BIN_DIR"' in script
    assert "timeout 300 apt-get -o Acquire::Retries=3 update" in script
    assert "timeout 600 apt-get -o Acquire::Retries=3 install" in script
    assert 'RUNTIME_MARKER="$VENV/.mas-runtime-abi.json"' in script
    assert 'read_bytes().replace(b"\\r\\n", b"\\n")' in script
    assert '"format": "mas-runtime-abi-marker-1"' in script
    for binding in (
        '"requirements_sha256"',
        'sys.implementation.name',
        'platform.python_version()',
        'sys.implementation.cache_tag',
        'sysconfig.get_config_var("SOABI")',
        'platform.machine()',
        '"uv_version"',
        '"installed_distributions"',
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
    assert 'runtime ABI difference: {path}' in script
    assert script.index('runtime ABI difference: {path}') < script.index(
        'immutable runtime ABI mismatch;'
    )
    assert 'MAS_ALLOW_C0E3_VENV_ADOPTION:-}" == "1"' in script
    assert '"$VENV" != "/workspace/ma-sub/.venv"' in script
    assert '"$REQUIREMENTS_SHA256" != "$C0E3_REQUIREMENTS_SHA256"' in script
    assert 'c0e3c8e40def2d7672a5dd201cd5d65fde086d6f' in script
    expected_requirements_sha = (
        "1a47075cdac4e504a915ac23badeb0524baaede883fb0608c874acab5206918f"
    )
    assert expected_requirements_sha in script
    assert 'uv pip install --python "$VENV/bin/python"' not in script
    assert 'importlib.metadata.distributions()' in script
    assert 'from packaging.specifiers import SpecifierSet' in script
    assert 'SpecifierSet(f"=={version}").contains(' in script
    assert 'installed[name], prereleases=True' in script
    assert 'timeout 300 uv pip check --python "$VENV/bin/python"' in script
    assert 'torch.ones(1, device="cuda")' in script
    assert 'ctranslate2.get_cuda_device_count() <= 0' in script
    assert 'c0e3 direct requirement mismatch:' in script
    assert 'stale_candidates=("$VENV".rebuild.*)' in script
    assert 'remove_rebuild_tree "$stale_candidate"' in script
    assert '"$VENV".previous.*)' not in script.split('stale_candidates=', 1)[1]
    assert 'uv venv --python 3.11 --relocatable "$candidate"' in script
    assert 'mv -- "$candidate" "$VENV"' in script
    assert 'remove_rebuild_tree "$backup"' in script
    assert 'uv pip install --python "$candidate/bin/python"' in script
    assert '--index-strategy unsafe-best-match --requirements requirements.lock' in script
    assert script.index('write_runtime_marker "$candidate/bin/python"') < script.index(
        'mv -- "$VENV" "$backup"'
    )
    cache_clean = "timeout 120 uv cache clean"
    pre_clean = script.index(cache_clean)
    post_clean = script.index(cache_clean, pre_clean + 1)
    assert script.count(cache_clean) == 2
    assert pre_clean < script.index('if ! validate_c0e3_venv; then')
    assert script.index('uv pip install --python "$candidate/bin/python"') < post_clean
    assert post_clean < script.index('if [[ -n "${MAS_NETWORK_VOLUME_QUOTA_BYTES:-}"')
    assert post_clean < script.index('PATH="$VENV/bin:$PATH" ./mas doctor --strict-runpod-worker')
    assert script.count('uv venv --python 3.11 --relocatable "$candidate"') == 1
    assert "uv pip sync" not in script
    assert 'PATH="$VENV/bin:$PATH" ./mas doctor --strict-runpod-worker' in script

    opt_in_branch = script.split(
        'if [[ "${MAS_ALLOW_C0E3_VENV_ADOPTION:-}" == "1" ]]; then', 1
    )[1].split('  else\n    candidate=', 1)[0]
    assert 'uv venv' not in opt_in_branch
    assert opt_in_branch.count("exit 1") >= 6
    for diagnostic in (
        "venv expected=/workspace/ma-sub/.venv",
        "requirements expected=$C0E3_REQUIREMENTS_SHA256",
        "release directory missing or unsafe",
        "existing venv missing or unsafe",
        "runtime marker already exists",
        "validation failed; refusing candidate rebuild",
    ):
        assert diagnostic in opt_in_branch

    runner = (ROOT / "runpod" / "run-episode.sh").read_text(encoding="utf-8")
    assert 'export PATH="${MAS_VENV_DIR:-$ROOT/.venv}/bin:' in runner
    assert '${MAS_BIN_DIR:-/workspace/.local/bin}:$PATH"' in runner


def test_immutable_image_runtime_and_entrypoint_do_not_depend_on_volume_code():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "WORKDIR /opt/ma-sub" in docker
    assert "MAS_VENV_DIR=/opt/venv" in docker
    assert "openssh-server" in docker
    assert "uv pip install --no-cache" in docker
    assert "MAS_RUNTIME_MODE=image-build MAS_BIN_DIR=/usr/local/bin" in docker
    assert 'ENTRYPOINT ["/opt/ma-sub/runpod/container-start.sh"]' in docker
    script = (ROOT / "runpod/bootstrap.sh").read_text(encoding="utf-8")
    assert script.index('immutable runtime missing:') < script.index('curl --fail')
    assert script.index('immutable runtime ABI mismatch;') < script.index('timeout 120 uv cache clean')
    startup = (ROOT / "runpod/container-start.sh").read_text(encoding="utf-8")
    assert 'exec /opt/ma-sub/mas "$@"' in startup
    assert 'exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no' in startup
    assert 'AuthenticationMethods=publickey' in startup
