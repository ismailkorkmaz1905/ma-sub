#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MAS_VENV_DIR:-$ROOT/.venv}"
VENV="$(realpath -m -- "$VENV")"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
MAS_BIN_DIR="${MAS_BIN_DIR:-/workspace/.local/bin}"
export UV_CACHE_DIR
export PATH="$MAS_BIN_DIR:$PATH"
cd "$ROOT"

mkdir -p "$MAS_BIN_DIR"
if ! command -v deno >/dev/null 2>&1; then
  deno_archive="$(mktemp)"
  deno_extract="$(mktemp -d)"
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --connect-timeout 15 --max-time 300 --retry 3 \
    https://github.com/denoland/deno/releases/download/v2.9.5/deno-x86_64-unknown-linux-gnu.zip \
    --output "$deno_archive"
  echo "8b010a3b1a4a0188a67cdb8a7a27348b2a501af78aec7fc74f2ace167368d530  $deno_archive" \
    | sha256sum --check --status
  python3 -m zipfile -e "$deno_archive" "$deno_extract"
  install -m 755 "$deno_extract/deno" "$MAS_BIN_DIR/deno"
  rm -f "$deno_archive"
  rm -rf "$deno_extract"
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  for attempt in 1 2 3; do
    if timeout 300 apt-get -o Acquire::Retries=3 update; then
      break
    fi
    [[ "$attempt" -lt 3 ]] || exit 1
    sleep $((attempt * 5))
  done
  for attempt in 1 2 3; do
    if timeout 600 apt-get -o Acquire::Retries=3 install -y --no-install-recommends ffmpeg; then
      break
    fi
    [[ "$attempt" -lt 3 ]] || exit 1
    sleep $((attempt * 5))
  done
fi

if ! command -v rclone >/dev/null 2>&1; then
  archive="$(mktemp)"
  extract="$(mktemp -d)"
  trap 'rm -f "$archive"; rm -rf "$extract"' EXIT
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --connect-timeout 15 --max-time 300 --retry 3 \
    https://downloads.rclone.org/v1.75.0/rclone-v1.75.0-linux-amd64.zip \
    --output "$archive"
  python3 -m zipfile -e "$archive" "$extract"
  install -m 755 "$extract/rclone-v1.75.0-linux-amd64/rclone" "$MAS_BIN_DIR/rclone"
fi

command -v uv >/dev/null 2>&1 || {
  export UV_INSTALL_DIR="$MAS_BIN_DIR"
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --connect-timeout 15 --max-time 300 --retry 3 \
    https://astral.sh/uv/0.8.14/install.sh | sh
}

mkdir -p "$UV_CACHE_DIR"
mkdir -p "$(dirname "$VENV")"
RUNTIME_MARKER="$VENV/.mas-runtime-abi.json"

write_runtime_marker() {
  local python="$1"
  local target="$2"
  local ffmpeg_path
  local ffprobe_path
  ffmpeg_path="$(readlink -f -- "$(command -v ffmpeg)")"
  ffprobe_path="$(readlink -f -- "$(command -v ffprobe)")"
  REQUIREMENTS_SHA256="$(sha256sum requirements.lock | awk '{print $1}')" \
  UV_VERSION="$(uv --version)" \
  FFMPEG_PATH="$ffmpeg_path" \
  FFMPEG_SHA256="$(sha256sum "$ffmpeg_path" | awk '{print $1}')" \
  FFMPEG_VERSION="$("$ffmpeg_path" -version | sed -n '1p')" \
  FFPROBE_PATH="$ffprobe_path" \
  FFPROBE_SHA256="$(sha256sum "$ffprobe_path" | awk '{print $1}')" \
  FFPROBE_VERSION="$("$ffprobe_path" -version | sed -n '1p')" \
  "$python" - "$target" <<'PY'
import hashlib
import json
import os
import platform
import sys
import sysconfig
from pathlib import Path

import torch

os_release_path = Path("/etc/os-release")
os_release = platform.freedesktop_os_release()
data = {
    "format": "mas-runtime-abi-marker-1",
    "requirements_sha256": os.environ["REQUIREMENTS_SHA256"],
    "python": {
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
    },
    "uv_version": os.environ["UV_VERSION"],
    "distro": {
        "os_release": os_release,
        "os_release_sha256": hashlib.sha256(os_release_path.read_bytes()).hexdigest(),
        "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        "libc": platform.libc_ver(),
    },
    "torch": {
        "version": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
    },
    "ffmpeg": {
        "path": os.environ["FFMPEG_PATH"],
        "sha256": os.environ["FFMPEG_SHA256"],
        "version": os.environ["FFMPEG_VERSION"],
    },
    "ffprobe": {
        "path": os.environ["FFPROBE_PATH"],
        "sha256": os.environ["FFPROBE_SHA256"],
        "version": os.environ["FFPROBE_VERSION"],
    },
}
canonical = json.dumps(
    data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
wrapped = {"data": data, "sha256": hashlib.sha256(canonical).hexdigest()}
target = Path(sys.argv[1])
temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
temporary.write_text(
    json.dumps(wrapped, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target)
PY
}

remove_rebuild_tree() {
  case "$1" in
    "$VENV".rebuild.*|"$VENV".previous.*) rm -rf -- "$1" ;;
    *) echo "refusing unsafe bootstrap cleanup target: $1" >&2; exit 1 ;;
  esac
}

runtime_reusable=0
observed_marker="$(mktemp)"
if [[ -x "$VENV/bin/python" && -f "$RUNTIME_MARKER" ]] \
    && write_runtime_marker "$VENV/bin/python" "$observed_marker" \
    && cmp -s -- "$RUNTIME_MARKER" "$observed_marker"; then
  runtime_reusable=1
fi
rm -f -- "$observed_marker"

if [[ "$runtime_reusable" -ne 1 ]]; then
  candidate="$(mktemp -d "$VENV.rebuild.XXXXXX")"
  rmdir -- "$candidate"
  if ! uv venv --python 3.11 --relocatable "$candidate"; then
    [[ ! -e "$candidate" ]] || remove_rebuild_tree "$candidate"
    exit 1
  fi
  if ! timeout 1800 uv pip install --python "$candidate/bin/python" \
      --index-strategy unsafe-best-match --requirements requirements.lock; then
    remove_rebuild_tree "$candidate"
    exit 1
  fi
  if ! write_runtime_marker "$candidate/bin/python" \
      "$candidate/.mas-runtime-abi.json"; then
    remove_rebuild_tree "$candidate"
    exit 1
  fi
  backup=""
  if [[ -e "$VENV" || -L "$VENV" ]]; then
    backup="$(mktemp -d "$VENV.previous.XXXXXX")"
    rmdir -- "$backup"
    mv -- "$VENV" "$backup"
  fi
  if ! mv -- "$candidate" "$VENV"; then
    [[ -z "$backup" ]] || mv -- "$backup" "$VENV"
    [[ ! -e "$candidate" ]] || remove_rebuild_tree "$candidate"
    exit 1
  fi
  [[ -z "$backup" ]] || remove_rebuild_tree "$backup"
fi
timeout 120 uv cache clean
if [[ -n "${MAS_NETWORK_VOLUME_QUOTA_BYTES:-}" && -n "${MAS_EPISODE:-}" ]]; then
  PYTHONPATH="$ROOT/src" "$VENV/bin/python" -c '
import os
from pathlib import Path
from mas.engine.burned_mp4 import inspect_encoding_storage
from mas.reliability import atomic_json
evidence = inspect_encoding_storage("/workspace", network_volume_root="/workspace",
    network_volume_quota_bytes=int(os.environ["MAS_NETWORK_VOLUME_QUOTA_BYTES"]))
target = Path("/workspace/ma-sub/EPISODES") / ("Muhtemel Ask " + str(int(os.environ["MAS_EPISODE"])) + ".Bolum")
atomic_json(target / "work" / "storage-preflight.json", evidence)
if evidence["network_volume_free_bytes"] < 1000000000:
    raise RuntimeError("Network volume has less than 1 GB available for source/audio preparation")
'
fi
PATH="$VENV/bin:$PATH" ./mas doctor --strict-runpod
