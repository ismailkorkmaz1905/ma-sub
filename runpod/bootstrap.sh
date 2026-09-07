#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MAS_VENV_DIR:-$ROOT/.venv}"
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
  timeout 300 apt-get update
  timeout 600 apt-get install -y --no-install-recommends ffmpeg
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
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.11 "$VENV"
fi
timeout 1800 uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match --requirements requirements.lock
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
