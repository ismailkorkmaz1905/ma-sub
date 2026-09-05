#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MAS_VENV_DIR:-$ROOT/.venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_CACHE_DIR
cd "$ROOT"

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
  install -m 755 "$extract/rclone-v1.75.0-linux-amd64/rclone" /usr/local/bin/rclone
fi

command -v uv >/dev/null 2>&1 || {
  export UV_INSTALL_DIR=/usr/local/bin
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    https://astral.sh/uv/0.8.14/install.sh | sh
}

mkdir -p "$UV_CACHE_DIR"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.11 "$VENV"
fi
uv pip install --python "$VENV/bin/python" --requirements requirements.lock
PATH="$VENV/bin:$PATH" ./mas doctor --strict-runpod
