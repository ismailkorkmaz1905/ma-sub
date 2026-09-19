#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MAS_VENV_DIR:-$ROOT/.venv}"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required" >&2
  exit 1
}
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required" >&2
  exit 1
}

[[ -x "$VENV/bin/python" ]] || uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -r requirements.lock
uv pip check --python "$VENV/bin/python"
echo "Ready. Run: ./mas doctor"
