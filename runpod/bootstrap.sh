#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  export UV_INSTALL_DIR=/usr/local/bin
  curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    https://astral.sh/uv/0.8.14/install.sh | sh
}

uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock
PATH="$ROOT/.venv/bin:$PATH" ./mas doctor
