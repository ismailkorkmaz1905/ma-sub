#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PATH="$ROOT/.venv/bin:$PATH" ./mas doctor --strict-runpod-worker
