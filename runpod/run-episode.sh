#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: ./runpod/run-episode.sh EPISODE [--source-url URL]" >&2
  exit 64
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISODE="$1"
MAX_RUNTIME_SECONDS="${MAS_MAX_RUNTIME_SECONDS:-14400}"
IDLE_TIMEOUT_SECONDS="${MAS_IDLE_TIMEOUT_SECONDS:-1800}"
CHECK_SECONDS="${MAS_WATCHDOG_CHECK_SECONDS:-15}"
LOG_DIR="$ROOT/EPISODES/Muhtemel Ask ${EPISODE}.Bolum/logs"
LOG_PATH="$LOG_DIR/runpod-session.log"

for value in "$MAX_RUNTIME_SECONDS" "$IDLE_TIMEOUT_SECONDS" "$CHECK_SECONDS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "watchdog values must be positive integer seconds" >&2
    exit 64
  }
done

mkdir -p "$LOG_DIR"
cd "$ROOT"
touch "$LOG_PATH"

pipeline_pid=""
stop_on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$pipeline_pid" ]] && kill -0 "$pipeline_pid" 2>/dev/null; then
    kill -TERM -- "-$pipeline_pid" 2>/dev/null || true
  fi
  if [[ "$rc" -ne 0 && -n "${RUNPOD_POD_ID:-}" && "${MAS_EXTERNAL_RUNPOD_CONTROLLER:-0}" != "1" ]]; then
    "$ROOT/runpod/stop-pod.sh" || {
      echo "RunPod stop request failed. Stop the pod from an external controller now." >&2
      rc=70
    }
  fi
  exit "$rc"
}
trap stop_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

bash "$ROOT/runpod/preflight.sh" 2>&1 | tee -a "$LOG_PATH"

setsid stdbuf -oL -eL ./mas run "$@" --local > >(tee -a "$LOG_PATH") 2>&1 &
pipeline_pid=$!
started="$(date +%s)"
stop_reason=""

while kill -0 "$pipeline_pid" 2>/dev/null; do
  now="$(date +%s)"
  last_output="$(stat -c %Y "$LOG_PATH")"
  if (( now - started >= MAX_RUNTIME_SECONDS )); then
    stop_reason="maximum runtime ${MAX_RUNTIME_SECONDS}s exceeded"
    break
  fi
  if (( now - last_output >= IDLE_TIMEOUT_SECONDS )); then
    stop_reason="no log progress for ${IDLE_TIMEOUT_SECONDS}s"
    break
  fi
  sleep "$CHECK_SECONDS"
done

if [[ -n "$stop_reason" ]]; then
  echo "WATCHDOG: $stop_reason" | tee -a "$LOG_PATH" >&2
  kill -TERM -- "-$pipeline_pid" 2>/dev/null || true
  for _ in {1..6}; do
    kill -0 "$pipeline_pid" 2>/dev/null || break
    sleep 5
  done
  kill -KILL -- "-$pipeline_pid" 2>/dev/null || true
fi

set +e
wait "$pipeline_pid"
rc=$?
set -e

if [[ -n "$stop_reason" ]]; then
  exit 124
fi
exit "$rc"
