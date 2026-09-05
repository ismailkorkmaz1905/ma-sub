#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_POD_ID:?RUNPOD_POD_ID is required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required}"
[[ "$RUNPOD_POD_ID" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "RUNPOD_POD_ID contains invalid characters" >&2
  exit 64
}
[[ "$RUNPOD_API_KEY" != *$'\n'* && "$RUNPOD_API_KEY" != *$'\r'* ]] || {
  echo "RUNPOD_API_KEY must not contain line breaks" >&2
  exit 64
}

base="https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
response="$(mktemp)"
trap 'rm -f "$response"' EXIT

printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" | curl --fail --silent --show-error \
  --request POST \
  --connect-timeout 10 \
  --max-time 30 \
  --retry 2 \
  --retry-all-errors \
  --retry-max-time 90 \
  --header 'Accept: application/json' \
  --header @- \
  --output "$response" \
  "${base}/stop"

echo "RunPod stop accepted for ${RUNPOD_POD_ID}."
echo "Verify provider state from outside the pod using docs/ASTRA_HANDOFF.md."
