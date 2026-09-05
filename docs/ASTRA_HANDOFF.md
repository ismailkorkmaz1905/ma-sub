# Astra handoff

## Current boundary

The maintained package, strict CLI orchestration, historical source archive, runtime policy, Docker definition, and RunPod control scripts are in this branch. Legacy notebooks are reference-only under `legacy/`.

Real GPU inference has not been performed. Live Google Drive byte/SHA-256 readback and provider-side RunPod shutdown have not been verified. Review `main` directly. Do not create a stable tag on local test results alone.

## Review commands

Windows:

```powershell
git fetch --all --prune
git switch main
git status --short --branch
git rev-parse HEAD
& .\.venv\Scripts\python.exe -m pytest -q tests
& .\.venv\Scripts\python.exe -m pytest -q tests/regression/test_ep12_reliability.py
& .\.venv\Scripts\python.exe -m pytest -q tests/test_pipeline_runtime.py
git diff --check
docker build --check .
docker build -t ma-sub:astra .
```

Linux or RunPod:

```bash
git fetch --all --prune
git switch main
git status --short --branch
git rev-parse HEAD
./runpod/bootstrap.sh
./mas doctor
./mas test
docker build --check .
docker build -t ma-sub:astra .
```

## First real RunPod run

Store secrets in RunPod environment variables, not the repository or shell history. Configure the `gdrive` rclone remote before starting.

```bash
export RUNPOD_POD_ID='POD_ID'
export RUNPOD_API_KEY='API_KEY'
export MAS_DRIVE_STRICT_REMOTE='gdrive:MyDrive/Muhtemel_Ask_Subtitles'
export MAS_MAX_RUNTIME_SECONDS=14400
export MAS_IDLE_TIMEOUT_SECONDS=1800
./runpod/run-episode.sh 13
```

For each ChatGPT handoff, preserve the returned ZIP exactly and resume with:

```bash
./runpod/run-episode.sh 13
```

## Evidence to retain

- Git commit and container image digest
- GPU model, CUDA, PyTorch, WhisperX, and model identifiers
- Source byte count and SHA-256 before and after all stages
- Stage state, completed UIDs, timings, retries, and checkpoint bindings
- Exact TR and ID input/output packs and their manifests
- Strict finalization report and subtitle QA reports
- Local final byte counts and SHA-256 values
- Drive temporary upload, remote readback receipt, and exact-name publication
- RunPod stop response, external `GET /v1/pods/{podId}` result, and billing screenshot/export

## External shutdown verification

Run this from outside the Pod after every exit path:

```bash
printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" | \
  curl --fail --silent --show-error --header @- \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
```

Accept compute shutdown only when provider state is `EXITED` and the console shows no active GPU compute charge. Stopped volume storage may continue to incur storage charges. A network-volume Pod may need explicit termination after retention requirements are satisfied.
