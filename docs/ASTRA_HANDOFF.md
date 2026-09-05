# Astra handoff

## Current boundary

The maintained package, strict CLI orchestration, historical source archive, runtime policy, Docker definition, and RunPod control scripts are in this branch. Legacy notebooks are reference-only under `legacy/`.

Real GPU inference has not been performed. Live Google Drive byte/SHA-256 readback and provider-side RunPod shutdown have not been verified. Review `main` directly. Do not create a stable tag on local test results alone.

The configured RunPod is `p54vvbyu76eztn` (`muhtemel-ask-ep12`). Its API credentials are stored outside Git in Windows User environment variables. The last recorded authenticated REST read observed `desiredStatus=EXITED`, but its exact observation time was not retained. Re-query before and after a real run. This does not verify automatic startup, remote execution, pipeline shutdown, or billing. The attached 50 GB network volume is `xgogcmey5o`; its last recorded rate was `$0.07/GB/month`, deriving to `$3.50/month`, but current pricing must be re-queried.

Episode 12 has retained historical MKV and subtitle material on Drive. Its exact remote sizes and hashes were not recorded. Do not use Episode 12 for the first real run. The local RunPod controller, complete source discovery, source identity checks, download watchdog, strict Pod preflight, handoff transfer, and fail-closed Drive publication are implemented and locally tested. No successful real-environment runtime verification is claimed.

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
.\mas.ps1 doctor
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

## In-Pod command after preflight

Store secrets in RunPod environment variables, not the repository or shell history. Configure the `gdrive` rclone remote before starting.

```bash
export RUNPOD_POD_ID='POD_ID'
export RUNPOD_API_KEY='API_KEY'
export MAS_DRIVE_STRICT_REMOTE='gdrive:MyDrive/Muhtemel_Ask_Subtitles/EPISODES'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
export MAS_YTDLP_COOKIES='/run/secrets/youtube-cookies.txt'
export MAS_MAX_RUNTIME_SECONDS=14400
export MAS_IDLE_TIMEOUT_SECONDS=1800
./runpod/run-episode.sh 11
```

The Windows entrypoint `.\mas.ps1 run 11` owns bounded Pod start, API and SSH readiness, deployment, streamed logs, handoff download/upload, secret cleanup, stop, and external EXITED polling. It performs local preflight before startup and refuses a dirty repository or non-EXITED Pod. These paths are locally tested but have not been exercised against the paid Pod.

For each ChatGPT handoff, preserve the returned ZIP exactly and resume with:

```bash
./runpod/run-episode.sh 11
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
- Pre-publication Drive inventory with byte counts and SHA-256 values for retained historical Episode 12 objects, plus their preservation receipt
- Post-publication Drive inventory proving that only the canonical final MKV, Turkish SRT, and Indonesian SRT files occupy the production episode target; preserved historical objects remain outside that target
- RunPod stop response, external `GET /v1/pods/{podId}` result, and billing screenshot/export

## External shutdown verification

Run this from outside the Pod after every exit path:

```bash
printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" | \
  curl --fail --silent --show-error --header @- \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
```

Accept compute shutdown only when provider state is `EXITED` and the console shows no active GPU compute charge. Stopped volume storage may continue to incur storage charges. A network-volume Pod may need explicit termination after retention requirements are satisfied.
