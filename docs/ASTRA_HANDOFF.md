# Astra handoff

## Current boundary

The maintained package, strict CLI orchestration, historical source archive, runtime policy, Docker definition, and RunPod control scripts are in this branch. Legacy notebooks are reference-only under `legacy/`.

Episode 11 reached the Turkish correction handoff on a real RTX 4090. Source
acquisition, audio extraction, raw ASR, correction-pack generation, controller
shutdown, and external `EXITED` verification passed. The successful controller
session took `1,168.031 seconds`; raw ASR took `986.5 seconds`. The complete
attempt and fix timeline is in `docs/EP11_FIRST_PRODUCTION_RUN.md`.

The Turkish correction return subsequently passed. Bounded audio review covered
245 records, resolved 107, and left 138 pending. After commit
`f581fcb59091d026e4f53a825916ba9e7abcf479`, an independent pinned-model CUDA
CTC probe processed 138/138 pending records with
`samil24/wav2vec-xlsr-53-turkish-v4` pinned to revision
`07d79597b78c56758045e3a2cd1c44bc1a19b1e8` on CUDA. It produced 0 normalized
exact reference matches and closed 0 strict decisions. Its report self SHA-256 is
`6081ddb6670ca0893bd94a704dd7eb94a6865785c0a9dea58698239bd48e65a6`; its
file SHA-256 is
`8312910ddd11065a321362cab253c1033833829bcba2f73b3f04ef3174c996d2`.
This is diagnostic execution evidence, not an acoustic PASS.

The latest controller run before that probe was attempt 23 at commit
`f581fcb59091d026e4f53a825916ba9e7abcf479`. Checkpoint-backed raw ASR passed
in 32.7 seconds. Bounded audio review processed 245/245 in 158.2 seconds and
stopped fail-closed with 107 resolved and 138 pending. The full 23-attempt
controller timeline, including capacity migrations and shutdown observations, is
in `docs/EP11_FIRST_PRODUCTION_RUN.md`.

Manual acoustic decisions, production forced alignment, live Google Drive
byte/SHA-256 readback, and the completed final episode have not been verified.
Review `main` directly. Do not create a stable tag from the partial Episode 11
run.

The latest independently observed probe Pod is `tccsb8991x84ua`; it was
externally verified `EXITED`. Credentials are stored outside Git in Windows User
environment variables. The retained 50 GB network-volume record is
`xgogcmey5o` in `EU-RO-1`; its last recorded rate was
`$0.07/GB/month`, deriving to `$3.50/month`, but current pricing must be
re-queried.

Episode 12 has retained historical MKV and subtitle material on Drive. Its exact remote sizes and hashes were not recorded. Do not use Episode 12 for the first real run. The local RunPod controller, complete source discovery, source identity checks, download watchdog, strict Pod preflight, handoff transfer, and fail-closed Drive publication are implemented and locally tested. Real GPU raw ASR and the independent CUDA CTC probe have runtime evidence; no successful end-to-end final episode or Drive delivery is claimed.

Bootstrap now reuses the persistent `/workspace` environment and caches, uses
dependency-resolving `uv pip install`, resolves across both trusted indexes, and
uses bounded download timeouts and retries. These corrections passed on the paid
Pod.

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

The Windows controller copies short-lived secret files and the local `gdrive` config to the Pod, then removes them before shutdown. Never place them in Git or shell history.

```bash
export RUNPOD_POD_ID='POD_ID'
export RUNPOD_API_KEY='API_KEY'
export MAS_DRIVE_STRICT_REMOTE='gdrive:Muhtemel_Ask_Subtitles/EPISODES'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
export MAS_YTDLP_COOKIES='/run/secrets/youtube-cookies.txt'
export MAS_MAX_RUNTIME_SECONDS=14400
export MAS_IDLE_TIMEOUT_SECONDS=1800
./runpod/run-episode.sh 11
```

The Windows entrypoint `.\mas.ps1 run 11` owns bounded Pod start, API and SSH readiness, deployment, streamed logs, handoff download/upload, secret cleanup, stop, and external EXITED polling. It performs local preflight before startup and refuses a dirty repository or non-EXITED Pod. Deployment, SSH, strict preflight, real RTX 4090 raw ASR, Turkish handoff download, controller shutdown, and external `EXITED` verification passed. The later pinned-model CUDA CTC probe also completed 138/138 and its Pod was externally verified `EXITED`, but it closed 0 strict decisions and is not an acoustic PASS. `.\mas.ps1 review-audio 11` now provides a localhost-only, hash-bound manual review UI without starting RunPod; the controller's byte/SHA-256-verified override upload is locally tested but has not yet been exercised in a real resumed Pod run. Manual acoustic decisions, production forced alignment, final pipeline completion, and Drive delivery remain unverified.

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
- Exact CTC probe report, both recorded SHA-256 values, pinned model/revision and CUDA identity, 138/138 coverage, 0 normalized exact matches, and 0 strict closures
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
