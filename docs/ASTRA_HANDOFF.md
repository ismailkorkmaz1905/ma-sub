> DELIVERY COMPLETE, 2026-09-06: Episode 12 1080p H.264/AAC MP4 with Indonesian subtitles burned in is complete and published. Drive object 1FAkmN0Z9HWcRrbfOT8C-ftmSXXiDdA2I was fully read before and after metadata-only rename: both actual reads match 8,978,040,872 bytes and SHA-256 3069df576bcf5d9ec88b1176ed3f016c2f210e511adcd216d38bf4b021f22675. Transport PASS; subtitle/perceptual quality remains REVIEW_REQUIRED. All owned MP4 Pods are externally ABSENT, retained Pod EXITED, volume preserved. Last balance USD 2.8764311039 at 14:01 UTC. No active GPU, encoder or delivery job remains.
>
> Local MP4: C:/Users/Ismail/CodeBase/ma-sub-archive-20260906/deliverables/Muhtemel Ask 12.Bolum.id.BURNED.REVIEW.mp4. Evidence: mp4-final-20260906T125824Z/recovery-complete.json under that archive. Earlier var/ and EPISODES/ paths also resolve under the archive. SSH -n -T and CUDA AV1 decode are in main; last local suite 742 passed, 5 skipped in 57.59 seconds. Pip/uv installer caches and failed video intermediates were cleaned; models, venv, sources and final artifacts preserved. Native single-stream download and direct-ID readback recovered the shared Google quota failure. Episode 13 has NOT started; use the updated desktop launcher and [plain-language report](SON_DURUM_VE_BOLUM_13.md). Earlier stopped/incomplete/active-job statements below are historical.

# Astra handoff

Latest Episode 13 takeover, 2026-09-07 Singapore: [preflight evidence and exact
remaining work](EP13_RUN_2026-09-07.md). BLOCKED_SOURCE_URL; operator was asked
for the official full-episode link. No paid compute or download started. External
inventory shows only retained Pod 781ct55zv4gkle EXITED and volume xgogcmey5o
50 GB. Balance observed at 2026-09-06 17:12:38 UTC: 2.8569866595 USD. SMTP quota
retry fix 8f2891b passed 744 local tests, 5 skipped; one status email was accepted.
Resume through `.\mas.ps1 run 13` only after the report's source, owned temporary
Pod, budget and sample prerequisites. Episode 12 remains unchanged. Earlier
Episode 11/12 next-action instructions below are historical.

Current pilot result, 2026-09-06: [five-clip Episode 11 ASR/SRT evidence](EP11_PILOT_RESULT_2026-09-06.md) supersedes runtime-only pilot status. Five real clips completed CUDA ASR; the final 54-cue draft remains REVIEW_REQUIRED with unresolved timing/readability/speaker review and no perceptual listening. Episode 12 was not started. Current blocker is acoustic/text acceptance of the produced candidate, not missing ASR output. Use its local `review.html` and retained raw evidence; the linked report gives the exact local replay command. Do not start another Pod merely to regenerate the draft. All five new pilot Pods were externally verified ABSENT; retained Pod 781ct55zv4gkle remains EXITED and volume xgogcmey5o is preserved. Gmail is blocked by its daily sending quota. Local cue repair has 706 passing tests, 5 skipped; no CI, acoustic or delivery PASS is implied. No new independent review round is requested by this continuation.

Current override, 2026-09-06: [Astra review](ASTRA_REVIEW_2026-09-06.md) supersedes the historical status and resume instructions below. Local repairs are not Episode 11 completion. Missing production speaker evidence and stale review binding remain blockers. No paid compute or Drive mutation without renewed explicit permission.

## Current boundary

The maintained package, strict CLI orchestration, historical source archive, runtime policy, Docker definition, and RunPod control scripts are in this branch. Legacy notebooks are reference-only under `legacy/`.

Episode 11 reached the Turkish correction handoff on a real RTX 4090. Source
acquisition, audio extraction, raw ASR, correction-pack generation, controller
shutdown, and external `EXITED` verification passed. The successful controller
session took `1,168.031 seconds`; raw ASR took `986.5 seconds`. The complete
attempt and fix timeline is in `docs/EP11_FIRST_PRODUCTION_RUN.md`.

The Turkish correction return subsequently passed. Bounded audio review covered
245 records, resolved 107, and left 138 pending under the policy used by that
run. After commit
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

Commit `1465c278486ad0f4ec70d85e84c287efdd30abfc` adds the hash-bound local
manual-review UI and byte/SHA-256-verified RunPod override transfer. Its local
suite passed with 504 tests and 32 skips in 44.10 seconds; focused UI/controller
tests passed 33/33 in 1.72 seconds. Main CI workflow run `33964604668` passed both
the test and Dockerfile jobs. The transfer implementation has not yet been
exercised in a real resumed Pod run.

The current operating decision is to replace a 138-clip manual-listening task
with a contextual machine-review policy using target audio and adjacent context.
The user will not manually listen to 138 clips. The new policy is not verified
until its code, focused/full tests, and a real resumed RunPod execution are
retained as evidence. Production forced alignment, live Google Drive
byte/SHA-256 readback, and the completed final episode have not been verified.
Review `main` directly. Do not create a stable tag from the partial Episode 11
run.

## Continuous execution rule

Answering status or operator questions must not terminate the active production
task. Continue until strict delivery or a genuine external blocker. If blocked,
record the exact blocker and retained evidence here, and record the exact resume
command for Astra. The current resume command is `.\mas.ps1 run 11`.

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

The Windows entrypoint `.\mas.ps1 run 11` owns bounded Pod start, API and SSH readiness, deployment, streamed logs, handoff download/upload, secret cleanup, stop, and external EXITED polling. It performs local preflight before startup and refuses a dirty repository or non-EXITED Pod. Deployment, SSH, strict preflight, real RTX 4090 raw ASR, Turkish handoff download, controller shutdown, and external `EXITED` verification passed. The later pinned-model CUDA CTC probe also completed 138/138 and its Pod was externally verified `EXITED`, but it closed 0 strict decisions and is not an acoustic PASS. The contextual machine-review replacement for manual review is the current plan, not a verified result; retain its code/test evidence and real resumed RunPod outcome before accepting it. Production forced alignment, final pipeline completion, and Drive delivery remain unverified.

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

## Urgent Astra takeover - 2026-09-06

The earlier boundary in this file is stale. Start from candidate `33637d3a31ee49c82813ca947ffd3b56174f9571` plus the documentation commit containing this section.

Current facts:

- Local full suite at the code candidate: `530 passed, 32 skipped in 39.74 s`. Focused forced-alignment suite: `35 passed in 0.33-0.40 s`.
- Real audio review completed `998/998` in `719.2 s` on one run. Three duplicate fragment resolutions were retained.
- Exact commit `33637d3` reran audio review from checkpoint in `18.5 s`, then forced alignment failed closed after `755.8 s` on exactly `MA11-TR-a09b20542760d351` versus `MA11-TR-1144d768dc92c4a9`.
- A regenerated local `Muhtemel Ask 11.Bolum_TR_TEXT_CORRECTED.zip` has those two exact UIDs pending audio review. Manifest SHA-256 is `195f7d1509cb3592fb6ed207d1c6b088930a02006d4cfc338a3ec3f92cc6c9bf`; ZIP SHA-256 is `DA278A26B22BC507D40577E2658F7E9A48374154BD978B940125AD1CB1C7BC2A`. Validate it before use. Remote upload/consumption is NOT VERIFIED.
- There is no Turkish SRT, Indonesian pack/return/SRT, strict mux, final QA, Drive byte/SHA-256 readback, or delivery PASS.
- User-reported impact is nearly `21 hours` and approximately `$20` RunPod spend. Do not present these as provider-verified billing.
- External API observation at `2026-09-06T01:35:52.8724964Z` showed Pod `781ct55zv4gkle` desired state `EXITED`, rate field `$0.74/hour`, and retained volume `xgogcmey5o`. Do not delete the volume.
- Paid RunPod compute is prohibited until the user explicitly reauthorizes it. The future command is `.\mas.ps1 run 11`; do not execute it now.
- YouTube cookies were rejected as stale/invalid by yt-dlp. Gmail `notify-test` submitted successfully, but long-term pipeline notification delivery is unresolved.

Review and repair priorities:

1. Reconcile every controller and remote log listed in `docs/EP11_FIRST_PRODUCTION_RUN.md` with code releases and checkpoints.
2. Establish raw-checkpoint generating-code provenance and actual recovery behavior. Code-driven raw ASR invalidation was not supported by the inspected input digest; do not treat it as a proven cause or silently rebind legacy artifacts.
3. Prove overlap candidate search is bounded and deterministic on adversarial and Episode 11-shaped fixtures. Audit the acoustic-lane provenance checks and the two remaining UIDs without inventing text or speaker identity.
4. Reduce fresh-Pod bootstrap repetition through the existing image/runtime design, preserving exact dependency evidence.
5. Audit transfer retries, finite timeouts, no-progress watchdogs, adoption/migration behavior, and shutdown on every exit path.
6. Audit Gmail event generation and delivery evidence without exposing secrets.
7. Evaluate GPT transcription APIs at medium priority using official OpenAI sources. Compare Turkish accuracy, word timestamps, diarization and overlap handling, cost, latency, privacy, rate limits, deterministic resume, hashes, and strict artifact contracts. Recommend replace, complement, or reject; do not implement blindly.
8. Make fixes, run focused tests, the full suite, `git diff --check`, and Docker/static checks possible without paid GPU. Commit and push meaningful stages to `main`.

Do not claim acoustic, Drive, billing, or end-to-end PASS from local tests. Do not start RunPod, mutate Drive, delete retained artifacts, touch Episode 12, weaken validation, or create a stable tag without new user authority and direct evidence.
