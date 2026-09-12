# Muhtemel Ask Subtitles

Current outcome and next-run checklist: [simple Episode 13 handoff](docs/SON_DURUM_VE_BOLUM_13.md). Main delivery now targets H.264/AAC MP4 with Indonesian subtitles burned into the image. Episode 12 MP4 is complete and delivered with two full remote byte/SHA-256 readbacks. Perceptual subtitle acceptance remains REVIEW_REQUIRED. Historical episode evidence moved outside the repository as described in that handoff.

Muhtemel Ask Subtitles is the single production pipeline for turning one episode source into strict Turkish and Indonesian subtitle deliverables.

```text
source -> audio -> Turkish ASR -> acoustic review -> correction handoff
       -> alignment -> Indonesian handoff -> subtitle QA -> Drive verification
       -> notification -> RunPod shutdown request
```

Production code lives under `src/mas/` and runs through `./mas`. Notebooks and the imported source snapshot under `legacy/` are read-only reference material, not production entrypoints.

> Current operator status, 2026-09-06: Episode 12 was explicitly authorized and produced separate REVIEW subtitles after full GPU ASR, CTC and targeted repairs. The user also authorized review MKV/Drive delivery and an Episode 13 launcher. This does not establish strict or release acceptance. See the [readable Episode 12 / Episode 13 report](docs/EP12_TO_EP13_OPERATOR_REPORT.md), [run evidence](docs/EP12_RUN_2026-09-06.md), and [reusable Episode X prompt](docs/EPISODE_OPERATOR_PROMPT.md). The earlier [Astra review](docs/ASTRA_REVIEW_2026-09-06.md) remains historical evidence of unresolved strict gates. No stable tag.

## Start here

- [Mimari harita](docs/ARCHITECTURE_MAP.md)
- [Operator and architecture decisions](docs/ARCHITECTURE_DECISIONS.md)
- [EP12 incident acceptance criteria](docs/EP12_ACCEPTANCE.md)
- [Codex continuation handoff](docs/CODEX_HANDOFF.md)
- [Astra handoff](docs/ASTRA_HANDOFF.md)
- [Astra review prompt](docs/ASTRA_REVIEW_PROMPT.md)
- [Episode 11 first production run record](docs/EP11_FIRST_PRODUCTION_RUN.md)

## Requirements

- Python 3.11
- Git, OpenSSH, and the generated RunPod SSH private key on the Windows controller
- A configured local `rclone` Google Drive remote; its config is copied to the Pod only for the run
- A RunPod Pod ID and API key for automatic start and externally verified stop
- For automatic capacity migration: the exact network volume ID, data center,
  GPU type, and maximum hourly price in the `MAS_RUNPOD_*` user environment
  variables
- A Gmail app password when email stage notifications are expected
- A Netscape-format YouTube cookies file when authenticated source download is required

## Local setup

Linux, macOS, or RunPod:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match --requirements requirements.lock
./mas doctor
./mas test
```

Windows PowerShell:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match --requirements requirements.lock
.\mas.ps1 doctor
.\mas.ps1 test
```

## Run an episode

Configure secrets outside the repository:

```bash
export MAS_DRIVE_STRICT_REMOTE='gdrive:Muhtemel_Ask_Subtitles/EPISODES'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
export MAS_YTDLP_COOKIES='/run/secrets/youtube-cookies.txt'
```

Start a new strict run. Episode 11 is the first non-EP12 test candidate:

```bash
./mas run 11
```

On Windows PowerShell, use `.\mas.ps1 run 11`. This production command validates
all local secrets and files before spending money, starts the configured EXITED
RunPod, waits for API and SSH readiness, uploads the clean Git commit and
short-lived secrets, streams output into the local run log, and externally stops
and polls the Pod to EXITED on every exit path. It refuses a dirty repository or
an already-running Pod. Offline fixture runs stay local.

For a new episode, MAS searches the official `@muhtemelaskdizi` videos page and
accepts only the exact full-episode title equivalent to `Muhtemel Ask 13. Bolum`.
Turkish accents and spacing may differ. Clips, trailers, previews, recaps, missing
matches, and ambiguous matches are rejected. The resolved watch URL is saved to
the episode's immutable `source/source.url`. Use `--source-url` only for an
explicit operator override before the episode is initialized.

Resume the same run after a Turkish correction or Indonesian translation handoff:

```bash
./mas run 11
```

When a handoff blocks progress, the controller downloads the exact input ZIP to
the local episode `translation_input/` directory and stops the Pod. Put only the
returned ZIP at the exact printed path under local `translation_output/`, then
run the same command. The controller uploads that return ZIP and resumes the
persistent remote checkpoint. Never edit source media, pack manifests, immutable
IDs, block order, Turkish text in the Indonesian return, or timing in a
translation return.

Episode 11's returned Turkish correction ZIP has passed the return gate. Its
previous bounded audio review processed 245 records, resolving 107 and leaving
138 pending. An independent pinned-model CUDA CTC probe executed for all 138
pending records, but its 0 normalized exact reference matches and 0 strict
closures remain diagnostic evidence. The current contextual policy resolves
non-orphan boundary fragments only after all bounded acoustic passes, records
the adjacent-context and decode audit, and keeps final forced alignment
mandatory. Real GPU verification of that policy is still pending.

Resume the production run without manually reviewing 138 clips:

```powershell
.\mas.ps1 run 11
```

The local review UI remains an operator fallback for orphan captions or records
without adjacent context, and does not start RunPod:

```powershell
.\mas.ps1 review-audio 11
```

The command opens a localhost-only browser page. Each save verifies the current
report, correction input/output bindings, and WAV SHA-256 before atomically
updating `review/audio_review_overrides.json`. Stop the local server with
`Ctrl+C`. If fallback decisions are needed, resume with `.\mas.ps1 run 11`; the
controller verifies the override file by byte count and SHA-256 on the Pod
before the pipeline starts.

Useful operator commands:

```bash
./mas status 13
./mas doctor
./mas test
./mas review-audio 11
./mas clean 13
./mas clean 13 --destroy
```

`clean` is a dry run unless `--destroy` is supplied.

## Run logs

Every `./mas run EPISODE` invocation writes a separate UTF-8 log under:

```text
EPISODES/Muhtemel Ask 13.Bolum/logs/run-<UTC timestamp>-<PID>.log
```

`logs/LATEST` contains the newest log filename. Each log records UTC start/end metadata, elapsed seconds, exit code, Git commit, non-secret configuration readiness, console output, errors, and full exception tracebacks. Source URLs are redacted from logged command arguments. Gmail passwords, cookie values, RunPod keys, and other secret values are never written. The resumable machine state remains separately available in `work/state.json`.

RunPod also mirrors process output into `logs/runpod-session.log` so provider/watchdog shutdown incidents retain their final console history.

## Gmail notifications

Notifications cover run start, stage progress, handoff waits, failures, and final readiness. They are enabled only when both `MAS_GMAIL_ADDRESS` and `MAS_GMAIL_APP_PASSWORD` are set. `MAS_NOTIFY_TO` defaults to the sending address.

Use a Google app password, never the account password. Keep it in an environment variable or RunPod secret and verify it before an episode run:

```bash
./mas notify-test
```

## YouTube cookies

`MAS_YTDLP_COOKIES` must point to an exported Netscape-format cookies file. Keep the file outside the repository, mount or copy it into RunPod as a secret, and never paste cookie contents into logs, commits, issues, or handoff documents. An authentication failure does not permit an unauthenticated or lower-quality fallback.

The same cookies file is used for official-channel source discovery and source
download. Discovery has finite yt-dlp retries, socket timeout, total timeout, and
a no-progress watchdog.

## Strict safety boundary

- Missing CUDA is a hard failure for GPU stages. There is no silent CPU fallback.
- Source media is immutable after its SHA-256 is recorded.
- Timing overrides can change timing only. Text mutation is rejected.
- Verified cross-speaker overlap remains separate. Speaker text is never merged.
- Strict outputs live under `final/`. Emergency outputs live under `emergency/` and cannot create or replace strict markers.
- Local file existence is not delivery. Drive publication passes only after remote byte count and SHA-256 readback match.
- Drive is a delivery target, not a repository mirror. Only the final MKV, Turkish SRT, and Indonesian SRT files are uploaded into each episode folder. Source media, logs, reports, code, and intermediate artifacts stay out of Drive.
- Episode 12 already contains retained historical deliverables. Before a real run publishes an existing exact filename, preflight must inventory the prior remote object and preserve auditable byte-count and SHA-256 evidence. Do not silently overwrite or delete it.
- A RunPod stop response is a request, not proof of zero billing. Verify provider state from outside the pod.

## RunPod

Bootstrap an interactive Pod checkout, configure provider and delivery secrets, then start the episode:

```bash
./runpod/bootstrap.sh
export RUNPOD_POD_ID='POD_ID'
export RUNPOD_API_KEY='API_KEY'
export MAS_DRIVE_STRICT_REMOTE='gdrive:Muhtemel_Ask_Subtitles/EPISODES'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
export MAS_YTDLP_COOKIES='/run/secrets/youtube-cookies.txt'
./runpod/run-episode.sh 11
```

`run-episode.sh` applies a 14,400-second maximum runtime and a 1,800-second no-log-progress timeout by default. Override them with `MAS_MAX_RUNTIME_SECONDS` and `MAS_IDLE_TIMEOUT_SECONDS`. A RunPod worker only prepares and exports the verified delivery, then exits with the collection handoff code. The external controller owns Drive publication, secret cleanup, and externally verified shutdown; a standalone in-Pod run cannot publish or claim shutdown PASS.

When `MAS_RUNPOD_AUTO_MIGRATE=1`, an exact provider "not enough free GPUs"
response triggers bounded migration instead of failing the episode. The
controller first verifies that the old Pod is `EXITED`, its configured network
volume and data center match, and no local Pod volume is at risk. It then
terminates only that Pod, attaches the same network volume to an available Pod
of the configured GPU type, rejects a price above
`MAS_RUNPOD_MAX_COST_PER_HR`, persists the new Pod ID, and resumes. Other HTTP
500 errors do not trigger migration.

The controller, complete channel discovery, source identity guards, download
watchdogs, strict GPU/Drive preflight, handoff transfer, and external shutdown
polling are implemented and covered by local tests. Episode 11 verified real Pod
startup, SSH, CUDA raw ASR, handoff download, controller shutdown, and external
`EXITED` polling. A later independent pinned-model CUDA CTC probe processed
138/138 pending review records on Pod `tccsb8991x84ua`; the Pod was externally
verified `EXITED`. The probe closed 0 strict decisions and is not an acoustic
PASS. Production forced alignment, final mux, and Drive upload/readback remain
unverified. The
controller never starts the Pod during setup, `doctor`, tests, status, or fixture
runs.

Keep provider keys and pipeline credentials in environment secrets, never in the repository or command history. Stopping a Pod releases its GPU but may retain billable volume storage. A Pod with a network volume may require termination instead of stop after artifacts are verified.
