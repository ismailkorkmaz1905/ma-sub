# Muhtemel Ask Subtitles

Muhtemel Ask Subtitles is the single production pipeline for turning one episode source into strict Turkish and Indonesian subtitle deliverables.

```text
source -> audio -> Turkish ASR -> acoustic review -> correction handoff
       -> alignment -> Indonesian handoff -> subtitle QA -> Drive verification
       -> notification -> RunPod shutdown request
```

Production code lives under `src/mas/` and runs through `./mas`. Notebooks and the imported source snapshot under `legacy/` are read-only reference material, not production entrypoints.

> Release status: real full-episode GPU inference has not been performed. Passing local tests do not verify model quality, CUDA compatibility, live Google Drive persistence, or provider shutdown. Do not create a stable tag until the Astra review and a real GPU episode run pass.

## Start here

- [Operator and architecture decisions](docs/ARCHITECTURE_DECISIONS.md)
- [EP12 incident acceptance criteria](docs/EP12_ACCEPTANCE.md)
- [Codex continuation handoff](docs/CODEX_HANDOFF.md)
- [Astra handoff](docs/ASTRA_HANDOFF.md)
- [Astra review prompt](docs/ASTRA_REVIEW_PROMPT.md)

## Requirements

- Python 3.11
- NVIDIA GPU with a working CUDA runtime for ASR, acoustic review, and forced alignment
- `ffmpeg`, `ffprobe`, Git, and `rclone`
- A configured `rclone` Google Drive remote
- A RunPod Pod ID and API key when automatic provider shutdown is expected
- A Gmail app password when email stage notifications are expected
- A Netscape-format YouTube cookies file when authenticated source download is required

## Local setup

Linux, macOS, or RunPod:

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock
./mas doctor
./mas test
```

Windows PowerShell:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv\Scripts\python.exe requirements.lock
.\mas.ps1 doctor
.\mas.ps1 test
```

## Run an episode

Configure secrets outside the repository:

```bash
export MAS_DRIVE_STRICT_REMOTE='gdrive:MyDrive/Muhtemel_Ask_Subtitles/EPISODES'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
export MAS_YTDLP_COOKIES='/run/secrets/youtube-cookies.txt'
```

Start a new strict run:

```bash
./mas run 13
```

On Windows PowerShell, use `.\mas.ps1 run 13`. No RunPod credentials are
required for `doctor`, tests, source discovery, or an offline fixture. A full
episode run requires a CUDA-capable environment. `RUNPOD_POD_ID` and
`RUNPOD_API_KEY` are needed only when the run is executing on a RunPod Pod and
automatic shutdown is expected.

For a new episode, MAS searches the official `@muhtemelaskdizi` videos page and
accepts only the exact full-episode title equivalent to `Muhtemel Ask 13. Bolum`.
Turkish accents and spacing may differ. Clips, trailers, previews, recaps, missing
matches, and ambiguous matches are rejected. The resolved watch URL is saved to
the episode's immutable `source/source.url`. Use `--source-url` only for an
explicit operator override before the episode is initialized.

Resume the same run after a Turkish correction or Indonesian translation handoff:

```bash
./mas run 13
```

The CLI prints the exact ZIP path it expects when a handoff blocks progress. Put only that returned ZIP in `translation_output/`, then run the same resume command. Never edit source media, pack manifests, immutable IDs, block order, Turkish text in the Indonesian return, or timing in a translation return.

Useful operator commands:

```bash
./mas status 13
./mas doctor
./mas test
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
- Drive is a delivery target, not a repository mirror. Only the final Turkish and Indonesian SRT files are uploaded into each episode folder. Source media, MKV output, logs, reports, code, and intermediate artifacts stay out of Drive.
- A RunPod stop response is a request, not proof of zero billing. Verify provider state from outside the pod.

## RunPod

Bootstrap an interactive Pod checkout, configure provider and delivery secrets, then start the episode:

```bash
./runpod/bootstrap.sh
export RUNPOD_POD_ID='POD_ID'
export RUNPOD_API_KEY='API_KEY'
export MAS_DRIVE_STRICT_REMOTE='gdrive:MyDrive/Muhtemel_Ask_Subtitles/EPISODES'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
export MAS_YTDLP_COOKIES='/run/secrets/youtube-cookies.txt'
./runpod/run-episode.sh 13
```

`run-episode.sh` applies a 14,400-second maximum runtime and a 1,800-second no-log-progress timeout by default. Override them with `MAS_MAX_RUNTIME_SECONDS` and `MAS_IDLE_TIMEOUT_SECONDS`. On completion, failure, handoff wait, or watchdog termination, it requests a provider-side stop through RunPod's REST API.

Keep provider keys and pipeline credentials in environment secrets, never in the repository or command history. Stopping a Pod releases its GPU but may retain billable volume storage. A Pod with a network volume may require termination instead of stop after artifacts are verified.
