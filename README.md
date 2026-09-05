# Muhtemel Ask Subtitles

MAS production engine turns one episode source into strict Turkish and Indonesian subtitle deliverables. Notebooks are retained only as read-only historical references under `legacy/`. Production execution uses Python modules and the `mas` CLI.

Real full-episode GPU inference has not been performed on this branch. A passing local suite is not evidence of model quality, CUDA compatibility, remote Drive persistence, or provider shutdown.

## Operator workflow

Requirements:

- Python 3.11
- NVIDIA GPU and working CUDA runtime for ASR, acoustic review, and forced alignment
- `ffmpeg`, `ffprobe`, Git, and `rclone`
- A configured `rclone` Google Drive remote
- RunPod Pod ID and API key when automatic provider shutdown is expected
- A Gmail app password when email stage notifications are expected

Local setup:

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock
./mas doctor
./mas test
```

Start a new strict run:

```bash
export MAS_DRIVE_STRICT_REMOTE='gdrive:MyDrive/Muhtemel_Ask_Subtitles'
export MAS_GMAIL_ADDRESS='your.account@gmail.com'
export MAS_GMAIL_APP_PASSWORD='GMAIL_APP_PASSWORD'
export MAS_NOTIFY_TO='your.account@gmail.com'
./mas run 13 --source-url 'SOURCE_URL'
```

Resume after either ChatGPT handoff:

```bash
./mas run 13
```

The CLI emits the exact expected ZIP path when it blocks for Turkish correction or Indonesian translation. Place only the returned exact ZIP in `translation_output/`, then run the same resume command. Never edit source media, pack manifests, immutable IDs, block order, Turkish text in the Indonesian return, or timing in a translation return.

Email notifications are enabled only when both `MAS_GMAIL_ADDRESS` and `MAS_GMAIL_APP_PASSWORD` are set. `MAS_NOTIFY_TO` defaults to the sending Gmail address. Use a Google app password, not the account password, and store it as an environment or RunPod secret.

Verify the credentials by sending one real message before starting an episode:

```bash
./mas notify-test
```

Useful commands:

```bash
./mas status 13
./mas doctor
./mas test
./mas clean 13
./mas clean 13 --destroy
```

`clean` is a dry run unless `--destroy` is supplied.

## Strict release boundary

- Missing CUDA is a hard failure for GPU stages. There is no silent CPU fallback.
- Source media is immutable after its SHA-256 is recorded.
- Timing overrides can change timing only. Text mutation is rejected.
- Verified cross-speaker overlap remains separate. Speaker text is never merged.
- Strict outputs live under `final/`. Emergency outputs live under `emergency/` and cannot create or replace strict markers.
- Local file existence is not delivery. Drive publication passes only after remote byte count and SHA-256 readback match.
- A RunPod stop response is a request, not proof of zero billing. Verify provider state from outside the pod.

## RunPod

Bootstrap an interactive Pod checkout:

```bash
./runpod/bootstrap.sh
export RUNPOD_POD_ID='POD_ID'
export RUNPOD_API_KEY='API_KEY'
export MAS_DRIVE_STRICT_REMOTE='gdrive:MyDrive/Muhtemel_Ask_Subtitles'
./runpod/run-episode.sh 13 --source-url 'SOURCE_URL'
```

`run-episode.sh` applies a 14,400-second maximum runtime and a 1,800-second no-log-progress timeout by default. Override them with `MAS_MAX_RUNTIME_SECONDS` and `MAS_IDLE_TIMEOUT_SECONDS`. On completion, failure, handoff wait, or watchdog termination, it requests a provider-side stop through RunPod's REST API. Keep the API key in an environment secret, never in the repository or command history.

Stopping a Pod releases its GPU but may retain billable volume storage. A Pod with a network volume may need termination instead of stop. Decide that separately after artifacts are verified.

See [architecture decisions](docs/ARCHITECTURE_DECISIONS.md), [incident acceptance](docs/EP12_ACCEPTANCE.md), and [Astra handoff](docs/ASTRA_HANDOFF.md).
