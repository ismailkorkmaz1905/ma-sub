# Muhtemel Ask Subtitles

One production pipeline downloads an episode, produces Turkish timing evidence,
creates Indonesian subtitles, burns the Indonesian subtitles into H.264/AAC MP4,
and verifies the published file by byte count and SHA-256 readback.

Production code is under `src/mas/`, configuration is under `config/`, tests are
under `tests/`, and the operator entrypoint is `./mas` or `.\mas.ps1`. The
`legacy/` tree is read-only reference material.

## Current policies

New Episode 15 and later states default to `semantic-block-v1`:

```text
source -> audio -> GPU Turkish ASR -> immutable word timeline
       -> semantic word-span handoff when needed -> Indonesian handoff
       -> subtitle QA -> local or remote encode -> Drive byte/SHA-256 readback
```

The ASR word timeline owns timing. ChatGPT Pro may correct Turkish text and select
owned word spans, but it cannot return timestamps. Python validates ownership,
derives display times, freezes block IDs, and runs release QA. First-hour and
whole-episode publication select from the same canonical final blocks.

Forced CTC remains available as the explicit `strict-ctc-v1` policy. Existing
Episode 14 and older states are not silently migrated. Episode 14 also retains
its separate delivery-first route. These policies must not share PASS markers or
mislabel semantic or emergency output as strict CTC output.

Read these current documents first:

- [Architecture decisions](docs/ARCHITECTURE_DECISIONS.md)
- [EP15 semantic alignment](docs/SEMANTIC_ALIGNMENT_EP15.md)
- [EP14 semantic replay](docs/EP14_SEMANTIC_REPLAY_2026-09-19.md)
- [EP14 launch repair](docs/EP14_READY_2026-09-18.md)
- [Delivery-first contract](docs/DELIVERY_FIRST_2026-09-15.md)
- [Codex handoff](docs/CODEX_HANDOFF.md)
- [Astra handoff](docs/ASTRA_HANDOFF.md)

Historical incident and run reports remain under `docs/` as evidence. They are
not current operator instructions when they conflict with the documents above.

## Requirements

- Python 3.11
- Git and OpenSSH
- FFmpeg and FFprobe
- NVIDIA CUDA for GPU-required worker stages
- RunPod credentials and SSH key for remote production
- A configured `rclone` Google Drive remote
- YouTube cookies when the source requires authentication
- Gmail credentials only when notifications are enabled

Keep all credentials outside Git in environment variables or provider secrets.

## Setup

Linux or RunPod:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match --requirements requirements.lock
./mas doctor
./mas test
```

Windows PowerShell controller:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match --requirements requirements.lock
.\mas.ps1 doctor --controller
.\mas.ps1 test
```

## Operator commands

```powershell
.\mas.ps1 doctor --controller
.\mas.ps1 status 15
.\mas.ps1 run 15 --source-url 'SOURCE_URL'
.\mas.ps1 run 15
```

The first run initializes the source and persists the run contract. Re-run the
same command after each requested handoff. When the semantic or Indonesian ZIP
is requested, use the exact printed input and output paths. Do not edit manifests,
IDs, order, source text, word ownership, timestamps, or hashes.

Exit code 29 means the semantic return is required. The controller collects the
pack, releases GPU capacity, and waits locally. A semantic return or Indonesian
return must not open a new GPU lease unless later work genuinely requires GPU.

Useful local commands:

```powershell
.\mas.ps1 status 14
.\mas.ps1 review-audio 13
.\mas.ps1 clean 13
```

`clean` is a dry run unless `--destroy` is explicitly supplied.

## Logs and checkpoints

Each run writes a separate local log under:

```text
EPISODES/Muhtemel Ask <N>.Bolum/logs/run-<UTC timestamp>-<PID>.log
```

`logs/LATEST` points to the newest log. Stage checkpoints and receipts live under
the episode `work/` and `final/` trees. Logs are evidence, not cache files. Do not
delete them as routine repository cleanup.

Long-running stages emit a bounded heartbeat. Useful progress is tracked through
the dedicated progress marker, not inferred from log chatter.

## Safety boundary

- Source media is immutable after its SHA-256 is recorded.
- GPU-required stages never silently fall back to CPU.
- Known different speakers may overlap, but their text is never merged.
- Same-speaker overlap fails; unknown-speaker overlap remains review evidence.
- Strict, semantic, delivery-first, and emergency artifacts keep separate identities.
- Local file existence is not delivery PASS.
- Drive PASS requires exact remote byte count and SHA-256 readback.
- Existing exact Drive filenames are inventoried and preserved before replacement.
- A provider stop response is not shutdown proof; verify state externally.
- Tests and fixtures do not authorize paid RunPod work.
- No stable release exists without independent review and a real GPU episode run.

## Development checks

```powershell
git status --short --branch
& .\.venv\Scripts\python.exe -m pytest -q tests
git diff --check
```

For runtime or release changes, also inspect `Dockerfile` and `runpod/`. Report
local tests, CI, real GPU execution, Drive readback, and external provider state
as separate evidence classes.
