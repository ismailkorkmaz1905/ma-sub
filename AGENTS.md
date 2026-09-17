# Muhtemel Ask Subtitles

This repository is the sole maintained Muhtemel Ask Subtitles production project. There is no separate V1 or V2 product. Public code belongs under `src/mas/`, configuration under `config/`, tests under `tests/`, and operator commands under `./mas`. Names containing `v2` inside legacy schemas or persisted artifact contracts are compatibility identifiers, not another product line.

The system downloads one episode source, extracts audio, performs GPU-only Turkish ASR and acoustic alignment, creates exact Turkish correction and Indonesian translation handoff packs, validates returned packs, emits strict subtitle artifacts, verifies Google Drive publication by byte count and SHA-256 readback, writes per-run local logs, sends Gmail status notifications when configured, and requests RunPod shutdown. Notebooks and the imported source snapshot under `legacy/` are read-only reference material and are never production entrypoints.

Start by reading `README.md`, `docs/CODEX_HANDOFF.md`, `docs/EP12_ACCEPTANCE.md`, `docs/ARCHITECTURE_DECISIONS.md`, and `docs/ASTRA_HANDOFF.md`. Use `docs/ASTRA_REVIEW_PROMPT.md` only for the independent Astra review.


## Delivery-first policy, 15 September 2026

For Episode 14 onward the default first-hour/local-QSV route prioritizes delivery.
See [the delivery-first contract](docs/DELIVERY_FIRST_2026-09-15.md). Bounded CTC
failures retain source cue timing, or omit only the unusable cue with a report.
Quality warnings must not become strict PASS, but they do not block this separate
delivery mode. Source/authentication/identity errors remain hard failures. Publish
the first approximately 60-minute burned-in MP4 first, then one full-episode MP4.
A real Indonesian return is required; the delivery path uses authenticated primary
Turkish ASR without requiring a separate manual Turkish correction/review handoff.
No paid run is authorized merely by editing or testing this code.

## EP14 launch repair, 18 September 2026

See `docs/EP14_READY_2026-09-18.md`. Existing-lease failure cleanup precedes no
new paid work and does not depend on a fresh quote or local readiness. Use
`./mas doctor --controller` for the Windows/controller role, not a local CUDA
worker check. EP14+ publishes part-001, keeps authenticated encoded tails locally,
then publishes one full MP4. Local tail completion is never a Drive PASS. Stored
status/quality metadata is not live service or perceptual certification.

## Repository boundaries

- Use `main` as the sole maintained production branch. PR #1 is merged and the obsolete feature branch has been deleted.
- Do not create a stable tag. A stable release requires Astra review and a real GPU episode run.
- Preserve user changes in a dirty worktree.
- Keep source media immutable after its SHA-256 is recorded.
- Never silently fall back to CPU for GPU-required stages.
- Never weaken schema, hashes, immutable fields, translation authority, timing rules, overlap rules, or subtitle QA to make a test pass.
- Keep strict and emergency outputs, receipts, markers, and paths completely separate.
- Do not report Drive delivery as PASS before remote byte-count and SHA-256 readback match.
- Before publishing an exact Drive filename that already exists, inventory and preserve the prior object with auditable byte-count and SHA-256 evidence. Never silently replace retained Episode 12 artifacts.
- Different known speakers may overlap, but their text must never be merged. Same-speaker overlap fails. Unknown-speaker overlap remains review evidence.
- Network work must have finite retries, connection/request timeouts, and a no-progress watchdog.
- Keep credentials, Gmail app passwords, RunPod keys, rclone configuration, and YouTube cookies out of Git. Use environment variables or provider secrets.
- Start paid RunPod compute only when an operator starts a real episode run. The controller must use bounded readiness timeouts, stream logs, preserve checkpoints, and verify shutdown from outside the Pod on every exit path.
- The last recorded 50 GB network-volume rate was `$0.07/GB/month`, deriving to `$3.50/month`; the exact observation time was not retained. Re-query provider state and pricing before cost decisions. File deletion does not reduce the allocation; permanent volume deletion requires preserved deliverables and explicit user approval.

## Required workflow

1. Read existing files before editing.
2. Run `git status --short --branch` and preserve unrelated changes.
3. Prefer small edits over rewrites.
4. Run focused tests, then the full suite before declaring completion.
5. Run `git diff --check`.
6. For runtime or release work, also check the Docker definition and the RunPod scripts.
7. Commit meaningful stages and push to `main` when authorized by the active task.
8. State explicitly whether a claim came from local tests, CI, a real GPU run, live Drive readback, or external RunPod observation.

## Commands

```powershell
git status --short --branch
& .\.venv\Scripts\python.exe -m pytest -q tests
git diff --check
.\mas.ps1 doctor
.\mas.ps1 status 13
```

Production entrypoints:

```bash
./mas run 11 --source-url 'SOURCE_URL'
./mas run 11
```

## Working style

- Do not load large JSON, SRT, or log files in full. Read the relevant ranges and records only.
- Inspect only the episode or job directory relevant to the task.
- Treat generated outputs as artifacts, not source code.
- Do not open audio or WAV files unless the user explicitly requests audio inspection.
- Read checkpoint and progress files selectively.
- Think before acting. Be concise in output and thorough in reasoning.
- Execute clearly scoped work without narration, status chatter, or confirmation requests.
- If a required step fails, state what failed, why, and what was attempted, then stop.
- Minimize expensive pipeline calls and output only information a person needs.
- Do not re-read a file unless it may have changed.
- Use the simplest working change. Do not add speculative features or premature abstractions.
- Three similar lines are preferable to a premature abstraction.
- Do not add docstrings or type annotations to untouched code.
- Do not add impossible-case error handling.
- For reviews, state the bug and the fix. Stop there.
- Do not praise code before or after a review and do not add out-of-scope suggestions.
- For debugging, read the relevant code and report what was found. If the cause is unknown, say `UNKNOWN`.
- Do not invent paths, APIs, functions, fields, test results, or citations.
- Distinguish observed facts from inference. Include units with numbers and say when data is missing.
- Lead reports with the finding. Prefer tables and bullets to long prose.
- Use plain ASCII punctuation in code and technical prose. Natural-language characters are allowed when needed.
- Return code first when code is requested. Add explanation only when it is necessary.
