# Muhtemel Ask Subtitles

This repository is the sole maintained Muhtemel Ask Subtitles production project. There is no separate V1 or V2 product. Public code belongs under `src/mas/`, configuration under `config/`, tests under `tests/`, and operator commands under `./mas`. Names containing `v2` inside legacy schemas or persisted artifact contracts are compatibility identifiers, not another product line.

The system downloads one episode source, extracts audio, performs GPU-only Turkish ASR and acoustic alignment, creates exact Turkish correction and Indonesian translation handoff packs, validates returned packs, emits strict subtitle artifacts, verifies Google Drive publication by byte count and SHA-256 readback, sends Gmail status notifications when configured, and requests RunPod shutdown. Notebooks and the imported source snapshot under `legacy/` are read-only reference material and are never production entrypoints.

Start by reading `README.md`, `docs/CODEX_HANDOFF.md`, `docs/EP12_ACCEPTANCE.md`, `docs/ARCHITECTURE_DECISIONS.md`, and `docs/ASTRA_HANDOFF.md`. Use `docs/ASTRA_REVIEW_PROMPT.md` only for the independent Astra review.

## Repository boundaries

- Use `main` as the sole maintained production branch. Consolidate pending implementation work into `main` and close obsolete feature branches and pull requests after their commits are preserved.
- Do not create a stable tag. A stable release requires Astra review and a real GPU episode run.
- Preserve user changes in a dirty worktree.
- Keep source media immutable after its SHA-256 is recorded.
- Never silently fall back to CPU for GPU-required stages.
- Never weaken schema, hashes, immutable fields, translation authority, timing rules, overlap rules, or subtitle QA to make a test pass.
- Keep strict and emergency outputs, receipts, markers, and paths completely separate.
- Do not report Drive delivery as PASS before remote byte-count and SHA-256 readback match.
- Different known speakers may overlap, but their text must never be merged. Same-speaker overlap fails. Unknown-speaker overlap remains review evidence.
- Network work must have finite retries, connection/request timeouts, and a no-progress watchdog.
- Keep credentials, Gmail app passwords, RunPod keys, rclone configuration, and YouTube cookies out of Git. Use environment variables or provider secrets.

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
./mas doctor
./mas status 13
```

Production entrypoints:

```bash
./mas run 13 --source-url 'SOURCE_URL'
./mas run 13
```

## Working style

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
