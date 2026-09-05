# Astra review prompt

Copy the following prompt into a fresh Astra review session after the candidate commit and real-run evidence are available.

```text
Perform an independent release-blocking review of C:\Users\Ismail\CodeBase\ma-sub on the sole maintained production branch, main.

Authority and prohibitions:
- You may read the full repository, run tests, install required test dependencies, build Docker, inspect Git and PR state, and inspect supplied real-run artifacts and logs.
- Do not create or move a stable tag and do not alter external production data.
- Review the exact candidate commit. Report its full SHA.
- Treat notebooks and everything under legacy/ as reference-only. Production must execute through normal Python modules and ./mas.
- Do not weaken tests, schema, hashes, source immutability, timing authority, overlap policy, subtitle QA, or strict/emergency separation.
- Never infer real GPU, Drive, or RunPod success from mocks, fixtures, local unit tests, CI, Dockerfile parsing, or a stop-request HTTP response.

Read completely before evaluating:
1. AGENTS.md
2. README.md
3. docs/CODEX_HANDOFF.md
4. docs/EP12_PIPELINE_INCIDENT_REPORT_FOR_ASTRA.md
5. docs/EP12_ACCEPTANCE.md
6. docs/ARCHITECTURE_DECISIONS.md
7. docs/ASTRA_HANDOFF.md
8. MIGRATION_NOTES.md
9. All production code, configuration, tests, Docker, RunPod scripts, workflow files, and the full legacy source snapshot relevant to the port

Run at minimum on Windows:

git fetch --all --prune
git switch main
git status --short --branch
git rev-parse HEAD
git log --oneline --decorate -10
git branch --all
& .\.venv\Scripts\python.exe -m pytest -q tests
& .\.venv\Scripts\python.exe -m pytest -q tests/regression/test_ep12_reliability.py
& .\.venv\Scripts\python.exe -m pytest -q tests/test_pipeline_runtime.py
git diff --check
.\mas.ps1 doctor

Run at minimum on Linux or RunPod:

git fetch --all --prune
git switch main
test -z "$(git status --porcelain)"
git rev-parse HEAD
./runpod/bootstrap.sh
./mas doctor
./mas test
docker build --check .
docker build -t ma-sub:astra .

Review every acceptance boundary:
- The public project is MAS, not separate V1/V2 products. Remaining v2 identifiers must be justified only by legacy persisted-contract compatibility.
- ./mas run EPISODE discovers the exact full episode on the official channel for a new run and resumes safely after initialization; --source-url remains an explicit pre-initialization operator override.
- Source bytes and SHA-256 remain immutable across every stage and resume.
- Checkpoints are atomic and bound to source, inputs, configuration, code identity, outputs, and completed UIDs. Stale bindings fail closed.
- GPU-required ASR, acoustic review, and forced alignment fail before work when CUDA is unavailable. No silent CPU fallback exists.
- Network retries, connection/request timeouts, total retry budget, and no-progress watchdog are finite.
- Timing overrides cannot change text and require fresh preconditions.
- Different known speakers can overlap but remain separate. Same-speaker overlap fails. Unknown-speaker overlap is unresolved evidence. Text is never merged across speakers.
- The 15 Episode 12 incident intervals remain regression fixtures and are not falsely described as acoustically resolved.
- Long Turkish content cannot silently collapse to Ha?, Ne?, Ya., or Evet. without a semantic-shrink alarm.
- Indonesian handoff validation preserves exact UID, count, order, timing, and immutable Turkish text.
- Subtitle QA enforces line length, line count, CPS, cue-duration, timing, and coverage requirements.
- Strict and emergency folders, filenames, markers, receipts, and release claims cannot overlap.
- Gmail stage notifications do not expose credentials and notification failure does not corrupt state.
- YouTube cookies are accepted only from an explicit Netscape-format file, never logged or committed, and authentication failures do not trigger unsafe fallback.
- Drive publication is restricted to the final MKV, Turkish SRT, and Indonesian SRT files under the episode folder. It uses a temporary name, reads remote bytes back, compares exact byte count and SHA-256, and only then publishes the exact final name. Source media, code, logs, reports, and intermediate artifacts are never synced to Drive.
- RunPod success, failure, translation-wait, maximum-runtime, idle-timeout, and signal paths preserve atomic state and request shutdown.
- A local real-run command starts an EXITED Pod only when work is requested, waits with bounded timeout, launches the remote pipeline securely, streams logs, and verifies shutdown externally. It must not leave paid compute running after any exit path.
- The retained 50 GB network volume cost is accounted for separately. Ordinary file deletion must not be claimed to reduce its fixed allocation cost; permanent deletion may happen only after required evidence and deliverables are preserved.
- External observation confirms provider desired state EXITED and no active GPU compute charge. Report retained volume/storage billing separately.

Real GPU truth boundary:
- Require evidence from one full real episode, including GPU/CUDA/model identities, immutable source hash, stage timings, retries, checkpoint bindings, exact Turkish and Indonesian packs and manifests, strict QA reports, local final hashes, Drive readback receipt, RunPod stop response, external provider-state query, and billing evidence.
- Spot-check source audio against representative final cue start/end times, all incident intervals, overlap cases, and semantic-shrink alarms.
- If any evidence is absent, label that gate NOT VERIFIED. Do not convert it to PASS because code or tests look correct.

Return exactly:
1. Decision: PASS, FAIL, or BLOCKED
2. Candidate commit SHA and worktree status
3. Branch consolidation state and CI results
4. Test and Docker results with pass/fail/skip counts
5. Findings ordered by severity, each with file and line evidence
6. Acceptance matrix: automated, GPU, acoustic quality, Drive readback, RunPod shutdown, billing
7. Real evidence inspected and hashes recorded
8. Remaining blockers before stable tag
9. Exact reproduction and first real-run commands

PASS is allowed only when every automated and real-environment gate is directly verified. Otherwise explicitly state that the stable tag remains prohibited.
```
