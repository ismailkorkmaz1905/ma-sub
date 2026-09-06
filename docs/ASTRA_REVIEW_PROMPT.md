# Astra review prompt

Read [the 2026-09-06 review](ASTRA_REVIEW_2026-09-06.md) before this historical prompt. It corrects unsupported raw-ASR invalidation claims and records the current paid-operation prohibition and missing speaker evidence.

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
8. docs/EP11_FIRST_PRODUCTION_RUN.md
9. MIGRATION_NOTES.md
10. All production code, configuration, tests, Docker, RunPod scripts, workflow files, and the full legacy source snapshot relevant to the port
11. `EPISODES/Muhtemel Ask 11.Bolum/prepare/independent_audio_probe.json`, verified against self SHA-256 `6081ddb6670ca0893bd94a704dd7eb94a6865785c0a9dea58698239bd48e65a6` and file SHA-256 `8312910ddd11065a321362cab253c1033833829bcba2f73b3f04ef3174c996d2`

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
& .\.venv\Scripts\python.exe -m pytest -q tests/test_audio_review_ui.py tests/test_runpod_controller.py
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
- Distinguish bounded CTC diagnostic execution from acoustic acceptance and production forced alignment. Episode 11's pinned-model CUDA probe processed 138/138 pending records but produced 0 normalized exact reference matches and closed 0 strict decisions; it is not an acoustic PASS.
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
- Existing exact-name Drive objects are inventoried by byte count and SHA-256 and preserved with an auditable receipt before replacement. Retained historical Episode 12 material is never silently overwritten or deleted.
- RunPod success, failure, translation-wait, maximum-runtime, idle-timeout, and signal paths preserve atomic state and request shutdown.
- A local real-run command starts an EXITED Pod only when work is requested, waits with bounded timeout, launches the remote pipeline securely, streams logs, and verifies shutdown externally. It must not leave paid compute running after any exit path.
- Automatic capacity migration is allowed only for the exact no-free-GPU provider error. It must verify the old Pod is EXITED, preserve the exact network volume and data center, enforce the configured GPU type and hourly price ceiling, persist the replacement Pod ID, and avoid retrying an ambiguous create request.
- The retained 50 GB network volume cost is accounted for separately. Ordinary file deletion must not be claimed to reduce its fixed allocation cost; permanent deletion may happen only after required evidence and deliverables are preserved.
- RunPod state and storage pricing are re-queried during the review with observation timestamps; stale handoff values are not treated as current facts.
- External observation confirms provider desired state EXITED and no active GPU compute charge. Report retained volume/storage billing separately.

Real GPU truth boundary:
- Require evidence from one full real episode, including GPU/CUDA/model identities, immutable source hash, stage timings, retries, checkpoint bindings, exact Turkish and Indonesian packs and manifests, strict QA reports, local final hashes, Drive readback receipt, RunPod stop response, external provider-state query, and billing evidence.
- Reconcile the Episode 11 structured controller logs with `docs/EP11_FIRST_PRODUCTION_RUN.md`; do not treat controller elapsed time as exact provider billing time.
- Reconcile the post-`f581fcb59091d026e4f53a825916ba9e7abcf479` CTC report with its two recorded hashes, model `samil24/wav2vec-xlsr-53-turkish-v4` pinned to revision `07d79597b78c56758045e3a2cd1c44bc1a19b1e8`, CUDA evidence, 138/138 coverage, 0 normalized exact reference matches, and 0 strict closures. Verify that Pod `tccsb8991x84ua` was externally `EXITED` without converting that shutdown evidence into acoustic success.
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

## Emergency review addendum - 2026-09-06

Apply this addendum before the prompt above. The review starts now, before final real-run evidence, because the user explicitly requested an urgent Astra takeover after a costly incomplete run.

- Read `AGENTS.md`, `README.md`, `docs/CODEX_HANDOFF.md`, `docs/EP12_ACCEPTANCE.md`, `docs/ARCHITECTURE_DECISIONS.md`, `docs/ASTRA_HANDOFF.md`, and the complete Episode 11 run record first.
- Review candidate `33637d3a31ee49c82813ca947ffd3b56174f9571` and the later documentation commit. Preserve the dirty worktree if any.
- Do not start or migrate RunPod, execute `.\mas.ps1 run 11`, mutate Drive, delete volume `xgogcmey5o`, or incur paid services without explicit new user authority.
- Deep-review the whole production codebase and all Episode 11 controller/remote logs. Repair locally provable defects; run focused/full tests, `git diff --check`, and available Docker/static checks; commit and push meaningful stages to `main`.
- Explain the observed delay chain: missing persistent alignment-call reuse, unbounded overlap candidate paths, provider capacity/SSH failures, repeated fresh-Pod bootstrap, a late two-UID review cycle, and transfer/network retries. Raw-stage duration alone does not prove repeated transcription; code-driven raw-ASR invalidation is unsupported by its input digest. Distinguish log-derived seconds from the user's unverified report of nearly `21 hours` and approximately `$20`.
- Validate the local corrected-return ZIP using manifest SHA-256 `195f7d1509cb3592fb6ed207d1c6b088930a02006d4cfc338a3ec3f92cc6c9bf` and file SHA-256 `DA278A26B22BC507D40577E2658F7E9A48374154BD978B940125AD1CB1C7BC2A`. Its remote consumption is NOT VERIFIED.
- The exact unresolved pair is `MA11-TR-a09b20542760d351` and `MA11-TR-1144d768dc92c4a9`. Do not invent speaker identities or corrected text.
- Audit stale YouTube-cookie handling and Gmail status notification event/delivery evidence without exposing credentials.
- At medium priority, research current GPT transcription APIs using only official OpenAI documentation. Evaluate Turkish accuracy, word timestamps, diarization/overlap, cost, latency, privacy/retention, limits, retries, deterministic resume, artifact hashing, and compatibility with strict correction/alignment contracts. Give a grounded replace/complement/reject decision. Do not implement a substitution that weakens timing or evidence requirements.
- Current Pod evidence is an external API observation at `2026-09-06T01:35:52.8724964Z`: `781ct55zv4gkle` desired state `EXITED`, rate field `$0.74/hour`, retained volume `xgogcmey5o`. Re-query only if it can be done without starting compute.
- There is no Turkish SRT, Indonesian pack/return/SRT, strict mux, Drive readback, final delivery, or stable release. Keep each gate NOT VERIFIED until direct evidence exists.
