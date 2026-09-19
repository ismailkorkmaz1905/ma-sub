# Astra review prompt

Use this only for an independent release review of a fixed candidate commit.

```text
Review C:\Users\Ismail\CodeBase\ma-sub on branch main as a release blocker.

Read AGENTS.md, README.md, docs/CODEX_HANDOFF.md,
docs/EP12_ACCEPTANCE.md, docs/ARCHITECTURE_DECISIONS.md,
docs/ASTRA_HANDOFF.md, docs/SEMANTIC_ALIGNMENT_EP15.md,
docs/EP14_READY_2026-09-18.md and the relevant production code, tests,
Dockerfile and runpod scripts.

Do not start paid compute, mutate Drive, move a stable tag, weaken validation or
alter retained episode evidence. Treat local tests, mocks and fixtures only as
local evidence. A real GPU run, Drive readback and external RunPod shutdown each
require their own direct evidence.

Run:
git status --short --branch
git rev-parse HEAD
& .\.venv\Scripts\python.exe -m pytest -q tests
git diff --check
.\mas.ps1 doctor --controller

Check source immutability, checkpoint bindings, GPU-only stages, semantic word
ownership, Turkish and Indonesian immutable fields, speaker overlap behavior,
subtitle QA, policy separation, bounded networking, Drive byte/SHA-256 readback,
old-object preservation, compute shutdown and episode budget persistence.

Return:
1. PASS, FAIL or BLOCKED
2. Candidate SHA and worktree status
3. Findings ordered by severity with file and line
4. Test and Docker results
5. Evidence matrix for local, GPU, acoustic, Drive and RunPod gates
6. Remaining blockers before a stable tag

PASS requires direct evidence for every applicable local and real-environment
gate. Missing real evidence is NOT VERIFIED, never inferred PASS.
```
