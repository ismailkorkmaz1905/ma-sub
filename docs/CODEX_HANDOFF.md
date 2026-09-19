# Codex handoff

## Current production boundary

New EP15+ states use `semantic-block-v1`. The immutable ASR word timeline owns
timing, the ChatGPT Pro ZIP return owns difficult Turkish text and word-span
choices, and Python owns validation, timestamp derivation, block identity, and
release QA. GPT timestamps and unowned word IDs are rejected.

Exit code 29 is `SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED`. The controller collects
the pack, verifies GPU release, and resumes semantic and Indonesian handoffs
locally. First-hour and whole-episode delivery use the same canonical
`final_blocks.jsonl`. Forced CTC remains explicit `strict-ctc-v1`; it is not an
automatic semantic fallback.

Read:

- [EP15 semantic architecture](SEMANTIC_ALIGNMENT_EP15.md)
- [EP14 real-artifact offline replay](EP14_SEMANTIC_REPLAY_2026-09-19.md)
- [EP14 launch repair](EP14_READY_2026-09-18.md)
- [Delivery-first contract](DELIVERY_FIRST_2026-09-15.md)
- [Architecture decisions](ARCHITECTURE_DECISIONS.md)

## Verified locally

- Semantic word ownership, handoff validation, metadata quarantine, explicit
  component fallback, translation locking, scoped delivery, and controller
  pause/resume paths are covered by local tests.
- The EP14 replay used retained real artifacts read-only with 0 forced CTC calls.
- The replay reached complete word ownership but did not produce a release PASS.

## Not verified

- No real ChatGPT Pro return exists for the 606 unresolved EP14 replay windows.
- No EP15 real GPU episode, perceptual subtitle acceptance, live Drive readback,
  or external RunPod observation has certified the semantic route.
- Local tests are not paid-run authorization.

## Working rules

1. Preserve the dirty worktree, source media, episode evidence, and Drive history.
2. Do not weaken hashes, immutable fields, timing ownership, overlap policy, or QA.
3. Use `.\mas.ps1 doctor --controller` for the Windows controller role.
4. Run focused tests, the full suite, and `git diff --check` before completion.
5. Inspect `Dockerfile` and `runpod/` for runtime or release changes.
6. Keep local, CI, GPU, Drive, and provider claims explicitly separate.

Historical run and incident documents under `docs/` are evidence only. When they
conflict with the current documents linked above, the current contracts win.
