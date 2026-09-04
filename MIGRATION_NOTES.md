# V2 migration notes

Inspected the 2026-09-04 V2 beta ZIP from Google Drive, including requested docs, four V2 notebooks, config, src and tests. The source suite ran 413 tests successfully before migration.

Precedence conflict: the older combined `TRANSLATION_INSTRUCTIONS.md` is superseded for V2 by `V2_PROJECT_INSTRUCTIONS.md`, which explicitly defines correction-first TR and later ID handoffs. Thresholds must not be loosened to obtain PASS.

Remaining v0.1.0 blocker: wire preserved GPU inference/reconciliation implementation into the new CLI and validate it on RunPod with a real episode.
