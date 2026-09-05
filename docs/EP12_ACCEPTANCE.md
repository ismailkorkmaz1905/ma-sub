# Episode 12 incident acceptance criteria

Source: `docs/EP12_PIPELINE_INCIDENT_REPORT_FOR_ASTRA.md`.

The incident's practical Indonesian SRT is an emergency/recovery artifact. It is not strict production evidence and cannot be promoted by renaming or copying it into `final/`.

## Automated gates

1. Process interruption resumes from the exact completed UID set.
2. Changed source, input ZIP, configuration, or code identity rejects stale checkpoints without deleting source media.
3. Different known speakers may overlap as separate cues; their text is never merged.
4. Same-speaker overlap fails; unknown-speaker overlap remains review evidence.
5. Review clip padding is context and is not counted as target speech coverage.
6. Silence or duplicate suffix passes only with explicit non-dialogue evidence.
7. A timing override containing text or stale preconditions fails schema validation.
8. Replacing a long sentence with `Ha?`, `Ne?`, `Ya.`, or `Evet.` raises a semantic-shrink review alarm.
9. Block count, UID, order, timing, or immutable Turkish text changes in an Indonesian return hard-fail.
10. Subtitle QA enforces at most 42 characters per line, at most 2 lines, configured CPS, and cue-duration limits.
11. Remote byte-count or SHA-256 mismatch prevents the final delivery marker.
12. Strict and emergency output paths, names, receipts, and markers cannot overlap.
13. The 15 intervals in `tests/fixtures/ep12_incident_intervals.json` remain regression fixtures anchored to report section 6.8.
14. GPU-required stages cannot silently execute on CPU.
15. Network retries, connection time, no-progress time, and total retry time are bounded.

The 15 interval fixture proves deterministic geometry and policy handling only. It does not claim that audio at those coordinates was listened to or acoustically resolved.

## RunPod and integration gates

These require a real external environment and are not satisfied by local tests:

1. CUDA preflight and model loading complete on the selected GPU.
2. One real episode completes ASR, acoustic review, forced alignment, strict finalization, and both handoffs.
3. Spot checks compare source playback with final cue start/end times.
4. Google Drive readback returns the exact uploaded bytes and SHA-256.
5. Success, failure, handoff wait, maximum runtime, and idle watchdog paths request Pod stop.
6. An observer outside the Pod confirms RunPod state `EXITED` and no active GPU compute charge.
7. Container image builds from a clean checkout and exposes a working `./mas` entrypoint.

## Non-negotiable decision rule

Elapsed-time targets never weaken translation quality, schema, hashes, immutable fields, acoustic evidence, or subtitle rules. Budget exhaustion preserves diagnostics and produces BLOCKED or FAIL, never PASS. No stable tag is allowed until Astra review and a real GPU episode run succeed.
