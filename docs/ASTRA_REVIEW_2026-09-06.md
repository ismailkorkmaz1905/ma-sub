# Astra review - Episode 11, 2026-09-06

Decision: FAIL for release readiness. Local repairs do not complete Episode 11. No paid GPU/API, Drive mutation, volume deletion, or Episode 12 operation was performed during this review.

Initial candidate: `4b90c2cf70935000103ef618b7013f469e6f3bdc`, clean `main`. Code reviewed beneath that handoff: `33637d3a31ee49c82813ca947ffd3b56174f9571`.

## Root cause: the acceptance contract outran the production evidence

The production correction schema in `src/mas/engine/tr_correction.py` has no speaker-identity field. `correction_records_to_alignment_inputs` in `src/mas/engine/workflow.py` constructs alignment inputs without speaker identity. Raw ASR has no diarization stage. Downstream validators correctly require independently known different speakers to accept overlapping words, but production has no path that supplies that evidence. The cross-speaker tests inject `speaker-a` and `speaker-b` directly. They prove validator behavior, not production speaker acquisition.

Consequently, the aligner attempts to remove unknown-speaker overlaps through joint, narrowed, and partitioned CTC windows, including cases that may represent real simultaneous dialogue. It cannot determine which case is a duplicate observation, an inaccurate boundary, or simultaneous speakers merely from the overlap. These require different evidence and recovery actions. No local text-only review can decide the remaining speakers reliably.

The retained `work/alignment_overlap_audit_result.json` records 2,805 inputs, 10,835 words, no normalization issues, and 1,434 word overlaps. Its first example binds `ikizine` in two UIDs to approximately the same 86,380 ms audio position. This establishes competing alignments; it does not independently establish whether either source utterance is a duplicate. `work/alignment_hybrid_audit_result.json` records 2,721 inputs, 1,306 initial overlaps, and 661 components. This was a systemic input/alignment ambiguity, not only two bad timestamps. These audit files are historical intermediate candidates, not final-run counts.

The old fallback then assigned distinct `acoustic-overlap-*` speaker IDs whenever each unresolved UID had `confirmed_dialogue`. That is invalid: confirming speech is not identifying different speakers. It manufactured the very evidence required by the validator. The fallback is removed, and persisted synthetic lane provenance is rejected. Unknown overlap remains unresolved.

## Why the loop consumed hours

1. Independent alignment processed every utterance again after a later failure. There was no persistent journal for successful model calls. Thousands of calls and follow-up candidates were repeated.
2. The Cartesian candidate search had an expanded-neighbor threshold but no hard bound on the fallback seed component or the earlier selection pass. Reducing the group did not remove exponential behavior.
3. The engine stopped at the first invalid utterance or displayed only the first final overlap pair. That encouraged repeated one-error-at-a-time full-episode attempts.
4. The four-hour `RunBudget` existed in a reliability helper but was not enforced across controller invocations. Each new invocation could obtain another per-run allowance. Handoff and stopped gaps did not consume a persisted episode budget.
5. Readiness timeouts could trigger migration despite the exact-capacity-error-only policy. Setup repeated on new hosts. Across the 63 logs, 26 contain fresh ffmpeg installation.
6. Turkish-return validation happened after starting and bootstrapping paid compute. Return transfer lacked the readback used for manual overrides. Network retry timeout restarted for each attempt.

All 63 controller invocations total 29,879.016 seconds. The 20 logged completed forced-alignment stages total 3,951.1 seconds. Interrupted stages and separate diagnostic probes are excluded from that alignment sum. The first logged alignment failure is in `run-20260905T122525.001811Z-4672.log`; the last is in `run-20260906T004815.388366Z-6628.log`. A 12-15-hour user wait must not be represented as one continuous alignment kernel.

The full wall interval from first start to retained external EXITED observation is 66,797.5752634 seconds: 2026-09-05 15:02:35.297233 through 2026-09-06 09:35:52.8724964 SGT. Every controller timestamp, commit, stage completion duration, and source-log hash is in [the complete inventory](EP11_CONTROLLER_LOG_INVENTORY.md). No billing export was supplied; approximately USD 20 remains user-reported.

## Corrections to the previous handoff

- The claim that Git changes broadly invalidated raw ASR is unsupported. `_raw_asr_input_sha256` contains configuration, audio, captions, names, terms and review UIDs, but no Git/code hash. The late 104-175-second raw stages cannot be labelled repeated GPU transcription from duration alone. Recovery can also validate and rebuild evidence. No raw ASR JSON/recovery file exists in the local retained prepare directory.
- `work/invalidate_raw_asr_checkpoint.py` explicitly moves ASR outputs/recovery and correction artifacts into a backup. It can start/migrate Pods. It was read only, never executed in this review. Future operators must not use it as a routine resume command.
- Raw ASR's missing stage-code identity is itself a release concern. Safe legacy evidence adoption requires the retained remote checkpoint and its generating implementation; silently changing its digest or deleting it is not a fix. That migration remains unverified.
- The local `TR_TEXT_CORRECTED.zip` contains 2,770 records and 1,000 pending flags, including the two newly requested UIDs. The retained `TR_CORRECTED.zip` contains 2,770 records and zero pending flags from the prior review.
- The old report states 998/998 resolved but binds provisional digest `820d627e1c2bcd06477858ca70fcd59ea5b9c4a08445861184877223bea813b1`. Validating it against the new provisional ZIP correctly raises `audio-review provisional SHA-256 mismatch`. Do not promote that historical PASS to the new return.

## Implemented local repairs

| Commit | Change | Evidence boundary |
|---|---|---|
| `651cdff` | Removed invented speaker lanes; hard candidate limit of 4,096 combinations on both selection paths; bounded outside-word search; checksum-bound model-call journal; completed alignment bound to current inputs, VAD, audio, relevant code and dependency lock; visible resumed flag | Local regression tests; GPU speedup not measured |
| `cd7dafa` | Persistent episode deadline anchored to earliest log; readiness/provider/setup/transfer/pipeline timeouts share it; bounded cleanup grace; precompute TR/cookie-format validation; return byte/SHA readback; capacity-only migration; finite cost guard and cleanup; shutdown GET failure still attempts stop; SMTP submission evidence; persistent uv/rclone binaries | Local controller/notification tests; no live provider mutation |
| `d12c46a` | Collect independent utterance validation failures in one pass; report all residual overlap UID pairs; adversarial 8,192-combination fixture must fail before search | Local synthetic tests; no acoustic decisions made |
| `0bea93a` | Bind journal and output provenance to actual loaded model tensor bytes, config and alignment dictionary; malformed provider JSON cannot skip stop; all post-run retrievals share 120 seconds before shutdown | Local weight-change and controller regressions; no GPU performance claim |

The journal stores raw model outputs under audio/model/version/device/code/dependency and exact-call hashes. The final review found that a model name alone could mix calls from changed model weights; actual loaded tensor bytes, configuration and alignment dictionary now also determine the journal namespace. This fingerprint is retained in output provenance, covered by the completed output digest. Cached results still pass current normalization, scores, timing, drift, VAD, and final validation. New correction context does not authorize new speaker IDs. An old completed alignment without the new input/output binding is preserved and rejected, not silently reused or overwritten. A fixed downloadable model snapshot is still not pinned; actual-weight fingerprinting prevents mixed cache reuse but does not guarantee future downloads reproduce old weights.

The final controller review found two additional cost risks: malformed JSON on the shutdown GET could bypass the stop call, and five diagnostic transfers could each receive 1,800 seconds before shutdown. These are repaired: invalid provider responses normalize into the caught controller error, and exit-code retrieval, diagnostics and handoff retrieval share one 120-second grace deadline. Expiry preserves remote artifacts and proceeds to shutdown instead of extending idle paid time.

## Notifications and environment

Fourteen historical SMTP warnings in `run-20260905T122525.001811Z-4672.log` report folded header linefeeds. Commit `d3aec98` had already repaired that header bug. This review adds sent/disabled/failed event records and Message-ID so later submission can be audited. SMTP acceptance does not prove inbox delivery; user-side receipt remains UNKNOWN. No test mail was sent during this review.

Cookie format validation now happens before compute, but cannot determine whether YouTube has revoked browser credentials. Retained yt-dlp logs report invalid cookies. A new episode still requires working authentication.

Local doctor returns successfully but reports ffmpeg, ffprobe and torch missing. Docker and Bash are unavailable on this Windows PATH. A local Linux image build and live GPU checks were not performed. Persistent uv/rclone reduces repeated downloads, but ffmpeg still needs a validated provider image; no image was deployed. The log-progress watchdog still measures output activity, including heartbeat lines; the newly enforced total deadline is the hard time bound, not proof of actual model progress.

## Architectural continuation boundary

Before another full paid run, the system needs a canonical utterance/evidence inventory that distinguishes duplicate observations, uncertain text, boundary uncertainty and independently established speaker overlap. Freeze correction identities against that inventory. Acquire speaker evidence before applying speaker-aware acceptance. Use the preserved call journal and one-pass diagnostics to repair only affected components. Do not run a series of full episodes to discover the next missing contract.

A verified speaker-evidence acquisition path, fixed model snapshot, raw-checkpoint adoption and real Turkish acoustic evaluation are still missing. They cannot be manufactured from local text or fake speaker fixtures. The four-hour target is not established until a representative real run meets it.

GPT transcription decision and current official sources are in [the evaluation](GPT_TRANSCRIPTION_EVALUATION.md): COMPLEMENT for an authorized sample, REJECT immediate forced-alignment replacement. It may improve text and avoid local ASR provisioning, but does not supply the exact word timing, acoustic scores and simultaneous-speaker evidence required here.

## Acceptance and resume

The user's latest practical target is synchronized, readable subtitles with different speakers kept separate, not perfect recognition. Finish Episode 11 before attempting another episode. This priority does not authorize invented speaker evidence, silent strict-gate bypass, or new spending. A simpler segment-level production contract may be considered separately, but the current word-level contract has not been silently relaxed.

| Gate | Result |
|---|---|
| Local regression suite | PASS: 563 passed, 32 skipped, 37.62 seconds |
| Local corrected-return schema and file/manifest SHA | PASS |
| Historical 998-review report against new provisional ZIP | FAIL, expected stale binding |
| Production speaker evidence | MISSING |
| Corrected real GPU forced alignment | NOT VERIFIED |
| TR SRT, ID pack/return/SRT, strict mux and final QA | NOT COMPLETE |
| Drive byte/SHA readback | NOT VERIFIED |
| Pod state | Historical external EXITED observation at 2026-09-06T01:35:52.8724964Z only |
| Billing total | UNKNOWN; user reports about USD 20 |
| Clean Linux image / end-to-end four-hour acceptance | NOT VERIFIED |

Paid compute remains prohibited. Eventual command: `.\mas.ps1 run 11`. The persisted episode origin remains 2026-09-05T07:02:35.297233+00:00. Because its 14,400-second allowance is exhausted, future explicitly authorized continuation also requires an operator-approved total `MAS_EPISODE_BUDGET_SECONDS` allowance; never reset or delete the origin. Preserve volume `xgogcmey5o` and all local/remote checkpoints. No stable tag.

## Local validation

- Full suite on final code candidate `0bea93a`: `563 passed, 32 skipped in 37.62s` using `.\.venv\Scripts\python.exe -m pytest -q tests`. Earlier candidate `d12c46a`: `559 passed, 32 skipped in 38.26s`.
- Final combined alignment/pipeline/controller focused suite: `100 passed in 2.30s`; controller focused suite: `42 passed in 0.55s`. Earlier controller/notification/preflight suite: `51 passed in 0.60s`.
- One new test initially failed because NumPy is absent locally. Its tensor fixture was replaced with standard-library packed float bytes; no runtime dependency was installed and the subsequent focused/full suites passed.
- `git diff --check`: PASS.
- Doctor: Python, Git and rclone available; ffmpeg, ffprobe and torch missing. Docker and Bash unavailable on PATH. Static runtime/script inspection only; no Linux image build or GPU run.
- Local schema and hash checks do not establish remote corrected-return consumption, SMTP inbox delivery, Drive readback or billing closure.
