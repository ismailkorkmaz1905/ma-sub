# Eleven-stage workflow review - 2026-09-06

Decision: FAIL for end-to-end readiness. Episode 11 acoustic pilot: BLOCKED before paid startup. This is a code/workflow review, not a completed repair or an acoustic quality certificate.

Reviewed candidate: `21fd937` on main. Root reviewed the production orchestration and relevant producer/consumer paths while GPT-5.6 Sol checked Episode 11 pilot execution readiness. Production source files were not modified during this review. Sol's separate record is [here](EP11_SOL_PILOT_EXECUTION.md).

## What the eleven steps actually are

The authoritative enumeration here is `pipeline.py:STAGE_NOTIFICATION_NAMES` and the eleven `_stage` calls in `run`, not the shorter README illustration. Notifications and compute shutdown are cross-cutting lifecycle operations after/between these steps.

| Step | Code stage | Finding and needed change |
|---:|---|---|
| 1 | download | Source discovery/authentication occur after paid startup. Resume repeats full-media EOF verification and may retry optional captions. Freeze source/evidence identity and move network-only readiness before GPU allocation. See F3 and F6. |
| 2 | audio | Source-relative A/V offset handling exists. Extraction and full decode verification have no subprocess deadline, while the shell watches heartbeat output rather than useful progress. Add bounded media progress; retain timeline/hash checks. See F1. |
| 3 | raw_asr | Main decode and rescue work are coupled until a late recovery save. Optional caption identity and postprocessing settings share the raw input hash. Separate immutable decoder results, VAD, rescue calls and enrichment checkpoints without weakening provenance. See F2/F3. |
| 4 | tr_pack | Text-only handoff copies every immutable record; new review requests can regenerate the pack. Instructions still refer to Colab/notebook commands. Stable cue correction and a supported repair command are not implemented. |
| 5 | tr_return | Exact structural validation exists, including a local controller precheck. It is not proof that corrected words were spoken. Missing audio decisions flow to another GPU review instead of completing the text handoff. |
| 6 | audio_review | Large-v3 is reused as a reviewer; pending cases may receive text-conditioned decoding and exact-crop decoding. Those are supporting evidence, not independent speaker/timing truth. A complete episode can still stop for any unresolved review. |
| 7 | forced_alignment | Strict per-word score/coverage/drift/overlap gates remain mandatory. The cue pilot does not replace this production path or add production diarization. No further per-UID threshold tuning is justified by the pilot's unit tests. |
| 8 | id_pack | All acoustic, coverage, segmentation and Turkish timing gates must pass before any Indonesian pack is emitted. This is the serial bottleneck that holds the entire deliverable behind residual local uncertainty. New cue-contract integration is unfinished. |
| 9 | id_return | Structural record validation is strong, but readability and semantic checks happen later. There is no matching local Indonesian return precheck before Pod start. See F4. |
| 10 | finalize | Reconstructs/validates evidence and runs semantic/timing checks, then muxes and round-trips SRT. No final checkpoint shortcut exists in `pipeline.run`. Hard line width is 84, despite the acceptance document saying 42. See F5/F7. |
| 11 | drive_readback | Real byte/SHA readback is implemented. Prior exact-name inventory/preservation and an episode-level publish transaction are not. Partial and final copies are both fully read back; budget/placement must include that traffic. See F8. |

## Confirmed weaknesses, ordered by impact

### F1 - The no-progress guard can be kept alive by a stuck stage

Evidence: `pipeline.py:158-160` prints RUNNING every 30 seconds independent of action progress. `runpod/run-episode.sh:58-64` uses log modification time as its idle signal. `remote.py:64` also treats any stdout/stderr bytes as progress, while rclone is asked to print statistics every 10 seconds at `remote.py:111`.

Consequently, a blocked extraction/model call can keep the no-log watchdog satisfied until the overall wall deadline. The global deadline still exists; this is not a claim of unlimited execution. Media extraction and verification pass `timeout_seconds=None` at `media.py:242,275,571`, making the distinction material.

Fix direction: distinguish liveness from completed units/audio seconds/transferred bytes. Use stage-specific hard deadlines plus an external process-group/Pod stop observer. Log heartbeat must never reset the useful-progress clock.

### F2 - A successful expensive main ASR pass is not saved immediately

Evidence: `raw_asr.py:2809` completes `run_main_pass`; the rescue-budget check at approximately `2885-2900` can then raise. Rescue calls run in the loop starting near `2920`. The recovery checkpoint is first written at `3015`, after rescue and speech-hole record construction. `consume_coarse_segments` collects the primary iterator in memory.

If execution stops before that save, a fresh run has no persisted primary result from this attempt and can repeat main inference. Existing final raw checkpoints do resume; this finding concerns interruptions/failures before the late save. Historical Episode 11 primary ASR was reported as 986.5 seconds in `EP11_FIRST_PRODUCTION_RUN.md`; this review did not measure a new GPU duration or attribute every historical retry to this defect.

Fix direction: save raw decoder output immediately, then independent VAD/rescue stage records, with audio, exact model, options, dependency and producer identity. Do not simply drop binding fields from the existing checkpoint.

### F3 - Optional captions can invalidate downstream resume

Evidence: `download.py:_caption_retry_on_resume` retries missing captions during later source resumes and publishes newly available metadata/captions. `raw_asr.py:410-432` includes caption SHA in the raw input identity. `recover_raw_asr_v2_from_checkpoint` rejects caption-hash mismatch at `2318-2319`. `pipeline.run` always enters download before raw ASR.

Thus, a previously missing optional caption becoming available can invalidate the completed raw artifact/recovery even though source audio did not change. This is a demonstrated dependency path in code, not proof it caused the final Episode 11 conflict.

Fix direction: freeze the episode's caption snapshot when creating the canonical evidence set, or support a separately versioned enrichment stage with explicit downstream invalidation. Never silently rebind old raw evidence to new captions.

### F4 - Invalid or impractical Indonesian text is rejected too late

Evidence: `runpod_controller.py:61` provides `_validate_local_tr_return`; `run_remote_episode` has no corresponding Indonesian precheck. `id_translation.py:905-1023` validates exact echoes, nonempty target text and field types. It accepts a true review flag. Final readability/semantic gates are at `finalize.py:1232-1265`.

Local synthetic probe: a 1,000-character target string for a 1,250-millisecond cue with `review_required=true` passed `validate_id_translation_records`. This does not mean finalization accepts it. It proves the early PASS is structural and the actionable rejection is delayed.

Fix direction: validate both returned packs locally before compute, including target-text length/CPS, glossary/number rules and unresolved review flags. Give translators per-cue duration and text budgets. Return one complete repair report; do not discover each class through another full remote resume. Keep immutable timing/text-authority checks.

### F5 - Resume is stage re-entry, not a phase-specific execution plan

Evidence: `pipeline.run` calls all prior stages unconditionally on every invocation. Individual lower-level caches may avoid inference, but full source hashes/EOF checks, pack construction, artifact construction, notifications and later validation still execute. `make_id_pack` reconstructs the strict artifact set. `finalize_episode` is called on every final-phase invocation.

`config/runtime_policy.json` contains a 240-minute plan, including reserve and handoff waits. Repository search found `planned_stage_minutes` consumed by a test, not by runtime scheduling. The total 14,400-second controller guard is real, but it is not stage-level budget enforcement. The `Progress` class is likewise exercised by tests, not integrated into these production stages.

Fix direction: explicit local-preflight, GPU-evidence, local-handoff/QA and local-delivery phases, with reusable hash-bound results. Persist human wait separately from compute/stage measurements while retaining an honest overall wall clock. Do not reset the old deadline to pretend the episode met four hours.

### F6 - Source replacement protection is after publication in one recovery path

Evidence: `pipeline.py` guards the old source before calling `download_source`, then checks the returned source in `acquire` only after the downloader returns. In `download.py:1066-1075`, a resumed source's EOF-check exception falls through to reacquisition. `_publish_outputs` moves prior targets to a backup, replaces them, and cleans the backup after its own successful marker commit (`download.py:826-914`). The downloader is not passed the controller state's expected immutable source SHA.

A transient EOF-check failure followed by a different valid download can therefore replace the recorded source before the outer source-hash mismatch is raised. This is a code-path finding, not an observed Episode 11 overwrite. Existing download transaction tests do not prove preservation of the caller's immutable identity across this successful replacement path.

Fix direction: once source identity is recorded, failed validation must preserve it and fail closed. Any reacquired candidate must be checked against the expected source SHA before commit; different bytes need a separate explicit source revision, never replacement of the locked file.

### F7 - The stated line-width contract differs from production configuration

Evidence: `EP12_ACCEPTANCE.md` says at most 42 characters per line. `series.yaml:16` sets `qa_max_chars_per_line: 84`; `finalize.py:999,1259,1271,1277` passes it as the hard limit. `srt.py:142` treats 42 as a soft target and 84 as the hard maximum.

Local synthetic probe: 24 repetitions of `kata` wrapped to two lines of 59 visible characters under the production values. This proves the layout function permits more than the documented 42-character ceiling; it is not a claim that those lines passed the complete final episode QA.

Fix direction: choose one approved cue-readability policy and wire it through segmentation, translation instructions and final QA. Do not change the number solely to make tests pass. Also distinguish model-estimated sync from exact acoustic acceptance in the new mode.

### F8 - Drive publication lacks the required prior-object transaction

Evidence: `remote.py:100-133` hashes local data, uploads a partial object, reads it back, moves with `--immutable`, then reads the final object. There is no prior exact-name inventory with retained byte/SHA evidence. `pipeline.py:540-545` publishes files sequentially and saves the overall receipt after all three.

`--immutable` is a useful refusal mechanism, not the required preservation receipt or an atomic three-file episode publication. A later-file failure can leave earlier files published without an episode receipt. This review made no Drive calls and does not claim any historical object was overwritten.

Fix direction: inventory/preserve exact-name conflicts before upload, persist per-file verified receipts, and finalize an episode delivery marker only when the complete set is verified. Never weaken byte/SHA readback to save time; perform this non-GPU phase outside a billed GPU lease.

### F9 - Text correction, acoustic review and speaker identity remain different authorities

Evidence: the TR pack instructions explicitly prohibit listening, but use obsolete Colab/`01_PREPARE_TR` instructions (`tr_correction.py:191-260`). `audio_review.py:1840-1890` can retry pending cases with text-conditioned decoding by the same ASR model family. `pipeline.py` uses the same configured Whisper model for primary ASR and acoustic review, and has no diarization stage.

Repeated agreement is not independent proof of a different speaker, and text-conditioned review is not independent reference truth. Existing strict alignment remains a later gate, so this is an architectural limitation rather than a claim that every confirmed review is false.

Fix direction: independent regular speaker-turn acquisition, explicit uncertainty, one bounded aggregated acoustic review, and cue-level corrections. Replace obsolete operator instructions while preserving existing artifact compatibility. No claim that merely installing pyannote solves missed overlapping words.

## Scope, tests and remaining uncertainty

### F10 - The tampering test can fail to change its input

The full suite in this review returned **1 failed, 586 passed, 32 skipped in 37.57 seconds**. The failure is `test_episode_budget_checkpoint_tampering_fails` at `tests/test_runpod_controller.py:434-443`. It replaces `started_at` with another immediate `datetime.now()` rather than a guaranteed different timestamp. The expected integrity exception did not occur. Inspection of the retained test artifact found a valid checksum; the controller only passes this path when the supplied body still matches its checksum. The likely cause is equal clock readings, not demonstrated acceptance of changed checkpoint bytes. The same focused suite passed earlier.

Fix direction: derive a definitely different timestamp from the saved value (for example, plus one second), keep the original checksum unchanged, and assert the body actually changed before expecting rejection. No test or production code was edited to hide this failure. The current full-suite result must be reported as FAIL, not replaced with the earlier 587-pass result.

- Syntax inventory parsed all 52 Python modules under `src/mas`, containing 829 function definitions. This is not a claim that every function received line-by-line semantic review.
- Deep review covered the eleven-stage orchestrator and relevant download/media, raw ASR/recovery, TR/ID contracts, review, alignment integration, segmentation/timing/finalization, mux, Drive, notification, state/progress and RunPod lifecycle paths. Legacy reference notebooks were not treated as production.
- Focused existing suites: 137 passed in 10.89 seconds across pipeline runtime, RunPod controller, raw ASR, ID translation, timing QA and finalization. Synthetic probes above ran locally without model or external calls.
- Sol's focused pilot suite: 24 passed in 0.26 seconds. No new GPU compute or API inference. No Episode 12 run or artifact mutation.
- These tests mostly validate contracts and mocked execution. They do not establish acoustic recall, speaker correctness, end-to-end speed, billing protection or live Drive delivery. The review found gaps despite green tests.
- Exact attribution of the historical 18-hour incident across these architectural weaknesses requires matching each specific log and artifact. No new percentage or fabricated stage duration is reported here.

## Implementation order after this review

1. Protect immutable source replacement, useful-progress shutdown, and the balance reserve before any real run.
2. Save expensive model results before enrichment/rescue and freeze optional input snapshots. Establish a phase-specific resume plan and unified effective configuration.
3. Validate the cue/speaker approach on fixed contextual Episode 11 samples, then integrate cue-level TR correction, ID translation and early aggregate QA. Do not advertise the isolated prototype as the new production pipeline.
4. Move finalization/publication out of the GPU lease and implement collision-preserving delivery receipts.
5. Only after Episode 11 pilot review, assess authorization/readiness for a fresh episode against the four-hour target. Episode 12 remains out of scope now.

The review does not implement these changes. The pilot is blocked by source/model/runtime access, not merely by the former USD 0.50 ceiling. A larger spending allowance alone does not remove those prerequisites.
