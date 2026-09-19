## Alignment follow-up - 19 September 2026

The independently reviewed B1-B4 patch is now applied: noncontiguous conflict
requests bind UID/text/window together, repeated context searches keep scanning,
joint/context word slicing preserves punctuation, and request UIDs are no longer
replaced by merged budget UIDs. Three regression files cover these changes.
The qualification workflow runs the full Python 3.11 suite before committing.
B1/B2 use the explicit `recovery_plan` route; legacy retry permissions are not
automatically upgraded. Existing EP14 evidence needs a freshly validated plan
and exact-commit retry authorization, not an unqualified `run 14` restart.
No episode, paid GPU, live media replay or Drive operation was performed.
This is a code fix, not proof of acoustic resolution of the two remaining UIDs.

# Codex handoff

Current EP14 launch repair: [18 September 2026 repair and operator checks](EP14_READY_2026-09-18.md).
This supersedes older controller cleanup, all-tail publication and doctor guidance below.
First-part Drive readback is followed by verified local tail encoding and one full MP4.
Real acoustic quality, target runtime and the live controller environment are not certified.

Current audit and repair: [16 September 2026 A-Z results](AZ_REPAIR_2026-09-16.md). Delivery-first remains the EP14+ policy; real acoustic quality and the four-to-six-hour target are not yet certified.

## Current architecture update - 2026-09-12

This section supersedes older architecture and resume descriptions below where
they conflict. The changes are locally implemented and tested; they do not by
themselves prove a real Episode 13 GPU, Drive, shutdown or delivery PASS.

Current flow:

```text
source -> audio -> raw ASR -> TR return -> audio review -> forced alignment
       -> ID return -> strict_finalize -> burn_mp4 -> exit 22 collection
       -> owned GPU Pods ABSENT -> Drive upload/readback
```

- Completed audio review hydrates only after the final Turkish ZIP and report
  match the exact current review input, configuration, implementation and manual
  overrides. `state.json` alone never skips this work.
- Forced alignment now creates `forced_alignment_v2.done.json` after the current
  audio, corrected text, VAD, code/dependency binding and aligned output pass.
  Its details bind the alignment digest and exact validated correction-output
  digest. A legacy aligned output is adopted only after the current full binding
  validator passes.
- `strict_finalize` and `burn_mp4` are separate. The strict marker binds all
  canonical evidence/configuration inputs, relevant producer code, PASS report,
  softsub MKV and both SRT files. MP4 reuse remains independently bound to the
  source, ID SRT, style, encoding settings, sample approval, output and receipt.
- Exit 22 is not recoverably complete until controller collection succeeds. If
  an expected external `/tmp/mas-epN-output/` MP4 or receipt is missing, a
  checksum-bound collection-failure record tied to the exact request and attempt
  advances the next remote-job identity. The episode budget is not reset.
- Drive publication now occurs after the capacity lease closes and bound
  capacity/shutdown evidence proves every controller-owned temporary Pod is
  `ABSENT`. If Drive then fails, a verified release marker enables a
  transfer-only retry before RunPod/Git/SSH preflight; that path cannot acquire
  GPU compute.
- Runtime bootstrap reuse now has an ABI marker, but there is still no proven
  immutable container image digest. The repository also does not yet demonstrate
  a complete transitive hash lock covering every Python dependency, native
  library, model and base-image component. Both remain open release risks.

> DELIVERY COMPLETE, 2026-09-06: Episode 12 1080p H.264/AAC MP4 with Indonesian subtitles burned in is complete and published. Drive object 1FAkmN0Z9HWcRrbfOT8C-ftmSXXiDdA2I was fully read before and after metadata-only rename: both actual reads match 8,978,040,872 bytes and SHA-256 3069df576bcf5d9ec88b1176ed3f016c2f210e511adcd216d38bf4b021f22675. Transport PASS; subtitle/perceptual quality remains REVIEW_REQUIRED. All owned MP4 Pods are externally ABSENT, retained Pod EXITED, volume preserved. Last balance USD 2.8764311039 at 14:01 UTC. No active GPU, encoder or delivery job remains.
>
> Local MP4: C:/Users/Ismail/CodeBase/ma-sub-archive-20260906/deliverables/Muhtemel Ask 12.Bolum.id.BURNED.REVIEW.mp4. Evidence: mp4-final-20260906T125824Z/recovery-complete.json under that archive. Earlier var/ and EPISODES/ paths also resolve under the archive. SSH -n -T and CUDA AV1 decode are in main; last local suite 742 passed, 5 skipped in 57.59 seconds. Pip/uv installer caches and failed video intermediates were cleaned; models, venv, sources and final artifacts preserved. Native single-stream download and direct-ID readback recovered the shared Google quota failure. Episode 13 has NOT started; use the updated desktop launcher and [plain-language report](SON_DURUM_VE_BOLUM_13.md). Earlier stopped/incomplete/active-job statements below are historical.

> DRIVE LAYOUT UPDATE, 2026-09-06: Muhtemel_Ask_Subtitles (1Gdn4WLjICGJNsYIMCSpSNyXgA_8mdL5p) now directly contains only the same verified MP4 object. Live listing confirmed one file and no subfolders. The old EPISODES tree was moved outside delivery to My Drive as "Muhtemel Ask - Eski teslim arsivi - 20260906" (same folder ID 1baEW_vbsemXqM-K1or6SOEAyP63DCGpI). Native trash attempts hit shared-project quota HTTP 403; browser access was unavailable. Old MKV/SRT are archived, NOT deleted. The old delivery path is historical; inventory current Drive before Episode 13 publication.

LATEST DELIVERY STATE AFTER INTERRUPTION, 2026-09-06: final local Episode 12
review SRTs and verified MKV are in `work/review-delivery-20260906-final-utf8/`.
All 228 native chunks completed; 3,055 source cues render as 2,838 rows. Sol's
final UTF-8 artifact verification passed; perceptual listening remains absent.
Drive upload failed at 11:22:02 UTC with Google shared OAuth project per-minute
quota HTTP 403. No remote byte/SHA receipt exists. No native/upload process
remained on interruption recovery. The newly required lean-ctx shell then blocked
rclone and its MCP transport closed during the suggested additive allow command.
Restore terminal access, inventory Drive, resume upload-only without rebuilding
or replacing existing MKV, then finish docs commit/push. No new GPU is needed.
Code commit: `9332295`; local suite 741 passed, 5 skipped. See
[readable final report and Episode 13 actions](EP12_TO_EP13_OPERATOR_REPORT.md).
Desktop Episode 13 Astra shortcut exists; Episode 13 has NOT started.

LATEST REAL PILOT, 2026-09-06: five Episode 11 contextual clips completed CUDA ASR on NVIDIA L4. Final local candidate is `work/subtitle-pilot/contextual-asr-20260906-4/` under Episode 11: 54 cues covering 155 seconds. Raw results and all failed setup attempts are retained. The pilot is REVIEW_REQUIRED, not accepted: 5 short-cue warnings, 12 reading-speed warnings and 41 uncertain-speaker words remain; perceptual listening was NOT_PERFORMED. Episode 12 was not started. See [actual pilot output, repairs and exact continuation](EP11_PILOT_RESULT_2026-09-06.md). The cue fragmentation repair passed 706 local tests, 5 skipped, in 58.93 seconds; all 54 real cues independently preserve text and exact original word boundaries. Latest external snapshot: balance USD 3.6213915897, retained Pod 781ct55zv4gkle EXITED, volume xgogcmey5o preserved at 50 GB; all five newly created pilot Pods externally ABSENT. Gmail is blocked by verified SMTP 550 5.4.5 daily sending limit, with later events retained locally. Do not rerun GPU ASR merely to regenerate this SRT. Existing uncommitted documentation and ignored var evidence were preserved; HEAD remains 2d7e71a7768037b1337e8f6b64d9ff072386d4ea with local cue/test/doc changes not committed.

## Current override - 2026-09-06 Astra review

LATEST EXECUTED QUALIFICATION: disposable RTX 4000 Ada Pod `ullkyc3b73lsc6` was created at USD 0.28/hour, inspected and explicitly terminated. External absence was confirmed at 2026-09-06T06:47:08.760033Z. Retained Pod `781ct55zv4gkle` remains EXITED and 50 GB volume `xgogcmey5o` remains present. CUDA 12.8 is available through retained PyTorch 2.8.0+cu128; faster-whisper 1.2.1, CTranslate2 4.8.1 and the large-v3 model are present. stable-ts and openai-whisper are missing. This is runtime inventory, NOT model inference or subtitle-quality PASS. Inventory worker took 58.75 seconds; all eight stage notifications were SMTP-accepted. Evidence: `var/disposable-qualification-20260906-1/`. Latest full local suite is 704 passed, 5 skipped in 73.06 seconds. Next: prepare isolated missing dependencies and execute five contextual ASR samples under another bounded lease, then evaluate cue/speaker evidence. Episode 12 must wait for pilot acceptance. Older readiness and test-count statements below are historical.

LATEST PROVIDER BOUNDARY: two bounded qualification attempts on retained Pod `781ct55zv4gkle` ended before SSH/ASR and externally confirmed EXITED. The second recorded provider_start HTTP 500 capacity rejection; first cause is UNKNOWN because the older diagnostic discarded it. Last guardian observation: 2026-09-06T06:17:27.154781Z; displayed balance USD 3.7349020257, not invoice evidence. Do not repeat the same capacity-bound start or use the old migration helper, which deletes the old Pod. A cheaper disposable qualification path with atomic provider `terminateAfter`, unique creation identity and explicit termination/readback is under development, not yet executed. ASR-only CUDA worker is implemented and focused-tested; final combined suite is pending draft qualification reconciliation fixes. Detailed attempts, durations, mail evidence and live price-query artifacts are in WORKFLOW_REPAIR_PROGRESS.md. Episode 11 final subtitles and Episode 12 remain incomplete.

LATEST LOCAL EVIDENCE: source reacquisition and five contextual Episode 11 clips are verified locally; no S3 key is needed for that source. Public CC-BY-4.0 Sortformer v2 ran on Intel N150 CPU: a 30-second clip took 18.89 seconds via CLI, and raw-probability C ABI export took 27.484 seconds. This is a new isolated diarization stage, not ASR CPU fallback or acoustic-quality PASS. Raw evidence is REVIEW_REQUIRED; default padded speaker segments are not proof of simultaneous speech. Native exporter hardening is still in progress. Real FFmpeg tests exposed a separate 21-millisecond mux timing bug; guessed AAC subtitle compensation was removed and mux timestamp shifting explicitly disabled. Latest combined local suite: 668 passed, 5 skipped in 73.15 seconds, before final native-exporter hardening. See WORKFLOW_REPAIR_PROGRESS.md and EP11_SOL_PILOT_EXECUTION.md for hashes, caveats and current prerequisites. No new Pod start or final subtitle delivery occurred. All older authority/budget/readiness paragraphs below are historical where they conflict with this record and the active task described next.

LATEST ACTIVE TASK: user explicitly requested persistent repair of the eleven-step approach, Sol Episode 11 pilot, and only after successful pilot review a fresh Episode 12 run with per-stage email updates. The prior unconditional Episode 12 prohibition below is superseded by this conditional authority; retained artifacts must still be preserved. See [repair progress and remaining work](WORKFLOW_REPAIR_PROGRESS.md). Local repair group 1 passed a combined working-tree suite of 604 tests, 32 skipped in 54.38 seconds; no paid run started. New cookies were header-validated without displaying values. Goal remains active and incomplete.

LATEST REVIEW: [eleven-stage workflow audit](ELEVEN_STAGE_REVIEW_2026-09-06.md) maps the actual eleven production stages and records remaining architectural/code weaknesses beyond alignment. No production code was changed in that review. Focused tests: 137 passed in 10.89 seconds; full suite: 1 failed, 586 passed, 32 skipped in 37.57 seconds. The failure is a nondeterministic timestamp mutation in the budget-tampering test; see F10, do not claim full-suite PASS. [Sol pilot execution](EP11_SOL_PILOT_EXECUTION.md) remains BLOCKED before paid startup because source/model/runtime and lease readiness are not established. Sol made no paid call. Episode 12 remains prohibited.

LATEST BUDGET OVERRIDE: preserve USD 1.00 from the existing RunPod balance; the former USD 0.50 cap below is superseded. Live balance at 10:59:24.8248931 SGT was USD 3.7543464701, giving at most USD 2.7543464701 before ongoing charges and shutdown reserves. Auto-pay was false; account spend USD 0.005/hour. See [current budget authority](EP11_BOUNDED_SAMPLE.md). No top-up, Episode 12 or GPT API authority; no paid operation started. Storage continues charging, so USD 1.00 cannot remain untouched forever while the retained volume exists.

Latest user direction: Episode 11 pilot FIRST, including cheaper-GPU feasibility; do not start Episode 12. A local cue-based draft prototype now exists; see [pilot contract and blockers](SUBTITLE_PILOT.md) and [hardware/whole-workflow review](HARDWARE_RUNPOD_REVIEW.md). Four real embedded WAV samples totaling 10,800 milliseconds were extracted and verified, but no new acoustic inference was run. The USD 0.50 combined allowance remains unspent. GPT transcription is out of scope. Sol is assigned bounded read-only pilot verification, not paid execution. Full production cue/translation migration is not complete.

Pilot implementation commit `979008f` was pushed to main. Full local suite: 587 passed, 32 skipped in 37.77 seconds. Sol completed independent checks; no paid runner was started. Pilot and hardware reports were submitted successfully through configured Gmail SMTP, message ID `<178866350961.8160.6195496528119390114@DESKTOP-3L7O2K0>`. SMTP acceptance is not inbox/read confirmation. The first local mail command failed PowerShell argument quoting before execution; retry via stdin succeeded, so only one message was submitted.

Research context: [workflow research](SUBTITLE_WORKFLOW_RESEARCH.md) documents the mismatch between the current all-word CTC contract and the user's usable-subtitle target. The proposal now has a separate pilot implementation, not a deployed production replacement. Local correction ZIPs contain embedded WAV clips despite no standalone media files; the final conflict neighborhood is not covered by the inspected clips. No new paid work occurred.

Latest authority update: the user authorized a combined maximum USD 0.50 for a small Episode 11 sample only, subsequently excluding GPT transcription. See [bounded sample preflight](EP11_BOUNDED_SAMPLE.md) as historical context, with the newer pilot document taking precedence on local WAV availability. Ready GPU image/models, source-audio access and a dedicated cost-bounded pilot lease remain unverified; no paid operation was initiated. Full-episode compute, Drive writes, migration and volume deletion are not authorized by this limited permission.

Read [the current review](ASTRA_REVIEW_2026-09-06.md), [all 63 controller logs and stage durations](EP11_CONTROLLER_LOG_INVENTORY.md), and [GPT transcription evaluation](GPT_TRANSCRIPTION_EVALUATION.md) first. They supersede historical status and continuation prompts below. Episode 11 is incomplete. Production has no speaker-evidence acquisition path; historical audio-review PASS does not bind the new provisional return. Local fixes passed 563 tests with 32 skipped in 37.62 seconds. No GPU or delivery PASS follows from those tests. Paid compute and Drive mutation remain prohibited without renewed explicit permission. Do not execute historical resume/invalidation commands.

## Repository snapshot

- Repository: `ismailkorkmaz1905/ma-sub`
- Production branch: `main`
- Consolidation: [PR #1](https://github.com/ismailkorkmaz1905/ma-sub/pull/1) is merged, and the obsolete `engineering/ep12-reliability` branch was deleted locally and remotely.
- Independent CTC probe baseline commit: `f581fcb59091d026e4f53a825916ba9e7abcf479`. Resolve the current review candidate with `git rev-parse HEAD`; the manual-review/controller implementation is newer than the probe baseline.
- Hash-bound manual-review UI and verified RunPod override-transfer implementation: `1465c278486ad0f4ec70d85e84c287efdd30abfc`.
- Recorded implementation worktree: clean and synchronized with `origin/main`
- Main CI at `c2adfcd6a23512219b0fb41ca176336c53f9bfd6`: test and Dockerfile jobs passed in workflow run `33951396788` on 2026-09-05.

Always re-run `git status --short --branch`, `git rev-parse HEAD`, and `gh pr view 1` because this snapshot can become stale.

## Continuous execution rule

Answering a status request or operator question does not end the active production task. Continue until strict delivery completes or a genuine external blocker prevents further safe work. When blocked, record the exact blocker, retained evidence, and the exact resume command in the Astra handoff before stopping.

## Completed work

- Removed the broken Base64 bootstrap workflow and payload.
- Preserved the complete imported source and notebooks under `legacy/SYSTEM_V2_BETA/` as reference-only material.
- Ported the maintained engine into `src/mas/`, configuration into `config/`, and historical executable tests into `tests/`.
- Added a neutral production API in `src/mas/engine/api.py`; legacy `v2` names remain only where required by stored schema and artifact compatibility.
- Wired the `./mas run EPISODE [--source-url URL]` state machine through source acquisition, audio extraction, GPU-only ASR, Turkish correction handoff, acoustic review, forced alignment, strict schema generation, Indonesian translation handoff, strict finalization, Drive publication, and RunPod stop request.
- Added hash-bound atomic resume state, source immutability checks, finite network retries and timeouts, and no-progress watchdogs.
- Added speaker-aware overlap handling. Different known speakers remain separate; same-speaker overlap fails; unknown-speaker overlap remains unresolved evidence.
- Enforced timing-only overrides and strict separation of emergency and strict outputs.
- Added Google Drive upload verification using remote byte-count and SHA-256 readback. Publication is restricted to the final MKV, Turkish SRT, and Indonesian SRT files under the Drive `EPISODES` tree; the repository, source media, logs, reports, and intermediate artifacts are not synced.
- Added Gmail notifications for run start, every stage start/completion/failure, translation waits, and final readiness. `./mas notify-test` verifies configured credentials.
- Added per-invocation local run logs with UTC metadata, Git SHA, stdout/stderr, exit code, elapsed time, and full tracebacks. Command source URLs and secret values are not logged; `logs/LATEST` points to the newest session.
- Added optional YouTube Netscape cookie input through `MAS_YTDLP_COOKIES`. Cookie contents are validated without being logged.
- Added Docker and RunPod bootstrap, watchdog, timeout, stop, and external-verification guidance.
- Preserved the 15 Episode 12 incident intervals in `tests/fixtures/ep12_incident_intervals.json`.
- Added operator, architecture, migration, incident acceptance, and Astra handoff documentation.

## Validation recorded

- Full local suite for `1465c278486ad0f4ec70d85e84c287efdd30abfc`: `504 passed, 32 skipped` in `44.10 seconds`.
- Manual-review UI plus RunPod override-transfer focused suite: `33 passed` in `1.72 seconds`.
- Full local suite at `83f0c9d`: `477 passed, 32 skipped` in `39.09 seconds`.
- Ported engine suite: `342 passed, 31 skipped`.
- Cross-speaker focused suite: `82 passed`.
- Local Python compilation and `git diff --check`: passed.
- Exact `main` CI at `c2adfcd6a23512219b0fb41ca176336c53f9bfd6`: test and Dockerfile jobs passed. PR #1 checks are historical because the PR is merged.
- Exact `main` CI at `1465c278486ad0f4ec70d85e84c287efdd30abfc`: test and Dockerfile jobs passed in workflow run `33964604668`.
- Docker was not available in the local Windows environment, so a local image build was not performed.
- Gmail SMTP notification was verified by a real test email to the configured recipient. The app password remains outside Git in the Windows user environment.
- A user-exported Netscape cookie file is stored outside Git at `C:\Users\Ismail\.config\ma-sub\youtube-cookies.txt`, restricted to the current Windows user, and bound through the persistent user-level `MAS_YTDLP_COOKIES` variable. Authenticated source acquisition and the equivalent short-lived RunPod secret-file transfer passed during the Episode 11 run. Cookie values are not logged.
- Official-channel discovery was exercised live with the cookie file and resolved Episode 12 to `https://www.youtube.com/watch?v=yVCzw_dFZA8`. Media download itself remains unverified in the production pipeline.
- Rclone `v1.75.0` is installed under the current Windows user. The `gdrive` OAuth config is stored outside Git with a restricted ACL and read access to `gdrive:Muhtemel_Ask_Subtitles/EPISODES` was verified. Rclone warned that its shared Google client ID is scheduled for retirement during 2026; a private Google OAuth client is a future continuity risk, not a blocker for the verified current connection.
- A dedicated ED25519 keypair exists outside Git at `C:\Users\Ismail\.ssh\ma-sub-runpod-ed25519`; the private-key path is persisted in `MAS_RUNPOD_SSH_KEY`. The public key is registered in RunPod as `ma-sub-controller` with fingerprint `SHA256:D95dKFhj4jlTCgNNbBy7rXL2FDoU5ZIZSwH55DAUI1Y`.
- The latest independently observed probe Pod is `tccsb8991x84ua`; it was externally verified `EXITED`. A dedicated `ma-sub-control` API key is stored outside Git in the Windows user environment. Episode 11 verified bounded startup, SSH, dependency bootstrap, strict cookie/CUDA/Drive preflight, source acquisition, audio extraction, real GPU raw ASR, Turkish handoff download, secret cleanup, controller stop, and external `EXITED` polling.
- The retained infrastructure record has a 30 GB temporary container disk and a 50 GB network volume `xgogcmey5o` in `EU-RO-1`. The stopped container disk was reported as not billed. The last recorded network-volume rate was `$0.07/GB/month`, deriving to `$3.50/month` for 50 GB, but its exact observation time was not retained. Re-query current pricing before cost decisions. Deleting files does not reduce this fixed allocation, and the recorded provider behavior did not allow shrinking it. Do not delete the volume before deciding whether its retained `/workspace` state is needed for the real run.
- Episode 11 raw ASR completed on a real RTX 4090 in `986.5 seconds`. The controller reached the Turkish correction handoff after `1,168.031 seconds`. The Turkish correction return later passed. Bounded audio review covered 245 records, resolved 107, and left 138 pending under the policy used by that run.
- After commit `f581fcb59091d026e4f53a825916ba9e7abcf479`, an independent CUDA CTC probe processed all 138 pending records with `samil24/wav2vec-xlsr-53-turkish-v4` pinned to revision `07d79597b78c56758045e3a2cd1c44bc1a19b1e8`. It produced 0 normalized exact reference matches and closed 0 strict decisions. Report self SHA-256: `6081ddb6670ca0893bd94a704dd7eb94a6865785c0a9dea58698239bd48e65a6`. Report file SHA-256: `8312910ddd11065a321362cab253c1033833829bcba2f73b3f04ef3174c996d2`. This is not an acoustic PASS.
- Live Google Drive upload/readback was not performed. Old system snapshots and Episode 12 intermediate/source folders were moved to Drive Trash during cleanup. Episode 12's existing MKV, root Indonesian SRT, and original `final/subtitles` tree were restored and verified present. They must be retained; no Turkish SRT was present in the inspected Drive folder. Exact remote byte counts, SHA-256 values, and a preservation receipt were not recorded, so current presence and collision safety must be re-verified before publishing canonical Episode 12 names.
- Controller-driven RunPod stop and external provider-state verification passed after the successful Turkish handoff. The independent CTC probe Pod `tccsb8991x84ua` was also externally verified `EXITED`. Final pipeline shutdown and console billing verification remain incomplete.
- The 23 logged Episode 11 controller attempts, their exact durations, observed failures, migrations, and corresponding actions are recorded in `docs/EP11_FIRST_PRODUCTION_RUN.md`. The first 16 through the successful handoff totalled `3,481.407 seconds`; attempts 17 through 23 added `2,767.171 seconds`, for `6,248.578 seconds` across all 23 controller invocations. Attempt 23 used checkpoint-backed raw ASR in `32.7 seconds`, then bounded audio review processed 245/245 in `158.2 seconds` and stopped fail-closed with 107 resolved and 138 pending.

Skipped local tests include Windows cases that require symlink privilege. Treat skipped tests as reported evidence, not passes.

## Open release work

1. Preserve the accepted Turkish correction return and the CTC report with both recorded hashes. Implement and validate the contextual machine-review policy that re-evaluates the pending set with target audio and adjacent context. The user will not manually listen to 138 clips. This plan is not verified until its code, focused/full tests, and a real resumed RunPod execution produce retained evidence; the CTC probe's 0 strict closures are not decisions.
2. After the contextual policy passes local validation, resume with `.\mas.ps1 run 11`. Record the resulting bounded-review counts, production forced alignment, and Indonesian handoff evidence. If the run still stops, preserve the exact report and checkpoint, state the exact external blocker, and retain `.\mas.ps1 run 11` as the resume command. The controller starts paid compute only after local preflight passes.
3. Complete the Indonesian translation return without changing immutable IDs, order, timing, or Turkish text, then resume the same command.
4. Inspect source playback against representative final cue boundaries and resolve every review interval with acoustic evidence.
5. Confirm strict Drive delivery with exact remote byte-count and SHA-256 readback receipts.
6. Exercise failure, maximum-runtime, and idle-watchdog shutdown paths separately; do not spend GPU time solely for synthetic tests when a bounded fixture can prove the path.
7. Replace rclone's retiring shared Google client ID with a private OAuth client before the provider disables it.
8. Re-query current storage pricing, then decide whether to delete the 50 GB network volume after durable Drive delivery. Deletion is permanent; ordinary file cleanup does not reduce its allocated size.
9. Run the Astra prompt in `docs/ASTRA_REVIEW_PROMPT.md` against the final commit and retained runtime evidence.
10. Do not create a stable tag until all real-environment gates pass and the user decides.

## Secret handling

Expected environment names:

```bash
MAS_GMAIL_ADDRESS
MAS_GMAIL_APP_PASSWORD
MAS_NOTIFY_TO
MAS_YTDLP_COOKIES
MAS_DRIVE_STRICT_REMOTE
RUNPOD_POD_ID
RUNPOD_API_KEY
```

## Emergency continuation boundary - 2026-09-06

This section supersedes the older Episode 11 status and continuation prompt above.

- Candidate commit before this documentation update: `33637d3a31ee49c82813ca947ffd3b56174f9571` on `main`, synchronized with `origin/main` and clean when checked.
- Local verification at that candidate: `530 passed, 32 skipped in 39.74 s`; focused forced-alignment tests: `35 passed in 0.33-0.40 s`. These are local tests, not real GPU or delivery evidence.
- Real Episode 11 audio review reached `998/998` PASS. Three acoustically duplicated fragments were resolved with retained provenance.
- The last exact GPU run still failed closed during forced alignment after `755.8 s` on one overlap: `MA11-TR-a09b20542760d351` and `MA11-TR-1144d768dc92c4a9`. No Turkish SRT exists yet.
- A new local Turkish-return ZIP was generated with those two UIDs pending exact audio review. Manifest SHA-256: `195f7d1509cb3592fb6ed207d1c6b088930a02006d4cfc338a3ec3f92cc6c9bf`. ZIP SHA-256: `DA278A26B22BC507D40577E2658F7E9A48374154BD978B940125AD1CB1C7BC2A`. It was not proven uploaded or consumed remotely before shutdown.
- Indonesian handoff/return/SRT, strict mux, subtitle QA, Drive upload and byte/SHA-256 readback, and final delivery remain incomplete. No stable tag is allowed.
- The user reported nearly `21 hours` elapsed and approximately `$20` of RunPod spend. These values are user reports, not provider billing measurements.
- External RunPod API observation at `2026-09-06T01:35:52.8724964Z`: Pod `781ct55zv4gkle` had desired state `EXITED`; its recorded rate field was `$0.74/hour`; retained network volume `xgogcmey5o` still existed. This proves stopped compute state only, not billing closure. Do not delete the volume.
- Do not start, restart, migrate, or otherwise incur paid RunPod compute until the user explicitly reauthorizes spend. The eventual resume command is `.\mas.ps1 run 11`, but it is currently prohibited.
- The current YouTube cookie file is stale/invalid according to real RunPod yt-dlp warnings. Episode 11 passed download only because immutable source media already existed. Refresh authentication before a new episode.
- `.\mas.ps1 notify-test` successfully submitted a real Gmail message to the configured recipient. The user's report of missing long-term status mail remains unresolved; audit pipeline notification events, delivery/log evidence, and recipient-side filtering without exposing credentials.

Observed causes of delay and cost:

1. Correction: the `104.0-175.2 s` raw-stage durations do not prove repeated transcription or code-driven invalidation. The raw input digest does not contain code/Git identity. See the current review for the evidence and missing legacy checkpoint provenance.
2. The first overlap candidate implementation performed unbounded external-word combination searches. One run was manually stopped after about `781 s`; commits `9e4ae5b` and `33637d3` bounded the search and avoided redundant candidates, but the exact candidate still spent `755.8 s` before the final fail-closed overlap.
3. Provider capacity and SSH readiness failures caused repeated start/stop cycles and migrations. Fresh Pods repeated bootstrap and ffmpeg installation.
4. The pipeline correctly failed closed on unresolved overlaps, but the review artifact did not initially include the final two exact conflict UIDs, causing another correction-return cycle.
5. The last resume attempt encountered transfer/network retries and was stopped before proving that the regenerated return reached the remote checkpoint.

Astra must review the complete codebase, controller logs, checkpoint binding granularity, overlap search complexity, bootstrap/image reuse, notification behavior, and exact current local/remote artifact boundary. It must fix and locally test defects without paid compute. It must also evaluate, using current official OpenAI documentation, whether GPT transcription APIs can replace or complement Whisper/CTC at the required Turkish accuracy, word-timestamp, diarization/overlap, privacy, cost, latency, retry, deterministic-resume, and hash-evidence boundaries. Do not implement an API substitution unless strict timing and evidence contracts can still be met.

Do not place secret values in this file, shell history, issue comments, test output, commits, or chat handoffs. `MAS_YTDLP_COOKIES` must point to a local Netscape-format cookies file outside the repository or in an ignored path.

## Fresh Codex CLI continuation prompt

```text
C:\Users\Ismail\CodeBase\ma-sub reposundaki calismayi devral. Once repo kokundeki AGENTS.md dosyasini, sonra README.md, docs/CODEX_HANDOFF.md ve docs/EP11_FIRST_PRODUCTION_RUN.md dosyalarini tamamen oku. Tek production dali main; once git status ve HEAD kontrol et, kirli agacta kullanici degisikliklerini koru. Guncel runtime candidate'i `git rev-parse HEAD` ile belirle. Episode 11 gercek RTX 4090 raw ASR asamasini ve Turkce correction return gate'ini tamamladi. Bounded audio review 245 kaydin 107'sini resolve etti, 138'ini o kosudaki politika altinda pending birakti. Kullanici 138 klibi elle dinlemeyecek. Pending seti hedef ses ve komsu baglamla yeniden degerlendiren contextual machine-review politikasini uygula ve dogrula; kod, focused/full testler ve gercek resumed RunPod kaniti olmadan bu plani basarili sayma. Bagimsiz pinned-model CUDA CTC probe 138/138 calisti fakat normalized exact match 0 ve strict closure 0; bu acoustic PASS degildir. Status veya soru yanitlamak aktif production gorevini bitirmez; strict delivery veya gercek external blocker'a kadar devam et. Blocker olursa tam nedeni, korunan kaniti ve exact resume komutu `.\mas.ps1 run 11` olarak Astra handoff'a yaz. Production forced alignment, ID handoff, strict mux ve Drive readback/delivery tamamlanmadi. Latest probe Pod tccsb8991x84ua disaridan EXITED dogrulandi; yeniden sorgulamadan guncel kabul etme. RunPod SSH key ve gdrive OAuth hazir; secret degerlerini asla loglama. Episode 12'ye dokunma. Gercek Drive readback, final pipeline kapanisi veya tamamlanmamis sonraki GPU asamalari yapilmissa yapilmis gibi raporlama. Schema, hash, source immutability, timing-only override, speaker overlap, strict/emergency ayrimi ve subtitle kalite kurallarini test gecsin diye gevsetme. Anlamli degisiklikleri test et, commit et ve main dalina push et. Astra incelemesi ve gercek tam episode kaniti tamamlanmadan stable tag basma.
```


### Episode 12 delivery progress, 2026-09-06

The user explicitly started Episode 12 with a four-hour deadline, overriding the earlier Episode 11 pilot gate. First complete TR/ID SRT pair: `EPISODES/Muhtemel Ask 12.Bolum/work/review-delivery-20260906-v1/` (2,992 display entries), with `delivery-review.json`, source-audio `review.html` and acoustic diagnostics. This is REVIEW_REQUIRED, not strict/listening PASS. Detailed current evidence is in `docs/EP12_RUN_2026-09-06.md`.

The root fixed new neighbor overlaps introduced by independent CTC windows, preserving raw ASR and falling back on conflicts. Full local suite: 724 passed, 5 skipped in 177.69 seconds. All temporary GPU Pods from both targeted repairs were deleted with external absence proof. Native CPU evidence continues as one resumed chain; 17 completed 119-second chunks were preserved and remaining audio uses 30-second chunks, absolute deadline 11:25 UTC. Sol is reviewing 619 fast Indonesian cues for concise wording and semantic errors. Gmail remains blocked by daily SMTP quota; no notification-delivery PASS. Current HEAD and pre-existing dirty/ignored files remain preserved.
