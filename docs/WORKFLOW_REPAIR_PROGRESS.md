# Workflow repair progress

## Active goal and authority

Repair the eleven-step production approach, then execute an Episode 11 pilot with Sol. Only after the pilot is reviewed as acceptable may Sol start Episode 12 from scratch, preserving existing Episode 12 artifacts. Notify the configured recipient at every run-stage start, completion and failure. Continue repairing recoverable errors instead of treating the first failure as completion.

RunPod spending remains bounded by existing balance minus USD 1.00 plus operational shutdown/billing margins. No top-ups, GPT transcription, volume deletion or automatic acceptance of gated-model terms. Increasing the budget does not prove runtime readiness.

## Completed local repair group 1

- F6: production passes the recorded source SHA into the downloader. A locked source with bad/missing checkpoint or failed EOF verification is preserved and fails before reacquisition. Force replacement of a locked source is prohibited.
- F3: once raw ASR or its recovery checkpoint exists, source resume no longer opportunistically fetches new captions. Existing evidence remains hash-bound; malformed/stale artifacts are not silently rebound.
- F1, partial: the Pod shell watches `useful-progress.json`, not log mtime. Stage transitions and actual ASR/review/alignment units update it; the 30-second heartbeat does not. Media extraction/read-through verification now has a 1,800-second subprocess ceiling. Rclone's own statistics-versus-progress classification and all phase-specific deadlines still need work.
- F10: checkpoint-tampering test now changes its timestamp by exactly one second, rather than relying on two wall-clock reads differing.
- A subsequent full test exposed Windows sharing violations while reading a concurrently atomically replaced progress file. A bounded JSON reader retries PermissionError only; malformed JSON remains fatal. State loading and the concurrent progress test use it.

Validation: source/controller/runtime group 70 passed in 1.99 seconds; expanded source/controller/runtime/alignment group 111 passed in 2.77 seconds; progress/reliability group 87 passed, 1 skipped in 3.90 seconds. Final combined working-tree suite, including Sol's uncommitted planning scaffold: 604 passed, 32 skipped in 54.38 seconds. The earlier transient run was 1 failed, 602 passed, 32 skipped in 77.87 seconds, and was repaired rather than hidden. No real GPU/Drive result follows from these tests.

## Cookies

User supplied `C:/Users/Ismail/Downloads/11 cookie.txt` and `12 cookie.txt`. Both passed the existing Netscape-header validator without exposing cookie values. Actual authenticated YouTube access remains untested. User-scope `MAS_YTDLP_COOKIES` now points to the Episode 11 file. Existing long-lived processes may still hold stale environment values; set the correct path explicitly for execution. Episode 12 file is retained for later conditional use, not consumed by an Episode 12 run.

## Remaining work

- Measure primary-checkpoint runtime on the real GPU; the local implementation and regression tests are complete.
- Exercise local Indonesian-return preflight with a real new return; structural/readability/semantic code and local tests are complete.
- Effective shared runtime/readability configuration and phase-specific resume.
- Cue correction/translation integration after measured pilot acceptance.
- Collision-preserving transactional Drive delivery outside GPU compute.
- Ready source/model/image and independently enforced pilot lease/shutdown, then real contextual Episode 11 evaluation.

Sol owns isolated pilot-runner planning/local-process enforcement code and tests. It is explicitly not approved for paid execution yet. Root found and rejected its initial cooperative-only timeout and excluded mail/retrieval budget assumptions. Local process/mail/HTTP bounds were added, but an independent external lease watchdog and runtime/model readiness remain unverified. No Pod was started.

The user authorized creating necessary access keys. The live RunPod S3 creation form exposes read/write access to all S3-compatible network volumes, with no read-only scope selector. At that final-action boundary an asynchronous confirmation describes the actual scope; creation remains pending. The logged-in browser is usable. No secret has been created or printed and no Pod has been started.

## Completed local repair group 2

- F2: primary inference is now checkpointed immediately, before independent VAD, rescue-budget validation and enrichment. Each content-addressed checkpoint binds actual model/config/tokenizer file hashes, installed runtime package file hashes, relevant producer function sources, exact transcription options, audio SHA, episode and runtime device/compute identity. Changed model/prompt causes a distinct checkpoint; rescue-policy changes do not invalidate unchanged primary evidence. Existing final/recovery validation contracts remain unchanged. Loading model weights and re-hashing dependencies still occur on primary-only resume; this is not a zero-startup-cost claim.
- Primary checkpoint regression interrupts VAD after inference, changes the rescue policy and resumes with exactly one synthetic main-model call across the two attempts. Separate cases reject tampering and isolate changed weights/prompt. Focused primary/raw/runtime suite: 70 passed in 6.47 seconds.
- F4: local Indonesian pack/return preflight now runs before provider start. It checks exact pack/schema/glossary/return identities, episode, pending reviews, semantic/layout QA and the final timing policy's hard Indonesian CPS limit. Complete per-UID issues and artifact hashes are persisted locally; changing either ZIP during validation fails. This is local preflight, not final acoustic acceptance or delivery evidence.
- Sol's isolated runner now bounds process trees and artifact-hash chunks, propagates shutdown failures and redacts provider errors/receipts. It remains an unapproved paid-execution scaffold until independent lease enforcement and real runtime/model readiness are established.

Combined full working-tree suite before the additional explicit semantic-review-count guard: 622 passed, 32 skipped in 45.66 seconds. Local doctor: Python/git/rclone OK; ffmpeg/ffprobe/torch missing at that observation. Docker definition inspected, but Docker is unavailable locally so no image build PASS is claimed. FFmpeg installation from the Windows build provider linked by ffmpeg.org is in progress, with vendor SHA-256 verification required before execution.

Previous group-1 status email was accepted by configured SMTP: `<178866529351.15200.2443881922230205678@DESKTOP-3L7O2K0>`. This is service submission evidence, not inbox/read confirmation.

## Source readiness and repair group 3

Group 2 commits `5453960`, `f161250` and `63f7510` were pushed to main. Final full suite after the semantic-review guard: 622 passed, 32 skipped in 56.42 seconds. Group-2 status email accepted by SMTP: `<178866628142.7688.696258260362935316@DESKTOP-3L7O2K0>`.

FFmpeg and ffprobe are now installed at `C:/Users/Ismail/AppData/Local/ma-sub-tools/ffmpeg-release-20260906/extracted/ffmpeg-n9.0-latest-win64-gpl-9.0/bin`, added to User PATH. Verified version: `n9.0.1-26-g5c8e7e2433-20260905`. BtbN archive SHA-256 matches its published checksum: `87C4729F3193F3BA562BADA0330A98338C9A858D25B15FC98338A1364667348B`. The first Gyan download was explicitly stopped after throughput stayed around 54 KB/s; its incomplete ZIP remains outside the repository and was never extracted/executed. The BtbN provider is linked by https://ffmpeg.org/download.html . Local doctor with refreshed PATH reports Python/ffmpeg/ffprobe/git/rclone OK, torch missing. No local NVIDIA GPU exists, so no CPU substitution for ASR is intended.

Free source reacquisition succeeded with a private working copy of the user's Episode 11 cookies. Official source: `https://www.youtube.com/watch?v=3CeNouZts7o`, selected format `251`, reported duration 8278 seconds. Audio download log reports 10 seconds for transfer; total extraction/handshake wall time was not instrumented. No RunPod was started. Files are isolated under `EPISODES/Muhtemel Ask 11.Bolum/work/subtitle-pilot/source-reacquisition-20260906/`:

- `source.webm`: 106448305 bytes, SHA-256 `ea3ee9b3a5f986f9f7dbc634abbbd8af4d3b1cd80e69a3d22d83c87f95c9e2d4`.
- `source.wav`: 264910244 bytes, SHA-256 `f4858dfcc4909d9f7331a20f233e205def9b5953bd4d51723982bde9105bdd64`. Immutable input recheck and conversion provenance persisted. Exact conversion elapsed seconds: UNKNOWN, not recorded by the conversion helper.
- All four historical PCM samples differ from the new decoded WAV by at most 1 signed 16-bit sample unit at zero lag; measured correlations are above 0.99999998. This is sampled diagnostic evidence, not whole-source byte equivalence. Old hashes/production sources were not replaced or rebound.
- `contextual-samples-20260906/samples.json` binds five newly extracted clips to the new source by exact PCM offset equality: 60000-90000, 350000-380000, 470000-505000, 600000-630000 and 4280000-4310000 milliseconds. Total 155000 milliseconds, preparation elapsed 2.25 seconds. Final conflict context included; acoustic acceptance remains NOT_EVALUATED.

Source access no longer depends on S3. The S3 key creation form was prepared but not submitted; access would cover all compatible volumes read/write. A separate question requests explicit approval for a noncommercial-only NVIDIA Sortformer pilot (CC-BY-NC-4.0); no model was downloaded or license accepted. Community-1 access is also unresolved. Model/runtime readiness remains a real pilot prerequisite, but a bounded GPU bootstrap/validation lease can be considered within the existing spending allowance; do not impose the circular requirement of proving GPU compatibility without ever using a bounded paid GPU check.

Uncommitted group 3: rclone progress now requires increasing transferred bytes/completed files rather than repeated statistics, binary readback counts stdout bytes only, and upload/hash/readbacks share one deadline. Targeted FFmpeg clip extraction has a 120-second subprocess ceiling. Focused transfer/controller/discovery tests: 88 passed in 3.01 seconds. Combined clip/transfer/guardian/pilot tests: 85 passed in 7.37 seconds. A sandbox run had 49 fixture permission errors and 39 passes; approved escalation reran the relevant tests successfully, without changing test gates.

Guardian remains under repair, not paid-ready: root replaced unsafe Windows `os.kill(pid, 0)` liveness probing before execution. Sol added ready receipts, sanitized errors and bounded shutdown on monitoring failure. Root then found a ready/start race where the parent could start after the guardian exited; Sol is repairing single-owner START sequencing and startup-budget accounting. Do not start a Pod from the draft runner until this regression is resolved and reviewed.
