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

- Persist primary ASR before rescue/enrichment, with actual-model and producer binding.
- Local Indonesian-return structural, readability and semantic preflight before compute.
- Effective shared runtime/readability configuration and phase-specific resume.
- Cue correction/translation integration after measured pilot acceptance.
- Collision-preserving transactional Drive delivery outside GPU compute.
- Ready source/model/image and independently enforced pilot lease/shutdown, then real contextual Episode 11 evaluation.

Sol owns isolated pilot-runner planning/local-process enforcement code and tests. It is explicitly not approved for paid execution yet. Root found and rejected its initial cooperative-only timeout and excluded mail/retrieval budget assumptions. Local process/mail/HTTP bounds were added, but an independent external lease watchdog and runtime/model readiness remain unverified. No Pod was started.

A non-blocking question asks the user for permission to create a RunPod S3 key for non-compute source retrieval; no key has been created. Local code repairs continue while access is unresolved.
