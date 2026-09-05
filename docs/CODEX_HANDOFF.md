# Codex handoff

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

Do not place secret values in this file, shell history, issue comments, test output, commits, or chat handoffs. `MAS_YTDLP_COOKIES` must point to a local Netscape-format cookies file outside the repository or in an ignored path.

## Fresh Codex CLI continuation prompt

```text
C:\Users\Ismail\CodeBase\ma-sub reposundaki calismayi devral. Once repo kokundeki AGENTS.md dosyasini, sonra README.md, docs/CODEX_HANDOFF.md ve docs/EP11_FIRST_PRODUCTION_RUN.md dosyalarini tamamen oku. Tek production dali main; once git status ve HEAD kontrol et, kirli agacta kullanici degisikliklerini koru. Guncel runtime candidate'i `git rev-parse HEAD` ile belirle. Episode 11 gercek RTX 4090 raw ASR asamasini ve Turkce correction return gate'ini tamamladi. Bounded audio review 245 kaydin 107'sini resolve etti, 138'ini o kosudaki politika altinda pending birakti. Kullanici 138 klibi elle dinlemeyecek. Pending seti hedef ses ve komsu baglamla yeniden degerlendiren contextual machine-review politikasini uygula ve dogrula; kod, focused/full testler ve gercek resumed RunPod kaniti olmadan bu plani basarili sayma. Bagimsiz pinned-model CUDA CTC probe 138/138 calisti fakat normalized exact match 0 ve strict closure 0; bu acoustic PASS degildir. Status veya soru yanitlamak aktif production gorevini bitirmez; strict delivery veya gercek external blocker'a kadar devam et. Blocker olursa tam nedeni, korunan kaniti ve exact resume komutu `.\mas.ps1 run 11` olarak Astra handoff'a yaz. Production forced alignment, ID handoff, strict mux ve Drive readback/delivery tamamlanmadi. Latest probe Pod tccsb8991x84ua disaridan EXITED dogrulandi; yeniden sorgulamadan guncel kabul etme. RunPod SSH key ve gdrive OAuth hazir; secret degerlerini asla loglama. Episode 12'ye dokunma. Gercek Drive readback, final pipeline kapanisi veya tamamlanmamis sonraki GPU asamalari yapilmissa yapilmis gibi raporlama. Schema, hash, source immutability, timing-only override, speaker overlap, strict/emergency ayrimi ve subtitle kalite kurallarini test gecsin diye gevsetme. Anlamli degisiklikleri test et, commit et ve main dalina push et. Astra incelemesi ve gercek tam episode kaniti tamamlanmadan stable tag basma.
```
