# Codex handoff

## Repository snapshot

- Repository: `ismailkorkmaz1905/ma-sub`
- Production branch: `main`
- Consolidation: [PR #1](https://github.com/ismailkorkmaz1905/ma-sub/pull/1) is merged, and the obsolete `engineering/ep12-reliability` branch was deleted locally and remotely.
- Controller, source reliability, and preflight implementation commit: `09320beb0a51e02b44612ae745d07f7ceaf72b5a`. Resolve the review candidate with `git rev-parse HEAD`; documentation-only commits may be newer.
- Recorded implementation worktree: clean and synchronized with `origin/main`
- Main CI at `c2adfcd6a23512219b0fb41ca176336c53f9bfd6`: test and Dockerfile jobs passed in workflow run `33951396788` on 2026-09-05.

Always re-run `git status --short --branch`, `git rev-parse HEAD`, and `gh pr view 1` because this snapshot can become stale.

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

- Full local suite at `09320be`: `470 passed, 32 skipped` in `34.25 seconds`.
- Ported engine suite: `342 passed, 31 skipped`.
- Cross-speaker focused suite: `82 passed`.
- Local Python compilation and `git diff --check`: passed.
- Exact `main` CI at `c2adfcd6a23512219b0fb41ca176336c53f9bfd6`: test and Dockerfile jobs passed. PR #1 checks are historical because the PR is merged.
- Docker was not available in the local Windows environment, so a local image build was not performed.
- Gmail SMTP notification was verified by a real test email to the configured recipient. The app password remains outside Git in the Windows user environment.
- A user-exported 3,225-byte Netscape cookie file was copied byte-for-byte to `C:\Users\Ismail\.config\ma-sub\youtube-cookies.txt`, restricted to the current Windows user, and bound through the persistent user-level `MAS_YTDLP_COOKIES` variable. Its 23 non-comment records and YouTube domain were validated without logging cookie values. Authenticated source acquisition and the equivalent RunPod secret-file mount remain unverified.
- Official-channel discovery was exercised live with the cookie file and resolved Episode 12 to `https://www.youtube.com/watch?v=yVCzw_dFZA8`. Media download itself remains unverified in the production pipeline.
- Rclone `v1.75.0` is installed under the current Windows user. The `gdrive` OAuth config is stored outside Git with a restricted ACL and read access to `gdrive:Muhtemel_Ask_Subtitles/EPISODES` was verified. Rclone warned that its shared Google client ID is scheduled for retirement during 2026; a private Google OAuth client is a future continuity risk, not a blocker for the verified current connection.
- A dedicated ED25519 keypair exists outside Git at `C:\Users\Ismail\.ssh\ma-sub-runpod-ed25519`; the private-key path is persisted in `MAS_RUNPOD_SSH_KEY`. The public key is registered in RunPod as `ma-sub-controller` with fingerprint `SHA256:D95dKFhj4jlTCgNNbBy7rXL2FDoU5ZIZSwH55DAUI1Y`.
- RunPod Pod `p54vvbyu76eztn` (`muhtemel-ask-ep12`) was identified with an RTX 4090 configuration and `/workspace` network-volume mount. A dedicated `ma-sub-control` API key is stored outside Git in the Windows user environment. The first Episode 11 attempt on 2026-09-05 started the Pod but never reached the pipeline because the existing Pod had no SSH public key. The controller timed out, issued stop, and externally verified `desiredStatus=EXITED`; the recorded run elapsed time was `347.187 seconds`. The Pod-specific `PUBLIC_KEY` and `SSH_PUBLIC_KEY` values were then populated through the REST API and read back as exact matches while the Pod remained `EXITED`. SSH and GPU execution still require a paid retry. The controller now prints startup and SSH wait progress and fails immediately on SSH authentication rejection.
- The Pod has a 30 GB temporary container disk and a 50 GB network volume `xgogcmey5o` in `EU-RO-1`. The stopped container disk was reported as not billed. The last recorded network-volume rate was `$0.07/GB/month`, deriving to `$3.50/month` for 50 GB, but its exact observation time was not retained. Re-query current pricing before cost decisions. Deleting files does not reduce this fixed allocation, and the recorded provider behavior did not allow shrinking it. Do not delete the volume before deciding whether its retained `/workspace` state is needed for the real run.
- Real full-episode GPU inference was not performed.
- Live Google Drive upload/readback was not performed. Old system snapshots and Episode 12 intermediate/source folders were moved to Drive Trash during cleanup. Episode 12's existing MKV, root Indonesian SRT, and original `final/subtitles` tree were restored and verified present. They must be retained; no Turkish SRT was present in the inspected Drive folder. Exact remote byte counts, SHA-256 values, and a preservation receipt were not recorded, so current presence and collision safety must be re-verified before publishing canonical Episode 12 names.
- Controller-driven RunPod stop and external provider-state verification were observed on the failed SSH attempt. The final REST state was `EXITED`. Pipeline-driven shutdown, successful SSH, GPU execution, and console billing verification were not performed.

Skipped local tests include Windows cases that require symlink privilege. Treat skipped tests as reported evidence, not passes.

## Open release work

1. Run `.\mas.ps1 run 11`. This retries the first non-EP12 candidate with the corrected Pod SSH configuration and starts paid compute only after local preflight passes.
2. Record successful SSH, strict GPU preflight, Docker/runtime dependency, authenticated source acquisition, and external EXITED evidence.
3. Complete both ChatGPT handoffs using the locally downloaded input ZIPs and exact local `translation_output` return paths.
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
C:\Users\Ismail\CodeBase\ma-sub reposundaki calismayi devral. Once repo kokundeki AGENTS.md dosyasini, sonra README.md ve docs/CODEX_HANDOFF.md dosyasini tamamen oku. Tek production dali main; once git status ve HEAD kontrol et, kirli agacta kullanici degisikliklerini koru. Controller `.\mas.ps1 run 11` ile Pod'u yalniz gercek run basladiginda baslatacak, loglari aktaracak, handoff ZIP'lerini indirip geri yukleyecek ve Pod'u disaridan EXITED durumuna kadar izleyecek sekilde kodlandi. RunPod SSH key ve gdrive OAuth hazir; secret degerlerini asla loglama. Pod p54vvbyu76eztn icin durumu yeniden sorgulamadan guncel kabul etme. Episode 12'ye dokunma; ilk test adayi Episode 11. Gercek GPU, Drive readback veya pipeline RunPod kapanisi yapilmadiysa yapilmis gibi raporlama. Schema, hash, source immutability, timing-only override, speaker overlap, strict/emergency ayrimi ve subtitle kalite kurallarini test gecsin diye gevsetme. Anlamli degisiklikleri test et, commit et ve main dalina push et. Astra incelemesi ve gercek GPU episode kaniti tamamlanmadan stable tag basma.
```
