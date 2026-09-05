# Codex handoff

## Repository snapshot

- Repository: `ismailkorkmaz1905/ma-sub`
- Production branch: `main`
- Consolidation: [PR #1](https://github.com/ismailkorkmaz1905/ma-sub/pull/1) is merged, and the obsolete `engineering/ep12-reliability` branch was deleted locally and remotely.
- Last implementation commit before this handoff refresh: `bb5f98ea815fa7624e1ef3ed6dafc888a0fd9375`. Resolve the review candidate with `git rev-parse HEAD`; documentation-only refresh commits may be newer.
- Recorded implementation worktree: clean and synchronized with `origin/main`
- Recorded pre-consolidation PR checks: both test jobs and both Dockerfile jobs passed on 2026-09-05

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

- Full local suite: `449 passed, 32 skipped` in `38.33 seconds`.
- Ported engine suite: `342 passed, 31 skipped`.
- Cross-speaker focused suite: `82 passed`.
- Local Python compilation and `git diff --check`: passed.
- Current PR test and Dockerfile checks: passed.
- Docker was not available in the local Windows environment, so a local image build was not performed.
- Gmail SMTP notification was verified by a real test email to the configured recipient. The app password remains outside Git in the Windows user environment.
- A user-exported 3,225-byte Netscape cookie file was copied byte-for-byte to `C:\Users\Ismail\.config\ma-sub\youtube-cookies.txt`, restricted to the current Windows user, and bound through the persistent user-level `MAS_YTDLP_COOKIES` variable. Its 23 non-comment records and YouTube domain were validated without logging cookie values. Authenticated source acquisition and the equivalent RunPod secret-file mount remain unverified.
- Official-channel discovery was exercised live with the cookie file and resolved Episode 12 to `https://www.youtube.com/watch?v=yVCzw_dFZA8`. Media download itself remains unverified in the production pipeline.
- RunPod Pod `p54vvbyu76eztn` (`muhtemel-ask-ep12`) was identified with an RTX 4090 configuration and `/workspace` network-volume mount. A dedicated `ma-sub-control` API key was created, stored outside Git in the Windows user-level `RUNPOD_API_KEY` environment variable, and authenticated successfully against the REST Pod endpoint. The observed Pod state was `EXITED`; this verifies credentials and read access only, not the pipeline shutdown path or zero billing after a real run.
- The Pod has a 30 GB temporary container disk and a 50 GB network volume `xgogcmey5o` in `EU-RO-1`. The stopped container disk is not billed. The network volume remains billed at the observed RunPod rate of `$0.07/GB/month`, which derives to `$3.50/month` for 50 GB. Deleting files does not reduce this fixed allocation, and RunPod does not allow shrinking it. Do not delete the volume before deciding whether its retained `/workspace` state is needed for the real run.
- Real full-episode GPU inference was not performed.
- Live Google Drive upload/readback was not performed. Old system snapshots and Episode 12 intermediate/source folders were moved to Drive Trash during cleanup. Episode 12's existing MKV, root Indonesian SRT, and original `final/subtitles` tree were restored and verified present. They must be retained; no Turkish SRT was present in the inspected Drive folder.
- Live RunPod stop and external billing-state verification were not performed.

Skipped local tests include Windows cases that require symlink privilege. Treat skipped tests as reported evidence, not passes.

## Open release work

1. Mount or copy the validated YouTube cookie file into RunPod outside the repository, set `MAS_YTDLP_COOKIES` to that Pod-local path, and verify authenticated source acquisition. Never commit the file.
2. Configure the Pod-local Gmail variables and the `gdrive` rclone remote without committing their values.
3. Implement and test the local RunPod controller. `mas run` currently executes locally; it does not start the stopped Pod, wait for readiness, or execute the remote command. Pod startup must happen only when the operator starts a real episode run.
4. Configure a secure remote execution path. Windows OpenSSH is installed, but no local `.ssh` key directory was present in the latest check.
5. Build the Docker image from a clean Linux or RunPod checkout.
6. Run `./mas doctor` on the selected GPU and record GPU, CUDA, PyTorch, WhisperX, and model versions.
7. Run one complete real episode through both ChatGPT handoffs.
8. Inspect source playback against representative final cue boundaries and resolve every review interval with acoustic evidence.
9. Confirm strict Drive delivery with exact remote byte-count and SHA-256 readback receipts.
10. Exercise success, failure, handoff wait, maximum-runtime, and idle-watchdog shutdown paths.
11. From outside the Pod, confirm provider state is `EXITED` and no active GPU compute charge remains.
12. Decide whether to delete the 50 GB network volume after durable Drive delivery. Deletion is permanent; ordinary file cleanup does not reduce its `$3.50/month` allocation cost.
13. Run the Astra prompt in `docs/ASTRA_REVIEW_PROMPT.md` against the final commit and retained runtime evidence.
14. Keep `main` as the sole maintained branch. Do not create a stable tag until all real-environment gates pass and the user decides.

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
C:\Users\Ismail\CodeBase\ma-sub reposundaki calismayi devral. Once repo kokundeki AGENTS.md dosyasini, sonra README.md ve docs/CODEX_HANDOFF.md dosyasini tamamen oku. Tek production dali olarak main uzerinde calis; once git status, HEAD ve varsa eski branch/PR durumunu kontrol et, kirli agacta kullanici degisikliklerini koru. CODEX_HANDOFF.md icindeki acik release islerinden devam et. Pod p54vvbyu76eztn su anda EXITED durumundadir; RUNPOD_POD_ID ve RUNPOD_API_KEY Windows User environment'ta hazirdir ancak degerlerini loglama. Gercek bolum komutu verildiginde Pod'u sinirli timeout ile baslatan, hazirligi bekleyen, uzak komutu calistiran ve her cikis yolunda kapanisi dogrulayan yerel controller'i tamamla; Pod'u onceden bosuna baslatma. Gercek GPU, Drive readback veya pipeline RunPod kapanisi yapilmadiysa yapilmis gibi raporlama. Schema, hash, source immutability, timing-only override, speaker overlap, strict/emergency ayrimi ve subtitle kalite kurallarini test gecsin diye gevsetme. Anlamli degisiklikleri test et, commit et ve main dalina push et. Astra incelemesi ve gercek GPU episode kaniti tamamlanmadan stable tag basma.
```
