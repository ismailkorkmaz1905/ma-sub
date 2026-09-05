# Codex handoff

## Repository snapshot

- Repository: `ismailkorkmaz1905/ma-sub`
- Branch: `engineering/ep12-reliability`
- Pull request: [#1](https://github.com/ismailkorkmaz1905/ma-sub/pull/1), open and draft against `main`
- Implementation baseline HEAD before this handoff-document change: `63bd7bbd353ceb5d30de579c92c6999586f68132`
- Recorded implementation worktree before this handoff-document change: clean and synchronized with `origin/engineering/ep12-reliability`
- Recorded PR checks: both test jobs and both Dockerfile jobs passed on 2026-09-05

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
- Added Google Drive upload verification using remote byte-count and SHA-256 readback.
- Added Gmail notifications for run start, every stage start/completion/failure, translation waits, and final readiness. `./mas notify-test` verifies configured credentials.
- Added optional YouTube Netscape cookie input through `MAS_YTDLP_COOKIES`. Cookie contents are validated without being logged.
- Added Docker and RunPod bootstrap, watchdog, timeout, stop, and external-verification guidance.
- Preserved the 15 Episode 12 incident intervals in `tests/fixtures/ep12_incident_intervals.json`.
- Added operator, architecture, migration, incident acceptance, and Astra handoff documentation.

## Validation recorded

- Full local suite: `431 passed, 32 skipped` in `35.30 seconds`.
- Ported engine suite: `342 passed, 31 skipped`.
- Cross-speaker focused suite: `82 passed`.
- Local Python compilation and `git diff --check`: passed.
- Current PR test and Dockerfile checks: passed.
- Docker was not available in the local Windows environment, so a local image build was not performed.
- Gmail SMTP notification was verified by a real test email to the configured recipient. The app password remains outside Git in the Windows user environment.
- Drive contains a private 2,987-byte `youtube-cookies.txt`, but the connector could not materialize it to the local filesystem. Fresh Chrome and Edge export also failed after both browsers were closed because Chromium App-Bound Encryption prevented `yt-dlp` DPAPI decryption. `MAS_YTDLP_COOKIES` remains unset and authenticated YouTube source acquisition remains unverified.
- Real full-episode GPU inference was not performed.
- Live Google Drive upload/readback was not performed.
- Live RunPod stop and external billing-state verification were not performed.

Skipped local tests include Windows cases that require symlink privilege. Treat skipped tests as reported evidence, not passes.

## Open release work

1. Export or manually download a fresh Netscape-format YouTube cookie file to `C:\Users\Ismail\.config\ma-sub\youtube-cookies.txt`, restrict it to the current Windows user, and set `MAS_YTDLP_COOKIES`. Never commit the file.
2. Build the Docker image from a clean Linux or RunPod checkout.
3. Run `./mas doctor` on the selected GPU and record GPU, CUDA, PyTorch, WhisperX, and model versions.
4. Run one complete real episode through both ChatGPT handoffs.
5. Inspect source playback against representative final cue boundaries and resolve every review interval with acoustic evidence.
6. Confirm strict Drive delivery with exact remote byte-count and SHA-256 readback receipts.
7. Exercise success, failure, handoff wait, maximum-runtime, and idle-watchdog shutdown paths.
8. From outside the Pod, confirm provider state is `EXITED` and no active GPU compute charge remains.
9. Run the Astra prompt in `docs/ASTRA_REVIEW_PROMPT.md` against the final commit and retained runtime evidence.
10. Keep PR #1 draft. Do not merge to `main` or create a stable tag until all real-environment gates pass and the user decides.

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
C:\Users\Ismail\CodeBase\ma-sub reposundaki calismayi devral. Once repo kokundeki AGENTS.md dosyasini, sonra README.md ve docs/CODEX_HANDOFF.md dosyasini tamamen oku. engineering/ep12-reliability dalinda kal; PR #1 acik ve draft kalsin. main dalina merge etme ve stable tag basma. Once git status, HEAD ve PR kontrollerini yap; kirli agacta kullanici degisikliklerini koru. CODEX_HANDOFF.md icindeki acik release islerinden devam et. Gercek GPU, Drive readback veya RunPod kapanisi yapilmadiysa yapilmis gibi raporlama. Schema, hash, source immutability, timing-only override, speaker overlap, strict/emergency ayrimi ve subtitle kalite kurallarini test gecsin diye gevsetme. Anlamli degisiklikleri test et, commit et ve yalniz engineering/ep12-reliability dalina push et.
```
