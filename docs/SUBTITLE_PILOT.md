# Episode 11 cue pilot

Status: local prototype, not acoustic acceptance or production migration. Episode 12 must not start before the Episode 11 pilot is reviewed. Combined new compute allowance is USD 0.50; no new paid operation has been initiated.

## Contract

`subtitle-pilot` is a separate draft mode within this project. Existing strict correction, alignment, translation and publication contracts are unchanged. The draft preserves source-relative estimated word boundaries, splits at supported speaker changes, and retains uncertain speakers as unknown. Simultaneous cues retain separate text and cue lineage. Readability warnings do not silently shorten spoken intervals or delete dialogue.

Estimated model timestamps are not measured speech boundaries. Local tests prove preservation of supplied intervals, not actual synchronization. Every result remains `REVIEW_REQUIRED`, `acoustic_acceptance=NOT_VERIFIED`, `strict_delivery=false`. There is no Drive publication or Indonesian translation path in this prototype.

## Commands

Use `python -m mas.cli subtitle-pilot 11 --help` in the project environment. Modes:

- `--pack PATH --uid UID`: validate and extract one to eight embedded correction-pack WAV clips. No inference or external calls.
- `--evidence PATH`: regenerate a draft from checksum-bound acoustic JSON. No inference.
- `--audio PATH --source-audio PATH --model-dir PATH --diarization-model-dir PATH --source-sha256 SHA --offset-ms INTEGER`: offline, CUDA-only inference using already provisioned models. WAV duration must not exceed 120 seconds. The source hash must identify the source audio, not the correction ZIP. The source WAV hash, format and exact PCM sample bytes at the requested offset are checked before and after inference.

Outputs stay under the episode's `work/subtitle-pilot` directory. Existing completed drafts are not overwritten. ASR and speaker-stage caches are separate, checksum-bound artifacts. Do not delete them to bypass identity failures.

The inference subprocess has a 180-second timeout. This is NOT a Pod lease manager or a USD spending cap. Do not start unmanaged paid compute to execute it. An externally verified shutdown and total-cost guard are prerequisites for a paid pilot.

## Prepared real samples

Artifact: `EPISODES/Muhtemel Ask 11.Bolum/work/subtitle-pilot/samples-ab61e8d655434f36/samples.json`. Its entries retain ZIP member names, SHA-256, source offsets and WAV durations. Extraction verified member bytes and durations; no acoustic model has processed these samples in this task.

| UID suffix | Historical selection category | Duration, milliseconds |
|---|---|---:|
| 065c477f792dd079 | Drift | 1880 |
| a78a03af33e196d7 | Word score | 4660 |
| 96edb78d62a01c4a | Edited-text score | 1880 |
| 87f362dfc9b77747 | Duration | 2380 |

Total: 10,800 milliseconds, derived by summing the manifest durations. These short excerpts do not establish full-scene diarization or normal-dialogue quality. Longer contextual normal, rapid-turn, music and overlap samples remain required. The final two conflict UIDs are not covered by this pack's inspected audio.

## Runtime and acceptance blockers

Local validation on 2026-09-06: focused suite 24 passed in 0.27 seconds; final full suite 587 passed, 32 skipped in 37.77 seconds. `git diff --check` passed. Sol independently reviewed the pilot and reran focused tests; its source-binding and model-identity findings were corrected. `doctor` reports missing ffmpeg, ffprobe and torch. Docker build and CUDA integration were not run. Decision: local prototype checks PASS; real acoustic pilot BLOCKED.

- Optional `requirements-pilot.lock` is not part of the production image. Linux dependency resolution, image build and actual CUDA imports remain unverified.
- stable-ts 2.19.1 is pinned for evaluation; its repository is archived. It is not adopted as an unconditionally maintained production dependency.
- Community-1 requires model access conditions and a complete offline cache. No gated terms were accepted by this task.
- Model directories are hashed, but external model references in a diarization configuration require additional closure verification before reusable GPU evidence can be trusted.
- Local machine has no NVIDIA CUDA GPU, torch or ffmpeg available. No CPU fallback is authorized.
- No cheap-GPU comparison has been measured. Cold provisioning, downloads and shutdown time count toward the USD 0.50 allowance.

Pilot acceptance requires listening/viewing the same fixed contextual clips, reporting missing dialogue, early/late boundaries, cut-off speech, speaker mixing, inference time and total lease cost. Do not infer four-hour episode readiness from a short clip or unit tests. Only after that review should production cue-level correction/translation integration and a fresh episode be considered.
