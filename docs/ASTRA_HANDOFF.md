# Astra handoff

## Review target

Review `semantic-block-v1` as a release policy distinct from strict CTC.

- Timing authority: immutable ASR word timeline.
- Linguistic authority: human-mediated ChatGPT Pro word-span return.
- Safety authority: Python validation and release QA.
- Translation authority: immutable block-UID Indonesian handoff.
- Delivery authority: remote byte count and SHA-256 readback.

New EP15+ states default to semantic alignment. Existing EP14 and older states
remain on their persisted policy. First-hour and whole-episode selections must
bind the same canonical final-block hash. Forced alignment remains available as
explicit `strict-ctc-v1` and must not be removed or silently invoked as fallback.

## Evidence boundary

The EP14 offline replay used real retained source identity, ASR word timings,
VAD, Turkish correction evidence, and emergency comparison subtitles. It made no
RunPod, Drive, Gmail, OpenAI API, or model-download call. Complete word ownership
passed, but the development-only return failed release QA. A real ChatGPT Pro
return for 606 unresolved windows is still required for linguistic acceptance.

Do not infer acoustic, perceptual, GPU, Drive, billing, or end-to-end PASS from
local tests or the replay.

## Review files

- [Semantic architecture](SEMANTIC_ALIGNMENT_EP15.md)
- [EP14 replay evidence](EP14_SEMANTIC_REPLAY_2026-09-19.md)
- [Architecture decisions](ARCHITECTURE_DECISIONS.md)
- [EP14 launch repair](EP14_READY_2026-09-18.md)
- [Delivery-first contract](DELIVERY_FIRST_2026-09-15.md)

Use [the independent review prompt](ASTRA_REVIEW_PROMPT.md) only for the formal
independent review.

## Local review commands

```powershell
git status --short --branch
git rev-parse HEAD
& .\.venv\Scripts\python.exe -m pytest -q tests
git diff --check
.\mas.ps1 doctor --controller
```

Also inspect `Dockerfile` and `runpod/` for release work. Docker results are
UNKNOWN when the host has no Docker executable. Do not create a stable tag or
start paid compute from this handoff alone.
