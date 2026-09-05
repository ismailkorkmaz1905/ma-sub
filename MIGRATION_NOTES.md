# Migration notes

## Source preservation

The complete 2026-09-04 beta source archive is preserved under `legacy/SYSTEM_V2_BETA/`. Its notebooks, scripts, tests, and reports are reference material only. No production command imports or executes a notebook.

The maintained engine now lives under `src/mas/`, configuration under `config/`, and executable tests under `tests/`. The public operator surface is:

```bash
./mas run 13 --source-url 'SOURCE_URL'
./mas run 13
```

## Contract carried forward

- Correction-first Turkish handoff followed by Indonesian translation handoff
- Exact pack manifest, UID, count, order, text, timing, and SHA-256 binding
- GPU-only ASR, acoustic review, and forced alignment
- Immutable source media and stale-checkpoint rejection
- Speaker-aware overlap without cross-speaker text merging
- Timing-only overrides
- Strict finalization gates for subtitle format, timing, coverage, and translation

The old broken Base64 bootstrap payload is removed. Dependencies are installed from `requirements.lock`; Docker uses the same lock. Notebook-specific transfer and monkeypatch procedures are not production mechanisms.

## Compatibility and validation status

Windows local validation does not prove CUDA or Linux container behavior. The imported historical suite contains platform-sensitive symlink and file-sync cases; those results must be reported separately from the maintained production suite.

No real full-episode GPU inference, live Drive upload/readback, or live RunPod shutdown verification has been performed on this branch. These remain release gates. No stable tag may be created before Astra review and the real GPU run.
