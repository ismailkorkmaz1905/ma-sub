# Review notes: Colab candidate

Update: the scheduled publisher-caption path is documented in
[`automation/README.md`](../automation/README.md). It avoids Colab and paid APIs.
The findings below describe the earlier, separate Colab implementation.

Base reviewed: `9352d3631e5ea58f860f6b2bf1da2e677e2f833b`.
Goal: natural Indonesian subtitles for newly released Muhtemel Aşk episodes,
without an online personal PC or RunPod, and without another post-edit forced
alignment loop. User explicitly authorized a new implementation if needed.

## Findings in the existing project

- `src/mas/cli.py` routes the normal `run` command into the RunPod controller.
  The controller has local SSH, rclone, source/return and encoder dependencies.
- The existing `forced_align.py` has edited-token acoustic gates and bounded
  recovery/search procedures. Corrected Turkish is therefore tied to a second
  acoustic alignment pass. Retaining these requirements in a new wrapper would
  retain the same operational failure mode.
- The newer semantic path already treats original ASR words as timing evidence,
  but remains integrated with the legacy orchestration, delivery and review stack.
- The production translation contract is useful: immutable UID/order/timing,
  neighboring context, natural Indonesian register, canonical names and explicit
  religious choices. It is bundled unchanged, with separate evidence notes.

## Chosen scope

Independent `src/mas/colab_flow.py`, self-contained notebook and dedicated tests.
No imports from the legacy controller/aligner. Existing production entry points
remain available. No paid resource, account, credential or scheduled job is created.

Prepare on Colab GPU -> Drive translation pack -> ChatGPT language work ->
Colab CPU finalization/review -> TR/ID SRT. Optional copied-video MKV preview or
CPU-burned MP4. A source file in Drive is supported when YouTube rejects downloads.

## Timing authority and invariants

1. Normalize the source audio timeline with FFmpeg; keep its content hash.
2. VAD proposes quiet chunk boundaries and missing-speech diagnostics only.
3. ASR runs once per chunk with word timestamps. A completed chunk is written
   immediately and bound to source/audio/model revision/package versions.
4. Partition source words into short, immutable cue spans. No global temporal
   redistribution based on corrected Turkish or Indonesian character counts.
5. Cue start equals its first ASR word start. Cue end never precedes any owned
   word end. Tail padding consumes at most 200 ms of available silence.
6. When source speech overlaps, report it. Never trim away speech to fake a
   non-overlapping result. Explicit listening decisions can correct ASR errors.
7. Language return cannot alter timing or UID fields. A review document has a
   separate authority and is bound to the exact schema and returned ZIP.
8. Draft output is available even when review is pending. Finality is honest;
   no silent quality bypass or automatic retry-until-pass loop exists.

## What is and is not demonstrated

Automated checks cover adversarial return ZIPs, identity/order/hash preservation,
source replacement refusal, subtitle overlap, early-end protection, reading speed,
Unicode, VAD gaps, multi-cue gap repair, interrupted/resumed ASR (fake model),
checkpoint tampering, notebook/source parity, subprocess timeout, and a real FFmpeg
fixture whose audio begins 1.2 seconds after media start.

These checks do **not** establish ASR quality, Indonesian naturalness, complete
speech coverage, Colab account readiness, or reliable YouTube access. No real GPU
inference or full-episode audiovisual acceptance was run in this workspace.

## Required live acceptance before Friday production

- Open notebook on the user's Colab account; confirm GPU allocation and Drive access.
- Use a real 5-10 minute episode excerpt, including silence, quiet speech, music,
  rapid turn-taking, names and at least one overlapping exchange.
- Run prepare, translate using the bundled contract, finalize, then listen with
  captions. Record actual onset/offset errors; do not infer accuracy from scores.
- Target: no observed cue appearing more than 200 ms before its spoken phrase;
  no observed premature disappearance. A cue may trail speech by up to 200 ms.
  These are review targets, not guarantees provided by the ASR timestamps.
- Check every detected coverage gap and representative samples throughout the
  full episode. VAD and ASR can miss the same soft speech; full playback is needed
  to assert episode-wide acceptance.
- Interrupt a Colab session after a saved chunk and resume from Drive. Verify
  package/runtime compatibility, source binding, and no repeated completed chunks.
- Test SRT with the intended mobile/TV player. Optional MKV/MP4 must be played back.

## Known tradeoffs / review focus

- Word timestamps come from Whisper attention/DTW. This removes post-edit CTC
  alignment, not all timing estimation. Bad word timestamps still need listening.
- VAD is imperfect under drama soundtracks. No speaker diarization or independent
  recognizer is claimed. Long continuous speech may hit a flagged hard chunk cut.
- New ASR installation is pinned at the direct inference-library level. Colab's
  driver, Python/base image and remaining transitives can change. Runtime versions
  and model revision are recorded; actual CUDA execution remains a live gate.
- Language work is a manual ZIP handoff, not an LLM API background service. No
  language-model name or speculative API entitlement is hard-coded.
- Drive-mounted copies are hashed after writing but are not independently read
  through the Drive API. Outputs say so rather than claiming verified cloud delivery.
- Colab is interactive with variable resource limits. No 06:00 autonomous start,
  no uptime promise, no keepalive tricks, no infinite retry or quota bypass.
- `REVIEWED` means deterministic gates passed plus recorded user review; it is
  not independent proof that a person actually listened or that every line is right.

## Suggested independent audit

Trace a word through normalization, chunk offset, cue ownership, pack, returned
translation and SRT. Try corrupt/missing/reordered records, delayed audio, skipped
speech, long target text, overlapping words and interrupted checkpoints. Check that
no validation flag quietly changes time or drops dialogue. Inspect the live Colab
run separately before approving this as the production path.

## Recorded local verification (2026-09-23)

Environment: Python 3.12.14, pytest 8.3.5, FFmpeg 6.1.1, CPU only.

| Command | Observed result |
|---|---|
| `PYTHONPATH=src python -m pytest tests/test_colab_flow.py -q` | 36 passed in 1.95 seconds |
| `PYTHONPATH=src python -m pytest -q` | 1,295 passed in 79.01 seconds |
| `python colab/build_notebook.py` | Self-contained notebook rebuilt; parity test passed |
| `git diff --cached --check` | Passed |

Both test runs emit one expected Python ZIP warning: an adversarial fixture writes
a duplicate filename to verify rejection. No real episode media, credentials or
notebook outputs are committed. The full suite includes the existing production
regressions; passing it does not substitute for the live acceptance above.
