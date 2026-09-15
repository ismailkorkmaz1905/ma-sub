# Episode 13 runtime audit

## Superseded preliminary report

Use [the deduplicated timeline audit](EP13_TIMELINE_AUDIT_2026-09-12.md)
for measured durations and [the quality-preserving implementation plan](FOUR_HOUR_PIPELINE_PLAN_2026-09-12.md)
for decisions. The historical figures below contain replayed log events and
unproven time attribution. In particular, making acoustic alignment optional
without a validated replacement is NOT an approved quality-preserving change.
These preliminary recommendations are retained only as audit history.

## Finding

The visible subtitle quality is not the main runtime bottleneck. The largest
measured stage is the strict word-level CTC alignment gate. Repeated controller
bootstrap, capacity, transfer, diagnostic collection and shutdown work consumed
more logged time than any individual stage.

The delivery path should keep Turkish text quality, natural Indonesian,
speaker separation, cue readability, overlap QA, burned-video inspection and
Drive byte/SHA-256 readback. It should remove strict word-level CTC proof from
the delivery-critical path and retain it only as an optional non-blocking audit.

## Observed window

- Source: 89 Episode 13 `run-*.log` files whose first event is at or after
  2026-09-11 06:20 SGT.
- First logged run: 2026-09-11 06:31:19 SGT.
- Last logged run ended: 2026-09-12 10:34:59 SGT.
- Wall-clock window between those events: 28 hours 3 minutes 40 seconds.
- Time inside the 89 non-overlapping CLI runs: 15 hours 43 minutes 45 seconds.
- Time between CLI runs: 12 hours 19 minutes 55 seconds. This includes code
  repair, handoffs, translation, diagnosis and idle gaps. It is not GPU time.

## Eleven-stage breakdown

The totals below sum terminal stage durations across retries. A stage that was
resumed or repeated is counted on every execution because it consumed real
time again.

| Stage | Attempts | PASS | FAIL | Total | Median | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| 1. download | 40 | 36 | 4 | 11.8 min | 14.9 s | 93.5 s |
| 2. audio | 36 | 36 | 0 | 3.0 min | 3.5 s | 43.9 s |
| 3. raw_asr | 33 | 33 | 0 | 89.5 min | 91.5 s | 40.1 min |
| 4. tr_pack | 33 | 33 | 0 | 11.9 min | 21.2 s | 28.4 s |
| 5. tr_return | 32 | 32 | 0 | 1.3 min | 2.3 s | 3.4 s |
| 6. audio_review | 32 | 31 | 1 | 31.8 min | 11.1 s | 10.4 min |
| 7. forced_alignment | 23 | 0 | 23 | 379.4 min | 16.4 min | 46.5 min |
| 8. id_pack | 0 | 0 | 0 | 0 min | N/A | N/A |
| 9. id_return | 0 | 0 | 0 | 0 min | N/A | N/A |
| 10. finalize / burn_mp4 | 0 | 0 | 0 | 0 min | N/A | N/A |
| 11. drive_readback | 0 | 0 | 0 | 0 min | N/A | N/A |

Measured terminal stage time totals 8 hours 48 minutes 42 seconds. Strict
forced alignment accounts for 71.8% of that total. Controller and work outside
terminal stage timers account for the remaining 6 hours 55 minutes 3 seconds
inside logged runs.

## What blocked delivery

- Strict forced alignment was executed 23 times and passed 0 times.
- The last run spent 1,260.9 seconds in forced alignment and still failed on
  36 unique unresolved neighbor pairs.
- The final diagnostics recorded 3,375 joint attempts, 1,942 candidate
  validation failures and 7,536 candidate options.
- 25 of the 36 unique pairs were already disjoint or touching in authoritative
  coarse cue geometry. All 11 coarse overlaps involved rescued speech holes.
- Fresh workers repeatedly installed 136 operating-system packages: 65.6 MB of
  downloads and 190 MB of installed files per clean worker.
- Source verification reread the 1,076,347,555-byte source locally and remotely
  on fresh sessions. Across 38 recorded seed verifications this is about 81.8 GB
  of hash input.
- Failure collection redownloaded about 1.849 GB; about 1.702 GB was duplicate
  retained evidence.
- Monitoring opened three SSH sessions every 15 seconds.
- Gmail notification sends were synchronous. The last run sent 14 messages,
  but their estimated 27 to 57 seconds is not a primary bottleneck.

## Changes required for a four-hour delivery

1. Replace delivery-blocking per-word CTC alignment with the validated segment
   timeline. Keep CTC as an optional audit after delivery. Keep text, speaker,
   overlap, CPS and visual QA gates unchanged.
2. Stop retrying an unchanged alignment failure. One repeated invariant must
   end the optional audit without restarting the complete worker.
3. Build and pin an immutable RunPod image containing ffmpeg, fonts, Python and
   model dependencies. Remove per-Pod package installation.
4. Cache source SHA-256 and remote object identity. Do not reread 1.076 GB when
   size, immutable object identity and the recorded digest are unchanged.
5. Upload Turkish and speaker evidence once by content hash. Reuse retained
   remote checkpoints.
6. Download only changed failure artifacts. Do not fetch duplicate diagnostics.
7. Replace three monitoring SSH sessions per poll with one bundled status call.
8. Move Gmail out of the paid worker path and send only stage transition and
   final result messages.
9. Start three parallel Indonesian language batches as soon as corrected
   Turkish text is stable. Do not keep a GPU alive while translation is done.
10. Use the already verified local Intel QSV encoder for the burned MP4. A
    30-second 1080p subtitle sample ran at 4.06 times realtime, deriving to
    about 36.4 minutes for the 8,855.161-second episode before final verification.
11. Start Drive upload immediately after encoding and verify the remote bytes
    once. Preserve collisions before replacement.

## Quality boundary

The cut is to proof and repeated infrastructure work, not to subtitle content.
The delivery must still require natural conversational Indonesian, correct
formal and informal register, canonical names, preserved numbers and religious
expressions, no missing UIDs, no display overlap, at most 20 characters per
second, visual burned-subtitle sampling, and exact Drive byte/SHA-256 readback.
