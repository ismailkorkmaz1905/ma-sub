# Muhtemel Ask Subtitle System V2 Beta - Verification Report

Verification date: 2026-09-04 UTC

## Release status

This build is suitable for continuing the isolated Episode 12 beta pilot. It is not yet a
production release and it does not repair an existing V1 SRT or MKV in place.
Production acceptance requires a complete GPU run on the real episode audio
followed by human playback review.

## Automated verification

The final source tree passed:

```text
python3 -m unittest discover -s tests -q
Ran 413 tests
OK

python -m compileall -q src tests
PASS
```

All code cells in the four V2 notebooks compile. Each V2 notebook resolves the
exact isolated Drive runtime
`MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA`; the three legacy V1 notebooks
remain bound to `SYSTEM` and contain no V2 beta import path.

The suite covers, among other cases:

- independent Silero-VAD policy and speech-hole rescue;
- contextual WAV review records and ZIP/hash/path tampering;
- ASR hallucination discard and confirmed-dialogue routing;
- text-only Turkish correction that cannot claim it opened or heard a WAV;
- review-only UID additions that reuse the completed hash-bound ASR/VAD
  checkpoint without loading Whisper or retranscribing the episode;
- separate limits for automatic ASR candidates and explicitly requested
  text-deletion review, so a bounded repair set cannot hide detector runaway;
- strict reuse of an existing text-only correction only when the refreshed pack
  differs solely by added immutable audio-review evidence;
- resumable, hash-bound short-WAV review in Colab with a blind first decode,
  one bounded evidence-conditioned retry, and fail-closed manual remainder;
- zero-duration faster-whisper boundary words normalized to a bounded 1 ms
  interval instead of aborting the remaining short-WAV review queue;
- finalization rejection when the provisional Turkish ZIP, final Turkish ZIP,
  WAV identity, audio-review report, or report digest is stale or tampered;
- Turkish correction before forced alignment;
- WhisperX word timing, acoustic-score, lexical-edit, deletion, duration and
  outward-drift gates;
- VAD overlap for each aligned word, including internal non-speech islands;
- speech-aware cue segmentation, the 999 ms internal-gap regression, cue
  overlap, CPS and provenance checks;
- Turkish/Indonesian semantic, name, number and religious-term validation;
- path-only finalization evidence, atomic output/report behavior and stream-copy
  MKV verification;
- finalizer-to-archive report compatibility and exact three-file retention;
- YouTube caption evidence with a first unexplained-run boundary of 229/230 ms.

The caption red-team replay used a real generated FLAC and contextual WAV
extraction. A caption covering 0-2000 ms with evidence only at 1000-2000 ms now
creates an immutable audio-review candidate. Leaving it pending stops
finalization before mux. A 229 ms unexplained run remains within the explicit
beta policy; the first 230 ms run requires review.

## Known residual limits

No finite automatic suite can prove zero subtitle errors. In particular:

- If coarse ASR, independent Silero VAD and available YouTube captions all miss
  the same short or quiet utterance, the system has no evidence from which to
  create a review candidate. The configured 120 ms Silero minimum is not an
  acoustic guarantee.
- A plausible but false ASR token wholly inside genuine speech can still pass
  if it receives a strong forced-alignment score and the Turkish correction or
  human review does not remove it.
- The beta deliberately accepts some bounded timing slack, including caption
  unexplained runs up to 229 ms and an effective aligned-word allowance up to
  180 ms beyond an unpadded audit-VAD core under the canonical policy.
- Model behavior on crowded dialogue, music and very short interjections remains
  subject to real-episode calibration and playback review.

These are declared pilot limits, not validator bypasses. Thresholds must not be
loosened merely to obtain PASS; a discovered failure should first become a
reproducible regression test.

## Pilot acceptance sequence

1. Install this release separately under `SYSTEM_V2_BETA`.
2. Rebuild the pilot episode through all three V2 production stages.
3. Let notebook 02 audit only the short contextual WAVs; explicitly listen only
   to any ambiguous remainder it refuses to decide.
4. Watch beginning, middle, end, fast-dialogue, music and every previously bad
   scene in the final MKV.
5. Run archive cleanup only after playback is accepted. Verified cleanup keeps
   one MKV and the two external SRTs, and removes the source/work tree only
   after A/V and embedded-subtitle verification.
