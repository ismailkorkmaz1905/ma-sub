# EP14 semantic alignment offline replay - 19 September 2026

## Boundary

The replay opened the real EP14 source identity, raw ASR, word timings, VAD,
validated Turkish correction return and emergency Turkish SRT read-only. All
derived files were created in a temporary directory and removed afterward. It
made no RunPod, GPU, Drive, Gmail, OpenAI API or model-download call. Forced CTC
call count was 0.

## Real artifact results

| Measure | Result |
|---|---:|
| Timing words | 11,833 |
| Eligible timing words | 11,749 |
| Semantic windows | 1,747 |
| Deterministic AUTO_PASS windows | 1,141 |
| Handoff-needed windows | 606 |
| Validated returned windows | 606 |
| Final replay blocks | 3,041 |
| Quarantine records in current raw ASR | 40 |
| Quarantined timing words | 84 |
| Unassigned eligible words | 0 |
| Duplicate word ownership | 0 |
| Internal gaps over 1,000 ms | 0 |
| Known mixed-speaker blocks | 0 |
| Known cross-scene blocks | 0 |
| GPT timestamp fields | 0 |
| Pack size | 435,175 bytes |
| Exact `Altyazı M.K.` strings in pack | 0 |

The historical first-hour Indonesian SRT contains 106 exact `Takarir M.K.`
occurrences. The current real raw-ASR segmentation represents detected metadata
as 40 records covering 84 timing words, so 106 is not truthfully reportable as
the current raw quarantine-record count. The mandatory synthetic regression uses
106 exact `Altyazı M.K.` records and proves that none reaches the semantic pack.

## Release-QA result and exact blocker

The real pack and return validators ran end to end and complete ownership passed.
No production ChatGPT Pro linguistic return exists for the 606 unresolved
windows. To exercise the validator and final-block builder offline, the replay
used an explicitly labeled development-only return generated from owned timing
word transcription. That return is not linguistic approval and correctly did
not pass release QA:

| Gate | Count |
|---|---:|
| Overlaps | 555 |
| CPS failures | 327 |
| More than two rendered lines | 30 |
| Lines over 84 characters | 0 |
| Long-line warnings | 0 |
| Release eligible | false |

This is the exact external acceptance boundary: a real ChatGPT Pro semantic
return for the generated 606 windows is required before EP14 could claim a
semantic release PASS. Synthetic ownership PASS is not presented as a real
linguistic or release PASS.

## Emergency-SRT comparison

The emergency Turkish SRT has 2,996 blocks. The semantic development replay has
3,041 blocks, a delta of +45. There are 68 exact start/end timing-pair matches and
1,576 exact normalized-text values shared between the two sets. Exact equality is
not expected because emergency timing is comparison evidence only and never
became semantic timing authority.
