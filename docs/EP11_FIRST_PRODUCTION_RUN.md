# Episode 11 first production run record

## Result

Episode 11 reached the Turkish correction handoff on 2026-09-05. A later run
accepted the Turkish correction return and exercised bounded audio review. This
is not a completed episode result. Real RTX 4090 inference, source acquisition,
audio extraction, raw ASR, correction-pack generation, controller shutdown, and
external `EXITED` verification were exercised successfully.

The evidence source is the UTF-8 structured controller logs under
`EPISODES/Muhtemel Ask 11.Bolum/logs/`. Secret values and the source URL are not
stored in those logs.

## Attempt timeline

| Attempt | Controller elapsed | Observed result | Fix or action |
| --- | ---: | --- | --- |
| 1 | 2.594 s | Local rclone configuration was missing. | Restored the external `gdrive` configuration. |
| 2 | 347.187 s | SSH readiness timed out after 300 s. | Registered and verified the dedicated Pod SSH key. |
| 3 | 299.485 s | Cookie, virtualenv, and rclone paths failed strict preflight. | Forced transferred `runtime.env` files to LF; commit `ff05d65`. |
| 4 | 360.016 s | Torch import failed because transitive dependencies such as `typing_extensions` were absent. | Replaced direct-only sync with dependency-resolving persistent installation; commit `e35f1ae`. |
| 5 | 99.234 s | uv selected an incompatible `requests` inventory from the PyTorch index. | Enabled trusted all-index resolution; commit `3b589b5`. |
| 6 | 120.453 s | A CUDA dependency download/extract exceeded the 30 s uv timeout. | Added a 120 s HTTP timeout, 3 retries, and bounded concurrency; commit `526e79b`. |
| 7 | 13.688 s | RunPod start returned HTTP 500 without useful controller detail. | Preserved safe provider response detail; commit `94dfcb1`. |
| 8 | 13.641 s | RunPod start again returned HTTP 500. | The next retry used the improved provider error reporting. |
| 9 | 13.578 s | Provider reported no free GPU on the stopped Pod's host. | Migrated to an available compatible Pod and added safe adoption; commit `ea5a4bf`. |
| 10 | 322.703 s | Strict Drive preflight timed out after runtime setup passed. | Added 3 bounded 60 s Drive probes with 1 s and 2 s backoff; commit `31c0e93`. |
| 11 | 13.453 s | The stopped host again had no free GPU. | Migrated to another compatible RTX 4090 Pod. |
| 12 | 240.281 s | YouTube rejected the stale cookie and reported a missing JavaScript runtime. | Refreshed the external cookie and added persistent Deno/EJS challenge support; commit `8d3933e`. |
| 13 | 12.360 s | The stopped host again had no free GPU. | Migrated to another compatible RTX 4090 Pod. |
| 14 | 201.375 s | Initial SSH succeeded, then the first remote setup command stopped making progress. | Added SSH liveness options and bounded setup retries; commit `cc03084`. |
| 15 | 253.328 s | Source download passed, then audio state persistence failed because `path` was supplied twice. | Renamed the conflicting stage metadata parameter; commit `83f0c9d`. |
| 16 | 1,168.031 s | Expected exit code 20: Turkish correction handoff created and downloaded. | No failure. The Pod was stopped and externally verified as `EXITED`. |
| 17 | 13.609 s | RunPod reported that the stopped Pod's host had no free GPU. | No pipeline stage ran. The Pod was stopped and externally verified as `EXITED`. |
| 18 | 1,151.875 s | The Turkish return passed, then bounded audio review processed 245 records and stopped fail-closed with 139 pending in that run. | Raw ASR had rerun for 812.3 s; diagnostics were preserved and Pod `oaombke16eg7er` was stopped. |
| 19 | 236.671 s | The returned correction ZIP no longer matched the regenerated correction-pack input SHA-256. | Diagnostics were downloaded; replacement Pod `6em6qtm2slpohd` was stopped and externally verified as `EXITED`. |
| 20 | 288.656 s | Raw ASR stopped after 3.0 s because recovery-checkpoint review UIDs would have been removed. | Capacity migration replaced `6em6qtm2slpohd` with `46jqfhnkgxg0vm`, preserving the network volume at $0.74/hour. Download passed in 19.0 s, audio passed in 4.7 s, diagnostics were downloaded, and shutdown was verified after 7.5 s. |
| 21 | 217.782 s | Startup remained `RUNNING` without reaching readiness and the controller was interrupted with `KeyboardInterrupt`; exit code 130. | Capacity migration replaced `46jqfhnkgxg0vm` with `cwz5z9h9ww6mxn`, preserving the network volume at $0.74/hour. No pipeline stage ran; shutdown was verified after 4.0 s. |
| 22 | 395.250 s | Replacement Pod startup timed out after 180 s while provider status remained `RUNNING`. | Capacity migrations replaced `cwz5z9h9ww6mxn` with `bfw4f0h5z4glgn`, then `bfw4f0h5z4glgn` with `ffzrh7o2ru1103`; both preserved the network volume at $0.74/hour. No pipeline stage ran; shutdown was verified after 4.2 s. |
| 23 | 463.328 s | Bounded audio review processed 245/245 and stopped fail-closed with 138 pending. | Capacity migration replaced `ffzrh7o2ru1103` with `x9fhfhzl1qjs3q`, preserving the network volume at $0.74/hour. Download passed in 10.3 s, audio in 3.1 s, checkpoint-backed raw ASR in 32.7 s, correction-pack generation in 7.7 s, and Turkish-return validation in 1.5 s. Audio review took 158.2 s; diagnostics were downloaded and shutdown was verified after 7.5 s. |

The first 15 failed controller invocations totalled 2,313.376 s, or 38 min
33.376 s. The first 16 controller invocations through the successful handoff
totalled 3,481.407 s, or 58 min 1.407 s. Attempts 17 through 23 added 2,767.171 s,
so the 23 recorded controller invocations totalled 6,248.578 s. Of these, the 22
non-handoff invocations totalled 5,080.547 s. The
wall-clock interval from the first start at 07:02:35.297233 UTC to the handoff at
09:09:24.143537 UTC was 7,608.846 s, or 2 h 6 min 48.846 s. These controller
durations include local checks, setup, shutdown, and stopped intervals. They are
not RunPod billing measurements.

The wall-clock interval from the first start to attempt 23 finishing at
11:16:15.406118 UTC was 15,220.109 s, or 4 h 13 min 40.109 s. This includes
stopped gaps and is not provider billing time.

## First successful handoff measurements

| Measurement | Result |
| --- | ---: |
| Controller session | 1,168.031 s |
| Resumed source download verification | 13.8 s |
| Audio extraction | 3.7 s |
| Raw ASR | 986.5 s |
| Turkish correction-pack generation | 7.7 s |
| Provider shutdown verification | 7.6 s |
| Utterances | 2,860 |
| Correction batches | 12 |
| Speech holes | 88 |
| ASR hallucination candidates | 122 |
| Correction-pack bytes | 11,853,192 bytes |
| Correction-pack SHA-256 | `AD2B9A18AD6DBEFEF53336A89249CE152F5414E8EBED71EC1A13049EE31873B9` |

The first complete media transfer occurred during attempt 15 and took 23.7 s.
Attempt 16 reused the immutable source and therefore reports resumed source
verification rather than a second full download.

The raw ASR stage used a real RTX 4090. Two `nvidia-smi` observations recorded
4,028 MB at 79.96 W and 9% utilization, then 3,932 MB at 139.31 W and 11%
utilization. A raw-ASR recovery artifact was present before stage completion.

## Later acoustic evidence

The accepted Turkish correction return entered bounded audio review. The current
review inventory contains 245 records: 107 resolved and 138 pending manual
acoustic decisions.

Attempt 23 exercised the corrected checkpoint path: raw ASR passed in 32.7 s,
then bounded audio review processed 245/245 in 158.2 s and stopped fail-closed
with the same 107 resolved and 138 pending boundary.

After commit `f581fcb59091d026e4f53a825916ba9e7abcf479`, an independent
pinned-model CUDA CTC probe processed 138/138 pending records on Pod
`tccsb8991x84ua`. The probe used model
`samil24/wav2vec-xlsr-53-turkish-v4` pinned to revision
`07d79597b78c56758045e3a2cd1c44bc1a19b1e8` on CUDA. It reported
`exact_reference_match_count=0` after normalized comparison and closed 0 strict
decisions. The report's self SHA-256 is
`6081ddb6670ca0893bd94a704dd7eb94a6865785c0a9dea58698239bd48e65a6`; the
report file SHA-256 is
`8312910ddd11065a321362cab253c1033833829bcba2f73b3f04ef3174c996d2`.
The Pod was externally verified `EXITED`.

The CTC probe proves bounded execution of the pinned model on CUDA. Its 0 exact
matches and 0 strict closures are not an acoustic PASS and do not replace manual
listening or production forced alignment.

## Current boundary

The Turkish correction return is complete. Manual acoustic decisions, production
forced alignment, Indonesian translation, strict subtitle QA, final mux, live
Drive byte/SHA-256 readback, and final delivery remain incomplete. The latest
probe Pod, `tccsb8991x84ua`, was externally verified `EXITED`. No stable tag may
be created from this partial run.

## Attempts after the original 23 - 2026-09-05 to 2026-09-06

| Log | Candidate | Observed stage timings and result |
|---|---|---|
| `run-20260905T235206.955372Z-1880.log` | `c4aec5c` | Download `13.7 s`; audio `5.2 s`; raw ASR `171.0 s`; TR pack `44.9 s`; TR return `3.1 s`; audio review `998/998` PASS in `719.2 s`; forced alignment FAIL in `380.7 s` on same/unknown-speaker overlap. Diagnostics retained; Pod `uziszfw1mrsj03` shutdown verified in `7.6 s`. |
| `run-20260906T002125.836371Z-14208.log` | `9e17b40` | Download `8.0 s`; audio `3.3 s`; raw ASR `104.0 s`; TR pack `22.4 s`; TR return `2.9 s`; checkpoint audio review `998/998` PASS in `9.8 s`. Forced alignment was manually interrupted at about `781 s` after an unbounded candidate-combination search was identified. Pod `h9wmdsf5fl7o5d` was externally stopped. |
| `run-20260906T004148.201847Z-1396.log` through `run-20260906T004805.705093Z-9416.log` | startup/adoption | Several starts reached desired `RUNNING` without usable SSH/machine readiness. They were interrupted after bounded waits and externally stopped. Capacity migration preserved the volume and replaced `h9wmdsf5fl7o5d` with `9aret9grguxm1i` at the recorded `$0.74/hour` rate. |
| `run-20260906T004815.388366Z-6628.log` | `33637d3` | Download `11.5 s`; audio `5.2 s`; raw ASR `175.2 s`; TR pack `36.9 s`; TR return `3.1 s`; checkpoint audio review `998/998` PASS in `18.5 s`; forced alignment FAIL in `755.8 s` on one remaining conflict: `MA11-TR-a09b20542760d351` and `MA11-TR-1144d768dc92c4a9`. Diagnostics retained; Pod `9aret9grguxm1i` shutdown verified in `7.3 s`. |
| `run-20260906T013246.588786Z-13720.log` | resume attempt | Transfer/network work retried and the controller migrated to `781ct55zv4gkle`. The user stopped further spending before stage completion. Shutdown was externally verified in `7.3 s`; subsequent API observation at `2026-09-06T01:35:52.8724964Z` showed desired state `EXITED`. |

The user reports nearly `21 hours` total elapsed and approximately `$20` of RunPod spend. Those are user-reported values, not provider billing measurements. The table records controller and stage timings only and must not be used as exact billing time.

Code changes across this interval:

- `c4aec5c`: resolve acoustically duplicated review fragments.
- `667799f`: resolve forced-alignment overlap components.
- `9e17b40`: deterministic overlap selection.
- `9e4ae5b`: bound overlap candidate search by the candidate time envelope.
- `33637d3`: avoid redundant narrowed/partition candidates when earlier selection already resolves a component.

Current local return artifact:

- Filename: `Muhtemel Ask 11.Bolum_TR_TEXT_CORRECTED.zip`
- Pending exact audio-review UIDs: `MA11-TR-a09b20542760d351`, `MA11-TR-1144d768dc92c4a9`
- Manifest SHA-256: `195f7d1509cb3592fb6ed207d1c6b088930a02006d4cfc338a3ec3f92cc6c9bf`
- ZIP SHA-256: `DA278A26B22BC507D40577E2658F7E9A48374154BD978B940125AD1CB1C7BC2A`
- Remote upload and checkpoint consumption: NOT VERIFIED

Current boundary: no Turkish SRT, Indonesian handoff/return/SRT, strict mux, final QA, Drive readback, or delivery exists. Pod `781ct55zv4gkle` is externally observed `EXITED`; retained volume `xgogcmey5o` remains. Paid compute must not be restarted until explicit user authorization. The eventual resume command is `.\mas.ps1 run 11`.
