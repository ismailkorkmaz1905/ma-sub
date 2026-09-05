# Episode 11 first production run record

## Result

Episode 11 reached the Turkish correction handoff on 2026-09-05. This is not a
completed episode result. Real RTX 4090 inference, source acquisition, audio
extraction, raw ASR, correction-pack generation, controller shutdown, and
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

The 15 failed controller invocations totalled 2,313.376 s, or 38 min 33.376 s.
All 16 controller invocations totalled 3,481.407 s, or 58 min 1.407 s. The
wall-clock interval from the first start at 07:02:35.297233 UTC to the handoff at
09:09:24.143537 UTC was 7,608.846 s, or 2 h 6 min 48.846 s. These controller
durations include local checks, setup, shutdown, and stopped intervals. They are
not RunPod billing measurements.

## Successful handoff measurements

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

## Current boundary

The Turkish correction pack is local and the Pod is stopped. Turkish correction,
acoustic review, forced alignment, Indonesian translation, strict subtitle QA,
final mux, live Drive byte/SHA-256 readback, and final delivery remain incomplete.
No stable tag may be created from this partial run.
