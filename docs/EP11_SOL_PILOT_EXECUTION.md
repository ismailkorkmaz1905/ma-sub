# Episode 11 Sol pilot execution

## Scope and authority

Episode 11 cue pilot only. The user authorized spending from the existing RunPod balance while retaining USD 1.00. Full Episode 11 execution, Episode 12, GPT API calls, Drive writes, top-ups, migration, volume deletion and accepting gated-model terms remain prohibited.

## Read-only readiness record

At `2026-09-06T03:04:21.7166936Z`, RunPod reported:

| Field | Value |
|---|---:|
| Account balance | USD 3.7543464701 |
| Current account spend | USD 0.005/hour |
| Auto-pay | false |
| Pod | 781ct55zv4gkle |
| Desired state | EXITED |
| Pod rate | USD 0.74/hour |
| Image | runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 |
| Network volume | xgogcmey5o, 50 GB, EU-RO-1 |

The process environment contained a stale Pod ID whose REST lookup returned HTTP 404. The Windows User environment identified the retained EXITED Pod above. No secret value was printed.

The mathematical balance above the required reserve was USD 2.7543464701 at the observation time. This is not an operational lease allowance because it excludes billing delay, continuing USD 0.005/hour account spend and shutdown reserve.

## Readiness decision

Status: BLOCKED before paid startup.

- Local storage has four validated historical issue-category WAV excerpts totaling 10,800 milliseconds. They do not cover the final conflict pair or full-scene diarization acceptance.
- Production source audio is `prepare/audio.flac`, while the pilot requires a source WAV. No local Episode 11 source audio exists, and no verified FLAC-to-WAV conversion and lineage artifact exists, so the required source-hash and exact-offset PCM verification cannot pass.
- RunPod S3 credentials were not found in the checked environment configuration. Other credential stores were not exhaustively inventoried. No usable non-compute access to the retained volume was established, and no new S3 key was created.
- No retained evidence proves that the volume contains the required stable-ts/faster-whisper and pyannote Community-1 offline model directories. Community-1 gated terms were not accepted.
- The retained Pod uses a generic image. The optional pilot dependency lock is not installed by the production bootstrap or image, and no ready pilot image has been verified.
- The existing `subtitle-pilot` command is a local 180-second worker, not a provider lease manager or dollar guard. The generic full-run controller is unsuitable for this pilot and its shell heartbeat does not establish output progress.

No Pod start, stop, migration, API inference, Drive operation, model download, gated-term acceptance or volume mutation was performed. New pilot compute spend initiated in this execution: USD 0.00.

## Isolated runner implementation

`src/mas/subtitle/pilot_runner.py` now provides local-only readiness primitives for a future approved lease:

- conservative balance calculation with a USD 1.00 floor, USD 0.10 billing-delay margin and combined account/Pod hourly rate;
- separate 180-second startup, 180-second inference and 120-second shutdown allocations;
- deterministic FLAC-to-16 kHz mono PCM WAV conversion with immutable input recheck and a checksum-bound provenance receipt;
- measured progress based only on increased completed-unit or artifact-byte counters;
- stage START/PASS/FAIL notification calls and atomic event records;
- artifact byte-count/SHA-256 snapshots;
- stop in `finally`, including an ambiguous start response, followed by external `EXITED` polling and a checksum-bound final Pod/account receipt.

Focused local validation after runner cleanup repairs: `36 passed in 3.77 seconds` across `tests/test_pilot_runner.py` and `tests/test_subtitle_pilot.py`. The timeout path now terminates the exact worker process tree on Windows and POSIX, artifact SHA-256 reading checks its deadline between 1 MiB chunks, and a failed external shutdown always raises even when inference had already failed. Python compilation and `git diff --check` passed. These results do not include CUDA, provider lifecycle or email-delivery execution.

At `2026-09-06T03:17:15.1826226Z`, a second read-only provider query returned balance USD 3.749485359, account spend USD 0.005/hour, auto-pay false, and the retained Pod still `EXITED` at USD 0.74/hour. If every readiness prerequisite were already satisfied, the planning scaffold would reserve 550 seconds and derive USD 0.1138194444 at the combined USD 0.745/hour rate, while preserving a USD 1.10 balance floor including the billing-delay margin. This is not permission to start: source, model and image readiness remain unproven, and the observed balance decrease since the earlier query demonstrates that displayed current hourly spend is not a complete real-time billing ledger.

The module remains a planning and local enforcement scaffold, not an approved paid runner. It bounds the child inference process, synchronous mail waits and provider HTTP calls, but no independent external lease watchdog has yet been proven to stop the Pod if the controller process itself hangs or terminates. Do not use it to start compute.

## Required lease plan before a future start

1. Configure existing RunPod S3 credentials in a secure local credential store or environment variables without committing or printing them. Use read-only S3 access to inventory the retained volume and retrieve only the required Episode 11 source/model evidence without starting a GPU. Do not create credentials unless the user separately authorizes it.
2. Retrieve `prepare/audio.flac` read-only and verify its retained byte count and SHA-256. Convert it deterministically to a PCM WAV without replacing or modifying the FLAC. Record the converter/version, command parameters, WAV format, byte count and SHA-256, and bind every clip to the WAV by exact frame offset and PCM-byte equality. Do not use a correction-pack hash as the source-audio hash.
3. Select contextual Episode 11 clips that include normal dialogue, rapid turns, music and overlap, including the final conflict pair when its complete neighborhood is available. Record source-relative offsets, durations and SHA-256 values. Keep the four existing 10,800-millisecond historical clips classified only as smoke samples.
4. The user must explicitly complete any required pyannote Community-1 access request and accept its model terms. Then provision a complete offline snapshot, including every externally referenced model in its configuration, and record content hashes. Automation must not accept gated terms or download a model during a billed pilot lease.
5. Provision and verify an image containing `requirements-pilot.lock`, ffmpeg/ffprobe, CUDA-compatible torch, stable-ts, faster-whisper and pyannote. Verify CUDA imports and model loading against the exact offline snapshots before treating it as ready; the generic retained Pod image is not this evidence.
6. Add an isolated controller that re-queries balance immediately before start, refuses a projected balance below USD 1.00 plus billing/shutdown margin, bounds startup and execution with a real no-output-progress watchdog, retrieves hash-bound artifacts, stops on every exit path and externally polls `EXITED`.
7. Submit the exact maximum lease seconds and conservative USD charge derived from the live Pod rate and reserves for approval before startup.
