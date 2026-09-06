# Episode 11 Sol pilot execution

## Scope and authority

Current execution scope is the Episode 11 cue pilot. The user authorized spending from the existing RunPod balance while retaining USD 1.00 and authorized creating necessary access keys. Episode 12 is conditional on pilot acceptance, not authorized to start yet. GPT API calls, top-ups, volume deletion and automatic acceptance of gated-model terms remain excluded. Existing deliverables must be preserved before any later publication. Creating an S3 key through the provider UI exposes an all-volume write-capable credential and remains pending its separate final-action confirmation; it is no longer necessary for the reacquired pilot source.

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

## Historical readiness decision before source reacquisition

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

Focused local validation after guardian repairs: `43 passed in 5.45 seconds` across `tests/test_pilot_guardian.py`, `tests/test_pilot_runner.py` and `tests/test_subtitle_pilot.py`. The timeout path terminates the exact worker process tree on Windows and POSIX, artifact SHA-256 reading checks its deadline between 1 MiB chunks, and a failed external shutdown always raises even when inference had already failed. A separate guardian now validates exact Pod/budget state, persists a ready handshake before startup, inherits credentials without placing them in arguments or files, overrides stale inherited Pod identity, triggers bounded stop on parent death, absolute deadline, cost guard or post-ready monitoring failure, and cannot honor disarm before external `EXITED` confirmation. Python compilation and `git diff --check` passed. These results do not include CUDA, real provider lifecycle or email-delivery execution.

At `2026-09-06T03:17:15.1826226Z`, a second read-only provider query returned balance USD 3.749485359, account spend USD 0.005/hour, auto-pay false, and the retained Pod still `EXITED` at USD 0.74/hour. If every readiness prerequisite were already satisfied, the planning scaffold would reserve 550 seconds and derive USD 0.1138194444 at the combined USD 0.745/hour rate, while preserving a USD 1.10 balance floor including the billing-delay margin. This is not permission to start: source, model and image readiness remain unproven, and the observed balance decrease since the earlier query demonstrates that displayed current hourly spend is not a complete real-time billing ledger.

The modules remain locally verified execution scaffolding, not a completed real-provider qualification. The guardian protects controller-process death and lease expiry, but cannot promise recovery from machine power loss or a provider/network outage. Do not start compute until source, model and image readiness pass and the real provider preflight fits the existing balance-minus-USD-1 authority.

## Superseded initial lease prerequisites

1. After the pending final-action confirmation, create the authorized RunPod S3 credential and configure it in a secure local credential store or environment variables without committing or printing it. Use access only to inventory the retained volume and retrieve the required Episode 11 source/model evidence without starting a GPU; do not write or delete volume objects.
2. Retrieve `prepare/audio.flac` read-only and verify its retained byte count and SHA-256. Convert it deterministically to a PCM WAV without replacing or modifying the FLAC. Record the converter/version, command parameters, WAV format, byte count and SHA-256, and bind every clip to the WAV by exact frame offset and PCM-byte equality. Do not use a correction-pack hash as the source-audio hash.
3. Select contextual Episode 11 clips that include normal dialogue, rapid turns, music and overlap, including the final conflict pair when its complete neighborhood is available. Record source-relative offsets, durations and SHA-256 values. Keep the four existing 10,800-millisecond historical clips classified only as smoke samples.
4. The user must explicitly complete any required pyannote Community-1 access request and accept its model terms. Then provision a complete offline snapshot, including every externally referenced model in its configuration, and record content hashes. Automation must not accept gated terms or download a model during a billed pilot lease.
5. Provision and verify an image containing `requirements-pilot.lock`, ffmpeg/ffprobe, CUDA-compatible torch, stable-ts, faster-whisper and the selected diarization backend. Local ffmpeg/ffprobe installation is verified as BtbN `n9.0.1-26-g5c8e7e2433-20260905`; the GPU image and CUDA imports remain unverified.
6. Re-query balance and Pod rate immediately before start. Start only when the exact computed lease, billing-delay margin and shutdown reserve preserve at least USD 1.00. No additional pilot spending approval is required when that calculation fits the existing authority.

## Current local pilot evidence, 2026-09-06

The source-access blockers above are historical. `docs/WORKFLOW_REPAIR_PROGRESS.md` records the free official-source reacquisition, immutable WAV lineage and five exact-offset contextual clips totaling 155000 milliseconds. They include the final conflict neighborhood. This new source does not replace or inherit the canonical remote source hash. An S3 key or gated pyannote model is not a prerequisite for evaluating the alternative below.

NVIDIA's public Sortformer v2 model is CC-BY-4.0. The official NeMo-Speech.cpp v0.1.0 native runtime supports standalone CPU diarization without PyTorch. This is an explicitly selected new pilot stage, not a CPU fallback for GPU-required ASR or alignment. Official references: [runtime release](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.1.0), [model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/blob/5240a64075176943f677d30fa2171c780229f341/README.md).

- Runtime archive: `nemo-speech-0.1.0-windows-x86_64-cpu.zip`, SHA-256 `5e4ea81046012edcd77fd8848de8eefb5a4ba38cc26f52eb544ab184695a75d6`, matched the published release checksum.
- GGUF revision: `5240a64075176943f677d30fa2171c780229f341`, SHA-256 `0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a`, 147075776 bytes, matched Hugging Face LFS metadata.
- Native doctor detected Intel N150 CPU and 15.8 GiB RAM. No paid compute was started.
- Sol's bounded CLI test processed `final-conflict-context.wav` (30 seconds) in 18.89 seconds, exit code 0. This is one measured clip, not a whole-episode speed guarantee.
- Output retained six segments, including two overlaps of 0.308 seconds. This proves nonexclusive output representation, not that speaker labels are correct.
- CLI JSON does not include raw frame probabilities. A separate C ABI exporter is being evaluated before acoustic evidence can be used for cue decisions.

Decision remains REVIEW_REQUIRED. Neither Turkish/Indonesian final SRT nor acoustic acceptance is established. Re-query live billing immediately before any bounded GPU qualification; do not make image compatibility proof depend circularly on an already completed GPU run. Any bootstrap qualification must have its own finite lease and shutdown guard.
