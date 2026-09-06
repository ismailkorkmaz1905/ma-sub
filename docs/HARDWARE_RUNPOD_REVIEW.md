# Local hardware and RunPod review - 2026-09-06

Budget override: references below to the original USD 0.50 allowance are historical. The user now authorizes existing RunPod balance minus a USD 1.00 reserve; see [current observed balance and limits](EP11_BOUNDED_SAMPLE.md). This does not authorize Episode 12, account top-ups or volume deletion.

## Read-only observations

Windows CIM reported Intel N150, four cores/four logical processors, 16,941,326,336 bytes of RAM and Intel Graphics. No NVIDIA device or `nvidia-smi` was found. The C volume reported 416,234,831,872 free bytes. The project environment has Python 3.11; torch, ffmpeg and ffprobe are unavailable. Docker is unavailable and WSL reports not installed. No installation was attempted.

This computer can host orchestration, hashes, pack handling and non-GPU validation. Actual media-processing throughput is not benchmarked. It cannot execute the current CUDA-required ASR/diarization stages as configured.

Live RunPod REST observation at `2026-09-06T02:44:13.446550+00:00`:

| Field | Observed value |
|---|---|
| Pod | 781ct55zv4gkle |
| Desired state | EXITED |
| Recorded hourly rate | USD 0.74/hour |
| Image | runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 |
| Container disk | 30 GB |
| Retained network volume | xgogcmey5o, 50 GB, EU-RO-1 |

The actual Pod uses a generic image, while the repository Dockerfile installs CUDA 12.8.1 runtime dependencies, ffmpeg and pinned Python packages. This explains an opportunity to remove repeated bootstrap work; it does not prove the historical environment was incapable of successful inference. No pilot image was built locally.

Historical controller logs contain 63 invocations. This is not a verified count of provider start/stop operations. The user's approximate USD 20 cost remains user-reported, not an invoice measurement.

## Workflow placement, not just alignment

The current production workflow enters paid remote execution for source download, extraction and ASR, waits for correction, returns for review/alignment and translation preparation, then returns again for finalization/publication. Non-GPU work and operator waits need not require repeated expensive provisioning.

Target placement, not yet implemented:

| Phase | Placement | Required evidence |
|---|---|---|
| Source authentication, download, hashes and sample selection | Local or storage access without a GPU | Valid source and complete input inventory |
| ASR and regular diarization | One pre-provisioned bounded GPU lease | Persisted raw results, actual model identity and externally observed shutdown |
| Cue construction, corrections, translation handoffs and QA | Local | Source lineage, text authority and complete issue report |
| Necessary acoustic rework only | One separately approved bounded sample batch | Aggregate failures, no repeated whole-episode retries |
| Mux and publication | Local after dependencies and authorization | Real file checks and remote SHA-256 readback |

The four-hour wall target includes human handoff delays under the existing deadline policy. It is not a demonstrated unattended service-level guarantee.

## Avoid opening a GPU to retrieve files

[RunPod's S3 API](https://docs.runpod.io/storage/s3-api) supports network-volume access without a running Pod, lists EU-RO-1 at `https://s3api-eu-ro-1.runpod.io/`, and states no additional S3 API charge. Retained storage charges continue. Its S3 credentials are distinct from the normal RunPod API key.

Checked Process/User environment scopes did not contain AWS or RunPod S3 credential variables. Other credential stores were not inventoried. No credentials were displayed, key created, volume changed or file fetched through S3.

[Network-volume documentation](https://docs.runpod.io/storage/network-volumes) describes locality constraints and storage pricing. An existing volume constrains the GPU comparison to available compatible capacity unless a separately authorized transfer is justified. File deletion does not shrink the allocation.

## Cheaper GPU test gate

Read-only GraphQL observation at `2026-09-06T02:56:11.1779681Z`, filtered to Secure Cloud, EU-RO-1 and one GPU, returned RTX A4500 (20 GB) at USD 0.25/hour with Low stock, versus RTX 4090 (24 GB) at USD 0.74/hour with High stock. `availableGpuCounts` was null, so this is a quoted candidate, not a confirmed allocation. Query shape follows [official GPU availability documentation](https://docs.runpod.io/sdks/graphql/manage-pods).

No cheaper GPU has yet been selected or benchmarked. At those quoted GPU rates alone, A4500 could take up to 2.96 times the 4090 lease duration before losing its compute-cost advantage (0.74 / 0.25); this is arithmetic, not a speed prediction, and excludes storage and other charges. Query current compatible capacity and prices again before a decision. Compare identical contextual audio, model versions, FP16 settings and output checks. Report both inference seconds and total billed-lease seconds, including startup, model setup, transfer and shutdown. Lower hourly price alone is not lower cost or equal speed.

Before spending: verify a ready image and complete model cache without a paid startup, reserve shutdown time within the combined USD 0.50 allowance, and provide an external stop observer. If cold setup plus two comparison leases cannot fit, report BLOCKED rather than silently extending the allowance. Do not reset Episode 11's historical deadline or run Episode 12.

[Pod management documentation](https://docs.runpod.io/pods/manage-pods) describes container/storage lifecycle. Persist models and checkpoints on retained storage, and bake reusable dependencies into a validated image. These are prerequisites, not claims that the current controller has already been migrated.
