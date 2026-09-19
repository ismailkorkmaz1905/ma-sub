# Architecture decisions

## Current production flows

```text
EP15+ semantic:
source -> audio -> GPU raw ASR -> immutable word timeline
       -> semantic handoff -> Indonesian translation -> semantic finalize
       -> burn_mp4 -> verified GPU release -> Drive byte/SHA-256 readback

Explicit legacy strict:
source -> audio -> GPU raw ASR -> Turkish correction -> acoustic review
       -> GPU forced alignment -> Indonesian translation -> strict finalize
       -> burn_mp4 -> verified GPU release -> Drive byte/SHA-256 readback
```

These diagrams describe code boundaries, not proof that a real EP15 GPU run,
Drive publication, perceptual review, or provider shutdown has passed.

## ADR-001: One production CLI

Production runs as normal Python modules through `./mas`. Notebooks are legacy references and are not execution dependencies.

## ADR-002: Resume is hash-bound

Resume is stage-specific; `state.json` is telemetry, not reuse authority. Download,
audio, raw ASR, forced alignment, acoustic review, strict finalization and MP4
encoding each use their own validators and bindings. Turkish and Indonesian return
ZIPs are always revalidated. A changed source, return ZIP, configuration, code
identity or output is rejected instead of being silently rebound.

A completed acoustic review can hydrate only when the final Turkish ZIP and its
report still validate against the exact provisional pack, current review input,
configuration, implementation hash and manual-override hash. Forced alignment
writes `forced_alignment_v2.done.json` only after the complete current
audio/text/VAD/code binding and aligned output validate; an older alignment may
receive this marker only through that same validation path.

Strict subtitle finalization and burned MP4 encoding are separate stages.
`strict_finalize.done.json` binds the canonical evidence and configuration files,
the relevant producer files, the PASS report, softsub MKV and both SRT files by
byte count and SHA-256. `burn_mp4` separately verifies the immutable source, ID
SRT, style, encoder settings, sample approval when required, output bytes and
encoding receipt. A legacy finalization report without the new marker is not a
resume checkpoint.

## ADR-003: Source media is immutable

The source remains in the episode `source/` directory. Its SHA-256 is recorded and checked before dependent stages. Production must not rename, move, rewrite, or substitute it after initialization.

## ADR-004: GPU requirements fail closed

ASR, acoustic review, and forced alignment require CUDA. Production passes CPU fallback as disabled. Missing GPU, incompatible CUDA, unavailable model, or failed preflight blocks the stage before expensive processing.

## ADR-005: Text and timing have separate authority

Turkish correction may change only approved text fields. Indonesian translation may change only Indonesian text fields. Timing overrides may change only timing and require the expected UID, input digest, old text, and old timing. `replacement_text` in a timing override is invalid.

## ADR-006: Overlap is speaker-aware

Words remain monotonic within one utterance or speaker lane. Proven different speakers may overlap and must produce separate cues. Same-speaker overlap fails. Unknown-speaker overlap remains unresolved review evidence. No operation may merge text across speakers.

## ADR-007: Strict and emergency are separate products

Strict artifacts and markers live under `final/`. Emergency artifacts and markers live under `emergency/`. Emergency output cannot satisfy a strict gate, replace a strict filename, alter a strict marker, or be described as a formal release.

## ADR-008: Remote readback defines delivery

Final files publish through `rclone` or a provider API using a temporary name. PASS requires reading the remote bytes back and matching both exact byte count and SHA-256 before the exact final name is published. A mounted path, copy exit code, or metadata-only hash is insufficient.

Drive is not a workspace or repository mirror. The strict publisher uploads the
final MP4 with Indonesian subtitles burned into the image to the configured
delivery root under the validated official episode title. Source media, both
SRTs, the verified softsub MKV, code, logs, reports, credentials and intermediate
artifacts remain local evidence. Existing Drive files are retained until a
separately authorized cleanup.

`strict_finalize` verifies every source, correction, acoustic, translation,
timing and subtitle-QA gate before `burn_mp4` may run. Burned video necessarily
has different encoded video bytes: its separate receipt binds immutable source
and ID SRT hashes, style, encoder, output bytes/SHA-256, dimensions, duration and
H.264/AAC stream checks. This encoding check is not perceptual subtitle
acceptance. Default production encoding is NVENC; QSV or libx264 must be
explicitly selected with `MAS_MP4_ENCODER`. The Drive publish set is MP4 only.

For 8-bit AV1 sources with NVENC output, decode through `av1_cuvid` on CUDA, then transfer frames for libass and encode with NVENC. This path completed a real 60-second 1080p Episode 12 sample on L4; full-episode delivery evidence is recorded separately. Automated SSH commands use `-n -T` to avoid inheriting controller input or requesting a terminal. The previously stalled subtitle hash command returned normally in the subsequent real run. Historical failures do not independently prove the exact original cause.

## ADR-009: Network work is bounded

Network operations use finite attempts, connection timeout, no-progress timeout, and total retry budget from `config/runtime_policy.json`. Authentication, schema, episode, and hash failures are not transient and are not retried.

## ADR-010: Compute shutdown needs two observers

The external controller owns temporary compute. On exit 22 it first collects and
validates the local delivery, then leaves the capacity lease. Drive publication
starts only after checksum-wrapped capacity state and shutdown evidence prove
that every owned temporary Pod is `ABSENT`. A stop or terminate response alone is
not proof of shutdown. The retained protected Pod and network volume remain
separate resources with their own observation and retention rules.

If Drive publication fails after that release, the checksum-bound
`gpu-released-for-delivery.json` permits a transfer-only retry before RunPod,
Git, SSH, handoff or GPU preflight. The retry revalidates both the delivery and
the referenced capacity/shutdown evidence and cannot reacquire compute.

## ADR-011: Cost control is fail-safe

`runpod/run-episode.sh` imposes a 14,400-second maximum run and a 1,800-second no-log-progress limit unless explicitly overridden. It terminates the process group, preserves already atomic checkpoints, and requests provider-side stop. Storage cost is reported separately from GPU compute cost.

## ADR-012: Reviewed dialogue is not speaker identity

Audio review can establish that words were spoken, not that overlapping utterances came from different speakers. Synthetic `acoustic-overlap-*` lanes and their persisted provenance are rejected. Production speaker acquisition is still missing; tests with injected identities do not establish an end-to-end capability. See [the review](ASTRA_REVIEW_2026-09-06.md).

## ADR-013: Reuse model calls, not unbound completed alignments

Successful alignment calls are journaled under exact transcript/options, audio hash, actual model tensor/config/dictionary digest, model/version/device and implementation/dependency hashes. Reused raw outputs still pass current acoustic/timing validators. Completed output additionally binds correction inputs, VAD and relevant code; its output digest covers the actual-model fingerprint in provenance. Legacy output without a matching marker is preserved and rejected. Downloadable model revision pinning and safe raw-ASR legacy adoption remain unresolved release concerns.

## ADR-014: Episode deadline survives controller restarts

The controller's default 14,400-second wall deadline is anchored to the earliest retained episode start and a checksum-bound persisted origin. Restarting does not reset it. Precompute and network waits share the remaining allowance; bounded shutdown/diagnostic grace remains available after expiry. Extending the total allowance requires operator approval, not deletion of history. This is a wall-time guard, not a GPU billing measurement or a demonstrated four-hour performance result.

## ADR-015: Evaluate cue drafts without relabeling strict evidence

The user authorized a natural cue-based workflow pilot on Episode 11 before Episode 12. `subtitle-pilot` preserves estimated speech intervals and separate speaker text, persists raw ASR/diarization independently, and emits a review-required draft. It never represents estimated timestamps as verified CTC evidence or publishes into strict paths. Existing strict contracts remain unchanged pending measured acoustic review and explicit consumer migration. See [pilot scope](SUBTITLE_PILOT.md). No full-run or four-hour acceptance follows from synthetic tests.

## ADR-016: Exit 22 collection failure gets a new job identity

Exit 22 is terminal only after every exported file is collected. If an expected
external `/tmp/mas-epN-output/` MP4 or receipt disappears before collection, the
controller writes a checksum-bound failure record tied to the exact request,
base input identity, attempt, exit code and missing path. Only that matching
record advances the attempt identity for a new bounded remote job. This prevents
the old terminal job token from suppressing recovery and does not reset the
episode wall-time budget.

## ADR-017: Runtime reuse is not an immutable image

Bootstrap runtime reuse is guarded by an ABI marker covering the requirements
file, Python implementation/version/cache tag/SOABI/machine, uv, distro and
glibc, Torch/CUDA/cuDNN/C++ ABI, and FFmpeg/FFprobe path, version and SHA-256.
Strict doctor checks still run. This reduces repeated installation but does not
establish an immutable container image digest. `requirements.lock` also is not
yet demonstrated here as a complete transitive, hash-pinned dependency lock for
every Python package, native library, model and base-image component. Immutable
image identity and complete transitive hash locking remain open release risks.

## ADR-018: Semantic block alignment is the default EP15+ release policy

New Episode 15 and later run contracts default to `semantic-block-v1`, independent
of whether delivery scope is `first-hour` or `whole-episode`. An immutable,
hash-bound word timeline is the only timing authority. ChatGPT Pro receives a
human-mediated ZIP and may choose corrected Turkish text plus owned
`first_word_id`/`last_word_id` spans, but it may not supply timestamps. Python is
the safety authority: it validates pack identity and ownership, rejects timestamp
fields, derives display times, locks block UIDs, and applies coverage, overlap,
speaker, scene, gap, subtitle and translation QA. Semantic PASS is explicitly not
strict CTC PASS. Existing forced-alignment code remains available only through
explicit `strict-ctc-v1` or diagnostic use. Exit 29 releases GPU capacity before
manual handoff wait; semantic return and ID handoff resume locally. Existing EP14
and older states are never silently migrated.

## Provider references (historical)

- [RunPod Stop a Pod REST API](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop)
- [RunPod Find a Pod by ID REST API](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
- [RunPod Pod management and storage behavior](https://docs.runpod.io/pods/manage-pods)
