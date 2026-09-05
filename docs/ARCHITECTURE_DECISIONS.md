# Architecture decisions

## ADR-001: One production CLI

Production runs as normal Python modules through `./mas`. Notebooks are legacy references and are not execution dependencies.

## ADR-002: Resume is hash-bound

Each completed stage is persisted atomically and bound to its inputs, configuration, code identity, outputs, and completed UIDs. Resume may reuse a stage only when those bindings still match. A changed source or returned ZIP invalidates dependent work instead of silently rebinding it.

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

Drive is not a workspace or repository mirror. The strict publisher uploads only the final MKV, Turkish SRT, and Indonesian SRT files under `EPISODES/Muhtemel Ask N.Bolum/`. Source media, code, logs, reports, credentials, and intermediate artifacts remain outside Drive.

## ADR-009: Network work is bounded

Network operations use finite attempts, connection timeout, no-progress timeout, and total retry budget from `config/runtime_policy.json`. Authentication, schema, episode, and hash failures are not transient and are not retried.

## ADR-010: Compute shutdown needs two observers

The in-pod controller requests `POST https://rest.runpod.io/v1/pods/{podId}/stop` with a bearer secret after durable checkpoint or delivery handling. The request has bounded connection, request, and retry times. An external observer must then query `GET https://rest.runpod.io/v1/pods/{podId}` and confirm `desiredStatus` is `EXITED` and billing shows no active compute. A successful stop response alone is not proof of shutdown. Network-volume Pods may require termination rather than stop.

## ADR-011: Cost control is fail-safe

`runpod/run-episode.sh` imposes a 14,400-second maximum run and a 1,800-second no-log-progress limit unless explicitly overridden. It terminates the process group, preserves already atomic checkpoints, and requests provider-side stop. Storage cost is reported separately from GPU compute cost.

## Provider references

- [RunPod Stop a Pod REST API](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop)
- [RunPod Find a Pod by ID REST API](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
- [RunPod Pod management and storage behavior](https://docs.runpod.io/pods/manage-pods)
