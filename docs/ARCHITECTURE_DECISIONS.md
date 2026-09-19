# Architecture decisions

## Production flows

```text
EP15+ semantic:
source -> GPU ASR -> immutable word timeline -> semantic handoff when needed
       -> Indonesian handoff -> QA -> MP4 -> Drive byte/SHA-256 readback

Explicit strict:
source -> GPU ASR -> Turkish correction -> acoustic alignment
       -> Indonesian handoff -> strict QA -> MP4 -> Drive readback
```

EP14 also retains its separate delivery-first route. A persisted episode policy
is never silently changed.

## Operator boundary

The public entrypoint is `./mas`. Normal status is concise; `status EPISODE
--json` exposes technical state. Operators exchange ZIP files only through
`translation_input/` and `translation_output/` and collect results from `final/`.
Other episode directories are internal resume evidence.

One current local run log lives at `.mas/run.log`. Remote output stays with its
remote job. Locks are transient process coordination, not episode history.

## Safety decisions

1. Stage resume is bound to its exact source, inputs, configuration and outputs.
   `state.json` is status, not reuse authority.
2. Source media remains immutable after its byte count and SHA-256 are recorded.
3. ASR and acoustic alignment require GPU and fail closed without CUDA.
4. ChatGPT may change only the authorized text or word-span fields. Python owns
   validation, timing derivation, block IDs and release QA.
5. Different known speakers may overlap as separate cues. Same-speaker overlap
   fails. Unknown-speaker overlap remains review evidence.
6. Semantic, strict, delivery-first and emergency outputs have separate
   identities and cannot satisfy each other's PASS markers.
7. Network work has finite retries, timeouts and no-progress limits.
8. Drive publication uses a temporary object and passes only after exact remote
   byte count and SHA-256 readback. Existing exact-name objects are preserved.
9. Temporary compute is released on every exit path and shutdown is checked by
   an observer outside the Pod.
10. The episode wall budget persists across controller retries. A human handoff
    pause or explicit extension cannot weaken QA or manufacture PASS.

## Policy details

New EP15+ states use `semantic-block-v1`. Its immutable ASR word timeline owns
timing. A semantic return cannot provide timestamps or use unowned word IDs.
First-hour and whole-episode delivery select from the same final block identity.

`strict-ctc-v1` remains available only when explicitly selected. EP14
delivery-first may retain usable source cue timing after bounded CTC failure and
reports quality warnings honestly. It still requires a real Indonesian return
and verified Drive delivery.

Names containing `v2` in schemas and stored artifacts are compatibility IDs,
not a second product.
