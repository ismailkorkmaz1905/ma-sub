# Episode 13 code preparation, 2026-09-07

Subsequent user-authorized paid technical testing and fixes are recorded in
[the RunPod smoke report](RUNPOD_SMOKE_2026-09-07.md). The no-paid-run statements
below describe the initial preparation, before that separate authorization.

The user confirmed Episode 13 has not been published. This work changes the
production code, not the episode's acceptance state. Initial HEAD: `192a19b`.
Episode 12 MP4, archived media, ignored evidence and recorded source hashes
remain unchanged. No paid Pod, episode download, long local encode or Drive
publication was started for this preparation.

## Production behavior

- `./mas run N` checks the exact official channel source before allocating a
  temporary Pod. It saves the official title and URL together. Main Drive MP4
  naming uses that title; strict receipts and review samples remain separate.
- The protected Pod `781ct55zv4gkle` is not restarted or deleted. Capacity uses
  uniquely named temporary Pods on retained volume `xgogcmey5o`, EU-RO-1.
  Fresh allowed offers prefer L4, then RTX 4000 Ada. Explicit no-capacity replies
  may advance to the next offer. Ambiguous create responses reconcile the unique
  name without issuing another create. A failed readiness candidate must be
  externally ABSENT before trying another. Capacity availability is not guaranteed.
- A local kernel lock and a remote Linux episode lock prevent duplicate work.
  SSH retains `-n -T`. The detached supervisor starts the long job once; the
  controller reads bounded status/log/checkpoint requests. A lost response never
  repeats the long start command. PID start times, input hashes and Git commit
  identify the running job. Stale state is not progress. Bootstrap is attempted
  once per lease, with bounded installation commands.
- The supervisor bounds runtime and terminates separate descendant process
  groups. Provider `terminateAfter` supplies an additional deadline. Controller
  lease cleanup deletes only proven owned Pods and verifies external absence.
  An unresolved ownership journal fails closed. Resume requires the original
  commit/source and journal; it observes the old job without redeploying it.
- ASR JSON snapshots are copied to local content-addressed storage as they appear.
  Source, handoff packs and exported files are checked by bytes and SHA-256.
  Existing conflicting local evidence is retained. Local stage state is mirrored
  while prior snapshots remain available.
- Exit 20 waits for Turkish correction; exit 21 waits for Indonesian translation;
  exit 23 waits for MP4 sample approval. These successful handoffs close the Pod.
  Only a downloaded handoff plus checksum-bound external shutdown evidence can
  pause the episode clock. Human waiting time is excluded on resume, without
  resetting prior active time. Crashes and unverified shutdown do not pause it.

## MP4 target and sample gates

Default target is **3,000,000,000 bytes**, not a hard ceiling. The video bitrate
plan derives from source duration, subtracting the AAC 192,000 bit/second budget.
The production encoder uses H.264 NVENC p4/VBR with the derived bitrate
instead of `-b:v 0`. The initial CQ 19 default was removed after the real smoke
showed it overrode the size objective; explicit quality overrides remain
available and require fresh reviewed samples. There is no file-size truncation or maximum-size acceptance
rule. Resolution is preserved; the code does not upscale a lower-resolution
source or silently switch GPU production to a CPU encoder.

On each new Pod, three short actual source scenes qualify the encoder before
audio/ASR work. Their synthetic test subtitle is explicitly technical evidence,
not a translation. GPU UUID/driver, FFmpeg version, input/settings hashes and
measured encode time are recorded. Projected full encode time is an estimate,
not a promise; encoding alone exceeding the allowance blocks further work.

After strict MKV/SRT finalization, three real subtitle samples are produced for
review. Empty fixed windows move to a nearby real cue and record actual source
ranges. Their measured size and time project full-file size and encode duration.
Quality must be assessed before selecting final settings; being close to 3 GB
does not justify visible damage. `MAS_MP4_TARGET_GB` and the allowlisted JSON
array `MAS_MP4_ENCODER_OPTIONS` can change the plan. Changed settings require
new bound samples and approval. Other codecs require a separately implemented
and tested encoder path; this change does not claim AV1 output support.

The operator's `review/mp4-sample-approval.json` requires:

- `approved: true`, after actual sample review;
- `encoding_settings_identity_sha256`, `source_sha256`,
  `source_duration_seconds`, `id_srt_sha256`, and exact `style` from the manifest;
- episode-relative `sample_manifest_path` under `work/encoding-samples/`, and
  `sample_manifest_sha256`.

The manifest and all three sample MP4/SRT byte/hash records are rechecked.
Approval is not generated automatically. Encoding receipts retain
`perceptual_acceptance: NOT_ASSERTED`; transport success is not perceptual or
audio acceptance. These sample checks do not replace acoustic subtitle review.

## Storage and publication

Volume availability is its provider allocation minus actual tree usage, not
`df` alone. Installer cache cleanup uses the supported
[`uv cache clean`](https://docs.astral.sh/uv/concepts/cache/#clearing-the-cache)
command after installation. Model directories and the environment are retained.
MP4 planning checks free space with headroom; the soft target still cannot
guarantee final size. If volume headroom is insufficient, the MP4 uses the exact
ephemeral path `/tmp/mas-epN-output/` and is copied locally with full verification.
For that case, the explicitly requested Drive readback occurs before Pod
deletion, bounded by the remaining paid allowance. Otherwise the Pod is deleted
before local Drive upload. A failed upload keeps local evidence; a transfer-only
retry does not acquire another GPU.

Existing same-name Drive objects are inventoried and fully read. Differing
objects are moved to unique retained paths and read back before final publication.
Duplicate names or preservation mismatch block publication. Upload uses unique
partial objects, immutable promotion and an actual full byte/SHA-256 readback.
Source, strict input packs, both SRTs, MKV, MP4, encoder receipt and approved
samples remain locally bound. SMTP permanent rejection remains `blocked` with
no cyclic retry.

## Live cost prerequisite

Before spending, the code refreshes account balance/spend, Pod inventory, volume
identity/allocation and GPU offers. The user's 2026-09-10 instruction removed the
1 USD balance floor and additional billing margin; the controller still reserves
120 seconds of lease time for shutdown. Affordability uses the allowed maximum
GPU rate and storage, so a more expensive allocation cannot reuse a cheaper
offer's duration. Auto-pay must be disabled. No top-up is performed.

The documented provider interface used here does not supply a verified live
storage unit-price field. Do not invent one or reuse a historic fixed default.
Before each real run, the operator/agent must inspect the official
[storage prices](https://docs.runpod.io/pods/storage/types) and save a local
`MAS_RUNPOD_STORAGE_QUOTE` JSON file. Its wrapper is `{data, sha256}`, where
`sha256` is `mas.reliability.digest(data)`. Data contains exactly:

- `observed_at_utc`, a timezone-qualified observation within 24 hours;
- `source_url: https://docs.runpod.io/pods/storage/types`;
- `network_volume_usd_per_gb_month` and `container_storage_usd_per_gb_month`,
  positive observed USD/GB/month values.

Missing, stale or altered quotes block acquisition. This file is operator
evidence, not a committed current-price configuration. Quotes and affordability
are recorded in the capacity audit. Actual run duration/cost still require the
new episode's measured scenes. The 2-4 hour and 1-2 USD planning aims are not SLAs.

For a public official source, cookies are optional. When a cookie file is
configured it retains the existing Netscape validation and verified transfer.
When it is absent, source discovery and download remain bounded and fail closed
if YouTube actually requires authentication; no empty or fabricated cookie file
is substituted.

## Validation and remaining work

Final local full suite: 813 passed, 11 skipped in 112.81 seconds.
Local targeted controller/capacity/encoder/delivery tests passed. Linux-only
process/lock tests require Linux CI; WSL is not installed here. Dockerfile and
RunPod scripts were inspected. No real GPU throughput, live Drive readback,
new provider observation or perceptual PASS is claimed for this code preparation.
Current account balance for this preparation is UNKNOWN, not a reused old value.

A mocked Drive preservation failure test initially reached configured Gmail:
its start/failure notices were SMTP-accepted. They were test notifications, not
an Episode 13 run or a Drive write. The test module now disables delivery
notifications automatically. Local incident evidence is retained in
`var/ep13-code-preparation-notification-incident.json`; final test output is in
`var/ep13-code-preparation-final-tests.log`.

All eleven Episode 13 stages remain NOT_STARTED: download, audio, raw_asr,
tr_pack, tr_return, audio_review, forced_alignment, id_pack, id_return, finalize,
drive_readback. The official unpublished source is the prerequisite. The next
run must refresh external state/prices, execute the real short qualification,
then follow the strict handoffs. Episode 14 must not start first.
