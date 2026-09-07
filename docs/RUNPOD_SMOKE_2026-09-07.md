# Authorized RunPod encoder smoke, 2026-09-07 UTC

Episode 13 is unpublished and was not started. These bounded technical tests
use the preserved Episode 12 source and review Indonesian SRT. They do not
replace the Episode 12 MP4, promote review evidence to strict acceptance, or
publish anything to Drive. All eleven Episode 13 stages remain NOT_STARTED:
download, audio, raw_asr, tr_pack, tr_return, audio_review, forced_alignment,
id_pack, id_return, finalize, drive_readback.

## Bugs found in real infrastructure

- Readiness REST calls lacked the application's User-Agent, while the capacity
  client already sent it. RunPod rejected the former with HTTP 403. Commit
  `89d6312` aligns both clients with `ma-sub-pilot/1.0`. A subsequent live
  protected-Pod GET succeeded. No credentials or browser identity were changed.
- Exact quota accounting repeated path metadata calls across a network volume.
  It exceeded the 60-second scan allowance. Commit `d1e1d97` uses cached
  `os.scandir` entries, retaining symlink exclusion, hardlink deduplication and
  permission failures. The bounded default is 180 seconds, with elapsed time
  recorded. The subsequent actual scan took 11.591 seconds; it counted
  34,652,817,433 bytes against 50,000,000,000 allocated bytes, leaving
  15,347,182,567 bytes. Host-wide `df` capacity was not used as volume quota.
- The initial 3 GB bitrate plan still included NVENC CQ 19. Three actual scenes
  projected 7,939,736,895 bytes despite the 3,000,000,000-byte target.
  [NVIDIA's rate-control documentation](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html)
  confirms that VBR-CQ targets quality instead of the average bitrate. The
  correction removes implicit CQ from the NVENC default while retaining
  explicit quality overrides and the mandatory sample-review gate. No file-size
  hard stop or automatic perceptual approval is added.

The second attempt also found FFmpeg absent from the fresh base container.
The production bootstrap already installs it. The isolated smoke harness was
corrected to perform a bounded minimal FFmpeg/font installation, without
reinstalling Python environments or models.

## Attempt receipts

| Attempt | Result | Controller elapsed | Owned Pod, externally ABSENT |
| --- | --- | --- | --- |
| 1 | HTTP 403 before SSH | 13.141 s | `dyhx4lm4ggta3b` |
| 2 | FFmpeg preflight failed before worker | 32.562 s | `wmgbfoffkbrhp9` |
| 3 | Exact volume scan exceeded 60 s, before encode | 191.437 s | `kyzyfwg8pl3tws` |
| 4 | Three actual MP4 samples, verified local readback | 228.297 s | `898uvpqr1u315e` |

Ignored local evidence is preserved under `var/runpod-smoke-20260907/` and
`var/runpod-smoke-20260907-attempt2/` through `-attempt4/`. Each contains
`result.json` and `capacity/capacity-shutdown.json`; successful sample exports
also contain `samples/encoding-samples.json`, hardware, MP4/SRT hashes and logs.
These files are local operational evidence, not committed media.

Each attempt used a uniquely owned temporary Pod and a finite lease. The five
short SSH checks, including the formerly hanging SRT hash command, passed on
attempts 3 and 4 with `-n -T`. The detached worker was started once and observed
through identity-bound status/log requests. No lost connection restarted a
long job. Gmail accepted the clearly labelled technical-test start/result
notifications. The protected Pod `781ct55zv4gkle` remained EXITED and volume
`xgogcmey5o` remained allocated.

## Source and initial measurements

- Source: `/workspace/ep12-nvenc/source.mkv`, 1,035,830,379 bytes;
  SHA-256 `3abb99badc9ec851972a3f5e78c989bae6242f4ba56f77bd2484d852f4e8d398`.
- ID review SRT: `/workspace/ep12-mp4-20260906T125824Z/id.srt`;
  SHA-256 `ef861cad7c9da89c5848f0894eaddf97d876bb1062ed94149fd856c1b4b44b78`.
- Episode 12 source duration: 8,352.921 seconds. The 3 GB plan derives
  2,681,246 video bit/second after the 192,000 audio bit/second allowance.
- Attempt 4: NVIDIA L4, driver 580.159.04, FFmpeg 4.4.2. Three 15-second
  1920x1080 H.264/AAC clips used CUDA AV1 decode and NVENC encode.
  Their combined encode elapsed was 5.522 seconds, projecting 1,025.064 seconds
  for encoding alone. This is a short-sample estimate, not an episode runtime
  guarantee. Their CQ 19 size projection was unsuitable for the desired target.
- Root inspected the three exported JPEG midframes. Subtitle placement was
  visible in frames 1 and 3; frame 2 fell between cues. Still images do not
  establish motion, sound, translation or full-episode perceptual acceptance.
  No audio was listened to and no sample approval was issued.

## Cost and validation

Initial live balance at 15:10:02 UTC was 2.7500422153 USD. After attempt 4 the
observed balance was 2.7114604652 USD, a 0.0385817501 USD decrease across these
observations, including storage and provider billing updates. This is not a
final invoice. Offers were refreshed for every lease: attempt 1 selected
RTX 4000 Ada at 0.28 USD/hour; attempts 2-4 selected L4 at 0.49 USD/hour.
The allowed rate was at most 0.50 USD/hour, with 1.10 USD retained and a
120-second shutdown reserve. Auto-pay was disabled and no top-up occurred.

Official [storage pricing](https://docs.runpod.io/pods/storage/types) was
observed before compute: network volume 0.07 USD/GB/month, container disk
0.10 USD/GB/month. The timestamped checksum-bound quote is retained locally.

Local header-fix tests: 40 focused tests passed; full suite 813 passed and
11 skipped. Logs and JSON status are under
`var/runpod-smoke-20260907/header-fix-tests-*`.
Local quota-fix tests: 9 focused tests passed; full suite 813 passed and
11 skipped in 78.43 seconds. These were terminal observations, without a
separate saved log. Root reviewed the diffs and `git diff --check` passed.
[Quota-fix Linux CI and Docker checks](https://github.com/ismailkorkmaz1905/ma-sub/actions/runs/34139279114)
passed for `d1e1d97`. These checks do not claim live Drive or full GPU pipeline
acceptance.
