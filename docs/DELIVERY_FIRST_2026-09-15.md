# Delivery-first policy from Episode 14

Operator instruction on 15 September 2026 supersedes the earlier requirement to
block the episode on subtitle quality failures. The maintained branch remains main.
No GPU or live Drive run was performed while implementing this change.

## Outcome

The default first-hour/local-QSV route uses delivery-first processing for Episode 14
and later. Retained Episode 12/13 and the strict finalizer remain unchanged.
First publish the first approximately 60-minute part with Indonesian subtitles
burned in. After the remaining burned parts are verified, assemble and publish one
full-episode H.264/AAC MP4. COMPLETE_PARTS alone is not the new completion condition.

## Timing and quality

Primary Turkish ASR remains GPU-only and uses the existing authenticated primary
checkpoint. This delivery route does not wait for a separate manual Turkish
correction or mandatory rescue/audio-review pass; primary transcription uncertainty
is reported. A real Indonesian translation return is still required. No translation
API, CPU ASR fallback or invented Indonesian text is added.

CTC refinement is optional, in an independently killable process. Each source cue
gets at most 20 seconds; all refinement shares a persisted 600-second episode
allowance, including bounded model startup. Keep 1800 seconds of remaining episode
allowance for delivery rather than spending it on refinement. On rejection or timeout,
keep the usable source interval. Omit only a cue with no usable bounded interval.
An incomplete parent longer than 10000 milliseconds cannot become a giant cue.
Cached successful, rejected and interrupted groups are not automatically retried.

The frozen ID pack labels timing honestly. It never claims strict CTC PASS for
fallback timestamps. Subtitle length, reading speed, uncertain meaning, names and
religious-expression issues remain visible warnings; they do not block delivery.
Instructions still require natural Indonesian and preservation of names and Allah.
An absent/blank translation removes only that cue and is reported. Unknown IDs,
changed immutable text, changed source bytes and forged signatures remain integrity
errors, not quality warnings.

## Publication

Delivery exports, warnings and checkpoints have distinct formats, filenames and
HMAC purposes from strict evidence. A successful transfer is not a quality PASS.
The full video is assembled with stream copy from already burned, compatible parts.
Exact part durations control offsets. Every planned source sample range must appear
once in order; a missing part cannot be called a full episode. The first part must
have verified Drive readback before the full file can be published.

Full assembly is checkpointed before upload. Upload/readback failure retains the
assembled video and resumes from it, without another alignment or encode. Existing
Drive filenames are preserved and byte count plus full SHA-256 readback are still
required. Quality warnings are retained under final/delivery-first/QUALITY-WARNINGS.json.

Hardware failure, unavailable source media, missing translation input, insufficient
disk or unreachable Drive cannot be turned into successful delivery by code. Existing
finite infrastructure timeouts and the six-hour compute budget remain. The four-hour
target and six-hour upper allowance have not been benchmarked on a real Episode 14.

FFmpeg concatenation contract: https://ffmpeg.org/ffmpeg-formats.html#concat
