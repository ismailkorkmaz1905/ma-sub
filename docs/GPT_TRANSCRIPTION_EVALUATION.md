# GPT transcription evaluation - 2026-09-06

Decision: COMPLEMENT for a future authorized evaluation; REJECT an immediate replacement of strict forced alignment. No paid request was made and no transcription API implementation was added. The OpenAI Docs skill determined the official-source research workflow.

## Capabilities and limits

The official file guide recommends `gpt-transcribe` for recorded speech. Uploads are limited to 25 MB; larger recordings need compression or chunks. Accepted formats include WAV, MP3, MP4, MPEG, MPGA, M4A and WebM, so the existing FLAC must be converted. Context, keyword and language hints are available. The guide directs word-timestamp users to `whisper-1`, not GPT Transcribe. Diarization uses `gpt-4o-transcribe-diarize` with `diarized_json` speaker/start/end segments and chunking for inputs longer than 30 seconds. These are segment annotations, not proof of exact word boundaries. The translation endpoint produces English, not Indonesian. [Official file transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text)

The inspected official pages do not establish Turkish-drama word-error rates, accuracy on background music, reliable separation of simultaneous voices, persistent speaker identity across independently submitted chunks, deterministic replay, or an Episode 11 latency guarantee. These remain UNKNOWN until an authorized, retained evaluation compares transcripts and timing against source audio. `gpt-4o-transcribe` has an official general accuracy improvement claim over original Whisper, but it is not evidence that this particular episode will pass. [GPT-4o Transcribe](https://developers.openai.com/api/docs/models/gpt-4o-transcribe)

## Price scenario

The validated local correction pack's last coarse end is 8,181,660 ms. That is 136.361 minutes from time zero, not a measured complete media duration. The following single-pass scenario uses that span. It excludes trailing media, retries, duplicated chunk context, conversion, network transfer, alignment, review, and translation.

| Model | Published estimated price | 136.361-minute scenario |
|---|---:|---:|
| gpt-transcribe | USD 0.0045/minute | USD 0.6136245 |
| gpt-4o-transcribe | USD 0.006/minute | USD 0.818166 |
| gpt-4o-mini-transcribe | USD 0.003/minute | USD 0.409083 |

Derivation: scenario minutes multiplied by the published per-minute estimate. Prices were checked on 2026-09-06. [Official API pricing](https://developers.openai.com/api/docs/pricing)

`gpt-4o-transcribe-diarize` lists audio input at USD 2.50 per million tokens and text output at USD 10 per million tokens. A precise Episode 11 estimate cannot be calculated without its billed token usage. [Official diarization model page](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize)

## Throughput, privacy, and reproducibility

GPT Transcribe's model page lists tier-dependent limits, including Tier 1 at 500 requests/minute and 200,000 tokens/minute; free-tier use is unsupported. Actual account access and limits have not been queried. The model page lists the `gpt-transcribe` alias, without a dated snapshot in the inspected snapshot list. Do not promise immutable model behavior from an alias. [GPT Transcribe model page](https://developers.openai.com/api/docs/models/gpt-transcribe)

The official data-control table lists `/v1/audio/transcriptions` as not used for training, with no abuse-monitoring or application-state retention and Zero Data Retention eligibility. Separate file-storage and Realtime endpoints have different retention rules; do not transfer that claim to them. Audio would still leave the local/RunPod environment for API processing. [Official data controls](https://developers.openai.com/api/docs/guides/your-data)

A future adapter must preserve the exact input bytes/hash, source-relative chunk offsets, model identifier, request parameters, context hashes, response bytes/hash, usage, request ID and completed-chunk journal. Resume should reuse stored responses rather than resubmit completed chunks. Retry only transient failures within a finite total deadline; ambiguous timeouts can incur duplicate charges. Account limits and retry billing have not been measured here.

## Compatibility decision

The repository requires corrected lexical tokens with validated acoustic word scores and intervals, independent VAD coverage, and exact immutable correction/translation contracts. GPT text-only output cannot substitute for those fields. Diarized segment labels cannot manufacture per-word scores or prove that two overlapping lines are different speakers.

The practical next evaluation, after explicit cost authorization, is a small representative Turkish sample containing the two residual UIDs, simultaneous dialogue, music, short replies, names, and quiet speech. Compare GPT transcription as a text/review input while retaining independent timing and speaker evidence. Preserve the current Episode 11 correction chain; do not rewrite it from a fresh API transcript. Full replacement requires an explicitly reviewed architecture/schema migration and real acoustic acceptance, not a model-name change.

The API scenario price is low relative to the user's reported USD 20 spend, but it does not prove that the episode would have finished within four hours: provisioning, invalidation scripts, transfer retries, missing deadline enforcement, and acceptance defects remain separate causes.
