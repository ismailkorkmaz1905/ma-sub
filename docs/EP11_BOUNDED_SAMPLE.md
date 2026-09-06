# Episode 11 bounded sample authorization - 2026-09-06

## Authority

LATEST OVERRIDE: the user replaced the USD 0.50 ceiling with permission to use existing RunPod balance while leaving USD 1.00. This applies to the Episode 11 pilot, not Episode 12, full-episode execution, GPT transcription, top-ups, migration, Drive changes or volume deletion. Do not reset the historical episode deadline. This is spending authority, not proof that the runner enforces it.

Read-only GraphQL `myself { clientBalance currentSpendPerHr isAutoPayEnabled }` at `2026-09-06T02:59:24.8248931Z` (10:59:24.8248931 SGT) returned balance USD 3.7543464701, account spend USD 0.005/hour and auto-pay false. The mathematical allowance at that instant is USD 2.7543464701 (balance minus USD 1.00), before subsequent charges and shutdown reserves. Operational spend must be lower to cover ongoing account charges, billing delay and shutdown. Re-query before any start, never count future top-ups as renewed permission, and never wait until the displayed balance equals USD 1.00 to stop.

Preserved storage continues charging even after compute stops. The USD 1.00 reserve cannot be promised indefinitely while storage remains retained; do not delete the volume to enforce that reserve without explicit permission. No new paid operation or billing-setting change was made during this authorization update.

The original USD 0.50 authorization and API-oriented preflight below are historical. Current workflow excludes GPT transcription; use `SUBTITLE_PILOT.md` for runtime blockers and corrected local sample availability.

Before compute starts, establish a bounded sample runner, current rate, startup and cleanup reserves, artifact retrieval, and the total allocation shared with API requests. No paid retry after an ambiguous response unless the reserved budget covers both attempts. Preserve request/audio/response hashes and source offsets. Sample speaker output is review evidence, not automatic strict acceptance or permission to guess overlapping speakers.

## Read-only preflight

- Candidate: `a87ffce`; initial tracked worktree clean on `main`.
- `OPENAI_API_KEY`: absent from Process, User and Machine environment scopes. No project `.env*` file was found by a filename-only search excluding legacy, dependencies, Git and episode artifacts. No credential values were printed or searched from unrelated stores.
- Local `EPISODES` inventory: zero files with WAV, MP3, FLAC, M4A, MP4, MKV, WebM or OGG extensions, including ignored files. This does not establish absence elsewhere on the user's computer or in the retained remote volume.
- External RunPod GET observation: `2026-09-06T02:21:57.789180+00:00` (10:21:57.789180 SGT). Pod `781ct55zv4gkle`: `desiredStatus=EXITED`, `costPerHr=0.74`, `networkVolumeId=xgogcmey5o`.
- No Pod start, migration, transcription request, Drive mutation or volume deletion was performed. New execution spend initiated by this sample task: USD 0.00. This is an action record, not a provider billing reconciliation; retained storage is separate existing liability.

## Next boundary

Status: BLOCKED for the API sample until an OpenAI API key is configured securely and source audio can be obtained. Do not spend on GPU startup while the API prerequisite is missing. The user should configure `OPENAI_API_KEY` locally, never paste it into chat. If a local source copy exists, inventory and verify it before considering paid retrieval.

The official OpenAI Docs workflow was used to check recorded-audio inputs and diarization availability. [File transcription](https://developers.openai.com/api/docs/guides/speech-to-text) lists a 25 MB upload limit, supported audio containers and specialized diarization output. [Diarization model](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize) and [pricing](https://developers.openai.com/api/docs/pricing) are the source of model/cost planning, not a guaranteed quality or billing result. No model adapter was implemented or paid request attempted in this preflight.
