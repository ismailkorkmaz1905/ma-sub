# Episode 11 bounded sample authorization - 2026-09-06

## Authority

The user explicitly authorized at most USD 0.50 additional combined RunPod/API spend in response to the proposed small real-audio test. This is a total cap for the sample workflow, not a per-request cap and not authorization for a full episode run, a migration, Drive changes, Episode 12 work or volume deletion. Do not reset the historical controller budget. Do not interpret this permission as proof that the existing controller can enforce a dollar ceiling.

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
