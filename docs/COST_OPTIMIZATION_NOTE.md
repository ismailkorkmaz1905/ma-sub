# GPU cost optimization note

RunPod already runs Linux; the main cost is rented GPU time, not the operating system.

Possible later architecture:

- Local: source discovery/download, audio extraction, packaging, final MKV/SRT, Drive upload and hash verification.
- RunPod GPU only: raw ASR, audio review and forced alignment.
- Stop the Pod during Turkish correction and Indonesian translation waits.
- Consider spot or another GPU provider only after measuring reliability and startup/model-download overhead.

Preferred future direction: keep RunPod, but change to `local preparation -> short GPU work -> local finalization`. That pipeline split has not been implemented.

## Persistence and resume

- The Python environment and uv/Hugging Face/Torch caches are kept under `/workspace`; bootstrap reuses the environment and installs only missing requirements.
- Completed download, audio, raw ASR and forced-alignment outputs use hash-verified markers and can be reused.
- An interrupted audio extraction or forced-alignment stage currently restarts that stage.
- Raw ASR can reuse its recovery checkpoint only after that checkpoint has been written; an earlier interruption restarts raw ASR.
- Audio review checkpoints every few clips and resumes completed decisions.
- Future work: add finer checkpoints inside long ASR and alignment work without weakening source, schema or hash validation.
