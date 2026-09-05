# GPU cost optimization note

RunPod already runs Linux; the main cost is rented GPU time, not the operating system.

Possible later architecture:

- Local: source discovery/download, audio extraction, packaging, final MKV/SRT, Drive upload and hash verification.
- RunPod GPU only: raw ASR, audio review and forced alignment.
- Stop the Pod during Turkish correction and Indonesian translation waits.
- Consider spot or another GPU provider only after measuring reliability and startup/model-download overhead.

Preferred direction: keep RunPod, but change to `local preparation -> short GPU work -> local finalization`. No implementation decision has been made.
