# Subtitle workflow research - 2026-09-06

Status: the research below preceded implementation. A separate [local pilot prototype](SUBTITLE_PILOT.md) now exists, but is NOT an acoustically verified production replacement. The user stopped region-by-region debugging and requested investigation of established end-to-end subtitle workflows. No paid GPU/API request, model installation or Drive mutation was made. The USD 0.50 sample allowance remains unspent by this task.

## Finding

The central mismatch is between a usable subtitle product and an all-words acoustic acceptance contract. Missing speaker acquisition is one defect, not a sufficient explanation or universal fix for every failing region. A normal subtitle needs readable cue boundaries and faithful dialogue; our mandatory path additionally demands exact corrected lexical coverage and scored CTC intervals for every word before even creating the Indonesian handoff. Small residual failures consequently block every downstream deliverable.

This is an engineering inference from the repository and primary sources below, not evidence that any replacement already meets the user's four-hour goal.

## What existing projects actually implement

| Primary source | Documented behavior | Implication for this project |
|---|---|---|
| [Subtitle Edit](https://subtitleedit.github.io/subtitleedit/features/speech-to-text.html) | Whisper-based transcription followed by optional waveform timing adjustment, splitting/merging, casing and quality reporting | Separate transcription from subtitle presentation and inspection. A desktop workflow is a reference, not proof of unattended automation. |
| [stable-ts](https://github.com/jianfch/stable-ts) | Faster-whisper integration, silence-aware timing, punctuation/gap/length regrouping, JSON persistence and later SRT export without repeating inference | Evaluate existing timing/post-processing instead of maintaining an expanding custom overlap search. It does not itself prove speaker separation. |
| [WhisperX](https://github.com/m-bain/whisperX) | Separate transcription, alignment and diarization; documented limitations on overlapping speech, speaker accuracy and unalignable characters | Requiring perfect word alignment and speaker certainty everywhere exceeds the tool's documented guarantees. |
| [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | Local diarization, regular and exclusive speaker timelines, model access conditions, public error benchmarks | A candidate for speaker turns, not a perfect oracle. Keep regular overlap evidence; an exclusive timeline cannot prove that a second simultaneous voice was absent. No gated-model terms were accepted here. |
| [Netflix subtitle templates](https://partnerhelp.netflixstudios.com/hc/en-us/articles/219375728-Timed-Text-Style-Guide-Subtitle-Templates) | Source-language timed templates support downstream translation; they need not be verbatim, and two-speaker cues can retain one speaker per line | Distinguish subtitle presentation from word-for-word transcription. This does not authorize automatic deletion, paraphrasing or mixing of dialogue. |
| [Netflix Turkish guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215342858-Turkish-Timed-Text-Style-Guide) | Dual-speaker line treatment, reading-speed and semantic segmentation guidance | Different speakers can remain distinct inside a carefully represented cue. These are editorial references, not mandatory certification requirements for this repository. |
| [Netflix timing guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines) | Subtitle timing is treated as a presentation task with audio and shot context | Word recognition confidence is not the sole measure of synchronization. |

The [faster-whisper benchmark](https://github.com/SYSTRAN/faster-whisper) reports 17 seconds for 13 minutes of audio with large-v2, FP16, batch size 8 on RTX 3070 Ti. This is transcription only on its benchmark, not Turkish drama, translation or end-to-end delivery. Our own historical Episode 11 raw ASR took 986.5 seconds (`docs/EP11_FIRST_PRODUCTION_RUN.md`). The 18-hour incident cannot be assigned to a single Whisper decoding pass.

## Concrete mismatches in current code

- `forced_align.py` applies per-word score, exact lexical coverage, duration and drift gates. `pipeline.py` requires full alignment success before `id_pack`.
- `segmentation.py:_mixed_speaker_risk` flags any two explicit dialogue lines. `validate_segmentation` turns this into an error. A read-only probe with two hyphen-prefixed dialogue lines returned `True`. This is incompatible with introducing an explicitly represented dual-speaker cue without a schema/policy change; it is not permission to concatenate unlabelled speakers.
- `correction_records_to_alignment_inputs` generated 2,717 alignment inputs from the currently retained accepted correction ZIP, with zero speaker IDs. This is a local routing result, not a new GPU result.
- The final target pair has coarse bounds separated by 60 ms, but 500 ms padding on both sides yields 940 ms of shared search space. That alone proves neither duplicate text nor simultaneous speech. Historical probes also show low scores and drift in this region. Adding diarization alone is not a demonstrated fix.
- Existing checks conflate machine uncertainty with structural impossibility. Invalid source identity is a fatal integrity error; a weak word score is an acoustic-review signal. Treating them as the same global delivery barrier makes failure recovery unbounded at the workflow level even when individual searches are bounded.

## Proposed general replacement, not another Episode 11 exception

1. Keep immutable source, source-relative offsets and one persisted primary faster-whisper transcript. Keep captions/rescue text as secondary evidence, not automatically additional canonical dialogue.
2. Evaluate stable-ts as the timing and cue-construction baseline. Preserve original model results and cue-to-source lineage. Do not automatically regroup across established speaker boundaries.
3. Produce speaker turns through a dedicated local diarization stage. Anonymous speaker labels are sufficient; character-name identification is not required. Retain uncertainty and regular overlap evidence instead of manufacturing certainty.
4. Create readable cue-level Turkish drafts. Support either separate cues or explicitly separate speaker lines without merging their underlying turns. Correct Turkish against those cues and source context; re-time affected spans only when edits require it.
5. Translate the stable cue template into Indonesian, then perform cue-level timing/readability, missing-dialogue and text-authority checks. Translation remains outside paid APIs under existing policy.
6. Always preserve a draft and a complete issue report. A draft with unresolved material problems is not a strict deliverable. Fatal hash/schema/source/billing failures still stop immediately. Acoustic uncertainty should enter one bounded review pass; exhausting it should return a truthful unresolved result, not another full-run retry.

Adoption requires an explicit new acceptance contract and migration of consumers/tests. Do not lower the old strict thresholds until tests pass, label new timing as old CTC evidence, rewrite legacy hashes, publish a draft as strict, or create a separate maintained V2 product. Preserve the existing artifacts and controller safety repairs.

## Evaluation boundary

The next experiment should compare this complete baseline on the SAME representative audio set against the current pipeline, with parameters fixed before inspecting individual failures. Include normal dialogue, quick exchanges, music/quiet speech and overlap. Measure draft production, cue synchronization, speaker-turn mixing, missing dialogue, processing time and total spend. Do not tune separately for every UID. A fresh episode end-to-end run is still required before claiming four-hour reliability.

This research did not prove stable-ts is more accurate for this Turkish episode or that pyannote eliminates speaker errors. It establishes a simpler, documented baseline worth evaluating and explains why local fixes to the old acceptance path were not enough.

## Local inventory correction

Earlier filename-only inventory found no standalone media. Inspection of the correction ZIP found 998 embedded WAV clips; therefore "no local audio exists" would be wrong. None of the clip records inspected covers 4,285,000-4,306,000 ms, the final conflict neighborhood. Other representative local samples are available. No WAV was submitted externally or extracted during this research. Before the user redirected the task, six existing synthetic alignment/cache/search tests passed in 0.24 seconds; they are not acoustic acceptance evidence.
