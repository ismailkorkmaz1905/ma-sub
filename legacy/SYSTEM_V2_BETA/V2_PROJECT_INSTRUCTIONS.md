# Muhtemel Ask Subtitle System V2 - Project Instructions

This Project assists the isolated Muhtemel Ask V2 beta workflow. Google Drive
is the source of truth. Never reuse an artifact from another episode, another
schema hash or the V1 workflow.

Drive root:

`MyDrive/Muhtemel_Ask_Subtitles`

Executable V2 root:

`MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA`

All four V2 notebooks must be opened from that beta root, in this order:

1. `01_PREPARE_TR.ipynb`
2. Turkish correction in ChatGPT Work
3. `02_ALIGN_PREPARE_ID.ipynb`
4. Indonesian translation in ChatGPT Work
5. `03_FINALIZE_V2.ipynb`
6. Human playback review
7. Optional `04_ARCHIVE_V2.ipynb`, first as a dry run

Never tell the user that adding a ZIP to Project Sources updates Colab. The
current executable release must be uploaded separately to
`SYSTEM_V2_BETA/`. Do not replace or import the stable V1 `SYSTEM/` directory
while evaluating V2.

V2 is a pilot, not a zero-error guarantee. A mechanical PASS proves only that
the hard checks passed for the exact hashed evidence. Existing V1 subtitles
and MKVs are not retimed or repaired by installing V2 or by archiving them; an
old episode needs a complete V2 rebuild and human playback review.

## Turkish correction pass

After `01_PREPARE_TR.ipynb` succeeds, read exactly:

`EPISODES/Muhtemel Ask X.Bolum/translation_input/Muhtemel Ask X.Bolum_TR_CORRECTION_PACK.zip`

Follow `TR_CORRECTION_INSTRUCTIONS.md` inside that ZIP. Copy every immutable
field exactly, including evidence, timing, identity, risk, audio and
`asr_audit` fields. Correct Turkish only; do not translate.

This pass is text-only. Never open, inspect, transcribe or listen to a linked
WAV. `02_ALIGN_PREPARE_ID.ipynb` owns the bounded, resumable Colab audio audit.
Never browse or search the web for this pass. Do not search Hugging Face,
GitHub, model repositories or websites; do not download or install Whisper or
another ASR model; do not look for a transcription plugin; and do not call an
external transcription or translation service. Correct the Turkish yourself
from the JSON text evidence in the package.
The only permitted external connector is Google Drive for the exact input and
output paths. Local ZIP/JSON handling is allowed; network access, package
installation and model/tool discovery are forbidden.

Every ordinary record without an immutable audio-review entry must remain
`non_dialogue=false`, use `audio_reviewed=false` and
`review_disposition=not_applicable`, and receive non-empty corrected Turkish;
never discard it as noise. For every exact hash-bound audio-review record, keep
`non_dialogue=false`, `review_required=true`, `audio_reviewed=false`, and
`review_disposition=pending_audio_review`. Correct available ASR/YouTube text,
but keep a blank `unresolved_vad_speech` record's `tr_corrected` empty. Never
infer speech or silence from neighboring text.

Do not set `audio_reviewed=true`, `reviewed_non_dialogue`,
`confirmed_dialogue`, or `discarded_asr_hallucination` yourself. The exact WAV,
its SHA-256, the isolated Colab transcript and the resulting disposition are
bound into `audio_review_v2.json` by the next notebook. Ambiguous clips remain
blocked and are surfaced for explicit listening instead of being guessed.

Deleting punctuation or changing case is allowed, but deleting any lexical
ASR token requires an exact hash-bound audio-review candidate. Without one,
use `review_required=true`, `audio_reviewed=false`, and
`review_disposition=pending_audio_review`; report the exact UID and tell the
user to rerun `01_PREPARE_TR.ipynb` with that UID in
`EXTRA_AUDIO_REVIEW_UIDS`. The notebook automatically rebinds the existing
validated text-only output when the pack change is provably review-only, so do
not redo all Turkish correction unless that rebind is rejected. Do not claim the
pass is complete while any review remains pending.

Validate every record and write only the provisional text result:

`EPISODES/Muhtemel Ask X.Bolum/translation_output/Muhtemel Ask X.Bolum_TR_TEXT_CORRECTED.zip`

Reopen the written ZIP and verify it before reporting completion. Then tell
the user the next executable step is `02_ALIGN_PREPARE_ID.ipynb` from
`SYSTEM_V2_BETA`, with the same episode number. That notebook alone creates the
final `_TR_CORRECTED.zip` after the acoustic gate passes.

## Indonesian translation pass

After `02_ALIGN_PREPARE_ID.ipynb` succeeds, read exactly:

`EPISODES/Muhtemel Ask X.Bolum/translation_input/Muhtemel Ask X.Bolum_ID_TRANSLATION_PACK.zip`

Follow `ID_TRANSLATION_INSTRUCTIONS.md` inside the ZIP. Translate only
`tr_text` into natural Indonesian. Never change block identity, order, timing,
Turkish text, alignment provenance or schema hashes.

Translate with the model itself. Never browse or search the web, use an online
translator, download a translation model, look for a translation plugin, or
send the text to an external service.

The Indonesian must sound like spoken dialogue, not literal machine
translation. Preserve each speaker's relationship, formality, emotion,
romance, anger, sarcasm, comedy, insults, hesitation and unfinished speech.
`aku`, `kamu`, `nggak`, `udah` and `aja` fit ordinary informal dialogue; use
`saya`, `Anda`, `Pak` and `Bu` when the scene requires respect or formality.
Do not force slang, censor or soften the meaning, add explanations, or move
dialogue between records. Keep every subtitle concise and natural while
following the exact canonical names, number/money repetitions and religious
expressions in `glossary.json`.

Validate every record and write only:

`EPISODES/Muhtemel Ask X.Bolum/translation_output/Muhtemel Ask X.Bolum_ID_TRANSLATED.zip`

Reopen the written ZIP and verify it before reporting completion. Then tell
the user the next executable step is `03_FINALIZE_V2.ipynb` from
`SYSTEM_V2_BETA`, with the same episode number. Do not describe translation
completion as final publication.

## Publication and storage language

Use these terms consistently:

- `source/` is the original resumable input and its working evidence.
- `final/` is the V2 publish output.
- Before archive cleanup, the source video and stream-copy final MKV coexist
  temporarily so their compressed A/V streams can be compared.
- After verified V2 cleanup, the episode retains exactly
  `final/Muhtemel Ask X.Bolum.mkv`,
  `final/subtitles/Muhtemel Ask X.Bolum-id.srt`, and
  `final/subtitles/Muhtemel Ask X.Bolum-tr.srt`.

The MKV embeds both subtitle tracks; the two small external SRTs are retained
once for editing/export. Do not call these two full-size duplicate videos.

Do not claim completion until the exact output ZIP has been written to Drive
and reopened successfully. Do not modify, move, rename or delete episode files
unless the user explicitly asks. Never read, copy or expose
`PRIVATE/youtube-cookies.txt`. Give concise progress updates during long work.
