# Muhtemel Ask Subtitle System V2 Beta

V2 beta is a clean rebuild path for the timing and speech-coverage defects
found while reviewing the V1 Episode 10/11 outputs. It is not an in-place
patch for an existing subtitle file, and it does not guarantee that every
subtitle will be perfect.

## Why V1 could look early or miss dialogue

The observed faults had several different causes:

- V1 used provisional faster-whisper word timestamps to create the locked
  subtitle schema. Its later timing check mainly proved that the final SRT
  still matched that same schema; it was not an independent acoustic proof.
- A phrase missed by ASR had no word record, so V1 had no independent
  full-audio speech inventory that could reliably expose the omission.
- Turkish correction happened after the block timings were locked. Inserted,
  replaced or deleted words therefore did not receive a fresh acoustic
  alignment against the audio.
- ASR hallucinations and partially unexplained YouTube captions could be
  mistaken for dialogue near a real speech boundary.

V2 changes that order: it creates coarse Turkish evidence and independent
Silero VAD coverage first, corrects Turkish text before timing is final, audits
only the bounded exceptional WAVs in Colab, and then force-aligns the corrected
Turkish with WhisperX. Final QA is rebuilt from the independent VAD, hashed
audio-review report and forced-alignment artifacts instead of validating a
timeline only against itself.

Installing V2 does not rewrite or repair any V1 SRT or MKV already on Drive.
Running `04_ARCHIVE_V2.ipynb` against an eligible V1 PASS report may migrate
its storage layout, but it does not retime the V1 subtitles. To repair an old
episode, rebuild that episode from `01_PREPARE_TR.ipynb` through
`03_FINALIZE_V2.ipynb` and review the new result.

## Install the isolated beta

1. Extract the current release package.
2. In Drive, keep the existing V1 folder
   `MyDrive/Muhtemel_Ask_Subtitles/SYSTEM/` unchanged.
3. Create `MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA/`.
4. Upload the current release contents into `SYSTEM_V2_BETA/`, preserving the
   complete `src/` and `config/` directories. Confirm that this beta folder
   contains all four V2 notebooks, `requirements-colab.txt`, and
   `requirements-v2-colab.txt`.
5. Close old Colab tabs. Open the V2 notebooks from `SYSTEM_V2_BETA/`, restart
   the runtime, and then run them. Every V2 notebook checks that its imported
   modules came from the beta tree.

The required notebooks are:

```text
SYSTEM_V2_BETA/
  01_PREPARE_TR.ipynb
  02_ALIGN_PREPARE_ID.ipynb
  03_FINALIZE_V2.ipynb
  04_ARCHIVE_V2.ipynb
  requirements-colab.txt
  requirements-v2-colab.txt
  config/
  src/
```

Adding the release ZIP to ChatGPT Project Sources does not install or update
the Colab runtime. The executable copy must exist at the Drive path above.

## One episode, in order

1. In `01_PREPARE_TR.ipynb`, set `EPISODE` and `SOURCE_URL`, then run all on a
   Colab GPU. It downloads or resumes the episode-named source, extracts audio,
   runs coarse Turkish ASR and independent Silero VAD, retries speech holes,
   and writes:

   ```text
   translation_input/Muhtemel Ask X.Bolum_TR_CORRECTION_PACK.zip
   ```

2. In ChatGPT Work, use the Turkish prompt from
   `V2_NEW_EPISODE_PROMPTS.md`. This is strictly text-only: Work must not open,
   transcribe or decide any WAV. Every linked audio record remains pending. The
   required provisional result is:

   ```text
   translation_output/Muhtemel Ask X.Bolum_TR_TEXT_CORRECTED.zip
   ```

3. In `02_ALIGN_PREPARE_ID.ipynb`, set the same `EPISODE` and run all. It
   validates the text correction, reviews only the hash-bound short WAVs with a
   resumable blind-first Colab ASR pass (with one bounded evidence-conditioned
   retry only when blind evidence is ambiguous), and writes the final
   `translation_output/Muhtemel Ask X.Bolum_TR_CORRECTED.zip` plus
   `prepare/audio_review_v2.json`. A genuinely ambiguous remainder is displayed
   in pages of ten and blocks publication until explicitly heard. The notebook
   then force-aligns corrected Turkish, recomputes speech coverage, locks the
   final blocks, and writes:

   ```text
   translation_input/Muhtemel Ask X.Bolum_ID_TRANSLATION_PACK.zip
   ```

4. In ChatGPT Work, use the Indonesian prompt from
   `V2_NEW_EPISODE_PROMPTS.md`. The required result is:

   ```text
   translation_output/Muhtemel Ask X.Bolum_ID_TRANSLATED.zip
   ```

5. In `03_FINALIZE_V2.ipynb`, set the same `EPISODE` and run all. It rebuilds
   and revalidates the evidence, publishes the two SRTs and soft-subtitle MKV,
   and writes the V2 PASS report last.
6. Watch beginning, middle, end, fast-dialogue, music and previously suspicious
   samples. Then run `04_ARCHIVE_V2.ipynb` first with
   `DELETE_INTERMEDIATES = False`. Only after reviewing its exact plan should
   you enable cleanup and type the episode-bound confirmation.

Do not skip a notebook or reuse a ZIP from another episode. A stage marker is
accepted only when its bound inputs and outputs still match.

## Source, final and the temporary duplicate

`source/` is the resumable working input. It contains the original downloaded
episode-named video plus download metadata and any available source caption
evidence. `prepare/`, `translation_input/`, and `translation_output/` contain
rebuild and validation evidence.

`final/` is the publish output. `03_FINALIZE_V2.ipynb` stream-copies the source
audio/video into the final MKV and embeds Indonesian and Turkish subtitles.
Because stream-copy verification needs both files, the source video and final
MKV intentionally coexist before cleanup. This is temporary full-size
duplication, not two independent masters.

After verified cleanup, the episode contains exactly these three files and no
source/work tree:

```text
final/
  Muhtemel Ask X.Bolum.mkv
  subtitles/
    Muhtemel Ask X.Bolum-id.srt
    Muhtemel Ask X.Bolum-tr.srt
```

The MKV is the single playback A/V file and already contains both subtitle
tracks. The two small external SRTs are retained once in `final/subtitles/`
for editing, export and recovery; Infuse does not need them to play the MKV.
Cleanup removes the original source only after compressed A/V stream hashes
match and both embedded subtitle tracks round-trip exactly.

## Canonical beta timing and coverage policy

These are the current code values. They are deliberately fixed for the pilot;
changing a notebook field does not make a looser run publishable.

| Stage | Current canonical bound or rule |
|---|---|
| Independent audit VAD | Silero speech span minimum `120 ms`; audit speech padding `60 ms`; VAD regions merge across gaps up to `150 ms` |
| Word-to-speech coverage | Aligned words receive `40 ms` coverage padding on each side and merge across gaps up to `40 ms` |
| Missing-speech gate | A remaining gap of at least `150 ms`, or a speech region below `70%` effective word coverage, requires rescue/review; rescue windows use `700 ms` padding and merge within `250 ms` |
| Word outside speech | No contiguous portion of an aligned word may extend more than `120 ms` outside the trusted speech union, unless an exact WAV-backed candidate was reviewed and confirmed as dialogue |
| Review WAV context | Target intervals are immutable; clips add up to `750 ms` context on each side and are expanded to at least `1500 ms` where media bounds allow |
| Forced-alignment window | Each corrected utterance is aligned within `900 ms` coarse-bound padding |
| Acoustic score | Every lexical word needs a finite score of at least `0.30`; inserted or replaced lexical tokens need at least `0.55`; unchanged scores from `0.30` through `0.549...` remain pilot-review warnings |
| Word timing | Each lexical word needs a positive interval, may last at most `2500 ms`, and may drift outward at most `500 ms` beyond the original unpadded coarse bounds |
| Cue timing | Cue start is the first reliable word (`0 ms` artificial lead); minimum duration `700 ms`; construction ceiling `7000 ms`; nominal end padding `220 ms`; next-speech guard `80 ms`; overlaps and negative word gaps are forbidden |
| Cue readability | Maximum `20` visible characters per second for both languages; at most two lines are designed around a `42`-character target; segmentation uses at most `24` words per block |
| Internal gap | A block with an internal aligned-word gap of `650 ms` or more is rejected as an unresolved speaker/boundary risk; `1000 ms` is the hard-silence split threshold |

The `7000 ms`, two-line, 42-character and 24-word values are construction
limits or layout targets; the publication report separately hard-requires zero
short cues, high-CPS cues, overlaps, invalid timing, missing Indonesian lines,
unaligned/synthetic words, missing or low scores, outward-drift violations,
unreviewed lexical deletions, alignment-provenance mismatches and unresolved
speech regions.

### YouTube caption gap policy

YouTube captions are evidence, not timing authority. A non-empty caption is
considered explained only when the union of coarse ASR intervals and
independent Silero audit-VAD intervals covers at least `50%` of its exact
interval **and** every contiguous unexplained run is at most `229 ms`. If
either condition fails, the entire exact caption interval becomes an
`orphan_youtube_caption` candidate with a hash-bound contextual WAV. A first
unexplained run of `230 ms` is therefore reviewable; a run of `229 ms` or less
is only tolerated when the 50% coverage condition also passes. This policy
does not copy YouTube caption timing into the final SRT.

## Pilot status: no zero-error guarantee

V2 beta is designed to expose and stop measurable failure modes; it cannot
prove that ASR text, VAD classification, forced alignment, human correction or
translation is semantically perfect. Automatic PASS means the recorded hard
gates passed for those exact hashed artifacts. It is not a promise of zero
early words, zero missed dialogue or zero translation errors.

Episode 10 remains the first real pilot. Do not call V2 production-ready until
the current automated suite passes, the complete Episode 10 workflow succeeds,
and a human playback review covers representative and previously bad scenes.
Thresholds must not be loosened merely to obtain PASS; record pilot findings
and change policy only with new regression tests.
