# Muhtemel Ask Subtitle System - Read This First

## Choose the workflow deliberately

V2 beta is the new rebuild path for early words, missing dialogue and circular
timing validation found while reviewing V1 Episode 10/11. V1 used provisional
Whisper word times as the locked timeline, had no independent full-audio VAD
inventory that could prove missed speech, and corrected Turkish only after
timing was fixed. V2 corrects Turkish first, then creates fresh WhisperX forced
alignment and checks it against independent Silero VAD evidence.

V2 is still a pilot, not a zero-error guarantee. Automatic PASS cannot prove
perfect ASR, VAD, alignment, correction or translation; representative human
playback review remains mandatory.

Installing the beta does **not** repair, retime or overwrite existing V1 SRTs
or MKVs. Even V2 archive migration preserves the timing already present in a
V1 PASS output. Rebuild an old episode through the complete V2 workflow if its
timing needs repair.

## V2 beta quick start

1. Keep `MyDrive/Muhtemel_Ask_Subtitles/SYSTEM/` unchanged for V1 rollback.
2. Extract the current release and upload its complete contents, including
   `src/`, `config/`, both requirements files and all four V2 notebooks, to:

   ```text
   MyDrive/Muhtemel_Ask_Subtitles/SYSTEM_V2_BETA/
   ```

3. Close old Colab tabs. Open notebooks from `SYSTEM_V2_BETA/`, restart the
   runtime, and run the same episode in this exact order:

   ```text
   01_PREPARE_TR.ipynb
   text-only Turkish correction in ChatGPT Work
   02_ALIGN_PREPARE_ID.ipynb (bounded Colab WAV audit + alignment)
   Indonesian translation in ChatGPT Work
   03_FINALIZE_V2.ipynb
   human playback review
   04_ARCHIVE_V2.ipynb (dry run first, cleanup only after review)
   ```

Adding the release ZIP to ChatGPT Project Sources is not a Colab installation;
the executable copy must exist in the Drive beta path. Use
`V2_NEW_EPISODE_PROMPTS.md` for the two Work passes and read
`V2_BETA_README.md` for the exact artifact names, review decisions and current
hard timing/coverage policy.

The Turkish Work pass must never browse for ASR tooling or listen to the WAVs
inside its pack. It writes only `_TR_TEXT_CORRECTED.zip` with every linked
audio decision still pending. Notebook 02 alone resolves the short hash-bound
clips, writes `audio_review_v2.json`, and creates the final
`_TR_CORRECTED.zip`; ambiguous clips fail closed and are shown for explicit
listening.

In V2, `source/` is the original resumable input and `final/` is the verified
publish output. A source video and final stream-copy MKV coexist temporarily so
their compressed A/V streams can be compared. After explicit verified cleanup,
the episode retains exactly:

```text
final/Muhtemel Ask X.Bolum.mkv
final/subtitles/Muhtemel Ask X.Bolum-id.srt
final/subtitles/Muhtemel Ask X.Bolum-tr.srt
```

The MKV contains both subtitle tracks. The two small external SRTs remain once
for editing/export; the original full-size source video and all work files are
removed only by the verified cleanup step.

## V1 legacy reference

This system creates locked Turkish subtitle blocks in Google Colab, lets
ChatGPT Work Ultra correct/translate only their text, and then validates and
publishes matching Turkish and Indonesian subtitles in Colab.

Everything below this heading describes V1. It is kept for rollback and for
understanding existing V1 workspaces; it is not the procedure for a new V2
pilot run.

It does not use the OpenAI API, a translation API, a local LLM, a translation
model, scene detection, or software installed on your computer.

## V1 one-time setup

1. Extract this ZIP normally.
2. In Google Drive create `MyDrive/Muhtemel_Ask_Subtitles/SYSTEM/`.
3. Upload the extracted contents into `SYSTEM/`, preserving `config/`, `src/`
   and the three notebooks.
4. Open `01_PREPARE.ipynb` from Drive with Google Colab and choose a GPU
   runtime: **Runtime > Change runtime type > T4 GPU** (or a better GPU).
5. Allow Colab to mount Google Drive when prompted.

The notebooks install all Python packages, ffmpeg, yt-dlp and Whisper inside
the temporary Colab runtime. Nothing is installed on your computer.

## V1 Drive layout

Each episode is isolated under:

```text
MyDrive/Muhtemel_Ask_Subtitles/
  SYSTEM/
  PRIVATE/
    youtube-cookies.txt
  ARCHIVE_REPORTS/
    Muhtemel Ask 12.Bolum_ARCHIVE_RECEIPT.json
  EPISODES/
    Muhtemel Ask 12.Bolum/
      source/
        Muhtemel Ask 12.Bolum.mkv
      prepare/
      translation_input/
      translation_output/
      review/
      final/
```

The notebooks refuse to consume a translation from another episode or schema.
If segmentation ever changes, increase the schema version and translate the
new pack. Old translations are not migrated by block number.

## V1 every new episode

### 1. Prepare

Open `01_PREPARE.ipynb`, change only:

```python
EPISODE = 12
SOURCE_URL = "https://www.youtube.com/watch?v=..."  # Dailymotion URLs also work
```

Select **Runtime > Run all**. Use only a video you are permitted to access and
download. The result is written to the episode's `translation_input/` folder:

```text
Muhtemel Ask 12.Bolum_TRANSLATION_PACK.zip
```

Completed stages are validated with hashes and reused after a runtime
disconnect. A file's existence alone never marks a stage complete.

#### If YouTube asks for browser verification

PREPARE first tries YouTube without account cookies. If YouTube returns the
specific "Sign in to confirm you're not a bot" challenge, the notebook stops
the repeated anonymous retries and checks the reusable credential at:

```text
MyDrive/Muhtemel_Ask_Subtitles/PRIVATE/youtube-cookies.txt
```

If no valid saved file exists, or YouTube rejects it, PREPARE opens a Colab
upload dialog once.

Upload one fresh `cookies.txt` in Netscape/Mozilla format. Export only
`youtube.com` cookies:

1. Open one new private/incognito window and sign in to YouTube.
2. In that same and only private tab, open
   `https://www.youtube.com/robots.txt`.
3. Export only the `youtube.com` cookies as Netscape `cookies.txt`.
4. Close the private window immediately and do not reopen that session.

After a fresh uploaded cookie completes an authenticated download, PREPARE
validates it again and saves it to the private Drive location above for later
runs. A rejected fresh cookie never replaces the last saved file. The Drive
copy is plain text and is an account session secret: keep the
`Muhtemel_Ask_Subtitles` folder private, use a throwaway YouTube account, and
never put the file in the project ZIP or send it through ChatGPT.

yt-dlp never reads the Drive file directly. PREPARE copies it into a random
temporary Colab `/content` directory with restricted runtime permissions and
removes that runtime copy after every authenticated attempt, whether the
attempt succeeds or fails. Cookie path, bytes, hash and presence remain
excluded from workspace identity, metadata and completion markers, so replacing
an expired cookie does not invalidate resumable downloads.

### 2. Translate in Work Ultra

Preferred connected-Drive flow: after PREPARE finishes, tell ChatGPT Work
Ultra that the episode pack is ready in the episode's `translation_input/`
folder. Work reads that exact pack from Drive, translates and validates it,
then writes the required ZIP directly to the same episode's
`translation_output/` folder. You do not need to download, upload or unpack
either ZIP.

Use this request:

```text
Read and follow TRANSLATION_INSTRUCTIONS.md inside the ZIP exactly. Complete
all batches, validate them, and return the one required TRANSLATED.zip. Do not
change segmentation, timing, order, block_uid or schema_sha256.
```

Work writes:

```text
Muhtemel Ask 12.Bolum_TRANSLATED.zip
```

If connected Drive is unavailable, the fallback is to upload the translation
pack manually and then place Work's returned ZIP in `translation_output/`.
In either flow, do not unpack or rename the ZIP or its contents.

### 3. Finalize

Open `02_FINALIZE.ipynb`, change only:

```python
EPISODE = 12
```

Run all cells. FINALIZE first validates the ZIP CRC, manifest identity, UID
set, UID order, schema hash, record count and semantic alignment. Any mismatch
stops the notebook before final files are written.

Successful output in `final/`:

```text
Muhtemel Ask 12.Bolum.tr-final.srt
Muhtemel Ask 12.Bolum.id-final.srt
```

FINALIZE also creates Infuse-ready sidecars beside the verified source video,
without copying the video:

```text
source/Muhtemel Ask 12.Bolum-id.srt
source/Muhtemel Ask 12.Bolum-tr.srt
```

Their basename always follows the actual downloaded video basename. New
downloads use the episode name; an older hash-verified `source.ext` workspace
continues to resume safely with legacy `source-id.srt` / `source-tr.srt` names.

Uncertain rows only are written to:

```text
review/Muhtemel Ask 12.Bolum_review.xlsx
```

If `CREATE_MKV = True`, FINALIZE also creates:

```text
Muhtemel Ask 12.Bolum - Endonezce + Turkce.mkv
```

The MKV is published only after both embedded subtitle tracks are extracted
again and compared with the source SRT files. Video and audio are stream-copied.

## V1 Infuse on Apple TV

Two supported modes:

- External subtitles: open the episode `source/` folder. FINALIZE already puts
  matching `-id.srt` and `-tr.srt` files beside the video, which lets Infuse
  discover both languages automatically. Indonesian and Turkish timings are
  identical.
- Soft-sub MKV: open the generated MKV. Indonesian is the default subtitle
  track; Turkish is available as the second subtitle track.

Add the Google Drive folder as an Infuse share and open the episode from there.

### 4. Archive and reclaim Drive space (optional)

After FINALIZE has produced a PASS report, open `03_ARCHIVE_CLEANUP.ipynb` and
change:

```python
EPISODE = 12
DELETE_INTERMEDIATES = False
KEEP_SOURCE_VIDEO = False
```

Run all cells. The notebook creates or revalidates the canonical soft-sub MKV
without re-encoding its video or audio. It extracts both subtitle tracks again,
compares them exactly with the final SRT files, and compares compressed A/V
stream hashes before any cleanup is offered. Atomic creation temporarily needs
about one source-video-sized block of free Drive space (roughly 0.5 GB for
Episode 11).

The default is a dry run: it creates/verifies the MKV and prints the exact files
and bytes that cleanup would remove. To perform cleanup, set
`DELETE_INTERMEDIATES = True`, run all again and type the exact confirmation
shown by the notebook. You may also set it to `True` before the first run; the
same run prints the complete preview before it opens the confirmation prompt.

With `KEEP_SOURCE_VIDEO = False`, the retained episode files are the verified
MKV and every SRT file. Any unexpected extra video is also preserved and
reported instead of being guessed to be disposable. This gives the useful storage reduction because the
original source video is replaced by an A/V-identical stream-copy inside the
MKV. Set it to `True` if you also want to retain the original MP4/MKV source;
that requires roughly one extra source-video-sized copy on Drive.

Cleanup removes the schema, ASR/audio, translation ZIPs, review workbook,
markers and finalization report. That episode can no longer resume in PREPARE
or FINALIZE; future subtitle changes require rebuilding its workspace. A small
verification receipt is kept outside the episode at `ARCHIVE_REPORTS/`
so the archived MKV and SRTs can still be rechecked safely.

Google Drive files in Trash continue to consume storage. The notebook never
empties all of Drive Trash because it may contain unrelated files; inspect and
empty Trash yourself after cleanup if you want the quota released immediately.

## V1 safe reruns and failures

- PREPARE writes `*.done.json` markers containing input and output hashes.
- Partial artifacts use temporary names and are renamed only after validation.
- Missing source captions do not stop ASR.
- Without a GPU, the notebook warns and can run a slower CPU fallback.
- A corrupt/incomplete translation ZIP, wrong episode, wrong schema, missing or
  duplicate UID, reordered output, shifted Turkish correction, empty text or
  final QA failure stops publication.
- Deterministic semantic QA binds corrected Turkish to same-block source
  evidence and checks Indonesian names, numbers and religious anchors. An
  anchor-free Indonesian-only semantic swap cannot be proven without a
  prohibited translation model/API; the Work contract's per-record self-check
  and first-episode human review cover that irreducible language-quality risk.
- FINALIZE is rerunnable; it rechecks every input hash before accepting an
  existing result and again immediately before publication. Existing outputs
  are validated before reuse.
- If `CREATE_MKV` is turned off for a successful rerun, any older canonical MKV
  is retired so it cannot silently retain subtitles from a previous translation.
- The system deliberately avoids a separate diarization model. Speaker-turn
  protection uses silence, punctuation, dialogue markers, Whisper boundaries
  and risk flags; ambiguous boundaries are sent to the review sheet.

## V1 advanced settings

Ordinary use requires only the variables above. Optional cells expose model,
batch size, MKV creation and strict QA settings. Keep `schema_version` unchanged
for an episode whose translation has begun. Any segmentation-rule change must
use a new schema version.
