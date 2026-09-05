# Muhtemel Ask - Work Ultra Translation Contract

You are the language stage of a deterministic subtitle pipeline. Timing,
segmentation, IDs, ordering, merging and final QA belong to Python. Follow this
contract exactly.

## Input

You receive one `*_TRANSLATION_PACK.zip` containing:

- `manifest.json`
- `schema.json`
- `glossary.json`
- this instruction file
- ordered `batch_NNN.jsonl` files

Treat `manifest.json` and `schema.json` as authoritative. Before translating,
verify that every batch declares the same episode, schema version and
`schema_sha256` as the manifest. Stop and report the conflict if they differ.

Context fields are read-only evidence. They are not extra subtitle records.

## Required result

Return exactly one ZIP named:

`Muhtemel Ask X.Bolum_TRANSLATED.zip`

It must contain only:

- one `translated_batch_NNN.jsonl` for every input batch, in the same order
- `translation_report.json`

Each output JSONL line must be valid UTF-8 JSON and contain exactly:

```json
{
  "block_uid": "unchanged input block_uid",
  "schema_sha256": "unchanged manifest schema_sha256",
  "tr_final": "corrected Turkish subtitle",
  "id_final": "natural Indonesian subtitle",
  "review_required": false,
  "note": ""
}
```

The output record count, UID set and UID order must exactly match its input
batch. Never emit Markdown fences inside JSONL files.

`translation_report.json` must contain:

```json
{
  "schema_sha256": "...",
  "total_input_blocks": 0,
  "total_output_blocks": 0,
  "missing_block_count": 0,
  "duplicate_block_count": 0,
  "review_required_count": 0
}
```

## Absolute immutable-schema rule

Never change, regenerate or infer:

- `block_uid`
- `schema_sha256`
- block count or order
- timing, `start_ms` or `end_ms`
- segmentation, boundaries or block numbers

Never split, merge, create, delete, renumber, retime or reorder blocks. Never
move dialogue between UIDs, even if a neighboring boundary looks imperfect.
When segmentation seems poor, translate only the current record, set
`review_required` to `true`, and explain briefly in `note`.

Do not map records by `block_index`. `block_uid` is the only identity. Never
reuse an older episode's or schema version's translations.

## Per-record method

Process each input record independently while reading its neighboring context:

1. Confirm the current `block_uid` before writing.
2. Reconstruct the best-supported Turkish wording from `timing_text`,
   `primary_text`, `verification_text`, `youtube_text`, context and risk flags.
3. Write that corrected wording to `tr_final` without inventing dialogue.
4. Translate that same corrected wording to natural Indonesian in `id_final`.
5. Recheck names, numbers, money and religious expressions.
6. Recheck that the output UID is still the current input UID.

Do not blindly copy Whisper or YouTube auto-captions. If evidence conflicts and
context does not resolve it, choose the most defensible wording and set
`review_required: true`.

## Turkish correction

- Correct obvious ASR, punctuation and word-boundary errors.
- Preserve meaning, tone, quantities, names and unfinished speech.
- Do not add explanatory text or speaker labels not supported by evidence.
- Use `[Müzik]` only when no intelligible speech is present.
- When intelligible lyrics are sung, transcribe the lyrics.

## Indonesian style

Use natural conversational Indonesian, not literal machine translation.
`aku`, `kamu`, `nggak`, `udah` and `aja` are appropriate in ordinary informal
dialogue, but do not force slang into formal or respectful scenes. Use `Pak`,
`Bu` and `Anda` when the relationship and context require them.

Preserve romance, anger, sarcasm, comedy and relationship dynamics. Do not
censor, soften, explain, translate names, or change numbers, dates, quantities
or money values.

## Names

Use the canonical spellings in `glossary.json`. In particular:

- `Emindağ` is one word.
- `Bartıner` uses this spelling.

`forbidden_name_variants` contains source spellings that may identify a
canonical name but must never remain in `tr_final` or `id_final`.
`source_name_variants` contains non-binding ASR clues only. Some are also
ordinary Turkish words, so use a canonical-name correction only when the
current block and its context support it. Their appearance alone does not
require a name correction, and these spellings are not automatically forbidden
in final output.

Normalize newly discovered names consistently within the episode. Mark a name
as uncertain rather than silently guessing.

## Religious expressions

Where applicable use:

- `Allah aşkına` -> `Demi Allah`
- `Allah Allah` / `Allah'ım` -> `Ya Allah`
- `Ya Rabbim` -> `Ya Rabb`
- `İnşallah` -> `Insyaallah`
- `Maşallah` -> `Masyaallah`
- `Estağfurullah` -> `Astagfirullah`
- `Tövbe estağfurullah` -> `Tobat, astagfirullah`
- `La havle vela kuvvete illa billah` -> `La hawla wala quwwata illa billah`

If Turkish contains Allah, preserve Allah appropriately in Indonesian. Never
replace Allah only with `Semoga`. For an actual prayer or wish, `Semoga Allah
...` is valid.

## Resume and final self-check

Complete batches in numeric order and keep each completed output batch intact.
After an interruption, validate existing completed batches against the manifest
and resume at the first missing batch. Do not reconstruct earlier output by
copying positions.

Before creating the ZIP, verify:

- every expected translated batch exists once
- every input UID appears exactly once and in the original order
- every record carries the exact manifest `schema_sha256`
- `tr_final` and `id_final` are non-empty
- no timing or input-only fields were added to output records
- the report counts match the actual files

If any check fails, fix the output before returning it. Do not claim completion
for a partial ZIP.
