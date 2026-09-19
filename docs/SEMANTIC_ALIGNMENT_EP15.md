# EP15+ semantic block alignment

## Authority split

- Timing authority: immutable `work/semantic_alignment/word_timeline.jsonl`.
- Linguistic authority: ChatGPT Pro file handoff for unresolved semantic windows.
- Safety authority: Python pack/return validator and release QA.
- Translation authority: the existing immutable block-UID Indonesian workflow.
- Optional legacy policy: explicit `strict-ctc-v1`; it is never called automatically by semantic mode.

Semantic release identity is `alignment_policy=semantic-block-v1`,
`timing_source=semantic_word_span_v1`, and `strict_ctc_pass=false`.
`semantic_alignment_pass` and `release_eligible` are separate gates.

## Run contract and artifacts

New EP15+ initialization persists `delivery_scope` separately from
`alignment_policy`. `MAS_PRODUCTION_PRIORITY=first-hour-v1` and
`whole-episode-v1` select publication scope from the same canonical
`final_blocks.jsonl`; they do not change alignment quality policy. Existing
episode state without this contract is not adopted as semantic state.

The semantic stages write exact input/output/code/config-bound markers:

- `semantic_word_timeline.done.json`
- `semantic_prepare.done.json`
- `semantic_return.done.json`
- `semantic_finalize.done.json`
- `final/semantic_release.done.json`

Known subtitle credits are retained in raw evidence but excluded through
`quarantine.json` before handoff. Safe exact contiguous matches become
`AUTO_PASS`. All other windows enter one ZIP with disjoint owned words and
overlapping read-only context.

## Operator flow

1. Run `./mas run 15` or `./mas.ps1 run 15`.
2. On `SEMANTIC_ALIGNMENT_HANDOFF_REQUIRED`, use the exact displayed
   `translation_input/..._SEMANTIC_ALIGNMENT_PACK.zip`.
3. In ChatGPT Pro, follow `SEMANTIC_ALIGNMENT_INSTRUCTIONS.md` and return one
   `..._SEMANTIC_ALIGNMENT_RETURN.zip`. Do not add timestamps.
4. Put it at the exact displayed `translation_output` path and run the same
   command again.
5. Complete the existing ID translation ZIP handoff when requested, then run the
   same command again.
6. The controller encodes/publishes only after semantic finalization and the
   existing byte/SHA-256 delivery gates.

Exit code 29 is a verified human-handoff pause. The controller collects the pack,
shuts down and externally verifies the Pod, then resumes semantic/ID work locally.
No semantic failure automatically opens a new GPU or enters CTC recovery.

`./mas status 15` reports `SEMANTIC_PREPARED`,
`WAITING_FOR_SEMANTIC_RETURN`, `SEMANTIC_RETURN_VALIDATED`, or
`SEMANTIC_FINALIZED` from stored stage evidence.

## Explicit component fallback

A returned block with `review_required=true` remains non-releasable. Only that
window may be replaced by a source/timeline/word-span-bound record in
`review/semantic_coarse_fallback_approvals.json`. Its `approval_sha256` and
`approval_auth_tag` must bind the exact approval body through the existing raw-ASR
authentication key. Production release additionally requires the explicit
operator policy `MAS_ALLOW_APPROVED_COARSE_SEMANTIC_RELEASE=1`. The receipt then
keeps `strict_ctc_pass=false`, sets `semantic_alignment_pass=false`, and records
the approved fallback count and `approved_coarse_component_fallback_v1` timing
source. Unapproved automatic coarse fallback does not exist.

## Safety invariants

The return must preserve manifest, source, word-timeline, producer, window and
batch identity. Every block is ordered, contiguous and limited to its owned word
IDs. Unknown IDs, context theft, duplicate/missing/reordered windows, timestamps,
metadata, hard gaps, known speaker crossings and known scene crossings fail
closed. All eligible words must be assigned once or supported by exact quarantine
evidence before normal release. TR and ID retain identical block UID, order and
timing.
