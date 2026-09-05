# Muhtemel Ask Subtitle System V1 - Test Report

Validation date: 2026-09-01 UTC

## Overall result: PREPARE PASS / TRANSLATION PASS / FINALIZE PASS / ARCHIVE PASS

Every deterministic test available in this environment passed. Episode 11 also
completed the full live PREPARE pipeline on a Google Colab T4 with DriveFS,
faster-whisper `large-v3`, targeted verification, locked segmentation and a
validated translation pack. The Work Ultra Turkish correction and Indonesian
translation also completed, passed independent cross-review and passed the
production FINALIZE validator. The live Episode 11 FINALIZE run completed with
hard QA PASS; human review and Infuse playback remain as viewing checks.

## Automated suite

Command:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Result: **141/141 PASS; 0 failures; 0 errors; 0 skips**.

Episode 10 semantic-validator regression coverage (5 methods):

1. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_isolated_remote_exact_text_is_not_position_evidence`
2. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_turkish_name_suffixes_preserve_canonical_identity`
3. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_reviewed_source_ambiguity_passes_but_tr_id_name_parity_stays_strict`
4. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_review_flag_without_note_does_not_bypass_source_evidence`
5. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_apostrophized_rabbim_and_vallahi_oath_align`

Episode 10 final-QA agreement regression coverage (2 methods):

1. `test_religious_terms.ReligiousAndAnchorTests.test_review_flag_alone_cannot_bypass_primary_name_anchor`
2. `test_religious_terms.ReligiousAndAnchorTests.test_trusted_reviewed_source_name_correction_passes_final_qa`

Episode 10 dotted-initialism regression coverage (3 methods):

1. `test_transcribe.TranscriptionCoverageTests.test_changed_dotted_initial_still_hard_fails`
2. `test_transcribe.TranscriptionCoverageTests.test_missing_leading_dotted_initial_is_repaired_in_place`
3. `test_transcribe.TranscriptionCoverageTests.test_non_initialism_partial_word_still_hard_fails`

New archive coverage (17 methods):

1. `test_episode_archive.EpisodeArchiveFFmpegIntegrationTests.test_real_stream_copy_archive_and_media_only_cleanup`
2. `test_episode_archive.EpisodeArchiveTests.test_completed_keep_source_archive_can_later_remove_exact_source`
3. `test_episode_archive.EpisodeArchiveTests.test_descendant_symlink_and_special_file_are_rejected`
4. `test_episode_archive.EpisodeArchiveTests.test_dry_run_deletes_nothing_and_writes_ready_receipt`
5. `test_episode_archive.EpisodeArchiveTests.test_episode_root_symlink_is_rejected`
6. `test_episode_archive.EpisodeArchiveTests.test_exact_episode_root_and_confirmation_are_required`
7. `test_episode_archive.EpisodeArchiveTests.test_invalid_present_report_never_falls_back_to_receipt`
8. `test_episode_archive.EpisodeArchiveTests.test_keep_source_video_disappearance_blocks_cleanup`
9. `test_episode_archive.EpisodeArchiveTests.test_keep_source_video_policy_preserves_source`
10. `test_episode_archive.EpisodeArchiveTests.test_mutated_plan_cannot_add_a_protected_mkv_to_deletion`
11. `test_episode_archive.EpisodeArchiveTests.test_notebook_contract_compiles_and_prompts_after_archive_preview`
12. `test_episode_archive.EpisodeArchiveTests.test_pass_report_identity_size_and_hash_are_required`
13. `test_episode_archive.EpisodeArchiveTests.test_plan_preserves_archive_media_and_deletes_intermediates`
14. `test_episode_archive.EpisodeArchiveTests.test_project_ancestor_and_dangling_receipt_symlinks_are_rejected`
15. `test_episode_archive.EpisodeArchiveTests.test_ready_receipt_recovers_a_partial_cleanup`
16. `test_episode_archive.EpisodeArchiveTests.test_successful_cleanup_keeps_only_mkv_and_srts`
17. `test_episode_archive.EpisodeArchiveTests.test_wrong_confirmation_and_tree_change_delete_nothing`

Baseline pipeline coverage (114 methods):

1. `test_archive_safety.ArchiveSafetyTests.test_extra_media_is_rejected_before_crc_or_read`
2. `test_archive_safety.ArchiveSafetyTests.test_oversize_metadata_is_rejected_before_crc_or_read`
3. `test_archive_safety.ArchiveSafetyTests.test_suspicious_ratio_metadata_is_rejected_before_crc_or_read`
4. `test_archive_safety.ArchiveSafetyTests.test_translated_zip_rejects_zero_record_batch`
5. `test_archive_safety.ArchiveSafetyTests.test_translation_pack_rejects_zero_record_batch`
6. `test_download_media.DownloadResumeTests.test_corrupt_optional_caption_is_dropped_when_retry_is_unavailable`
7. `test_download_media.DownloadResumeTests.test_hash_valid_marker_cannot_resume_a_truncated_source`
8. `test_download_media.DownloadResumeTests.test_interrupted_marker_commit_leaves_no_uncommitted_final_source`
9. `test_download_media.DownloadResumeTests.test_legacy_marker_without_eof_evidence_does_not_resume`
10. `test_download_media.DownloadResumeTests.test_missing_caption_retries_without_redownloading_video`
11. `test_download_media.DownloadResumeTests.test_publish_removes_only_stale_workflow_source_artifacts`
12. `test_download_media.DownloadResumeTests.test_stable_workspace_reuses_partial_after_interruption`
13. `test_download_media.MediaIntegrityTests.test_audio_duration_tolerance_rejects_one_second_loss`
14. `test_download_media.MediaIntegrityTests.test_audio_marker_without_alignment_evidence_is_rejected`
15. `test_download_media.MediaIntegrityTests.test_audio_preserves_nonzero_source_playback_offset`
16. `test_download_media.MediaIntegrityTests.test_supplied_source_digest_must_match_current_bytes`
17. `test_download_media.MediaIntegrityTests.test_truncated_faststart_source_fails_eof_validation`
18. `test_e2e.SyntheticWorkflowTests.test_prepare_pack_to_translated_zip_to_final_srt_and_review`
19. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_all_code_cells_compile`
20. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_input_hash_guard_covers_resume_publish_and_commit`
21. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_non_mkv_run_cannot_accept_or_leave_a_canonical_mkv`
22. `test_mux.MKVRoundTripTests.test_24_mkv_subtitle_extraction_round_trip_and_stream_copy`
23. `test_mux.MKVRoundTripTests.test_h264_decode_reordering_does_not_shift_subtitle_presentation`
24. `test_religious_terms.ReligiousAndAnchorTests.test_15_required_religious_phrase_preservation`
25. `test_religious_terms.ReligiousAndAnchorTests.test_16_allah_cannot_be_replaced_only_with_semoga`
26. `test_religious_terms.ReligiousAndAnchorTests.test_17_special_name_spelling_is_enforced`
27. `test_religious_terms.ReligiousAndAnchorTests.test_18_numeric_and_money_values_cannot_change`
28. `test_review.ReviewSelectionTests.test_explicit_rows_are_never_dropped_when_they_exceed_cap`
29. `test_review.ReviewSelectionTests.test_five_percent_is_total_cap_when_explicit_rows_fit`
30. `test_schema.BlockIdentityTests.test_01_stable_block_uid_generation`
31. `test_schema.BlockIdentityTests.test_02_same_input_creates_same_block_uid`
32. `test_schema.BlockIdentityTests.test_03_changed_timing_creates_different_uid`
33. `test_schema.BlockIdentityTests.test_04_schema_sha_consistency_and_tamper_detection`
34. `test_schema.BlockIdentityTests.test_adjacent_overlap_hard_fails_build_and_tamper_validation`
35. `test_schema.BlockIdentityTests.test_missing_per_block_episode_cannot_be_hidden_by_recomputed_sha`
36. `test_schema.BlockIdentityTests.test_schema_rejects_translation_fields`
37. `test_schema.BlockIdentityTests.test_schema_version_changes_block_identity`
38. `test_segmentation_drift.SegmentationDriftRegressionTests.test_11_old_2505_block_schema_cannot_be_reused_for_2470_blocks`
39. `test_segmentation_drift.SegmentationDriftRegressionTests.test_26_translation_batch_from_wrong_episode_hard_fails`
40. `test_segmentation_drift.SegmentationDriftRegressionTests.test_27_translation_batch_from_wrong_schema_version_hard_fails`
41. `test_segmentation_drift.SegmentationDriftRegressionTests.test_explicit_dialogue_turn_starts_a_new_block`
42. `test_segmentation_drift.SegmentationDriftRegressionTests.test_segmentation_preserves_all_words_and_hard_silence_boundaries`
43. `test_segmentation_drift.SegmentationDriftRegressionTests.test_segmentation_validator_rejects_mixed_turn_and_internal_gap`
44. `test_segmentation_drift.SegmentationDriftRegressionTests.test_short_whisper_segment_change_is_a_neutral_hard_boundary`
45. `test_srt.SRTTests.test_19_overlap_is_reported_and_never_silently_retimed`
46. `test_srt.SRTTests.test_20_more_than_two_visible_lines_fails_qa`
47. `test_srt.SRTTests.test_21_line_over_84_characters_fails_qa`
48. `test_srt.SRTTests.test_22_utf8_srt_round_trip`
49. `test_srt.SRTTests.test_23_timing_round_trip_is_exact_to_one_millisecond`
50. `test_srt.SRTTests.test_25_empty_translation_is_rejected`
51. `test_transcribe.TranscriptionCoverageTests.test_nonempty_segment_without_timed_words_hard_fails`
52. `test_transcribe.TranscriptionCoverageTests.test_normalized_punctuation_and_spacing_preserve_full_coverage`
53. `test_transcribe.TranscriptionCoverageTests.test_persisted_transcription_rejects_silent_segment_loss`
54. `test_transcribe.TranscriptionCoverageTests.test_segment_text_replacement_still_hard_fails`
55. `test_translation_alignment.TranslationContractTests.test_05_missing_translation_uid_hard_fails`
56. `test_translation_alignment.TranslationContractTests.test_06_duplicate_translation_uid_hard_fails`
57. `test_translation_alignment.TranslationContractTests.test_07_extra_translation_uid_hard_fails`
58. `test_translation_alignment.TranslationContractTests.test_08_reordered_translation_records_hard_fail`
59. `test_translation_alignment.TranslationContractTests.test_09_changed_schema_sha_hard_fails`
60. `test_translation_alignment.TranslationContractTests.test_10_wrong_block_number_mapping_hard_fails`
61. `test_translation_alignment.TranslationContractTests.test_12_shifted_translation_beginning_at_block_12_is_detected`
62. `test_translation_alignment.TranslationContractTests.test_13_split_block_attempt_is_rejected`
63. `test_translation_alignment.TranslationContractTests.test_14_merge_block_attempt_is_rejected`
64. `test_translation_alignment.TranslationContractTests.test_valid_translation_contract_exposes_uid_map`
65. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_anchor_free_id_final_is_not_pseudo_aligned_cross_lingually`
66. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_conflicting_name_candidates_allow_one_supported_correction`
67. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_conflicting_religious_candidates_do_not_impose_their_union`
68. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_id_final_shift_is_detected_from_preserved_anchors`
69. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_invented_quantity_and_known_name_are_rejected`
70. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_known_asr_name_variant_can_be_corrected_to_canonical_spelling`
71. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_numeric_tokens_preserve_decimal_and_whitespace_boundaries`
72. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_supported_correction_is_not_rejected_for_matching_other_block`
73. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_thousands_separator_formatting_can_change_without_changing_value`
74. `test_segmentation_drift.SegmentationDriftRegressionTests.test_synthetic_timing_words_are_not_split_into_impossible_cues`
75. `test_transcribe.TranscriptionCoverageTests.test_combined_timed_word_is_split_before_omission_repair`
76. `test_transcribe.TranscriptionCoverageTests.test_internal_omission_without_gap_preserves_text_order`
77. `test_transcribe.TranscriptionCoverageTests.test_internal_omissions_get_explicit_synthetic_timing`
78. `test_transcribe.TranscriptionCoverageTests.test_missing_leading_word_gets_explicit_synthetic_timing`
79. `test_transcribe.TranscriptionCoverageTests.test_missing_trailing_word_gets_explicit_synthetic_timing`
80. `test_transcribe.TranscriptionCoverageTests.test_trailing_omission_without_gap_preserves_text_order`
81. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_name_variant_maps_stay_separate_across_notebooks`
82. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_ambiguous_source_variant_allows_literal_and_name_readings`
83. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_apostrophe_and_joined_religious_source_forms_are_equivalent`
84. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_bartiner_remains_forbidden_in_final_output`
85. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_default_source_variants_match_names_config`
86. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_explicit_empty_source_variants_do_not_use_defaults`
87. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_overlapping_source_variant_preserves_literal_name_reading`
88. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_religious_source_normalization_keeps_id_mapping_strict`
89. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_source_name_variant_correction_is_not_misaligned_to_exact_name`
90. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_source_name_variant_is_not_automatically_forbidden_in_final`
91. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_source_name_variant_supports_canonical_final_name`
92. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_source_variants_support_natural_turkish_address_forms`
93. `test_translation_validation_regressions.TranslationValidationRegressionTests.test_zip_loader_forwards_source_name_variants`
94. `test_religious_terms.ReligiousAndAnchorTests.test_verification_or_context_name_is_not_a_current_block_anchor`
95. `test_religious_terms.ReligiousAndAnchorTests.test_same_block_primary_name_remains_a_required_anchor`
96. `test_download_media.DownloadResumeTests.test_episode_named_publish_retires_only_legacy_and_matching_stem_artifacts`
97. `test_download_media.DownloadResumeTests.test_episode_named_source_resumes_with_marker_verified_path`
98. `test_download_media.DownloadResumeTests.test_legacy_source_marker_still_resumes_when_episode_stem_is_requested`
99. `test_download_media.DownloadResumeTests.test_output_stem_rejects_paths_and_media_extensions`
100. `test_download_media.DownloadResumeTests.test_cookie_path_does_not_change_resume_identity`
101. `test_download_media.DownloadResumeTests.test_cookie_reaches_media_and_caption_without_persisting_secret`
102. `test_download_media.DownloadResumeTests.test_cookie_reaches_resumed_caption_probe_and_download`
103. `test_download_media.DownloadResumeTests.test_invalid_cookie_file_fails_before_yt_dlp`
104. `test_download_media.DownloadResumeTests.test_missing_cookie_is_not_required_for_verified_marker_resume`
105. `test_download_media.DownloadResumeTests.test_youtube_bot_auth_error_fails_fast`
106. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_prepare_uses_drive_backed_cookie_fallback`
107. `test_download_media.DownloadResumeTests.test_authenticated_caption_failure_redacts_secret`
108. `test_download_media.DownloadResumeTests.test_authenticated_media_failure_redacts_secret`
109. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_anonymous_success_never_reads_or_writes_cookie`
110. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_cookie_helpers_copy_saved_drive_cookie_to_private_runtime`
111. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_cookie_helpers_reject_non_youtube_saved_file_without_leak`
112. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_cookie_save_failure_redacts_path_and_secret`
113. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_rejected_fresh_cookie_is_not_persisted_or_retried`
114. `test_finalize_notebook_contract.FinalizeNotebookContractTests.test_stale_saved_cookie_falls_back_once_and_persists_fresh_cookie`

## Episode 11 live PREPARE acceptance

- Official full episode source acquired from Dailymotion after the equivalent
  YouTube URL was blocked from Colab; duration **2:17:58**.
- Source MP4 and extracted FLAC passed the hash/marker resume checks after a
  clean runtime restart.
- T4 `large-v3` Turkish ASR and bounded targeted verification completed and
  resumed from its verified marker on the final clean run.
- Locked segmentation: **2,816 blocks**, **0 overlaps**, **0 invalid timings**,
  **0 source-text mismatches**, **0 validator warnings**.
- Schema SHA-256:
  `23c3e1e85447633e4fd040e977296d66edc27c25b9826a9263a108ffcaf79ba2`.
- Translation pack: **7 JSONL batches**, **11 ZIP members**, CRC PASS,
  `ZipFile.testzip()` returned `None`.
- Translation-pack SHA-256:
  `b1a6b756940796b84e87ce520df6aa6bed046ee8328123ef1d1097462c230b21`.
- Translated ZIP: **2,816 Turkish + 2,816 Indonesian records**, **8 ZIP
  members**, CRC PASS, UID/order/schema PASS, and every production semantic
  mismatch counter at zero.
- Translated-ZIP SHA-256:
  `ede1f3bb20623e85647a1ca0a275cdb6970b8bb5b0c5aa5cebfcb9e952034c87`.
- Independent second-pass language QA corrected **149** definite ASR,
  same-block alignment, Indonesian meaning and register issues. **316**
  uncertain rows remain explicitly marked for the FINALIZE review workbook.
- The exact Episode 11 translated records also passed the complete FINALIZE
  subtitle QA preflight: **0 special-name mismatches**, **0 anchor mismatches**,
  **316 review-required rows**, and **1,556 non-blocking high-CPS warnings**.
  A regression now prevents adjacent speech present only in targeted
  `verification_text` or context from becoming a false hard name anchor.
- The live Episode 11 FINALIZE publication completed: **2,816 blocks**, **0
  positional mismatches**, **0 overlaps**, and final hard QA **PASS**.
- Source video and Infuse sidecars now use the episode-labelled stem (for
  example, `Muhtemel Ask 11.Bolum.mp4` and `Muhtemel Ask 11.Bolum-tr.srt`).
  Legacy `source.*` workspaces remain resumable without a redownload.

## Additional validation executed

- `python -m compileall -q src tests`: **PASS**.
- `01_PREPARE.ipynb`: JSON parse **PASS**; **8/8** code cells compile.
- `02_FINALIZE.ipynb`: JSON parse **PASS**; **7/7** code cells compile.
- `03_ARCHIVE_CLEANUP.ipynb`: JSON parse **PASS**; **5/5** code cells compile.
- YouTube authentication fallback: exact bot challenge fails fast after one
  anonymous attempt; a private Drive cookie is copied into a random restricted
  runtime directory before yt-dlp sees it; a stale saved credential falls back
  once to a fresh upload; rejected fresh cookies never overwrite Drive; cookie
  path/content never enters metadata or markers; changing or deleting the
  credential does not invalidate a completed source marker: **PASS**.
- Actual non-Colab-runtime cells of `02_FINALIZE.ipynb` executed on a synthetic
  ten-block episode with a real H.264/AAC source, locked schema, validated
  translation pack and translated ZIP: **PASS**. Canonical SRTs, matching
  Infuse sidecars, review workbook and report were published; an immediate
  hash/mtime-verified rerun replaced nothing.
- Synthetic translated-ZIP replacement after validation: rejected before
  staging; no outputs changed. Replacement during workbook staging: rejected
  before publication; the prior commit remained intact and staged files were
  cleaned.
- Stale canonical MKV followed by `CREATE_MKV=False`: stale MKV retired and a
  five-output non-MKV PASS report produced; immediate resume replaced nothing.
- Real ffmpeg H.264 B-frame/AAC and Matroska fixtures: ID first/default, TR
  second/non-default, SRT extraction exact to 1 ms and text, video/audio packet
  hashes unchanged: **PASS**.
- Real FFmpeg post-finalization fixture: stream-copy MKV creation, external
  READY/PASS receipt, exact-plan media-only cleanup, source-video replacement,
  retained SRT verification and post-clean no-op rerun: **PASS**.
- Review XLSX: exact `Kontrol!A1:J4` values/headers inspected, formula-error
  scan returned zero matches, and a PNG render was visually checked: **PASS**.
- Episode 10 production translation ZIP: **3,153/3,153 TR and ID records;
  schema, UID, order, batches, timing, names, numbers, religious expressions
  and positional validation all PASS; exact final-QA replay PASS with
  `special_name_mismatch_count=0`, `anchor_mismatch_count=0`, and 122 explicit
  review rows retained**.
- Final release ZIP: **38 file members; CRC PASS; `ZipFile.testzip()` returned
  `None`; no media, `.pyc`, or `__pycache__` members; required-member check
  PASS; all three notebooks parsed from the archive; fresh extraction compile
  and 141/141 test rerun PASS**.

## Remaining live acceptance

The following are not yet marked PASS:

1. Stable yt-dlp `.part` continuation after an actual VM loss and the CPU
   `int8` fallback on a runtime where GPU initialization fails.
2. Direct YouTube acquisition from Colab against current blocking behavior,
   including the first live save/reuse of `PRIVATE/youtube-cookies.txt`;
   Episode 11 used the matching official Dailymotion source instead.
3. Listening-based human assessment of flagged ASR/timing, short speaker turns,
   music/lyrics, names and religious expressions. Speaker protection is
   heuristic rather than acoustic diarization.
4. Human review of the 316 explicitly flagged rows and any anchor-free
   Indonesian alignment risk, plus Infuse on Apple TV discovery/playback of
   both external sidecars and optional soft-sub MKV.
5. Live Episode 11 execution of `03_ARCHIVE_CLEANUP.ipynb`, followed by manual
   Infuse playback and user-controlled Google Drive Trash emptying.

