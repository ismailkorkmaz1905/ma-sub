from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from mas.engine.download import sha256_file, write_stage_marker
import mas.engine.raw_asr as raw_asr_v2_module
from mas.engine.raw_asr import (
    RawASRV2Config,
    build_asr_hallucination_records,
    build_correction_utterances,
    build_coverage_intervals,
    build_speech_hole_records,
    effective_rescue_span_limit,
    include_orphan_youtube_captions_for_correction,
    include_speech_holes_for_correction,
    load_valid_raw_asr_v2,
    consume_coarse_segments,
    merge_rescue_evidence,
    transcribe_raw_audio_v2,
    validate_persisted_raw_asr_v2,
    validate_publishable_raw_asr_v2_policy,
)
from mas.engine.speech_coverage import analyze_speech_coverage
from mas.engine.tr_correction import (
    create_tr_correction_pack,
    read_tr_correction_pack,
    validate_input_utterances,
)
from mas.engine.transcribe import TranscriptionError


@dataclass
class Word:
    start: float | None
    end: float | None
    word: str
    probability: float = 0.9


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]
    avg_logprob: float | None = -0.10
    no_speech_prob: float | None = 0.05
    compression_ratio: float | None = 1.10
    temperature: float | None = 0.0


def _fake_extract_clip(
    _audio_path: Path, output_path: Path, start_ms: int, end_ms: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(
            b"\x00\x00" * round(16_000 * (end_ms - start_ms) / 1000)
        )


@dataclass
class Info:
    language: str = "tr"


class FakeCTranslate2:
    @staticmethod
    def get_cuda_device_count() -> int:
        return 1


class FakeCPUOnlyCTranslate2:
    @staticmethod
    def get_cuda_device_count() -> int:
        return 0


def _complete_segment(text: str = "Merhaba") -> Segment:
    return Segment(0.0, 1.0, text, [Word(0.0, 1.0, f" {text}")])


class RawASRV2Tests(unittest.TestCase):
    def test_incomplete_coarse_parent_is_bounded_without_omitting_untimed_residual(self) -> None:
        words = [dict(start_ms=1000 + index * 1200, end_ms=2000 + index * 1200, text=" kelime")
                 for index in range(10)]
        segment = {"segment_id": "main-1000", "start_ms": 0, "end_ms": 184000,
                   "text": "Başka ve eksik kaynak metni", "words": words,
                   "word_timing_complete": True}
        islands = raw_asr_v2_module.required_acoustic_review_regions([segment], [])
        self.assertEqual([(item["start_ms"], item["end_ms"]) for item in islands],
                         [(start, min(start + 10000, 184000)) for start in range(0, 184000, 10000)])
        self.assertTrue(all(item["end_ms"] - item["start_ms"] <= 10000 for item in islands))
        self.assertEqual(segment["end_ms"], 184000)
        untimed = raw_asr_v2_module.required_acoustic_review_regions([dict(segment, words=[])], [])
        self.assertEqual([(item["start_ms"], item["end_ms"]) for item in untimed],
                         [(item["start_ms"], item["end_ms"]) for item in islands])
        self.assertEqual(sum(item["end_ms"] - item["start_ms"] for item in untimed), 184000)

    def test_outside_complement_keeps_partial_vad_word_runs_without_duplicate_islands(self) -> None:
        segment = {"segment_id": "incomplete", "start_ms": 0, "end_ms": 9000,
                   "text": "Eksik başka metin", "words": [
                       {"start_ms": 0, "end_ms": 500, "text": " Bir"},
                       {"start_ms": 700, "end_ms": 900, "text": " iki"},
                       {"start_ms": 1100, "end_ms": 1300, "text": " üç"},
                       {"start_ms": 3000, "end_ms": 3200, "text": " dört"},
                   ]}
        vad = [{"start_ms": 400, "end_ms": 600}, {"start_ms": 950, "end_ms": 1050}]
        islands = raw_asr_v2_module.required_acoustic_review_regions([segment], vad)
        self.assertEqual([(r["start_ms"], r["end_ms"]) for r in islands],
                         [(0, 400), (600, 950), (1050, 9000)])
        all_required = raw_asr_v2_module.required_speech_coverage_regions([segment], vad)
        self.assertEqual([(r["start_ms"], r["end_ms"]) for r in all_required],
                         [(0, 9000)])
        partial = dict(segment, start_ms=400, end_ms=600, words=[{"start_ms": 0, "end_ms": 500, "text": " Bir"}])
        self.assertEqual([(r["start_ms"], r["end_ms"]) for r in
                          raw_asr_v2_module.required_acoustic_review_regions([partial], vad)], [(0, 400)])

    def test_only_mandatory_evidence_can_use_hard_budget_headroom(self) -> None:
        spans = [{"start_ms": index * 2000, "end_ms": index * 2000 + 1000}
                 for index in range(3)]
        with self.assertRaisesRegex(TranscriptionError, "cannot fit"):
            raw_asr_v2_module._bounded_rescue_batches(spans, limit=2, hard_limit=3)
        spans[-1]["required_source_records"] = [{"source_id": "incomplete", "source_sha256": "a" * 64}]
        batches = raw_asr_v2_module._bounded_rescue_batches(spans, limit=2, hard_limit=3)
        self.assertEqual(len(batches), 3)
        self.assertEqual(raw_asr_v2_module._rescue_budget_audit(batches, limit=2, hard_limit=3), {
            "adaptive_target": 2, "hard_limit": 3, "mandatory_hard_limit": 3, "actual_batch_count": 3,
            "mandatory_batch_count": 1, "mandatory_overflow_count": 1,
            "mandatory_overflow_reason": "preserve_required_incomplete_source_evidence",
        })
        with self.assertRaisesRegex(TranscriptionError, "above hard limit"):
            raw_asr_v2_module._bounded_rescue_batches(spans, limit=2, hard_limit=2)
        too_many_ordinary = spans + [{"start_ms": 9000, "end_ms": 10000}]
        with self.assertRaisesRegex(TranscriptionError, "cannot fit"):
            raw_asr_v2_module._bounded_rescue_batches(too_many_ordinary, limit=2, hard_limit=4)

    def test_outside_island_batch_bound_survives_generic_longer_batches(self) -> None:
        spans = [{"start_ms": 0, "end_ms": 13000},
                 {"start_ms": 20000, "end_ms": 28000, "max_duration_ms": 10000,
                  "required_source_records": [{"source_id": "outside", "source_sha256": "a" * 64}]},
                 {"start_ms": 28000, "end_ms": 31000}]
        batches = raw_asr_v2_module._bounded_rescue_batches(spans, limit=2, hard_limit=3)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(batch["duration_ms"] <= 10000 for batch in batches if "max_duration_ms" in batch))

    def test_mandatory_absolute_ceiling_does_not_unlock_ordinary_192_cap(self) -> None:
        spans = [{"start_ms": index * 2000, "end_ms": index * 2000 + 1000,
                  "required_source_records": [{"source_id": "mandatory", "source_sha256": "a" * 64}]}
                 for index in range(256)]
        batches = raw_asr_v2_module._bounded_rescue_batches(
            spans, limit=173, hard_limit=192, mandatory_hard_limit=256)
        self.assertEqual(len(batches), 256)
        with self.assertRaisesRegex(TranscriptionError, "257 spans, above hard limit 256"):
            raw_asr_v2_module._bounded_rescue_batches(
                spans + [dict(spans[-1], start_ms=600000, end_ms=601000)],
                limit=173, hard_limit=192, mandatory_hard_limit=256)
        ordinary = [{"start_ms": span["start_ms"], "end_ms": span["end_ms"]} for span in spans[:193]]
        with self.assertRaisesRegex(TranscriptionError, "193 spans, above hard limit 192"):
            raw_asr_v2_module._bounded_rescue_batches(
                ordinary, limit=173, hard_limit=192, mandatory_hard_limit=256)
        with self.assertRaisesRegex(ValueError, "256"):
            RawASRV2Config(rescue_mandatory_hard_max_spans=257)

    def test_coalescer_cannot_absorb_ordinary_span_into_mandatory_attribution(self) -> None:
        spans = [{"start_ms": 0, "end_ms": 10000},
                 {"start_ms": 11000, "end_ms": 11500},
                 {"start_ms": 11500, "end_ms": 12000,
                  "required_source_records": [{"source_id": "small", "source_sha256": "a" * 64}]}]
        result = raw_asr_v2_module._bounded_rescue_batches(spans, limit=2, hard_limit=192, mandatory_hard_limit=256)
        self.assertEqual(len(result), 3)
        self.assertEqual(sum(bool(item.get("required_source_records")) for item in result), 1)
        self.assertEqual(result[-1]["source_rescue_span_indices"], [3])

    def test_rescue_execution_proof_recomputes_policy_metrics_and_batch_identities(self) -> None:
        segments, words = consume_coarse_segments(iter([_complete_segment(), Segment(1, 3, "Eksik", [])]))
        vad = [{"start_ms": 0, "end_ms": 1000, "source": "silero_vad"}]
        settings = RawASRV2Config()
        initial = raw_asr_v2_module._analyze_raw_speech_coverage(vad, segments, words, config=settings.speech_coverage_config())
        batches, budget = raw_asr_v2_module._plan_rescue_batches(initial, vad, segments, words, settings)
        raw = {"model": {"settings": asdict(settings)}, "segments": segments, "words": words, "vad_regions": vad,
               "initial_speech_coverage": initial, "rescue_batches": batches, "rescue_budget_audit": budget}
        raw_asr_v2_module.validate_raw_rescue_plan(raw)
        changes = (
            lambda r: r["rescue_budget_audit"].update(adaptive_target=192),
            lambda r: r["rescue_budget_audit"].update(hard_limit=256),
            lambda r: r["rescue_budget_audit"].update(mandatory_hard_limit=999),
            lambda r: r["rescue_batches"][0].update(start_ms=0),
            lambda r: r["rescue_batches"][0].update(source_rescue_span_indices=[999]),
            lambda r: r["rescue_batches"][0].update(required_source_records=[]),
            lambda r: r["initial_speech_coverage"]["metrics"].update(independent_vad_speech_coverage_ratio=.1),
            lambda r: r["model"]["settings"].update(rescue_max_spans=192),
            lambda r: r.pop("rescue_budget_audit"),
            lambda r: r.pop("rescue_batches"),
        )
        for change in changes:
            changed = json.loads(json.dumps(raw))
            change(changed)
            with self.subTest(change=changes.index(change)), self.assertRaises(TranscriptionError):
                raw_asr_v2_module.validate_raw_rescue_plan(changed)

    def test_ep13_provisional_scene_gap_fragments_require_acoustic_review(self) -> None:
        fixtures = (
            ("main-465", (
                (1936030, 1936670, " Bugün"),
                (1973840, 1974480, " de"),
                (2098780, 2099280, " atlatacaksın"),
                (2099280, 2099840, " Defne'm."),
            ), ["Bugün", "de", "atlatacaksın Defne'm."]),
            ("main-2305", (
                (8428310, 8429150, " Sosyal"),
                (8472270, 8472830, " hizmetlerden"),
                (8472830, 8473150, " geldiler,"),
                (8473290, 8473770, " sizi"),
            ), ["Sosyal", "hizmetlerden geldiler, sizi"]),
        )
        flag = "provisional_word_gap_requires_audio_review"
        for segment_id, timed_words, expected in fixtures:
            with self.subTest(segment_id=segment_id), tempfile.TemporaryDirectory() as directory:
                segment = {
                    "segment_id": segment_id,
                    "start_ms": timed_words[0][0], "end_ms": timed_words[-1][1],
                    "text": "".join(word[2] for word in timed_words).strip(),
                    "word_timing_complete": True,
                    "words": [dict(start_ms=s, end_ms=e, text=t) for s, e, t in timed_words],
                    "asr_audit": dict(avg_logprob=-.1, no_speech_prob=.01,
                                      compression_ratio=1.1, temperature=0.),
                }
                records = build_correction_utterances([segment], episode=13)
                self.assertEqual([record["asr_text"] for record in records], expected)
                self.assertTrue(all(flag in record["risk_flags"] for record in records))
                root = Path(directory)
                audio = root / "audio.flac"
                audio.write_bytes(b"synthetic source")
                with patch.object(raw_asr_v2_module, "_extract_clip", side_effect=_fake_extract_clip):
                    updated, candidates = build_asr_hallucination_records(
                        records,
                        [{"start_ms": segment["start_ms"], "end_ms": segment["end_ms"],
                          "source": "silero_vad", "vad_region_index": 1}],
                        episode=13, audio_path=audio, audio_output_root=root,
                    )
                self.assertEqual(len(candidates), len(expected))
                self.assertTrue(all(flag in item["reason"] for item in candidates))
                self.assertTrue(all("manual_audio_review_required" in item["risk_flags"] for item in updated))

    def test_serious_gap_threshold_and_only_adjacent_fragments_are_flagged(self) -> None:
        # The 5 s internal/edge policy preserves adjacent fragments for
        # mandatory audio review without treating them as detector output.
        flag = "provisional_word_gap_requires_audio_review"
        for gap, expected in ((4999, [False, False, False]), (5000, [True, True, False])):
            with self.subTest(gap=gap):
                segment = {"segment_id": "main-1", "start_ms": 0,
                           "end_ms": gap + 2000, "text": "A B C", "words": [
                    {"start_ms": 0, "end_ms": 500, "text": " A"},
                    {"start_ms": gap + 500, "end_ms": gap + 900, "text": " B"},
                    {"start_ms": gap + 1700, "end_ms": gap + 2000, "text": " C"},
                ]}
                records = build_correction_utterances([segment], episode=13)
                self.assertEqual([flag in record["risk_flags"] for record in records], expected)

    def test_serious_provisional_segment_edge_gaps_require_review(self) -> None:
        flag = "provisional_word_gap_requires_audio_review"
        for start, end, offset, expected in ((0, 6600, 5000, [True, False]),
                                              (0, 6600, 0, [False, True]),
                                              (1, 6600, 5000, [False, False])):
            with self.subTest(start=start, end=end, offset=offset):
                words = [{"start_ms": offset, "end_ms": offset + 400, "text": " Bir"},
                         {"start_ms": offset + 1200, "end_ms": offset + 1600, "text": " söz"}]
                records = build_correction_utterances(
                    [{"segment_id": "main-edge", "start_ms": start, "end_ms": end,
                      "text": "Bir söz", "words": words}], episode=13,
                )
                self.assertEqual([flag in record["risk_flags"] for record in records], expected)

    def test_incomplete_ep13_inventory_never_proves_vad_coverage(self) -> None:
        segments, words = consume_coarse_segments(iter([
            Segment(3758.670, 3942.660, "Nefes al, " * 24, [
                Word(3758.670, 3760.000, " Nefes"),
                Word(3790.000, 3791.000, " Bismillah"),
            ]),
        ]))
        self.assertFalse(segments[0]["word_timing_complete"])
        segments[0]["word_timing_complete"] = True
        records = build_correction_utterances(segments, episode=13)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["asr_text"], ("Nefes al, " * 24).strip())
        self.assertIn("incomplete_provisional_word_timing", records[0]["risk_flags"])
        self.assertIn("provisional_word_gap_requires_audio_review", records[0]["risk_flags"])
        self.assertEqual(raw_asr_v2_module._coverage_intervals_for_segments(segments, words), [])
        coverage = analyze_speech_coverage(
            [{"start_ms": 3758670, "end_ms": 3760000},
             {"start_ms": 3790000, "end_ms": 3791000}], [],
            config=RawASRV2Config().speech_coverage_config(),
        )
        self.assertEqual(coverage["unresolved_speech_region_count"], 2)
        self.assertEqual(len(coverage["rescue_spans"]), 2)

    def test_incomplete_primary_words_cannot_suppress_complete_rescue(self) -> None:
        primary, primary_words = consume_coarse_segments(iter([
            Segment(0, 184, "Nefes al " * 24, [Word(1, 2, " Nefes")]),
        ]))
        rescue, rescue_words = consume_coarse_segments(iter([
            Segment(1, 2, "Nefes", [Word(1, 2, " Nefes")]),
        ]), source="rescue-1")
        segments, words = merge_rescue_evidence(primary, primary_words, rescue, rescue_words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(words), 2)
        self.assertEqual(raw_asr_v2_module._coverage_intervals_for_segments(segments, words),
                         [{"start_ms": 1000, "end_ms": 2000}])

    def test_rescue_batches_preserve_all_targets_and_original_duration_bound(self) -> None:
        spans = [
            {"start_ms": 0, "end_ms": 10000, "issue_ids": ["a"]},
            {"start_ms": 11000, "end_ms": 12000, "issue_ids": ["b"]},
            {"start_ms": 12500, "end_ms": 13500, "issue_ids": ["c"]},
            {"start_ms": 20000, "end_ms": 21000, "issue_ids": ["d"]},
        ]
        batches = raw_asr_v2_module._bounded_rescue_batches(spans, limit=3, hard_limit=4)
        self.assertEqual([b["source_rescue_span_indices"] for b in batches], [[1], [2, 3], [4]])
        self.assertEqual(batches[1]["issue_ids"], ["b", "c"])
        self.assertEqual(batches[1]["start_ms"], 11000)
        self.assertEqual(batches[1]["end_ms"], 13500)
        self.assertLessEqual(max(b["duration_ms"] for b in batches), 10000)
        self.assertEqual(batches, raw_asr_v2_module._bounded_rescue_batches(spans, limit=3, hard_limit=4))
        self.assertNotIn("source_rescue_span_indices", spans[0])
        unchanged = raw_asr_v2_module._bounded_rescue_batches(spans, limit=4, hard_limit=4)
        self.assertEqual([(b["start_ms"], b["end_ms"]) for b in unchanged],
                         [(b["start_ms"], b["end_ms"]) for b in spans])
        with self.assertRaisesRegex(TranscriptionError, "above hard limit"):
            raw_asr_v2_module._bounded_rescue_batches(spans, limit=2, hard_limit=3)
        with self.assertRaisesRegex(TranscriptionError, "without exceeding"):
            raw_asr_v2_module._bounded_rescue_batches(spans[:2], limit=1, hard_limit=4)
        with self.assertRaisesRegex(TranscriptionError, "cannot fit"):
            raw_asr_v2_module._bounded_rescue_batches(
                spans[:3], limit=2, hard_limit=4,
                trusted_intervals=[{"start_ms": 12200, "end_ms": 12300}],
            )

    def test_serious_gap_threshold_is_part_of_canonical_hash_bound_policy(self) -> None:
        settings = asdict(RawASRV2Config())
        settings["correction_review_word_gap_ms"] = 20000
        with self.assertRaisesRegex(TranscriptionError, "correction_review_word_gap_ms"):
            validate_publishable_raw_asr_v2_policy({"model": {"settings": settings}})

    def test_hole_wav_reuse_requires_exact_canonical_metadata(self) -> None:
        issue = {"issue_id": "speech-1", "start_ms": 1000, "end_ms": 2000,
                 "classification": "unresolved_speech", "reasons": ["missing"]}
        before = {"coarse_start_ms": 0, "coarse_end_ms": 500, "asr_text": "Önce"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.flac"
            audio.write_bytes(b"synthetic source")
            with patch.object(raw_asr_v2_module, "_extract_clip", side_effect=_fake_extract_clip):
                original = build_speech_hole_records(
                    {"coverage_issues": [issue]}, episode=13, audio_path=audio,
                    audio_output_root=root, utterances=[before],
                )
            with patch.object(raw_asr_v2_module, "_extract_clip", side_effect=AssertionError("no repeated extraction")):
                reused = build_speech_hole_records(
                    {"coverage_issues": [issue]}, episode=13, audio_path=audio,
                    audio_output_root=root, utterances=[before], existing_records=original,
                )
            self.assertEqual(reused, original)
            for changed_issue, context in ((dict(issue, reasons=["changed"]), before),
                                           (issue, dict(before, asr_text="Başka")),
                                           (dict(issue, issue_id="speech-2"), before)):
                with self.subTest(issue=changed_issue, context=context):
                    with patch.object(raw_asr_v2_module, "_extract_clip", side_effect=_fake_extract_clip) as extract:
                        rebuilt = build_speech_hole_records(
                            {"coverage_issues": [changed_issue]}, episode=13, audio_path=audio,
                            audio_output_root=root, utterances=[context], existing_records=original,
                        )
                    self.assertEqual(extract.call_count, 1)
                    self.assertEqual(rebuilt[0]["reason"], ", ".join(changed_issue["reasons"]))
                    self.assertEqual(rebuilt[0]["context_before"], context["asr_text"])
            retained = root / "speech_hole_audio" / "retained" / original[0]["audio_sha256"] / Path(original[0]["audio_member"]).name
            self.assertEqual(sha256_file(retained), original[0]["audio_sha256"])

    def test_explicit_review_budget_does_not_relax_automatic_detector_bound(self) -> None:
        raw_asr_v2_module._require_asr_hallucination_candidate_budget(
            automatic_count=192,
            structural_count=0,
            explicit_only_count=832,
            total_count=1024,
            reviewable_count=900,
            candidate_reason_counts={},
        )
        with self.assertRaisesRegex(TranscriptionError, "automatic_candidates=193"):
            raw_asr_v2_module._require_asr_hallucination_candidate_budget(
                automatic_count=193,
                structural_count=0,
                explicit_only_count=0,
                total_count=193,
                reviewable_count=500,
                candidate_reason_counts={},
            )
        with self.assertRaisesRegex(
            TranscriptionError, "explicit_only_candidates=833"
        ):
            raw_asr_v2_module._require_asr_hallucination_candidate_budget(
                automatic_count=0,
                structural_count=0,
                explicit_only_count=833,
                total_count=833,
                reviewable_count=900,
                candidate_reason_counts={},
            )

    def test_mandatory_structural_reviews_do_not_consume_detector_budget(self) -> None:
        raw_asr_v2_module._require_asr_hallucination_candidate_budget(
            automatic_count=126,
            structural_count=215,
            explicit_only_count=0,
            total_count=313,
            reviewable_count=2972,
            candidate_reason_counts={
                "provisional_word_gap_requires_audio_review": 157,
                "orphan_youtube_caption_without_asr_or_vad_overlap": 58,
            },
        )

    def test_rescue_budget_adapts_only_for_healthy_long_episode(self) -> None:
        config = RawASRV2Config()
        self.assertEqual(
            effective_rescue_span_limit(
                config,
                vad_region_count=1_468,
                speech_coverage_ratio=0.9541,
            ),
            162,
        )
        self.assertEqual(
            effective_rescue_span_limit(
                config,
                vad_region_count=1_468,
                speech_coverage_ratio=0.8999,
            ),
            96,
        )
        self.assertEqual(
            effective_rescue_span_limit(
                config,
                vad_region_count=10_000,
                speech_coverage_ratio=0.99,
            ),
            192,
        )

    def test_one_weak_confidence_signal_does_not_force_audio_review(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "text": "Merhaba",
                    "source": "main",
                    "word_timing_complete": True,
                    "asr_audit": {
                        "avg_logprob": -0.90,
                        "no_speech_prob": 0.05,
                        "compression_ratio": 1.10,
                        "temperature": 0.0,
                    },
                }
            ],
            episode=12,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "audio.flac"
            source_audio.write_bytes(b"synthetic-audio-source")
            updated, candidates = build_asr_hallucination_records(
                utterances,
                [
                    {
                        "vad_region_index": 1,
                        "start_ms": 900,
                        "end_ms": 2_100,
                        "source": "silero_vad",
                    }
                ],
                episode=12,
                audio_path=source_audio,
                audio_output_root=root,
            )
        self.assertEqual(candidates, [])
        self.assertNotIn("suspected_asr_hallucination", updated[0]["risk_flags"])

    def test_two_weak_confidence_signals_force_audio_review(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "text": "Merhaba",
                    "source": "main",
                    "word_timing_complete": True,
                    "asr_audit": {
                        "avg_logprob": -0.90,
                        "no_speech_prob": 0.60,
                        "compression_ratio": 1.10,
                        "temperature": 0.0,
                    },
                }
            ],
            episode=12,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "audio.flac"
            source_audio.write_bytes(b"synthetic-audio-source")
            with patch(
                "mas.engine.raw_asr._extract_clip",
                side_effect=_fake_extract_clip,
            ):
                updated, candidates = build_asr_hallucination_records(
                    utterances,
                    [
                        {
                            "vad_region_index": 1,
                            "start_ms": 900,
                            "end_ms": 2_100,
                            "source": "silero_vad",
                        }
                    ],
                    episode=12,
                    audio_path=source_audio,
                    audio_output_root=root,
                )
        self.assertEqual(len(candidates), 1)
        self.assertIn("low_asr_average_log_probability", candidates[0]["reason"])
        self.assertIn("high_asr_no_speech_probability", candidates[0]["reason"])
        self.assertIn("suspected_asr_hallucination", updated[0]["risk_flags"])

    def test_missing_word_time_is_not_synthesized(self) -> None:
        segments, words = consume_coarse_segments(
            [Segment(1.0, 2.0, "Merhaba dünya", [Word(1.1, 1.4, " Merhaba"), Word(None, None, " dünya")])]
        )
        self.assertEqual(len(words), 1)
        self.assertNotIn("synthetic", repr(segments).casefold())
        self.assertNotIn("synthetic", repr(words).casefold())

    def test_rescue_adds_new_evidence_without_overwriting_primary(self) -> None:
        primary_segments = [{"start_ms": 1000, "end_ms": 1500, "text": "Evet", "source": "main"}]
        primary_words = [{"start_ms": 1050, "end_ms": 1400, "text": " Evet"}]
        rescue_segments = [{"start_ms": 2000, "end_ms": 2600, "text": "Hayır", "source": "rescue"}]
        rescue_words = [{"start_ms": 2100, "end_ms": 2500, "text": " Hayır"}]
        segments, words = merge_rescue_evidence(
            primary_segments, primary_words, rescue_segments, rescue_words
        )
        self.assertEqual([item["text"] for item in segments], ["Evet", "Hayır"])
        self.assertEqual(len(words), 2)

    def test_rescue_fragment_already_inside_primary_text_is_not_duplicated(self) -> None:
        primary_segments = [
            {
                "start_ms": 1_000,
                "end_ms": 2_000,
                "text": "Lütfen, rica ediyorum.",
                "source": "main",
            }
        ]
        rescue_segments = [
            {
                "start_ms": 1_400,
                "end_ms": 1_800,
                "text": "Lütfen.",
                "source": "rescue-1",
            }
        ]
        segments, _ = merge_rescue_evidence(
            primary_segments,
            [],
            rescue_segments,
            [],
        )
        self.assertEqual([item["text"] for item in segments], ["Lütfen, rica ediyorum."])

    def test_rescue_fragment_with_only_temporal_touch_is_preserved(self) -> None:
        primary_segments = [
            {
                "start_ms": 1_000,
                "end_ms": 2_000,
                "text": "Lütfen, rica ediyorum.",
                "source": "main",
            }
        ]
        rescue_segments = [
            {
                "start_ms": 1_990,
                "end_ms": 2_500,
                "text": "Lütfen.",
                "source": "rescue-1",
            }
        ]
        segments, _ = merge_rescue_evidence(
            primary_segments,
            [],
            rescue_segments,
            [],
        )
        self.assertEqual(len(segments), 2)

    def test_known_subtitle_credit_hallucination_accepts_initial_order(self) -> None:
        self.assertTrue(
            raw_asr_v2_module._is_known_subtitle_hallucination("Altyazı .K. M")
        )

    def test_correction_uid_binds_time_and_text(self) -> None:
        records = build_correction_utterances(
            [{"start_ms": 1000, "end_ms": 2000, "text": "Merhaba", "source": "main"}],
            episode=12,
        )
        changed = build_correction_utterances(
            [{"start_ms": 1001, "end_ms": 2000, "text": "Merhaba", "source": "main"}],
            episode=12,
        )
        self.assertNotEqual(records[0]["utterance_uid"], changed[0]["utterance_uid"])
        self.assertEqual(
            set(records[0]),
            {
                "utterance_uid",
                "utterance_index",
                "coarse_start_ms",
                "coarse_end_ms",
                "asr_text",
                "youtube_text",
                "context_before",
                "context_after",
                "risk_flags",
                "asr_audit",
            },
        )

    def test_correction_utterance_splits_proven_long_internal_word_gap(self) -> None:
        records = build_correction_utterances(
            [
                {
                    "segment_id": "main-1",
                    "start_ms": 1_000,
                    "end_ms": 43_400,
                    "text": "Abi, iyi misin?",
                    "source": "main",
                    "word_timing_complete": True,
                    "words": [
                        {"start_ms": 1_000, "end_ms": 1_200, "text": " Abi,"},
                        {"start_ms": 42_600, "end_ms": 42_900, "text": " iyi"},
                        {"start_ms": 42_900, "end_ms": 43_400, "text": " misin?"},
                    ],
                }
            ],
            episode=12,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(
            [
                (item["coarse_start_ms"], item["coarse_end_ms"], item["asr_text"])
                for item in records
            ],
            [(1_000, 1_200, "Abi,"), (42_600, 43_400, "iyi misin?")],
        )
        self.assertEqual(records[0]["context_after"], "iyi misin?")
        self.assertEqual(records[1]["context_before"], "Abi,")

    def test_incomplete_word_inventory_is_not_split_or_dropped(self) -> None:
        records = build_correction_utterances(
            [
                {
                    "segment_id": "main-1",
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "text": "Anne olmaz.",
                    "source": "main",
                    "word_timing_complete": True,
                    "words": [
                        {"start_ms": 1_000, "end_ms": 1_200, "text": " Anne"}
                    ],
                }
            ],
            episode=12,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["asr_text"], "Anne olmaz.")
        self.assertEqual(
            (records[0]["coarse_start_ms"], records[0]["coarse_end_ms"]),
            (1_000, 2_000),
        )
        self.assertIn(
            "incomplete_provisional_word_timing", records[0]["risk_flags"]
        )

    def test_short_outside_vad_candidate_has_context_wav_and_pack_binding(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 1_000,
                    "end_ms": 1_019,
                    "text": "Altyazı",
                    "source": "main",
                    "word_timing_complete": True,
                    "asr_audit": {
                        "avg_logprob": -0.10,
                        "no_speech_prob": 0.05,
                        "compression_ratio": 1.10,
                        "temperature": 0.0,
                    },
                }
            ],
            episode=12,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "audio.flac"
            source_audio.write_bytes(b"synthetic-audio-source")
            with patch(
                "mas.engine.raw_asr._extract_clip",
                side_effect=_fake_extract_clip,
            ):
                updated, candidates = build_asr_hallucination_records(
                    utterances,
                    [
                        {
                            "vad_region_index": 1,
                            "start_ms": 1_200,
                            "end_ms": 1_450,
                            "source": "silero_vad",
                        }
                    ],
                    episode=12,
                    audio_path=source_audio,
                    audio_output_root=root,
                )

            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertEqual((candidate["start_ms"], candidate["end_ms"]), (1_000, 1_019))
            self.assertEqual(
                (candidate["clip_start_ms"], candidate["clip_end_ms"]),
                (250, 1_769),
            )
            self.assertIn("zero_unpadded_independent_vad_overlap", candidate["reason"])
            self.assertIn("music_or_subtitle_text_marker", candidate["reason"])
            self.assertIn("suspected_asr_hallucination", updated[0]["risk_flags"])
            clip_path = root.joinpath(*Path(candidate["audio_member"]).parts)
            with wave.open(str(clip_path), "rb") as review_audio:
                self.assertEqual(review_audio.getnframes(), round(16_000 * 1.519))

            pack_path = root / "candidate-pack.zip"
            create_tr_correction_pack(
                updated,
                [],
                pack_path,
                episode=12,
                asr_hallucination_records=candidates,
                asr_hallucination_audio_root=root,
            )
            reopened = read_tr_correction_pack(pack_path)

        self.assertEqual(list(reopened.utterances), updated)
        self.assertEqual(list(reopened.asr_hallucination_records), candidates)

    def test_known_subtitle_hallucination_is_reviewed_even_inside_vad(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 1_000,
                    "end_ms": 1_900,
                    "text": "Altyazı M.K.",
                    "source": "main",
                    "word_timing_complete": True,
                    "asr_audit": {
                        "avg_logprob": -0.10,
                        "no_speech_prob": 0.05,
                        "compression_ratio": 1.10,
                        "temperature": 0.0,
                    },
                }
            ],
            episode=12,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "audio.flac"
            source_audio.write_bytes(b"synthetic-audio-source")
            with patch(
                "mas.engine.raw_asr._extract_clip",
                side_effect=_fake_extract_clip,
            ):
                updated, candidates = build_asr_hallucination_records(
                    utterances,
                    [
                        {
                            "vad_region_index": 1,
                            "start_ms": 900,
                            "end_ms": 2_000,
                            "source": "silero_vad",
                        }
                    ],
                    episode=12,
                    audio_path=source_audio,
                    audio_output_root=root,
                )

        self.assertEqual(len(candidates), 1)
        self.assertIn("music_or_subtitle_text_marker", candidates[0]["reason"])
        self.assertIn(
            "known_subtitle_hallucination_signature", candidates[0]["reason"]
        )
        self.assertIn("suspected_asr_hallucination", updated[0]["risk_flags"])

    def test_one_ms_caption_touch_still_becomes_orphan_review_candidate(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 999,
                    "end_ms": 1_001,
                    "text": "x",
                    "source": "main",
                }
            ],
            episode=12,
        )
        combined = include_orphan_youtube_captions_for_correction(
            utterances,
            [
                {
                    "caption_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "text": "Duyulmuş olabilecek tam cümle",
                }
            ],
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 999,
                    "end_ms": 1_001,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )

        orphan = combined[0]
        self.assertEqual((orphan["coarse_start_ms"], orphan["coarse_end_ms"]), (0, 1_000))
        self.assertEqual(orphan["asr_text"], "")
        self.assertEqual(orphan["youtube_text"], "Duyulmuş olabilecek tam cümle")
        self.assertIn("orphan_youtube_caption", orphan["risk_flags"])
        self.assertTrue(all(value is None for value in orphan["asr_audit"].values()))

    def test_matching_asr_text_explains_rolling_caption_boundary(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 700,
                    "end_ms": 1_500,
                    "text": "Kadir burada ve bekliyor",
                    "source": "main",
                }
            ],
            episode=12,
        )
        combined = include_orphan_youtube_captions_for_correction(
            utterances,
            [
                {
                    "caption_index": 1,
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "text": "Kadir burada",
                }
            ],
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 700,
                    "end_ms": 1_500,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )

        self.assertEqual(len(combined), 1)
        self.assertNotIn("orphan_youtube_caption", combined[0]["risk_flags"])

    def test_matching_text_with_one_ms_overlap_remains_orphan(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 1_999,
                    "end_ms": 2_100,
                    "text": "Kadir burada",
                    "source": "main",
                }
            ],
            episode=12,
        )
        combined = include_orphan_youtube_captions_for_correction(
            utterances,
            [
                {
                    "caption_index": 1,
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "text": "Kadir burada",
                }
            ],
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 1_999,
                    "end_ms": 2_100,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )

        self.assertEqual(len(combined), 2)
        self.assertTrue(
            any(
                "orphan_youtube_caption" in item["risk_flags"]
                for item in combined
            )
        )

    def test_half_caption_with_long_unexplained_run_is_audio_review_candidate(self) -> None:
        combined = include_orphan_youtube_captions_for_correction(
            build_correction_utterances(
                [
                    {
                        "start_ms": 1_000,
                        "end_ms": 2_000,
                        "text": "ikinci yarı",
                        "source": "main",
                    }
                ],
                episode=12,
            ),
            [
                {
                    "caption_index": 1,
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "text": "Yarısı kanıtlı cümle",
                }
            ],
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )

        self.assertEqual(len(combined), 2)
        orphan = next(
            item
            for item in combined
            if "orphan_youtube_caption" in item["risk_flags"]
        )
        self.assertEqual((orphan["coarse_start_ms"], orphan["coarse_end_ms"]), (0, 2_000))
        self.assertEqual(orphan["youtube_text"], "Yarısı kanıtlı cümle")
        self.assertIn("manual_audio_review_required", orphan["risk_flags"])

    def test_normalized_rolling_caption_accepts_bounded_trailing_display(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 0,
                    "end_ms": 2_500,
                    "text": "kanıtlı cümle",
                    "source": "main",
                }
            ],
            episode=12,
        )
        combined = include_orphan_youtube_captions_for_correction(
            utterances,
            [
                {
                    "caption_index": 1,
                    "start_ms": 0,
                    "end_ms": 3_000,
                    "text": "Ekranda yarım saniye kalan cümle",
                }
            ],
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 2_500,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )

        self.assertEqual(len(combined), 1)
        self.assertNotIn("orphan_youtube_caption", combined[0]["risk_flags"])

    def test_caption_unexplained_run_boundary_accepts_229ms_rejects_230ms(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 230,
                    "end_ms": 1_000,
                    "text": "kanıtlı devam",
                    "source": "main",
                }
            ],
            episode=12,
        )
        captions = [
            {
                "caption_index": 1,
                "start_ms": 0,
                "end_ms": 1_000,
                "text": "Sınırdaki cümle",
            }
        ]

        accepted = include_orphan_youtube_captions_for_correction(
            utterances,
            captions,
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 229,
                    "end_ms": 1_000,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )
        rejected = include_orphan_youtube_captions_for_correction(
            utterances,
            captions,
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 230,
                    "end_ms": 1_000,
                    "source": "silero_vad",
                }
            ],
            episode=12,
        )

        self.assertEqual(len(accepted), 1)
        self.assertNotIn("orphan_youtube_caption", accepted[0]["risk_flags"])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            any(
                "orphan_youtube_caption" in item["risk_flags"]
                for item in rejected
            )
        )

    def test_overlapping_provisional_words_become_strict_coverage_union(self) -> None:
        intervals = build_coverage_intervals(
            [
                {"start_ms": 1000, "end_ms": 1400},
                {"start_ms": 1300, "end_ms": 1600},
                {"start_ms": 2000, "end_ms": 2200},
            ]
        )
        self.assertEqual(
            intervals,
            [
                {"start_ms": 1000, "end_ms": 1600},
                {"start_ms": 2000, "end_ms": 2200},
            ],
        )

    def test_unresolved_speech_becomes_explicit_hole_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "audio.flac"
            source_audio.write_bytes(b"synthetic-audio-source")
            with patch(
                "mas.engine.raw_asr._extract_clip", side_effect=_fake_extract_clip
            ):
                holes = build_speech_hole_records(
                    {
                        "coverage_issues": [
                            {
                                "issue_id": "speech-coverage-0001",
                                "classification": "unresolved_speech",
                                "start_ms": 3000,
                                "end_ms": 3900,
                                "reasons": ["speech_hole_above_min_duration"],
                            }
                        ]
                    },
                    episode=12,
                    audio_path=source_audio,
                    audio_output_root=root,
                )
            clip_path = root.joinpath(*Path(holes[0]["audio_member"]).parts)
            self.assertTrue(clip_path.is_file())
            self.assertEqual(clip_path.stat().st_size, holes[0]["audio_size_bytes"])
        self.assertEqual(len(holes), 1)
        self.assertIn("manual_audio_review_required", holes[0]["risk_flags"])
        combined = include_speech_holes_for_correction([], holes)
        self.assertEqual(combined[0]["utterance_uid"], holes[0]["hole_uid"])
        self.assertEqual(combined[0]["asr_text"], "")
        self.assertIn("unresolved_vad_speech", combined[0]["risk_flags"])

    def test_raw_asr_records_round_trip_through_tr_correction_pack(self) -> None:
        utterances = build_correction_utterances(
            [
                {
                    "start_ms": 1000,
                    "end_ms": 1900,
                    "text": "Birinci ham cümle",
                    "source": "main",
                    "word_timing_complete": True,
                },
                {
                    "start_ms": 5000,
                    "end_ms": 5900,
                    "text": "İkinci ham cümle",
                    "source": "rescue-1",
                    "word_timing_complete": False,
                },
            ],
            episode=12,
            youtube_captions=[
                {
                    "caption_index": 1,
                    "start_ms": 1100,
                    "end_ms": 1800,
                    "text": "Birinci YouTube cümlesi",
                }
            ],
            context_count=1,
        )
        self.assertEqual(utterances[0]["youtube_text"], "Birinci YouTube cümlesi")
        self.assertEqual(utterances[0]["context_after"], "İkinci ham cümle")
        self.assertEqual(utterances[1]["context_before"], "Birinci ham cümle")
        self.assertEqual(
            utterances[1]["risk_flags"],
            ["incomplete_provisional_word_timing", "speech_hole_rescue_asr"],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "audio.flac"
            source_audio.write_bytes(b"synthetic-audio-source")
            with patch(
                "mas.engine.raw_asr._extract_clip", side_effect=_fake_extract_clip
            ):
                holes = build_speech_hole_records(
                    {
                        "coverage_issues": [
                            {
                                "issue_id": "speech-coverage-0001",
                                "classification": "unresolved_speech",
                                "start_ms": 3000,
                                "end_ms": 3700,
                                "reasons": ["speech_hole_above_min_duration"],
                            }
                        ]
                    },
                    episode=12,
                    audio_path=source_audio,
                    audio_output_root=root,
                    utterances=utterances,
                )
            combined = include_speech_holes_for_correction(utterances, holes)
            self.assertEqual(
                [item["utterance_index"] for item in combined], [1, 2, 3]
            )
            hole_utterance = combined[1]
            self.assertEqual(hole_utterance["utterance_uid"], holes[0]["hole_uid"])
            self.assertEqual(hole_utterance["asr_text"], "")
            self.assertEqual(hole_utterance["youtube_text"], "")
            self.assertEqual(
                hole_utterance["context_before"], "Birinci ham cümle"
            )
            self.assertEqual(
                hole_utterance["context_after"], "İkinci ham cümle"
            )
            self.assertIn("unresolved_vad_speech", hole_utterance["risk_flags"])
            self.assertEqual(validate_input_utterances(combined), combined)

            pack_path = root / "tr-correction.zip"
            manifest = create_tr_correction_pack(
                combined,
                holes,
                pack_path,
                episode=12,
                batch_size=2,
                speech_hole_audio_root=root,
            )
            packed = read_tr_correction_pack(pack_path)
        self.assertEqual(manifest["utterance_count"], 3)
        self.assertEqual(manifest["speech_hole_count"], 1)
        self.assertEqual(list(packed.utterances), combined)
        self.assertEqual(list(packed.speech_holes), holes)
        # The VAD hole is carried as blank immutable evidence; this stage never
        # fabricates dialogue from neighboring context.
        self.assertEqual(packed.utterances[1]["asr_text"], "")


class RawASRV2RuntimeTests(unittest.TestCase):
    def setUp(self):
        auth_patcher = patch.dict(
            "os.environ", {"MAS_RAW_ASR_AUTH_KEY": "1" * 64}
        )
        auth_patcher.start()
        self.addCleanup(auth_patcher.stop)
        model_directory = tempfile.TemporaryDirectory()
        self.addCleanup(model_directory.cleanup)
        model_path = Path(model_directory.name)
        (model_path / "model.bin").write_bytes(b"synthetic-model")
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        for name, value in (("resolve_model", model_path),
                            ("producer_identity", {"test_runtime": "synthetic-1"})):
            patcher = patch.object(raw_asr_v2_module.primary_checkpoint, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_primary_survives_vad_failure_but_noncanonical_budget_cannot_publish(self):
        calls = []

        class Model:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, **_kwargs):
                calls.append(_path)
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with patch.object(raw_asr_v2_module, "_import_whisper", return_value=(Model, FakeCTranslate2)):
                with patch.object(raw_asr_v2_module, "_extract_vad_regions", side_effect=RuntimeError("VAD interrupted")):
                    with self.assertRaisesRegex(RuntimeError, "VAD interrupted"):
                        transcribe_raw_audio_v2(audio, prepare, episode=11)
                self.assertEqual(len(list((prepare / "primary_asr").glob("*.json"))), 1)
                self.assertFalse((prepare / "raw_asr_v2.recovery.json").exists())
                with (patch.object(raw_asr_v2_module, "_extract_vad_regions", return_value=self._vad()),
                      self.assertRaisesRegex(TranscriptionError, "rescue policy is not canonical")):
                    transcribe_raw_audio_v2(
                        audio, prepare, episode=11,
                        config=RawASRV2Config(rescue_max_spans=97))
                self.assertEqual(len(calls), 1)
                self.assertFalse((prepare / "raw_asr_v2.done.json").exists())

    def test_primary_does_not_reuse_changed_model_or_prompt(self):
        calls = []

        class Model:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, **_kwargs):
                calls.append(_path)
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (patch.object(raw_asr_v2_module, "_import_whisper", return_value=(Model, FakeCTranslate2)),
                  patch.object(raw_asr_v2_module, "_extract_vad_regions", side_effect=RuntimeError("VAD interrupted"))):
                for names in ((), ("Omer",)):
                    with self.assertRaisesRegex(RuntimeError, "VAD interrupted"):
                        transcribe_raw_audio_v2(audio, prepare, episode=11, canonical_names=names)
                model_path = raw_asr_v2_module.primary_checkpoint.resolve_model("large-v3")
                (model_path / "model.bin").write_bytes(b"changed-model")
                with self.assertRaisesRegex(RuntimeError, "VAD interrupted"):
                    transcribe_raw_audio_v2(audio, prepare, episode=11)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(list((prepare / "primary_asr").glob("*.json"))), 3)

    @staticmethod
    def _audio_and_prepare(directory: str) -> tuple[Path, Path]:
        root = Path(directory)
        audio = root / "audio.flac"
        audio.write_bytes(b"synthetic-audio-for-runtime-tests")
        return audio, root / "work"

    @staticmethod
    def _vad() -> tuple[list[dict[str, object]], None]:
        return (
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "source": "silero_vad",
                }
            ],
            None,
        )

    def test_production_auth_key_is_required_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            for value, message in (
                ("", "is required"),
                ("short", "strong 256-bit"),
            ):
                with (
                    patch.dict(
                        "os.environ", {"MAS_RAW_ASR_AUTH_KEY": value}
                    ),
                    patch.object(
                        raw_asr_v2_module,
                        "_import_whisper",
                        side_effect=AssertionError("inference setup must not run"),
                    ) as imported,
                    self.assertRaisesRegex(TranscriptionError, message),
                ):
                    transcribe_raw_audio_v2(audio, prepare, episode=12)
                imported.assert_not_called()

    def test_cuda_model_initialization_falls_back_once_to_cpu(self) -> None:
        attempts: list[tuple[str, str]] = []

        class Model:
            def __init__(self, _name: str, *, device: str, compute_type: str, **_kwargs: object):
                attempts.append((device, compute_type))
                if device == "cuda":
                    raise RuntimeError("CUDA driver initialization failed")

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                data = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    config=RawASRV2Config(allow_cpu_fallback=True),
                )

        self.assertEqual(attempts, [("cuda", "float16"), ("cpu", "int8")])
        self.assertEqual(data["model"]["device"], "cpu")
        self.assertEqual(data["model"]["compute_type"], "int8")
        self.assertEqual(data["model"]["final_runtime_device"], "cpu")
        self.assertIn("CUDA initialization", data["model"]["runtime_fallback_reason"])

    def test_legacy_recovery_without_producer_receipts_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            prepare.mkdir(parents=True, exist_ok=True)
            captions_path = Path(directory) / "captions.vtt"
            captions_path.write_text("WEBVTT\n", encoding="utf-8")
            words = [
                {
                    "word_index": 1,
                    "segment_id": "main-1",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "text": " Merhaba",
                    "probability": 0.99,
                    "timing_source": "faster_whisper_provisional",
                }
            ]
            segments = [
                {
                    "segment_id": "main-1",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "text": "Merhaba",
                    "word_timing_complete": True,
                    "words": words,
                    "source": "main",
                    "asr_audit": {
                        "avg_logprob": -0.10,
                        "no_speech_prob": 0.05,
                        "compression_ratio": 1.10,
                        "temperature": 0.0,
                    },
                }
            ]
            vad_regions = [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "source": "silero_vad",
                }
            ]
            coverage = raw_asr_v2_module._analyze_raw_speech_coverage(
                vad_regions, segments, words, config=RawASRV2Config().speech_coverage_config())
            batches, budget = raw_asr_v2_module._plan_rescue_batches(
                coverage, vad_regions, segments, words, RawASRV2Config())
            saved_settings = asdict(RawASRV2Config())
            # Match the first field-complete recovery format written before
            # post-processing-only split/lexical settings were introduced.
            for field in (
                "correction_max_internal_word_gap_ms",
                "correction_review_word_gap_ms",
                "orphan_caption_min_lexical_asr_overlap_ms",
                "orphan_caption_min_shared_tokens",
                "orphan_caption_min_character_bigram_dice",
            ):
                saved_settings.pop(field)
            checkpoint = {
                "format": "raw-asr-v2-recovery-1",
                "episode": 12,
                "audio_path": str(audio.resolve()),
                "audio_sha256": sha256_file(audio),
                "captions_path": str(captions_path.resolve()),
                "caption_sha256": sha256_file(captions_path),
                "canonical_names": [],
                "religious_terms": [],
                "hallucination_review_utterance_uids": [],
                "model": {
                    "main_pass_device": "cuda",
                    "main_pass_compute_type": "float16",
                    "final_runtime_device": "cuda",
                    "final_runtime_compute_type": "float16",
                    "runtime_fallback_reason": None,
                    "language": "tr",
                    "settings": saved_settings,
                },
                "segments": segments,
                "words": words,
                "vad_regions": vad_regions,
                "vad_fallback_reason": None,
                "independent_vad": True,
                "initial_speech_coverage": coverage,
                "speech_coverage": coverage,
                "rescue_failures": [],
                "rescue_span_count": 0,
                "rescue_batches": batches,
                "rescue_budget_audit": budget,
                "youtube_captions": [],
                "speech_hole_records": [],
            }
            (prepare / "raw_asr_v2.recovery.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )

            with patch.object(
                raw_asr_v2_module,
                "_import_whisper",
                side_effect=AssertionError("model inference must not run"),
            ):
                with self.assertRaisesRegex(
                    TranscriptionError, "lacks immutable producer receipts"
                ):
                    transcribe_raw_audio_v2(
                        audio,
                        prepare,
                        episode=12,
                        captions_path=captions_path,
                    )

            self.assertTrue((prepare / "raw_asr_v2.recovery.json").is_file())
            self.assertFalse((prepare / "raw_asr_v2.json").exists())
            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())
            self.assertIsNone(
                load_valid_raw_asr_v2(
                    prepare,
                    audio_path=audio,
                    episode=12,
                )
            )

    def test_low_primary_coverage_cannot_coalesce_around_safe_rescue_limit(self) -> None:
        calls = []

        class Model:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, path, **_kwargs):
                calls.append(path)
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            vad = [{"start_ms": index * 5000, "end_ms": index * 5000 + 1000,
                    "vad_region_index": index + 1, "source": "silero_vad"}
                   for index in range(100)]
            with (
                patch.object(raw_asr_v2_module, "_import_whisper", return_value=(Model, FakeCTranslate2)),
                patch.object(raw_asr_v2_module, "_extract_vad_regions", return_value=(vad, None)),
                patch.object(raw_asr_v2_module, "_extract_clip") as extract,
                self.assertRaisesRegex(TranscriptionError, "99 spans, above hard limit 96"),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=13)
            self.assertEqual(len(calls), 1)
            extract.assert_not_called()
            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())

    def test_ordinary_incomplete_parent_is_preserved_but_only_rescue_enters_correction(self) -> None:
        calls = []

        class Model:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, path, **_kwargs):
                calls.append(path)
                if len(calls) == 1:
                    return iter([Segment(0, 2, "Merhaba bugün", [Word(0, 1, " Merhaba")])]), Info()
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(raw_asr_v2_module, "_import_whisper", return_value=(Model, FakeCTranslate2)),
                patch.object(raw_asr_v2_module, "_extract_vad_regions", return_value=self._vad()),
                patch.object(raw_asr_v2_module, "_extract_clip", side_effect=_fake_extract_clip) as clips,
            ):
                data = transcribe_raw_audio_v2(audio, prepare, episode=13)
            self.assertEqual(len(calls), 2)
            self.assertEqual(clips.call_count, 2)
            self.assertTrue(all(call.args[3] - call.args[2] <= 10000 for call in clips.call_args_list))
            self.assertEqual(next(item["text"] for item in data["segments"] if item["segment_id"] == "main-1"), "Merhaba bugün")
            self.assertEqual(len(data["excluded_incomplete_segments"]), 1)
            self.assertEqual([item["asr_text"] for item in data["correction_utterances"]], ["Merhaba", ""])
            self.assertIn("unresolved_vad_speech", data["correction_utterances"][1]["risk_flags"])
            self.assertEqual(data["asr_hallucination_records"], [])
            self.assertFalse((prepare / "asr_hallucination_audio").exists())
            self.assertEqual(data["rescue_budget_audit"]["actual_batch_count"], 1)
            self.assertEqual(data["rescue_budget_audit"]["mandatory_overflow_count"], 0)
            tampered = json.loads(json.dumps(data))
            tampered["rescue_budget_audit"]["actual_batch_count"] += 1
            with self.assertRaisesRegex(TranscriptionError, "rescue budget audit"):
                validate_persisted_raw_asr_v2(tampered, expected_input_sha256=data["input_sha256"],
                    audio_sha256=data["audio_sha256"], episode=13, prepare_dir=prepare, require_independent_vad=False)

    def test_excluded_500ms_evet_is_rescued_and_persisted_as_mandatory_hole(self) -> None:
        calls = []

        class Model:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, path, **_kwargs):
                calls.append(path)
                if len(calls) == 1:
                    return iter([Segment(0, 4.5, "Tamam", [Word(0, 4.5, " Tamam")]),
                                 Segment(4.5, 5, "Evet", [])]), Info()
                return iter([]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            vad = [{"start_ms": 0, "end_ms": 5000, "source": "silero_vad", "vad_region_index": 1}]
            with (
                patch.object(raw_asr_v2_module, "_import_whisper", return_value=(Model, FakeCTranslate2)),
                patch.object(raw_asr_v2_module, "_extract_vad_regions", return_value=(vad, None)),
                patch.object(raw_asr_v2_module, "_extract_clip", side_effect=_fake_extract_clip),
            ):
                data = transcribe_raw_audio_v2(audio, prepare, episode=13)
            self.assertEqual(len(calls), 2)
            self.assertEqual(data["speech_coverage"]["status"], "FAIL")
            self.assertEqual([(h["start_ms"], h["end_ms"]) for h in data["speech_hole_records"]], [(4500, 5000)])
            issue = data["speech_coverage"]["coverage_issues"][0]
            self.assertEqual(issue["required_source_records"][0]["source_id"], "main-2")
            self.assertEqual(data["excluded_incomplete_segments"][0]["utterance_uid"],
                             build_correction_utterances([data["segments"][1]], episode=13)[0]["utterance_uid"])
            hole_uid = data["speech_hole_records"][0]["hole_uid"]
            self.assertIn(hole_uid, {u["utterance_uid"] for u in data["correction_utterances"]})

            generic = analyze_speech_coverage(vad, [{"start_ms": 0, "end_ms": 4500}],
                                               config=RawASRV2Config().speech_coverage_config())
            self.assertEqual(generic["status"], "PASS")
            tampered = json.loads(json.dumps(data))
            tampered["speech_coverage"]["coverage_issues"][0]["required_source_records"][0]["source_sha256"] = "0" * 64
            with self.assertRaisesRegex(TranscriptionError, "complete lexical word inventory"):
                validate_persisted_raw_asr_v2(
                    tampered, expected_input_sha256=data["input_sha256"], audio_sha256=data["audio_sha256"],
                    episode=13, prepare_dir=prepare, require_independent_vad=False,
                )

            checkpoint_path = prepare / "raw_asr_v2.recovery.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint.update(initial_speech_coverage=generic, speech_coverage=generic, speech_hole_records=[])
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            with (
                patch.object(raw_asr_v2_module, "_import_whisper", side_effect=AssertionError("no recovery inference")),
                patch.object(raw_asr_v2_module, "_extract_clip", side_effect=AssertionError("no new clips")),
                self.assertRaisesRegex(TranscriptionError, "canonical primary evidence"),
            ):
                raw_asr_v2_module.recover_raw_asr_v2_from_checkpoint(audio, prepare, episode=13)

    def test_legacy_incomplete_recovery_without_canonical_execution_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            prepare.mkdir(parents=True)
            segments, words = consume_coarse_segments(iter([
                Segment(0, 184, "Nefes al, " * 24, [Word(0, 1, " Nefes")]),
            ]))
            segments[0]["word_timing_complete"] = True
            vad = [
                {"start_ms": 0, "end_ms": 1000, "source": "silero_vad", "vad_region_index": 1},
                {"start_ms": 3000, "end_ms": 4000, "source": "silero_vad", "vad_region_index": 2},
            ]
            old_coverage = analyze_speech_coverage(
                vad, build_coverage_intervals(words),
                config=RawASRV2Config().speech_coverage_config(),
            )
            with patch.object(raw_asr_v2_module, "_extract_clip", side_effect=_fake_extract_clip):
                old_holes = build_speech_hole_records(
                    old_coverage, episode=13, audio_path=audio, audio_output_root=prepare,
                )
            self.assertEqual(len(old_holes), 1)
            saved_settings = asdict(RawASRV2Config())
            saved_settings.pop("correction_review_word_gap_ms")
            checkpoint = {
                "format": "raw-asr-v2-recovery-1", "episode": 13,
                "audio_sha256": sha256_file(audio), "caption_sha256": None,
                "canonical_names": [], "religious_terms": [],
                "hallucination_review_utterance_uids": [],
                "model": {"settings": saved_settings, "main_pass_device": "cuda",
                          "main_pass_compute_type": "float16"},
                "segments": segments, "words": words, "vad_regions": vad,
                "vad_fallback_reason": None, "independent_vad": True,
                "initial_speech_coverage": old_coverage, "speech_coverage": old_coverage,
                "rescue_failures": [], "rescue_span_count": 0,
                "youtube_captions": [], "speech_hole_records": old_holes,
            }
            (prepare / "raw_asr_v2.recovery.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            with (
                patch.object(raw_asr_v2_module, "_import_whisper", side_effect=AssertionError("no inference")),
                patch.object(raw_asr_v2_module, "_extract_clip", side_effect=AssertionError("no new decode")) as clips,
                self.assertRaisesRegex(TranscriptionError, "canonical primary evidence"),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=13)
            old_wav = prepare / old_holes[0]["audio_member"]
            self.assertEqual(sha256_file(old_wav), old_holes[0]["audio_sha256"])
            self.assertEqual(old_wav.stat().st_size, old_holes[0]["audio_size_bytes"])
            clips.assert_not_called()
            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())
            self.assertEqual(json.loads((prepare / "raw_asr_v2.recovery.json").read_text(encoding="utf-8")),
                             json.loads(json.dumps(checkpoint)))
            for key, value in (("rescue_max_span_fraction", 0.10),
                               ("rescue_max_span_fraction", 0.09), ("beam_size", 99)):
                invalid_checkpoint = json.loads(json.dumps(checkpoint))
                invalid_checkpoint["model"]["settings"][key] = value
                (prepare / "raw_asr_v2.recovery.json").write_text(json.dumps(invalid_checkpoint), encoding="utf-8")
                with (
                    patch.object(raw_asr_v2_module, "_import_whisper", side_effect=AssertionError("no inference")),
                    self.assertRaisesRegex(TranscriptionError, "ASR settings mismatch"),
                ):
                    raw_asr_v2_module.recover_raw_asr_v2_from_checkpoint(audio, prepare, episode=13)

    def test_added_review_uid_reuses_completed_checkpoint_without_inference(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment("Merhaba")]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            captions_path = Path(directory) / "captions.vtt"
            captions_path.write_text("WEBVTT\n", encoding="utf-8")
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                initial = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    captions_path=captions_path,
                )

            uid = initial["correction_utterances"][0]["utterance_uid"]
            initial_input_sha = initial["input_sha256"]
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    side_effect=AssertionError("model inference must not run"),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_clip",
                    side_effect=_fake_extract_clip,
                ),
            ):
                recovered = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    captions_path=captions_path,
                    config=RawASRV2Config(extra_audio_review_uids=(uid,)),
                )

            self.assertTrue(recovered["resumed"])
            self.assertNotEqual(recovered["input_sha256"], initial_input_sha)
            self.assertEqual(
                recovered["hallucination_review_utterance_uids"], [uid]
            )
            self.assertEqual(
                recovered["model"]["settings"]["extra_audio_review_uids"],
                (uid,),
            )
            self.assertEqual(len(recovered["asr_hallucination_records"]), 1)
            review = recovered["asr_hallucination_records"][0]
            self.assertEqual(review["utterance_uid"], uid)
            self.assertTrue((prepare / review["audio_member"]).is_file())
            self.assertIsNotNone(
                load_valid_raw_asr_v2(
                    prepare,
                    audio_path=audio,
                    episode=12,
                    expected_input_sha256=recovered["input_sha256"],
                )
            )

    def test_production_transcribe_emits_orphan_caption_context_wav(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment("Merhaba")]), Info()

        captions = [
            {
                "caption_index": 1,
                "start_ms": 2_000,
                "end_ms": 2_200,
                "text": "Duyulmayan altyazı adayı",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            captions_path = Path(directory) / "captions.vtt"
            captions_path.write_text("WEBVTT\n", encoding="utf-8")
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "load_vtt_captions",
                    return_value=captions,
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_clip",
                    side_effect=_fake_extract_clip,
                ),
            ):
                data = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    captions_path=captions_path,
                )

            self.assertTrue(data["independent_vad"])
            self.assertEqual(len(data["asr_hallucination_records"]), 1)
            candidate = data["asr_hallucination_records"][0]
            self.assertEqual((candidate["start_ms"], candidate["end_ms"]), (2_000, 2_200))
            self.assertEqual(candidate["asr_text"], "")
            self.assertEqual(candidate["youtube_text"], captions[0]["text"])
            self.assertIn("orphan_youtube_caption", candidate["risk_flags"])
            self.assertNotIn("suspected_asr_hallucination", candidate["risk_flags"])
            self.assertTrue((prepare / candidate["audio_member"]).is_file())
            persisted = load_valid_raw_asr_v2(
                prepare,
                audio_path=audio,
                episode=12,
            )
            self.assertIsNotNone(persisted)
            pack_path = Path(directory) / "orphan-pack.zip"
            create_tr_correction_pack(
                data["correction_utterances"],
                data["speech_hole_records"],
                pack_path,
                episode=12,
                asr_hallucination_records=data[
                    "asr_hallucination_records"
                ],
                asr_hallucination_audio_root=prepare,
            )
            reopened = read_tr_correction_pack(pack_path)
            self.assertEqual(
                list(reopened.asr_hallucination_records),
                data["asr_hallucination_records"],
            )

            loose = json.loads(json.dumps(data))
            loose["speech_coverage"]["config"]["min_hole_ms"] = 10_000
            loose = raw_asr_v2_module._sign_raw_asr_artifact(
                loose, bytes.fromhex("1" * 64)
            )
            with self.assertRaisesRegex(
                TranscriptionError, "canonical V2 beta policy"
            ):
                validate_persisted_raw_asr_v2(
                    loose,
                    expected_input_sha256=data["input_sha256"],
                    audio_sha256=data["audio_sha256"],
                    episode=12,
                    prepare_dir=prepare,
                )
            weakened_caption_gap = json.loads(json.dumps(data))
            weakened_caption_gap["model"]["settings"][
                "orphan_caption_max_unexplained_run_ms"
            ] = 10_000
            weakened_caption_gap = raw_asr_v2_module._sign_raw_asr_artifact(
                weakened_caption_gap, bytes.fromhex("1" * 64)
            )
            with self.assertRaisesRegex(
                TranscriptionError, "canonical V2 beta publication policy"
            ):
                validate_persisted_raw_asr_v2(
                    weakened_caption_gap,
                    expected_input_sha256=data["input_sha256"],
                    audio_sha256=data["audio_sha256"],
                    episode=12,
                    prepare_dir=prepare,
                )
            missing_orphan = json.loads(json.dumps(data))
            missing_orphan["correction_utterances"] = [
                item
                for item in missing_orphan["correction_utterances"]
                if "orphan_youtube_caption" not in item["risk_flags"]
            ]
            for utterance_index, item in enumerate(
                missing_orphan["correction_utterances"], start=1
            ):
                item["utterance_index"] = utterance_index
            missing_orphan["asr_hallucination_records"] = []
            missing_orphan = raw_asr_v2_module._sign_raw_asr_artifact(
                missing_orphan, bytes.fromhex("1" * 64)
            )
            with self.assertRaisesRegex(
                TranscriptionError, "persisted orphan YouTube-caption candidates"
            ):
                validate_persisted_raw_asr_v2(
                    missing_orphan,
                    expected_input_sha256=data["input_sha256"],
                    audio_sha256=data["audio_sha256"],
                    episode=12,
                    prepare_dir=prepare,
                )
            self.assertEqual(
                validate_publishable_raw_asr_v2_policy(data)[
                    "coverage_policy_label"
                ],
                "v2-beta-full-length-dialogue-v1",
            )

    def test_lazy_cuda_inference_failure_retries_full_pass_on_cpu(self) -> None:
        attempts: list[str] = []

        class Model:
            def __init__(self, _name: str, *, device: str, **_kwargs: object):
                self.device = device
                attempts.append(device)

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                if self.device == "cuda":
                    def fail_during_iteration() -> object:
                        raise RuntimeError("CUDA out of memory while decoding")
                        yield None

                    return fail_during_iteration(), Info()
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                data = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    config=RawASRV2Config(allow_cpu_fallback=True),
                )

        self.assertEqual(attempts, ["cuda", "cpu"])
        self.assertEqual(data["model"]["device"], "cpu")
        self.assertIn("CUDA inference", data["model"]["runtime_fallback_reason"])

    def test_non_cuda_main_failure_does_not_trigger_cpu_retry_or_marker(self) -> None:
        attempts: list[str] = []

        class Model:
            def __init__(self, _name: str, *, device: str, **_kwargs: object):
                attempts.append(device)

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                raise RuntimeError("decoder input is corrupt")

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with patch.object(
                raw_asr_v2_module,
                "_import_whisper",
                return_value=(Model, FakeCTranslate2),
            ):
                with self.assertRaisesRegex(
                    TranscriptionError, "Full Turkish raw ASR pass failed"
                ):
                    transcribe_raw_audio_v2(audio, prepare, episode=12)

            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())
            self.assertFalse((prepare / "raw_asr_v2.json").exists())
        self.assertEqual(attempts, ["cuda"])

    def test_cuda_rescue_failure_switches_remaining_work_to_cpu(self) -> None:
        attempts: list[str] = []

        class Model:
            def __init__(self, _name: str, *, device: str, **_kwargs: object):
                self.device = device
                self.calls = 0
                attempts.append(device)

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                self.calls += 1
                if self.device == "cuda" and self.calls == 1:
                    primary = Segment(
                        0.0,
                        0.4,
                        "Merhaba",
                        [Word(0.0, 0.4, " Merhaba")],
                    )
                    return iter([primary]), Info()
                if self.device == "cuda":
                    def fail_during_iteration() -> object:
                        raise RuntimeError("CUDA allocation failed during rescue")
                        yield None

                    return fail_during_iteration(), Info()
                rescue = Segment(
                    0.0,
                    0.5,
                    "Dünya",
                    [Word(0.0, 0.5, " Dünya")],
                )
                return iter([rescue]), Info()

        initial_report = {
            "metrics": {"speech_coverage_ratio": 0.4},
            "config": asdict(RawASRV2Config().speech_coverage_config()),
            "rescue_spans": [{"start_ms": 400, "end_ms": 1_000}],
            "coverage_issues": [],
            "unresolved_speech_region_count": 1,
        }
        final_report = {
            "config": asdict(RawASRV2Config().speech_coverage_config()),
            "rescue_spans": [],
            "coverage_issues": [],
            "unresolved_speech_region_count": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
                patch.object(raw_asr_v2_module, "_extract_clip"),
                patch.object(
                    raw_asr_v2_module,
                    "analyze_speech_coverage",
                    side_effect=[initial_report, final_report, final_report, initial_report],
                ),
            ):
                data = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    config=RawASRV2Config(allow_cpu_fallback=True),
                )

        self.assertEqual(attempts, ["cuda", "cpu"])
        self.assertEqual(data["model"]["device"], "cuda")
        self.assertEqual(data["model"]["final_runtime_device"], "cpu")
        self.assertIn(
            "CUDA rescue inference", data["model"]["runtime_fallback_reason"]
        )
        self.assertEqual(data["rescue_failures"], [])
        self.assertEqual(
            [segment["text"] for segment in data["segments"]],
            ["Merhaba", "Dünya"],
        )

    def test_valid_marker_resumes_without_importing_model(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            production_config = RawASRV2Config(allow_cpu_fallback=False)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                first = transcribe_raw_audio_v2(
                    audio, prepare, episode=12, config=production_config
                )
            loaded = load_valid_raw_asr_v2(
                prepare,
                audio_path=audio,
                episode=12,
                require_independent_vad=True,
            )
            with patch.object(
                raw_asr_v2_module,
                "_import_whisper",
                side_effect=AssertionError("model import must not run on resume"),
            ):
                resumed = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    config=production_config,
                    require_resume=True,
                )
            persisted = json.loads(
                (prepare / "raw_asr_v2.json").read_text(encoding="utf-8")
            )

        self.assertFalse(first["resumed"])
        self.assertFalse(first["model"]["settings"]["allow_cpu_fallback"])
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertFalse(loaded["resumed"])
        self.assertTrue(resumed["resumed"])
        self.assertEqual(persisted["status"], "completed")
        self.assertFalse(persisted["resumed"])
        self.assertEqual(resumed["input_sha256"], first["input_sha256"])

    def test_completed_cache_rejects_missing_or_rotated_key_until_force(self) -> None:
        calls = 0

        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                nonlocal calls
                calls += 1
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)
            self.assertEqual(calls, 1)
            persisted_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in prepare.rglob("*.json")
            )
            self.assertNotIn("1" * 64, persisted_text)

            with (
                patch.dict("os.environ", {"MAS_RAW_ASR_AUTH_KEY": ""}),
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    side_effect=AssertionError("missing key must not infer"),
                ) as imported,
                self.assertRaisesRegex(TranscriptionError, "is required"),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)
            imported.assert_not_called()

            with (
                patch.dict(
                    "os.environ", {"MAS_RAW_ASR_AUTH_KEY": "2" * 64}
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    side_effect=AssertionError("rotated key must not infer"),
                ) as imported,
                self.assertRaisesRegex(
                    TranscriptionError, "explicit force rerun is required"
                ),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)
            imported.assert_not_called()
            self.assertEqual(calls, 1)

            with (
                patch.dict(
                    "os.environ", {"MAS_RAW_ASR_AUTH_KEY": "2" * 64}
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                refreshed = transcribe_raw_audio_v2(
                    audio, prepare, episode=12, force=True
                )
                loaded = load_valid_raw_asr_v2(
                    prepare, audio_path=audio, episode=12
                )
            self.assertEqual(calls, 2)
            self.assertIsNotNone(loaded)
            self.assertFalse(refreshed["resumed"])

    def test_coherent_recovery_tamper_is_rejected_without_done_marker(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)
            (prepare / "raw_asr_v2.json").unlink()
            (prepare / "raw_asr_v2.done.json").unlink()
            recovery_path = prepare / "raw_asr_v2.recovery.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            recovery["segments"][0]["text"] = "Değiştirilmiş"
            recovery["segments"][0]["words"][0]["text"] = " Değiştirilmiş"
            recovery["words"][0]["text"] = " Değiştirilmiş"
            recovery_path.write_text(json.dumps(recovery), encoding="utf-8")

            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    side_effect=AssertionError("model inference must not run"),
                ),
                self.assertRaisesRegex(
                    TranscriptionError, "differs from immutable producer receipts"
                ),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)

            self.assertFalse((prepare / "raw_asr_v2.json").exists())
            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())

    def test_coherent_producer_receipt_replacement_is_rejected(self) -> None:
        calls = 0

        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return iter(
                        [Segment(0, 2, "Merhaba bugün", [Word(0, 1, " Merhaba")])]
                    ), Info()
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_clip",
                    side_effect=_fake_extract_clip,
                ),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)

            (prepare / "raw_asr_v2.json").unlink()
            (prepare / "raw_asr_v2.done.json").unlink()
            recovery_path = prepare / "raw_asr_v2.recovery.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            primary_record = recovery["producer_receipts"]["primary"]
            primary_path = prepare / primary_record["relative_path"]
            envelope = json.loads(primary_path.read_text(encoding="utf-8"))
            original_identity = envelope["data"]["identity"]
            batches = recovery["rescue_batches"]
            rescue_binding = raw_asr_v2_module._rescue_journal_binding(
                audio_sha256=recovery["audio_sha256"],
                primary_receipt_sha256=primary_record["sha256"],
                model_identity=original_identity["model"],
                producer=original_identity["producer"],
                settings=RawASRV2Config(),
                prompt=None,
                rescue_batches=batches,
            )
            batch_uid = raw_asr_v2_module._rescue_batch_uid(1, batches[0])
            rescue_result = raw_asr_v2_module.UnitJournal(
                prepare / "raw_asr_units" / "rescue", rescue_binding
            ).read(batch_uid)
            self.assertIsNotNone(rescue_result)

            vad_binding = raw_asr_v2_module._vad_journal_binding(
                audio_sha256=recovery["audio_sha256"],
                primary_receipt_sha256=primary_record["sha256"],
                settings=RawASRV2Config(),
                producer=original_identity["producer"],
            )
            saved_vad_result = raw_asr_v2_module.UnitJournal(
                prepare / "raw_asr_units" / "vad", vad_binding
            ).read("independent-vad")
            self.assertIsNotNone(saved_vad_result)

            envelope["data"]["segments"][0]["text"] = "Sahte ana çıktı"
            envelope["data"]["segments"][0]["words"][0]["text"] = " Sahte"
            envelope["data"]["words"][0]["text"] = " Sahte"
            recovery["segments"][0]["text"] = "Sahte ana çıktı"
            recovery["segments"][0]["words"][0]["text"] = " Sahte"
            next(
                word
                for word in recovery["words"]
                if word["segment_id"] == "main-1"
            )["text"] = " Sahte"
            envelope["sha256"] = raw_asr_v2_module.sha256_json(envelope["data"])
            primary_path.write_text(json.dumps(envelope), encoding="utf-8")
            forged_primary_sha = sha256_file(primary_path)
            recovery["producer_receipts"]["primary"] = {
                "relative_path": primary_path.relative_to(prepare).as_posix(),
                "sha256": forged_primary_sha,
                "auth_tag": primary_record["auth_tag"],
            }

            auth_path = raw_asr_v2_module._primary_auth_record_path(
                prepare, original_identity
            )
            auth_record = json.loads(auth_path.read_text(encoding="utf-8"))
            auth_record["receipt_sha256"] = forged_primary_sha
            auth_path.write_text(json.dumps(auth_record), encoding="utf-8")
            forged_vad_binding = raw_asr_v2_module._vad_journal_binding(
                audio_sha256=recovery["audio_sha256"],
                primary_receipt_sha256=forged_primary_sha,
                settings=RawASRV2Config(),
                producer=original_identity["producer"],
            )
            raw_asr_v2_module.UnitJournal(
                prepare / "raw_asr_units" / "vad",
                forged_vad_binding,
            ).write("independent-vad", saved_vad_result)
            recovery["producer_receipts"]["vad_result_sha256"] = (
                raw_asr_v2_module._unit_result_sha256(saved_vad_result)
            )
            forged_rescue_binding = raw_asr_v2_module._rescue_journal_binding(
                audio_sha256=recovery["audio_sha256"],
                primary_receipt_sha256=forged_primary_sha,
                model_identity=original_identity["model"],
                producer=original_identity["producer"],
                settings=RawASRV2Config(),
                prompt=None,
                rescue_batches=batches,
            )
            raw_asr_v2_module.UnitJournal(
                prepare / "raw_asr_units" / "rescue", forged_rescue_binding
            ).write(batch_uid, rescue_result)
            recovery["producer_receipts"]["rescue_result_sha256"][batch_uid] = (
                raw_asr_v2_module._unit_result_sha256(rescue_result)
            )
            recovery_path.write_text(json.dumps(recovery), encoding="utf-8")

            with self.assertRaisesRegex(
                TranscriptionError, "primary producer receipt authentication failed"
            ):
                raw_asr_v2_module.recover_raw_asr_v2_from_checkpoint(
                    audio, prepare, episode=12
                )
            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())

    def test_completed_rescue_batch_is_not_rerun_after_interruption(self) -> None:
        calls: list[str] = []

        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, path: str, **_kwargs: object) -> tuple[object, Info]:
                calls.append(path)
                if len(calls) == 1:
                    return iter(
                        [Segment(0, 2, "Merhaba bugün", [Word(0, 1, " Merhaba")])]
                    ), Info()
                return iter([_complete_segment()]), Info()

        original_write = raw_asr_v2_module.UnitJournal.write
        interrupted = False

        def write_then_interrupt(journal, uid, result):
            nonlocal interrupted
            original_write(journal, uid, result)
            if uid.startswith("rescue-") and not interrupted:
                interrupted = True
                raise RuntimeError("interrupted after committed rescue batch")

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_clip",
                    side_effect=_fake_extract_clip,
                ),
                patch.object(
                    raw_asr_v2_module.UnitJournal,
                    "write",
                    new=write_then_interrupt,
                ),
                self.assertRaisesRegex(RuntimeError, "interrupted after committed"),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)
            self.assertFalse((prepare / "raw_asr_v2.recovery.json").exists())

            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    side_effect=AssertionError("VAD must be reused"),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_clip",
                    side_effect=_fake_extract_clip,
                ),
            ):
                data = transcribe_raw_audio_v2(audio, prepare, episode=12)

            self.assertEqual(len(calls), 2)
            self.assertEqual(data["rescue_budget_audit"]["actual_batch_count"], 1)
            self.assertTrue((prepare / "raw_asr_v2.done.json").is_file())

    def test_marker_cannot_redirect_resume_to_another_file(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                data = transcribe_raw_audio_v2(audio, prepare, episode=12)
            alternate = prepare / "alternate.json"
            alternate.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            write_stage_marker(
                prepare / "raw_asr_v2.done.json",
                stage="raw_asr_v2",
                input_sha256=data["input_sha256"],
                outputs={"raw_asr_v2": alternate},
            )
            with patch.object(
                raw_asr_v2_module,
                "_import_whisper",
                side_effect=AssertionError("model inference must not run"),
            ):
                recovered = transcribe_raw_audio_v2(audio, prepare, episode=12)
            self.assertTrue(recovered["resumed"])
            marker = json.loads(
                (prepare / "raw_asr_v2.done.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                Path(marker["outputs"]["raw_asr_v2"]["path"]).resolve(),
                (prepare / "raw_asr_v2.json").resolve(),
            )

    def test_rehashed_marker_cannot_hide_unauthenticated_completed_artifact(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                data = transcribe_raw_audio_v2(audio, prepare, episode=12)
            output = prepare / "raw_asr_v2.json"
            tampered = json.loads(output.read_text(encoding="utf-8"))
            tampered["input_sha256"] = "0" * 64
            output.write_text(
                json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
            )
            write_stage_marker(
                prepare / "raw_asr_v2.done.json",
                stage="raw_asr_v2",
                input_sha256=data["input_sha256"],
                outputs={"raw_asr_v2": output},
            )
            with patch.object(
                raw_asr_v2_module,
                "_import_whisper",
                side_effect=AssertionError("model inference must not run"),
            ):
                with self.assertRaisesRegex(
                    TranscriptionError, "explicit force rerun is required"
                ):
                    transcribe_raw_audio_v2(audio, prepare, episode=12)

    def test_missing_caption_fails_before_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with patch.object(
                raw_asr_v2_module,
                "_import_whisper",
                side_effect=AssertionError("model import happened too early"),
            ) as import_mock:
                with self.assertRaisesRegex(
                    TranscriptionError, "Caption file does not exist"
                ):
                    transcribe_raw_audio_v2(
                        audio,
                        prepare,
                        episode=12,
                        captions_path=Path(directory) / "missing.vtt",
                    )
            import_mock.assert_not_called()

    def test_main_pass_disables_previous_text_conditioning_by_default(self) -> None:
        observed: list[bool] = []

        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **kwargs: object) -> tuple[object, Info]:
                observed.append(bool(kwargs["condition_on_previous_text"]))
                return iter([_complete_segment()]), Info()

        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                transcribe_raw_audio_v2(audio, prepare, episode=12)

        self.assertEqual(observed, [False])
        self.assertFalse(RawASRV2Config().condition_on_previous_text)
        self.assertFalse(
            RawASRV2Config().transcription_config().condition_on_previous_text
        )

    def test_full_length_policy_rejects_first_990ms_original_word_gap(self) -> None:
        settings = RawASRV2Config()
        largest_accepted = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 5_000}],
            [
                {"start_ms": 0, "end_ms": 1_000},
                {"start_ms": 1_989, "end_ms": 5_000},
            ],
            config=settings.speech_coverage_config(),
        )
        first_rejected = analyze_speech_coverage(
            [{"start_ms": 0, "end_ms": 5_000}],
            [
                {"start_ms": 0, "end_ms": 1_000},
                {"start_ms": 1_990, "end_ms": 5_000},
            ],
            config=settings.speech_coverage_config(),
        )

        self.assertEqual(
            settings.coverage_policy_label,
            "v2-beta-full-length-dialogue-v1",
        )
        self.assertEqual(settings.word_padding_ms, 120)
        self.assertEqual(settings.word_merge_gap_ms, 300)
        self.assertEqual(settings.rescue_min_hole_ms, 750)
        self.assertEqual(settings.rescue_min_coverage_ratio, 0.70)
        self.assertEqual(largest_accepted["status"], "PASS")
        self.assertEqual(first_rejected["status"], "FAIL")
        self.assertEqual(first_rejected["unresolved_speech_region_count"], 1)
        self.assertEqual(
            [
                (issue["start_ms"], issue["end_ms"])
                for issue in first_rejected["coverage_issues"]
            ],
            [(1_120, 1_870)],
        )

    def test_beta_vad_configuration_keeps_sub_250ms_speech_eligible(self) -> None:
        settings = RawASRV2Config()
        transcription = settings.transcription_config()

        self.assertEqual(settings.vad_min_speech_ms, 120)
        self.assertEqual(
            transcription.vad_parameters["min_speech_duration_ms"],
            120,
        )

    def test_word_timing_vad_fallback_cannot_complete_default_v2_stage(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        fallback_vad = (
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "source": "vad_filtered_word_timing_fallback",
                }
            ],
            "silero_vad_returned_no_regions",
        )
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=fallback_vad,
                ),
            ):
                with self.assertRaisesRegex(
                    TranscriptionError, "requires independent audio VAD"
                ):
                    transcribe_raw_audio_v2(audio, prepare, episode=12)

            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())
            self.assertFalse((prepare / "raw_asr_v2.json").exists())

    def test_explicit_debug_override_marks_non_independent_vad_artifact(self) -> None:
        class Model:
            def __init__(self, _name: str, **_kwargs: object):
                pass

            def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                return iter([_complete_segment()]), Info()

        fallback_vad = (
            [
                {
                    "vad_region_index": 1,
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "source": "vad_filtered_word_timing_fallback",
                }
            ],
            "silero_vad_returned_no_regions",
        )
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=fallback_vad,
                ),
            ):
                data = transcribe_raw_audio_v2(
                    audio,
                    prepare,
                    episode=12,
                    config=RawASRV2Config(require_independent_vad=False),
                )
            production_load = load_valid_raw_asr_v2(
                prepare,
                audio_path=audio,
                episode=12,
                require_independent_vad=True,
            )
            diagnostic_load = load_valid_raw_asr_v2(
                prepare,
                audio_path=audio,
                episode=12,
                require_independent_vad=False,
            )

        self.assertFalse(data["independent_vad"])
        self.assertFalse(data["model"]["settings"]["require_independent_vad"])
        self.assertIsNone(production_load)
        self.assertIsNotNone(diagnostic_load)

    def test_audio_change_during_lazy_asr_prevents_output_and_marker_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)

            class Model:
                def __init__(self, _name: str, **_kwargs: object):
                    pass

                def transcribe(self, _path: str, **_kwargs: object) -> tuple[object, Info]:
                    audio.write_bytes(b"changed-while-the-lazy-model-was-running")
                    return iter([_complete_segment()]), Info()

            with (
                patch.object(
                    raw_asr_v2_module,
                    "_import_whisper",
                    return_value=(Model, FakeCTranslate2),
                ),
                patch.object(
                    raw_asr_v2_module,
                    "_extract_vad_regions",
                    return_value=self._vad(),
                ),
            ):
                with self.assertRaisesRegex(
                    TranscriptionError, "Audio input changed during raw ASR"
                ):
                    transcribe_raw_audio_v2(audio, prepare, episode=12)

            self.assertFalse((prepare / "raw_asr_v2.json").exists())
            self.assertFalse((prepare / "raw_asr_v2.done.json").exists())

    def test_invalid_episode_fails_before_creating_stage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio, prepare = self._audio_and_prepare(directory)
            for invalid in (0, -1, True):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(
                        TranscriptionError, "episode must be a positive integer"
                    ):
                        transcribe_raw_audio_v2(audio, prepare, episode=invalid)
            self.assertFalse(prepare.exists())


if __name__ == "__main__":
    unittest.main()
