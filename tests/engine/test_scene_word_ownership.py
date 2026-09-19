import copy

import pytest

from mas.engine.segmentation import SegmentationError, build_blocks, validate_segmentation
from mas.engine.srt import wrap_text


def scene_transcription():
    return {
        "words": [
            {
                "word_index": index,
                "segment_id": 1 if start < 30_000 else 2,
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "probability": 0.99,
            }
            for index, (start, end, text) in enumerate(
                [
                    (1_000, 1_300, "Ben"),
                    (1_320, 1_620, "elma"),
                    (1_640, 2_000, "aldım."),
                    (32_000, 32_300, "Sen"),
                    (32_320, 32_620, "armut"),
                    (32_640, 33_000, "al."),
                ],
                start=1,
            )
        ]
    }


def test_thirty_second_silence_preserves_next_scene_first_word():
    transcription = scene_transcription()
    blocks = build_blocks(transcription, episode=13)
    assert [block["timing_text"] for block in blocks] == [
        "Ben elma aldım.", "Sen armut al."
    ]
    assert blocks[0]["end_ms"] < 32_000
    assert blocks[1]["start_ms"] == 32_000
    assert validate_segmentation(blocks, transcription["words"]).valid


def test_validator_rejects_scene_leak_even_when_total_text_is_unchanged():
    transcription = scene_transcription()
    blocks = build_blocks(transcription, episode=13)
    blocks[0]["timing_text"] = blocks[0]["primary_text"] = "Ben elma aldım. Sen"
    blocks[1]["timing_text"] = blocks[1]["primary_text"] = "armut al."
    report = validate_segmentation(blocks, transcription["words"])
    assert report.source_text_mismatch_count == 0
    assert not report.valid
    assert any("word outside its acoustic interval" in error for error in report.errors)


def test_validator_rejects_missing_spoken_scene():
    transcription = scene_transcription()
    blocks = build_blocks(transcription, episode=13)
    report = validate_segmentation(blocks[:1], transcription["words"])
    assert not report.valid
    assert report.source_text_mismatch_count == 1


def test_validator_rejects_leak_in_display_text_even_if_timing_text_is_unchanged():
    transcription = scene_transcription()
    blocks = build_blocks(transcription, episode=13)
    blocks[0]["primary_text"] = "Ben elma aldım. Sen"
    blocks[1]["primary_text"] = "armut al."
    report = validate_segmentation(blocks, transcription["words"])
    assert not report.valid
    assert report.source_text_mismatch_count == 1
    assert any("primary_text differs" in error for error in report.errors)


def test_validator_rejects_cue_ending_before_its_last_spoken_word():
    transcription = scene_transcription()
    blocks = build_blocks(transcription, episode=13)
    blocks[0]["end_ms"] = 1_950
    report = validate_segmentation(blocks, transcription["words"])
    assert not report.valid
    assert any("word outside its acoustic interval" in error for error in report.errors)


@pytest.mark.parametrize("phrases", [
    ("Ben elma aldım.", "Sen armut al."),
    ("Zeytin ekmek getirdim.", "Gökyüzü bugün kapalı."),
    ("Burada seni bekledik.", "Yarın yine döneriz."),
])
@pytest.mark.parametrize("lead_ms", [1, 80, 4_000])
def test_validator_rejects_early_scene_start_from_source_not_mutable_metadata(phrases, lead_ms):
    transcription = scene_transcription()
    for word, text in zip(transcription["words"], " ".join(phrases).split(), strict=True):
        word["text"] = text
    blocks = build_blocks(transcription, episode=14)
    blocks[1]["start_ms"] -= lead_ms
    blocks[1]["vad_info"]["first_word_start_ms"] = blocks[1]["start_ms"]
    original = copy.deepcopy(blocks)
    report = validate_segmentation(blocks, transcription["words"])
    assert not report.valid
    assert any("block 2: cue starts before its first acoustic word" in error
               for error in report.errors)
    assert not any(error.startswith("block 1:") for error in report.errors)
    assert blocks == original


def test_word_ownership_preserves_different_known_speaker_overlap():
    transcription = scene_transcription()
    for word in transcription["words"]:
        word["speaker_id"] = "A" if word["start_ms"] < 30_000 else "B"
        if word["speaker_id"] == "B":
            word["start_ms"] -= 30_500
            word["end_ms"] -= 30_500
    transcription["words"].sort(key=lambda word: word["start_ms"])
    for index, word in enumerate(transcription["words"], start=1):
        word["word_index"] = index
    original = copy.deepcopy(transcription)
    blocks = build_blocks(transcription, episode=13)
    assert len(blocks) == 2
    assert blocks[0]["end_ms"] > blocks[1]["start_ms"]
    assert validate_segmentation(blocks, transcription["words"]).valid
    assert transcription == original


def test_segmentation_splits_for_42_character_lines_before_pack_freeze():
    transcription = {"words": [
        {"word_index": index, "segment_id": 1, "start_ms": 1000 + index * 400,
         "end_ms": 1350 + index * 400, "text": "kelime"}
        for index in range(1, 16)
    ]}
    blocks = build_blocks(transcription, episode=13)
    assert len(blocks) > 1
    assert validate_segmentation(blocks, transcription["words"]).valid
    assert " ".join(block["timing_text"] for block in blocks) == " ".join(["kelime"] * 15)
    for block in blocks:
        assert all(len(line) <= 42 for line in wrap_text(
            block["timing_text"], target=42, max_lines=2, hard_limit=42
        ).splitlines())


def test_unsplittable_long_word_is_not_truncated_or_timed_synthetically():
    transcription = {"words": [
        {"word_index": 1, "segment_id": 1, "start_ms": 1000,
         "end_ms": 4000, "text": "x" * 43}
    ]}
    with pytest.raises(SegmentationError, match="No safe segmentation"):
        build_blocks(transcription, episode=13)


def test_validator_rejects_invented_dialogue_on_a_new_speaker_lane():
    transcription = scene_transcription()
    for word in transcription["words"]:
        word["speaker_id"] = "A"
    blocks = build_blocks(transcription, episode=13)
    invented = copy.deepcopy(blocks[-1])
    invented.update(block_index=3, speaker_id="B", start_ms=40_000, end_ms=42_000,
                    primary_text="Uydurma.", timing_text="Uydurma.")
    report = validate_segmentation([*blocks, invented], transcription["words"])
    assert not report.valid
    assert report.source_text_mismatch_count == 1
