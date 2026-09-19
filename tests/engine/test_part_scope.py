import copy

import pytest

from mas.engine import part_scope
from mas.engine.part_scope import PartScopeError, build_part_plan, project_part_vad, validate_part_plan


def plan_case(*, samples=7200000 * 16 + 7):
    record = lambda name: {"relative_path": name, "sha256": "a" * 64, "size_bytes": 100}
    audio = {**record("work/audio.flac"), "sample_rate_hz": 16000,
             "channels": 1, "sample_count": samples}
    regions = [
        {"vad_region_index": 1, "start_ms": 1000, "end_ms": 3599000, "source": "silero_vad"},
        {"vad_region_index": 2, "start_ms": 3603000, "end_ms": 7199000, "source": "silero_vad"},
    ]
    return build_part_plan(
        episode=14, source=record("source/episode.mp4"), audio=audio,
        vad={"independent_vad": True, "audio_sha256": audio["sha256"],
             "sample_count": samples, "config": {"threshold": .5},
             "model": {"test_model_sha256": "b" * 64}, "producer_sha256": "c" * 64,
             "regions": regions},
        captions={**record("source/captions.vtt"), "records": [
            {"caption_index": 1, "start_ms": 1000, "end_ms": 2000, "text": "Zeytin getirdim."},
            {"caption_index": 2, "start_ms": 3603100, "end_ms": 3604000, "text": "Yarın döneriz."},
        ]},
    )


def test_plan_keeps_exact_sample_partition_and_rebases_only_derived_vad():
    plan = plan_case()
    original = copy.deepcopy(plan)
    first, second = plan["parts"]
    assert first["part_id"] == "part-001"
    assert first["end_ms"] == 3602880
    assert first["end_sample"] == second["start_sample"]
    assert second["end_sample"] == 7200000 * 16 + 7
    assert second["end_ms"] == 7200001
    assert project_part_vad(plan, "part-002") == [
        {"vad_region_index": 2, "start_ms": 120,
         "end_ms": 3596120, "source": "silero_vad"}
    ]
    assert validate_part_plan(plan) == plan
    assert original == plan


@pytest.mark.parametrize("mutation", [
    lambda plan: plan["parts"].pop(),
    lambda plan: plan["parts"][0]["vad_region_indices"].clear(),
    lambda plan: plan["parts"][1]["caption_indices"].clear(),
    lambda plan: plan["parts"][1].update(start_sample=0),
    lambda plan: plan["parts"][0].update(end_sample=3601000 * 16 - 1),
    lambda plan: plan["parts"][0].update(end_ms=3598000),
    lambda plan: plan["parts"][1].update(part_id="../../source"),
    lambda plan: plan["parts"].reverse(),
    lambda plan: plan["parts"][1].update(end_sample=7200000 * 16),
])
def test_plan_rejects_omission_overlap_gaps_and_changed_boundaries(mutation):
    plan = plan_case()
    mutation(plan)
    with pytest.raises(PartScopeError):
        validate_part_plan(plan)


def test_speaker_overlap_and_crossing_caption_are_kept_together():
    plan = plan_case()
    plan["vad"]["regions"].insert(1, {
        "vad_region_index": 2, "start_ms": 3598500, "end_ms": 3606000,
        "source": "silero_vad", "speaker_id": "B",
    })
    plan["vad"]["regions"][2]["vad_region_index"] = 3
    with pytest.raises(PartScopeError, match="no proven speech/caption gap"):
        build_part_plan(**{key: plan[key] for key in ("episode", "source", "audio", "vad", "captions")})
    plan = plan_case()
    plan["captions"]["records"].insert(1, {
        "caption_index": 2, "start_ms": 3598500, "end_ms": 3606000, "text": "Konuşma devam ediyor."
    })
    plan["captions"]["records"][2]["caption_index"] = 3
    with pytest.raises(PartScopeError, match="no proven speech/caption gap"):
        build_part_plan(**{key: plan[key] for key in ("episode", "source", "audio", "vad", "captions")})


def test_boundary_reserves_last_short_cue_duration_and_next_word_guard():
    from mas.engine.segmentation import build_blocks

    plan = plan_case()
    plan["captions"] = None
    plan["vad"]["regions"][0]["end_ms"] = 3599500
    plan["vad"]["regions"][1]["start_ms"] = 3600500
    plan = build_part_plan(**{key: plan[key] for key in ("episode", "source", "audio", "vad", "captions")})
    first, second = plan["parts"]
    assert first["end_ms"] == 3600380
    assert second["start_sample"] == first["end_sample"]
    assert project_part_vad(plan, second["part_id"])[0]["start_ms"] == 120
    words = [{"word_index": 1, "start_ms": 3599400, "end_ms": 3599500,
              "text": "Tamam.", "utterance_uid": "scene-final"}]
    blocks = build_blocks({"words": words}, 14, end_boundary_ms=first["end_ms"])
    assert blocks[0]["start_ms"] == 3599400
    assert blocks[0]["end_ms"] == 3600100
    assert blocks[0]["end_ms"] <= first["end_ms"]


@pytest.mark.parametrize("path", ["../audio.flac", "/audio.flac", "C:/audio.flac", "a/../audio.flac", "a\\audio.flac"])
def test_manifest_paths_cannot_escape_episode_root(path):
    plan = plan_case()
    plan["audio"]["relative_path"] = path
    with pytest.raises(PartScopeError):
        validate_part_plan(plan)


def test_boundary_requires_independent_vad_not_asr_fallback():
    plan = plan_case()
    plan["vad"]["regions"][0]["source"] = "vad_filtered_word_timing_fallback"
    with pytest.raises(PartScopeError, match="ASR-derived"):
        validate_part_plan(plan)


def test_part_count_is_bounded(monkeypatch):
    plan = plan_case()
    monkeypatch.setattr(part_scope, "TARGET_DURATION_MS", 2000)
    plan["audio"]["sample_count"] = plan["vad"]["sample_count"] = 200000 * 16
    plan["vad"]["regions"] = [
        {"vad_region_index": index + 1, "start_ms": index * 2500,
         "end_ms": index * 2500 + 1000, "source": "silero_vad"}
        for index in range(80)
    ]
    with pytest.raises(PartScopeError, match="maximum 64"):
        build_part_plan(episode=14, source=plan["source"], audio=plan["audio"], vad=plan["vad"])
