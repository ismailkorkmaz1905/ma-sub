from mas.pipeline import STRICT_DRIVE_OUTPUTS


def test_drive_contains_only_final_episode_outputs():
    assert STRICT_DRIVE_OUTPUTS == ("mp4",)
