from mas.pipeline import STRICT_DRIVE_OUTPUTS


def test_drive_contains_only_final_subtitles():
    assert STRICT_DRIVE_OUTPUTS == ("tr_srt", "id_srt")
