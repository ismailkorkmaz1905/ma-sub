import shutil
import subprocess

import pytest

from mas.engine.burned_mp4 import burn_indonesian_mp4


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_burn_renders_text_preserves_source_and_binds_reuse(tmp_path):
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=black:s=320x180:r=25', '-f', 'lavfi', '-i', 'sine=frequency=440',
                    '-t', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)],
                   check=True, timeout=30)
    source_bytes = source.read_bytes()
    subtitles = tmp_path / 'id.srt'
    subtitles.write_text('1\n00:00:00,200 --> 00:00:01,800\nHalo dunia.\n', encoding='utf-8')
    output = tmp_path / 'burned.mp4'
    report = burn_indonesian_mp4(source, subtitles, output, encoder='libx264', timeout_seconds=30)
    assert source.read_bytes() == source_bytes
    assert report['width'] == 320 and report['height'] == 180
    assert report['subtitle_blocks'] == 1 and report['perceptual_acceptance'] == 'NOT_ASSERTED'
    frame = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '1', '-i', str(output),
                            '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                           capture_output=True, check=True, timeout=30).stdout
    assert max(frame) > 200
    assert sum(value > 100 for value in frame) > 20
    assert burn_indonesian_mp4(source, subtitles, output, encoder='libx264') == report
    subtitles.write_text('1\n00:00:00,200 --> 00:00:01,800\nChanged.\n', encoding='utf-8')
    with pytest.raises(ValueError, match='preserve'):
        burn_indonesian_mp4(source, subtitles, output, encoder='libx264')
