import json
from pathlib import Path
import shutil
import subprocess

import pytest

from mas import full_delivery as full


def test_concat_quotes_paths_without_shell_interpolation(tmp_path):
    line = full._concat_line(tmp_path / "video 'quoted'.mp4", 12.5)
    assert "'\\''" in line
    assert 'duration 12.500000000' in line
    with pytest.raises(ValueError):
        full._concat_line(tmp_path / 'unsafe\n.mp4', 1)


def test_incompatible_streams_are_rejected():
    with pytest.raises(ValueError):
        full._streams({'streams': [{'codec_type': 'video'}]})


def test_real_cpu_media_assembles_without_reencoding(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/FFprobe unavailable')
    parts = []
    for i in range(2):
        output = tmp_path / f'part-{i}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=160x90:rate=25',
            '-f', 'lavfi', '-i', 'sine=frequency=1000:sample_rate=48000', '-t', '1',
            '-vf', "drawtext=text='caption':x=20:y=20:fontsize=12:fontcolor=white",
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2', '-y', str(output)],
            capture_output=True, timeout=30, check=True)
        parts.append((output, 1))
    output = tmp_path / 'full.mp4'
    result = full.assemble_parts(parts, output, duration_seconds=2, remaining=lambda: 30)
    assert output.is_file() and abs(float(result['format']['duration']) - 2) <= .2
    assert full._streams(result)[0]['codec_name'] == 'h264'
    assert all(path.is_file() for path, _ in parts)


@pytest.fixture
def staged_episode(tmp_path, monkeypatch):
    from mas import delivery_first as df, partial_delivery
    from mas.engine import part_audio, partial_finalize, partial_encode
    from mas.engine.episode_archive import file_record
    from mas.reliability import atomic_json
    from mas.source_discovery import CHANNEL_VIDEOS_URL
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '4' * 64)
    (tmp_path / 'work').mkdir()
    (tmp_path / 'source').mkdir()
    atomic_json(tmp_path / 'work/part-plan.json', {'test': 'original plan'})
    atomic_json(tmp_path / 'source/official-source.json', {
        'episode': 14, 'title': 'Muhtemel Ask 14. Bolum', 'channel_url': CHANNEL_VIDEOS_URL})
    parts, artifacts = [], {}
    for i in range(2):
        ident = f'part-{i + 1:03d}'
        scope = {'part_id': ident, 'start_sample': i * 16000, 'end_sample': (i + 1) * 16000}
        parts.append(scope)
        folder = tmp_path / 'parts' / ident
        (folder / 'work').mkdir(parents=True)
        (folder / 'final').mkdir()
        atomic_json(folder / 'work/partial-export.json', {'mode': df.EXPORT_MODE, 'part_id': ident})
        mp4 = folder / 'final/burned.mp4'
        mp4.write_bytes(b'already burned caption')
        artifacts[ident] = mp4
    plan = {'parts': parts, 'audio': {'sample_count': 32000}, 'source': {'sha256': 'f' * 64}}
    monkeypatch.setattr(part_audio, 'load_part_plan', lambda *a, **kw: plan)
    monkeypatch.setattr(partial_delivery, 'validate_published_part', lambda *a, **kw: {'verified': True})
    monkeypatch.setattr(partial_finalize, 'validate_partial_export', lambda *a, **kw: (
        {'mode': df.EXPORT_MODE}, {'warnings': [{'reason': 'test quality warning'}]}))
    monkeypatch.setattr(partial_encode, 'validate_partial_encoding', lambda root, ep, ident, **kw: ({}, artifacts[ident]))
    from mas import notify
    monkeypatch.setattr(notify, 'enqueue_notification', lambda *a, **kw: None)
    return tmp_path, plan


def test_full_delivery_waits_for_first_part_readback(staged_episode, monkeypatch):
    from mas import partial_delivery
    root, _ = staged_episode
    monkeypatch.setattr(partial_delivery, 'validate_published_part', lambda *a, **kw: None)
    monkeypatch.setattr(full, 'assemble_parts', lambda *a, **kw: pytest.fail('assembled before first readback'))
    assert full.publish_full_episode(root, 14, 'drive:delivery', total_timeout=30) is None
    assert not (root / 'final/delivery-first/full-drive-receipt.json').exists()


def test_upload_retry_reuses_assembled_video_and_never_passes_bad_hash(staged_episode, monkeypatch):
    from mas import remote
    from mas.hashing import sha256_file
    root, _ = staged_episode
    calls = {'assemble': 0, 'upload': 0}
    def assemble(parts, output, **kwargs):
        calls['assemble'] += 1
        output.write_bytes(b'full already burned episode')
        return {'format': {'duration': 2}}
    def upload(path, target, **kwargs):
        calls['upload'] += 1
        return {'remote': target, 'bytes': path.stat().st_size,
                'sha256': ('0' * 64 if calls['upload'] == 1 else sha256_file(path))}
    monkeypatch.setattr(full, 'assemble_parts', assemble)
    monkeypatch.setattr(remote, 'upload_verified', upload)
    with pytest.raises(ValueError, match='readback'):
        full.publish_full_episode(root, 14, 'drive:delivery', total_timeout=30)
    assert not (root / 'final/delivery-first/full-drive-receipt.json').exists()
    result = full.publish_full_episode(root, 14, 'drive:delivery', total_timeout=30)
    assert result['status'] == 'DELIVERED_WITH_WARNINGS'
    assert result['single_full_episode_file'] is True
    assert calls == {'assemble': 1, 'upload': 2}
    assert full.publish_full_episode(root, 14, 'drive:delivery', total_timeout=30) == result
    assert calls == {'assemble': 1, 'upload': 2}
