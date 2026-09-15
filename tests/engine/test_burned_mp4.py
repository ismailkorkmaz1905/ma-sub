import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import mas.engine.burned_mp4 as burned_module
from mas.engine.burned_mp4 import (_run_encode, _tree_bytes, burn_indonesian_mp4,
                                   create_encoding_samples, inspect_encoding_storage,
                                   plan_encoding_settings, qualify_encoding)


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
    report = burn_indonesian_mp4(source, subtitles, output, encoder='libx264', timeout_seconds=30,
                                target_size_gb=.001, idle_timeout_seconds=10)
    assert source.read_bytes() == source_bytes
    assert report['width'] == 320 and report['height'] == 180
    assert report['subtitle_blocks'] == 1 and report['perceptual_acceptance'] == 'NOT_ASSERTED'
    frame = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '1', '-i', str(output),
                            '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                           capture_output=True, check=True, timeout=30).stdout
    assert max(frame) > 200
    assert sum(value > 100 for value in frame) > 20
    assert report['encoding_settings']['target_policy'] == 'SOFT_TARGET'
    assert report['encoding_settings']['planned_video_bitrate_bps'] == 3_808_000
    assert report['encoding_settings']['identity_sha256']
    assert burn_indonesian_mp4(source, subtitles, output, encoder='libx264',
                               target_size_gb=.001) == report
    with pytest.raises(ValueError, match='encoding'):
        burn_indonesian_mp4(source, subtitles, output, encoder='libx264', target_size_gb=.002)
    subtitles.write_text('1\n00:00:00,200 --> 00:00:01,800\nChanged.\n', encoding='utf-8')
    with pytest.raises(ValueError, match='preserve'):
        burn_indonesian_mp4(source, subtitles, output, encoder='libx264', target_size_gb=.001)


def test_custom_options_require_hash_bound_sample_approval(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    subtitles = tmp_path / 'id.srt'
    output = tmp_path / 'burned.mp4'
    source.write_bytes(b'source')
    subtitles.write_text('1\n00:00:00,000 --> 00:00:01,000\nHalo.\n', encoding='utf-8')
    monkeypatch.setattr('mas.engine.burned_mp4._probe', lambda _: {
        'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'pix_fmt': 'yuv420p',
                     'width': 320, 'height': 180}], 'format': {'duration': '2'}})
    with pytest.raises(ValueError, match='approval record'):
        burn_indonesian_mp4(source, subtitles, output, encoder='libx264',
                            encoder_options=['-crf', '20'], target_size_gb=.001)
    plan, _ = plan_encoding_settings(source, encoder='libx264', target_size_gb=.001,
                                     encoder_options=['-crf', '20'])
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'approved': True,
                                    'encoding_settings_identity_sha256': plan['identity_sha256'],
                                    'source_sha256': '0' * 64,
                                    'source_duration_seconds': plan['source_duration_seconds']}), encoding='utf-8')
    with pytest.raises(ValueError, match='does not approve'):
        burn_indonesian_mp4(source, subtitles, output, encoder='libx264',
                            encoder_options=['-crf', '20'], sample_approval_path=approval,
                            target_size_gb=.001)


def test_plan_rejects_bool_nan_duration_and_unapproved_option(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    monkeypatch.setattr(burned_module, '_probe', lambda _: {'format': {'duration': '2'}})
    with pytest.raises(ValueError, match='finite'):
        plan_encoding_settings(source, target_size_gb=True)
    with pytest.raises(ValueError, match='allowlist'):
        plan_encoding_settings(source, encoder='libx264', encoder_options=['-filter_complex', 'null'])
    monkeypatch.setattr(burned_module, '_probe', lambda _: {'format': {'duration': 'nan'}})
    with pytest.raises(ValueError, match='duration'):
        plan_encoding_settings(source)


def test_nvenc_default_uses_target_bitrate_without_cq_and_explicit_cq_changes_identity(
        tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    monkeypatch.setattr(burned_module, '_probe', lambda _: {
        'format': {'duration': '3600'},
        'streams': [{'codec_type': 'video', 'codec_name': 'av1', 'pix_fmt': 'yuv420p',
                     'width': 1920, 'height': 1080}]})
    default, _ = plan_encoding_settings(source, encoder='h264_nvenc', target_size_gb=3.0)
    explicit, _ = plan_encoding_settings(
        source, encoder='h264_nvenc', target_size_gb=3.0,
        encoder_options=['-preset', 'p4', '-rc', 'vbr', '-cq', '19'])
    assert default['encoder_options'] == ['-preset', 'p4', '-rc', 'vbr']
    assert '-cq' not in default['encoder_options']
    assert explicit['identity_sha256'] != default['identity_sha256']


def test_network_volume_storage_uses_declared_quota_and_actual_tree_bytes(tmp_path):
    (tmp_path / 'one').write_bytes(b'1234')
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'two').write_bytes(b'56789')
    result = inspect_encoding_storage(nested, network_volume_root=tmp_path,
                                      network_volume_quota_bytes=100)
    assert result['network_volume_used_bytes'] == 9
    assert result['network_volume_free_bytes'] == 91
    assert result['network_volume_scan_elapsed_seconds'] >= 0
    with pytest.raises(ValueError, match='actual quota'):
        inspect_encoding_storage(tmp_path, network_volume_root=tmp_path)


def test_network_volume_usage_deduplicates_hardlinks_and_surfaces_stat_error(tmp_path, monkeypatch):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.write_bytes(b'12345')
    try:
        os.link(first, second)
    except OSError:
        pytest.skip('hardlinks unavailable')
    assert _tree_bytes(tmp_path) == 5
    original = os.scandir
    class Entry:
        def __init__(self, entry):
            self.entry = entry
            self.name, self.path = entry.name, entry.path
        def is_symlink(self): return self.entry.is_symlink()
        def is_dir(self, **kwargs): return self.entry.is_dir(**kwargs)
        def is_file(self, **kwargs): return self.entry.is_file(**kwargs)
        def stat(self, **kwargs):
            if self.name == 'first':
                raise PermissionError('denied')
            return self.entry.stat(**kwargs)
    class Scan:
        def __init__(self, path): self.scan = original(path)
        def __enter__(self): return [Entry(entry) for entry in self.scan.__enter__()]
        def __exit__(self, *args): return self.scan.__exit__(*args)
    monkeypatch.setattr(os, 'scandir', Scan)
    with pytest.raises(PermissionError, match='denied'):
        _tree_bytes(tmp_path)


def test_encode_idle_watchdog_preserves_log_and_partial(tmp_path):
    partial = tmp_path / 'partial.mp4'
    log = tmp_path / 'encode.log'
    script = "from pathlib import Path; import time; Path('partial.mp4').write_bytes(b'partial'); time.sleep(5)"
    with pytest.raises(TimeoutError, match='frame progress watchdog expired'):
        _run_encode([sys.executable, '-c', script], tmp_path, log, partial, 2, .2)
    assert 'frame progress watchdog expired' in log.read_text(encoding='utf-8')
    assert log.with_suffix('.failed.mp4').read_bytes() == b'partial'


def test_encode_marks_real_progress_and_cleans_up_on_observer_interrupt(tmp_path, monkeypatch):
    marks = []
    monkeypatch.setattr(burned_module, 'mark_work_progress',
                        lambda stage, **fields: marks.append((stage, fields)))
    script = "print('frame=1', flush=True); print('out_time_us=1000', flush=True)"
    _run_encode([sys.executable, '-u', '-c', script], tmp_path, tmp_path / 'ok.log',
                tmp_path / 'none.mp4', 2, 1)
    assert marks == [('burned_mp4', {'completed': 1}), ('burned_mp4', {'completed': 1})]

    processes = []
    original = subprocess.Popen
    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(burned_module.subprocess, 'Popen', start)
    monkeypatch.setattr(burned_module, 'mark_work_progress',
                        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    script = "import time; print('frame=1', flush=True); time.sleep(5)"
    with pytest.raises(KeyboardInterrupt):
        _run_encode([sys.executable, '-u', '-c', script], tmp_path, tmp_path / 'interrupt.log',
                    tmp_path / 'none.mp4', 2, 1)
    assert processes[0].poll() is not None


def test_create_three_fixed_rebased_samples_without_approval_claim(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    subtitles = tmp_path / 'id.srt'
    subtitles.write_text(
        '1\n00:00:05,000 --> 00:00:07,000\nFirst.\n\n'
        '2\n00:00:25,000 --> 00:00:27,000\nMiddle.\n\n'
        '3\n00:00:50,000 --> 00:00:52,000\nLast.\n', encoding='utf-8')
    source_probe = {'streams': [{'codec_type': 'video', 'codec_name': 'h264',
                                 'pix_fmt': 'yuv420p', 'width': 320, 'height': 180}],
                    'format': {'duration': '60'}}
    sample_probe = {'streams': [{'codec_type': 'video', 'codec_name': 'h264',
                                 'pix_fmt': 'yuv420p', 'width': 320, 'height': 180},
                                {'codec_type': 'audio', 'codec_name': 'aac'}],
                    'format': {'duration': '15'}}
    monkeypatch.setattr(burned_module, '_probe',
                        lambda path: sample_probe if Path(path).name.startswith('sample-') else source_probe)
    calls = []
    def encode(command, cwd, log, output, total, idle):
        calls.append((command, total))
        output.write_bytes(b'sample-' + str(len(calls)).encode())
    monkeypatch.setattr(burned_module, '_run_encode', encode)
    manifest_path = create_encoding_samples(source, subtitles, tmp_path / 'samples',
                                            encoder='libx264', target_size_gb=.001)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert len(calls) == 3 and [item['source_start_seconds'] for item in manifest['samples']] == [0.0, 22.5, 45.0]
    assert manifest['status'] == 'REVIEW_REQUIRED'
    assert manifest['perceptual_acceptance'] == 'NOT_ASSERTED'
    assert manifest['encoding_settings']['source_sha256']
    assert manifest['inputs']['id_srt_sha256']
    assert manifest['style'] == burned_module.SUBTITLE_STYLE
    assert (tmp_path / 'samples' / 'sample-2.srt').read_text(encoding='utf-8').startswith(
        '1\n00:00:02,500 --> 00:00:04,500\nMiddle.')


def test_technical_qualification_is_bound_reusable_and_never_quality_approval(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    output = tmp_path / 'technical'
    settings = {'identity_sha256': 'a' * 64, 'source_sha256': hashlib.sha256(b'source').hexdigest(),
                'source_duration_seconds': 60.0}
    monkeypatch.setattr(burned_module, 'plan_encoding_settings',
                        lambda *args, **kwargs: (settings, {'format': {'duration': '60'}}))
    hardware = {'ffmpeg_version': 'ffmpeg test', 'gpu': 'GPU, uuid, driver'}
    monkeypatch.setattr(burned_module, '_encoding_hardware', lambda: hardware)
    def samples(source_path, subtitle_path, output_dir, **kwargs):
        records = []
        for index in range(1, 4):
            path = Path(output_dir) / f'sample-{index}.mp4'
            subtitle = Path(output_dir) / f'sample-{index}.srt'
            path.write_bytes(f'sample-{index}'.encode())
            subtitle.write_bytes(f'subtitle-{index}'.encode())
            records.append({'output_file': path.name, 'output_bytes': path.stat().st_size,
                            'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'subtitle_file': subtitle.name, 'subtitle_bytes': subtitle.stat().st_size,
                            'subtitle_sha256': hashlib.sha256(subtitle.read_bytes()).hexdigest()})
        manifest = {'measured_encode_seconds': 3.0, 'projected_full_encode_seconds': 12.0,
                    'samples': records}
        path = Path(output_dir) / 'encoding-samples.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path
    monkeypatch.setattr(burned_module, 'create_encoding_samples', samples)
    receipt_path = qualify_encoding(source, output)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['status'] == 'TECHNICALLY_VERIFIED'
    assert receipt['scope'] == 'ENCODER_ONLY'
    assert receipt['subtitle_kind'] == 'SYNTHETIC_TECHNICAL_TEST'
    assert receipt['perceptual_acceptance'] == receipt['translation_acceptance'] == 'NOT_ASSERTED'
    assert receipt['hardware'] == hardware and receipt['projected_full_encode_seconds'] == 12.0
    assert qualify_encoding(source, output) == receipt_path
    monkeypatch.setattr(burned_module, '_encoding_hardware',
                        lambda: {'ffmpeg_version': 'ffmpeg test', 'gpu': 'different GPU'})
    with pytest.raises(ValueError, match='identity changed'):
        qualify_encoding(source, output)
