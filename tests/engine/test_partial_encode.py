import json

import pytest

from test_partial_finalize import partial_fixture
from mas import partial_delivery
from mas.engine import partial_encode, partial_finalize
from mas.reliability import atomic_json


def encoded_fixture(partial_fixture, monkeypatch):
    root, child, _, _, _, _ = partial_fixture
    partial_finalize.finalize_partial_episode(root, 12, 'part-001')
    calls = []
    def release(*args, **kwargs):
        assert 0 < kwargs['total_timeout'] <= 120
        calls.append('released')
    monkeypatch.setattr(partial_delivery, 'validate_part_release', release)
    monkeypatch.setattr(partial_encode.burn, '_qsv_hardware', lambda **kw: {'qsv_hardware_probe': 'PASS'})
    def probe(path):
        duration = '7' if path.name == 'partial.mp4' else '15' if path.name.startswith('sample-') else '60'
        return {'format': {'duration': duration}, 'streams': [
            {'codec_type': 'video', 'codec_name': 'h264', 'pix_fmt': 'yuv420p', 'width': 320, 'height': 180},
            {'codec_type': 'audio', 'codec_name': 'aac'}]}
    monkeypatch.setattr(partial_encode.burn, '_probe', probe)
    def encode(command, cwd, log, output, total, idle):
        assert 0 < idle <= total <= 120
        assert calls and calls[0] == 'released'
        calls.append(command)
        output.write_bytes(b'encoded synthetic output')
        log.write_text('fixture encoder', encoding='utf-8')
    monkeypatch.setattr(partial_encode.burn, '_run_encode', encode)
    return root, child, calls


def test_partial_qsv_encode_has_range_lineage_proportional_target_and_no_whole_authority(partial_fixture, monkeypatch):
    root, child, calls = encoded_fixture(partial_fixture, monkeypatch)
    result = partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120)
    receipt, output = partial_encode.validate_partial_encoding(root, 12, 'part-001')
    assert result['output']['relative_path'] == output.relative_to(root).as_posix()
    assert receipt['mode'] == 'strict-partial' and receipt['full_episode_complete'] is False
    assert receipt['source_range_seconds'] == [0, 7]
    assert receipt['soft_target_bytes'] == 350000000
    command = calls[-1]
    assert command[command.index('-ss') + 1] == '0.000000000'
    assert command[command.index('-t') + 1] == '7.000000000'
    assert command[command.index('-c:v') + 1] == 'h264_qsv'
    assert command[command.index('-global_quality') + 1] == '18'
    assert not output.with_suffix('.burn.json').exists()
    assert not (root / 'final/burned_mp4_delivery.json').exists()
    count = len(calls)
    assert partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120) == result
    assert len(calls) == count + 1


def test_partial_encoder_requires_release_before_probe_or_encode(partial_fixture, monkeypatch):
    root, child, _, _, _, _ = partial_fixture
    partial_finalize.finalize_partial_episode(root, 12, 'part-001')
    monkeypatch.setattr(partial_encode.burn, 'plan_encoding_settings', lambda *a, **k: pytest.fail('probe before release'))
    with pytest.raises(FileNotFoundError):
        partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120)


@pytest.mark.parametrize('change', ['mode', 'range', 'output', 'sample', 'subtitle'])
def test_partial_encoding_validation_rejects_changed_authority_or_bytes(partial_fixture, monkeypatch, change):
    root, child, _ = encoded_fixture(partial_fixture, monkeypatch)
    result = partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120)
    path = child / 'final/partial-encoding.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if change == 'mode':
        receipt['mode'] = 'strict'
        atomic_json(path, receipt)
    elif change == 'range':
        receipt['source_range_seconds'][0] = 30
        atomic_json(path, receipt)
    elif change == 'output':
        (root / result['output']['relative_path']).write_bytes(b'changed')
    elif change == 'sample':
        folder = (root / receipt['qualification']['relative_path']).parent
        (folder / 'sample-1.mp4').write_bytes(b'changed')
    else:
        (child / 'final/Muhtemel Ask 12.Bolum_part-001-id.srt').write_bytes(b'changed')
    with pytest.raises(ValueError):
        partial_encode.validate_partial_encoding(root, 12, 'part-001')


def test_partial_expired_budget_never_starts_source_validation(partial_fixture, monkeypatch):
    root, _, _, _, _, _ = partial_fixture
    monkeypatch.setattr(partial_encode, 'validate_partial_export', lambda *a, **k: pytest.fail('late validation'))
    with pytest.raises(ValueError):
        partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=0)


def test_range_encoder_uses_exact_sample_offset_not_rounded_milliseconds(partial_fixture, monkeypatch):
    root, child, calls = encoded_fixture(partial_fixture, monkeypatch)
    real = partial_encode.validate_partial_export
    def nonzero_scope(*args, **kwargs):
        export, report = real(*args, **kwargs)
        report['scope'] = {**report['scope'], 'start_sample': 480001, 'end_sample': 592001,
                           'start_ms': 30000, 'end_ms': 37001}
        return export, report
    monkeypatch.setattr(partial_encode, 'validate_partial_export', nonzero_scope)
    partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120)
    command = calls[-1]
    assert command[command.index('-ss') + 1] == '30.000062500'
    assert command[command.index('-t') + 1] == '7.000000000'
    assert command[command.index('-seek_timestamp') + 1] == '0'


@pytest.mark.parametrize('seam', ['receipt', 'move'])
def test_verified_partial_output_recovers_publication_interruption_without_reencoding(partial_fixture, monkeypatch, seam):
    root, child, calls = encoded_fixture(partial_fixture, monkeypatch)
    write = partial_encode.atomic_json
    stage = partial_encode.burn._stage_output
    if seam == 'receipt':
        def crash(path, data):
            if path.name == 'partial-encoding.json':
                raise KeyboardInterrupt()
            write(path, data)
        monkeypatch.setattr(partial_encode, 'atomic_json', crash)
    else:
        def crash_move(*args):
            raise KeyboardInterrupt()
        monkeypatch.setattr(partial_encode.burn, '_stage_output', crash_move)
    with pytest.raises(KeyboardInterrupt):
        partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120)
    count = sum(isinstance(item, list) for item in calls)
    monkeypatch.setattr(partial_encode, 'atomic_json', write)
    monkeypatch.setattr(partial_encode.burn, '_stage_output', stage)
    result = partial_encode.burn_partial_indonesian_mp4(root, 12, 'part-001', total_timeout=120)
    assert partial_encode.validate_partial_encoding(root, 12, 'part-001')[1].is_file()
    assert sum(isinstance(item, list) for item in calls) == count
    assert (root / result['receipt']['relative_path']).exists()
