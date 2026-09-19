import json

import pytest

from mas import notify
from mas.engine import burned_mp4
from mas.reliability import file_digest as sha256_file


def test_corrupt_outbox_records_do_not_starve_terminal_delivery(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(notify, 'send_email',
                        lambda *args, **kwargs: sent.append(args[1]) or {'status': 'sent'})
    notify.enqueue_notification(14, 'ready', 'verified', root=tmp_path, kind='terminal')
    outbox = tmp_path / 'work/notification-outbox'
    for index in range(3):
        (outbox / f'000{index}.json').write_text('{')
    notify.drain_outbox(tmp_path, max_messages=3)
    assert sent == ['ready']
    assert len(list(outbox.glob('000*.json'))) == 3


def qualification_fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'immutable source')
    folder = tmp_path / 'qualification'
    def probe(path):
        return {'format': {'duration': '15' if path.name.startswith('sample-') else '60'},
                'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'pix_fmt': 'yuv420p',
                             'width': 320, 'height': 180},
                            {'codec_type': 'audio', 'codec_name': 'aac'}]}
    monkeypatch.setattr(burned_mp4, '_probe', probe)
    monkeypatch.setattr(burned_mp4, '_qsv_hardware', lambda **kw: {'qsv_hardware_probe': 'PASS'})
    return source, folder


def test_interrupted_qualification_reuses_completed_samples_and_preserves_attempt(tmp_path, monkeypatch):
    source, folder = qualification_fixture(tmp_path, monkeypatch)
    calls = []
    def encode(command, cwd, log, output, *args):
        calls.append(output.name)
        log.write_bytes(b'encode log')
        output.write_bytes(b'partial' if len(calls) == 2 else b'verified sample')
        if len(calls) == 2:
            raise TimeoutError('interrupted sample encode')
    monkeypatch.setattr(burned_mp4, '_run_encode', encode)
    with pytest.raises(TimeoutError, match='interrupted sample'):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    first = sha256_file(folder / 'sample-1.mp4')
    receipt = burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    assert burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv') == receipt
    assert calls == ['sample-1.mp4', 'sample-2.mp4', 'sample-2.mp4', 'sample-3.mp4']
    assert sha256_file(folder / 'sample-1.mp4') == first
    assert (folder / 'interrupted/sample-2.mp4.attempt-1').read_bytes() == b'partial'
    assert json.loads(receipt.read_text())['perceptual_acceptance'] == 'NOT_ASSERTED'


def test_deterministic_sample_failure_is_not_retried(tmp_path, monkeypatch):
    source, folder = qualification_fixture(tmp_path, monkeypatch)
    calls = []
    def encode(*args):
        calls.append('encode')
        raise RuntimeError('ffmpeg exit code 1')
    monkeypatch.setattr(burned_mp4, '_run_encode', encode)
    with pytest.raises(RuntimeError, match='exit code'):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    with pytest.raises(ValueError, match='BLOCKED'):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    assert calls == ['encode']


def test_interruption_sample_attempts_remain_bounded_across_restarts(tmp_path, monkeypatch):
    source, folder = qualification_fixture(tmp_path, monkeypatch)
    calls = []
    def encode(*args):
        calls.append('encode')
        raise TimeoutError('interrupted')
    monkeypatch.setattr(burned_mp4, '_run_encode', encode)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    with pytest.raises(ValueError, match='retry budget exhausted'):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    assert len(calls) == 3


def test_completed_samples_survive_crash_before_qualification_receipt(tmp_path, monkeypatch):
    source, folder = qualification_fixture(tmp_path, monkeypatch)
    calls = []
    def encode(command, cwd, log, output, *args):
        calls.append(output.name)
        output.write_bytes(b'sample')
    monkeypatch.setattr(burned_mp4, '_run_encode', encode)
    write = burned_mp4.atomic_write_json
    def crash(path, data):
        if path.name == 'technical-qualification.json':
            raise KeyboardInterrupt()
        write(path, data)
    monkeypatch.setattr(burned_mp4, 'atomic_write_json', crash)
    with pytest.raises(KeyboardInterrupt):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    monkeypatch.setattr(burned_mp4, 'atomic_write_json', write)
    assert burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv').is_file()
    assert calls == ['sample-1.mp4', 'sample-2.mp4', 'sample-3.mp4']


def test_changed_completed_sample_cannot_unlock_next_encode(tmp_path, monkeypatch):
    source, folder = qualification_fixture(tmp_path, monkeypatch)
    calls = []
    def encode(command, cwd, log, output, *args):
        calls.append(output.name)
        if len(calls) == 2:
            raise TimeoutError('interrupted')
        output.write_bytes(b'sample')
    monkeypatch.setattr(burned_mp4, '_run_encode', encode)
    with pytest.raises(TimeoutError):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    (folder / 'sample-1.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    assert len(calls) == 2


@pytest.mark.parametrize('change', ['source', 'hardware', 'subtitle', 'checkpoint'])
def test_qualification_restart_rejects_changed_evidence(tmp_path, monkeypatch, change):
    source, folder = qualification_fixture(tmp_path, monkeypatch)
    def interrupt(*args):
        raise TimeoutError('interrupted')
    monkeypatch.setattr(burned_mp4, '_run_encode', interrupt)
    with pytest.raises(TimeoutError):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
    if change == 'source':
        source.write_bytes(b'changed source')
    elif change == 'hardware':
        monkeypatch.setattr(burned_mp4, '_qsv_hardware', lambda **kw: {'qsv_hardware_probe': 'PASS', 'driver': 'new'})
    elif change == 'subtitle':
        (folder / 'technical-encoder-test.srt').write_bytes(b'changed subtitle')
    else:
        path = folder / 'sample-checkpoint.json'
        wrapped = json.loads(path.read_text())
        wrapped['data']['attempts'] = {}
        path.write_text(json.dumps(wrapped))
    monkeypatch.setattr(burned_mp4, '_run_encode', lambda *a: pytest.fail('changed evidence encoded'))
    with pytest.raises(ValueError, match='changed'):
        burned_mp4.qualify_encoding(source, folder, encoder='h264_qsv')
