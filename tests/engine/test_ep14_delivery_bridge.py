import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

from test_part_audio import synthetic_parents
from test_delivery_audit_regressions import prepared, test_key, _ready
from mas import delivery_first as df, partial_delivery as pd, full_delivery as full
from mas import pipeline, remote, runpod_controller as rc
from mas.engine import burned_mp4 as burn, part_audio as pa, id_translation as it
from mas.engine import partial_finalize as pf
from mas.hashing import sha256_file
from mas.reliability import atomic_json, digest


def release(root, part_id):
    audit = root / ('work/capacity/' + part_id)
    state = {'format': 'mas-capacity-lease-state-1', 'episode': 14, 'status': 'RELEASED',
             'owned_pod_ids': ['owned'], 'shutdown': [{'pod_id': 'owned', 'status': 'ABSENT'}]}
    atomic_json(audit / 'capacity-state.json', {'data': state, 'sha256': digest(state)})
    data = {'state_sha256': digest(state), 'owned_pods': state['shutdown']}
    atomic_json(audit / 'capacity-shutdown.json', {'data': data, 'sha256': digest(data)})
    pd.write_part_release(root, 14, part_id, 'owned', audit, total_timeout=120)


@pytest.fixture
def bridge(prepared, monkeypatch):
    root, options, _, _ = prepared
    monkeypatch.setenv('MAS_DRIVE_STRICT_REMOTE', 'drive:delivery')
    uploads, encodes, joins = [], [], []
    durations = {}
    # Replace only hardware/encoder and remote boundaries. All source lineage,
    # translation, release, export, qualification and HMAC validators are real.
    monkeypatch.setattr(burn, '_qsv_hardware', lambda **kw: {'qsv_hardware_probe': 'PASS'})
    def probe(path):
        duration = durations.get(str(path), 80007 / 16000)
        return {'format': {'duration': str(duration)}, 'streams': [
            {'codec_type': 'video', 'codec_name': 'h264', 'pix_fmt': 'yuv420p', 'width': 320, 'height': 180},
            {'codec_type': 'audio', 'codec_name': 'aac'}]}
    monkeypatch.setattr(burn, '_probe', probe)
    def encode(command, cwd, log, output, total, idle):
        assert 0 < idle <= total <= 120
        encodes.append(command)
        durations[str(output)] = float(command[command.index('-t') + 1])
        output.write_bytes(('synthetic encoded ' + str(output)).encode())
        log.write_text('fixture encode')
    monkeypatch.setattr(burn, '_run_encode', encode)
    def upload(path, target, **kw):
        uploads.append(target)
        return {'remote': target, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    monkeypatch.setattr(pd, 'upload_verified', upload)
    monkeypatch.setattr(remote, 'upload_verified', upload)
    def assemble(parts, output, *, duration_seconds, remaining):
        assert remaining() > 0
        joins.append(parts)
        output.write_bytes(b''.join(path.read_bytes() for path, _ in parts))
        return {'format': {'duration': str(duration_seconds)}}
    monkeypatch.setattr(full, 'assemble_parts', assemble)
    _ready(prepared)
    release(root, 'part-001')
    assert pd.complete_local_part(root, 14, 'part-001', 'drive:delivery', total_timeout=120) == rc.NEXT_PART
    assert df.run_worker(**options) == rc.WAIT_PART_RETURN
    child = root / 'parts/part-002'
    schema = df.read_signed(child / 'prepare/delivery-schema.json', 'schema')['schema']
    pack = child / 'translation_input/Muhtemel Ask 14.Bolum_ID_TRANSLATION_PACK.zip'
    out = child / 'translation_output/Muhtemel Ask 14.Bolum_ID_TRANSLATED.zip'
    records = [{**r, 'id_final': 'Halo.'} for r in it.build_id_translation_records(schema)]
    it.create_id_translation_output_zip(schema, records, out, input_manifest=it.validate_id_translation_pack(pack))
    assert df.run_worker(**options) == rc.READY_FOR_PARTIAL_ENCODE
    release(root, 'part-002')
    assert pd.complete_local_part(root, 14, 'part-002', 'drive:delivery', total_timeout=120) == rc.NEXT_PART
    return root, options, uploads, encodes, joins


def test_first_remote_then_local_tail_then_full_and_upload_only_resume(bridge, monkeypatch, capsys):
    root, options, uploads, encodes, joins = bridge
    tail = root / 'parts/part-002'
    assert len(uploads) == 1 and '.part-001.' in uploads[0]
    assert not (tail / 'final/drive_readback_receipt.json').exists()
    assert pd.validate_local_tail(root, 14, 'part-002')['status'] == 'LOCAL_ENCODED_NOT_PUBLISHED'
    assert df.run_worker(**options) == rc.NEXT_PART
    count = len(encodes)
    assert pd.complete_local_part(root, 14, 'part-002', 'drive:delivery', total_timeout=120) == rc.NEXT_PART
    assert len(encodes) == count and len(uploads) == 1
    assert any(p.name == 'controller-tail-ack.json' for p in rc._part_resume_files(root, 14))
    good_upload = remote.upload_verified
    monkeypatch.setattr(remote, 'upload_verified', lambda *a, **k: (_ for _ in ()).throw(TimeoutError('transfer')))
    with pytest.raises(TimeoutError, match='transfer'):
        pd.complete_parts(root, 14, total_timeout=120)
    assert len(joins) == 1
    assert (root / 'final/delivery-first/full-assembled.json').is_file()
    monkeypatch.setattr(remote, 'upload_verified', good_upload)
    data = pd.complete_parts(root, 14, total_timeout=120)
    assert data['status'] == 'DELIVERED_WITH_WARNINGS' and data['single_full_episode_file'] is True
    assert len(uploads) == 2 and '.full.id.mp4' in uploads[1] and len(joins) == 1
    assert pd.complete_parts(root, 14, total_timeout=120) == data
    assert len(uploads) == 2 and len(encodes) == count
    # Status is small signed metadata, not expensive byte or live service validation.
    monkeypatch.setattr(pipeline, 'episode_dir', lambda _: root)
    capsys.readouterr()
    assert pipeline.status(14) == 0
    status = json.loads(capsys.readouterr().out)['progressive']
    assert status['full_delivery']['stored_status'] == 'DELIVERED_WITH_WARNINGS'
    assert status['full_delivery']['live_verification'] is False
    assert status['parts'][1]['local_tail']['remote_publication'] is False
    # A retained late run delivers locally without authorizing GPU or changing its clock.
    budget = {'episode': 14, 'started_at': '2020-01-01T00:00:00+00:00', 'limit_seconds': 21600}
    atomic_json(root / 'work/controller_budget.json', {'data': budget, 'sha256': digest(budget)})
    monkeypatch.setattr(rc, '_drain_notifications', lambda *a: None)
    assert rc._resume_partial_delivery(root, 14) == 0
    assert len(uploads) == 2
    assert json.loads((root / 'work/controller_budget.json').read_text())['data'] == budget


def test_tail_ack_tampering_missing_first_publication_and_changed_tail_bytes_fail(bridge):
    root, options, uploads, encodes, joins = bridge
    first = root / 'parts/part-001'
    tail = root / 'parts/part-002'
    ack_path = tail / 'work/controller-tail-ack.json'
    original = ack_path.read_bytes()
    envelope = json.loads(original)
    envelope['data']['local_tail']['status'] = 'PASS_PARTIAL'
    atomic_json(ack_path, envelope)
    with pytest.raises(ValueError, match='authentication'):
        df.run_worker(**options)
    ack_path.write_bytes(original)
    # Even a correctly signed local-tail envelope cannot skip first-part readback.
    first_ack = first / 'work/controller-delivery-ack.json'
    saved_ack = first_ack.read_bytes()
    first_ack.unlink()
    df.write_signed(first / 'work/controller-tail-ack.json',
                    {'format': 'mas-controller-tail-ack-1'}, 'controller-tail-ack')
    with pytest.raises(ValueError, match='first-part'):
        pd.validate_worker_completed_part(root, 14, 'part-001')
    (first / 'work/controller-tail-ack.json').unlink()
    first_ack.write_bytes(saved_ack)
    receipt = first / 'final/drive_readback_receipt.json'
    saved = receipt.read_bytes(); receipt.unlink()
    assert pd.complete_parts(root, 14, total_timeout=120) is None
    assert not joins and len(uploads) == 1
    receipt.write_bytes(saved)
    local = pd.validate_local_tail(root, 14, 'part-002')
    (root / local['mp4']['relative_path']).write_bytes(b'changed encoded content')
    with pytest.raises(ValueError, match='identity mismatch'):
        pd.complete_parts(root, 14, total_timeout=120)
    assert not joins and len(uploads) == 1


def test_hash_dedup_is_one_validation_only_and_detects_changed_bytes(prepared, monkeypatch):
    root, child, export = _ready(prepared)
    real = pa._sha
    counts = Counter()
    def counting(path, deadline=None):
        counts[str(Path(path))] += 1
        return real(path, deadline)
    monkeypatch.setattr(pa, '_sha', counting)
    df.validate_export(root, 14, 'part-001', export, total_timeout=120)
    plan = pa.load_part_plan(root, 14, verify_files=False)
    reads_per_validation = 2 if sys.platform == 'win32' else 1
    for name in ('source', 'audio'):
        assert counts[str(root / plan[name]['relative_path'])] == reads_per_validation
    df.validate_export(root, 14, 'part-001', export, total_timeout=120)
    assert counts[str(root / plan['source']['relative_path'])] == 2 * reads_per_validation
    record = plan['source']; path = root / record['relative_path']
    cache = {}
    pa._verify_file(root, record, verified=cache)
    stamp = path.stat()
    path.write_bytes(b'x' * stamp.st_size)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='identity mismatch'):
        pa._verify_file(root, record, verified=cache)


def test_empty_subtitle_and_unresolved_hint_warnings_remain_visible(prepared, monkeypatch, capsys):
    root, options, decoded, _ = prepared
    decoded['segments'] = []
    assert df.run_worker(**options) == rc.READY_FOR_PARTIAL_ENCODE
    _, report = pf.validate_partial_export(root, 14, 'part-001')
    summary = df.quality_summary(report)
    assert summary['no_subtitles_warning'] is True and summary['unresolved_hint_ms'] > 0
    assert report['quality_status'] == 'NOT_STRICT'
    monkeypatch.setattr(pipeline, 'episode_dir', lambda _: root)
    capsys.readouterr(); pipeline.status(14)
    progress = json.loads(capsys.readouterr().out)['progressive']
    assert progress['parts'][0]['quality']['no_subtitles_warning'] is True
    report_path = root / 'parts/part-001/final/DELIVERY-QUALITY-REPORT.json'
    report_path.write_text('{}')
    pipeline.status(14)
    progress = json.loads(capsys.readouterr().out)['progressive']
    assert progress['parts'][0]['quality']['metadata_status'] == 'UNVERIFIED_LOCAL_METADATA'


def test_windows_never_uses_ctime_as_content_change_proof(tmp_path, monkeypatch):
    path = tmp_path / 'source.bin'
    path.write_bytes(b'original')
    record = {'relative_path': 'source.bin', 'sha256': sha256_file(path), 'size_bytes': 8}
    stamp = path.stat()
    key = (str(path), record['sha256'], 8)
    cache = {key: (stamp.st_dev, stamp.st_ino, 8, stamp.st_mtime_ns, stamp.st_ctime_ns)}
    monkeypatch.setattr(pa.sys, 'platform', 'win32')
    reads = []
    real = pa._sha
    def counting(path, deadline=None):
        reads.append(path)
        return real(path, deadline)
    monkeypatch.setattr(pa, '_sha', counting)
    pa._verify_file(tmp_path, record, verified=cache)
    pa._verify_file(tmp_path, record, verified=cache)
    assert len(reads) == 2
