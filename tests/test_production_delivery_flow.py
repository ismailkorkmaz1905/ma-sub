import hashlib
import json
from pathlib import Path

import pytest

from mas import delivery, pipeline, runpod_controller
from mas.reliability import atomic_json


def _record(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'relative_path': relative, 'size_bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def _strict_delivery(root, episode=15, with_samples=False):
    name = f'Muhtemel Ask {episode}.Bolum'
    inputs = {
        'source_video': _record(root, 'source/source.mp4', b'source'),
        'tr_correction_output': _record(root, 'handoff/tr.zip', b'tr return'),
        'id_translation_output': _record(root, 'handoff/id.zip', b'id return'),
    }
    outputs = {
        'tr_srt': _record(root, f'output/{name}.tr.srt', b'tr srt'),
        'id_srt': _record(root, f'output/{name}.id.srt', b'id srt'),
        'mkv': _record(root, f'output/{name}.mkv', b'mkv'),
    }
    report = {'episode': episode, 'status': 'PASS', 'input_files': inputs, 'outputs': outputs}
    report_path = root / 'output' / f'{name}_FINALIZATION_REPORT_V2.json'
    atomic_json(report_path, report)
    mp4 = _record(root, f'output/{name}.id.bound.mp4', b'mp4')
    receipt_path = root / Path(mp4['relative_path']).with_suffix('.burn.json')
    receipt = {'status': 'VERIFIED_ENCODING', 'inputs': {
        'source_sha256': inputs['source_video']['sha256'],
        'id_srt_sha256': outputs['id_srt']['sha256']},
        'output_sha256': mp4['sha256'], 'output_bytes': mp4['size_bytes']}
    if with_samples:
        sample_root = root / 'work' / 'encoding-samples' / 'bound'
        samples = []
        for index in range(1, 4):
            video = _record(root, f'work/encoding-samples/bound/sample-{index}.mp4',
                            f'video-{index}'.encode())
            subtitle = _record(root, f'work/encoding-samples/bound/sample-{index}.srt',
                               f'subtitle-{index}'.encode())
            samples.append({'output_file': Path(video['relative_path']).name,
                            'output_bytes': video['size_bytes'], 'output_sha256': video['sha256'],
                            'subtitle_file': Path(subtitle['relative_path']).name,
                            'subtitle_bytes': subtitle['size_bytes'], 'subtitle_sha256': subtitle['sha256']})
        sample_manifest = sample_root / 'encoding-samples.json'
        atomic_json(sample_manifest, {'samples': samples})
        approval_path = root / 'work' / 'mp4-sample-approval.json'
        atomic_json(approval_path, {
            'approved': True,
            'sample_manifest_path': 'work/encoding-samples/bound/encoding-samples.json',
            'sample_manifest_sha256': delivery.sha256_file(sample_manifest)})
        receipt['sample_approval'] = {'bytes': approval_path.stat().st_size,
                                      'sha256': delivery.sha256_file(approval_path)}
    atomic_json(receipt_path, receipt)
    manifest = {'mode': 'strict', 'strict_finalization_sha256': delivery.sha256_file(report_path),
                'encoding_receipt_sha256': delivery.sha256_file(receipt_path),
                'outputs': {'mp4': mp4}}
    atomic_json(root / 'output' / 'burned_mp4_delivery.json', manifest)
    return report, manifest


def _controller_preflight(monkeypatch, root):
    monkeypatch.setattr(runpod_controller, 'ROOT', root)
    monkeypatch.setattr(runpod_controller, 'episode_dir', lambda _: root / 'episode')
    monkeypatch.setattr(runpod_controller, '_required_environment', lambda: {
        'MAS_DRIVE_STRICT_REMOTE': 'drive:delivery', 'MAS_YTDLP_COOKIES': 'cookies'})
    monkeypatch.setattr(runpod_controller, '_local_preflight', lambda _: ('commit', Path('rclone')))
    monkeypatch.setattr(runpod_controller, '_validate_local_tr_return', lambda *args: None)
    monkeypatch.setattr(runpod_controller, 'preflight_local_id_return', lambda *args: None)


def test_transfer_only_verified_local_delivery_never_acquires_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runpod_controller, 'ROOT', tmp_path)
    monkeypatch.setattr(runpod_controller, 'episode_dir', lambda _: tmp_path / 'episode')
    monkeypatch.setenv('MAS_DRIVE_STRICT_REMOTE', 'drive:delivery')
    for name in ('RUNPOD_POD_ID', 'RUNPOD_API_KEY', 'MAS_RUNPOD_SSH_KEY',
                 'MAS_YTDLP_COOKIES'):
        monkeypatch.delenv(name, raising=False)
    episode_root = tmp_path / 'episode'
    _strict_delivery(episode_root)
    delivery.validate_delivery(episode_root, 15)
    audit = episode_root / 'work' / 'capacity' / 'completed'
    audit.mkdir(parents=True)
    shutdown_result = [{'pod_id': 'owned', 'status': 'ABSENT', 'error_type': None}]
    state = {'format': 'mas-capacity-lease-state-1', 'episode': 15,
             'owned_pod_ids': ['owned'], 'shutdown': shutdown_result, 'status': 'RELEASED'}
    atomic_json(audit / 'capacity-state.json',
                {'data': state, 'sha256': runpod_controller.digest(state)})
    shutdown = {'state_sha256': runpod_controller.digest(state), 'owned_pods': shutdown_result}
    atomic_json(audit / 'capacity-shutdown.json',
                {'data': shutdown, 'sha256': runpod_controller.digest(shutdown)})
    runpod_controller._write_delivery_release(episode_root, 15, 'owned', audit)
    def publish(root, episode, remote, **kwargs):
        delivery.validate_delivery(root, episode)
        assert remote == 'drive:delivery'
        assert 0 < kwargs['total_timeout'] <= 21600
        return 0
    monkeypatch.setattr(runpod_controller, 'publish_local_delivery', publish)
    monkeypatch.setattr(runpod_controller, '_required_environment',
                        lambda: pytest.fail('transfer-only retry required RunPod environment'))
    monkeypatch.setattr(runpod_controller, '_local_preflight',
                        lambda *_: pytest.fail('transfer-only retry ran RunPod/git/SSH preflight'))
    monkeypatch.setattr(runpod_controller, '_validate_local_tr_return',
                        lambda *_: pytest.fail('transfer-only retry revalidated TR handoff'))
    monkeypatch.setattr(runpod_controller, 'preflight_local_id_return',
                        lambda *_: pytest.fail('transfer-only retry revalidated ID handoff'))
    monkeypatch.setattr(runpod_controller, 'CapacityProvider',
                        lambda *args: pytest.fail('transfer-only retry acquired GPU capacity'))
    assert runpod_controller.run_remote_episode(15) == 0

    with (audit / 'capacity-state.json').open('a', encoding='utf-8') as handle:
        handle.write('\n')
    with pytest.raises(runpod_controller.RunPodControllerError,
                       match='capacity evidence changed'):
        runpod_controller.run_remote_episode(15)


def test_wait_23_transfers_only_samples_and_does_not_acquire_gpu(tmp_path, monkeypatch):
    _controller_preflight(monkeypatch, tmp_path)
    episode_root = tmp_path / 'episode'
    (episode_root / 'work').mkdir(parents=True)
    atomic_json(episode_root / 'work' / 'remote-job-status.json',
                {'status': 'EXITED', 'exit_code': delivery.WAIT_MP4_SAMPLE})
    sample_export = episode_root / 'work' / 'sample-export.json'
    atomic_json(sample_export, {'episode': 15, 'mode': 'review', 'files': []})
    shutdown = episode_root / 'work' / 'capacity-shutdown.json'
    body = {'owned_pods': [{'pod_id': 'owned', 'status': 'ABSENT'}]}
    atomic_json(shutdown, {'data': body, 'sha256': runpod_controller.digest(body)})
    runpod_controller._episode_budget(episode_root, 15)
    runpod_controller._pause_episode_budget(episode_root, 15, delivery.WAIT_MP4_SAMPLE,
                                           sample_export, shutdown)
    monkeypatch.setattr(runpod_controller, 'CapacityProvider',
                        lambda *args: pytest.fail('sample wait acquired GPU capacity'))
    assert runpod_controller.run_remote_episode(15) == delivery.WAIT_MP4_SAMPLE

    stages = []
    monkeypatch.setattr(pipeline, 'set_stage', lambda *args, **kwargs: stages.append((args, kwargs)))
    monkeypatch.setattr(pipeline, 'mark_work_progress', lambda *args, **kwargs: None)
    result = pipeline._stage(tmp_path / 'state.json', {'episode': 15}, 'finalize',
                             lambda: {'sample_review_required': True, 'samples': 'manifest'})
    assert result['sample_review_required'] is True
    assert stages[-1][0][3] == 'blocked'
    assert all(call[0][3] != 'pass' for call in stages)


def test_terminal_wait_status_without_verified_wait_ledger_does_not_short_circuit(tmp_path, monkeypatch):
    _controller_preflight(monkeypatch, tmp_path)
    episode_root = tmp_path / 'episode'
    (episode_root / 'work').mkdir(parents=True)
    atomic_json(episode_root / 'work' / 'remote-job-status.json',
                {'status': 'EXITED', 'exit_code': delivery.WAIT_MP4_SAMPLE})
    class ReachedSourcePreflight(Exception):
        pass
    monkeypatch.setattr(runpod_controller, '_prepare_official_source',
                        lambda *args, **kwargs: (_ for _ in ()).throw(ReachedSourcePreflight()))
    with pytest.raises(ReachedSourcePreflight):
        runpod_controller.run_remote_episode(15)


def test_external_mp4_export_is_exact_and_downloads_to_canonical_local(tmp_path, monkeypatch):
    local_root = tmp_path / 'episode'
    remote_root = '/workspace/ma-sub/EPISODES/Muhtemel Ask 15.Bolum'
    data = b'external mp4'
    record = {'relative_path': 'output/Muhtemel Ask 15.Bolum.id.bound.mp4',
              'storage_path': '/tmp/mas-ep15-output/Muhtemel Ask 15.Bolum.id.bound.mp4',
              'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    calls = []
    def transfer(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(data)
    monkeypatch.setattr(runpod_controller, '_network_retry', transfer)
    budget = type('Budget', (), {'check': lambda self: 60})()
    runpod_controller._download_record(record, local_root, remote_root,
                                       ['scp'], 'host', budget)
    canonical = local_root / record['relative_path']
    assert canonical.read_bytes() == data
    assert calls[0][-2] == f"root@host:{record['storage_path']}"
    delivery_record = dict(record, storage_path=str(Path('/tmp/mas-ep15-output') / canonical.name))
    assert delivery.remote_storage_path(local_root, delivery_record, 15) == Path(delivery_record['storage_path'])
    with pytest.raises(ValueError, match='unexpected external'):
        delivery.remote_storage_path(local_root, dict(delivery_record, storage_path='elsewhere'), 15)
    bad = dict(record, storage_path='/tmp/mas-ep16-output/' + canonical.name)
    with pytest.raises(runpod_controller.RunPodControllerError, match='escapes episode'):
        runpod_controller._download_record(bad, tmp_path / 'other', remote_root,
                                           ['scp'], 'host', budget)


def test_changed_external_mp4_preserves_prior_canonical_by_sha(tmp_path, monkeypatch):
    local_root = tmp_path / 'episode'
    remote_root = '/workspace/ma-sub/EPISODES/Muhtemel Ask 15.Bolum'
    filename = 'Muhtemel Ask 15.Bolum.id.bound.mp4'
    canonical = local_root / 'output' / filename
    canonical.parent.mkdir(parents=True)
    prior = b'prior external mp4'
    current = b'new external mp4'
    canonical.write_bytes(prior)
    record = {'relative_path': f'output/{filename}',
              'storage_path': f'/tmp/mas-ep15-output/{filename}',
              'size_bytes': len(current), 'sha256': hashlib.sha256(current).hexdigest()}
    def transfer(command, **kwargs):
        Path(command[-1]).write_bytes(current)
    monkeypatch.setattr(runpod_controller, '_network_retry', transfer)
    budget = type('Budget', (), {'check': lambda self: 60})()

    runpod_controller._download_record(record, local_root, remote_root,
                                       ['scp'], 'host', budget)

    retained = (local_root / 'work' / 'remote-checkpoints' /
                hashlib.sha256(prior).hexdigest() / filename)
    assert retained.read_bytes() == prior
    assert canonical.read_bytes() == current


def test_strict_source_returns_subtitles_and_mkv_remain_hash_bound(tmp_path):
    report, _ = _strict_delivery(tmp_path)
    delivery.validate_delivery(tmp_path, 15)
    for record in list(report['input_files'].values()) + list(report['outputs'].values()):
        path = tmp_path / record['relative_path']
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with pytest.raises(ValueError, match='byte/SHA'):
            delivery.validate_delivery(tmp_path, 15)
        path.write_bytes(original)


def test_remote_inventory_preservation_mismatch_aborts_publication(tmp_path, monkeypatch):
    _strict_delivery(tmp_path)
    (tmp_path / 'source' / 'source.url').write_text('https://example.test/15', encoding='utf-8')
    atomic_json(tmp_path / 'source' / 'official-source.json', {
        'episode': 15, 'url': 'https://example.test/15', 'channel_url': 'channel',
        'title': 'Muhtemel Ask 15. Bolum'})
    atomic_json(tmp_path / 'work' / 'state.json', {'episode': 15, 'stages': {}})
    monkeypatch.setattr(delivery, 'CHANNEL_VIDEOS_URL', 'channel')
    monkeypatch.setattr(delivery, 'is_exact_episode_title', lambda title, episode: True)
    monkeypatch.setattr(delivery, 'upload_verified',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            RuntimeError('retained Drive byte/SHA-256 readback mismatch')))
    with pytest.raises(RuntimeError, match='retained Drive'):
        delivery.publish_local_delivery(tmp_path, 15, 'drive:delivery')
    assert not (tmp_path / 'output' / 'drive_readback_receipt.json').exists()


def test_local_publication_forwards_scratch_lifetime_timeout(tmp_path, monkeypatch):
    _strict_delivery(tmp_path)
    (tmp_path / 'source' / 'source.url').write_text('https://example.test/15', encoding='utf-8')
    atomic_json(tmp_path / 'source' / 'official-source.json', {
        'episode': 15, 'url': 'https://example.test/15', 'channel_url': 'channel',
        'title': 'Muhtemel Ask 15. Bolum'})
    atomic_json(tmp_path / 'work' / 'state.json', {'episode': 15, 'stages': {}})
    monkeypatch.setattr(delivery, 'CHANNEL_VIDEOS_URL', 'channel')
    monkeypatch.setattr(delivery, 'is_exact_episode_title', lambda title, episode: True)
    observed = {}
    def upload(source, remote, **kwargs):
        observed.update(source=source, remote=remote, kwargs=kwargs)
        return {'bytes': source.stat().st_size, 'sha256': delivery.sha256_file(source),
                'remote': remote}
    monkeypatch.setattr(delivery, 'upload_verified', upload)
    assert delivery.publish_local_delivery(tmp_path, 15, 'drive:delivery', total_timeout=37) == 0
    assert 0 < observed['kwargs']['total_timeout'] <= 37
    assert observed['kwargs']['preservation_receipt'] == tmp_path / 'output' / 'drive-preservation.json'
    assert observed['kwargs']['require_drive_preflight'] is True
    receipt = json.loads((tmp_path / 'output' / 'drive_readback_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'PASS'
    assert receipt['delivery_sha256'] == hashlib.sha256(
        (tmp_path / 'output' / 'burned_mp4_delivery.json').read_bytes()).hexdigest()


def test_publication_local_validation_consumes_same_wall_budget(tmp_path, monkeypatch):
    _strict_delivery(tmp_path)
    (tmp_path / 'source' / 'source.url').write_text('https://example.test/15', encoding='utf-8')
    atomic_json(tmp_path / 'source' / 'official-source.json', {
        'episode': 15, 'url': 'https://example.test/15', 'channel_url': 'channel',
        'title': 'Muhtemel Ask 15. Bolum'})
    monkeypatch.setattr(delivery, 'CHANNEL_VIDEOS_URL', 'channel')
    monkeypatch.setattr(delivery, 'is_exact_episode_title', lambda *args: True)
    clock = [100.0]
    monkeypatch.setattr(delivery.time, 'monotonic', lambda: clock[0])
    validate = delivery.validate_delivery
    def slow_validate(*args):
        clock[0] += 11
        return validate(*args)
    monkeypatch.setattr(delivery, 'validate_delivery', slow_validate)
    monkeypatch.setattr(delivery, 'upload_verified', lambda *args, **kwargs: pytest.fail('upload started'))
    with pytest.raises(TimeoutError, match='wall-time budget'):
        delivery.publish_local_delivery(tmp_path, 15, 'drive:delivery', total_timeout=10)
    assert not (tmp_path / 'output' / 'drive_readback_receipt.json').exists()


def test_mutated_approved_sample_blocks_local_delivery(tmp_path):
    _strict_delivery(tmp_path, with_samples=True)
    delivery.validate_delivery(tmp_path, 15)
    (tmp_path / 'source' / 'source.url').write_text('https://example.test/15', encoding='utf-8')
    atomic_json(tmp_path / 'source' / 'official-source.json', {'episode': 15})
    export = delivery.write_delivery_export(tmp_path, 15)
    exported = {record['relative_path'] for record in export['files']}
    assert 'work/mp4-sample-approval.json' in exported
    assert 'work/encoding-samples/bound/encoding-samples.json' in exported
    assert {f'work/encoding-samples/bound/sample-{index}.{suffix}'
            for index in range(1, 4) for suffix in ('mp4', 'srt')} <= exported
    sample = tmp_path / 'work' / 'encoding-samples' / 'bound' / 'sample-2.mp4'
    sample.write_bytes(sample.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='byte/SHA'):
        delivery.validate_delivery(tmp_path, 15)
