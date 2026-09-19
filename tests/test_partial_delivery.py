import importlib
from pathlib import Path

import pytest

from mas import partial_delivery as delivery
from mas import runpod_controller
from mas.engine import part_audio, partial_encode
from mas.engine.episode_archive import file_record
from mas.reliability import atomic_json, digest


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / 'engine'))
    fixture_module = importlib.import_module('test_partial_finalize')
    encode_tests = importlib.import_module('test_partial_encode')
    fixture = fixture_module.partial_fixture.__wrapped__(tmp_path, monkeypatch)
    root, child, _, _, plan, _ = fixture
    atomic_json(root / 'source/official-source.json', {'episode': 15, 'url': 'https://example.invalid/15',
                'channel_url': delivery.CHANNEL_VIDEOS_URL, 'title': 'Muhtemel Ask 15. Bolum'})
    real_release = delivery.validate_part_release
    root, child, calls = encode_tests.encoded_fixture(fixture, monkeypatch)
    monkeypatch.setattr(delivery, 'validate_part_release', real_release)
    monkeypatch.setattr(part_audio, 'load_part_plan', lambda *a, **kw: {**plan, 'audio': {**plan['audio'], 'sample_count': 112000}})
    audit = root / 'work/capacity/test'
    state = {'format': 'mas-capacity-lease-state-1', 'episode': 15, 'status': 'RELEASED',
             'owned_pod_ids': ['owned'], 'shutdown': [{'pod_id': 'owned', 'status': 'ABSENT'}]}
    atomic_json(audit / 'capacity-state.json', {'data': state, 'sha256': digest(state)})
    shutdown = {'state_sha256': digest(state), 'owned_pods': state['shutdown']}
    atomic_json(audit / 'capacity-shutdown.json', {'data': shutdown, 'sha256': digest(shutdown)})
    capacity_check = runpod_controller._capacity_release_evidence
    def capacity(*args):
        capacity_check(*args)
        calls.append('released')
    monkeypatch.setattr(runpod_controller, '_capacity_release_evidence', capacity)
    delivery.write_part_release(root, 15, 'part-001', 'owned', audit)
    uploads = []
    def upload(path, target, **kwargs):
        assert kwargs['require_drive_preflight'] is True
        assert 0 < kwargs['total_timeout'] <= 120
        uploads.append(target)
        record = file_record(path, root)
        return {'remote': target, 'bytes': record['size_bytes'], 'sha256': record['sha256']}
    monkeypatch.setattr(delivery, 'upload_verified', upload)
    return root, child, calls, uploads


def test_bridge_emits_only_partial_receipt_ack_and_explicit_parts_completion(bridge):
    root, child, _, uploads = bridge
    assert delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120) == delivery.NEXT_PART
    published = delivery.validate_published_part(root, 15, 'part-001')
    assert published['status'] == 'PASS_PARTIAL' and len(uploads) == 1
    assert delivery.validate_worker_published_part(root, 15, 'part-001')['delivery'] == published
    completed = delivery.complete_parts(root, 15)
    assert completed['status'] == 'COMPLETE_PARTS' and completed['single_full_episode_file'] is False
    assert not (root / 'output/burned_mp4_delivery.json').exists()


@pytest.mark.parametrize('failure', ['bytes', 'sha256', 'remote', 'exception'])
def test_bad_upload_result_never_leaves_partial_pass_or_ack(bridge, monkeypatch, failure):
    root, child, _, _ = bridge
    def upload(path, target, **kwargs):
        if failure == 'exception':
            raise RuntimeError('cloud byte/SHA verification failed')
        record = file_record(path, root)
        result = {'bytes': record['size_bytes'], 'sha256': record['sha256'], 'remote': target}
        result[failure] = 0 if failure == 'bytes' else 'mismatch'
        return result
    monkeypatch.setattr(delivery, 'upload_verified', upload)
    with pytest.raises((ValueError, RuntimeError)):
        delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120)
    assert not (child / 'output/drive_readback_receipt.json').exists()
    assert not (child / 'work/controller-delivery-ack.json').exists()


def test_cached_published_part_restores_ack_without_encoder_or_upload(bridge, monkeypatch):
    root, child, _, _ = bridge
    delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120)
    (child / 'work/controller-delivery-ack.json').unlink()
    monkeypatch.setattr(partial_encode, 'burn_partial_indonesian_mp4', lambda *a, **kw: pytest.fail('re-encode'))
    monkeypatch.setattr(delivery, 'upload_verified', lambda *a, **kw: pytest.fail('re-upload'))
    assert delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120) == delivery.NEXT_PART
    assert delivery.validate_worker_published_part(root, 15, 'part-001') is not None


def test_cached_publication_cannot_claim_a_different_destination(bridge, monkeypatch):
    root, child, _, _ = bridge
    delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120)
    before = (child / 'output/drive_readback_receipt.json').read_bytes()
    monkeypatch.setattr(partial_encode, 'burn_partial_indonesian_mp4', lambda *a, **kw: pytest.fail('re-encode'))
    with pytest.raises(ValueError, match='destination differs'):
        delivery.complete_local_part(root, 15, 'part-001', 'drive:different', total_timeout=120)
    assert (child / 'output/drive_readback_receipt.json').read_bytes() == before


def test_missing_gpu_release_forbids_encoder(bridge, monkeypatch):
    root, child, _, _ = bridge
    (child / 'work/gpu-released-for-encode.json').unlink()
    monkeypatch.setattr(partial_encode, 'burn_partial_indonesian_mp4', lambda *a, **kw: pytest.fail('encoder started'))
    with pytest.raises(FileNotFoundError):
        delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120)


def test_worker_ack_needs_no_mp4_but_local_completion_requires_it(bridge):
    root, child, _, _ = bridge
    delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120)
    receipt = delivery.validate_published_part(root, 15, 'part-001')
    (root / receipt['mp4']['relative_path']).unlink()
    assert delivery.validate_worker_published_part(root, 15, 'part-001') is not None
    with pytest.raises(ValueError):
        delivery.complete_parts(root, 15)


def test_later_part_failure_preserves_earlier_delivery(bridge, monkeypatch):
    root, child, _, _ = bridge
    delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=120)
    before = (child / 'output/drive_readback_receipt.json').read_bytes()
    with pytest.raises((FileNotFoundError, ValueError)):
        delivery.complete_local_part(root, 15, 'part-002', 'drive:strict', total_timeout=120)
    assert (child / 'output/drive_readback_receipt.json').read_bytes() == before
    assert delivery.validate_published_part(root, 15, 'part-001')['status'] == 'PASS_PARTIAL'


def test_expired_shared_budget_never_writes_delivery_receipt(bridge, monkeypatch):
    root, child, _, _ = bridge
    clock = [0.0]
    monkeypatch.setattr(delivery.time, 'monotonic', lambda: clock[0])
    def upload(path, target, **kwargs):
        assert kwargs['total_timeout'] <= 1
        clock[0] = 2
        record = file_record(path, root)
        return {'remote': target, 'bytes': record['size_bytes'], 'sha256': record['sha256']}
    monkeypatch.setattr(delivery, 'upload_verified', upload)
    with pytest.raises(TimeoutError):
        delivery.complete_local_part(root, 15, 'part-001', 'drive:strict', total_timeout=1)
    assert not (child / 'output/drive_readback_receipt.json').exists()
