import re
import time
from pathlib import Path

from .delivery import NEXT_PART, safe_relative
from .engine.episode_archive import file_record
from .engine.part_audio import _deadline, _remaining, _verify_file
from .hashing import sha256_file
from .notify import enqueue_notification
from .reliability import atomic_json, digest, read_json
from .remote import upload_verified
from .source_discovery import CHANNEL_VIDEOS_URL, is_exact_episode_title


def _read_bound(path):
    saved = read_json(path)
    data = saved.get('data')
    if not isinstance(data, dict) or saved.get('sha256') != digest(data):
        raise ValueError('partial delivery evidence checksum mismatch')
    return data


def part_directory(root, part_id):
    if not isinstance(part_id, str) or not re.fullmatch(r'part-[0-9]{3}', part_id):
        raise ValueError('invalid part identifier')
    return safe_relative(Path(root), 'parts/' + part_id)


def validate_part_release(root, episode, part_id, *, total_timeout=300):
    deadline = _deadline(total_timeout)
    root = Path(root)
    folder = part_directory(root, part_id)
    from .engine.partial_finalize import validate_partial_export
    validate_partial_export(root, episode, part_id, total_timeout=_remaining(deadline))
    release = _read_bound(folder / 'work/gpu-released-for-encode.json')
    if (release.get('format') != 'mas-part-gpu-release-1' or release.get('episode') != episode
            or release.get('part_id') != part_id or release.get('status') != 'ABSENT'
            or release.get('export_sha256') != sha256_file(folder / 'work/partial-export.json')):
        raise ValueError('partial encode requires its exact external GPU release')
    from .runpod_controller import _capacity_release_evidence
    _capacity_release_evidence(root, episode, release['pod_id'], release['capacity_state'], release['capacity_shutdown'])
    _remaining(deadline)
    return release


def write_part_release(root, episode, part_id, pod_id, audit, *, total_timeout=300):
    deadline = _deadline(total_timeout)
    root, audit = Path(root), Path(audit)
    folder = part_directory(root, part_id)
    from .engine.partial_finalize import validate_partial_export
    validate_partial_export(root, episode, part_id, total_timeout=_remaining(deadline))
    records = {name: {'relative_path': (audit / filename).relative_to(root).as_posix(),
                      'sha256': sha256_file(audit / filename)} for name, filename in
               (('capacity_state', 'capacity-state.json'), ('capacity_shutdown', 'capacity-shutdown.json'))}
    from .runpod_controller import _capacity_release_evidence
    _capacity_release_evidence(root, episode, pod_id, records['capacity_state'], records['capacity_shutdown'])
    data = {'format': 'mas-part-gpu-release-1', 'episode': episode, 'part_id': part_id,
            'status': 'ABSENT', 'pod_id': pod_id,
            'export_sha256': sha256_file(folder / 'work/partial-export.json'), **records}
    atomic_json(folder / 'work/gpu-released-for-encode.json', {'data': data, 'sha256': digest(data)})


def validate_published_part(root, episode, part_id, *, total_timeout=300):
    deadline = _deadline(total_timeout)
    root = Path(root)
    folder = part_directory(root, part_id)
    path = folder / 'final/drive_readback_receipt.json'
    if not path.is_file():
        return None
    data = _read_bound(path)
    validate_part_release(root, episode, part_id, total_timeout=_remaining(deadline))
    from .engine.partial_encode import validate_partial_encoding
    encoding, mp4 = validate_partial_encoding(root, episode, part_id, total_timeout=_remaining(deadline))
    if (data.get('format') != 'mas-part-delivery-1' or data.get('status') != 'PASS_PARTIAL'
            or data.get('episode') != episode or data.get('part_id') != part_id
            or data.get('plan_sha256') != sha256_file(root / 'work/part-plan.json')
            or data.get('export_sha256') != sha256_file(folder / 'work/partial-export.json')
            or _verify_file(root, data['mp4'], deadline) != mp4
            or _verify_file(root, data['encoding_receipt'], deadline) != folder / 'final/partial-encoding.json'
            or len(data.get('files', [])) != 1):
        raise ValueError('published part authority changed')
    remote = data['files'][0]
    if (remote.get('bytes') != data['mp4']['size_bytes'] or remote.get('sha256') != data['mp4']['sha256']
            or not remote.get('remote') or f'.{part_id}.' not in remote['remote']):
        raise ValueError('partial Drive byte/SHA readback binding changed')
    return data


def write_worker_delivery_ack(root, episode, part_id, *, total_timeout=300):
    root = Path(root)
    delivery = validate_published_part(root, episode, part_id, total_timeout=total_timeout)
    if delivery is None:
        raise ValueError('cannot acknowledge an unpublished part')
    data = {'format': 'mas-controller-part-ack-1', 'episode': episode, 'part_id': part_id,
            'delivery': delivery, 'delivery_data_sha256': digest(delivery)}
    path = part_directory(root, part_id) / 'work/controller-delivery-ack.json'
    if path.exists() and _read_bound(path) != data:
        raise ValueError('Existing controller part acknowledgement changed; preserve it')
    atomic_json(path, {'data': data, 'sha256': digest(data)})
    return path


def validate_worker_published_part(root, episode, part_id, *, total_timeout=300):
    root = Path(root)
    folder = part_directory(root, part_id)
    path = folder / 'work/controller-delivery-ack.json'
    if not path.is_file():
        return None
    ack = _read_bound(path)
    delivery = ack.get('delivery')
    from .engine.partial_finalize import validate_partial_export
    validate_partial_export(root, episode, part_id, total_timeout=total_timeout)
    if (ack.get('format') != 'mas-controller-part-ack-1' or ack.get('episode') != episode
            or ack.get('part_id') != part_id or not isinstance(delivery, dict)
            or ack.get('delivery_data_sha256') != digest(delivery)
            or delivery.get('format') != 'mas-part-delivery-1' or delivery.get('status') != 'PASS_PARTIAL'
            or delivery.get('episode') != episode or delivery.get('part_id') != part_id
            or delivery.get('plan_sha256') != sha256_file(root / 'work/part-plan.json')
            or delivery.get('export_sha256') != sha256_file(folder / 'work/partial-export.json')
            or len(delivery.get('files', [])) != 1):
        raise ValueError('controller part acknowledgement binding changed')
    record, remote = delivery['mp4'], delivery['files'][0]
    relative = record['relative_path']
    if (not relative.startswith(f'parts/{part_id}/final/')
            or safe_relative(root, relative).suffix != '.mp4'
            or remote.get('bytes') != record['size_bytes'] or remote.get('sha256') != record['sha256']
            or f'.{part_id}.' not in remote.get('remote', '')):
        raise ValueError('controller part acknowledgement lacks matching readback evidence')
    return ack


def complete_local_part(root, episode, part_id, remote_root, *, total_timeout):
    root = Path(root)
    deadline = _deadline(total_timeout)

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('partial delivery episode budget exhausted')
        return value

    metadata = read_json(safe_relative(root, 'source/official-source.json'))
    source_url = safe_relative(root, 'source/source.url').read_text(encoding='utf-8').strip()
    title = metadata.get('title', '')
    if (metadata.get('episode') != episode or metadata.get('url') != source_url
            or metadata.get('channel_url') != CHANNEL_VIDEOS_URL or not is_exact_episode_title(title, episode)
            or re.search(r'[\\/:*?"<>|\x00-\x1f]', title)):
        raise ValueError('partial publication official source identity mismatch')
    if not remote_root or '\r' in remote_root or '\n' in remote_root:
        raise ValueError('invalid partial Drive destination')
    target = remote_root.rstrip('/') + '/' + title + '.' + part_id + '.id.mp4'
    published = validate_published_part(root, episode, part_id, total_timeout=remaining())
    if published is not None:
        if published['files'][0]['remote'] != target:
            raise ValueError('Previously published partial destination differs; preserve existing receipt')
        write_worker_delivery_ack(root, episode, part_id, total_timeout=remaining())
        return NEXT_PART
    validate_part_release(root, episode, part_id, total_timeout=remaining())
    from .engine.partial_encode import burn_partial_indonesian_mp4, validate_partial_encoding
    result = burn_partial_indonesian_mp4(root, episode, part_id, total_timeout=remaining())
    _, mp4 = validate_partial_encoding(root, episode, part_id, total_timeout=remaining())
    if file_record(mp4, root) != result['output']:
        raise ValueError('partial encoder output identity changed')
    folder = part_directory(root, part_id)
    receipt = upload_verified(mp4, target, total_timeout=remaining(),
                              preservation_receipt=folder / 'final/drive-preservation.json',
                              require_drive_preflight=True)
    if (not isinstance(receipt, dict) or receipt.get('remote') != target
            or type(receipt.get('bytes')) is not int or receipt['bytes'] != result['output']['size_bytes']
            or receipt.get('sha256') != result['output']['sha256']):
        raise ValueError('Partial upload readback does not match exact target bytes/SHA-256')
    validate_partial_encoding(root, episode, part_id, total_timeout=remaining())
    remaining()
    data = {'format': 'mas-part-delivery-1', 'status': 'PASS_PARTIAL', 'episode': episode,
            'part_id': part_id, 'plan_sha256': sha256_file(root / 'work/part-plan.json'),
            'export_sha256': sha256_file(folder / 'work/partial-export.json'),
            'encoding_receipt': result['receipt'], 'mp4': result['output'], 'files': [receipt],
            'perceptual_acceptance': 'NOT_ASSERTED'}
    from .engine.partial_finalize import validate_partial_export
    export, report = validate_partial_export(root, episode, part_id, total_timeout=remaining())
    if report.get('mode') == 'delivery-first-v1':
        data.update(quality_status='NOT_STRICT', quality_report=export['report'])
    atomic_json(folder / 'final/drive_readback_receipt.json', {'data': data, 'sha256': digest(data)})
    validate_published_part(root, episode, part_id, total_timeout=remaining())
    write_worker_delivery_ack(root, episode, part_id, total_timeout=remaining())
    enqueue_notification(episode, part_id + ' Drive teslimi tamamlandi', target, root=root, kind='terminal')
    return NEXT_PART


def complete_parts(root, episode, *, total_timeout=300):
    deadline = _deadline(total_timeout)
    root = Path(root)
    from .engine.part_audio import load_part_plan
    plan = load_part_plan(root, episode, verify_files=False)
    published = []
    for part in plan['parts']:
        if validate_published_part(root, episode, part['part_id'], total_timeout=_remaining(deadline)) is None:
            return None
        published.append(file_record(part_directory(root, part['part_id']) / 'final/drive_readback_receipt.json', root))
    from .delivery_first import EXPORT_MODE
    first_export = read_json(part_directory(root, plan['parts'][0]['part_id']) / 'work/partial-export.json')
    if first_export.get('mode') == EXPORT_MODE:
        import os
        from .full_delivery import publish_full_episode
        return publish_full_episode(root, episode, os.getenv('MAS_DRIVE_STRICT_REMOTE', ''),
                                    total_timeout=_remaining(deadline))
    data = {'format': 'mas-complete-parts-1', 'status': 'COMPLETE_PARTS', 'episode': episode,
            'plan_sha256': sha256_file(root / 'work/part-plan.json'), 'source': plan['source'],
            'audio_sample_count': plan['audio']['sample_count'], 'parts': published,
            'single_full_episode_file': False, 'perceptual_acceptance': 'NOT_ASSERTED'}
    atomic_json(root / 'final/parts-delivery.json', {'data': data, 'sha256': digest(data)})
    enqueue_notification(episode, 'Butun parcalar Drive teslimi tamamlandi',
                         'COMPLETE_PARTS: verified separate part files; no single full-file claim.',
                         root=root, kind='terminal')
    return data
