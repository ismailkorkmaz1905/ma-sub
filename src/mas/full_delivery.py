"""Assemble already burned parts into one episode, then verify Drive readback."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .delivery import safe_relative, verified_record
from .delivery_first import MODE, EXPORT_MODE, _clock, read_signed, write_signed
from .engine.episode_archive import file_record
from .reliability import digest, file_digest as sha256_file


def _probe(path, timeout):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(path)], capture_output=True, text=True,
                            timeout=max(.001, min(60, timeout)), check=True)
    return json.loads(result.stdout)


def _streams(probe):
    fields = ('codec_type', 'codec_name', 'profile', 'width', 'height', 'pix_fmt',
              'time_base', 'sample_rate', 'channels', 'channel_layout')
    streams = probe.get('streams', [])
    if [s.get('codec_type') for s in streams] != ['video', 'audio']:
        raise ValueError('Full delivery requires exactly one video and one audio stream')
    return [{k: s.get(k) for k in fields} for s in streams]


def _concat_line(path, duration):
    value = str(Path(path).resolve()).replace('\\', '/')
    if any(c in value for c in ('\n', '\r', '\x00')):
        raise ValueError('Unsafe concat input path')
    return "file '" + value.replace("'", "'\\''") + "'\nduration " + f'{duration:.9f}' + '\n'


def assemble_parts(parts, output, *, duration_seconds, remaining):
    from .engine.burned_mp4 import _run_encode
    probes = [_probe(path, remaining()) for path, _ in parts]
    streams = [_streams(p) for p in probes]
    if not streams or any(value != streams[0] for value in streams[1:]):
        raise ValueError('Burned parts have incompatible streams; cannot concatenate safely')
    if any(abs(float(p['format']['duration']) - expected) > .1 for p, (_, expected) in zip(probes, parts)):
        raise ValueError('Burned part duration differs from its source scope')
    output = Path(output)
    concat = output.with_suffix('.ffconcat')
    concat.write_text('ffconcat version 1.0\n' + ''.join(_concat_line(p, d) for p, d in parts),
                      encoding='utf-8', newline='\n')
    temporary = output.with_suffix('.assembling.mp4')
    if temporary.exists():
        temporary.replace(output.with_suffix('.interrupted-' + sha256_file(temporary)[:12] + '.mp4'))
    needed = sum(Path(path).stat().st_size for path, _ in parts)
    if shutil.disk_usage(output.parent).free < needed * 1.05:
        raise OSError('Insufficient local space for the full episode; burned parts retained')
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', '-f', 'concat', '-safe', '0',
               '-i', str(concat), '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy',
               '-t', f'{duration_seconds:.9f}', '-movflags', '+faststart',
               '-progress', 'pipe:1', '-stats_period', '1', str(temporary)]
    _run_encode(command, output.parent, output.with_suffix('.concat.log'), temporary,
                max(.001, remaining()), min(120, max(.001, remaining())))
    probe = _probe(temporary, remaining())
    if _streams(probe) != streams[0] or abs(float(probe['format']['duration']) - duration_seconds) > .2:
        raise ValueError('Assembled full-episode streams or duration do not match')
    temporary.replace(output)
    return probe


def publish_full_episode(root, episode, remote_root, *, total_timeout):
    from .engine.part_audio import load_part_plan
    from .engine.partial_finalize import validate_partial_export
    from .engine.partial_encode import validate_partial_encoding
    from .partial_delivery import validate_published_part, validate_local_tail
    from .remote import upload_verified
    from .source_discovery import CHANNEL_VIDEOS_URL, is_exact_episode_title
    root = Path(root)
    remaining = _clock(total_timeout)
    plan = load_part_plan(root, episode, verify_files=False)
    parts, records, warnings = [], [], []
    expected_start = 0
    for part in plan['parts']:
        # In particular, part-001 must have real readback before a full release.
        published = validate_published_part(root, episode, part['part_id'], total_timeout=max(.001, remaining()))
        if published is None:
            if part['part_id'] == plan['parts'][0]['part_id']:
                return None
            if validate_local_tail(root, episode, part['part_id'], total_timeout=max(.001, remaining())) is None:
                return None
        export, report = validate_partial_export(root, episode, part['part_id'], total_timeout=max(.001, remaining()))
        if export['mode'] != EXPORT_MODE:
            raise ValueError('Full delivery-first assembly cannot mix strict or unrelated parts')
        encoding, mp4 = validate_partial_encoding(root, episode, part['part_id'], total_timeout=max(.001, remaining()))
        if part['start_sample'] != expected_start:
            raise ValueError('Full episode has an omitted or duplicated sample range')
        expected_start = part['end_sample']
        duration = (part['end_sample'] - part['start_sample']) / 16000
        parts.append((mp4, duration))
        records.append({'part_id': part['part_id'], 'scope': part, 'mp4': file_record(mp4, root),
                        'export': file_record(root / f"parts/{part['part_id']}/work/partial-export.json", root)})
        warnings.extend({'part_id': part['part_id'], **note} for note in report['warnings'])
    if expected_start != plan['audio']['sample_count']:
        raise ValueError('Full episode does not cover the final source sample')
    identity = {'episode': episode, 'mode': MODE, 'plan_sha256': sha256_file(root / 'work/part-plan.json'),
                'source': plan['source'], 'parts': records, 'assembler': digest(Path(__file__).read_text(encoding='utf-8'))}
    folder = root / 'output/delivery-first'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / ('full-' + digest(identity)[:16] + '.id.mp4')
    encoded_path, receipt_path = folder / 'full-assembled.json', folder / 'full-drive-receipt.json'
    metadata = json.loads(safe_relative(root, 'source/official-source.json').read_text(encoding='utf-8'))
    title = metadata.get('title', '')
    if (metadata.get('episode') != episode or metadata.get('channel_url') != CHANNEL_VIDEOS_URL
            or not is_exact_episode_title(title, episode) or any(c in title for c in '\\/:*?"<>|\r\n\x00')
            or not remote_root or any(c in remote_root for c in '\r\n\x00')):
        raise ValueError('Invalid full-episode publication destination')
    target = remote_root.rstrip('/') + '/' + title + '.full.id.mp4'
    if receipt_path.exists():
        delivered = read_signed(receipt_path, 'full-delivery')
        if delivered['identity'] != identity or delivered['remote']['remote'] != target:
            raise ValueError('Existing full delivery belongs to different source or destination')
        verified_record(root, delivered['mp4'])
        if (delivered['remote']['sha256'] != delivered['mp4']['sha256']
                or delivered['remote']['bytes'] != delivered['mp4']['size_bytes']):
            raise ValueError('Full delivery readback identity changed')
        return delivered
    if encoded_path.exists():
        encoded = read_signed(encoded_path, 'full-assembled')
        if encoded['identity'] != identity or verified_record(root, encoded['output']) != output:
            raise ValueError('Full assembly checkpoint identity changed')
    else:
        if output.exists():
            output.replace(output.with_suffix('.unverified-' + sha256_file(output)[:12] + '.mp4'))
        assembled = assemble_parts(parts, output, duration_seconds=expected_start / 16000, remaining=remaining)
        encoded = {'identity': identity, 'output': file_record(output, root),
                   'duration_seconds': float(assembled['format']['duration'])}
        write_signed(encoded_path, encoded, 'full-assembled')
    quality_path = folder / 'QUALITY-WARNINGS.json'
    write_signed(quality_path, {'episode': episode, 'quality_status': 'NOT_STRICT',
                               'warnings': warnings, 'mp4': encoded['output']}, 'full-quality')
    remote = upload_verified(output, target, total_timeout=max(.001, remaining()),
        preservation_receipt=folder / 'drive-preservation.json', require_drive_preflight=True)
    if (remote.get('remote') != target or remote.get('bytes') != encoded['output']['size_bytes']
            or remote.get('sha256') != encoded['output']['sha256']):
        raise ValueError('Full episode remote byte/SHA-256 readback mismatch')
    verified_record(root, encoded['output'])
    result = {'status': 'DELIVERED_WITH_WARNINGS', 'quality_status': 'NOT_STRICT',
              'single_full_episode_file': True, 'identity': identity, 'mp4': encoded['output'],
              'remote': remote, 'quality_report': file_record(quality_path, root),
              'perceptual_acceptance': 'NOT_ASSERTED'}
    write_signed(receipt_path, result, 'full-delivery')
    return result
