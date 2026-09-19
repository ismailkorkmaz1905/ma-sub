from pathlib import Path
from contextlib import contextmanager
import math
import shutil
import tempfile
import time

from . import burned_mp4 as burn
from .episode_archive import file_record
from .part_audio import _verify_file
from .partial_finalize import validate_partial_export
from ..delivery import safe_relative, verified_record
from ..reliability import atomic_json, digest, file_digest as sha256_file, read_json


@contextmanager
def _encode_workspace(parent, log_path):
    with tempfile.TemporaryDirectory(prefix='.partial-burn-', dir=parent) as folder:
        work = Path(folder)
        try:
            yield work
        except BaseException:
            encoded = work / 'partial.mp4'
            if encoded.is_file():
                burn._preserve_failed_partial(encoded, log_path)
            raise


def burn_partial_indonesian_mp4(root, episode, part_id, *, total_timeout, target_size_gb=3):
    if (type(total_timeout) not in (int, float) or not math.isfinite(total_timeout)
            or not 0 < total_timeout <= 21600):
        raise ValueError('Partial encode requires a finite remaining episode budget within 6 hours')
    root = Path(root)
    deadline = time.monotonic() + total_timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Partial encode episode wall budget exhausted')
        return value

    remaining()
    export, report = validate_partial_export(root, episode, part_id, total_timeout=remaining())
    from ..partial_delivery import validate_part_release
    validate_part_release(root, episode, part_id, total_timeout=remaining())
    source = verified_record(root, report['input_files']['source_video'])
    subtitles = verified_record(root, report['outputs']['id_srt'])
    settings, probe = burn.plan_encoding_settings(source, encoder='h264_qsv', target_size_gb=target_size_gb)
    scope = report['scope']
    start = scope['start_sample'] / 16000
    duration = (scope['end_sample'] - scope['start_sample']) / 16000
    if not 0 <= start < start + duration <= settings['source_duration_seconds'] + .001:
        raise ValueError('Partial range exceeds immutable source video')
    identity = {'episode': episode, 'part_id': part_id, 'scope': scope,
        'source_sha256': settings['source_sha256'], 'id_srt_sha256': sha256_file(subtitles),
        'lineage_sha256': report['lineage_sha256'], 'report_sha256': export['report']['sha256'],
        'export_sha256': sha256_file(safe_relative(root, f'parts/{part_id}/work/partial-export.json')),
        'settings': settings, 'style': burn.SUBTITLE_STYLE}
    child = safe_relative(root, f'parts/{part_id}')
    output = child / 'output' / f'Muhtemel Ask {episode}.Bolum_{part_id}.id.{digest(identity)[:16]}.mp4'
    receipt_path = child / 'output/partial-encoding.json'
    pending_path = child / 'work/partial-encode-pending.json'
    if not receipt_path.exists() and pending_path.exists():
        wrapped = read_json(pending_path)
        pending = wrapped.get('data')
        if (not isinstance(pending, dict) or wrapped.get('sha256') != digest(pending)
                or pending.get('identity') != identity
                or pending.get('output', {}).get('relative_path') != output.relative_to(root).as_posix()):
            raise ValueError('Pending partial encode publication identity changed')
        if report['mode'] == 'delivery-first-v1':
            from ..delivery_first import verify_evidence
            verify_evidence(pending, 'partial-encoding')
        candidates = (output, output.with_suffix('.encode.partial.mp4'), output.with_suffix('.encode.interrupted.mp4'),
                      output.with_suffix('.partial-encode.failed.mp4'))
        for candidate in candidates:
            if candidate.exists():
                record = {**pending['output'], 'relative_path': candidate.relative_to(root).as_posix()}
                verified_record(root, record)
                if candidate != output:
                    candidate.rename(output)
                atomic_json(receipt_path, pending)
                break
        else:
            raise ValueError('Verified pending partial output is absent; retained evidence cannot authorize re-encode')
    if receipt_path.exists():
        saved, mp4 = validate_partial_encoding(root, episode, part_id, total_timeout=remaining())
        if saved.get('identity') != identity or mp4 != output:
            raise ValueError('Existing partial encoding differs; preserve published output')
        remaining()
        return {'receipt': file_record(receipt_path, root), 'output': file_record(mp4, root)}
    qualification_path = burn.qualify_encoding(
        source, root / 'work/encoding-qualification' / digest(settings)[:16],
        encoder='h264_qsv', target_size_gb=target_size_gb, timeout_seconds=min(240, remaining()))
    settings_bytes = round(settings['target_bytes'] * duration / settings['source_duration_seconds'])
    storage = burn.inspect_encoding_storage(output.parent)
    required = round(settings_bytes * 1.05)
    if storage['filesystem_free_bytes'] < required:
        raise ValueError('Insufficient local storage for proportional partial encode target')
    if output.exists():
        raise ValueError('Unbound partial MP4 exists; preserve it')
    video = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    with _encode_workspace(output.parent, output.with_suffix('.partial-encode.log')) as work:
        shutil.copyfile(subtitles, work / 'id.srt')
        encoded = work / 'partial.mp4'
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-n', '-accurate_seek', '-seek_timestamp', '0',
            '-ss', f'{start:.9f}', '-i', str(source.resolve()), '-t', f'{duration:.9f}',
            '-map', '0:v:0', '-map', '0:a:0', '-sn', '-dn',
            '-vf', ("subtitles=id.srt:force_style='" + burn.SUBTITLE_STYLE + "'"
                    if subtitles.read_text(encoding='utf-8').strip() else 'null'),
            '-c:v', 'h264_qsv', *settings['encoder_options'],
            '-b:v', str(settings['planned_video_bitrate_bps']), '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-metadata:s:a:0', 'language=tur',
            '-movflags', '+faststart', '-progress', 'pipe:1', '-stats_period', '1', str(encoded.resolve())]
        timeout = min(14400, remaining())
        started = time.monotonic()
        burn._run_encode(command, work, output.with_suffix('.partial-encode.log'), encoded,
                         timeout, min(120, timeout))
        encoded_probe = burn._probe(encoded)
        streams = encoded_probe['streams']
        encoded_video = next(stream for stream in streams if stream['codec_type'] == 'video')
        encoded_audio = next(stream for stream in streams if stream['codec_type'] == 'audio')
        if (len(streams) != 2 or encoded_video.get('codec_name') != 'h264'
                or encoded_video.get('pix_fmt') != 'yuv420p' or encoded_audio.get('codec_name') != 'aac'
                or (encoded_video['width'], encoded_video['height']) != (video['width'], video['height'])
                or abs(float(encoded_probe['format']['duration']) - duration) > .1):
            raise ValueError('Partial encoded stream/dimensions/duration verification failed')
        if validate_partial_export(root, episode, part_id, total_timeout=remaining()) != (export, report):
            raise ValueError('Partial subtitle export changed during encoding')
        remaining()
        receipt = {'format': 'mas-partial-burned-id-mp4-1', 'mode': report['mode'],
            'status': 'VERIFIED_PARTIAL_ENCODING', 'full_episode_complete': False,
            'episode': episode, 'part_id': part_id, 'identity': identity,
            'source_range_seconds': [start, start + duration], 'duration_seconds': float(encoded_probe['format']['duration']),
            'soft_target_bytes': settings_bytes, 'qualification': file_record(qualification_path, root),
            'perceptual_acceptance': 'NOT_ASSERTED', 'encode_elapsed_seconds': time.monotonic() - started,
            'storage_preflight': {'required_bytes': required, **storage},
            'output': {'relative_path': output.relative_to(root).as_posix(), 'size_bytes': encoded.stat().st_size,
                       'sha256': sha256_file(encoded)}, 'command': command}
        remaining()
        if report['mode'] == 'delivery-first-v1':
            from ..delivery_first import sign_evidence
            receipt = sign_evidence(receipt, 'partial-encoding')
        atomic_json(pending_path, {'data': receipt, 'sha256': digest(receipt)})
        try:
            burn._stage_output(encoded, output)
        except BaseException:
            burn._preserve_failed_partial(encoded, output.with_suffix('.partial-encode.log'))
            raise
        atomic_json(receipt_path, receipt)
    return {'receipt': file_record(receipt_path, root), 'output': file_record(output, root)}


def validate_partial_encoding(root, episode, part_id, *, total_timeout=300):
    if type(total_timeout) not in (int, float) or not math.isfinite(total_timeout) or not 0 < total_timeout <= 21600:
        raise ValueError('Partial encoding validation requires a bounded remaining episode allowance')
    deadline = time.monotonic() + total_timeout
    root = Path(root)
    export, report = validate_partial_export(root, episode, part_id, total_timeout=total_timeout)
    path = safe_relative(root, f'parts/{part_id}/output/partial-encoding.json')
    receipt = read_json(path)
    if report['mode'] == 'delivery-first-v1':
        from ..delivery_first import verify_evidence
        verify_evidence(receipt, 'partial-encoding')
    identity = receipt.get('identity', {})
    settings = identity.get('settings', {})
    unsigned_settings = {key: value for key, value in settings.items() if key != 'identity_sha256'}
    if (receipt.get('format') != 'mas-partial-burned-id-mp4-1' or receipt.get('mode') != report['mode']
            or receipt.get('status') != 'VERIFIED_PARTIAL_ENCODING' or receipt.get('episode') != episode
            or receipt.get('part_id') != part_id or receipt.get('full_episode_complete') is not False
            or identity.get('scope') != report['scope'] or identity.get('episode') != episode
            or identity.get('part_id') != part_id or identity.get('lineage_sha256') != report['lineage_sha256']
            or identity.get('report_sha256') != export['report']['sha256']
            or identity.get('source_sha256') != report['input_files']['source_video']['sha256']
            or identity.get('id_srt_sha256') != report['outputs']['id_srt']['sha256']
            or identity.get('export_sha256') != sha256_file(safe_relative(root, f'parts/{part_id}/work/partial-export.json'))
            or identity.get('style') != burn.SUBTITLE_STYLE or settings.get('encoder') != 'h264_qsv'
            or settings.get('encoder_options') != burn.ENCODERS['h264_qsv']
            or settings.get('source_sha256') != identity.get('source_sha256')
            or settings.get('identity_sha256') != digest(unsigned_settings)):
        raise ValueError('Partial encoding authority or source/subtitle/range binding changed')
    scope = report['scope']
    expected_range = [scope['start_sample'] / 16000, scope['end_sample'] / 16000]
    if (receipt.get('source_range_seconds') != expected_range
            or abs(receipt['duration_seconds'] - (expected_range[1] - expected_range[0])) > .1):
        raise ValueError('Partial encoded duration or source offset changed')
    qualification_path = _verify_file(root, receipt['qualification'], deadline)
    qualification = read_json(qualification_path)
    if (qualification.get('format') != 'mas-technical-encoder-qualification-1'
            or qualification.get('status') != 'TECHNICALLY_VERIFIED'
            or qualification.get('source_sha256') != identity['source_sha256']
            or qualification.get('settings_identity_sha256') != settings.get('identity_sha256')
            or qualification.get('hardware', {}).get('qsv_hardware_probe') != 'PASS'
            or len(qualification.get('samples', [])) != 3):
        raise ValueError('Partial QSV technical qualification changed')
    for filename, field in (('encoding-samples.json', 'sample_manifest_sha256'),
                            ('technical-encoder-test.srt', 'technical_subtitle_sha256')):
        if sha256_file(qualification_path.parent / filename) != qualification[field]:
            raise ValueError('Partial encoder qualification manifest changed')
    sample_manifest = read_json(qualification_path.parent / 'encoding-samples.json')
    if sample_manifest.get('encoding_settings') != settings or sample_manifest.get('samples') != qualification['samples']:
        raise ValueError('Partial encoder qualification settings or sample records changed')
    for sample in qualification['samples']:
        for key in ('output', 'subtitle'):
            name = sample[key + '_file']
            if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Partial qualification artifact escaped its directory')
            _verify_file(root, {'relative_path': (qualification_path.parent / name).relative_to(root).as_posix(),
                'sha256': sample[key + '_sha256'], 'size_bytes': sample[key + '_bytes']}, deadline)
    expected = f'parts/{part_id}/output/Muhtemel Ask {episode}.Bolum_{part_id}.id.{digest(identity)[:16]}.mp4'
    if receipt['output']['relative_path'] != expected:
        raise ValueError('Partial encoding output escaped its identity-scoped path')
    return receipt, _verify_file(root, receipt['output'], deadline)
