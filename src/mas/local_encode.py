import math
import time
from pathlib import Path

from .delivery import verified_record
from .engine.burned_mp4 import (
    SUBTITLE_STYLE, burn_indonesian_mp4, plan_encoding_settings, qualify_encoding,
)
from .engine.episode_archive import file_record
from .engine.srt import parse_srt
from .hashing import sha256_file
from .reliability import atomic_json, digest, read_json


def write_subtitle_export(root, episode, *, target_size_gb=3):
    root = Path(root)
    report_path = root / 'final' / f'Muhtemel Ask {episode}.Bolum_FINALIZATION_REPORT_V2.json'
    report = read_json(report_path)
    if report.get('status') != 'PASS' or report.get('episode') != episode:
        raise ValueError('local encode requires strict subtitle finalization')
    records = list(report['input_files'].values()) + list(report['outputs'].values())
    for record in records:
        verified_record(root, record)
    plan = {'format': 'mas-local-encode-plan-1', 'episode': episode,
            'execution': 'local-qsv-v1', 'encoder': 'h264_qsv', 'style': SUBTITLE_STYLE,
            'target_size_gb': target_size_gb, 'strict_report_sha256': sha256_file(report_path),
            'source': report['input_files']['source_video'], 'id_srt': report['outputs']['id_srt']}
    plan_path = root / 'work/local-encode-plan.json'
    atomic_json(plan_path, {'data': plan, 'sha256': digest(plan)})
    records += [file_record(path, root) for path in (
        report_path, plan_path, root / 'source/official-source.json', root / 'source/source.url')]
    manifest = {'episode': episode, 'mode': 'strict-subtitles',
                'files': list({item['relative_path']: item for item in records}.values())}
    atomic_json(root / 'work/subtitle-export.json', manifest)
    return manifest


def validate_subtitle_export(root, episode):
    root = Path(root)
    manifest = read_json(root / 'work/subtitle-export.json')
    if manifest.get('episode') != episode or manifest.get('mode') != 'strict-subtitles':
        raise ValueError('subtitle export identity mismatch')
    records = manifest.get('files', [])
    names = [item['relative_path'] for item in records]
    if not records or len(names) != len(set(names)):
        raise ValueError('subtitle export inventory is empty or duplicated')
    for record in records:
        verified_record(root, record)
    bound = read_json(root / 'work/local-encode-plan.json')
    plan = bound.get('data')
    if (not isinstance(plan, dict) or bound.get('sha256') != digest(plan)
            or plan.get('format') != 'mas-local-encode-plan-1' or plan.get('episode') != episode
            or plan.get('execution') != 'local-qsv-v1' or plan.get('encoder') != 'h264_qsv'
            or plan.get('style') != SUBTITLE_STYLE):
        raise ValueError('local encode plan identity mismatch')
    report_path = root / 'final' / f'Muhtemel Ask {episode}.Bolum_FINALIZATION_REPORT_V2.json'
    report = read_json(report_path)
    if (sha256_file(report_path) != plan['strict_report_sha256'] or report.get('status') != 'PASS'
            or report.get('episode') != episode or plan['source'] != report['input_files']['source_video']
            or plan['id_srt'] != report['outputs']['id_srt']):
        raise ValueError('local encode source/subtitle authority mismatch')
    required = list(report['input_files'].values()) + list(report['outputs'].values())
    required += [file_record(report_path, root), file_record(root / 'work/local-encode-plan.json', root)]
    if any(record not in records for record in required):
        raise ValueError('subtitle export omitted required strict evidence')
    tr = parse_srt(verified_record(root, report['outputs']['tr_srt']))
    id_entries = parse_srt(verified_record(root, plan['id_srt']))
    if (not tr or [(e.index, e.start_ms, e.end_ms) for e in tr]
            != [(e.index, e.start_ms, e.end_ms) for e in id_entries]):
        raise ValueError('local encode TR/ID timelines differ')
    return plan, report


def complete_local_encode(root, episode, *, total_timeout):
    root = Path(root)
    deadline = time.monotonic() + total_timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('local encode episode wall budget exhausted')
        return min(value, math.nextafter(float(total_timeout), 0.0))

    plan, report = validate_subtitle_export(root, episode)
    release = read_json(root / 'work/gpu-released-for-encode.json')
    data = release.get('data')
    if (not isinstance(data, dict) or release.get('sha256') != digest(data)
            or data.get('format') != 'mas-gpu-released-for-encode-1'
            or data.get('episode') != episode or data.get('status') != 'ABSENT'
            or data.get('plan_sha256') != sha256_file(root / 'work/local-encode-plan.json')):
        raise ValueError('local encode requires exact external GPU release evidence')
    from .runpod_controller import _capacity_release_evidence
    _capacity_release_evidence(root, episode, data['pod_id'], data['capacity_state'], data['capacity_shutdown'])
    source, subtitles = verified_record(root, plan['source']), verified_record(root, plan['id_srt'])
    settings, _ = plan_encoding_settings(source, encoder='h264_qsv', target_size_gb=plan['target_size_gb'])
    identity = digest({'settings': settings, 'id_srt': plan['id_srt']['sha256'], 'style': SUBTITLE_STYLE})[:16]
    qualification = root / 'work/encoding-qualification' / identity
    qualification_path = qualify_encoding(source, qualification, encoder='h264_qsv',
                                          target_size_gb=plan['target_size_gb'],
                                          timeout_seconds=min(240, remaining()))
    output = root / 'final' / f'Muhtemel Ask {episode}.Bolum.id.{identity}.mp4'
    timeout = min(14400, remaining())
    burn_indonesian_mp4(source, subtitles, output, encoder='h264_qsv',
                        target_size_gb=plan['target_size_gb'], timeout_seconds=timeout,
                        idle_timeout_seconds=min(120, timeout))
    validate_subtitle_export(root, episode)
    remaining()
    delivery = {'format': 'mas-burned-mp4-delivery-1', 'mode': 'strict',
                'strict_finalization_sha256': plan['strict_report_sha256'],
                'encoding_receipt_sha256': sha256_file(output.with_suffix('.burn.json')),
                'execution_plan_sha256': sha256_file(root / 'work/local-encode-plan.json'),
                'qualification': file_record(qualification_path, root),
                'outputs': {'mp4': file_record(output, root)}}
    atomic_json(root / 'final/burned_mp4_delivery.json', delivery)
    return delivery


def validate_local_encoding_evidence(root, episode, delivery, receipt):
    root = Path(root)
    plan, _ = validate_subtitle_export(root, episode)
    if (delivery.get('execution_plan_sha256') != sha256_file(root / 'work/local-encode-plan.json')
            or receipt.get('encoder') != plan['encoder'] or receipt.get('style') != plan['style']):
        raise ValueError('local encoding execution plan binding changed')
    path = verified_record(root, delivery['qualification'])
    qualification = read_json(path)
    settings = receipt['encoding_settings']
    if (qualification.get('format') != 'mas-technical-encoder-qualification-1'
            or qualification.get('status') != 'TECHNICALLY_VERIFIED'
            or qualification.get('source_sha256') != plan['source']['sha256']
            or qualification.get('settings_identity_sha256') != settings['identity_sha256']
            or settings['target_size_gb'] != plan['target_size_gb']
            or qualification.get('hardware', {}).get('qsv_hardware_probe') != 'PASS'
            or len(qualification.get('samples', [])) != 3):
        raise ValueError('local encoding qualification binding changed')
    if (sha256_file(path.parent / 'encoding-samples.json') != qualification['sample_manifest_sha256']
            or sha256_file(path.parent / 'technical-encoder-test.srt') != qualification['technical_subtitle_sha256']):
        raise ValueError('local encoding qualification samples changed')
    for sample in qualification['samples']:
        for kind in ('output', 'subtitle'):
            name = sample[kind + '_file']
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('local qualification sample escaped its directory')
            verified_record(root, {'relative_path': (path.parent / name).relative_to(root).as_posix(),
                                   'size_bytes': sample[kind + '_bytes'], 'sha256': sample[kind + '_sha256']})
