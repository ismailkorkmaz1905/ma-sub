import math
import time
from pathlib import Path

from .delivery import finalization_report_path, verified_record
from .engine.burned_mp4 import (
    SUBTITLE_STYLE, burn_indonesian_mp4, plan_encoding_settings, qualify_encoding,
)
from .engine.episode_archive import file_record
from .engine.semantic_alignment import prepare_semantic_delivery_scope, read_jsonl
from .engine.srt import parse_srt
from .hashing import sha256_file
from .reliability import atomic_json, digest, read_json


def write_subtitle_export(root, episode, *, target_size_gb=3):
    root = Path(root)
    report_path = finalization_report_path(root, episode)
    report = read_json(report_path)
    if report.get('status') != 'PASS' or report.get('episode') != episode:
        raise ValueError('local encode requires subtitle finalization')
    semantic = report.get('alignment_policy') == 'semantic-block-v1'
    if semantic and (report.get('strict_ctc_pass') is not False
                     or report.get('release_eligible') is not True):
        raise ValueError('local encode semantic release gate has not passed')
    records = list(report['input_files'].values()) + list(report['outputs'].values())
    for record in records:
        verified_record(root, record)
    plan = {'format': 'mas-local-encode-plan-1', 'episode': episode,
            'execution': 'local-qsv-v1', 'encoder': 'h264_qsv', 'style': SUBTITLE_STYLE,
            'target_size_gb': target_size_gb,
            'source': report['input_files']['source_video'], 'id_srt': report['outputs']['id_srt']}
    if semantic:
        state = read_json(root / 'work/state.json')
        delivery_scope = state.get('run_contract', {}).get('delivery_scope')
        selection = prepare_semantic_delivery_scope(
            root, report, delivery_scope=delivery_scope,
            finalization_sha256=sha256_file(report_path))
        plan.update(
            id_srt=file_record(root / selection['id_srt_relative_path'], root),
            tr_srt=file_record(root / selection['tr_srt_relative_path'], root),
            delivery_scope=delivery_scope,
            duration_limit_seconds=(selection['duration_limit_ms'] / 1000.0
                                    if selection['duration_limit_ms'] is not None else None),
            delivery_scope_selection=file_record(selection['path'], root),
        )
        plan.update(alignment_policy='semantic-block-v1',
                    finalization_report_sha256=sha256_file(report_path))
        records += [plan['id_srt'], plan['tr_srt'], plan['delivery_scope_selection']]
    else:
        plan['strict_report_sha256'] = sha256_file(report_path)
    plan_path = root / 'work/local-encode-plan.json'
    atomic_json(plan_path, {'data': plan, 'sha256': digest(plan)})
    records += [file_record(path, root) for path in (
        report_path, plan_path, root / 'source/official-source.json', root / 'source/source.url')]
    manifest = {'episode': episode, 'mode': 'semantic-subtitles' if semantic else 'strict-subtitles',
                'files': list({item['relative_path']: item for item in records}.values())}
    atomic_json(root / 'work/subtitle-export.json', manifest)
    return manifest


def validate_subtitle_export(root, episode):
    root = Path(root)
    manifest = read_json(root / 'work/subtitle-export.json')
    if manifest.get('episode') != episode or manifest.get('mode') not in {'strict-subtitles', 'semantic-subtitles'}:
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
    report_path = finalization_report_path(root, episode)
    report = read_json(report_path)
    semantic = report.get('alignment_policy') == 'semantic-block-v1'
    report_sha = plan.get('finalization_report_sha256') if semantic else plan.get('strict_report_sha256')
    if (sha256_file(report_path) != report_sha or report.get('status') != 'PASS'
            or report.get('episode') != episode or plan['source'] != report['input_files']['source_video']
            or semantic and (manifest.get('mode') != 'semantic-subtitles'
                             or plan.get('alignment_policy') != 'semantic-block-v1'
                             or report.get('release_eligible') is not True)
            or not semantic and plan['id_srt'] != report['outputs']['id_srt']
            or not semantic and manifest.get('mode') != 'strict-subtitles'):
        raise ValueError('local encode source/subtitle authority mismatch')
    required = list(report['input_files'].values()) + list(report['outputs'].values())
    required += [file_record(report_path, root), file_record(root / 'work/local-encode-plan.json', root)]
    if semantic:
        selection_path = verified_record(root, plan['delivery_scope_selection'])
        selection = read_json(selection_path)
        final_blocks_path = verified_record(root, report['input_files']['final_blocks'])
        final_blocks = read_jsonl(final_blocks_path)
        expected_count = (len(final_blocks) if plan.get('delivery_scope') == 'whole-episode'
                          else sum(block['start_ms'] < 3_600_000 for block in final_blocks)
                          if plan.get('delivery_scope') == 'first-hour' else -1)
        expected_duration = (None if plan.get('delivery_scope') == 'whole-episode'
                             else max(3_600_000, final_blocks[expected_count - 1]['end_ms']) / 1000.0
                             if expected_count > 0 else -1)
        if (selection.get('format') != 'mas-semantic-delivery-scope-1'
                or selection.get('episode') != episode
                or selection.get('delivery_scope') != plan.get('delivery_scope')
                or selection.get('finalization_sha256') != sha256_file(report_path)
                or selection.get('canonical_final_blocks_sha256') != sha256_file(final_blocks_path)
                or selection.get('selected_block_count') != expected_count
                or selection.get('selected_block_uids') != [block['block_uid'] for block in final_blocks[:expected_count]]
                or plan.get('duration_limit_seconds') != expected_duration
                or plan['id_srt'] != file_record(root / selection['id_srt_relative_path'], root)
                or plan['tr_srt'] != file_record(root / selection['tr_srt_relative_path'], root)):
            raise ValueError('semantic delivery scope binding changed')
        required += [plan['delivery_scope_selection'], plan['id_srt'], plan['tr_srt']]
    if any(record not in records for record in required):
        raise ValueError('subtitle export omitted required strict evidence')
    tr = parse_srt(verified_record(root, plan['tr_srt'] if semantic else report['outputs']['tr_srt']))
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
    settings, _ = plan_encoding_settings(
        source, encoder='h264_qsv', target_size_gb=plan['target_size_gb'],
        duration_limit_seconds=plan.get('duration_limit_seconds'))
    identity = digest({'settings': settings, 'id_srt': plan['id_srt']['sha256'], 'style': SUBTITLE_STYLE})[:16]
    qualification = root / 'work/encoding-qualification' / identity
    qualification_path = qualify_encoding(source, qualification, encoder='h264_qsv',
                                          target_size_gb=plan['target_size_gb'],
                                          timeout_seconds=min(240, remaining()),
                                          duration_limit_seconds=plan.get('duration_limit_seconds'))
    output = root / 'final' / f'Muhtemel Ask {episode}.Bolum.id.{identity}.mp4'
    timeout = min(14400, remaining())
    burn_indonesian_mp4(source, subtitles, output, encoder='h264_qsv',
                        target_size_gb=plan['target_size_gb'], timeout_seconds=timeout,
                        idle_timeout_seconds=min(120, timeout),
                        duration_limit_seconds=plan.get('duration_limit_seconds'))
    validate_subtitle_export(root, episode)
    remaining()
    semantic = report.get('alignment_policy') == 'semantic-block-v1'
    delivery = {'format': 'mas-burned-mp4-delivery-1',
                'mode': 'semantic-block-v1' if semantic else 'strict',
                'encoding_receipt_sha256': sha256_file(output.with_suffix('.burn.json')),
                'execution_plan_sha256': sha256_file(root / 'work/local-encode-plan.json'),
                'qualification': file_record(qualification_path, root),
                'outputs': {'mp4': file_record(output, root)}}
    if semantic:
        delivery['finalization_sha256'] = plan['finalization_report_sha256']
        delivery['delivery_scope'] = plan['delivery_scope']
        delivery['delivery_scope_sha256'] = plan['delivery_scope_selection']['sha256']
    else:
        delivery['strict_finalization_sha256'] = plan['strict_report_sha256']
    atomic_json(root / 'final/burned_mp4_delivery.json', delivery)
    if semantic:
        atomic_json(
            root / 'final/delivery-scopes' / plan['delivery_scope'] / 'burned_mp4_delivery.json',
            delivery,
        )
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
