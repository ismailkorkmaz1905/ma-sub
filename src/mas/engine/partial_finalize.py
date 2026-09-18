from pathlib import Path
import math
import hmac
import time

from . import finalize as full
from .raw_asr import (RawASRV2Config, _raw_asr_auth_key, _raw_asr_auth_tag,
                      _verify_raw_asr_artifact_auth, _verify_completed_raw_asr_producer)
from .episode_archive import file_record
from .part_audio import _verify_file, load_part_plan, validate_part_audio
from .srt import assert_srt_roundtrip, parse_srt, write_srt
from .translation_workspace import validate_id_workspace_output
from ..config import ROOT
from ..delivery import safe_relative, verified_record
from ..hashing import sha256_file
from ..reliability import atomic_json, digest, read_json


_PARTIAL_REQUIRED_INPUTS = frozenset({
    'source_video', 'download_metadata', 'download_marker', 'official_source', 'source_url',
    'parent_audio', 'parent_audio_metadata', 'parent_audio_marker', 'part_plan', 'derived_audio',
    'parent_vad', 'derived_audio_marker', 'raw_asr_v2', 'raw_asr_v2_marker',
    'forced_alignment_v2', 'forced_alignment_v2_marker', 'audio_review', 'aligned_schema',
    'tr_pack', 'tr_text_output', 'tr_output', 'id_pack', 'id_output', 'id_workspace_receipt',
})
_PARTIAL_PRODUCER_FILES = (
    'partial_finalize.py', 'finalize.py', 'workflow.py', 'raw_asr.py', 'primary_checkpoint.py',
    'forced_align.py', 'alignment_recovery.py', 'subtitle_metadata.py', 'speaker.py', 'audio_review.py', 'part_audio.py', 'part_scope.py',
    'tr_correction.py', 'id_translation.py', 'translation_workspace.py', 'aligned_schema.py',
    'segmentation.py', 'speech_coverage.py', 'timing_qa.py', 'subtitle_qa.py', 'srt.py',
    'translation_validation.py', 'speaker_evidence.py', 'download.py', 'media.py',
    'episode_archive.py', 'segment.py', 'schema.py',
)


def _partial_export_producer_identity():
    engine = Path(__file__).resolve().parent
    paths = {name: engine / name for name in _PARTIAL_PRODUCER_FILES}
    paths.update({name: ROOT / 'config/production' / name
                  for name in ('series.yaml', 'names.yaml', 'religious_terms.yaml')})
    paths['requirements.lock'] = engine.parents[2] / 'requirements.lock'
    # Git may check out CRLF on the controller and LF on the worker.
    # Normalize only line endings, not source text or configuration values.
    return {name: digest(path.read_text(encoding='utf-8')) for name, path in paths.items()}


def _partial_auth_key():
    key = _raw_asr_auth_key(RawASRV2Config())
    if key is None:
        raise ValueError('Partial export authentication key is unavailable')
    return key


def _sign_partial_export(export, *, key, producer_identity):
    signed = dict(export)
    signed.pop('producer_auth_tag', None)
    signed['auth_format'] = 'mas-strict-partial-export-auth-1'
    signed['producer_identity'] = producer_identity
    signed['producer_auth_tag'] = _raw_asr_auth_tag(key, 'strict-partial-export', signed)
    return signed


def _verify_partial_export_auth(export):
    if not isinstance(export, dict) or export.get('auth_format') != 'mas-strict-partial-export-auth-1':
        raise ValueError('Partial export is unsigned; preserve legacy evidence')
    body = dict(export)
    tag = body.pop('producer_auth_tag', None)
    expected = _raw_asr_auth_tag(_partial_auth_key(), 'strict-partial-export', body)
    if not isinstance(tag, str) or not hmac.compare_digest(tag, expected):
        raise ValueError('Partial export authentication failed for the current key')
    if export.get('producer_identity') != _partial_export_producer_identity():
        raise ValueError('Partial export producer/configuration identity changed')


def _require_partial_inventory(report, plan, derived):
    required = set(_PARTIAL_REQUIRED_INPUTS)
    if plan.get('captions') is not None:
        required.add('parent_captions')
    if derived.get('captions_path') is not None:
        required.add('derived_captions')
    if not isinstance(report.get('input_files'), dict) or set(report['input_files']) != required:
        raise ValueError('Partial export requires the complete canonical strict evidence inventory')
    if not isinstance(report.get('outputs'), dict) or set(report['outputs']) != {'tr_srt', 'id_srt'}:
        raise ValueError('Partial export requires exactly the verified TR and ID subtitles')


def finalize_partial_episode(root, episode, part_id, *, config_dir=None, total_timeout=300):
    if type(total_timeout) not in (int, float) or not math.isfinite(total_timeout) or not 0 < total_timeout <= 21600:
        raise ValueError('Partial finalization requires a bounded remaining episode allowance')
    deadline = time.monotonic() + total_timeout
    auth_key = _partial_auth_key()
    producer_identity = _partial_export_producer_identity()

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Partial finalization episode budget exhausted')
        return value

    root = Path(root)
    if (root.name != full._episode_identity(episode) or root.parent.name != 'EPISODES'
            or root.is_symlink() or not root.is_dir()
            or root.parent.resolve() != (ROOT / 'EPISODES').resolve()):
        raise ValueError('Partial episode root must be the canonical safe EPISODES child')
    plan = load_part_plan(root, episode, verify_files=False)
    part = next((item for item in plan['parts'] if item['part_id'] == part_id), None)
    if part is None:
        raise ValueError('Unknown partial episode scope')
    derived = validate_part_audio(root, episode, part_id, total_timeout=remaining())
    remaining()
    lineage = derived['lineage']
    child = safe_relative(root, f'parts/{part_id}')
    name = full._episode_identity(episode)
    source, source_record = full._source_record(root, verified_record(root, plan['source']), name)
    remaining()
    source_metadata, download_marker, _ = full._validate_source_chain(root, source, source_record)
    remaining()
    parent_audio, audio_metadata, audio_marker, _ = full._validate_audio_chain(root, source, source_record['sha256'])
    remaining()
    if (sha256_file(parent_audio) != lineage['parent_audio_sha256']
            or source_record['sha256'] != lineage['parent_source_sha256']):
        raise ValueError('Partial source lineage differs from verified parent chain')
    prepare = full._require_workflow_directory(child, 'prepare')
    incoming = full._require_workflow_directory(child, 'translation_input')
    outgoing = full._require_workflow_directory(child, 'translation_output')
    full._require_workflow_directory(child, 'work')
    paths = {
        'source_video': source, 'download_metadata': source_metadata, 'download_marker': download_marker,
        'official_source': root / 'source/official-source.json', 'source_url': root / 'source/source.url',
        'parent_audio': parent_audio, 'parent_audio_metadata': audio_metadata, 'parent_audio_marker': audio_marker,
        'part_plan': root / 'work/part-plan.json', 'derived_audio': derived['audio_path'],
        'parent_vad': root / 'work/part-vad.json',
        'derived_audio_marker': derived['lineage_path'],
        'raw_asr_v2': prepare / 'raw_asr_v2.json', 'raw_asr_v2_marker': prepare / 'raw_asr_v2.done.json',
        'forced_alignment_v2': prepare / 'forced_alignment_v2.json',
        'forced_alignment_v2_marker': prepare / 'forced_alignment_v2.done.json',
        'audio_review': prepare / 'audio_review_v2.json', 'aligned_schema': prepare / 'aligned_tr_schema_v2.json',
        'tr_pack': incoming / f'{name}_TR_CORRECTION_PACK.zip',
        'tr_text_output': outgoing / f'{name}_TR_TEXT_CORRECTED.zip',
        'tr_output': outgoing / f'{name}_TR_CORRECTED.zip',
        'id_pack': incoming / f'{name}_ID_TRANSLATION_PACK.zip',
        'id_output': outgoing / f'{name}_ID_TRANSLATED.zip',
        'id_workspace_receipt': outgoing / f'{name}_ID_TRANSLATED.zip.workspace.json',
    }
    if derived['captions_path'] is not None:
        paths['derived_captions'] = derived['captions_path']
    if plan.get('captions') is not None:
        paths['parent_captions'] = verified_record(root, plan['captions'])
    project_root = ROOT
    canonical_config = project_root / 'config/production'
    if config_dir is not None and Path(config_dir).resolve() != canonical_config.resolve():
        raise ValueError('Partial finalization configuration escaped canonical production directory')
    configs = {key: canonical_config / filename for key, filename in (
        ('series', 'series.yaml'), ('names', 'names.yaml'), ('religious', 'religious_terms.yaml'))}
    config_data = {key: full._load_yaml_file(path, key) for key, path in configs.items()}
    all_paths = {**paths, **configs}
    remaining()
    before = full._evidence_snapshot(all_paths)
    remaining()
    raw, _ = full._load_json_file(paths['raw_asr_v2'], 'Child raw ASR')
    _verify_raw_asr_artifact_auth(raw, auth_key)
    _verify_completed_raw_asr_producer(raw)
    child_audio_sha = lineage['audio']['sha256']
    if (raw.get('episode') != episode or raw.get('audio_sha256') != child_audio_sha
            or Path(raw.get('audio_path', '')).resolve() != derived['audio_path'].resolve()):
        raise ValueError('Child raw ASR source identity changed')
    raw_marker = full._validated_stage_marker(
        paths['raw_asr_v2_marker'], stage='raw_asr_v2',
        expected_outputs={'raw_asr_v2': paths['raw_asr_v2']}, expected_input_sha256=raw.get('input_sha256'))
    if raw_marker.get('details', {}).get('audio_sha256') != child_audio_sha:
        raise ValueError('Child raw ASR marker audio identity changed')
    correction_sha = full.compute_input_sha256(
        raw.get('correction_utterances', []), raw.get('speech_hole_records', []), episode=episode,
        asr_hallucination_records=raw.get('asr_hallucination_records', []))
    correction_pack = full.read_tr_correction_pack(paths['tr_pack'], expected_input_sha256=correction_sha)
    if correction_pack.manifest.get('episode') != episode:
        raise ValueError('Child Turkish correction pack episode changed')
    correction = full.validate_tr_correction_output(paths['tr_pack'], paths['tr_output'])
    if any(record.get('review_required') is True for record in correction.records):
        raise ValueError('Child Turkish correction has unresolved review')
    audio_review = full.validate_audio_review_v2_report(
        paths['tr_pack'], paths['tr_text_output'], paths['tr_output'], paths['audio_review'])
    forced, _ = full._load_json_file(paths['forced_alignment_v2'], 'Child forced alignment')
    full.validate_forced_alignment_data(forced)
    if forced.get('audio_sha256') != child_audio_sha:
        raise ValueError('Child forced alignment audio changed')
    forced_marker = full._validated_stage_marker(
        paths['forced_alignment_v2_marker'], stage='forced_alignment_v2',
        expected_outputs={'forced_alignment_v2': paths['forced_alignment_v2']})
    details = forced_marker.get('details', {})
    if (details.get('alignment_sha256') != forced['alignment_sha256']
            or details.get('correction_output_sha256') != correction.output_sha256):
        raise ValueError('Child forced alignment marker/correction identity changed')
    policy = full.build_production_translation_policy(config_data['series'], config_data['names'], config_data['religious'])
    schema, _ = full._load_json_file(paths['aligned_schema'], 'Child aligned schema')
    if schema.get('production_policy') != policy:
        raise ValueError('Partial production policy is absent or changed')
    artifacts = full.build_strict_v2_artifacts(
        raw, correction.records, forced, episode=episode, acoustic_audio_review=audio_review,
        production_policy=policy, part_lineage=lineage, parent_vad_regions=derived['parent_vad_regions'])
    trusted = full.validate_aligned_turkish_schema(schema)
    if trusted != artifacts.schema or trusted['audio_sha256'] != child_audio_sha:
        raise ValueError('Child persisted schema differs from strict evidence rebuild')
    remaining()
    reviewed = full._digest_bound_report(artifacts.audio_review_report,
        digest_field='audio_review_sha256', label='Child acoustic review')
    word_vad = full._digest_bound_report(artifacts.strict_word_vad_report,
        digest_field='strict_word_vad_sha256', label='Child strict word/VAD')
    if (reviewed.get('pending_audio_review_count') != 0 or word_vad.get('unsafe_word_count') != 0
            or word_vad.get('confirmed_dialogue_coverage_fail_count') != 0
            or word_vad.get('audio_sha256') != child_audio_sha
            or word_vad.get('alignment_sha256') != forced['alignment_sha256']
            or word_vad.get('audio_review_sha256') != reviewed['audio_review_sha256']
            or artifacts.speech_coverage_report.get('audio_review_v2') != reviewed
            or artifacts.speech_coverage_report.get('strict_word_vad_v2') != word_vad
            or trusted.get('speech_coverage_sha256') != digest(artifacts.speech_coverage_report)):
        raise ValueError('Child speech/acoustic evidence has unresolved or unbound risk')
    manifest = full.validate_id_translation_pack(paths['id_pack'], expected_schema=trusted)
    translations = full.load_and_validate_id_translation_zip(trusted, paths['id_output'], input_manifest=manifest)
    validate_id_workspace_output(paths['id_pack'], paths['id_output'])
    ordered = translations.ordered_records(trusted)
    if any(record.get('review_required') is True for record in ordered):
        raise ValueError('Child Indonesian translation has unresolved review')
    id_text = {record['block_uid']: record['id_final'] for record in ordered}
    tr_text = {block['block_uid']: block['tr_text'] for block in trusted['blocks']}
    timing = full.run_timing_qa_v2(trusted['blocks'], speech_coverage_report=artifacts.speech_coverage_report,
        alignment_report=artifacts.alignment_report, id_text_by_uid=id_text,
        config=full.TimingQAV2Config(**policy['timing_qa']))
    full.assert_timing_qa_v2(timing)
    subtitle_config = config_data['series']['subtitle']
    target = full._positive_int(subtitle_config['target_chars_per_line'], 'subtitle target line width')
    hard = full._positive_int(subtitle_config['qa_max_chars_per_line'], 'subtitle hard line width')
    if hard < target:
        raise ValueError('Subtitle hard line limit is below target')
    semantic = full.run_subtitle_qa(
        [{**block, 'schema_sha256': trusted['schema_sha256'], 'primary_text': block['tr_text']}
         for block in trusted['blocks']],
        [{**record, 'schema_sha256': trusted['schema_sha256'], 'tr_final': block['tr_text']}
         for block, record in zip(trusted['blocks'], ordered)],
        names_config=config_data['names'], religious_config=config_data['religious'],
        preferred_max_cps=subtitle_config['preferred_max_cps'], line_limit=hard)
    full.assert_final_qa(semantic)
    if semantic.get('review_required_count') != 0:
        raise ValueError('Child semantic QA has unresolved review')
    remaining()
    entries = {language: full._entries(trusted['blocks'], text,
        target_chars_per_line=target, hard_line_limit=hard) for language, text in (('tr', tr_text), ('id', id_text))}
    duration_ms = (lineage['end_sample'] - lineage['start_sample']) * 1000 / lineage['sample_rate_hz']
    if not entries['tr'] or any(entry.start_ms < 0 or entry.end_ms > duration_ms for entry in entries['tr']):
        raise ValueError('Partial subtitle timeline exceeds exact derived audio range')
    final = child / 'final'
    full._safe_output_directory(final, child)
    output_paths = {language + '_srt': final / f'{name}_{part_id}-{language}.srt' for language in entries}
    for language, values in entries.items():
        path = output_paths[language + '_srt']
        if path.is_symlink():
            raise ValueError('Partial subtitle output must not be a symlink')
        if path.exists():
            assert_srt_roundtrip(path, values)
        else:
            write_srt(path, values)
    outputs = {key: file_record(path, root) for key, path in output_paths.items()}
    report = {'format': 'mas-partial-finalization-1', 'status': 'PASS', 'mode': 'strict-partial',
        'episode': episode, 'part_id': part_id, 'full_episode_complete': False,
        'scope': part, 'part_plan_identity_sha256': lineage['plan_sha256'],
        'lineage_sha256': sha256_file(derived['lineage_path']),
        'schema_sha256': trusted['schema_sha256'], 'block_count': trusted['block_count'],
        'input_files': {key: file_record(path, root) for key, path in paths.items()},
        'configuration_files': full._configuration_records(project_root, configs),
        'raw_policy_v2': full._raw_policy_report(raw, raw_artifact_sha256=before['raw_asr_v2']['sha256'], audio_sha256=child_audio_sha),
        'alignment_policy_v2': full._alignment_policy_report(forced, audio_sha256=child_audio_sha),
        'alignment_edit_audit_v2': full._alignment_edit_audit_report(forced),
        'audio_review_v2': reviewed, 'strict_word_vad_v2': word_vad,
        'speech_coverage_v2': artifacts.speech_coverage_report, 'timing_qa_v2': timing,
        'semantic_subtitle_qa': semantic, 'id_translation_validation': full._translation_report(translations, ordered),
        'outputs': outputs}
    remaining()
    if full._evidence_snapshot(all_paths) != before:
        raise ValueError('Partial finalization evidence changed before publication')
    remaining()
    report_path = final / f'{name}_{part_id}_PARTIAL_FINALIZATION_REPORT.json'
    if report_path.exists() and read_json(report_path) != report:
        raise ValueError('Existing partial report differs; preserve published evidence')
    atomic_json(report_path, report)
    records = list(report['input_files'].values()) + list(outputs.values()) + [file_record(report_path, root)]
    export = {'episode': episode, 'part_id': part_id, 'mode': 'strict-partial-subtitles',
              'plan_sha256': sha256_file(root / 'work/part-plan.json'), 'report': file_record(report_path, root),
              'files': list({item['relative_path']: item for item in records}.values())}
    _require_partial_inventory(report, plan, derived)
    if _partial_auth_key() != auth_key or _partial_export_producer_identity() != producer_identity:
        raise ValueError('Partial producer identity changed during finalization')
    export = _sign_partial_export(export, key=auth_key, producer_identity=producer_identity)
    export_path = child / 'work/partial-export.json'
    if export_path.exists() and read_json(export_path) != export:
        raise ValueError('Existing partial export differs; preserve published evidence')
    atomic_json(export_path, export)
    remaining()
    atomic_json(root / 'work/partial-export.json', export)
    return report


def validate_partial_export(root, episode, part_id=None, *, total_timeout=300):
    if type(total_timeout) not in (int, float) or not math.isfinite(total_timeout) or not 0 < total_timeout <= 21600:
        raise ValueError('Partial export requires a bounded remaining episode allowance')
    deadline = time.monotonic() + total_timeout

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Partial export validation budget exhausted')
        return value

    root = Path(root)
    current = (root / 'work/partial-export.json' if part_id is None
               else safe_relative(root, f'parts/{part_id}/work/partial-export.json'))
    export = read_json(current)
    from ..delivery_first import EXPORT_MODE, validate_export
    if export.get('mode') == EXPORT_MODE:
        return validate_export(root, episode, part_id or export.get('part_id'), export,
                               total_timeout=remaining())
    _verify_partial_export_auth(export)
    remaining()
    if (export.get('episode') != episode or export.get('mode') != 'strict-partial-subtitles'
            or (part_id is not None and export.get('part_id') != part_id)):
        raise ValueError('Partial export identity changed')
    part_id = export['part_id']
    plan = load_part_plan(root, episode, verify_files=False)
    derived = validate_part_audio(root, episode, part_id, total_timeout=remaining())
    part = next(item for item in plan['parts'] if item['part_id'] == part_id)
    if export.get('plan_sha256') != sha256_file(root / 'work/part-plan.json'):
        raise ValueError('Partial export plan file identity changed')
    records = export.get('files', [])
    names = [item['relative_path'] for item in records]
    if not records or len(names) != len(set(names)):
        raise ValueError('Partial export inventory empty or duplicated')
    for record in records:
        _verify_file(root, record, deadline)
    expected_report = f'parts/{part_id}/final/{full._episode_identity(episode)}_{part_id}_PARTIAL_FINALIZATION_REPORT.json'
    if export['report']['relative_path'] != expected_report or export['report'] not in records:
        raise ValueError('Partial report escaped scoped evidence inventory')
    report = read_json(verified_record(root, export['report']))
    _require_partial_inventory(report, plan, derived)
    if (report.get('format') != 'mas-partial-finalization-1' or report.get('mode') != 'strict-partial'
            or report.get('status') != 'PASS' or report.get('episode') != episode
            or report.get('part_id') != part_id or report.get('full_episode_complete') is not False
            or report.get('scope') != part or report.get('part_plan_identity_sha256') != derived['lineage']['plan_sha256']
            or report.get('lineage_sha256') != sha256_file(derived['lineage_path'])):
        raise ValueError('Partial report scope/lineage authority changed')
    if any(record not in records for record in list(report['input_files'].values()) + list(report['outputs'].values())):
        raise ValueError('Partial export omitted strict evidence')
    if (report['input_files']['source_video'] != {key: plan['source'][key] for key in ('relative_path', 'sha256', 'size_bytes')}
            or report['input_files']['derived_audio'] != derived['lineage']['audio']):
        raise ValueError('Partial source or audio record changed')
    tr = parse_srt(verified_record(root, report['outputs']['tr_srt']))
    id_entries = parse_srt(verified_record(root, report['outputs']['id_srt']))
    duration = (part['end_sample'] - part['start_sample']) / 16
    if (not tr or len(tr) != report['block_count']
            or [(e.index, e.start_ms, e.end_ms) for e in tr] != [(e.index, e.start_ms, e.end_ms) for e in id_entries]
            or any(e.start_ms < 0 or e.end_ms > duration for e in tr)):
        raise ValueError('Partial TR/ID subtitle range or timeline changed')
    remaining()
    return export, report
