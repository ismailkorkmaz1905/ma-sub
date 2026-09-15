"""Delivery-first episode policy. Quality warnings are not strict PASS evidence."""
import copy
import gc
import hmac
import json
import math
import os
from pathlib import Path
import textwrap
import time

from .hashing import sha256_file
from .reliability import atomic_json, digest, read_json

MODE = 'delivery-first-v1'
EXPORT_MODE = 'delivery-first-subtitles'
POLICY = {'mode': MODE, 'first_episode': 14, 'group_seconds': 20,
          'alignment_seconds': 600, 'delivery_reserve_seconds': 1800,
          'maximum_source_cue_ms': 10000}


def enabled(episode):
    return episode >= POLICY['first_episode']


def _key():
    from .engine.raw_asr import RawASRV2Config, _raw_asr_auth_key
    key = _raw_asr_auth_key(RawASRV2Config())
    if key is None:
        raise ValueError('Delivery evidence authentication key is unavailable')
    return key


def _tag(body, purpose):
    from .engine.raw_asr import _raw_asr_auth_tag
    return _raw_asr_auth_tag(_key(), 'delivery-first-' + purpose, body)


def write_signed(path, body, purpose):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Unsafe delivery checkpoint')
    atomic_json(path, {'data': body, 'auth_tag': _tag(body, purpose)})


def read_signed(path, purpose):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Unsafe delivery checkpoint')
    envelope = read_json(path)
    body, tag = envelope.get('data'), envelope.get('auth_tag')
    if (not isinstance(body, dict) or not isinstance(tag, str)
            or not hmac.compare_digest(tag, _tag(body, purpose))):
        raise ValueError('Delivery checkpoint authentication failed')
    return body


def producer():
    root = Path(__file__).resolve().parents[2]
    names = ['src/mas/delivery_first.py', 'src/mas/engine/delivery_align.py',
             'src/mas/full_delivery.py', 'src/mas/engine/id_translation.py',
             'src/mas/engine/translation_workspace.py', 'src/mas/engine/part_audio.py',
             'src/mas/engine/part_scope.py', 'requirements.lock']
    names += ['config/production/' + x for x in ('series.yaml', 'names.yaml', 'religious_terms.yaml')]
    return {name: digest((root / name).read_text(encoding='utf-8')) for name in names}


def _clock(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 21600:
        raise ValueError('Delivery requires a finite remaining episode allowance')
    deadline = time.monotonic() + seconds
    def remaining():
        return max(0.0, deadline - time.monotonic())
    return remaining


def primary_transcript(audio_path, prepare, episode, series, names, religious):
    # Reuse the existing authenticated primary receipt, not the strict rescue gate.
    from .engine import raw_asr as raw, primary_checkpoint as checkpoint
    from .progress import mark_work_progress
    audio_path, prepare = Path(audio_path), Path(prepare)
    settings = raw.RawASRV2Config(model_name=series['whisper_model'])
    key = _key()
    audio_sha = sha256_file(audio_path)
    model_dir = checkpoint.resolve_model(settings.model_name)
    model_id = checkpoint.model_identity(model_dir)
    prompt = raw._prompt_text(tuple(names['canonical_names']), tuple(x['source'] for x in religious['terms']))
    identity = raw._canonical_primary_identity(
        episode=episode, audio_sha256=audio_sha, settings=settings, prompt=prompt,
        model_identity=model_id, producer_identity=raw._raw_asr_producer_identity())
    path = prepare / 'primary_asr' / (raw.sha256_json(identity) + '.json')
    cached = checkpoint.load_primary(path, identity)
    if cached is not None:
        if raw._primary_receipt_record(prepare, path, identity, key, create=False) is None:
            raise ValueError('Primary transcript lacks authenticated producer evidence')
        return cached
    WhisperModel, ctranslate2 = raw._import_whisper()
    device, compute = raw._select_device(settings.transcription_config(), ctranslate2)
    if device != 'cuda':
        raise ValueError('Delivery ASR requires CUDA; CPU fallback is not enabled')
    model = WhisperModel(str(model_dir), device=device, compute_type=compute,
                         cpu_threads=max(1, os.cpu_count() or 1), num_workers=1)
    try:
        iterator, info = model.transcribe(str(audio_path), **raw._primary_asr_options(settings, prompt))
        def progress():
            for index, segment in enumerate(iterator, 1):
                mark_work_progress('delivery_primary_asr', completed=index)
                yield segment
        segments, words = raw.consume_coarse_segments(progress(), source='main')
        if sha256_file(audio_path) != audio_sha or checkpoint.model_identity(model_dir) != model_id:
            raise ValueError('Source audio or model changed during delivery ASR')
        checkpoint.save_primary(path, identity, str(info.language), segments, words)
        raw._primary_receipt_record(prepare, path, identity, key, create=True)
        return checkpoint.load_primary(path, identity)
    finally:
        model = None
        gc.collect()


def source_cues(segments, duration_ms):
    cues, warnings = [], []
    for index, segment in enumerate(segments):
        start, end = segment.get('start_ms'), segment.get('end_ms')
        text = str(segment.get('text', '')).strip()
        uid = digest({'segment': segment, 'index': index})[:20]
        if not text or type(start) is not int or type(end) is not int or not 0 <= start < end <= duration_ms:
            warnings.append({'uid': uid, 'reason': 'unusable_source_interval', 'action': 'omitted'})
            continue
        words = segment.get('words', [])
        complete = (bool(words) and all(isinstance(w, dict) and isinstance(w.get('text'), str)
                    and type(w.get('start_ms')) is int and type(w.get('end_ms')) is int
                    and start <= w['start_ms'] < w['end_ms'] <= end for w in words)
                    and ''.join(text.split()).casefold() == ''.join(''.join(w['text'].split()) for w in words).casefold()
                    and all(a['end_ms'] <= b['start_ms'] for a, b in zip(words, words[1:])))
        groups = []
        if complete:
            group = []
            for word in words:
                if group and (word['end_ms'] - group[0]['start_ms'] > 6000
                              or len(' '.join(w['text'].strip() for w in group + [word])) > 78
                              or word['start_ms'] - group[-1]['end_ms'] > 700):
                    groups.append((group[0]['start_ms'], group[-1]['end_ms'],
                                   ' '.join(w['text'].strip() for w in group)))
                    group = []
                group.append(word)
            if group:
                groups.append((group[0]['start_ms'], group[-1]['end_ms'], ' '.join(w['text'].strip() for w in group)))
        else:
            groups = [(start, end, text)]
        for number, (a, b, value) in enumerate(groups):
            ident = uid + '-' + str(number)
            if b - a > POLICY['maximum_source_cue_ms']:
                warnings.append({'uid': ident, 'start_ms': a, 'end_ms': b,
                                 'reason': 'unbounded_incomplete_source', 'action': 'omitted'})
                continue
            cues.append({'uid': ident, 'start_ms': a, 'end_ms': b, 'text': value,
                         'timing_source': 'asr_word_interval' if complete else 'source_interval_fallback'})
    return cues, warnings


def build_schema(cues, episode, duration_ms, production_policy):
    from .engine.id_translation import validate_aligned_turkish_schema
    blocks, warnings = [], []
    last_end = 0
    for cue in sorted(cues, key=lambda x: (x['start_ms'], x['end_ms'], x['uid'])):
        start, end = max(last_end, cue['start_ms']), min(duration_ms, cue['end_ms'])
        if end - start < 80:
            warnings.append({'uid': cue['uid'], 'start_ms': cue['start_ms'], 'end_ms': cue['end_ms'],
                             'reason': 'no_display_interval_after_overlap', 'action': 'omitted'})
            continue
        if start != cue['start_ms']:
            warnings.append({'uid': cue['uid'], 'start_ms': cue['start_ms'], 'end_ms': end,
                             'reason': 'source_overlap', 'action': 'trimmed_inside_source_interval'})
        blocks.append({'block_uid': 'DELIVERY-' + cue['uid'], 'block_index': len(blocks) + 1,
                       'start_ms': start, 'end_ms': end, 'tr_text': cue['text'],
                       'alignment_provenance': {'timing_source': cue['timing_source'],
                                                'strict_alignment_pass': False,
                                                'source_interval_ms': [cue['start_ms'], cue['end_ms']]}})
        last_end = end
    if not blocks:
        raise ValueError('No usable Turkish dialogue is available for Indonesian translation')
    return validate_aligned_turkish_schema({
        'schema_version': '2.0', 'episode': episode, 'block_count': len(blocks), 'blocks': blocks,
        'publication_mode': MODE, 'quality_status': 'NOT_STRICT', 'delivery_policy': POLICY,
        'production_policy': production_policy}), warnings


def read_translations(schema, translated_zip):
    from .engine.id_translation import (_open_checked_zip, _parse_jsonl, OUTPUT_BATCH_RE,
                                        build_id_translation_records, ID_OUTPUT_FIELDS, _json_equal)
    expected = {r['block_uid']: r for r in build_id_translation_records(schema)}
    accepted, warnings, seen = {}, [], set()
    with _open_checked_zip(Path(translated_zip)) as archive:
        batches = sorted(name for name in archive.namelist() if OUTPUT_BATCH_RE.fullmatch(name))
        if not batches:
            raise ValueError('No Indonesian translation batches have been supplied')
        for name in batches:
            for record in _parse_jsonl(archive.read(name), member=name):
                uid = record.get('block_uid')
                if uid not in expected or uid in seen:
                    raise ValueError('Unknown or duplicated Indonesian cue identity')
                seen.add(uid)
                source = expected[uid]
                if (set(source) - set(record) or set(record) - set(source) - ID_OUTPUT_FIELDS
                        or any(not _json_equal(record[k], v) for k, v in source.items())):
                    raise ValueError('Indonesian return changed immutable source evidence')
                text = record.get('id_final')
                if not isinstance(text, str) or not text.strip() or any(ord(c) < 32 and c not in '\n\r\t' for c in text):
                    warnings.append({'uid': uid, 'reason': 'missing_indonesian_translation', 'action': 'omitted'})
                else:
                    accepted[uid] = record
                    if record.get('review_required'):
                        warnings.append({'uid': uid, 'reason': 'translator_review_required', 'action': 'delivered_with_warning'})
    warnings += [{'uid': uid, 'reason': 'missing_indonesian_translation', 'action': 'omitted'}
                 for uid in expected if uid not in seen]
    return accepted, warnings


def display_text(text):
    text = ' '.join(str(text).split())
    if any(ord(c) < 32 for c in text):
        raise ValueError('Subtitle contains control characters')
    lines = textwrap.wrap(text, width=42, break_long_words=False, break_on_hyphens=False)
    if len(lines) <= 2:
        return '\n'.join(lines)
    words = text.split()
    middle = min(range(1, len(words)), key=lambda n: abs(len(' '.join(words[:n])) - len(' '.join(words[n:]))))
    return ' '.join(words[:middle]) + '\n' + ' '.join(words[middle:])


def create_export(root, episode, part, derived, schema, transcript_path, schema_path,
                  id_pack, id_output, warnings, *, remaining):
    from .delivery import safe_relative
    from .engine.episode_archive import file_record
    from .engine.part_audio import load_part_plan
    from .engine.srt import SubtitleEntry, write_srt
    from .engine.subtitle_qa import run_subtitle_qa
    plan = load_part_plan(root, episode, verify_files=False)
    child = safe_relative(root, 'parts/' + part['part_id'])
    accepted, notes = read_translations(schema, id_output)
    warnings = list(warnings) + notes
    tr, ind, quality_records = [], [], []
    for block in schema['blocks']:
        record = accepted.get(block['block_uid'])
        if record is None:
            continue
        index = len(ind) + 1
        tr.append(SubtitleEntry(index, block['start_ms'], block['end_ms'], display_text(block['tr_text'])))
        ind.append(SubtitleEntry(index, block['start_ms'], block['end_ms'], display_text(record['id_final'])))
        quality_records.append({**record, 'tr_final': block['tr_text']})
    policy = schema['production_policy']
    qa = run_subtitle_qa([{**b, 'primary_text': b['tr_text']} for b in schema['blocks']], quality_records,
        names_config=policy['names'], religious_config=policy['religious'],
        preferred_max_cps=policy['timing_qa']['maximum_cps'], line_limit=42)
    warnings += [{'reason': issue.get('code', 'subtitle_quality'), 'detail': issue,
                  'action': 'delivered_with_warning'} for issue in qa.get('issues', [])]
    output_paths = {language: child / 'final' / ('DELIVERY-' + language + '.srt') for language in ('tr', 'id')}
    for language, entries in (('tr', tr), ('id', ind)):
        write_srt(output_paths[language], entries)
        if not entries:
            output_paths[language].write_text('\n', encoding='utf-8')
    paths = {'source_video': root / plan['source']['relative_path'],
             'parent_audio': root / plan['audio']['relative_path'],
             'official_source': root / 'source/official-source.json', 'source_url': root / 'source/source.url',
             'part_plan': root / 'work/part-plan.json', 'parent_vad': root / 'work/part-vad.json',
             'derived_audio': derived['audio_path'], 'derived_audio_marker': derived['lineage_path'],
             'transcript': transcript_path, 'schema': schema_path, 'id_pack': id_pack, 'id_output': id_output}
    report = {'format': 'mas-delivery-first-part-1', 'mode': MODE, 'status': 'READY_WITH_WARNINGS',
              'quality_status': 'NOT_STRICT', 'perceptual_acceptance': 'NOT_ASSERTED',
              'episode': episode, 'part_id': part['part_id'], 'scope': part, 'full_episode_complete': False,
              'lineage_sha256': sha256_file(derived['lineage_path']), 'block_count': len(ind),
              'omitted_cue_count': sum(w.get('action') == 'omitted' for w in warnings),
              'warnings': warnings, 'input_files': {k: file_record(p, root) for k, p in paths.items()},
              'outputs': {k + '_srt': file_record(p, root) for k, p in output_paths.items()}}
    report_path = child / 'final/DELIVERY-QUALITY-REPORT.json'
    atomic_json(report_path, report)
    files = list(report['input_files'].values()) + list(report['outputs'].values()) + [file_record(report_path, root)]
    export = {'mode': EXPORT_MODE, 'episode': episode, 'part_id': part['part_id'],
              'plan_sha256': sha256_file(root / 'work/part-plan.json'), 'producer': producer(),
              'report': file_record(report_path, root), 'files': files}
    export['auth_tag'] = _tag(export, 'export')
    if remaining() <= 0:
        raise TimeoutError('Delivery export budget exhausted; outputs retained locally')
    atomic_json(child / 'work/partial-export.json', export)
    atomic_json(root / 'work/partial-export.json', export)
    return export, report


REQUIRED_INPUTS = {'source_video', 'parent_audio', 'official_source', 'source_url', 'part_plan', 'parent_vad',
                   'derived_audio', 'derived_audio_marker', 'transcript', 'schema', 'id_pack', 'id_output'}


def validate_export(root, episode, part_id, export, *, total_timeout):
    from .delivery import safe_relative, verified_record
    from .engine.part_audio import validate_part_audio, load_part_plan, _verify_file
    from .engine.srt import parse_srt
    remaining = _clock(total_timeout)
    root = Path(root)
    body = dict(export)
    tag = body.pop('auth_tag', None)
    if (body.get('mode') != EXPORT_MODE or body.get('episode') != episode or body.get('part_id') != part_id
            or not isinstance(tag, str) or not hmac.compare_digest(tag, _tag(body, 'export'))
            or body.get('producer') != producer()):
        raise ValueError('Delivery export authentication, scope or producer changed')
    plan = load_part_plan(root, episode, verify_files=False)
    part = next(x for x in plan['parts'] if x['part_id'] == part_id)
    derived = validate_part_audio(root, episode, part_id, total_timeout=max(.001, remaining()))
    if export['plan_sha256'] != sha256_file(root / 'work/part-plan.json'):
        raise ValueError('Delivery export plan changed')
    files = export['files']
    if not files or len(files) > 512 or len({r['relative_path'] for r in files}) != len(files):
        raise ValueError('Delivery evidence inventory is invalid')
    deadline = time.monotonic() + remaining()
    for record in files:
        _verify_file(root, record, deadline)
    expected_report = f'parts/{part_id}/final/DELIVERY-QUALITY-REPORT.json'
    if export['report'] not in files or export['report']['relative_path'] != expected_report:
        raise ValueError('Delivery report escaped evidence inventory')
    report = read_json(verified_record(root, export['report']))
    if (report.get('mode') != MODE or report.get('status') != 'READY_WITH_WARNINGS'
            or report.get('quality_status') != 'NOT_STRICT' or report.get('scope') != part
            or report.get('episode') != episode or report.get('part_id') != part_id
            or set(report.get('input_files', {})) != REQUIRED_INPUTS
            or set(report.get('outputs', {})) != {'tr_srt', 'id_srt'}
            or report.get('lineage_sha256') != sha256_file(derived['lineage_path'])
            or report['input_files']['source_video'] != {k: plan['source'][k] for k in ('relative_path', 'sha256', 'size_bytes')}
            or report['input_files']['derived_audio'] != derived['lineage']['audio']):
        raise ValueError('Delivery report contract changed')
    inventory = list(report['input_files'].values()) + list(report['outputs'].values()) + [export['report']]
    if sorted(files, key=lambda r: r['relative_path']) != sorted(inventory, key=lambda r: r['relative_path']):
        raise ValueError('Delivery export requires its complete authenticated evidence')
    tr = parse_srt(verified_record(root, report['outputs']['tr_srt']))
    ind = parse_srt(verified_record(root, report['outputs']['id_srt']))
    duration = (part['end_sample'] - part['start_sample']) / 16
    if (len(ind) != report['block_count'] or [(x.start_ms, x.end_ms) for x in ind] != [(x.start_ms, x.end_ms) for x in tr]
            or any(x.start_ms < 0 or x.end_ms > duration for x in ind)
            or any(a.end_ms > b.start_ms for a, b in zip(ind, ind[1:]))):
        raise ValueError('Delivery subtitle timeline changed')
    if remaining() <= 0:
        raise TimeoutError('Delivery validation allowance exhausted')
    return export, report


def run_worker(root, episode, source_video, audio_path, captions_path=None, *, total_timeout,
               config_dir, series, names, religious):
    from .delivery import NEXT_PART, READY_FOR_PARTIAL_ENCODE, WAIT_PART_RETURN, safe_relative
    from .engine.part_audio import prepare_episode_parts, extract_part_audio
    from .engine.part_scope import get_part
    from .engine.partial_finalize import validate_partial_export
    from .engine.id_translation import (build_production_translation_policy, create_id_translation_pack,
                                        validate_id_translation_pack)
    from .engine.translation_workspace import prepare_id_translation_workspaces
    from .engine.delivery_align import refine_cues
    from .engine.episode_archive import file_record
    from .partial_delivery import validate_worker_published_part
    from .progressive import write_partial_handoff
    root = Path(root).resolve()
    remaining = _clock(total_timeout)
    plan = prepare_episode_parts(root, episode, source_video, audio_path, captions_path,
                                 total_timeout=max(.001, remaining()))
    part = next((p for p in plan['parts'] if validate_worker_published_part(
        root, episode, p['part_id'], total_timeout=max(.001, remaining())) is None), None)
    if part is None:
        return NEXT_PART
    part_id = part['part_id']
    child = safe_relative(root, 'parts/' + part_id)
    for name in ('prepare', 'work', 'final', 'translation_input', 'translation_output'):
        safe_relative(root, f'parts/{part_id}/{name}').mkdir(parents=True, exist_ok=True)
    if (child / 'work/partial-export.json').exists():
        export, _ = validate_partial_export(root, episode, part_id, total_timeout=max(.001, remaining()))
        atomic_json(root / 'work/partial-export.json', export)
        return READY_FOR_PARTIAL_ENCODE
    derived = extract_part_audio(root, episode, part_id, total_timeout=max(.001, remaining()))
    binding = {'episode': episode, 'part_id': part_id, 'audio_sha256': derived['lineage']['audio']['sha256'],
               'plan_sha256': sha256_file(root / 'work/part-plan.json'), 'producer': producer()}
    transcript_path = child / 'prepare/delivery-primary.json'
    schema_path = child / 'prepare/delivery-schema.json'
    if schema_path.exists():
        saved = read_signed(schema_path, 'schema')
        if saved['binding'] != binding:
            raise ValueError('Frozen delivery schema identity changed')
        schema, warnings = saved['schema'], saved['warnings']
    else:
        if transcript_path.exists():
            transcript = read_signed(transcript_path, 'primary')
            if transcript['binding'] != binding:
                raise ValueError('Delivery primary identity changed')
        else:
            raw = primary_transcript(derived['audio_path'], child / 'prepare', episode, series, names, religious)
            transcript = {'binding': binding, 'primary': raw}
            write_signed(transcript_path, transcript, 'primary')
        duration_ms = (part['end_sample'] - part['start_sample']) // 16
        cues, warnings = source_cues(transcript['primary']['segments'], duration_ms)
        refined, notes = refine_cues(cues, derived['audio_path'], child / 'work/delivery-align', binding,
            remaining=remaining, budget_path=root / 'work/delivery-alignment-budget.json',
            group_seconds=POLICY['group_seconds'], total_seconds=POLICY['alignment_seconds'],
            reserve_seconds=POLICY['delivery_reserve_seconds'])
        schema, overlap_notes = build_schema(refined, episode, duration_ms,
            build_production_translation_policy(series, names, religious))
        warnings += notes + overlap_notes
        warnings.append({'reason': 'primary_transcript_without_mandatory_manual_review',
                         'action': 'delivered_with_warning'})
        write_signed(schema_path, {'binding': binding, 'schema': schema, 'warnings': warnings}, 'schema')
    name = f'Muhtemel Ask {episode}.Bolum'
    id_pack = child / 'translation_input' / (name + '_ID_TRANSLATION_PACK.zip')
    id_output = child / 'translation_output' / (name + '_ID_TRANSLATED.zip')
    if id_pack.exists():
        validate_id_translation_pack(id_pack, expected_schema=schema)
    else:
        create_id_translation_pack(schema, id_pack, batch_size=series['batch_size'],
                                   glossary=schema['production_policy']['glossary'])
    workspace = prepare_id_translation_workspaces(id_pack, child / 'translation_input/id-workers')
    if not id_output.exists():
        context = [derived['lineage_path'], transcript_path, schema_path, Path(workspace)]
        context += [Path(workspace).parent / f'worker-{i:02d}/input.json' for i in range(1, 4)]
        write_partial_handoff(root, episode, part_id, 'id', id_pack, id_output, context)
        return WAIT_PART_RETURN
    create_export(root, episode, part, derived, schema, transcript_path, schema_path,
                  id_pack, id_output, warnings, remaining=remaining)
    validate_partial_export(root, episode, part_id, total_timeout=max(.001, remaining()))
    return READY_FOR_PARTIAL_ENCODE


def collect_workspace(pack_path, workspace_path, worker_returns, *, reviews=(), out_zip=None):
    """Keep cue ownership strict; pending linguistic review becomes a warning."""
    from .engine.translation_workspace import _pack, _json_equal
    from .engine.id_translation import _write_zip_atomic, _jsonl_bytes, _pretty_json_bytes, ID_OUTPUT_FIELDS
    schema, manifest, records, freeze = _pack(pack_path)
    envelope = read_json(workspace_path)
    workspace = envelope.get('payload')
    if not isinstance(workspace, dict) or envelope.get('sha256') != digest(workspace) or workspace.get('freeze') != freeze:
        raise ValueError('Delivery translation workspace identity changed')
    expected = {r['block_uid']: r for r in records}
    workers = workspace.get('workers', [])
    owned = [uid for w in workers for uid in w['owned_uids']]
    seeds = workspace.get('seed_records', [])
    seed_uids = [r['block_uid'] for r in seeds]
    if (len(owned + seed_uids) != len(set(owned + seed_uids)) or set(owned + seed_uids) != set(expected)
            or not isinstance(worker_returns, dict) or set(worker_returns) - {w['worker_id'] for w in workers}):
        raise ValueError('Delivery translation ownership is invalid')
    received = list(seeds)
    waiting = []
    for worker in workers:
        values = worker_returns.get(worker['worker_id'])
        if values is None:
            if worker['owned_uids']:
                waiting.append(worker['worker_id'])
            continue
        if not isinstance(values, list) or any(r.get('block_uid') not in worker['owned_uids'] for r in values):
            raise ValueError('Translation worker returned a cue it does not own')
        received.extend(values)
    if waiting:
        if out_zip is not None:
            raise ValueError('Indonesian worker returns are still missing: ' + ', '.join(waiting))
        return {'status': 'WAIT_TRANSLATION', 'waiting_workers': waiting, 'quality_status': 'NOT_STRICT'}
    accepted = {}
    for record in received:
        uid = record.get('block_uid')
        if uid not in expected or uid in accepted:
            raise ValueError('Unknown or duplicated translation cue')
        source = expected[uid]
        if (set(source) - set(record) or set(record) - set(source) - ID_OUTPUT_FIELDS
                or any(not _json_equal(record[k], v) for k, v in source.items())):
            raise ValueError('Translation changed immutable source evidence')
        accepted[uid] = record
    ordered = [accepted[r['block_uid']] for r in records if r['block_uid'] in accepted]
    result = {'status': 'DELIVERY_RETURN_READY', 'quality_status': 'NOT_STRICT',
              'scope_uids': [r['block_uid'] for r in records], 'accepted_records': ordered,
              'missing_uids': [r['block_uid'] for r in records if r['block_uid'] not in accepted],
              'reviews': list(reviews), 'freeze': freeze}
    if out_zip is not None:
        payloads = {}
        offset = 0
        for batch in manifest['batches']:
            group = records[offset:offset + batch['block_count']]
            payloads[batch['output_file']] = _jsonl_bytes([accepted[r['block_uid']] for r in group if r['block_uid'] in accepted])
            offset += batch['block_count']
        payloads['translation_report.json'] = _pretty_json_bytes({
            'kind': 'delivery-first-indonesian-output', 'schema_sha256': schema['schema_sha256'],
            'total_input_blocks': len(records), 'total_output_blocks': len(ordered), 'quality_status': 'NOT_STRICT'})
        _write_zip_atomic(Path(out_zip), payloads)
        read_translations(schema, out_zip)
        result['status'] = 'DELIVERY_RETURN_CREATED'
        write_signed(Path(str(out_zip) + '.workspace.json'), result, 'translation-return')
    return result
