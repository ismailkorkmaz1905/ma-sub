import copy
import textwrap

from ..reliability import IntegrityError, digest


def make_pack(draft, cues):
    if draft.get('sha256') != digest(draft.get('data')):
        raise IntegrityError('draft checksum mismatch')
    body = draft['data']
    if body.get('mode') != 'draft' or body.get('strict_delivery') is not False:
        raise IntegrityError('translation requires a separate review draft')
    if any(cue not in body['cues'] for cue in cues):
        raise IntegrityError('translation cue differs from draft')
    pack = {'format': 'mas-draft-translation-pack-1', 'episode': body['episode'],
            'source_sha256': body['source_sha256'], 'audio_sha256': body['audio_sha256'],
            'cues': [{'cue_id': c['cue_id'], 'source_sha256': digest(c),
                      'tr_text': c['text']} for c in cues]}
    return {'data': pack, 'sha256': digest(pack)}


def validate_return(pack, returned):
    if pack.get('sha256') != digest(pack.get('data')):
        raise IntegrityError('translation pack checksum mismatch')
    if set(returned) != {'pack_sha256', 'translations'} or returned['pack_sha256'] != pack['sha256']:
        raise IntegrityError('returned translation pack binding differs')
    translations = returned['translations']
    expected = pack['data']['cues']
    if not isinstance(translations, list) or len(translations) != len(expected):
        raise IntegrityError('translation count differs')
    for cue, translated in zip(expected, translations):
        if (set(translated) != {'cue_id', 'source_sha256', 'id_text'}
                or translated['cue_id'] != cue['cue_id']
                or translated['source_sha256'] != cue['source_sha256']):
            raise IntegrityError('translation identity or immutable fields differ')
        text = translated['id_text']
        if not isinstance(text, str) or not text.strip() or '\x00' in text:
            raise IntegrityError('translation text is empty or invalid')
    return copy.deepcopy(translations)


def render_translation(draft, translations):
    if draft.get('sha256') != digest(draft.get('data')):
        raise IntegrityError('draft checksum mismatch')
    cues = draft['data']['cues']
    if len(translations) != len(cues):
        raise IntegrityError('full translation is incomplete')
    by_id = {}
    for cue, translated in zip(cues, translations):
        if translated['cue_id'] != cue['cue_id'] or translated['source_sha256'] != digest(cue):
            raise IntegrityError('full translation order or source binding differs')
        by_id[cue['cue_id']] = ' '.join(translated['id_text'].split())
    rendered, issues = [], []
    for cue in draft['data']['rendered']:
        texts = [by_id[key] for key in cue['source_cue_ids']]
        lines = ['-' + text for text in texts] if len(texts) > 1 else textwrap.wrap(
            texts[0], width=42, break_long_words=False, break_on_hyphens=False)
        record = dict(cue, text='\n'.join(lines))
        rendered.append(record)
        if len(lines) > 2 or any(len(line) > 42 for line in lines):
            issues.append({'kind': 'id_line_length_review', 'start_ms': cue['start_ms']})
        if sum(map(len, texts)) / ((cue['end_ms'] - cue['start_ms']) / 1000) > 20:
            issues.append({'kind': 'id_reading_speed_review', 'start_ms': cue['start_ms']})
    return rendered, issues
