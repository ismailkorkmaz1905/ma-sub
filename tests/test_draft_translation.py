import copy

import pytest

from mas.reliability import IntegrityError, digest
from mas.subtitle.draft_translation import make_pack, render_translation, validate_return


def fixture():
    cue = {'cue_id': 'cue-00001', 'text': 'Merhaba.', 'start_ms': 1200,
           'end_ms': 2300, 'speaker': None, 'segment_index': 0}
    body = {'mode': 'draft', 'strict_delivery': False, 'episode': 12,
            'source_sha256': 'a' * 64, 'audio_sha256': 'b' * 64, 'cues': [cue],
            'rendered': [dict(cue, source_cue_ids=[cue['cue_id']])]}
    draft = {'data': body, 'sha256': digest(body)}
    pack = make_pack(draft, [cue])
    returned = {'pack_sha256': pack['sha256'], 'translations': [
        {'cue_id': cue['cue_id'], 'source_sha256': digest(cue), 'id_text': 'Halo.'}]}
    return draft, pack, returned


@pytest.mark.parametrize('mutation', ['timing', 'binding', 'missing', 'empty'])
def test_translation_return_cannot_change_source_or_omit_dialogue(mutation):
    _, pack, returned = fixture()
    if mutation == 'timing':
        returned['translations'][0]['start_ms'] = 1000
    elif mutation == 'binding':
        returned['translations'][0]['source_sha256'] = 'c' * 64
    elif mutation == 'missing':
        returned['translations'] = []
    else:
        returned['translations'][0]['id_text'] = ' '
    with pytest.raises(IntegrityError):
        validate_return(pack, returned)


def test_translation_preserves_timing_and_rejects_stale_cues():
    draft, pack, returned = fixture()
    before = copy.deepcopy(draft)
    translated = validate_return(pack, returned)
    rendered, issues = render_translation(draft, translated)
    assert rendered[0]['text'] == 'Halo.'
    assert (rendered[0]['start_ms'], rendered[0]['end_ms']) == (1200, 2300)
    assert not issues
    assert draft == before
    draft['data']['cues'][0]['end_ms'] = 2400
    draft['sha256'] = digest(draft['data'])
    with pytest.raises(IntegrityError):
        render_translation(draft, translated)
