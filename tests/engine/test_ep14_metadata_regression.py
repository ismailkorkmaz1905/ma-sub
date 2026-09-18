"""Synthetic reproduction of the reported count; no real EP14 media is included."""

import copy
import pytest

from mas import delivery_first as df
from mas.engine.id_translation import IDTranslationError, create_id_translation_pack
from mas.engine.subtitle_metadata import is_subtitle_credit, repetition_warnings
from test_id_translation import _schema


@pytest.mark.parametrize('text', ['Altyazı M.K.', 'ALTYAZI K . M .', 'Takarir M.K.',
                                 'Takarir K.M.', 'TAKARIR M.K.', 'SUBTITLE M.K.', 'Ｓｕｂｔｉｔｌｅ Ｍ．Ｋ．', 'Taka\u200brir M.K.'])
def test_exact_metadata_variants_are_detected(text):
    assert is_subtitle_credit(text)


def test_106_credit_records_are_quarantined_and_time_distribution_remains_visible():
    segments = [{'start_ms': index * 30000, 'end_ms': index * 30000 + 500,
                 'text': 'Altyazı M.K.', 'words': []} for index in range(106)]
    segments.append({'start_ms': 3200000, 'end_ms': 3200800,
                     'text': 'Defne, buraya gel.', 'words': []})
    original = copy.deepcopy(segments)
    cues, warnings = df.source_cues(segments, 3210000)
    assert segments == original
    assert [cue['text'] for cue in cues] == ['Defne, buraya gel.']
    assert sum(warning['action'] == 'omitted' for warning in warnings) == 106
    repeated = [warning for warning in warnings if warning['reason'] == 'repeated_subtitle_text']
    assert len(repeated) == 1
    assert repeated[0]['count'] == 106
    assert repeated[0]['span_ms'] == 3150500
    assert len(repeated[0]['minute_bins']) == 53
    assert len(set(repeated[0]['uids'])) == 106
    assert repeated[0]['known_credit'] is True


def test_repeated_real_dialogue_is_reported_not_removed():
    segments = [{'uid': str(index), 'start_ms': index * 30000,
                 'end_ms': index * 30000 + 500, 'text': 'Evet.', 'words': []} for index in range(7)]
    assert repetition_warnings(segments)[0]['known_credit'] is False
    cues, warnings = df.source_cues(segments, 200000)
    assert len(cues) == 7
    assert all(warning['action'] != 'omitted' for warning in warnings)
    assert not is_subtitle_credit('Altyazı M.K. yazıyor.')


def test_translation_pack_rejects_credit_before_writing_any_zip(tmp_path):
    schema = _schema(1)
    schema['blocks'][0]['tr_text'] = 'Altyazı M.K.'
    target = tmp_path / 'translation.zip'
    with pytest.raises(IDTranslationError, match='metadata cannot enter'):
        create_id_translation_pack(schema, target)
    assert not target.exists()


@pytest.mark.parametrize('text', ['Takarir M.K.', 'TAKARIR K.M.'])
def test_strict_translation_return_cannot_bypass_metadata_gate(tmp_path, text):
    from mas.engine.id_translation import (build_id_translation_records,
                                           create_id_translation_output_zip,
                                           validate_id_translation_records)
    schema = _schema(1)
    records = [{**record, 'id_final': text} for record in build_id_translation_records(schema)]
    result = validate_id_translation_records(schema, records, raise_on_error=False)
    assert not result.ok
    assert any('subtitle-credit metadata' in issue for issue in result.issues)
    target = tmp_path / 'returned.zip'
    with pytest.raises(IDTranslationError):
        create_id_translation_output_zip(schema, records, target)
    assert not target.exists()
