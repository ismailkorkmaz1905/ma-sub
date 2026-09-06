import copy

import pytest

from mas.reliability import IntegrityError, digest
from mas.subtitle.draft_asr import apply_ctc_alignment, coalesce_zero_duration_fragments, reject_new_alignment_overlaps


def test_zero_duration_suffix_preserves_text_and_positive_acoustic_span():
    raw = [{'start': 1, 'end': 2, 'text': 'Degil mi?', 'words': [
        {'start': 1, 'end': 2, 'word': 'Degil'},
        {'start': 2, 'end': 2, 'word': ' mi?'}]}]
    before = copy.deepcopy(raw)
    prepared, issues = coalesce_zero_duration_fragments(raw)
    assert prepared[0]['words'] == [{'start': 1, 'end': 2, 'word': 'Degil mi?'}]
    assert len(issues) == 1
    assert raw == before


def test_zero_duration_prefix_attaches_only_at_identical_onset():
    raw = [{'words': [{'start': 1, 'end': 1, 'word': 'Bana'},
                      {'start': 1, 'end': 2, 'word': ' bak.'}]}]
    prepared, _ = coalesce_zero_duration_fragments(raw)
    assert prepared[0]['words'] == [{'start': 1, 'end': 2, 'word': 'Bana bak.'}]
    raw[0]['words'][1]['start'] = 1.1
    with pytest.raises(IntegrityError):
        coalesce_zero_duration_fragments(raw)


def test_fully_untimed_segment_requires_acoustic_repair():
    with pytest.raises(IntegrityError):
        coalesce_zero_duration_fragments([{'words': [{'start': 1, 'end': 1, 'word': 'A'}]}])


def test_ctc_replaces_only_complete_acoustically_scored_word_intervals():
    source = [{'start': 1, 'end': 2, 'text': 'Merhaba.',
               'words': [{'start': 1, 'end': 2, 'word': 'Merhaba.'}]}]
    record = {'index': 0, 'input_segment_sha256': digest(source[0]),
              'coarse': {'start': .3, 'end': 2.7},
              'raw': {'word_segments': [{'word': 'Merhaba.', 'start': 1.2, 'end': 1.8, 'score': .8}]}}
    aligned, issues, count = apply_ctc_alignment(source, [record])
    assert count == 1 and not issues
    assert (aligned[0]['start'], aligned[0]['end']) == (1.2, 1.8)
    assert source[0]['start'] == 1
    record['coarse']['start'] = 1.2000000000000002
    assert apply_ctc_alignment(source, [record])[2] == 1
    record['coarse']['start'] = 1.21
    assert apply_ctc_alignment(source, [record])[2] == 0
    record['coarse']['start'] = .3
    record['raw']['word_segments'][0]['score'] = .1
    aligned, issues, count = apply_ctc_alignment(source, [record])
    assert count == 0 and aligned == source
    assert issues[0]['kind'] == 'ctc_word_quality_review'
    record['input_segment_sha256'] = 'f' * 64
    with pytest.raises(IntegrityError):
        apply_ctc_alignment(source, [record])


def test_independent_alignment_cannot_introduce_neighbor_overlap():
    original = [dict(start=1, end=2), dict(start=2, end=3), dict(start=3, end=4)]
    aligned = [dict(start=1.1, end=2.3), dict(start=2.1, end=2.9), dict(start=2.95, end=4)]
    prepared, issues = reject_new_alignment_overlaps(original, aligned)
    assert prepared == original
    assert {i['segment_index'] for i in issues} == {0, 1, 2}
    assert aligned[0]['end'] == 2.3


def test_existing_source_overlap_is_not_silently_trimmed():
    original = [dict(start=1, end=2.2), dict(start=2, end=3)]
    aligned = [dict(start=1.1, end=2.3), dict(start=2.1, end=2.9)]
    prepared, issues = reject_new_alignment_overlaps(original, aligned)
    assert prepared == aligned and not issues
