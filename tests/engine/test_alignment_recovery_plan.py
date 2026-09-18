import copy
import json

import pytest

from mas.engine import forced_align as fa
from mas.engine.alignment_recovery import (
    RecoveryCallBudget, RecoveryPlanError, build_recovery_plan,
    request_matches_plan, validate_recovery_plan,
)
from mas.reliability import UnitJournal, atomic_json, digest


def source(count=40):
    return fa.validate_coarse_segments([
        {'utterance_uid': f'u{index:02d}', 'start_ms': index * 500,
         'end_ms': index * 500 + 400, 'text': f'Söz{index}',
         'asr_text': f'Söz{index}', 'deletion_audio_reviewed': False}
        for index in range(count)])


def scope_for(items, roots=None, calls=3):
    roots = roots or ['u08', 'u09']
    identity = {key: 'a' * 64 for key in (
        'audio_sha256', 'model_state_sha256', 'raw_alignment_binding')}
    identity['source_sha256'] = digest(items)
    scope = {'stage': 'forced_alignment', 'target_uids': roots,
             'context_uids': fa._alignment_resume_context_uids(items, roots), **identity,
             'recovery_plan': build_recovery_plan(items, roots, max_new_ctc_calls=calls)}
    return scope, identity


def test_full_actual_joint_call_metadata_matches_frozen_closure():
    items = source()
    scope, identity = scope_for(items)
    request = [{'start': 0.0, 'end': 19.9,
                'text': fa._alignment_model_text(' '.join(item['text'] for item in items))}]
    uids = [item['utterance_uid'] for item in items]
    legacy = {key: value for key, value in scope.items() if key != 'recovery_plan'}
    assert not fa._alignment_request_matches_scope(request, uids, items, legacy, identity)
    assert fa._alignment_request_matches_scope(request, uids, items, scope, identity)
    assert scope['recovery_plan']['components'] == [uids]
    # All successive neighborhoods and component merges were authorized up front.
    for group in (items[:4], items[3:12], items[10:30], items[29:]):
        transcript = [{'start': group[0]['start_ms'] / 1000,
                       'end': group[-1]['end_ms'] / 1000,
                       'text': fa._alignment_model_text(' '.join(item['text'] for item in group))}]
        assert fa._alignment_request_matches_scope(
            transcript, [item['utterance_uid'] for item in group], items, scope, identity)


def test_closure_does_not_cross_unrelated_scene_or_accept_changed_text():
    items = source(3)
    items.append({**items[-1], 'utterance_uid': 'elsewhere', 'text': 'Uzak', 'asr_text': 'Uzak',
                  'start_ms': 30000, 'end_ms': 30500, 'coarse_start_ms': 30000, 'coarse_end_ms': 30500})
    scope, identity = scope_for(items, ['u01'])
    assert scope['recovery_plan']['components'] == [['u00', 'u01', 'u02']]
    assert not fa._alignment_request_matches_scope(
        [{'start': 30.0, 'end': 30.5, 'text': 'Uzak'}], ['elsewhere'], items, scope, identity)
    assert not request_matches_plan([{'start': 0.0, 'end': 0.4, 'text': 'Invented'}],
                                    ['u00'], items, scope['recovery_plan'])
    with pytest.raises(RecoveryPlanError, match='exact source'):
        validate_recovery_plan({**scope['recovery_plan'], 'components': [['u00', 'u01', 'u02', 'elsewhere']]},
                               items, ['u01'])


@pytest.mark.parametrize('start,end,uids', [
    (float('nan'), 1.0, ['u00']), (0.0, float('nan'), ['u00']),
    (0.0, float('inf'), ['u00']), (True, 2.0, ['u00']),
    (1.0, 0.0, ['u00']), (0.0, 0.0, ['u00']),
    ('0', 1.0, ['u00']), (0.0, 0.4, ['u00', 'u00']),
    (0.0, 0.4, [[]]),
])
def test_invalid_request_rejected_before_ctc_for_legacy_and_closure(start, end, uids):
    items = source(3)
    scope, identity = scope_for(items, ['u00'])
    for candidate in (scope, {key: value for key, value in scope.items() if key != 'recovery_plan'}):
        assert not fa._alignment_request_matches_scope(
            [{'start': start, 'end': end, 'text': 'Soz0'}], uids, items, candidate, identity)


def test_plan_tampering_source_changes_and_mixed_authority_fail():
    items = source()
    scope, identity = scope_for(items)
    changed = copy.deepcopy(items)
    changed[15]['text'] = 'Değişti'
    with pytest.raises(RecoveryPlanError):
        validate_recovery_plan(scope['recovery_plan'], changed, scope['target_uids'])
    for field in ('discovery_component_limit', 'continuation_component_limit'):
        with pytest.raises(fa.ForcedAlignmentError, match='mix dynamic'):
            fa._alignment_resume_groups({**scope, field: 3}, items, identity)
    with pytest.raises(RecoveryPlanError):
        build_recovery_plan(items, ['u00', 'u00'], max_new_ctc_calls=3)


def test_call_budget_survives_restart_and_charges_interrupted_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '2' * 64)
    scope, _ = scope_for(source(), calls=2)
    budget = RecoveryCallBudget(tmp_path, scope)
    budget.reserve('a' * 64)
    # Missing raw output after an interrupted call consumes another slot.
    restarted = RecoveryCallBudget(tmp_path, scope)
    restarted.reserve('a' * 64)
    with pytest.raises(RecoveryPlanError, match='allowance exhausted'):
        RecoveryCallBudget(tmp_path, scope).reserve('b' * 64)
    saved = json.loads(budget.path.read_text())
    saved['data']['attempts'] = []
    atomic_json(budget.path, saved)
    with pytest.raises(RecoveryPlanError, match='authentication'):
        RecoveryCallBudget(tmp_path, scope)


def test_closure_requires_auth_and_rejects_rotated_key(tmp_path, monkeypatch):
    scope, _ = scope_for(source())
    monkeypatch.delenv('MAS_RAW_ASR_AUTH_KEY', raising=False)
    with pytest.raises(RecoveryPlanError, match='requires'):
        RecoveryCallBudget(tmp_path, scope)
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '2' * 64)
    RecoveryCallBudget(tmp_path, scope).reserve('a' * 64)
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '3' * 64)
    with pytest.raises(RecoveryPlanError, match='authentication'):
        RecoveryCallBudget(tmp_path, scope)


def test_exact_closure_predecessor_rehydrates_on_restart_without_restoring_latest(tmp_path):
    items = source(3)
    scope, _ = scope_for(items, ['u01'])
    journal = UnitJournal(tmp_path, {'raw': 'fixture'})
    key = digest({'call': 'cached'})
    raw = {'segments': [], 'word_segments': []}
    journal.write(key, raw)
    details = {'status': 'BLOCKED', 'reason': 'recovery_time_budget',
               'component_uids': ['u01'], 'source_sha256': digest([items[1]]),
               'raw_results': {key: digest(raw)}, 'total_recovery_seconds': 900.1,
               'limits': {'total_recovery_seconds': 900}}
    latest = tmp_path / 'components/latest-conflict-failure.json'
    atomic_json(latest, {'data': details, 'sha256': digest(details)})
    archived = fa._archive_authorized_recovery_failure(tmp_path, scope, items, journal)
    assert not latest.exists()
    assert fa._archive_authorized_recovery_failure(tmp_path, scope, items, journal) == archived
    changed = copy.deepcopy(scope)
    changed['recovery_plan']['max_new_ctc_calls'] += 1
    with pytest.raises(fa.ForcedAlignmentError, match='retained conflict'):
        fa._archive_authorized_recovery_failure(tmp_path, changed, items, journal)
    journal.write(key, {'changed': True})
    with pytest.raises(fa.ForcedAlignmentError, match='missing or changed'):
        fa._archive_authorized_recovery_failure(tmp_path, scope, items, journal)


def test_real_alignment_wrapper_reuses_completed_calls_and_persisted_budget(tmp_path, monkeypatch):
    from test_forced_align import ForcedAlignmentTests, _TwoPairOverlapWhisperX
    monkeypatch.setattr(fa, '_model_state_sha256', lambda *args: 'a' * 64)
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '2' * 64)
    audio = tmp_path / 'audio.flac'
    audio.write_bytes(b'synthetic audio placeholder; fake model boundary')
    checkpoint = tmp_path / 'units'
    coarse = ForcedAlignmentTests()._two_pair_coarse()
    cold = fa.align_corrected_segments(audio, coarse, whisperx_module=_TwoPairOverlapWhisperX(),
                                      checkpoint_dir=checkpoint)
    identity = json.loads((checkpoint / 'resume-identity.json').read_text())['data']
    items = identity['source']
    scope = {key: identity[key] for key in ('stage', 'audio_sha256', 'model_state_sha256',
                                           'raw_alignment_binding', 'source_sha256')}
    scope.update(target_uids=['utt-alpha'], context_uids=['utt-alpha', 'utt-bravo'],
                 recovery_plan=build_recovery_plan(items, ['utt-alpha'], max_new_ctc_calls=1))
    retained = None
    for path in checkpoint.glob('*/*.json'):
        saved = json.loads(path.read_text())
        result = saved.get('data', {}).get('result')
        if not isinstance(result, dict):
            continue
        words = result.get('word_segments', [])
        if words and [word['word'] for word in words] == ['Alpha']:
            path.unlink()  # Simulate one unfinished model call; keep all others.
        elif retained is None:
            retained = (saved['data']['uid'], result)
    assert retained is not None
    details = {'status': 'BLOCKED', 'reason': 'recovery_time_budget',
               'component_uids': ['utt-alpha'], 'source_sha256': digest([items[0]]),
               'raw_results': {retained[0]: digest(retained[1])},
               'total_recovery_seconds': 900.1, 'limits': {'total_recovery_seconds': 900}}
    atomic_json(checkpoint / 'components/latest-conflict-failure.json',
                {'data': details, 'sha256': digest(details)})
    warm = _TwoPairOverlapWhisperX()
    assert fa.align_corrected_segments(audio, coarse, whisperx_module=warm,
                                      checkpoint_dir=checkpoint, resume_scope=scope) == cold
    assert warm.align_calls == [{'text': 'Alpha'}]
    restarted = _TwoPairOverlapWhisperX()
    assert fa.align_corrected_segments(audio, coarse, whisperx_module=restarted,
                                      checkpoint_dir=checkpoint, resume_scope=scope) == cold
    assert restarted.align_calls == []
    budget = RecoveryCallBudget(checkpoint, scope)
    assert len(json.loads(budget.path.read_text())['data']['attempts']) == 1


def test_changed_resolver_revalidates_without_new_model_calls(tmp_path, monkeypatch):
    from test_forced_align import ForcedAlignmentTests, _TwoPairOverlapWhisperX
    from pathlib import Path
    from unittest.mock import Mock
    monkeypatch.setattr(fa, '_model_state_sha256', lambda *args: 'a' * 64)
    audio = tmp_path / 'audio.flac'
    audio.write_bytes(b'fake-boundary audio')
    checkpoint = tmp_path / 'units'
    coarse = ForcedAlignmentTests()._two_pair_coarse()
    fa.align_corrected_segments(audio, coarse, whisperx_module=_TwoPairOverlapWhisperX(),
                                checkpoint_dir=checkpoint)
    raw_producer = fa._raw_alignment_producer_sha256()
    original_hash = fa._audio_sha256
    monkeypatch.setattr(fa, '_audio_sha256', lambda path:
                        'f' * 64 if Path(path) == Path(fa.__file__) else original_hash(path))
    normalize = Mock(wraps=fa._normalize_aligned_words_uncached)
    monkeypatch.setattr(fa, '_normalize_aligned_words_uncached', normalize)
    model = _TwoPairOverlapWhisperX()
    fa.align_corrected_segments(audio, coarse, whisperx_module=model, checkpoint_dir=checkpoint)
    assert fa._raw_alignment_producer_sha256() == raw_producer
    assert normalize.call_count > 0
    assert model.align_calls == []
