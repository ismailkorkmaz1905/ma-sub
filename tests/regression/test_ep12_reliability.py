import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from mas.reliability import (
    BudgetExceeded, IntegrityError, OperationFailed, RetryPolicy, RunBudget,
    UnitJournal, atomic_json, bounded_retry, checkpoint_reusable, digest,
    make_checkpoint, run_command,
)
from mas.evidence_guard import (
    ReviewWindow, TimingOverride, classify_word_lanes, semantic_shrink,
)
from mas.progress import Progress

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / 'tests/fixtures/ep12_incident_intervals.json').read_text())['cases']
SHA = 'a' * 64


def test_wall_budget_includes_handoff_and_survives_restart():
    start = datetime(2026, 9, 5, tzinfo=timezone.utc)
    budget = RunBudget(start.isoformat())
    recovered = RunBudget(**json.loads(json.dumps(budget.__dict__)))
    assert recovered.remaining(start + timedelta(hours=3, minutes=30)) == 1800
    with pytest.raises(BudgetExceeded):
        recovered.check(start + timedelta(hours=4))


def test_naive_clock_is_rejected():
    with pytest.raises(ValueError, match='timezone'):
        RunBudget('2026-09-05T00:00:00')


@pytest.mark.parametrize('bad', [0, -1, True, float('nan'), float('inf')])
def test_bad_budget_rejected(bad):
    with pytest.raises(ValueError):
        RunBudget('2026-09-05T00:00:00+00:00', bad)


def test_policy_allocates_exactly_four_hours():
    policy = json.loads((ROOT / 'config/runtime_policy.json').read_text())
    assert sum(policy['planned_stage_minutes'].values()) * 60 == 14400
    assert policy['include_chatgpt_handoff_wait'] is True
    assert policy['allow_automatic_cpu_fallback'] is False
    assert policy['rules']['no_nllb_fallback'] is True


def test_transient_retry_is_bounded():
    calls = []
    clock = [0.0]
    def operation(timeout):
        calls.append(timeout)
        clock[0] += timeout
        raise OperationFailed('UPLOAD', 'timeout', retryable=True)
    with pytest.raises(BudgetExceeded):
        bounded_retry(operation, policy=RetryPolicy(), remaining_seconds=125,
                      clock=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0] + n))
    assert len(calls) == 2
    assert clock[0] == 125


@pytest.mark.parametrize('error', [IntegrityError('hash mismatch'), OperationFailed('AUTH', '401')])
def test_permanent_failures_are_not_retried(error):
    calls = []
    def operation(timeout):
        calls.append(timeout)
        raise error
    with pytest.raises(type(error)):
        bounded_retry(operation, policy=RetryPolicy(), remaining_seconds=300)
    assert len(calls) == 1


def test_success_after_transient_failure():
    calls = []
    def operation(timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise OperationFailed('UPLOAD', 'temporary', retryable=True)
        return 'verified'
    assert bounded_retry(operation, policy=RetryPolicy(), remaining_seconds=300,
                         sleep=lambda n: None) == 'verified'
    assert len(calls) == 2


def test_command_timeout_preserves_log(tmp_path):
    log = tmp_path / 'timeout.log'
    began = time.monotonic()
    with pytest.raises(OperationFailed, match='timed out'):
        run_command([sys.executable, '-S', '-u', '-c', 'import time;print("started");time.sleep(30)'],
                    stage='UPLOAD', log_path=log, timeout_seconds=0.3)
    assert time.monotonic() - began < 5
    assert 'started' in log.read_text()


@pytest.mark.skipif(os.name != 'posix', reason='process groups require POSIX')
def test_timeout_terminates_child_process(tmp_path):
    child = tmp_path / 'child_alive.txt'
    child_code = f'import time;from pathlib import Path;time.sleep(2);Path({str(child)!r}).write_text("alive")'
    parent_code = f'import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,"-S","-c",{child_code!r}]);Path({str(tmp_path / "ready")!r}).write_text("spawned");time.sleep(30)'
    with pytest.raises(OperationFailed):
        run_command([sys.executable, '-S', '-c', parent_code], stage='TRANSFER',
                    log_path=tmp_path/'kill.log', timeout_seconds=0.5)
    assert (tmp_path/'ready').read_text() == 'spawned'
    time.sleep(2.1)
    assert not child.exists()


def test_process_failure_is_not_assumed_retryable(tmp_path):
    with pytest.raises(OperationFailed) as error:
        run_command([sys.executable, '-c', 'raise SystemExit(7)'],
                    stage='ASR', log_path=tmp_path/'exit.log', timeout_seconds=2)
    assert not error.value.retryable


def checkpoint(tmp_path):
    out = tmp_path / 'asr.json'
    out.write_text('[]')
    return make_checkpoint(stage='ASR', run_id='run12', inputs={'audio': SHA},
                           config={'model': 'large-v3'}, outputs=[out],
                           completed_uids=['u1'], git_commit='old-commit',
                           code_sha256='code-hash', container_digest=None,
                           started_at='2026-09-05T00:00:00+00:00',
                           finished_at='2026-09-05T00:01:00+00:00',
                           provider='test', gpu=None, model_versions={'asr': 'fixture'})


def reuse(saved, **changes):
    kwargs = dict(stage='ASR', inputs={'audio': SHA}, config={'model': 'large-v3'}, code_sha256='code-hash')
    kwargs.update(changes)
    return checkpoint_reusable(saved, **kwargs)


def test_asr_checkpoint_survives_id_translation_change(tmp_path):
    saved = checkpoint(tmp_path)
    (tmp_path/'ID_TRANSLATED.zip').write_bytes(b'a completely new translation')
    assert reuse(saved)


@pytest.mark.parametrize('change', [dict(inputs={'audio':'changed'}), dict(config={'model':'changed'}),
                                   dict(code_sha256='changed'), dict(stage='ALIGN')])
def test_changed_dependencies_invalidate_only_matching_stage(tmp_path, change):
    assert not reuse(checkpoint(tmp_path), **change)


def test_corrupt_output_cannot_be_reused(tmp_path):
    saved = checkpoint(tmp_path)
    (tmp_path/'asr.json').write_text('{}')  # Same length, different hash.
    assert not reuse(saved)


def test_missing_output_cannot_be_reused(tmp_path):
    saved = checkpoint(tmp_path)
    (tmp_path/'asr.json').unlink()
    assert not reuse(saved)


def test_empty_outputs_cannot_pass(tmp_path):
    saved = checkpoint(tmp_path)
    saved['data']['outputs'] = []
    saved['sha256'] = digest(saved['data'])
    assert not reuse(saved)


def test_corrupt_checkpoint_is_fatal_not_silently_reset(tmp_path):
    saved = checkpoint(tmp_path)
    saved['data']['inputs']['audio'] = 'changed'
    with pytest.raises(IntegrityError, match='checksum'):
        reuse(saved)


def test_atomic_write_failure_preserves_old_checkpoint(tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    atomic_json(path, {'complete':['u1']})
    import mas.reliability as module
    def fail(*args):
        raise OSError('simulated crash before rename')
    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError):
        atomic_json(path, {'complete':['u1','u2']})
    assert json.loads(path.read_text()) == {'complete':['u1']}


def test_real_process_exit_and_resume_same_completed_uids(tmp_path):
    script = ('import os\nfrom mas.reliability import UnitJournal\nfrom pathlib import Path\n'
              f'j=UnitJournal(Path({str(tmp_path)!r}),{{"audio":"a","config":"b"}})\n'
              'for n in range(5):j.write(str(n),{"text":str(n)})\n'
              'os._exit(17)\n')
    completed = subprocess.run([sys.executable, '-c', script], check=False)
    assert completed.returncode == 17
    journal = UnitJournal(tmp_path, {'audio':'a','config':'b'})
    assert [journal.read(str(n)) for n in range(5)] == [{'text':str(n)} for n in range(5)]
    assert journal.read('5') is None
    assert UnitJournal(tmp_path, {'audio':'new','config':'b'}).read('0') is None


def test_journal_tamper_is_fatal(tmp_path):
    journal = UnitJournal(tmp_path, {'input':SHA})
    journal.write('uid', {'text':'do not change'})
    path = next(journal.root.glob('*.json'))
    value = json.loads(path.read_text())
    value['data']['result']['text'] = 'Ya.'
    path.write_text(json.dumps(value))
    with pytest.raises(IntegrityError, match='checksum'):
        journal.read('uid')


def test_live_progress_does_not_fabricate_completion(tmp_path):
    with Progress(episode=12, run_id='r', stage='ASR', status_path=tmp_path/'status.json',
                  total=10, stream=io.StringIO(), interval_seconds=0.02) as progress:
        progress.advance(3, uid='u3', checkpoint_saved=True)
        time.sleep(0.06)
        snapshot = json.loads((tmp_path/'status.json').read_text())
        assert snapshot['percent'] == 30
        assert snapshot['processed'] == 3
        assert snapshot['status'] == 'RUNNING'
        assert snapshot['last_checkpoint_at']
        assert snapshot['seconds_since_progress'] > 0
        with pytest.raises(ValueError):
            progress.set_state('PASS')
        progress.advance(10)
        progress.set_state('PASS')
    assert json.loads((tmp_path/'status.json').read_text())['status'] == 'PASS'


def test_unknown_workload_has_no_fake_percent_or_eta(tmp_path):
    p = Progress(episode=12, run_id='r', stage='MODEL', status_path=tmp_path/'s', stream=io.StringIO())
    assert p.snapshot()['percent'] is None
    assert p.snapshot()['eta_sec'] is None


def test_review_context_padding_is_not_speech():
    window = ReviewWindow(0, 5000, 1000, 4000, SHA)
    assert window.missing_speech([(2200,2800)], [(2200,2800)]) == []
    assert window.missing_speech([(2200,2800)], [(2200,2500)]) == [(2500,2800)]


@pytest.mark.parametrize('case', CASES, ids=[c['case_id'] for c in CASES])
def test_each_reported_interval_uses_target_not_clip_padding(case):
    # Synthetic interval geometry anchored to the report. Not an audio test or
    # proof that the report's acoustic classification is correct.
    a, b = case['start_ms'], case['end_ms']
    window = ReviewWindow(max(0,a-750),b+750,a,b,SHA)
    midpoint = a + max(1,(b-a)//2)
    assert window.missing_speech([(a,b)], [(a,b)]) == []
    assert window.missing_speech([(a,b)], [(a,midpoint)]) == [(midpoint,b)]
    assert case['evidence_status'] == 'report_only_not_audio_verified'


def override():
    return TimingOverride(SHA,'MA12-TR-f123d5936e04720f',
                          'Kadir Bey bu konuda acaba ne düşünecek?',1000,4000,1100,4100)


def record():
    o=override()
    return dict(utterance_uid=o.utterance_uid,tr_corrected=o.expected_old_text,
                start_ms=1000,end_ms=4000,asr_audit={'source':'fixture'})


def test_timing_override_never_changes_text_or_audit():
    original=record(); saved=copy.deepcopy(original)
    edited=override().apply(original,input_sha256=SHA)
    assert original == saved
    assert edited['tr_corrected'] == original['tr_corrected']
    assert edited['asr_audit'] == original['asr_audit']
    assert (edited['start_ms'],edited['end_ms']) == (1100,4100)


@pytest.mark.parametrize('field', ['replacement_text','tr_corrected','text','non_dialogue'])
def test_timing_override_rejects_text_mutation_keys(field):
    data=override().__dict__.copy();data[field]='Ha?'
    with pytest.raises(IntegrityError,match='keys'):
        TimingOverride.from_dict(data)


@pytest.mark.parametrize('change', [{'tr_corrected':'different text'},{'start_ms':1001},
                                   {'end_ms':4001},{'utterance_uid':'wrong'}])
def test_stale_override_preconditions_rejected(change):
    value=record();value.update(change)
    with pytest.raises(IntegrityError,match='stale'):
        override().apply(value,input_sha256=SHA)


def test_changed_input_hash_rejects_override():
    with pytest.raises(IntegrityError,match='stale'):
        override().apply(record(),input_sha256='b'*64)


@pytest.mark.parametrize('short', ['Ha?','Ne?','Ya.','Evet.',''])
def test_semantic_shrink_alarm(short):
    assert semantic_shrink('Ya Rabbim, sen yardım et Allah\'ım, ne olur.',short)


def test_punctuation_only_correction_does_not_trigger_shrink():
    assert not semantic_shrink('Kadir Bey bu konuda acaba ne düşünecek',
                               'Kadir Bey, bu konuda acaba ne düşünecek?')


def test_cross_speaker_overlap_keeps_both_texts():
    words=[dict(utterance_uid='a',speaker_id='A',start_ms=0,end_ms=1000,text='Hayır.'),
           dict(utterance_uid='b',speaker_id='B',start_ms=500,end_ms=1500,text='Evet.')]
    original=copy.deepcopy(words)
    result=classify_word_lanes(words)
    assert len(result['accepted_overlaps']) == 1
    assert not result['review_required']
    assert words == original


def test_unknown_speaker_overlap_requires_review():
    words=[dict(utterance_uid='a',start_ms=0,end_ms=1000,text='Hayır.'),
           dict(utterance_uid='b',start_ms=500,end_ms=1500,text='Evet.')]
    result=classify_word_lanes(words)
    assert not result['accepted_overlaps']
    assert result['review_required'][0]['reason'] == 'unknown_speaker_overlap'


def test_same_utterance_word_overlap_is_invalid():
    words=[dict(utterance_uid='a',start_ms=0,end_ms=1000),
           dict(utterance_uid='a',start_ms=500,end_ms=1500)]
    with pytest.raises(IntegrityError,match='UID a'):
        classify_word_lanes(words)


def test_rapid_speaker_switches_not_merged():
    words=[dict(utterance_uid=str(n),speaker_id=str(n%2),start_ms=n*200,
                end_ms=(n+1)*200,text=str(n)) for n in range(20)]
    assert classify_word_lanes(words)['word_count'] == 20


def test_resumed_progress_does_not_use_old_work_as_current_speed(tmp_path):
    p = Progress(episode=12, run_id='r', stage='ASR', status_path=tmp_path/'s',
                 total=10, initial_processed=8, stream=io.StringIO())
    assert p.snapshot()['percent'] == 80
    assert p.snapshot()['eta_sec'] is None
    p.advance(9)
    assert p.snapshot()['eta_sec'] is not None
