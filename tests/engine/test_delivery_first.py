import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from mas import delivery_first as df
from mas.engine import delivery_align as align
from mas.engine.id_translation import (build_production_translation_policy, build_id_translation_records,
                                       create_id_translation_output_zip, validate_id_translation_pack)
from mas.engine.part_audio import extract_part_audio, prepare_episode_parts
from mas.engine.partial_finalize import validate_partial_export
from mas.reliability import atomic_json, digest
from test_part_audio import synthetic_parents


@pytest.fixture(autouse=True)
def auth(monkeypatch):
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '2' * 64)


@pytest.fixture
def policy():
    from mas.pipeline import _load_configs
    _, series, names, religious = _load_configs()
    return series, names, religious, build_production_translation_policy(series, names, religious)


def test_policy_preserves_retained_episodes():
    assert not df.enabled(13)
    assert df.enabled(14)


def test_long_incomplete_parent_only_omits_its_own_cue():
    cues, warnings = df.source_cues([
        {'start_ms': 0, 'end_ms': 30000, 'text': 'Eksik uzun metin', 'words': []},
        {'start_ms': 30000, 'end_ms': 31000, 'text': 'Merhaba', 'words': []}], 32000)
    assert [c['text'] for c in cues] == ['Merhaba']
    assert warnings[0]['action'] == 'omitted'


def test_timed_words_split_without_losing_text():
    words = [{'text': str(i), 'start_ms': i * 1000, 'end_ms': i * 1000 + 500} for i in range(15)]
    text = ' '.join(w['text'] for w in words)
    cues, _ = df.source_cues([{'start_ms': 0, 'end_ms': 15000, 'text': text, 'words': words}], 15000)
    assert ' '.join(c['text'] for c in cues) == text
    assert max(c['end_ms'] - c['start_ms'] for c in cues) <= 6000


def test_overlap_never_moves_a_cue_into_a_distant_scene(policy):
    schema, warnings = df.build_schema([
        {'uid': 'one', 'start_ms': 0, 'end_ms': 1000, 'text': 'Merhaba', 'timing_source': 'source_interval_fallback'},
        {'uid': 'two', 'start_ms': 900, 'end_ms': 1800, 'text': 'İyiyim', 'timing_source': 'source_interval_fallback'},
        {'uid': 'three', 'start_ms': 30000, 'end_ms': 31000, 'text': 'Sonra', 'timing_source': 'source_interval_fallback'}
    ], 14, 32000, policy[3])
    assert schema['blocks'][1]['start_ms'] == 1000
    assert schema['blocks'][2]['start_ms'] == 30000
    assert warnings and schema['quality_status'] == 'NOT_STRICT'


class FakeSession:
    calls = []
    def __init__(self, *a, **kw):
        pass
    def align(self, cue, seconds):
        self.calls.append(cue['uid'])
        if cue['uid'] == 'bad':
            raise align.AlignmentUnavailable('alignment_timeout')
        return {'uid': cue['uid'], 'status': 'aligned', 'start_ms': cue['start_ms'], 'end_ms': cue['end_ms']}
    def close(self):
        pass


def test_failed_group_continues_and_does_not_repeat_on_resume(tmp_path):
    cues = [{'uid': 'bad', 'start_ms': 0, 'end_ms': 900, 'text': 'Merhaba'},
            {'uid': 'good', 'start_ms': 1000, 'end_ms': 1900, 'text': 'İyiyim'}]
    FakeSession.calls = []
    kwargs = dict(remaining=lambda: 4000, session_factory=FakeSession)
    first, notes = align.refine_cues(cues, 'audio', tmp_path, {'episode': 14}, **kwargs)
    assert [x['text'] for x in first] == ['Merhaba', 'İyiyim']
    assert first[0]['timing_source'] == 'source_interval_fallback'
    assert first[1]['timing_source'] == 'ctc_cue_bounds'
    assert notes[0]['reason'] == 'alignment_timeout'
    second, _ = align.refine_cues(cues, 'audio', tmp_path, {'episode': 14}, **kwargs)
    assert first == second
    assert FakeSession.calls == ['bad', 'good']


def test_low_budget_never_launches_alignment(tmp_path):
    cue = {'uid': 'a', 'start_ms': 0, 'end_ms': 1000, 'text': 'Merhaba'}
    def forbidden(*a, **kw):
        pytest.fail('spent delivery reserve on alignment')
    result, notes = align.refine_cues([cue], 'audio', tmp_path, {'episode': 14},
                                    remaining=lambda: 1799, session_factory=forbidden)
    assert result[0]['timing_source'] == 'source_interval_fallback'
    assert notes[0]['reason'] == 'delivery_time_reserved'


def test_inflight_group_is_not_relaunched(tmp_path):
    cue = {'uid': 'a', 'start_ms': 0, 'end_ms': 1000, 'text': 'Merhaba'}
    binding = {'episode': 14}
    uid = digest({'binding': binding, 'cue': cue})
    df.write_signed(tmp_path / (uid + '.json'), {'binding': binding, 'cue': cue, 'result': None}, 'alignment-group')
    result, notes = align.refine_cues([cue], 'audio', tmp_path, binding, remaining=lambda: 4000,
                                    session_factory=lambda *a, **kw: pytest.fail('relaunched inflight group'))
    assert notes[0]['reason'] == 'interrupted_group'


def test_actual_hung_subprocess_is_killed(tmp_path):
    code = "import sys,time;print('MAS_ALIGN {\"event\":\"ready\"}',flush=True);sys.stdin.readline();time.sleep(30)"
    session = align.BoundedAligner('unused', tmp_path / 'worker.log', startup_seconds=5,
                                   command=[sys.executable, '-u', '-c', code])
    started = time.monotonic()
    with pytest.raises(align.AlignmentUnavailable, match='timeout'):
        session.align({'uid': 'one'}, .1)
    session.close()
    assert session.process.poll() is not None
    assert time.monotonic() - started < 5


def test_actual_lost_subprocess_is_bounded(tmp_path):
    code = "print('MAS_ALIGN {\"event\":\"ready\"}',flush=True)"
    session = align.BoundedAligner('unused', tmp_path / 'worker.log', startup_seconds=5,
                                   command=[sys.executable, '-u', '-c', code])
    try:
        with pytest.raises(align.AlignmentUnavailable):
            session.align({'uid': 'one'}, .5)
    finally:
        session.close()


def test_handoff_to_real_authenticated_delivery_export(synthetic_parents, monkeypatch, policy):
    root, source, audio, captions, calls = synthetic_parents
    atomic_json(root / 'source/official-source.json', {'episode': 14, 'title': 'Muhtemel Ask 14. Bolum'})
    (root / 'source/source.url').write_text('https://example.invalid/14', encoding='utf-8')
    primary_calls = []
    def primary(*a, **kw):
        primary_calls.append(1)
        return {'identity': {'model': {'fixture': True}, 'producer': {}},
                'segments': [{'start_ms': 500, 'end_ms': 1000, 'text': 'Merhaba', 'words': []}]}
    monkeypatch.setattr(df, 'primary_transcript', primary)
    from mas.engine import delivery_coverage
    monkeypatch.setattr(delivery_coverage, 'model_binding', lambda _: {'model': {'fixture': True}, 'producer': {}})
    series, names, religious, _ = policy
    from mas.progressive import run_progressive_worker
    args = dict(total_timeout=1200, config_dir=None, series=series, names=names, religious=religious)
    assert run_progressive_worker(root, 14, source, audio, captions, **args) == 25
    child = root / 'parts/part-001'
    saved = df.read_signed(child / 'prepare/delivery-schema.json', 'schema')
    schema = saved['schema']
    pack = child / 'translation_input/Muhtemel Ask 14.Bolum_ID_TRANSLATION_PACK.zip'
    out = child / 'translation_output/Muhtemel Ask 14.Bolum_ID_TRANSLATED.zip'
    records = [{**r, 'id_final': 'Halo. ' * 40, 'review_required': True, 'note': 'needs review'}
               for r in build_id_translation_records(schema)]
    create_id_translation_output_zip(schema, records, out, input_manifest=validate_id_translation_pack(pack))
    assert run_progressive_worker(root, 14, source, audio, captions, **args) == 26
    export, report = validate_partial_export(root, 14, 'part-001')
    assert report['quality_status'] == 'NOT_STRICT'
    assert report['status'] == 'READY_WITH_WARNINGS' and report['warnings']
    assert primary_calls == [1]
    assert not (child / 'prepare/forced_alignment_v2.json').exists()
    assert not list(child.glob('final/*PARTIAL_FINALIZATION_REPORT*'))
    # Changing a subtitle and all public checksums cannot forge the export.
    srt = root / report['outputs']['id_srt']['relative_path']
    srt.write_text('1\n00:00:00,500 --> 00:00:01,000\nchanged\n\n', encoding='utf-8')
    from mas.engine.episode_archive import file_record
    replacement = file_record(srt, root)
    export['files'] = [replacement if r['relative_path'] == replacement['relative_path'] else r for r in export['files']]
    atomic_json(child / 'work/partial-export.json', export)
    with pytest.raises(ValueError, match='authentication'):
        validate_partial_export(root, 14, 'part-001')


def test_delivery_workspace_allows_quality_warning_not_changed_identity(tmp_path, policy):
    from mas.engine.translation_workspace import prepare_id_translation_workspaces, collect_id_translation_workspaces, validate_id_workspace_output
    from mas.engine.id_translation import create_id_translation_pack
    schema, _ = df.build_schema([{'uid': 'a', 'start_ms': 0, 'end_ms': 1000,
        'text': 'Allah yardım etsin.', 'timing_source': 'source_interval_fallback'}], 14, 2000, policy[3])
    pack, out = tmp_path / 'pack.zip', tmp_path / 'out.zip'
    create_id_translation_pack(schema, pack, glossary=schema['production_policy']['glossary'])
    workspace = prepare_id_translation_workspaces(pack, tmp_path / 'workers')
    plan = json.loads(workspace.read_text())['payload']
    returns = {}
    for worker in plan['workers']:
        data = json.loads((workspace.parent / worker['input_file']).read_text())
        returns[worker['worker_id']] = [{**r, 'id_final': 'Semoga.', 'review_required': True} for r in data['records']]
    result = collect_id_translation_workspaces(pack, workspace, returns, out_zip=out)
    assert result['status'] == 'DELIVERY_RETURN_CREATED'
    assert validate_id_workspace_output(pack, out)['quality_status'] == 'NOT_STRICT'
    owner = next(key for key, values in returns.items() if values)
    returns[owner][0]['tr_text'] = 'changed'
    with pytest.raises(ValueError, match='immutable'):
        collect_id_translation_workspaces(pack, workspace, returns, out_zip=out)


def test_key_rotation_rejects_delivery_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'saved.json'
    df.write_signed(path, {'episode': 14}, 'schema')
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '3' * 64)
    with pytest.raises(ValueError, match='authentication'):
        df.read_signed(path, 'schema')
