import copy
import json
from pathlib import Path
import shutil

import pytest

import test_finalize as finalize_fixture
from mas.engine import partial_finalize as partial
from mas.engine.download import write_stage_marker
from mas.engine.episode_archive import file_record
from mas.engine.id_translation import create_id_translation_pack
from mas.engine.translation_workspace import REVIEW_CHECKS, collect_id_translation_workspaces, prepare_id_translation_workspaces
from mas.reliability import atomic_json, digest


@pytest.fixture
def partial_fixture(tmp_path, monkeypatch):
    old = finalize_fixture.FinalizeV2Tests()
    root, paths = old._workspace(str(tmp_path))
    old._inputs(root, paths)
    old.doCleanups()
    atomic_json(root / 'source/official-source.json', {'episode': 12, 'url': 'https://example.invalid/12'})
    (root / 'source/source.url').write_text('https://example.invalid/12', encoding='utf-8')
    project = root.parent.parent
    config = Path(__file__).resolve().parents[2] / 'config/production'
    for source in config.glob('*.yaml'):
        shutil.copyfile(source, project / 'config/production' / source.name)
    monkeypatch.setattr(partial, 'ROOT', project)
    child = root / 'parts/part-001'
    for name in ('prepare', 'translation_input', 'translation_output', 'work', 'final'):
        (child / name).mkdir(parents=True, exist_ok=True)
    keys = ('audio', 'raw', 'raw_marker', 'forced', 'forced_marker', 'tr_pack', 'tr_text_output',
            'tr_output', 'audio_review', 'schema', 'id_pack', 'translated')
    child_paths = {key: child / paths[key].relative_to(root) for key in keys}
    for key in keys:
        shutil.copyfile(paths[key], child_paths[key])
    raw = json.loads(child_paths['raw'].read_text(encoding='utf-8'))
    raw['audio_path'] = str(child_paths['audio'].resolve())
    atomic_json(child_paths['raw'], raw)
    write_stage_marker(child_paths['raw_marker'], stage='raw_asr_v2', input_sha256=raw['input_sha256'],
                       outputs={'raw_asr_v2': child_paths['raw']}, details={'audio_sha256': raw['audio_sha256']})
    forced = json.loads(child_paths['forced'].read_text(encoding='utf-8'))
    correction = partial.full.validate_tr_correction_output(child_paths['tr_pack'], child_paths['tr_output'])
    write_stage_marker(child_paths['forced_marker'], stage='forced_alignment_v2', input_sha256='f' * 64,
        outputs={'forced_alignment_v2': child_paths['forced']},
        details={'alignment_sha256': forced['alignment_sha256'], 'correction_output_sha256': correction.output_sha256})
    scope = {'part_id': 'part-001', 'part_index': 1, 'start_ms': 0, 'end_ms': 7000,
             'start_sample': 0, 'end_sample': 112000, 'vad_region_indices': [1, 2], 'caption_indices': []}
    plan = {'episode': 12, 'source': file_record(paths['source'], root),
            'audio': file_record(paths['audio'], root), 'captions': None, 'parts': [scope],
            'vad': {'regions': raw['vad_regions']}}
    atomic_json(root / 'work/part-plan.json', {'data': plan, 'sha256': digest(plan)})
    atomic_json(root / 'work/part-vad.json', {'data': plan['vad'], 'sha256': digest(plan['vad'])})
    lineage = {'format': 'mas-derived-audio-part-1', 'episode': 12, 'part_id': 'part-001',
        'plan_sha256': digest(plan), 'parent_source_sha256': plan['source']['sha256'],
        'parent_audio_sha256': plan['audio']['sha256'], 'audio': file_record(child_paths['audio'], root),
        'sample_rate_hz': 16000, 'sample_count': 112000, 'start_sample': 0, 'end_sample': 112000,
        'parent_vad_sha256': digest(raw['vad_regions'])}
    lineage_path = child / 'prepare/audio-part.done.json'
    atomic_json(lineage_path, {'data': lineage, 'sha256': digest(lineage)})
    derived = {'lineage': lineage, 'lineage_path': lineage_path, 'audio_path': child_paths['audio'],
               'captions_path': None, 'parent_vad_regions': raw['vad_regions']}
    monkeypatch.setattr(partial, 'load_part_plan', lambda *a, **kw: plan)
    monkeypatch.setattr(partial, 'validate_part_audio', lambda *a, **kw: derived)
    values = [partial.full._load_yaml_file(project / 'config/production' / filename, filename)
              for filename in ('series.yaml', 'names.yaml', 'religious_terms.yaml')]
    policy = partial.full.build_production_translation_policy(*values)
    reviewed = partial.full.validate_audio_review_v2_report(child_paths['tr_pack'], child_paths['tr_text_output'],
                                                            child_paths['tr_output'], child_paths['audio_review'])
    artifacts = partial.full.build_strict_v2_artifacts(raw, correction.records, forced,
        episode=12, acoustic_audio_review=reviewed, production_policy=policy,
        part_lineage=lineage, parent_vad_regions=raw['vad_regions'])
    atomic_json(child_paths['schema'], artifacts.schema)
    child_paths['id_pack'].unlink()
    create_id_translation_pack(artifacts.schema, child_paths['id_pack'], glossary=policy['glossary'])
    workspace = prepare_id_translation_workspaces(child_paths['id_pack'], child / 'translation_input/id-workers')
    workers = json.loads(workspace.read_text(encoding='utf-8'))['payload']['workers']
    translations = ['Defne 12 datang.', 'Apa kabar?']
    returns = {}
    for worker in workers:
        data = json.loads((workspace.parent / worker['input_file']).read_text(encoding='utf-8'))
        returns[data['worker_id']] = [{**record, 'id_final': translations[record['block_index'] - 1]}
                                     for record in data['records']]
    pending = collect_id_translation_workspaces(child_paths['id_pack'], workspace, returns)
    reviews = [{**{key: item[key] for key in ('block_uid', 'source_record_sha256', 'translation_sha256',
                                             'context_sha256', 'policy_sha256')},
                'reviewer': 'independent-test-reviewer', 'checks': {key: True for key in REVIEW_CHECKS}}
               for item in pending['risk_review_items']]
    child_paths['translated'].unlink()
    collect_id_translation_workspaces(child_paths['id_pack'], workspace, returns,
                                     reviews=reviews, out_zip=child_paths['translated'])
    return root, child, paths, child_paths, plan, derived


def test_partial_finalization_rebuilds_real_child_quality_without_fake_full_markers(partial_fixture):
    root, child, _, _, _, _ = partial_fixture
    report = partial.finalize_partial_episode(root, 12, 'part-001')
    assert report['mode'] == 'strict-partial' and report['full_episode_complete'] is False
    assert report['speech_coverage_v2']['parent_part_vad_v1']['status'] == 'PASS'
    assert not (child / 'prepare/audio.done.json').exists()
    assert not (child / 'source/download.done.json').exists()
    assert not (root / 'final/Muhtemel Ask 12.Bolum_FINALIZATION_REPORT_V2.json').exists()
    assert partial.validate_partial_export(root, 12, 'part-001')[1] == report
    assert partial.finalize_partial_episode(root, 12, 'part-001') == report


@pytest.mark.parametrize('changed', ['raw_marker', 'forced_marker', 'tr_text_output', 'audio_review', 'translated'])
def test_partial_finalization_rejects_tampered_child_evidence(partial_fixture, changed):
    root, child, _, paths, _, _ = partial_fixture
    paths[changed].write_bytes(b'changed')
    with pytest.raises(Exception):
        partial.finalize_partial_episode(root, 12, 'part-001')
    assert not (child / 'work/partial-export.json').exists()


def test_prior_part_export_remains_valid_when_current_pointer_changes(partial_fixture):
    root, child, _, _, _, _ = partial_fixture
    partial.finalize_partial_episode(root, 12, 'part-001')
    atomic_json(root / 'work/partial-export.json', {'part_id': 'part-002'})
    assert partial.validate_partial_export(root, 12, 'part-001')[0]['part_id'] == 'part-001'


def test_missing_parent_speech_cannot_pass_partial_finalization(partial_fixture):
    root, child, _, _, _, derived = partial_fixture
    derived['parent_vad_regions'] = copy.deepcopy(derived['parent_vad_regions'])
    derived['parent_vad_regions'].append({'vad_region_index': 3, 'start_ms': 6300, 'end_ms': 6900,
                                          'source': 'silero_vad'})
    derived['lineage']['parent_vad_sha256'] = digest(derived['parent_vad_regions'])
    with pytest.raises(Exception, match='complete projected parent VAD'):
        partial.finalize_partial_episode(root, 12, 'part-001')


def test_export_validation_cannot_reset_budget_after_audio_proof(partial_fixture, monkeypatch):
    root, _, _, _, _, derived = partial_fixture
    partial.finalize_partial_episode(root, 12, 'part-001')
    clock = [0.0]
    monkeypatch.setattr(partial.time, 'monotonic', lambda: clock[0])
    def validate(*args, **kwargs):
        assert 0 < kwargs['total_timeout'] <= 1
        clock[0] = 2
        return derived
    monkeypatch.setattr(partial, 'validate_part_audio', validate)
    with pytest.raises(TimeoutError):
        partial.validate_partial_export(root, 12, 'part-001', total_timeout=1)


def test_finalizer_does_not_continue_or_publish_after_child_proof_exhausts_budget(partial_fixture, monkeypatch):
    root, child, _, _, _, derived = partial_fixture
    clock = [0.0]
    monkeypatch.setattr(partial.time, 'monotonic', lambda: clock[0])
    def validate(*args, **kwargs):
        assert 0 < kwargs['total_timeout'] <= 1
        clock[0] = 2
        return derived
    monkeypatch.setattr(partial, 'validate_part_audio', validate)
    monkeypatch.setattr(partial.full, '_source_record', lambda *a: pytest.fail('continued after deadline'))
    with pytest.raises(TimeoutError):
        partial.finalize_partial_episode(root, 12, 'part-001', total_timeout=1)
    assert not (child / 'work/partial-export.json').exists()
