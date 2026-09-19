import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas import partial_delivery, pipeline, progressive, runpod_controller as controller
from mas.delivery import NEXT_PART, READY_FOR_PARTIAL_ENCODE, WAIT_PART_RETURN, verified_record
from mas.engine import part_audio, partial_finalize
from mas.engine.episode_archive import file_record
from mas.engine.part_scope import build_part_plan
from mas.reliability import BudgetExceeded, atomic_json, digest, file_digest as sha256_file
from mas.remote_job import _identity, checkpoint_manifest


class Budget:
    def check(self):
        return 600


def _bound(path, data):
    atomic_json(path, {'data': data, 'sha256': digest(data)})


def _plan(root, captions=None):
    source = {'relative_path': 'source/movie.mp4', 'size_bytes': 10, 'sha256': '1' * 64}
    audio = {'relative_path': 'work/audio.flac', 'size_bytes': 10, 'sha256': '2' * 64,
             'sample_rate_hz': 16000, 'channels': 1, 'sample_count': 7200000 * 16}
    regions = [{'vad_region_index': i, 'start_ms': start, 'end_ms': end, 'source': 'silero_vad'}
               for i, (start, end) in enumerate([(1000, 2500), (3600000, 3601000),
                                                  (3603000, 3605000), (7197000, 7199000)], 1)]
    plan = build_part_plan(episode=15, source=source, audio=audio,
        vad={'independent_vad': True, 'audio_sha256': audio['sha256'], 'sample_count': audio['sample_count'],
             'config': {'policy': 'canonical'}, 'model': {'name': 'silero'},
             'producer_sha256': '3' * 64, 'regions': regions}, captions=captions)
    _bound(root / 'work/part-plan.json', plan)
    controller._episode_budget(root, 15)
    return plan


def _handoff(root, kind='tr'):
    child = root / 'parts/part-001'
    name = 'Muhtemel Ask 15.Bolum'
    pack = child / 'handoff' / (name + ('_TR_CORRECTION_PACK.zip' if kind == 'tr' else '_ID_TRANSLATION_PACK.zip'))
    output = child / 'handoff' / (name + ('_TR_TEXT_CORRECTED.zip' if kind == 'tr' else '_ID_TRANSLATED.zip'))
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_bytes(kind.encode())
    return progressive.write_partial_handoff(root, 15, 'part-001', kind, pack, output, [pack])


def _transport(monkeypatch, remote):
    calls = []
    def network(command, **kwargs):
        source = command[-2].split(':', 1)[1]
        shutil.copyfile(remote / source.removeprefix('/episode/'), command[-1])
    def download(record, local, remote_root, *args, **kwargs):
        calls.append(record['relative_path'])
        source = verified_record(remote, record)
        target = local / record['relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return verified_record(local, record)
    monkeypatch.setattr(controller, '_network_retry', network)
    monkeypatch.setattr(controller, '_download_record', download)
    return calls


def _release(root, code):
    audit = root / 'work/capacity/run'
    pods = [{'pod_id': 'owned', 'status': 'ABSENT', 'error_type': None}]
    state = {'format': 'mas-capacity-lease-state-1', 'episode': 15, 'status': 'RELEASED',
             'owned_pod_ids': ['owned'], 'shutdown': pods}
    _bound(audit / 'capacity-state.json', state)
    _bound(audit / 'capacity-shutdown.json', {'state_sha256': digest(state), 'owned_pods': pods})
    _bound(root / 'work/controller-lease.json', {'commit': 'a' * 40, 'audit': 'work/capacity/run'})
    request = {'episode': 15, 'commit': 'a' * 40, 'input_sha256': 'b' * 64}
    _bound(root / 'work/remote-job-request.json', request)
    atomic_json(root / 'work/remote-job-status.json', {'status': 'EXITED', 'exit_code': code,
                'identity': _identity(15, 'a' * 40, 'b' * 64)})
    return audit


def test_handoff_transport_keeps_frozen_phases_without_parent_media(tmp_path, monkeypatch):
    remote, local, staging = (tmp_path / x for x in ('remote', 'local', 'staging'))
    staging.mkdir()
    _plan(remote)
    calls = _transport(monkeypatch, remote)
    monkeypatch.setattr('mas.engine.translation_workspace.prepare_id_translation_workspaces', lambda *a: None)
    for kind in ('tr', 'id'):
        _handoff(remote, kind)
        controller._collect_remote_results(WAIT_PART_RETURN, 15, local, '/episode', [], ['scp'],
                                           'host', Budget(), staging)
        assert progressive.validate_partial_handoff(local, 15)['kind'] == kind
    assert (local / 'parts/part-001/work/partial-handoff-tr.json').is_file()
    assert (local / 'parts/part-001/work/partial-handoff-id.json').is_file()
    assert not (local / 'source/movie.mp4').exists()
    assert not any(name.startswith('parts/part-002/') for name in calls)


def test_partial_export_transport_uses_plain_manifest_and_parent_vad(tmp_path, monkeypatch):
    remote, local, staging = (tmp_path / x for x in ('remote', 'local', 'staging'))
    staging.mkdir()
    _plan(remote)
    atomic_json(remote / 'work/part-vad.json', {'independent_vad': True})
    export = {'episode': 15, 'part_id': 'part-001', 'mode': 'strict-partial-subtitles',
              'files': [file_record(remote / 'work/part-plan.json', remote),
                        file_record(remote / 'work/part-vad.json', remote)]}
    atomic_json(remote / 'work/partial-export.json', export)
    _transport(monkeypatch, remote)
    checks = []
    def validate(root, episode, part_id, **kwargs):
        assert json.loads((root / 'parts/part-001/work/partial-export.json').read_text()) == export
        assert (root / 'work/part-vad.json').exists()
        checks.append(kwargs['total_timeout'])
    monkeypatch.setattr(partial_finalize, 'validate_partial_export', validate)
    controller._collect_remote_results(READY_FOR_PARTIAL_ENCODE, 15, local, '/episode', [], ['scp'],
                                       'host', Budget(), staging)
    assert checks == [600]


def test_partial_export_transport_restores_caption_dependency_from_signed_plan(tmp_path, monkeypatch):
    remote, local, staging = (tmp_path / x for x in ('remote', 'local', 'staging'))
    staging.mkdir()
    caption = remote / 'source/youtube.tr.vtt'
    caption.parent.mkdir(parents=True)
    caption.write_text('WEBVTT\n', encoding='utf-8')
    _plan(remote, captions={**file_record(caption, remote), 'records': []})
    atomic_json(remote / 'work/part-vad.json', {'independent_vad': True})
    derived_caption = remote / 'parts/part-001/work/captions.vtt'
    derived_caption.parent.mkdir(parents=True)
    derived_caption.write_text('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nMerhaba\n', encoding='utf-8')
    lineage_path = remote / 'parts/part-001/work/audio-part.done.json'
    _bound(lineage_path, {'captions': file_record(derived_caption, remote)})
    export = {'episode': 15, 'part_id': 'part-001', 'mode': 'strict-partial-subtitles',
              'files': [file_record(remote / 'work/part-plan.json', remote),
                        file_record(remote / 'work/part-vad.json', remote), file_record(lineage_path, remote)]}
    atomic_json(remote / 'work/partial-export.json', export)
    calls = _transport(monkeypatch, remote)
    monkeypatch.setattr(partial_finalize, 'validate_partial_export', lambda *a, **k: None)
    controller._collect_remote_results(READY_FOR_PARTIAL_ENCODE, 15, local, '/episode', [], ['scp'],
                                       'host', Budget(), staging)
    assert (local / 'source/youtube.tr.vtt').read_bytes() == caption.read_bytes()
    assert (local / 'parts/part-001/work/captions.vtt').read_bytes() == derived_caption.read_bytes()
    assert calls[-2:] == ['source/youtube.tr.vtt', 'parts/part-001/work/captions.vtt']


def test_partial_transfer_rejects_crosspart_inventory(tmp_path, monkeypatch):
    remote, local, staging = (tmp_path / x for x in ('remote', 'local', 'staging'))
    staging.mkdir()
    _plan(remote)
    data = _handoff(remote)
    data['files'][0]['relative_path'] = 'parts/part-002/work/evidence.json'
    _bound(remote / 'work/partial-handoff.json', data)
    _transport(monkeypatch, remote)
    with pytest.raises(controller.RunPodControllerError, match='escaped'):
        controller._collect_remote_results(WAIT_PART_RETURN, 15, local, '/episode', [], ['scp'],
                                           'host', Budget(), staging)


def test_missing_return_recovers_pause_after_verified_shutdown_without_gpu(tmp_path, monkeypatch):
    _plan(tmp_path)
    handoff = _handoff(tmp_path)
    _release(tmp_path, WAIT_PART_RETURN)
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'episode_dir', lambda _: tmp_path)
    monkeypatch.setattr(controller, '_required_environment', lambda: pytest.fail('GPU environment requested'))
    assert controller.run_remote_episode(15) == WAIT_PART_RETURN
    budget = partial_delivery._read_bound(tmp_path / 'work/controller_budget.json')
    assert budget['wait']['reason'] == WAIT_PART_RETURN
    assert budget['wait']['evidence_sha256'] == sha256_file(tmp_path / 'work/partial-handoff.json')
    assert not (tmp_path / handoff['expected_return']).exists()
    assert controller.run_remote_episode(15) == WAIT_PART_RETURN


def test_expired_part_wait_cannot_reset_six_hour_clock(tmp_path, monkeypatch):
    _plan(tmp_path)
    _handoff(tmp_path)
    saved = partial_delivery._read_bound(tmp_path / 'work/controller_budget.json')
    saved['started_at'] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    _bound(tmp_path / 'work/controller_budget.json', saved)
    monkeypatch.setattr(controller, '_required_environment', lambda: pytest.fail('GPU requested'))
    with pytest.raises(BudgetExceeded):
        controller._resume_partial_delivery(tmp_path, 15)


def _current_part(root, part_id='part-001', stage='forced_alignment'):
    lineage = root / f'parts/{part_id}/work/audio-part.done.json'
    atomic_json(lineage, {'fixture': 'derived audio'})
    hashes = {'part_plan_sha256': sha256_file(root / 'work/part-plan.json'),
              'part_audio_lineage_sha256': sha256_file(lineage)}
    state_path = root / f'parts/{part_id}/work/state.json'
    atomic_json(state_path, {'episode': 15, 'part_id': part_id, **hashes,
                            'stages': {stage: {'status': 'failed'}}})
    data = {'format': 'mas-current-part-1', 'episode': 15, 'part_id': part_id, 'stage': stage,
            'state': file_record(state_path, root), **hashes}
    _bound(root / 'work/current-part.json', data)
    return data


def test_partial_failure_identity_is_exact_current_part_not_old_conflict(tmp_path):
    _plan(tmp_path)
    current = _current_part(tmp_path)
    result = controller._partial_failure_identity(tmp_path, 15)
    assert result == {'part_id': 'part-001', 'part_stage': 'forced_alignment',
                      **{key: current[key] for key in ('part_plan_sha256', 'part_audio_lineage_sha256')}}
    _current_part(tmp_path, 'part-002', 'id_return')
    assert controller._partial_failure_identity(tmp_path, 15) == {'part_id': 'part-002', 'part_stage': 'id_return'}


def test_partial_failure_rejects_changed_state_with_rehashed_current_pointer(tmp_path):
    _plan(tmp_path)
    current = _current_part(tmp_path)
    state_path = tmp_path / current['state']['relative_path']
    state = json.loads(state_path.read_text())
    state['part_audio_lineage_sha256'] = 'f' * 64
    atomic_json(state_path, state)
    current['state'] = file_record(state_path, tmp_path)
    _bound(tmp_path / 'work/current-part.json', current)
    with pytest.raises(controller.RunPodControllerError, match='audio evidence changed'):
        controller._partial_failure_identity(tmp_path, 15)


def test_new_job_failure_cannot_adopt_previous_job_alignment_marker(tmp_path):
    _plan(tmp_path)
    current = _current_part(tmp_path)
    _release(tmp_path, 1)
    assert controller._partial_failure_identity(tmp_path, 15) == {}
    current['remote_job_token'] = _identity(15, 'a' * 40, 'b' * 64)['token']
    _bound(tmp_path / 'work/current-part.json', current)
    assert controller._partial_failure_identity(tmp_path, 15)['part_stage'] == 'forced_alignment'


def test_partial_diagnostics_only_snapshot_active_part(tmp_path):
    root = tmp_path / 'EPISODES/Muhtemel Ask 15.Bolum'
    _plan(root)
    _current_part(root)
    for part_id in ('part-001', 'part-002'):
        atomic_json(root / f'parts/{part_id}/work/forced_alignment_units/resume-identity.json', {'part': part_id})
        atomic_json(root / f'parts/{part_id}/work/raw_asr_v2.json', {'fixture': part_id})
    manifest = checkpoint_manifest(tmp_path, 15, 'a' * 40, diagnostics=True)
    relatives = {record['relative_path'] for record in manifest['files']}
    assert manifest['part_id'] == 'part-001'
    assert 'work/current-part.json' in relatives
    assert 'parts/part-001/work/forced_alignment_units/resume-identity.json' in relatives
    assert 'parts/part-001/work/raw_asr_v2.json' in relatives
    assert not any(relative.startswith('parts/part-002/') for relative in relatives)


def test_expired_partial_worker_is_left_to_existing_lease_cleanup(tmp_path):
    _plan(tmp_path)
    saved = partial_delivery._read_bound(tmp_path / 'work/controller_budget.json')
    saved['started_at'] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    _bound(tmp_path / 'work/controller_budget.json', saved)
    _bound(tmp_path / 'work/controller-lease.json', {'audit': 'work/capacity/run'})
    _bound(tmp_path / 'work/capacity/run/capacity-state.json', {'status': 'RUNNING'})
    assert controller._resume_partial_delivery(tmp_path, 15) is None


def test_whole_episode_mode_ignores_retained_partial_delivery(tmp_path, monkeypatch):
    (tmp_path / 'work').mkdir()
    (tmp_path / 'work/part-plan.json').write_text('not a reusable partial plan', encoding='utf-8')
    (tmp_path / 'work/partial-handoff.json').write_text('not a reusable handoff', encoding='utf-8')
    monkeypatch.setenv('MAS_PRODUCTION_PRIORITY', 'whole-episode-v1')
    assert controller._resume_partial_delivery(tmp_path, 15) is None
    assert controller._preflight_part_return(tmp_path, 15) is None
    assert controller._part_resume_files(tmp_path, 15) == []


def test_complete_part_advances_loop_but_wait_does_not(tmp_path, monkeypatch):
    calls = []
    def once(*args):
        calls.append(args)
        return NEXT_PART if len(calls) == 1 else WAIT_PART_RETURN
    monkeypatch.setattr(controller, '_run_remote_episode_once', once)
    assert controller.run_remote_episode(15) == WAIT_PART_RETURN
    assert len(calls) == 2


def test_worker_ack_only_completion_cannot_start_another_gpu_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, '_network_retry', lambda *a, **k: pytest.fail('unexpected transfer'))
    with pytest.raises(controller.RunPodControllerError, match='no-progress GPU continuation'):
        controller._collect_remote_results(NEXT_PART, 15, tmp_path, '/episode', [], [], 'host', Budget(), tmp_path)


def test_partial_collection_retry_matches_plain_export(tmp_path):
    relative = 'parts/part-001/output/strict.srt'
    atomic_json(tmp_path / 'work/partial-export.json', {'episode': 15, 'mode': 'strict-partial-subtitles',
                                                     'files': [{'relative_path': relative}]})
    failure = {'exit_code': READY_FOR_PARTIAL_ENCODE, 'relative_path': relative, 'storage_path': None}
    assert controller._collection_failure_paths_match(failure, {'episode': 15}, tmp_path / 'work/remote-job-status.json')


def test_part_review_uploads_keep_namespace_and_do_not_authorize_unvalidated_retry(tmp_path):
    _plan(tmp_path)
    before, evidence_before = controller._remote_input_binding(tmp_path, 'source', 'a' * 40, 15)
    for filename in ('audio_review_overrides.json', 'speaker_evidence_v1.json'):
        atomic_json(tmp_path / 'parts/part-001/work' / filename, {'fixture': 'unvalidated'})
    paths = controller._part_resume_files(tmp_path, 15)
    assert {path.relative_to(tmp_path).as_posix() for path in paths} == {
        'parts/part-001/work/audio_review_overrides.json', 'parts/part-001/work/speaker_evidence_v1.json'}
    after, evidence_after = controller._remote_input_binding(tmp_path, 'source', 'a' * 40, 15)
    assert before != after
    assert evidence_before == evidence_after


def test_runtime_priority_is_first_hour_on_qsv_and_explicit_whole_on_nvenc(monkeypatch):
    monkeypatch.delenv('MAS_PRODUCTION_PRIORITY', raising=False)
    monkeypatch.delenv('MAS_DELIVERY_EXECUTION_PLAN', raising=False)
    assert controller._production_priority() == 'first-hour-v1'
    monkeypatch.setenv('MAS_DELIVERY_EXECUTION_PLAN', 'remote-nvenc-v1')
    assert controller._production_priority() == 'whole-episode-v1'
    monkeypatch.setenv('MAS_PRODUCTION_PRIORITY', 'first-hour-v1')
    with pytest.raises(controller.RunPodControllerError, match='unsupported'):
        controller._production_priority()


@pytest.mark.parametrize('valid_budget', [True, False])
def test_first_hour_dispatch_precedes_full_episode_asr(tmp_path, monkeypatch, valid_budget):
    monkeypatch.setattr(pipeline, 'episode_dir', lambda _: tmp_path)
    monkeypatch.setattr(pipeline, '_load_configs', lambda: (tmp_path, {'whisper_model': 'model'}, {}, {}))
    monkeypatch.setattr(pipeline, '_guard_existing_source', lambda *a: None)
    monkeypatch.setattr(pipeline, '_resolve_source_url', lambda *a: 'https://example.invalid/15')
    monkeypatch.setenv('MAS_EXTERNAL_RUNPOD_CONTROLLER', '1')
    monkeypatch.setenv('MAS_DELIVERY_EXECUTION_PLAN', 'local-qsv-v1')
    monkeypatch.setenv('MAS_PRODUCTION_PRIORITY', 'first-hour-v1')
    monkeypatch.setenv('MAS_ALIGNMENT_POLICY', 'strict-ctc-v1')
    monkeypatch.delenv('MAS_CODE_FIX_RESUME', raising=False)
    stages = []
    def stage(path, state, name, action):
        stages.append(name)
        return action()
    monkeypatch.setattr(pipeline, '_stage', stage)
    def download(url, folder, **kwargs):
        video = folder / 'source.mkv'
        video.write_bytes(b'fixture source')
        return SimpleNamespace(video_path=video, resumed=False, captions_path=None)
    def extract(video, folder, **kwargs):
        audio = folder / 'audio.flac'
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b'fixture audio')
        return SimpleNamespace(audio_path=audio, resumed=False)
    monkeypatch.setattr(pipeline, 'download_source', download)
    monkeypatch.setattr(pipeline, 'extract_audio', extract)
    monkeypatch.setattr(pipeline, 'transcribe_raw_audio', lambda *a, **k: pytest.fail('full episode ASR started'))
    calls = []
    monkeypatch.setattr(progressive, 'run_progressive_worker', lambda *a, **k: calls.append((a, k)) or WAIT_PART_RETURN)
    controller._episode_budget(tmp_path, 15)
    if not valid_budget:
        atomic_json(tmp_path / 'work/controller_budget.json', {'data': {'episode': 15, 'limit_seconds': 21601}})
        with pytest.raises(RuntimeError, match='original bounded episode clock'):
            pipeline.run(15)
        assert not calls
    else:
        assert pipeline.run(15) == WAIT_PART_RETURN
        assert len(calls) == 1
        assert 0 < calls[0][1]['total_timeout'] <= 21600
        assert calls[0][0][2].name == 'source.mkv'
    assert stages == ['download', 'audio']


def test_whole_episode_never_reads_stale_partial_failure_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv('MAS_PRODUCTION_PRIORITY', 'whole-episode-v1')
    path = tmp_path / 'work/current-part.json'
    path.parent.mkdir(parents=True)
    path.write_text('intentionally invalid stale partial JSON')
    assert controller._partial_failure_identity(tmp_path, 15) == {}
