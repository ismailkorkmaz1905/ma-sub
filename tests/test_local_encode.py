import json

import pytest

from mas import local_encode
from mas.engine.episode_archive import file_record
from mas.reliability import atomic_json, digest, file_digest as sha256_file


def subtitle_export(root):
    def record(name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return file_record(path, root)
    inputs = {'source_video': record('source/source.mp4', 'source')}
    outputs = {language + '_srt': record('final/' + language + '.srt',
               '1\n00:00:01,000 --> 00:00:02,000\nMerhaba.\n') for language in ('tr', 'id')}
    report = {'episode': 14, 'status': 'PASS', 'input_files': inputs, 'outputs': outputs}
    atomic_json(root / 'final/Muhtemel Ask 14.Bolum_FINALIZATION_REPORT_V2.json', report)
    record('source/official-source.json', '{}')
    record('source/source.url', 'https://example.invalid/source')
    local_encode.write_subtitle_export(root, 14)
    return report


def release(root):
    body = {'format': 'mas-gpu-released-for-encode-1', 'episode': 14, 'status': 'ABSENT',
            'plan_sha256': sha256_file(root / 'work/local-encode-plan.json'),
            'pod_id': 'owned', 'capacity_state': {}, 'capacity_shutdown': {}}
    atomic_json(root / 'work/gpu-released-for-encode.json', {'data': body, 'sha256': digest(body)})


def test_export_is_strict_and_independent_of_mutable_mail(tmp_path):
    subtitle_export(tmp_path)
    atomic_json(tmp_path / 'work/notification-outbox/event.json', {'status': 'sent'})
    assert local_encode.validate_subtitle_export(tmp_path, 14)[0]['encoder'] == 'h264_qsv'
    (tmp_path / 'final/id.srt').write_text('changed')
    with pytest.raises(ValueError):
        local_encode.validate_subtitle_export(tmp_path, 14)


def test_export_rejects_forged_equal_hash_inventory_with_changed_timeline(tmp_path):
    report = subtitle_export(tmp_path)
    path = tmp_path / 'final/id.srt'
    path.write_text('1\n00:00:00,500 --> 00:00:02,000\nHalo.\n')
    report['outputs']['id_srt'] = file_record(path, tmp_path)
    atomic_json(tmp_path / 'final/Muhtemel Ask 14.Bolum_FINALIZATION_REPORT_V2.json', report)
    local_encode.write_subtitle_export(tmp_path, 14)
    with pytest.raises(ValueError, match='timelines differ'):
        local_encode.validate_subtitle_export(tmp_path, 14)


def test_semantic_first_hour_export_selects_from_canonical_blocks(tmp_path):
    def record(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return file_record(path, tmp_path)

    inputs = {
        'source_video': record('source/source.mp4', 'source'),
        'final_blocks': record(
            'work/semantic_alignment/final_blocks.jsonl',
            json.dumps({'block_uid': 'one', 'block_index': 1, 'start_ms': 3_599_000,
                        'end_ms': 3_600_100}) + '\n'
            + json.dumps({'block_uid': 'two', 'block_index': 2, 'start_ms': 3_601_000,
                          'end_ms': 3_602_000}) + '\n'),
    }
    srt = ('1\n00:59:59,000 --> 01:00:00,100\nBir.\n\n'
           '2\n01:00:01,000 --> 01:00:02,000\nİki.\n')
    outputs = {
        'tr_srt': record('final/subtitles/episode-tr.srt', srt),
        'id_srt': record('final/subtitles/episode-id.srt', srt),
    }
    report = {
        'episode': 15, 'status': 'PASS', 'alignment_policy': 'semantic-block-v1',
        'strict_ctc_pass': False, 'semantic_alignment_pass': True,
        'release_eligible': True, 'input_files': inputs, 'outputs': outputs,
    }
    atomic_json(tmp_path / 'final/Muhtemel Ask 15.Bolum_SEMANTIC_FINALIZATION_REPORT.json', report)
    record('source/official-source.json', '{}')
    record('source/source.url', 'https://example.invalid/source')
    atomic_json(tmp_path / 'work/state.json', {
        'episode': 15,
        'run_contract': {'format': 'mas-run-contract-1', 'episode': 15,
                         'delivery_scope': 'first-hour',
                         'alignment_policy': 'semantic-block-v1'},
        'stages': {},
    })

    local_encode.write_subtitle_export(tmp_path, 15)
    plan, _ = local_encode.validate_subtitle_export(tmp_path, 15)

    assert plan['delivery_scope'] == 'first-hour'
    assert plan['duration_limit_seconds'] == 3600.1
    assert len(local_encode.parse_srt(tmp_path / plan['id_srt']['relative_path'])) == 1


def test_no_hardware_or_encoder_call_before_external_gpu_release(tmp_path, monkeypatch):
    subtitle_export(tmp_path)
    monkeypatch.setattr(local_encode, 'plan_encoding_settings', lambda *a, **k: pytest.fail('encoded before release'))
    with pytest.raises(FileNotFoundError):
        local_encode.complete_local_encode(tmp_path, 14, total_timeout=600)


def test_qsv_resume_uses_only_remaining_budget_and_emits_bound_delivery(tmp_path, monkeypatch):
    subtitle_export(tmp_path)
    release(tmp_path)
    from mas import runpod_controller
    events = []
    monkeypatch.setattr(runpod_controller, '_capacity_release_evidence', lambda *a: events.append('released'))
    monkeypatch.setattr(local_encode, 'plan_encoding_settings', lambda *a, **k: ({'test': True}, {}))
    def qualify(source, folder, **kwargs):
        assert kwargs['encoder'] == 'h264_qsv' and 0 < kwargs['timeout_seconds'] < 120
        events.append('qualified')
        path = folder / 'technical-qualification.json'
        atomic_json(path, {'test': True})
        return path
    def burn(source, subtitles, output, **kwargs):
        assert kwargs['encoder'] == 'h264_qsv'
        assert 0 < kwargs['idle_timeout_seconds'] <= kwargs['timeout_seconds'] < 120
        events.append('burned')
        output.write_bytes(b'mp4')
        atomic_json(output.with_suffix('.burn.json'), {'test': True})
    monkeypatch.setattr(local_encode, 'qualify_encoding', qualify)
    monkeypatch.setattr(local_encode, 'burn_indonesian_mp4', burn)
    result = local_encode.complete_local_encode(tmp_path, 14, total_timeout=120)
    assert events == ['released', 'qualified', 'burned']
    assert result['execution_plan_sha256'] == sha256_file(tmp_path / 'work/local-encode-plan.json')
    assert result['mode'] == 'strict'


def test_expired_budget_never_starts_qualification(tmp_path, monkeypatch):
    subtitle_export(tmp_path)
    release(tmp_path)
    from mas import runpod_controller
    monkeypatch.setattr(runpod_controller, '_capacity_release_evidence', lambda *a: None)
    monkeypatch.setattr(local_encode, 'plan_encoding_settings', lambda *a, **k: ({}, {}))
    monkeypatch.setattr(local_encode, 'qualify_encoding', lambda *a, **k: pytest.fail('late qualification'))
    with pytest.raises(TimeoutError, match='budget exhausted'):
        local_encode.complete_local_encode(tmp_path, 14, total_timeout=0)


def test_local_delivery_qualification_rejects_changed_samples(tmp_path):
    subtitle_export(tmp_path)
    plan, _ = local_encode.validate_subtitle_export(tmp_path, 14)
    folder = tmp_path / 'work/encoding-qualification/example'
    folder.mkdir(parents=True)
    samples = []
    for number in range(3):
        item = {}
        for kind in ('output', 'subtitle'):
            path = folder / f'{kind}-{number}.dat'
            path.write_bytes(b'sample')
            item.update({kind + '_file': path.name, kind + '_bytes': path.stat().st_size,
                         kind + '_sha256': sha256_file(path)})
        samples.append(item)
    atomic_json(folder / 'encoding-samples.json', {'samples': samples})
    (folder / 'technical-encoder-test.srt').write_bytes(b'technical')
    qualification = {'format': 'mas-technical-encoder-qualification-1', 'status': 'TECHNICALLY_VERIFIED',
                     'source_sha256': plan['source']['sha256'], 'settings_identity_sha256': 'settings',
                     'hardware': {'qsv_hardware_probe': 'PASS'}, 'samples': samples,
                     'sample_manifest_sha256': sha256_file(folder / 'encoding-samples.json'),
                     'technical_subtitle_sha256': sha256_file(folder / 'technical-encoder-test.srt')}
    path = folder / 'technical-qualification.json'
    atomic_json(path, qualification)
    delivery = {'execution_plan_sha256': sha256_file(tmp_path / 'work/local-encode-plan.json'),
                'qualification': file_record(path, tmp_path)}
    receipt = {'encoder': plan['encoder'], 'style': plan['style'],
               'encoding_settings': {'identity_sha256': 'settings', 'target_size_gb': plan['target_size_gb']}}
    local_encode.validate_local_encoding_evidence(tmp_path, 14, delivery, receipt)
    (folder / samples[0]['output_file']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        local_encode.validate_local_encoding_evidence(tmp_path, 14, delivery, receipt)
