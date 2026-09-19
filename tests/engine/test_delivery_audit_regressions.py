import copy
import json
from pathlib import Path
import zipfile

import pytest

from mas import delivery_first as df, partial_delivery as pd, runpod_controller as rc
from mas.engine import delivery_align as da, delivery_coverage as dc, id_translation as it
from mas.engine import partial_finalize as pf, partial_encode as pe, part_scope as ps
from mas.reliability import atomic_json, digest, file_digest as sha256_file
from test_part_audio import synthetic_parents


@pytest.fixture(autouse=True)
def test_key(monkeypatch):
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', 'a' * 64)


@pytest.fixture
def prepared(synthetic_parents, monkeypatch):
    root, source, audio, captions, calls = synthetic_parents
    from mas.pipeline import _load_configs
    _, series, names, religious = _load_configs()
    atomic_json(root / 'source/official-source.json', {
        'episode': 14, 'title': 'Muhtemel Ask 14. Bolum', 'url': 'https://example.invalid/14',
        'channel_url': pd.CHANNEL_VIDEOS_URL})
    (root / 'source/source.url').write_text('https://example.invalid/14', encoding='utf-8')
    runtime = {'model': {'sha256': 'model-fixture'}, 'producer': {'sha256': 'decoder-fixture'}}
    monkeypatch.setattr(dc, 'model_binding', lambda _series: copy.deepcopy(runtime))
    decoded = {'identity': copy.deepcopy(runtime), 'segments': [{'start_ms': 500, 'end_ms': 1000, 'text': 'Merhaba', 'words': []}]}
    # Only the model boundary is replaced. Real plan, PCM, pack, HMAC and export validators run.
    monkeypatch.setattr(df, 'primary_transcript', lambda *a, **kw: copy.deepcopy(decoded))
    return root, dict(root=root, episode=14, source_video=source, audio_path=audio,
        captions_path=captions, total_timeout=1200, config_dir=None,
        series=series, names=names, religious=religious), decoded, runtime


def _ready(prepared):
    root, options, _, _ = prepared
    assert df.run_worker(**options) == 25
    child = root / 'parts/part-001'
    schema = df.read_signed(child / 'prepare/delivery-schema.json', 'schema')['schema']
    pack = child / 'translation_input/Muhtemel Ask 14.Bolum_ID_TRANSLATION_PACK.zip'
    out = child / 'translation_output/Muhtemel Ask 14.Bolum_ID_TRANSLATED.zip'
    records = [{**r, 'id_final': 'Halo.'} for r in it.build_id_translation_records(schema)]
    it.create_id_translation_output_zip(schema, records, out, input_manifest=it.validate_id_translation_pack(pack))
    assert df.run_worker(**options) == 26
    return root, child, pf.validate_partial_export(root, 14, 'part-001')[0]


def test_no_dialogue_scope_reaches_authenticated_export_without_translation_handoff(prepared):
    root, options, decoded, _ = prepared
    decoded['segments'] = []
    assert df.run_worker(**options) == 26
    export, report = pf.validate_partial_export(root, 14, 'part-001')
    assert report['block_count'] == 0 and report['quality_status'] == 'NOT_STRICT'
    assert any(w['reason'] == 'speech_gap_unresolved' for w in report['warnings'])
    assert not (root / 'work/partial-handoff.json').exists()
    assert (root / report['outputs']['id_srt']['relative_path']).read_text().strip() == ''
    assert df.run_worker(**options) == 26
    assert pf.validate_partial_export(root, 14, 'part-001')[0] == export
    with pytest.raises(it.IDTranslationError):
        it.validate_aligned_turkish_schema({'schema_version': '2.0', 'episode': 12, 'blocks': []})


def test_changed_actual_model_rejects_frozen_schema_before_another_model_call(prepared, monkeypatch):
    root, options, _, runtime = prepared
    assert df.run_worker(**options) == 25
    runtime['model']['sha256'] = 'changed-weight-files'
    monkeypatch.setattr(df, 'primary_transcript', lambda *a, **kw: pytest.fail('repeated ASR'))
    with pytest.raises(ValueError, match='schema identity'):
        df.run_worker(**options)


def test_real_ctc_interval_is_not_clipped_back_to_inaccurate_coarse_bounds():
    cue = {'uid': 'one', 'text': 'Merhaba', 'start_ms': 1000, 'end_ms': 2000}
    decoded = {'word_segments': [{'word': 'Merhaba', 'start': .9, 'end': 2.1, 'score': .9}]}
    assert da._interval(decoded, cue) == (900, 2100)
    decoded['word_segments'][0]['start'] = .4
    with pytest.raises(ValueError, match='timing'):
        da._interval(decoded, cue)


def test_gap_plan_catches_vad_hole_and_joint_vad_asr_miss_with_caption_hint():
    cues = [{'start_ms': 1000, 'end_ms': 2000}]
    plan = dc.plan_gaps(cues, [{'start_ms': 1000, 'end_ms': 3000}],
        [{'start_ms': 7000, 'end_ms': 8000, 'text': 'Never copy this caption'}], [], 9000)
    assert plan == [{'start_ms': 2000, 'end_ms': 3000}, {'start_ms': 7000, 'end_ms': 8000}]
    plan = dc.plan_gaps([], [], [], [{'start_ms': 0, 'end_ms': 30001, 'text': 'incomplete'}], 30001)
    assert max(x['end_ms'] - x['start_ms'] for x in plan) <= 10000
    assert sum(x['end_ms'] - x['start_ms'] for x in plan) == 30001


class RescueSession:
    calls = []
    def __init__(self, *a, **kw):
        pass
    def align(self, target, seconds):
        self.calls.append(target['uid'])
        return {'uid': target['uid'], 'status': 'rescued', 'segments': [{
            'start_ms': target['start_ms'], 'end_ms': target['end_ms'], 'text': 'Gerçek ses sonucu', 'words': []}]}
    def close(self):
        pass


def test_recovery_uses_owned_audio_result_and_reuses_completed_jobs(tmp_path):
    child = tmp_path / 'parts/part-001'
    binding = {'episode': 14, 'plan_sha256': 'p', 'duration_ms': 4000,
               'primary_producer': {'model': {'sha256': 'm'}}}
    cues = [{'uid': 'primary', 'start_ms': 0, 'end_ms': 1000, 'text': 'Korunan metin'}]
    RescueSession.calls = []
    options = dict(remaining=lambda: 4000, model_name='fixture', session_factory=RescueSession)
    args = (cues, [], [{'start_ms': 2000, 'end_ms': 3000}], None, 'not opened', child, binding)
    first, notes = dc.recover_gaps(*args, **options)
    again, _ = dc.recover_gaps(*args, **options)
    assert first == again and len(RescueSession.calls) == 1
    assert first[0] == cues[0] and first[1]['start_ms'] == 2000 and first[1]['end_ms'] == 3000
    assert any(n['reason'] == 'speech_gap_recovered' for n in notes)
    budget = df.read_signed(tmp_path / 'work/delivery-rescue-budget.json', 'rescue-budget')
    assert budget['jobs'] == 1
    saved_path = next((child / 'work/delivery-rescue').glob('*.json'))
    envelope = json.loads(saved_path.read_text()); envelope['data']['result']['segments'][0]['text'] = 'changed'
    atomic_json(saved_path, envelope)
    with pytest.raises(ValueError, match='authentication'):
        dc.recover_gaps(*args, **options)


def test_recovery_deadline_and_ordinary_job_cap_do_not_block_delivery(tmp_path):
    child = tmp_path / 'parts/part-001'
    binding = {'episode': 14, 'plan_sha256': 'p', 'duration_ms': 4000, 'primary_producer': {'model': {}}}
    def forbidden(*a, **kw):
        pytest.fail('spent reserve or bypassed job cap')
    args = ([], [], [{'start_ms': 0, 'end_ms': 3000}], None, 'not opened', child, binding)
    result, notes = dc.recover_gaps(*args, remaining=lambda: 1799, model_name='fixture', session_factory=forbidden)
    assert result == [] and notes[0]['reason'] == 'speech_gap_unresolved'
    # Another part has no result cache, but the same persistent episode work allowance.
    df.write_signed(tmp_path / 'work/delivery-rescue-budget.json', {
        'episode': 14, 'plan_sha256': 'p', 'limit_seconds': dc.TOTAL_SECONDS,
        'max_jobs': dc.MAX_JOBS, 'jobs': 192, 'spent_seconds': 0.0}, 'rescue-budget')
    args = (*args[:5], tmp_path / 'parts/part-002', binding)
    result, _ = dc.recover_gaps(*args, remaining=lambda: 4000, model_name='fixture', session_factory=forbidden)
    assert result == []


def test_caption_hint_cannot_become_unheard_dialogue(tmp_path):
    target = {'uid': 'target', 'start_ms': 2000, 'end_ms': 3000}
    assert dc._valid_rescue({'status': 'unresolved', 'text': 'caption'}, target) == []
    assert dc._valid_rescue({'status': 'rescued', 'segments': [{
        'start_ms': 20000, 'end_ms': 21000, 'text': 'foreign scene'}]}, target) == []


def test_rehashed_controller_ack_cannot_skip_first_part(prepared):
    root, child, export = _ready(prepared)
    delivery = {'format': 'mas-part-delivery-1', 'status': 'PASS_PARTIAL', 'episode': 14,
        'part_id': 'part-001', 'plan_sha256': sha256_file(root / 'work/part-plan.json'),
        'export_sha256': sha256_file(child / 'work/partial-export.json'),
        'mp4': {'relative_path': 'parts/part-001/final/fake.mp4', 'size_bytes': 1, 'sha256': '0' * 64},
        'files': [{'remote': 'drive:x.part-001.id.mp4', 'bytes': 1, 'sha256': '0' * 64}]}
    ack = {'format': 'mas-controller-part-ack-1', 'episode': 14, 'part_id': 'part-001',
           'delivery': delivery, 'delivery_data_sha256': digest(delivery)}
    path = child / 'work/controller-delivery-ack.json'
    atomic_json(path, {'data': ack, 'sha256': digest(ack)})
    with pytest.raises(ValueError, match='authentication'):
        pd.validate_worker_published_part(root, 14, 'part-001')
    # The downstream path validates the real export before demanding controller-only evidence.
    assert pf.validate_partial_export(root, 14, 'part-001')[0] == export


def test_part_publication_and_encoding_reject_unsigned_self_asserted_receipts(prepared):
    root, child, _ = _ready(prepared)
    atomic_json(child / 'final/partial-encoding.json', {'format': 'mas-partial-burned-id-mp4-1'})
    with pytest.raises(ValueError, match='authentication'):
        pe.validate_partial_encoding(root, 14, 'part-001')
    value = {'format': 'mas-part-delivery-1', 'status': 'PASS_PARTIAL'}
    atomic_json(child / 'final/drive_readback_receipt.json', {'data': value, 'sha256': digest(value)})
    with pytest.raises(ValueError, match='authentication'):
        pd.validate_published_part(root, 14, 'part-001')


def test_real_delivery_export_authorizes_collection_retry_not_forged_export(prepared):
    root, child, export = _ready(prepared)
    record = next(r for r in export['files'] if r['relative_path'].endswith('delivery-schema.json'))
    failure = {'exit_code': rc.READY_FOR_PARTIAL_ENCODE, 'relative_path': record['relative_path'], 'storage_path': None}
    status = root / 'work/remote-job-status.json'
    assert rc._collection_failure_paths_match(failure, {'episode': 14}, status)
    export['files'] = []
    atomic_json(root / 'work/partial-export.json', export)
    assert not rc._collection_failure_paths_match(failure, {'episode': 14}, status)


def test_expired_compute_budget_permits_only_retained_local_delivery(prepared, monkeypatch):
    root, child, _ = _ready(prepared)
    budget = {'episode': 14, 'started_at': '2020-01-01T00:00:00+00:00', 'limit_seconds': 21600}
    path = root / 'work/controller_budget.json'
    atomic_json(path, {'data': budget, 'sha256': digest(budget)})
    original = path.read_bytes()
    # External release validation is exercised by complete_local_part; this checks controller routing.
    atomic_json(child / 'work/gpu-released-for-encode.json', {'retained': 'release'})
    seen = []
    def complete(*a, **kw):
        assert 0 < kw['total_timeout'] <= 3600
        seen.append('local-only')
        return rc.NEXT_PART
    monkeypatch.setattr(pd, 'complete_local_part', complete)
    assert rc._resume_partial_delivery(root, 14) == rc.NEXT_PART
    assert seen == ['local-only'] and path.read_bytes() == original
    assert json.loads((root / 'work/delivery-deadline-missed.json').read_text())['gpu_authorized'] is False
    with pytest.raises(rc.BudgetExceeded):
        rc.RunBudget(budget['started_at'], budget['limit_seconds']).check()


def test_encoder_workspace_preserves_media_on_post_encode_validation_failure(tmp_path):
    log = tmp_path / 'partial-encode.log'
    with pytest.raises(ValueError, match='quality'):
        with pe._encode_workspace(tmp_path, log) as work:
            (work / 'partial.mp4').write_bytes(b'already encoded media')
            raise ValueError('post-encode quality validation interrupted')
    assert log.with_suffix('.failed.mp4').read_bytes() == b'already encoded media'


def test_id_zip_size_guard_runs_before_crc_or_decompression(tmp_path, monkeypatch):
    file = tmp_path / 'large.zip'
    with zipfile.ZipFile(file, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('batch_001.jsonl', 'a' * 4096)
    monkeypatch.setattr(it, 'MAX_ID_ZIP_MEMBER_BYTES', 1024)
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda *a: pytest.fail('decompressed before limit'))
    with pytest.raises(it.IDTranslationError, match='resource limit'):
        it._open_checked_zip(file)


def test_continuous_speech_or_empty_vad_cannot_deadlock_first_hour(monkeypatch):
    monkeypatch.setattr(ps, 'TARGET_DURATION_MS', 2000)
    source = {'relative_path': 'source/video.mp4', 'sha256': '1' * 64, 'size_bytes': 1}
    audio = {'relative_path': 'prepare/audio.flac', 'sha256': '2' * 64, 'size_bytes': 1,
             'sample_rate_hz': 16000, 'channels': 1, 'sample_count': 80000}
    vad = {'independent_vad': True, 'audio_sha256': audio['sha256'], 'sample_count': 80000,
           'config': {'valid': True}, 'model': {'sha256': '3' * 64}, 'producer_sha256': '4' * 64,
           'regions': [{'vad_region_index': 1, 'start_ms': 0, 'end_ms': 5000, 'source': 'silero_vad'}]}
    plan = ps.build_part_plan(episode=14, source=source, audio=audio, vad=vad,
                             boundary_policy=ps.DELIVERY_BOUNDARY_POLICY)
    assert [p['end_sample'] for p in plan['parts']] == [32000, 64000, 80000]
    assert plan['parts'][0]['boundary_warning']
    assert sum(r['end_ms'] - r['start_ms'] for p in plan['parts']
               for r in ps.project_part_vad(plan, p['part_id'])) == 5000
    assert ps.validate_part_plan(plan) == plan
    with pytest.raises(ps.PartScopeError):
        ps.build_part_plan(episode=12, source=source, audio=audio, vad=vad)
    vad['regions'] = []
    empty = ps.build_part_plan(episode=14, source=source, audio=audio, vad=vad,
                              boundary_policy=ps.DELIVERY_BOUNDARY_POLICY)
    assert len(empty['parts']) == 3


def test_default_worker_recovers_detected_gap_then_freezes_real_id_pack(prepared, monkeypatch):
    root, options, decoded, _ = prepared
    decoded['segments'] = []
    RescueSession.calls = []
    monkeypatch.setattr(dc, 'BoundedRescuer', RescueSession)
    def unavailable(*a, **kw):
        raise da.AlignmentUnavailable('fixture model unavailable')
    monkeypatch.setattr(da, 'BoundedAligner', unavailable)
    options = {**options, 'total_timeout': 4000}
    assert df.run_worker(**options) == 25
    child = root / 'parts/part-001'
    saved = df.read_signed(child / 'prepare/delivery-schema.json', 'schema')
    assert saved['schema']['blocks'][0]['tr_text'] == 'Gerçek ses sonucu'
    assert saved['schema']['blocks'][0]['start_ms'] == 500
    assert saved['schema']['blocks'][0]['end_ms'] == 1000
    assert len(RescueSession.calls) == 1
    assert any(w['reason'] == 'speech_gap_recovered' for w in saved['warnings'])
    # All real pack and export validators run after only the GPU boundary was substituted.
    pack = child / 'translation_input/Muhtemel Ask 14.Bolum_ID_TRANSLATION_PACK.zip'
    out = child / 'translation_output/Muhtemel Ask 14.Bolum_ID_TRANSLATED.zip'
    it.create_id_translation_output_zip(saved['schema'],
        [{**r, 'id_final': 'Hasil suara asli.'} for r in it.build_id_translation_records(saved['schema'])],
        out, input_manifest=it.validate_id_translation_pack(pack))
    assert df.run_worker(**options) == 26
    _, report = pf.validate_partial_export(root, 14, 'part-001')
    assert report['block_count'] == 1 and len(RescueSession.calls) == 1
    assert not (root / 'parts/part-002/prepare/delivery-primary.json').exists()


def test_changed_primary_identity_is_not_signed_as_current_result(prepared):
    root, options, decoded, _ = prepared
    decoded['identity']['model']['sha256'] = 'wrong actual producer'
    with pytest.raises(ValueError, match='producer changed'):
        df.run_worker(**options)
    assert not (root / 'parts/part-001/prepare/delivery-primary.json').exists()
