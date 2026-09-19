"""Production raw-ASR entrypoint regressions; inference/VAD are the fake boundaries."""
import copy
import json
from pathlib import Path

import pytest

import test_raw_asr as fixtures
from mas.engine import raw_asr as raw


@pytest.fixture
def runtime_case(tmp_path, monkeypatch):
    helper = fixtures.RawASRV2RuntimeTests()
    monkeypatch.setenv('MAS_RAW_ASR_AUTH_KEY', '1' * 64)
    model_dir = tmp_path / 'synthetic-model'
    model_dir.mkdir()
    for name in ('model.bin', 'config.json', 'tokenizer.json'):
        (model_dir / name).write_bytes(b'synthetic-model')
    monkeypatch.setattr(raw.primary_checkpoint, 'resolve_model', lambda *_: model_dir)
    monkeypatch.setattr(raw.primary_checkpoint, 'producer_identity',
                        lambda *_: {'test_runtime': 'synthetic-1'})
    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            calls.append(path)
            return iter([fixtures._complete_segment()]), fixtures.Info()

    audio, prepare = helper._audio_and_prepare(str(tmp_path))
    monkeypatch.setattr(raw, '_import_whisper', lambda: (Model, fixtures.FakeCTranslate2))
    monkeypatch.setattr(raw, '_extract_vad_regions', lambda *args: helper._vad())
    yield audio, prepare, calls


def test_completed_producer_is_persisted_and_unchanged_resume_does_not_infer(runtime_case):
    audio, prepare, calls = runtime_case
    first = raw.transcribe_raw_audio_v2(audio, prepare, episode=14)
    assert first['producer_identity'] == raw._completed_raw_asr_producer_identity()
    before = (prepare / 'raw_asr_v2.json').read_bytes()
    second = raw.transcribe_raw_audio_v2(audio, prepare, episode=14, require_resume=True)
    assert second['resumed'] is True
    assert len(calls) == 1
    assert (prepare / 'raw_asr_v2.json').read_bytes() == before


@pytest.mark.parametrize('changed', ['model', 'producer'])
def test_completed_cache_rejects_changed_actual_producer_without_retranscribing(runtime_case, monkeypatch, changed):
    audio, prepare, calls = runtime_case
    raw.transcribe_raw_audio_v2(audio, prepare, episode=14)
    output = prepare / 'raw_asr_v2.json'
    before = output.read_bytes()
    if changed == 'model':
        model = raw.primary_checkpoint.resolve_model('large-v3')
        (model / 'model.bin').write_bytes(b'changed-actual-weights')
    else:
        monkeypatch.setattr(raw.primary_checkpoint, 'producer_identity', lambda *args: {'test_runtime': 'changed'})
    assert raw.load_valid_raw_asr_v2(prepare, audio_path=audio, episode=14) is None
    with pytest.raises(raw.TranscriptionError, match='producer/model identity changed'):
        raw.transcribe_raw_audio_v2(audio, prepare, episode=14)
    assert len(calls) == 1
    assert output.read_bytes() == before


def test_recovery_signature_covers_review_wav_record_before_reuse(runtime_case, monkeypatch):
    audio, prepare, calls = runtime_case
    original = raw.build_asr_hallucination_records
    def interrupt(*args, **kwargs):
        raise RuntimeError('interrupted after recovery commit')
    monkeypatch.setattr(raw, 'build_asr_hallucination_records', interrupt)
    with pytest.raises(RuntimeError, match='after recovery commit'):
        raw.transcribe_raw_audio_v2(audio, prepare, episode=14)
    checkpoint_path = prepare / raw.RAW_ASR_V2_RECOVERY_FILENAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
    assert checkpoint['recovery_auth_tag']
    # The primary, VAD and rescue receipts are untouched. Only the formerly
    # unauthenticated review-WAV metadata is replaced.
    checkpoint['speech_hole_records'] = [{'audio_sha256': '0' * 64, 'audio_size_bytes': 32044}]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding='utf-8')
    monkeypatch.setattr(raw, 'build_asr_hallucination_records', original)
    monkeypatch.setattr(raw, '_validate_persisted_review_audio',
                        lambda *args, **kwargs: pytest.fail('untrusted review WAV was used'))
    with pytest.raises(raw.TranscriptionError, match='Recovery checkpoint authentication failed'):
        raw.recover_raw_asr_v2_from_checkpoint(audio, prepare, episode=14)
    assert len(calls) == 1
    assert not (prepare / 'raw_asr_v2.json').exists()
    assert not (prepare / 'raw_asr_v2.done.json').exists()


def test_signed_recovery_resumes_without_model_inference(runtime_case, monkeypatch):
    audio, prepare, calls = runtime_case
    original = raw.build_asr_hallucination_records
    monkeypatch.setattr(raw, 'build_asr_hallucination_records',
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('after checkpoint')))
    with pytest.raises(RuntimeError, match='after checkpoint'):
        raw.transcribe_raw_audio_v2(audio, prepare, episode=14)
    monkeypatch.setattr(raw, 'build_asr_hallucination_records', original)
    monkeypatch.setattr(raw, '_import_whisper', lambda: pytest.fail('recovery launched inference'))
    result = raw.recover_raw_asr_v2_from_checkpoint(audio, prepare, episode=14)
    assert result['producer_identity'] == raw._completed_raw_asr_producer_identity()
    assert raw.load_valid_raw_asr_v2(prepare, audio_path=audio, episode=14) is not None
    assert len(calls) == 1


@pytest.mark.parametrize('field', ['episode', 'audio_sha256', 'speech_hole_records', 'producer_receipts'])
def test_recovery_auth_rejects_changes_and_deletion(field):
    body = {'episode': 14, 'audio_sha256': 'a' * 64,
            'speech_hole_records': [{'audio_sha256': 'b' * 64}], 'producer_receipts': {'primary': 'c'}}
    key = bytes.fromhex('1' * 64)
    signed = raw._sign_raw_asr_recovery(body, key)
    raw._verify_raw_asr_recovery_auth(signed, key)
    changed = copy.deepcopy(signed)
    changed.pop(field)
    with pytest.raises(raw.TranscriptionError, match='authentication failed'):
        raw._verify_raw_asr_recovery_auth(changed, key)
    with pytest.raises(raw.TranscriptionError, match='authentication failed'):
        raw._verify_raw_asr_recovery_auth(signed, bytes.fromhex('2' * 64))


def test_recovery_cannot_use_completed_artifact_tag():
    key = bytes.fromhex('1' * 64)
    signed = raw._sign_raw_asr_artifact({'episode': 14}, key)
    signed['recovery_auth_tag'] = signed.pop('raw_asr_auth_tag')
    with pytest.raises(raw.TranscriptionError, match='authentication failed'):
        raw._verify_raw_asr_recovery_auth(signed, key)
