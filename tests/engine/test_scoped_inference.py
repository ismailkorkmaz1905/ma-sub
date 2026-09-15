import pytest

from mas.engine import raw_asr
from mas.engine.transcribe import TranscriptionError


@pytest.mark.parametrize('force', [True, False])
def test_scoped_alignment_cannot_start_raw_inference(tmp_path, monkeypatch, force):
    audio = tmp_path / 'synthetic.wav'
    audio.write_bytes(b'synthetic-not-opened')
    monkeypatch.setattr(raw_asr, 'load_valid_raw_asr_v2', lambda *a, **k: None)
    monkeypatch.setattr(raw_asr, 'recover_raw_asr_v2_from_checkpoint',
                        lambda *a, **k: pytest.fail('recovery inference was attempted'))
    with pytest.raises(TranscriptionError, match='Scoped alignment retry'):
        raw_asr.transcribe_raw_audio_v2(audio, tmp_path / 'prepare', episode=14,
                                       require_resume=True, force=force)
