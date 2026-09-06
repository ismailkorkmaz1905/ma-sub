import pytest

from mas.reliability import IntegrityError
from mas.subtitle.episode_alignment_worker import model_identity


def test_ctc_identity_uses_ctc_files_and_detects_changed_weights(tmp_path):
    for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin',
                 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json'):
        (tmp_path / name).write_bytes(name.encode())
    original = model_identity(tmp_path)
    (tmp_path / 'pytorch_model.bin').write_bytes(b'changed')
    assert model_identity(tmp_path)['sha256'] != original['sha256']
    (tmp_path / 'vocab.json').unlink()
    with pytest.raises(IntegrityError):
        model_identity(tmp_path)
