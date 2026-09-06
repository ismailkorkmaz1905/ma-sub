import json

import pytest

from mas.reliability import IntegrityError, digest
from mas.subtitle.episode_asr_worker import consume


def test_interruption_retains_bound_raw_segments(tmp_path):
    segment = {'text': ' Merhaba.', 'start': 1.0, 'end': 2.0, 'words': []}
    def interrupted():
        yield segment
        raise RuntimeError('interrupted')
    request = {'output': str(tmp_path), 'maximum_seconds': 60}
    binding = {'source': 'fixed'}
    with pytest.raises(RuntimeError, match='interrupted'):
        consume(request, interrupted(), binding)
    raw = json.loads((tmp_path / 'segment-000000.json').read_text())
    assert raw['sha256'] == digest(raw['data'])
    assert raw['data']['segment'] == segment
    assert raw['data']['binding_sha256'] == digest(binding)
    assert not (tmp_path / 'result.json').exists()
    with pytest.raises(IntegrityError, match='preserve prior'):
        consume(request, [segment], binding)


def test_worker_checks_deadline_before_writing(tmp_path):
    clock = iter([0, 61])
    with pytest.raises(TimeoutError):
        consume({'output': str(tmp_path), 'maximum_seconds': 60}, [{'end': 1}], {},
                clock=lambda: next(clock))
    assert not list(tmp_path.glob('segment-*.json'))
