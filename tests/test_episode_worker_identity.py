import json

import pytest

from mas.reliability import IntegrityError, digest
from mas.subtitle import episode_asr_worker, episode_alignment_worker, episode_repair_worker


@pytest.mark.parametrize('worker', [episode_asr_worker, episode_alignment_worker, episode_repair_worker])
@pytest.mark.parametrize('episode', [None, True, 0, -1, '13'])
def test_invalid_episode_fails_before_gpu_or_output(tmp_path, worker, episode):
    output = tmp_path / 'output'
    request = {'episode': episode, 'maximum_seconds': 60, 'output': str(output)}
    if worker is episode_repair_worker:
        body = {'episode': episode, 'clips': []}
        manifest = tmp_path / 'manifest.json'
        manifest.write_text(json.dumps({'data': body, 'sha256': digest(body)}))
        request.update(manifest=str(manifest), manifest_sha256=digest(body))
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(request))
    with pytest.raises(IntegrityError, match='invalid Episode'):
        worker.run_worker(path)
    assert not output.exists()
