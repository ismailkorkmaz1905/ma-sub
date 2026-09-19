import json

import pytest

from mas import runpod_controller as controller
from mas.engine import burned_mp4
from mas.reliability import BudgetExceeded, digest


class Budget:
    def check(self):
        return 12


def test_qsv_preflight_is_bounded_and_records_only_synthetic_authority(tmp_path, monkeypatch):
    monkeypatch.delenv('MAS_DELIVERY_EXECUTION_PLAN', raising=False)
    monkeypatch.setattr(controller.shutil, 'which', lambda name: name)
    calls = []
    monkeypatch.setattr(burned_mp4, '_qsv_hardware',
                        lambda **kw: calls.append(kw) or {'qsv_hardware_probe': 'PASS'})
    controller._local_encoder_preflight(tmp_path, 14, Budget())
    assert calls == [{'timeout_seconds': 12}]
    envelope = json.loads((tmp_path / 'work/controller-local-encoder-preflight.json').read_text())
    assert envelope['sha256'] == digest(envelope['data'])
    assert envelope['data']['scope'] == 'SYNTHETIC_ENCODER_ONLY'
    assert envelope['data']['perceptual_acceptance'] == 'NOT_ASSERTED'


@pytest.mark.parametrize('missing', ['ffmpeg', 'ffprobe'])
def test_qsv_preflight_rejects_missing_tools_before_hardware_call(tmp_path, monkeypatch, missing):
    monkeypatch.delenv('MAS_DELIVERY_EXECUTION_PLAN', raising=False)
    monkeypatch.setattr(controller.shutil, 'which', lambda name: None if name == missing else name)
    monkeypatch.setattr(burned_mp4, '_qsv_hardware', lambda **kw: pytest.fail('hardware probe'))
    with pytest.raises(controller.RunPodControllerError, match=missing):
        controller._local_encoder_preflight(tmp_path, 14, Budget())


def test_remote_nvenc_does_not_require_local_qsv(tmp_path, monkeypatch):
    monkeypatch.setenv('MAS_DELIVERY_EXECUTION_PLAN', 'remote-nvenc-v1')
    monkeypatch.setattr(controller.shutil, 'which', lambda name: pytest.fail('local tool check'))
    controller._local_encoder_preflight(tmp_path, 14, Budget())


def test_unknown_execution_plan_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('MAS_DELIVERY_EXECUTION_PLAN', 'cpu-fallback')
    with pytest.raises(controller.RunPodControllerError, match='unsupported'):
        controller._local_encoder_preflight(tmp_path, 14, Budget())


def test_expired_probe_does_not_persist_pass(tmp_path, monkeypatch):
    monkeypatch.delenv('MAS_DELIVERY_EXECUTION_PLAN', raising=False)
    monkeypatch.setattr(controller.shutil, 'which', lambda name: name)
    class Expiring:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls > 1:
                raise BudgetExceeded('expired')
            return 1
    monkeypatch.setattr(burned_mp4, '_qsv_hardware', lambda **kw: {'qsv_hardware_probe': 'PASS'})
    with pytest.raises(BudgetExceeded):
        controller._local_encoder_preflight(tmp_path, 14, Expiring())
    assert not (tmp_path / 'work/controller-local-encoder-preflight.json').exists()
