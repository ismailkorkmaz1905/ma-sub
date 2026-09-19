import json

import pytest

from mas import local_encode
from mas import runpod_controller as controller
from mas.reliability import atomic_json, digest


class Budget:
    def check(self):
        return 600


def test_subtitle_ready_exit_collects_only_strict_subtitle_export(tmp_path, monkeypatch):
    export = {"episode": 14, "mode": "strict-subtitles", "files": [{"relative_path": "work/local-encode-plan.json"}]}
    def network(command, **kwargs):
        assert "subtitle-export.json" in command[-2]
        from pathlib import Path
        Path(command[-1]).write_text(json.dumps(export), encoding="utf-8")
    monkeypatch.setattr(controller, "_network_retry", network)
    records = []
    monkeypatch.setattr(controller, "_download_record", lambda record, *a, **kw: records.append(record))
    controller._collect_remote_results(24, 14, tmp_path, "/workspace/ma-sub/EPISODES/Muhtemel Ask 15.Bolum",
                                       ["ssh"], ["scp"], "host", Budget(), tmp_path)
    assert records == export["files"]


def test_encode_resume_never_requires_gpu_environment_or_acquires_provider(tmp_path, monkeypatch):
    release = {"pod_id": "owned", "capacity_state": {"relative_path": "work/capacity/run/capacity-state.json"}}
    atomic_json(tmp_path / "work/gpu-released-for-encode.json", {"data": release, "sha256": digest(release)})
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "episode_dir", lambda _: tmp_path)
    monkeypatch.setattr(controller, "_episode_budget", lambda *_: Budget())
    monkeypatch.setenv("MAS_DRIVE_STRICT_REMOTE", "gdrive:target")
    monkeypatch.setattr(controller, "_required_environment", lambda: pytest.fail("GPU environment required"))
    monkeypatch.setattr(controller, "CapacityProvider", lambda *_: pytest.fail("GPU acquired"))
    events = []
    monkeypatch.setattr(local_encode, "complete_local_encode", lambda *a, **kw: events.append("local-encode"))
    monkeypatch.setattr(controller, "validate_delivery", lambda *a: events.append("validate"))
    monkeypatch.setattr(controller, "_write_delivery_release", lambda *a: events.append("delivery-release"))
    monkeypatch.setattr(controller, "publish_local_delivery", lambda *a, **kw: events.append("publish") or 0)
    assert controller.run_remote_episode(15) == 0
    assert events == ["local-encode", "validate", "delivery-release", "publish"]


def test_encode_failure_preserves_resume_boundary_without_acquiring_gpu(tmp_path, monkeypatch):
    atomic_json(tmp_path / "work/gpu-released-for-encode.json", {})
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "episode_dir", lambda _: tmp_path)
    monkeypatch.setattr(controller, "_episode_budget", lambda *_: Budget())
    monkeypatch.setenv("MAS_DRIVE_STRICT_REMOTE", "gdrive:target")
    monkeypatch.setattr(controller, "CapacityProvider", lambda *_: pytest.fail("GPU acquired"))
    monkeypatch.setattr(local_encode, "complete_local_encode", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("QSV unavailable")))
    with pytest.raises(RuntimeError, match="QSV unavailable"):
        controller.run_remote_episode(15)
    assert (tmp_path / "work/gpu-released-for-encode.json").exists()


def test_capacity_release_evidence_is_written_even_on_failure(tmp_path, monkeypatch):
    audit = tmp_path / "work/capacity/run"
    state = {"owned_pod_ids": ["owned"]}
    atomic_json(audit / "capacity-state.json", {"data": state, "sha256": digest(state)})
    atomic_json(audit / "capacity-shutdown.json", {})
    events = []
    monkeypatch.setattr(controller, "_capacity_release_evidence", lambda *a: events.append("verified-absent"))
    with pytest.raises(RuntimeError):
        with controller._record_release_after_capacity(tmp_path, 15, audit):
            events.append("work")
            raise RuntimeError("failure")
    assert events == ["work", "verified-absent"]
