import hashlib
import json
from types import SimpleNamespace

import pytest

from mas import remote_job
from mas import runpod_controller as controller
from mas.reliability import atomic_json, digest


def test_verified_unchanged_handoff_skips_scp_but_reads_remote_hash(tmp_path, monkeypatch):
    source = tmp_path / "return.zip"
    source.write_bytes(b"same exact handoff")
    calls = []
    monkeypatch.setattr(controller, "_network_retry", lambda command, **kwargs: calls.append(command) or b"PRESENT")
    hashes = []
    expected = (source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(controller, "_remote_file_signature", lambda *a, **kw: hashes.append(a[1]) or expected)
    receipt = controller._upload_episode_file_verified(source, "/remote/return.zip", ssh=["ssh"],
        scp=["scp"], host="host", reuse_verified=True)
    assert receipt == {"bytes": expected[0], "sha256": expected[1]}
    assert len(calls) == 1 and calls[0][0] == "ssh"
    assert hashes == ["/remote/return.zip"]


def test_changed_handoff_uploads_and_requires_both_readbacks(tmp_path, monkeypatch):
    source = tmp_path / "return.zip"
    source.write_bytes(b"new exact handoff")
    calls = []
    expected = (source.stat().st_size, hashlib.sha256(source.read_bytes()).hexdigest())
    signatures = iter([(expected[0], "0" * 64), expected, expected])
    monkeypatch.setattr(controller, "_remote_file_signature", lambda *a, **kw: next(signatures))
    monkeypatch.setattr(controller, "_network_retry", lambda command, **kwargs: calls.append(command) or b"PRESENT")
    controller._upload_episode_file_verified(source, "/remote/return.zip", ssh=["ssh"],
        scp=["scp"], host="host", reuse_verified=True)
    assert sum(command[0] == "scp" for command in calls) == 1
    assert any("mv -f" in command[-1] for command in calls)


def _manifest_case(root):
    identity = remote_job._identity(14, "a" * 40, "b" * 64)
    request = {"episode": 14, "commit": identity["commit"], "input_sha256": identity["input_sha256"]}
    atomic_json(root / "work/remote-job-request.json", {"data": request, "sha256": digest(request)})
    payload = b"retained diagnostic"
    signature = hashlib.sha256(payload).hexdigest()
    relative = "prepare/audio_review_v2.json"
    remote_root = "/workspace/ma-sub/EPISODES/Muhtemel Ask 14.Bolum"
    record = {"relative_path": relative, "sha256": signature, "size_bytes": len(payload),
              "snapshot_path": remote_root + "/work/remote-jobs/snapshots/" + signature + ".json"}
    manifest = {"identity": identity, "kind": "diagnostics", "files": [record], "missing": [], "unstable": []}
    return remote_root, manifest, payload


@pytest.mark.parametrize("location", ["canonical", "retained"])
def test_diagnostic_manifest_reuses_exact_local_bytes(tmp_path, monkeypatch, location):
    remote_root, manifest, payload = _manifest_case(tmp_path)
    record = manifest["files"][0]
    path = (tmp_path / record["relative_path"] if location == "canonical" else
            tmp_path / "work/remote-checkpoints" / record["sha256"] / "audio_review_v2.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    calls = []
    def network(command, **kwargs):
        calls.append(command)
        assert command[0] == "ssh", "unchanged diagnostic was downloaded again"
        return json.dumps(manifest).encode()
    monkeypatch.setattr(controller, "_network_retry", network)
    controller._collect_diagnostics(14, tmp_path, remote_root, ["ssh"], ["scp"], "host", controller.time.monotonic() + 120)
    assert len(calls) == 1


def test_diagnostic_manifest_downloads_changed_hash_and_preserves_canonical(tmp_path, monkeypatch):
    remote_root, manifest, payload = _manifest_case(tmp_path)
    canonical = tmp_path / manifest["files"][0]["relative_path"]
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"old evidence")
    from pathlib import Path
    def network(command, **kwargs):
        if command[0] == "ssh":
            return json.dumps(manifest).encode()
        Path(command[-1]).write_bytes(payload)
        return b""
    monkeypatch.setattr(controller, "_network_retry", network)
    controller._collect_diagnostics(14, tmp_path, remote_root, ["ssh"], ["scp"], "host", controller.time.monotonic() + 120)
    assert canonical.read_bytes() == b"old evidence"
    retained = tmp_path / "work/remote-checkpoints" / manifest["files"][0]["sha256"] / canonical.name
    assert retained.read_bytes() == payload


def test_diagnostic_manifest_rejects_forged_path_before_download(tmp_path, monkeypatch):
    remote_root, manifest, _ = _manifest_case(tmp_path)
    manifest["files"][0]["relative_path"] = "../../credentials"
    monkeypatch.setattr(controller, "_network_retry", lambda *a, **kw: json.dumps(manifest).encode())
    with pytest.raises(controller.RunPodControllerError, match="path set"):
        controller._collect_diagnostics(14, tmp_path, remote_root, ["ssh"], ["scp"], "host", controller.time.monotonic() + 120)


def test_remote_diagnostic_manifest_snapshots_only_bounded_failure_set(tmp_path):
    root = tmp_path / "EPISODES/Muhtemel Ask 14.Bolum"
    diagnostic = root / "prepare/audio_review_v2.json"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_bytes(b"review")
    (diagnostic.parent / "unrelated.json").write_bytes(b"unrelated")
    manifest = remote_job.checkpoint_manifest(tmp_path, 14, "a" * 40, diagnostics=True)
    assert manifest["kind"] == "diagnostics"
    assert [record["relative_path"] for record in manifest["files"]] == ["prepare/audio_review_v2.json"]
    assert len(manifest["missing"]) == 9
    diagnostic.write_bytes(b"changed after snapshot")
    from pathlib import Path
    assert Path(manifest["files"][0]["snapshot_path"]).read_bytes() == b"review"
