import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas.engine import primary_checkpoint as checkpoint
from mas.engine.transcribe import TranscriptionError


def test_model_hash_binds_weights_and_tokenizer(tmp_path):
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"synthetic")
    first = checkpoint.model_identity(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"different")
    second = checkpoint.model_identity(tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"changed tokenizer")
    assert first != second != checkpoint.model_identity(tmp_path)


def test_incomplete_model_cannot_silently_fetch_unbound_tokenizer(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"synthetic")
    (tmp_path / "config.json").write_bytes(b"{}")
    with pytest.raises(TranscriptionError, match="incomplete"):
        checkpoint.model_identity(tmp_path)


def test_checkpoint_resume_and_tamper_detection(tmp_path):
    path = tmp_path / "primary.json"
    identity = {"model": "bound-test-model", "audio": "bound-test-audio"}
    checkpoint.save_primary(path, identity, "tr", [], [])
    assert checkpoint.load_primary(path, identity)["language"] == "tr"
    assert checkpoint.load_primary(path, {"model": "changed"}) is None
    saved = json.loads(path.read_text())
    saved["data"]["words"] = [{"text": "tampered"}]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(TranscriptionError, match="integrity"):
        checkpoint.load_primary(path, identity)


def test_runtime_binding_includes_versioned_native_libraries(tmp_path, monkeypatch):
    library = tmp_path / "libtest.so.1.2"
    library.write_bytes(b"native-one")
    dist = SimpleNamespace(version="test-1", files=[Path(library.name)],
                           locate_file=lambda item: tmp_path / item)
    monkeypatch.setattr(checkpoint.importlib.metadata, "distribution", lambda name: dist)
    first = checkpoint.producer_identity((test_runtime_binding_includes_versioned_native_libraries,))
    library.write_bytes(b"native-two")
    assert checkpoint.producer_identity((test_runtime_binding_includes_versioned_native_libraries,)) != first


def test_missing_runtime_binding_fails_closed(monkeypatch):
    monkeypatch.setattr(checkpoint.importlib.metadata, "distribution",
                        lambda name: SimpleNamespace(version="test", files=[]))
    with pytest.raises(TranscriptionError, match="cannot bind"):
        checkpoint.producer_identity(())
