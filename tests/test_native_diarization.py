import json
import wave

import pytest

from mas.reliability import IntegrityError, digest, file_digest
from mas.subtitle import native_diarization as module


def wav(path, seconds=1):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * (16000 * seconds))


def test_worker_preserves_probabilities_segments_and_hashes(tmp_path, monkeypatch):
    audio, model, runtime = tmp_path / "clip.wav", tmp_path / "model.gguf", tmp_path / "bin"
    wav(audio)
    model.write_bytes(b"model")
    runtime.mkdir()
    (runtime / "nemo_speech_asr_c.dll").write_bytes(b"dll")
    (runtime / "nemo-speech.exe").write_bytes(b"exe")
    monkeypatch.setattr(module, "_native_result",
                        lambda api, model, samples, rate: (0.08, [[0.8, 0.7, 0.0, 0.0]],
                                                           [{"start": 0.0, "end": 0.08, "speaker": 1},
                                                            {"start": 0.0, "end": 0.08, "speaker": 2}]))
    output = tmp_path / "result.json"
    request = {"audio": str(audio), "model": str(model), "runtime_dir": str(runtime),
               "output": str(output), "episode": 11, "clip_identity": "conflict-context",
               "input_sha256": file_digest(audio), "model_sha256": file_digest(model),
               "runtime": module._runtime_identity(runtime)}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    module.worker(request_path, api_factory=lambda path: object())
    wrapped = json.loads(output.read_text())
    assert wrapped["sha256"] == digest(wrapped["data"])
    assert wrapped["data"]["status"] == "REVIEW_REQUIRED"
    assert wrapped["data"]["probabilities"] == [[0.8, 0.7, 0.0, 0.0]]
    assert [item["speaker"] for item in wrapped["data"]["segments"]] == [1, 2]
    assert {item["name"] for item in wrapped["data"]["runtime"]} == {
        "nemo_speech_asr_c.dll", "nemo-speech.exe"}


def test_rejects_overlong_or_non_pcm16_input(tmp_path):
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0" * 32000)
    with pytest.raises(IntegrityError, match="mono.*PCM16"):
        module._read_wav(stereo)


def test_rejects_truncated_wav(tmp_path):
    path = tmp_path / "truncated.wav"
    wav(path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(IntegrityError, match="truncated"):
        module._read_wav(path)


def test_worker_rejects_mutated_input_before_native_call(tmp_path):
    audio, model, runtime = tmp_path / "clip.wav", tmp_path / "model.gguf", tmp_path / "bin"
    wav(audio)
    model.write_bytes(b"model")
    runtime.mkdir()
    request = {"audio": str(audio), "model": str(model), "runtime_dir": str(runtime),
               "output": str(tmp_path / "result.json"), "episode": 11,
               "clip_identity": "conflict-context", "input_sha256": "0" * 64,
               "model_sha256": file_digest(model), "runtime": []}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    with pytest.raises(IntegrityError, match="input or model changed"):
        module.worker(request_path, api_factory=lambda path: None)


def test_worker_rejects_invalid_native_probabilities(tmp_path, monkeypatch):
    audio, model, runtime = tmp_path / "clip.wav", tmp_path / "model.gguf", tmp_path / "bin"
    wav(audio)
    model.write_bytes(b"model")
    runtime.mkdir()
    (runtime / "native.dll").write_bytes(b"dll")
    monkeypatch.setattr(module, "_native_result",
                        lambda api, model, samples, rate: (0.08, [[float("nan"), 0, 0, 0]], []))
    request = {"audio": str(audio), "model": str(model), "runtime_dir": str(runtime),
               "output": str(tmp_path / "result.json"), "episode": 11,
               "clip_identity": "conflict-context", "input_sha256": file_digest(audio),
               "model_sha256": file_digest(model), "runtime": module._runtime_identity(runtime)}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    with pytest.raises(IntegrityError, match="frame probabilities"):
        module.worker(request_path, api_factory=lambda path: None)


def test_rejects_impossible_frame_count_from_native_boundary(tmp_path, monkeypatch):
    audio, model, runtime = tmp_path / "clip.wav", tmp_path / "model.gguf", tmp_path / "bin"
    wav(audio)
    model.write_bytes(b"model")
    runtime.mkdir()
    (runtime / "native.dll").write_bytes(b"dll")
    monkeypatch.setattr(module, "_native_result",
                        lambda api, model, samples, rate: (0.08, [[0, 0, 0, 0]] * 100, []))
    request = {"audio": str(audio), "model": str(model), "runtime_dir": str(runtime),
               "output": str(tmp_path / "result.json"), "episode": 11,
               "clip_identity": "conflict-context", "input_sha256": file_digest(audio),
               "model_sha256": file_digest(model), "runtime": module._runtime_identity(runtime)}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    with pytest.raises(IntegrityError, match="frame probabilities"):
        module.worker(request_path, api_factory=lambda path: None)


@pytest.mark.parametrize("timeout", [0, -1, 181, float("nan"), True])
def test_export_rejects_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        module.export_native(tmp_path / "audio", tmp_path / "model", tmp_path / "runtime",
                             tmp_path / "out", episode=11, clip_identity="clip", timeout=timeout)


def test_export_rejects_forged_output_binding(tmp_path, monkeypatch):
    audio, model, runtime = tmp_path / "clip.wav", tmp_path / "model.gguf", tmp_path / "bin"
    wav(audio)
    model.write_bytes(b"model")
    runtime.mkdir()
    (runtime / "native.dll").write_bytes(b"dll")
    output = tmp_path / "evidence.json"
    class Process:
        returncode = 0
        def __init__(self, *args, **kwargs): pass
        def wait(self, timeout):
            body = {"status": "REVIEW_REQUIRED", "production_acceptance": False,
                    "input_sha256": "0" * 64}
            output.write_text(json.dumps({"data": body, "sha256": digest(body)}))
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    with pytest.raises(IntegrityError, match="binding"):
        module.export_native(audio, model, runtime, output, episode=11, clip_identity="clip")


def test_export_cleans_process_on_wait_interrupt(tmp_path, monkeypatch):
    audio, model, runtime = tmp_path / "clip.wav", tmp_path / "model.gguf", tmp_path / "bin"
    wav(audio)
    model.write_bytes(b"model")
    runtime.mkdir()
    (runtime / "native.dll").write_bytes(b"dll")
    process = object()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(module, "_kill_process", lambda value: (_ for _ in ()).throw(RuntimeError("cleaned")))
    with pytest.raises(RuntimeError, match="cleaned"):
        module.export_native(audio, model, runtime, tmp_path / "out.json",
                             episode=11, clip_identity="clip")
