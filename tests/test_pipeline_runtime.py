import json
import hashlib
import sys
from types import SimpleNamespace

import pytest

from mas import pipeline
from mas.remote import RemoteVerificationError, _run_watchdog, upload_verified
from mas.pipeline import _stage
from mas.runpod import RunPodShutdownError, stop_current_pod


def test_offline_fixture_interruption_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "episode_dir", lambda episode: tmp_path / str(episode))
    assert pipeline.run(13, fixture=True, stop_after=2) == 75
    state_path = tmp_path / "13" / "work" / "fixture-state.json"
    interrupted = json.loads(state_path.read_text(encoding="utf-8"))
    assert list(interrupted["stages"]) == ["fixture_source", "fixture_transform"]
    first_hashes = {name: value["sha256"] for name, value in interrupted["stages"].items()}
    assert pipeline.run(13, fixture=True) == 0
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert {name: resumed["stages"][name]["sha256"] for name in first_hashes} == first_hashes
    assert resumed["stages"]["fixture_verify"]["status"] == "pass"


@pytest.mark.parametrize("change", ["text", "vad", "code", "audio", "output", "missing_marker"])
def test_alignment_checkpoint_rejects_changed_bindings(tmp_path, monkeypatch, change):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"original audio")
    alignment = tmp_path / "alignment.json"
    inputs = [{"utterance_uid": "u1", "text": "Merhaba"}]
    vad = [{"start_ms": 0, "end_ms": 1000}]
    for name in ("src/mas/engine/forced_align.py", "src/mas/engine/speaker.py",
                 "src/mas/engine/workflow.py", "requirements.lock"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    calls = []

    def align(*args, **kwargs):
        calls.append(args)
        return {"audio_sha256": pipeline.sha256_file(audio), "provenance": {"device": "cuda"}}

    monkeypatch.setattr(pipeline, "align_corrected_segments", align)
    monkeypatch.setattr(pipeline, "validate_forced_alignment_data", lambda result: None)
    assert pipeline._aligned_checkpoint(alignment, audio, inputs, vad)[1] is False
    assert pipeline._aligned_checkpoint(alignment, audio, inputs, vad)[1] is True
    original = alignment.read_bytes()
    if change == "text":
        inputs[0]["text"] = "Selam"
    elif change == "vad":
        vad[0]["end_ms"] = 1100
    elif change == "code":
        (tmp_path / "src/mas/engine/forced_align.py").write_text("changed", encoding="utf-8")
    elif change == "audio":
        audio.write_bytes(b"different audio")
    elif change == "output":
        alignment.write_text("{}", encoding="utf-8")
        original = alignment.read_bytes()
    else:
        alignment.with_suffix(".binding.json").unlink()
    with pytest.raises(RuntimeError, match="binding is stale or missing"):
        pipeline._aligned_checkpoint(alignment, audio, inputs, vad)
    assert len(calls) == 1
    assert alignment.read_bytes() == original


def test_drive_upload_cannot_target_emergency(tmp_path):
    source = tmp_path / "strict.srt"
    source.write_text("strict", encoding="utf-8")
    with pytest.raises(RemoteVerificationError, match="strict destination"):
        upload_verified(source, "drive:emergency/strict.srt")


def test_stage_records_elapsed_seconds_on_success_and_failure(tmp_path, monkeypatch, capsys):
    state = {"episode": 13}
    values = iter((10.0, 12.5, 20.0, 23.25))
    monkeypatch.setattr("mas.pipeline.time.monotonic", lambda: next(values))
    monkeypatch.setattr("mas.pipeline.notify", lambda *args: None)
    path = tmp_path / "state.json"

    _stage(path, state, "ok", lambda: {"value": 1, "path": "audio.wav"})
    assert state["stages"]["ok"]["elapsed_seconds"] == 2.5
    assert state["stages"]["ok"]["path"] == "audio.wav"

    with pytest.raises(ValueError, match="broken"):
        _stage(path, state, "failed", lambda: (_ for _ in ()).throw(ValueError("broken")))
    assert state["stages"]["failed"]["elapsed_seconds"] == 3.25
    assert capsys.readouterr().out.splitlines() == [
        "[STAGE] ok: START",
        "[STAGE] ok: PASS elapsed=2.5s",
        "[STAGE] failed: START",
        "[STAGE] failed: FAIL elapsed=3.2s",
    ]


def test_pending_extra_audio_review_uids_are_collected_once(tmp_path, monkeypatch):
    input_pack = tmp_path / "input.zip"
    output_pack = tmp_path / "output.zip"
    input_pack.touch()
    output_pack.touch()
    monkeypatch.setattr(
        pipeline,
        "read_tr_correction_pack",
        lambda _path: SimpleNamespace(
            speech_holes=({"hole_uid": "hole"},),
            asr_hallucination_records=(
                {"utterance_uid": "candidate", "reason": "automatic"},
                {
                    "utterance_uid": "explicit",
                    "reason": "high_asr_no_speech_probability, explicit_uid_review_request",
                },
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_tr_correction_output",
        lambda *_paths: SimpleNamespace(
            records=(
                {"utterance_uid": "hole", "review_required": True, "asr_text": "a", "tr_corrected": "a"},
                {"utterance_uid": "candidate", "review_required": True, "asr_text": "b", "tr_corrected": "b"},
                {"utterance_uid": "extra-1", "review_required": True, "asr_text": "c", "tr_corrected": "c"},
                {"utterance_uid": "ignored", "review_required": False, "asr_text": "d", "tr_corrected": "d"},
                {"utterance_uid": "lexical-delete", "review_required": False, "asr_text": "lüks lüks", "tr_corrected": "lüks"},
                {"utterance_uid": "explicit", "review_required": True, "asr_text": "e", "tr_corrected": "e"},
                {"utterance_uid": "extra-2", "review_required": True, "asr_text": "f", "tr_corrected": "f"},
            )
        ),
    )

    assert pipeline._pending_extra_audio_review_uids(input_pack, output_pack) == (
        "extra-1",
        "lexical-delete",
        "explicit",
        "extra-2",
    )


def test_stage_heartbeat_does_not_mark_useful_work(tmp_path, monkeypatch, capsys):
    marks = []
    class Event:
        def __init__(self): self.waits = 0
        def wait(self, timeout):
            self.waits += 1
            return self.waits > 2
        def set(self): pass
    class Thread:
        def __init__(self, target, daemon): self.target = target
        def start(self): self.target()
        def join(self): pass
    monkeypatch.setattr(pipeline.threading, "Event", Event)
    monkeypatch.setattr(pipeline.threading, "Thread", Thread)
    monkeypatch.setattr(pipeline, "notify", lambda *a: None)
    monkeypatch.setattr(pipeline, "mark_work_progress", lambda stage, **kw: marks.append((stage, kw)))
    pipeline._stage(tmp_path / "state.json", {"episode": 11}, "audio", lambda: {})
    assert capsys.readouterr().out.count("RUNNING") == 2
    assert marks == [("audio", {}), ("audio", {"completed": True})]


def test_work_marker_and_shell_watchdog_are_separate_from_log(tmp_path, monkeypatch):
    from mas.progress import mark_work_progress
    marker = tmp_path / "progress.json"
    monkeypatch.setenv("MAS_PROGRESS_FILE", str(marker))
    mark_work_progress("raw_asr", completed=7)
    data = json.loads(marker.read_text())
    assert data["completed"] == 7
    shell = (pipeline.ROOT / "runpod/run-episode.sh").read_text()
    assert 'stat -c %Y "$MAS_PROGRESS_FILE"' in shell
    assert 'stat -c %Y "$LOG_PATH"' not in shell


def test_json_reader_retries_sharing_violation_but_not_invalid_json(tmp_path, monkeypatch):
    from mas.reliability import read_json
    import mas.reliability as reliability
    path = tmp_path / "progress.json"
    attempts = []
    def read(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("Windows atomic replacement sharing violation")
        return '{"processed":3}'
    monkeypatch.setattr(type(path), "read_text", read)
    monkeypatch.setattr(reliability.time, "sleep", lambda *a: None)
    assert read_json(path) == {"processed": 3}
    assert len(attempts) == 2
    monkeypatch.setattr(type(path), "read_text", lambda *a, **k: "invalid")
    with pytest.raises(json.JSONDecodeError):
        read_json(path)


def test_drive_readback_hashes_stream_without_buffering_file(tmp_path, monkeypatch):
    source = tmp_path / "strict.mkv"
    payload = (b"episode-data-" * 100_000) + b"end"
    source.write_bytes(payload)
    calls = []

    def fake_run(command, *, idle_timeout, total_timeout, stdout_handler=None):
        calls.append((command, stdout_handler is not None))
        if command[1] == "cat":
            for offset in range(0, len(payload), 8191):
                stdout_handler(payload[offset:offset + 8191])
        return b""

    monkeypatch.setattr("mas.remote.shutil.which", lambda executable: "rclone")
    monkeypatch.setattr("mas.remote._run_watchdog", fake_run)

    result = upload_verified(source, "drive:strict/episode.mkv")

    assert result == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "remote": "drive:strict/episode.mkv",
    }
    assert [(command[1], streamed) for command, streamed in calls] == [
        ("copyto", False),
        ("cat", True),
        ("moveto", False),
        ("cat", True),
    ]
    assert "--immutable" in calls[2][0]


def test_network_watchdog_observes_small_progress_chunks():
    script = (
        "import sys,time\n"
        "for _ in range(5):\n"
        " sys.stderr.write('x'); sys.stderr.flush(); time.sleep(0.1)\n"
        "print('done')\n"
    )
    output = _run_watchdog(
        [sys.executable, "-u", "-c", script], idle_timeout=0.2, total_timeout=3
    )
    assert output.strip() == b"done"


def test_network_watchdog_total_timeout_survives_closed_pipes():
    script = "import os,time; os.close(1); os.close(2); time.sleep(5)"
    with pytest.raises(RemoteVerificationError, match="watchdog expired"):
        _run_watchdog(
            [sys.executable, "-c", script], idle_timeout=0.2, total_timeout=0.4
        )


def test_runpod_stop_requires_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-id")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunPodShutdownError, match="RUNPOD_API_KEY"):
        stop_current_pod()


def test_non_runpod_shutdown_is_noop(monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert stop_current_pod() == {"requested": False, "reason": "not_running_on_runpod"}


def test_stage_notifies_start_and_failure(tmp_path, monkeypatch):
    state = {"episode": 13, "stages": {}}
    events = []
    monkeypatch.setattr(pipeline, "notify", lambda episode, event, details=None: events.append((episode, event, details)))

    def fail():
        raise ValueError("invalid returned ZIP")

    with pytest.raises(ValueError, match="invalid returned ZIP"):
        pipeline._stage(tmp_path / "state.json", state, "id_return", fail)

    assert state["stages"]["id_return"]["status"] == "failed"
    assert events == [
        (
            13,
            "Endonezce çeviri dönüşü doğrulaması başladı",
            "Dönen Endonezce çevirinin kimliği, sırası ve değişmez alanları doğrulanacak.",
        ),
        (
            13,
            "Endonezce çeviri dönüşü doğrulaması başarısız",
            "Sonuç: aşama tamamlanamadı.\n"
            "Süre: 0.0 saniye.\n"
            "Hata: ValueError: invalid returned ZIP\n"
            "Sonraki adım: hatayı giderip aynı bölüm komutuyla güvenli devam edin.",
        ),
    ]


def test_download_notification_explains_resume_check(tmp_path, monkeypatch):
    state = {"episode": 13, "stages": {}}
    events = []
    monkeypatch.setattr(
        pipeline,
        "notify",
        lambda episode, event, details=None: events.append(
            (episode, event, details)
        ),
    )

    pipeline._stage(tmp_path / "state.json", state, "download", lambda: {})

    assert events[0][1] == "kaynak dosyası kontrolü başladı"
    assert "SHA-256 bütünlüğü" in events[0][2]
    assert events[1][1] == "kaynak dosyası kontrolü tamamlandı"
    assert "Sonuç: kaynak dosyası kontrolü tamamlandı." in events[1][2]
    assert "Süre: " in events[1][2]
    assert "Sonraki adım: ses dosyasını hazırlamak." in events[1][2]


@pytest.mark.parametrize(
    ("pod_id", "api_key", "message"),
    (("pod/id", "key", "POD_ID"), ("pod-id", "key\nheader", "line breaks")),
)
def test_runpod_shutdown_rejects_unsafe_environment(monkeypatch, pod_id, api_key, message):
    monkeypatch.setenv("RUNPOD_POD_ID", pod_id)
    monkeypatch.setenv("RUNPOD_API_KEY", api_key)
    with pytest.raises(RunPodShutdownError, match=message):
        stop_current_pod()
