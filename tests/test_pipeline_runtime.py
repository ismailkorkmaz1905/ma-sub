import json
import hashlib
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas import pipeline
from mas.remote import RemoteVerificationError, _run_watchdog, upload_verified
from mas.pipeline import _stage
from mas.runpod import RunPodShutdownError, stop_current_pod


def test_state_module_has_no_unverified_generic_reuse_api():
    from mas import state

    assert not hasattr(state, "reusable")


def test_runpod_worker_requires_external_delivery(monkeypatch):
    monkeypatch.delenv("MAS_EXTERNAL_RUNPOD_CONTROLLER", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")

    assert pipeline._requires_external_delivery() is True


def test_local_worker_keeps_local_delivery(monkeypatch):
    monkeypatch.delenv("MAS_EXTERNAL_RUNPOD_CONTROLLER", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

    assert pipeline._requires_external_delivery() is False


def test_external_controller_requires_external_delivery_without_pod_env(monkeypatch):
    monkeypatch.setenv("MAS_EXTERNAL_RUNPOD_CONTROLLER", "1")
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

    assert pipeline._requires_external_delivery() is True


def test_audio_review_reset_preserves_bound_stale_artifacts(tmp_path, monkeypatch):
    name = "Muhtemel Ask 14.Bolum"
    artifacts = []
    for relative, payload in (
        ("prepare/audio_review_v2.json", b"report"),
        ("prepare/audio_review_v2.recovery.json", b"recovery"),
        (f"translation_output/{name}_TR_CORRECTED.zip", b"final"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append({
            "relative_path": relative,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    body = {
        "format": "mas-audio-review-reset-1",
        "episode": 14,
        "correction_input_sha256": "a" * 64,
        "provisional_output_sha256": "b" * 64,
        "reason": "validated correction evidence changed",
        "artifacts": artifacts,
    }
    marker = {"data": body, "sha256": pipeline.sha256_json(body)}
    marker_path = tmp_path / "review/audio_review_reset.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "read_tr_correction_pack",
        lambda _: SimpleNamespace(manifest={"input_sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_tr_correction_output",
        lambda *_: SimpleNamespace(output_sha256="b" * 64),
    )

    receipt = pipeline._apply_audio_review_reset(
        tmp_path, 14, name, tmp_path / "pack.zip", tmp_path / "output.zip"
    )

    archive = tmp_path / "review/superseded-audio-review" / marker["sha256"]
    assert receipt["artifacts"] == artifacts
    for item in artifacts:
        assert not (tmp_path / item["relative_path"]).exists()
        assert (archive / item["relative_path"]).is_file()
    replacement = tmp_path / f"translation_output/{name}_TR_CORRECTED.zip"
    replacement.write_bytes(b"new-final")
    pipeline._apply_audio_review_reset(
        tmp_path, 14, name, tmp_path / "pack.zip", tmp_path / "output.zip"
    )
    assert replacement.read_bytes() == b"new-final"


def test_strict_finalize_input_binding_covers_inputs_and_producer(tmp_path, monkeypatch):
    producer = tmp_path / "producer.py"
    input_path = tmp_path / "input.json"
    producer.write_text("producer-v1", encoding="utf-8")
    input_path.write_text("input-v1", encoding="utf-8")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_STRICT_FINALIZE_PRODUCER_FILES", ("producer.py",))

    original = pipeline._strict_finalize_input_sha256(13, {"input": input_path})
    input_path.write_text("input-v2", encoding="utf-8")
    changed_input = pipeline._strict_finalize_input_sha256(13, {"input": input_path})
    producer.write_text("producer-v2", encoding="utf-8")
    changed_producer = pipeline._strict_finalize_input_sha256(13, {"input": input_path})

    assert original != changed_input != changed_producer


def test_strict_finalize_checkpoint_hydrates_only_exact_bound_outputs(
    tmp_path, monkeypatch
):
    root = tmp_path / "episode"
    final = root / "final"
    subtitles = final / "subtitles"
    subtitles.mkdir(parents=True)
    outputs = {
        "report": final / "report.json",
        "mkv": final / "episode.mkv",
        "id_srt": subtitles / "episode-id.srt",
        "tr_srt": subtitles / "episode-tr.srt",
    }
    for name, path in outputs.items():
        if name != "report":
            path.write_text(name, encoding="utf-8")
    report = {
        "status": "PASS",
        "episode": 13,
        "outputs": {
            name: pipeline.file_record(path, root)
            for name, path in outputs.items()
            if name != "report"
        },
    }
    outputs["report"].write_text(json.dumps(report), encoding="utf-8")
    marker = final / "strict_finalize.done.json"
    input_sha256 = "a" * 64
    pipeline.write_stage_marker(
        marker,
        stage="strict_finalize_v2",
        input_sha256=input_sha256,
        outputs=outputs,
    )

    assert pipeline._load_strict_finalize_checkpoint(
        marker, outputs["report"], outputs, root, 13, input_sha256
    ) == report

    other_report = final / "other-report.json"
    other_report.write_bytes(outputs["report"].read_bytes())
    with pytest.raises(RuntimeError, match="report path is missing or unsafe"):
        pipeline._load_strict_finalize_checkpoint(
            marker, other_report, outputs, root, 13, input_sha256
        )

    real_lstat = Path.lstat
    report_details = real_lstat(outputs["report"])

    def symlink_lstat(path):
        if path == outputs["report"]:
            return SimpleNamespace(
                st_mode=stat.S_IFLNK,
                st_size=report_details.st_size,
            )
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", symlink_lstat)
    with pytest.raises(RuntimeError, match="report path is missing or unsafe"):
        pipeline._load_strict_finalize_checkpoint(
            marker, outputs["report"], outputs, root, 13, input_sha256
        )
    monkeypatch.setattr(Path, "lstat", real_lstat)

    outputs["id_srt"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint binding changed"):
        pipeline._load_strict_finalize_checkpoint(
            marker, outputs["report"], outputs, root, 13, input_sha256
        )


def test_strict_finalize_checkpoint_does_not_adopt_legacy_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"status":"PASS","episode":13}', encoding="utf-8")

    assert pipeline._load_strict_finalize_checkpoint(
        tmp_path / "missing.done.json", report, {}, tmp_path, 13, "a" * 64
    ) is None


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


@pytest.mark.parametrize('execution', ['remote-nvenc-v1', 'local-qsv-v1'])
def test_production_stop_after_download_does_not_start_audio(tmp_path, monkeypatch, execution):
    monkeypatch.setattr(pipeline, "episode_dir", lambda episode: tmp_path / str(episode))
    monkeypatch.setattr(
        pipeline,
        "_load_configs",
        lambda: (tmp_path, {"whisper_model": "model"}, {"canonical_names": []}, {"terms": []}),
    )
    monkeypatch.setattr(pipeline, "_guard_existing_source", lambda *args: None)
    monkeypatch.setattr(pipeline, "_resolve_source_url", lambda *args: "https://example.test/13")

    def download(url, source_dir, **kwargs):
        video = source_dir / "source.mkv"
        video.write_bytes(b"source")
        return SimpleNamespace(video_path=video, resumed=False, captions_path=None)

    monkeypatch.setattr(pipeline, "download_source", download)
    monkeypatch.setenv("MAS_EXTERNAL_RUNPOD_CONTROLLER", "1")
    monkeypatch.setenv("MAS_DELIVERY_EXECUTION_PLAN", execution)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    monkeypatch.setenv("MAS_MAX_RUNTIME_SECONDS", "1")

    def qualify(*args, **kwargs):
        assert execution == 'remote-nvenc-v1', 'local QSV must not qualify a paid GPU encoder'
        receipt = tmp_path / "qualification.json"
        receipt.write_text(
            json.dumps({"projected_full_encode_seconds": 999}), encoding="utf-8"
        )
        return receipt

    monkeypatch.setattr(pipeline, "qualify_encoding", qualify)
    monkeypatch.setattr(
        pipeline,
        "extract_audio",
        lambda *args, **kwargs: pytest.fail("audio stage must not start"),
    )
    monkeypatch.setattr(
        pipeline,
        "_stage",
        lambda path, state, name, action: action(),
    )

    assert pipeline.run(13, stop_after=1) == 75


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
        return {"audio_sha256": pipeline.sha256_file(audio),
                "alignment_sha256": "a" * 64, "provenance": {"device": "cuda"}}

    monkeypatch.setattr(pipeline, "align_corrected_segments", align)
    monkeypatch.setattr(pipeline, "validate_forced_alignment_data", lambda result: None)
    correction_sha256 = "b" * 64
    assert pipeline._aligned_checkpoint(
        alignment, audio, inputs, vad, correction_sha256
    )[1] is False
    assert pipeline._aligned_checkpoint(
        alignment, audio, inputs, vad, correction_sha256
    )[1] is True
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
        pipeline._aligned_checkpoint(alignment, audio, inputs, vad, correction_sha256)
    assert len(calls) == 1
    assert alignment.read_bytes() == original


def test_verified_legacy_alignment_gets_bound_stage_marker(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    alignment = tmp_path / "forced_alignment_v2.json"
    inputs = [{"utterance_uid": "u1", "text": "Merhaba"}]
    vad = [{"start_ms": 0, "end_ms": 1000}]
    for name in ("src/mas/engine/forced_align.py", "src/mas/engine/speaker.py",
                 "src/mas/engine/workflow.py", "requirements.lock"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    calls = []
    result = {"audio_sha256": pipeline.sha256_file(audio),
              "alignment_sha256": "a" * 64, "provenance": {"device": "cuda"}}
    monkeypatch.setattr(
        pipeline,
        "align_corrected_segments",
        lambda *args, **kwargs: calls.append(1) or result,
    )
    monkeypatch.setattr(pipeline, "validate_forced_alignment_data", lambda value: None)
    correction_sha256 = "b" * 64

    pipeline._aligned_checkpoint(alignment, audio, inputs, vad, correction_sha256)
    done = alignment.with_suffix(".done.json")
    done.unlink()
    assert pipeline._aligned_checkpoint(
        alignment, audio, inputs, vad, correction_sha256
    )[1] is True

    marker = json.loads(done.read_text(encoding="utf-8"))
    assert marker["stage"] == "forced_alignment_v2"
    assert marker["details"] == {
        "alignment_sha256": "a" * 64,
        "correction_output_sha256": correction_sha256,
    }
    assert marker["outputs"]["forced_alignment_v2"]["sha256"] == pipeline.sha256_file(
        alignment
    )
    assert calls == [1]

    marker["details"]["correction_output_sha256"] = "c" * 64
    done.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stage marker binding changed"):
        pipeline._aligned_checkpoint(alignment, audio, inputs, vad, correction_sha256)
    assert calls == [1]


def test_drive_upload_cannot_target_emergency(tmp_path):
    source = tmp_path / "strict.srt"
    source.write_text("strict", encoding="utf-8")
    with pytest.raises(RemoteVerificationError, match="strict destination"):
        upload_verified(source, "drive:emergency/strict.srt")


def test_stage_records_elapsed_seconds_on_success_and_failure(tmp_path, monkeypatch, capsys):
    state = {"episode": 13}
    values = iter((10.0, 12.5, 20.0, 23.25))
    monkeypatch.setattr("mas.pipeline.time.monotonic", lambda: next(values))
    monkeypatch.setattr("mas.pipeline.notify", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(pipeline, "notify", lambda *a, **kw: None)
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

    def fake_run(command, *, idle_timeout, total_timeout, stdout_handler=None, progress_observer=None):
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
        [sys.executable, "-S", "-u", "-c", script], idle_timeout=0.2, total_timeout=3
    )
    assert output.strip() == b"done"


def test_network_watchdog_total_timeout_survives_closed_pipes():
    script = "import os,time; os.close(1); os.close(2); time.sleep(5)"
    with pytest.raises(RemoteVerificationError, match="watchdog expired"):
        _run_watchdog(
            [sys.executable, "-S", "-c", script], idle_timeout=0.2, total_timeout=0.4
        )


def test_network_watchdog_accepts_external_byte_progress_for_silent_process():
    progress = iter((True, True, True, True, True))
    _run_watchdog(
        [sys.executable, "-S", "-c", "import time; time.sleep(0.35)"],
        idle_timeout=0.1,
        total_timeout=1,
        progress_probe=lambda: next(progress, False),
        progress_probe_interval=0.05,
    )


def test_repeated_rclone_stats_do_not_count_as_transfer_progress():
    from mas.remote import _RcloneProgress
    observe = _RcloneProgress()
    chunk = b'{"stats":{"bytes":10,"transfers":0}}\n'
    assert not observe("stderr", chunk[:12])
    assert observe("stderr", chunk[12:])
    assert not observe("stderr", chunk)
    assert not observe("stderr", b'{"stats":{"bytes":0,"elapsedTime":100}}\n')
    assert not observe("stdout", b"heartbeat")
    assert observe("stderr", b'{"stats":{"bytes":11,"transfers":0}}\n')


def test_rclone_chatter_cannot_keep_stalled_process_alive():
    import time
    from mas.remote import _RcloneProgress
    script = "import time\nwhile True:\n print('{\"stats\":{\"bytes\":0}}', flush=True); time.sleep(0.02)"
    started = time.monotonic()
    with pytest.raises(RemoteVerificationError, match="watchdog expired"):
        _run_watchdog([sys.executable, "-u", "-c", script],
                      idle_timeout=0.3, total_timeout=5,
                      progress_observer=_RcloneProgress())
    assert time.monotonic() - started < 3


def test_upload_readbacks_share_one_deadline(tmp_path, monkeypatch):
    import mas.remote as remote_module
    source = tmp_path / "strict.srt"
    source.write_bytes(b"subtitle")
    clock = [100.0]
    deadlines = []
    monkeypatch.setattr(remote_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(remote_module.shutil, "which", lambda name: "rclone")
    def run(command, *, total_timeout, stdout_handler=None, **kwargs):
        deadlines.append(total_timeout)
        clock[0] += 3
        if stdout_handler:
            stdout_handler(b"subtitle")
        return b""
    monkeypatch.setattr(remote_module, "_run_watchdog", run)
    with pytest.raises(RemoteVerificationError, match="transaction deadline"):
        upload_verified(source, "drive:strict/subtitle.srt", total_timeout=8)
    assert deadlines == [8, 5, 2]


def test_failed_output_handler_reaps_network_process(monkeypatch):
    import mas.remote as remote_module
    original = remote_module.subprocess.Popen
    processes = []
    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    def fail(chunk):
        raise RuntimeError("consumer failed")
    monkeypatch.setattr(remote_module.subprocess, "Popen", start)
    with pytest.raises(RuntimeError, match="consumer failed"):
        _run_watchdog([sys.executable, "-u", "-c", "import time; print('data', flush=True); time.sleep(30)"],
                      stdout_handler=fail, total_timeout=5)
    assert processes[0].poll() is not None


def test_targeted_audio_clip_extraction_has_finite_deadline(tmp_path, monkeypatch):
    from mas.engine import transcribe
    import subprocess
    monkeypatch.setattr(transcribe.shutil, "which", lambda name: "ffmpeg")
    def run(command, **kwargs):
        assert kwargs["timeout"] == 120
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(transcribe.subprocess, "run", run)
    with pytest.raises(transcribe.TranscriptionError, match="exceeded 120 seconds"):
        transcribe._extract_clip(tmp_path / "source.flac", tmp_path / "clip.wav", 1000, 2000)


def test_runpod_stop_requires_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-id")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunPodShutdownError, match="RUNPOD_API_KEY"):
        stop_current_pod()


def test_non_runpod_shutdown_is_noop(monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert stop_current_pod() == {"requested": False, "reason": "not_running_on_runpod"}


def test_stage_only_notifies_failure_without_start_chatter(tmp_path, monkeypatch):
    state = {"episode": 13, "stages": {}}
    events = []
    monkeypatch.setattr(pipeline, "notify", lambda episode, event, details=None, **kwargs: events.append((episode, event, details)))

    def fail():
        raise ValueError("invalid returned ZIP")

    with pytest.raises(ValueError, match="invalid returned ZIP"):
        pipeline._stage(tmp_path / "state.json", state, "id_return", fail)

    assert state["stages"]["id_return"]["status"] == "failed"
    assert events == [
        (
            13,
            "Endonezce çeviri dönüşü doğrulaması başarısız",
            "Sonuç: aşama tamamlanamadı.\n"
            "Hata: ValueError: invalid returned ZIP\n"
            "Sonraki adım: hatayı giderip aynı bölüm komutuyla güvenli devam edin.",
        ),
    ]


def test_only_fresh_major_milestones_enqueue_mail(tmp_path, monkeypatch):
    state = {"episode": 13, "stages": {}}
    events = []
    monkeypatch.setattr(
        pipeline,
        "notify",
        lambda episode, event, details=None, **kwargs: events.append(
            (episode, event, kwargs)
        ),
    )

    pipeline._stage(tmp_path / "state.json", state, "download", lambda: {})

    assert events == []
    pipeline._stage(tmp_path / 'state.json', state, 'raw_asr', lambda: {'resumed': False})
    pipeline._stage(tmp_path / 'state.json', state, 'raw_asr', lambda: {'resumed': True})
    assert len(events) == 1 and events[0][2]['kind'] == 'milestone'


@pytest.mark.parametrize(
    ("pod_id", "api_key", "message"),
    (("pod/id", "key", "POD_ID"), ("pod-id", "key\nheader", "line breaks")),
)
def test_runpod_shutdown_rejects_unsafe_environment(monkeypatch, pod_id, api_key, message):
    monkeypatch.setenv("RUNPOD_POD_ID", pod_id)
    monkeypatch.setenv("RUNPOD_API_KEY", api_key)
    with pytest.raises(RunPodShutdownError, match=message):
        stop_current_pod()
