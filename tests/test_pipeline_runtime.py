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
                {"utterance_uid": "hole", "review_required": True},
                {"utterance_uid": "candidate", "review_required": True},
                {"utterance_uid": "extra-1", "review_required": True},
                {"utterance_uid": "ignored", "review_required": False},
                {"utterance_uid": "explicit", "review_required": True},
                {"utterance_uid": "extra-2", "review_required": True},
            )
        ),
    )

    assert pipeline._pending_extra_audio_review_uids(input_pack, output_pack) == (
        "extra-1",
        "explicit",
        "extra-2",
    )


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
        (13, "Endonezce çeviri dönüşü doğrulaması başladı", None),
        (
            13,
            "Endonezce çeviri dönüşü doğrulaması başarısız",
            "ValueError: invalid returned ZIP",
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
    assert "yeniden indirilmeyecek" in events[0][2]
    assert events[1][1] == "kaynak dosyası kontrolü tamamlandı"
    assert events[1][2].startswith("Süre: ")


@pytest.mark.parametrize(
    ("pod_id", "api_key", "message"),
    (("pod/id", "key", "POD_ID"), ("pod-id", "key\nheader", "line breaks")),
)
def test_runpod_shutdown_rejects_unsafe_environment(monkeypatch, pod_id, api_key, message):
    monkeypatch.setenv("RUNPOD_POD_ID", pod_id)
    monkeypatch.setenv("RUNPOD_API_KEY", api_key)
    with pytest.raises(RunPodShutdownError, match=message):
        stop_current_pod()
