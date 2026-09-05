import copy
import hashlib
import http.client
import io
import json
import threading
import wave
import zipfile
from http.server import ThreadingHTTPServer

import pytest

from mas.audio_review_ui import AudioReviewStore, AudioReviewUIError, make_handler
from mas.engine.tr_correction import create_tr_correction_output, create_tr_correction_pack
from mas.hashing import sha256_json


def _wav(duration_ms):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * round(16_000 * duration_ms / 1_000))
    return output.getvalue()


def _episode(tmp_path, *, pending=("hole-1", "candidate-1")):
    root = tmp_path / "Muhtemel Ask 11.Bolum"
    (root / "prepare").mkdir(parents=True)
    (root / "translation_input").mkdir()
    (root / "translation_output").mkdir()
    hole_audio = _wav(800)
    candidate_audio = _wav(1_519)
    audit = {
        "avg_logprob": -0.25,
        "no_speech_prob": 0.08,
        "compression_ratio": 1.1,
        "temperature": 0.0,
    }
    utterances = [
        {
            "utterance_uid": "candidate-1",
            "utterance_index": 1,
            "coarse_start_ms": 1_000,
            "coarse_end_ms": 1_019,
            "asr_text": "Eski ASR metni",
            "youtube_text": "",
            "context_before": "önce",
            "context_after": "sonra",
            "risk_flags": ["suspected_asr_hallucination"],
            "asr_audit": copy.deepcopy(audit),
        },
        {
            "utterance_uid": "hole-1",
            "utterance_index": 2,
            "coarse_start_ms": 3_100,
            "coarse_end_ms": 3_900,
            "asr_text": "",
            "youtube_text": "",
            "context_before": "önce",
            "context_after": "sonra",
            "risk_flags": ["unresolved_vad_speech", "speech_without_asr"],
            "asr_audit": {
                "avg_logprob": None,
                "no_speech_prob": None,
                "compression_ratio": None,
                "temperature": None,
            },
        },
    ]
    hole = {
            "hole_uid": "hole-1",
            "hole_index": 1,
            "start_ms": 3_100,
            "end_ms": 3_900,
            "clip_start_ms": 3_100,
            "clip_end_ms": 3_900,
            "reason": "VAD speech has no ASR words",
            "audio_member": "speech_hole_audio/hole-1.wav",
            "audio_sha256": hashlib.sha256(hole_audio).hexdigest(),
            "audio_size_bytes": len(hole_audio),
            "context_before": "önce",
            "context_after": "sonra",
            "risk_flags": ["unresolved_vad_speech", "speech_without_asr"],
        }
    candidate = {
            "candidate_uid": "candidate-evidence-1",
            "candidate_index": 1,
            "utterance_uid": "candidate-1",
            "utterance_index": 1,
            "start_ms": 1_000,
            "end_ms": 1_019,
            "clip_start_ms": 250,
            "clip_end_ms": 1_769,
            "reason": "ASR target has no independent-VAD overlap",
            "audio_member": "asr_hallucination_audio/candidate-evidence-1.wav",
            "audio_sha256": hashlib.sha256(candidate_audio).hexdigest(),
            "audio_size_bytes": len(candidate_audio),
            "asr_text": "Eski ASR metni",
            "youtube_text": "",
            "context_before": "önce",
            "context_after": "sonra",
            "risk_flags": ["suspected_asr_hallucination"],
            "asr_audit": copy.deepcopy(audit),
        }
    evidence = {
        "hole-1": hole,
        "candidate-1": candidate,
    }
    for record, payload in ((hole, hole_audio), (candidate, candidate_audio)):
        path = root.joinpath(*record["audio_member"].split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    pack = root / "translation_input" / f"{root.name}_TR_CORRECTION_PACK.zip"
    pack_manifest = create_tr_correction_pack(
        utterances,
        [hole],
        pack,
        episode=11,
        speech_hole_audio_root=root,
        asr_hallucination_records=[candidate],
        asr_hallucination_audio_root=root,
    )
    records = []
    for record in utterances:
        output = copy.deepcopy(record)
        output.update(
            {
                "tr_corrected": (
                    "Güvenilir düzeltilmiş metin"
                    if record["utterance_uid"] == "candidate-1"
                    else ""
                ),
                "non_dialogue": False,
                "review_required": True,
                "audio_reviewed": False,
                "review_disposition": "pending_audio_review",
                "note": "Metin hazır; ses kararı bekleniyor.",
            }
        )
        records.append(output)
    output_path = root / "translation_output" / f"{root.name}_TR_TEXT_CORRECTED.zip"
    output_manifest = create_tr_correction_output(pack, records, output_path)
    kinds = {"hole-1": "speech_hole", "candidate-1": "asr_caption_candidate"}
    report = {
        "correction_input_sha256": pack_manifest["input_sha256"],
        "provisional_output_sha256": output_manifest["output_sha256"],
        "pending_count": len(pending),
        "pending_utterance_uids": list(pending),
        "outcomes": [
            {
                "utterance_uid": uid,
                "evidence_kind": kinds[uid],
                "decision": (
                    "pending_audio_review" if uid in pending else "confirmed_dialogue"
                ),
                "audio_member": evidence[uid]["audio_member"],
                "audio_sha256": evidence[uid]["audio_sha256"],
                "start_ms": 100,
                "end_ms": 300,
                "secondary_transcript": "",
            }
            for uid in ("hole-1", "candidate-1")
        ],
    }
    report["audio_review_sha256"] = sha256_json(report)
    (root / "prepare" / "audio_review_v2.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return root


def _update_report(root, change):
    path = root / "prepare" / "audio_review_v2.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.pop("audio_review_sha256")
    change(report)
    report["audio_review_sha256"] = sha256_json(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_store_serves_only_pending_hash_bound_audio_and_partial_saves(tmp_path):
    root = _episode(tmp_path)
    store = AudioReviewStore(root)

    assert [item["utterance_uid"] for item in store.items] == ["hole-1", "candidate-1"]
    assert store.audio("hole-1") == _wav(800)
    candidate = next(item for item in store.items if item["utterance_uid"] == "candidate-1")
    assert candidate["tr_corrected"] == "Güvenilir düzeltilmiş metin"
    store.save(
        "hole-1",
        {
            "disposition": "confirmed_dialogue",
            "tr_corrected": "Evet.",
            "note": "WAV dinlendi; Evet duyuluyor.",
        },
    )
    saved = json.loads(store.overrides_path.read_text(encoding="utf-8"))
    assert list(saved) == ["hole-1"]
    assert AudioReviewStore(root).overrides == saved


@pytest.mark.parametrize(
    "uid,value,message",
    [
        (
            "hole-1",
            {"disposition": "discarded_asr_hallucination", "tr_corrected": "", "note": "WAV dinlendi."},
            "disposition",
        ),
        (
            "candidate-1",
            {"disposition": "confirmed_dialogue", "tr_corrected": "", "note": "WAV dinlendi."},
            "requires Turkish text",
        ),
        (
            "candidate-1",
            {"disposition": "discarded_asr_hallucination", "tr_corrected": "metin", "note": "WAV dinlendi."},
            "requires empty Turkish text",
        ),
        (
            "candidate-1",
            {"disposition": "discarded_asr_hallucination", "tr_corrected": "", "note": ""},
            "listening note",
        ),
    ],
)
def test_store_rejects_invalid_decisions(tmp_path, uid, value, message):
    store = AudioReviewStore(_episode(tmp_path))
    with pytest.raises(AudioReviewUIError, match=message):
        store.save(uid, value)


def test_store_fails_closed_for_unknown_stale_override(tmp_path):
    root = _episode(tmp_path)
    (root / "review").mkdir()
    (root / "review" / "audio_review_overrides.json").write_text(
        json.dumps(
            {
                "old-uid": {
                    "disposition": "reviewed_non_dialogue",
                    "tr_corrected": "",
                    "note": "WAV dinlendi; konuşma yok.",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AudioReviewUIError, match="stale or unknown"):
        AudioReviewStore(root)


def test_store_detects_audio_hash_mismatch_before_serving(tmp_path):
    root = _episode(tmp_path, pending=("hole-1",))
    pack = root / "translation_input" / f"{root.name}_TR_CORRECTION_PACK.zip"
    with zipfile.ZipFile(pack, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["speech_hole_audio/hole-1.wav"] = b"x" * len(
        members["speech_hole_audio/hole-1.wav"]
    )
    with zipfile.ZipFile(pack, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(AudioReviewUIError, match="SHA-256 mismatch"):
        AudioReviewStore(root)


def test_store_detects_report_change_before_save(tmp_path):
    root = _episode(tmp_path)
    store = AudioReviewStore(root)
    report = root / "prepare" / "audio_review_v2.json"
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AudioReviewUIError, match="report changed"):
        store.save(
            "hole-1",
            {
                "disposition": "reviewed_non_dialogue",
                "tr_corrected": "",
                "note": "WAV dinlendi; konuşma duyulmuyor.",
            },
        )


@pytest.mark.parametrize(
    "field",
    ["correction_input_sha256", "provisional_output_sha256"],
)
def test_store_rejects_report_correction_binding_mismatch(tmp_path, field):
    root = _episode(tmp_path)
    _update_report(root, lambda report: report.__setitem__(field, "0" * 64))
    with pytest.raises(AudioReviewUIError, match=field):
        AudioReviewStore(root)


def test_resolved_manual_override_is_preserved_across_partial_resume(tmp_path):
    root = _episode(tmp_path)
    first = AudioReviewStore(root)
    first.save(
        "candidate-1",
        {
            "disposition": "confirmed_dialogue",
            "tr_corrected": "Güvenilir düzeltilmiş metin",
            "note": "WAV dinlendi; konuşma ve metin doğrulandı.",
        },
    )

    def resolve_candidate(report):
        report["pending_utterance_uids"] = ["hole-1"]
        report["pending_count"] = 1
        candidate = next(
            item for item in report["outcomes"] if item["utterance_uid"] == "candidate-1"
        )
        candidate["decision"] = "confirmed_dialogue"
        candidate["source"] = "manual"

    _update_report(root, resolve_candidate)
    resumed = AudioReviewStore(root)
    assert [item["utterance_uid"] for item in resumed.items] == ["hole-1"]
    assert set(resumed.overrides) == {"candidate-1"}
    resumed.save(
        "hole-1",
        {
            "disposition": "reviewed_non_dialogue",
            "tr_corrected": "",
            "note": "WAV dinlendi; hedef aralıkta konuşma duyulmadı.",
        },
    )
    assert set(json.loads(resumed.overrides_path.read_text(encoding="utf-8"))) == {
        "candidate-1",
        "hole-1",
    }


def test_cli_registers_review_audio(monkeypatch):
    from mas import cli

    called = []
    monkeypatch.setattr(cli, "run_audio_review_ui", lambda episode: called.append(episode) or 0)
    assert cli.main(["review-audio", "11"]) == 0
    assert called == [11]


def test_local_http_ui_serves_audio_and_saves(tmp_path):
    store = AudioReviewStore(_episode(tmp_path, pending=("hole-1",)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, "secret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", "/api/state?token=secret")
        response = connection.getresponse()
        state = json.loads(response.read())
        assert response.status == 200
        assert state["pending_count"] == 1

        connection.request("GET", "/audio/hole-1?token=secret")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == _wav(800)

        payload = json.dumps(
            {
                "token": "secret",
                "uid": "hole-1",
                "disposition": "reviewed_non_dialogue",
                "tr_corrected": "",
                "note": "WAV dinlendi; yalnız müzik duyuluyor.",
            }
        )
        connection.request(
            "POST",
            "/api/save",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["override"]["disposition"] == "reviewed_non_dialogue"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
