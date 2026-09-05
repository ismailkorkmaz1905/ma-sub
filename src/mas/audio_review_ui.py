import hashlib
import json
import os
import secrets
import tempfile
import threading
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .config import episode_dir
from .engine.tr_correction import (
    TRCorrectionError,
    read_tr_correction_pack,
    validate_tr_correction_output,
)
from .hashing import sha256_file, sha256_json


class AudioReviewUIError(RuntimeError):
    pass


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_override(uid, kind, raw):
    if not isinstance(raw, dict):
        raise AudioReviewUIError(f"Override {uid} must be an object")
    if set(raw) != {"disposition", "tr_corrected", "note"}:
        raise AudioReviewUIError(
            f"Override {uid} must contain only disposition, tr_corrected and note"
        )
    disposition = raw["disposition"]
    allowed = (
        {"confirmed_dialogue", "reviewed_non_dialogue"}
        if kind == "speech_hole"
        else {"confirmed_dialogue", "discarded_asr_hallucination"}
    )
    if disposition not in allowed:
        raise AudioReviewUIError(
            f"Override {uid} disposition must be one of {sorted(allowed)}"
        )
    corrected = raw["tr_corrected"]
    note = raw["note"]
    if not isinstance(corrected, str):
        raise AudioReviewUIError(f"Override {uid} tr_corrected must be a string")
    if (
        not isinstance(note, str)
        or len(note.strip()) < 10
        or "dinlendi" not in note.casefold()
    ):
        raise AudioReviewUIError(f"Override {uid} requires a concrete listening note")
    if disposition == "confirmed_dialogue" and not corrected.strip():
        raise AudioReviewUIError(
            f"Override {uid} confirmed dialogue requires Turkish text"
        )
    if disposition != "confirmed_dialogue" and corrected.strip():
        raise AudioReviewUIError(
            f"Override {uid} non-dialogue decision requires empty Turkish text"
        )
    return {
        "disposition": disposition,
        "tr_corrected": corrected.strip(),
        "note": note.strip(),
    }


class AudioReviewStore:
    def __init__(self, root):
        self.root = Path(root)
        self.report_path = self.root / "prepare" / "audio_review_v2.json"
        name = self.root.name
        self.pack_path = self.root / "translation_input" / f"{name}_TR_CORRECTION_PACK.zip"
        self.output_path = self.root / "translation_output" / f"{name}_TR_TEXT_CORRECTED.zip"
        self.overrides_path = self.root / "review" / "audio_review_overrides.json"
        self.lock = threading.Lock()
        try:
            self.report_digest = sha256_file(self.report_path)
        except FileNotFoundError as exc:
            raise AudioReviewUIError(f"Missing audio review report: {self.report_path}") from exc
        self.items = self._load_items()
        if sha256_file(self.report_path) != self.report_digest:
            raise AudioReviewUIError("Audio review report changed while the UI was opening")
        self.by_uid = {item["utterance_uid"]: item for item in self.items}
        self.overrides = self._load_overrides()

    def _load_items(self):
        try:
            report = json.loads(self.report_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AudioReviewUIError(f"Missing audio review report: {self.report_path}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AudioReviewUIError(f"Invalid audio review report: {self.report_path}") from exc
        report_sha = report.get("audio_review_sha256")
        if (
            not isinstance(report_sha, str)
            or report_sha
            != sha256_json(
                {key: value for key, value in report.items() if key != "audio_review_sha256"}
            )
        ):
            raise AudioReviewUIError("Audio review report digest mismatch")
        pending = report.get("pending_utterance_uids")
        outcomes = report.get("outcomes")
        if not isinstance(pending, list) or not all(
            isinstance(uid, str) and uid for uid in pending
        ):
            raise AudioReviewUIError("Audio review pending UID list is malformed")
        if len(pending) != len(set(pending)) or report.get("pending_count") != len(pending):
            raise AudioReviewUIError("Audio review pending UID count is inconsistent")
        if not isinstance(outcomes, list):
            raise AudioReviewUIError("Audio review outcomes are malformed")
        pending_set = set(pending)
        all_outcomes_by_uid = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                raise AudioReviewUIError("Audio review outcome must be an object")
            uid = outcome.get("utterance_uid")
            kind = outcome.get("evidence_kind")
            if not isinstance(uid, str) or not uid or uid in all_outcomes_by_uid:
                raise AudioReviewUIError(f"Audio review outcome UID is invalid or duplicated: {uid}")
            if kind not in {"speech_hole", "asr_caption_candidate"}:
                raise AudioReviewUIError(f"Unsupported evidence kind for {uid}: {kind}")
            all_outcomes_by_uid[uid] = outcome
            if uid in pending_set:
                if outcome.get("decision") != "pending_audio_review":
                    raise AudioReviewUIError(f"Pending outcome is inconsistent: {uid}")
        if not pending_set.issubset(all_outcomes_by_uid):
            raise AudioReviewUIError("Audio review pending outcomes do not match pending UID list")

        if not self.pack_path.is_file():
            raise AudioReviewUIError(f"Missing correction pack: {self.pack_path}")
        if not self.output_path.is_file():
            raise AudioReviewUIError(f"Missing provisional correction output: {self.output_path}")
        try:
            pack = read_tr_correction_pack(self.pack_path)
            provisional = validate_tr_correction_output(self.pack_path, self.output_path)
        except (TRCorrectionError, OSError, zipfile.BadZipFile) as exc:
            raise AudioReviewUIError(f"Invalid correction artifact: {exc}") from exc
        if report.get("correction_input_sha256") != provisional.input_sha256:
            raise AudioReviewUIError("Audio review correction_input_sha256 mismatch")
        if report.get("provisional_output_sha256") != provisional.output_sha256:
            raise AudioReviewUIError("Audio review provisional_output_sha256 mismatch")
        if pack.manifest.get("input_sha256") != provisional.input_sha256:
            raise AudioReviewUIError("Correction pack and provisional output input mismatch")
        evidence_by_uid = {
            str(record["hole_uid"]): record for record in pack.speech_holes
        }
        for record in pack.asr_hallucination_records:
            uid = str(record["utterance_uid"])
            if uid in evidence_by_uid:
                raise AudioReviewUIError(f"Duplicate correction evidence UID: {uid}")
            evidence_by_uid[uid] = record
        if set(all_outcomes_by_uid) != set(evidence_by_uid):
            raise AudioReviewUIError("Audio review outcomes do not match correction review inventory")
        self.review_kinds = {}
        for uid, evidence in evidence_by_uid.items():
            expected_kind = (
                "speech_hole" if "hole_uid" in evidence else "asr_caption_candidate"
            )
            if all_outcomes_by_uid[uid]["evidence_kind"] != expected_kind:
                raise AudioReviewUIError(f"Audio review evidence kind mismatch for {uid}")
            self.review_kinds[uid] = expected_kind
        provisional_by_uid = {
            str(record["utterance_uid"]): record for record in provisional.records
        }
        with zipfile.ZipFile(self.pack_path, "r") as archive:
            items = []
            for uid in pending:
                outcome = all_outcomes_by_uid[uid]
                evidence = evidence_by_uid.get(uid)
                if evidence is None:
                    raise AudioReviewUIError(f"Correction pack has no evidence for {uid}")
                provisional_record = provisional_by_uid.get(uid)
                if provisional_record is None:
                    raise AudioReviewUIError(f"Provisional correction has no record for {uid}")
                kind = outcome.get("evidence_kind")
                if kind not in {"speech_hole", "asr_caption_candidate"}:
                    raise AudioReviewUIError(f"Unsupported evidence kind for {uid}: {kind}")
                member = outcome.get("audio_member")
                digest = outcome.get("audio_sha256")
                if (
                    member != evidence.get("audio_member")
                    or digest != evidence.get("audio_sha256")
                    or not isinstance(member, str)
                    or not isinstance(digest, str)
                ):
                    raise AudioReviewUIError(f"Audio evidence binding mismatch for {uid}")
                try:
                    info = archive.getinfo(member)
                except KeyError as exc:
                    raise AudioReviewUIError(f"Correction pack is missing audio for {uid}") from exc
                if info.is_dir():
                    raise AudioReviewUIError(f"Audio member is not a file for {uid}")
                items.append(
                    {
                        "utterance_uid": uid,
                        "evidence_kind": kind,
                        "audio_member": member,
                        "audio_sha256": digest,
                        "start_ms": outcome.get("start_ms"),
                        "end_ms": outcome.get("end_ms"),
                        "asr_text": evidence.get("asr_text", ""),
                        "youtube_text": evidence.get("youtube_text", ""),
                        "context_before": evidence.get("context_before", ""),
                        "context_after": evidence.get("context_after", ""),
                        "secondary_transcript": outcome.get("secondary_transcript", ""),
                        "tr_corrected": provisional_record["tr_corrected"],
                    }
                )
        return items

    def _load_overrides(self):
        if not self.overrides_path.is_file():
            return {}
        try:
            raw = json.loads(self.overrides_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AudioReviewUIError(f"Invalid override file: {self.overrides_path}") from exc
        if not isinstance(raw, dict):
            raise AudioReviewUIError("Audio review overrides must be an object")
        unknown = sorted(set(raw).difference(self.review_kinds))
        if unknown:
            raise AudioReviewUIError(f"Override file contains stale or unknown UIDs: {unknown}")
        return {
            uid: _validate_override(uid, self.review_kinds[uid], value)
            for uid, value in raw.items()
        }

    def assert_current(self):
        if sha256_file(self.report_path) != self.report_digest:
            raise AudioReviewUIError(
                "Audio review report changed after the UI opened; restart review-audio"
            )

    def audio(self, uid):
        self.assert_current()
        item = self.by_uid.get(uid)
        if item is None:
            raise AudioReviewUIError(f"Stale or unknown pending UID: {uid}")
        with zipfile.ZipFile(self.pack_path, "r") as archive:
            try:
                payload = archive.read(item["audio_member"])
            except KeyError as exc:
                raise AudioReviewUIError(f"Correction pack is missing audio for {uid}") from exc
        if hashlib.sha256(payload).hexdigest() != item["audio_sha256"]:
            raise AudioReviewUIError(f"Audio SHA-256 mismatch for {uid}")
        return payload

    def save(self, uid, raw):
        with self.lock:
            self.assert_current()
            item = self.by_uid.get(uid)
            if item is None:
                raise AudioReviewUIError(f"Stale or unknown pending UID: {uid}")
            value = _validate_override(uid, item["evidence_kind"], raw)
            self.audio(uid)
            updated = dict(self.overrides)
            updated[uid] = value
            _atomic_json(self.overrides_path, updated)
            self.overrides = updated
            return value

    def state(self):
        self.assert_current()
        return {
            "items": self.items,
            "overrides": self.overrides,
            "saved_count": sum(uid in self.overrides for uid in self.by_uid),
            "pending_count": len(self.items),
        }


_PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAS Ses İnceleme</title><style>
body{font:16px system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 18px;background:#f5f6f8;color:#18202a}main{background:white;padding:24px;border-radius:14px;box-shadow:0 4px 20px #0001}button{padding:10px 14px;margin:4px;border:1px solid #789;border-radius:8px;background:#fff;cursor:pointer}button.primary{background:#174ea6;color:white}textarea,input{box-sizing:border-box;width:100%;padding:10px;margin:5px 0 14px;border:1px solid #abb4bf;border-radius:7px;font:inherit}audio{width:100%;margin:12px 0}.muted{color:#5f6b77}.context{white-space:pre-wrap;background:#f2f4f7;padding:10px;border-radius:7px}.saved{color:#08783e;font-weight:600}.error{color:#b00020;font-weight:600}</style></head>
<body><main><h1>MAS Ses İnceleme</h1><div id="progress"></div><h2 id="uid"></h2><div id="kind" class="muted"></div><audio id="audio" controls preload="metadata"></audio><p><b>Önce:</b></p><div id="before" class="context"></div><p><b>ASR / YouTube / İkinci ASR:</b></p><div id="evidence" class="context"></div><p><b>Sonra:</b></p><div id="after" class="context"></div><label>Doğru Türkçe metin</label><textarea id="text" rows="3"></textarea><label>Somut dinleme notu</label><textarea id="note" rows="3" placeholder="WAV dinlendi; ... duyuluyor."></textarea><div id="choices"></div><div><button id="prev">Önceki</button><button id="next">Sonraki</button></div><p id="message"></p></main>
<script>
const token=__TOKEN__;let state,index=0;
const $=id=>document.getElementById(id);
function current(){return state.items[index]}
function render(){const x=current(),v=state.overrides[x.utterance_uid]||{};$('progress').textContent=`${index+1} / ${state.pending_count} - kaydedilen ${state.saved_count}`;$('uid').textContent=x.utterance_uid;const kind=x.evidence_kind==='speech_hole'?'ASR tarafından kaçırılmış olası konuşma':'Şüpheli ASR metni';$('kind').textContent=`${kind} - ${x.start_ms} ms / ${x.end_ms} ms`;$('before').textContent=x.context_before||'(boş)';$('after').textContent=x.context_after||'(boş)';$('evidence').textContent=[x.asr_text,x.youtube_text,x.secondary_transcript].filter(Boolean).join(' / ')||'(boş)';$('text').value=v.tr_corrected??x.tr_corrected??'';$('note').value=v.note||'';$('audio').src=`/audio/${encodeURIComponent(x.utterance_uid)}?token=${encodeURIComponent(token)}`;$('choices').replaceChildren();const choices=x.evidence_kind==='speech_hole'?[['confirmed_dialogue','Konuşma var'],['reviewed_non_dialogue','Konuşma yok']]:[['confirmed_dialogue','Metin doğru, konuşma var'],['discarded_asr_hallucination','ASR uydurmuş, konuşma yok']];for(const [value,label] of choices){const b=document.createElement('button');b.textContent=label;b.className=value==='confirmed_dialogue'?'primary':'';b.onclick=()=>save(value);$('choices').appendChild(b)}$('message').textContent=v.disposition?'Bu kayıt daha önce kaydedildi':'';$('message').className=v.disposition?'saved':''}
async function save(disposition){const x=current(),tr=disposition==='confirmed_dialogue'?$('text').value:'';let note=$('note').value;if(!note.trim()){note=disposition==='confirmed_dialogue'?'WAV dinlendi; hedef aralıkta konuşma duyuldu ve Türkçe metin doğrulandı.':disposition==='reviewed_non_dialogue'?'WAV dinlendi; hedef aralıkta konuşma duyulmadı.':'WAV dinlendi; hedef aralıkta ASR metnini destekleyen konuşma duyulmadı.';$('note').value=note}const body={token,uid:x.utterance_uid,disposition,tr_corrected:tr,note};const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok){$('message').textContent=data.error;$('message').className='error';return}state.overrides[body.uid]=data.override;state.saved_count=state.items.filter(item=>state.overrides[item.utterance_uid]).length;const next=state.items.findIndex((item,position)=>position>index&&!state.overrides[item.utterance_uid]);const wrapped=state.items.findIndex(item=>!state.overrides[item.utterance_uid]);if(next>=0||wrapped>=0){index=next>=0?next:wrapped;render();return}$('message').textContent='Kaydedildi';$('message').className='saved';$('progress').textContent=`${index+1} / ${state.pending_count} - kaydedilen ${state.saved_count}`}
$('prev').onclick=()=>{index=(index-1+state.items.length)%state.items.length;render()};$('next').onclick=()=>{index=(index+1)%state.items.length;render()};
fetch(`/api/state?token=${encodeURIComponent(token)}`).then(async r=>{const data=await r.json();if(!r.ok)throw Error(data.error);state=data;if(!state.items.length)throw Error('Bekleyen ses incelemesi yok');const first=state.items.findIndex(item=>!state.overrides[item.utterance_uid]);index=first>=0?first:0;render()}).catch(e=>{document.body.textContent=e.message});
</script></body></html>"""


def make_handler(store, token):
    page = _PAGE.replace("__TOKEN__", json.dumps(token, ensure_ascii=False)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; media-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, value):
            return secrets.compare_digest(value or "", token)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._reply(200, page, "text/html; charset=utf-8")
                return
            query = parse_qs(parsed.query)
            if not self._authorized((query.get("token") or [""])[0]):
                self._reply(403, {"error": "Invalid review session token"})
                return
            try:
                if parsed.path == "/api/state":
                    self._reply(200, store.state())
                elif parsed.path.startswith("/audio/"):
                    uid = unquote(parsed.path[len("/audio/") :])
                    self._reply(200, store.audio(uid), "audio/wav")
                else:
                    self._reply(404, {"error": "Not found"})
            except AudioReviewUIError as exc:
                self._reply(409, {"error": str(exc)})

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                self._reply(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64_000:
                    raise AudioReviewUIError("Invalid request size")
                raw = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(raw, dict) or not self._authorized(raw.get("token")):
                    self._reply(403, {"error": "Invalid review session token"})
                    return
                uid = raw.get("uid")
                override = {
                    "disposition": raw.get("disposition"),
                    "tr_corrected": raw.get("tr_corrected"),
                    "note": raw.get("note"),
                }
                self._reply(200, {"override": store.save(uid, override)})
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, AudioReviewUIError) as exc:
                self._reply(400, {"error": str(exc)})

        def log_message(self, format, *args):
            return

    return Handler


def run_audio_review_ui(episode):
    store = AudioReviewStore(episode_dir(episode))
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, token))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Audio review: {url}")
    print(f"Pending: {len(store.items)}; saved: {len(store.overrides)}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "AudioReviewStore",
    "AudioReviewUIError",
    "make_handler",
    "run_audio_review_ui",
]
