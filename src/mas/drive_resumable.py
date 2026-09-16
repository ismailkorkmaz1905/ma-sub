import argparse
import base64
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .reliability import atomic_json, digest, read_json


class ResumableUploadError(RuntimeError):
    pass


def _dpapi(data, *, decrypt=False):
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    operation = library.CryptUnprotectData if decrypt else library.CryptProtectData
    operation.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    operation.restype = wintypes.BOOL
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ResumableUploadError("private upload session protection failed")
    free = ctypes.WinDLL("kernel32").LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.memset(result.data, 0, result.size)
        free(result.data)


def session_directory(directory=None):
    if directory is None:
        directory = os.getenv("MAS_DRIVE_SESSION_DIR")
    if directory is None:
        root = os.getenv("LOCALAPPDATA") if os.name == "nt" else os.getenv("XDG_STATE_HOME")
        directory = (Path(root) if root else Path.home() / ".local" / "state") / "ma-sub" / "drive-sessions"
    path = Path(directory).resolve()
    project = Path(__file__).resolve().parents[2]
    if path == project or project in path.parents or any((p / ".git").exists() for p in (path, *path.parents)):
        raise ResumableUploadError("private upload sessions must be stored outside Git and the project")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix" and (path.stat().st_mode & 0o077 or path.stat().st_uid != os.geteuid()):
        raise ResumableUploadError("upload session directory must be owned by the user with mode 0700")
    return path


class SessionStore:
    def __init__(self, partial, directory=None):
        self.path = session_directory(directory) / (digest({"partial": partial}) + ".json")
        self.lock = None

    def __enter__(self):
        lock_path = self.path.with_suffix(".lock")
        if lock_path.is_symlink() or self.path.is_symlink():
            raise ResumableUploadError("unsafe private session path")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        self.lock = os.fdopen(fd, "r+b")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            self.lock = None
            raise ResumableUploadError("another process owns this upload session") from None
        return self

    def __exit__(self, *_args):
        self.lock.close()

    def load(self):
        if not self.path.exists():
            return None
        try:
            if os.name == "posix" and (self.path.stat().st_mode & 0o077
                                        or self.path.stat().st_uid != os.geteuid()):
                raise ValueError("permissions")
            envelope = read_json(self.path)
            if os.name == "nt":
                if set(envelope) != {"dpapi_user"}:
                    raise ValueError("protection")
                envelope = json.loads(_dpapi(base64.b64decode(envelope["dpapi_user"], validate=True), decrypt=True))
            payload = envelope["payload"]
            if not isinstance(payload, dict) or envelope["sha256"] != digest(payload):
                raise ValueError("integrity")
            return payload
        except (OSError, ValueError, KeyError, TypeError):
            raise ResumableUploadError("private upload session cannot be validated") from None

    def save(self, state):
        envelope = {"payload": state, "sha256": digest(state)}
        if os.name == "nt":
            envelope = {"dpapi_user": base64.b64encode(_dpapi(json.dumps(envelope).encode())).decode("ascii")}
        atomic_json(self.path, envelope)


def prepare_session(partial, binding, *, allow_create, session_dir=None):
    # Durable local intent precedes the controller's "started" flag and any HTTP create.
    with SessionStore(partial, session_dir) as store:
        state = store.load()
        if state is None:
            if not allow_create:
                raise ResumableUploadError("saved upload session is missing; restart is not authorized")
            store.save({"binding": binding, "status": "PREPARED", "acked": 0, "sent_through": 0})
        elif state.get("binding") != binding:
            raise ResumableUploadError("upload session source/destination/OAuth binding changed")


def _session_url(url, object_id=None):
    try:
        parts = urlsplit(url)
        valid = (parts.scheme == "https" and parts.hostname == "www.googleapis.com"
                 and parts.port in (None, 443) and not parts.username and not parts.password
                 and parts.path in {"/upload/drive/v3/files", "/upload/drive/v3/files/" + str(object_id)}
                 and not parts.fragment
                 and "upload_id=" in parts.query)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ResumableUploadError("Drive returned an unsafe upload session location")
    return url


def _request(method, url, headers, body, timeout):
    parts = urlsplit(url)
    connection = http.client.HTTPSConnection(parts.hostname, timeout=timeout)
    try:
        connection.request(method, parts.path + ("?" + parts.query if parts.query else ""), body, headers)
        response = connection.getresponse()
        data = response.read(65537)
        if len(data) > 65536:
            raise ResumableUploadError("Drive response exceeds the metadata bound")
        return response.status, {key.lower(): value for key, value in response.getheaders()}, data
    finally:
        connection.close()


def _offset(headers, size):
    value = headers.get("range")
    if value is None:
        return 0
    matched = re.fullmatch(r"bytes=0-([0-9]+)", value)
    if not matched or not 0 < int(matched[1]) + 1 <= size:
        raise ResumableUploadError("Drive acknowledged an invalid byte range")
    return int(matched[1]) + 1


def upload(source, remote, partial, parent_id, size, sha256, credentials, *, total_timeout=3600,
           idle_timeout=120, session_dir=None, chunk_size=8 * 1024 * 1024,
           transport=None, progress=None, allow_session_create=True):
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in (total_timeout, idle_timeout))
            or type(chunk_size) is not int or chunk_size <= 0 or chunk_size % (256 * 1024)):
        raise ResumableUploadError("invalid upload bounds")
    if (type(size) is not int or size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or not re.fullmatch(re.escape(remote) + r"\.partial-[0-9a-f]{32}", partial)
            or "emergency" in remote.casefold() or not parent_id):
        raise ResumableUploadError("invalid strict upload binding")
    transport = transport or _request
    progress = progress or (lambda _phase, _count: None)
    started = last_progress = time.monotonic()
    deadline = started + total_timeout

    def remaining():
        now = time.monotonic()
        value = min(deadline - now, idle_timeout - (now - last_progress), 30)
        if value <= 0:
            raise ResumableUploadError("Drive upload deadline or no-progress bound expired")
        return value

    def advance(phase, count):
        nonlocal last_progress
        last_progress = time.monotonic()
        progress(phase, count)

    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ResumableUploadError("unsafe upload source")
    checksum = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        stamp = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            remaining()
            count += len(chunk)
            checksum.update(chunk)
            advance("hash", count)
        if (count, checksum.hexdigest()) != (size, sha256):
            raise ResumableUploadError("upload source byte/SHA-256 binding changed")
        binding = {"source": str(path.resolve()), "remote": remote, "partial": partial,
                   "parent_id": parent_id, "bytes": size, "sha256": sha256,
                   "credential_identity_sha256": credentials["credential_identity_sha256"]}
        with SessionStore(partial, session_dir) as store:
            state = store.load()
            if state is None and not allow_session_create:
                raise ResumableUploadError("saved upload session is missing; restart is not authorized")
            if state is not None and state.get("binding") != binding:
                raise ResumableUploadError("upload session source/destination/OAuth binding changed")
            if state is not None:
                if (type(state.get("acked")) is not int or type(state.get("sent_through")) is not int
                        or not 0 <= state["acked"] <= state["sent_through"] <= size):
                    raise ResumableUploadError("invalid saved upload acknowledgement")
                reserved = state.get("reserved_id")
                if reserved is not None and (not isinstance(reserved, str)
                        or re.fullmatch(r"[A-Za-z0-9_-]{1,256}", reserved) is None):
                    raise ResumableUploadError("invalid reserved Drive object identity")
                if state.get("status") == "COMPLETE":
                    if (state["acked"] != size or not isinstance(state.get("object_id"), str) or not state["object_id"]
                            or reserved is not None and state["object_id"] != reserved):
                        raise ResumableUploadError("saved completion evidence is invalid")
                    return _completion(state)
                if state.get("status") == "PREPARED":
                    if state["acked"] != 0 or state["sent_through"] != 0 or state.get("session_uri"):
                        raise ResumableUploadError("prepared upload intent is invalid")
                elif state.get("status") in {"INITIATING", "EXPIRED"} and reserved is not None:
                    pass  # Reconcile this exact pre-generated object, never create a second identity.
                else:
                    if state.get("status") != "ACTIVE" or not state.get("session_uri"):
                        raise ResumableUploadError("upload session initiation is ambiguous; evidence preserved")
                    _session_url(state["session_uri"], reserved)
            else:
                state = {"binding": binding, "status": "PREPARED", "acked": 0, "sent_through": 0}
                store.save(state)

            token = None
            refreshes = 0

            def refresh():
                nonlocal token, refreshes
                if refreshes >= 3:
                    raise ResumableUploadError("Drive OAuth refresh retry allowance exhausted")
                refreshes += 1
                payload = urlencode({key: credentials[key] for key in
                    ("client_id", "client_secret", "refresh_token")} | {"grant_type": "refresh_token"}).encode()
                try:
                    code, _, data = transport("POST", "https://oauth2.googleapis.com/token",
                        {"Content-Type": "application/x-www-form-urlencoded"}, payload, remaining())
                    remaining()
                    value = json.loads(data)
                    token = value.get("access_token") if isinstance(value, dict) else None
                    if (code != 200 or not isinstance(token, str) or not token
                            or str(value.get("token_type", "Bearer")).lower() != "bearer"):
                        raise ValueError("token")
                except (OSError, http.client.HTTPException, ValueError):
                    raise ResumableUploadError("private Drive OAuth refresh failed") from None

            def start_session():
                # Generate and persist an ID before create. A lost POST response can
                # then be retried without creating duplicate Drive objects.
                api = "https://www.googleapis.com/drive/v3/files"
                endpoint = "https://www.googleapis.com/upload/drive/v3/files"
                fields = "id,name,size,parents,sha256Checksum"
                if not state.get("reserved_id"):
                    try:
                        code, _, data = transport("GET", api + "/generateIds?count=1&space=drive",
                            {"Authorization": "Bearer " + token}, None, remaining())
                    except (OSError, http.client.HTTPException):
                        raise ResumableUploadError("Drive ID reservation failed; prepared intent retained") from None
                    remaining()
                    try:
                        ids = json.loads(data).get("ids")
                        if (code != 200 or not isinstance(ids, list) or len(ids) != 1
                                or not isinstance(ids[0], str)
                                or re.fullmatch(r"[A-Za-z0-9_-]{1,256}", ids[0]) is None):
                            raise ValueError("id")
                    except (ValueError, AttributeError):
                        raise ResumableUploadError("Drive returned invalid reserved file identity") from None
                    state["reserved_id"] = ids[0]
                    store.save(state)
                object_id = state["reserved_id"]
                for _ in range(3):
                    method, suffix = "POST", ""
                    metadata = {"id": object_id, "name": partial.rsplit("/", 1)[-1], "parents": [parent_id]}
                    if state["status"] in {"INITIATING", "EXPIRED"}:
                        try:
                            code, _, data = transport("GET", api + "/" + object_id + "?" + urlencode({
                                "supportsAllDrives": "true", "fields": fields}),
                                {"Authorization": "Bearer " + token}, None, remaining())
                        except (OSError, http.client.HTTPException):
                            continue
                        remaining()
                        if code == 401:
                            refresh()
                            continue
                        if code == 200:
                            try:
                                observed = json.loads(data)
                                observed_size = observed.get("size", "0")
                                if (observed.get("id") != object_id or observed.get("name") != metadata["name"]
                                        or observed.get("parents") != [parent_id]
                                        or not isinstance(observed_size, str) or not observed_size.isdigit()
                                        or int(observed_size) > size):
                                    raise ValueError("identity")
                            except (ValueError, AttributeError):
                                raise ResumableUploadError("reserved Drive object ownership changed") from None
                            if int(observed_size) == size:
                                if observed.get("sha256Checksum", sha256) != sha256:
                                    raise ResumableUploadError("reserved Drive object content changed")
                                # This is upload completion only. The publisher still
                                # reads every remote byte before any delivery receipt.
                                state.update(status="COMPLETE", acked=size, sent_through=size, object_id=object_id)
                                state.pop("session_uri", None)
                                store.save(state)
                                advance("upload", size)
                                return True
                            method, suffix, metadata = "PATCH", "/" + object_id, {}
                        elif code != 404:
                            if code in {408, 429, 500, 502, 503, 504}:
                                continue
                            raise ResumableUploadError(f"Drive object reconciliation rejected (HTTP {code})")
                    state["status"] = "INITIATING"
                    store.save(state)
                    try:
                        code, headers, _ = transport(method, endpoint + suffix + "?" + urlencode({
                            "uploadType": "resumable", "supportsAllDrives": "true", "fields": fields}),
                            {"Authorization": "Bearer " + token, "Content-Type": "application/json; charset=UTF-8",
                             "X-Upload-Content-Type": "application/octet-stream", "X-Upload-Content-Length": str(size)},
                            json.dumps(metadata).encode(), remaining())
                    except (OSError, http.client.HTTPException):
                        continue
                    remaining()
                    if code == 200:
                        state.update(status="ACTIVE", acked=0, sent_through=0,
                                     session_uri=_session_url(headers.get("location"), object_id))
                        store.save(state)
                        return False
                    if code == 401:
                        refresh()
                    elif code not in {408, 409, 429, 500, 502, 503, 504}:
                        raise ResumableUploadError(f"Drive upload initiation rejected (HTTP {code}); intent retained")
                raise ResumableUploadError("Drive initiation retry allowance exhausted; reserved identity retained")

            refresh()
            if state["status"] != "ACTIVE" and start_session():
                return _completion(state)
            session_restarts = 0
            query = True
            stalled = 0
            while True:
                timeout = remaining()
                if stalled >= 4:
                    raise ResumableUploadError("Drive upload retry/no-progress allowance exhausted; session preserved")
                if not query:
                    current = os.fstat(handle.fileno())
                    if (current.st_size, current.st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
                        raise ResumableUploadError("upload source changed during transfer")
                    handle.seek(state["acked"])
                    payload = handle.read(min(chunk_size, size - state["acked"]))
                    if not payload:
                        raise ResumableUploadError("Drive upload source ended before completion")
                    end = state["acked"] + len(payload)
                    state["sent_through"] = max(state["sent_through"], end)
                    store.save(state)
                    content_range = f"bytes {state['acked']}-{end - 1}/{size}"
                else:
                    payload = b""
                    content_range = f"bytes */{size}"
                try:
                    code, headers, data = transport("PUT", state["session_uri"],
                        {"Authorization": "Bearer " + token, "Content-Length": str(len(payload)),
                         "Content-Type": "application/octet-stream", "Content-Range": content_range}, payload, timeout)
                except (OSError, http.client.HTTPException):
                    query = True
                    stalled += 1
                    continue
                remaining()
                if code in (200, 201):
                    try:
                        metadata = json.loads(data)
                    except ValueError:
                        raise ResumableUploadError("Drive upload completion metadata is invalid") from None
                    if (not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str) or not metadata["id"]
                            or metadata.get("name") != partial.rsplit("/", 1)[-1]
                            or metadata.get("parents") != [parent_id] or metadata.get("size") != str(size)
                            or state["sent_through"] != size
                            or state.get("reserved_id", metadata["id"]) != metadata["id"]):
                        raise ResumableUploadError("Drive upload completion does not match exact file binding")
                    state.update(status="COMPLETE", acked=size, object_id=metadata["id"])
                    state.pop("session_uri", None)
                    store.save(state)
                    advance("upload", size)
                    return _completion(state)
                if code == 308:
                    offset = _offset(headers, size)
                    if not state["acked"] <= offset <= state["sent_through"]:
                        raise ResumableUploadError("Drive acknowledgement regressed or exceeds sent bytes")
                    if offset > state["acked"]:
                        state["acked"] = offset
                        store.save(state)
                        advance("upload", offset)
                        stalled = 0
                    elif not query or offset == size:
                        stalled += 1
                    query = offset == size
                    continue
                if code == 401:
                    refresh()
                elif code == 404:
                    state["status"] = "EXPIRED"
                    store.save(state)
                    if not state.get("reserved_id") or session_restarts >= 2:
                        raise ResumableUploadError("Drive upload session expired; bounded restart unavailable")
                    session_restarts += 1
                    if start_session():
                        return _completion(state)
                    query, stalled = True, 0
                    continue
                elif code not in (408, 429, 500, 502, 503, 504):
                    raise ResumableUploadError(f"Drive upload rejected (HTTP {code}); session preserved")
                stalled += 1
                query = True


def _completion(state):
    return {"status": "UPLOAD_COMPLETE", "bytes": state["binding"]["bytes"],
            "sha256": state["binding"]["sha256"], "object_id": state["object_id"],
            "session_binding_sha256": digest(state["binding"])}


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "remote", "partial", "parent-id", "sha256", "identity", "config"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--total-timeout", required=True, type=float)
    parser.add_argument("--idle-timeout", required=True, type=float)
    parser.add_argument("--allow-session-create", action="store_true")
    args = parser.parse_args()
    try:
        from .remote import _drive_credentials
        credentials = _drive_credentials(args.remote, config_path=args.config)
        if credentials["credential_identity_sha256"] != args.identity:
            raise ResumableUploadError("Drive OAuth identity changed before session upload")
        result = upload(args.source, args.remote, args.partial, args.parent_id, args.size, args.sha256,
                        credentials, total_timeout=args.total_timeout, idle_timeout=args.idle_timeout,
                        allow_session_create=args.allow_session_create,
                        progress=lambda phase, count: print(json.dumps({"phase": phase, "bytes": count}),
                                                            file=sys.stderr, flush=True))
    except Exception as error:
        # Network exceptions may contain session URLs or OAuth payloads. Never print them.
        message = str(error) if isinstance(error, ResumableUploadError) else "private Drive upload failed; evidence preserved"
        print(json.dumps({"error": message}), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
