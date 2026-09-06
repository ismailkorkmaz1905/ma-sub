import importlib.metadata
import inspect
import json
from pathlib import Path

from .download import atomic_write_json, sha256_file, sha256_json
from .transcribe import TranscriptionError


def resolve_model(model_name):
    if Path(model_name).is_dir():
        return Path(model_name).resolve()
    from faster_whisper.utils import download_model
    return Path(download_model(model_name)).resolve()


def model_identity(model_dir):
    root = Path(model_dir)
    if not all((root / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
        raise TranscriptionError("primary ASR model files are incomplete")
    files = {path.relative_to(root).as_posix(): sha256_file(path)
             for path in sorted(root.rglob("*")) if path.is_file()}
    return {"files": files, "sha256": sha256_json(files)}


def producer_identity(functions):
    packages = {}
    for name in ("faster-whisper", "ctranslate2", "tokenizers", "av", "onnxruntime"):
        dist = importlib.metadata.distribution(name)
        files = {}
        for item in dist.files or ():
            if not str(item).endswith(".pyc"):
                path = Path(dist.locate_file(item))
                files[str(item)] = sha256_file(path)
        if not files:
            raise TranscriptionError(f"cannot bind primary ASR runtime package: {name}")
        packages[name] = {"version": dist.version, "files_sha256": sha256_json(files)}
    return {"checkpoint_code_sha256": sha256_file(Path(__file__)),
            "functions": {function.__name__: sha256_json(inspect.getsource(function))
                          for function in functions}, "packages": packages}


def load_primary(path, identity):
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or set(saved) != {"data", "sha256"}:
            raise ValueError("invalid checkpoint envelope")
        data = saved["data"]
        if not isinstance(data, dict):
            raise ValueError("invalid checkpoint body")
        if saved["sha256"] != sha256_json(data):
            raise ValueError("content hash mismatch")
        if data["identity"] != identity:
            return None
        if set(data) != {"identity", "language", "segments", "words"}:
            raise ValueError("invalid fields")
        if not isinstance(data["language"], str) or not all(
            isinstance(data[key], list) for key in ("segments", "words")
        ):
            raise ValueError("invalid primary evidence")
        return data
    except (ValueError, KeyError, TypeError) as exc:
        raise TranscriptionError("primary ASR checkpoint integrity failure") from exc


def save_primary(path, identity, language, segments, words):
    data = {"identity": identity, "language": language,
            "segments": segments, "words": words}
    atomic_write_json(path, {"data": data, "sha256": sha256_json(data)})
