from datetime import datetime, timezone
from pathlib import Path

from . import RULES_VERSION, SCHEMA_VERSION, __version__
from .reliability import atomic_json, read_json


def load(path, episode):
    path = Path(path)
    if not path.exists():
        return {
            "episode": episode,
            "pipeline_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "stages": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    state = read_json(path)
    if state.get("episode") != episode:
        raise ValueError("state episode mismatch")
    return state


def save(path, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(Path(path), state)


def set_stage(state_path, state, name, status, **details):
    if status not in {"pending", "running", "pass", "blocked", "failed"}:
        raise ValueError(f"invalid stage status: {status}")
    state.setdefault("stages", {})[name] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    save(state_path, state)
