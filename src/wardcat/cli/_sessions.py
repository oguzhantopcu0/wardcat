"""Saved sessions of the interactive screen, and ``wardcat sessions``.

A session file holds the screen's configuration — filters, layers, the NER
language, the LLM backend and model — so it can be picked up again. It never
holds a secret or a scanned text: the salt and an API key are referred to by
the name of the environment variable that carries them, and a key typed into
the screen lives in memory only. Files are readable by their owner alone.

Sessions live under ``$XDG_STATE_HOME/wardcat/sessions`` (``~/.local/state``
when unset), or ``%LOCALAPPDATA%\\wardcat\\sessions`` on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wardcat.exceptions import ConfigError

SCHEMA = 2
LAYERS = ("regex", "ner", "llm")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def state_dir() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    path = base / "wardcat"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def sessions_dir() -> Path:
    path = state_dir() / "sessions"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def new_session() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "schema": SCHEMA,
        "id": secrets.token_hex(4),
        "created_at": now,
        "updated_at": now,
        "filters": {},
        "layers": ["regex"],
        "ner": None,
        "llm": None,
        "salt_env": None,
    }


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Overlay a loaded file on the defaults; anything malformed falls back.

    A file written by wardcat-cli (schema 1) is read too: its filters, layers,
    NER language and LLM model carry over; its salt is ignored, never reused.
    """
    state = new_session()
    if isinstance(raw.get("id"), str) and _ID.match(raw["id"]):
        state["id"] = raw["id"]
    for key in ("created_at", "updated_at"):
        if isinstance(raw.get(key), str):
            state[key] = raw[key]
    filters = raw.get("filters")
    if isinstance(filters, dict):
        state["filters"] = {
            str(k).upper(): str(v) for k, v in filters.items() if isinstance(v, str)
        }
    layers = raw.get("layers", raw.get("active_layers"))
    if isinstance(layers, list):
        kept = [layer for layer in LAYERS if layer in layers]
        state["layers"] = kept or ["regex"]
    ner = raw.get("ner")
    if isinstance(ner, dict) and isinstance(ner.get("language"), str):
        state["ner"] = {"language": ner["language"], "size": ner.get("size")}
    elif isinstance(raw.get("ner_language"), str):  # wardcat-cli
        state["ner"] = {"language": raw["ner_language"], "size": None}
    llm = raw.get("llm")
    if isinstance(llm, dict) and isinstance(llm.get("model"), str):
        state["llm"] = {
            "backend": str(llm.get("backend") or "ollama"),
            "model": llm["model"],
            "base_url": llm.get("base_url") if isinstance(llm.get("base_url"), str) else None,
            "api_key_env": llm.get("api_key_env")
            if isinstance(llm.get("api_key_env"), str)
            else None,
            "allow_http": bool(llm.get("allow_http")),
            "adjudicate": bool(llm.get("adjudicate")),
            "key_typed": bool(llm.get("key_typed")),
        }
    elif isinstance(raw.get("llm_model"), str):  # wardcat-cli
        state["llm"] = {
            "backend": str(raw.get("llm_backend") or "ollama"),
            "model": raw["llm_model"],
            "base_url": raw.get("llm_base_url")
            if isinstance(raw.get("llm_base_url"), str)
            else None,
            "api_key_env": None,
            "allow_http": False,
            "adjudicate": False,
            "key_typed": False,
        }
    if isinstance(raw.get("salt_env"), str):
        state["salt_env"] = raw["salt_env"]
    if state["ner"] is None and "ner" in state["layers"]:
        state["layers"].remove("ner")
    if state["llm"] is None and "llm" in state["layers"]:
        state["layers"].remove("llm")
    if not state["layers"]:
        state["layers"] = ["regex"]
    return state


def path_for(session_id: str) -> Path:
    if not _ID.match(session_id):
        raise ConfigError(f"not a session id: {session_id!r}")
    return sessions_dir() / f"{session_id}.json"


def load(session_id: str) -> tuple[dict[str, Any], bool]:
    """The session, and whether its file carried a salt (a wardcat-cli file)."""
    path = path_for(session_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"no saved session {session_id!r}; see `wardcat sessions`") from None
    except (OSError, ValueError):
        raise ConfigError(f"session {session_id!r} cannot be read") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"session {session_id!r} cannot be read")
    raw.setdefault("id", session_id)
    return normalize(raw), bool(raw.get("salt"))


def all_sessions() -> list[dict[str, Any]]:
    """Every readable session, most recently used first."""
    found = []
    for path in sessions_dir().glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            raw.setdefault("id", path.stem)
            found.append(normalize(raw))
    return sorted(found, key=lambda s: s["updated_at"], reverse=True)


def save(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_private(path_for(state["id"]), json.dumps(state, indent=2) + "\n")


def write_private(path: Path, text: str) -> None:
    """Write *text* to a file only its owner can read, whatever mode it had."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    if os.name == "posix":
        os.chmod(path, 0o600)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "sessions",
        help="list the saved sessions of the interactive screen, or delete one",
        allow_abbrev=False,
    )
    p.add_argument("--delete", metavar="ID", help="delete this session")
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN

    if args.delete:
        path = path_for(args.delete)
        if not path.exists():
            raise ConfigError(f"no saved session {args.delete!r}")
        path.unlink()
        print(f"deleted session {args.delete}")
        return EXIT_CLEAN
    saved = all_sessions()
    if not saved:
        print("no saved sessions; run `wardcat` in a terminal to start one")
        return EXIT_CLEAN
    for s in saved:
        print(
            f"{s['id']}  {s['updated_at']}  layers: {', '.join(s['layers'])}  "
            f"{len(s['filters'])} filter(s)"
        )
    print("\nresume one with: wardcat --resume ID")
    return EXIT_CLEAN
