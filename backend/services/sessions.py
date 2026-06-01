"""Chat session management — create, load, save, delete sessions."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from backend.config import SESSIONS_DIR
from backend.models.schemas import SessionInfo, ChatMessage


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def create_session(title: str = "New Chat", system_prompt: Optional[str] = None) -> SessionInfo:
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "id": session_id,
        "title": title,
        "system_prompt": system_prompt or "",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _session_path(session_id).write_text(json.dumps(session, indent=2))
    return SessionInfo(
        id=session_id,
        title=title,
        message_count=0,
        created_at=now,
        updated_at=now,
        system_prompt=session["system_prompt"],
    )


def get_session(session_id: str) -> Optional[dict]:
    path = _session_path(session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_sessions() -> list[SessionInfo]:
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(f.read_text())
        sessions.append(SessionInfo(
            id=data["id"],
            title=data["title"],
            message_count=len(data.get("messages", [])),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            system_prompt=data.get("system_prompt"),
        ))
    return sessions


def add_message(session_id: str, message: ChatMessage):
    session = get_session(session_id)
    if not session:
        return
    session["messages"].append({"role": message.role.value, "content": message.content})
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    _session_path(session_id).write_text(json.dumps(session, indent=2))


def rename_session(session_id: str, title: str):
    session = get_session(session_id)
    if not session:
        return
    session["title"] = title
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    _session_path(session_id).write_text(json.dumps(session, indent=2))


def delete_session(session_id: str):
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
