"""Hermes-to-Dashboard bridge — message queue for direct agent communication."""

import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from backend.config import DATA_DIR

QUEUE_DIR = DATA_DIR / "bridge_queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

INBOX_FILE = QUEUE_DIR / "inbox.json"    # Messages from dashboard -> Hermes
OUTBOX_FILE = QUEUE_DIR / "outbox.json"  # Messages from Hermes -> dashboard


def _load_json(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return []
    return []


def _save_json(path: Path, data: list):
    path.write_text(json.dumps(data, indent=2))


def submit_message(session_id: str, message: str, system_prompt: str = "") -> str:
    """Submit a chat message from the dashboard to Hermes' inbox.
    
    Returns a request_id that the dashboard can poll for responses.
    """
    request_id = str(uuid.uuid4())[:12]
    inbox = _load_json(INBOX_FILE)
    inbox.append({
        "request_id": request_id,
        "session_id": session_id,
        "message": message,
        "system_prompt": system_prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    })
    _save_json(INBOX_FILE, inbox)
    return request_id


def get_pending_messages() -> list:
    """Get all pending messages for Hermes to process.
    Called by Hermes via session_search or a cron job.
    """
    inbox = _load_json(INBOX_FILE)
    return [m for m in inbox if m["status"] == "pending"]


def mark_processing(request_id: str):
    """Mark a request as being processed by Hermes."""
    inbox = _load_json(INBOX_FILE)
    for m in inbox:
        if m["request_id"] == request_id:
            m["status"] = "processing"
            m["processing_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(INBOX_FILE, inbox)


def submit_response(request_id: str, session_id: str, content: str):
    """Submit Hermes' response back to the outbox."""
    outbox = _load_json(OUTBOX_FILE)
    outbox.append({
        "request_id": request_id,
        "session_id": session_id,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_json(OUTBOX_FILE, outbox)

    # Mark inbox item as completed
    inbox = _load_json(INBOX_FILE)
    for m in inbox:
        if m["request_id"] == request_id:
            m["status"] = "completed"
            m["completed_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(INBOX_FILE, inbox)


def get_response(request_id: str) -> Optional[dict]:
    """Get the response for a specific request_id. Called by dashboard polling."""
    outbox = _load_json(OUTBOX_FILE)
    for m in outbox:
        if m["request_id"] == request_id:
            return m
    return None


def get_responses_for_session(session_id: str) -> list:
    """Get all responses for a session."""
    outbox = _load_json(OUTBOX_FILE)
    return [m for m in outbox if m["session_id"] == session_id]


def get_queue_status(request_id: str) -> dict:
    """Get the current status of a request."""
    inbox = _load_json(INBOX_FILE)
    for m in inbox:
        if m["request_id"] == request_id:
            resp = get_response(request_id)
            if resp:
                return {"status": "completed", "response": resp}
            return {"status": m["status"]}
    return {"status": "not_found"}
