"""Hermes bridge API endpoints — message queue status and management."""

from fastapi import APIRouter
from backend.services.hermes_bridge import (
    get_pending_messages, get_queue_status, get_responses_for_session,
    INBOX_FILE, OUTBOX_FILE,
)

router = APIRouter()


@router.get("/inbox")
async def get_inbox():
    """Get all inbox messages (for debugging)."""
    import json
    if INBOX_FILE.exists():
        return json.loads(INBOX_FILE.read_text())
    return []


@router.get("/outbox")
async def get_outbox():
    """Get all outbox messages (for debugging)."""
    import json
    if OUTBOX_FILE.exists():
        return json.loads(OUTBOX_FILE.read_text())
    return []


@router.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get the status of a specific request."""
    return get_queue_status(request_id)


@router.get("/pending")
async def get_pending():
    """Get pending messages (for Hermes to process)."""
    return get_pending_messages()


@router.get("/session/{session_id}")
async def get_session_responses(session_id: str):
    """Get all responses for a session."""
    return get_responses_for_session(session_id)
