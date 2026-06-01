"""Chat API endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.models.schemas import (
    ChatRequest, ChatResponse, SessionCreate, SessionInfo,
    SessionRename, ChatMessage, MessageRole,
)
from backend.services import sessions as svc
from backend.services.llm import chat_stream

router = APIRouter()


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint via SSE."""
    session = svc.get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    sys_prompt = session.get("system_prompt", "")
    messages = []
    if sys_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt))
    for m in session.get("messages", []):
        messages.append(ChatMessage(role=MessageRole(m["role"]), content=m["content"]))
    messages.append(ChatMessage(role=MessageRole.USER, content=request.message))

    svc.add_message(request.session_id, ChatMessage(role=MessageRole.USER, content=request.message))

    async def generate():
        full_content = ""
        async for chunk in chat_stream(messages, request.temperature, request.max_tokens):
            full_content += chunk
            yield f"data: {chunk}\n\n"
        svc.add_message(request.session_id, ChatMessage(role=MessageRole.ASSISTANT, content=full_content))
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/sessions", response_model=SessionInfo)
async def create_session(req: SessionCreate):
    return svc.create_session(req.title, req.system_prompt)


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    return svc.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = svc.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.patch("/sessions/rename")
async def rename_session(req: SessionRename):
    svc.rename_session(req.session_id, req.title)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    svc.delete_session(session_id)
    return {"ok": True}
