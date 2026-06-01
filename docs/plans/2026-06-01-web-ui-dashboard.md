# Project Glitch — Web UI Dashboard Implementation Plan

> **Goal:** Build a full-featured web UI dashboard for Project Glitch — an on-premise AI assistant. Chat interface, system monitoring, model management, RLHF training panel, and document management. All running locally, dark theme, streaming responses.

**Architecture:** Python FastAPI backend + vanilla HTML/CSS/JS frontend. FastAPI serves the API and static files. Frontend talks to backend via REST + Server-Sent Events (SSE) for streaming chat. System monitoring reads hardware stats via psutil and nvidia-ml-py.

**Tech Stack:** Python 3.11+, FastAPI, psutil, nvidia-ml-py3 (optional), asyncio, vanilla JS (no framework), CSS custom properties for theming.

---

## Phase 1: Project Structure & Backend Setup

### Task 1: Create project scaffold

**Files:**
- Create: `backend/main.py`
- Create: `backend/config.py`
- Create: `backend/requirements.txt`
- Create: `backend/__init__.py`
- Create: `frontend/index.html`
- Create: `frontend/css/style.css`
- Create: `frontend/js/app.js`
- Create: `.gitignore`

**Directory structure:**
```
project-glitch/
├── backend/
│   ├── main.py          # FastAPI app entry
│   ├── config.py         # Settings, paths, defaults
│   ├── requirements.txt  # Python deps
│   ├── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py       # Chat endpoints
│   │   ├── system.py     # Hardware monitoring
│   │   ├── models.py     # Model management
│   │   ├── training.py   # RLHF training
│   │   └── documents.py  # Knowledge base
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py        # LLM inference abstraction
│   │   ├── hardware.py   # GPU/CPU/RAM monitoring
│   │   └── sessions.py   # Chat session management
│   └── models/
│       ├── __init__.py
│       └── schemas.py    # Pydantic models
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js
│       ├── chat.js
│       ├── monitor.js
│       ├── settings.js
│       └── api.js
├── docs/
├── scripts/
├── data/
│   ├── sessions/         # Chat history JSON
│   ├── models/           # GGUF model files
│   └── documents/        # Uploaded docs
├── README.md
└── .gitignore
```

**Step 1: Create directories**

```bash
cd /c/Users/Naff
mkdir -p project-glitch/{backend/{routers,services,models},frontend/{css,js},data/{sessions,models,documents},docs,scripts}
```

**Step 2: Write requirements.txt**

```txt
fastapi==0.115.0
uvicorn==0.34.0
pydantic==2.10.0
psutil==6.1.0
python-multipart==0.0.18
aiofiles==24.1.0
```

**Step 3: Write config.py**

```python
"""Project Glitch configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MODELS_DIR = DATA_DIR / "models"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Create dirs if missing
for d in [SESSIONS_DIR, MODELS_DIR, DOCUMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "mistral-7b-v0.2.Q4_K_M.gguf"
DEFAULT_SYSTEM_PROMPT = "You are Glitch, a helpful AI assistant running locally."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CONTEXT_LENGTH = 4096

# LLM backend settings
LLM_BACKEND = "llama_cpp"  # or "openai_compatible"
LLM_HOST = "localhost"
LLM_PORT = 8080  # llama.cpp server port
```

**Step 4: Write main.py**

```python
"""FastAPI entry point for Project Glitch dashboard."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.routers import chat, system, models, training, documents

app = FastAPI(title="Project Glitch", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])

# Static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok", "name": "Project Glitch", "version": "1.0.0"}
```

**Step 5: Write .gitignore**

```gitignore
__pycache__/
*.pyc
.env
venv/
data/models/*.gguf
data/sessions/*.json
data/documents/
dist/
build/
node_modules/
.DS_Store
Thumbs.db
```

**Step 6: Commit**

```bash
git add .
git commit -m "feat: project scaffold for web UI dashboard"
```

---

## Phase 2: Pydantic Schemas & API Layer

### Task 2: Define all Pydantic models

**Files:**
- Create: `backend/models/schemas.py`

**Step 1: Write schemas.py**

```python
"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# --- Chat ---

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class ChatResponse(BaseModel):
    session_id: str
    role: str = "assistant"
    content: str
    tokens_used: Optional[int] = None
    tokens_per_second: Optional[float] = None
    response_time_ms: Optional[int] = None


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"
    system_prompt: Optional[str] = None


class SessionInfo(BaseModel):
    id: str
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str
    system_prompt: Optional[str] = None


class SessionRename(BaseModel):
    session_id: str
    title: str


# --- System ---

class GPUInfo(BaseModel):
    name: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    temperature_c: Optional[int] = None
    utilization_pct: Optional[int] = None


class SystemStats(BaseModel):
    cpu_percent: float
    cpu_count: int
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    gpu: Optional[GPUInfo] = None
    uptime_seconds: int


class InferenceStats(BaseModel):
    model_name: Optional[str] = None
    context_length: int
    tokens_per_second: float = 0.0
    total_tokens_generated: int = 0
    is_generating: bool = False


# --- Models ---

class ModelInfo(BaseModel):
    name: str
    path: str
    size_gb: float
    quantization: Optional[str] = None
    context_length: int = 4096
    is_loaded: bool = False


class ModelLoadRequest(BaseModel):
    model_name: str
    context_length: Optional[int] = 4096


class SettingsUpdate(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_length: Optional[int] = None
    system_prompt: Optional[str] = None


# --- Training ---

class TrainingRequest(BaseModel):
    base_model: str
    dataset_path: str
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    lora_rank: int = 16


class TrainingStatus(BaseModel):
    job_id: str
    status: str  # "idle", "running", "completed", "failed"
    progress_pct: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    started_at: Optional[str] = None
    error: Optional[str] = None


# --- Documents ---

class DocumentInfo(BaseModel):
    name: str
    path: str
    size_kb: float
    uploaded_at: str
    status: str  # "uploading", "processing", "indexed", "error"
    chunks: int = 0


class DocumentUploadResponse(BaseModel):
    success: bool
    document: Optional[DocumentInfo] = None
    error: Optional[str] = None
```

**Step 2: Commit**

```bash
git add backend/models/schemas.py
git commit -m "feat: add Pydantic schemas for all API models"
```

---

## Phase 3: Backend Services

### Task 3: Hardware monitoring service

**Files:**
- Create: `backend/services/hardware.py`

```python
"""Hardware monitoring — CPU, RAM, GPU stats."""

import psutil
import time
from typing import Optional
from backend.models.schemas import GPUInfo, SystemStats

_start_time = time.time()


def get_system_stats() -> SystemStats:
    """Get current system resource usage."""
    mem = psutil.virtual_memory()

    gpu = _get_gpu_info()

    return SystemStats(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        cpu_count=psutil.cpu_count(),
        ram_total_gb=round(mem.total / (1024**3), 1),
        ram_used_gb=round(mem.used / (1024**3), 1),
        ram_percent=mem.percent,
        gpu=gpu,
        uptime_seconds=int(time.time() - _start_time),
    )


def _get_gpu_info() -> Optional[GPUInfo]:
    """Try to get NVIDIA GPU info. Returns None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)

        return GPUInfo(
            name=name.decode() if isinstance(name, bytes) else name,
            memory_total_mb=mem_info.total // (1024**2),
            memory_used_mb=mem_info.used // (1024**2),
            memory_free_mb=mem_info.free // (1024**2),
            temperature_c=temp,
            utilization_pct=util.gpu,
        )
    except Exception:
        return None
```

### Task 4: Session management service

**Files:**
- Create: `backend/services/sessions.py`

```python
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
    info = SessionInfo(**session, message_count=0, system_prompt=session["system_prompt"])
    return info


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
```

### Task 5: LLM inference service

**Files:**
- Create: `backend/services/llm.py`

```python
"""LLM inference service — talks to llama.cpp server or runs inline."""

import httpx
import time
from typing import Optional, AsyncGenerator
from backend.config import LLM_HOST, LLM_PORT
from backend.models.schemas import InferenceStats, ChatMessage

_current_model: Optional[str] = None
_total_tokens: int = 0
_is_generating: bool = False


def get_inference_stats() -> InferenceStats:
    return InferenceStats(
        model_name=_current_model,
        context_length=4096,
        tokens_per_second=0.0,
        total_tokens_generated=_total_tokens,
        is_generating=_is_generating,
    )


async def chat_stream(
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """Stream chat response from llama.cpp server via SSE."""
    global _is_generating, _total_tokens
    _is_generating = True
    start_time = time.time()
    token_count = 0

    url = f"http://{LLM_HOST}:{LLM_PORT}/completion"
    payload = {
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                token_count += 1
                                yield content
                        except Exception:
                            pass
    except Exception as e:
        yield f"\n[Error: Could not connect to LLM server — {e}]"
    finally:
        _total_tokens += token_count
        _is_generating = False


def load_model(model_name: str, context_length: int = 4096):
    """Signal which model is loaded. Actual loading is handled by llama.cpp server."""
    global _current_model
    _current_model = model_name
```

**Commit:** `feat: add hardware, session, and LLM services`

---

## Phase 4: API Routers

### Task 6: Chat router

**Files:**
- Create: `backend/routers/chat.py`

```python
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

    # Build messages
    sys_prompt = session.get("system_prompt", "")
    messages = []
    if sys_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt))
    for m in session.get("messages", []):
        messages.append(ChatMessage(role=MessageRole(m["role"]), content=m["content"]))
    messages.append(ChatMessage(role=MessageRole.USER, content=request.message))

    # Save user message
    svc.add_message(request.session_id, ChatMessage(role=MessageRole.USER, content=request.message))

    async def generate():
        full_content = ""
        async for chunk in chat_stream(messages, request.temperature, request.max_tokens):
            full_content += chunk
            yield f"data: {chunk}\n\n"
        # Save assistant response
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
```

### Task 7: System router

**Files:**
- Create: `backend/routers/system.py`

```python
"""System monitoring endpoints."""

from fastapi import APIRouter
from backend.models.schemas import SystemStats
from backend.services.hardware import get_system_stats

router = APIRouter()


@router.get("/stats", response_model=SystemStats)
async def system_stats():
    return get_system_stats()


@router.get("/inference")
async def inference_stats():
    from backend.services.llm import get_inference_stats
    return get_inference_stats()
```

### Task 8: Models router

**Files:**
- Create: `backend/routers/models.py`

```python
"""Model management endpoints."""

import os
from fastapi import APIRouter
from backend.models.schemas import ModelInfo, ModelLoadRequest, SettingsUpdate
from backend.config import MODELS_DIR, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_CONTEXT_LENGTH
from backend.services.llm import load_model

router = APIRouter()

_settings = {
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "context_length": DEFAULT_CONTEXT_LENGTH,
}


@router.get("/", response_model=list[ModelInfo])
async def list_models():
    models = []
    for f in MODELS_DIR.glob("*.gguf"):
        size_gb = round(f.stat().st_size / (1024**3), 2)
        # Try to detect quantization from filename
        name = f.stem
        quant = None
        for q in ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", "Q5_0", "F16", "F32"]:
            if q.lower() in name.lower():
                quant = q
                break
        models.append(ModelInfo(
            name=name,
            path=str(f),
            size_gb=size_gb,
            quantization=quant,
            context_length=4096,
        ))
    return models


@router.post("/load")
async def load_model_endpoint(req: ModelLoadRequest):
    model_path = MODELS_DIR / req.model_name
    if not model_path.exists():
        return {"error": f"Model not found: {req.model_name}"}
    load_model(req.model_name, req.context_length)
    return {"ok": True, "model": req.model_name}


@router.get("/settings")
async def get_settings():
    return _settings


@router.patch("/settings")
async def update_settings(req: SettingsUpdate):
    if req.temperature is not None:
        _settings["temperature"] = req.temperature
    if req.max_tokens is not None:
        _settings["max_tokens"] = req.max_tokens
    if req.context_length is not None:
        _settings["context_length"] = req.context_length
    return _settings
```

### Task 9: Training router

**Files:**
- Create: `backend/routers/training.py`

```python
"""RLHF training endpoints."""

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter
from backend.models.schemas import TrainingRequest, TrainingStatus
from backend.config import MODELS_DIR

router = APIRouter()

_training_jobs: dict[str, TrainingStatus] = {}


@router.post("/start", response_model=TrainingStatus)
async def start_training(req: TrainingRequest):
    job_id = str(uuid.uuid4())[:8]
    status = TrainingStatus(
        job_id=job_id,
        status="running",
        progress_pct=0.0,
        current_epoch=0,
        total_epochs=req.epochs,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    _training_jobs[job_id] = status
    # TODO: kick off actual training subprocess
    return status


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def training_status(job_id: str):
    if job_id not in _training_jobs:
        return TrainingStatus(job_id=job_id, status="not_found")
    return _training_jobs[job_id]


@router.get("/jobs", response_model=list[TrainingStatus])
async def list_training_jobs():
    return list(_training_jobs.values())


@router.post("/stop/{job_id}")
async def stop_training(job_id: str):
    if job_id in _training_jobs:
        _training_jobs[job_id].status = "stopped"
    return {"ok": True}
```

### Task 10: Documents router

**Files:**
- Create: `backend/routers/documents.py`

```python
"""Document / knowledge base endpoints."""

import json
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File
from backend.models.schemas import DocumentInfo, DocumentUploadResponse
from backend.config import DOCUMENTS_DIR

router = APIRouter()

_index_file = DOCUMENTS_DIR / "_index.json"


def _load_index() -> dict:
    if _index_file.exists():
        return json.loads(_index_file.read_text())
    return {"documents": []}


def _save_index(index: dict):
    _index_file.write_text(json.dumps(index, indent=2))


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    index = _load_index()
    return [DocumentInfo(**d) for d in index.get("documents", [])]


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        dest = DOCUMENTS_DIR / file.filename
        dest.write_bytes(content)
        size_kb = round(len(content) / 1024, 1)

        doc = DocumentInfo(
            name=file.filename,
            path=str(dest),
            size_kb=size_kb,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            status="processing",
        )

        # Update index
        index = _load_index()
        index["documents"].append(doc.model_dump())
        _save_index(index)

        # TODO: trigger background indexing
        doc.status = "indexed"
        doc.chunks = 0

        return DocumentUploadResponse(success=True, document=doc)
    except Exception as e:
        return DocumentUploadResponse(success=False, error=str(e))


@router.delete("/{filename}")
async def delete_document(filename: str):
    dest = DOCUMENTS_DIR / filename
    if dest.exists():
        dest.unlink()
    index = _load_index()
    index["documents"] = [d for d in index["documents"] if d["name"] != filename]
    _save_index(index)
    return {"ok": True}
```

**Commit:** `feat: add all API routers (chat, system, models, training, documents)`

---

## Phase 5: Frontend

### Task 11: HTML layout and navigation

**Files:**
- Create: `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Glitch — Dashboard</title>
  <link rel="stylesheet" href="css/style.css">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>">
</head>
<body class="noise">
  <div class="app-layout">
    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="logo">
          <span class="logo-icon">⚡</span>
          <span class="logo-text">Glitch<span class="dot">.</span></span>
        </div>
      </div>

      <nav class="sidebar-nav">
        <button class="nav-item active" data-page="chat">
          <span class="nav-icon">💬</span>
          <span>Chat</span>
        </button>
        <button class="nav-item" data-page="monitor">
          <span class="nav-icon">📊</span>
          <span>Monitor</span>
        </button>
        <button class="nav-item" data-page="models">
          <span class="nav-icon">🧠</span>
          <span>Models</span>
        </button>
        <button class="nav-item" data-page="training">
          <span class="nav-icon">🔥</span>
          <span>Training</span>
        </button>
        <button class="nav-item" data-page="documents">
          <span class="nav-icon">📄</span>
          <span>Documents</span>
        </button>
        <button class="nav-item" data-page="settings">
          <span class="nav-icon">⚙️</span>
          <span>Settings</span>
        </button>
      </nav>

      <div class="sidebar-footer">
        <div class="system-mini" id="sidebar-system">
          <div class="mini-stat">
            <span class="mini-label">CPU</span>
            <div class="mini-bar"><div class="mini-bar-fill" id="cpu-mini">0%</div></div>
          </div>
          <div class="mini-stat">
            <span class="mini-label">RAM</span>
            <div class="mini-bar"><div class="mini-bar-fill" id="ram-mini">0%</div></div>
          </div>
        </div>
        <div class="version">v1.0.0</div>
      </div>
    </aside>

    <!-- Main content -->
    <main class="main-content">
      <!-- Chat page -->
      <div class="page active" id="page-chat">
        <div class="chat-layout">
          <!-- Session list -->
          <div class="session-list">
            <div class="session-list-header">
              <h3>Chats</h3>
              <button class="btn-icon" id="new-chat-btn" title="New chat">+</button>
            </div>
            <div class="sessions" id="sessions-list">
              <!-- Filled by JS -->
            </div>
          </div>

          <!-- Chat area -->
          <div class="chat-area">
            <div class="chat-header" id="chat-header">
              <span class="chat-title" id="chat-title">No chat selected</span>
              <button class="btn-icon btn-danger" id="delete-chat-btn" title="Delete chat" style="display:none">🗑️</button>
            </div>
            <div class="messages" id="messages">
              <div class="empty-state" id="empty-state">
                <div class="empty-icon">🤖</div>
                <p>Start a new chat or select an existing one</p>
              </div>
            </div>
            <div class="chat-input-area">
              <div class="chat-input-wrap">
                <textarea id="chat-input" placeholder="Message Glitch..." rows="1"></textarea>
                <button class="btn-send" id="send-btn" title="Send">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                </button>
              </div>
              <div class="input-hints">
                <span>Enter to send</span>
                <span>Shift+Enter for new line</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Monitor page -->
      <div class="page" id="page-monitor">
        <div class="page-header">
          <h1>System Monitor</h1>
          <p>Real-time hardware and inference stats</p>
        </div>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-header">
              <span class="stat-icon">🖥️</span>
              <span class="stat-title">CPU</span>
            </div>
            <div class="stat-value" id="cpu-stat">0%</div>
            <div class="stat-bar"><div class="stat-bar-fill" id="cpu-bar"></div></div>
            <div class="stat-sub" id="cpu-cores">— cores</div>
          </div>
          <div class="stat-card">
            <div class="stat-header">
              <span class="stat-icon">💾</span>
              <span class="stat-title">RAM</span>
            </div>
            <div class="stat-value" id="ram-stat">0 / 0 GB</div>
            <div class="stat-bar"><div class="stat-bar-fill" id="ram-bar"></div></div>
            <div class="stat-sub" id="ram-pct">0%</div>
          </div>
          <div class="stat-card" id="gpu-card" style="display:none">
            <div class="stat-header">
              <span class="stat-icon">🎮</span>
              <span class="stat-title" id="gpu-name">GPU</span>
            </div>
            <div class="stat-value" id="gpu-mem">0 / 0 MB</div>
            <div class="stat-bar"><div class="stat-bar-fill" id="gpu-bar"></div></div>
            <div class="stat-sub" id="gpu-temp">—°C</div>
          </div>
          <div class="stat-card">
            <div class="stat-header">
              <span class="stat-icon">⚡</span>
              <span class="stat-title">Inference</span>
            </div>
            <div class="stat-value" id="inf-model">—</div>
            <div class="stat-sub" id="inf-status">Idle</div>
            <div class="stat-sub" id="inf-tokens">0 tokens total</div>
          </div>
        </div>

        <div class="monitor-logs">
          <h3>System Info</h3>
          <div class="log-output" id="system-logs">
            <div class="log-line">Uptime: <span id="uptime">—</span></div>
            <div class="log-line">GPU Util: <span id="gpu-util">—</span></div>
            <div class="log-line">Context: <span id="ctx-length">4096</span></div>
          </div>
        </div>
      </div>

      <!-- Models page -->
      <div class="page" id="page-models">
        <div class="page-header">
          <h1>Model Manager</h1>
          <p>Load and switch between GGUF models</p>
        </div>
        <div class="model-list" id="model-list">
          <!-- Filled by JS -->
        </div>
        <div class="model-dropzone" id="model-dropzone">
          <p>Drop .gguf files here or click to browse</p>
          <input type="file" id="model-file-input" accept=".gguf" style="display:none">
        </div>
      </div>

      <!-- Training page -->
      <div class="page" id="page-training">
        <div class="page-header">
          <h1>RLHF Training</h1>
          <p>Fine-tune models with reinforcement learning from human feedback</p>
        </div>
        <div class="training-layout">
          <div class="training-form card">
            <h3>Start Training Job</h3>
            <div class="form-group">
              <label>Base Model</label>
              <select id="train-model">
                <option>Loading models...</option>
              </select>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Epochs</label>
                <input type="number" id="train-epochs" value="3" min="1" max="20">
              </div>
              <div class="form-group">
                <label>Learning Rate</label>
                <input type="number" id="train-lr" value="0.0002" step="0.0001">
              </div>
              <div class="form-group">
                <label>Batch Size</label>
                <input type="number" id="train-batch" value="4" min="1" max="32">
              </div>
              <div class="form-group">
                <label>LoRA Rank</label>
                <input type="number" id="train-lora" value="16" min="1" max="64">
              </div>
            </div>
            <div class="form-group">
              <label>Dataset Path</label>
              <input type="text" id="train-dataset" placeholder="data/training_data.jsonl">
            </div>
            <button class="btn btn-primary" id="start-training-btn">Start Training</button>
          </div>
          <div class="training-jobs card">
            <h3>Training Jobs</h3>
            <div id="training-jobs-list">
              <p class="empty-text">No training jobs yet</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Documents page -->
      <div class="page" id="page-documents">
        <div class="page-header">
          <h1>Knowledge Base</h1>
          <p>Upload documents for the AI to reference</p>
        </div>
        <div class="doc-upload" id="doc-dropzone">
          <div class="upload-icon">📄</div>
          <p>Drop files here or click to upload</p>
          <p class="upload-hint">.txt, .md, .pdf — max 50MB</p>
          <input type="file" id="doc-file-input" multiple style="display:none">
        </div>
        <div class="doc-list" id="doc-list">
          <!-- Filled by JS -->
        </div>
      </div>

      <!-- Settings page -->
      <div class="page" id="page-settings">
        <div class="page-header">
          <h1>Settings</h1>
          <p>Configure model parameters and defaults</p>
        </div>
        <div class="settings-grid">
          <div class="settings-card card">
            <h3>Generation</h3>
            <div class="form-group">
              <label>Temperature: <span id="temp-val">0.7</span></label>
              <input type="range" id="setting-temp" min="0" max="2" step="0.05" value="0.7">
            </div>
            <div class="form-group">
              <label>Max Tokens: <span id="maxtok-val">2048</span></label>
              <input type="range" id="setting-maxtok" min="256" max="8192" step="256" value="2048">
            </div>
            <div class="form-group">
              <label>Context Length: <span id="ctx-val">4096</span></label>
              <input type="range" id="setting-ctx" min="1024" max="32768" step="1024" value="4096">
            </div>
          </div>
          <div class="settings-card card">
            <h3>System Prompt</h3>
            <textarea id="setting-sysprompt" rows="8" placeholder="You are a helpful assistant...">You are Glitch, a helpful AI assistant running locally.</textarea>
            <button class="btn btn-primary" id="save-settings-btn" style="margin-top:16px">Save Settings</button>
          </div>
          <div class="settings-card card">
            <h3>LLM Server</h3>
            <div class="form-group">
              <label>Backend URL</label>
              <input type="text" id="setting-llm-url" value="http://localhost:8080">
            </div>
            <button class="btn btn-outline" id="test-connection-btn">Test Connection</button>
            <div id="connection-result" style="margin-top:8px;font-size:0.85rem;"></div>
          </div>
          <div class="settings-card card">
            <h3>About</h3>
            <p><strong>Project Glitch</strong> v1.0.0</p>
            <p>On-premise AI assistant dashboard.</p>
            <p>Built by <a href="https://github.com/Lijnaff" target="_blank" style="color:var(--accent-cyan)">Nafyad Fantaye</a></p>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script src="js/api.js"></script>
  <script src="js/chat.js"></script>
  <script src="js/monitor.js"></script>
  <script src="js/settings.js"></script>
  <script src="js/app.js"></script>
</body>
</html>
```

### Task 12: Full CSS stylesheet

**Files:**
- Create: `frontend/css/style.css`

```css
/* ============================================
   Project Glitch — Dashboard Styles
   Dark theme with cyan/amber accents
   ============================================ */

:root {
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-tertiary: #1a1a25;
  --bg-hover: #222233;
  --border: #2a2a3a;
  --border-light: #3a3a4a;
  --text-primary: #e8e8f0;
  --text-secondary: #9898a8;
  --text-muted: #585868;
  --accent-cyan: #06b6d4;
  --accent-cyan-dim: rgba(6, 182, 212, 0.15);
  --accent-amber: #f59e0b;
  --accent-amber-dim: rgba(245, 158, 11, 0.15);
  --accent-green: #22c55e;
  --accent-red: #ef4444;
  --accent-purple: #a855f7;
  --radius: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  --transition: 0.2s ease;
  --sidebar-width: 220px;
  --sidebar-collapsed: 56px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body {
  height: 100%;
  font-family: var(--font);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.5;
  overflow: hidden;
}

body.noise::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
}

/* ---- Layout ---- */

.app-layout {
  display: flex;
  height: 100vh;
  position: relative;
  z-index: 1;
}

/* ---- Sidebar ---- */

.sidebar {
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  transition: width var(--transition);
  overflow: hidden;
}

.sidebar-header {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--text-primary);
}

.logo-icon { font-size: 1.4rem; }
.logo-text .dot { color: var(--accent-cyan); }

.sidebar-nav {
  flex: 1;
  padding: 12px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition);
  border: none;
  background: none;
  font-size: 0.9rem;
  font-weight: 500;
  text-align: left;
  width: 100%;
  font-family: var(--font);
}

.nav-item:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.nav-item.active {
  background: var(--accent-cyan-dim);
  color: var(--accent-cyan);
}

.nav-icon {
  font-size: 1.1rem;
  width: 24px;
  text-align: center;
}

.sidebar-footer {
  padding: 12px 12px 16px;
  border-top: 1px solid var(--border);
}

.system-mini {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 8px;
}

.mini-stat {
  display: flex;
  align-items: center;
  gap: 8px;
}

.mini-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  width: 28px;
}

.mini-bar {
  flex: 1;
  height: 4px;
  background: var(--bg-tertiary);
  border-radius: 2px;
  overflow: hidden;
}

.mini-bar-fill {
  height: 100%;
  background: var(--accent-cyan);
  border-radius: 2px;
  transition: width 0.5s ease;
  font-size: 0;
}

.version {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-align: center;
}

/* ---- Main content ---- */

.main-content {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.page { display: none; }
.page.active { display: block; }

.page-header {
  padding: 32px 32px 0;
  margin-bottom: 24px;
}

.page-header h1 {
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.page-header p {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

/* ---- Buttons ---- */

.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  border-radius: var(--radius);
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all var(--transition);
  font-family: var(--font);
  text-decoration: none;
}

.btn-primary {
  background: var(--accent-cyan);
  color: #000;
}

.btn-primary:hover {
  background: #22d3f0;
}

.btn-outline {
  background: transparent;
  border-color: var(--border);
  color: var(--text-secondary);
}

.btn-outline:hover {
  border-color: var(--accent-cyan);
  color: var(--accent-cyan);
}

.btn-danger {
  color: var(--accent-red);
}

.btn-danger:hover {
  background: rgba(239, 68, 68, 0.1);
}

.btn-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: var(--radius);
  border: none;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 1rem;
  transition: all var(--transition);
  font-family: var(--font);
}

.btn-icon:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

/* ---- Chat layout ---- */

.chat-layout {
  display: flex;
  height: 100vh;
}

.session-list {
  width: 260px;
  min-width: 260px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
}

.session-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border);
}

.session-list-header h3 {
  font-size: 1rem;
  font-weight: 600;
}

.sessions {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.session-item {
  padding: 10px 12px;
  border-radius: var(--radius);
  cursor: pointer;
  transition: all var(--transition);
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.session-item:hover { background: var(--bg-hover); }
.session-item.active {
  background: var(--accent-cyan-dim);
  color: var(--accent-cyan);
}

.session-item-title {
  font-size: 0.85rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.session-item-meta {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 2px;
}

/* ---- Chat area ---- */

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.chat-title {
  font-size: 1rem;
  font-weight: 600;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
}

.empty-state .empty-icon {
  font-size: 3rem;
  margin-bottom: 12px;
  opacity: 0.5;
}

.message {
  display: flex;
  gap: 12px;
  max-width: 800px;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.user { align-self: flex-end; flex-direction: row-reverse; }

.message-avatar {
  width: 32px;
  height: 32px;
  min-width: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.9rem;
  background: var(--bg-tertiary);
}

.message.assistant .message-avatar { background: var(--accent-cyan-dim); }
.message.user .message-avatar { background: var(--accent-amber-dim); }

.message-content {
  padding: 12px 16px;
  border-radius: var(--radius-lg);
  font-size: 0.9rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.message.user .message-content {
  background: var(--accent-amber-dim);
  border: 1px solid rgba(245, 158, 11, 0.15);
  color: var(--text-primary);
}

.message.assistant .message-content {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
}

.message.assistant .message-content.streaming::after {
  content: '▊';
  animation: blink 0.8s infinite;
  color: var(--accent-cyan);
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* ---- Chat input ---- */

.chat-input-area {
  padding: 16px 24px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}

.chat-input-wrap {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 8px;
  transition: border-color var(--transition);
}

.chat-input-wrap:focus-within {
  border-color: var(--accent-cyan);
}

#chat-input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-family: var(--font);
  font-size: 0.9rem;
  resize: none;
  max-height: 150px;
  line-height: 1.5;
  padding: 4px;
}

#chat-input::placeholder {
  color: var(--text-muted);
}

.btn-send {
  width: 36px;
  height: 36px;
  min-width: 36px;
  border-radius: var(--radius);
  border: none;
  background: var(--accent-cyan);
  color: #000;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}

.btn-send:hover { background: #22d3f0; }
.btn-send:disabled { opacity: 0.4; cursor: not-allowed; }

.input-hints {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  font-size: 0.7rem;
  color: var(--text-muted);
  padding: 0 4px;
}

/* ---- Stats / Monitor ---- */

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
  padding: 0 32px;
}

.stat-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
}

.stat-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.stat-icon { font-size: 1.2rem; }
.stat-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.stat-value {
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.stat-bar {
  height: 6px;
  background: var(--bg-tertiary);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 6px;
}

.stat-bar-fill {
  height: 100%;
  background: var(--accent-cyan);
  border-radius: 3px;
  transition: width 0.5s ease;
}

.stat-sub {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.monitor-logs {
  margin: 24px 32px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
}

.monitor-logs h3 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 12px;
}

.log-output {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text-secondary);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.log-line {
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
}

.log-line span {
  color: var(--accent-cyan);
}

/* ---- Models page ---- */

.model-list {
  padding: 0 32px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  transition: all var(--transition);
}

.model-item:hover { border-color: var(--border-light); }

.model-info { flex: 1; }

.model-name {
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 4px;
}

.model-meta {
  display: flex;
  gap: 12px;
  font-size: 0.8rem;
  color: var(--text-muted);
}

.model-badge {
  padding: 4px 10px;
  border-radius: 100px;
  font-size: 0.75rem;
  font-weight: 500;
}

.model-badge.loaded {
  background: var(--accent-cyan-dim);
  color: var(--accent-cyan);
}

.model-badge.available {
  background: var(--bg-tertiary);
  color: var(--text-muted);
}

.model-dropzone {
  margin: 24px 32px;
  padding: 48px;
  border: 2px dashed var(--border);
  border-radius: var(--radius-xl);
  text-align: center;
  color: var(--text-muted);
  cursor: pointer;
  transition: all var(--transition);
}

.model-dropzone:hover {
  border-color: var(--accent-cyan);
  color: var(--text-secondary);
}

.model-dropzone.drag-over {
  border-color: var(--accent-cyan);
  background: var(--accent-cyan-dim);
}

/* ---- Training page ---- */

.training-layout {
  padding: 0 32px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
}

.card h3 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 16px;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.form-group input,
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 10px 12px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-primary);
  font-family: var(--font);
  font-size: 0.9rem;
  outline: none;
  transition: border-color var(--transition);
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  border-color: var(--accent-cyan);
}

.form-group textarea {
  resize: vertical;
  min-height: 80px;
  line-height: 1.6;
}

.form-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px;
}

/* ---- Progress bar ---- */

.progress-bar {
  height: 8px;
  background: var(--bg-tertiary);
  border-radius: 4px;
  overflow: hidden;
  margin: 8px 0;
}

.progress-bar-fill {
  height: 100%;
  background: var(--accent-cyan);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-bar-fill.green { background: var(--accent-green); }
.progress-bar-fill.amber { background: var(--accent-amber); }
.progress-bar-fill.red { background: var(--accent-red); }

/* ---- Documents page ---- */

.doc-upload {
  margin: 0 32px;
  padding: 48px;
  border: 2px dashed var(--border);
  border-radius: var(--radius-xl);
  text-align: center;
  cursor: pointer;
  transition: all var(--transition);
  margin-bottom: 24px;
}

.doc-upload:hover {
  border-color: var(--accent-cyan);
}

.upload-icon {
  font-size: 2rem;
  margin-bottom: 8px;
}

.upload-hint {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 4px;
}

.doc-list {
  padding: 0 32px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.doc-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.doc-name {
  font-weight: 500;
  font-size: 0.9rem;
}

.doc-meta {
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* ---- Settings ---- */

.settings-grid {
  padding: 0 32px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}

.settings-card .form-group input[type="range"] {
  -webkit-appearance: none;
  height: 4px;
  background: var(--bg-tertiary);
  border-radius: 2px;
  cursor: pointer;
  padding: 0;
}

.settings-card .form-group input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent-cyan);
}

/* ---- Scrollbar ---- */

::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--border-light);
}

/* ---- Range input val display ---- */

.form-group label span {
  color: var(--accent-cyan);
  font-family: var(--font-mono);
}

/* ---- Text utilities ---- */

.empty-text {
  color: var(--text-muted);
  font-size: 0.9rem;
  text-align: center;
  padding: 24px;
}

/* ---- Connection status indicator ---- */

.conn-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 6px;
}

.conn-dot.ok { background: var(--accent-green); }
.conn-dot.bad { background: var(--accent-red); }

/* ---- Responsive ---- */

@media (max-width: 768px) {
  .sidebar {
    width: var(--sidebar-collapsed);
    min-width: var(--sidebar-collapsed);
  }
  .nav-item span:not(.nav-icon),
  .logo-text,
  .sidebar-footer {
    display: none;
  }
  .session-list { width: 200px; min-width: 200px; }
  .training-layout { grid-template-columns: 1fr; }
}
```

### Task 13: Frontend JavaScript (API layer + app logic)

**Files:**
- Create: `frontend/js/api.js`

```javascript
/** API client for Project Glitch backend */
const BASE = '';

const api = {
  async get(url) {
    const r = await fetch(BASE + url);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(BASE + url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
  async patch(url, body) {
    const r = await fetch(BASE + url, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  },
  async del(url) {
    await fetch(BASE + url, { method: 'DELETE' });
  },
  postFormData(url, formData) {
    return fetch(BASE + url, { method: 'POST', body: formData });
  },
};

// Chat
api.chat = {
  getSessions: () => api.get('/api/chat/sessions'),
  getSession: (id) => api.get(`/api/chat/sessions/${id}`),
  createSession: (title, prompt) => api.post('/api/chat/sessions', { title, system_prompt: prompt }),
  renameSession: (id, title) => api.patch('/api/chat/sessions/rename', { session_id: id, title }),
  deleteSession: (id) => api.del(`/api/chat/sessions/${id}`),
  stream: (session_id, message, temp, maxTok) =>
    fetch(`${BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id, message, temperature: temp, max_tokens: maxTok }),
    }),
};

// System
api.system = {
  stats: () => api.get('/api/system/stats'),
  inference: () => api.get('/api/system/inference'),
};

// Models
api.models = {
  list: () => api.get('/api/models/'),
  load: (name, ctx) => api.post('/api/models/load', { model_name: name, context_length: ctx }),
  settings: () => api.get('/api/models/settings'),
  updateSettings: (s) => api.patch('/api/models/settings', s),
};

// Training
api.training = {
  start: (cfg) => api.post('/api/training/start', cfg),
  status: (id) => api.get(`/api/training/status/${id}`),
  list: () => api.get('/api/training/jobs'),
  stop: (id) => api.post(`/api/training/stop/${id}`),
};

// Documents
api.docs = {
  list: () => api.get('/api/documents/'),
  upload: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return api.postFormData('/api/documents/upload', fd).then(r => r.json());
  },
  delete: (name) => api.del(`/api/documents/${name}`),
};
```

**Files:**
- Create: `frontend/js/chat.js`

```javascript
/** Chat logic — sessions, messages, streaming */
let currentSession = null;
let settings = { temperature: 0.7, max_tokens: 2048 };

async function loadSessions() {
  const sessions = await api.chat.getSessions();
  const list = document.getElementById('sessions-list');
  list.innerHTML = sessions.map(s =>
    `<div class="session-item ${currentSession === s.id ? 'active' : ''}" data-id="${s.id}">
      <div>
        <div class="session-item-title">${escHtml(s.title)}</div>
        <div class="session-item-meta">${s.message_count} messages · ${formatDate(s.updated_at)}</div>
      </div>
    </div>`
  ).join('');

  list.querySelectorAll('.session-item').forEach(el => {
    el.addEventListener('click', () => selectSession(el.dataset.id));
  });
}

async function selectSession(id) {
  currentSession = id;
  const session = await api.chat.getSession(id);
  document.getElementById('chat-title').textContent = session.title;
  document.getElementById('delete-chat-btn').style.display = 'inline-flex';
  document.getElementById('empty-state').style.display = 'none';

  // Render messages
  const msgs = document.getElementById('messages');
  // Keep or remove empty state
  const hasMsgs = session.messages && session.messages.length > 0;
  msgs.innerHTML = '';

  if (!hasMsgs) {
    msgs.innerHTML = `<div class="empty-state" id="empty-state">
      <div class="empty-icon">💬</div>
      <p>Start the conversation</p>
    </div>`;
  } else {
    session.messages.forEach(m => appendMessage(m.role, m.content));
  }

  msgs.scrollTop = msgs.scrollHeight;
  loadSessions(); // refresh to update active state
  renderModelList();
}

function appendMessage(role, content, streaming = false) {
  const msgs = document.getElementById('messages');
  const empty = document.getElementById('empty-state');
  if (empty) empty.remove();

  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = `
    <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div class="message-content ${streaming ? 'streaming' : ''}">${escHtml(content)}</div>
  `;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
  return el;
}

function updateStreamingMessage(el, content) {
  const mc = el.querySelector('.message-content');
  mc.textContent = content;
  mc.classList.add('streaming');
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function finalizeStreaming(el) {
  const mc = el.querySelector('.message-content');
  mc.classList.remove('streaming');
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || !currentSession) return;

  input.value = '';
  input.style.height = 'auto';

  appendMessage('user', text);
  const el = appendMessage('assistant', '', true);
  document.getElementById('send-btn').disabled = true;

  try {
    const res = await api.chat.stream(currentSession, text, settings.temperature, settings.max_tokens);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n\n');
      for (const line of lines) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          const data = line.slice(6);
          full += data;
          updateStreamingMessage(el, full);
        }
      }
    }
    finalizeStreaming(el);
  } catch (e) {
    updateStreamingMessage(el, '[Error: ' + e.message + ']');
    finalizeStreaming(el);
  }

  document.getElementById('send-btn').disabled = false;
  loadSessions();
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function formatDate(d) {
  try { return new Date(d).toLocaleDateString(); } catch { return d; }
}

// New chat
document.getElementById('new-chat-btn').addEventListener('click', async () => {
  const title = prompt('Chat title:', 'New Chat') || 'New Chat';
  const s = await api.chat.createSession(title);
  await loadSessions();
  selectSession(s.id);
});

// Delete chat
document.getElementById('delete-chat-btn').addEventListener('click', async () => {
  if (!currentSession) return;
  if (!confirm('Delete this chat?')) return;
  await api.chat.deleteSession(currentSession);
  currentSession = null;
  document.getElementById('chat-title').textContent = 'No chat selected';
  document.getElementById('delete-chat-btn').style.display = 'none';
  document.getElementById('messages').innerHTML = `<div class="empty-state" id="empty-state">
    <div class="empty-icon">🤖</div>
    <p>Start a new chat or select an existing one</p>
  </div>`;
  loadSessions();
});

// Send message
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// Auto-resize textarea
document.getElementById('chat-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});
```

**Files:**
- Create: `frontend/js/monitor.js`

```javascript
/** System monitoring — polls backend for hardware stats */
async function updateStats() {
  try {
    const s = await api.system.stats();
    const inf = await api.system.inference();

    // CPU
    const cpu = Math.round(s.cpu_percent);
    document.getElementById('cpu-stat').textContent = cpu + '%';
    document.getElementById('cpu-bar').style.width = cpu + '%';
    document.getElementById('cpu-cores').textContent = s.cpu_count + ' cores';
    document.getElementById('cpu-mini').style.width = cpu + '%';

    // RAM
    const ramPct = Math.round(s.ram_percent);
    document.getElementById('ram-stat').textContent = `${s.ram_used_gb} / ${s.ram_total_gb} GB`;
    document.getElementById('ram-bar').style.width = ramPct + '%';
    document.getElementById('ram-pct').textContent = ramPct + '%';
    document.getElementById('ram-mini').style.width = ramPct + '%';

    // GPU
    if (s.gpu) {
      document.getElementById('gpu-card').style.display = 'block';
      document.getElementById('gpu-name').textContent = s.gpu.name || 'GPU';
      const gpuPct = Math.round((s.gpu.memory_used_mb / s.gpu.memory_total_mb) * 100);
      document.getElementById('gpu-mem').textContent = `${s.gpu.memory_used_mb} / ${s.gpu.memory_total_mb} MB`;
      document.getElementById('gpu-bar').style.width = gpuPct + '%';
      document.getElementById('gpu-temp').textContent = (s.gpu.temperature_c || '—') + '°C';
      document.getElementById('gpu-util').textContent = (s.gpu.utilization_pct || '—') + '%';
    }

    // Inference
    document.getElementById('inf-model').textContent = inf.model_name || '—';
    document.getElementById('inf-status').textContent = inf.is_generating ? '🟢 Generating' : 'Idle';
    document.getElementById('inf-tokens').textContent = `${inf.total_tokens_generated} tokens total`;

    // System info
    document.getElementById('uptime').textContent = formatUptime(s.uptime_seconds);
  } catch (e) {
    console.warn('Stats error:', e);
  }
}

function formatUptime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

// Poll every 2 seconds when visible
setInterval(updateStats, 2000);
updateStats();
```

**Files:**
- Create: `frontend/js/settings.js`

```javascript
/** Settings and models page logic */

async function renderModelList() {
  try {
    const models = await api.models.list();
    const modelsPage = document.getElementById('model-list');
    const trainSelect = document.getElementById('train-model');

    if (models.length === 0) {
      modelsPage.innerHTML = '<p class="empty-text">No models found. Place .gguf files in data/models/</p>';
    } else {
      modelsPage.innerHTML = models.map(m => `
        <div class="model-item">
          <div class="model-info">
            <div class="model-name">${escHtml(m.name)}</div>
            <div class="model-meta">
              <span>${m.size_gb} GB</span>
              <span>${m.quantization || 'unknown'}</span>
              <span>ctx: ${m.context_length}</span>
            </div>
          </div>
          <div>
            <span class="model-badge ${m.is_loaded ? 'loaded' : 'available'}">
              ${m.is_loaded ? 'Loaded' : 'Available'}
            </span>
            ${!m.is_loaded ? `<button class="btn btn-outline" onclick="loadModel('${escHtml(m.name)}')">Load</button>` : ''}
          </div>
        </div>
      `).join('');
    }

    // Training model selector
    if (trainSelect) {
      trainSelect.innerHTML = models.map(m => `<option>${escHtml(m.name)}</option>`).join('') ||
        '<option>No models available</option>';
    }
  } catch (e) {
    console.warn('Models error:', e);
  }
}

async function loadModel(name) {
  await api.models.load(name);
  renderModelList();
}

// Settings sliders
document.getElementById('setting-temp')?.addEventListener('input', function() {
  document.getElementById('temp-val').textContent = this.value;
});
document.getElementById('setting-maxtok')?.addEventListener('input', function() {
  document.getElementById('maxtok-val').textContent = this.value;
});
document.getElementById('setting-ctx')?.addEventListener('input', function() {
  document.getElementById('ctx-val').textContent = this.value;
});

// Save settings
document.getElementById('save-settings-btn')?.addEventListener('click', async () => {
  const s = {
    temperature: parseFloat(document.getElementById('setting-temp').value),
    max_tokens: parseInt(document.getElementById('setting-maxtok').value),
    context_length: parseInt(document.getElementById('setting-ctx').value),
    system_prompt: document.getElementById('setting-sysprompt').value,
  };
  await api.models.updateSettings(s);
  settings.temperature = s.temperature;
  settings.max_tokens = s.max_tokens;
  alert('Settings saved!');
});

// Test connection
document.getElementById('test-connection-btn')?.addEventListener('click', async () => {
  const url = document.getElementById('setting-llm-url').value;
  const result = document.getElementById('connection-result');
  result.innerHTML = 'Testing...';
  try {
    const r = await fetch(url.replace('/completion', '/health') || url);
    const d = await r.json();
    result.innerHTML = `<span class="conn-dot ok"></span>OK — ${JSON.stringify(d)}`;
  } catch (e) {
    result.innerHTML = `<span class="conn-dot bad"></span>Failed — ${e.message}`;
  }
});

// Training
document.getElementById('start-training-btn')?.addEventListener('click', async () => {
  const cfg = {
    base_model: document.getElementById('train-model').value,
    dataset_path: document.getElementById('train-dataset').value,
    epochs: parseInt(document.getElementById('train-epochs').value),
    learning_rate: parseFloat(document.getElementById('train-lr').value),
    batch_size: parseInt(document.getElementById('train-batch').value),
    lora_rank: parseInt(document.getElementById('train-lora').value),
  };
  const job = await api.training.start(cfg);
  alert(`Training job ${job.job_id} started!`);
  renderTrainingJobs();
});

async function renderTrainingJobs() {
  const jobs = await api.training.list();
  const list = document.getElementById('training-jobs-list');
  if (jobs.length === 0) {
    list.innerHTML = '<p class="empty-text">No training jobs yet</p>';
    return;
  }
  list.innerHTML = jobs.map(j => `
    <div class="doc-item">
      <div>
        <div class="doc-name">Job ${j.job_id}</div>
        <div class="doc-meta">${j.status} · Epoch ${j.current_epoch}/${j.total_epochs} · Loss: ${j.loss || '—'}</div>
      </div>
      <div>
        <div class="progress-bar" style="width:120px;display:inline-block">
          <div class="progress-bar-fill ${j.status === 'completed' ? 'green' : j.status === 'failed' ? 'red' : ''}" style="width:${j.progress_pct}%"></div>
        </div>
        <span style="font-size:0.8rem;margin-left:8px">${Math.round(j.progress_pct)}%</span>
      </div>
    </div>
  `).join('');
}

// Documents
async function renderDocs() {
  const docs = await api.docs.list();
  const list = document.getElementById('doc-list');
  if (docs.length === 0) {
    list.innerHTML = '<p class="empty-text">No documents uploaded yet</p>';
    return;
  }
  list.innerHTML = docs.map(d => `
    <div class="doc-item">
      <div>
        <div class="doc-name">📄 ${escHtml(d.name)}</div>
        <div class="doc-meta">${d.size_kb} KB · ${d.status} · ${d.chunks} chunks</div>
      </div>
      <button class="btn-icon btn-danger" onclick="deleteDoc('${escHtml(d.name)}')" title="Delete">🗑️</button>
    </div>
  `).join('');
}

async function deleteDoc(name) {
  if (!confirm(`Delete ${name}?`)) return;
  await api.docs.delete(name);
  renderDocs();
}

// Document upload
const docDropzone = document.getElementById('doc-dropzone');
const docInput = document.getElementById('doc-file-input');

docDropzone?.addEventListener('click', () => docInput.click());
docDropzone?.addEventListener('dragover', (e) => { e.preventDefault(); docDropzone.style.borderColor = 'var(--accent-cyan)'; });
docDropzone?.addEventListener('dragleave', () => { docDropzone.style.borderColor = ''; });
docDropzone?.addEventListener('drop', async (e) => {
  e.preventDefault();
  docDropzone.style.borderColor = '';
  for (const file of e.dataTransfer.files) {
    await api.docs.upload(file);
  }
  renderDocs();
});
docInput?.addEventListener('change', async () => {
  for (const file of docInput.files) {
    await api.docs.upload(file);
  }
  renderDocs();
  docInput.value = '';
});

// Model file upload dropzone
const modelDropzone = document.getElementById('model-dropzone');
const modelInput = document.getElementById('model-file-input');

modelDropzone?.addEventListener('click', () => modelInput.click());
modelDropzone?.addEventListener('dragover', (e) => { e.preventDefault(); modelDropzone.classList.add('drag-over'); });
modelDropzone?.addEventListener('dragleave', () => { modelDropzone.classList.remove('drag-over'); });
modelDropzone?.addEventListener('drop', async (e) => {
  e.preventDefault();
  modelDropzone.classList.remove('drag-over');
  // Note: large .gguf files should be placed directly on disk, not uploaded via browser
  alert('For large models, place .gguf files directly in the data/models/ directory on the server.');
});
modelInput?.addEventListener('change', () => {
  alert('For large models, place .gguf files directly in the data/models/ directory on the server.');
});

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
```

**Files:**
- Create: `frontend/js/app.js`

```javascript
/** Main app — navigation, init */

// Page navigation
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const page = item.dataset.page;
    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    item.classList.add('active');
    // Update page
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');

    // Lazy-load page content
    if (page === 'models') renderModelList();
    if (page === 'training') { renderModelList(); renderTrainingJobs(); }
    if (page === 'documents') renderDocs();
  });
});

// Init
(async () => {
  // Load settings
  try {
    const s = await api.models.settings();
    settings.temperature = s.temperature;
    settings.max_tokens = s.max_tokens;
    const tempEl = document.getElementById('setting-temp');
    if (tempEl) { tempEl.value = s.temperature; document.getElementById('temp-val').textContent = s.temperature; }
    const maxEl = document.getElementById('setting-maxtok');
    if (maxEl) { maxEl.value = s.max_tokens; document.getElementById('maxtok-val').textContent = s.max_tokens; }
    const ctxEl = document.getElementById('setting-ctx');
    if (ctxEl) { ctxEl.value = s.context_length; document.getElementById('ctx-val').textContent = s.context_length; }
    const sysEl = document.getElementById('setting-sysprompt');
    if (sysEl && s.system_prompt) sysEl.value = s.system_prompt;
  } catch (e) { console.warn('Settings load failed:', e); }

  // Load sessions
  await loadSessions();
})();
```

**Commit:** `feat: add full frontend — chat, monitor, models, training, docs, settings`

---

## Phase 6: Polish & Deploy

### Task 14: Add __init__.py files, install & run

**Files:**
- Create: `backend/__init__.py` (empty)
- Create: `backend/routers/__init__.py` (empty)
- Create: `backend/services/__init__.py` (empty)
- Create: `backend/models/__init__.py` (empty)

**Step 1: Create empty init files**

```bash
touch project-glitch/backend/__init__.py
touch project-glitch/backend/routers/__init__.py
touch project-glitch/backend/services/__init__.py
touch project-glitch/backend/models/__init__.py
```

**Step 2: Install and run**

```bash
cd project-glitch/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

**Step 3: Commit everything**

```bash
cd project-glitch
git add .
git commit -m "feat: full web UI dashboard for Project Glitch"
git push origin main
```
