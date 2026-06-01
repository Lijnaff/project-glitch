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
    status: str = "idle"
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
    status: str = "indexed"
    chunks: int = 0


class DocumentUploadResponse(BaseModel):
    success: bool
    document: Optional[DocumentInfo] = None
    error: Optional[str] = None
