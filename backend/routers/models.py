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
    "system_prompt": "",
}


@router.get("/", response_model=list[ModelInfo])
async def list_models():
    models = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.gguf"):
            size_gb = round(f.stat().st_size / (1024**3), 2)
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
    if req.system_prompt is not None:
        _settings["system_prompt"] = req.system_prompt
    return _settings
