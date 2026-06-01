"""Hermes bridge endpoints — direct agent communication via message queue."""

import os
import httpx
import time
import json as _json
from typing import Optional, AsyncGenerator
from backend.config import LLM_HOST, LLM_PORT
from backend.models.schemas import InferenceStats, ChatMessage

_current_model: Optional[str] = None
_total_tokens: int = 0
_is_generating: bool = False

# Backend selection: "openrouter", "llama_cpp", or "hermes"
_llm_backend = os.environ.get("GLITCH_LLM_BACKEND", "hermes")

# OpenRouter settings
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL = os.environ.get("GLITCH_OPENROUTER_MODEL", "openrouter/owl-alpha")


def get_inference_stats() -> InferenceStats:
    return InferenceStats(
        model_name=_current_model,
        context_length=4096,
        tokens_per_second=0.0,
        total_tokens_generated=_total_tokens,
        is_generating=_is_generating,
    )


def set_backend(backend: str):
    global _llm_backend
    _llm_backend = backend


def get_backend() -> str:
    return _llm_backend


async def chat_stream(
    messages: list[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """Stream chat response from the configured LLM backend."""
    global _is_generating, _total_tokens
    _is_generating = True
    token_count = 0

    try:
        if _llm_backend == "hermes":
            async for chunk in _hermes_stream(messages, temperature, max_tokens):
                token_count += 1
                yield chunk
        elif _llm_backend == "openrouter":
            async for chunk in _openrouter_stream(messages, temperature, max_tokens):
                token_count += 1
                yield chunk
        else:
            async for chunk in _llamacpp_stream(messages, temperature, max_tokens):
                token_count += 1
                yield chunk
    finally:
        _total_tokens += token_count
        _is_generating = False


async def _hermes_stream(
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Stream chat via Hermes Agent bridge (message queue).
    
    1. Submit the user message to the inbox queue
    2. Poll the outbox queue for Hermes' response
    3. Yield the response as it arrives
    """
    from backend.services.hermes_bridge import (
        submit_message, get_response, get_queue_status
    )

    # Get the last user message
    user_msg = next((m for m in reversed(messages) if m.role.value == "user"), None)
    if not user_msg:
        yield "\n[Error: No user message found]"
        return

    # Get system prompt if any
    sys_msg = next((m for m in messages if m.role.value == "system"), None)
    system_prompt = sys_msg.content if sys_msg else ""

    # Submit to Hermes inbox
    session_id = "hermes-bridge"
    request_id = submit_message(session_id, user_msg.content, system_prompt)

    yield ""  # Initial empty chunk to establish SSE connection

    # Poll for response (up to 300 seconds = 5 minutes)
    start_time = time.time()
    timeout = 300
    poll_interval = 2  # seconds

    while time.time() - start_time < timeout:
        time.sleep(poll_interval)

        # Check status
        status = get_queue_status(request_id)
        if status["status"] == "completed":
            response = status.get("response", {})
            content = response.get("content", "")
            if content:
                yield content
            else:
                yield "\n[Error: Empty response from Hermes]"
            return

    yield "\n[Timeout: Hermes did not respond within 5 minutes. Make sure Hermes is running and checking the inbox.]"


async def _openrouter_stream(
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Stream from OpenRouter API."""
    if not _OPENROUTER_API_KEY:
        yield "\n[Error: No OpenRouter API key configured]"
        return

    url = f"{_OPENROUTER_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Lijnaff/project-glitch",
        "X-Title": "Project Glitch Dashboard",
    }
    payload = {
        "model": _OPENROUTER_MODEL,
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield f"\n[Error: OpenRouter returned {response.status_code}: {body.decode()[:200]}]"
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass
    except Exception as e:
        yield f"\n[Error: OpenRouter connection failed — {e}]"


async def _llamacpp_stream(
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Stream from local llama.cpp server."""
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
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass
    except Exception as e:
        yield f"\n[Error: Could not connect to llama.cpp at {url} — {e}]"


def load_model(model_name: str, context_length: int = 4096):
    global _current_model
    _current_model = model_name
