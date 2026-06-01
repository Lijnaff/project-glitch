"""LLM inference service — supports Hermes Agent (CLI), OpenRouter (cloud), and llama.cpp (local) backends."""

import os
import httpx
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
    """Stream chat via Hermes Agent CLI subprocess.

    Calls `hermes chat -q <message>` and yields the response word-by-word
    to simulate streaming. Hermes handles tool-calling, memory, skills, etc.
    """
    import asyncio
    from pathlib import Path

    # Build conversation context from message history
    # Format: system prompt + recent messages so Hermes has context
    conversation_parts = []
    for msg in messages:
        role = msg.role.value
        if role == "system":
            conversation_parts.append(f"[System: {msg.content}]")
        elif role == "user":
            conversation_parts.append(f"User: {msg.content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {msg.content}")

    # The last user message is the actual query
    user_msg = next((m for m in reversed(messages) if m.role.value == "user"), None)
    if not user_msg:
        yield "\n[Error: No user message found]"
        return

    # Build the full context string
    context = "\n\n".join(conversation_parts)

    # Path to Hermes CLI
    hermes_python = Path(r"C:\Users\Naff\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe")
    hermes_main = Path(r"C:\Users\Naff\AppData\Local\hermes\hermes-agent\hermes_cli\main.py")

    if not hermes_python.exists():
        yield "\n[Error: Hermes Agent not found. Install at C:\\Users\\Naff\\AppData\\Local\\hermes\\hermes-agent]"
        return

    yield ""  # Initial empty chunk to establish SSE connection

    try:
        proc = await asyncio.create_subprocess_exec(
            str(hermes_python),
            "-m", "hermes_cli.main",
            "chat",
            "-q", context,
            "-Q",
            "--yolo",
            "--max-turns", "5",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120
        )

        if proc.returncode != 0 and stderr:
            err_text = stderr.decode("utf-8", errors="replace")[:200]
            yield f"\n[Hermes error: {err_text}]"
            return

        response = stdout.decode("utf-8", errors="replace").strip()

        # Strip session_id footer Hermes appends
        if "\nsession_id:" in response:
            response = response[:response.rfind("\nsession_id:")].strip()

        if not response:
            yield "\n[Error: Hermes returned empty response]"
            return

        # Simulate streaming: yield word by word for natural feel
        words = response.split(" ")
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word

    except asyncio.TimeoutError:
        yield "\n[Timeout: Hermes did not respond within 2 minutes]"
    except FileNotFoundError:
        yield "\n[Error: Hermes Python or main.py not found. Check installation path.]"
    except Exception as e:
        yield f"\n[Error: Hermes subprocess failed — {e}]"


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
