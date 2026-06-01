"""LLM inference service — talks to llama.cpp server via HTTP."""

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
                        if data.strip() == "[DONE]":
                            break
                        try:
                            import json as _json
                            chunk = _json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                token_count += 1
                                yield content
                        except Exception:
                            pass
    except Exception as e:
        yield f"\n[Error: Could not connect to LLM server at {url} — {e}]"
    finally:
        _total_tokens += token_count
        _is_generating = False


def load_model(model_name: str, context_length: int = 4096):
    """Signal which model is loaded. Actual loading is handled by llama.cpp server."""
    global _current_model
    _current_model = model_name
