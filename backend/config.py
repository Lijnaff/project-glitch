"""Project Glitch configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MODELS_DIR = DATA_DIR / "models"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Load .env file if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Create dirs if missing
for d in [SESSIONS_DIR, MODELS_DIR, DOCUMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "mistral-7b-v0.2.Q4_K_M.gguf"
DEFAULT_SYSTEM_PROMPT = "You are Glitch, a helpful AI assistant running locally."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_CONTEXT_LENGTH = 4096

# LLM backend: "openrouter" (cloud) or "llama_cpp" (local)
LLM_BACKEND = os.environ.get("GLITCH_LLM_BACKEND", "openrouter")

# OpenRouter settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("GLITCH_OPENROUTER_MODEL", "openrouter/owl-alpha")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# llama.cpp local server settings
LLM_HOST = os.environ.get("LLM_HOST", "localhost")
LLM_PORT = int(os.environ.get("LLM_PORT", "8080"))
