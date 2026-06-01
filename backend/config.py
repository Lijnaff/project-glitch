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
LLM_BACKEND = "llama_cpp"
LLM_HOST = "localhost"
LLM_PORT = 8080
