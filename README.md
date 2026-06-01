# 🤖 Project Glitch

> On-premise personal AI assistant dashboard. Chat, monitor, models, training, and documents — all in one place.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Connecting to an LLM

The dashboard supports two backends:

### OpenRouter (Default — Cloud)
Uses the same API and model as Hermes Agent. In the dashboard:
1. Go to **Settings → LLM Backend**
2. Select **OpenRouter**
3. Enter your API key (get one at https://openrouter.ai/keys)
4. Model: `openrouter/owl-alpha` (or any OpenRouter model)
5. Click **Save & Apply**

### llama.cpp (Local)
Run your own GGUF model locally:
1. Start llama.cpp server on port 8080
2. In Settings, select **llama.cpp**
3. Enter your server URL
4. Click **Save & Apply**

## Features

- 💬 **Chat** — Streaming chat with session management (SSE)
- 📊 **Monitor** — Real-time CPU, RAM, GPU, and inference stats
- 🧠 **Models** — List, load, and switch between GGUF models
- 🔥 **Training** — RLHF training panel with job management
- 📄 **Documents** — Upload and manage knowledge base documents
- ⚙️ **Settings** — Configure temperature, max tokens, context length, system prompt

## Auto-Start on Windows

The dashboard starts automatically on boot via Windows Startup folder:
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\ProjectGlitch.lnk`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
| LLM | OpenRouter API or llama.cpp |

## About

Built by [Nafyad Fantaye](https://github.com/Lijnaff) — Full-Stack Software Engineer & AI Architect.

## License

MIT License.
