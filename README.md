# 🤖 Project Glitch

> On-premise personal AI assistant built on the OpenClaw platform. Local execution, zero external API costs, absolute data privacy.

## Overview

Project Glitch is a fully localized, high-speed personal AI assistant. No cloud dependencies, no API keys, no data leaving your machine.

## Key Features

- **100% Local Execution** — Runs entirely on your hardware
- **Zero API Costs** — No external API bills
- **Full Data Privacy** — Your data never leaves your machine
- **Custom RLHF Training** — Fine-tuned on domain-specific documentation
- **Low Latency** — Sub-second response times on consumer GPU

## Web UI Dashboard

This repo includes a full web-based dashboard:

- 💬 **Chat** — Streaming chat with session management
- 📊 **Monitor** — Real-time CPU, RAM, GPU, and inference stats
- 🧠 **Models** — List, load, and switch between GGUF models
- 🔥 **Training** — RLHF training panel with job management
- 📄 **Documents** — Upload and manage knowledge base documents
- ⚙️ **Settings** — Configure temperature, max tokens, context length, system prompt

### Quick Start

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

### Auto-Start on Windows

The dashboard is configured to start automatically on Windows boot:

1. A shortcut was placed in the Windows Startup folder: `start-dashboard.vbs`
2. You can verify it exists at: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\ProjectGlitch.lnk`
3. On every boot, the dashboard starts silently on port 8000
4. To open it, connect to http://localhost:8000 in your browser

If you need to start manually (no visible console window):

```cmd
wscript.exe "C:\Users\Naff\project-glitch\start-dashboard.vbs"
```

Or to see console output for debugging:

```cmd
C:\Users\Naff\project-glitch\start-dashboard.bat
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
| Platform | OpenClaw |
| Model | Local LLM (GGUF quantized) |

## Requirements

- Python 3.11+
- llama.cpp server running on port 8080 (for chat inference)
- NVIDIA GPU recommended (8GB+ VRAM) or Apple Silicon
- Windows 11 (for auto-start feature)

## About

Built by [Nafyad Fantaye](https://github.com/Lijnaff) — Full-Stack Software Engineer & AI Architect based in Addis Ababa, Ethiopia.

## License

MIT License.
