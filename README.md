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

### Running the Dashboard

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
| Platform | OpenClaw |
| Model | Local LLM (GGUF quantized) |

## Roadmap

- [x] Core OpenClaw integration
- [x] Local LLM inference
- [x] Custom RLHF training pipeline
- [x] Document ingestion and vector search
- [x] Web UI dashboard
- [ ] Multi-agent workflows
- [ ] Tool chaining
- [ ] Mobile companion app

## About

Built by [Nafyad Fantaye](https://github.com/Lijnaff) — Full-Stack Software Engineer & AI Architect based in Addis Ababa, Ethiopia.

## License

MIT License.
