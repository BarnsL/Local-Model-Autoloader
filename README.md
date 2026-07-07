# Local Model Autoloader

A local Ollama model swapper/proxy that prevents VRAM fragmentation and CPU fallback by intelligently managing model residency.

## Problem

When using multiple Ollama models on limited VRAM:

```
vision model (e.g., qwen2.5vl:3b) stays loaded in VRAM
→ text model (e.g., qwen2.5:7b) tries to load
→ not enough VRAM
→ 7B falls back to CPU
→ inference speed dies
```

## Solution

Proxy all local model calls through this swapper. Before every request it:

1. Decides which model is needed
2. Unloads models that should not be resident
3. Loads the target model
4. Runs the job
5. If the job used the vision model, immediately evicts it
6. Reloads the text model as the default resident

## Quick Start

```bash
# Install
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Edit config.yaml to match your Ollama model tags

# Run the proxy server
uvicorn server:app --host 127.0.0.1 --port 8787
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, loaded models, free VRAM |
| POST | `/swap/load/{model_key}` | Force-load a model |
| POST | `/swap/unload/{model_key}` | Force-unload a model |
| POST | `/chat` | Chat with text model |
| POST | `/screenshot/translate` | OCR + translate screenshot |
| POST | `/screenshot/ask` | Screenshot → OCR → text model reasoning |

## Direct Ollama Commands (emergency cleanup)

```bash
# Check what's loaded
curl http://127.0.0.1:11434/api/ps

# Unload vision model
curl http://127.0.0.1:11434/api/chat -d '{"model":"qwen2.5vl:3b","messages":[],"keep_alive":0}'

# Load text model
curl http://127.0.0.1:11434/api/chat -d '{"model":"qwen2.5:7b","messages":[],"keep_alive":"30m"}'
```

## Policy

Default resident: `qwen2.5:7b` (text model).
Vision models are burst tools — loaded only for OCR/translation, then evicted immediately.
`keep_alive: 0` on vision requests prevents squatting.
