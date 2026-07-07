# Local Model Autoloader

A local Ollama model swapper/proxy that prevents VRAM fragmentation and CPU fallback by intelligently managing model residency across multiple GPUs.

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
3. Loads the target model (with optional VRAM safety gate)
4. Runs the job
5. If the job used the vision model, immediately evicts it
6. Reloads the text model as the default resident

All via Ollama's native API — empty chat messages load/unload models, `/api/ps` checks residency, `keep_alive: 0` prevents squatting.

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
| `GET` | `/health` | Status, loaded models, total free VRAM across all GPUs |
| `POST` | `/swap/load/{model_key}` | Force-load a model (evicts others if `single_model_mode`) |
| `POST` | `/swap/unload/{model_key}` | Force-unload a model |
| `POST` | `/chat` | Chat with text model (`model_key` defaults to `text_primary`) |
| `POST` | `/screenshot/translate` | OCR + translate screenshot via vision model, then evict it |
| `POST` | `/screenshot/ask` | Full pipeline: screenshot → vision OCR → unload vision → text model reasoning |

### Request Schemas

**POST /chat**
```json
{
  "model_key": "text_primary",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**POST /screenshot/translate**
```json
{
  "image_path": "screenshot.png",
  "target_language": "English",
  "source_hint": "auto"
}
```

**POST /screenshot/ask**
```json
{
  "image_path": "screenshot.png",
  "target_language": "English",
  "user_task": "Explain what this error means and what to do"
}
```

## Verified Test Results

Tested on a dual-GPU system (8GB + 10GB) with Ollama models: `qwen2.5:7b` (4.7GB), `qwen2.5vl:3b` (3.2GB), `qwen2.5vl:7b` (6.0GB).

### /health
```
GET /health → {"status":"ok","loaded_models":[],"free_vram_mb":16736}
```
No models loaded, VRAM summed across both GPUs.

### /swap/load + /swap/unload
```
POST /swap/load/text_primary
→ {"loaded":"qwen2.5:7b","resident":["qwen2.5:7b"],"free_vram_mb":16736}

ollama ps → qwen2.5:7b  5.6 GB  100% GPU

POST /swap/unload/text_primary
→ {"unloaded":"qwen2.5:7b","resident":[],"free_vram_mb":16736}

ollama ps → (empty)
```
Model loads to GPU, confirmed resident via Ollama's own API, then cleanly unloaded.

### /chat
```
POST /chat  {"messages":[{"role":"user","content":"Say hello in 3 words"}]}
→ {"response":"Hello there!","resident":["qwen2.5:7b"]}
```
Text model loads on demand, responds correctly, remains resident.

### /screenshot/translate (vision OCR + auto-eviction)
Test image: German error dialog — "Systemfehler: Zugriff verweigert / Error Code: 0x80070005"
```
POST /screenshot/translate
→ {
    "raw": "ORIGINAL_TEXT:\nSystemfehler. Zugriff verweigert\n...\nTRANSLATED_TEXT:\nSystem Error: Access Denied\n...",
    "vision_model": "qwen2.5vl:3b",
    "resident_model": "qwen2.5:7b",
    "resident": ["qwen2.5:7b"]
  }
```
Vision model loaded → OCR + translated German to English → vision evicted → text model reloaded as resident. Full cycle verified via `ollama ps` — only `qwen2.5:7b` remained after the call.

### /screenshot/ask (full pipeline)
```
POST /screenshot/ask  {"user_task":"Explain the error in one sentence"}
→ {"response":"The system error is Access Denied (0x80070005); contact the administrator.",
    "resident":["qwen2.5:7b"]}
```
Vision model extracts text → unloaded → text model receives OCR output and reasons about it. Single round-trip, clean handoff.

## Emergency Ollama Commands

```bash
# Check what's loaded
curl http://127.0.0.1:11434/api/ps

# Unload vision model
curl http://127.0.0.1:11434/api/chat -d '{"model":"qwen2.5vl:3b","messages":[],"keep_alive":0}'

# Load text model
curl http://127.0.0.1:11434/api/chat -d '{"model":"qwen2.5:7b","messages":[],"keep_alive":"30m"}'
```

## Configuration

`config.yaml` controls everything:

```yaml
models:
  text_primary:       # your main resident model
    name: "qwen2.5:7b"
    keep_alive: "30m"   # stay resident
    unload_after_use: false

  vision_ocr:         # burst tool — loaded only for OCR
    name: "qwen2.5vl:3b"
    keep_alive: 0       # evict immediately
    unload_after_use: true

policy:
  single_model_mode: true              # only one model resident at a time
  preferred_resident_model: "qwen2.5:7b"  # always reload after vision jobs
  evict_vision_after_job: true         # auto-evict vision after OCR
  check_vram: true                     # gate loading behind VRAM check
  min_free_vram_mb_before_load: 1500   # refuse to load if below this
```

## Policy

- **Text models are residents.** They stay loaded with `keep_alive: "30m"`.
- **Vision models are burst tools.** Loaded only for OCR/translation, evicted immediately after with `keep_alive: 0`.
- **Single-model mode** prevents two models from fighting over VRAM.
- **VRAM gate** refuses to load if free VRAM drops below the configured threshold.
