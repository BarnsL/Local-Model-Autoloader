from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from swapper import OllamaSwapper


app = FastAPI(title="Local Model Swapper")
swapper = OllamaSwapper("config.yaml")


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model_key: str = "text_primary"


class ScreenshotTranslateRequest(BaseModel):
    image_path: str
    target_language: str = "English"
    source_hint: str = "auto"


class ScreenshotAskRequest(BaseModel):
    image_path: str
    user_task: str
    target_language: str = "English"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": await swapper.loaded_models(),
        "free_vram_mb": swapper.get_free_vram_mb()
    }


@app.post("/swap/load/{model_key}")
async def load_model(model_key: str):
    model = await swapper.ensure_model(model_key)
    return {
        "loaded": model,
        "resident": await swapper.loaded_models(),
        "free_vram_mb": swapper.get_free_vram_mb()
    }


@app.post("/swap/unload/{model_key}")
async def unload_model(model_key: str):
    model = swapper.model_name(model_key)
    await swapper.unload_model(model)
    return {
        "unloaded": model,
        "resident": await swapper.loaded_models(),
        "free_vram_mb": swapper.get_free_vram_mb()
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    response = await swapper.chat_text(
        messages=req.messages,
        model_key=req.model_key
    )
    return {
        "response": response,
        "resident": await swapper.loaded_models(),
        "free_vram_mb": swapper.get_free_vram_mb()
    }


@app.post("/screenshot/translate")
async def screenshot_translate(req: ScreenshotTranslateRequest):
    result = await swapper.vision_extract_translate(
        image_path=req.image_path,
        target_language=req.target_language,
        source_hint=req.source_hint
    )
    return {
        **result,
        "resident": await swapper.loaded_models(),
        "free_vram_mb": swapper.get_free_vram_mb()
    }


@app.post("/screenshot/ask")
async def screenshot_ask(req: ScreenshotAskRequest):
    response = await swapper.translated_screenshot_to_text_llm(
        image_path=req.image_path,
        user_task=req.user_task,
        target_language=req.target_language
    )
    return {
        "response": response,
        "resident": await swapper.loaded_models(),
        "free_vram_mb": swapper.get_free_vram_mb()
    }
