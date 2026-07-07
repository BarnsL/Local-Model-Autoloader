import asyncio
import base64
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml


class OllamaSwapper:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.base_url = self.config["ollama"]["base_url"].rstrip("/")
        self.models = self.config["models"]
        self.policy = self.config["policy"]
        self.lock = asyncio.Lock()

    def model_name(self, key: str) -> str:
        return self.models[key]["name"]

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{self.base_url}{path}", json=payload)
            r.raise_for_status()
            return r.json()

    async def _get(self, path: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{self.base_url}{path}")
            r.raise_for_status()
            return r.json()

    async def loaded_models(self) -> List[str]:
        """
        Uses Ollama /api/ps to see what is currently resident.
        """
        try:
            data = await self._get("/api/ps")
            return [m.get("name") or m.get("model") for m in data.get("models", [])]
        except Exception:
            return []

    async def unload_model(self, model: str) -> None:
        """
        Unload through Ollama chat endpoint.
        Empty messages + keep_alive 0 = unload.
        """
        try:
            await self._post("/api/chat", {
                "model": model,
                "messages": [],
                "keep_alive": 0,
                "stream": False
            })
        except Exception:
            # Fallback to CLI if API unload fails.
            subprocess.run(["ollama", "stop", model], capture_output=True, text=True)

    async def load_model(self, model: str, keep_alive: Any = "30m") -> None:
        """
        Empty messages loads the model into memory.
        """
        await self._post("/api/chat", {
            "model": model,
            "messages": [],
            "keep_alive": keep_alive,
            "stream": False
        })

    def get_free_vram_mb(self) -> Optional[int]:
        """
        Returns total free VRAM across all GPUs in MB.
        If nvidia-smi is unavailable, returns None.
        """
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"
                ],
                text=True
            )
            values = [int(x.strip()) for x in out.splitlines() if x.strip()]
            return sum(values) if values else None
        except Exception:
            return None

    async def evict_everything_except(self, target_model: str) -> None:
        loaded = await self.loaded_models()
        for model in loaded:
            if model and model != target_model:
                await self.unload_model(model)

    async def ensure_model(self, model_key: str) -> str:
        """
        Central model swap function.
        Checks VRAM before loading if policy.check_vram is enabled.
        """
        target = self.model_name(model_key)
        keep_alive = self.models[model_key].get("keep_alive", "30m")

        async with self.lock:
            if self.policy.get("single_model_mode", True):
                await self.evict_everything_except(target)

            loaded = await self.loaded_models()
            if target not in loaded:
                # VRAM safety check before loading
                if self.policy.get("check_vram", False):
                    free = self.get_free_vram_mb()
                    min_req = self.policy.get("min_free_vram_mb_before_load", 0)
                    if free is not None and free < min_req:
                        raise RuntimeError(
                            f"VRAM too low to load {target}: "
                            f"{free}MB free, need {min_req}MB"
                        )
                await self.load_model(target, keep_alive=keep_alive)

            return target

    async def chat_text(self, messages: List[Dict[str, str]], model_key: str = "text_primary") -> str:
        model = await self.ensure_model(model_key)
        keep_alive = self.models[model_key].get("keep_alive", "30m")

        data = await self._post("/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive
        })

        return data.get("message", {}).get("content", "")

    async def vision_extract_translate(
        self,
        image_path: str,
        target_language: str = "English",
        source_hint: str = "auto"
    ) -> Dict[str, str]:
        """
        Uses the small vision model only for OCR/translation.
        Then immediately evicts it and reloads the 7B text model.
        """
        vision_key = "vision_ocr"
        vision_model = await self.ensure_model(vision_key)

        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

        prompt = f"""
You are an OCR and screenshot translation engine.

Task:
1. Read all visible text in the screenshot.
2. Preserve numbers, dates, names, warnings, buttons, labels, URLs, and error messages.
3. Translate the text into {target_language}.
4. Do not explain.
5. Return this exact format:

ORIGINAL_TEXT:
...

TRANSLATED_TEXT:
...

Source language hint: {source_hint}
"""

        data = await self._post("/api/chat", {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64]
                }
            ],
            "stream": False,

            # Critical: do not let the vision model squat in VRAM.
            "keep_alive": 0
        })

        content = data.get("message", {}).get("content", "")

        if self.policy.get("evict_vision_after_job", True):
            await self.unload_model(vision_model)

        # Put the 7B back on GPU.
        preferred = self.policy.get("preferred_resident_model")
        if preferred:
            for key, meta in self.models.items():
                if meta["name"] == preferred:
                    await self.ensure_model(key)
                    break

        return {
            "raw": content,
            "vision_model": vision_model,
            "resident_model": self.policy.get("preferred_resident_model", "")
        }

    async def translated_screenshot_to_text_llm(
        self,
        image_path: str,
        user_task: str,
        target_language: str = "English"
    ) -> str:
        """
        Full pipeline:
        image -> vision OCR/translation -> unload vision -> text 7B analysis.
        """
        extraction = await self.vision_extract_translate(
            image_path=image_path,
            target_language=target_language
        )

        text_prompt = f"""
The following text was extracted from a screenshot by a vision OCR model.

{extraction["raw"]}

User task:
{user_task}

Instructions:
- Use the translated text as the main source.
- Preserve important numbers, dates, names, buttons, labels, and warnings.
- Mention if the OCR appears uncertain.
"""

        return await self.chat_text([
            {"role": "system", "content": "You are a precise text-only assistant."},
            {"role": "user", "content": text_prompt}
        ])
