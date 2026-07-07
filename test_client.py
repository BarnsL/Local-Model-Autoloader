import requests


BASE = "http://127.0.0.1:8787"


def health():
    r = requests.get(f"{BASE}/health")
    print(r.json())


def load_7b():
    r = requests.post(f"{BASE}/swap/load/text_primary")
    print(r.json())


def unload_vision():
    r = requests.post(f"{BASE}/swap/unload/vision_ocr")
    print(r.json())


def ask_text():
    payload = {
        "model_key": "text_primary",
        "messages": [
            {"role": "system", "content": "You are concise and precise."},
            {"role": "user", "content": "Say whether you are running."}
        ]
    }
    r = requests.post(f"{BASE}/chat", json=payload)
    print(r.json())


def translate_screenshot():
    payload = {
        "image_path": "screenshot.png",
        "target_language": "English",
        "source_hint": "auto"
    }
    r = requests.post(f"{BASE}/screenshot/translate", json=payload)
    print(r.json())


def screenshot_to_text_model():
    payload = {
        "image_path": "screenshot.png",
        "target_language": "English",
        "user_task": "Explain what this screenshot says and what action I should take."
    }
    r = requests.post(f"{BASE}/screenshot/ask", json=payload)
    print(r.json())


if __name__ == "__main__":
    health()
    unload_vision()
    load_7b()
    ask_text()
