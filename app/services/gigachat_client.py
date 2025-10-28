import os
from typing import Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class GigaChatClient:
    def __init__(self) -> None:
        self.api_url = os.getenv(
            "GIGACHAT_API_URL",
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        )
        self.api_key = os.getenv("GIGACHAT_API_KEY", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    def extract_characteristics(self, product_name: str, manufacturer: str, article: str) -> Dict[str, Any]:
        if not self.api_key:
            # Fallback for local dev without credentials
            return {}

        prompt = (
            "Извлеки технические характеристики из описания/названия товара и артикула. "
            "Ответ верни строгим JSON объектом: {\"характеристика\": \"значение\"}.\n\n"
            f"Наименование: {product_name}\nПроизводитель: {manufacturer}\nАртикул: {article}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "system", "content": "Ты извлекаешь характеристики товаров."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(self.api_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # The actual parsing depends on GigaChat schema; keep robust fallback
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return {}

        # Attempt to parse JSON block inside content
        import json
        try:
            return json.loads(content)
        except Exception:
            return {}


