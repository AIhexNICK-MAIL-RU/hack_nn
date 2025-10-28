from typing import Dict, Any

from .gigachat_client import GigaChatClient


class CharacteristicExtractor:
    def __init__(self) -> None:
        self.gigachat = GigaChatClient()
        self.cache: Dict[str, Dict[str, Any]] = {}

    def extract(self, product_name: str, manufacturer: str, article: str) -> Dict[str, Any]:
        cache_key = f"{manufacturer}::{article}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        characteristics = self.gigachat.extract_characteristics(
            product_name, manufacturer, article
        )
        self.cache[cache_key] = characteristics
        return characteristics


