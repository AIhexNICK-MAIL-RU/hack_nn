from typing import List, Dict, Any
import os
import pandas as pd


class CatalogManager:
    def __init__(self) -> None:
        self.catalogs_dir = os.path.join(os.getcwd(), "catalogs")
        self._catalog_cache: List[Dict[str, Any]] = []

    def load_catalogs(self) -> List[Dict[str, Any]]:
        if self._catalog_cache:
            return self._catalog_cache
        if not os.path.isdir(self.catalogs_dir):
            return []

        aggregated: List[Dict[str, Any]] = []
        for file in os.listdir(self.catalogs_dir):
            if not file.lower().endswith(".xlsx"):
                continue
            path = os.path.join(self.catalogs_dir, file)
            try:
                df = pd.read_excel(path)
            except Exception:
                continue
            # expected columns heuristic
            article_col = None
            for candidate in ["Артикул", "article", "Article", "арт."]:
                if candidate in df.columns:
                    article_col = candidate
                    break
            if article_col is None:
                continue

            for _, row in df.iterrows():
                product = {
                    "article": str(row.get(article_col, "")),
                    "characteristics": {},
                }
                aggregated.append(product)

        self._catalog_cache = aggregated
        return aggregated


