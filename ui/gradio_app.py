import os
from typing import List, Dict, Any

import gradio as gr
import pandas as pd
import joblib

from app.services.extractor import CharacteristicExtractor
from app.services.catalog_manager import CatalogManager
from app.services.matching import find_analogs


def load_model(model_path: str):
    if not os.path.isfile(model_path):
        return None
    blob = joblib.load(model_path)
    return blob


def predict_flow(name: str, manufacturer: str, article: str, catalogs_file, dataset_file):
    # Optionally accept uploaded Excel files to override defaults
    work_dir = os.getcwd()
    temp_catalogs_path = None
    temp_dataset_path = None

    if catalogs_file is not None:
        temp_catalogs_path = os.path.join(work_dir, "_uploaded_catalogs.xlsx")
        catalogs_file.save(temp_catalogs_path)
    if dataset_file is not None:
        temp_dataset_path = os.path.join(work_dir, "_uploaded_dataset.xlsx")
        dataset_file.save(temp_dataset_path)

    extractor = CharacteristicExtractor()
    target_char = extractor.extract(name, manufacturer, article)

    catalog_mgr = CatalogManager()
    catalog = catalog_mgr.load_catalogs()

    prelim = find_analogs(target_char, catalog, threshold=0.0, limit=20)

    model_path = os.getenv("MODEL_PATH", os.path.join("artifacts", "mlp_model.joblib"))
    model_blob = load_model(model_path)

    if model_blob is not None and prelim:
        keys = model_blob["keys"]
        pipeline = model_blob["pipeline"]
        import numpy as np
        from app.ml.model import build_pair_features

        X = np.vstack([build_pair_features(target_char, p.get("characteristics", {}), keys) for p in prelim])
        proba = pipeline.predict_proba(X)[:, 1]
        for p, sc in zip(prelim, proba):
            p["score"] = float(sc)
        prelim.sort(key=lambda x: x["score"], reverse=True)

    top = prelim[:6]
    return [p.get("article", "") for p in top]


with gr.Blocks(title="Analog Matcher (Gradio)") as demo:
    gr.Markdown("## Поиск аналогов продукции — Gradio UI")
    with gr.Row():
        name = gr.Textbox(label="Наименование")
        manufacturer = gr.Textbox(label="Производитель")
        article = gr.Textbox(label="Артикул")
    with gr.Row():
        catalogs_file = gr.File(label="Каталоги (Excel)", file_types=[".xlsx"], type="filepath")
        dataset_file = gr.File(label="Обучающая выборка (Excel)", file_types=[".xlsx"], type="filepath")
    run_btn = gr.Button("Найти аналоги")
    output = gr.JSON(label="Результат")

    def _wrapped(name, manufacturer, article, catalogs_path, dataset_path):
        # Gradio passes file path when type="filepath"
        class _F:
            def __init__(self, path):
                self._p = path
            def save(self, to):
                import shutil
                shutil.copy2(self._p, to)

        cf = _F(catalogs_path) if catalogs_path else None
        df = _F(dataset_path) if dataset_path else None
        analogs = predict_flow(name, manufacturer, article, cf, df)
        return {"analogs": analogs}

    run_btn.click(_wrapped, inputs=[name, manufacturer, article, catalogs_file, dataset_file], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)


