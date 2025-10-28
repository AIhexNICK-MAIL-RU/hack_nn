from __future__ import annotations

from typing import List, Dict, Any, Tuple
import os
import pandas as pd

from .model import SiameseLikeMLP


def load_pairs_from_dataset(dataset_path: str) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[int]]:
    # Expect columns: se_article, competitor_article, is_match, plus optional attributes
    df = pd.read_excel(dataset_path)
    keys = [c for c in df.columns if c not in {"se_article", "competitor_article", "is_match"}]
    X_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    y: List[int] = []
    for _, row in df.iterrows():
        target = {k: row.get(k) for k in keys}
        candidate = {k: row.get(k) for k in keys}
        # In absence of separate fields per side, use same features; real dataset should split
        X_pairs.append((target, candidate))
        y.append(int(row.get("is_match", 0)))
    return X_pairs, y


def train_and_save(model_path: str, dataset_path: str) -> None:
    X_pairs, y = load_pairs_from_dataset(dataset_path)
    keys = list(X_pairs[0][0].keys()) if X_pairs else []
    model = SiameseLikeMLP.create(keys)
    model.fit(X_pairs, y)
    import joblib
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"keys": keys, "pipeline": model.pipeline}, model_path)


if __name__ == "__main__":
    ds_path = os.getenv("TRAIN_DATASET", os.path.join("dataset", "train.xlsx"))
    out_path = os.getenv("MODEL_PATH", os.path.join("artifacts", "mlp_model.joblib"))
    train_and_save(out_path, ds_path)


