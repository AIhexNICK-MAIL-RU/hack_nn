from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


def _feature_diff(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 0.0 if a == b else 1.0


def build_pair_features(target: Dict[str, Any], candidate: Dict[str, Any], keys: List[str]) -> np.ndarray:
    feats: List[float] = []
    for k in keys:
        feats.append(_feature_diff(_safe_str(target.get(k)), _safe_str(candidate.get(k))))
    return np.asarray(feats, dtype=np.float32)


@dataclass
class SiameseLikeMLP:
    keys: List[str]
    pipeline: Pipeline

    @classmethod
    def create(cls, keys: List[str]) -> "SiameseLikeMLP":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, random_state=42)),
        ])
        return cls(keys=keys, pipeline=pipeline)

    def fit(self, X_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], y: List[int]) -> None:
        X = np.vstack([build_pair_features(a, b, self.keys) for a, b in X_pairs])
        self.pipeline.fit(X, np.asarray(y))

    def predict_proba(self, X_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> np.ndarray:
        X = np.vstack([build_pair_features(a, b, self.keys) for a, b in X_pairs])
        proba = self.pipeline.predict_proba(X)
        return proba[:, 1]


