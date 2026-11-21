"""Loads the trained classifier and turns one hand-crop image into a prediction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/hand_clf.joblib"))


@dataclass
class Prediction:
    label: str           # "holding" or "empty"
    confidence: float    # probability of the predicted label
    prob_holding: float


class HandClassifier:
    def __init__(self, classifier, embedder, labels: list[str],
                 mlflow_run_id: str | None = None):
        self.classifier = classifier
        self.embedder = embedder
        self.labels = labels
        # Lineage: which MLflow run trained this head. None for models saved
        # before promote.py wrote it, so older hand_clf.json files still load.
        self.mlflow_run_id = mlflow_run_id

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH) -> "HandClassifier":
        from src.features import DinoEmbedder

        meta = json.loads(model_path.with_suffix(".json").read_text())
        classifier = joblib.load(model_path)
        embedder = DinoEmbedder(meta["backbone"], device="cpu")
        return cls(classifier, embedder, meta["labels"], meta.get("mlflow_run_id"))

    def predict(self, image: Image.Image) -> Prediction:
        features = self.embedder.embed([image])
        probs = self.classifier.predict_proba(features)[0]
        idx = int(np.argmax(probs))
        holding_idx = self.labels.index("holding")
        return Prediction(
            label=self.labels[idx],
            confidence=float(probs[idx]),
            prob_holding=float(probs[holding_idx]),
        )
