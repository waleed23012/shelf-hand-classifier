"""Shared test fixtures.

The real DINOv2 backbone is ~350 MB, so the fast tests replace it with a small
deterministic stand-in that returns a 768-dim vector, the same shape DINOv2-base
gives. The trained classifier itself (models/hand_clf.joblib) is the real one.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import main
from src.app.model import HandClassifier
from src.features import l2_normalize

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_clf.joblib"
META = json.loads(MODEL_PATH.with_suffix(".json").read_text())
FEATURE_DIM = META["feature_dim"]


class FakeEmbedder:
    """Deterministic image -> unit vector, same output size as DINOv2-base."""

    def __init__(self, dim: int = FEATURE_DIM):
        self.projection = np.random.RandomState(0).randn(16 * 16 * 3, dim)

    def embed(self, images):
        rows = []
        for img in images:
            pixels = np.asarray(img.convert("RGB").resize((16, 16)), dtype=np.float64) / 255.0
            rows.append(pixels.reshape(-1) @ self.projection)
        return l2_normalize(np.vstack(rows))


@pytest.fixture
def client():
    classifier = joblib.load(MODEL_PATH)
    main.state["model"] = HandClassifier(classifier, FakeEmbedder(), META["labels"])
    with TestClient(main.app) as c:
        yield c
    main.state["model"] = None


def make_image(size=(96, 96), color=(180, 140, 120), mode="RGB", fmt="JPEG") -> bytes:
    """A small image with a gradient so it is not a flat colour."""
    img = Image.new(mode, size, color if mode == "RGB" else 128)
    if mode == "RGB":
        arr = np.asarray(img).copy()
        arr[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)[None, :]
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()
