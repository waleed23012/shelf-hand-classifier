"""Feature extraction shared by training and serving.

Both train.py and the API import from here, so an image is turned into a
vector in exactly the same way at training time and at prediction time.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from PIL import Image

# dinov2-base gives 768-dim features (what the current classifier was trained on).
# dinov2-small (384-dim) makes a much smaller, faster image but needs retraining.
DEFAULT_BACKBONE = os.getenv("BACKBONE", "facebook/dinov2-base")


class DinoEmbedder:
    """Wraps a frozen DINOv2 model and returns one L2-normalised vector per image."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone).to(self.device).eval()

    def embed(self, images: Iterable[Image.Image], batch_size: int = 32) -> np.ndarray:
        images = [img.convert("RGB") for img in images]
        feats = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=images[i : i + batch_size], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                out = self.model(**inputs)
            feats.append(out.pooler_output.cpu().numpy())
        return l2_normalize(np.vstack(feats))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
