"""Embed the train and test splits once and cache them to disk.

Reads hand-crop images from data/raw/dataset (train) and data/raw/dataset_test
(test), only the "empty" and "holding" folders, embeds them with the frozen
DINOv2 backbone and saves everything to data/embeddings.npz.

Leakage fix: test images whose video id also appears in train are moved into
the train set, so no video is in both splits.

Usage:
    python -m src.train.extract_embeddings
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.features import DEFAULT_BACKBONE, DinoEmbedder
from src.train.train import LABELS, collect_paths, video_id

TRAIN_ROOT = "data/raw/dataset"
TEST_ROOT = "data/raw/dataset_test"
OUT_PATH = Path("data/embeddings.npz")


def describe(name: str, y: np.ndarray, groups: np.ndarray) -> None:
    videos = sorted(set(groups.tolist()))
    print(f"{name}: {len(y)} images ({(y == 0).sum()} empty / {(y == 1).sum()} holding), "
          f"{len(videos)} videos: {videos}")


def main() -> None:
    train_paths, y_train = collect_paths([TRAIN_ROOT])
    test_paths, y_test = collect_paths([TEST_ROOT])
    if not train_paths or not test_paths:
        raise SystemExit(
            f"No images found: train={len(train_paths)} ({TRAIN_ROOT}), "
            f"test={len(test_paths)} ({TEST_ROOT})"
        )
    groups_train = np.array([video_id(p) for p in train_paths])
    groups_test = np.array([video_id(p) for p in test_paths])

    print("Before leakage fix:")
    describe("  train", y_train, groups_train)
    describe("  test ", y_test, groups_test)

    shared = sorted(set(groups_train.tolist()) & set(groups_test.tolist()))
    move = np.isin(groups_test, shared)
    print(f"Leakage fix: moving {int(move.sum())} test images from {len(shared)} "
          f"shared videos into train: {shared}")

    train_paths = train_paths + [p for p, m in zip(test_paths, move, strict=True) if m]
    y_train = np.concatenate([y_train, y_test[move]])
    groups_train = np.concatenate([groups_train, groups_test[move]])
    test_paths = [p for p, m in zip(test_paths, move, strict=True) if not m]
    y_test = y_test[~move]
    groups_test = groups_test[~move]

    assert not set(groups_train) & set(groups_test), "video ids still shared"

    print("After leakage fix:")
    describe("  train", y_train, groups_train)
    describe("  test ", y_test, groups_test)

    embedder = DinoEmbedder(DEFAULT_BACKBONE)
    print(f"embedding with {embedder.backbone} on {embedder.device}")
    X_train = embedder.embed(Image.open(p) for p in train_paths).astype(np.float32)
    X_test = embedder.embed(Image.open(p) for p in test_paths).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        groups_train=groups_train.astype(str),
        groups_test=groups_test.astype(str),
        labels=np.array(LABELS),
        backbone=np.array(DEFAULT_BACKBONE),
    )
    print(f"Saved {OUT_PATH}: X_train {X_train.shape}, X_test {X_test.shape}")


if __name__ == "__main__":
    main()
