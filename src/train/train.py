"""Train the hand-state classifier (holding vs empty).

Pipeline: hand-crop images -> frozen DINOv2 features -> logistic regression
(linear probing). Evaluated with GroupKFold by video so frames from the same
video never appear in both train and test (no leakage).

Expected data layout (one or more roots):
    <root>/holding/*.jpg
    <root>/empty/*.jpg
File names start with the video id, e.g. 10_1_crop_p3R_0012.50.jpg

Usage:
    python -m src.train.train --data dataset dataset_v2 --out models/hand_clf.joblib
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from src.features import DEFAULT_BACKBONE, DinoEmbedder

LABELS = ["empty", "holding"]  # index 0 = empty, 1 = holding
SEED = 42


def collect_paths(roots: list[str]) -> tuple[list[str], np.ndarray]:
    paths, y = [], []
    for root in roots:
        for label_idx, label in enumerate(LABELS):
            found = sorted(glob.glob(os.path.join(root, label, "*.jpg")))
            paths += found
            y += [label_idx] * len(found)
    return paths, np.array(y)


def video_id(path: str) -> str:
    return re.sub(r"_p\d+[LR]_.*", "", os.path.basename(path))


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000, C=0.1, class_weight="balanced", random_state=SEED
    )


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> dict:
    accs, f1s = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups), start=1):
        clf = build_classifier().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        print(f"fold {k}: acc={accs[-1]:.3f}  f1={f1s[-1]:.3f}")
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", nargs="+", required=True, help="dataset root folder(s)")
    parser.add_argument("--out", default="models/hand_clf.joblib")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    paths, y = collect_paths(args.data)
    if len(paths) == 0:
        raise SystemExit(f"No images found under {args.data}")
    groups = np.array([video_id(p) for p in paths])
    print(f"{len(paths)} images ({(y == 1).sum()} holding, {(y == 0).sum()} empty) "
          f"from {len(set(groups))} videos")

    embedder = DinoEmbedder(args.backbone)
    X = embedder.embed(Image.open(p) for p in paths)

    metrics = cross_validate(X, y, groups, args.folds)
    print(f"CV accuracy {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")

    final = build_classifier().fit(X, y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out)

    meta = {
        "backbone": args.backbone,
        "labels": LABELS,
        "feature_dim": int(X.shape[1]),
        "n_images": len(paths),
        "n_videos": len(set(groups)),
        "cv": metrics,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
