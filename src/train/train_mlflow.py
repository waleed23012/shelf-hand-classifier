"""Train candidate heads on cached DINOv2 embeddings and log every run to MLflow.

Each run records: params (model settings + data_sha256), metrics (overall F1,
F1 per test video = data slice, p95 latency, size), tags (git commit, batch)
and the model with its signature.

Usage:
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    python -m src.train.train_mlflow
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA = Path("data/embeddings.npz")
EXPERIMENT = "shelf-hand"
SEED = 42
# skops (MLflow 3's sklearn format) rejects this MLP optimizer type unless it is trusted.
TRUSTED_TYPES = ["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]

# One deliberately weak model (C=0.00001 regularises so strongly it barely learns),
# to test that the promotion gate blocks it.
CANDIDATES = {
    "weak": LogisticRegression(C=0.00001, max_iter=1000),
}


def sha256(path: Path) -> str:
    """Fingerprint of the training data: same hash = same data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def p95_latency_ms(model, X: np.ndarray, n: int = 300) -> float:
    """One image at a time, like the API. Head only: the backbone cost is the same for all."""
    times = []
    for i in range(n):
        x = X[i % len(X)][None, :]
        t0 = time.perf_counter()
        model.predict_proba(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def size_kb(model) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.joblib"
        joblib.dump(model, path)
        return path.stat().st_size / 1024


def main() -> None:
    d = np.load(DATA)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test, groups_test = d["X_test"], d["y_test"], d["groups_test"]
    labels = [str(x) for x in d["labels"]]
    data_hash = sha256(DATA)
    batch = os.getenv("BATCH", time.strftime("%Y%m%d-%H%M%S"))

    mlflow.set_experiment(EXPERIMENT)
    for name, model in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "f1_macro": f1_score(y_test, pred, average="macro"),
                "accuracy": accuracy_score(y_test, pred),
                "auc": roc_auc_score(y_test, proba),
            }
            for label, f1 in zip(labels, f1_score(y_test, pred, average=None), strict=True):
                metrics[f"f1_class_{label}"] = f1
            # Data slices: each test video is a different recording session
            slice_f1 = {}
            for video in np.unique(groups_test):
                m = groups_test == video
                slice_f1[str(video)] = f1_score(y_test[m], pred[m], average="macro")
                metrics[f"f1_video_{video}"] = slice_f1[str(video)]
            metrics["f1_worst_slice"] = min(slice_f1.values())
            metrics["p95_latency_ms"] = p95_latency_ms(model, X_test)
            metrics["model_size_kb"] = size_kb(model)

            mlflow.log_params({
                "candidate": name,
                "data_sha256": data_hash,
                "backbone": str(d["backbone"]),
                "n_train": len(y_train),
                "n_test": len(y_test),
                **{f"model__{k}": v for k, v in model.get_params().items()},
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"git_sha": git_sha(), "batch": batch})
            signature = infer_signature(X_test[:5], model.predict_proba(X_test[:5]))
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_test[:2],
                skops_trusted_types=TRUSTED_TYPES,
            )

            print(f"{name:8s} f1={metrics['f1_macro']:.3f} "
                  f"worst_slice={metrics['f1_worst_slice']:.3f} "
                  f"auc={metrics['auc']:.3f} p95={metrics['p95_latency_ms']:.2f}ms "
                  f"size={metrics['model_size_kb']:.0f}KB")


if __name__ == "__main__":
    main()
