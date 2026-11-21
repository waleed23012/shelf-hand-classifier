"""Promotion gate: only a better model gets to production.

Rules (reasons in MODEL_SELECTION.md):
  1. The head's p95 latency must be within LATENCY_BUDGET_MS.
  2. Its F1 must beat the current champion by at least MIN_F1_GAIN.
Exit code 0 = PROMOTED, 1 = BLOCKED (this makes the CI check fail).

On promotion it registers the model, moves the @champion alias to it, and
writes what the API and the Dockerfile read:
  models/hand_clf.joblib   the classifier head
  models/hand_clf.json     backbone, labels and lineage (MLflow run id)
  models/champion.json     the champion's numbers, kept in Git for CI

Usage:
    python scripts/promote.py                      # compare with models/champion.json
    python scripts/promote.py --champion base.json # CI: compare with the base branch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

EXPERIMENT = "shelf-hand"
MODEL_NAME = "shelf-hand-model"
LABELS = ["empty", "holding"]  # same order as LABELS in src/train/train.py
MODELS_DIR = Path("models")

LATENCY_BUDGET_MS = 5.0  # head only; laptop p95 is 0.1-0.3 ms, CI runners are slower
# with 68 test images, one image moves F1 by ~0.015: smaller gains are noise
MIN_F1_GAIN = 0.02


def batch_runs(client: MlflowClient, exp_id: str, batch: str | None):
    """Runs of one training batch (BATCH env var), or of the latest batch."""
    finished = "attributes.status = 'FINISHED'"
    if batch is None:
        latest = client.search_runs(
            [exp_id], finished, order_by=["attributes.start_time DESC"], max_results=1
        )
        if not latest:
            sys.exit("BLOCKED: no finished training runs found")
        batch = latest[0].data.tags["batch"]
    return batch, client.search_runs([exp_id], f"{finished} AND tags.batch = '{batch}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", default=str(MODELS_DIR / "champion.json"),
                        help="champion record to beat (CI passes the one from the base branch)")
    args = parser.parse_args()

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        sys.exit(f"BLOCKED: experiment {EXPERIMENT!r} not found")
    batch, runs = batch_runs(client, exp.experiment_id, os.getenv("BATCH"))

    print(f"batch {batch}: {len(runs)} candidates")
    for r in runs:
        m = r.data.metrics
        print(f"  {r.info.run_name:8s} f1={m['f1_macro']:.3f} "
              f"worst_slice={m['f1_worst_slice']:.3f} "
              f"p95={m['p95_latency_ms']:.2f}ms")

    # Rule 1: latency budget
    within = [r for r in runs if r.data.metrics["p95_latency_ms"] <= LATENCY_BUDGET_MS]
    if not within:
        print(f"BLOCKED: no candidate within the latency budget of {LATENCY_BUDGET_MS} ms")
        sys.exit(1)
    best = max(within, key=lambda r: r.data.metrics["f1_macro"])
    best_f1 = best.data.metrics["f1_macro"]

    # Rule 2: must beat the champion by MIN_F1_GAIN
    champ_path = Path(args.champion)
    if champ_path.exists() and champ_path.stat().st_size > 0:
        champ = json.loads(champ_path.read_text())
        needed = champ["f1_macro"] + MIN_F1_GAIN
        print(f"champion: {champ['candidate']} f1={champ['f1_macro']:.3f}, "
              f"a new model needs f1 >= {needed:.3f}")
        if best_f1 < needed:
            print(f"BLOCKED: best candidate {best.info.run_name} "
                  f"has f1={best_f1:.3f} < {needed:.3f}")
            sys.exit(1)
    else:
        print("no champion yet: the best candidate within the budget is promoted")

    # Register, move the alias, and write the files the image packages
    version = mlflow.register_model(f"runs:/{best.info.run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    m, params = best.data.metrics, best.data.params
    record = {
        "candidate": best.info.run_name,
        "mlflow_run_id": best.info.run_id,
        "registered_model": MODEL_NAME,
        "model_version": version.version,
        "data_sha256": params["data_sha256"],
        "trained_at_commit": best.data.tags.get("git_sha"),
        "f1_macro": m["f1_macro"],
        "f1_worst_slice": m["f1_worst_slice"],
        "p95_latency_ms": m["p95_latency_ms"],
        "model_size_kb": m["model_size_kb"],
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "hand_clf.joblib")
    meta = {"backbone": params["backbone"], "labels": LABELS, "feature_dim": 768, **record}
    (MODELS_DIR / "hand_clf.json").write_text(json.dumps(meta, indent=2))
    (MODELS_DIR / "champion.json").write_text(json.dumps(record, indent=2))

    print(f"PROMOTED: {best.info.run_name} f1={best_f1:.3f} -> {MODEL_NAME} "
          f"v{version.version} @champion (run {best.info.run_id})")


if __name__ == "__main__":
    main()
