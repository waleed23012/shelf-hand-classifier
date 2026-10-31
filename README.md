# Shelf Hand Classifier

[![CI](https://github.com/waleed23012/shelf-hand-classifier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/waleed23012/shelf-hand-classifier/actions/workflows/ci.yml)

A small FastAPI service that looks at a cropped image of a shopper's hand at a
supermarket shelf and says whether the hand is **holding a product** or is
**empty**. It is one part of a shelf-monitoring system that compares what the
camera saw being taken with what staff recorded, and flags the differences for
a manager.

AI381 MLOps · Lab 2 (Container + CI)

## Run it (one command)

Requires Docker.

```bash
docker run --rm -p 8000:8000 ghcr.io/waleed23012/shelf-hand-classifier:latest
```

Then open <http://localhost:8000/docs>, or:

```bash
curl http://localhost:8000/health
curl -F "file=@hand.jpg;type=image/jpeg" http://localhost:8000/predict
```

```json
{"label": "holding", "confidence": 0.83, "prob_holding": 0.83, "decision": "HOLDING", "threshold": 0.7}
```

When the model is less sure than `threshold` the decision is `REVIEW`, so a
person checks it instead of the system guessing. Set it per request
(`/predict?threshold=0.6`, allowed 0.5 to 1.0) or for the container with
`-e REVIEW_THRESHOLD=0.6`.

## How it works

```
camera frame -> hand detector -> hand crop -> [ this service ]
                                               DINOv2-base (frozen, 768 features)
                                               -> logistic regression
                                               -> holding / empty + confidence
```

* **DINOv2-base** (Meta, pre-trained on 142M images) turns the crop into 768
  numbers. It is not trained here; it is used as a fixed feature extractor.
* **Logistic regression** on top of those features is the only part trained on
  our data (linear probing): 1,003 hand crops from 29 videos of the MERL
  Shopping dataset.
* Accuracy is **74.9% ± 6.5%** with 5-fold `GroupKFold`, split **by video** so
  frames of the same person never appear in both train and test.

Finding the hand in the frame and the shelf logic that turns "hand left the
shelf holding something" into a *take* event live outside this container.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_loaded": true}` once the model is ready |
| POST | `/predict` | Multipart upload `file` (JPEG, PNG or WebP, 16 px to 5 MB) |

Invalid input (not an image, empty, too small, wrong type, threshold out of
range) returns `422` instead of crashing the server.

## Project layout

```
src/features.py       image -> DINOv2 features (shared by training and serving)
src/train/train.py    extract features, cross-validate, save the classifier
src/app/model.py      load classifier + backbone, predict one image
src/app/main.py       FastAPI app: /health, /predict, input validation
models/               hand_clf.joblib (classifier) + hand_clf.json (metadata)
tests/                pytest suite
Dockerfile            multi-stage, non-root, pinned base, HEALTHCHECK
.github/workflows/    CI
```

## Develop locally

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
ruff check .
pytest                               # fast tests, fake backbone
RUN_SLOW=1 pytest                    # also runs the real DINOv2 (downloads ~350 MB)
python -m uvicorn src.app.main:app --reload
```

Retrain (expects `<root>/holding/*.jpg` and `<root>/empty/*.jpg`):

```bash
python -m src.train.train --data dataset dataset_v2 --out models/hand_clf.joblib
```

Build the image yourself:

```bash
docker build -t shelf-hand-classifier .
docker run --rm -p 8000:8000 shelf-hand-classifier
```

## CI

`.github/workflows/ci.yml`

* **Every pull request:** `ruff` lint and `pytest`.
* **Every push to `main`:** the same, then build the image, scan it with
  **Trivy** (full HIGH/CRITICAL report; the job fails on any CRITICAL issue
  that has a fix), and only then push it to **GHCR** tagged with the full
  commit SHA (plus `latest`). The digest is written to the run summary.
* Every third-party action is pinned to a full commit SHA.
* `main` is protected: changes go through a pull request and CI must pass.

## Design decisions

* **Same feature code for training and serving** (`src/features.py`), so the
  API cannot silently prepare images differently from training
  (training-serving skew).
* **Pinned versions everywhere.** `scikit-learn==1.6.1` matches the version the
  classifier was saved with; a test fails if it drifts. The base image is
  pinned by digest, actions by SHA.
* **DINOv2 is baked into the image at build time** and the container runs
  offline, so it starts without internet and the health check is not waiting
  on a 350 MB download.
* **CPU-only PyTorch** keeps the image around 0.8 GB compressed instead of
  several GB with CUDA, at the cost of slower inference.
* **Fast tests use a stand-in backbone** so CI does not download DINOv2 on
  every pull request; the real backbone is covered by the optional slow test.

## Limitations

* Trained on lab footage from a top-down camera; about half of the crops were
  too ambiguous to label, which caps accuracy. Footage from the real store
  camera (angled 30 to 40 degrees) is the next step.
* Confidences cluster near 50%, so the `REVIEW` threshold should be tuned on
  held-out data rather than fixed at 0.7.
* MERL Shopping is licensed for non-commercial research only.
