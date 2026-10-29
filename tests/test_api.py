"""Tests for the shelf hand-classifier service.

1. API contract    - /health and /predict return the documented shape
2. Model loads     - the saved classifier opens with the pinned scikit-learn
3. Input schema    - bad uploads and bad parameters are rejected with 4xx
4. Edge cases      - tiny, grayscale and PNG images
5. Deterministic   - the same image always gets the same answer
"""

from __future__ import annotations

import io
import os
import warnings

import joblib
import pytest
from PIL import Image

from tests.conftest import META, MODEL_PATH, make_image

JPEG = "image/jpeg"


def post_image(client, data: bytes, content_type: str = JPEG, **params):
    return client.post(
        "/predict", params=params, files={"file": ("hand.jpg", data, content_type)}
    )


# 1. API contract -------------------------------------------------------------

def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_documented_fields(client):
    resp = post_image(client, make_image())
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"label", "confidence", "prob_holding", "decision", "threshold"}
    assert body["label"] in {"holding", "empty"}
    assert 0.5 <= body["confidence"] <= 1.0          # binary: winner is >= 50%
    assert 0.0 <= body["prob_holding"] <= 1.0
    # decision must follow the threshold rule
    if body["confidence"] < body["threshold"]:
        assert body["decision"] == "REVIEW"
    else:
        assert body["decision"] == body["label"].upper()


def test_threshold_one_always_sends_to_review(client):
    resp = post_image(client, make_image(), threshold=1.0)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "REVIEW"


# 2. Model loads ---------------------------------------------------------------

def test_classifier_artifact_matches_metadata():
    # Fail if scikit-learn drifted from the version the model was saved with.
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        clf = joblib.load(MODEL_PATH)

    assert clf.n_features_in_ == META["feature_dim"]
    assert list(clf.classes_) == list(range(len(META["labels"])))
    assert META["labels"] == ["empty", "holding"]


# 3. Input schema --------------------------------------------------------------

@pytest.mark.parametrize(
    "data, content_type",
    [
        (b"", JPEG),                               # empty upload
        (b"this is not an image", JPEG),           # wrong bytes, right header
        (make_image(), "text/plain"),              # right bytes, wrong type
    ],
    ids=["empty", "corrupt", "wrong-type"],
)
def test_rejects_invalid_uploads(client, data, content_type):
    assert post_image(client, data, content_type).status_code == 422


def test_rejects_missing_file(client):
    assert client.post("/predict").status_code == 422


@pytest.mark.parametrize("threshold", [0.2, 1.5])
def test_rejects_threshold_out_of_range(client, threshold):
    assert post_image(client, make_image(), threshold=threshold).status_code == 422


# 4. Edge cases ----------------------------------------------------------------

def test_rejects_image_too_small(client):
    assert post_image(client, make_image(size=(8, 8))).status_code == 422


@pytest.mark.parametrize(
    "data, content_type",
    [
        (make_image(mode="L"), JPEG),              # grayscale camera frame
        (make_image(fmt="PNG"), "image/png"),      # PNG instead of JPEG
        (make_image(size=(1200, 40)), JPEG),       # very thin crop
    ],
    ids=["grayscale", "png", "thin-crop"],
)
def test_accepts_unusual_but_valid_images(client, data, content_type):
    assert post_image(client, data, content_type).status_code == 200


# 5. Deterministic -------------------------------------------------------------

def test_same_image_gives_same_prediction(client):
    img = make_image()
    first = post_image(client, img).json()
    second = post_image(client, img).json()
    assert first == second


# Optional: real DINOv2 end to end (slow, downloads ~350 MB). Run with RUN_SLOW=1.

@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run")
def test_real_backbone_end_to_end():
    from src.app.model import HandClassifier

    model = HandClassifier.load(MODEL_PATH)
    a = model.predict(Image.open(io.BytesIO(make_image())))
    b = model.predict(Image.open(io.BytesIO(make_image())))
    assert a == b
    assert a.label in {"holding", "empty"}
