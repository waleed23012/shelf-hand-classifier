"""FastAPI service for the shelf hand-state classifier.

GET  /health   -> is the service up and the model loaded?
POST /predict  -> upload a hand-crop image, get holding/empty + a decision
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.app.model import HandClassifier

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_SIDE_PX = 16
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.70"))

state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set state["model"] to a fake before startup, so only load if empty.
    if state["model"] is None:
        state["model"] = HandClassifier.load()
    yield


app = FastAPI(title="Shelf Hand Classifier", version="1.0.0", lifespan=lifespan)


def _openapi_with_file_picker() -> dict:
    """Swagger UI only draws a 'Choose File' button for format: binary,
    which FastAPI's OpenAPI 3.1 output leaves out. Add it back."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    for comp in schema.get("components", {}).get("schemas", {}).values():
        for prop in comp.get("properties", {}).values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_file_picker


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class PredictResponse(BaseModel):
    label: Literal["holding", "empty"]
    confidence: float = Field(ge=0.0, le=1.0)
    prob_holding: float = Field(ge=0.0, le=1.0)
    decision: Literal["HOLDING", "EMPTY", "REVIEW"]
    threshold: float


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=state["model"] is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Cropped image of one hand at the shelf"),
    threshold: float = Query(
        DEFAULT_THRESHOLD, ge=0.5, le=1.0,
        description="Below this confidence the decision is REVIEW (sent to a human)",
    ),
) -> PredictResponse:
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported type {file.content_type}")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="File is not a valid image") from None

    if min(image.size) < MIN_SIDE_PX:
        raise HTTPException(status_code=422, detail=f"Image smaller than {MIN_SIDE_PX}px")

    result = model.predict(image.convert("RGB"))

    if result.confidence < threshold:
        decision = "REVIEW"
    else:
        decision = result.label.upper()

    return PredictResponse(
        label=result.label,
        confidence=round(result.confidence, 4),
        prob_holding=round(result.prob_holding, 4),
        decision=decision,
        threshold=threshold,
    )
