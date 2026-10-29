# syntax=docker/dockerfile:1

# Pinned base image. After the first `docker pull`, replace the tag below with
# the exact digest (see README) so every build uses byte-identical layers.
ARG BASE_IMAGE=python:3.13-slim-trixie@sha256:2d44738ebf4ce5f63b9d9ea871e9c7cfdc0426f7b0c97cac915fd88ded1d86bb
# ---------- Stage 1: builder ----------
# Installs dependencies and downloads the DINOv2 weights. None of the build
# tools or pip caches from this stage end up in the final image.
FROM ${BASE_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the backbone into the image: the container then starts without internet
# and the health check does not wait on a 350 MB download.
COPY models/hand_clf.json /tmp/hand_clf.json
RUN python -c "import json; from transformers import AutoImageProcessor, AutoModel; \
b = json.load(open('/tmp/hand_clf.json'))['backbone']; \
AutoImageProcessor.from_pretrained(b); AutoModel.from_pretrained(b); print('cached', b)"

# ---------- Stage 2: runtime ----------
FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="shelf-hand-classifier" \
      org.opencontainers.image.description="DINOv2 + logistic regression: is a hand at the shelf holding a product?"

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/hand_clf.joblib

# Unprivileged user: if the app is compromised it cannot modify the system.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf /opt/hf
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models

USER app
EXPOSE 8000

# The slim image has no curl, so the check uses Python's standard library.
# start-period gives DINOv2 time to load before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
