# =============================================================================
# LISS-IV Cloud Removal — Production Dockerfile
#
# Multi-stage build: a shared base with GDAL + Python deps, then thin
# stage-specific entrypoints for training vs. serving so the deployed API
# image doesn't carry training-only dev tooling.
#
# Build:
#   docker build -t liss4-cloud-removal:latest .
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints \
#       liss4-cloud-removal:latest python -m src.training.train --config config/config.yaml
#
# Run the API:
#   docker run --gpus all -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints \
#       -e CLOUD_REMOVAL_CHECKPOINT=/app/checkpoints/ckpt_best.pt \
#       liss4-cloud-removal:latest uvicorn api.app:app --host 0.0.0.0 --port 8000
# =============================================================================

FROM python:3.11-slim AS base

# --- System dependencies: GDAL and friends for rasterio/geospatial I/O ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_VERSION=3.6.2 \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Python dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- Application code ---
COPY . .

RUN mkdir -p /app/data/raw /app/data/processed /app/checkpoints /app/logs /app/outputs

EXPOSE 8000

# Default entrypoint runs the API; override the command for training/inference/eval.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
