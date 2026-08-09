"""
Production-style FastAPI service exposing the LISS-IV cloud removal model
for operational integration (e.g. behind an internal geospatial processing
gateway, or triggered by a ground-station ingestion pipeline).

Endpoints:
    GET  /health              — liveness/readiness + loaded model info
    POST /predict              — upload a cloudy GeoTIFF (+ optional SAR
                                  GeoTIFF), get back a reconstructed
                                  cloud-free GeoTIFF
    POST /evaluate              — upload a (cloudy, cloud_free) pair to
                                  compute quantitative metrics on the fly

Run:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

Note on --workers: keep at 1 per GPU process (the model is loaded once at
startup into a single process/device); scale horizontally with a load
balancer / multiple containers instead of multiple uvicorn workers sharing
one GPU.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.schemas import HealthResponse, InferenceResponse, MetricsResponse
from src.evaluation.metrics import compute_all_metrics
from src.inference.predict import load_model, run_inference
from src.utils.config import default_config_path, load_config
from src.utils.geo_utils import percentile_normalize, read_raster
from src.utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="LISS-IV Cloud Removal Service",
    description="Generative AI-based cloud removal and reconstruction for LISS-IV satellite imagery.",
    version="1.0.0",
)

_STATE = {"model": None, "cfg": None, "device": None}


@app.on_event("startup")
def load_model_on_startup():
    cfg_path = os.environ.get("CLOUD_REMOVAL_CONFIG", default_config_path())
    ckpt_path = os.environ.get("CLOUD_REMOVAL_CHECKPOINT")

    cfg = load_config(cfg_path)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    _STATE["cfg"] = cfg
    _STATE["device"] = device

    if ckpt_path and Path(ckpt_path).exists():
        model, cfg = load_model(cfg, ckpt_path, device)
        _STATE["model"] = model
        _STATE["cfg"] = cfg
        logger.info(f"Loaded model from {ckpt_path} onto {device}")
    else:
        logger.warning(
            "No CLOUD_REMOVAL_CHECKPOINT set (or file not found) — API will "
            "start but /predict will return 503 until a checkpoint is "
            "configured. Set the env var and restart, or POST to /reload."
        )


@app.get("/health", response_model=HealthResponse)
def health():
    cfg = _STATE["cfg"]
    return HealthResponse(
        status="ok",
        model_loaded=_STATE["model"] is not None,
        architecture=cfg.model.architecture if cfg else None,
        device=str(_STATE["device"]) if _STATE["device"] else None,
    )


@app.post("/reload")
def reload_model(checkpoint_path: str):
    cfg = _STATE["cfg"]
    device = _STATE["device"]
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    model, cfg = load_model(cfg, checkpoint_path, device)
    _STATE["model"] = model
    _STATE["cfg"] = cfg
    return {"status": "reloaded", "checkpoint": checkpoint_path}


@app.post("/predict", response_model=InferenceResponse)
async def predict(cloudy_tif: UploadFile = File(...), sar_tif: UploadFile | None = File(None)):
    if _STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model checkpoint loaded. Set CLOUD_REMOVAL_CHECKPOINT and restart, or call /reload.",
        )

    job_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, f"cloudy_{job_id}.tif")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(cloudy_tif.file, f)

        sar_path = None
        if sar_tif is not None:
            sar_path = os.path.join(tmp_dir, f"sar_{job_id}.tif")
            with open(sar_path, "wb") as f:
                shutil.copyfileobj(sar_tif.file, f)

        output_dir = Path(_STATE["cfg"].paths.outputs_dir) / "api_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"reconstructed_{job_id}.tif")

        start = time.time()
        _, cloud_mask = run_inference(
            _STATE["cfg"], None, input_path, output_path, sar_path,
            model=_STATE["model"], device=_STATE["device"],
        )
        elapsed = time.time() - start

        return InferenceResponse(
            job_id=job_id,
            output_path=output_path,
            cloud_fraction_detected=float(cloud_mask.mean()),
            processing_time_seconds=round(elapsed, 3),
        )


@app.get("/download/{job_id}")
def download_result(job_id: str):
    output_dir = Path(_STATE["cfg"].paths.outputs_dir) / "api_predictions"
    path = output_dir / f"reconstructed_{job_id}.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found (expired or invalid job_id)")
    return FileResponse(str(path), media_type="image/tiff", filename=path.name)


@app.post("/evaluate", response_model=MetricsResponse)
async def evaluate_pair(
    reconstructed_tif: UploadFile = File(...),
    ground_truth_tif: UploadFile = File(...),
):
    cfg = _STATE["cfg"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_path = os.path.join(tmp_dir, "pred.tif")
        gt_path = os.path.join(tmp_dir, "gt.tif")
        with open(pred_path, "wb") as f:
            shutil.copyfileobj(reconstructed_tif.file, f)
        with open(gt_path, "wb") as f:
            shutil.copyfileobj(ground_truth_tif.file, f)

        pred_arr, _ = read_raster(pred_path)
        gt_arr, _ = read_raster(gt_path)
        pred_norm = percentile_normalize(pred_arr)
        gt_norm = percentile_normalize(gt_arr)

        metrics = compute_all_metrics(pred_norm, gt_norm, cfg.evaluation.metrics)
        return MetricsResponse(**metrics)
