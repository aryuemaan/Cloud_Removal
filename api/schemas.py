"""Pydantic schemas for the cloud-removal serving API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    architecture: Optional[str] = None
    device: Optional[str] = None


class InferenceResponse(BaseModel):
    job_id: str
    output_path: str
    cloud_fraction_detected: float
    processing_time_seconds: float


class MetricsResponse(BaseModel):
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    sam: Optional[float] = None
    rmse: Optional[float] = None
    ergas: Optional[float] = None
    mae: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str
