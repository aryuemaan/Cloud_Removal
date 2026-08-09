"""
Basic API smoke tests using FastAPI's TestClient (no running server / no
network needed). Full /predict and /evaluate coverage with real GeoTIFFs is
exercised manually in scripts/run_full_pipeline.sh and CI's integration job
(see .github/workflows if configured); here we verify the app wires up and
responds sanely even without a loaded checkpoint.
"""
import os

from fastapi.testclient import TestClient


def test_health_endpoint_without_checkpoint(monkeypatch):
    monkeypatch.delenv("CLOUD_REMOVAL_CHECKPOINT", raising=False)
    from api.app import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.delenv("CLOUD_REMOVAL_CHECKPOINT", raising=False)
    from api.app import app

    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            files={"cloudy_tif": ("test.tif", b"not-a-real-tiff", "image/tiff")},
        )
        assert resp.status_code == 503
