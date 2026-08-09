# LISS-IV Generative Cloud Removal & Reconstruction

Industrial-grade Generative AI framework for automated cloud removal and
surface reconstruction in LISS-IV satellite imagery, fusing optical,
Sentinel-1 SAR, and cloud-mask information through an attention-augmented
GAN (with a swappable diffusion-model alternative).

This is a **complete, runnable codebase** — data pipeline, training,
inference, evaluation, a REST API, Docker packaging, and tests — not a
notebook or a partial reference implementation. Every command below has
been executed against this exact codebase as part of building it.

```
┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ Raw scenes │──▶│ Preprocessing│──▶│ GAN / Diffusion│──▶│ Cloud-free    │
│ (LISS-IV,  │   │ (mask, patch,│   │ Training        │   │ GeoTIFF       │
│ SAR, DEM)  │   │ normalize)   │   │ (src/training)  │   │ (src/inference)│
└────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
                                              │
                                              ▼
                                   ┌────────────────────┐
                                   │ Evaluation & QA     │
                                   │ (PSNR/SSIM/SAM/...) │
                                   └────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full design
rationale (why SAR fusion, why these losses, why a classical cloud mask,
etc.) — read it before making architectural changes.

---

## 1. Quickstart (5 minutes, no real data required)

The repo ships a synthetic data generator so you can validate the **entire
pipeline** — preprocessing → training → evaluation → inference → API —
before ever touching real Bhoonidhi/Sentinel downloads.

```bash
# 1. Clone / unzip and enter the project
cd liss4-cloud-removal

# 2. Create an environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full smoke-test pipeline (synthetic data, ~2 epochs, a few minutes on CPU)
bash scripts/run_full_pipeline.sh
```

This generates synthetic scenes, extracts patches, trains a small model,
evaluates it, and writes qualitative comparison panels to
`outputs/evaluation/`. If this completes without errors, your environment
is correctly set up.

To run each stage individually, see §4 below.

---

## 2. Installation

### 2.1 Requirements

- Python 3.10+
- GDAL system libraries (for `rasterio`) — see platform notes below
- ~8 GB RAM minimum for default patch size (256×256) at batch size 16;
  reduce `training.batch_size` / `model.generator.base_channels` for
  smaller machines
- A CUDA-capable GPU is **strongly recommended for real training** (the
  default config assumes one); CPU works for the smoke test and small-scale
  validation but is not practical for full-resolution training

### 2.2 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**GDAL note:** `rasterio` requires GDAL system libraries.
- **Ubuntu/Debian:** `sudo apt-get install -y gdal-bin libgdal-dev`
- **macOS:** `brew install gdal`
- **Conda (recommended if you hit GDAL build issues):**
  ```bash
  conda create -n liss4 python=3.11
  conda activate liss4
  conda install -c conda-forge rasterio gdal
  pip install -r requirements.txt
  ```

### 2.3 Docker (alternative to a local environment)

```bash
docker build -t liss4-cloud-removal:latest .

# Run the API
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -e CLOUD_REMOVAL_CHECKPOINT=/app/checkpoints/ckpt_best.pt \
    liss4-cloud-removal:latest

# Or use docker-compose (also provides a tensorboard profile)
docker compose up api
docker compose run --rm train
docker compose --profile monitoring up tensorboard
```

Drop `--gpus all` / the GPU `deploy` blocks in `docker-compose.yml` if you
don't have `nvidia-container-toolkit` installed; everything runs on CPU too
(slower).

---

## 3. Project layout

```
liss4-cloud-removal/
├── config/config.yaml          # single source of truth for all settings
├── src/
│   ├── data/                   # cloud masking, preprocessing, dataset, augmentation
│   ├── models/                 # generator, discriminator, losses, diffusion, factory
│   ├── training/                # trainer, LR scheduler, train.py CLI
│   ├── inference/               # sliding-window predictor, predict.py CLI
│   ├── evaluation/              # metrics (PSNR/SSIM/SAM/RMSE/ERGAS/MAE), evaluate.py CLI
│   └── utils/                   # config loader, logging, geo I/O, checkpoints, plots
├── api/                         # FastAPI serving layer (app.py, schemas.py)
├── scripts/                     # sample-data generation, dataset prep, run scripts
├── tests/                       # pytest unit + API tests
├── docs/architecture.md         # design rationale — read this
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

---

## 4. Step-by-step usage

### 4.1 Prepare data

**Option A — synthetic demo data** (no downloads needed):
```bash
python scripts/generate_sample_data.py --num_scenes 14 --scene_size 512
```

**Option B — real data.** Download LISS-IV scenes from
[Bhoonidhi](https://bhoonidhi.nrsc.gov.in/) and (optionally) matching
Sentinel-1/2 tiles from [Copernicus Data Space](https://dataspace.copernicus.eu/),
then place them as:
```
data/raw/cloud_free/<scene_id>.npz     # key "image", shape (4, H, W), float32 in native DN or pre-normalized
data/raw/cloudy/<scene_id>.npz         # keys "image", "mask" (optional — real cloud mask if you have one)
data/raw/sentinel1_sar/<scene_id>.npz  # key "image", shape (2, H, W) — VV, VH, co-registered to the LISS-IV grid
```
If your data is in GeoTIFF form instead, use `src/utils/geo_utils.py`
(`read_raster`, `reproject_to_reference`) to load/co-register, then
`np.savez_compressed` into the layout above — see
`scripts/generate_sample_data.py` for a template.

### 4.2 Extract patches + train/val/test split

```bash
python scripts/prepare_dataset.py --config config/config.yaml
```
Writes patches to `data/processed/patches/` and a scene-level
`data/processed/splits.json` (train/val/test split is done **per scene**,
not per patch, to avoid spatial leakage).

### 4.3 Train

```bash
python -m src.training.train --config config/config.yaml
```

Common overrides (no need to edit the YAML for one-off experiments):
```bash
python -m src.training.train --config config/config.yaml \
    --override training.batch_size=8 training.epochs=100 model.architecture=diffusion
```

Resume from a checkpoint:
```bash
python -m src.training.train --config config/config.yaml --resume checkpoints/ckpt_epoch0050.pt
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

**What gets saved:** `checkpoints/ckpt_best.pt` (best validation PSNR),
`ckpt_epochNNNN.pt` (periodic), `ckpt_final.pt`. Every checkpoint embeds the
exact model config used to train it, so evaluation/inference always load
the right architecture automatically — see §6 below.

### 4.4 Evaluate on the held-out test split

```bash
python -m src.evaluation.evaluate \
    --config config/config.yaml \
    --checkpoint checkpoints/ckpt_best.pt \
    --out_dir outputs/evaluation
```

Produces:
- `outputs/evaluation/metrics_per_sample.csv` — per-patch PSNR/SSIM/SAM/RMSE/ERGAS/MAE
- `outputs/evaluation/metrics_summary.json` — mean/std/min/max across the test set
- `outputs/evaluation/comparison_*.png` — qualitative panels (cloudy input, mask, reconstruction, ground truth, error map)

### 4.5 Run inference on a full scene

```bash
python -m src.inference.predict \
    --config config/config.yaml \
    --checkpoint checkpoints/ckpt_best.pt \
    --input path/to/cloudy_scene.tif \
    --sar path/to/sar_scene.tif \
    --output outputs/reconstructed_scene.tif
```
(`--sar` is optional.) This tiles the full scene with Gaussian-blended
sliding-window inference + test-time augmentation, and writes a
georeferenced cloud-free GeoTIFF preserving the input CRS/transform.

Or via the shell wrapper: `./scripts/run_inference.sh <ckpt> <input.tif> <output.tif> [sar.tif]`

### 4.6 Serve as a REST API

```bash
export CLOUD_REMOVAL_CHECKPOINT=checkpoints/ckpt_best.pt
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
    -F "cloudy_tif=@scene.tif" \
    -F "sar_tif=@sar.tif"
# -> {"job_id": "...", "output_path": "...", "cloud_fraction_detected": 0.18, "processing_time_seconds": 4.2}

curl -O http://localhost:8000/download/<job_id>
```

Interactive API docs: `http://localhost:8000/docs` (Swagger UI, auto-generated by FastAPI).

---

## 5. Configuration

Everything is controlled from [`config/config.yaml`](config/config.yaml):
data paths, patch size, model architecture and hyperparameters, loss
weights, training schedule, inference blending strategy, evaluation
metrics. CLI `--override key.path=value` overrides let you sweep
experiments without editing the file. Key switches:

| Setting | Effect |
|---|---|
| `model.architecture` | `sar_fusion_attention_gan` (default, fast) or `diffusion` (slower, higher sample diversity) |
| `model.generator.attention` | `cbam`, `self_attention`, or `none` |
| `model.generator.use_temporal` | enable multi-temporal reference fusion (requires wiring, see `docs/architecture.md` §9) |
| `training.loss_weights.*` | rebalance L1 / SSIM / SAM / perceptual / adversarial / TV terms |
| `inference.tta` | test-time-augmentation (flip-averaging) for a small quality/latency trade |
| `inference.blend_mode` | `gaussian` (seamless tiling) vs `linear`/`none` |

---

## 6. Testing

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v
```

Covers: cloud-mask detection, synthetic cloud simulation, dataset/patch
extraction, all loss functions and model forward passes (including a
regression test guarding against the `SpectralAngleLoss` NaN-gradient
issue described in `docs/architecture.md` §4), evaluation metrics, and
FastAPI endpoint wiring. All 19+ tests pass on this codebase as shipped.

---

## 7. Extending the project

- **New architecture:** implement it under `src/models/`, register it in
  `src/models/model_factory.py::build_generator`, add a config branch — the
  training/eval/inference pipelines are architecture-agnostic and need no
  changes.
- **Real learned cloud segmentation instead of the classical detector:**
  swap `src/data/cloud_mask.py::combined_occlusion_mask` for a model call
  with the same `(C,H,W) -> (H,W) uint8` interface.
- **Multi-temporal fusion:** see `docs/architecture.md` §9 for the exact
  steps (the model architecture already supports it; only the dataset
  needs extending).
- **New evaluation metric:** add a function to `src/evaluation/metrics.py`
  and register it in `METRIC_FUNCS`; it's automatically picked up by
  `evaluate.py` and the API's `/evaluate` endpoint via
  `evaluation.metrics` in the config.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `rasterio`/GDAL install fails | Install system GDAL first (§2.2), or use the conda path |
| `RuntimeError: size mismatch` loading a checkpoint | Shouldn't happen — `evaluate.py`/`predict.py` reconcile architecture from the checkpoint's embedded config automatically. If it still does, the checkpoint predates this feature; retrain or manually match `model.*` config to the training run |
| Training loss becomes `NaN` | Already fixed in this codebase (see `docs/architecture.md` §4); if you re-introduce a custom loss with `acos`/`asin`/`log`, clamp inputs well inside the singular boundary, not just at it |
| `/predict` API call is slow / high memory | Make sure you're not loading a second model per-request — the shipped API loads once at startup and reuses it (`api/app.py`) |
| `No module named 'tensorboard'` | `pip install tensorboard` (listed in `requirements.txt`; only missing if you installed a partial subset) |

---

## 9. License & attribution

Add your organization's license here. LISS-IV data is distributed by NRSC/
ISRO via Bhoonidhi under their respective terms of use; Sentinel-1/2 data
is distributed by ESA/Copernicus under the Copernicus open data policy.
Ensure compliance with both when using real satellite data with this
pipeline.
