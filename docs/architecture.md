# Architecture & Design Rationale

This document explains **why** the system is built the way it is, so future
maintainers (or you, in six months) don't have to reverse-engineer the
reasoning from the code alone.

## 1. Problem framing

LISS-IV is a 4-band (Green, Red, NIR, and a fourth band depending on product
variant) high-resolution optical sensor. Optical sensors cannot see through
cloud, so "cloud removal" is really **conditional image inpainting**: given
a cloudy observation (and optionally auxiliary cloud-penetrating data), 
predict what the surface looked like under the cloud.

Two data realities shape the whole design:

1. **Paired (cloudy, cloud-free) images of the exact same ground location
   and near-identical date are rare.** Real supervision mostly comes from
   *different* acquisition dates of the same tile, which introduces
   confounds (seasonal change, illumination, actual land-cover change).
2. **Cloud-free target images are comparatively abundant.** This motivates
   the synthetic cloud simulation pipeline (`src/data/synthetic_clouds.py`):
   procedurally cloud genuine cloud-free scenes to manufacture unlimited,
   perfectly-aligned supervised pairs, and mix in a smaller amount of real
   cloudy/cloud-free pairs where available.

## 2. Why SAR fusion

Sentinel-1 SAR (C-band radar) penetrates cloud cover entirely. It carries
real information about surface roughness/structure that a purely optical
inpainting model has to hallucinate. The generator (`src/models/generator.py`)
fuses SAR through a dedicated encoder stem rather than naive early
concatenation, because SAR and optical statistics differ enough (speckle
noise, very different dynamic range and texture) that treating them
identically at the first convolution tends to let the higher-fidelity
optical channels dominate and the network learns to mostly ignore SAR.

SAR is optional at every stage: if unavailable, the pipeline zero-fills the
SAR input tensor and the model still functions purely as an optical
inpainting network (weaker on large/thick cloud gaps, still useful for
patchy cirrus).

## 3. Why a GAN (with a diffusion alternative)

- **GAN (`sar_fusion_attention_gan`, default)**: fast single-pass inference
  (critical for operational throughput over large mosaics), well-understood
  training recipe, mature literature for satellite image
  restoration/fusion (pix2pix-style conditional GANs, DSen2-CR, GLF-CR).
- **Diffusion (`diffusion`)**: offered as a swappable alternative
  (`src/models/diffusion_model.py`) for cases where sample diversity /
  avoiding GAN mode collapse matters more than inference latency — e.g.
  offline batch reprocessing of an archive where you can afford
  20-50 step DDIM sampling per tile. Selected via
  `model.architecture: diffusion` in `config/config.yaml`; the rest of the
  training/eval/inference pipeline is written to be architecture-agnostic
  through `src/models/model_factory.py`.

Both share the same conditioning inputs (cloudy optical + cloud mask + SAR)
and the same evaluation harness, so architectures can be benchmarked
head-to-head on identical data splits — directly supporting the "Comparative
assessment of different Generative AI architectures" project objective.

## 4. Loss design

A single L1 loss is not sufficient for multispectral reconstruction:

| Loss | Purpose |
|---|---|
| Masked L1 | Pixel-accurate reconstruction, upweighted inside cloud regions so clear-region accuracy (usually already easy) doesn't dilute gradient signal from the actually-hard region |
| SSIM | Structural/perceptual fidelity — avoids blurry, "regression-to-the-mean" outputs that pure L1 encourages |
| Spectral Angle Mapper (SAM) | Preserves per-pixel *spectral shape*, independent of brightness — important because downstream LULC classification cares about spectral signature, not just per-band absolute error |
| VGG perceptual | High-frequency texture realism (LISS-IV has no native 3-band RGB, so a learned 1×1 projection maps 4 bands → pseudo-RGB before the VGG features) |
| Total variation | Suppresses checkerboard/high-frequency GAN artifacts at reconstruction boundaries |
| Adversarial (LSGAN) | Pushes the *distribution* of reconstructed regions to look like real imagery, not just minimize pointwise error |

See `src/models/losses.py::CompositeGeneratorLoss` for the weighted
combination, and `config/config.yaml::training.loss_weights` for the
default weighting (typical of pix2pix-family recipes: L1 dominates by
~1-2 orders of magnitude over the adversarial term).

**Numerical stability note:** `SpectralAngleLoss` originally used
`torch.acos` clamped only to `[-1+eps, 1-eps]` with `eps=1e-8`. Because
`d/dx[arccos(x)] → -∞` as `x → ±1`, and well-trained pixels have
`cos_angle` extremely close to 1, this produced `NaN` gradients in
practice (not just in edge cases) once the model started reconstructing
well. The fix widens the clamp margin (`eps=1e-4`) — a deliberate,
documented trade-off of small numerical bias very close to zero loss in
exchange for bounded gradients. See the docstring in `losses.py` and the
corresponding regression test in `tests/test_models.py`.

## 5. Mask-aware skip connections

Standard U-Net skip connections copy encoder features straight to the
decoder. For inpainting, this is a problem: inside heavily-clouded regions,
the "clean" encoder feature *is itself corrupted* (it encodes the cloud, not
the surface). `UpBlock.forward` in `generator.py` gates skip connections by
the (downsampled) cloud mask, blending toward a spatially-averaged feature
inside cloud regions so the decoder leans more on the SAR/context pathway
there, while still using full-resolution skip detail in clear regions where
it's trustworthy.

## 6. Residual reconstruction, not full regeneration

The generator predicts a *correction* `residual`, and the final output is
`cloudy + residual * cloud_mask`, clamped to valid range — **not** a
from-scratch image. This means:
- Clear-region pixels pass through almost unchanged (residual is masked out
  there), preserving fine detail the network didn't need to learn to
  reproduce.
- Training is more stable (the network starts near an identity mapping
  rather than needing to learn the entire image distribution from noise).

## 7. Operational inference: sliding window + blending

Full LISS-IV scenes vastly exceed the model's training patch size and
typical GPU memory. `src/inference/sliding_window.py` tiles the scene,
predicts each tile independently (batched), and recombines with
**Gaussian-weighted feathering** so overlapping tile predictions blend
smoothly rather than showing visible seams at tile boundaries — a standard
requirement for any tool whose output will be visually inspected or mosaic
-ked with adjacent scenes.

## 8. Checkpoint/config coupling

Every checkpoint embeds the *exact* model config used to train it
(`Trainer._save`). `src/utils/checkpoint.py::reconcile_model_config` uses
this to rebuild the correct architecture at evaluation/inference time,
**regardless of what `config/config.yaml` currently says** — because config
files get edited between experiments, and a silent architecture mismatch
otherwise fails with a wall of cryptic `size mismatch` errors (or worse,
silently loads garbage if shapes happen to coincide). This was caught and
fixed during integration testing of this codebase; see the regression
coverage implied by `evaluate.py`/`predict.py` always calling
`reconcile_model_config` before building the model.

## 9. Extending with temporal fusion

`TemporalEncoder` in `generator.py` and `model.generator.use_temporal` in
the config implement the architecture for fusing a stack of prior
reference observations (e.g. the last 3 cloud-free-ish acquisitions of the
same tile) as extra context. **This is currently disabled by default**
(`use_temporal: false`) because the reference data pipeline
(`src/data/dataset.py`) does not populate a `temporal_stack` tensor — doing
so requires a multi-temporal archive indexed by tile ID, which is
site/deployment-specific. To enable it:

1. Extend `CloudRemovalDataset.__getitem__` to load and stack N prior
   observations for the same scene/tile into a `(C*N, H, W)` tensor.
2. Pass it through the batch dict as `"temporal_stack"`.
3. Set `model.generator.use_temporal: true` in config.
4. Update `Trainer.train_step` / `validate` to pass
   `batch["temporal_stack"]` into the generator call.

The forward pass already handles the case gracefully (zero-fills if
`use_temporal=True` but no stack is supplied for a given batch), but for
best results you want it populated consistently.

## 10. Why classical cloud masking rather than a learned segmentation model

`src/data/cloud_mask.py` uses a deterministic, multi-cue classical detector
(brightness + NDVI exclusion + local smoothness + morphological cleanup)
rather than a trained segmentation network. Reasons:

- It needs **no labeled cloud-mask training data** to bootstrap the rest of
  the pipeline — it directly produces the occlusion masks used as both
  training supervision targets (where to focus reconstruction loss) and
  inference-time triggers (where to reconstruct at all).
- LISS-IV lacks a thermal/cirrus band, ruling out standard Fmask-style
  detection; the implemented heuristic is a practical substitute tuned for
  what's actually available (4 optical bands).
- The interface is intentionally swappable: any future learned cloud
  segmentation model can replace `combined_occlusion_mask` as a drop-in, as
  long as it returns a binary `(H, W)` array.

## 11. Serving architecture

The FastAPI service (`api/app.py`) loads the model **once** at process
startup (`@app.on_event("startup")`) and every `/predict` call reuses that
resident model via `run_inference(..., model=..., device=...)`. An earlier
version of this endpoint reloaded a fresh model from checkpoint on every
request — functionally correct but wasteful (double memory footprint,
extra checkpoint I/O latency per request) — this was caught during
integration testing and fixed; the fix is directly visible as the
`model=`/`device=` parameters on `run_inference` in
`src/inference/predict.py`.

Run one `uvicorn` worker per GPU/device; scale horizontally with multiple
container replicas behind a load balancer rather than multiple workers
sharing one GPU process.
