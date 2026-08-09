#!/usr/bin/env bash
# End-to-end demo pipeline: generate sample data -> prepare dataset -> train
# a few epochs -> evaluate -> run inference on one scene.
#
# Use this to smoke-test the whole codebase after installation. It uses a
# small model and short training schedule so it finishes in a few minutes
# even on CPU; for real experiments, run each stage directly with your full
# config (see README section 4 for individual commands).
set -euo pipefail

echo "=================================================================="
echo " LISS-IV Cloud Removal - Full Pipeline Smoke Test"
echo "=================================================================="

echo "[1/5] Generating synthetic sample dataset..."
# 14 scenes is the minimum that reliably yields a non-empty val/test split
# under the default 80/10/10 scene-level split (src/data/preprocessing.py
# split_scenes) -- 6 scenes rounds val/test down to 0 scenes each.
python scripts/generate_sample_data.py --num_scenes 14 --scene_size 384

echo "[2/5] Preparing patches + train/val/test split..."
python scripts/prepare_dataset.py --config config/config.yaml

echo "[3/5] Training (short smoke-test run: small model, 2 epochs)..."
# base_channels/depth are reduced from the config.yaml production defaults
# (64/5) purely so this smoke test finishes quickly on a CPU-only machine.
# Drop these overrides (or set them larger) for real training on a GPU.
python -m src.training.train --config config/config.yaml \
    --override \
        training.epochs=2 \
        training.batch_size=4 \
        training.checkpoint_every=1 \
        training.validate_every=1 \
        model.generator.base_channels=16 \
        model.generator.depth=3 \
        model.discriminator.base_channels=16

echo "[4/5] Evaluating on test split..."
python -m src.evaluation.evaluate --config config/config.yaml \
    --checkpoint checkpoints/ckpt_best.pt --out_dir outputs/evaluation

echo "[5/5] Done."
echo "  - Metrics:    outputs/evaluation/metrics_summary.json"
echo "  - Per-sample: outputs/evaluation/metrics_per_sample.csv"
echo "  - Visual QA:  outputs/evaluation/comparison_*.png"
echo
echo "This was a smoke test on a small model / tiny dataset to verify the"
echo "pipeline runs end-to-end. For real results, use real LISS-IV data"
echo "(README section 4.1) and the full-size model (drop the overrides in"
echo "this script, or edit config/config.yaml directly)."
