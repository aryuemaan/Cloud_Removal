"""
Training entry point.

Usage:
    python -m src.training.train --config config/config.yaml
    python -m src.training.train --config config/config.yaml --override training.batch_size=8
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from src.data.dataset import build_dataloaders
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_overrides
from src.utils.logger import get_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for p in pairs or []:
        key, value = p.split("=", 1)
        # naive type coercion
        try:
            value = eval(value, {}, {})
        except Exception:
            pass
        overrides[key] = value
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Train the LISS-IV cloud removal model.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = merge_overrides(cfg, parse_overrides(args.override))

    set_seed(cfg.project.seed)
    logger = get_logger("train_main", cfg.paths.logs_dir, cfg.logging.level)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Architecture: {cfg.model.architecture} | Device: {cfg.project.device}")

    loaders = build_dataloaders(cfg)
    logger.info(
        f"Dataset sizes -> train: {len(loaders['train'].dataset)}, "
        f"val: {len(loaders['val'].dataset)}, test: {len(loaders['test'].dataset)}"
    )

    trainer = Trainer(cfg, loaders)

    if args.resume:
        from src.utils.checkpoint import load_checkpoint
        state = load_checkpoint(args.resume, map_location=str(trainer.device))
        trainer.generator.load_state_dict(state["generator"])
        trainer.discriminator.load_state_dict(state["discriminator"])
        trainer.opt_g.load_state_dict(state["opt_g"])
        trainer.opt_d.load_state_dict(state["opt_d"])
        trainer.best_psnr = state.get("best_psnr", -float("inf"))
        logger.info(f"Resumed training from {args.resume} (epoch {state.get('epoch')})")

    trainer.fit()


if __name__ == "__main__":
    main()
