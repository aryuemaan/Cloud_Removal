"""Model checkpointing helpers: save/load/resume training state, and keep
only the top-K + latest checkpoints on disk to control storage costs."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    tag: str,
    keep_last_n: int = 5,
) -> str:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{tag}.pt")
    torch.save(state, ckpt_path)

    # Rotate old checkpoints (excluding 'best')
    ckpts = sorted(
        [p for p in glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch*.pt"))],
        key=os.path.getmtime,
    )
    while len(ckpts) > keep_last_n:
        os.remove(ckpts.pop(0))
    return ckpt_path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "ckpt_epoch*.pt")),
        key=os.path.getmtime,
    )
    return ckpts[-1] if ckpts else None


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    path = os.path.join(checkpoint_dir, "ckpt_best.pt")
    return path if os.path.exists(path) else None


def reconcile_model_config(cfg, checkpoint_state: Dict[str, Any]):
    """
    Rebuild the `model` section of `cfg` from whatever was actually saved
    inside the checkpoint at training time, so evaluate/inference always
    load a checkpoint into an architecturally-matching model — regardless
    of what `config/config.yaml` currently says (it may have been edited
    since training, e.g. for a different experiment).

    Returns a new config object; the original `cfg` is left untouched
    except for its `model` subsection being replaced. Non-model sections
    (paths, inference, evaluation settings, etc.) always come from the
    config passed in, since those are run-specific rather than
    architecture-specific.
    """
    from src.utils.config import ConfigDict
    from src.utils.logger import get_logger

    logger = get_logger("checkpoint")
    ckpt_config = checkpoint_state.get("config")
    if not ckpt_config or "model" not in ckpt_config:
        logger.warning(
            "Checkpoint has no embedded config/model section (older "
            "checkpoint format?) — falling back to the model architecture "
            "defined in the provided --config. This will fail to load if "
            "the architectures don't match."
        )
        return cfg

    current_model = cfg.to_dict().get("model", {})
    if current_model != ckpt_config["model"]:
        logger.info(
            "Model architecture in the provided config differs from the "
            "checkpoint's embedded training config; using the checkpoint's "
            "architecture so weights load correctly."
        )

    merged = cfg.to_dict()
    merged["model"] = ckpt_config["model"]
    return ConfigDict(merged)
