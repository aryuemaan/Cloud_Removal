"""Learning-rate scheduler factory with linear warmup + cosine decay, the
standard recipe for stable GAN/diffusion training at scale."""
from __future__ import annotations

import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(optimizer, cfg):
    return WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        total_epochs=cfg.training.epochs,
    )
