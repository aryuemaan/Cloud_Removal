"""
Trainer class encapsulating the full adversarial training loop:
  - alternating generator / discriminator updates
  - mixed precision (AMP)
  - gradient clipping
  - EMA of generator weights (for smoother inference-time outputs)
  - checkpointing (latest + best-by-validation-PSNR)
  - TensorBoard logging
  - early stopping
"""
from __future__ import annotations

import copy
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluation.metrics import compute_psnr, compute_ssim
from src.models.losses import AdversarialLoss, CompositeGeneratorLoss
from src.models.model_factory import build_discriminator, build_generator
from src.training.scheduler import build_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger


class EMA:
    """Exponential moving average of generator weights — reduces high
    frequency GAN artifacts in the deployed inference model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)


class Trainer:
    def __init__(self, cfg, loaders: dict):
        self.cfg = cfg
        self.loaders = loaders
        self.device = torch.device(cfg.project.device if torch.cuda.is_available() or cfg.project.device == "cpu" else "cpu")
        self.logger = get_logger("trainer", cfg.paths.logs_dir, cfg.logging.level)

        self.generator = build_generator(cfg).to(self.device)
        self.discriminator = build_discriminator(cfg).to(self.device)
        self.ema = EMA(self.generator, decay=0.999)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=cfg.training.lr_generator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.training.lr_discriminator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.sched_g = build_scheduler(self.opt_g, cfg)
        self.sched_d = build_scheduler(self.opt_d, cfg)

        self.gen_loss_fn = CompositeGeneratorLoss(
            cfg.training.loss_weights.to_dict(), in_channels=cfg.model.generator.in_channels_optical
        ).to(self.device)
        self.adv_loss_fn = AdversarialLoss().to(self.device)

        self.scaler_g = GradScaler(enabled=cfg.project.mixed_precision)
        self.scaler_d = GradScaler(enabled=cfg.project.mixed_precision)

        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(cfg.paths.tensorboard_dir) if cfg.logging.use_tensorboard else None
        )

        self.best_psnr = -float("inf")
        self.patience_counter = 0
        self.global_step = 0

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_step(self, batch: dict) -> dict:
        cloudy, mask, target, sar = batch["cloudy"], batch["cloud_mask"], batch["cloud_free"], batch["sar"]

        # ---- Generator forward ----
        with autocast(enabled=self.cfg.project.mixed_precision):
            fake, residual = self.generator(cloudy, mask, sar)

        # ---- Discriminator update ----
        self.opt_d.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg.project.mixed_precision):
            pred_real = self.discriminator(target, mask)
            pred_fake = self.discriminator(fake.detach(), mask)
            loss_d = 0.5 * (
                self.adv_loss_fn(pred_real, True) + self.adv_loss_fn(pred_fake, False)
            )
        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.unscale_(self.opt_d)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.training.gradient_clip_norm)
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()

        # ---- Generator update ----
        self.opt_g.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg.project.mixed_precision):
            pred_fake_for_g = self.discriminator(fake, mask)
            adv_loss = self.adv_loss_fn(pred_fake_for_g, True) * self.cfg.training.loss_weights.adversarial
            recon_losses = self.gen_loss_fn(fake, target, mask)
            total_g_loss = adv_loss + recon_losses["total"]

        self.scaler_g.scale(total_g_loss).backward()
        self.scaler_g.unscale_(self.opt_g)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.cfg.training.gradient_clip_norm)
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        self.ema.update(self.generator)

        return {
            "loss_d": loss_d.item(),
            "loss_g_total": total_g_loss.item(),
            "loss_g_adv": adv_loss.item(),
            **{f"loss_g_{k}": v.item() for k, v in recon_losses.items() if k != "total"},
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.generator.eval()
        psnr_vals, ssim_vals = [], []
        for batch in self.loaders["val"]:
            batch = self._to_device(batch)
            fake, _ = self.generator(batch["cloudy"], batch["cloud_mask"], batch["sar"])
            for i in range(fake.shape[0]):
                pred_np = ((fake[i] + 1) / 2).clamp(0, 1).cpu().numpy()
                target_np = ((batch["cloud_free"][i] + 1) / 2).clamp(0, 1).cpu().numpy()
                psnr_vals.append(compute_psnr(pred_np, target_np))
                ssim_vals.append(compute_ssim(pred_np, target_np))
        self.generator.train()
        return {
            "val_psnr": sum(psnr_vals) / max(len(psnr_vals), 1),
            "val_ssim": sum(ssim_vals) / max(len(ssim_vals), 1),
        }

    def fit(self):
        cfg = self.cfg
        for epoch in range(cfg.training.epochs):
            epoch_start = time.time()
            running = {}
            pbar = tqdm(self.loaders["train"], desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
            for batch in pbar:
                batch = self._to_device(batch)
                logs = self.train_step(batch)
                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + v
                self.global_step += 1
                pbar.set_postfix({"g_loss": f"{logs['loss_g_total']:.3f}", "d_loss": f"{logs['loss_d']:.3f}"})
                if self.writer:
                    for k, v in logs.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)

            self.sched_g.step()
            self.sched_d.step()

            n_batches = len(self.loaders["train"])
            avg = {k: v / n_batches for k, v in running.items()}
            self.logger.info(f"Epoch {epoch+1} | " + " | ".join(f"{k}={v:.4f}" for k, v in avg.items()))

            if (epoch + 1) % cfg.training.validate_every == 0:
                val_metrics = self.validate()
                self.logger.info(f"Epoch {epoch+1} | Validation: {val_metrics}")
                if self.writer:
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch + 1)

                improved = val_metrics["val_psnr"] > self.best_psnr
                if improved:
                    self.best_psnr = val_metrics["val_psnr"]
                    self.patience_counter = 0
                    self._save(epoch, tag="best")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= cfg.training.early_stopping_patience:
                    self.logger.info("Early stopping triggered.")
                    break

            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                self._save(epoch, tag=f"epoch{epoch+1:04d}")

            self.logger.info(f"Epoch {epoch+1} took {time.time() - epoch_start:.1f}s")

        self._save(cfg.training.epochs - 1, tag="final")
        if self.writer:
            self.writer.close()

    def _save(self, epoch: int, tag: str):
        state = {
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "ema_generator": self.ema.shadow.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "best_psnr": self.best_psnr,
            "config": self.cfg.to_dict(),
        }
        path = save_checkpoint(state, self.cfg.paths.checkpoints_dir, tag)
        self.logger.info(f"Saved checkpoint: {path}")
