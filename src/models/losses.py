"""
Composite loss functions for training the cloud-removal GAN.

Combines:
  - Adversarial loss (LSGAN-style, more stable than vanilla BCE-GAN)
  - L1 reconstruction loss, weighted higher inside cloud-occluded regions
  - Perceptual loss via a pretrained VGG16 (applied on a 3-band RGB proxy
    built from the 4-band LISS-IV composite, since VGG expects 3 channels)
  - SSIM structural loss for perceptual/structural fidelity
  - Spectral Angle Mapper (SAM) loss to preserve per-pixel spectral shape
    (critical for multispectral fidelity — a purely per-band L1 loss can
    minimize error while distorting the spectral signature used in land
    cover analysis)
  - Total variation loss for smoothness / artifact suppression
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class AdversarialLoss(nn.Module):
    """LSGAN loss: MSE against real/fake target labels."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.criterion(prediction, target)


class MaskedL1Loss(nn.Module):
    """L1 loss with extra weight applied to cloud-occluded pixels, so the
    model is pushed harder to get the actually-reconstructed region right
    while still supervising the (usually already-correct) clear region."""

    def __init__(self, cloud_weight: float = 2.0):
        super().__init__()
        self.cloud_weight = cloud_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + (self.cloud_weight - 1.0) * mask
        l1 = torch.abs(pred - target) * weight
        return l1.mean()


class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper as a differentiable loss: the angle between the
    predicted and target spectral vectors at each pixel, averaged. Encourages
    spectrally-consistent (not just per-band-accurate) reconstructions.

    Numerical note: `acos` has an unbounded derivative as its argument
    approaches +/-1 (i.e. whenever prediction and target are nearly
    spectrally identical, which is the common/desired case once training
    converges). A naive `torch.acos(cos_angle)` therefore produces NaN
    gradients in practice, not just at pathological inputs. We instead
    clamp well inside the valid range (not just at the boundary) so the
    gradient stays bounded, trading a small amount of loss-value accuracy
    very close to 0 for training stability.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        dot = (pred * target).sum(dim=1)
        pred_norm = pred.norm(dim=1).clamp_min(1e-8)
        target_norm = target.norm(dim=1).clamp_min(1e-8)
        cos_angle = (dot / (pred_norm * target_norm)).clamp(-1 + eps, 1 - eps)
        angle = torch.acos(cos_angle)
        return angle.mean()


class TotalVariationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w


class SSIMLoss(nn.Module):
    """Differentiable structural similarity loss (1 - SSIM), implemented
    directly to avoid a hard dependency on external SSIM packages for the
    multi-channel case."""

    def __init__(self, window_size: int = 11, channels: int = 4):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._create_window(window_size, channels))

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = pred.shape[1]
        if c != self.channels:
            window = self._create_window(self.window_size, c).to(pred.device)
        else:
            window = self.window.to(pred.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=c)
        mu2 = F.conv2d(target, window, padding=pad, groups=c)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=c) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss on ImageNet-pretrained VGG16 features. LISS-IV has 4
    bands, so we project to a pseudo-RGB triplet (first 3 bands, or a
    learned 1x1 projection) before feeding VGG."""

    def __init__(self, in_channels: int = 4, resize: bool = True):
        super().__init__()
        try:
            vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        except Exception:
            # Offline / air-gapped environments (no access to download
            # pretrained weights): fall back to randomly-initialized VGG.
            # Perceptual loss is then a fixed-random-projection feature
            # matching term rather than a true ImageNet-perceptual term —
            # still provides useful high-frequency structural gradient
            # signal, but weights should be downloaded when connectivity
            # is available for best results.
            vgg = tv_models.vgg16(weights=None).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.resize = resize
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.proj.weight, 0)
        with torch.no_grad():
            for i in range(min(3, in_channels)):
                self.proj.weight[i, i, 0, 0] = 1.0
        for p in self.proj.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rgb = self._prep(pred)
        target_rgb = self._prep(target)
        f_pred = self.vgg(pred_rgb)
        with torch.no_grad():
            f_target = self.vgg(target_rgb)
        return F.l1_loss(f_pred, f_target)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2  # from [-1,1] to [0,1]
        x = self.proj(x).clamp(0, 1)
        if self.resize and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class CompositeGeneratorLoss(nn.Module):
    """Weighted sum of all reconstruction-side losses, used for the
    generator's optimization step (adversarial term computed separately by
    the trainer using the discriminator output)."""

    def __init__(self, weights: dict, in_channels: int = 4):
        super().__init__()
        self.weights = weights
        self.l1 = MaskedL1Loss(cloud_weight=weights.get("cloud_mask_focus", 2.0))
        self.ssim = SSIMLoss(channels=in_channels)
        self.sam = SpectralAngleLoss()
        self.tv = TotalVariationLoss()
        self.perceptual = VGGPerceptualLoss(in_channels=in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
        losses = {
            "l1": self.l1(pred, target, mask) * self.weights.get("l1_reconstruction", 100.0),
            "ssim": self.ssim(pred, target) * self.weights.get("ssim", 5.0),
            "sam": self.sam(pred, target) * self.weights.get("spectral_angle", 5.0),
            "tv": self.tv(pred) * self.weights.get("total_variation", 0.1),
            "perceptual": self.perceptual(pred, target) * self.weights.get("perceptual_vgg", 10.0),
        }
        losses["total"] = sum(losses.values())
        return losses
