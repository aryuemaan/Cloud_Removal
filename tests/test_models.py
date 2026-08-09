import torch

from src.models.discriminator import PatchGANDiscriminator
from src.models.generator import SARFusionAttentionGenerator
from src.models.losses import (
    AdversarialLoss,
    CompositeGeneratorLoss,
    MaskedL1Loss,
    SpectralAngleLoss,
    SSIMLoss,
    TotalVariationLoss,
)


def _make_batch(batch=2, c_opt=4, c_sar=2, size=64):
    cloudy = torch.rand(batch, c_opt, size, size) * 2 - 1
    mask = torch.randint(0, 2, (batch, 1, size, size)).float()
    sar = torch.rand(batch, c_sar, size, size) * 2 - 1
    target = torch.rand(batch, c_opt, size, size) * 2 - 1
    return cloudy, mask, sar, target


def test_generator_forward_shape():
    model = SARFusionAttentionGenerator(
        in_channels_optical=4, in_channels_sar=2, in_channels_mask=1,
        base_channels=16, depth=3, attention="cbam", use_temporal=False,
    )
    cloudy, mask, sar, _ = _make_batch(size=64)
    out, residual = model(cloudy, mask, sar)
    assert out.shape == cloudy.shape
    assert residual.shape == cloudy.shape
    assert out.min() >= -1.0001 and out.max() <= 1.0001


def test_generator_forward_self_attention_variant():
    model = SARFusionAttentionGenerator(
        in_channels_optical=4, in_channels_sar=2, in_channels_mask=1,
        base_channels=16, depth=2, attention="self_attention",
    )
    cloudy, mask, sar, _ = _make_batch(size=32)
    out, _ = model(cloudy, mask, sar)
    assert out.shape == cloudy.shape


def test_discriminator_forward_shape():
    disc = PatchGANDiscriminator(in_channels=4, base_channels=16, n_layers=3)
    _, mask, _, target = _make_batch(size=64)
    out = disc(target, mask)
    assert out.dim() == 4
    assert out.shape[0] == target.shape[0]


def test_losses_are_finite_and_positive():
    cloudy, mask, sar, target = _make_batch(size=32)
    pred = torch.tanh(torch.rand_like(target))

    l1 = MaskedL1Loss()(pred, target, mask)
    ssim = SSIMLoss(channels=4)(pred, target)
    sam = SpectralAngleLoss()(pred, target)
    tv = TotalVariationLoss()(pred)
    adv = AdversarialLoss()

    for val in [l1, ssim, sam, tv]:
        assert torch.isfinite(val)
        assert val.item() >= 0

    disc_out = torch.rand(2, 1, 6, 6)
    adv_real = adv(disc_out, True)
    adv_fake = adv(disc_out, False)
    assert torch.isfinite(adv_real) and torch.isfinite(adv_fake)


def test_composite_generator_loss():
    weights = {
        "l1_reconstruction": 100.0, "ssim": 5.0, "spectral_angle": 5.0,
        "total_variation": 0.1, "perceptual_vgg": 10.0, "cloud_mask_focus": 2.0,
    }
    loss_fn = CompositeGeneratorLoss(weights, in_channels=4)
    cloudy, mask, sar, target = _make_batch(size=32)
    pred = torch.tanh(torch.rand_like(target))
    losses = loss_fn(pred, target, mask)
    assert "total" in losses
    assert torch.isfinite(losses["total"])
