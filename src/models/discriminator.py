"""PatchGAN discriminator (Isola et al., pix2pix-style) operating on the
reconstructed vs. real optical imagery, conditioned on the cloud mask so it
can focus adversarial pressure on reconstructed regions specifically."""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 64, n_layers: int = 3, condition_on_mask: bool = True):
        super().__init__()
        self.condition_on_mask = condition_on_mask
        eff_in = in_channels + (1 if condition_on_mask else 0)

        layers = [
            nn.Conv2d(eff_in, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(ch * 2, base_channels * 8)
            layers += [
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch

        out_ch = min(ch * 2, base_channels * 8)
        layers += [
            nn.Conv2d(ch, out_ch, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, cloud_mask: torch.Tensor = None) -> torch.Tensor:
        if self.condition_on_mask and cloud_mask is not None:
            img = torch.cat([img, cloud_mask], dim=1)
        return self.model(img)
