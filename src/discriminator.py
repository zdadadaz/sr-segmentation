"""
VGG-style discriminator for GAN-based SR training.

Architecture follows SRGAN / Real-ESRGAN: stacked Conv-BN-LeakyReLU blocks
with stride-2 downsampling, ending with a linear classifier.
"""

import torch
import torch.nn as nn


def _disc_block(in_ch, out_ch, stride):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)]
    if stride == 2:          # first conv of each level has no BN
        layers += [nn.BatchNorm2d(out_ch)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class VGGDiscriminator(nn.Module):
    """
    VGG-style discriminator operating on HR / SR patches.

    Input:  (B, num_in_ch, H, W)   — typically HR or SR image (3 channels)
    Output: (B, 1)                  — real/fake logit (no sigmoid; use with BCEWithLogitsLoss)

    Architecture (default num_feat=64):
      Conv(3→64) LeakyReLU
      Conv(64→64,  stride=2) BN LeakyReLU
      Conv(64→128, stride=1) BN LeakyReLU
      Conv(128→128, stride=2) BN LeakyReLU
      Conv(128→256, stride=1) BN LeakyReLU
      Conv(256→256, stride=2) BN LeakyReLU
      Conv(256→512, stride=1) BN LeakyReLU
      Conv(512→512, stride=2) BN LeakyReLU
      AdaptiveAvgPool → Linear(512→1024) LeakyReLU → Linear(1024→1)
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64):
        super().__init__()
        F = num_feat
        self.features = nn.Sequential(
            # no BN on first layer
            nn.Conv2d(num_in_ch, F, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # level 1
            nn.Conv2d(F,    F,    3, stride=2, padding=1), nn.BatchNorm2d(F),    nn.LeakyReLU(0.2, inplace=True),
            # level 2
            nn.Conv2d(F,    F*2,  3, stride=1, padding=1), nn.BatchNorm2d(F*2),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(F*2,  F*2,  3, stride=2, padding=1), nn.BatchNorm2d(F*2),  nn.LeakyReLU(0.2, inplace=True),
            # level 3
            nn.Conv2d(F*2,  F*4,  3, stride=1, padding=1), nn.BatchNorm2d(F*4),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(F*4,  F*4,  3, stride=2, padding=1), nn.BatchNorm2d(F*4),  nn.LeakyReLU(0.2, inplace=True),
            # level 4
            nn.Conv2d(F*4,  F*8,  3, stride=1, padding=1), nn.BatchNorm2d(F*8),  nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(F*8,  F*8,  3, stride=2, padding=1), nn.BatchNorm2d(F*8),  nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(F * 8, F * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(F * 16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
