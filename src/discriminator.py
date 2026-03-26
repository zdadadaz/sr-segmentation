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


class UNetDiscriminatorSN(nn.Module):
    """
    U-Net style discriminator with Spectral Normalization (Real-ESRGAN default).
    Operates spatially (predicts per-pixel real/fake or patch-based),
    enhancing detail capture across scales.
    """
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # Downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # Extra convs for UNet
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # downsample
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # out
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

from torch.nn import functional as F
