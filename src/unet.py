"""
UNet Super-Resolution Architecture with Segmentation Guidance.

Two variants:
  UNetSR           — base UNet-SR; set num_in_ch=4 for mask-concat mode
  SegGuidedUNetSR  — UNet-SR with SFT injection at bottleneck + decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        kwargs = {} if self.mode in ('nearest', 'area') else {'align_corners': False}
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, **kwargs)


def _conv(in_ch, out_ch):
    """Conv2d 3×3 + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
    )


class UNetSR(nn.Module):
    """
    U-Net Super-Resolution model.

    Architecture: 2-level encoder-decoder with additive skip connections
    and PixelShuffle sub-pixel upsampling. A bilinear anchor (residual
    upsampling path) is added to the PixelShuffle output for stability.

    Modes:
      num_in_ch=3 — standard RGB input
      num_in_ch=4 — mask-concat: concatenate binary mask as 4th channel
                    before the network (no SFT layers)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.num_out_ch = num_out_ch
        F_ = num_feat

        self.anchor = Interpolate(scale_factor=scale, mode='bilinear')
        self.pooling = nn.AvgPool2d(2, 2)

        # Encoder
        self.conv_first = _conv(num_in_ch, F_)
        self.conv1 = _conv(F_, F_)          # (B, F, H, W)
        self.conv2 = _conv(F_, F_)          # after pool → (B, F, H/2, W/2)
        self.conv3 = _conv(F_, F_)
        self.conv4 = _conv(F_, F_)          # after pool → (B, F, H/4, W/4)
        self.conv5 = _conv(F_, F_)          # bottleneck

        # Decoder — skip connections via addition (same channel count at all levels)
        self.conv6 = _conv(F_, F_)
        self.conv7 = _conv(F_, F_)
        self.conv8 = _conv(F_, F_)
        self.conv9 = _conv(F_, F_)

        # Sub-pixel output
        self.conv_last = nn.Conv2d(F_, num_out_ch * scale ** 2, 3, padding=1)
        self.depth_to_space = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor, seg_map=None) -> torch.Tensor:
        if seg_map is not None and x.shape[1] == 3:
            # Mask-concat mode internal handling
            m = seg_map if seg_map.ndim == 4 else seg_map.unsqueeze(1)
            if m.shape[2:] != x.shape[2:]:
                m = F.interpolate(m, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, m.float()], dim=1)

        # anchor uses only the first 3 channels (RGB)
        xup = self.anchor(x[:, :3])

        # Encoder
        x1 = self.conv_first(x)
        x1 = self.conv1(x1)           # skip-1: (B, F, H, W)

        x2 = self.pooling(x1)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)           # skip-2: (B, F, H/2, W/2)

        x3 = self.pooling(x2)
        x3 = self.conv4(x3)
        x3 = self.conv5(x3)           # bottleneck: (B, F, H/4, W/4)

        # Decoder
        x2r = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2r = x2r + x2                # additive skip
        x2r = self.conv6(x2r)
        x2r = self.conv7(x2r)

        x1r = F.interpolate(x2r, scale_factor=2, mode='bilinear', align_corners=False)
        x1r = x1r + x1               # additive skip
        x1r = self.conv8(x1r)
        x1r = self.conv9(x1r)

        xr = self.conv_last(x1r)
        return self.depth_to_space(xr) + xup


class SegGuidedUNetSR(nn.Module):
    """
    UNetSR with SFT (Spatial Feature Transform) injection.

    Segmentation mask is injected at three points:
      - After the bottleneck (deepest encoder features)
      - After each decoder stage

    This mirrors the design of SegGuidedRRDBNet in realesrgan_arch.py.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        scale: int = 4,
        num_seg_classes: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.num_seg_classes = num_seg_classes
        self.num_out_ch = num_out_ch
        F_ = num_feat

        from .sr_integration import SFTBlock

        self.anchor = Interpolate(scale_factor=scale, mode='bilinear')
        self.pooling = nn.AvgPool2d(2, 2)

        # Encoder (identical to UNetSR)
        self.conv_first = _conv(num_in_ch, F_)
        self.conv1 = _conv(F_, F_)
        self.conv2 = _conv(F_, F_)
        self.conv3 = _conv(F_, F_)
        self.conv4 = _conv(F_, F_)
        self.conv5 = _conv(F_, F_)

        # Decoder
        self.conv6 = _conv(F_, F_)
        self.conv7 = _conv(F_, F_)
        self.conv8 = _conv(F_, F_)
        self.conv9 = _conv(F_, F_)

        self.conv_last = nn.Conv2d(F_, num_out_ch * scale ** 2, 3, padding=1)
        self.depth_to_space = nn.PixelShuffle(scale)

        # SFT blocks — one per injection point, all operating on F_ channels
        self.sft_bottleneck = SFTBlock(F_, num_seg_classes)
        self.sft_dec2 = SFTBlock(F_, num_seg_classes)
        self.sft_dec1 = SFTBlock(F_, num_seg_classes)

    def _seg(self, seg_map: torch.Tensor, target_size) -> torch.Tensor:
        """Resize and one-hot-encode seg_map to match a feature map spatial size."""
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        if seg_map.shape[2:] != target_size:
            seg_map = F.interpolate(seg_map, size=target_size,
                                    mode='bilinear', align_corners=False)
        if seg_map.shape[1] != self.num_seg_classes:
            s = torch.zeros(seg_map.shape[0], 2, *seg_map.shape[2:],
                            device=seg_map.device)
            s[:, 0] = 1 - seg_map[:, 0]
            s[:, 1] = seg_map[:, 0]
            seg_map = s
        return seg_map

    def forward(self, x: torch.Tensor, seg_map=None) -> torch.Tensor:
        xup = self.anchor(x[:, :self.num_out_ch])

        # Encoder
        x1 = self.conv_first(x)
        x1 = self.conv1(x1)

        x2 = self.pooling(x1)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)

        x3 = self.pooling(x2)
        x3 = self.conv4(x3)
        x3 = self.conv5(x3)

        if seg_map is not None:
            x3 = self.sft_bottleneck(x3, self._seg(seg_map, x3.shape[2:]))

        # Decoder
        x2r = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2r = x2r + x2
        x2r = self.conv6(x2r)
        x2r = self.conv7(x2r)

        if seg_map is not None:
            x2r = self.sft_dec2(x2r, self._seg(seg_map, x2r.shape[2:]))

        x1r = F.interpolate(x2r, scale_factor=2, mode='bilinear', align_corners=False)
        x1r = x1r + x1
        x1r = self.conv8(x1r)
        x1r = self.conv9(x1r)

        if seg_map is not None:
            x1r = self.sft_dec1(x1r, self._seg(seg_map, x1r.shape[2:]))

        xr = self.conv_last(x1r)
        return self.depth_to_space(xr) + xup
