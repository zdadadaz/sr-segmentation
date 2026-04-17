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


# ---------------------------------------------------------------------------
# Collapsible Linear Block (CLB) — from SESR
# "Collapsible Linear Blocks for Super-Efficient Super Resolution" (WACV 2022)
# https://arxiv.org/abs/2103.09404
#
# Training:  expand(3×3, no bias) → squeeze(1×1, bias) → activation
# Inference: mathematically equivalent single 3×3 conv → activation
#            obtained by collapsing expand+squeeze into one kernel via the
#            impulse-response trick (see collapse()).
# ---------------------------------------------------------------------------

class CollapsibleLinearBlock(nn.Module):
    """
    CLB (non-residual). Suitable when in_ch != out_ch (e.g. first conv).

    Args:
        in_ch:   input channels
        out_ch:  output channels
        tmp_ch:  intermediate channels (expand width). Default: out_ch * 4.
    """

    def __init__(self, in_ch: int, out_ch: int, tmp_ch: int = None):
        super().__init__()
        if tmp_ch is None:
            tmp_ch = out_ch * 4
        # bias=False on expand is REQUIRED for correct collapse math
        self.conv_expand  = nn.Conv2d(in_ch,   tmp_ch,  3, padding=1, bias=False)
        self.conv_squeeze = nn.Conv2d(tmp_ch,  out_ch,  1)
        self.activation   = nn.LeakyReLU(0.2, inplace=True)
        self.collapsed    = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        """
        Merge expand(3×3) → squeeze(1×1) into a single equivalent 3×3 conv.

        Method: pass an identity (delta) signal through both layers to read
        out the composite impulse response, then flip for cross-correlation.
        """
        if self.collapsed:
            return

        C_in  = self.conv_expand.in_channels
        k     = self.conv_expand.kernel_size[0]
        pad   = k // 2
        dev   = self.conv_expand.weight.device

        # Build delta: batch of C_in images, each with a 1 at (c, pad, pad)
        delta = torch.zeros(C_in, C_in, k, k, device=dev)
        for i in range(C_in):
            delta[i, i, pad, pad] = 1.0

        with torch.no_grad():
            inter         = self.conv_expand(delta)           # [C_in, tmp_ch, k, k]
            kernel_biased = self.conv_squeeze(inter)          # [C_in, C_out,  k, k]
            bias          = self.conv_squeeze.bias.clone()
            kernel        = kernel_biased - bias[None, :, None, None]
            kernel        = torch.flip(kernel, [2, 3])        # cross-corr correction
            kernel        = kernel.permute(1, 0, 2, 3).contiguous()  # [C_out, C_in, k, k]

            C_out    = self.conv_squeeze.out_channels
            new_conv = nn.Conv2d(C_in, C_out, k, padding=pad).to(dev)
            new_conv.weight = nn.Parameter(kernel)
            new_conv.bias   = nn.Parameter(bias)

        self.conv_expand  = new_conv
        self.conv_squeeze = nn.Identity()
        self.collapsed    = True


class ResidualCollapsibleLinearBlock(CollapsibleLinearBlock):
    """
    CLB with residual connection (requires in_ch == out_ch).

    Training:  x + squeeze(expand(x))  → activation
    Collapsed: equivalent single 3×3 conv (residual folded into center pixel)
               → activation
    """

    def __init__(self, in_ch: int, out_ch: int, tmp_ch: int = None):
        assert in_ch == out_ch, "ResidualCollapsibleLinearBlock requires in_ch == out_ch"
        super().__init__(in_ch, out_ch, tmp_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.collapsed:
            return self.activation(self.conv_expand(x))
        return self.activation(x + self.conv_squeeze(self.conv_expand(x)))

    def collapse(self):
        if self.collapsed:
            return
        super().collapse()
        # Fold the residual identity into the collapsed kernel:
        # add 1 to the center spatial position for each (out_ch, in_ch) diagonal.
        k   = self.conv_expand.kernel_size[0]
        mid = k // 2
        with torch.no_grad():
            C = self.conv_expand.in_channels
            for i in range(C):
                self.conv_expand.weight[i, i, mid, mid] += 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv(in_ch: int, out_ch: int, block_type: str = 'conv', tmp_ch: int = None):
    """
    Factory for a single feature-extraction unit.

    block_type='conv'   : standard Conv2d 3×3 + LeakyReLU  (default)
    block_type='clb'    : CollapsibleLinearBlock or ResidualCLB (when in_ch==out_ch)
    block_type='dwconv' : Depthwise separable convolution (3×3 DW + 1×1 PW) + LeakyReLU
    """
    if block_type == 'clb':
        if in_ch == out_ch:
            return ResidualCollapsibleLinearBlock(in_ch, out_ch, tmp_ch)
        else:
            return CollapsibleLinearBlock(in_ch, out_ch, tmp_ch)
    elif block_type == 'dwconv':
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),  # Depthwise
            nn.Conv2d(in_ch, out_ch, 1),                          # Pointwise
            nn.LeakyReLU(0.2, inplace=True),
        )
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

    block_type:
      'conv' — standard Conv2d 3×3 + LeakyReLU (default)
      'clb'  — Collapsible Linear Block (SESR); call collapse_clb() before
               deployment to merge expand+squeeze into one 3×3 conv
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        scale: int = 4,
        block_type: str = 'conv',
    ):
        super().__init__()
        self.scale = scale
        self.num_out_ch = num_out_ch
        F_ = num_feat
        tmp_ch = F_ * 4   # CLB intermediate channels (4× expansion, like SESR)

        B = lambda ic, oc: _conv(ic, oc, block_type, tmp_ch)

        self.anchor = Interpolate(scale_factor=scale, mode='bilinear')
        self.pooling = nn.AvgPool2d(2, 2)

        # Encoder
        self.conv_first = B(num_in_ch, F_)   # non-residual (channels change)
        self.conv1 = B(F_, F_)               # (B, F, H, W)
        self.conv2 = B(F_, F_)               # after pool → (B, F, H/2, W/2)
        self.conv3 = B(F_, F_)
        self.conv4 = B(F_, F_)               # after pool → (B, F, H/4, W/4)
        self.conv5 = B(F_, F_)               # bottleneck

        # Decoder — skip connections via addition (same channel count at all levels)
        self.conv6 = B(F_, F_)
        self.conv7 = B(F_, F_)
        self.conv8 = B(F_, F_)
        self.conv9 = B(F_, F_)

        # Sub-pixel output — plain conv (channels differ; not collapsed)
        self.conv_last = nn.Conv2d(F_, num_out_ch * scale ** 2, 3, padding=1)
        self.depth_to_space = nn.PixelShuffle(scale)

    def collapse_clb(self):
        """Collapse all CLBs into single-conv equivalents (call before deployment)."""
        for m in self.modules():
            if isinstance(m, CollapsibleLinearBlock):
                m.collapse()

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

    block_type: same as UNetSR — 'conv' (default) or 'clb'.
    Note: SFT gamma/beta convolutions always use standard conv2d.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        scale: int = 4,
        num_seg_classes: int = 2,
        block_type: str = 'conv',
        num_levels: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.num_seg_classes = num_seg_classes
        self.num_out_ch = num_out_ch
        self.num_levels = num_levels
        F_ = num_feat
        tmp_ch = F_ * 4

        B = lambda ic, oc: _conv(ic, oc, block_type, tmp_ch)

        from .sr_integration import SFTBlock

        self.anchor = Interpolate(scale_factor=scale, mode='bilinear')
        self.pooling = nn.AvgPool2d(2, 2)

        # Encoder
        self.conv_first = B(num_in_ch, F_)
        self.conv1 = B(F_, F_)
        self.conv2 = B(F_, F_)
        self.conv3 = B(F_, F_)
        
        if self.num_levels >= 2:
            self.conv4 = B(F_, F_)
            self.conv5 = B(F_, F_)

        # Decoder
        if self.num_levels >= 2:
            self.conv6 = B(F_, F_)
            self.conv7 = B(F_, F_)
            
        self.conv8 = B(F_, F_)
        self.conv9 = B(F_, F_)

        self.conv_last = nn.Conv2d(F_, num_out_ch * scale ** 2, 3, padding=1)
        self.depth_to_space = nn.PixelShuffle(scale)

        # SFT blocks
        self.sft_bottleneck = SFTBlock(F_, num_seg_classes)
        if self.num_levels >= 2:
            self.sft_dec2 = SFTBlock(F_, num_seg_classes)
        self.sft_dec1 = SFTBlock(F_, num_seg_classes)

    def collapse_clb(self):
        """Collapse all CLBs into single-conv equivalents (call before deployment)."""
        for m in self.modules():
            if isinstance(m, CollapsibleLinearBlock):
                m.collapse()

    def _seg(self, seg_map: torch.Tensor, target_size) -> torch.Tensor:
        """Resize and one-hot-encode seg_map to match a feature map spatial size."""
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        if seg_map.shape[2:] != target_size:
            seg_map = F.interpolate(seg_map, size=target_size,
                                    mode='bilinear', align_corners=False)
        if seg_map.shape[1] != self.num_seg_classes:
            if self.num_seg_classes == 2 and seg_map.shape[1] == 1:
                s = torch.zeros(seg_map.shape[0], 2, *seg_map.shape[2:],
                                device=seg_map.device)
                s[:, 0] = 1 - seg_map[:, 0]
                s[:, 1] = seg_map[:, 0]
                seg_map = s
            else:
                # Generic multi-class one-hot
                seg_map = F.one_hot(seg_map.squeeze(1).long(), num_classes=self.num_seg_classes)
                seg_map = seg_map.permute(0, 3, 1, 2).float()
        return seg_map

    def forward(self, x: torch.Tensor, seg_map=None, return_feat=False):
        xup = self.anchor(x[:, :self.num_out_ch])

        # Encoder Level 0
        x1 = self.conv_first(x)
        x1 = self.conv1(x1)

        # Encoder Level 1
        x2 = self.pooling(x1)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        
        feat = x2

        if self.num_levels >= 2:
            # Encoder Level 2
            x3 = self.pooling(x2)
            x3 = self.conv4(x3)
            x3 = self.conv5(x3)
            feat = x3

        # SFT Bottleneck
        if seg_map is not None:
            feat = self.sft_bottleneck(feat, self._seg(seg_map, feat.shape[2:]))
            
        bottleneck_feat = feat

        # Decoder
        if self.num_levels >= 2:
            x2r = F.interpolate(bottleneck_feat, scale_factor=2, mode='bilinear', align_corners=False)
            x2r = x2r + x2
            x2r = self.conv6(x2r)
            x2r = self.conv7(x2r)

            if seg_map is not None:
                x2r = self.sft_dec2(x2r, self._seg(seg_map, x2r.shape[2:]))
        else:
            x2r = bottleneck_feat

        x1r = F.interpolate(x2r, scale_factor=2, mode='bilinear', align_corners=False)
        x1r = x1r + x1
        x1r = self.conv8(x1r)
        x1r = self.conv9(x1r)

        if seg_map is not None:
            x1r = self.sft_dec1(x1r, self._seg(seg_map, x1r.shape[2:]))

        xr = self.conv_last(x1r)
        
        out = self.depth_to_space(xr) + xup
        
        if return_feat:
            # For KD: return the output AND the bottleneck feature map
            return out, bottleneck_feat
        return out
