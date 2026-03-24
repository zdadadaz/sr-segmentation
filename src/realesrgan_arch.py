"""
Real-ESRGAN Architecture with Segmentation Integration
Contains RRDBNet and SegGuidedRRDBNet implementation.
"""

import math
import torch
from torch import nn as nn
from torch.nn import functional as F

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """Standard RRDBNet without SFT.

    Used for the mask-concat training mode where the segmentation mask is
    concatenated to the LR input channels before the network (no SFT needed).
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if self.scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, seg_map=None):
        if seg_map is not None and x.shape[1] == 3:
            # Mask-concat mode internal handling
            m = seg_map if seg_map.ndim == 4 else seg_map.unsqueeze(1)
            if m.shape[2:] != x.shape[2:]:
                m = F.interpolate(m, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, m.float()], dim=1)

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))

        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SegGuidedRRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN and Real-ESRGAN, integrated with segmentation guidance (SFT).
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4, num_seg_classes=2):
        super(SegGuidedRRDBNet, self).__init__()
        self.scale = scale
        self.num_seg_classes = num_seg_classes
        
        # SFT block import
        from .sr_integration import SFTBlock

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Modify to hold a list of RRDB blocks and SFT blocks
        self.body_blocks = nn.ModuleList([RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        
        # Inject SFT at specific intervals (e.g., every 5 blocks) to save memory, or after the whole body
        # For simplicity and effectiveness, we inject one SFT block before the body, and one after.
        # This is a light-weight B2 implementation.
        self.sft_pre = SFTBlock(num_feat, num_seg_classes)
        self.sft_post = SFTBlock(num_feat, num_seg_classes)

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if self.scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _prepare_seg_map(self, seg_map: torch.Tensor, target_size) -> torch.Tensor:
        """Prepare seg_map for injection"""
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        
        if seg_map.shape[2:] != target_size:
            seg_map = F.interpolate(seg_map, size=target_size, mode='bilinear', align_corners=False)
            
        if seg_map.shape[1] != self.num_seg_classes:
            seg_map_onehot = torch.zeros(seg_map.shape[0], 2, *seg_map.shape[2:], device=seg_map.device)
            seg_map_onehot[:, 0] = 1 - seg_map[:, 0]
            seg_map_onehot[:, 1] = seg_map[:, 0]
            seg_map = seg_map_onehot
        return seg_map

    def forward(self, x, seg_map=None):
        feat = self.conv_first(x)
        
        if seg_map is not None:
            seg_map_resized = self._prepare_seg_map(seg_map, feat.shape[2:])
            feat = self.sft_pre(feat, seg_map_resized)
            
        body_feat = feat
        for block in self.body_blocks:
            body_feat = block(body_feat)
            
        if seg_map is not None:
            body_feat = self.sft_post(body_feat, seg_map_resized)
            
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
            
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
