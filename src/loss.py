import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PerceptualLoss(nn.Module):
    """L1 loss on multiple VGG19 feature maps (Real-ESRGAN style)."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Layer indices for VGG19 (ReLU outputs):
        # conv1_2: index  3, weight 0.1
        # conv2_2: index  8, weight 0.1
        # conv3_4: index 17, weight 1.0
        # conv4_4: index 26, weight 1.0
        # conv5_4: index 35, weight 1.0
        self.layer_weights = {
            '3': 0.1, '8': 0.1, '17': 1.0, '26': 1.0, '35': 1.0
        }
        
        self.features = vgg.features[:36].to(device)
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Input images are expected to be in [0, 1] range; clamp just in case.
        x_sr = torch.clamp(sr, 0, 1)
        x_hr = torch.clamp(hr, 0, 1)
        
        loss = 0
        for name, module in self.features._modules.items():
            x_sr = module(x_sr)
            x_hr = module(x_hr)
            if name in self.layer_weights:
                loss += self.layer_weights[name] * F.l1_loss(x_sr, x_hr)
        return loss

class SegAwareLoss(nn.Module):
    """
    Segmentation-aware loss function.
    Applies different loss weights to hair/fur regions vs other regions.
    """
    
    def __init__(
        self,
        hair_weight: float = 1.0,
        other_weight: float = 1.0,
        use_perceptual: bool = False,
        use_ssim: bool = False,
        loss_type: str = 'l1'
    ):
        super().__init__()
        
        self.hair_weight = hair_weight
        self.other_weight = other_weight
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        self.loss_type = loss_type.lower()
        
        # Pixel loss
        if self.loss_type == 'l2':
            self.pixel_loss = nn.MSELoss()
        else:
            self.pixel_loss = nn.L1Loss()
        
        if use_perceptual:
            # Fallback legacy VGG16 perceptual loss (if requested outside GAN)
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.vgg_layers = vgg.features[:23]
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
            self.vgg_layers.eval()

    def forward(
        self,
        sr_output: torch.Tensor,
        hr_target: torch.Tensor,
        seg_map: torch.Tensor
    ) -> torch.Tensor:
        # 1. Handle potential size mismatch due to scale rounding
        if sr_output.shape[2:] != hr_target.shape[2:]:
            h_h, w_h = sr_output.shape[2:]
            hr_target = hr_target[:, :, :h_h, :w_h]
            if seg_map.shape[2:] != (h_h, w_h):
                seg_map = F.interpolate(seg_map, size=(h_h, w_h), mode='nearest')

        # 2. Prepare seg_map
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        
        # Binary mask
        hair_mask = (seg_map > 0.5).float()
        other_mask = 1 - hair_mask
        
        # Pixel loss
        loss_hair = self.pixel_loss(sr_output * hair_mask, hr_target * hair_mask)
        loss_other = self.pixel_loss(sr_output * other_mask, hr_target * other_mask)
        
        loss = (
            self.hair_weight * loss_hair +
            self.other_weight * loss_other
        )
        
        # Perceptual loss
        if self.use_perceptual:
            loss += self._perceptual_loss(sr_output, hr_target, hair_mask, other_mask)
        
        # SSIM loss
        if self.use_ssim:
            loss += self._ssim_loss(sr_output, hr_target, hair_mask, other_mask)
        
        return loss
    
    def _perceptual_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        hair_mask: torch.Tensor,
        other_mask: torch.Tensor
    ) -> torch.Tensor:
        # Legacy VGG16 implementation
        device = sr.device
        self.vgg_layers = self.vgg_layers.to(device)
        
        sr_clipped = torch.clamp(sr, 0, 1)
        hr_clipped = torch.clamp(hr, 0, 1)
        sr_feat = self.vgg_layers(sr_clipped)
        hr_feat = self.vgg_layers(hr_clipped)
        
        # Compute loss with masking
        loss_hair = F.l1_loss(sr_feat * hair_mask[:, :, :sr_feat.size(2), :sr_feat.size(3)],
                             hr_feat * hair_mask[:, :, :hr_feat.size(2), :hr_feat.size(3)])
        loss_other = F.l1_loss(sr_feat * other_mask[:, :, :sr_feat.size(2), :sr_feat.size(3)],
                              hr_feat * other_mask[:, :, :hr_feat.size(2), :hr_feat.size(3)])
        
        return 0.1 * (self.hair_weight * loss_hair + self.other_weight * loss_other)
    
    def _ssim_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        hair_mask: torch.Tensor,
        other_mask: torch.Tensor
    ) -> torch.Tensor:
        from piqa import SSIM
        ssim = SSIM().to(sr.device)
        
        sr_clipped = torch.clamp(sr, 0, 1)
        hr_clipped = torch.clamp(hr, 0, 1)
        
        hair_ssim = 1 - ssim(sr_clipped * hair_mask, hr_clipped * hair_mask)
        other_ssim = 1 - ssim(sr_clipped * other_mask, hr_clipped * other_mask)
        
        return 0.2 * (self.hair_weight * hair_ssim + self.other_weight * other_ssim)
