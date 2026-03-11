"""
SR model integration with SFT (PR6)

Implements:
- B1: Input concat (seg map as additional channels)
- B2: SFT (Spatial Feature Transform) multi-layer injection
- B3: Seg-conditioned loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class SegInputConcatSR(nn.Module):
    """
    B1: SR model with segmentation map as additional input channels
    
    Modifies first conv layer to accept RGB + Seg channels
    """
    
    def __init__(
        self,
        sr_model: nn.Module,
        num_seg_classes: int = 2,
        freeze_pretrained: bool = True
    ):
        """
        Initialize with segmentation-concatenated input
        
        Args:
            sr_model: Base SR model
            num_seg_classes: Number of segmentation classes
            freeze_pretrained: Whether to freeze pretrained weights
        """
        super().__init__()
        self.sr_model = sr_model
        self.num_seg_classes = num_seg_classes
        
        # Modify first conv layer
        self._modify_first_conv(freeze_pretrained)
    
    def _modify_first_conv(self, freeze_pretrained: bool):
        """Modify first conv to accept seg channels"""
        # Get original conv
        original_conv = self.sr_model.head
        
        # Create new conv with extra input channels
        in_channels = 3 + self.num_seg_classes
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Copy pretrained weights for original channels
        if hasattr(original_conv, 'weight'):
            with torch.no_grad():
                # Keep original RGB weights
                new_conv.weight[:, :3, :, :] = original_conv.weight
                # Initialize new channels to zero
                new_conv.weight[:, 3:, :, :].zero_()
        
        # Copy bias if exists
        if original_conv.bias is not None:
            with torch.no_grad():
                new_conv.bias = original_conv.bias.clone()
        
        # Replace conv
        self.sr_model.head = new_conv
        
        # Optionally freeze pretrained layers
        if freeze_pretrained:
            for param in self.sr_model.parameters():
                param.requires_grad = False
            
            # Unfreeze the new head
            for param in self.sr_model.head.parameters():
                param.requires_grad = True
    
    def forward(
        self,
        image: torch.Tensor,
        seg_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with segmentation map
        
        Args:
            image: Input image (B, 3, H, W)
            seg_map: Segmentation map (B, num_classes, H, W) or (B, 1, H, W)
            
        Returns:
            SR output (B, 3, H*scale, W*scale)
        """
        # Ensure seg_map has correct shape
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        
        # Resize seg_map to match image if needed
        if seg_map.shape[2:] != image.shape[2:]:
            seg_map = F.interpolate(
                seg_map, size=image.shape[2:], mode='bilinear', align_corners=False
            )
        
        # One-hot encode if needed
        if seg_map.shape[1] != self.num_seg_classes:
            seg_classes = seg_map.shape[1]
            if seg_classes == 1:
                # Binary - expand to 2 classes
                seg_onehot = torch.zeros(
                    seg_map.shape[0], 2, *seg_map.shape[2:],
                    device=seg_map.device
                )
                seg_onehot[:, 0] = 1 - seg_map[:, 0]  # background
                seg_onehot[:, 1] = seg_map[:, 0]      # hair/fur
                seg_map = seg_onehot
            else:
                # Already one-hot or class indices
                pass
        
        # Concatenate
        x = torch.cat([image, seg_map], dim=1)
        
        # Forward through SR model
        return self.sr_model(x)


class SFTBlock(nn.Module):
    """
    Spatial Feature Transform block
    
    Applies per-pixel scale (gamma) and shift (beta) to features
    """
    
    def __init__(self, in_channels: int, seg_channels: int):
        """
        Initialize SFT block
        
        Args:
            in_channels: Number of input feature channels
            seg_channels: Number of segmentation map channels
        """
        super().__init__()
        
        # Segment encoder
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(seg_channels, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
        )
        
        # Gamma (scale) network
        self.gamma_net = nn.Sequential(
            nn.Conv2d(64, in_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        
        # Beta (shift) network
        self.beta_net = nn.Sequential(
            nn.Conv2d(64, in_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        seg_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply SFT to features
        
        Args:
            features: Input features (B, C, H, W)
            seg_map: Segmentation map (B, seg_channels, H, W)
            
        Returns:
            Transformed features
        """
        # Encode segmentation
        seg_feat = self.seg_encoder(seg_map)
        
        # Generate gamma and beta
        gamma = self.gamma_net(seg_feat)
        beta = self.beta_net(seg_feat)
        
        # Apply transform: output = (1 + gamma) * features + beta
        output = (1 + gamma) * features + beta
        
        return output


class SegGuidedSR(nn.Module):
    """
    B2: SFT-based SR model with multi-layer segmentation injection
    
    Injects segmentation information at multiple layers via SFT
    """
    
    def __init__(
        self,
        sr_model: nn.Module,
        num_seg_classes: int = 2,
        injection_layers: list = None,
        freeze_pretrained: bool = True
    ):
        """
        Initialize SFT-guided SR model
        
        Args:
            sr_model: Base SR model
            num_seg_classes: Number of segmentation classes
            injection_layers: List of layer indices to inject SFT
            freeze_pretrained: Whether to freeze pretrained weights
        """
        super().__init__()
        self.sr_model = sr_model
        self.num_seg_classes = num_seg_classes
        
        # Default injection after each residual block
        self.injection_layers = injection_layers or [0, 1, 2, 3, 4]
        
        # Create SFT blocks
        self.sft_blocks = nn.ModuleList([
            SFTBlock(64, num_seg_classes),   # For early layers
            SFTBlock(64, num_seg_classes),
            SFTBlock(128, num_seg_classes),
            SFTBlock(256, num_seg_classes),
            SFTBlock(512, num_seg_classes),
        ])
        
        # Freeze pretrained if requested
        if freeze_pretrained:
            for param in self.sr_model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        image: torch.Tensor,
        seg_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with SFT injection
        
        Args:
            image: Input image (B, 3, H, W)
            seg_map: Segmentation map
            
        Returns:
            SR output
        """
        # Prepare seg_map
        seg_map = self._prepare_seg_map(seg_map, image.shape[2:])
        
        # Forward through SR body with SFT injection
        x = self.sr_model.head(image)
        
        # Apply SFT at specified layers
        for i, block in enumerate(self.sr_model.body):
            x = block(x)
            
            if i in self.injection_layers and i < len(self.sft_blocks):
                x = self.sft_blocks[i](x, seg_map)
        
        # Tail
        x = self.sr_model.tail(x)
        
        return x
    
    def _prepare_seg_map(
        self,
        seg_map: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Prepare seg_map for injection"""
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        
        # Resize to match feature map size
        if seg_map.shape[2:] != target_size:
            seg_map = F.interpolate(
                seg_map, size=target_size, mode='bilinear', align_corners=False
            )
        
        # Convert to one-hot if needed
        if seg_map.shape[1] != self.num_seg_classes:
            seg_map = F.one_hot(
                seg_map.squeeze(1).long(),
                num_classes=self.num_seg_classes
            ).permute(0, 3, 1, 2).float()
        
        return seg_map


class SegAwareLoss(nn.Module):
    """
    B3: Segmentation-aware loss function
    
    Applies different loss weights to hair/fur regions vs other regions
    """
    
    def __init__(
        self,
        hair_weight: float = 1.0,
        other_weight: float = 1.0,
        use_perceptual: bool = True,
        use_ssim: bool = True
    ):
        """
        Initialize seg-aware loss
        
        Args:
            hair_weight: Weight for hair/fur region loss
            other_weight: Weight for other region loss
            use_perceptual: Include perceptual loss
            use_ssim: Include SSIM loss
        """
        super().__init__()
        
        self.hair_weight = hair_weight
        self.other_weight = other_weight
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        
        # L1 loss
        self.l1_loss = nn.L1Loss()
        
        if use_perceptual:
            # VGG-based perceptual loss
            from torchvision.models import vgg16
            
            vgg = vgg16(pretrained=True)
            self.vgg_layers = vgg.features[:23]  # Up to relu4_3
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
            self.vgg_layers.eval()
    
    def forward(
        self,
        sr_output: torch.Tensor,
        hr_target: torch.Tensor,
        seg_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute segmentation-aware loss
        
        Args:
            sr_output: SR output (B, 3, H, W)
            hr_target: HR target (B, 3, H, W)
            seg_map: Hair/fur mask (B, 1, H, W) or (B, H, W)
            
        Returns:
            Combined loss
        """
        # Prepare seg_map
        if seg_map.ndim == 3:
            seg_map = seg_map.unsqueeze(1)
        
        # Resize to match if needed
        if seg_map.shape[2:] != sr_output.shape[2:]:
            seg_map = F.interpolate(
                seg_map, size=sr_output.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        # Binary mask
        hair_mask = (seg_map > 0.5).float()
        other_mask = 1 - hair_mask
        
        # L1 loss
        loss_hair_l1 = self.l1_loss(sr_output * hair_mask, hr_target * hair_mask)
        loss_other_l1 = self.l1_loss(sr_output * other_mask, hr_target * other_mask)
        
        loss = (
            self.hair_weight * loss_hair_l1 +
            self.other_weight * loss_other_l1
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
        """Compute perceptual loss"""
        def extract_features(x):
            return self.vgg_layers(x)
        
        # Extract features
        sr_feat = extract_features(sr * 255)
        hr_feat = extract_features(hr * 255)
        
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
        """Compute SSIM loss"""
        from piqa import SSIM
        
        ssim = SSIM().to(sr.device)
        
        # Compute SSIM with masking
        hair_ssim = 1 - ssim(sr * hair_mask, hr * hair_mask)
        other_ssim = 1 - ssim(sr * other_mask, hr * other_mask)
        
        return 0.2 * (self.hair_weight * hair_ssim + self.other_weight * other_ssim)


def create_seg_sr_model(
    model_type: str = 'B2',
    sr_model: nn.Module = None,
    num_seg_classes: int = 2,
    config: dict = None
) -> nn.Module:
    """
    Factory function to create segmentation-aware SR model
    
    Args:
        model_type: 'B1', 'B2', or 'B3' (B3 is loss, not model)
        sr_model: Base SR model
        num_seg_classes: Number of segmentation classes
        config: Configuration dictionary
        
    Returns:
        SR model with segmentation integration
    """
    if model_type == 'B1':
        return SegInputConcatSR(
            sr_model=sr_model,
            num_seg_classes=num_seg_classes,
            freeze_pretrained=config.get('freeze_pretrained', True) if config else True
        )
    elif model_type == 'B2':
        return SegGuidedSR(
            sr_model=sr_model,
            num_seg_classes=num_seg_classes,
            injection_layers=config.get('injection_layers') if config else None,
            freeze_pretrained=config.get('freeze_pretrained', True) if config else True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_seg_loss(
    config: dict = None
) -> SegAwareLoss:
    """
    Factory function to create segmentation-aware loss
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SegAwareLoss instance
    """
    config = config or {}
    
    return SegAwareLoss(
        hair_weight=config.get('hair_weight', 1.0),
        other_weight=config.get('other_weight', 1.0),
        use_perceptual=config.get('use_perceptual', True),
        use_ssim=config.get('use_ssim', True)
    )
