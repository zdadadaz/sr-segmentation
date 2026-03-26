import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

def filter2D(img, kernel):
    """
    Apply 2D filter to image.
    img: (B, C, H, W)
    kernel: (B, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    # Reshape for depthwise convolution
    img = img.view(1, b * c, img.size(-2), img.size(-1))
    kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
    out = F.conv2d(img, kernel, groups=b * c)
    return out.view(b, c, h, w)

class USMSharp(nn.Module):
    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        # Static gaussian kernel for USM
        # In real-esrgan it's often a fixed kernel
        pass

    def forward(self, img, weight=0.5, radius=50, threshold=10):
        # Simple USM: out = img + weight * (img - blur)
        # For simplicity in this script, we'll use a 5x5 gaussian blur
        # This is a placeholder for the more complex BasicSR implementation
        blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
        residual = img - blur
        mask = torch.abs(residual) * 255. > threshold
        mask = mask.float()
        out = img + weight * residual * mask
        return torch.clamp(out, 0, 1)

def random_add_gaussian_noise_pt(img, sigma_range=(1, 30), gray_prob=0.1):
    sigma = random.uniform(sigma_range[0], sigma_range[1]) / 255.
    noise = torch.randn_like(img) * sigma
    if random.random() < gray_prob:
        noise = noise[:, :1, ...].repeat(1, 3, 1, 1)
    return torch.clamp(img + noise, 0, 1)

def random_add_poisson_noise_pt(img, scale_range=(0.05, 3.0), gray_prob=0.1):
    scale = random.uniform(scale_range[0], scale_range[1])
    img_tmp = torch.clamp(img, 0, 1)
    noise = torch.randn_like(img_tmp) * torch.sqrt(img_tmp * scale) / 255.
    if random.random() < gray_prob:
        noise = noise[:, :1, ...].repeat(1, 3, 1, 1)
    return torch.clamp(img + noise, 0, 1)

class ESRGANSynthesizer:
    """
    Synthesizes LQ images from GT images on-the-fly.
    Follows the Real-ESRGAN two-order degradation process.
    """
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.usm = USMSharp()
        
    def synthesize(self, hr, mask=None, opt=None):
        """
        Synthesize LR from HR.
        Returns (lr, gt_usm)
        """
        # Hardcoded default options matching BasicSR/Real-ESRGAN defaults
        if opt is None:
            opt = {
                'resize_prob': [0.2, 0.7, 0.1], # up, down, keep
                'resize_range': [0.15, 1.5],
                'gaussian_noise_prob': 0.5,
                'noise_range': [1, 30],
                'poisson_scale_range': [0.05, 3.0],
                'gray_noise_prob': 0.4,
                'jpeg_range': [30, 95],
                
                'second_blur_prob': 0.8,
                'resize_prob2': [0.3, 0.4, 0.3],
                'resize_range2': [0.3, 1.2],
                'gaussian_noise_prob2': 0.5,
                'noise_range2': [1, 25],
                'poisson_scale_range2': [0.05, 2.5],
                'gray_noise_prob2': 0.4,
                'jpeg_range2': [30, 95]
            }

        gt = hr
        gt_usm = self.usm(gt)
        ori_h, ori_w = gt.size()[2:4]

        # 1. First degradation
        out = gt_usm
        # Blur (skipped here, adding if needed)
        
        # Resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'up':
            s = random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            s = random.uniform(opt['resize_range'][0], 1)
        else:
            s = 1
        out = F.interpolate(out, scale_factor=s, mode='bilinear')
        
        # Noise
        if random.random() < opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(out, opt['noise_range'], opt['gray_noise_prob'])
        else:
            out = random_add_poisson_noise_pt(out, opt['poisson_scale_range'], opt['gray_noise_prob'])
        
        # JPEG (Simplified approximation using resize for now as we don't have Differentiable JPEG)
        # Real-ESRGAN uses DiffJPEG, here we skip or use a simple quality reduction if possible
        
        # 2. Second degradation
        if random.random() < opt['second_blur_prob']:
             out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
             
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            s = random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            s = random.uniform(opt['resize_range2'][0], 1)
        else:
            s = 1
        
        target_h, target_w = ori_h // self.scale, ori_w // self.scale
        out = F.interpolate(out, size=(int(target_h * s), int(target_w * s)), mode='bilinear')
        
        # Noise 2
        if random.random() < opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(out, opt['noise_range2'], opt['gray_noise_prob2'])
        else:
            out = random_add_poisson_noise_pt(out, opt['poisson_scale_range2'], opt['gray_noise_prob2'])
            
        # Resize Back
        out = F.interpolate(out, size=(target_h, target_w), mode='bicubic')
        
        lr = torch.clamp(out, 0, 1)
        
        # Note: In a real system, we'd also sync kernels and JPEG quality.
        # Here we provide a functional skeleton.
        
        return lr, gt_usm
