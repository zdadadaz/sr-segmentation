"""
PyTorch Dataset for Segmentation-Guided SR Training.
Loads HR, LR, and Mask images.
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SRSegDataset(Dataset):
    """
    Dataset for paired HR, LR and Segmentation Mask reading.
    Assumes all images share the same stem name across directories.
    """
    def __init__(self, hr_dir, lr_dir, mask_dir, patch_size=256, scale=4, augment=True, is_train=True):
        super(SRSegDataset, self).__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.is_train = is_train

        # Get all valid HR images
        self.hr_paths = sorted([str(p) for p in self.hr_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
    def __len__(self):
        return len(self.hr_paths)

    def _augment(self, img_l, img_h, mask):
        """Random horizontal/vertical flip and 90 degree rotation."""
        # Horizontal flip
        if random.random() < 0.5:
            img_l = TF.hflip(img_l)
            img_h = TF.hflip(img_h)
            mask = TF.hflip(mask)
        
        # Vertical flip
        if random.random() < 0.5:
            img_l = TF.vflip(img_l)
            img_h = TF.vflip(img_h)
            mask = TF.vflip(mask)
            
        # Rotation 90, 180, 270 degrees
        rot_angle = random.choice([0, 90, 180, 270])
        if rot_angle > 0:
            img_l = TF.rotate(img_l, rot_angle)
            img_h = TF.rotate(img_h, rot_angle)
            mask = TF.rotate(mask, rot_angle)
            
        return img_l, img_h, mask

    def __getitem__(self, index):
        hr_path = self.hr_paths[index]
        filename = Path(hr_path).name
        
        lr_path = self.lr_dir / filename
        mask_path = self.mask_dir / filename
        
        # In case we generate dummy data with different extensions
        if not lr_path.exists():
             lr_path = self.lr_dir / (Path(hr_path).stem + '.png')
        if not mask_path.exists():
             mask_path = self.mask_dir / (Path(hr_path).stem + '.png')

        # Load images
        img_h = Image.open(hr_path).convert('RGB')
        img_l = Image.open(lr_path).convert('RGB')
        
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
        else:
            # Fallback empty mask if missing
            mask = Image.new('L', img_h.size, 0)
            
        # Optional random crop for training
        if self.is_train and self.patch_size > 0:
            w_h, h_h = img_h.size
            if w_h >= self.patch_size and h_h >= self.patch_size:
                lr_patch_size = self.patch_size // self.scale
                # Determine top left
                x = random.randint(0, img_l.size[0] - lr_patch_size)
                y = random.randint(0, img_l.size[1] - lr_patch_size)
                
                # Crop LR
                img_l = TF.crop(img_l, y, x, lr_patch_size, lr_patch_size)
                # Crop HR & Mask
                hx, hy = x * self.scale, y * self.scale
                img_h = TF.crop(img_h, hy, hx, self.patch_size, self.patch_size)
                mask = TF.crop(mask, hy, hx, self.patch_size, self.patch_size)
            else:
                # Resize if too small (mostly for robustness on edge cases)
                img_h = TF.resize(img_h, (self.patch_size, self.patch_size))
                mask = TF.resize(mask, (self.patch_size, self.patch_size))
                img_l = TF.resize(img_l, (self.patch_size // self.scale, self.patch_size // self.scale))
        
        # Apply data augmentations
        if self.augment and self.is_train:
            img_l, img_h, mask = self._augment(img_l, img_h, mask)
            
        # Convert to tensors
        img_l = TF.to_tensor(img_l)  # [3, H_l, W_l]
        img_h = TF.to_tensor(img_h)  # [3, H_h, W_h]
        mask = TF.to_tensor(mask)    # [1, H_h, W_h]
        
        # Ensure mask is binary or scaled hair mask: 
        # Typically class 1 = hair. A simple binarize:
        mask = (mask > 0).float()
        
        return {'lr': img_l, 'hr': img_h, 'mask': mask, 'lq_path': str(lr_path), 'gt_path': hr_path}

def create_dataloader(hr_dir, lr_dir, mask_dir, batch_size=4, patch_size=256, scale=4, num_workers=4, is_train=True):
    dataset = SRSegDataset(hr_dir, lr_dir, mask_dir, patch_size, scale, augment=is_train, is_train=is_train)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
