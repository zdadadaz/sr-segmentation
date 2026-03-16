"""
Training script for the standalone Hair/Fur Segmentation model.
Uses DeepLabV3 with a MobileNetV3 backbone.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

class HairSegDataset(Dataset):
    """
    Dataset for Training Hair/Fur Segmentation model.
    Loads HR images and corresponding masks as binary labels.
    """
    def __init__(self, images_dir, masks_dir, img_size=256, is_train=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.is_train = is_train

        # Use png, jpg
        self.image_paths = sorted([str(p) for p in self.images_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = Path(img_path).name
        
        # Assume mask has the same name but is .png
        mask_path = self.masks_dir / (Path(img_path).stem + '.png')
        
        img = Image.open(img_path).convert('RGB')
        
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L') # Grayscale
        else:
            mask = Image.new('L', img.size, 0)
            
        # Resize to fixed size
        img = TF.resize(img, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
        
        # Convert to tensor
        img = TF.to_tensor(img) # [3, H, W], [0, 1]
        
        # Normalize image using ImageNet stats (expected by torchvision models)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask to tensor [0, 1]
        mask = TF.to_tensor(mask)
        # Binarize mask
        mask = (mask > 0).float()
        
        return img, mask

def create_model(num_classes=1):
    """
    Create DeepLabV3 model with MobileNetV3-Large backbone.
    """
    # Load pretrained DeepLabV3
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )
    
    # Replace the classifier for our specific number of classes
    # Original classifier has 21 classes (PASCAL VOC)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Hair Segmentation Model')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to training masks')
    parser.add_argument('--epochs', type=int, default=20, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image resize dimension')
    parser.add_argument('--save_dir', type=str, default='models/segmentation', help='Directory to save model weights')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"🚀 Using device: {args.device}")
    
    # Dataset and DataLoader
    print("📦 Loading dataset...")
    train_dataset = HairSegDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        img_size=args.img_size,
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"📉 Dataset size: {len(train_dataset)} images")
    
    # Model
    print("🧠 Initializing DeepLabV3 (MobileNetV3 Backbone)...")
    # For binary segmentation, we predict 1 channel and use BCEWithLogitsLoss
    model = create_model(num_classes=1).to(args.device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("🔥 Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(args.device)
            masks = masks.to(args.device) # [B, 1, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass
            # DeepLabV3 returns an OrderedDict with 'out' and 'aux'
            outputs = model(images)['out']
            
            # Loss computation
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"📊 Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically or at the end
        if epoch % 5 == 0 or epoch == args.epochs:
            save_path = os.path.join(args.save_dir, f"hair_seg_mobilenet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved model checkpoint to {save_path}")

if __name__ == '__main__':
    main()
