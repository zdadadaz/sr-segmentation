"""
Training Script for Real-ESRGAN with Segmentation Guidance
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.realesrgan_arch import SegGuidedRRDBNet, RRDBNet
from src.dataset import create_dataloader
from src.sr_integration import SegAwareLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train Seg-Guided Real-ESRGAN')
    parser.add_argument('--hr_dir', type=str, required=True, help='Path to HR images')
    parser.add_argument('--lr_dir', type=str, required=True, help='Path to LR images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to segmentation masks')
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=256, help='HR crop size')
    parser.add_argument('--scale', type=int, default=4, help='Super resolution scale')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    # Model type
    parser.add_argument(
        '--model_type', type=str, default='sft', choices=['sft', 'mask_concat'],
        help=(
            'sft: SegGuidedRRDBNet with SFT injection (default). '
            'mask_concat: standard RRDBNet with mask concatenated to LR input (4-ch), no SFT.'
        )
    )

    # Loss args
    parser.add_argument('--hair_weight', type=float, default=2.0, help='Weight for hair/fur regions')
    parser.add_argument('--other_weight', type=float, default=1.0, help='Weight for non-hair regions')
    parser.add_argument('--no_perceptual', action='store_true', help='Disable perceptual loss')
    parser.add_argument('--no_ssim', action='store_true', help='Disable SSIM loss')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {args.device}")
    
    # Initialize DataLoader
    print("Initializing DataLoader...")
    train_loader = create_dataloader(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        scale=args.scale,
        num_workers=4,
        is_train=True
    )
    print(f"Dataset size: {len(train_loader.dataset)}")
    
    # Initialize Model
    print(f"Initializing Model (model_type={args.model_type})...")
    if args.model_type == 'mask_concat':
        # Mask is concatenated to LR as an extra channel; no SFT.
        model = RRDBNet(
            num_in_ch=4,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=args.scale,
        ).to(args.device)
    else:
        model = SegGuidedRRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=args.scale,
            num_seg_classes=2,
        ).to(args.device)
    
    # Initialize Loss
    print("Initializing SegAwareLoss...")
    criterion = SegAwareLoss(
        hair_weight=args.hair_weight,
        other_weight=args.other_weight,
        use_perceptual=not args.no_perceptual,
        use_ssim=not args.no_ssim
    ).to(args.device)
    
    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            hr_img = batch['hr'].to(args.device)
            lr_img = batch['lr'].to(args.device)
            mask = batch['mask'].to(args.device)
            
            optimizer.zero_grad()

            # Forward pass
            if args.model_type == 'mask_concat':
                # Ensure mask has shape (B, 1, H, W) before concatenating
                mask_ch = mask if mask.ndim == 4 else mask.unsqueeze(1)
                mask_ch = mask_ch.float()
                if mask_ch.shape[2:] != lr_img.shape[2:]:
                    mask_ch = torch.nn.functional.interpolate(
                        mask_ch, size=lr_img.shape[2:], mode='nearest'
                    )
                lr_with_mask = torch.cat([lr_img, mask_ch], dim=1)
                sr_out = model(lr_with_mask)
            else:
                sr_out = model(lr_img, seg_map=mask)

            # Compute loss
            loss = criterion(sr_out, hr_img, mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_type': args.model_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Saved checkpoint: {save_path}")

if __name__ == '__main__':
    main()
