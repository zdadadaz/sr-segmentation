"""
GAN Training Script for SR with Segmentation Guidance.

Implements a Real-ESRGAN-style training loop:
  - Generator  : UNetSR / SegGuidedUNetSR / RRDBNet / SegGuidedRRDBNet
                 (selected via --arch and --model_type, same as train.py)
  - Discriminator: VGGDiscriminator (VGG-style, outputs real/fake logit)
  - G losses   : L1 pixel + VGG perceptual + SegAwareLoss weighting + adversarial
  - D loss     : LSGAN (MSE on real=1 / fake=0 targets, more stable than BCE)

Usage:
  python train_gan.py \\
    --hr_dir data/hr --lr_dir data/lr --mask_dir data/masks \\
    --arch unet --model_type sft \\
    --epochs 100 --save_dir experiments/gan_unet_sft
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.dataset import create_dataloader
from src.discriminator import VGGDiscriminator
from src.sr_integration import SegAwareLoss
from train import build_model, forward_model


# ---------------------------------------------------------------------------
# Perceptual (VGG feature) loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """L1 loss on VGG16 relu4_3 feature maps."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:23]   # up to relu4_3
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_feat = self.features(torch.clamp(sr, 0, 1))
        hr_feat = self.features(torch.clamp(hr, 0, 1))
        return nn.functional.l1_loss(sr_feat, hr_feat)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='GAN Training for Seg-Guided SR')
    # Data
    parser.add_argument('--hr_dir',   type=str, required=True)
    parser.add_argument('--lr_dir',   type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    # Training
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--patch_size', type=int,   default=256)
    parser.add_argument('--scale',      type=int,   default=4)
    parser.add_argument('--save_dir',   type=str,   default='experiments/gan')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    # Architecture (same choices as train.py)
    parser.add_argument('--arch', type=str, default='unet', choices=['rrdb', 'unet'],
                        help='Generator architecture')
    parser.add_argument('--model_type', type=str, default='sft',
                        choices=['sft', 'mask_concat'],
                        help='sft: SFT injection | mask_concat: 4-ch input')
    # Pretrained generator (optional)
    parser.add_argument('--pretrained_g', type=str, default=None,
                        help='Path to pretrained generator checkpoint (from train.py)')
    # Learning rates
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generator LR')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminator LR')
    # Loss weights
    parser.add_argument('--w_pixel',       type=float, default=1.0,
                        help='Weight for L1 pixel loss')
    parser.add_argument('--w_perceptual',  type=float, default=0.1,
                        help='Weight for VGG perceptual loss')
    parser.add_argument('--w_adv',         type=float, default=0.01,
                        help='Weight for adversarial loss (G)')
    parser.add_argument('--hair_weight',   type=float, default=2.0,
                        help='SegAwareLoss weight for hair/fur regions')
    parser.add_argument('--other_weight',  type=float, default=1.0,
                        help='SegAwareLoss weight for non-hair regions')
    # Discriminator
    parser.add_argument('--d_feat', type=int, default=64,
                        help='Discriminator base feature channels')
    # Misc
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    print(f"Device: {device}  |  arch={args.arch}  |  model_type={args.model_type}")

    # ---- DataLoader --------------------------------------------------------
    print("Initializing DataLoader...")
    train_loader = create_dataloader(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        scale=args.scale,
        num_workers=4,
        is_train=True,
    )
    print(f"Dataset size: {len(train_loader.dataset)}")

    # ---- Generator ---------------------------------------------------------
    print("Initializing Generator...")
    netG = build_model(args.arch, args.model_type, args.scale, device)

    if args.pretrained_g:
        ckpt = torch.load(args.pretrained_g, map_location=device)
        netG.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded pretrained G from {args.pretrained_g}")

    # ---- Discriminator -----------------------------------------------------
    print("Initializing Discriminator...")
    netD = VGGDiscriminator(num_in_ch=3, num_feat=args.d_feat).to(device)

    # ---- Losses ------------------------------------------------------------
    criterion_pixel = SegAwareLoss(
        hair_weight=args.hair_weight,
        other_weight=args.other_weight,
        use_perceptual=False,   # perceptual handled separately below
        use_ssim=False,
    ).to(device)

    criterion_perceptual = PerceptualLoss().to(device)

    # LSGAN: real target=1, fake target=0
    criterion_gan = nn.MSELoss()

    # ---- Optimizers --------------------------------------------------------
    optG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.9, 0.99))
    optD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.9, 0.99))

    real_label = torch.ones(args.batch_size, 1, device=device)
    fake_label = torch.zeros(args.batch_size, 1, device=device)

    # ---- Training loop -----------------------------------------------------
    print("Starting GAN Training...")
    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()

        totals = {'g': 0., 'g_pix': 0., 'g_per': 0., 'g_adv': 0., 'd': 0.}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            hr  = batch['hr'].to(device)
            lr  = batch['lr'].to(device)
            mask = batch['mask'].to(device)

            B = hr.size(0)
            # Resize label tensors if batch is smaller than expected (last batch)
            rl = real_label[:B]
            fl = fake_label[:B]

            # ----------------------------------------------------------------
            # (1) Update Discriminator
            # ----------------------------------------------------------------
            optD.zero_grad()

            with torch.no_grad():
                sr = forward_model(netG, lr, mask, args.model_type)

            d_real = netD(hr)
            d_fake = netD(sr.detach())

            loss_d_real = criterion_gan(d_real, rl)
            loss_d_fake = criterion_gan(d_fake, fl)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            loss_d.backward()
            optD.step()

            # ----------------------------------------------------------------
            # (2) Update Generator
            # ----------------------------------------------------------------
            optG.zero_grad()

            sr = forward_model(netG, lr, mask, args.model_type)

            # Pixel loss (seg-aware L1)
            loss_pix = criterion_pixel(sr, hr, mask)

            # Perceptual loss
            loss_per = criterion_perceptual(sr, hr)

            # Adversarial loss (G wants D to output 1 for fake)
            loss_adv = criterion_gan(netD(sr), rl)

            loss_g = (
                args.w_pixel      * loss_pix +
                args.w_perceptual * loss_per +
                args.w_adv        * loss_adv
            )

            loss_g.backward()
            optG.step()

            # ----------------------------------------------------------------
            totals['g']     += loss_g.item()
            totals['g_pix'] += loss_pix.item()
            totals['g_per'] += loss_per.item()
            totals['g_adv'] += loss_adv.item()
            totals['d']     += loss_d.item()

            pbar.set_postfix({
                'G': f"{loss_g.item():.4f}",
                'D': f"{loss_d.item():.4f}",
            })

        n = len(train_loader)
        print(
            f"Epoch [{epoch}/{args.epochs}]  "
            f"G={totals['g']/n:.4f}  "
            f"(pix={totals['g_pix']/n:.4f}  per={totals['g_per']/n:.4f}  adv={totals['g_adv']/n:.4f})  "
            f"D={totals['d']/n:.4f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            base = os.path.join(args.save_dir, f"epoch_{epoch}")
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'model_type': args.model_type,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optG.state_dict(),
            }, base + '_G.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optD.state_dict(),
            }, base + '_D.pth')
            print(f"Saved: {base}_G.pth  /  {base}_D.pth")


if __name__ == '__main__':
    main()
