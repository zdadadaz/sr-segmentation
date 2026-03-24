"""
GAN Training Script for SR with Segmentation Guidance.

Implements a Real-ESRGAN-style training loop:
  - Generator  : UNetSR / SegGuidedUNetSR / RRDBNet / SegGuidedRRDBNet
                 (selected via --arch and --model_type, same as train.py)
  - Discriminator: VGGDiscriminator (VGG-style, outputs real/fake logit)
  - G losses   : L1 pixel + VGG perceptual + SegAwareLoss weighting + adversarial
  - D loss     : LSGAN (MSE on real=1 / fake=0 targets, more stable than BCE)

Usage:
  python train_gan.py --config configs/train.yaml
  python train_gan.py --config configs/train.yaml --epochs 100 --save_dir experiments/gan
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import create_dataloader
from src.discriminator import VGGDiscriminator
from src.sr_integration import SegAwareLoss
from train import build_model, run_validation


# ---------------------------------------------------------------------------
# Perceptual (VGG feature) loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """L1 loss on multiple VGG19 feature maps."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Layer indices for VGG19 (ReLU outputs):
        # conv1_2: index 3, weight 0.1
        # conv2_2: index 8, weight 0.1
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
                loss += self.layer_weights[name] * nn.functional.l1_loss(x_sr, x_hr)
        return loss


def update_ema(netG_ema, netG, decay):
    """
    Update EMA model: netG_ema = netG_ema * decay + netG * (1 - decay)
    """
    with torch.no_grad():
        msd = netG.state_dict()
        emsd = netG_ema.state_dict()
        for key in msd:
            emsd[key].copy_(emsd[key] * decay + (1.0 - decay) * msd[key])


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_config(args, cfg):
    """Fill unset args (None) from YAML config sections, then apply hardcoded defaults."""
    data      = cfg.get('data',  {}) or {}
    model_cfg = cfg.get('model', {}) or {}
    train_cfg = cfg.get('train', {}) or {}
    loss_cfg  = cfg.get('loss',  {}) or {}
    gan_cfg   = cfg.get('gan',   {}) or {}

    # Data paths
    if args.hr_dir   is None: args.hr_dir   = data.get('hr_dir')
    if args.lr_dir   is None: args.lr_dir   = data.get('lr_dir')
    if args.mask_dir is None: args.mask_dir = data.get('mask_dir')

    # Model
    if args.arch       is None: args.arch       = model_cfg.get('arch',       'unet')
    if args.model_type is None: args.model_type = model_cfg.get('model_type', 'sft')
    if args.scale      is None: args.scale      = model_cfg.get('scale',      4)

    # Train
    if args.epochs     is None: args.epochs     = train_cfg.get('epochs',     50)
    if args.batch_size is None: args.batch_size = train_cfg.get('batch_size', 4)
    if args.patch_size is None: args.patch_size = train_cfg.get('patch_size', 256)
    if args.save_dir   is None: args.save_dir   = train_cfg.get('save_dir',   'experiments/gan')
    if args.device     is None:
        d = train_cfg.get('device', 'auto')
        args.device = ('cuda' if torch.cuda.is_available() else 'cpu') if d == 'auto' else d

    # Loss (shared with train.py)
    if args.hair_weight  is None: args.hair_weight  = loss_cfg.get('hair_weight',  2.0)
    if args.other_weight is None: args.other_weight = loss_cfg.get('other_weight', 1.0)
    if args.pixel_loss_type is None: args.pixel_loss_type = loss_cfg.get('pixel_loss_type', 'l1')

    # GAN-specific
    if args.pretrained_g  is None: args.pretrained_g  = gan_cfg.get('pretrained_g')
    if args.lr_g          is None: args.lr_g          = gan_cfg.get('lr_g',         1e-4)
    if args.lr_d          is None: args.lr_d          = gan_cfg.get('lr_d',         1e-4)
    if args.w_pixel       is None: args.w_pixel       = gan_cfg.get('w_pixel',      1.0)
    if args.w_perceptual  is None: args.w_perceptual  = gan_cfg.get('w_perceptual', 0.1)
    if args.w_adv         is None: args.w_adv         = gan_cfg.get('w_adv',        0.01)
    if args.d_feat        is None: args.d_feat        = gan_cfg.get('d_feat',       64)
    if args.save_every    is None: args.save_every    = gan_cfg.get('save_every',   5)
    if args.ema_decay     is None: args.ema_decay     = gan_cfg.get('ema_decay',    0.999)
    if args.gan_loss_type is None: args.gan_loss_type = gan_cfg.get('gan_loss_type', 'l2')

    # Validation
    val_cfg = cfg.get('validation', {}) or {}
    if args.val_enabled is None:  args.val_enabled = val_cfg.get('enabled', False)
    if args.val_lr_dir is None:   args.val_lr_dir = val_cfg.get('val_lr_dir')
    if args.val_mask_dir is None: args.val_mask_dir = val_cfg.get('val_mask_dir')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='GAN Training for Seg-Guided SR')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to training YAML config (e.g. configs/train.yaml)')

    # Data — optional here; can be provided via config
    parser.add_argument('--hr_dir',   type=str, default=None)
    parser.add_argument('--lr_dir',   type=str, default=None)
    parser.add_argument('--mask_dir', type=str, default=None)

    # Training
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--patch_size', type=int,   default=None)
    parser.add_argument('--scale',      type=int,   default=None)
    parser.add_argument('--save_dir',   type=str,   default=None)
    parser.add_argument('--device',     type=str,   default=None, help='cuda | cpu | auto')

    # Architecture
    parser.add_argument('--arch',       type=str, default=None, choices=['rrdb', 'unet'],
                        help='Generator architecture')
    parser.add_argument('--model_type', type=str, default=None, choices=['sft', 'mask_concat'],
                        help='sft: SFT injection | mask_concat: 4-ch input')

    # Loss weights
    parser.add_argument('--hair_weight',  type=float, default=None, help='SegAwareLoss hair weight')
    parser.add_argument('--other_weight', type=float, default=None, help='SegAwareLoss non-hair weight')
    parser.add_argument('--pixel_loss_type', type=str, default=None, choices=['l1', 'l2'], help='Pixel loss: l1 | l2')

    # GAN-specific
    parser.add_argument('--pretrained_g', type=str,   default=None, help='Pretrained generator checkpoint')
    parser.add_argument('--lr_g',         type=float, default=None, help='Generator LR')
    parser.add_argument('--lr_d',         type=float, default=None, help='Discriminator LR')
    parser.add_argument('--w_pixel',      type=float, default=None, help='L1 pixel loss weight')
    parser.add_argument('--w_perceptual', type=float, default=None, help='VGG perceptual loss weight')
    parser.add_argument('--w_adv',        type=float, default=None, help='Adversarial loss weight (G)')
    parser.add_argument('--d_feat',       type=int,   default=None, help='Discriminator base channels')
    parser.add_argument('--save_every',   type=int,   default=None, help='Save checkpoint every N epochs')
    parser.add_argument('--ema_decay',    type=float, default=None, help='Generator EMA decay')
    parser.add_argument('--gan_loss_type', type=str,  default=None, choices=['l1', 'l2'], help='GAN loss: l1 | l2')

    # Validation
    parser.add_argument('--val_enabled', action='store_true', default=None, help='Enable validation during training')
    parser.add_argument('--val_lr_dir', type=str, default=None, help='Path to validation LR images')
    parser.add_argument('--val_mask_dir', type=str, default=None, help='Path to validation masks')

    args = parser.parse_args()

    # Load YAML config and fill unset args
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    _apply_config(args, cfg)

    # Validate required fields
    missing = [n for n, v in [('--hr_dir', args.hr_dir), ('--lr_dir', args.lr_dir), ('--mask_dir', args.mask_dir)] if not v]
    if missing:
        parser.error(f"Missing required fields: {', '.join(missing)} (pass via CLI or config data section)")

    return args


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

    # ---- EMA Generator -----------------------------------------------------
    netG_ema = build_model(args.arch, args.model_type, args.scale, device).to(device)
    netG_ema.load_state_dict(netG.state_dict())
    netG_ema.eval()
    for p_ema in netG_ema.parameters():
        p_ema.requires_grad = False

    # ---- Discriminator -----------------------------------------------------
    print("Initializing Discriminator...")
    netD = VGGDiscriminator(num_in_ch=3, num_feat=args.d_feat).to(device)

    # ---- Losses ------------------------------------------------------------
    criterion_pixel = SegAwareLoss(
        hair_weight=args.hair_weight,
        other_weight=args.other_weight,
        use_perceptual=False,   # perceptual handled separately below
        use_ssim=False,
        loss_type=args.pixel_loss_type,
    ).to(device)

    criterion_perceptual = PerceptualLoss(device).to(device)

    # Adverarial loss: l1 or l2
    if args.gan_loss_type == 'l1':
        criterion_gan = nn.L1Loss()
    else:
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
            hr   = batch['hr'].to(device)
            lr   = batch['lr'].to(device)
            mask = batch['mask'].to(device)

            B  = hr.size(0)
            rl = real_label[:B]
            fl = fake_label[:B]

            # ----------------------------------------------------------------
            # (1) Update Discriminator
            # ----------------------------------------------------------------
            optD.zero_grad()

            with torch.no_grad():
                sr = netG(lr, mask)

            d_real = netD(hr)
            d_fake = netD(sr.detach())

            loss_d = 0.5 * (criterion_gan(d_real, rl) + criterion_gan(d_fake, fl))
            loss_d.backward()
            optD.step()

            # ----------------------------------------------------------------
            # (2) Update Generator
            # ----------------------------------------------------------------
            optG.zero_grad()

            sr = netG(lr, mask)

            loss_pix = criterion_pixel(sr, hr, mask)
            loss_per = criterion_perceptual(sr, hr)
            loss_adv = criterion_gan(netD(sr), rl)

            loss_g = (
                args.w_pixel      * loss_pix +
                args.w_perceptual * loss_per +
                args.w_adv        * loss_adv
            )

            loss_g.backward()
            optG.step()

            # Update EMA
            if args.ema_decay > 0:
                update_ema(netG_ema, netG, args.ema_decay)

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

        # ---- Save Checkpoints ----------------------------------------------
        if epoch % args.save_every == 0 or epoch == args.epochs:
            # Run validation
            if args.val_enabled:
                run_validation(netG, args, epoch, name_prefix="G_")
                if netG_ema:
                    run_validation(netG_ema, args, epoch, name_prefix="G_EMA_")

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
            
            # Save EMA
            torch.save({
                'epoch': epoch,
                'model_state_dict': netG_ema.state_dict(),
            }, base + '_G_ema.pth')
            
            print(f"Saved: {base}_G.pth  /  {base}_D.pth  /  {base}_G_ema.pth")


if __name__ == '__main__':
    main()
