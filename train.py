"""
Training Script for Real-ESRGAN with Segmentation Guidance
"""

import argparse
import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from src.realesrgan_arch import SegGuidedRRDBNet, RRDBNet
from src.unet import UNetSR, SegGuidedUNetSR
from src.dataset import create_dataloader
from src.sr_integration import SegAwareLoss


def build_model(arch: str, model_type: str, scale: int, device: str):
    """Instantiate generator based on arch and model_type."""
    if arch == 'unet':
        if model_type == 'mask_concat':
            return UNetSR(num_in_ch=4, num_out_ch=3, num_feat=64, scale=scale).to(device)
        else:  # sft
            return SegGuidedUNetSR(num_in_ch=3, num_out_ch=3, num_feat=64, scale=scale, num_seg_classes=2).to(device)
    else:  # rrdb
        if model_type == 'mask_concat':
            return RRDBNet(num_in_ch=4, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale).to(device)
        else:  # sft
            return SegGuidedRRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale, num_seg_classes=2).to(device)




def _apply_config(args, cfg):
    """Fill unset args (None) from YAML config sections, then apply hardcoded defaults."""
    data      = cfg.get('data',  {}) or {}
    model_cfg = cfg.get('model', {}) or {}
    train_cfg = cfg.get('train', {}) or {}
    loss_cfg  = cfg.get('loss',  {}) or {}

    # Data paths
    if args.hr_dir   is None: args.hr_dir   = data.get('hr_dir')
    if args.lr_dir   is None: args.lr_dir   = data.get('lr_dir')
    if args.mask_dir is None: args.mask_dir = data.get('mask_dir')

    # Model
    if args.arch       is None: args.arch       = model_cfg.get('arch',       'rrdb')
    if args.model_type is None: args.model_type = model_cfg.get('model_type', 'sft')
    if args.scale      is None: args.scale      = model_cfg.get('scale',      4)

    # Train
    if args.epochs     is None: args.epochs     = train_cfg.get('epochs',     10)
    if args.batch_size is None: args.batch_size = train_cfg.get('batch_size', 4)
    if args.patch_size is None: args.patch_size = train_cfg.get('patch_size', 256)
    if args.lr         is None: args.lr         = train_cfg.get('lr',         1e-4)
    if args.save_dir   is None: args.save_dir   = train_cfg.get('save_dir',   'experiments')
    if args.device     is None:
        d = train_cfg.get('device', 'auto')
        args.device = ('cuda' if torch.cuda.is_available() else 'cpu') if d == 'auto' else d

    # Loss
    if args.hair_weight  is None: args.hair_weight  = loss_cfg.get('hair_weight',  2.0)
    if args.other_weight is None: args.other_weight = loss_cfg.get('other_weight', 1.0)
    if args.pixel_loss_type is None: args.pixel_loss_type = loss_cfg.get('pixel_loss_type', 'l1')

    # Bool flags: --no_perceptual / --no_ssim default to False (not set).
    # If not explicitly passed on CLI, read from config.
    if not args.no_perceptual:
        args.no_perceptual = not loss_cfg.get('use_perceptual', True)
    if not args.no_ssim:
        args.no_ssim = not loss_cfg.get('use_ssim', True)

    # Validation
    val_cfg = cfg.get('validation', {}) or {}
    if args.val_enabled is None:  args.val_enabled = val_cfg.get('enabled', False)
    if args.val_lr_dir is None:   args.val_lr_dir = val_cfg.get('val_lr_dir')
    if args.val_mask_dir is None: args.val_mask_dir = val_cfg.get('val_mask_dir')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seg-Guided SR (RRDB or UNet)')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to training YAML config (e.g. configs/train.yaml)')

    # Data — optional here; can be provided via config
    parser.add_argument('--hr_dir',   type=str, default=None, help='Path to HR images')
    parser.add_argument('--lr_dir',   type=str, default=None, help='Path to LR images')
    parser.add_argument('--mask_dir', type=str, default=None, help='Path to segmentation masks')

    # Training
    parser.add_argument('--epochs',     type=int,   default=None, help='Total training epochs')
    parser.add_argument('--batch_size', type=int,   default=None, help='Batch size')
    parser.add_argument('--lr',         type=float, default=None, help='Learning rate')
    parser.add_argument('--patch_size', type=int,   default=None, help='HR crop size')
    parser.add_argument('--scale',      type=int,   default=None, help='Super resolution scale')
    parser.add_argument('--save_dir',   type=str,   default=None, help='Directory to save checkpoints')
    parser.add_argument('--device',     type=str,   default=None, help='Device: cuda | cpu | auto')

    # Architecture
    parser.add_argument('--arch',       type=str, default=None, choices=['rrdb', 'unet'],
                        help='Generator architecture')
    parser.add_argument('--model_type', type=str, default=None, choices=['sft', 'mask_concat'],
                        help='sft: SFT injection | mask_concat: 4-ch input')

    # Loss
    parser.add_argument('--hair_weight',  type=float, default=None, help='Weight for hair/fur regions')
    parser.add_argument('--other_weight', type=float, default=None, help='Weight for non-hair regions')
    parser.add_argument('--no_perceptual', action='store_true', help='Disable perceptual loss')
    parser.add_argument('--no_ssim',       action='store_true', help='Disable SSIM loss')
    parser.add_argument('--pixel_loss_type', type=str, default=None, choices=['l1', 'l2'], 
                        help='Pixel loss type: l1 | l2 (mse)')

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
    missing = [n for n, v in [('--hr_dir', args.hr_dir), ('--lr_dir', args.lr_dir)] if not v]
    if missing:
        parser.error(f"Missing required fields: {', '.join(missing)} (pass via CLI or config data section)")

    return args


@torch.no_grad()
def run_validation(model, args, epoch, name_prefix=""):
    """
    Run inference on validation folder and save images.
    """
    from pathlib import Path
    from PIL import Image
    import torchvision.transforms.functional as TF

    if not args.val_enabled or not args.val_lr_dir:
        return

    val_out_dir = os.path.join(args.save_dir, 'validation', f'epoch_{epoch}')
    os.makedirs(val_out_dir, exist_ok=True)
    
    val_lr_path = Path(args.val_lr_dir)
    exts = ['.png', '.jpg', '.jpeg', '.webp']
    images = sorted([p for p in val_lr_path.glob('*') if p.suffix.lower() in exts])
    
    if not images:
        print(f"  [Val] No images found in {args.val_lr_dir}")
        return

    model.eval()
    device = next(model.parameters()).device
    
    print(f"  [Val] Running validation on {len(images)} images -> {val_out_dir}")
    # Process max 10 images for speed during training if desired, but here we do all
    for img_path in images:
        # Load LR
        lr_pil = Image.open(img_path).convert('RGB')
        lr_tensor = TF.to_tensor(lr_pil).unsqueeze(0).to(device)
        
        # Load Mask
        mask_tensor = None
        if args.val_mask_dir:
            val_mask_path = Path(args.val_mask_dir)
            m_path = val_mask_path / (img_path.stem + '.png')
            if not m_path.exists(): m_path = val_mask_path / img_path.name
            
            if m_path.exists():
                mask_pil = Image.open(m_path).convert('L')
                mask_tensor = TF.to_tensor(mask_pil).unsqueeze(0).to(device)
                mask_tensor = (mask_tensor > 0.5).float()
                
        if mask_tensor is None:
            # Fallback empty mask if directory not provided or filename not found
            # mask_tensor should be target resolution (HR)
            mask_tensor = torch.zeros((1, 1, h * args.scale, w * args.scale), device=device)
            
        # Pad to multiple of 4 (for UNet architectures)
        h, w = lr_tensor.shape[2:]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        
        if pad_h > 0 or pad_w > 0:
            # Pad LR image
            lr_tensor = torch.nn.functional.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            # Pad Mask (respecting its relative scale to LR)
            mh, mw = mask_tensor.shape[2:]
            sh, sw = mh // h, mw // w # usually scale
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, pad_w * sw, 0, pad_h * sh), mode='constant', value=0)

        # Inference
        sr_tensor = model(lr_tensor, mask_tensor)
        
        # Crop back to original size
        sr_tensor = sr_tensor[:, :, :h * args.scale, :w * args.scale]
        
        # Save
        sr_pil = TF.to_pil_image(sr_tensor.squeeze(0).clamp(0, 1))
        sr_pil.save(os.path.join(val_out_dir, f"{name_prefix}{img_path.stem}.png"))
    
    model.train()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Device: {args.device}  |  arch={args.arch}  |  model_type={args.model_type}")

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

    print(f"Initializing Model (arch={args.arch}, model_type={args.model_type})...")
    model = build_model(args.arch, args.model_type, args.scale, args.device)

    print("Initializing SegAwareLoss...")
    # If no mask_dir provided, we should probably use equal weights (standard loss)
    if not args.mask_dir:
        print("  [Note] No mask_dir provided. Using standard (equal) weights for loss.")
        args.hair_weight = 1.0
        args.other_weight = 1.0

    criterion = SegAwareLoss(
        hair_weight=args.hair_weight,
        other_weight=args.other_weight,
        use_perceptual=not args.no_perceptual,
        use_ssim=not args.no_ssim,
        loss_type=args.pixel_loss_type,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            hr_img = batch['hr'].to(args.device)
            lr_img = batch['lr'].to(args.device)
            mask   = batch['mask'].to(args.device)

            optimizer.zero_grad()
            sr_out = model(lr_img, mask)
            loss = criterion(sr_out, hr_img, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        # Run validation
        if args.val_enabled:
            run_validation(model, args, epoch)

        save_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'arch': args.arch,
            'model_type': args.model_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Saved checkpoint: {save_path}")


if __name__ == '__main__':
    main()
