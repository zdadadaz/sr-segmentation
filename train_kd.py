import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.unet import SegGuidedUNetSR
from src.dataset import create_dataloader

def train_kd():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for SegGuidedUNetSR")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Pretrained teacher checkpoint')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='Epochs to train with only L1 pixel loss')
    parser.add_argument('--kd_epochs', type=int, default=100, help='Epochs to train with KD loss')
    parser.add_argument('--w_pixel', type=float, default=1.0, help='Weight for L1 pixel loss')
    parser.add_argument('--w_feat', type=float, default=0.5, help='Weight for L1 feature distillation')
    parser.add_argument('--w_out', type=float, default=0.3, help='Weight for L1 output distillation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for student')
    args = parser.parse_args()

    # Load config for data
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    total_epochs = args.warmup_epochs + args.kd_epochs

    train_cfg = cfg.get('train', {})
    batch_size = train_cfg.get('batch_size', 4)
    patch_size = train_cfg.get('patch_size', 256)
    scale = cfg.get('model', {}).get('scale', 4)

    data_cfg = cfg.get('data', {})
    hr_dir = data_cfg.get('hr_dir')
    lr_dir = data_cfg.get('lr_dir')
    mask_dir = data_cfg.get('mask_dir')

    if not hr_dir:
        raise ValueError("hr_dir is required in config.")

    print("Initializing DataLoader...")
    train_loader = create_dataloader(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        mask_dir=mask_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        scale=scale,
        is_train=True
    )

    print("Initializing Models...")
    # Teacher: FP32, num_feat=64, conv, num_levels=2
    teacher = SegGuidedUNetSR(num_in_ch=3, num_feat=64, scale=scale, block_type='conv', num_levels=2).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device)['model'])
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # Student: num_feat=32, dwconv, num_levels=1
    student = SegGuidedUNetSR(num_in_ch=3, num_feat=32, scale=scale, block_type='dwconv', num_levels=1).to(device)

    # Feature adapter: Project student 32-ch features to teacher 64-ch features for L1 distance
    adapter = nn.Conv2d(32, 64, 1).to(device)

    # Opt
    optimizer = optim.Adam(list(student.parameters()) + list(adapter.parameters()), lr=args.lr)
    
    criterion_l1 = nn.L1Loss()

    save_dir = os.path.join(train_cfg.get('save_dir', 'experiments/run'), 'kd')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting Training: {args.warmup_epochs} Warmup + {args.kd_epochs} KD = {total_epochs} Epochs Total")

    for epoch in range(1, total_epochs + 1):
        student.train()
        adapter.train()

        # KD schedule Phase:
        is_kd_phase = epoch > args.warmup_epochs

        total_loss = 0.
        total_l_pix = 0.
        total_l_feat = 0.
        total_l_out = 0.

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} (KD: {is_kd_phase})")
        for batch in pbar:
            hr = batch['hr'].to(device)
            # If no LR, fall back to downsampling HR directly for student KD training
            if batch.get('has_lr', torch.ones(1)).all():
                lr = batch['lr'].to(device)
            else:
                lr = F.interpolate(hr, scale_factor=1/scale, mode='bicubic', align_corners=False)
                lr = torch.clamp(lr, 0, 1)

            mask = batch['mask'].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                t_out, t_feat = teacher(lr, mask, return_feat=True)

            s_out, s_feat = student(lr, mask, return_feat=True)

            # 1. Pixel Loss
            l_pix = criterion_l1(s_out, hr)

            # 2. KD Losses (if active)
            l_feat = torch.tensor(0.0).to(device)
            l_out = torch.tensor(0.0).to(device)

            if is_kd_phase:
                # Align spatial size if different levels produce different scales
                # Teacher bottleneck is H/4, W/4
                # Student 1-level bottleneck is H/2, W/2
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    # Pool student features to match teacher resolution
                    s_feat_pooled = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                else:
                    s_feat_pooled = s_feat

                # Map channels 32 -> 64
                s_feat_adapted = adapter(s_feat_pooled)

                l_feat = criterion_l1(s_feat_adapted, t_feat.detach())
                l_out  = criterion_l1(s_out, t_out.detach())

            # Total Loss
            loss = (args.w_pixel * l_pix)
            
            if is_kd_phase:
                loss += (args.w_feat * l_feat) + (args.w_out * l_out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l_pix += l_pix.item()
            if is_kd_phase:
                total_l_feat += l_feat.item()
                total_l_out += l_out.item()

            postfix = {'loss': f"{loss.item():.4f}", 'pix': f"{l_pix.item():.4f}"}
            if is_kd_phase:
                postfix['feat'] = f"{l_feat.item():.4f}"
                postfix['out'] = f"{l_out.item():.4f}"
            pbar.set_postfix(postfix)

        # Logging
        n = len(train_loader)
        msg = f"Epoch [{epoch}/{total_epochs}] Loss: {total_loss/n:.4f} | Pix: {total_l_pix/n:.4f}"
        if is_kd_phase:
            msg += f" | Feat: {total_l_feat/n:.4f} | Out: {total_l_out/n:.4f}"
        print(msg)

        # Save checkpoint
        if epoch % 5 == 0 or epoch == total_epochs:
            ckpt_path = os.path.join(save_dir, f"student_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model': student.state_dict(),
                'adapter': adapter.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt_path)
            print(f"Saved student checkpoint to {ckpt_path}")

if __name__ == '__main__':
    train_kd()
