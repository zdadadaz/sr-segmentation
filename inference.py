"""
Inference script for Seg-Guided Super-Resolution.
Processes an input folder and a mask folder using a trained SR model.
"""

import argparse
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from train import build_model, forward_model

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Seg-Guided SR')
    parser.add_argument('--input', type=str, required=True, help='Path to input images (LR or test images)')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to matching masks')
    parser.add_argument('--output', type=str, default='output/inference', help='Output directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained generator (.pth)')
    parser.add_argument('--config', type=str, default=None, help='Path to training YAML config to load model settings')
    
    # Model config (overrides config file if provided)
    parser.add_argument('--arch', type=str, default=None, choices=['rrdb', 'unet'], help='Generator architecture')
    parser.add_argument('--model_type', type=str, default=None, choices=['sft', 'mask_concat'], help='Mask integration mode')
    parser.add_argument('--scale', type=int, default=None, help='SR scale')
    parser.add_argument('--device', type=str, default='auto', help='cuda | cpu | auto')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            model_cfg = cfg.get('model', {})
            if args.arch is None: args.arch = model_cfg.get('arch', 'rrdb')
            if args.model_type is None: args.model_type = model_cfg.get('model_type', 'sft')
            if args.scale is None: args.scale = model_cfg.get('scale', 4)
    
    # Defaults if not in config or CLI
    if args.arch is None: args.arch = 'rrdb'
    if args.model_type is None: args.model_type = 'sft'
    if args.scale is None: args.scale = 4
    
    return args

@torch.no_grad()
def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"🚀 Using device: {device}")
    
    # Create output dir
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print(f"🧠 Loading model: {args.arch} ({args.model_type}) from {args.model_path}")
    model = build_model(args.arch, args.model_type, args.scale, device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle both full checkpoints (with optimizer etc) and state_dicts
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get image paths
    input_dir = Path(args.input)
    mask_dir = Path(args.mask_dir)
    extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_paths = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in extensions])
    
    if not image_paths:
        print(f"❌ No images found in {args.input}")
        return
    
    print(f"🔍 Found {len(image_paths)} images. Processing...")
    
    for img_path in tqdm(image_paths, desc="Inference"):
        # 1. Load Image
        image = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device) # [1, 3, H, W]
        
        # 2. Load Mask
        # Try to find mask with same stem
        mask_path = mask_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            # Try same extension as image
            mask_path = mask_dir / img_path.name
            
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            mask_tensor = TF.to_tensor(mask).unsqueeze(0).to(device) # [1, 1, H_h, W_h]
            # Ensure mask is binary [0, 1]
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            print(f"  ⚠️ Warning: Mask not found for {img_path.name}. Using empty mask.")
            mask_tensor = torch.zeros((1, 1, image.height * args.scale, image.width * args.scale), device=device)
        
        # 3. Model Forward
        # SFT models expect seg_map at output resolution? 
        # actually, src.pipeline.forward_model handles resizing if needed.
        # But wait, SegGuidedRRDBNet and SegGuidedUNetSR in train.py:
        # they handle it.
        
        try:
            sr_tensor = forward_model(model, img_tensor, mask_tensor, args.model_type)
            
            # 4. Save result
            sr_image = TF.to_pil_image(sr_tensor.squeeze(0).clamp(0, 1))
            save_path = os.path.join(args.output, f"{img_path.stem}_sr.png")
            sr_image.save(save_path)
            
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✨ Finished! Results saved to {args.output}")

if __name__ == '__main__':
    main()
