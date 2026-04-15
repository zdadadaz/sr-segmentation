import argparse
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import cv2

from train import build_model
from src.post_processing import apply_vegetation_post_processing

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
    
    # Vegetation Post-Processing
    parser.add_argument('--veg_method', type=str, default='none', choices=['none', 'noise_synthesis', 'guided_sharpness'], 
                        help='Post-processing method for vegetation areas')
    parser.add_argument('--veg_mask_dir', type=str, default=None, help='Specific path to vegetation masks (if different from --mask_dir)')
    parser.add_argument('--veg_strength', type=float, default=0.15, help='Strength for vegetation enhancement')
    parser.add_argument('--veg_type', type=str, default='auto', choices=['grass', 'tree', 'flower', 'auto'], help='Type of vegetation')
    parser.add_argument('--hr_ref_dir', type=str, default=None, help='Path to HR reference images (required for guided_sharpness)')
    
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
    veg_mask_dir = Path(args.veg_mask_dir) if args.veg_mask_dir else mask_dir
    hr_ref_dir = Path(args.hr_ref_dir) if args.hr_ref_dir else None
    
    extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_paths = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in extensions])
    
    if not image_paths:
        print(f"❌ No images found in {args.input}")
        return
    
    print(f"🔍 Found {len(image_paths)} images. Processing...")
    if args.veg_method != 'none':
        print(f"🌿 Vegetation post-processing enabled: {args.veg_method}")
    
    for img_path in tqdm(image_paths, desc="Inference"):
        # 1. Load Image
        image = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device) # [1, 3, H, W]
        
        # 2. Load Vegetation Mask (if needed)
        veg_mask_np = None
        if args.veg_method != 'none':
            veg_mask_path = veg_mask_dir / (img_path.stem + '.png')
            if not veg_mask_path.exists():
                veg_mask_path = veg_mask_dir / img_path.name
                
            if veg_mask_path.exists():
                veg_mask_pil = Image.open(veg_mask_path).convert('L')
                veg_mask_np = np.array(veg_mask_pil)
                # If it's the official multi-class mask from DatasetGenerator, vegetation is index 4
                if 4 in np.unique(veg_mask_np) and veg_mask_np.max() < 10:
                    veg_mask_np = (veg_mask_np == 4).astype(np.uint8) * 255
                else:
                    veg_mask_np = (veg_mask_np > 127).astype(np.uint8) * 255
        
        # 3. Load SR Mask (hair/fur etc)
        mask_path = mask_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            mask_path = mask_dir / img_path.name
            
        if mask_path.exists():
            mask_pil = Image.open(mask_path).convert('L')
            mask_np_orig = np.array(mask_pil)
            
            # If vegetation post-processing is enabled and we have a veg mask,
            # we should exclude it from the hair/fur SR mask to follow the strategy:
            # "hair_mask = hair_fur_mask & ~veg_mask"
            if veg_mask_np is not None:
                if mask_np_orig.shape[:2] != veg_mask_np.shape[:2]:
                    veg_mask_resized = cv2.resize(veg_mask_np, (mask_np_orig.shape[1], mask_np_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    veg_mask_resized = veg_mask_np
                
                # Subtract vegetation from hair mask
                mask_np_orig = np.where(veg_mask_resized > 127, 0, mask_np_orig)
            
            mask_tensor = TF.to_tensor(Image.fromarray(mask_np_orig)).unsqueeze(0).to(device)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, 1, image.height * args.scale, image.width * args.scale), device=device)


        # 4. Model Forward
        try:
            h, w = img_tensor.shape[2:]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                img_tensor_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                mh, mw = mask_tensor.shape[2:]
                sh, sw = mh // h, mw // w
                mask_tensor_padded = torch.nn.functional.pad(mask_tensor, (0, pad_w * sw, 0, pad_h * sh), mode='constant', value=0)
            else:
                img_tensor_padded = img_tensor
                mask_tensor_padded = mask_tensor

            sr_tensor = model(img_tensor_padded, mask_tensor_padded)
            sr_tensor = sr_tensor[:, :, :h * args.scale, :w * args.scale]
            
            sr_image_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            
            # 5. Vegetation Post-Processing
            if args.veg_method != 'none' and veg_mask_np is not None:
                # Resize veg mask to match SR resolution if needed
                if veg_mask_np.shape[:2] != sr_image_np.shape[:2]:
                    veg_mask_np = cv2.resize(veg_mask_np, (sr_image_np.shape[1], sr_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                hr_image = None
                if args.veg_method == 'guided_sharpness' and hr_ref_dir:
                    hr_path = hr_ref_dir / (img_path.stem + '.png')
                    if not hr_path.exists():
                        hr_path = hr_ref_dir / img_path.name
                    if hr_path.exists():
                        hr_image = np.array(Image.open(hr_path).convert('RGB'))
                    else:
                        print(f"  ⚠️ HR reference not found for {img_path.name}. Skipping guided sharpness.")
                
                sr_image_np = apply_vegetation_post_processing(
                    sr_output=sr_image_np,
                    vegetation_mask=(veg_mask_np > 127),
                    method=args.veg_method,
                    hr_image=hr_image,
                    strength=args.veg_strength,
                    vegetation_type=args.veg_type
                )
            
            # 6. Save result
            sr_image = Image.fromarray(sr_image_np)
            save_path = os.path.join(args.output, f"{img_path.stem}_sr.png")
            sr_image.save(save_path)
            
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"✨ Finished! Results saved to {args.output}")

if __name__ == '__main__':
    main()
