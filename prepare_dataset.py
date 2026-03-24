"""
Script to prepare Super-Resolution dataset from a folder of HR images.
Generates HR, LR (downsampled), and segmentation masks automatically.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SegmentationPipeline
from src.dataset_generator import DatasetGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare SR-Segmentation Dataset')
    parser.add_argument('--src_dir', type=str, required=True, help='Path to source HR images')
    parser.add_argument('--output_dir', type=str, default='data/custom_dataset', help='Output dataset directory')
    parser.add_argument('--mask_dir', type=str, default=None, help='Path to matching masks (if provided, skips segmentation)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--scale', type=int, default=4, help='Downsampling scale')
    parser.add_argument('--recursive', action='store_true', help='Search for images recursively')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Training set ratio for the split')
    parser.add_argument('--tile', type=int, default=None, help='Tile HR image into specified size (e.g. 512)')
    parser.add_argument('--step', type=int, default=None, help='Stride for tiling (defaults to tile size)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🚀 Initializing preparation pipeline...")
    print(f"  Source: {args.src_dir}")
    if args.mask_dir:
        print(f"  Masks:  {args.mask_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Scale: x{args.scale}")
    
    # Initialize segmentation pipeline
    pipeline = SegmentationPipeline(config_path=args.config)
    
    # Initialize dataset generator
    generator = DatasetGenerator(
        pipeline=pipeline,
        output_dir=args.output_dir,
        scale=args.scale,
        tile_size=args.tile,
        step=args.step
    )
    
    # Process the directory
    print(f"\n🔍 Searching for images in {args.src_dir}...")
    processed_files = generator.process_directory(
        source_dir=args.src_dir,
        mask_dir=args.mask_dir,
        recursive=args.recursive,
        max_images=args.max_images
    )
    
    if not processed_files:
        print("❌ No images found or successfully processed.")
        return
    
    print(f"\n✅ Finished processing {len(processed_files)} images.")
    
    # Generate train/val split
    split_file = os.path.join(args.output_dir, 'split.json')
    print(f"📊 Generating train/val split (ratio={args.split_ratio})...")
    generator.generate_split(train_ratio=args.split_ratio, output_file=split_file)
    print(f"  Saved split info to: {split_file}")
    
    # Print statistics
    generator.print_statistics()
    
    print(f"\n✨ Dataset preparation complete!")
    print(f"  HR images: {os.path.join(args.output_dir, 'hr')}")
    print(f"  LR images: {os.path.join(args.output_dir, 'lr')}")
    print(f"  Masks:     {os.path.join(args.output_dir, 'mask')}")
    print(f"\nYou can now use these directories to start training with train.py.")

if __name__ == '__main__':
    main()
