"""
Test pipeline with real images
Downloads sample images and runs the full segmentation pipeline
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SegmentationPipeline
from utils.visualization import visualize_masks_grid, draw_bboxes, create_side_by_side
from PIL import Image
import numpy as np


def download_test_images():
    """Download test images (animals and humans)"""
    test_dir = Path(__file__).parent / 'test_images'
    test_dir.mkdir(exist_ok=True)
    
    # Test image URLs
    images = {
        'cat.jpg': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800',
        'dog.jpg': 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800',
        'person.jpg': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800',
    }
    
    downloaded = []
    for name, url in images.items():
        path = test_dir / name
        if not path.exists():
            print(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  ✅ Saved to {path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        else:
            print(f"  ⏭️  Already exists: {path}")
        downloaded.append(path)
    
    return downloaded


def test_pipeline_with_image(pipeline, image_path, output_dir):
    """Test pipeline with a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print('='*60)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    print(f"Image shape: {img_np.shape}")
    
    # Run segmentation
    print("Running segmentation...")
    result = pipeline.segment(img_np)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"  Animals detected: {len(result.animal_bboxes)}")
    print(f"  Persons detected: {len(result.person_bboxes)}")
    
    if result.animal_bboxes:
        print(f"  Animal bboxes:")
        for bbox in result.animal_bboxes:
            print(f"    - {bbox.label}: conf={bbox.confidence:.2f}")
    
    if result.person_bboxes:
        print(f"  Person bboxes:")
        for bbox in result.person_bboxes:
            print(f"    - Person: conf={bbox.confidence:.2f}")
    
    # Calculate hair coverage
    hair_pixels = result.final_mask.sum()
    total_pixels = result.final_mask.size
    coverage = hair_pixels / total_pixels * 100
    print(f"  Hair/fur coverage: {coverage:.2f}%")
    
    print(f"\n🤖 Models used:")
    for model, version in result.model_versions.items():
        print(f"    - {model}: {version}")
    
    # Save visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Draw bboxes on image
    bboxes = [b.xyxy for b in result.animal_bboxes] + [b.xyxy for b in result.person_bboxes]
    labels = [b.label for b in result.animal_bboxes] + [b.label for b in result.person_bboxes]
    
    vis_img = draw_bboxes(img_np.copy(), bboxes, labels=labels)
    
    # Save
    from PIL import Image as PILImage
    output_path = output_dir / f"{image_path.stem}_result.jpg"
    PILImage.fromarray(vis_img).save(output_path)
    print(f"\n💾 Saved visualization to: {output_path}")
    
    # Also save the mask
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    mask_img = Image.fromarray(result.final_mask * 255)
    mask_img.save(mask_path)
    print(f"💾 Saved mask to: {mask_path}")
    
    return result


def main():
    print("="*60)
    print("SR Segmentation Pipeline - Real Image Test")
    print("="*60)
    
    # Download test images
    print("\n📥 Downloading test images...")
    test_images = download_test_images()
    
    if not test_images:
        print("❌ No test images available!")
        return
    
    print(f"\n✅ Got {len(test_images)} test images")
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    pipeline = SegmentationPipeline(config_path='configs/default.yaml')
    print("Pipeline initialized!")
    
    # Output directory
    output_dir = Path(__file__).parent / 'output' / 'test_results'
    
    # Test each image
    results = []
    for img_path in test_images:
        result = test_pipeline_with_image(pipeline, img_path, output_dir)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    
    total_time = sum(r.processing_time_ms for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"  Total images: {len(results)}")
    print(f"  Average processing time: {avg_time:.1f}ms")
    print(f"  Output directory: {output_dir}")
    
    print("\n✅ All tests completed!")


if __name__ == '__main__':
    main()
