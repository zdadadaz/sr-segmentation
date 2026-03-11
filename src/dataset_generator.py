"""
Dataset auto-labeling pipeline (PR5)
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


class DatasetGenerator:
    """
    Generate segmentation dataset using the full pipeline
    
    Workflow:
    1. Load images from source directory
    2. Run segmentation pipeline (PR2-4)
    3. Save images and masks in standard format
    4. Generate train/val split
    5. Provide statistics and quality metrics
    """
    
    # Class definitions for output dataset
    CLASSES = ['background', 'hair_fur', 'face', 'skin', 'vegetation']
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    
    def __init__(
        self,
        pipeline,
        output_dir: str = 'dataset',
        images_subdir: str = 'images',
        masks_subdir: str = 'masks'
    ):
        """
        Initialize dataset generator
        
        Args:
            pipeline: SegmentationPipeline instance
            output_dir: Base output directory
            images_subdir: Subdirectory for images
            masks_subdir: Subdirectory for masks
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / images_subdir
        self.masks_dir = self.output_dir / masks_subdir
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_hair': 0,
            'images_with_animal': 0,
            'images_with_human': 0,
            'hair_coverage': [],  # Percentage of image covered by hair
            'class_distribution': {c: 0 for c in self.CLASSES},
        }
    
    def process_directory(
        self,
        source_dir: str,
        extensions: List[str] = None,
        recursive: bool = True,
        max_images: int = None
    ) -> List[str]:
        """
        Process all images in a directory
        
        Args:
            source_dir: Source directory containing images
            extensions: List of file extensions to process
            recursive: Whether to search recursively
            max_images: Maximum number of images to process
            
        Returns:
            List of processed image paths
        """
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.webp']
        source_dir = Path(source_dir)
        
        # Find all images
        image_paths = []
        pattern = '**/*' if recursive else '*'
        
        for ext in extensions:
            image_paths.extend(source_dir.glob(f'{pattern}{ext}'))
            image_paths.extend(source_dir.glob(f'{pattern}{ext.upper()}'))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        # Process each image
        processed = []
        
        for img_path in tqdm(image_paths, desc="Generating dataset"):
            try:
                result = self.process_image(img_path)
                if result:
                    processed.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return processed
    
    def process_image(
        self,
        image_path: str
    ) -> Optional[str]:
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Output image path or None if failed
        """
        image_path = Path(image_path)
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        # Run segmentation
        result = self.pipeline.segment(image_np)
        
        # Generate output filename
        output_name = image_path.stem + '.png'
        output_image_path = self.images_dir / output_name
        output_mask_path = self.masks_dir / output_name
        
        # Save image
        Image.fromarray(image_np).save(output_image_path)
        
        # Create and save mask
        mask = self._create_output_mask(result, image_np.shape[:2])
        mask_img = Image.fromarray(mask)
        mask_img.save(output_mask_path)
        
        # Update statistics
        self._update_stats(result, mask)
        
        return str(output_image_path)
    
    def _create_output_mask(
        self,
        result,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create output mask in standard format
        
        Args:
            result: SegmentationResult
            original_shape: (H, W) of original image
            
        Returns:
            Class index mask (H, W)
        """
        h, w = original_shape
        
        # Initialize with background (0)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add vegetation (placeholder - would need vegetation detection)
        
        # Add skin (class 3)
        if result.skin_mask is not None:
            skin = cv2.resize(result.skin_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask[skin > 0] = self.CLASS_TO_IDX['skin']
        
        # Add face (class 2)
        if result.face_mask is not None:
            face = cv2.resize(result.face_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask[face > 0] = self.CLASS_TO_IDX['face']
        
        # Add hair/fur (class 1)
        final_mask = result.final_mask
        if final_mask is not None:
            final = cv2.resize(final_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask[final > 0] = self.CLASS_TO_IDX['hair_fur']
        
        return mask
    
    def _update_stats(self, result, mask: np.ndarray):
        """Update running statistics"""
        self.stats['total_images'] += 1
        
        # Count hair/fur pixels
        hair_pixels = np.sum(mask == self.CLASS_TO_IDX['hair_fur'])
        total_pixels = mask.size
        hair_coverage = hair_pixels / total_pixels
        
        self.stats['hair_coverage'].append(hair_coverage)
        
        if hair_coverage > 0:
            self.stats['images_with_hair'] += 1
        
        if result.animal_mask is not None and result.animal_mask.any():
            self.stats['images_with_animal'] += 1
        
        if result.human_hair_mask is not None and result.human_hair_mask.any():
            self.stats['images_with_human'] += 1
        
        # Class distribution
        for i, class_name in enumerate(self.CLASSES):
            self.stats['class_distribution'][class_name] += np.sum(mask == i)
    
    def generate_split(
        self,
        train_ratio: float = 0.8,
        output_file: str = None
    ) -> Dict[str, List[str]]:
        """
        Generate train/val split files
        
        Args:
            train_ratio: Ratio of training samples
            output_file: Optional path to save split JSON
            
        Returns:
            Dictionary with 'train' and 'val' lists
        """
        # Get all image files
        image_files = sorted([f.stem for f in self.images_dir.glob('*.png')])
        
        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(image_files))
        
        # Split
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_files = [image_files[i] for i in train_indices]
        val_files = [image_files[i] for i in val_indices]
        
        split = {
            'train': train_files,
            'val': val_files,
        }
        
        # Save if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(split, f, indent=2)
        
        return split
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        
        # Compute derived statistics
        if stats['hair_coverage']:
            stats['avg_hair_coverage'] = np.mean(stats['hair_coverage'])
            stats['median_hair_coverage'] = np.median(stats['hair_coverage'])
        
        # Percentage of images with each type
        if stats['total_images'] > 0:
            stats['pct_with_hair'] = stats['images_with_hair'] / stats['total_images']
            stats['pct_with_animal'] = stats['images_with_animal'] / stats['total_images']
            stats['pct_with_human'] = stats['images_with_human'] / stats['total_images']
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()
        
        print("=" * 50)
        print("Dataset Generation Statistics")
        print("=" * 50)
        print(f"Total images: {stats['total_images']}")
        print(f"Images with hair/fur: {stats['images_with_hair']} ({stats.get('pct_with_hair', 0)*100:.1f}%)")
        print(f"Images with animals: {stats['images_with_animal']} ({stats.get('pct_with_animal', 0)*100:.1f}%)")
        print(f"Images with humans: {stats['images_with_human']} ({stats.get('pct_with_human', 0)*100:.1f}%)")
        
        if 'avg_hair_coverage' in stats:
            print(f"\nHair coverage:")
            print(f"  Average: {stats['avg_hair_coverage']*100:.2f}%")
            print(f"  Median: {stats['median_hair_coverage']*100:.2f}%")
        
        print("\nClass distribution:")
        total_pixels = sum(stats['class_distribution'].values())
        for class_name, count in stats['class_distribution'].items():
            pct = count / total_pixels * 100 if total_pixels > 0 else 0
            print(f"  {class_name}: {pct:.2f}%")
        
        print("=" * 50)


class QualityChecker:
    """
    Check quality of generated dataset
    
    Provides tools for manual review and acceptance/rejection
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize quality checker
        
        Args:
            dataset_dir: Dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images'
        self.masks_dir = self.dataset_dir / 'masks'
        
        # Load existing quality data if available
        self.quality_file = self.dataset_dir / 'quality.json'
        self.quality_data = self._load_quality_data()
    
    def _load_quality_data(self) -> Dict:
        """Load existing quality data"""
        if self.quality_file.exists():
            with open(self.quality_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_quality_data(self):
        """Save quality data"""
        with open(self.quality_file, 'w') as f:
            json.dump(self.quality_data, f, indent=2)
    
    def check_image(self, image_name: str) -> Dict:
        """
        Check a single image's quality
        
        Args:
            image_name: Image filename (without extension)
            
        Returns:
            Quality check results
        """
        image_path = self.images_dir / f"{image_name}.png"
        mask_path = self.masks_dir / f"{image_name}.png"
        
        if not image_path.exists() or not mask_path.exists():
            return {'error': 'File not found'}
        
        # Load image and mask
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        
        # Check mask validity
        unique_classes = np.unique(mask)
        
        # Check for common issues
        issues = []
        
        # Issue: empty mask
        if len(unique_classes) == 1 and unique_classes[0] == 0:
            issues.append('empty_mask')
        
        # Issue: mask too small
        hair_pixels = np.sum(mask > 0)
        if hair_pixels < 100:
            issues.append('very_small_mask')
        
        # Compute coverage
        hair_coverage = hair_pixels / mask.size
        
        return {
            'image_name': image_name,
            'image_shape': image.shape,
            'mask_shape': mask.shape,
            'unique_classes': unique_classes.tolist(),
            'hair_coverage': hair_coverage,
            'issues': issues,
            'status': 'reviewed' if image_name in self.quality_data else 'pending',
        }
    
    def mark_accepted(self, image_name: str):
        """Mark image as accepted"""
        self.quality_data[image_name] = 'accepted'
        self._save_quality_data()
    
    def mark_rejected(self, image_name: str):
        """Mark image as rejected"""
        self.quality_data[image_name] = 'rejected'
        self._save_quality_data()
    
    def get_pending_images(self) -> List[str]:
        """Get list of pending images"""
        all_images = [f.stem for f in self.images_dir.glob('*.png')]
        return [img for img in all_images if img not in self.quality_data]
    
    def get_accepted_count(self) -> int:
        """Get count of accepted images"""
        return sum(1 for v in self.quality_data.values() if v == 'accepted')
    
    def get_rejected_count(self) -> int:
        """Get count of rejected images"""
        return sum(1 for v in self.quality_data.values() if v == 'rejected')
