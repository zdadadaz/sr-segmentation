"""
SR Segmentation Pipeline
Main entry point for the hair/fur segmentation system
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path


@dataclass
class BBox:
    """Bounding box in XYXY format"""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float = 1.0
    
    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        x = self.x1
        y = self.y1
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return (x, y, w, h)
    
    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy)


@dataclass
class SegmentationResult:
    """
    Unified segmentation result data structure
    """
    # Original image shape
    original_shape: Tuple[int, int]  # (H, W)
    
    # Binary mask for hair/fur regions (0 = non-hair, 1 = hair/fur)
    hair_fur_mask: Optional[np.ndarray] = None
    
    # Detailed masks (for debugging/analysis)
    animal_mask: Optional[np.ndarray] = None      # Fur from animals
    human_hair_mask: Optional[np.ndarray] = None   # Hair from humans
    face_mask: Optional[np.ndarray] = None         # Face regions (to exclude)
    skin_mask: Optional[np.ndarray] = None         # Skin regions (to exclude)
    vegetation_mask: Optional[np.ndarray] = None  # Vegetation (to exclude)
    
    # Bounding boxes
    animal_bboxes: List[BBox] = None              # Detected animals
    person_bboxes: List[BBox] = None               # Detected persons
    
    # Metadata
    model_versions: dict = None                   # Which models were used
    processing_time_ms: float = 0.0
    
    # Confidence scores
    hair_confidence: float = 1.0
    
    def __post_init__(self):
        if self.animal_bboxes is None:
            self.animal_bboxes = []
        if self.person_bboxes is None:
            self.person_bboxes = []
        if self.model_versions is None:
            self.model_versions = {}
    
    @property
    def final_mask(self) -> np.ndarray:
        """
        Get the final merged mask for SR processing
        hair_fur_mask = (animal_mask OR human_hair_mask) AND NOT (face_mask OR skin_mask)
        """
        h, w = self.original_shape
        
        # Start with hair/fur mask if available
        if self.hair_fur_mask is not None:
            return self.hair_fur_mask
        
        # Otherwise merge from detailed masks
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add animal fur
        if self.animal_mask is not None:
            mask = np.logical_or(mask, self.animal_mask).astype(np.uint8)
        
        # Add human hair
        if self.human_hair_mask is not None:
            mask = np.logical_or(mask, self.human_hair_mask).astype(np.uint8)
        
        # Exclude face and skin regions
        exclude_mask = np.zeros((h, w), dtype=np.uint8)
        if self.face_mask is not None:
            exclude_mask = np.logical_or(exclude_mask, self.face_mask).astype(np.uint8)
        if self.skin_mask is not None:
            exclude_mask = np.logical_or(exclude_mask, self.skin_mask).astype(np.uint8)
        
        mask = mask * (1 - exclude_mask)
        return mask
    
    def get_soft_mask(self, sigma: float = 3.0) -> np.ndarray:
        """
        Get soft mask with Gaussian blur on edges for smooth SR blending
        """
        from scipy.ndimage import gaussian_filter
        
        binary_mask = self.final_mask
        soft_mask = gaussian_filter(binary_mask.astype(float), sigma=sigma)
        return soft_mask
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'original_shape': self.original_shape,
            'has_hair_fur_mask': self.hair_fur_mask is not None,
            'animal_bboxes': [b.__dict__ for b in self.animal_bboxes],
            'person_bboxes': [b.__dict__ for b in self.person_bboxes],
            'model_versions': self.model_versions,
            'processing_time_ms': self.processing_time_ms,
            'hair_confidence': self.hair_confidence,
        }


class SegmentationPipeline:
    """
    Main pipeline for hair/fur segmentation
    
    Pipeline flow:
    1. SpeciesNet → detect animals → animal bbox
    2. SAM (bbox prompt) → animal pixel mask → animal_mask
    3. Person detection → BiSeNet → hair_mask + face_mask + skin_mask
    4. Merge: final_mask = (animal_mask | hair_mask) & ~(face_mask | skin_mask)
    """
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        import yaml
        from utils.config_parser import load_config
        
        self.config = load_config(config_path)
        
        # Model placeholders (will be loaded lazily)
        self.speciesnet_model = None
        self.sam_model = None
        self.bisenet_model = None
        self.person_detector = None
        
        # Device
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Run full segmentation pipeline
        
        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image
            
        Returns:
            SegmentationResult with masks and metadata
        """
        import time
        from utils.image_utils import preprocess_image
        
        start_time = time.time()
        
        # Convert to numpy if needed
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        h, w = image.shape[:2]
        
        # Initialize result
        result = SegmentationResult(
            original_shape=(h, w),
            model_versions={},
        )
        
        # TODO: Implement PR2 (SpeciesNet + SAM)
        # TODO: Implement PR3 (BiSeNet face parsing)
        # TODO: Implement PR4 (Mask merging)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def segment_animals_only(self, image: np.ndarray) -> SegmentationResult:
        """Run only animal detection and masking (PR2)"""
        # TODO: Implement in PR2
        pass
    
    def segment_humans_only(self, image: np.ndarray) -> SegmentationResult:
        """Run only human hair segmentation (PR3)"""
        # TODO: Implement in PR3
        pass
    
    def merge_masks(self, result: SegmentationResult) -> SegmentationResult:
        """Merge animal and human masks (PR4)"""
        # TODO: Implement in PR4
        pass


# Import torch at module level for device checking
import torch
