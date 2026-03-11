"""
SR Segmentation Pipeline
Main entry point for the hair/fur segmentation system
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path
import time
import torch


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
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class SegmentationResult:
    """Unified segmentation result data structure"""
    original_shape: Tuple[int, int]  # (H, W)
    
    # Binary mask for hair/fur regions
    hair_fur_mask: Optional[np.ndarray] = None
    
    # Detailed masks
    animal_mask: Optional[np.ndarray] = None
    human_hair_mask: Optional[np.ndarray] = None
    face_mask: Optional[np.ndarray] = None
    skin_mask: Optional[np.ndarray] = None
    vegetation_mask: Optional[np.ndarray] = None
    
    # Bounding boxes
    animal_bboxes: List[BBox] = field(default_factory=list)
    person_bboxes: List[BBox] = field(default_factory=list)
    
    # Metadata
    model_versions: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    hair_confidence: float = 1.0
    
    @property
    def final_mask(self) -> np.ndarray:
        """
        Get the final merged mask for SR processing
        hair_fur_mask = (animal_mask | hair_mask) & ~(face_mask | skin_mask)
        """
        if self.hair_fur_mask is not None:
            return self.hair_fur_mask
        
        h, w = self.original_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.animal_mask is not None:
            mask = np.logical_or(mask, self.animal_mask).astype(np.uint8)
        if self.human_hair_mask is not None:
            mask = np.logical_or(mask, self.human_hair_mask).astype(np.uint8)
        
        exclude = np.zeros((h, w), dtype=np.uint8)
        if self.face_mask is not None:
            exclude = np.logical_or(exclude, self.face_mask).astype(np.uint8)
        if self.skin_mask is not None:
            exclude = np.logical_or(exclude, self.skin_mask).astype(np.uint8)
        
        mask = mask * (1 - exclude)
        return mask
    
    def get_soft_mask(self, sigma: float = 3.0) -> np.ndarray:
        """Get soft mask with Gaussian blur for smooth SR blending"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(self.final_mask.astype(float), sigma=sigma)
    
    def to_dict(self) -> dict:
        return {
            'original_shape': self.original_shape,
            'has_hair_fur_mask': self.hair_fur_mask is not None,
            'num_animals': len(self.animal_bboxes),
            'num_persons': len(self.person_bboxes),
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
        from utils.config_parser import load_config
        
        self.config = load_config(config_path)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy-loaded models
        self._speciesnet = None
        self._sam = None
        self._bisenet = None
        self._person_detector = None
        self._mask_merger = None
    
    @property
    def speciesnet(self):
        """Lazy-load SpeciesNet detector"""
        if self._speciesnet is None:
            from src.speciesnet import create_speciesnet_detector
            self._speciesnet = create_speciesnet_detector(
                model_path=self.config.get('models', {}).get('speciesnet'),
                config=self.config.get('speciesnet', {}),
                device=self.device
            )
        return self._speciesnet
    
    @property
    def sam(self):
        """Lazy-load SAM"""
        if self._sam is None:
            from src.sam import create_sam_generator
            self._sam = create_sam_generator(
                model_type=self.config.get('sam', {}).get('model_type', 'vit_h'),
                checkpoint_path=self.config.get('models', {}).get('sam'),
                device=self.device,
                config=self.config.get('sam', {})
            )
        return self._sam
    
    @property
    def bisenet(self):
        """Lazy-load BiSeNet"""
        if self._bisenet is None:
            from src.bisenet import create_bisenet_parser
            self._bisenet = create_bisenet_parser(
                model_path=self.config.get('models', {}).get('bisenet'),
                device=self.device,
                config=self.config.get('bisenet', {})
            )
        return self._bisenet
    
    @property
    def person_detector(self):
        """Lazy-load person detector"""
        if self._person_detector is None:
            from src.bisenet import create_person_detector
            self._person_detector = create_person_detector(
                model_path=self.config.get('models', {}).get('person_detector'),
                device=self.device,
                config=self.config.get('person_detector', {})
            )
        return self._person_detector
    
    @property
    def mask_merger(self):
        """Lazy-load mask merger"""
        if self._mask_merger is None:
            from src.mask_merger import create_mask_merger
            self._mask_merger = create_mask_merger(config=self.config)
        return self._mask_merger
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Run full segmentation pipeline
        
        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image
            
        Returns:
            SegmentationResult with masks and metadata
        """
        start_time = time.time()
        
        # Convert PIL to numpy
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        h, w = image.shape[:2]
        result = SegmentationResult(original_shape=(h, w))
        
        # Step 1: Detect animals (PR2)
        animal_detections = self.speciesnet.detect_animals(image, filter_furry=True)
        result.animal_bboxes = [
            BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=cls, confidence=conf)
            for b, cls, conf in animal_detections
        ]
        
        # Step 2: Generate animal masks with SAM (PR2)
        if animal_detections:
            animal_masks = self.sam.generate_masks_from_bboxes(image, animal_detections)
            result.animal_mask = self.sam.combine_masks(animal_masks, (h, w))
        
        # Step 3: Detect persons and parse faces (PR3)
        person_detections = self.person_detector.detect(image)
        result.person_bboxes = [
            BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label='person', confidence=conf)
            for b, conf in person_detections
        ]
        
        # Parse hair/face/skin for each detected person
        combined_hair = np.zeros((h, w), dtype=np.uint8)
        combined_face = np.zeros((h, w), dtype=np.uint8)
        combined_skin = np.zeros((h, w), dtype=np.uint8)
        
        for bbox, conf in person_detections:
            face_result = self.bisenet.parse(image, crop_box=bbox)
            combined_hair = np.logical_or(combined_hair, face_result['hair']).astype(np.uint8)
            combined_face = np.logical_or(combined_face, face_result['face']).astype(np.uint8)
            combined_skin = np.logical_or(combined_skin, face_result['skin']).astype(np.uint8)
        
        result.human_hair_mask = combined_hair
        result.face_mask = combined_face
        result.skin_mask = combined_skin
        
        # Step 4: Merge masks (PR4)
        merged = self.mask_merger.merge(
            animal_mask=result.animal_mask,
            human_hair_mask=result.human_hair_mask,
            face_mask=result.face_mask,
            skin_mask=result.skin_mask,
            original_size=(h, w)
        )
        result.hair_fur_mask = merged['final_mask']
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.model_versions = {
            'speciesnet': 'yolov8n',
            'sam': self.sam.model_type if self.sam.sam else 'fallback_grabcut',
            'bisenet': 'bisenet_face_parsing' if self.bisenet.model else 'fallback_color',
        }
        
        return result
    
    def segment_animals_only(self, image: np.ndarray) -> SegmentationResult:
        """Run only animal detection and masking (PR2)"""
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        h, w = image.shape[:2]
        result = SegmentationResult(original_shape=(h, w))
        
        detections = self.speciesnet.detect_animals(image, filter_furry=True)
        result.animal_bboxes = [
            BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=cls, confidence=conf)
            for b, cls, conf in detections
        ]
        
        if detections:
            masks = self.sam.generate_masks_from_bboxes(image, detections)
            result.animal_mask = self.sam.combine_masks(masks, (h, w))
        
        return result
    
    def segment_humans_only(self, image: np.ndarray) -> SegmentationResult:
        """Run only human hair segmentation (PR3)"""
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        h, w = image.shape[:2]
        result = SegmentationResult(original_shape=(h, w))
        
        persons = self.person_detector.detect(image)
        result.person_bboxes = [
            BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label='person', confidence=conf)
            for b, conf in persons
        ]
        
        combined_hair = np.zeros((h, w), dtype=np.uint8)
        combined_face = np.zeros((h, w), dtype=np.uint8)
        combined_skin = np.zeros((h, w), dtype=np.uint8)
        
        for bbox, conf in persons:
            masks = self.bisenet.parse(image, crop_box=bbox)
            combined_hair = np.logical_or(combined_hair, masks['hair']).astype(np.uint8)
            combined_face = np.logical_or(combined_face, masks['face']).astype(np.uint8)
            combined_skin = np.logical_or(combined_skin, masks['skin']).astype(np.uint8)
        
        result.human_hair_mask = combined_hair
        result.face_mask = combined_face
        result.skin_mask = combined_skin
        
        return result
