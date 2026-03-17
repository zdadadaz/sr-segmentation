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
    texture_mask: Optional[np.ndarray] = None
    
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
        if self.texture_mask is not None:
            mask = np.logical_or(mask, self.texture_mask).astype(np.uint8)
        
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
        self._texture_classifier = None
    
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
        """Lazy-load face parser (BiSeNet or FaceXFormer depending on config)"""
        if self._bisenet is None:
            parser_name = self.config.get('face_parser', 'bisenet')
            if parser_name == 'facexformer':
                from src.facexformer_parser import create_facexformer_parser
                self._bisenet = create_facexformer_parser(
                    model_path=self.config.get('models', {}).get('facexformer'),
                    device=self.device,
                    config={}
                )
            else:
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
    
    @property
    def texture_classifier(self):
        """Lazy-load texture classifier (PR7)"""
        if self._texture_classifier is None:
            from src.texture_classifier import create_texture_classifier
            self._texture_classifier = create_texture_classifier(
                config=self.config.get('texture_classifier', {})
            )
        return self._texture_classifier
    
    @staticmethod
    def _bboxes_heavily_overlap(bboxes: list, iou_threshold: float = 0.5) -> bool:
        """Return True if any pair of bboxes has IoU > iou_threshold."""
        if len(bboxes) < 2:
            return False
        for i in range(len(bboxes)):
            x1i, y1i, x2i, y2i = bboxes[i]
            ai = max(0, x2i - x1i) * max(0, y2i - y1i)
            for j in range(i + 1, len(bboxes)):
                x1j, y1j, x2j, y2j = bboxes[j]
                aj = max(0, x2j - x1j) * max(0, y2j - y1j)
                ix = max(0, min(x2i, x2j) - max(x1i, x1j))
                iy = max(0, min(y2i, y2j) - max(y1i, y1j))
                inter = ix * iy
                union = ai + aj - inter
                if union > 0 and inter / union > iou_threshold:
                    return True
        return False

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
        
        # Parse hair/face/skin for each detected person.
        # When multiple person bboxes heavily overlap (IoU > 0.5) the per-crop
        # approach degrades — run BiSeNet on the full image instead so all
        # people are parsed in one pass.
        combined_hair = np.zeros((h, w), dtype=np.uint8)
        combined_face = np.zeros((h, w), dtype=np.uint8)
        combined_skin = np.zeros((h, w), dtype=np.uint8)

        if person_detections and self._bboxes_heavily_overlap(
            [b for b, _ in person_detections]
        ):
            face_result = self.bisenet.parse(image, crop_box=None)
            combined_hair = face_result['hair']
            combined_face = face_result['face']
            combined_skin = face_result['skin']
        else:
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
        
        # Step 5: Texture fallback (PR7) on remaining areas
        exclude_mask = merged['exclude_mask']
        # Also exclude regions already designated as animal or human hair
        exclude_mask = np.logical_or(exclude_mask, merged['combined_hair']).astype(np.uint8)
        
        texture_mask = self.texture_classifier.classify_texture(image, exclude_mask=exclude_mask)
        result.texture_mask = texture_mask
        
        # Merge again with texture
        merged_with_texture = self.mask_merger.merge(
            animal_mask=result.animal_mask,
            human_hair_mask=result.human_hair_mask,
            face_mask=result.face_mask,
            skin_mask=result.skin_mask,
            texture_mask=result.texture_mask,
            original_size=(h, w)
        )
        result.hair_fur_mask = merged_with_texture['final_mask']
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        parser_name = self.config.get('face_parser', 'bisenet')
        result.model_versions = {
            'speciesnet': 'speciesnet_md_v5a',
            'sam': self.sam.model_type if self.sam.sam else 'fallback_grabcut',
            'face_parser': parser_name,
            'texture': 'gabor_filter_bank',
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
