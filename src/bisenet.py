"""
BiSeNet face parsing module (PR3)
"""

import numpy as np
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from pathlib import Path


class BiSeNetParser:
    """
    BiSeNet face parsing for hair/face/skin segmentation
    
    Outputs:
    - hair_mask: Hair regions
    - face_mask: Face regions (to exclude)
    - skin_mask: Skin regions (to exclude)
    """
    
    # Standard face parsing class indices (from CelebAMask-HQ)
    CLASS_MAP = {
        0: 'background',
        1: 'skin',
        2: 'l_brow',      # left eyebrow
        3: 'r_brow',      # right eyebrow
        4: 'l_eye',       # left eye
        5: 'r_eye',       # right eye
        6: 'nose',
        7: 'mouth',
        8: 'u_lip',       # upper lip
        9: 'l_lip',       # lower lip
        10: 'hair',
    }
    
    # Simplified mapping for our use case
    HAIR_CLASSES = {10}  # hair class
    FACE_CLASSES = {6}   # nose
    SKIN_CLASSES = {1, 2, 3, 4, 5, 7, 8, 9}  # skin + facial features
    
    def __init__(
        self,
        model_path: str = None,
        num_classes: int = 19,
        device: str = 'cuda',
        config: dict = None
    ):
        """
        Initialize BiSeNet parser
        
        Args:
            model_path: Path to model weights
            num_classes: Number of segmentation classes
            device: Device to run on
            config: Configuration dictionary
        """
        self.device = device
        self.num_classes = num_classes
        self.config = config or {}
        
        self.model = None
        
        # TODO: Load model in PR3
        # self._load_model()
    
    def _load_model(self):
        """Load BiSeNet model"""
        # TODO: Implement actual model loading
        # from models.bisenet import BiSeNet
        # self.model = BiSeNet(n_classes=self.num_classes, backbone='resnet18')
        # self.model.load_state_dict(torch.load(self.model_path))
        # self.model.to(self.device)
        # self.model.eval()
        pass
    
    def parse(
        self,
        image: np.ndarray,
        crop_box: Tuple[int, int, int, int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parse face regions from image
        
        Args:
            image: RGB image (H, W, 3)
            crop_box: Optional (x1, y1, x2, y2) to crop to face region
            
        Returns:
            Dictionary with 'hair', 'face', 'skin' masks
        """
        # TODO: Implement actual parsing in PR3
        
        h, w = image.shape[:2]
        
        # Placeholder: return empty masks
        return {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'skin': np.zeros((h, w), dtype=np.uint8),
        }
    
    def parse_hair_only(
        self,
        image: np.ndarray,
        crop_box: Tuple[int, int, int, int] = None
    ) -> np.ndarray:
        """
        Get only hair mask
        
        Args:
            image: RGB image
            crop_box: Optional crop box
            
        Returns:
            Binary hair mask
        """
        result = self.parse(image, crop_box)
        return result['hair']
    
    def _prediction_to_masks(
        self,
        prediction: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Convert model prediction to separate masks
        
        Args:
            prediction: Model output (H, W) with class indices
            original_size: (H, W) of original image
            
        Returns:
            Dictionary of masks
        """
        h, w = original_size
        
        # Resize if needed
        if prediction.shape[:2] != (h, w):
            import cv2
            prediction = cv2.resize(
                prediction, (w, h), interpolation=cv2.INTER_NEAREST
            )
        
        # Create masks
        masks = {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'skin': np.zeros((h, w), dtype=np.uint8),
        }
        
        for class_idx, class_name in self.CLASS_MAP.items():
            if class_idx in self.HAIR_CLASSES:
                masks['hair'][prediction == class_idx] = 1
            elif class_idx in self.FACE_CLASSES:
                masks['face'][prediction == class_idx] = 1
            elif class_idx in self.SKIN_CLASSES:
                masks['skin'][prediction == class_idx] = 1
        
        return masks


class PersonDetector:
    """
    Person detector for finding humans in images
    
    Used to determine where to apply BiSeNet face parsing
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        config: dict = None
    ):
        """
        Initialize person detector
        
        Args:
            model_path: Path to model weights
            device: Device to run on
            config: Configuration dictionary
        """
        self.device = device
        self.config = config or {}
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        self.model = None
        
        # TODO: Load model in PR3
        # self._load_model()
    
    def _load_model(self):
        """Load person detection model"""
        # TODO: Implement actual model loading
        # Use YOLOv8 or similar
        pass
    
    def detect(
        self,
        image: np.ndarray
    ) -> list:
        """
        Detect persons in image
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            List of (bbox, confidence) where bbox = [x1, y1, x2, y2]
        """
        # TODO: Implement actual detection in PR3
        # Placeholder: return empty list
        return []
    
    def detect_and_crop(
        self,
        image: np.ndarray
    ) -> list:
        """
        Detect persons and return cropped images
        
        Args:
            image: RGB image
            
        Returns:
            List of (cropped_image, bbox, confidence)
        """
        detections = self.detect(image)
        
        results = []
        for bbox, conf in detections:
            x1, y1, x2, y2 = map(int, bbox)
            crop = image[y1:y2, x1:x2]
            results.append((crop, bbox, conf))
        
        return results


def create_bisenet_parser(
    model_path: str = None,
    num_classes: int = 19,
    device: str = 'cuda',
    config: dict = None
) -> BiSeNetParser:
    """
    Factory function to create BiSeNet parser
    
    Args:
        model_path: Path to model weights
        num_classes: Number of classes
        device: Device to run on
        config: Configuration dictionary
        
    Returns:
        BiSeNetParser instance
    """
    return BiSeNetParser(
        model_path=model_path,
        num_classes=num_classes,
        device=device,
        config=config
    )


def create_person_detector(
    model_path: str = None,
    device: str = 'cuda',
    config: dict = None
) -> PersonDetector:
    """
    Factory function to create person detector
    
    Args:
        model_path: Path to model weights
        device: Device to run on
        config: Configuration dictionary
        
    Returns:
        PersonDetector instance
    """
    return PersonDetector(
        model_path=model_path,
        device=device,
        config=config
    )
