"""
SpeciesNet animal detection module (PR2)
"""

import numpy as np
from typing import List, Tuple, Optional
import torch
from pathlib import Path


class SpeciesNetDetector:
    """
    SpeciesNet for detecting animals in images
    
    Filters for furry animals only (excludes birds, reptiles, fish, insects)
    """
    
    # PASCAL VOC/COCO class mapping for furry animals
    FURRY_CLASSES = {
        15: 'cat',  # COCO: cat
        16: 'dog',  # COCO: dog
        17: 'horse',  # COCO: horse
        18: 'cow',  # COCO: cow
        # Additional that might have fur:
        # 19: 'sheep',  # COCO: sheep
        # 20: 'bird',
        # 21: 'dog',
        # 22: 'squirrel',
    }
    
    # Expanded list based on actual model classes
    DEFAULT_FURRY_CLASSES = [
        'cat', 'dog', 'horse', 'cow', 'sheep', 'rabbit',
        'bear', 'fox', 'wolf', 'deer', 'squirrel', 'mouse',
        'rat', 'hamster', 'gerbil', 'ferret', 'weasel', 'otter',
        'badger', 'hedgehog', 'llama', 'alpaca', 'goat', 'pig'
    ]
    
    def __init__(
        self,
        model_path: str = None,
        config: dict = None,
        device: str = 'cuda'
    ):
        """
        Initialize SpeciesNet detector
        
        Args:
            model_path: Path to model weights
            config: Configuration dictionary
            device: Device to run on
        """
        self.device = device
        self.config = config or {}
        
        # Get furry classes from config or use default
        self.furry_classes = set(
            self.config.get('furry_classes', self.DEFAULT_FURRY_CLASSES)
        )
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        self.model = None
        self.model_path = model_path
        
        # TODO: Load actual model in PR2
        # self._load_model()
    
    def _load_model(self):
        """Load the SpeciesNet model"""
        # TODO: Implement actual model loading
        # This will be implemented in PR2
        pass
    
    def detect_animals(
        self,
        image: np.ndarray,
        filter_furry: bool = True
    ) -> List[Tuple[List[float], str, float]]:
        """
        Detect animals in image
        
        Args:
            image: RGB image (H, W, 3)
            filter_furry: If True, only return furry animals
            
        Returns:
            List of (bbox, class_name, confidence) tuples
            bbox is [x1, y1, x2, y2]
        """
        # TODO: Implement actual detection in PR2
        # This is a placeholder that returns empty results
        
        results = []
        
        # Placeholder: return sample detection for testing
        # Will be replaced with actual model inference
        pass
        
        return results
    
    def filter_furry_animals(
        self,
        detections: List[Tuple[List[float], str, float]]
    ) -> List[Tuple[List[float], str, float]]:
        """
        Filter detections to only furry animals
        
        Args:
            detections: List of (bbox, class_name, confidence)
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for bbox, class_name, conf in detections:
            if class_name.lower() in self.furry_classes:
                filtered.append((bbox, class_name, conf))
        
        return filtered


def create_speciesnet_detector(
    model_path: str = None,
    config: dict = None,
    device: str = 'cuda'
) -> SpeciesNetDetector:
    """
    Factory function to create SpeciesNet detector
    
    Args:
        model_path: Path to model weights
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        SpeciesNetDetector instance
    """
    return SpeciesNetDetector(
        model_path=model_path,
        config=config,
        device=device
    )
