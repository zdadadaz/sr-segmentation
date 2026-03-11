"""
SpeciesNet animal detection module (PR2)

Uses YOLOv8 with animal class filtering as SpeciesNet alternative
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from pathlib import Path
import cv2


class SpeciesNetDetector:
    """
    SpeciesNet for detecting animals in images
    
    Filters for furry animals only (excludes birds, reptiles, fish, insects)
    Uses YOLOv8 as the underlying detector with class filtering
    """
    
    # COCO class indices for animals
    # 15: cat, 16: dog, 17: horse, 18: cow, 19: sheep, 20: bird
    # 21: rabbit, 22: hamster (not in COCO), 23: squirrel (not in COCO)
    
    COCO_ANIMAL_CLASSES = {
        15: 'cat',
        16: 'dog', 
        17: 'horse',
        18: 'cow',
        19: 'sheep',
        20: 'bird',
        21: 'rabbit',
        # Extended classes that may be present in models
        22: 'hamster',
        23: 'squirrel',
    }
    
    # Furry animals - exclude birds
    FURRY_CLASSES = {
        'cat', 'dog', 'horse', 'cow', 'sheep', 'rabbit',
        'bear', 'fox', 'wolf', 'deer', 'squirrel', 'mouse',
        'rat', 'hamster', 'gerbil', 'ferret', 'weasel', 'otter',
        'badger', 'hedgehog', 'llama', 'alpaca', 'goat', 'pig'
    }
    
    # Bird classes to exclude
    BIRD_CLASSES = {'bird', 'bird'}
    
    def __init__(
        self,
        model_path: str = None,
        config: dict = None,
        device: str = 'cuda'
    ):
        """
        Initialize SpeciesNet detector
        
        Args:
            model_path: Path to model weights (YOLOv8)
            config: Configuration dictionary
            device: Device to run on
        """
        self.device = device
        self.config = config or {}
        
        # Get furry classes from config or use default
        self.furry_classes = set(
            self.config.get('furry_classes', self.FURRY_CLASSES)
        )
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        self.model = None
        self.model_path = model_path
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained yolov8n (nano) model
                self.model = YOLO('yolov8n.pt')
            
            # Move to device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                
        except ImportError:
            print("Warning: ultralytics not installed. Using fallback detection.")
            self.model = None
    
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
        if self.model is None:
            return self._fallback_detect(image)
        
        # Run inference
        results = self.model(image, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Get class name
                class_name = self.COCO_ANIMAL_CLASSES.get(class_id, f'class_{class_id}')
                
                # Filter for furry animals if requested
                if filter_furry and class_name.lower() not in self.furry_classes:
                    continue
                
                detections.append(([float(x1), float(y1), float(x2), float(y2)], class_name, confidence))
        
        return detections
    
    def _fallback_detect(
        self,
        image: np.ndarray
    ) -> List[Tuple[List[float], str, float]]:
        """
        Fallback detection using basic image processing
        Used when model is not available
        """
        # Simple edge-based detection as placeholder
        # This is just for testing the pipeline
        
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple skin/fur color detection in HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect brownish/orange regions (common fur colors)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum area threshold
                x, y, cw, ch = cv2.boundingRect(contour)
                detections.append((
                    [float(x), float(y), float(x + cw), float(y + ch)],
                    'animal',
                    0.5
                ))
        
        return detections
    
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
    
    def filter_exclude_birds(
        self,
        detections: List[Tuple[List[float], str, float]]
    ) -> List[Tuple[List[float], str, float]]:
        """
        Exclude bird detections
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered detections without birds
        """
        return [
            (bbox, class_name, conf)
            for bbox, class_name, conf in detections
            if class_name.lower() not in self.BIRD_CLASSES
        ]
    
    def get_animal_count(self, detections: List[Tuple]) -> int:
        """Get count of animal detections"""
        return len(detections)
    
    def has_furry_animals(self, detections: List[Tuple]) -> bool:
        """Check if any furry animals detected"""
        return len(self.filter_furry_animals(detections)) > 0


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
