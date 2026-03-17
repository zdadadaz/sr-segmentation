"""
BiSeNet face parsing module (PR3)

Outputs hair_mask, face_mask, skin_mask from person regions
Uses pretrained BiSeNet face parsing model
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import cv2


class BiSeNetParser:
    """
    BiSeNet face parsing for hair/face/skin segmentation
    
    Standard face parsing class indices (CelebAMask-HQ):
    0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
    6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth,
    12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    """
    
    # Hair class indices
    HAIR_CLASSES = {17}  # hair
    
    # Face classes (to exclude from SR)
    FACE_CLASSES = {2, 3, 4, 5, 6, 10, 11, 12, 13}  # brows, eyes, nose, mouth, lips
    
    # Skin classes (to exclude from SR)
    SKIN_CLASSES = {1, 7, 8, 14}  # skin, ears, neck
    
    # All person classes
    PERSON_CLASSES = HAIR_CLASSES | FACE_CLASSES | SKIN_CLASSES
    
    def __init__(
        self,
        model_path: str = None,
        num_classes: int = 19,
        device: str = 'cuda',
        config: dict = None
    ):
        self.device = device
        self.num_classes = num_classes
        self.config = config or {}
        self.model_path = model_path
        self.model = None
        self.input_size = 512
        
        self._load_model()
    
    def _load_model(self):
        """Load BiSeNet model"""
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"BiSeNet model not found: '{self.model_path}'. "
                "Download face_parsing.pth and set models.bisenet in configs/default.yaml."
            )
        from src.model import BiSeNet
        self.model = BiSeNet(self.num_classes, 'resnet18')
        state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state_dict)
        device = self.device if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()

    def _build_bisenet(self) -> nn.Module:
        from src.model import BiSeNet
        return BiSeNet(self.num_classes, 'resnet18')
    
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
        h, w = image.shape[:2]
        
        # Crop if bbox provided
        if crop_box is not None:
            x1, y1, x2, y2 = map(int, crop_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = image[y1:y2, x1:x2]
        else:
            cropped = image
            x1, y1 = 0, 0
        
        prediction = self._run_inference(cropped)
        masks = self._prediction_to_masks(prediction, cropped.shape[:2])
        
        # Map back to original image coordinates if cropped
        if crop_box is not None:
            full_masks = {
                'hair': np.zeros((h, w), dtype=np.uint8),
                'face': np.zeros((h, w), dtype=np.uint8),
                'skin': np.zeros((h, w), dtype=np.uint8),
            }
            for key in masks:
                full_masks[key][y1:y2, x1:x2] = masks[key]
            return full_masks
        
        return masks
    
    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run model inference
        
        Args:
            image: RGB image
            
        Returns:
            Class prediction map (H, W)
        """
        # Preprocess
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # To tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        
        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]
            
            prediction = output.argmax(dim=1).squeeze().cpu().numpy()
        
        # Resize back to original size
        oh, ow = image.shape[:2]
        prediction = cv2.resize(prediction.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
        
        return prediction
    
    def _prediction_to_masks(
        self,
        prediction: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Convert model prediction to separate masks"""
        h, w = original_size
        
        if prediction.shape[:2] != (h, w):
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
        
        masks = {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'skin': np.zeros((h, w), dtype=np.uint8),
        }
        
        for class_idx in self.HAIR_CLASSES:
            masks['hair'][prediction == class_idx] = 1
        
        for class_idx in self.FACE_CLASSES:
            masks['face'][prediction == class_idx] = 1
        
        for class_idx in self.SKIN_CLASSES:
            masks['skin'][prediction == class_idx] = 1
        
        return masks
    
    def parse_hair_only(
        self,
        image: np.ndarray,
        crop_box: Tuple[int, int, int, int] = None
    ) -> np.ndarray:
        """Get only hair mask"""
        result = self.parse(image, crop_box)
        return result['hair']





class PersonDetector:
    """
    Person detector for finding humans in images
    Uses YOLOv8 or OpenCV's Haar cascade as fallback
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        config: dict = None
    ):
        self.device = device
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.model = None
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str = None):
        """Load person detection model"""
        from ultralytics import YOLO
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('models/yolov8n.pt')

    def detect(
        self,
        image: np.ndarray
    ) -> List[Tuple[List[float], float]]:
        """
        Detect persons in image

        Args:
            image: RGB image (H, W, 3)

        Returns:
            List of (bbox, confidence) where bbox = [x1, y1, x2, y2]
        """
        return self._detect_yolo(image)

    def _detect_yolo(self, image: np.ndarray) -> List[Tuple[List[float], float]]:
        """Detect using YOLOv8"""
        results = self.model(image, verbose=False, conf=self.confidence_threshold, classes=[0])  # class 0 = person
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item())
                detections.append(([float(x1), float(y1), float(x2), float(y2)], conf))
        
        return detections
    
    def detect_and_crop(
        self,
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, List[float], float]]:
        """Detect persons and return cropped images"""
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
    """Factory function to create BiSeNet parser"""
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
    """Factory function to create person detector"""
    return PersonDetector(
        model_path=model_path,
        device=device,
        config=config
    )
