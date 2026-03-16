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
        try:
            # Try loading from provided path
            if self.model_path and Path(self.model_path).exists():
                self.model = self._build_bisenet()
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.model.load_state_dict(state_dict)
                device = self.device if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
                self.model.to(device)
                self.model.eval()
            else:
                print("Warning: BiSeNet model not found. Using fallback face parsing.")
                self.model = None
        except Exception as e:
            print(f"Warning: Failed to load BiSeNet: {e}. Using fallback.")
            self.model = None
    
    def _build_bisenet(self) -> nn.Module:
        """
        Build BiSeNet model architecture
        """
        try:
            from src.model import BiSeNet
            model = BiSeNet(self.num_classes, 'resnet18')
            return model
        except ImportError:
            print("Warning: Could not import BiSeNet from src.model. Make sure model.py and resnet.py are present.")
            # Fallback will be triggered dynamically
            return None
    
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
        
        if self.model is not None:
            # Run model inference
            prediction = self._run_inference(cropped)
            masks = self._prediction_to_masks(prediction, cropped.shape[:2])
        else:
            # Fallback: use color-based segmentation
            masks = self._fallback_parse(cropped)
        
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
    
    def _fallback_parse(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fallback face parsing using color-based segmentation
        """
        h, w = image.shape[:2]
        
        masks = {
            'hair': np.zeros((h, w), dtype=np.uint8),
            'face': np.zeros((h, w), dtype=np.uint8),
            'skin': np.zeros((h, w), dtype=np.uint8),
        }
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Skin detection in YCrCb space
        lower_skin = np.array([0, 133, 77])
        upper_skin = np.array([255, 173, 127])
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Clean up skin mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        masks['skin'] = (skin_mask > 0).astype(np.uint8)
        
        # Hair detection: dark regions above skin
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Hair is dark and above the skin center
        if skin_mask.any():
            skin_center_y = np.mean(np.where(skin_mask > 0)[0])
            dark_mask[int(skin_center_y):, :] = 0  # Remove dark below skin center
        
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        masks['hair'] = (dark_mask > 0).astype(np.uint8)
        
        # Face is skin in the upper portion
        face_mask = skin_mask.copy()
        masks['face'] = (face_mask > 0).astype(np.uint8)
        
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
        try:
            from ultralytics import YOLO
            
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                self.model = YOLO('models/yolov8n.pt')
            
        except ImportError:
            print("Warning: ultralytics not installed. Using OpenCV cascade for person detection.")
            self.model = None
    
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
        if self.model is not None:
            return self._detect_yolo(image)
        else:
            return self._detect_cascade(image)
    
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
    
    def _detect_cascade(self, image: np.ndarray) -> List[Tuple[List[float], float]]:
        """Fallback: detect using OpenCV Haar cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        detections = []
        h, w = image.shape[:2]
        
        for (x, y, fw, fh) in faces:
            # Expand to approximate person bbox (face → upper body)
            margin_x = int(fw * 0.3)
            margin_y_top = int(fh * 0.5)
            margin_y_bottom = int(fh * 2.0)
            
            px1 = max(0, x - margin_x)
            py1 = max(0, y - margin_y_top)
            px2 = min(w, x + fw + margin_x)
            py2 = min(h, y + fh + margin_y_bottom)
            
            detections.append(([float(px1), float(py1), float(px2), float(py2)], 0.7))
        
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
    def parse(
        self, 
        image: np.ndarray, 
        person_boxes: List['BBox'] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parse face features from image.
        
        Args:
            image: RGB image structure (H,W,3)
            person_boxes: List of bounding boxes for people. If none provided,
                         will attempt detection if detector configured.
                         
        Returns:
            Dictionary mapping feature names ('hair', 'face', 'skin') to boolean masks
        """
        if self.model is None:
            # Fallback to color-based skin/hair heuristics if no model available
            return self._fallback_parse(image, person_boxes)
            
        # Determine ROI if person boxes provided
        # In a generic situation, we would process each person box.
        # For simplicity, we process the whole image or the largest box crop.
        # This implementation processes the whole resized image.
        
        img_tensor = self._preprocess(image)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            try:
                # Actual face parsing network typically outputs a list of scales;
                # often index 0 is the primary output shape [1, num_classes, H, W]
                out = self.model(img_tensor)
                
                # Check output format and extract primary output
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                    
                # Get predicted class indices
                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                
            except Exception as e:
                print(f"Error during BiSeNet inference: {e}")
                return self._fallback_parse(image, person_boxes)
                
        # Resize mask back to original image size
        parsing_full = cv2.resize(
            parsing.astype(np.uint8), 
            (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Extract specific masks based on standard FaceParsing ID mapping
        # 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g, 7: l_ear, 8: r_ear, 
        # 9: ear_r, 10: nose, 11: mouth, 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 
        # 17: hair, 18: hat
        
        skin_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        face_ids = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
        
        # Merge parsing map into binary masks
        skin_mask = np.isin(parsing_full, skin_ids).astype(np.uint8)
        face_mask = np.isin(parsing_full, face_ids).astype(np.uint8)
        hair_mask = (parsing_full == 17).astype(np.uint8)
        
        # Mask off areas where no person is detected to reduce false positives
        if person_boxes:
            person_mask = np.zeros_like(hair_mask)
            for box in person_boxes:
                person_mask[box.ymin:box.ymax, box.xmin:box.xmax] = 1
            hair_mask = hair_mask * person_mask
            skin_mask = skin_mask * person_mask
            face_mask = face_mask * person_mask
            
        return {
            'hair': hair_mask,
            'face': face_mask,
            'skin': skin_mask
        }
        
    def _fallback_parse(
        self, 
        image: np.ndarray, 
        person_boxes: List['BBox'] = None
    ) -> Dict[str, np.ndarray]:
        """
        Very simple fallback method relying on color heuritsics and basic assumptions
        used when actual BiSeNet is not available.
        """
        h, w = image.shape[:2]
        
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        result = {
            'hair': empty_mask.copy(),
            'face': empty_mask.copy(),
            'skin': empty_mask.copy()
        }
        
        if person_boxes:
            for box in person_boxes:
                h_box = box.ymax - box.ymin
                head_ymin = box.ymin
                head_ymax = box.ymin + int(h_box * 0.25)
                face_ymin = box.ymin + int(h_box * 0.1)
                face_ymax = box.ymin + int(h_box * 0.4)
                
                result['hair'][head_ymin:head_ymax, box.xmin:box.xmax] = 1
                result['face'][face_ymin:face_ymax, box.xmin:box.xmax] = 1
                result['skin'][face_ymin:face_ymax, box.xmin:box.xmax] = 1
                
        return result

    def get_hair_mask(self, image: np.ndarray, person_boxes: List['BBox'] = None) -> np.ndarray:
        """Utility method to get just the hair mask"""
        result = self.parse(image, person_boxes)
        return result['hair']
