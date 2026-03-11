"""
SAM (Segment Anything Model) for pixel-level mask generation (PR2)

Uses bbox prompts from SpeciesNet to generate precise animal masks
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from pathlib import Path
import cv2


class SAMMaskGenerator:
    """
    SAM-based pixel mask generator
    Uses bbox prompts from SpeciesNet to generate precise animal masks
    """
    
    def __init__(
        self,
        model_type: str = 'vit_h',
        checkpoint_path: str = None,
        device: str = 'cuda',
        config: dict = None
    ):
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.config = config or {}
        
        self.points_per_side = self.config.get('points_per_side', 32)
        self.pred_iou_thresh = self.config.get('pred_iou_thresh', 0.88)
        self.stability_score_thresh = self.config.get('stability_score_thresh', 0.95)
        self.min_mask_region_area = self.config.get('min_mask_region_area', 100)
        
        self.sam = None
        self.predictor = None
        self._current_image_set = False
        
        self._load_model()
    
    def _load_model(self):
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            else:
                # Try to find checkpoint
                possible_paths = [
                    'models/sam_vit_h_4b8939.pth',
                    'models/sam_vit_l_0b3195.pth',
                    'models/sam_vit_b_01ec64.pth',
                ]
                
                loaded = False
                for path in possible_paths:
                    if Path(path).exists():
                        model_type = 'vit_h' if 'vit_h' in path else ('vit_l' if 'vit_l' in path else 'vit_b')
                        sam = sam_model_registry[model_type](checkpoint=path)
                        loaded = True
                        break
                
                if not loaded:
                    print("Warning: SAM checkpoint not found. Using fallback mask generation.")
                    return
            
            device = self.device if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            sam.to(device=device)
            self.sam = sam
            self.predictor = SamPredictor(sam)
            
        except ImportError:
            print("Warning: segment_anything not installed. Using fallback mask generation.")
            self.sam = None
            self.predictor = None
    
    def set_image(self, image: np.ndarray):
        """
        Set image for SAM (pre-compute image embedding)
        
        Args:
            image: RGB image (H, W, 3)
        """
        if self.predictor is not None:
            self.predictor.set_image(image)
            self._current_image_set = True
        self._current_image = image
    
    def generate_mask_from_bbox(
        self,
        image: np.ndarray,
        bbox: List[float],
        multimask_output: bool = False
    ) -> np.ndarray:
        """
        Generate mask from bounding box prompt
        
        Args:
            image: RGB image (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks
            
        Returns:
            Binary mask (H, W)
        """
        h, w = image.shape[:2]
        
        if self.predictor is not None:
            # Set image if not already set
            if not self._current_image_set:
                self.set_image(image)
            
            # Convert bbox to numpy array
            input_box = np.array(bbox)
            
            # Generate mask
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=multimask_output,
            )
            
            if multimask_output:
                # Return the highest scoring mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
            else:
                mask = masks[0]
            
            return mask.astype(np.uint8)
        
        # Fallback: create mask from bbox directly with GrabCut
        return self._fallback_mask_from_bbox(image, bbox)
    
    def _fallback_mask_from_bbox(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> np.ndarray:
        """
        Fallback mask generation using GrabCut
        
        Args:
            image: RGB image
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Binary mask
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Use GrabCut for better segmentation
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        try:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except cv2.error:
            # If GrabCut fails, fallback to simple bbox mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def generate_masks_from_bboxes(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[List[float], str, float]]
    ) -> List[np.ndarray]:
        """
        Generate masks for multiple bounding boxes
        
        Args:
            image: RGB image (H, W, 3)
            bboxes: List of (bbox, class_name, confidence)
            
        Returns:
            List of binary masks
        """
        # Set image once for efficiency
        self.set_image(image)
        
        masks = []
        for bbox, class_name, conf in bboxes:
            mask = self.generate_mask_from_bbox(image, bbox)
            masks.append(mask)
        
        # Reset image state
        self._current_image_set = False
        
        return masks
    
    def combine_masks(
        self,
        masks: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Combine multiple masks into a single fur mask
        
        Args:
            masks: List of binary masks
            image_shape: (H, W) of the image
            
        Returns:
            Combined binary mask
        """
        h, w = image_shape
        combined = np.zeros((h, w), dtype=np.uint8)
        
        for mask in masks:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            combined = np.logical_or(combined, mask).astype(np.uint8)
        
        return combined
    
    def postprocess_mask(
        self,
        mask: np.ndarray,
        min_area: int = None,
        apply_morphology: bool = True
    ) -> np.ndarray:
        """
        Postprocess generated mask
        
        Args:
            mask: Binary mask
            min_area: Minimum connected component area to keep
            apply_morphology: Apply morphological operations
            
        Returns:
            Processed mask
        """
        min_area = min_area or self.min_mask_region_area
        
        # Ensure binary
        binary = (mask > 0.5).astype(np.uint8)
        
        if apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Remove small components
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            filtered = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered[labels == i] = 1
            
            return filtered
        
        return binary


def create_sam_generator(
    model_type: str = 'vit_h',
    checkpoint_path: str = None,
    device: str = 'cuda',
    config: dict = None
) -> SAMMaskGenerator:
    """Factory function to create SAM mask generator"""
    return SAMMaskGenerator(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device,
        config=config
    )
