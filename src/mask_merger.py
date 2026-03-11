"""
Mask merging logic module (PR4)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.ndimage import gaussian_filter
import cv2


class MaskMerger:
    """
    Merges animal masks and human hair masks with proper exclusion zones
    
    Logic:
    - final_mask = (animal_mask | human_hair_mask) & ~(face_mask | skin_mask)
    - Apply soft edges for smooth SR blending
    """
    
    def __init__(
        self,
        gaussian_sigma: float = 3.0,
        edge_blur_radius: int = 5,
        min_hair_area: int = 100
    ):
        """
        Initialize mask merger
        
        Args:
            gaussian_sigma: Sigma for Gaussian blur on final mask
            edge_blur_radius: Radius for edge blur
            min_hair_area: Minimum connected component area
        """
        self.gaussian_sigma = gaussian_sigma
        self.edge_blur_radius = edge_blur_radius
        self.min_hair_area = min_hair_area
    
    def merge(
        self,
        animal_mask: np.ndarray = None,
        human_hair_mask: np.ndarray = None,
        face_mask: np.ndarray = None,
        skin_mask: np.ndarray = None,
        original_size: Tuple[int, int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Merge all masks according to logic
        
        Args:
            animal_mask: Binary mask from animals
            human_hair_mask: Binary mask from human hair
            face_mask: Face regions to exclude
            skin_mask: Skin regions to exclude
            original_size: (H, W) of original image
            
        Returns:
            Dictionary with merged and individual masks
        """
        h, w = original_size
        
        # Initialize empty masks if not provided
        animal = self._ensure_size(animal_mask, (h, w))
        human_hair = self._ensure_size(human_hair_mask, (h, w))
        face = self._ensure_size(face_mask, (h, w))
        skin = self._ensure_size(skin_mask, (h, w))
        
        # Compute exclusion mask
        exclude_mask = np.logical_or(face, skin).astype(np.uint8)
        
        # Compute combined hair/fur mask
        combined_hair = np.logical_or(animal, human_hair).astype(np.uint8)
        
        # Apply exclusion
        final_mask = combined_hair * (1 - exclude_mask)
        
        # Post-process
        final_mask = self._postprocess_mask(final_mask)
        
        return {
            'animal_mask': animal,
            'human_hair_mask': human_hair,
            'face_mask': face,
            'skin_mask': skin,
            'exclude_mask': exclude_mask,
            'combined_hair': combined_hair,
            'final_mask': final_mask,
        }
    
    def merge_from_result(
        self,
        segmentation_result
    ) -> np.ndarray:
        """
        Merge from SegmentationResult object
        
        Args:
            segmentation_result: SegmentationResult with individual masks
            
        Returns:
            Final merged mask
        """
        result = self.merge(
            animal_mask=segmentation_result.animal_mask,
            human_hair_mask=segmentation_result.human_hair_mask,
            face_mask=segmentation_result.face_mask,
            skin_mask=segmentation_result.skin_mask,
            original_size=segmentation_result.original_shape
        )
        
        return result['final_mask']
    
    def create_soft_mask(
        self,
        binary_mask: np.ndarray,
        sigma: float = None
    ) -> np.ndarray:
        """
        Create soft mask with Gaussian blur on edges
        
        Args:
            binary_mask: Binary mask (0 or 1)
            sigma: Gaussian sigma (uses default if None)
            
        Returns:
            Soft mask (float, 0-1)
        """
        sigma = sigma or self.gaussian_sigma
        
        # Apply Gaussian blur
        soft_mask = gaussian_filter(binary_mask.astype(float), sigma=sigma)
        soft_mask = np.clip(soft_mask, 0, 1)
        
        return soft_mask
    
    def create_edge_blurred_mask(
        self,
        binary_mask: np.ndarray,
        radius: int = None
    ) -> np.ndarray:
        """
        Create mask with blurred edges using morphological operations
        
        Args:
            binary_mask: Binary mask
            radius: Blur radius (uses default if None)
            
        Returns:
            Edge-blurred mask
        """
        radius = radius or self.edge_blur_radius
        
        # Create blurred version using multiple dilations/erosions
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Erode then dilate to get inner boundary
        inner = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
        
        # Dilate then erode to get outer boundary
        outer = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
        
        # Edge region is the difference
        edge = outer - inner
        
        # Create soft transition
        soft_mask = binary_mask.astype(float)
        
        # Blend edges
        if edge.any():
            edge_blurred = gaussian_filter(edge.astype(float), sigma=radius/2)
            soft_mask = np.where(edge > 0, edge_blurred, soft_mask)
        
        return soft_mask
    
    def _ensure_size(
        self,
        mask: Optional[np.ndarray],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Ensure mask is correct size
        
        Args:
            mask: Input mask or None
            target_size: (H, W) target size
            
        Returns:
            Correctly sized mask
        """
        if mask is None:
            return np.zeros(target_size, dtype=np.uint8)
        
        h, w = target_size
        
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        return mask.astype(np.uint8)
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Postprocess mask: remove small components, smooth edges
        
        Args:
            mask: Binary mask
            
        Returns:
            Processed mask
        """
        # Ensure binary
        binary = (mask > 0).astype(np.uint8)
        
        # Remove small connected components
        if self.min_hair_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            filtered = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_hair_area:
                    filtered[labels == i] = 1
            
            binary = filtered
        
        return binary
    
    def blend_sr_outputs(
        self,
        image: np.ndarray,
        sr_hair: np.ndarray,
        sr_general: np.ndarray,
        soft_mask: np.ndarray
    ) -> np.ndarray:
        """
        Blend two SR outputs using soft mask
        
        Args:
            image: Original image (for reference)
            sr_hair: SR output for hair/fur regions
            sr_general: SR output for general regions
            soft_mask: Soft mask for hair/fur (0-1)
            
        Returns:
            Blended SR output
        """
        # Ensure soft_mask is 3D for broadcasting
        if soft_mask.ndim == 2:
            soft_mask = soft_mask[:, :, np.newaxis]
        
        # Blend
        blended = soft_mask * sr_hair + (1 - soft_mask) * sr_general
        blended = np.clip(blended, 0, 1)
        
        return blended


def create_mask_merger(
    config: dict = None
) -> MaskMerger:
    """
    Factory function to create MaskMerger
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MaskMerger instance
    """
    merging_config = config.get('merging', {}) if config else {}
    
    return MaskMerger(
        gaussian_sigma=merging_config.get('gaussian_sigma', 3.0),
        edge_blur_radius=merging_config.get('edge_blur_radius', 5),
        min_hair_area=merging_config.get('min_hair_area', 100)
    )
