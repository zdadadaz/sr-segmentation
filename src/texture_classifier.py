"""
Texture classifier fallback (PR7)

Gabor filter bank for detecting fur/hair-like textures in regions
not covered by SpeciesNet or BiSeNet (e.g., plush toys, blankets, carpets)
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from scipy import ndimage


class TextureClassifier:
    """
    Gabor filter bank texture classifier for hair/fur-like textures
    
    Detects fur-like patterns in regions not already labeled by
    SpeciesNet or BiSeNet
    """
    
    # Gabor filter parameters tuned for fur/hair textures
    DEFAULT_FREQUENCIES = [0.05, 0.1, 0.15, 0.2, 0.3]
    DEFAULT_ORIENTATIONS = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    
    def __init__(
        self,
        frequencies: List[float] = None,
        orientations: List[float] = None,
        sigma: float = 3.0,
        threshold: float = 0.5,
        min_area: int = 200,
        config: dict = None
    ):
        """
        Initialize texture classifier
        
        Args:
            frequencies: Gabor filter frequencies
            orientations: Gabor filter orientations (radians)
            sigma: Gaussian envelope sigma
            threshold: Classification threshold
            min_area: Minimum region area to consider
            config: Configuration dictionary
        """
        self.config = config or {}
        
        self.frequencies = frequencies or self.DEFAULT_FREQUENCIES
        self.orientations = orientations or self.DEFAULT_ORIENTATIONS
        self.sigma = sigma
        self.threshold = self.config.get('threshold', threshold)
        self.min_area = self.config.get('min_area', min_area)
        
        # Build Gabor filter bank
        self.filters = self._build_filter_bank()
    
    def _build_filter_bank(self) -> List[np.ndarray]:
        """Build Gabor filter bank"""
        filters = []
        
        ksize = 31  # Kernel size
        
        for freq in self.frequencies:
            for theta in self.orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(ksize, ksize),
                    sigma=self.sigma,
                    theta=theta,
                    lambd=1.0 / freq,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                
                # Normalize
                kernel /= 1.5 * kernel.sum()
                
                filters.append(kernel)
        
        return filters
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gabor texture features from image
        
        Args:
            image: Grayscale image (H, W)
            
        Returns:
            Feature map (H, W, num_filters)
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = image.shape
        features = np.zeros((h, w, len(self.filters)), dtype=np.float32)
        
        for i, kernel in enumerate(self.filters):
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            features[:, :, i] = np.abs(filtered)
        
        return features
    
    def classify_texture(
        self,
        image: np.ndarray,
        exclude_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Classify fur/hair-like texture regions
        
        Args:
            image: RGB image (H, W, 3)
            exclude_mask: Binary mask of regions already labeled (to exclude)
            
        Returns:
            Binary mask of fur-like texture regions
        """
        h, w = image.shape[:2]
        
        # Extract Gabor features
        features = self.extract_features(image)
        
        # Compute energy (average response across all filters)
        energy = np.mean(features, axis=2)
        
        # Compute variance (texture complexity)
        variance = np.var(features, axis=2)
        
        # Fur/hair has high energy + high variance (lots of fine detail)
        # Smooth surfaces have low variance
        # Regular textures (bricks, tiles) have specific frequency responses
        
        # Normalize
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        variance_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
        
        # Compute fur score: high energy AND high variance
        fur_score = energy_norm * variance_norm
        
        # Additional: check for directional consistency (fur tends to have direction)
        direction_score = self._compute_direction_score(features)
        
        # Combined score
        combined_score = 0.5 * fur_score + 0.5 * direction_score
        
        # Threshold
        fur_mask = (combined_score > self.threshold).astype(np.uint8)
        
        # Apply exclusion mask
        if exclude_mask is not None:
            if exclude_mask.shape[:2] != (h, w):
                exclude_mask = cv2.resize(exclude_mask.astype(np.uint8), (w, h))
            fur_mask = fur_mask * (1 - exclude_mask)
        
        # Post-process
        fur_mask = self._postprocess(fur_mask)
        
        return fur_mask
    
    def _compute_direction_score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute directional consistency score
        Fur/hair tends to have strong responses in specific directions
        
        Args:
            features: Gabor features (H, W, num_filters)
            
        Returns:
            Direction score map (H, W)
        """
        h, w, n = features.shape
        num_orientations = len(self.orientations)
        num_frequencies = len(self.frequencies)
        
        # Reshape to (H, W, num_freq, num_orient)
        features_4d = features.reshape(h, w, num_frequencies, num_orientations)
        
        # For each frequency, find the dominant orientation
        # High directional consistency = high score
        max_orient_response = np.max(features_4d, axis=3)
        mean_orient_response = np.mean(features_4d, axis=3) + 1e-8
        
        # Directional ratio: max/mean. Higher means more directional
        direction_ratio = max_orient_response / mean_orient_response
        
        # Average across frequencies
        direction_score = np.mean(direction_ratio, axis=2)
        
        # Normalize
        direction_score = (direction_score - direction_score.min()) / (direction_score.max() - direction_score.min() + 1e-8)
        
        return direction_score
    
    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Post-process texture mask"""
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small components
        if self.min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            filtered = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                    filtered[labels == i] = 1
            
            mask = filtered
        
        return mask
    
    def get_fur_score_map(self, image: np.ndarray) -> np.ndarray:
        """
        Get continuous fur probability map
        
        Args:
            image: RGB image
            
        Returns:
            Float score map (H, W) in [0, 1]
        """
        features = self.extract_features(image)
        
        energy = np.mean(features, axis=2)
        variance = np.var(features, axis=2)
        
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        variance_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
        
        direction_score = self._compute_direction_score(features)
        
        return 0.5 * (energy_norm * variance_norm) + 0.5 * direction_score


def create_texture_classifier(config: dict = None) -> TextureClassifier:
    """Factory function to create TextureClassifier"""
    config = config or {}
    return TextureClassifier(
        threshold=config.get('threshold', 0.5),
        min_area=config.get('min_area', 200),
        config=config
    )
