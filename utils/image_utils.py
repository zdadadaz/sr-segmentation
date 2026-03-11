"""
Image loading and preprocessing utilities
"""

import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple
import torch
from pathlib import Path


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file path
    
    Args:
        path: Image file path
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def load_image_pil(path: Union[str, Path]) -> Image.Image:
    """Load image as PIL Image"""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save numpy array as image
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        path: Output path
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    Image.fromarray(image).save(path)


def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = None,
    normalize: bool = True
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess image for model inference
    
    Args:
        image: Input image (numpy array or PIL)
        target_size: (H, W) target size, None to keep original
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Preprocessed tensor and original shape
    """
    original_shape = None
    
    # Convert PIL to numpy
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
    
    original_shape = image.shape[:2]
    
    # Resize if needed
    if target_size is not None:
        image = resize_image(image, target_size)
    
    # Convert to tensor
    image = torch.from_numpy(image).float()
    
    # HWC to CHW
    image = image.permute(2, 0, 1)
    
    # Normalize
    if normalize:
        image = image / 255.0
    
    return image, original_shape


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image (H, W, C)
        target_size: (H, W) target size
        
    Returns:
        Resized image
    """
    from cv2 import resize
    
    h, w = target_size
    return resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def pad_image_to_square(image: np.ndarray, pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to square shape
    
    Args:
        image: Input image (H, W, C)
        pad_value: Padding value
        
    Returns:
        Padded image and (pad_top, pad_left) offset
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    
    padded = np.full((max_dim, max_dim, image.shape[2]), pad_value, dtype=image.dtype)
    padded[pad_top:pad_top+h, pad_left:pad_left+w] = image
    
    return padded, (pad_top, pad_left)


def crop_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Crop image to bounding box
    
    Args:
        image: Input image (H, W, C)
        bbox: (x1, y1, x2, y2) bounding box
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return image[y1:y2, x1:x2]


def batch_images(images: list, size: Tuple[int, int] = None) -> torch.Tensor:
    """
    Batch process multiple images
    
    Args:
        images: List of images (numpy arrays)
        size: Target size for all images
        
    Returns:
        Batched tensor (B, C, H, W)
    """
    processed = []
    for img in images:
        img_tensor, _ = preprocess_image(img, target_size=size)
        processed.append(img_tensor)
    return torch.stack(processed)
