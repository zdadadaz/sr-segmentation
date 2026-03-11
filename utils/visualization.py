"""
Visualization utilities for segmentation masks
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List
from pathlib import Path


def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
    apply_colormap: bool = False
) -> np.ndarray:
    """
    Overlay mask on image
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Transparency
        color: RGB color for mask
        apply_colormap: Use colormap instead of solid color
        
    Returns:
        Visualization image
    """
    h, w = image.shape[:2]
    
    # Ensure mask is correct size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    if apply_colormap:
        mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    else:
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
    
    # Blend
    result = (image * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    return result


def visualize_masks_grid(
    image: np.ndarray,
    masks: dict,
    titles: List[str] = None
) -> np.ndarray:
    """
    Create grid visualization of multiple masks
    
    Args:
        image: Original RGB image
        masks: Dict of {name: mask_array}
        titles: Optional list of titles
        
    Returns:
        Grid visualization image
    """
    n = len(masks)
    if titles is None:
        titles = list(masks.keys())
    
    # Create figure
    h, w = image.shape[:2]
    cell_h, cell_w = h, w
    
    # Calculate grid size
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # Create canvas
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for i, (name, mask) in enumerate(masks.items()):
        row = i // cols
        col = i % cols
        
        y1 = row * cell_h
        x1 = col * cell_w
        
        # Visualize mask
        color = colors[i % len(colors)]
        vis = visualize_mask(image, mask, alpha=0.5, color=color)
        
        canvas[y1:y1+cell_h, x1:x1+cell_w] = vis
        
        # Add title
        if titles and i < len(titles):
            cv2.putText(
                canvas, titles[i], (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
    
    return canvas


def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Tuple],
    labels: List[str] = None,
    scores: List[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: RGB image
        bboxes: List of (x1, y1, x2, y2) tuples
        labels: Optional list of labels
        scores: Optional list of confidence scores
        color: Box color
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    result = image.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if labels or scores:
            label = labels[i] if labels else ""
            if scores and i < len(scores):
                label += f" {scores[i]:.2f}"
            
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(result, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result


def create_side_by_side(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Input",
    title2: str = "Output"
) -> np.ndarray:
    """
    Create side-by-side comparison
    
    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
        
    Returns:
        Side-by-side image
    """
    # Resize to match height
    h = max(image1.shape[0], image2.shape[0])
    
    def resize_match(img, target_h):
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            w = int(img.shape[1] * scale)
            return cv2.resize(img, (w, target_h))
        return img
    
    img1 = resize_match(image1, h)
    img2 = resize_match(image2, h)
    
    result = np.hstack([img1, img2])
    
    # Add titles
    cv2.putText(result, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, title2, (img1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result


def save_visualization(
    image: np.ndarray,
    path: str,
    masks: dict = None,
    bboxes: list = None
) -> None:
    """
    Save visualization to file
    
    Args:
        image: Base image
        path: Output path
        masks: Optional dict of masks to overlay
        bboxes: Optional list of bboxes to draw
    """
    result = image.copy()
    
    # Draw masks
    if masks:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for i, (name, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            result = visualize_mask(result, mask, alpha=0.4, color=color)
    
    # Draw bboxes
    if bboxes:
        result = draw_bboxes(result, bboxes)
    
    # Save
    cv2.imwrite(str(path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def create_mask_legend() -> np.ndarray:
    """
    Create a color legend for common segmentation classes
    
    Returns:
        Legend image
    """
    classes = [
        ("Background", (0, 0, 0)),
        ("Hair/Fur", (0, 255, 0)),
        ("Face", (255, 0, 0)),
        ("Skin", (255, 128, 128)),
        ("Vegetation", (0, 128, 0)),
    ]
    
    h = len(classes) * 40 + 20
    w = 300
    legend = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    y = 10
    for name, color in classes:
        cv2.rectangle(legend, (10, y), (40, y + 30), color, -1)
        cv2.putText(legend, name, (50, y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y += 40
    
    return legend
