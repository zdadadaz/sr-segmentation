"""
Test segmentation pipeline on data/test images
Saves visualizations with colored mask overlays to output/data_test_results/
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SegmentationPipeline
from PIL import Image, ImageDraw, ImageFont


MASK_COLORS = {
    'final': (255, 100, 0, 140),       # orange
    'animal': (0, 200, 100, 120),       # green
    'human_hair': (100, 100, 255, 120), # blue
    'face': (255, 50, 50, 100),         # red
    'skin': (255, 200, 50, 80),         # yellow
    'texture': (200, 0, 255, 80),       # purple
}


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, color_rgba: tuple) -> np.ndarray:
    """Overlay a binary mask on an RGB image with given RGBA color."""
    img_pil = Image.fromarray(img_rgb).convert('RGBA')
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b, a = color_rgba
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                draw.point((x, y), fill=(r, g, b, a))
    return np.array(Image.alpha_composite(img_pil, overlay).convert('RGB'))


def overlay_mask_fast(img_rgb: np.ndarray, mask: np.ndarray, color_rgba: tuple) -> np.ndarray:
    """Fast vectorized mask overlay."""
    result = img_rgb.copy().astype(np.float32)
    r, g, b, a = color_rgba
    alpha = a / 255.0
    color = np.array([r, g, b], dtype=np.float32)
    m = mask.astype(bool)
    result[m] = result[m] * (1 - alpha) + color * alpha
    return result.clip(0, 255).astype(np.uint8)


def draw_bbox_on_array(img: np.ndarray, bbox, label: str, color=(255, 255, 0)) -> np.ndarray:
    """Draw bounding box on image array."""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1 + 4, y1 + 4), label, fill=color)
    return np.array(img_pil)


def create_result_grid(img_rgb: np.ndarray, result) -> np.ndarray:
    """
    Create a 2x3 grid showing:
    [original+bboxes] [final mask overlay]
    [animal mask]     [human hair mask]
    [face/skin mask]  [texture mask]
    """
    h, w = img_rgb.shape[:2]
    thumb_w, thumb_h = min(w, 640), min(h, 480)
    scale = min(thumb_w / w, thumb_h / h)
    nw, nh = int(w * scale), int(h * scale)

    def resize(arr):
        return np.array(Image.fromarray(arr).resize((nw, nh), Image.LANCZOS))

    def mask_to_rgb(mask, color_key):
        if mask is None:
            return np.zeros((nh, nw, 3), dtype=np.uint8)
        m = (mask > 0).astype(np.uint8)
        base = resize(img_rgb)
        return overlay_mask_fast(base, np.array(Image.fromarray(m * 255).resize((nw, nh), Image.NEAREST)) > 127, MASK_COLORS[color_key])

    def label(arr, text):
        out = arr.copy()
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, len(text) * 8 + 10, 20], fill=(0, 0, 0))
        draw.text((5, 3), text, fill=(255, 255, 255))
        return np.array(pil)

    # Panel 1: original with bboxes
    orig = resize(img_rgb).copy()
    for bbox in result.animal_bboxes:
        orig = draw_bbox_on_array(orig, BBoxScaled(bbox, scale), f"{bbox.label} {bbox.confidence:.2f}", color=(0, 255, 100))
    for bbox in result.person_bboxes:
        orig = draw_bbox_on_array(orig, BBoxScaled(bbox, scale), f"person {bbox.confidence:.2f}", color=(100, 100, 255))

    # Panel 2: final mask
    final_panel = mask_to_rgb(result.hair_fur_mask, 'final')

    # Panel 3: animal mask
    animal_panel = mask_to_rgb(result.animal_mask, 'animal')

    # Panel 4: human hair mask
    hair_panel = mask_to_rgb(result.human_hair_mask, 'human_hair')

    # Panel 5: face+skin mask
    if result.face_mask is not None and result.skin_mask is not None:
        fs_mask = np.logical_or(result.face_mask, result.skin_mask).astype(np.uint8)
    elif result.face_mask is not None:
        fs_mask = result.face_mask
    else:
        fs_mask = result.skin_mask
    face_skin_panel = mask_to_rgb(fs_mask, 'face')

    # Panel 6: texture mask
    texture_panel = mask_to_rgb(result.texture_mask, 'texture')

    panels = [
        label(orig, 'Original + Detections'),
        label(final_panel, 'Final Mask (Hair/Fur)'),
        label(animal_panel, 'Animal Mask'),
        label(hair_panel, 'Human Hair Mask'),
        label(face_skin_panel, 'Face + Skin Mask'),
        label(texture_panel, 'Texture Mask'),
    ]

    pad = 4
    cell_w, cell_h = nw + pad, nh + pad
    grid = np.zeros((cell_h * 3 + pad, cell_w * 2 + pad, 3), dtype=np.uint8)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    for panel, (row, col) in zip(panels, positions):
        y0 = pad + row * cell_h
        x0 = pad + col * cell_w
        grid[y0:y0 + nh, x0:x0 + nw] = panel

    return grid


class BBoxScaled:
    """Wrapper to apply scale to bbox for visualization."""
    def __init__(self, bbox, scale):
        self.x1 = bbox.x1 * scale
        self.y1 = bbox.y1 * scale
        self.x2 = bbox.x2 * scale
        self.y2 = bbox.y2 * scale
        self.label = bbox.label
        self.confidence = bbox.confidence


def test_image(pipeline, img_path: Path, output_dir: Path):
    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")
    print('='*60)

    img = np.array(Image.open(img_path).convert('RGB'))
    print(f"  Shape: {img.shape}")

    result = pipeline.segment(img)

    # Stats
    final = result.hair_fur_mask if result.hair_fur_mask is not None else result.final_mask
    coverage = final.sum() / final.size * 100 if final is not None else 0
    print(f"  Time:     {result.processing_time_ms:.0f}ms")
    print(f"  Animals:  {len(result.animal_bboxes)}")
    print(f"  Persons:  {len(result.person_bboxes)}")
    print(f"  Coverage: {coverage:.2f}%")

    for bbox in result.animal_bboxes:
        print(f"    Animal: {bbox.label} (conf={bbox.confidence:.2f})")
    for bbox in result.person_bboxes:
        print(f"    Person (conf={bbox.confidence:.2f})")

    # Save mask
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{img_path.stem}_mask.png"
    Image.fromarray(final * 255).save(mask_path)
    print(f"  Mask saved: {mask_path}")

    # Save grid visualization
    grid = create_result_grid(img, result)
    grid_path = output_dir / f"{img_path.stem}_grid.jpg"
    Image.fromarray(grid).save(grid_path, quality=90)
    print(f"  Grid saved: {grid_path}")

    # Save side-by-side (original | mask overlay)
    overlay = overlay_mask_fast(img, final.astype(bool), MASK_COLORS['final'])
    h, w = img.shape[:2]
    side = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
    side[:, :w] = img
    side[:, w + 4:] = overlay
    side_path = output_dir / f"{img_path.stem}_side_by_side.jpg"
    Image.fromarray(side).save(side_path, quality=90)
    print(f"  Side-by-side saved: {side_path}")

    return result


def main():
    test_dir = Path(__file__).parent / 'data' / 'test'
    output_dir = Path(__file__).parent / 'output' / 'data_test_results'

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in exts)

    if not images:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(images)} images in {test_dir}")
    for p in images:
        print(f"  {p.name}")

    print("\nInitializing pipeline...")
    pipeline = SegmentationPipeline(config_path='configs/default.yaml')
    print("Pipeline ready.")

    results = []
    for img_path in images:
        r = test_image(pipeline, img_path, output_dir)
        results.append(r)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"  Images tested:       {len(results)}")
    avg_time = np.mean([r.processing_time_ms for r in results])
    print(f"  Avg processing time: {avg_time:.0f}ms")
    detected_animals = sum(len(r.animal_bboxes) for r in results)
    detected_persons = sum(len(r.person_bboxes) for r in results)
    print(f"  Total animals:       {detected_animals}")
    print(f"  Total persons:       {detected_persons}")
    print(f"  Output dir:          {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
