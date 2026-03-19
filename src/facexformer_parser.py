"""
FaceXFormer face parser module.

Alternative to BiSeNet for face parsing; uses LaPa 11-class segmentation.
LaPa classes:
  0=background, 1=skin, 2=left_brow, 3=right_brow, 4=left_eye,
  5=right_eye, 6=nose, 7=upper_lip, 8=inner_mouth, 9=lower_lip, 10=hair
"""

import sys
import os
import numpy as np
import torch
import torchvision
from pathlib import Path
from typing import Optional, List, Tuple
from torchvision.transforms import InterpolationMode


HAIR_CLASS = 10
FACE_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9}   # brows, eyes, nose, lips, mouth
SKIN_CLASSES = {1}                           # face skin


def _add_network_to_path():
    """Add FaceXFormer network directory to sys.path."""
    network_parent = Path(__file__).parent.parent / "tmp" / "facexformer-main"
    if str(network_parent) not in sys.path:
        sys.path.insert(0, str(network_parent))


def _adjust_bbox(x_min, y_min, x_max, y_max, img_w, img_h, margin_pct=15):
    w = x_max - x_min
    h = y_max - y_min
    dw = w * margin_pct / 100.0 / 2
    dh = h * margin_pct / 100.0 / 2
    return (
        max(0, x_min - dw),
        max(0, y_min - dh),
        min(img_w, x_max + dw),
        min(img_h, y_max + dh),
    )


_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
])

_DUMMY_LABELS = {
    "segmentation": torch.zeros([224, 224]),
    "lnm_seg":      torch.zeros([5, 2]),
    "landmark":     torch.zeros([68, 2]),
    "headpose":     torch.zeros([3]),
    "attribute":    torch.zeros([40]),
    "a_g_e":        torch.zeros([3]),
    "visibility":   torch.zeros([29]),
}

_PARSE_TASK = torch.tensor([0])


class FaceXFormerParser:
    """Face parser using FaceXFormer (LaPa 11-class)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        config: dict = None,
    ):
        self.config = config or {}
        self.device = device
        self.model_path = model_path
        self.model = None
        self.mtcnn = None
        self._load_model()

    def _load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"FaceXFormer model not found: '{self.model_path}'. "
                "Set models.facexformer in configs/default.yaml."
            )

        _add_network_to_path()
        from network import FaceXFormer
        from facenet_pytorch import MTCNN

        model = FaceXFormer().to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict_backbone"])
        model.eval()
        self.model = model

        # keep_all=True so we can detect multiple faces
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def _run_on_face_crop(self, face_crop_rgb: np.ndarray) -> np.ndarray:
        """
        Run FaceXFormer on a face crop (numpy RGB uint8).
        Returns a 224×224 class map (uint8).
        """
        from PIL import Image as PILImage

        pil = PILImage.fromarray(face_crop_rgb)
        tensor = _TRANSFORMS(pil).unsqueeze(0).to(self.device)

        labels = {k: v.unsqueeze(0).to(self.device) for k, v in _DUMMY_LABELS.items()}
        task = _PARSE_TASK.to(self.device)

        with torch.no_grad():
            *_, seg_output = self.model(tensor, labels, task)

        preds = seg_output.softmax(dim=1)
        mask = torch.argmax(preds, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return mask  # (224, 224)

    def _detect_faces(self, image_rgb: np.ndarray):
        """
        Detect faces with MTCNN. Returns list of (x1,y1,x2,y2) pixel boxes,
        or empty list if none found.
        """
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image_rgb)
        boxes, probs = self.mtcnn.detect(pil)
        if boxes is None:
            return []
        results = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.9:
                continue
            results.append(box)  # [x1, y1, x2, y2] float
        return results

    def parse(self, image: np.ndarray, crop_box=None) -> dict:
        """
        Parse face regions in image.

        Args:
            image:    RGB numpy array (H, W, 3)
            crop_box: optional (x1,y1,x2,y2) pixel box to restrict search area

        Returns:
            {'hair': uint8 mask, 'face': uint8 mask, 'skin': uint8 mask}
            all masks are full-image size (H, W)
        """
        H, W = image.shape[:2]
        hair_mask = np.zeros((H, W), dtype=np.uint8)
        face_mask = np.zeros((H, W), dtype=np.uint8)
        skin_mask = np.zeros((H, W), dtype=np.uint8)

        # Region to search in (full image or crop)
        if crop_box is not None:
            cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(W, cx2), min(H, cy2)
            region = image[cy1:cy2, cx1:cx2]
            offset_x, offset_y = cx1, cy1
        else:
            region = image
            offset_x, offset_y = 0, 0

        rH, rW = region.shape[:2]
        if rH == 0 or rW == 0:
            return {"hair": hair_mask, "face": face_mask, "skin": skin_mask}

        face_boxes = self._detect_faces(region)
        if not face_boxes:
            return {"hair": hair_mask, "face": face_mask, "skin": skin_mask}

        for box in face_boxes:
            bx1, by1, bx2, by2 = box
            # Add margin
            bx1, by1, bx2, by2 = _adjust_bbox(bx1, by1, bx2, by2, rW, rH)
            bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)

            face_crop = region[by1:by2, bx1:bx2]
            if face_crop.size == 0:
                continue

            crop_h = by2 - by1
            crop_w = bx2 - bx1

            seg_224 = self._run_on_face_crop(face_crop)  # (224, 224)

            # Reject likely-failed segmentations: when background (class 0)
            # covers < 20% of the crop, the model probably could not parse
            # the face (e.g. side profile) and classified everything as skin,
            # producing a solid-block artifact.
            bg_ratio = (seg_224 == 0).sum() / seg_224.size
            if bg_ratio < 0.20:
                continue

            # Resize back to crop size
            seg_full = np.array(
                __import__("PIL").Image.fromarray(seg_224).resize(
                    (crop_w, crop_h), __import__("PIL").Image.NEAREST
                )
            )

            # Map into full-image masks using full-image coords
            fy1 = offset_y + by1
            fy2 = offset_y + by2
            fx1 = offset_x + bx1
            fx2 = offset_x + bx2

            # Clamp to image
            py1, py2 = max(0, fy1), min(H, fy2)
            px1, px2 = max(0, fx1), min(W, fx2)
            sy1 = py1 - fy1
            sy2 = sy1 + (py2 - py1)
            sx1 = px1 - fx1
            sx2 = sx1 + (px2 - px1)

            region_seg = seg_full[sy1:sy2, sx1:sx2]

            hair_mask[py1:py2, px1:px2] = np.logical_or(
                hair_mask[py1:py2, px1:px2], region_seg == HAIR_CLASS
            ).astype(np.uint8)

            face_region = np.zeros_like(region_seg, dtype=bool)
            for cls in FACE_CLASSES:
                face_region |= region_seg == cls
            face_mask[py1:py2, px1:px2] = np.logical_or(
                face_mask[py1:py2, px1:px2], face_region
            ).astype(np.uint8)

            skin_region = np.zeros_like(region_seg, dtype=bool)
            for cls in SKIN_CLASSES:
                skin_region |= region_seg == cls
            skin_mask[py1:py2, px1:px2] = np.logical_or(
                skin_mask[py1:py2, px1:px2], skin_region
            ).astype(np.uint8)

        return {"hair": hair_mask, "face": face_mask, "skin": skin_mask}


def create_facexformer_parser(
    model_path: str = None,
    device: str = "cpu",
    config: dict = None,
) -> FaceXFormerParser:
    """Factory function to create FaceXFormerParser."""
    return FaceXFormerParser(model_path=model_path, device=device, config=config)
