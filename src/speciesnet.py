"""
SpeciesNet animal detection module

Uses Google SpeciesNet (MegaDetector v5 + EfficientNet V2 classifier)
for detecting animals in images.
"""

import tempfile
import numpy as np
from typing import List, Tuple
from pathlib import Path
from PIL import Image


class SpeciesNetDetector:
    """
    Animal detector backed by Google SpeciesNet (MegaDetector v5).

    SpeciesNet detect() output per image:
        detections: list of {
            'category': '1' (animal) | '2' (human) | '3' (vehicle),
            'label':    'animal' | 'human' | 'vehicle',
            'conf':     float,
            'bbox':     [xmin, ymin, width, height]  — normalized [0, 1]
        }
    """

    def __init__(
        self,
        model_path: str = None,
        config: dict = None,
        device: str = 'cpu',
    ):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        from speciesnet import SpeciesNet
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"SpeciesNet model directory not found: '{self.model_path}'. "
                "Set models.speciesnet in configs/default.yaml."
            )
        # detector-only, no geofencing needed
        self.model = SpeciesNet(self.model_path, components='detector', geofence=False)

    def detect_animals(
        self,
        image: np.ndarray,
        filter_furry: bool = True,
    ) -> List[Tuple[List[float], str, float]]:
        """
        Detect animals in image.

        Args:
            image:        RGB numpy array (H, W, 3)
            filter_furry: unused — kept for API compatibility

        Returns:
            List of (bbox_xyxy, label, confidence)
            bbox_xyxy = [x1, y1, x2, y2] in pixel coordinates
        """
        h, w = image.shape[:2]

        # SpeciesNet needs a file path; write to a temp PNG
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        Image.fromarray(image).save(tmp_path)

        result = self.model.detect(filepaths=[tmp_path], progress_bars=False)
        Path(tmp_path).unlink(missing_ok=True)

        detections = []
        for pred in result.get('predictions', []):
            for d in pred.get('detections', []):
                if d.get('category') != '1':          # only animals
                    continue
                if d['conf'] < self.confidence_threshold:
                    continue

                xmin, ymin, bw, bh = d['bbox']        # normalized xywh
                x1 = xmin * w
                y1 = ymin * h
                x2 = (xmin + bw) * w
                y2 = (ymin + bh) * h
                detections.append(([x1, y1, x2, y2], 'animal', float(d['conf'])))

        return detections


def create_speciesnet_detector(
    model_path: str = None,
    config: dict = None,
    device: str = 'cpu',
) -> SpeciesNetDetector:
    """Factory function to create SpeciesNetDetector."""
    return SpeciesNetDetector(
        model_path=model_path,
        config=config,
        device=device,
    )
