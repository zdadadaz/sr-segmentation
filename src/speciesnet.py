"""
SpeciesNet animal detection module

Uses Google SpeciesNet (MegaDetector v5 + EfficientNet V2 classifier)
for detecting animals in images.
"""

import json
import tempfile
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image


class SpeciesNetDetector:
    """
    Animal detector backed by Google SpeciesNet (MegaDetector v5).

    In detector-only mode (use_classifier=False, default):
        detect() output per image:
            detections: list of {
                'category': '1' (animal) | '2' (human) | '3' (vehicle),
                'label':    'animal' | 'human' | 'vehicle',
                'conf':     float,
                'bbox':     [xmin, ymin, width, height]  — normalized [0, 1]
            }

    In full-pipeline mode (use_classifier=True):
        predict() output per image additionally contains:
            'prediction': full label string "UUID;class;order;family;genus;species;common_name"

        Taxonomy filtering (only active when use_classifier=True):
          - If 'allowed_taxonomy' is NOT set in config → keep mammalia only (default)
          - If 'allowed_taxonomy' IS set → filter by specified levels (class/order/family/genus/species)
            Each non-empty level acts as an allowlist; ALL specified levels must match.
            Sentinel predictions (ANIMAL, UNKNOWN) are always kept conservatively.

        Example config:
            speciesnet:
              use_classifier: true
              allowed_taxonomy:
                family: [felidae, canidae, ursidae]  # cats, dogs, bears only
    """

    def __init__(
        self,
        model_path: str = None,
        config: dict = None,
        device: str = 'cpu',
    ):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.use_classifier = self.config.get('use_classifier', False)
        self.model_path = model_path
        self.model = None
        self._sentinels: Optional[set] = None   # Classification.ANIMAL / UNKNOWN
        self._load_model()

    def _load_model(self):
        from speciesnet import SpeciesNet
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"SpeciesNet model directory not found: '{self.model_path}'. "
                "Set models.speciesnet in configs/default.yaml."
            )
        if self.use_classifier:
            # Full pipeline: detector + EfficientNet V2 classifier + ensemble.
            # Significantly slower than detector-only (classifier runs on each crop).
            self.model = SpeciesNet(self.model_path, components='all', geofence=False)
            self._sentinels = self._load_sentinels()
        else:
            # Detector-only mode — fast, returns all animals regardless of species.
            self.model = SpeciesNet(self.model_path, components='detector', geofence=False)

    def _load_sentinels(self) -> set:
        """Load Classification.ANIMAL and Classification.UNKNOWN sentinel strings."""
        from speciesnet.constants import Classification
        return {Classification.ANIMAL.value, Classification.UNKNOWN.value}

    def _is_allowed(self, prediction_label: Optional[str]) -> bool:
        """
        Return True if the prediction passes the active taxonomy filter.

        Filter logic:
          - None / malformed label          → conservative keep (True)
          - Sentinel (ANIMAL / UNKNOWN)     → conservative keep (True)
          - allowed_taxonomy not configured → keep mammalia only (default)
          - allowed_taxonomy configured     → each non-empty level acts as an
            allowlist; ALL specified levels must match.

        Taxonomy label format:
            UUID ; class ; order ; family ; genus ; species ; common_name
              0      1       2       3        4       5          6
        """
        if prediction_label is None:
            return True
        if prediction_label in self._sentinels:
            return True

        parts = prediction_label.split(';')
        if len(parts) < 2:
            return True  # malformed → conservative keep

        taxonomy = self.config.get('allowed_taxonomy') or {}
        level_map = {'class': 1, 'order': 2, 'family': 3, 'genus': 4, 'species': 5}
        has_whitelist = any(taxonomy.get(lvl) for lvl in level_map)

        if not has_whitelist:
            # Default behaviour: mammalia only
            return parts[1] == 'mammalia'

        # Custom whitelist: every specified level must match
        for level, idx in level_map.items():
            allowed = taxonomy.get(level) or []
            if not allowed:
                continue                    # level not restricted
            if idx >= len(parts) or not parts[idx]:
                continue                    # label doesn't reach this level → keep
            if parts[idx] not in allowed:
                return False
        return True

    def detect_animals(
        self,
        image: np.ndarray,
        filter_furry: bool = True,
    ) -> List[Tuple[List[float], str, float]]:
        """
        Detect animals in image.

        Args:
            image:        RGB numpy array (H, W, 3)
            filter_furry: deprecated/unused — mammal filtering is controlled by
                          the 'use_classifier' config flag instead

        Returns:
            List of (bbox_xyxy, label, confidence)
            bbox_xyxy = [x1, y1, x2, y2] in pixel coordinates
        """
        h, w = image.shape[:2]

        # SpeciesNet needs a file path; write to a temp PNG
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        Image.fromarray(image).save(tmp_path)

        if self.use_classifier:
            # predict() runs detector → classifier → ensemble and supports
            # single_thread mode (detect() does not).
            result = self.model.predict(
                filepaths=[tmp_path],
                run_mode='single_thread',
                progress_bars=False,
            )
        else:
            result = self.model.detect(filepaths=[tmp_path], progress_bars=False)

        Path(tmp_path).unlink(missing_ok=True)

        detections = []
        for pred in result.get('predictions', []):
            prediction_label = pred.get('prediction') if self.use_classifier else None

            for d in pred.get('detections', []):
                if d.get('category') != '1':          # only animals
                    continue
                if d['conf'] < self.confidence_threshold:
                    continue
                if self.use_classifier and not self._is_allowed(prediction_label):
                    continue  # discard by taxonomy filter

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
