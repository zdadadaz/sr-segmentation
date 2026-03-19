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
        Detections where 'prediction' resolves to a non-mammalia class are discarded.
        Ambiguous results (Classification.ANIMAL, Classification.UNKNOWN, or missing
        prediction key) are kept conservatively.
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
        self.furry_labels: Optional[set] = None
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
            self.furry_labels = self._load_furry_labels()
        else:
            # Detector-only mode — fast, returns all animals regardless of species.
            self.model = SpeciesNet(self.model_path, components='detector', geofence=False)

    def _load_furry_labels(self) -> set:
        """
        Build a set of SpeciesNet label strings that represent furry (mammal) animals.

        The label file format (one per line):
            UUID;class;order;family;genus;species;common_name

        We keep labels where class == 'mammalia'.  Additionally, two sentinel values
        from speciesnet.constants are added as conservative fallbacks:
          - Classification.ANIMAL: ensemble could not narrow to a species → keep
          - Classification.UNKNOWN: classifier returned 'no cv result' → keep
        """
        model_dir = Path(self.model_path)
        info = json.loads((model_dir / 'info.json').read_text(encoding='utf-8'))
        labels_path = model_dir / info['classifier_labels']

        furry = set()
        with open(labels_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) > 1 and parts[1] == 'mammalia':
                    furry.add(line)

        # Conservative: keep detections when classifier is uncertain
        from speciesnet.constants import Classification
        furry.add(Classification.ANIMAL.value)   # generic "animal" sentinel
        furry.add(Classification.UNKNOWN.value)  # "no cv result"
        return furry

    def _is_furry(self, prediction_label: Optional[str]) -> bool:
        """
        Return True if the ensemble prediction indicates a furry (mammal) animal,
        or if the prediction is ambiguous/missing (conservative keep).

        Args:
            prediction_label: Full label string from SpeciesNet ensemble, e.g.
                "UUID;mammalia;carnivora;felidae;panthera;leo;lion", or None.

        Returns:
            True  → keep detection (mammal or uncertain)
            False → discard detection (confirmed non-mammal: aves/reptilia/etc.)
        """
        if prediction_label is None:
            return True  # missing key → conservative keep
        if prediction_label in self.furry_labels:
            return True  # O(1) lookup covers all mammalia + sentinels
        parts = prediction_label.split(';')
        if len(parts) < 2:
            return True  # malformed → conservative keep
        return parts[1] == 'mammalia'  # safety net for any label not in set

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
                if self.use_classifier and not self._is_furry(prediction_label):
                    continue  # discard confirmed non-mammal (bird, reptile, etc.)

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
