# SR Segmentation Project
### Super Resolution + Semantic Segmentation for Hair/Fur Enhancement

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a Super-Resolution (SR) pipeline that uses semantic segmentation masks to specifically enhance hair and fur textures in images of animals and humans.

---

## Quick Start

```bash
# 1. Create venv with Python 3.12 (speciesnet requires Python <3.13)
python3.12 -m venv venv
source venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Prepare dataset
python prepare_dataset.py --src_dir my_photos --output_dir data/v1

# 4. Train
python train.py --hr_dir data/v1/hr --lr_dir data/v1/lr --mask_dir data/v1/mask
```

---

## Project Structure

```
sr-segmentation/
├── src/
│   ├── pipeline.py             # Main segmentation pipeline
│   ├── speciesnet.py           # Animal detection (SpeciesNet / MegaDetector v5)
│   ├── sam.py                  # SAM pixel mask generation
│   ├── bisenet.py              # BiSeNet face parser
│   ├── facexformer_parser.py   # FaceXFormer face parser (alternative to BiSeNet)
│   ├── mask_merger.py          # Mask merging logic
│   ├── texture_classifier.py   # Gabor texture fallback
│   ├── realesrgan_arch.py      # RRDBNet / SegGuidedRRDBNet architectures
│   ├── dataset.py              # Training dataset loader
│   ├── dataset_generator.py    # Auto-labeling tool
│   └── sr_integration.py       # SFT blocks + SegAwareLoss
├── train.py                    # Main training script
├── prepare_dataset.py          # Auto-labeling & dataset preparation
├── generate_dummy_data.py      # Quick-start dummy data generator
├── configs/
│   └── default.yaml            # Configuration (models, thresholds, face_parser)
├── models/                     # Pre-trained model weights
├── tmp/
│   └── facexformer-main/       # FaceXFormer network code (cloned from GitHub)
└── data/                       # Dataset directory
```

---

## Installation

**Requires Python 3.12** (speciesnet depends on yolov5 which is incompatible with Python ≥3.13).

```bash
# Create venv
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# reverse_geocoder (speciesnet dependency) must be installed via conda on macOS
conda install -c conda-forge reverse_geocoder
```

---

## Model Weights

Download and place in `models/`:

| Model | File | Purpose | Download |
|-------|------|---------|----------|
| SpeciesNet (MegaDetector v5) | `models/speciesNet/` | Animal detection | [cameratrapai releases](https://github.com/google/cameratrapai) |
| SAM ViT-H | `sam_vit_h.pth` | Pixel mask generation | [fbaipublicfiles](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| BiSeNet | `face_parsing.pth` | Face parsing (default) | [GDrive](https://docs.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) |
| FaceXFormer | `facexfromer.pt` | Face parsing (alternative) | [FaceXFormer releases](https://github.com/pranavphoenix/FaceXFormer) |
| YOLOv8n | `yolov8n.pt` | Person detection | [ultralytics](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) |

FaceXFormer also requires its network code in `tmp/facexformer-main/`:
```bash
git clone https://github.com/pranavphoenix/FaceXFormer tmp/facexformer-main
```

---

## Segmentation Pipeline

The pipeline runs 5 steps on each image:

```
1. SpeciesNet (MegaDetector v5)  →  animal bounding boxes
2. SAM (bbox prompt)             →  animal pixel masks
3. YOLOv8 person detector        →  person bounding boxes
4. Face parser (BiSeNet or       →  hair / face / skin masks
   FaceXFormer, per config)
5. Mask merger + Gabor texture   →  final hair/fur mask
   classifier fallback
```

**Face parser selection** — edit `configs/default.yaml`:
```yaml
face_parser: bisenet       # default, CelebAMask-HQ 19-class
# face_parser: facexformer  # LaPa 11-class, better for non-frontal faces
```

### Programmatic Usage

```python
from PIL import Image
from src.pipeline import SegmentationPipeline

pipeline = SegmentationPipeline()  # loads configs/default.yaml

img = Image.open("photo.jpg")
result = pipeline.segment(img)

hair_mask  = result.final_mask          # (H, W) uint8 binary mask
soft_mask  = result.get_soft_mask(3.0)  # Gaussian-blurred for SR blending
```

---

## Dataset Preparation

### A. Auto-label real images

```bash
python prepare_dataset.py \
  --src_dir path/to/hr_images \
  --output_dir data/my_dataset \
  --scale 4
```

Produces `hr/`, `lr/`, and `mask/` subdirectories and a `split.json`.

### B. Dummy data (quick smoke test)

```bash
python generate_dummy_data.py
```

---

## Training

Two training modes are available via `--model_type`:

### `sft` (default) — Segmentation-guided with SFT injection

Uses `SegGuidedRRDBNet`: the mask is injected into the feature space via **Spatial Feature Transform (SFT)** blocks before and after the RRDB body.

```bash
python train.py \
  --hr_dir data/my_dataset/hr \
  --lr_dir data/my_dataset/lr \
  --mask_dir data/my_dataset/mask \
  --model_type sft \
  --epochs 50 --batch_size 4 \
  --save_dir experiments/sft
```

### `mask_concat` — Mask concatenated to input, no SFT

Uses a plain `RRDBNet(num_in_ch=4)`: the mask is simply concatenated to the LR image as a 4th channel before the network. No SFT layers.

```bash
python train.py \
  --hr_dir data/my_dataset/hr \
  --lr_dir data/my_dataset/lr \
  --mask_dir data/my_dataset/mask \
  --model_type mask_concat \
  --epochs 50 --batch_size 4 \
  --save_dir experiments/mask_concat
```

### Common training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `sft` | `sft` or `mask_concat` |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--patch_size` | 256 | HR crop size |
| `--scale` | 4 | SR scale factor |
| `--hair_weight` | 2.0 | Loss weight for hair/fur regions |
| `--no_perceptual` | — | Disable perceptual loss |
| `--no_ssim` | — | Disable SSIM loss |

Both modes use `SegAwareLoss` which applies higher pixel-level loss weight to hair/fur regions.

---

## PR Progress

- [x] PR1: Project scaffold + inference pipeline skeleton
- [x] PR2: SpeciesNet (MegaDetector v5) + SAM pixel mask
- [x] PR3: BiSeNet face parsing
- [x] PR4: Mask merging logic + full pipeline
- [x] PR5: Dataset auto-labeling pipeline
- [x] PR6: SR model SFT integration & training script
- [x] PR7: Texture classifier fallback
- [x] PR8: Replace YOLOv8 COCO with SpeciesNet for better wildlife detection
- [x] PR9: FaceXFormer as alternative face parser (config-selectable)
- [x] PR10: `mask_concat` training mode (RRDBNet 4-ch input, no SFT)
