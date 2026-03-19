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

# 4. Edit training config
#    Set data paths and hyperparameters in configs/train.yaml

# 5. Train (standard)
python train.py --config configs/train.yaml

# 6. GAN fine-tune
python train_gan.py --config configs/train.yaml \
  --pretrained_g experiments/run/epoch_50.pth
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
│   ├── unet.py                 # UNetSR / SegGuidedUNetSR architectures
│   ├── discriminator.py        # VGGDiscriminator for GAN training
│   ├── dataset.py              # Training dataset loader
│   ├── dataset_generator.py    # Auto-labeling tool
│   └── sr_integration.py       # SFTBlock + SegAwareLoss
├── train.py                    # Standard (non-GAN) training script
├── train_gan.py                # GAN training script
├── prepare_dataset.py          # Auto-labeling & dataset preparation
├── generate_dummy_data.py      # Quick-start dummy data generator
├── configs/
│   ├── default.yaml            # Segmentation pipeline configuration (models, thresholds)
│   └── train.yaml              # SR training configuration (data paths, hyperparameters)
├── models/                     # Pre-trained model weights
├── tmp/
│   └── facexformer-main/       # FaceXFormer network code (cloned from GitHub)
└── data/                       # Dataset directory
```

---

## Installation

**Requires Python 3.12** (speciesnet depends on yolov5 which is incompatible with Python ≥3.13).

```bash
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

FaceXFormer also requires its network code:
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

**Face parser selection** — `configs/default.yaml`:
```yaml
face_parser: bisenet       # CelebAMask-HQ 19-class
# face_parser: facexformer  # LaPa 11-class; use_classifier for mammals only
```

**SpeciesNet classifier** (opt-in, filters non-mammal animals):
```yaml
speciesnet:
  use_classifier: true   # enables EfficientNet V2; slower but filters birds/reptiles
```

### Programmatic Usage

```python
from PIL import Image
from src.pipeline import SegmentationPipeline

pipeline = SegmentationPipeline()  # loads configs/default.yaml
result = pipeline.segment(Image.open("photo.jpg"))

hair_mask = result.final_mask          # (H, W) uint8 binary mask
soft_mask = result.get_soft_mask(3.0)  # Gaussian-blurred for SR blending
```

---

## Dataset Preparation

```bash
# Auto-label real images
python prepare_dataset.py \
  --src_dir path/to/hr_images \
  --output_dir data/my_dataset \
  --scale 4

# Quick smoke test with synthetic data
python generate_dummy_data.py
```

Produces `hr/`, `lr/`, and `mask/` subdirectories plus a `split.json`.

---

## Training

### Generator Architectures

Two architectures are available via `--arch`:

| `--arch` | Model | Description |
|----------|-------|-------------|
| `rrdb` (default) | `RRDBNet` / `SegGuidedRRDBNet` | Real-ESRGAN backbone, 23 RRDB blocks |
| `unet` | `UNetSR` / `SegGuidedUNetSR` | 2-level U-Net with PixelShuffle output |

### Mask Integration Modes

Two modes are available via `--model_type`:

| `--model_type` | Description |
|----------------|-------------|
| `sft` (default) | Mask injected into feature space via **SFT** (Spatial Feature Transform) blocks |
| `mask_concat` | Mask concatenated to LR image as 4th channel (`num_in_ch=4`); no SFT layers |

All 4 combinations (`rrdb+sft`, `rrdb+mask_concat`, `unet+sft`, `unet+mask_concat`) are supported.

---

### Standard Training (`train.py`)

Pixel-level supervision only (SegAwareLoss = weighted L1 + perceptual + SSIM).

Set data paths and hyperparameters in `configs/train.yaml`, then:

```bash
# Config-driven (recommended)
python train.py --config configs/train.yaml

# CLI overrides (any config value can be overridden)
python train.py --config configs/train.yaml \
  --arch unet --model_type sft \
  --epochs 50 --save_dir experiments/unet_sft
```

| Argument | Config key | Default | Description |
|----------|------------|---------|-------------|
| `--config` | — | — | Path to training YAML config |
| `--hr_dir` | `data.hr_dir` | — | Path to HR images (required) |
| `--lr_dir` | `data.lr_dir` | — | Path to LR images (required) |
| `--mask_dir` | `data.mask_dir` | — | Path to masks (required) |
| `--arch` | `model.arch` | `rrdb` | `rrdb` or `unet` |
| `--model_type` | `model.model_type` | `sft` | `sft` or `mask_concat` |
| `--scale` | `model.scale` | 4 | SR scale factor |
| `--epochs` | `train.epochs` | 10 | Training epochs |
| `--batch_size` | `train.batch_size` | 4 | Batch size |
| `--lr` | `train.lr` | 1e-4 | Learning rate |
| `--patch_size` | `train.patch_size` | 256 | HR crop size |
| `--save_dir` | `train.save_dir` | `experiments` | Checkpoint directory |
| `--device` | `train.device` | `auto` | `cuda` / `cpu` / `auto` |
| `--hair_weight` | `loss.hair_weight` | 2.0 | Loss weight for hair/fur regions |
| `--no_perceptual` | `loss.use_perceptual` | — | Disable perceptual loss |
| `--no_ssim` | `loss.use_ssim` | — | Disable SSIM loss |

---

### GAN Training (`train_gan.py`)

Real-ESRGAN-style adversarial training. Typically used to fine-tune a pretrained generator.

**Losses:**
- **Generator**: `w_pixel` × SegAwareLoss(L1) + `w_perceptual` × VGG perceptual + `w_adv` × LSGAN
- **Discriminator**: LSGAN (MSE, real=1 / fake=0) — more stable than standard BCE

```bash
# Step 1: pretrain generator with pixel loss
python train.py --config configs/train.yaml \
  --epochs 50 --save_dir experiments/pretrain

# Step 2: GAN fine-tune (set gan.pretrained_g in config or pass via CLI)
python train_gan.py --config configs/train.yaml \
  --pretrained_g experiments/pretrain/epoch_50.pth \
  --epochs 100 --save_dir experiments/gan
```

| Argument | Config key | Default | Description |
|----------|------------|---------|-------------|
| `--config` | — | — | Path to training YAML config |
| `--pretrained_g` | `gan.pretrained_g` | — | Warm-start from `train.py` checkpoint |
| `--lr_g` | `gan.lr_g` | 1e-4 | Generator learning rate |
| `--lr_d` | `gan.lr_d` | 1e-4 | Discriminator learning rate |
| `--w_pixel` | `gan.w_pixel` | 1.0 | SegAwareLoss weight |
| `--w_perceptual` | `gan.w_perceptual` | 0.1 | VGG perceptual loss weight |
| `--w_adv` | `gan.w_adv` | 0.01 | Adversarial loss weight |
| `--d_feat` | `gan.d_feat` | 64 | Discriminator base channels |
| `--save_every` | `gan.save_every` | 5 | Save checkpoint every N epochs |

Checkpoints are saved as `epoch_N_G.pth` and `epoch_N_D.pth` separately.

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
- [x] PR11: UNetSR + SegGuidedUNetSR architectures; VGGDiscriminator; GAN training (`train_gan.py`)
- [x] PR12: `configs/train.yaml` — unified training config; `--config` flag for both training scripts
