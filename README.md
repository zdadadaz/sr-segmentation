# 🐾 SR Segmentation Project
### Super Resolution + Semantic Segmentation for Hair/Fur Enhancement

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a state-of-the-art Super-Resolution (SR) pipeline that leverages **Spatial Feature Transform (SFT)**. By using semantic segmentation masks as guidance, the model specifically enhances hair and fur textures in images of animals and humans, preserving fine details while scaling.

---

## 🚀 Quick Start in 3 Steps

1. **Install**: `pip install -r requirements.txt`
2. **Prepare**: `python prepare_dataset.py --src_dir my_photos --output_dir data/v1`
3. **Train**: `python train.py --hr_dir data/v1/hr --lr_dir data/v1/lr --mask_dir data/v1/mask`

---

## 📂 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Workflow](#-workflow)
  - [1. Dataset Generation](#1-dataset-generation)
  - [2. Training](#2-training)
  - [3. Testing and Inference](#3-testing-and-inference)
- [Usage Examples](#-usage-examples)

---

## 📂 Project Structure

```
sr-segmentation/
├── src/
│   ├── pipeline.py          # Main segmentation pipeline
│   ├── realesrgan_arch.py   # Seg-Guided RRDBNet architecture
│   ├── dataset.py           # Training dataset loader
│   ├── dataset_generator.py # Auto-labeling tool
│   └── sr_integration.py    # SR loss and integration
├── train.py                 # Main training script
├── prepare_dataset.py       # Auto-labeling & dataset preparation
├── generate_dummy_data.py   # Quick-start data generator
├── test_pipeline.py         # Component testing
├── test_real_images.py      # Inference testing with real images
├── configs/                 # Configuration files
├── models/                  # Pre-trained model weights (yolov8n.pt, etc.)
└── data/                    # Dataset directory
```

---

## ⚙️ Installation

```bash
# Clone the repository and navigate to it
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📥 Model Weights

Before running the pipeline, you need to download the pre-trained weights and place them in the `models/` directory.

| Model | Weight File | Purpose | Download Link |
|-------|------------|---------|---------------|
| YOLOv8n | `yolov8n.pt` | Detection | [Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) |
| SAM (ViT-H) | `sam_vit_h.pth` | Segmentation | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| BiSeNet | `face_parsing.pth` | Face Parsing | [Download (GDrive)](https://docs.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) |

Alternatively, you can use the provided script to download the BiSeNet weights:
```bash
python download_gdrive.py
```

---

## 🔄 Workflow

### 1. Dataset Generation

There are two ways to prepare your dataset:

#### A. Generate Dummy Data (Quick Start)
Use this if you want to test the training pipeline quickly without real images.
```bash
python generate_dummy_data.py
```
This creates a `dummy_data/` folder with random HR, LR, and Mask images.

#### B. Auto-Labeling Real Images
If you have a collection of high-resolution images, use the `prepare_dataset.py` script to automatically create the training triplets (HR, LR, and Mask).

```bash
python prepare_dataset.py \
  --src_dir path/to/your/hr_images \
  --output_dir data/my_dataset \
  --scale 4
```

This script will:
1.  **Detect & Segment**: Run the full pipeline to find hair/fur and exclude faces/skin.
2.  **Generate HR**: Resize images to be perfectly divisible by the scale factor.
3.  **Generate LR**: Downsample the HR images (using Bicubic interpolation).
4.  **Save Triplets**: Organize them into `hr/`, `lr/`, and `mask/` subdirectories.
5.  **Split**: Generate a `split.json` for training/validation.

### 2. Training

Train the `SegGuidedRRDBNet` using the generated dataset. The model uses `SegAwareLoss` to prioritize hair/fur texture recovery.

```bash
python train.py \
  --hr_dir data/my_dataset/hr \
  --lr_dir data/my_dataset/lr \
  --mask_dir data/my_dataset/mask \
  --epochs 50 \
  --batch_size 4 \
  --save_dir experiments/v1
```

**Key Arguments:**
- `--hair_weight`: Loss weight for hair/fur regions (default: 2.0).
- `--patch_size`: HR crop size (default: 256).
- `--scale`: SR scaling factor (default: 4).

### 3. Hair Segmentation Pre-training (Optional)

If you want to train a dedicated, lightweight hair extraction model (based on MobileNetV3 + DeepLabV3) instead of relying on the full PR2-4 pipeline:

```bash
python train_segment.py \
  --images_dir data/my_dataset/hr \
  --masks_dir data/my_dataset/mask \
  --epochs 20 \
  --batch_size 8 \
  --save_dir models/segmentation
```
This produces a fast hair segmentation model specifically tailored to your domain.

### 4. Testing and Inference

#### A. Component Testing
Run the test suite to verify all modules (YOLO, SAM, BiSeNet) are working correctly with fallbacks.
```bash
python test_pipeline.py
```

#### B. Inference on Real Images
Run the pipeline on sample images and visualize the segmentation results.
```bash
python test_real_images.py
```
This will download sample images (cat, dog, person) and save the results in `output/test_results/`.

---

## 💡 Usage Examples

Run the segmentation pipeline programmatically:

```python
import cv2
from src.pipeline import SegmentationPipeline
from utils.visualization import visualize_mask

# Initialize
pipeline = SegmentationPipeline()

# Load image
img = cv2.imread("test_images/cat.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Segment hair/fur
result = pipeline.segment(img)

# Access masks
hair_mask = result.final_mask
soft_mask = result.get_soft_mask(sigma=3.0)

# Visualize
vis = visualize_mask(img, hair_mask)
cv2.imwrite("output.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
```

---

## PR Progress (Current Status)

- [x] PR1: Project scaffold + inference pipeline skeleton
- [x] PR2: SpeciesNet + SAM pixel mask
- [x] PR3: BiSeNet face parsing
- [x] PR4: Mask merging logic + full pipeline
- [x] PR5: Dataset auto-labeling pipeline
- [x] PR6: SR model SFT integration & Training Script
- [x] PR7: Texture classifier fallback
