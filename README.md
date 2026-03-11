# SR Segmentation Project
Super Resolution + Semantic Segmentation for Hair/Fur Enhancement

## Project Structure

```
sr-segmentation/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main segmentation pipeline
│   ├── speciesnet.py        # Animal detection (PR2)
│   ├── sam.py               # SAM mask generation (PR2)
│   ├── bisenet.py           # Face parsing (PR3)
│   ├── mask_merger.py       # Mask merging logic (PR4)
│   ├── dataset_generator.py # Auto-labeling (PR5)
│   └── sr_integration.py    # SR model integration (PR6)
├── models/                  # Model weights
├── configs/                 # Configuration files
├── data/                    # Dataset directory
├── test_images/            # Test images
├── output/                  # Output results
├── utils/
│   ├── __init__.py
│   ├── image_utils.py      # Image loading, preprocessing
│   ├── visualization.py     # Mask visualization
│   └── config_parser.py     # Config parsing
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.pipeline import SegmentationPipeline
from utils.image_utils import load_image
from utils.visualization import visualize_masks

# Initialize pipeline
pipeline = SegmentationPipeline()

# Run segmentation
image = load_image("test_images/sample.jpg")
result = pipeline.segment(image)

# Visualize
visualize_masks(image, result)
```

## PR Progress

- [x] PR1: Project scaffold + inference pipeline skeleton
- [x] PR2: SpeciesNet + SAM pixel mask
- [x] PR3: BiSeNet face parsing
- [x] PR4: Mask merging logic + full pipeline
- [x] PR5: Dataset auto-labeling pipeline
- [x] PR6: SR model SFT integration (code pipeline, no training)
- [x] PR7: Texture classifier fallback
