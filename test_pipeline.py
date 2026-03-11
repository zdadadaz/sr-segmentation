"""
Tests for PR1-PR4: Segmentation pipeline end-to-end
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_data_structures():
    """Test SegmentationResult and BBox data structures"""
    from src.pipeline import SegmentationResult, BBox
    
    # Test BBox
    bbox = BBox(x1=10, y1=20, x2=100, y2=200, label='cat', confidence=0.9)
    assert bbox.xyxy == (10, 20, 100, 200)
    assert bbox.xywh == (10, 20, 90, 180)
    assert bbox.center == (55.0, 110.0)
    print("✅ BBox data structure works")
    
    # Test SegmentationResult
    result = SegmentationResult(original_shape=(480, 640))
    assert result.original_shape == (480, 640)
    assert result.animal_bboxes == []
    assert result.person_bboxes == []
    
    # Test final_mask with no masks
    final = result.final_mask
    assert final.shape == (480, 640)
    assert final.sum() == 0
    print("✅ SegmentationResult data structure works")
    
    # Test final_mask with masks
    result.animal_mask = np.zeros((480, 640), dtype=np.uint8)
    result.animal_mask[100:200, 100:200] = 1
    result.human_hair_mask = np.zeros((480, 640), dtype=np.uint8)
    result.human_hair_mask[50:100, 200:300] = 1
    result.face_mask = np.zeros((480, 640), dtype=np.uint8)
    result.face_mask[150:180, 130:170] = 1  # Small face region inside animal
    
    final = result.final_mask
    assert final.shape == (480, 640)
    assert final[110, 110] == 1  # Animal region (not face)
    assert final[160, 150] == 0  # Face region excluded
    assert final[60, 210] == 1   # Hair region
    print("✅ Mask merging logic works")
    
    # Test soft mask
    soft = result.get_soft_mask(sigma=3.0)
    assert soft.shape == (480, 640)
    assert soft.max() <= 1.0
    assert soft.min() >= 0.0
    print("✅ Soft mask generation works")
    
    # Test to_dict
    d = result.to_dict()
    assert 'original_shape' in d
    assert 'processing_time_ms' in d
    print("✅ Serialization works")


def test_config():
    """Test config loading"""
    from utils.config_parser import load_config, get_default_config, Config
    
    # Test default config
    config = get_default_config()
    assert 'device' in config
    assert 'models' in config
    assert 'sam' in config
    print("✅ Default config works")
    
    # Test Config class
    cfg = Config(config)
    assert cfg.get('device') is not None
    assert cfg.get('sam.model_type') == 'vit_h'
    assert cfg.get('nonexistent', 'default') == 'default'
    print("✅ Config class works")
    
    # Test loading from YAML
    config = load_config('configs/default.yaml')
    assert isinstance(config, dict)
    print("✅ YAML config loading works")


def test_image_utils():
    """Test image utilities"""
    from utils.image_utils import preprocess_image, pad_image_to_square, crop_bbox
    
    # Create a dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test preprocess
    tensor, original_shape = preprocess_image(image, normalize=True)
    assert tensor.shape == (3, 480, 640)
    assert tensor.max() <= 1.0
    assert original_shape == (480, 640)
    print("✅ Image preprocessing works")
    
    # Test pad to square
    padded, offset = pad_image_to_square(image)
    assert padded.shape[0] == padded.shape[1]
    assert padded.shape[0] == 640  # Max of H, W
    print("✅ Pad to square works")
    
    # Test crop bbox
    cropped = crop_bbox(image, (10, 20, 100, 200))
    assert cropped.shape == (180, 90, 3)
    print("✅ Crop bbox works")


def test_visualization():
    """Test visualization utilities"""
    from utils.visualization import visualize_mask, draw_bboxes, create_side_by_side
    
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    
    # Test mask overlay
    vis = visualize_mask(image, mask, alpha=0.5)
    assert vis.shape == (480, 640, 3)
    print("✅ Mask visualization works")
    
    # Test bbox drawing
    bboxes = [(10, 20, 100, 200), (200, 100, 400, 300)]
    labels = ['cat', 'dog']
    result = draw_bboxes(image, bboxes, labels=labels)
    assert result.shape == (480, 640, 3)
    print("✅ BBox drawing works")
    
    # Test side by side
    sbs = create_side_by_side(image, image)
    assert sbs.shape[0] == 480
    assert sbs.shape[1] == 1280
    print("✅ Side-by-side comparison works")


def test_mask_merger():
    """Test mask merger (PR4)"""
    from src.mask_merger import MaskMerger
    
    merger = MaskMerger(gaussian_sigma=3.0, min_hair_area=50)
    
    h, w = 480, 640
    
    # Create test masks
    animal = np.zeros((h, w), dtype=np.uint8)
    animal[100:200, 100:200] = 1
    
    hair = np.zeros((h, w), dtype=np.uint8)
    hair[50:100, 200:300] = 1
    
    face = np.zeros((h, w), dtype=np.uint8)
    face[60:90, 220:280] = 1  # Face inside hair region
    
    skin = np.zeros((h, w), dtype=np.uint8)
    skin[150:180, 130:170] = 1  # Skin inside animal region
    
    # Test merge
    result = merger.merge(animal, hair, face, skin, (h, w))
    
    assert 'final_mask' in result
    assert result['final_mask'].shape == (h, w)
    
    # Animal region should be in final (except where skin is)
    assert result['final_mask'][110, 110] == 1  # Animal, not skin
    assert result['final_mask'][160, 150] == 0  # Skin excluded
    
    # Hair region should be in final (except where face is)
    assert result['final_mask'][55, 210] == 1   # Hair, not face
    assert result['final_mask'][70, 250] == 0   # Face excluded
    
    print("✅ Mask merger works correctly")
    
    # Test soft mask
    soft = merger.create_soft_mask(result['final_mask'])
    assert soft.shape == (h, w)
    assert 0 <= soft.min() <= soft.max() <= 1.0
    print("✅ Soft mask generation works")
    
    # Test blending
    sr_hair = np.random.rand(h, w, 3).astype(np.float32)
    sr_general = np.random.rand(h, w, 3).astype(np.float32)
    blended = merger.blend_sr_outputs(None, sr_hair, sr_general, soft)
    assert blended.shape == (h, w, 3)
    print("✅ SR blending works")


def test_speciesnet_detector():
    """Test SpeciesNet detector (PR2)"""
    from src.speciesnet import SpeciesNetDetector
    
    # Test initialization (without actual model)
    detector = SpeciesNetDetector(config={'confidence_threshold': 0.5})
    
    # Test with dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect_animals(image)
    
    # Should return a list (empty or with detections depending on fallback)
    assert isinstance(detections, list)
    print("✅ SpeciesNet detector initializes and runs")
    
    # Test filtering
    fake_detections = [
        ([10, 20, 100, 200], 'cat', 0.9),
        ([200, 100, 400, 300], 'bird', 0.8),
        ([50, 50, 150, 150], 'dog', 0.7),
    ]
    filtered = detector.filter_furry_animals(fake_detections)
    assert len(filtered) == 2  # cat and dog, not bird
    assert filtered[0][1] == 'cat'
    assert filtered[1][1] == 'dog'
    print("✅ Furry animal filtering works")


def test_sam_generator():
    """Test SAM mask generator (PR2)"""
    from src.sam import SAMMaskGenerator
    
    # Test initialization (will use fallback)
    sam = SAMMaskGenerator(device='cpu')
    
    # Test mask from bbox
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = sam.generate_mask_from_bbox(image, [100, 100, 300, 300])
    
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    print("✅ SAM mask generation works (fallback)")
    
    # Test combine masks
    masks = [
        np.zeros((480, 640), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.uint8),
    ]
    masks[0][100:200, 100:200] = 1
    masks[1][150:250, 200:300] = 1
    
    combined = sam.combine_masks(masks, (480, 640))
    assert combined.shape == (480, 640)
    assert combined[110, 110] == 1
    assert combined[160, 210] == 1
    print("✅ SAM mask combining works")
    
    # Test postprocess
    noisy_mask = np.zeros((480, 640), dtype=np.uint8)
    noisy_mask[100:200, 100:200] = 1
    noisy_mask[5, 5] = 1  # Small noise
    
    cleaned = sam.postprocess_mask(noisy_mask, min_area=50)
    assert cleaned[110, 110] == 1  # Large region preserved
    assert cleaned[5, 5] == 0     # Small noise removed
    print("✅ SAM postprocessing works")


def test_bisenet_parser():
    """Test BiSeNet face parser (PR3)"""
    from src.bisenet import BiSeNetParser, PersonDetector
    
    # Test initialization (will use fallback)
    parser = BiSeNetParser(device='cpu')
    
    # Test parsing
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = parser.parse(image)
    
    assert 'hair' in result
    assert 'face' in result
    assert 'skin' in result
    assert result['hair'].shape == (480, 640)
    print("✅ BiSeNet parsing works (fallback)")
    
    # Test with crop
    result_cropped = parser.parse(image, crop_box=[100, 100, 400, 400])
    assert result_cropped['hair'].shape == (480, 640)
    print("✅ BiSeNet parsing with crop works")
    
    # Test person detector
    detector = PersonDetector(device='cpu')
    persons = detector.detect(image)
    assert isinstance(persons, list)
    print("✅ Person detector works (fallback)")


def test_pipeline_e2e():
    """Test full pipeline end-to-end (PR4)"""
    from src.pipeline import SegmentationPipeline
    
    # Initialize pipeline
    pipeline = SegmentationPipeline(config_path='configs/default.yaml')
    
    # Create test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run full pipeline
    result = pipeline.segment(image)
    
    assert result.original_shape == (480, 640)
    assert result.hair_fur_mask is not None
    assert result.hair_fur_mask.shape == (480, 640)
    assert result.processing_time_ms > 0
    assert isinstance(result.model_versions, dict)
    print("✅ Full pipeline works end-to-end")
    
    # Test animals-only
    result_animals = pipeline.segment_animals_only(image)
    assert result_animals.original_shape == (480, 640)
    print("✅ Animals-only pipeline works")
    
    # Test humans-only
    result_humans = pipeline.segment_humans_only(image)
    assert result_humans.original_shape == (480, 640)
    print("✅ Humans-only pipeline works")
    
    # Test to_dict
    d = result.to_dict()
    assert isinstance(d, dict)
    print("✅ Result serialization works")


def test_dataset_generator():
    """Test dataset generator (PR5)"""
    from src.dataset_generator import DatasetGenerator, QualityChecker
    from src.pipeline import SegmentationPipeline
    
    pipeline = SegmentationPipeline(config_path='configs/default.yaml')
    generator = DatasetGenerator(pipeline, output_dir='/tmp/test_dataset')
    
    assert generator.images_dir.exists()
    assert generator.masks_dir.exists()
    print("✅ Dataset generator initializes correctly")
    
    # Test statistics
    stats = generator.get_statistics()
    assert 'total_images' in stats
    print("✅ Dataset statistics work")
    
    # Test quality checker
    checker = QualityChecker('/tmp/test_dataset')
    pending = checker.get_pending_images()
    assert isinstance(pending, list)
    print("✅ Quality checker works")


def test_sr_integration():
    """Test SR integration (PR6) - structure only"""
    import torch
    from src.sr_integration import SFTBlock, SegAwareLoss
    
    # Test SFT block
    sft = SFTBlock(in_channels=64, seg_channels=2)
    features = torch.randn(1, 64, 32, 32)
    seg_map = torch.randn(1, 2, 32, 32)
    output = sft(features, seg_map)
    assert output.shape == (1, 64, 32, 32)
    print("✅ SFT block works")


if __name__ == '__main__':
    print("=" * 60)
    print("Running SR Segmentation Tests")
    print("=" * 60)
    
    tests = [
        ("Data Structures (PR1)", test_data_structures),
        ("Config (PR1)", test_config),
        ("Image Utils (PR1)", test_image_utils),
        ("Visualization (PR1)", test_visualization),
        ("SpeciesNet Detector (PR2)", test_speciesnet_detector),
        ("SAM Generator (PR2)", test_sam_generator),
        ("BiSeNet Parser (PR3)", test_bisenet_parser),
        ("Mask Merger (PR4)", test_mask_merger),
        ("Pipeline E2E (PR4)", test_pipeline_e2e),
        ("Dataset Generator (PR5)", test_dataset_generator),
        ("SR Integration (PR6)", test_sr_integration),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"❌ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  ❌ {name}: {err}")
    print("=" * 60)
