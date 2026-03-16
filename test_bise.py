import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bisenet import BiSeNetParser

def test_bise():
    print("Testing BiSeNet Parser initialization...")
    parser = BiSeNetParser(model_path='models/face_parsing.pth', device='cpu')
    
    if parser.model is None:
        print("❌ Model failed to load.")
        return
        
    print("✅ Model loaded successfully!")
    
    # Create fake RGB image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Running inference...")
    res = parser.parse(img)
    
    print(f"Result keys: {res.keys()}")
    for k, v in res.items():
        print(f"  {k}: shape {v.shape}, max {v.max()}, min {v.min()}")
        
    print("✅ Inference runs successfully!")

if __name__ == '__main__':
    test_bise()
