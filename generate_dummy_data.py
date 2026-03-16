import os
import numpy as np
from PIL import Image

def create_dummy_data(base_dir='dummy_data', num_samples=5):
    hr_dir = os.path.join(base_dir, 'hr')
    lr_dir = os.path.join(base_dir, 'lr')
    mask_dir = os.path.join(base_dir, 'mask')
    
    for d in [hr_dir, lr_dir, mask_dir]:
        os.makedirs(d, exist_ok=True)
        
    for i in range(num_samples):
        # Create 256x256 HR image
        hr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        # Create 64x64 LR image
        lr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Create 256x256 mask
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        
        Image.fromarray(hr).save(os.path.join(hr_dir, f'img_{i}.png'))
        Image.fromarray(lr).save(os.path.join(lr_dir, f'img_{i}.png'))
        Image.fromarray(mask).save(os.path.join(mask_dir, f'img_{i}.png'))
        
    print(f"Created {num_samples} dummy samples in {base_dir}")

if __name__ == '__main__':
    create_dummy_data()
