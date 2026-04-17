
import torch
import torch.nn as nn
from src.loss import SegAwareLoss

def test_binary_backward_compatibility():
    print("Testing binary backward compatibility...")
    criterion = SegAwareLoss(hair_weight=2.0, other_weight=1.0)
    
    sr = torch.ones((1, 3, 10, 10)) * 0.5
    hr = torch.ones((1, 3, 10, 10)) * 0.0
    
    # Binary mask: all hair (class 1)
    mask = torch.ones((1, 1, 10, 10))
    
    loss = criterion(sr, hr, mask)
    # expected: 2.0 * L1(0.5, 0.0) = 2.0 * 0.5 = 1.0
    print(f"Loss (All Hair): {loss.item():.4f} (Expected: 1.0000)")
    
    # Binary mask: all other (class 0)
    mask = torch.zeros((1, 1, 10, 10))
    loss = criterion(sr, hr, mask)
    # expected: 1.0 * L1(0.5, 0.0) = 1.0 * 0.5 = 0.5
    print(f"Loss (All Other): {loss.item():.4f} (Expected: 0.5000)")

def test_multiclass_index():
    print("\nTesting multi-class with index map...")
    # 3 classes with weights [1.0, 2.0, 5.0]
    weights = [1.0, 2.0, 5.0]
    criterion = SegAwareLoss(class_weights=weights)
    
    sr = torch.ones((1, 3, 10, 10)) * 0.5
    hr = torch.ones((1, 3, 10, 10)) * 0.0
    
    # Mask with class index 2 everywhere
    mask = torch.ones((1, 1, 10, 10)) * 2.0
    loss = criterion(sr, hr, mask)
    # Each pixel has m2=1, m0=0, m1=0
    # Mean loss across the whole image for class 2: 
    # self.pixel_loss(sr*m2, hr*m2) = mean(abs(0.5*1 - 0.0*1)) = 0.5
    # Total loss = 5.0 * 0.5 = 2.5
    print(f"Loss (All Class 2): {loss.item():.4f} (Expected: 2.5000)")

def test_multiclass_onehot():
    print("\nTesting multi-class with one-hot map...")
    weights = [1.0, 2.0, 5.0]
    criterion = SegAwareLoss(class_weights=weights)
    
    sr = torch.ones((1, 3, 10, 10)) * 0.5
    hr = torch.ones((1, 3, 10, 10)) * 0.0
    
    # One-hot mask for 3 classes, let's say class 1 is active
    mask = torch.zeros((1, 3, 10, 10))
    mask[:, 1, :, :] = 1.0
    
    loss = criterion(sr, hr, mask)
    # Expected: 2.0 * 0.5 = 1.0
    print(f"Loss (All Class 1 One-Hot): {loss.item():.4f} (Expected: 1.0000)")

if __name__ == "__main__":
    test_binary_backward_compatibility()
    test_multiclass_index()
    test_multiclass_onehot()
