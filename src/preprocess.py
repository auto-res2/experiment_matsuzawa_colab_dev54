#!/usr/bin/env python3
"""
CoCoDiff Data Preprocessing Module
=================================
Handles synthetic dataset generation for CoCoDiff experiments.
Creates ImageNet-style synthetic data with multiple texture patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random

class ToyImageNet(Dataset):
    """32×32 synthetic images in 3 texture patterns → simulates ImageNet-256."""
    
    def __init__(self, size: int = 4096, seed: int = 42):
        super().__init__()
        self.size = size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.x, self.y = self._make_dataset()

    def _make_dataset(self):
        """Generate synthetic images with different texture patterns."""
        imgs, labels = [], []
        for i in range(self.size):
            mode = i % 3
            if mode == 0:  # low-freq smooth
                img = np.random.randn(3, 32, 32) * 0.1
            elif mode == 1:  # checkerboard high-freq
                base = np.indices((32, 32)).sum(axis=0) % 2
                img = np.tile(base, (3, 1, 1)).astype(float)
            else:  # random noise
                img = np.random.randn(3, 32, 32)
            imgs.append(img.astype(np.float32))
            labels.append(mode)
        return torch.tensor(imgs), torch.tensor(labels)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_synthetic_prompts(num_prompts: int = 5000) -> List[str]:
    """Generate COCO-style prompts with repeating patterns."""
    return [f"A photo of class {i%100} in style {i%5}" for i in range(num_prompts)]

def get_data_loaders(batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, List[str]]:
    """Create data loaders for training and evaluation."""
    real_dataset = ToyImageNet(size=4096)
    real_loader = DataLoader(
        real_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    prompts = create_synthetic_prompts()
    
    return real_loader, prompts

if __name__ == "__main__":
    print("Testing CoCoDiff preprocessing module...")
    
    real_loader, prompts = get_data_loaders()
    print(f"Created real data loader with {len(real_loader.dataset)} samples")
    print(f"Generated {len(prompts)} synthetic prompts")
    
    batch = next(iter(real_loader))
    imgs, labels = batch
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
    print(f"Image range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    
    print("Preprocessing module test completed successfully!")
