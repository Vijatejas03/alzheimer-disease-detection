"""
PyTorch Dataset and DataLoader loader module.
Handles multi-class MRI dataset loading, stratified splitting, and class imbalance weight computation.
"""

import os
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.data.validation import CLASSES, CLASS_TO_IDX, VALID_EXTENSIONS
from src.data.augmentation import get_training_transforms, get_validation_transforms
from src.utils.logger import logger


class AlzheimerMRIDataset(Dataset):
    """
    PyTorch Dataset for 4-class Alzheimer MRI scans.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_counts: Dict[str, int] = {cls_name: 0 for cls_name in CLASSES}
        
        self._load_samples()
        
    def _load_samples(self):
        """Scan directory and index valid image files across all 4 classes."""
        if not os.path.exists(self.root_dir):
            return
            
        for cls_name in CLASSES:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            label = CLASS_TO_IDX[cls_name]
            files = sorted(os.listdir(cls_dir))
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in VALID_EXTENSIONS:
                    file_path = os.path.join(cls_dir, f)
                    self.samples.append((file_path, label))
                    self.class_counts[cls_name] += 1
                    
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            # Fallback to zero image
            image = Image.new('RGB', (224, 224), color=0)
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


class TransformedSubset(Dataset):
    """
    Dataset wrapper allowing distinct transforms on train/val/test subsets.
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=0)
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def compute_class_weights(class_counts: Dict[str, int]) -> torch.Tensor:
    """
    Compute balanced class weights for Cross-Entropy Loss to handle class imbalance.
    
    Formula: weight[c] = total_samples / (num_classes * count[c])
    """
    counts = [max(1, class_counts.get(cls_name, 0)) for cls_name in CLASSES]
    total = sum(counts)
    num_classes = len(CLASSES)
    
    weights = [total / (num_classes * c) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Normalize so mean weight is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def create_data_loaders(
    dataset_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], torch.Tensor, Dict[str, int]]:
    """
    Create stratified PyTorch DataLoaders for training, validation, and testing.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights_tensor, class_counts_dict)
    """
    base_dataset = AlzheimerMRIDataset(root_dir=dataset_dir)
    total_samples = len(base_dataset)
    
    if total_samples == 0:
        logger.warning(f"No samples found in dataset directory: {dataset_dir}")
        return None, None, None, torch.ones(len(CLASSES)), base_dataset.class_counts
        
    labels = [sample[1] for sample in base_dataset.samples]
    indices = list(range(total_samples))
    
    # Check if all classes have at least 2 samples for stratified split
    can_stratify = all(base_dataset.class_counts[cls_name] >= 2 for cls_name in CLASSES if base_dataset.class_counts[cls_name] > 0)
    stratify_arg = labels if can_stratify else None
    
    # Split into Train+Val vs Test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_seed,
        stratify=stratify_arg
    )
    
    # Stratify train vs val
    if can_stratify:
        train_val_labels = [labels[i] for i in train_val_idx]
        stratify_train_val = train_val_labels
    else:
        stratify_train_val = None
        
    val_rel_size = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_rel_size,
        random_state=random_seed,
        stratify=stratify_train_val
    )
    
    # Create subsets with distinct transforms
    train_transform = get_training_transforms()
    val_test_transform = get_validation_transforms()
    
    train_subset = TransformedSubset(Subset(base_dataset, train_idx), transform=train_transform)
    val_subset = TransformedSubset(Subset(base_dataset, val_idx), transform=val_test_transform)
    test_subset = TransformedSubset(Subset(base_dataset, test_idx), transform=val_test_transform)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Compute class weights based on training subset
    train_counts = {cls_name: 0 for cls_name in CLASSES}
    for idx in train_idx:
        lbl = base_dataset.samples[idx][1]
        train_counts[CLASSES[lbl]] += 1
        
    class_weights = compute_class_weights(train_counts)
    
    logger.info(f"DataLoaders created - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, class_weights, base_dataset.class_counts
