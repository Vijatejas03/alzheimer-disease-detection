"""
PyTorch Dataset and DataLoader loader module.
Handles multi-class MRI dataset loading, stratified splitting, split CSV manifests,
and class imbalance weight computation.
"""

import os
import csv
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.data.validation import CLASSES, CLASS_TO_IDX, VALID_EXTENSIONS
from src.data.augmentation import get_training_transforms, get_validation_transforms
from src.utils.logger import logger

# Canonical mapping for CSV manifests
CANONICAL_NAME_TO_IDX = {
    'Non-Demented': 0,
    'Very Mild Demented': 1,
    'Mild Demented': 2,
    'Moderate Demented': 3,
    'Non_Demented': 0,
    'Very_Mild_Demented': 1,
    'Mild_Demented': 2,
    'Moderate_Demented': 3,
}


class AlzheimerMRISplitDataset(Dataset):
    """
    PyTorch Dataset loading strictly from a split manifest CSV (train.csv, validation.csv, test.csv).
    Guarantees that split assignment is fixed, verifiable, and free of data leakage.
    Includes in-memory PIL caching to eliminate disk I/O overhead on repeated epochs.
    """
    def __init__(self, csv_file: str, project_root: Optional[str] = None, transform=None):
        self.csv_file = csv_file
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_counts: Dict[str, int] = {cls_name: 0 for cls_name in CLASSES}
        self._cache: Dict[str, Image.Image] = {}
        
        self._load_from_csv()
        
    def _load_from_csv(self):
        if not os.path.exists(self.csv_file):
            logger.warning(f"Split CSV file not found: {self.csv_file}")
            return
            
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row['filepath']
                abs_path = os.path.join(self.project_root, rel_path) if not os.path.isabs(rel_path) else rel_path
                
                c_name = row['canonical_class']
                label = CANONICAL_NAME_TO_IDX.get(c_name, 0)
                
                self.samples.append((abs_path, label))
                # Count under standard canonical folder name
                std_cls = CLASSES[label]
                self.class_counts[std_cls] += 1
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        if img_path in self._cache:
            image = self._cache[img_path]
        else:
            try:
                # Grayscale MRI converted to RGB (3-channel) for pretrained CNN backbones
                image = Image.open(img_path).convert('RGB')
                self._cache[img_path] = image
            except Exception as e:
                logger.error(f"Error reading image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color=0)
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


class AlzheimerMRIDataset(Dataset):
    """
    PyTorch Dataset for 4-class Alzheimer MRI scans loading directly from directory structure.
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


def create_split_data_loaders(
    splits_dir: str = "reports/splits",
    project_root: Optional[str] = None,
    batch_size: int = 32,
    pin_memory: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, Dict[str, int]]:
    """
    Create PyTorch DataLoaders strictly from the generated split CSV manifests.
    Ensures training receives augmentation while validation and test receive deterministic preprocessing.
    """
    root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_csv = os.path.join(root, splits_dir, "train.csv") if not os.path.isabs(splits_dir) else os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(root, splits_dir, "validation.csv") if not os.path.isabs(splits_dir) else os.path.join(splits_dir, "validation.csv")
    test_csv = os.path.join(root, splits_dir, "test.csv") if not os.path.isabs(splits_dir) else os.path.join(splits_dir, "test.csv")
    
    train_transform = get_training_transforms()
    val_test_transform = get_validation_transforms()
    
    train_dataset = AlzheimerMRISplitDataset(train_csv, project_root=root, transform=train_transform)
    val_dataset = AlzheimerMRISplitDataset(val_csv, project_root=root, transform=val_test_transform)
    test_dataset = AlzheimerMRISplitDataset(test_csv, project_root=root, transform=val_test_transform)
    
    # Class weights computed strictly from training split counts
    class_weights = compute_class_weights(train_dataset.class_counts)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    logger.info(f"Split DataLoaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, class_weights, train_dataset.class_counts
