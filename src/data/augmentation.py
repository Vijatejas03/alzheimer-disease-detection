"""
Medical data augmentation pipeline for Brain MRI scans.
Applies domain-safe transformations that preserve anatomical validity while mitigating overfitting.
"""

from typing import Tuple
from torchvision import transforms
from src.data.preprocessing import DEFAULT_IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_training_transforms(img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> transforms.Compose:
    """
    Data augmentation pipeline for training split.
    
    Included transforms:
    - Resize to target size.
    - Random Horizontal Flip: Preserves bilateral axial symmetry.
    - Random Rotation (-10 to +10 deg): Simulates slight patient head positioning variations.
    - Random Affine (translation +/-5%, scaling 0.95 to 1.05): Simulates slight voxel shift.
    - ColorJitter (brightness 0.1, contrast 0.1): Simulates scanner intensity variations.
    - Tensor conversion & ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_validation_transforms(img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> transforms.Compose:
    """
    Deterministic preprocessing transforms for validation and testing splits.
    """
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
