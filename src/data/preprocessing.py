"""
MRI image preprocessing and transform utilities.
Standardizes resolution, normalization, and tensor conversions for deep learning models.
"""

from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Standard input dimensions for MobileNet, EfficientNet, and ResNet backbones
DEFAULT_IMG_SIZE = (224, 224)

# Standard ImageNet normalization parameters used for transfer learning backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_transforms(img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> transforms.Compose:
    """
    Standard preprocessing transforms for inference and validation.
    
    1. Resizes to target dimensions (224, 224).
    2. Converts PIL Image to PyTorch Tensor in range [0.0, 1.0].
    3. Normalizes using standard mean and standard deviation.
    """
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def crop_brain_contour(image: Image.Image, threshold: int = 15) -> Image.Image:
    """
    Crop empty black margins around brain MRI slice to focus on neuroanatomical tissue.
    
    Args:
        image: PIL RGB or Grayscale image.
        threshold: Pixel intensity threshold to distinguish brain from black background.
        
    Returns:
        Cropped PIL Image (or original if contour detection fails).
    """
    try:
        gray = image.convert('L')
        np_gray = np.array(gray)
        
        # Mask where intensity > threshold
        mask = np_gray > threshold
        if not np.any(mask):
            return image
            
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add small margin
        h, w = np_gray.shape
        rmin = max(0, rmin - 2)
        rmax = min(h - 1, rmax + 2)
        cmin = max(0, cmin - 2)
        cmax = min(w - 1, cmax + 2)
        
        return image.crop((cmin, rmin, cmax, rmax))
    except Exception:
        return image


def preprocess_single_image(
    image_input: Union[str, Image.Image],
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    device: str = 'cpu',
    crop_contour: bool = False
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Process a single input image for direct model inference.
    
    Args:
        image_input: Path to image or PIL Image object.
        img_size: Target dimensions (default 224x224).
        device: Device to place tensor on ('cpu' or 'cuda').
        crop_contour: Whether to crop black background margins.
        
    Returns:
        Tuple of (preprocessed_tensor of shape [1, 3, H, W], processed_pil_image).
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    else:
        pil_img = image_input.convert('RGB')
        
    if crop_contour:
        pil_img = crop_brain_contour(pil_img)
        
    transform = get_preprocessing_transforms(img_size=img_size)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    
    return tensor, pil_img


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized PyTorch tensor back to a uint8 RGB numpy array [0, 255] for display.
    
    Args:
        tensor: Tensor of shape [3, H, W] or [1, 3, H, W].
        
    Returns:
        Numpy array of shape (H, W, 3) in range [0, 255] uint8.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
        
    tensor_cpu = tensor.detach().cpu().clone()
    for t, m, s in zip(tensor_cpu, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
        
    tensor_cpu = torch.clamp(tensor_cpu, 0.0, 1.0)
    np_img = tensor_cpu.permute(1, 2, 0).numpy()
    return (np_img * 255.0).astype(np.uint8)
