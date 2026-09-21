"""
Model factory for building, loading, and inspecting candidate architectures.
Provides unified model instantiation, checkpoint loading, and parameter profiling.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from src.models.architectures import (
    build_mobilenet_v2,
    build_efficientnet_b0,
    build_resnet18,
    build_resnet50,
    get_target_convolutional_layer
)
from src.utils.logger import logger

# Supported model registry with human-readable labels
MODEL_REGISTRY = {
    'mobilenet_v2': {
        'name': 'MobileNetV2',
        'builder': build_mobilenet_v2,
        'family': 'Inverted Residual / Lightweight',
        'description': 'Optimized for high-speed, parameter-efficient inference.'
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'builder': build_efficientnet_b0,
        'family': 'Compound Scaled CNN',
        'description': 'Balanced scaling across depth, width, and input resolution.'
    },
    'resnet18': {
        'name': 'ResNet-18',
        'builder': build_resnet18,
        'family': 'Residual Network (18 layers)',
        'description': 'Compact residual architecture with skip connections.'
    },
    'resnet50': {
        'name': 'ResNet-50',
        'builder': build_resnet50,
        'family': 'Residual Network (50 layers)',
        'description': 'Deep residual network with bottleneck blocks for rich representations.'
    }
}


def build_model(model_name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Instantiate a candidate architecture by name.
    
    Args:
        model_name: Identifier key in MODEL_REGISTRY.
        num_classes: Number of output stages (default 4).
        pretrained: Whether to initialize with ImageNet pre-trained backbone weights.
    """
    key = model_name.lower().replace('-', '_')
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture '{model_name}'. Supported: {list(MODEL_REGISTRY.keys())}")
        
    builder_fn = MODEL_REGISTRY[key]['builder']
    model = builder_fn(num_classes=num_classes, pretrained=pretrained)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_trained_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int = 4,
    device: str = 'cpu'
) -> Tuple[Optional[nn.Module], str]:
    """
    Load a trained model and restore weights from checkpoint.
    
    Returns:
        Tuple of (model, status_message)
    """
    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint file not found: {checkpoint_path}"
        
    try:
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint is dict with state_dict or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        logger.info(f"Loaded checkpoint for {model_name} from {checkpoint_path}")
        return model, "Model loaded successfully."
    except Exception as e:
        logger.error(f"Failed to load checkpoint for {model_name}: {e}")
        return None, f"Error loading checkpoint: {str(e)}"


def get_available_checkpoints(models_dir: str = 'results/models') -> Dict[str, str]:
    """
    Scan the models directory and return available trained model checkpoint paths.
    
    Returns:
        Dictionary mapping model key to checkpoint file path.
    """
    checkpoints = {}
    if not os.path.exists(models_dir):
        return checkpoints
        
    for fname in os.listdir(models_dir):
        if fname.endswith('.pt') or fname.endswith('.pth'):
            # Match with known model keys
            for model_key in MODEL_REGISTRY.keys():
                if model_key in fname.lower():
                    checkpoints[model_key] = os.path.join(models_dir, fname)
                    break
    return checkpoints
