"""
Candidate Deep Learning Model Architectures for Alzheimer's Multi-Stage Detection.
Defines MobileNet, EfficientNet, and ResNet models adapted for 4-class MRI classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


def build_mobilenet_v2(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build MobileNetV2 architecture with adapted 4-class classification head.
    Lightweight inverted residual architecture ideal for fast inference.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_efficientnet_b0(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 architecture with adapted 4-class classification head.
    Compound-scaled backbone with high parameter efficiency.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_resnet18(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-18 architecture with adapted 4-class classification head.
    Deep residual network with identity shortcut connections.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_resnet50(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-50 architecture with adapted 4-class classification head.
    Deeper bottleneck residual network.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_target_convolutional_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Identify and return the final convolutional layer for Grad-CAM computation.
    
    Args:
        model: PyTorch model instance.
        model_name: Model identifier name.
        
    Returns:
        Target nn.Module layer.
    """
    name = model_name.lower()
    if 'resnet' in name:
        # Final block in layer4
        return model.layer4[-1]
    elif 'efficientnet' in name:
        # Final conv layer in features
        return model.features[-1]
    elif 'mobilenet' in name:
        # Final conv layer in features
        return model.features[-1]
    else:
        # Fallback: search for last Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise ValueError(f"Could not find a valid convolutional layer for Grad-CAM in model '{model_name}'.")
