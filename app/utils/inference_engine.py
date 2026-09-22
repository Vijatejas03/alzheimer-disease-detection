"""
Inference Engine and Model Management for Alzheimer's Detection Streamlit App.
Provides cached model loading, device detection, inference execution, and Grad-CAM generation.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import (
    build_mobilenet_v2,
    build_efficientnet_b0,
    build_resnet18,
    get_target_convolutional_layer
)
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.data.augmentation import get_validation_transforms

CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

CLASS_COLORS = {
    "Non-Demented": "#2563eb",         # Medical Blue
    "Very Mild Demented": "#6366f1",   # Blue-Purple
    "Mild Demented": "#d97706",        # Amber
    "Moderate Demented": "#dc2626"     # Restrained Red
}

MODEL_CONFIGS = {
    "EfficientNet-B0": {
        "slug": "efficientnet_b0",
        "ckpt_path": PROJECT_ROOT / "results" / "models" / "efficientnet_b0_best.pt",
        "builder": lambda: build_efficientnet_b0(num_classes=4, pretrained=False),
        "target_layer_desc": "features.8 (Conv2dNormActivation: 1280 channels)",
        "total_parameters": 4012672,
        "params": "4,012,672 (~4.01M)",
        "params_formatted": "4,012,672 (~4.01M)"
    },
    "ResNet18": {
        "slug": "resnet18",
        "ckpt_path": PROJECT_ROOT / "results" / "models" / "resnet18_best.pt",
        "builder": lambda: build_resnet18(num_classes=4, pretrained=False),
        "target_layer_desc": "layer4.1 (BasicBlock: 512 channels)",
        "total_parameters": 11178564,
        "params": "11,178,564 (~11.18M)",
        "params_formatted": "11,178,564 (~11.18M)"
    },
    "MobileNetV2": {
        "slug": "mobilenet_v2",
        "ckpt_path": PROJECT_ROOT / "results" / "models" / "mobilenet_v2_best.pt",
        "builder": lambda: build_mobilenet_v2(num_classes=4, pretrained=False),
        "target_layer_desc": "features.18 (Conv2dNormActivation: 1280 channels)",
        "total_parameters": 2228996,
        "params": "2,228,996 (~2.23M)",
        "params_formatted": "2,228,996 (~2.23M)"
    }
}


def get_inference_device() -> Tuple[torch.device, str]:
    """Detect CUDA availability and return device along with descriptive string."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        device = torch.device('cuda:0')
        desc = f"CUDA (NVIDIA {gpu_name}, {vram_mb:.0f} MiB VRAM)"
    else:
        device = torch.device('cpu')
        desc = "CPU (CUDA unavailable)"
    return device, desc


RESNET18_RELEASE_URL = "https://github.com/Vijatejas03/alzheimer-disease-detection/releases/download/v1.0.0/resnet18_best.pt"


def ensure_checkpoint_available(model_name: str, ckpt_path: Path) -> Tuple[bool, str]:
    """
    Ensure model checkpoint file is present on disk.
    If missing in cloud deployment, downloads from GitHub Release asset.
    """
    if ckpt_path.exists():
        return True, "Checkpoint present on disk"

    if model_name == "ResNet18":
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = ckpt_path.with_suffix(".tmp")
        try:
            import urllib.request
            with urllib.request.urlopen(RESNET18_RELEASE_URL, timeout=120) as resp:
                if resp.status != 200:
                    return False, f"GitHub Release returned HTTP {resp.status} for ResNet-18 checkpoint"
                with open(temp_path, "wb") as out_f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
            if temp_path.exists() and temp_path.stat().st_size > 1000000:
                temp_path.replace(ckpt_path)
                return True, "Downloaded successfully from GitHub Release"
            else:
                if temp_path.exists():
                    temp_path.unlink()
                return False, "Downloaded file was invalid or incomplete"
        except Exception as dl_err:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False, f"Failed to download ResNet-18 checkpoint: {dl_err}"

    return False, f"Checkpoint not found at: {ckpt_path}"


@st.cache_resource(show_spinner="Loading trained neural architecture checkpoint...")
def load_cached_model(model_name: str) -> Tuple[Optional[torch.nn.Module], str]:
    """
    Cached model loader. Instantiates architecture and loads best checkpoint weights.
    Caches model in GPU/CPU memory to avoid disk I/O on UI interactions.
    """
    if model_name not in MODEL_CONFIGS:
        return None, f"Unknown model name '{model_name}'"
        
    cfg = MODEL_CONFIGS[model_name]
    ckpt_path = cfg["ckpt_path"]
    
    ok, msg = ensure_checkpoint_available(model_name, ckpt_path)
    if not ok:
        return None, msg
        
    device, _ = get_inference_device()
    try:
        model = cfg["builder"]()
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        return model, "Loaded successfully"
    except Exception as e:
        return None, f"Error loading checkpoint: {e}"


def preprocess_image_for_inference(pil_img: Image.Image) -> Tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
    """
    Convert raw input MRI to standardized 224x224 RGB representation and normalized PyTorch tensor.
    
    Returns:
        - tensor: PyTorch tensor [1, 3, 224, 224] with ImageNet normalization
        - display_img: Standardized 224x224 RGB PIL image for overlay and display
        - meta: Dictionary with image dimensions and format info
    """
    orig_mode = pil_img.mode
    orig_size = pil_img.size
    
    # Standardize to 3-channel RGB (standard CNN backbone requirement)
    rgb_img = pil_img.convert('RGB')
    resized_img = rgb_img.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Inference transform (ImageNet mean & std normalization)
    val_transform = get_validation_transforms(img_size=(224, 224))
    tensor = val_transform(rgb_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    
    meta = {
        "original_mode": orig_mode,
        "original_width": orig_size[0],
        "original_height": orig_size[1],
        "channels": 3,
        "inference_size": (224, 224)
    }
    return tensor, resized_img, meta


def run_model_inference(model: torch.nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Execute forward inference on input tensor using torch.no_grad().
    
    Returns complete prediction metrics, confidence, probabilities, and execution latency.
    """
    device, _ = get_inference_device()
    input_tensor = input_tensor.to(device)
    
    start_t = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probs_tensor = F.softmax(logits, dim=1).squeeze()
        pred_idx = int(torch.argmax(probs_tensor).item())
        probs_np = probs_tensor.cpu().numpy()
        confidence = float(probs_np[pred_idx])
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    
    prob_dict = {CLASS_NAMES[i]: float(probs_np[i]) for i in range(4)}
    
    return {
        "predicted_idx": pred_idx,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": prob_dict,
        "latency_ms": latency_ms
    }


def generate_gradcam(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    display_img: Image.Image,
    target_class: Optional[int] = None
) -> Tuple[np.ndarray, Image.Image, str]:
    """
    Generate Grad-CAM heatmap and blended overlay for the given input image.
    Uses existing GradCAM class and target layer hook mechanism.
    
    Returns:
        - heatmap_np: 2D numpy array [0.0, 1.0] of shape (224, 224)
        - overlay_img: PIL Image with blended jet colormap overlay
        - target_layer_desc: Human-readable description of target layer
    """
    device, _ = get_inference_device()
    input_tensor = input_tensor.to(device)
    
    cfg = MODEL_CONFIGS[model_name]
    slug = cfg["slug"]
    target_layer_desc = cfg["target_layer_desc"]
    
    target_layer = get_target_convolutional_layer(model, slug)
    cam_engine = GradCAM(model, target_layer=target_layer, model_name=slug)
    
    # Generate CAM
    cam_np, _, _ = cam_engine.generate_cam(input_tensor, target_class=target_class)
    
    # Generate overlay
    overlay_img = overlay_heatmap(display_img, cam_np, alpha=0.45, colormap_name='jet')
    
    return cam_np, overlay_img, target_layer_desc
