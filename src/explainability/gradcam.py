"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Explainability Engine.
Produces visual heatmaps highlighting salient neuroanatomical regions in Brain MRI slices.
"""

from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures import get_target_convolutional_layer


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, model_name: str = "resnet18"):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            self.target_layer = get_target_convolutional_layer(model, model_name)
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains the gradient of the loss/score wrt layer output
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Generate 2D normalized Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: Tensor of shape [1, 3, H, W].
            target_class: Integer class index to explain. If None, uses model's predicted class.
            
        Returns:
            Tuple of:
                - heatmap: 2D numpy array of shape (H, W) normalized to [0.0, 1.0]
                - predicted_class: Index of predicted class
                - probabilities: 1D numpy array of softmax probabilities across all 4 stages
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1).detach().cpu().numpy().squeeze()
        pred_class = int(torch.argmax(output, dim=1).item())
        
        if target_class is None:
            target_class = pred_class
            
        # Target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")
            
        # Global average pooling of gradients: weights alpha_k = (1/Z) * sum(gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus only on features with positive influence on the target class
        cam = F.relu(cam)
        
        # Upsample heatmap to match input image resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        
        # Squeeze to 2D numpy array
        cam_np = cam.squeeze().cpu().numpy()
        
        # Min-max normalization to [0.0, 1.0]
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np, pred_class, probs


def overlay_heatmap(
    original_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = 'jet'
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on top of the original MRI image.
    
    Args:
        original_image: Original MRI as PIL Image or numpy array (RGB).
        heatmap: 2D numpy array [0.0, 1.0].
        alpha: Heatmap blend opacity (0.0 to 1.0).
        colormap_name: Matplotlib colormap ('jet', 'turbo', 'viridis', 'plasma').
        
    Returns:
        Blended RGB PIL Image.
    """
    if isinstance(original_image, np.ndarray):
        pil_img = Image.fromarray(original_image).convert('RGB')
    else:
        pil_img = original_image.convert('RGB')
        
    w, h = pil_img.size
    
    # Resize heatmap if dimensions differ
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((w, h), Image.Resampling.BILINEAR)
        heatmap = np.array(heatmap_pil) / 255.0
        
    # Get colormap
    try:
        cmap = cm.get_cmap(colormap_name)
    except Exception:
        cmap = cm.jet
        
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Extract RGB, ignore alpha
    colored_heatmap = (colored_heatmap * 255.0).astype(np.uint8)
    
    # Blend with original image
    orig_np = np.array(pil_img).astype(np.float32)
    heat_np = colored_heatmap.astype(np.float32)
    
    blended = (1.0 - alpha) * orig_np + alpha * heat_np
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)
