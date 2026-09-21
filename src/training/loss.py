"""
Loss functions for handling multi-class imbalance in Alzheimer's MRI datasets.
Provides class-weighted cross-entropy and focal loss implementations.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for mitigating extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    loss_type: str = 'weighted_ce',
    device: str = 'cpu'
) -> nn.Module:
    """
    Configure loss function.
    
    Args:
        class_weights: Tensor of shape [num_classes] representing inverse frequency weights.
        loss_type: 'weighted_ce' or 'focal'.
        device: 'cpu' or 'cuda'.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
        
    if loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)
