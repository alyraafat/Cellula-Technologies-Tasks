import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, List
from .metrics import dice_coefficient

class DiceLoss(nn.Module):
    def __init__(self, smooth: float=1.0) -> None:
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(y_pred)
        dice = dice_coefficient(y_true=y_true, y_pred=y_pred, smooth=self.smooth, flat_tensors=True)
        return 1 - dice
    

class FocalLoss(nn.Module):
    def __init__(self, gamma: float=2.0, alpha: Union[float, Tuple[float, float]]=1.0, reduction: str='mean') -> None:
        super().__init__()
        self.gamma = gamma

        # alpha[0] is for class 0 (negative) and alpha[1] is for class 1 (positive)
        if isinstance(alpha, (tuple, list)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = torch.tensor([alpha, alpha])

        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        p_t = torch.exp(-bce_loss)
        modulating_factor = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha[1] * y_true + self.alpha[0] * (1 - y_true)
        loss = alpha_t * modulating_factor * bce_loss
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        

class CombinedLoss(nn.Module):
    def __init__(self, dice_loss_weight: float=0.5, focal_loss_weight: float=0.5, dice_loss_kwargs: dict={}, focal_loss_kwargs: dict={}) -> None:
        super().__init__()
        self.dice_loss_weight = dice_loss_weight
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss = DiceLoss(**dice_loss_kwargs)
        self.focal_loss = FocalLoss(**focal_loss_kwargs)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        combined_loss = self.dice_loss_weight * dice_loss + self.focal_loss_weight * focal_loss
        return combined_loss
        
