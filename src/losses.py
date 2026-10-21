import torch
import torch.nn as nn


def _ensure_4d_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    return target


def dice_soft(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = probs.contiguous()
    target = target.contiguous()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean()


class DiceBCELoss(nn.Module):
    """Binary segmentation loss: loss = w * (1 - Dice) + (1-w) * BCEWithLogits."""

    def __init__(self, dice_weight: float = 0.7, pos_weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target: torch.Tensor) -> torch.Tensor:
        target = _ensure_4d_target(target)
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = dice_soft(probs, target)
        dice_loss = 1.0 - dice
        return self.dice_weight * dice_loss + (1.0 - self.dice_weight) * bce
