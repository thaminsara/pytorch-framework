import torch
import torch.nn as nn
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1, dim: int = -1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=self.dim), dim=self.dim))


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return bce + (1 - dice)


class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss(pred, target)


class MAELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        return self.loss(pred, target)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, pred, target):
        return self.loss(pred, target)


class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, zero_infinity: bool = False):
        super().__init__()
        self.loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)
        return self.loss(log_probs, targets, input_lengths, target_lengths)


_LOSS_MAP = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "mse": MSELoss,
    "mae": MAELoss,
    "huber": HuberLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "dice": DiceLoss,
    "dice_bce": DiceBCELoss,
    "ctc": CTCLoss,
    "nll": nn.NLLLoss,
    "kl_div": nn.KLDivLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    if name not in _LOSS_MAP:
        raise ValueError(f"Loss '{name}' not found. Available: {list(_LOSS_MAP.keys())}")
    return _LOSS_MAP[name](**kwargs)
