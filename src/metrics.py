import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class Accuracy:
    def __init__(self, top_k: int = 1):
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            if preds.dim() == 1:
                preds = preds.unsqueeze(1)
            _, top_preds = preds.topk(self.top_k, dim=1)
            correct = top_preds.eq(targets.view(-1, 1).expand_as(top_preds))
            self.correct += correct.sum().item()
            self.total += targets.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Precision:
    def __init__(self, average: str = "macro"):
        self.average = average
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            pred_labels = torch.argmax(preds, dim=1)
            self.tp += (pred_labels == targets).sum().item()
            self.fp += (pred_labels != targets).sum().item()

    def compute(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)


class Recall:
    def __init__(self, average: str = "macro"):
        self.average = average
        self.reset()

    def reset(self):
        self.tp = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            pred_labels = torch.argmax(preds, dim=1)
            self.tp += (pred_labels == targets).sum().item()
            self.fn += (pred_labels != targets).sum().item()

    def compute(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)


class F1Score:
    def __init__(self, average: str = "macro"):
        self.average = average
        self.precision = Precision(average)
        self.recall = Recall(average)

    def reset(self):
        self.precision.reset()
        self.recall.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute(self) -> float:
        p = self.precision.compute()
        r = self.recall.compute()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)


class IoU:
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            preds = torch.argmax(preds, dim=1) if preds.dim() > 1 else preds
            for c in range(self.num_classes):
                if self.ignore_index is not None and c == self.ignore_index:
                    continue
                pred_c = (preds == c)
                target_c = (targets == c)
                self.intersection[c] += (pred_c & target_c).sum().item()
                self.union[c] += (pred_c | target_c).sum().item()

    def compute(self) -> float:
        iou_per_class = self.intersection / (self.union + 1e-6)
        return iou_per_class.mean().item()

    def compute_per_class(self) -> Dict[int, float]:
        iou_per_class = self.intersection / (self.union + 1e-6)
        return {c: iou_per_class[c].item() for c in range(self.num_classes)}


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            preds = torch.argmax(preds, dim=1) if preds.dim() > 1 else preds
            for t, p in zip(targets.view(-1), preds.view(-1)):
                self.matrix[t.long(), p.long()] += 1

    def compute(self) -> torch.Tensor:
        return self.matrix


class AverageMeter:
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


_METRIC_MAP = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1": F1Score,
    "iou": IoU,
    "confusion_matrix": ConfusionMatrix,
    "average_meter": AverageMeter,
}


def get_metric(name: str, **kwargs):
    if name not in _METRIC_MAP:
        raise ValueError(f"Metric '{name}' not found. Available: {list(_METRIC_MAP.keys())}")
    return _METRIC_MAP[name](**kwargs)
