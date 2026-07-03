import torch
import torch.optim as optim
from typing import Optional, Dict, Any


def get_optimizer(name: str, model_params, lr: float = 0.001, **kwargs) -> optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return optim.SGD(model_params, lr=lr, **kwargs)
    elif name == "adam":
        return optim.Adam(model_params, lr=lr, **kwargs)
    elif name == "adamw":
        return optim.AdamW(model_params, lr=lr, **kwargs)
    elif name == "rmsprop":
        return optim.RMSprop(model_params, lr=lr, **kwargs)
    elif name == "adagrad":
        return optim.Adagrad(model_params, lr=lr, **kwargs)
    elif name == "adadelta":
        return optim.Adadelta(model_params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Optimizer '{name}' not found. Available: sgd, adam, adamw, rmsprop, adagrad, adadelta")


def get_scheduler(name: str, optimizer: optim.Optimizer, **kwargs):
    if name is None or name == "none":
        return None
    name = name.lower()
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif name == "cyclic":
        return optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif name == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler '{name}' not found.")
