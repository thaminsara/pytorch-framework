import torch
import os
import time
from typing import Optional, Dict, Any, List
import numpy as np


class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 10,
                 min_delta: float = 0.0, mode: str = "min", verbose: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.wait = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None or self.monitor not in logs:
            return
        current = logs[self.monitor]
        improved = False
        if self.mode == "min":
            if current < self.best_value - self.min_delta:
                improved = True
        else:
            if current > self.best_value + self.min_delta:
                improved = True
        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best {self.monitor}: {self.best_value:.4f}")


class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = "val_loss",
                 mode: str = "min", save_best_only: bool = True, verbose: bool = True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None or self.monitor not in logs:
            return
        current = logs[self.monitor]
        if not self.save_best_only:
            torch.save(logs.get("model_state_dict", None), self.filepath.format(epoch=epoch, **logs))
            if self.verbose:
                print(f"Model saved to {self.filepath}")
            return
        improved = False
        if self.mode == "min":
            if current < self.best_value:
                improved = True
        else:
            if current > self.best_value:
                improved = True
        if improved:
            self.best_value = current
            torch.save(logs.get("model_state_dict", None), self.filepath)
            if self.verbose:
                print(f"Model improved. Saved to {self.filepath}")


class LearningRateScheduler(Callback):
    def __init__(self, scheduler, verbose: bool = True):
        super().__init__()
        self.scheduler = scheduler
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if logs and self.scheduler.mode == "min":
                self.scheduler.step(logs.get("val_loss", logs.get("loss", 0)))
            elif logs:
                self.scheduler.step(logs.get("val_accuracy", logs.get("accuracy", 0)))
        else:
            self.scheduler.step()
        if self.verbose:
            lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: learning rate = {lr:.6f}")


class History(Callback):
    def __init__(self):
        super().__init__()
        self.history: Dict[str, List[Any]] = {}

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        self.history = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)


class ProgressCallback(Callback):
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.verbose:
            print(f"\nEpoch {epoch + 1}")

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        if self.verbose and logs and batch % 10 == 0:
            loss = logs.get("loss", 0.0)
            print(f"  Batch {batch}: loss = {loss:.4f}")


class TimerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        if logs is not None:
            logs["epoch_time"] = epoch_time

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        total_time = time.time() - self.start_time
        if logs is not None:
            logs["total_time"] = total_time
