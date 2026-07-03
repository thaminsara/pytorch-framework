import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Callable, Tuple
from tqdm import tqdm
import time
import os
from .utils import count_parameters, get_device, ensure_dir, get_timestamp
from .callbacks import Callback, EarlyStopping, ModelCheckpoint, History, TimerCallback, ProgressCallback
from .optimizers import get_optimizer, get_scheduler
from .losses import get_loss
from .metrics import Accuracy, Precision, Recall, F1Score, AverageMeter
from .visualization import Visualizer


class Engine:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 criterion: Optional[nn.Module] = None, optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = "auto", config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = get_device(device)
        self.model.to(self.device)
        self.config = config or {}
        self.history: Dict[str, List[float]] = {}
        self.callbacks: List[Callback] = []
        self.visualizer = Visualizer(save_dir=self.config.get("paths", {}).get("plots_dir", "./gui/static/plots"))
        self._setup_default_callbacks()

    def _setup_default_callbacks(self):
        self.callbacks.append(TimerCallback())
        self.callbacks.append(History())
        self.callbacks.append(ProgressCallback())

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)
        return self

    def _log_epoch(self, logs: Dict[str, Any]):
        log_str = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
        print(f"  {log_str}")

    def train_epoch(self, epoch: int, pbar: Optional[tqdm] = None) -> Dict[str, float]:
        self.model.train()
        train_loss = AverageMeter("train_loss")
        train_acc = Accuracy()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            for cb in self.callbacks:
                cb.on_batch_begin(batch_idx)
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            if self.config.get("training", {}).get("gradient_clip"):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip"])
            self.optimizer.step()
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)
            running_loss += loss.item()
            logs = {"loss": loss.item(), "train_loss": train_loss.avg}
            if batch_idx % 10 == 0 and pbar:
                pbar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in logs.items()})
            for cb in self.callbacks:
                cb.on_batch_end(batch_idx, logs)

        return {"train_loss": train_loss.avg, "train_accuracy": train_acc.compute()}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        val_loss = AverageMeter("val_loss")
        val_acc = Accuracy()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(output, target)
                all_preds.append(output.argmax(dim=1).cpu())
                all_targets.append(target.cpu())
        return {
            "val_loss": val_loss.avg,
            "val_accuracy": val_acc.compute(),
            "val_preds": torch.cat(all_preds).numpy(),
            "val_targets": torch.cat(all_targets).numpy(),
        }

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        self.history = {k: [] for k in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]}
        logs = {"model_state_dict": self.model.state_dict()}
        for cb in self.callbacks:
            cb.on_train_begin(logs)
        for epoch in range(epochs):
            epoch_logs = {}
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False) as pbar:
                train_metrics = self.train_epoch(epoch, pbar)
                val_metrics = self.validate_epoch(epoch)
                epoch_logs.update(train_metrics)
                epoch_logs.update(val_metrics)
                epoch_logs["epoch"] = epoch + 1
                epoch_logs["model_state_dict"] = self.model.state_dict()
                for key, value in epoch_logs.items():
                    if key not in self.history:
                        self.history[key] = []
                    if isinstance(value, (int, float)):
                        self.history[key].append(value)
                if pbar:
                    pbar.close()
                self._log_epoch(epoch_logs)
                stop = False
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, epoch_logs)
                    if hasattr(cb, "stop_training") and cb.stop_training:
                        stop = True
                if stop:
                    print(f"Training stopped early at epoch {epoch + 1}")
                    break
        logs = {}
        for cb in self.callbacks:
            cb.on_train_end(logs)
        return self.history

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        if loader is None:
            loader = self.val_loader
        if loader is None:
            raise ValueError("No validation loader provided.")
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                all_preds.append(output.argmax(dim=1).cpu())
                all_targets.append(target.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        accuracy = (all_preds == all_targets).mean()
        return {
            "loss": total_loss / total_samples,
            "accuracy": accuracy,
            "predictions": all_preds.tolist(),
            "targets": all_targets.tolist(),
            "num_samples": total_samples,
        }

    def save_model(self, filepath: str):
        ensure_dir(os.path.dirname(filepath) or ".")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
            "history": self.history,
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("history"):
            self.history = checkpoint["history"]
        if checkpoint.get("config"):
            self.config = checkpoint["config"]
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return torch.softmax(self.model(x), dim=1)

    def get_model_summary(self) -> str:
        from .utils import count_parameters
        params = count_parameters(self.model)
        lines = [
            "=" * 50,
            "Model Summary",
            "=" * 50,
            f"Device: {self.device}",
            f"Total Parameters: {params['total']:,}",
            f"Trainable Parameters: {params['trainable']:,}",
            f"Non-trainable Parameters: {params['total'] - params['trainable']:,}",
            "=" * 50,
        ]
        return "\n".join(lines)
