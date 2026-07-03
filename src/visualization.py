import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime


class Visualizer:
    def __init__(self, save_dir: str = "./gui/static/plots", dpi: int = 150):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        self.plots = {}

    def plot_loss_curves(self, history: Dict[str, List[float]], title: str = "Loss Curves") -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in ["train_loss", "val_loss", "loss"]:
            if key in history:
                ax.plot(history[key], label=key.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        filepath = os.path.join(self.save_dir, f"loss_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["loss_curves"] = filepath
        return filepath

    def plot_accuracy_curves(self, history: Dict[str, List[float]], title: str = "Accuracy Curves") -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in ["train_accuracy", "val_accuracy", "accuracy"]:
            if key in history:
                ax.plot(history[key], label=key.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        filepath = os.path.join(self.save_dir, f"accuracy_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["accuracy_curves"] = filepath
        return filepath

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                              title: str = "Confusion Matrix") -> str:
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        filepath = os.path.join(self.save_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["confusion_matrix"] = filepath
        return filepath

    def plot_learning_rate(self, history: Dict[str, List[float]], title: str = "Learning Rate Schedule") -> str:
        if "learning_rate" not in history:
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["learning_rate"], label="Learning Rate", color="orange")
        ax.set_xlabel("Step/Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        filepath = os.path.join(self.save_dir, f"learning_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["learning_rate"] = filepath
        return filepath

    def plot_epoch_times(self, history: Dict[str, List[float]], title: str = "Epoch Times") -> str:
        if "epoch_time" not in history:
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(history["epoch_time"]) + 1), history["epoch_time"], color="skyblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        filepath = os.path.join(self.save_dir, f"epoch_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["epoch_times"] = filepath
        return filepath

    def plot_class_distribution(self, class_counts: Dict[int, int], title: str = "Class Distribution") -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax.bar([str(c) for c in classes], counts, color="steelblue")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        filepath = os.path.join(self.save_dir, f"class_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["class_distribution"] = filepath
        return filepath

    def plot_model_architecture(self, model: torch.nn.Module, input_shape: List[int],
                                title: str = "Model Architecture") -> str:
        try:
            from torchviz import make_dot
            dummy_input = torch.randn(1, *input_shape)
            output = model(dummy_input)
            dot = make_dot(output, params=dict(model.named_parameters()))
            filepath = os.path.join(self.save_dir, f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            dot.render(filepath.replace(".png", ""), format="png", cleanup=True)
            filepath = filepath.replace(".png", "") + ".png"
            self.plots["architecture"] = filepath
            return filepath
        except Exception:
            return self._fallback_architecture_plot(model)

    def _fallback_architecture_plot(self, model: torch.nn.Module) -> str:
        layer_types = []
        layer_params = []
        for name, module in model.named_modules():
            if name == "":
                continue
            layer_type = type(module).__name__
            layer_types.append(layer_type)
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_params.append(f"{name}: {layer_type} ({module.in_features if hasattr(module, 'in_features') else module.in_channels})")
            else:
                layer_params.append(f"{name}: {layer_type}")
        fig, ax = plt.subplots(figsize=(12, len(layer_types) * 0.4 + 2))
        y_pos = np.arange(len(layer_types))
        ax.barh(y_pos, [1] * len(layer_types), color="lightcoral")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_params, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_title("Model Architecture Overview")
        ax.invert_yaxis()
        ax.set_xticks([])
        filepath = os.path.join(self.save_dir, f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.plots["architecture"] = filepath
        return filepath

    def plot_all_metrics(self, history: Dict[str, List[float]], model: Optional[torch.nn.Module] = None,
                         input_shape: Optional[List[int]] = None, class_names: Optional[List[str]] = None,
                         confusion_matrix: Optional[np.ndarray] = None) -> Dict[str, str]:
        paths = {}
        if history:
            if any("loss" in k for k in history.keys()):
                paths["loss_curves"] = self.plot_loss_curves(history)
            if any("accuracy" in k for k in history.keys()):
                paths["accuracy_curves"] = self.plot_accuracy_curves(history)
            if "learning_rate" in history:
                paths["learning_rate"] = self.plot_learning_rate(history)
            if "epoch_time" in history:
                paths["epoch_times"] = self.plot_epoch_times(history)
        if model is not None and input_shape:
            paths["architecture"] = self.plot_model_architecture(model, input_shape)
        if confusion_matrix is not None and class_names:
            paths["confusion_matrix"] = self.plot_confusion_matrix(confusion_matrix, class_names)
        return paths

    def get_plot_path(self, plot_name: str) -> str:
        return self.plots.get(plot_name, "")

    def clear_plots(self):
        self.plots = {}
