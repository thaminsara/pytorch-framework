import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
from .layers import LayerRegistry


class ModelBuilder:
    def __init__(self, input_shape: List[int] = None):
        self.input_shape = input_shape
        self.layers = []
        self.model = None

    def add_layer(self, layer_type: str, **kwargs) -> "ModelBuilder":
        if layer_type not in LayerRegistry.list_layers():
            available = ", ".join(LayerRegistry.list_layers())
            raise ValueError(f"Layer '{layer_type}' not found. Available layers: {available}")
        self.layers.append({"type": layer_type, "params": kwargs})
        return self

    def build(self) -> nn.Module:
        if not self.layers:
            raise ValueError("No layers added. Use add_layer() to build your model.")
        layers = []
        for layer_config in self.layers:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            layer_class = LayerRegistry.get(layer_type)
            layer_instance = layer_class(**params)
            layers.append(layer_instance)
        self.model = nn.Sequential(*layers)
        return self.model

    def summary(self) -> str:
        if self.model is None:
            self.build()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lines = [
            "=" * 60,
            f"Model Summary",
            "=" * 60,
            f"Input Shape: {self.input_shape}",
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Non-trainable Parameters: {total_params - trainable_params:,}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def from_config(self, config: List[Dict[str, Any]]) -> nn.Module:
        self.layers = []
        for layer_config in config:
            layer_type = layer_config.get("type")
            params = layer_config.get("params", {})
            self.add_layer(layer_type, **params)
        return self.build()

    def reset(self):
        self.layers = []
        self.model = None

    def get_config(self) -> List[Dict[str, Any]]:
        return self.layers


class AutoModelBuilder:
    @staticmethod
    def build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int,
                  dropout: float = 0.0, activation: str = "relu") -> nn.Sequential:
        builder = ModelBuilder(input_shape=[input_dim])
        builder.add_layer("linear", in_features=input_dim, out_features=hidden_dims[0], dropout=dropout)
        builder.add_layer(activation)
        for i in range(1, len(hidden_dims)):
            builder.add_layer("linear", in_features=hidden_dims[i-1], out_features=hidden_dims[i], dropout=dropout)
            builder.add_layer(activation)
        builder.add_layer("linear", in_features=hidden_dims[-1], out_features=output_dim)
        return builder.build()

    @staticmethod
    def build_cnn(input_channels: int, num_classes: int, img_size: int = 32,
                  dropout: float = 0.0) -> nn.Sequential:
        builder = ModelBuilder(input_shape=[input_channels, img_size, img_size])
        builder.add_layer("conv2d", in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, dropout=dropout)
        builder.add_layer("maxpool2d", kernel_size=2, stride=2)
        builder.add_layer("conv2d", in_channels=32, out_channels=64, kernel_size=3, padding=1, dropout=dropout)
        builder.add_layer("maxpool2d", kernel_size=2, stride=2)
        builder.add_layer("conv2d", in_channels=64, out_channels=128, kernel_size=3, padding=1, dropout=dropout)
        builder.add_layer("maxpool2d", kernel_size=2, stride=2)
        builder.add_layer("flatten")
        builder.add_layer("linear", in_features=128 * (img_size // 8) * (img_size // 8),
                         out_features=256, dropout=dropout)
        builder.add_layer("relu")
        builder.add_layer("linear", in_features=256, out_features=num_classes)
        return builder.build()

    @staticmethod
    def build_lstm(input_dim: int, hidden_dim: int, num_layers: int,
                   output_dim: int, dropout: float = 0.0) -> nn.Sequential:
        builder = ModelBuilder(input_shape=[None, input_dim])
        builder.add_layer("lstm", input_size=input_dim, hidden_size=hidden_dim,
                         num_layers=num_layers, dropout=dropout)
        builder.add_layer("linear", in_features=hidden_dim, out_features=output_dim)
        return builder.build()
