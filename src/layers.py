import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class LayerRegistry:
    _layers = {}

    @classmethod
    def register(cls, name: str):
        def decorator(layer_class):
            cls._layers[name] = layer_class
            return layer_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._layers:
            raise ValueError(f"Layer '{name}' not found. Available: {list(cls._layers.keys())}")
        return cls._layers[name]

    @classmethod
    def list_layers(cls) -> List[str]:
        return list(cls._layers.keys())


@LayerRegistry.register("linear")
class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.dropout(self.linear(x))


@LayerRegistry.register("relu")
class CustomReLU(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)


@LayerRegistry.register("sigmoid")
class CustomSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


@LayerRegistry.register("tanh")
class CustomTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


@LayerRegistry.register("softmax")
class CustomSoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


@LayerRegistry.register("batchnorm1d")
class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


@LayerRegistry.register("batchnorm2d")
class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


@LayerRegistry.register("dropout")
class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)


@LayerRegistry.register("conv2d")
class CustomConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


@LayerRegistry.register("maxpool2d")
class CustomMaxPool2d(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


@LayerRegistry.register("avgpool2d")
class CustomAvgPool2d(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


@LayerRegistry.register("flatten")
class CustomFlatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


@LayerRegistry.register("reshape")
class CustomReshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


@LayerRegistry.register("lstm")
class CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output


@LayerRegistry.register("gru")
class CustomGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, x):
        output, hidden = self.gru(x)
        return output


@LayerRegistry.register("embedding")
class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)


@LayerRegistry.register("multihead_attention")
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return output


@LayerRegistry.register("transformer_encoder")
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int = 1,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


@LayerRegistry.register("layer_norm")
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.ln(x)


@LayerRegistry.register("sequential")
class CustomSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
