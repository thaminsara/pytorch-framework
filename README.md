# PyTorch Training Framework

A comprehensive Python framework for training deep learning models from scratch with a Flask-based GUI.

## Features

- **Custom Model Builder**: Visual layer-based model construction with 20+ layer types
- **Training Engine**: Full-featured training loop with callbacks, metrics, and logging
- **Visualization**: Real-time plots for loss, accuracy, confusion matrix, learning rate, etc.
- **Dataset Manager**: Built-in support for CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN and custom datasets
- **Pretrained Models**: Integration with torchvision model zoo
- **Loss Functions**: Cross-entropy, MSE, MAE, Huber, Focal, Label Smoothing, Dice, etc.
- **Optimizers**: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta
- **Schedulers**: Step, MultiStep, Exponential, Cosine, ReduceLROnPlateau, Cyclic, OneCycle
- **Callbacks**: EarlyStopping, ModelCheckpoint, LearningRateScheduler, History, Timer

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the GUI

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

### Programmatic Usage

```python
import torch
import torch.nn as nn
from src.engine import Engine
from src.model_builder import ModelBuilder, AutoModelBuilder
from src.datasets import DatasetManager
from src.losses import get_loss
from src.optimizers import get_optimizer, get_scheduler
from src.callbacks import EarlyStopping, ModelCheckpoint

# Load dataset
dm = DatasetManager()
loaders = dm.get_common_dataset("cifar10", batch_size=64)

# Build custom model
builder = ModelBuilder(input_shape=[3, 32, 32])
builder.add_layer("conv2d", in_channels=3, out_channels=32, kernel_size=3, padding=1)
builder.add_layer("maxpool2d", kernel_size=2, stride=2)
builder.add_layer("relu")
builder.add_layer("conv2d", in_channels=32, out_channels=64, kernel_size=3, padding=1)
builder.add_layer("maxpool2d", kernel_size=2, stride=2)
builder.add_layer("flatten")
builder.add_layer("linear", in_features=64*8*8, out_features=128)
builder.add_layer("relu")
builder.add_layer("linear", in_features=128, out_features=10)
model = builder.build()

# Train
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer("adam", model.parameters(), lr=0.001)
scheduler = get_scheduler("step", optimizer, step_size=5, gamma=0.5)

engine = Engine(model, loaders["train"], loaders["val"], criterion, optimizer, scheduler)
engine.add_callback(EarlyStopping(monitor="val_loss", patience=10))
history = engine.fit(epochs=20)

# Evaluate
results = engine.evaluate()
print(f"Accuracy: {results['accuracy']:.4f}")

# Save model
engine.save_model("./models/my_model.pt")
```

### Using Pretrained Models

```python
import torchvision.models as models

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 10)
```

## Directory Structure

```
pytorch-framework/
├── app.py                    # Flask application entry point
├── requirements.txt
├── config/
│   └── default.yaml          # Default configuration
├── src/
│   ├── engine.py             # Core training engine
│   ├── model_builder.py      # Custom model builder
│   ├── layers.py             # Layer registry and custom layers
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   ├── callbacks.py          # Training callbacks
│   ├── optimizers.py         # Optimizers and schedulers
│   ├── datasets.py           # Data loading utilities
│   ├── visualization.py      # Visualization tools
│   └── utils.py              # Utility functions
├── gui/
│   ├── routes.py             # Flask routes
│   ├── templates/            # HTML templates
│   └── static/               # CSS, JS, and plot images
├── models/                   # Saved model weights
├── experiments/              # Experiment logs
└── data/                     # Dataset storage
```

## Available Layers

Linear, ReLU, Sigmoid, Tanh, Softmax, BatchNorm1D, BatchNorm2D, Dropout, Conv2D, MaxPool2D, AvgPool2D, Flatten, Reshape, LSTM, GRU, Embedding, MultiHeadAttention, TransformerEncoder, LayerNorm, Sequential

## GUI Pages

- **Home**: Overview and quick start guide
- **Train**: Build custom models and configure training
- **Models**: Browse saved models
- **Visualize**: View training metrics and plots
- **Results**: Detailed training results

## License

MIT License
