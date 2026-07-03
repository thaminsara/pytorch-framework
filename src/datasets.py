import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
import glob
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class DatasetManager:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.classes = []
        self.num_classes = 0
        self.dataset = None
        self.dataloaders = {}
        self.sizes = {}

    def list_available_datasets(self) -> List[str]:
        return [
            "cifar10", "cifar100", "mnist", "fashionmnist",
            "imagenet", "svhn", "stl10", "caltech101"
        ]

    def load_imagefolder(self, root_dir: str, transform=None, split_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
        dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        classes = dataset.classes
        num_classes = len(classes)
        train_size = int(len(dataset) * split_ratio)
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        self.classes = classes
        self.num_classes = num_classes
        return train_ds, val_ds

    def create_dataloaders(self, train_ds: Dataset, val_ds: Dataset,
                           batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
        self.dataloaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        }
        self.sizes = {
            "train": len(train_ds),
            "val": len(val_ds)
        }
        return self.dataloaders

    def get_transform(self, img_size: int = 224, mode: str = "train") -> transforms.Compose:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if mode == "train":
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

    def get_common_dataset(self, name: str, root: str = "./data",
                           img_size: int = 224, batch_size: int = 32) -> Dict[str, DataLoader]:
        name = name.lower()
        train_transform = self.get_transform(img_size, mode="train")
        val_transform = self.get_transform(img_size, mode="val")

        dataset_loaders = {
            "mnist": datasets.MNIST,
            "fashionmnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "svhn": datasets.SVHN,
        }

        if name not in dataset_loaders:
            raise ValueError(f"Dataset '{name}' not supported. Available: {self.list_available_datasets()}")

        if name in ["svhn"]:
            train_ds = dataset_loaders[name](root=root, split="train", download=True, transform=train_transform)
            val_ds = dataset_loaders[name](root=root, split="test", download=True, transform=val_transform)
        else:
            train_ds = dataset_loaders[name](root=root, train=True, download=True, transform=train_transform)
            val_ds = dataset_loaders[name](root=root, train=False, download=True, transform=val_transform)

        self.classes = train_ds.classes if hasattr(train_ds, "classes") else [str(i) for i in range(train_ds.targets.max() + 1)]
        self.num_classes = len(self.classes)

        self.dataloaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        }
        self.sizes = {"train": len(train_ds), "val": len(val_ds)}
        return self.dataloaders
