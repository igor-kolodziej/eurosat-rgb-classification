from __future__ import annotations

from pathlib import Path

from torchvision import transforms

from src.data.dataset import get_dataset_paths
from src.preprocessing.transforms import CLAHETransform, RandomRotate90
from src.utils.io import load_json


def _load_train_stats(config_path: str | Path | None = None) -> tuple[list[float], list[float]]:
    stats = load_json(get_dataset_paths(config_path).train_stats_json)
    return stats["mean"], stats["std"]


def get_transforms(variant_id: str, split: str, config_path: str | Path | None = None):
    if variant_id == "v0_raw":
        return transforms.Compose([transforms.ToTensor()])
    if variant_id == "v1_normalized":
        mean, std = _load_train_stats(config_path)
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if variant_id == "v2_enhanced":
        mean, std = _load_train_stats(config_path)
        shared = [CLAHETransform(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        if split == "train":
            return transforms.Compose(
                [
                    CLAHETransform(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    RandomRotate90(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        return transforms.Compose(shared)
    raise ValueError(f"Unknown preprocessing variant: {variant_id}")
