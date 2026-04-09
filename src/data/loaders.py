from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import load_split_metadata
from src.preprocessing.registry import get_transforms
from src.utils.config import load_config
from src.utils.reproducibility import seed_worker, set_global_seed


class EuroSATSplitDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        data_root: Path,
        split: str,
        transform=None,
    ) -> None:
        self.metadata = metadata.loc[metadata["split"] == split].reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.data_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = int(row["label_idx"])
        return image, target


def maybe_subset_dataset(dataset: Dataset, subset_fraction: float, seed: int) -> Dataset:
    if subset_fraction >= 1.0:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    subset_size = max(1, int(len(dataset) * subset_fraction))
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_datasets(
    variant_id: str,
    config_path: str | Path | None = None,
    subset_fraction: float | None = None,
) -> dict[str, Dataset]:
    config = load_config(config_path)
    if subset_fraction is not None:
        config["training"]["subset_fraction"] = subset_fraction
    metadata = load_split_metadata(config_path)
    data_root = config["paths"]["data_raw"]
    datasets: dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
        transform = get_transforms(variant_id, split, config_path)
        dataset = EuroSATSplitDataset(metadata=metadata, data_root=data_root, split=split, transform=transform)
        active_subset_fraction = config["training"].get("subset_fraction", 1.0)
        if split == "train" and active_subset_fraction < 1.0:
            dataset = maybe_subset_dataset(dataset, subset_fraction=active_subset_fraction, seed=config["seed"])
        datasets[split] = dataset
    return datasets


def build_dataloaders(
    variant_id: str,
    config_path: str | Path | None = None,
    subset_fraction: float | None = None,
) -> dict[str, DataLoader]:
    config = load_config(config_path)
    if subset_fraction is not None:
        config["training"]["subset_fraction"] = subset_fraction
    set_global_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    datasets = build_datasets(variant_id=variant_id, config_path=config_path, subset_fraction=subset_fraction)

    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=split == "train",
            num_workers=config["training"]["num_workers"],
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders
