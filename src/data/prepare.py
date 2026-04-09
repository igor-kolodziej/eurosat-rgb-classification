from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import EuroSAT

from src.data.dataset import get_dataset_paths
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.reproducibility import set_global_seed


def download_eurosat_dataset(config_path: str | Path | None = None) -> EuroSAT:
    config = load_config(config_path)
    set_global_seed(config["seed"])
    ensure_dir(config["paths"]["data_raw"])
    return EuroSAT(root=config["paths"]["data_raw"], download=config["dataset"]["download"])


def build_split_dataframe(dataset: EuroSAT, config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    seed = config["seed"]
    ratios = config["dataset"]["split_ratios"]

    records: list[dict[str, object]] = []
    for sample_path, label_idx in dataset.samples:
        sample_path = Path(sample_path)
        records.append(
            {
                "relative_path": str(sample_path.relative_to(config["paths"]["data_raw"])),
                "label_name": dataset.classes[label_idx],
                "label_idx": int(label_idx),
            }
        )

    metadata = pd.DataFrame(records)
    indices = metadata.index.to_numpy()
    labels = metadata["label_idx"].to_numpy()

    first_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - ratios["train"],
        random_state=seed,
    )
    train_idx, temp_idx = next(first_split.split(indices, labels))

    temp_fraction = ratios["val"] + ratios["test"]
    val_fraction_within_temp = ratios["val"] / temp_fraction
    second_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - val_fraction_within_temp,
        random_state=seed,
    )
    temp_labels = labels[temp_idx]
    val_sub_idx, test_sub_idx = next(second_split.split(temp_idx, temp_labels))
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    metadata["split"] = ""
    metadata.loc[train_idx, "split"] = "train"
    metadata.loc[val_idx, "split"] = "val"
    metadata.loc[test_idx, "split"] = "test"

    validate_split(metadata)
    return metadata.sort_values(["split", "label_name", "relative_path"]).reset_index(drop=True)


def validate_split(metadata: pd.DataFrame) -> None:
    if metadata["split"].eq("").any():
        raise ValueError("All rows must be assigned to a split.")

    if metadata["relative_path"].duplicated().any():
        duplicates = metadata.loc[metadata["relative_path"].duplicated(), "relative_path"].tolist()
        raise ValueError(f"Duplicate sample paths found: {duplicates[:5]}")

    if metadata["split"].value_counts().sum() != len(metadata):
        raise ValueError("Split counts do not sum to dataset size.")

    overlap_count = metadata.groupby("relative_path")["split"].nunique().gt(1).sum()
    if overlap_count:
        raise ValueError("Some files appear in multiple splits.")

    class_per_split = metadata.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    if (class_per_split == 0).any().any():
        raise ValueError("Each split must contain every class.")


def save_split_artifacts(dataset: EuroSAT, metadata: pd.DataFrame, config_path: str | Path | None = None) -> None:
    config = load_config(config_path)
    paths = get_dataset_paths(config_path)

    ensure_dir(paths.split_csv.parent)
    ensure_dir(paths.split_summary_json.parent)
    metadata.to_csv(paths.split_csv, index=False)
    save_json(dataset.class_to_idx, paths.class_map_json)

    split_counts = metadata.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    summary = {
        "seed": config["seed"],
        "dataset_size": int(len(metadata)),
        "split_sizes": {key: int(value) for key, value in metadata["split"].value_counts().to_dict().items()},
        "class_distribution_by_split": {
            split: {label: int(count) for label, count in counts.items()}
            for split, counts in split_counts.to_dict(orient="index").items()
        },
    }
    save_json(summary, paths.split_summary_json)


def compute_train_channel_stats(metadata: pd.DataFrame, config_path: str | Path | None = None) -> dict[str, list[float]]:
    config = load_config(config_path)
    paths = get_dataset_paths(config_path)
    train_rows = metadata.loc[metadata["split"] == "train"]

    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for relative_path in train_rows["relative_path"]:
        image_path = config["paths"]["data_raw"] / relative_path
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        flattened = image.reshape(-1, 3)
        sums += flattened.sum(axis=0)
        squared_sums += np.square(flattened).sum(axis=0)
        pixel_count += flattened.shape[0]

    mean = sums / pixel_count
    std = np.sqrt((squared_sums / pixel_count) - np.square(mean))
    stats = {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "num_train_images": int(len(train_rows)),
        "pixel_count": int(pixel_count),
    }
    save_json(stats, paths.train_stats_json)
    return stats


def build_preprocessing_manifest(config_path: str | Path | None = None) -> dict[str, dict[str, object]]:
    return {
        "v0_raw": {
            "description": "Minimal baseline with tensor conversion only.",
            "train_steps": ["Convert PIL image to tensor in [0, 1]."],
            "eval_steps": ["Convert PIL image to tensor in [0, 1]."],
            "rationale": "Keeps the baseline close to raw RGB patches and isolates model performance without stronger preprocessing.",
        },
        "v1_normalized": {
            "description": "Standardized pipeline with train-split RGB normalization.",
            "train_steps": ["Convert to tensor.", "Normalize with train-split mean and std."],
            "eval_steps": ["Convert to tensor.", "Normalize with train-split mean and std."],
            "rationale": "Improves optimization stability and gives a standard reference pipeline for later experiments.",
        },
        "v2_enhanced": {
            "description": "Contrast-enhanced pipeline with CLAHE and light satellite-safe augmentation.",
            "train_steps": [
                "Apply CLAHE on luminance in LAB color space.",
                "Random horizontal flip.",
                "Random vertical flip.",
                "Random 90-degree rotation.",
                "Convert to tensor.",
                "Normalize with train-split mean and std.",
            ],
            "eval_steps": [
                "Apply CLAHE on luminance in LAB color space.",
                "Convert to tensor.",
                "Normalize with train-split mean and std.",
            ],
            "rationale": "Boosts local contrast and uses orientation-safe augmentations suitable for land-cover patches. Denoising is omitted to avoid removing texture cues.",
        },
    }


def summarize_metadata(metadata: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset_size": int(len(metadata)),
        "class_counts": {key: int(value) for key, value in Counter(metadata["label_name"]).items()},
        "split_counts": {key: int(value) for key, value in Counter(metadata["split"]).items()},
    }
