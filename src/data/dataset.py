from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io import load_json


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    split_csv: Path
    class_map_json: Path
    split_summary_json: Path
    train_stats_json: Path


def get_dataset_paths(config_path: str | Path | None = None) -> DatasetPaths:
    config = load_config(config_path)
    return DatasetPaths(
        root=config["paths"]["data_raw"],
        split_csv=config["paths"]["data_splits"] / "eurosat_rgb_split.csv",
        class_map_json=config["paths"]["data_splits"] / "class_to_idx.json",
        split_summary_json=config["paths"]["outputs_metrics"] / "split_summary.json",
        train_stats_json=config["paths"]["outputs_metrics"] / "train_channel_stats.json",
    )


def load_split_metadata(config_path: str | Path | None = None) -> pd.DataFrame:
    paths = get_dataset_paths(config_path)
    return pd.read_csv(paths.split_csv)


def load_class_names(config_path: str | Path | None = None) -> list[str]:
    class_to_idx = load_json(get_dataset_paths(config_path).class_map_json)
    return [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
