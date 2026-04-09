from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import (
    build_preprocessing_manifest,
    build_split_dataframe,
    compute_train_channel_stats,
    download_eurosat_dataset,
    save_split_artifacts,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json


def main() -> None:
    config = load_config()
    ensure_dir(config["paths"]["outputs_metrics"])

    dataset = download_eurosat_dataset()
    metadata = build_split_dataframe(dataset)
    save_split_artifacts(dataset, metadata)
    compute_train_channel_stats(metadata)
    save_json(
        build_preprocessing_manifest(),
        config["paths"]["outputs_metrics"] / "preprocessing_variants.json",
    )
    print("Dataset prepared successfully.")


if __name__ == "__main__":
    main()
