from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = resolve_path(config_path or "configs/project.yaml")
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    path_keys = {
        "data_raw",
        "data_splits",
        "outputs_root",
        "outputs_figures",
        "outputs_metrics",
        "outputs_models",
        "outputs_samples",
    }
    for key in path_keys:
        config["paths"][key] = resolve_path(config["paths"][key])

    return config
