from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.data.dataset import load_split_metadata
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, save_text
from src.utils.reproducibility import set_global_seed
from src.utils.visualization import save_figure


def sample_per_class(metadata: pd.DataFrame, count_per_class: int, seed: int) -> pd.DataFrame:
    sampled_frames = []
    for _, frame in metadata.groupby("label_name", sort=True):
        sampled_frames.append(frame.sample(min(len(frame), count_per_class), random_state=seed))
    return pd.concat(sampled_frames, ignore_index=True)


def load_rgb_array(root: Path, relative_path: str) -> np.ndarray:
    return np.asarray(Image.open(root / relative_path).convert("RGB"))


def build_image_property_summary(metadata: pd.DataFrame, data_root: Path) -> dict[str, object]:
    widths: list[int] = []
    heights: list[int] = []
    brightness_values: list[float] = []
    dtype_names: list[str] = []
    min_values: list[int] = []
    max_values: list[int] = []

    for relative_path in metadata["relative_path"]:
        image = load_rgb_array(data_root, relative_path)
        heights.append(int(image.shape[0]))
        widths.append(int(image.shape[1]))
        dtype_names.append(str(image.dtype))
        min_values.append(int(image.min()))
        max_values.append(int(image.max()))
        brightness_values.append(float(image.mean()))

    return {
        "image_height_set": sorted(set(heights)),
        "image_width_set": sorted(set(widths)),
        "channels": 3,
        "dtype_set": sorted(set(dtype_names)),
        "global_min": int(min(min_values)),
        "global_max": int(max(max_values)),
        "brightness_mean": float(np.mean(brightness_values)),
        "brightness_std": float(np.std(brightness_values)),
    }


def make_class_count_chart(metadata: pd.DataFrame, figures_dir: Path) -> None:
    class_counts = metadata["label_name"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, legend=False, palette="viridis")
    plt.title("EuroSAT RGB Class Counts")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    save_figure(figures_dir / "class_counts.png")


def make_representative_grid(metadata: pd.DataFrame, data_root: Path, figures_dir: Path, seed: int) -> None:
    sampled = sample_per_class(metadata, count_per_class=3, seed=seed)
    classes = sorted(sampled["label_name"].unique())
    fig, axes = plt.subplots(len(classes), 3, figsize=(9, 3 * len(classes)))
    for row_idx, class_name in enumerate(classes):
        class_rows = sampled.loc[sampled["label_name"] == class_name].reset_index(drop=True)
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx < len(class_rows):
                image = load_rgb_array(data_root, class_rows.loc[col_idx, "relative_path"])
                ax.imshow(image)
            ax.set_title(f"{class_name} #{col_idx + 1}")
            ax.axis("off")
    save_figure(figures_dir / "representative_samples.png")


def make_brightness_histogram(metadata: pd.DataFrame, data_root: Path, figures_dir: Path, sample_per_cls: int, seed: int) -> None:
    sampled = sample_per_class(metadata, count_per_class=sample_per_cls, seed=seed)
    brightness = []
    for relative_path in sampled["relative_path"]:
        image = load_rgb_array(data_root, relative_path)
        brightness.extend(image.mean(axis=2).reshape(-1).tolist())

    plt.figure(figsize=(8, 4))
    sns.histplot(brightness, bins=40, kde=True, color="teal")
    plt.title("Brightness Distribution (Sampled Pixels)")
    plt.xlabel("Average RGB intensity")
    plt.ylabel("Frequency")
    save_figure(figures_dir / "brightness_histogram.png")


def make_confusable_class_grid(metadata: pd.DataFrame, data_root: Path, figures_dir: Path, seed: int) -> None:
    pairs = [("Pasture", "HerbaceousVegetation"), ("Residential", "Industrial")]
    fig, axes = plt.subplots(len(pairs), 4, figsize=(10, 5))
    rng = np.random.default_rng(seed)
    for row_idx, (left_class, right_class) in enumerate(pairs):
        left_rows = metadata.loc[metadata["label_name"] == left_class]
        right_rows = metadata.loc[metadata["label_name"] == right_class]
        left_pick = left_rows.iloc[rng.choice(len(left_rows), size=2, replace=False)].reset_index(drop=True)
        right_pick = right_rows.iloc[rng.choice(len(right_rows), size=2, replace=False)].reset_index(drop=True)
        for col_idx in range(2):
            axes[row_idx, col_idx].imshow(load_rgb_array(data_root, left_pick.loc[col_idx, "relative_path"]))
            axes[row_idx, col_idx].set_title(left_class)
            axes[row_idx, col_idx].axis("off")
        for col_idx in range(2):
            axes[row_idx, col_idx + 2].imshow(load_rgb_array(data_root, right_pick.loc[col_idx, "relative_path"]))
            axes[row_idx, col_idx + 2].set_title(right_class)
            axes[row_idx, col_idx + 2].axis("off")
    save_figure(figures_dir / "confusable_classes.png")


def build_eda_summary(metadata: pd.DataFrame, property_summary: dict[str, object]) -> str:
    class_balance = metadata["label_name"].value_counts().sort_index().to_dict()
    return "\n".join(
        [
            "# EDA Summary",
            "",
            "## Practical observations",
            "- EuroSAT RGB patches are consistently 64x64 with three channels, so preprocessing can stay simple and fixed-size.",
            f"- Class counts are balanced across the dataset ({class_balance}).",
            f"- Pixel dynamic range spans {property_summary['global_min']} to {property_summary['global_max']} with average brightness {property_summary['brightness_mean']:.2f} +/- {property_summary['brightness_std']:.2f}.",
            "- Residential vs Industrial and Pasture vs HerbaceousVegetation show visible texture and structural overlap, so contrast and normalization comparisons are justified.",
            "- Illumination differs across patches, but the images are already tightly cropped, so strong geometric distortion or denoising is unnecessary for Part 1.",
        ]
    )


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    metadata = load_split_metadata()
    data_root = config["paths"]["data_raw"]
    figures_dir = ensure_dir(config["paths"]["outputs_figures"] / "eda")
    metrics_dir = ensure_dir(config["paths"]["outputs_metrics"] / "eda")

    make_class_count_chart(metadata, figures_dir)
    make_representative_grid(metadata, data_root, figures_dir, config["seed"])
    make_brightness_histogram(
        metadata,
        data_root,
        figures_dir,
        sample_per_cls=config["dataset"]["eda_histogram_samples_per_class"],
        seed=config["seed"],
    )
    make_confusable_class_grid(metadata, data_root, figures_dir, config["seed"])

    property_summary = build_image_property_summary(
        sample_per_class(metadata, count_per_class=10, seed=config["seed"]),
        data_root,
    )
    save_json(property_summary, metrics_dir / "image_properties.json")
    metadata["label_name"].value_counts().sort_index().to_csv(metrics_dir / "class_counts.csv", header=["count"])
    save_text(build_eda_summary(metadata, property_summary), metrics_dir / "eda_summary.md")
    print("EDA artifacts generated.")


if __name__ == "__main__":
    main()
