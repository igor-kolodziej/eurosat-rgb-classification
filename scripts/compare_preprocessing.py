from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.data.dataset import get_dataset_paths, load_split_metadata
from src.preprocessing.registry import get_transforms
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, save_json, save_text
from src.utils.reproducibility import set_global_seed
from src.utils.visualization import denormalize_image, image_to_numpy, save_figure


VARIANT_ORDER = ["v0_raw", "v1_normalized", "v2_enhanced"]


def sample_fixed_subset(metadata: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    sampled_frames = []
    for _, frame in metadata.groupby("label_name", sort=True):
        sampled_frames.append(frame.sample(min(len(frame), per_class), random_state=seed))
    return pd.concat(sampled_frames, ignore_index=True).sort_values(["label_name", "relative_path"]).reset_index(drop=True)


def apply_variant(image: Image.Image, variant_id: str) -> np.ndarray:
    transformed = get_transforms(variant_id=variant_id, split="val")(image.copy())
    array = image_to_numpy(transformed)
    if variant_id in {"v1_normalized", "v2_enhanced"}:
        stats = load_json(get_dataset_paths().train_stats_json)
        array = denormalize_image(array, stats["mean"], stats["std"])
    return np.clip(array, 0.0, 1.0)


def make_side_by_side_grid(subset: pd.DataFrame, data_root: Path, figures_dir: Path) -> None:
    selected = subset.groupby("label_name", group_keys=False).head(1).reset_index(drop=True)
    fig, axes = plt.subplots(len(selected), len(VARIANT_ORDER), figsize=(10, 2.5 * len(selected)))

    for row_idx, row in selected.iterrows():
        image = Image.open(data_root / row["relative_path"]).convert("RGB")
        for col_idx, variant_id in enumerate(VARIANT_ORDER):
            axes[row_idx, col_idx].imshow(apply_variant(image, variant_id))
            axes[row_idx, col_idx].set_title(f"{row['label_name']} | {variant_id}")
            axes[row_idx, col_idx].axis("off")
    save_figure(figures_dir / "variant_side_by_side.png")


def luminance_values(image_array: np.ndarray) -> np.ndarray:
    image_uint8 = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    return lab[..., 0].reshape(-1)


def make_histograms_and_summary(subset: pd.DataFrame, data_root: Path, figures_dir: Path, metrics_dir: Path) -> None:
    stats_rows = []
    plt.figure(figsize=(9, 5))

    for variant_id in VARIANT_ORDER:
        luminance = []
        for relative_path in subset["relative_path"]:
            image = Image.open(data_root / relative_path).convert("RGB")
            processed = apply_variant(image, variant_id)
            values = luminance_values(processed)
            luminance.extend(values.tolist())

        luminance_array = np.asarray(luminance, dtype=np.float32)
        sns.kdeplot(luminance_array, label=variant_id, fill=False)
        stats_rows.append(
            {
                "variant_id": variant_id,
                "mean_brightness": float(luminance_array.mean()),
                "std_brightness": float(luminance_array.std()),
                "contrast_proxy": float(np.percentile(luminance_array, 95) - np.percentile(luminance_array, 5)),
            }
        )

    plt.title("Luminance Distribution by Preprocessing Variant")
    plt.xlabel("Luminance (LAB L channel)")
    plt.ylabel("Density")
    plt.legend()
    save_figure(figures_dir / "variant_luminance_histograms.png")

    summary = pd.DataFrame(stats_rows)
    summary.to_csv(metrics_dir / "variant_summary.csv", index=False)
    save_json(summary.to_dict(orient="records"), metrics_dir / "variant_summary.json")


def build_interpretation() -> str:
    return "\n".join(
        [
            "# Preprocessing Comparison Interpretation",
            "",
            "- `v0_raw` preserves the untouched RGB patches and serves as the reference point for all downstream comparisons.",
            "- `v1_normalized` keeps the image appearance close to raw after denormalization, but stabilizes channel scaling for model optimization.",
            "- `v2_enhanced` increases local luminance contrast through CLAHE, which makes field boundaries and urban textures more explicit in several classes.",
            "- The contrast shift is moderate rather than destructive, so `v2_enhanced` is a reasonable preprocessing candidate for Person 2 to test against the baseline.",
        ]
    )


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    figures_dir = ensure_dir(config["paths"]["outputs_figures"] / "preprocessing")
    metrics_dir = ensure_dir(config["paths"]["outputs_metrics"] / "preprocessing")
    metadata = load_split_metadata()
    subset = sample_fixed_subset(
        metadata.loc[metadata["split"] == "val"],
        per_class=config["dataset"]["comparison_samples_per_class"],
        seed=config["seed"],
    )
    subset.to_csv(metrics_dir / "comparison_subset.csv", index=False)

    make_side_by_side_grid(subset, config["paths"]["data_raw"], figures_dir)
    make_histograms_and_summary(subset, config["paths"]["data_raw"], figures_dir, metrics_dir)
    save_text(build_interpretation(), metrics_dir / "preprocessing_interpretation.md")
    print("Preprocessing comparison artifacts generated.")


if __name__ == "__main__":
    main()
