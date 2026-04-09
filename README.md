# EuroSAT RGB Preprocessing Package

This repository implements the full 1 part of a two-person university project on satellite image preprocessing, plus one small baseline classification contribution. The scope is limited to the RGB version of EuroSAT and is designed to be reproducible, easy to explain, and realistic to run on a MacBook.

## What Is Included

- Reproducible EuroSAT RGB download and loading
- Frozen stratified train/validation/test split with saved metadata
- Compact preprocessing-focused EDA
- Three frozen preprocessing variants
- Technical preprocessing comparisons with saved figures and summary tables
- One lightweight CNN baseline trained on the raw/minimal variant
- Handoff documentation for Part 2

## Project Structure

```text
.
├── README.md
├── TECHNICAL_SUMMARY.md
├── requirements.txt
├── configs/
│   └── project.yaml
├── data/
│   ├── raw/
│   └── splits/
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_preprocessing_comparison.ipynb
│   └── 03_baseline_classification.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_eda.py
│   ├── compare_preprocessing.py
│   └── train_baseline.py
└── src/
    ├── data/
    ├── preprocessing/
    ├── models/
    ├── training/
    ├── evaluation/
    └── utils/
```

## Environment Setup

The local system Python is externally managed, so use a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .mplconfig .cache
export MPLCONFIGDIR="$PWD/.mplconfig"
export XDG_CACHE_HOME="$PWD/.cache"
```

These cache exports avoid Matplotlib/font-cache issues in sandboxed or MacBook-local environments.

## How To Run

Run the full pipeline in this order:

```bash
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.mplconfig"
export XDG_CACHE_HOME="$PWD/.cache"

python scripts/prepare_dataset.py
python scripts/run_eda.py
python scripts/compare_preprocessing.py
python scripts/train_baseline.py
```

## MacBook Note

- The baseline is intentionally lightweight and uses a small CNN from scratch.
- The default training budget in `configs/project.yaml` is MacBook-friendly: `8` epochs max with early stopping patience `3`.
- The training script automatically prefers Apple `mps` when available, then `cuda`, then `cpu`.
- `num_workers` is left at `0` to avoid common laptop multiprocessing issues.

## Frozen Split

The split is stratified and fixed with seed `42`.

- Dataset size: `27,000`
- Train: `18,899`
- Validation: `4,050`
- Test: `4,051`

Saved split artifacts:

- `data/splits/eurosat_rgb_split.csv`
- `data/splits/class_to_idx.json`
- `outputs/metrics/split_summary.json`

Person 2 should reuse these files exactly and must not resample the dataset.

## Preprocessing Variants

| Variant         | Purpose                         | Train Transform                                           | Validation/Test Transform                |
| --------------- | ------------------------------- | --------------------------------------------------------- | ---------------------------------------- |
| `v0_raw`        | Minimal baseline                | `ToTensor()`                                              | `ToTensor()`                             |
| `v1_normalized` | Standardized reference pipeline | `ToTensor()` + train-split normalization                  | `ToTensor()` + train-split normalization |
| `v2_enhanced`   | Contrast-enhanced candidate     | CLAHE + flips + random 90 degree rotation + normalization | CLAHE + normalization                    |

The frozen variant manifest is saved in `outputs/metrics/preprocessing_variants.json`.

## Main Saved Outputs

EDA:

- `outputs/figures/eda/class_counts.png`
- `outputs/figures/eda/representative_samples.png`
- `outputs/figures/eda/brightness_histogram.png`
- `outputs/figures/eda/confusable_classes.png`
- `outputs/metrics/eda/image_properties.json`
- `outputs/metrics/eda/eda_summary.md`

Preprocessing comparison:

- `outputs/figures/preprocessing/variant_side_by_side.png`
- `outputs/figures/preprocessing/variant_luminance_histograms.png`
- `outputs/metrics/preprocessing/variant_summary.csv`
- `outputs/metrics/preprocessing/preprocessing_interpretation.md`

Baseline classification:

- `outputs/metrics/baseline/baseline_cnn_v0.json`
- `outputs/metrics/baseline/baseline_interpretation.md`
- `outputs/figures/baseline/confusion_matrix_v0.png`
- `outputs/figures/baseline/training_history_v0.png`
- `outputs/models/baseline_cnn_v0.pt`

## Baseline Results

Baseline model: small CNN from scratch trained on `v0_raw`.

- Validation accuracy: `0.7467`
- Validation macro F1: `0.7290`
- Test accuracy: `0.7388`
- Test macro F1: `0.7213`

This is meant to validate the frozen data pipeline and give Person 2 a clean starting point, not to be the final modeling study.

## For Person 2

Use the existing split and preprocessing registry as fixed infrastructure.

1. Do not regenerate the split.
2. Do not change the definitions of `v0_raw`, `v1_normalized`, or `v2_enhanced`.
3. Load data through `src.data.loaders.build_dataloaders(variant_id, config_path=None)`.
4. Use `outputs/metrics/train_channel_stats.json` and `outputs/metrics/preprocessing_variants.json` as the source of truth.
5. Build new classifiers on top of the frozen split and preprocessing interface.

Recommended starting point for Person 2:

- Compare a stronger CNN or transfer-learning baseline across `v0_raw`, `v1_normalized`, and `v2_enhanced`
- Reuse the saved confusion matrix and preprocessing comparison to choose which classes need extra modeling attention

## Important Assumptions

- RGB-only EuroSAT
- Native patch size kept at `64x64`
- Fixed seed `42`
- No denoising in the enhanced pipeline because the patches are small and texture is important
- No presentation slides are included
