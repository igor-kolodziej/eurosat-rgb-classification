# EuroSAT RGB Preprocessing & Classification Package

This repository implements a two-part university project on satellite image preprocessing and classification. The scope is limited to the RGB version of EuroSAT and is designed to be reproducible, easy to explain, and realistic to run on a MacBook.

## What Is Included

### Part 1: Preprocessing & Baseline

- Reproducible EuroSAT RGB download and loading
- Frozen stratified train/validation/test split with saved metadata
- Compact preprocessing-focused EDA
- Three frozen preprocessing variants
- Technical preprocessing comparisons with saved figures and summary tables
- One lightweight CNN baseline trained on the raw/minimal variant

### Part 2: Transfer Learning & Cross-Variant Comparison

- ResNet18 transfer learning model (pretrained ImageNet, fine-tuned)
- Cross-variant comparison training across all three preprocessing variants
- Per-class F1 analysis identifying which classes benefit from enhanced preprocessing
- Comparison figures, confusion matrices, and interpretation reports

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
│   ├── 03_baseline_classification.ipynb
│   └── 04_part2_classification.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_eda.py
│   ├── compare_preprocessing.py
│   ├── train_baseline.py
│   ├── train_transfer.py
│   └── compare_variants.py
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

# Part 1
python scripts/prepare_dataset.py
python scripts/run_eda.py
python scripts/compare_preprocessing.py
python scripts/train_baseline.py

# Part 2
python scripts/train_transfer.py --variant v1_normalized
python scripts/compare_variants.py
```

To run a quick smoke test with reduced data:

```bash
python scripts/train_transfer.py --variant v1_normalized --subset-fraction 0.05
python scripts/compare_variants.py --subset-fraction 0.05
```

## MacBook Note

- The Part 1 baseline is intentionally lightweight and uses a small CNN from scratch.
- Part 2 uses ResNet18 which is still laptop-friendly (~11M parameters).
- The default training budget in `configs/project.yaml`:
  - Part 1 baseline: `8` epochs max with early stopping patience `3`.
  - Part 2 transfer: `15` epochs max with early stopping patience `5`.
- The training scripts automatically prefer Apple `mps` when available, then `cuda`, then `cpu`.
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

Part 2 should reuse these files exactly and must not resample the dataset.

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

Baseline classification (Part 1):

- `outputs/metrics/baseline/baseline_cnn_v0.json`
- `outputs/metrics/baseline/baseline_interpretation.md`
- `outputs/figures/baseline/confusion_matrix_v0.png`
- `outputs/figures/baseline/training_history_v0.png`
- `outputs/models/baseline_cnn_v0.pt`

Transfer learning (Part 2):

- `outputs/metrics/transfer/resnet18_finetune_<variant>.json`
- `outputs/figures/transfer/confusion_matrix_resnet18_finetune_<variant>.png`
- `outputs/figures/transfer/training_history_resnet18_finetune_<variant>.png`
- `outputs/models/resnet18_finetune_<variant>.pt`

Cross-variant comparison (Part 2):

- `outputs/metrics/comparison/variant_comparison.csv`
- `outputs/metrics/comparison/comparison_interpretation.md`
- `outputs/figures/comparison/variant_comparison_bar.png`
- `outputs/figures/comparison/per_class_f1_by_variant.png`
- `outputs/figures/comparison/confusion_matrix_resnet18_<variant>.png`

## Baseline Results (Part 1)

Baseline model: small CNN from scratch trained on `v0_raw`.

- Validation accuracy: `0.7467`
- Validation macro F1: `0.7290`
- Test accuracy: `0.7388`
- Test macro F1: `0.7213`

This validates the frozen data pipeline and gives Part 2 a clean starting point.

## Transfer Learning Results (Part 2)

Model: ResNet18 pretrained (ImageNet), fully fine-tuned on each variant.

| Variant | Val Acc | Val F1 | Test Acc | Test F1 |
|---------|---------|--------|----------|--------|
| `v0_raw` | 0.8338 | 0.8262 | 0.8400 | 0.8329 |
| `v1_normalized` | 0.8183 | 0.8111 | 0.8398 | 0.8317 |
| `v2_enhanced` | 0.8837 | 0.8806 | **0.8953** | **0.8923** |

Key findings:

- Transfer learning improves test accuracy by +10.1 points over the Part 1 baseline even without preprocessing changes.
- `v2_enhanced` (CLAHE + augmentation) adds another +5.5 points on top of the architecture improvement.
- Classes with subtle texture differences (Highway, River, PermanentCrop) benefit most from CLAHE.
- The previously identified confusable pairs (Pasture/HerbaceousVegetation, Residential/Industrial) also show meaningful F1 gains.

## Infrastructure Contracts

The following are frozen and must not be modified:

1. The stratified split in `data/splits/eurosat_rgb_split.csv`.
2. The preprocessing variant definitions in `src/preprocessing/registry.py`.
3. The variant manifest in `outputs/metrics/preprocessing_variants.json`.
4. Channel statistics in `outputs/metrics/train_channel_stats.json`.
5. Data loading interface: `src.data.loaders.build_dataloaders(variant_id, config_path=None)`.

## Important Assumptions

- RGB-only EuroSAT
- Native patch size kept at `64x64`
- Fixed seed `42`
- No denoising in the enhanced pipeline because the patches are small and texture is important
- No presentation slides are included
