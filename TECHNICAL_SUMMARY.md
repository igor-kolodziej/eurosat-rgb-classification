# Technical Summary

## Dataset Assumptions

- Dataset: EuroSAT RGB only
- Total images: `27,000`
- Image size: `64x64`
- Channels: `3`
- Seed: `42`
- Split strategy: stratified `70/15/15`

Saved split contract:

- `data/splits/eurosat_rgb_split.csv`
- `data/splits/class_to_idx.json`
- `outputs/metrics/split_summary.json`

## Preprocessing Decisions

### `v0_raw`

- Keeps the baseline close to original RGB patches
- Uses `ToTensor()` only
- Useful as the clean reference point for downstream experiments

### `v1_normalized`

- Uses train-split RGB mean/std:
  - mean: `[0.3438, 0.3799, 0.4075]`
  - std: `[0.2023, 0.1367, 0.1154]`
- Rationale: stable optimization and a standard comparison point

### `v2_enhanced`

- Applies CLAHE on the luminance channel in LAB space
- Keeps augmentation satellite-safe:
  - horizontal flip
  - vertical flip
  - random 90 degree rotation
- Denoising was intentionally omitted because EuroSAT patches are small and texture differences are important for visually similar classes

## Compact Findings

- EuroSAT RGB patches are uniform in size, so preprocessing can stay simple.
- The sampled brightness analysis suggests illumination variation is present, which justifies normalization and contrast comparison.
- `Pasture` vs `HerbaceousVegetation` and `Residential` vs `Industrial` remain visually close enough to motivate preprocessing experiments.
- CLAHE increases local contrast without visibly destroying structure, so `v2_enhanced` is a valid candidate for Person 2 experiments.

## Baseline Classifier

- Model: small CNN from scratch
- Train variant: `v0_raw`
- Device in this run: `cpu`
- MacBook-friendly defaults:
  - batch size `64`
  - max epochs `8`
  - early stopping patience `3`

Results:

- Validation accuracy: `0.7467`
- Validation macro F1: `0.7290`
- Test accuracy: `0.7388`
- Test macro F1: `0.7213`

Saved baseline artifacts:

- `outputs/metrics/baseline/baseline_cnn_v0.json`
- `outputs/figures/baseline/confusion_matrix_v0.png`
- `outputs/figures/baseline/training_history_v0.png`
- `outputs/models/baseline_cnn_v0.pt`

## Handoff Instructions For Person 2

1. Reuse the frozen split in `data/splits/eurosat_rgb_split.csv`. Do not regenerate it.
2. Reuse the preprocessing registry in `src/preprocessing/registry.py` and the manifest in `outputs/metrics/preprocessing_variants.json`. Do not silently redefine variants.
3. Load data through `src/data/loaders.py`.
4. Treat the baseline metrics as the reference floor, not the final result.
5. Focus only on model-side experimentation next: stronger CNNs, transfer learning, and cross-variant comparison.

## Known Limitations

- The baseline is intentionally simple and not optimized for best possible accuracy.
- The EDA is compact and focused on preprocessing, not a full remote-sensing domain analysis.
- Preprocessing variants are applied on the fly rather than materializing transformed image copies to disk.
