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
- CLAHE increases local contrast without visibly destroying structure, so `v2_enhanced` is a valid candidate for Part 2 experiments.

## Baseline Classifier (Part 1)

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

## Transfer Learning Classifier (Part 2)

- Model: ResNet18 pretrained on ImageNet, fully fine-tuned
- Trained independently on each of the three frozen preprocessing variants
- Training defaults:
  - batch size `64`
  - max epochs `15`
  - learning rate `0.0001` (Adam)
  - early stopping patience `5`
- Variant list loaded from `outputs/metrics/preprocessing_variants.json` (source of truth)

### Cross-Variant Results

| Variant | Val Acc | Val F1 | Test Acc | Test F1 |
|---------|---------|--------|----------|--------|
| `v0_raw` | 0.8338 | 0.8262 | 0.8400 | 0.8329 |
| `v1_normalized` | 0.8183 | 0.8111 | 0.8398 | 0.8317 |
| `v2_enhanced` | 0.8837 | 0.8806 | 0.8953 | 0.8923 |

Best variant: `v2_enhanced`. CLAHE contrast enhancement combined with satellite-safe augmentation produces the highest accuracy and F1.

### Per-Class Findings (v2 vs v0 F1 Improvement)

- Highway: +0.1087
- River: +0.1033
- PermanentCrop: +0.0908
- Pasture: +0.0632
- Industrial: +0.0543
- Residential: +0.0499

Classes with subtle texture differences (Highway, River, PermanentCrop) benefit most from CLAHE. The previously identified confusable pairs (Pasture/HerbaceousVegetation, Residential/Industrial) also show meaningful improvement.

### Improvement Over Part 1 Baseline

- Best Part 2 model (`v2_enhanced`): test accuracy `0.8953` vs Part 1 baseline `0.7388`, a **+15.6 percentage point** gain.
- Transfer learning alone (v0_raw, no preprocessing change): test accuracy `0.8400`, a **+10.1 point** gain. This confirms that the model architecture contributes most.
- Preprocessing adds another **+5.5 points** on top of the architecture improvement.

Saved transfer artifacts:

- `outputs/metrics/transfer/resnet18_finetune_<variant>.json`
- `outputs/figures/transfer/confusion_matrix_resnet18_finetune_<variant>.png`
- `outputs/figures/transfer/training_history_resnet18_finetune_<variant>.png`
- `outputs/models/resnet18_finetune_<variant>.pt`

Saved comparison artifacts:

- `outputs/metrics/comparison/variant_comparison.csv`
- `outputs/metrics/comparison/comparison_interpretation.md`
- `outputs/figures/comparison/variant_comparison_bar.png`
- `outputs/figures/comparison/per_class_f1_by_variant.png`
- `outputs/figures/comparison/confusion_matrix_resnet18_<variant>.png`

## Part 2 Design Decisions

1. Reused the frozen split and preprocessing registry from Part 1 without modification.
2. Variant list loaded dynamically from `outputs/metrics/preprocessing_variants.json` rather than hard-coded.
3. Part 1 baseline metrics loaded as the reference floor for improvement analysis.
4. ResNet18 chosen over deeper architectures as a practical trade-off: strong ImageNet features while remaining laptop-friendly.
5. Full fine-tuning preferred over linear probing because the domain gap between ImageNet and satellite imagery is non-trivial.
6. Each variant trained independently with the same seed for fair comparison.

## Known Limitations

- The Part 1 baseline is intentionally simple and not optimized for best possible accuracy.
- The EDA is compact and focused on preprocessing, not a full remote-sensing domain analysis.
- Preprocessing variants are applied on the fly rather than materializing transformed image copies to disk.
- Part 2 results were obtained with full fine-tuning only; linear probing and deeper architectures (ResNet50, EfficientNet) remain unexplored.
- No learning rate scheduling or warm-up was applied; these could further improve convergence.
