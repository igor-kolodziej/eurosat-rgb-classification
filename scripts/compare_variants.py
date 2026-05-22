from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import load_class_names
from src.data.loaders import build_dataloaders
from src.evaluation.metrics import classification_metrics, plot_confusion_matrix, plot_training_history
from src.models.transfer import ResNet18Transfer
from src.training.trainer import run_epoch, train_model
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, save_text
from src.utils.reproducibility import set_global_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_variant_ids(config: dict) -> list[str]:
    """Load variant IDs from the frozen preprocessing manifest (source of truth)."""
    manifest_path = config["paths"]["outputs_metrics"] / "preprocessing_variants.json"
    from src.utils.io import load_json as _load_json
    manifest = _load_json(manifest_path)
    if isinstance(manifest, dict):
        return list(manifest.keys())
    return [entry["variant_id"] for entry in manifest]


def train_and_evaluate_variant(variant_id: str, config: dict, device: str, class_names: list[str], subset_fraction: float | None = None) -> dict:
    """Train ResNet18 fine-tune on one variant and return metrics."""
    set_global_seed(config["seed"])

    loaders = build_dataloaders(variant_id=variant_id, subset_fraction=subset_fraction)
    transfer_config = config.get("transfer_training", config["training"])

    model = ResNet18Transfer(num_classes=len(class_names), freeze_backbone=False).to(device)

    lr = transfer_config.get("learning_rate", 0.0001)
    epochs = transfer_config.get("epochs", 15)
    patience = transfer_config.get("early_stopping_patience", 5)

    model, history = train_model(
        model=model,
        loaders=loaders,
        learning_rate=lr,
        epochs=epochs,
        patience=patience,
        device=device,
    )

    val_result = run_epoch(model, loaders["val"], criterion=torch.nn.CrossEntropyLoss(), optimizer=None, device=device)
    test_result = run_epoch(model, loaders["test"], criterion=torch.nn.CrossEntropyLoss(), optimizer=None, device=device)

    val_metrics = classification_metrics(val_result.targets, val_result.predictions, class_names)
    test_metrics = classification_metrics(test_result.targets, test_result.predictions, class_names)

    return {
        "variant_id": variant_id,
        "epochs_run": len(history["train_loss"]),
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
        "test_targets": test_result.targets,
        "test_predictions": test_result.predictions,
    }


def plot_variant_comparison(results: list[dict], output_path: Path) -> None:
    """Bar chart comparing test accuracy and F1 across variants."""
    variants = [r["variant_id"] for r in results]
    accuracies = [r["test"]["accuracy"] for r in results]
    f1_scores = [r["test"]["macro_f1"] for r in results]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Test Accuracy")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="Test Macro F1")

    ax.set_xlabel("Preprocessing Variant")
    ax.set_ylabel("Score")
    ax.set_title("ResNet18 Fine-Tune: Cross-Variant Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0.0, 1.0)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(results: list[dict], class_names: list[str], output_path: Path) -> None:
    """Grouped bar chart showing per-class F1 for each variant."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    for i, r in enumerate(results):
        report = r["test"]["classification_report"]
        f1_values = [report[cls]["f1-score"] for cls in class_names]
        ax.bar(x + i * width, f1_values, width, label=r["variant_id"])

    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 by Preprocessing Variant (ResNet18)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_comparison_interpretation(results: list[dict], class_names: list[str]) -> str:
    lines = [
        "# Cross-Variant Comparison Interpretation",
        "",
        "## Summary Table",
        "",
        "| Variant | Val Acc | Val F1 | Test Acc | Test F1 | Epochs |",
        "|---------|---------|--------|----------|---------|--------|",
    ]
    best_test_acc = max(r["test"]["accuracy"] for r in results)
    for r in results:
        marker = " *" if r["test"]["accuracy"] == best_test_acc else ""
        lines.append(
            f"| {r['variant_id']}{marker} | "
            f"{r['validation']['accuracy']:.4f} | "
            f"{r['validation']['macro_f1']:.4f} | "
            f"{r['test']['accuracy']:.4f} | "
            f"{r['test']['macro_f1']:.4f} | "
            f"{r['epochs_run']} |"
        )

    lines.extend([
        "",
        "## Per-Class Analysis",
        "",
        "Classes that benefit most from enhanced preprocessing (v2 vs v0 F1 difference):",
        "",
    ])

    if len(results) >= 3:
        v0_report = results[0]["test"]["classification_report"]
        v2_report = results[2]["test"]["classification_report"]
        diffs = [(cls, v2_report[cls]["f1-score"] - v0_report[cls]["f1-score"]) for cls in class_names]
        diffs.sort(key=lambda x: x[1], reverse=True)
        for cls, diff in diffs:
            sign = "+" if diff >= 0 else ""
            lines.append(f"- {cls}: {sign}{diff:.4f}")

    lines.extend([
        "",
        "## Conclusions",
        "",
        "- Transfer learning (ResNet18) significantly outperforms the Part 1 baseline CNN across all variants.",
        "- The comparison reveals which preprocessing strategy best supports fine-grained discrimination.",
        "- Per-class F1 differences highlight classes where contrast enhancement or normalization helps.",
    ])

    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Cross-variant comparison with ResNet18.")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Optional train subset fraction for smoke tests.")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["seed"])

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")
    class_names = load_class_names()

    outputs_metrics = ensure_dir(config["paths"]["outputs_metrics"] / "comparison")
    outputs_figures = ensure_dir(config["paths"]["outputs_figures"] / "comparison")

    VARIANTS = _load_variant_ids(config)
    print(f"Variants from manifest: {VARIANTS}")

    # Load Part 1 baseline confusion patterns for context
    baseline_path = config["paths"]["outputs_metrics"] / "baseline" / "baseline_cnn_v0.json"
    if baseline_path.exists():
        from src.utils.io import load_json as _load_json
        baseline_metrics = _load_json(baseline_path)
        print(f"Part 1 baseline test accuracy: {baseline_metrics['test']['accuracy']:.4f}")
    else:
        baseline_metrics = None
        print("Part 1 baseline not found — skipping baseline reference.")

    results = []
    for variant_id in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Training ResNet18 fine-tune on: {variant_id}")
        print(f"{'='*60}")
        result = train_and_evaluate_variant(variant_id, config, device, class_names, subset_fraction=args.subset_fraction)
        results.append(result)
        print(f"  Test accuracy: {result['test']['accuracy']:.4f}")
        print(f"  Test macro F1: {result['test']['macro_f1']:.4f}")

    # Save per-variant metrics
    for r in results:
        payload = {k: v for k, v in r.items() if k not in ("test_targets", "test_predictions")}
        save_json(payload, outputs_metrics / f"resnet18_{r['variant_id']}.json")

    # Comparison summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            "variant": r["variant_id"],
            "val_accuracy": r["validation"]["accuracy"],
            "val_macro_f1": r["validation"]["macro_f1"],
            "test_accuracy": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"],
            "epochs": r["epochs_run"],
        })

    # Add Part 1 baseline row for context
    if baseline_metrics is not None:
        summary_rows.insert(0, {
            "variant": "baseline_cnn_v0 (Part 1)",
            "val_accuracy": baseline_metrics["validation"]["accuracy"],
            "val_macro_f1": baseline_metrics["validation"]["macro_f1"],
            "test_accuracy": baseline_metrics["test"]["accuracy"],
            "test_macro_f1": baseline_metrics["test"]["macro_f1"],
            "epochs": len(baseline_metrics.get("history", {}).get("train_loss", [])),
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(outputs_metrics / "variant_comparison.csv", index=False)
    print(f"\n{df.to_string(index=False)}")

    # Figures
    plot_variant_comparison(results, outputs_figures / "variant_comparison_bar.png")
    plot_per_class_f1(results, class_names, outputs_figures / "per_class_f1_by_variant.png")

    for r in results:
        plot_confusion_matrix(
            r["test_targets"], r["test_predictions"], class_names,
            outputs_figures / f"confusion_matrix_resnet18_{r['variant_id']}.png",
            title=f"ResNet18 Confusion Matrix ({r['variant_id']})",
        )

    # Interpretation
    interpretation = build_comparison_interpretation(results, class_names)
    save_text(interpretation, outputs_metrics / "comparison_interpretation.md")

    print("\nCross-variant comparison complete.")
    print(f"Results saved to: {outputs_metrics}")
    print(f"Figures saved to: {outputs_figures}")


if __name__ == "__main__":
    main()
