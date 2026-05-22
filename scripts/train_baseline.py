from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import load_class_names
from src.data.loaders import build_dataloaders
from src.evaluation.metrics import classification_metrics, plot_confusion_matrix, plot_training_history
from src.models.baseline import BaselineCNN
from src.training.trainer import run_epoch, train_model
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, save_text
from src.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline EuroSAT CNN.")
    parser.add_argument("--variant", default="v0_raw", help="Preprocessing variant to train on.")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Optional train subset fraction for smoke tests.")
    return parser.parse_args()


def build_result_interpretation(metrics: dict[str, object]) -> str:
    val = metrics["validation"]
    test = metrics["test"]
    return "\n".join(
        [
            "# Baseline Interpretation",
            "",
            f"- Validation accuracy: {val['accuracy']:.4f}, macro F1: {val['macro_f1']:.4f}.",
            f"- Test accuracy: {test['accuracy']:.4f}, macro F1: {test['macro_f1']:.4f}.",
            "- This small CNN is intentionally modest: it validates the frozen split and dataloader setup rather than maximizing leaderboard performance.",
            "- Confusion patterns should be used by Part 2 to choose stronger models and decide whether enhanced preprocessing improves similar land-cover classes.",
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_config()
    set_global_seed(config["seed"])

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    loaders = build_dataloaders(variant_id=args.variant, subset_fraction=args.subset_fraction)
    class_names = load_class_names()

    model = BaselineCNN(num_classes=len(class_names)).to(device)
    model, history = train_model(
        model=model,
        loaders=loaders,
        learning_rate=config["training"]["learning_rate"],
        epochs=config["training"]["epochs"],
        patience=config["training"]["early_stopping_patience"],
        device=device,
    )

    val_result = run_epoch(model, loaders["val"], criterion=torch.nn.CrossEntropyLoss(), optimizer=None, device=device)
    test_result = run_epoch(model, loaders["test"], criterion=torch.nn.CrossEntropyLoss(), optimizer=None, device=device)

    val_metrics = classification_metrics(val_result.targets, val_result.predictions, class_names)
    test_metrics = classification_metrics(test_result.targets, test_result.predictions, class_names)

    outputs_models = ensure_dir(config["paths"]["outputs_models"])
    outputs_metrics = ensure_dir(config["paths"]["outputs_metrics"] / "baseline")
    outputs_figures = ensure_dir(config["paths"]["outputs_figures"] / "baseline")
    torch.save(model.state_dict(), outputs_models / "baseline_cnn_v0.pt")

    result_payload = {
        "variant_id": args.variant,
        "device": device,
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_json(result_payload, outputs_metrics / "baseline_cnn_v0.json")
    save_text(build_result_interpretation(result_payload), outputs_metrics / "baseline_interpretation.md")
    plot_confusion_matrix(test_result.targets, test_result.predictions, class_names, outputs_figures / "confusion_matrix_v0.png", title="Baseline CNN Confusion Matrix")
    plot_training_history(history, outputs_figures / "training_history_v0.png")
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
