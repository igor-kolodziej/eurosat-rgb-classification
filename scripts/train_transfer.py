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
from src.models.transfer import ResNet18Transfer
from src.training.trainer import run_epoch, train_model
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, save_text
from src.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 transfer model on EuroSAT.")
    parser.add_argument("--variant", default="v1_normalized", help="Preprocessing variant.")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Optional train subset fraction for smoke tests.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet backbone (linear probe).")
    return parser.parse_args()


def build_result_interpretation(metrics: dict, variant_id: str, freeze: bool) -> str:
    val = metrics["validation"]
    test = metrics["test"]
    mode = "linear probe (frozen backbone)" if freeze else "full fine-tune"
    return "\n".join(
        [
            "# Transfer Learning Interpretation",
            "",
            f"- Model: ResNet18 pretrained ({mode})",
            f"- Variant: `{variant_id}`",
            f"- Validation accuracy: {val['accuracy']:.4f}, macro F1: {val['macro_f1']:.4f}.",
            f"- Test accuracy: {test['accuracy']:.4f}, macro F1: {test['macro_f1']:.4f}.",
            "- Transfer learning from ImageNet provides strong feature representations even for satellite imagery.",
            "- Compare across variants to determine whether preprocessing improves classification of confusable classes.",
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_config()
    set_global_seed(config["seed"])

    transfer_config = config.get("transfer_training", config["training"])

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"Variant: {args.variant}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    loaders = build_dataloaders(variant_id=args.variant, subset_fraction=args.subset_fraction)
    class_names = load_class_names()

    model = ResNet18Transfer(num_classes=len(class_names), freeze_backbone=args.freeze_backbone).to(device)

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

    mode_tag = "frozen" if args.freeze_backbone else "finetune"
    model_name = f"resnet18_{mode_tag}_{args.variant}"

    outputs_models = ensure_dir(config["paths"]["outputs_models"])
    outputs_metrics = ensure_dir(config["paths"]["outputs_metrics"] / "transfer")
    outputs_figures = ensure_dir(config["paths"]["outputs_figures"] / "transfer")

    torch.save(model.state_dict(), outputs_models / f"{model_name}.pt")

    result_payload = {
        "model_name": model_name,
        "variant_id": args.variant,
        "freeze_backbone": args.freeze_backbone,
        "device": device,
        "learning_rate": lr,
        "epochs_run": len(history["train_loss"]),
        "history": history,
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_json(result_payload, outputs_metrics / f"{model_name}.json")
    save_text(
        build_result_interpretation(result_payload, args.variant, args.freeze_backbone),
        outputs_metrics / f"{model_name}_interpretation.md",
    )
    plot_confusion_matrix(
        test_result.targets, test_result.predictions, class_names,
        outputs_figures / f"confusion_matrix_{model_name}.png",
        title=f"ResNet18 Confusion Matrix ({args.variant})",
    )
    plot_training_history(history, outputs_figures / f"training_history_{model_name}.png")

    print(f"Transfer training complete: {model_name}")
    print(f"  Val accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
