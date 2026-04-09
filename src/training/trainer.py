from __future__ import annotations

from dataclasses import dataclass
import sys

import torch
from torch import nn
from tqdm.auto import tqdm


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    targets: list[int]
    predictions: list[int]


def run_epoch(model, loader, criterion, optimizer=None, device: str = "cpu") -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.set_grad_enabled(is_training):
        for inputs, targets in tqdm(loader, leave=False, disable=not sys.stdout.isatty()):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions == targets).sum().item()
            total += batch_size
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    return EpochResult(
        loss=running_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        targets=all_targets,
        predictions=all_predictions,
    )


def train_model(
    model,
    loaders,
    learning_rate: float,
    epochs: int,
    patience: int,
    device: str = "cpu",
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(epochs):
        train_result = run_epoch(model, loaders["train"], criterion, optimizer=optimizer, device=device)
        val_result = run_epoch(model, loaders["val"], criterion, optimizer=None, device=device)

        history["train_loss"].append(train_result.loss)
        history["val_loss"].append(val_result.loss)
        history["train_accuracy"].append(train_result.accuracy)
        history["val_accuracy"].append(val_result.accuracy)

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    return model, history
