from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.evaluation.metrics import classification_metrics
from src.models.baseline import BaselineCNN
from src.preprocessing.registry import get_transforms
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed


def test_baseline_forward_shape():
    model = BaselineCNN(num_classes=10)
    output = model(torch.rand(2, 3, 64, 64))
    assert output.shape == (2, 10)


def test_raw_transform_on_synthetic_image():
    image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    tensor = get_transforms("v0_raw", "test")(image)
    assert tensor.shape == (3, 64, 64)
    assert torch.isfinite(tensor).all()


def test_metrics_have_expected_bounds():
    classes = [f"class-{index}" for index in range(10)]
    truth = list(range(10))
    prediction = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    metrics = classification_metrics(truth, prediction, classes)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 10


def test_configuration_and_seed_are_stable():
    config = load_config()
    assert config["seed"] == 42
    assert config["dataset"]["split_ratios"] == {"train": 0.70, "val": 0.15, "test": 0.15}
    set_global_seed(42)
    first = np.random.random(3)
    set_global_seed(42)
    second = np.random.random(3)
    assert np.array_equal(first, second)
