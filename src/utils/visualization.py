from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_figure(path: str | Path, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def image_to_numpy(image) -> np.ndarray:
    if hasattr(image, "permute"):
        image = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return np.clip(image, 0.0, 1.0)


def denormalize_image(image: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    mean_array = np.asarray(list(mean)).reshape(1, 1, 3)
    std_array = np.asarray(list(std)).reshape(1, 1, 3)
    restored = image * std_array + mean_array
    return np.clip(restored, 0.0, 1.0)
