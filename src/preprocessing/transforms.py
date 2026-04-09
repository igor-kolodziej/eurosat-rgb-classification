from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)


class RandomRotate90:
    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.choice((0, 90, 180, 270))
        if angle == 0:
            return image
        return image.rotate(angle)
