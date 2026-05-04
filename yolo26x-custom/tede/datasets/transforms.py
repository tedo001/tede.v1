"""Detection-aware augmentations.

Each transform takes ``(image, target)`` and returns ``(image, target)``,
keeping bounding boxes synchronised with the underlying pixel content.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    """Chain transforms together."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """Letterbox-style resize that preserves boxes."""

    def __init__(self, size: int = 640) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        W, H = image.size
        s = self.size / max(W, H)
        new_w, new_h = int(round(W * s)), int(round(H * s))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        # Pad to square
        canvas = Image.new("RGB", (self.size, self.size), color=(114, 114, 114))
        canvas.paste(image, (0, 0))
        if target.get("boxes") is not None and target["boxes"].numel() > 0:
            target["boxes"] = target["boxes"] * s
        return canvas, target


class RandomHorizontalFlip:
    """Horizontal flip with bounding-box mirroring."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        if random.random() < self.p:
            W, _ = image.size
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if target.get("boxes") is not None and target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class ColorJitter:
    """Brightness/contrast/saturation jitter (no effect on boxes)."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2) -> None:
        self.t = __import__("torchvision").transforms.ColorJitter(brightness, contrast, saturation)

    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        return self.t(image), target


class ToTensor:
    """PIL -> torch.FloatTensor in [0, 1] (C, H, W)."""

    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        return TF.to_tensor(image), target


def build_transforms(image_size: int = 640, training: bool = True) -> Compose:
    """Standard train/eval transform pipelines."""
    if training:
        return Compose([
            Resize(image_size),
            RandomHorizontalFlip(0.5),
            ColorJitter(0.2, 0.2, 0.2),
            ToTensor(),
        ])
    return Compose([Resize(image_size), ToTensor()])
