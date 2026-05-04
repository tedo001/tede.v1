"""PyTorch ``Dataset`` for YOLO-format detection data.

Each item is a tuple ``(image, target)`` where:

- ``image`` is a ``torch.FloatTensor`` of shape (3, H, W) in [0, 1].
- ``target`` is a dict with ``boxes`` (xyxy in pixels) and ``labels``
  (int64). This is exactly what torchvision detection models consume.

Labels on disk follow the standard YOLO convention: one ``.txt`` per
image, lines ``class_id cx cy w h`` normalised to ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from tede.utils import get_logger, load_yaml

LOGGER = get_logger("tede.datasets")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_split_paths(data_yaml: str, split: str) -> Tuple[Path, List[Path]]:
    """Read a YOLO dataset YAML and resolve the image directory for ``split``."""
    cfg = load_yaml(data_yaml)
    base = Path(cfg.get("path", "")).expanduser()
    if not base.is_absolute():
        base = (Path(data_yaml).resolve().parent / base).resolve()
    img_subdir = cfg.get(split)
    if img_subdir is None:
        raise KeyError(f"split '{split}' missing in dataset YAML {data_yaml}")
    images_dir = base / img_subdir if not Path(img_subdir).is_absolute() else Path(img_subdir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found for split '{split}': {images_dir}")
    images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    return images_dir, images


def _label_path_for(image_path: Path, images_root: Path) -> Path:
    """Mirror the images/<split>/... structure under labels/<split>/..."""
    rel = image_path.relative_to(images_root)
    labels_root = images_root.parent.parent / "labels" / images_root.name
    return labels_root / rel.with_suffix(".txt")


class YOLODataset(Dataset):
    """YOLO-format detection dataset compatible with torchvision detectors.

    Args:
        data_yaml: Path to a YOLO dataset YAML (with ``path``, ``train``,
            ``val``, ``test``, ``nc``, ``names``).
        split: ``"train"`` | ``"val"`` | ``"test"``.
        transforms: Optional callable applied as
            ``image, target = transforms(image, target)``.
        label_offset: Added to every class id (``0`` for RetinaNet/FCOS,
            ``1`` for FasterRCNN/SSD that reserve 0 for background).
    """

    def __init__(
        self,
        data_yaml: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        label_offset: int = 0,
    ) -> None:
        self.data_yaml = data_yaml
        self.split = split
        self.transforms = transforms
        self.label_offset = int(label_offset)

        cfg = load_yaml(data_yaml)
        self.num_classes = int(cfg["nc"])
        self.class_names = cfg.get("names", {})
        if isinstance(self.class_names, list):
            self.class_names = {i: n for i, n in enumerate(self.class_names)}

        self.images_root, self.images = _load_split_paths(data_yaml, split)
        if not self.images:
            raise RuntimeError(f"No images found for split '{split}' under {self.images_root}")
        LOGGER.info("YOLODataset[%s] loaded %d images from %s", split, len(self.images), self.images_root)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.images[idx]
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        W, H = image.size

        boxes: List[List[float]] = []
        labels: List[int] = []
        lbl_path = _label_path_for(img_path, self.images_root)
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    continue
                x1 = max(0.0, (cx - w / 2) * W)
                y1 = max(0.0, (cy - h / 2) * H)
                x2 = min(float(W), (cx + w / 2) * W)
                y2 = min(float(H), (cy + h / 2) * H)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + self.label_offset)

        target: Dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([H, W], dtype=torch.int64),
            "path": str(img_path),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Variable-size collate for detection: keep images and targets as lists."""
    images, targets = zip(*batch)
    return list(images), list(targets)
