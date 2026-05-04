"""TEDE model factory.

Wraps torchvision's detection models behind a single ``build_model`` call.
The supported architectures (``retinanet``, ``fcos``, ``fasterrcnn``,
``ssd``) are all standard PyTorch components — no Ultralytics, no YOLO
codebase — but they all consume the same ``(images, targets)`` training
contract and emit the same per-image ``{boxes, labels, scores}`` dicts at
inference, which lets the engine layer stay model-agnostic.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

SUPPORTED_ARCHS = ("retinanet", "fcos", "fasterrcnn", "ssd")


def build_model(num_classes: int, arch: str = "retinanet", pretrained_backbone: bool = True) -> nn.Module:
    """Construct a torchvision detection model with a fresh head.

    Args:
        num_classes: Number of foreground classes (matches YOLO ``nc``).
            Background is handled internally by the model when needed.
        arch: One of ``retinanet | fcos | fasterrcnn | ssd``.
        pretrained_backbone: If True, the backbone uses ImageNet-pretrained
            weights from torchvision. The detection head is always randomly
            initialised so it can be retrained on your classes.

    Returns:
        ``torch.nn.Module`` ready for training. During training call
        ``model(images, targets)`` and sum the returned loss dict; during
        eval call ``model(images)`` and parse the per-image dicts.
    """
    arch = arch.lower()
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {SUPPORTED_ARCHS}")

    weights_backbone = "DEFAULT" if pretrained_backbone else None

    if arch == "retinanet":
        from torchvision.models.detection import retinanet_resnet50_fpn_v2

        return retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes,
        )
    if arch == "fcos":
        from torchvision.models.detection import fcos_resnet50_fpn

        return fcos_resnet50_fpn(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes,
        )
    if arch == "fasterrcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        # FasterRCNN counts background as class 0, so pass num_classes + 1.
        return fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes + 1,
        )
    if arch == "ssd":
        from torchvision.models.detection import ssd300_vgg16

        # SSD also reserves class 0 for background.
        return ssd300_vgg16(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes + 1,
        )
    raise AssertionError("unreachable")  # pragma: no cover


def label_offset(arch: str) -> int:
    """Index offset to apply to YOLO class IDs for the given architecture.

    YOLO labels are zero-indexed in ``[0, nc-1]``. ``retinanet`` and ``fcos``
    take labels in that exact range; ``fasterrcnn`` and ``ssd`` reserve 0
    for background and expect labels in ``[1, nc]``.
    """
    return 1 if arch.lower() in {"fasterrcnn", "ssd"} else 0
