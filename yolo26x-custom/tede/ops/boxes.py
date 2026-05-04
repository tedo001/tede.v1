"""Box utility wrappers (re-exports torchvision primitives)."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou as _tv_box_iou


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes. Shape (N, M)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    return _tv_box_iou(boxes1, boxes2)


def clip_boxes(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Clip xyxy boxes to ``(H, W)``."""
    H, W = image_size
    boxes = boxes.clone()
    boxes[:, 0::2].clamp_(min=0, max=W)
    boxes[:, 1::2].clamp_(min=0, max=H)
    return boxes
