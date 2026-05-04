"""Non-maximum suppression."""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.ops import batched_nms


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    top_k: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter low-score detections, run class-wise NMS, return top-k.

    Args:
        boxes: (N, 4) xyxy.
        scores: (N,).
        labels: (N,) int64.
        iou_threshold: IoU above which overlapping boxes are suppressed.
        score_threshold: Drop detections below this confidence first.
        top_k: Keep at most this many detections per image.
    """
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep_idx = batched_nms(boxes, scores, labels, iou_threshold)
    keep_idx = keep_idx[:top_k]
    return boxes[keep_idx], scores[keep_idx], labels[keep_idx]
