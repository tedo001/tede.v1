"""Validation loop and mAP computation."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from tede.nn.model import label_offset
from tede.ops import compute_map
from tede.utils import get_logger

LOGGER = get_logger("tede.validator")


class Validator:
    """Run inference over a DataLoader and report mAP50 / mAP50-95."""

    def __init__(self, model: torch.nn.Module, loader: DataLoader, device: torch.device, arch: str = "retinanet") -> None:
        self.model = model
        self.loader = loader
        self.device = device
        self.arch = arch

    @torch.no_grad()
    def run(self, score_threshold: float = 0.05) -> Dict[str, Any]:
        self.model.eval()
        all_preds: list = []
        all_targets: list = []
        offset = label_offset(self.arch)
        for images, targets in self.loader:
            images = [img.to(self.device, non_blocking=True) for img in images]
            outputs = self.model(images)
            for out, tgt in zip(outputs, targets):
                keep = out["scores"] >= score_threshold
                all_preds.append({
                    "boxes": out["boxes"][keep].detach().cpu(),
                    "scores": out["scores"][keep].detach().cpu(),
                    "labels": (out["labels"][keep].detach().cpu() - offset),
                })
                all_targets.append({
                    "boxes": tgt["boxes"].detach().cpu(),
                    "labels": (tgt["labels"].detach().cpu() - offset),
                })
        return compute_map(all_preds, all_targets)
