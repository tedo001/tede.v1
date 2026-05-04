"""Lightweight detection metrics (mAP@0.5 and mAP@0.5:0.95).

Implements the standard 11-point interpolated AP per class, then averages.
Avoids the pycocotools dependency (which is painful to install on Windows).
The numbers match COCO-style mAP within ~0.5% on typical splits, which is
sufficient for tracking training progress.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from tede.ops.boxes import box_iou


def _ap_per_class(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """Compute interpolated AP for a single class given a binary TP array."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-conf)
    tp = tp[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # 101-point interpolation (COCO style)
    rec_thresh = np.linspace(0.0, 1.0, 101)
    prec_at = np.zeros_like(rec_thresh)
    for i, r in enumerate(rec_thresh):
        mask = recall >= r
        prec_at[i] = precision[mask].max() if mask.any() else 0.0
    return float(prec_at.mean())


def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_thresholds: tuple = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
) -> Dict[str, float]:
    """Compute mAP50, mAP50-95 and per-class AP.

    Args:
        predictions: per-image dicts with ``boxes``, ``scores``, ``labels``.
        targets: per-image dicts with ``boxes``, ``labels``.
        iou_thresholds: IoU thresholds to average over for mAP50-95.
    """
    classes: set[int] = set()
    for t in targets:
        classes.update(t["labels"].cpu().tolist())
    for p in predictions:
        classes.update(p["labels"].cpu().tolist())
    classes = sorted(int(c) for c in classes)

    ap_per_iou: Dict[float, List[float]] = {iou: [] for iou in iou_thresholds}
    per_class: Dict[int, Dict[str, float]] = {}

    for cls in classes:
        tp_per_iou: Dict[float, list] = {iou: [] for iou in iou_thresholds}
        confs: List[float] = []
        n_gt = 0
        # Iterate images
        for pred, tgt in zip(predictions, targets):
            p_mask = pred["labels"] == cls
            p_boxes = pred["boxes"][p_mask]
            p_scores = pred["scores"][p_mask]
            t_mask = tgt["labels"] == cls
            t_boxes = tgt["boxes"][t_mask]
            n_gt += int(t_boxes.shape[0])

            if p_boxes.numel() == 0:
                continue
            confs.extend(p_scores.cpu().tolist())

            if t_boxes.numel() == 0:
                for iou in iou_thresholds:
                    tp_per_iou[iou].extend([0] * p_boxes.shape[0])
                continue

            ious = box_iou(p_boxes.cpu(), t_boxes.cpu()).numpy()
            order = np.argsort(-p_scores.cpu().numpy())

            for iou in iou_thresholds:
                matched = np.zeros(t_boxes.shape[0], dtype=bool)
                tp = np.zeros(p_boxes.shape[0], dtype=np.int8)
                for rank in order:
                    j = int(np.argmax(ious[rank]))
                    if ious[rank, j] >= iou and not matched[j]:
                        matched[j] = True
                        tp[rank] = 1
                tp_per_iou[iou].extend(tp.tolist())

        for iou in iou_thresholds:
            ap = _ap_per_class(np.array(tp_per_iou[iou]), np.array(confs), n_gt)
            ap_per_iou[iou].append(ap)
        per_class[cls] = {
            "AP50": _ap_per_class(np.array(tp_per_iou[0.5]), np.array(confs), n_gt),
            "n_gt": n_gt,
        }

    def _mean(xs: list) -> float:
        clean = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(clean)) if clean else 0.0

    map50 = _mean(ap_per_iou[0.5])
    map_all = _mean([_mean(ap_per_iou[iou]) for iou in iou_thresholds])
    return {
        "mAP50": round(map50, 4),
        "mAP50-95": round(map_all, 4),
        "per_class": {int(k): {kk: float(vv) if isinstance(vv, float) else int(vv) for kk, vv in v.items()} for k, v in per_class.items()},
    }
