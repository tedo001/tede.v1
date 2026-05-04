"""Detection ops: NMS and metrics. Built on torchvision primitives only."""

from tede.ops.boxes import box_iou
from tede.ops.metrics import compute_map
from tede.ops.nms import multiclass_nms

__all__ = ["box_iou", "multiclass_nms", "compute_map"]
