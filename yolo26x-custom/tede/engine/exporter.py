"""Export TEDE checkpoints to ONNX (and via ONNX, to TensorRT)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from tede.nn import build_model
from tede.utils import auto_device, get_logger

LOGGER = get_logger("tede.exporter")


def export_onnx(
    weights: str,
    output: Optional[str] = None,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = True,
    device: Optional[str] = None,
) -> str:
    """Export a TEDE checkpoint to ONNX. Returns the artifact path."""
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    out = Path(output) if output else weights_path.with_suffix(".onnx")

    dev = torch.device("cuda:0" if str(auto_device(device)) == "0" else auto_device(device))
    ckpt = torch.load(weights_path, map_location=dev)
    arch = ckpt.get("arch", "retinanet")
    num_classes = ckpt["num_classes"]
    model = build_model(num_classes=num_classes, arch=arch, pretrained_backbone=False)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=dev)
    dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}} if dynamic else None
    LOGGER.info("Exporting %s to %s (imgsz=%d, opset=%d)", weights, out, imgsz, opset)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    LOGGER.info("Wrote %s", out)
    return str(out)
