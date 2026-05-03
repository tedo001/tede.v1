"""Standalone export helper. ``TEDE.export()`` is the recommended entrypoint."""

from __future__ import annotations

from pathlib import Path

from tede.utils import auto_device, get_logger

LOGGER = get_logger("tede.export")
SUPPORTED_FORMATS = {"onnx", "engine", "torchscript", "openvino", "coreml", "tflite"}


def export(
    weights: str,
    fmt: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    device: str | None = None,
    simplify: bool = True,
) -> str:
    """Export a TEDE/Ultralytics checkpoint to a deployable format.

    Returns the artifact path.
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Supported: {sorted(SUPPORTED_FORMATS)}")
    if not Path(weights).is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required for export.") from exc

    device = auto_device(device)
    LOGGER.info("Exporting %s to format=%s (imgsz=%d, device=%s)", weights, fmt, imgsz, device)
    try:
        artifact = YOLO(weights).export(
            format=fmt, imgsz=imgsz, half=half, dynamic=dynamic, simplify=simplify, device=device
        )
    except Exception as exc:
        LOGGER.exception("Export failed.")
        raise RuntimeError(f"Export failed: {exc}") from exc
    LOGGER.info("Exported artifact: %s", artifact)
    return str(artifact)
