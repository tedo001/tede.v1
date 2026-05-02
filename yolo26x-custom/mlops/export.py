"""Export trained YOLO26x weights to ONNX / TensorRT / TorchScript.

Usage:
    python -m mlops.export --weights runs/.../best.pt --format onnx
    python -m mlops.export --weights runs/.../best.pt --format engine --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import auto_device, get_logger

LOGGER = get_logger("mlops.export")
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
    """Export a YOLO26x model to a deployable runtime format.

    Returns the path of the generated artifact.
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

    model = YOLO(weights)
    try:
        artifact = model.export(
            format=fmt,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            device=device,
        )
    except Exception as exc:
        LOGGER.exception("Export failed.")
        raise RuntimeError(f"Export failed: {exc}") from exc

    LOGGER.info("Exported artifact: %s", artifact)
    return str(artifact)


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(description="Export YOLO26x weights.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--format", type=str, default="onnx", choices=sorted(SUPPORTED_FORMATS))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-simplify", action="store_true")
    return p.parse_args()


def main() -> None:
    """Module entrypoint."""
    args = parse_args()
    try:
        export(
            weights=args.weights,
            fmt=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            device=args.device,
            simplify=not args.no_simplify,
        )
    except Exception as exc:
        LOGGER.error("Export pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
