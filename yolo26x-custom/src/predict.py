"""YOLO26x inference: image / video / directory / webcam.

Usage:
    python -m src.predict --weights runs/.../best.pt --source path/to/image.jpg
    python -m src.predict --weights runs/.../best.pt --source 0  # webcam
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils import auto_device, get_logger

LOGGER = get_logger("predict")


def predict(
    weights: str,
    source: Union[str, int],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[str] = None,
    save: bool = True,
    project: str = "runs/predict",
    name: str = "exp",
) -> List[Dict[str, Any]]:
    """Run YOLO26x inference and return a JSON-serializable detections list.

    Args:
        weights: Path to a trained ``.pt`` checkpoint.
        source: File path, glob, directory, URL, or webcam index.
        imgsz: Inference image size.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        device: Override device. Default = auto.
        save: Save annotated images / videos to disk.
        project: Output project directory.
        name: Run subdirectory name.

    Returns:
        A list of per-image detection dicts: ``{"image": ..., "detections": [...]}``.
    """
    if not Path(weights).is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required for prediction.") from exc

    device = auto_device(device)
    LOGGER.info("Loading weights %s on device %s", weights, device)
    model = YOLO(weights)

    try:
        results = model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=save,
            project=project,
            name=name,
            verbose=False,
        )
    except Exception as exc:
        LOGGER.exception("Prediction failed.")
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    payload: List[Dict[str, Any]] = []
    names = getattr(model, "names", {}) or {}
    for r in results:
        detections: List[Dict[str, Any]] = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy().tolist()
                clses = boxes.cls.cpu().numpy().astype(int).tolist()
                for box, c, k in zip(xyxy, confs, clses):
                    detections.append(
                        {
                            "bbox_xyxy": [float(x) for x in box],
                            "confidence": float(c),
                            "class_id": int(k),
                            "class_name": names.get(int(k), str(k)),
                        }
                    )
            except (AttributeError, ValueError) as exc:
                LOGGER.warning("Failed to parse boxes: %s", exc)
        payload.append({"image": getattr(r, "path", None), "detections": detections})

    LOGGER.info("Inference complete. %d image(s) processed.", len(payload))
    return payload


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(description="YOLO26x inference.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-save", action="store_true", help="Do not write annotated outputs")
    p.add_argument("--project", type=str, default="runs/predict")
    p.add_argument("--name", type=str, default="exp")
    return p.parse_args()


def main() -> None:
    """Module entrypoint."""
    args = parse_args()
    src: Union[str, int]
    src = int(args.source) if args.source.isdigit() else args.source
    try:
        results = predict(
            weights=args.weights,
            source=src,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=not args.no_save,
            project=args.project,
            name=args.name,
        )
        for r in results:
            LOGGER.info("%s -> %d detections", r["image"], len(r["detections"]))
    except Exception as exc:
        LOGGER.error("Prediction failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
