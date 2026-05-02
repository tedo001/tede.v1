"""YOLO26x evaluation: mAP, per-class metrics, confusion matrix and FPS benchmarks.

Usage:
    python -m src.evaluate --weights runs/detect/yolo26x_custom/weights/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils import auto_device, get_logger, load_yaml, save_json, save_yaml

LOGGER = get_logger("evaluate")


def resolve_dataset_yaml(data_path: str) -> str:
    """Materialize a sibling YAML with absolute paths so Ultralytics doesn't
    resolve a relative ``path:`` against its global ``datasets_dir``."""
    src = Path(data_path)
    if not src.is_file():
        return data_path
    src = src.resolve()
    cfg = load_yaml(src)
    base = src.parent
    if "path" in cfg and cfg["path"]:
        p = Path(cfg["path"])
        if not p.is_absolute():
            cfg["path"] = str((base / p).resolve())
    out = src.with_name(f".{src.stem}.resolved.yaml")
    save_yaml(cfg, out)
    return str(out)


def evaluate(
    weights: str,
    data: str = "configs/dataset.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    split: str = "val",
    output_dir: str = "runs/eval",
) -> Dict[str, Any]:
    """Run validation and return a metrics report.

    Args:
        weights: Path to a trained ``best.pt``.
        data: Dataset YAML path.
        imgsz: Image size.
        batch: Batch size.
        device: Compute device override.
        split: One of ``"val"`` | ``"test"``.
        output_dir: Directory to write the JSON report.

    Returns:
        Dict containing mAP50, mAP50-95, per-class metrics, confusion matrix path
        and FPS benchmarks for both available compute targets.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required for evaluation.") from exc

    if not Path(weights).is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    device = auto_device(device)
    LOGGER.info("Evaluating %s on split=%s (device=%s)", weights, split, device)

    data = resolve_dataset_yaml(data)
    model = YOLO(weights)
    try:
        metrics = model.val(
            data=data,
            imgsz=imgsz,
            batch=batch,
            device=device,
            split=split,
            plots=True,
            verbose=True,
        )
    except Exception as exc:
        LOGGER.exception("Validation failed.")
        raise RuntimeError(f"Validation failed: {exc}") from exc

    box = metrics.box
    report: Dict[str, Any] = {
        "weights": str(weights),
        "data": data,
        "split": split,
        "device": device,
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "mean_precision": float(box.mp),
        "mean_recall": float(box.mr),
    }

    # Per-class metrics
    names = getattr(model, "names", {}) or {}
    per_class: List[Dict[str, Any]] = []
    try:
        for i, cls_id in enumerate(box.ap_class_index):
            per_class.append(
                {
                    "class_id": int(cls_id),
                    "class_name": names.get(int(cls_id), str(cls_id)),
                    "precision": float(box.p[i]) if i < len(box.p) else None,
                    "recall": float(box.r[i]) if i < len(box.r) else None,
                    "mAP50": float(box.ap50[i]) if i < len(box.ap50) else None,
                    "mAP50-95": float(box.ap[i]) if i < len(box.ap) else None,
                }
            )
    except (AttributeError, IndexError) as exc:
        LOGGER.warning("Could not compute per-class breakdown: %s", exc)
    report["per_class"] = per_class

    save_dir = Path(getattr(metrics, "save_dir", output_dir))
    cm_path = save_dir / "confusion_matrix.png"
    if cm_path.is_file():
        report["confusion_matrix"] = str(cm_path)

    # Speed benchmark
    report["fps"] = benchmark_fps(model, imgsz=imgsz, device=device)

    out = Path(output_dir) / "evaluation_report.json"
    save_json(report, out)
    LOGGER.info("Evaluation report written to %s", out)
    LOGGER.info("mAP50=%.4f | mAP50-95=%.4f", report["mAP50"], report["mAP50-95"])
    return report


def benchmark_fps(model: Any, imgsz: int = 640, device: str = "cpu", warmup: int = 5, iters: int = 30) -> Dict[str, float]:
    """Benchmark inference speed in FPS on a synthetic input.

    Tries the requested device first then falls back to CPU. Reports both when
    a CUDA device is available so the report covers GPU and CPU.
    """
    results: Dict[str, float] = {}

    def _bench(dev: str) -> Optional[float]:
        try:
            dummy = np.random.randint(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)
            for _ in range(warmup):
                model.predict(dummy, device=dev, imgsz=imgsz, verbose=False)
            t0 = time.perf_counter()
            for _ in range(iters):
                model.predict(dummy, device=dev, imgsz=imgsz, verbose=False)
            elapsed = time.perf_counter() - t0
            return iters / elapsed if elapsed > 0 else 0.0
        except Exception as exc:
            LOGGER.warning("FPS benchmark failed on device=%s: %s", dev, exc)
            return None

    fps_primary = _bench(device)
    if fps_primary is not None:
        results[device] = round(fps_primary, 2)

    if device != "cpu":
        fps_cpu = _bench("cpu")
        if fps_cpu is not None:
            results["cpu"] = round(fps_cpu, 2)

    return results


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(description="Evaluate a YOLO26x model.")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p.add_argument("--data", type=str, default="configs/dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    p.add_argument("--output", type=str, default="runs/eval")
    return p.parse_args()


def main() -> None:
    """Module entrypoint."""
    args = parse_args()
    try:
        evaluate(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            output_dir=args.output,
        )
    except Exception as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
