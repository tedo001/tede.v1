"""YOLO26x training entrypoint with MLflow experiment tracking.

Usage:
    python -m src.train --config configs/model.yaml
    python -m src.train --config configs/model.yaml --epochs 50 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import auto_device, get_logger, gpu_memory_mb, load_yaml, save_yaml

LOGGER = get_logger("train")


def resolve_dataset_yaml(data_path: str) -> str:
    """Rewrite a dataset YAML so all paths are absolute.

    Ultralytics resolves a relative ``path:`` against its global
    ``datasets_dir`` setting, which often points at an unrelated old project
    on Windows. To avoid that, we materialize a sibling YAML with absolute
    paths and pass that to Ultralytics instead. Pre-shipped names like
    ``coco8.yaml`` (no path separators, no .yaml on disk in cwd) are passed
    through unchanged so the registry still resolves them.
    """
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
    LOGGER.info("Resolved dataset YAML written to %s (path=%s)", out, cfg.get("path"))
    return str(out)


def build_train_kwargs(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config + CLI overrides into a kwargs dict for ``YOLO.train``.

    Keys ``model`` and ``cfg``/``mode`` are removed because Ultralytics
    expects them on the model loader / not at all.
    """
    merged = {**cfg, **{k: v for k, v in overrides.items() if v is not None}}
    merged.pop("model", None)
    merged.pop("mode", None)
    merged["device"] = auto_device(merged.get("device") or None)
    if "data" in merged and merged["data"]:
        merged["data"] = resolve_dataset_yaml(merged["data"])
    return merged


def train(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Path:
    """Train a YOLO26x model and return the path to the best checkpoint.

    Args:
        config_path: Path to a model config YAML (see ``configs/model.yaml``).
        overrides: Optional dict that overrides any key in the config.

    Returns:
        Filesystem path to ``best.pt``.
    """
    overrides = overrides or {}
    cfg = load_yaml(config_path)
    model_path = cfg.get("model", "yolo26x.pt")
    train_kwargs = build_train_kwargs(cfg, overrides)

    LOGGER.info("Loading YOLO26x model: %s", model_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        model = YOLO(model_path)
    except Exception as exc:
        LOGGER.exception("Failed to load model %s", model_path)
        raise RuntimeError(f"Could not load YOLO26x weights from {model_path}: {exc}") from exc

    # Optional MLflow tracking — soft dependency
    tracker = None
    try:
        from mlops.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(experiment_name="yolo26x_custom")
        tracker.start_run(run_name=train_kwargs.get("name", "yolo26x_run"))
        tracker.log_params(train_kwargs)
    except Exception as exc:  # pragma: no cover - optional path
        LOGGER.warning("MLflow tracking disabled: %s", exc)
        tracker = None

    LOGGER.info("Training kwargs: %s", train_kwargs)
    LOGGER.info("Pre-train CUDA memory: %s MB", gpu_memory_mb())

    try:
        results = model.train(**train_kwargs)
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user.")
        raise
    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        if tracker is not None:
            tracker.end_run(status="FAILED")
        raise

    LOGGER.info("Post-train CUDA memory: %s MB", gpu_memory_mb())

    save_dir = Path(getattr(results, "save_dir", train_kwargs.get("project", "runs/detect")))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    if tracker is not None:
        try:
            metrics = getattr(results, "results_dict", None) or {}
            tracker.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            for art in (best, last):
                if art.is_file():
                    tracker.log_artifact(str(art))
            tracker.end_run(status="FINISHED")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to log MLflow artifacts: %s", exc)

    if not best.is_file():
        LOGGER.warning("best.pt not found at %s; returning save_dir.", best)
        return save_dir
    LOGGER.info("Best checkpoint: %s", best)
    return best


def parse_args() -> argparse.Namespace:
    """Build the CLI."""
    p = argparse.ArgumentParser(description="Train YOLO26x on a custom dataset.")
    p.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    p.add_argument("--data", type=str, help="Override dataset YAML path")
    p.add_argument("--epochs", type=int, help="Override number of epochs")
    p.add_argument("--batch", type=int, help="Override batch size")
    p.add_argument("--imgsz", type=int, help="Override image size")
    p.add_argument("--device", type=str, help="Override device, e.g. '0', 'cpu', '0,1'")
    p.add_argument("--name", type=str, help="Override run name")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return p.parse_args()


def main() -> None:
    """Module entrypoint."""
    args = parse_args()
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k != "config" and v not in (None, False)
    }
    try:
        train(args.config, overrides=overrides)
    except SystemExit:
        raise
    except Exception as exc:
        LOGGER.error("Training pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
