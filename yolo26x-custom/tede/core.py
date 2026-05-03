"""TEDE: a thin orchestration layer on top of Ultralytics with MLOps hooks.

The class wraps ``ultralytics.YOLO`` so users get a single entrypoint for
train / val / predict / export / serve / register, plus optional MLflow
tracking and a local model registry — without losing access to the underlying
Ultralytics power-user API via ``TEDE.yolo``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tede.utils import auto_device, get_logger, gpu_memory_mb

LOGGER = get_logger("tede")


class TEDE:
    """High-level TEDE detector.

    Examples:
        >>> from tede import TEDE
        >>> model = TEDE("yolo26s.pt")
        >>> model.train(data="data.yaml", epochs=50, batch=8)
        >>> metrics = model.val(data="data.yaml")
        >>> dets = model.predict("image.jpg")
        >>> model.export(format="onnx")
    """

    def __init__(self, weights: str = "yolo26s.pt") -> None:
        """Construct a TEDE detector.

        Args:
            weights: Path to a ``.pt`` checkpoint, or a name like
                ``yolo26n.pt|yolo26s.pt|yolo26m.pt|yolo26l.pt|yolo26x.pt``.
                Ultralytics auto-downloads the official weights if missing.
        """
        self.weights = weights
        self._yolo = None
        self._tracker = None

    @property
    def yolo(self) -> Any:
        """Underlying ``ultralytics.YOLO`` instance (lazy-loaded)."""
        if self._yolo is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Ultralytics is required. Install with: pip install ultralytics"
                ) from exc
            LOGGER.info("Loading model: %s", self.weights)
            self._yolo = YOLO(self.weights)
        return self._yolo

    @property
    def names(self) -> Dict[int, str]:
        """Class id -> name mapping from the loaded model."""
        return getattr(self.yolo, "names", {}) or {}

    def train(
        self,
        data: str,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        device: Optional[str] = None,
        workers: Optional[int] = None,
        track: bool = True,
        register_after: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Train on a dataset YAML.

        Args:
            data: Path to a YOLO-format dataset YAML.
            epochs: Number of epochs.
            batch: Batch size.
            imgsz: Image size.
            device: Compute device override (``"0"``, ``"cpu"``, ``"0,1"``).
            workers: Dataloader workers (use ``0`` on low-RAM Windows).
            track: Enable MLflow tracking when MLflow is installed.
            register_after: If True, register ``best.pt`` to the model registry.
            **kwargs: Additional kwargs forwarded to ``YOLO.train``.

        Returns:
            Path to ``best.pt``.
        """
        from tede.data import resolve_dataset_yaml

        device = auto_device(device)
        train_kwargs: Dict[str, Any] = {
            "data": resolve_dataset_yaml(data),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            **kwargs,
        }
        if workers is not None:
            train_kwargs["workers"] = workers

        if track:
            try:
                from tede.tracking import ExperimentTracker

                self._tracker = ExperimentTracker(experiment_name="tede")
                self._tracker.start_run(run_name=train_kwargs.get("name", "tede_run"))
                self._tracker.log_params(train_kwargs)
            except Exception as exc:
                LOGGER.warning("Tracking disabled: %s", exc)
                self._tracker = None

        LOGGER.info("Training kwargs: %s", train_kwargs)
        LOGGER.info("Pre-train CUDA memory: %s MB", gpu_memory_mb())

        try:
            results = self.yolo.train(**train_kwargs)
        except KeyboardInterrupt:
            LOGGER.warning("Training interrupted by user.")
            if self._tracker:
                self._tracker.end_run("KILLED")
            raise
        except Exception as exc:
            LOGGER.exception("Training failed.")
            if self._tracker:
                self._tracker.end_run("FAILED")
            raise RuntimeError(f"Training failed: {exc}") from exc

        save_dir = Path(getattr(results, "save_dir", "runs/detect"))
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"

        if self._tracker:
            try:
                metrics = getattr(results, "results_dict", None) or {}
                self._tracker.log_metrics(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                for art in (best, last):
                    if art.is_file():
                        self._tracker.log_artifact(str(art))
                self._tracker.end_run("FINISHED")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to log artifacts: %s", exc)

        LOGGER.info("Post-train CUDA memory: %s MB", gpu_memory_mb())
        if best.is_file():
            self.weights = str(best)
            self._yolo = None  # force reload from best.pt on next use
            if register_after:
                metrics_dict = getattr(results, "results_dict", None) or {}
                self.register(
                    metrics={k: float(v) for k, v in metrics_dict.items() if isinstance(v, (int, float))},
                    dataset=data,
                )
            LOGGER.info("Best checkpoint: %s", best)
            return best
        LOGGER.warning("best.pt not found at %s; returning save_dir.", best)
        return save_dir

    def val(self, data: str, **kwargs: Any) -> Dict[str, Any]:
        """Validate the model and return a metrics dict."""
        from tede.data import resolve_dataset_yaml

        kwargs.setdefault("device", auto_device(kwargs.get("device")))
        data = resolve_dataset_yaml(data)
        try:
            metrics = self.yolo.val(data=data, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Validation failed: {exc}") from exc
        box = metrics.box
        report = {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "mean_precision": float(box.mp),
            "mean_recall": float(box.mr),
        }
        try:
            per_class = []
            for i, cls_id in enumerate(box.ap_class_index):
                per_class.append(
                    {
                        "class_id": int(cls_id),
                        "class_name": self.names.get(int(cls_id), str(cls_id)),
                        "precision": float(box.p[i]) if i < len(box.p) else None,
                        "recall": float(box.r[i]) if i < len(box.r) else None,
                        "mAP50": float(box.ap50[i]) if i < len(box.ap50) else None,
                        "mAP50-95": float(box.ap[i]) if i < len(box.ap) else None,
                    }
                )
            report["per_class"] = per_class
        except (AttributeError, IndexError):
            report["per_class"] = []
        return report

    def predict(
        self,
        source: Union[str, int],
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: Optional[str] = None,
        save: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run inference and return JSON-serializable detections."""
        device = auto_device(device)
        try:
            results = self.yolo.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=save,
                verbose=False,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Prediction failed: {exc}") from exc
        names = self.names
        payload: List[Dict[str, Any]] = []
        for r in results:
            dets: List[Dict[str, Any]] = []
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                try:
                    xyxy = boxes.xyxy.cpu().numpy().tolist()
                    confs = boxes.conf.cpu().numpy().tolist()
                    clses = boxes.cls.cpu().numpy().astype(int).tolist()
                    for box, c, k in zip(xyxy, confs, clses):
                        dets.append(
                            {
                                "bbox_xyxy": [float(x) for x in box],
                                "confidence": float(c),
                                "class_id": int(k),
                                "class_name": names.get(int(k), str(k)),
                            }
                        )
                except (AttributeError, ValueError) as exc:
                    LOGGER.warning("Failed to parse boxes: %s", exc)
            payload.append({"image": getattr(r, "path", None), "detections": dets})
        return payload

    def export(self, format: str = "onnx", imgsz: int = 640, **kwargs: Any) -> str:
        """Export to a deployable runtime format (onnx | engine | torchscript | ...)."""
        try:
            artifact = self.yolo.export(format=format, imgsz=imgsz, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Export failed: {exc}") from exc
        return str(artifact)

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Launch the bundled FastAPI inference server with this checkpoint."""
        import os

        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install serving extras: pip install tede[serving]") from exc
        os.environ["TEDE_WEIGHTS"] = self.weights
        from tede.api import app

        uvicorn.run(app, host=host, port=port)

    def register(
        self,
        metrics: Dict[str, float],
        dataset: Optional[str] = None,
        notes: Optional[str] = None,
        bump: str = "minor",
    ) -> Dict[str, Any]:
        """Register the current weights to the local model registry."""
        from tede.registry import ModelRegistry

        return ModelRegistry().register(
            self.weights, metrics=metrics, dataset=dataset, notes=notes, bump=bump
        )
