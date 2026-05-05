"""TEDE high-level API.

The ``TEDE`` class is the public entrypoint for the framework. It is a
thin orchestration layer over ``tede.engine`` and exposes the same five
verbs you'd expect from any detection framework: ``train``, ``val``,
``predict``, ``export``, ``serve``.

This implementation is fully independent of Ultralytics — every detection
component is built on torchvision and pure PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tede.utils import get_logger

LOGGER = get_logger("tede")


class TEDE:
    """High-level TEDE detector.

    Examples:
        >>> from tede import TEDE
        >>> model = TEDE(arch="retinanet")
        >>> best = model.train(data="data.yaml", epochs=50, batch=8)
        >>> metrics = model.val(data="data.yaml")
        >>> dets = model.predict("image.jpg")
        >>> model.export(format="onnx")
    """

    def __init__(self, weights: Optional[str] = None, arch: str = "retinanet") -> None:
        """Construct a TEDE detector.

        Args:
            weights: Optional path to a previously trained ``.pt`` checkpoint.
                If ``None``, a fresh model is built when ``train`` is called.
            arch: One of ``retinanet | fcos | fasterrcnn | ssd``. Ignored
                when loading from a checkpoint (the checkpoint records its
                own architecture).
        """
        self.weights = weights
        self.arch = arch
        self._predictor = None
        self._tracker = None

    def _load_predictor(self):
        from tede.engine import Predictor

        if self.weights is None:
            raise RuntimeError("No weights loaded. Train a model first or pass weights=... .")
        if self._predictor is None or self._predictor.weights != self.weights:
            self._predictor = Predictor(self.weights)
        return self._predictor

    def train(
        self,
        data: str,
        epochs: int = 50,
        batch: int = 8,
        imgsz: int = 640,
        device: Optional[str] = None,
        workers: int = 2,
        lr0: float = 0.005,
        optimizer: str = "sgd",
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        amp: bool = True,
        save_period: int = 10,
        patience: int = 20,
        project: str = "runs/detect",
        name: str = "tede",
        pretrained: bool = True,
        track: bool = True,
        register_after: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Train on a YOLO-format dataset YAML and return the best checkpoint."""
        from tede.engine import Trainer

        # Optional MLflow tracking — soft dependency.
        if track:
            try:
                from tede.tracking import ExperimentTracker

                self._tracker = ExperimentTracker(experiment_name="tede")
                self._tracker.start_run(run_name=name)
                params = {
                    "data": data, "arch": self.arch, "epochs": epochs, "batch": batch,
                    "imgsz": imgsz, "device": str(device), "workers": workers,
                    "lr0": lr0, "optimizer": optimizer, "weight_decay": weight_decay,
                    "momentum": momentum, "amp": amp, "pretrained": pretrained,
                }
                self._tracker.log_params(params)
            except Exception as exc:
                LOGGER.warning("Tracking disabled: %s", exc)
                self._tracker = None

        trainer = Trainer(
            data=data, arch=self.arch, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, workers=workers, lr0=lr0, optimizer=optimizer,
            weight_decay=weight_decay, momentum=momentum, amp=amp,
            save_period=save_period, patience=patience, project=project,
            name=name, pretrained=pretrained, resume=self.weights if kwargs.get("resume") else None,
        )
        try:
            best = trainer.fit()
        except Exception as exc:
            if self._tracker:
                self._tracker.end_run("FAILED")
            raise RuntimeError(f"Training failed: {exc}") from exc

        self.weights = str(best)
        self._predictor = None  # force reload from new best.pt

        if self._tracker:
            try:
                if best.is_file():
                    self._tracker.log_artifact(str(best))
                self._tracker.end_run("FINISHED")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to log artifacts: %s", exc)

        if register_after:
            try:
                metrics = self.val(data=data)
                self.register(metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))}, dataset=data)
            except Exception as exc:
                LOGGER.warning("Auto-register failed: %s", exc)

        return best

    def val(self, data: str, batch: int = 8, imgsz: int = 640, device: Optional[str] = None,
            workers: int = 0, split: str = "val") -> Dict[str, Any]:
        """Run validation on a dataset split and return mAP metrics."""
        import torch
        from torch.utils.data import DataLoader

        from tede.datasets import YOLODataset, build_transforms, collate_fn
        from tede.engine import Validator
        from tede.nn import build_model
        from tede.nn.model import label_offset
        from tede.utils import auto_device

        if self.weights is None:
            raise RuntimeError("val requires weights. Train first or pass weights=... .")

        dev = torch.device("cuda:0" if str(auto_device(device)) == "0" else auto_device(device))
        ckpt = torch.load(self.weights, map_location=dev)
        arch = ckpt.get("arch", self.arch)
        nc = ckpt["num_classes"]

        ds = YOLODataset(
            data, split=split,
            transforms=build_transforms(imgsz, training=False),
            label_offset=label_offset(arch),
        )
        loader = DataLoader(
            ds, batch_size=batch, shuffle=False, num_workers=workers,
            collate_fn=collate_fn, pin_memory=dev.type == "cuda",
        )
        model = build_model(num_classes=nc, arch=arch, pretrained_backbone=False)
        model.load_state_dict(ckpt["model"])
        model.to(dev).eval()

        return Validator(model, loader, dev, arch=arch).run()

    def predict(self, source: Union[str, int], conf: float = 0.25, imgsz: int = 640,
                device: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Run inference and return a list of per-image detection dicts."""
        from tede.engine import Predictor

        if self.weights is None:
            raise RuntimeError("predict requires weights. Train first or pass weights=... .")
        if self._predictor is None or self._predictor.weights != self.weights:
            self._predictor = Predictor(self.weights, device=device, imgsz=imgsz, conf=conf)
        return self._predictor(source)

    def export(self, format: str = "onnx", imgsz: int = 640, **kwargs: Any) -> str:
        """Export to a deployable runtime format. Currently supports ``onnx``."""
        from tede.engine import export_onnx

        if self.weights is None:
            raise RuntimeError("export requires weights. Train first or pass weights=... .")
        if format.lower() != "onnx":
            raise NotImplementedError(
                f"Format '{format}' not supported by the standalone TEDE engine. "
                "Export to ONNX first, then convert to TensorRT/OpenVINO/CoreML "
                "with their respective toolchains."
            )
        return export_onnx(self.weights, imgsz=imgsz, **kwargs)

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Launch the bundled FastAPI inference server."""
        import os

        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install serving extras: pip install tede[serving]") from exc

        if self.weights is None:
            raise RuntimeError("serve requires weights. Train first or pass weights=... .")
        os.environ["TEDE_WEIGHTS"] = self.weights
        from tede.api import app

        uvicorn.run(app, host=host, port=port)

    def register(self, metrics: Dict[str, float], dataset: Optional[str] = None,
                 notes: Optional[str] = None, bump: str = "minor") -> Dict[str, Any]:
        """Register the current weights to the local model registry."""
        from tede.registry import ModelRegistry

        if self.weights is None:
            raise RuntimeError("register requires weights.")
        return ModelRegistry().register(
            self.weights, metrics=metrics, dataset=dataset, notes=notes, bump=bump,
        )
