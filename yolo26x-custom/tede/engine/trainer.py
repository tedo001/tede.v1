"""Pure-PyTorch training loop for TEDE detection models.

Reads a YOLO dataset YAML, builds the chosen torchvision detector, and
runs a standard SGD/AdamW loop with cosine LR, AMP, gradient clipping,
periodic checkpointing and per-epoch validation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from tede.datasets import YOLODataset, build_transforms, collate_fn
from tede.engine.validator import Validator
from tede.nn import build_model
from tede.nn.model import label_offset
from tede.utils import auto_device, ensure_dirs, get_logger, gpu_memory_mb, save_json

LOGGER = get_logger("tede.trainer")


def _build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name in ("sgd", "musgd"):  # MuSGD is treated as SGD with Nesterov for parity
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")


class Trainer:
    """High-level training orchestrator.

    Args:
        data: YOLO dataset YAML path.
        arch: Detection architecture (``retinanet | fcos | fasterrcnn | ssd``).
        epochs: Number of epochs.
        batch: Batch size.
        imgsz: Image size (square letterbox).
        device: Compute device. ``None`` -> auto.
        workers: Dataloader workers (use 0 on low-RAM Windows).
        lr0: Initial learning rate.
        optimizer: ``sgd | adamw | adam``.
        weight_decay: L2 regularization.
        momentum: SGD momentum.
        amp: Use mixed precision when CUDA is available.
        save_period: Save a checkpoint every N epochs.
        patience: Early-stop after N epochs without mAP50 improvement.
        project, name: Output directory = ``project/name``.
        pretrained: ImageNet-pretrained backbone (recommended).
    """

    def __init__(
        self,
        data: str,
        arch: str = "retinanet",
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
        resume: Optional[str] = None,
    ) -> None:
        self.data = data
        self.arch = arch
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.imgsz = int(imgsz)
        self.device = torch.device(auto_device(device) if str(auto_device(device)) != "cpu" else "cpu")
        if str(self.device) == "0":
            self.device = torch.device("cuda:0")
        self.workers = int(workers)
        self.lr0 = float(lr0)
        self.optimizer_name = optimizer
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.save_period = int(save_period)
        self.patience = int(patience)
        self.project = Path(project)
        self.name = name
        self.pretrained = bool(pretrained)
        self.resume = resume

        self.save_dir = self.project / self.name
        self.weights_dir = self.save_dir / "weights"
        ensure_dirs(self.save_dir, self.weights_dir)

        self._build_data()
        self._build_model()

    def _build_data(self) -> None:
        offset = label_offset(self.arch)
        self.train_ds = YOLODataset(
            self.data, split="train",
            transforms=build_transforms(self.imgsz, training=True),
            label_offset=offset,
        )
        try:
            self.val_ds = YOLODataset(
                self.data, split="val",
                transforms=build_transforms(self.imgsz, training=False),
                label_offset=offset,
            )
        except (FileNotFoundError, KeyError):
            LOGGER.warning("No val split found — skipping validation.")
            self.val_ds = None
        self.num_classes = self.train_ds.num_classes
        self.class_names = self.train_ds.class_names
        LOGGER.info("Classes (%d): %s", self.num_classes, self.class_names)

        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.batch, shuffle=True,
            num_workers=self.workers, collate_fn=collate_fn,
            pin_memory=self.device.type == "cuda", drop_last=True,
            persistent_workers=self.workers > 0,
        )
        if self.val_ds is not None:
            self.val_loader = DataLoader(
                self.val_ds, batch_size=self.batch, shuffle=False,
                num_workers=self.workers, collate_fn=collate_fn,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.workers > 0,
            )
        else:
            self.val_loader = None

    def _build_model(self) -> None:
        self.model = build_model(
            num_classes=self.num_classes,
            arch=self.arch,
            pretrained_backbone=self.pretrained,
        ).to(self.device)
        if self.resume:
            self._load_checkpoint(self.resume)
        self.optimizer = _build_optimizer(
            self.model, self.optimizer_name, self.lr0, self.weight_decay, self.momentum,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.epochs, 1), eta_min=self.lr0 * 0.01,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        LOGGER.info("Resumed weights from %s", path)

    def _save_checkpoint(self, path: Path, **extra: Any) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "arch": self.arch,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "imgsz": self.imgsz,
            **extra,
        }, path)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        n_batches = 0
        t0 = time.perf_counter()
        for images, targets in self.train_loader:
            images = [img.to(self.device, non_blocking=True) for img in images]
            t_cuda: list = []
            for t in targets:
                t_cuda.append({k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()})

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                loss_dict = self.model(images, t_cuda)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                LOGGER.warning("Non-finite loss at epoch %d; skipping batch.", epoch)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v.detach().cpu())
            running["loss"] = running.get("loss", 0.0) + float(loss.detach().cpu())
            n_batches += 1

        elapsed = time.perf_counter() - t0
        means = {k: v / max(n_batches, 1) for k, v in running.items()}
        LOGGER.info(
            "epoch %d/%d | loss %.4f | lr %.2e | gpu_mem %s MB | %.1fs",
            epoch, self.epochs, means.get("loss", 0.0),
            self.optimizer.param_groups[0]["lr"], gpu_memory_mb(), elapsed,
        )
        return means

    def fit(self) -> Path:
        """Run the full training loop and return the path to ``best.pt``."""
        best_map50 = -1.0
        epochs_no_improve = 0
        history: list = []

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            self.scheduler.step()

            val_metrics: Dict[str, Any] = {}
            if self.val_loader is not None:
                validator = Validator(self.model, self.val_loader, self.device, self.arch)
                val_metrics = validator.run()
                LOGGER.info(
                    "  val mAP50=%.4f | mAP50-95=%.4f",
                    val_metrics.get("mAP50", 0.0), val_metrics.get("mAP50-95", 0.0),
                )

            row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
            history.append(row)
            save_json({"history": history}, self.save_dir / "results.json")

            map50 = float(val_metrics.get("mAP50", 0.0))
            if map50 > best_map50:
                best_map50 = map50
                epochs_no_improve = 0
                self._save_checkpoint(self.weights_dir / "best.pt", epoch=epoch, mAP50=best_map50)
                LOGGER.info("  new best.pt @ mAP50=%.4f", best_map50)
            else:
                epochs_no_improve += 1

            self._save_checkpoint(self.weights_dir / "last.pt", epoch=epoch, mAP50=map50)
            if self.save_period > 0 and epoch % self.save_period == 0:
                self._save_checkpoint(self.weights_dir / f"epoch_{epoch}.pt", epoch=epoch, mAP50=map50)

            if self.patience > 0 and epochs_no_improve >= self.patience:
                LOGGER.warning("Early stopping at epoch %d (no improvement for %d epochs).", epoch, self.patience)
                break

        best = self.weights_dir / "best.pt"
        if not best.is_file():
            best = self.weights_dir / "last.pt"
        return best
