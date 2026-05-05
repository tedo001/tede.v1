"""Inference engine — loads a TEDE checkpoint and produces detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from tede.nn import build_model
from tede.nn.model import label_offset
from tede.utils import auto_device, get_logger

LOGGER = get_logger("tede.predictor")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _letterbox(img: Image.Image, size: int) -> tuple[Image.Image, float, tuple[int, int]]:
    """Letterbox-resize a PIL image to (size, size). Returns (image, scale, (W, H))."""
    W, H = img.size
    s = size / max(W, H)
    new_w, new_h = int(round(W * s)), int(round(H * s))
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(114, 114, 114))
    canvas.paste(img_r, (0, 0))
    return canvas, s, (W, H)


class Predictor:
    """Load a TEDE checkpoint and run inference on images / video frames.

    The class is callable: ``predictor("image.jpg")``.
    """

    def __init__(
        self,
        weights: str,
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.25,
    ) -> None:
        self.weights = weights
        self.device = torch.device("cuda:0" if str(auto_device(device)) == "0" else auto_device(device))
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        ckpt = torch.load(weights, map_location=self.device)
        self.arch = ckpt.get("arch", "retinanet")
        self.num_classes = ckpt["num_classes"]
        self.class_names = ckpt.get("class_names", {}) or {i: str(i) for i in range(self.num_classes)}
        if isinstance(self.class_names, list):
            self.class_names = {i: n for i, n in enumerate(self.class_names)}
        self.label_offset = label_offset(self.arch)

        self.model = build_model(num_classes=self.num_classes, arch=self.arch, pretrained_backbone=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        LOGGER.info("Loaded %s (arch=%s, nc=%d) on %s", weights, self.arch, self.num_classes, self.device)

    def _load_image(self, src: Any) -> List[Image.Image]:
        if isinstance(src, Image.Image):
            return [src.convert("RGB")]
        if isinstance(src, np.ndarray):
            return [Image.fromarray(src).convert("RGB")]
        if isinstance(src, (str, Path)):
            p = Path(src)
            if p.is_dir():
                return [Image.open(f).convert("RGB") for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]
            if p.is_file():
                return [Image.open(p).convert("RGB")]
            raise FileNotFoundError(f"Source not found: {src}")
        raise TypeError(f"Unsupported source type: {type(src)}")

    @torch.no_grad()
    def __call__(self, source: Any) -> List[Dict[str, Any]]:
        images = self._load_image(source)
        results: List[Dict[str, Any]] = []
        for img in images:
            canvas, scale, (W, H) = _letterbox(img, self.imgsz)
            tensor = TF.to_tensor(canvas).to(self.device)
            outputs = self.model([tensor])[0]

            keep = outputs["scores"] >= self.conf
            boxes = outputs["boxes"][keep].detach().cpu()
            scores = outputs["scores"][keep].detach().cpu()
            labels = (outputs["labels"][keep].detach().cpu() - self.label_offset)

            # Map letterboxed coords back to original image space
            boxes = boxes / scale
            boxes[:, 0::2].clamp_(min=0, max=W)
            boxes[:, 1::2].clamp_(min=0, max=H)

            detections: List[Dict[str, Any]] = []
            for box, c, k in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                detections.append({
                    "bbox_xyxy": [float(x) for x in box],
                    "confidence": float(c),
                    "class_id": int(k),
                    "class_name": self.class_names.get(int(k), str(int(k))),
                })
            results.append({"image": getattr(img, "filename", None), "detections": detections})
        return results
