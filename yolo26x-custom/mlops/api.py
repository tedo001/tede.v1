"""FastAPI inference server for YOLO26x.

Run:
    uvicorn mlops.api:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
    YOLO_WEIGHTS    Path to a ``best.pt`` (default: weights/best.pt)
    YOLO_DEVICE     Device override (default: auto)
    YOLO_IMGSZ      Inference image size (default: 640)
    YOLO_CONF       Confidence threshold (default: 0.25)
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from mlops.monitor import Monitor
from src.utils import auto_device, get_logger

LOGGER = get_logger("mlops.api")
STATE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the YOLO26x model once at startup."""
    weights = os.getenv("YOLO_WEIGHTS", "weights/best.pt")
    device = auto_device(os.getenv("YOLO_DEVICE") or None)
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Ultralytics not installed.") from exc

    if not os.path.isfile(weights):
        LOGGER.warning("Weights file not found at %s — API will return 503 until provided.", weights)
        STATE["model"] = None
    else:
        LOGGER.info("Loading model %s on %s", weights, device)
        STATE["model"] = YOLO(weights)
    STATE["device"] = device
    STATE["weights"] = weights
    STATE["imgsz"] = int(os.getenv("YOLO_IMGSZ", "640"))
    STATE["conf"] = float(os.getenv("YOLO_CONF", "0.25"))
    STATE["monitor"] = Monitor()
    yield
    STATE.clear()


app = FastAPI(
    title="YOLO26x Inference API",
    description="Production inference endpoint for the YOLO26x custom detector.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok" if STATE.get("model") is not None else "no_model",
        "device": STATE.get("device"),
        "weights": STATE.get("weights"),
    }


@app.get("/metrics/summary")
def metrics_summary(last_n: int = 100) -> Dict[str, Any]:
    """Return monitoring summary over the last ``last_n`` predictions."""
    monitor: Monitor = STATE.get("monitor")
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialised.")
    return monitor.summary(last_n=last_n)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Run inference on a single uploaded image."""
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload.")
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        arr = np.asarray(image)
        t0 = time.perf_counter()
        try:
            results = model.predict(
                arr,
                imgsz=STATE["imgsz"],
                conf=STATE["conf"],
                device=STATE["device"],
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("Inference failed.")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        names = getattr(model, "names", {}) or {}
        detections: List[Dict[str, Any]] = []
        if results:
            r = results[0]
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
                    LOGGER.warning("Could not parse boxes: %s", exc)

        STATE["monitor"].log_prediction(
            detections=detections,
            latency_ms=latency_ms,
            image_id=file.filename,
            model_version=os.path.basename(STATE.get("weights", "")),
        )
        return JSONResponse(
            {
                "filename": file.filename,
                "latency_ms": round(latency_ms, 2),
                "detections": detections,
            }
        )
    finally:
        await file.close()
