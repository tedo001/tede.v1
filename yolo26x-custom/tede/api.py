"""FastAPI inference server for TEDE.

Run programmatically:
    >>> from tede import TEDE
    >>> TEDE("best.pt").serve(port=8000)

Or directly:
    $ TEDE_WEIGHTS=best.pt uvicorn tede.api:app --host 0.0.0.0 --port 8000

Environment variables:
    TEDE_WEIGHTS    Path to the weights file (default: weights/best.pt).
    TEDE_DEVICE     Device override (default: auto).
    TEDE_IMGSZ      Inference image size (default: 640).
    TEDE_CONF       Confidence threshold (default: 0.25).
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from tede.monitor import Monitor
from tede.utils import auto_device, get_logger

LOGGER = get_logger("tede.api")
STATE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup."""
    weights = os.getenv("TEDE_WEIGHTS", "weights/best.pt")
    device = auto_device(os.getenv("TEDE_DEVICE") or None)
    if not os.path.isfile(weights):
        LOGGER.warning("Weights file not found at %s — API will return 503 until provided.", weights)
        STATE["predictor"] = None
    else:
        from tede.engine import Predictor

        LOGGER.info("Loading model %s on %s", weights, device)
        STATE["predictor"] = Predictor(
            weights,
            device=device,
            imgsz=int(os.getenv("TEDE_IMGSZ", "640")),
            conf=float(os.getenv("TEDE_CONF", "0.25")),
        )
    STATE["device"] = device
    STATE["weights"] = weights
    STATE["monitor"] = Monitor()
    yield
    STATE.clear()


app = FastAPI(
    title="TEDE Inference API",
    description="Production inference endpoint for the TEDE detection framework.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok" if STATE.get("predictor") is not None else "no_model",
        "device": STATE.get("device"),
        "weights": STATE.get("weights"),
    }


@app.get("/metrics/summary")
def metrics_summary(last_n: int = 100) -> Dict[str, Any]:
    """Return a summary of the last ``last_n`` predictions."""
    monitor: Monitor = STATE.get("monitor")
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialised.")
    return monitor.summary(last_n=last_n)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Run inference on a single uploaded image."""
    predictor = STATE.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload.")
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        t0 = time.perf_counter()
        try:
            results = predictor(image)
        except Exception as exc:
            LOGGER.exception("Inference failed.")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = results[0]["detections"] if results else []
        STATE["monitor"].log_prediction(
            detections=detections,
            latency_ms=latency_ms,
            image_id=file.filename,
            model_version=os.path.basename(STATE.get("weights", "")),
        )
        return JSONResponse({
            "filename": file.filename,
            "latency_ms": round(latency_ms, 2),
            "detections": detections,
        })
    finally:
        await file.close()
