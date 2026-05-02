"""Production monitoring: confidence drift, accuracy alerts, latency tracking.

Persistence is JSONL so the monitor can be tailed by external tooling
(e.g. shipped to S3, Splunk, Loki). Stateless across processes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from src.utils import get_logger

LOGGER = get_logger("mlops.monitor")


class Monitor:
    """Append-only monitor for prediction-time signals."""

    def __init__(
        self,
        log_dir: str = "monitoring",
        accuracy_threshold: float = 0.6,
        latency_threshold_ms: float = 200.0,
    ) -> None:
        """Initialise log directory and alert thresholds."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_log = self.log_dir / "predictions.jsonl"
        self.alerts_log = self.log_dir / "alerts.jsonl"
        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold_ms = latency_threshold_ms

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # pragma: no cover
            LOGGER.warning("Failed to write monitoring record: %s", exc)

    def log_prediction(
        self,
        detections: List[Dict[str, Any]],
        latency_ms: float,
        image_id: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist one prediction event and emit alerts when thresholds are crossed."""
        confidences = [d.get("confidence", 0.0) for d in detections if "confidence" in d]
        avg_conf = float(mean(confidences)) if confidences else 0.0
        record = {
            "timestamp": self._now(),
            "image_id": image_id,
            "model_version": model_version,
            "num_detections": len(detections),
            "avg_confidence": avg_conf,
            "min_confidence": float(min(confidences)) if confidences else 0.0,
            "max_confidence": float(max(confidences)) if confidences else 0.0,
            "latency_ms": float(latency_ms),
        }
        self._append(self.predictions_log, record)

        if latency_ms > self.latency_threshold_ms:
            self.raise_alert(
                "high_latency",
                f"Inference {latency_ms:.1f}ms exceeded {self.latency_threshold_ms:.1f}ms.",
                context={"latency_ms": latency_ms, "image_id": image_id},
            )
        if confidences and avg_conf < self.accuracy_threshold:
            self.raise_alert(
                "low_confidence",
                f"avg_confidence {avg_conf:.3f} below threshold {self.accuracy_threshold:.3f}.",
                context={"avg_confidence": avg_conf, "image_id": image_id},
            )
        return record

    def raise_alert(self, alert_type: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Append an alert record and log a warning."""
        record = {
            "timestamp": self._now(),
            "type": alert_type,
            "message": message,
            "context": context or {},
        }
        self._append(self.alerts_log, record)
        LOGGER.warning("[ALERT:%s] %s", alert_type, message)

    def summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Aggregate stats across the last ``last_n`` prediction events."""
        if not self.predictions_log.is_file():
            return {"records": 0}
        try:
            with self.predictions_log.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()[-last_n:]
        except OSError:
            return {"records": 0}

        records: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                continue

        if not records:
            return {"records": 0}

        latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
        confs = [r["avg_confidence"] for r in records if "avg_confidence" in r]
        return {
            "records": len(records),
            "avg_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
            "avg_confidence": round(mean(confs), 4) if confs else 0.0,
            "low_confidence_rate": round(
                sum(1 for c in confs if c < self.accuracy_threshold) / len(confs), 4
            ) if confs else 0.0,
        }
