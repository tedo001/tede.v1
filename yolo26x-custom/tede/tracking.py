"""MLflow experiment tracking — soft dependency, never blocks training."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from tede.utils import get_logger

LOGGER = get_logger("tede.tracking")


class ExperimentTracker:
    """High-level wrapper around the MLflow tracking API."""

    def __init__(
        self,
        experiment_name: str = "tede",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._mlflow = None
        self._active_run = None
        try:
            import mlflow

            self._mlflow = mlflow
            uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or f"file:{Path('mlruns').resolve()}"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            LOGGER.info("MLflow tracking URI=%s, experiment=%s", uri, experiment_name)
        except ImportError:
            LOGGER.warning("MLflow not installed — tracker is disabled.")

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def start_run(self, run_name: Optional[str] = None) -> None:
        if not self.enabled:
            return
        try:
            self._active_run = self._mlflow.start_run(run_name=run_name)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to start MLflow run: %s", exc)

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.enabled or self._active_run is None:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to end MLflow run: %s", exc)
        finally:
            self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        flat: Dict[str, Any] = {}
        for k, v in params.items():
            flat[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
        try:
            self._mlflow.log_params(flat)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to log params: %s", exc)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to log metrics: %s", exc)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if not Path(path).exists():
            return
        try:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to log artifact %s: %s", path, exc)

    @contextmanager
    def run(self, run_name: Optional[str] = None) -> Iterator["ExperimentTracker"]:
        self.start_run(run_name=run_name)
        try:
            yield self
            self.end_run("FINISHED")
        except Exception:
            self.end_run("FAILED")
            raise
