"""Filesystem-backed model registry with semantic-style versioning.

Each registered model is copied into ``model_registry/<name>/v<major>.<minor>/``
alongside a ``metadata.json``. The registry tracks the current "best" model
based on a primary metric (default ``mAP50-95``).

Usage:
    from mlops.model_registry import ModelRegistry
    reg = ModelRegistry()
    info = reg.register("runs/.../weights/best.pt", metrics={"mAP50-95": 0.74},
                       dataset="configs/dataset.yaml")
    reg.compare_with_best(info["version"])
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import get_logger

LOGGER = get_logger("mlops.registry")
_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


class ModelRegistry:
    """Local model registry with semantic versions and a ``best`` pointer."""

    def __init__(self, root: str = "model_registry", model_name: str = "yolo26x_custom") -> None:
        """Initialise the registry directory layout."""
        self.root = Path(root)
        self.model_name = model_name
        self.model_dir = self.root / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.model_dir / "index.json"
        if not self.index_path.is_file():
            self._write_index({"name": model_name, "versions": [], "best": None, "primary_metric": "mAP50-95"})

    def _read_index(self) -> Dict[str, Any]:
        with self.index_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_index(self, data: Dict[str, Any]) -> None:
        with self.index_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    def _next_version(self, bump: str = "minor") -> str:
        """Compute the next semantic version string (``vMAJOR.MINOR``)."""
        index = self._read_index()
        versions = index.get("versions", [])
        if not versions:
            return "v1.0"
        latest = versions[-1]["version"]
        m = _VERSION_RE.match(latest)
        major, minor = (int(m.group(1)), int(m.group(2))) if m else (1, 0)
        if bump == "major":
            return f"v{major + 1}.0"
        return f"v{major}.{minor + 1}"

    def register(
        self,
        weights_path: str,
        metrics: Dict[str, float],
        dataset: Optional[str] = None,
        notes: Optional[str] = None,
        bump: str = "minor",
    ) -> Dict[str, Any]:
        """Copy weights into the registry and record metadata."""
        src = Path(weights_path)
        if not src.is_file():
            raise FileNotFoundError(f"Weights not found: {src}")

        index = self._read_index()
        version = self._next_version(bump=bump)
        version_dir = self.model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        target = version_dir / src.name
        shutil.copy2(src, target)

        metadata: Dict[str, Any] = {
            "name": self.model_name,
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "weights": str(target.resolve()),
            "weights_size_bytes": target.stat().st_size,
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            "dataset": dataset,
            "notes": notes,
        }
        with (version_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        index["versions"].append({"version": version, "path": str(version_dir.resolve())})
        primary = index.get("primary_metric", "mAP50-95")
        current_best = index.get("best")
        new_score = metadata["metrics"].get(primary)
        best_score = current_best.get("metrics", {}).get(primary) if current_best else None
        if new_score is not None and (best_score is None or new_score > best_score):
            index["best"] = metadata
            LOGGER.info("New best model: %s with %s=%.4f", version, primary, new_score)
        self._write_index(index)
        LOGGER.info("Registered %s -> %s", src, version_dir)
        return metadata

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all registered versions."""
        return self._read_index().get("versions", [])

    def get_best(self) -> Optional[Dict[str, Any]]:
        """Return metadata for the current best model, or ``None``."""
        return self._read_index().get("best")

    def compare_with_best(self, version: str) -> Dict[str, Any]:
        """Compare a registered version against the current best model.

        Returns a dict of metric deltas. Positive = improvement.
        """
        index = self._read_index()
        best = index.get("best")
        version_dir = self.model_dir / version
        meta_path = version_dir / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Unknown version: {version}")
        with meta_path.open("r", encoding="utf-8") as fh:
            target = json.load(fh)

        if not best:
            return {"baseline": None, "candidate": target, "deltas": {}, "is_better": True}

        deltas = {
            k: round(target["metrics"].get(k, 0.0) - best["metrics"].get(k, 0.0), 6)
            for k in set(target["metrics"]) | set(best["metrics"])
        }
        primary = index.get("primary_metric", "mAP50-95")
        is_better = deltas.get(primary, 0.0) > 0
        return {"baseline": best, "candidate": target, "deltas": deltas, "is_better": is_better}
