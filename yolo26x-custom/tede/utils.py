"""Shared helpers used across the TEDE package."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_logger(name: str = "tede", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML file, raising informative errors."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"YAML config not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse YAML at {p}: {exc}") from exc


def save_yaml(data: Dict[str, Any], path: str | os.PathLike) -> None:
    """Persist a dict to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def save_json(data: Dict[str, Any], path: str | os.PathLike) -> None:
    """Persist a dict to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def auto_device(preferred: Optional[str] = None) -> str:
    """Choose the best available compute device."""
    if preferred not in (None, "", "auto"):
        return str(preferred)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def gpu_memory_mb() -> Optional[float]:
    """Return current allocated CUDA memory in MB, or None if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        return None
    return None


def hash_directory(path: str | os.PathLike, algorithm: str = "sha256") -> str:
    """Deterministic content hash of every file under ``path``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    hasher = hashlib.new(algorithm)
    for fp in sorted(p.rglob("*")):
        if fp.is_file():
            hasher.update(str(fp.relative_to(p)).encode("utf-8"))
            try:
                with fp.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        hasher.update(chunk)
            except OSError:
                continue
    return hasher.hexdigest()


def ensure_dirs(*paths: str | os.PathLike) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
