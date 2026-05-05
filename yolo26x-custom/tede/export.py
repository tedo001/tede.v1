"""Backward-compatible export shim.

Earlier versions of TEDE wrapped Ultralytics' export. The framework is
now independent of Ultralytics, and the real implementation lives in
``tede.engine.exporter``. This module preserves the ``tede.export.export``
import path for any user code that still references it.
"""

from __future__ import annotations

from typing import Optional

from tede.engine.exporter import export_onnx
from tede.utils import get_logger

LOGGER = get_logger("tede.export")
SUPPORTED_FORMATS = {"onnx"}


def export(
    weights: str,
    fmt: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = True,
    device: Optional[str] = None,
    simplify: bool = True,
) -> str:
    """Export a TEDE checkpoint to a deployable format.

    Currently only ``onnx`` is supported by the standalone engine. Convert
    the resulting ``.onnx`` file to TensorRT, OpenVINO, etc. with their
    respective external toolchains.
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise NotImplementedError(
            f"Format '{fmt}' not supported by the standalone TEDE engine. "
            "Export to ONNX first, then convert with the target runtime's tools."
        )
    return export_onnx(weights=weights, imgsz=imgsz, dynamic=dynamic, device=device)
