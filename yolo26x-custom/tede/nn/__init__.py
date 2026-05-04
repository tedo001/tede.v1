"""TEDE neural network components — model factory built on torchvision.

The framework exposes a single ``build_model`` factory that returns a
torchvision detection model configured for ``num_classes`` foreground
classes. No dependency on Ultralytics or any YOLO codebase.
"""

from tede.nn.model import SUPPORTED_ARCHS, build_model

__all__ = ["build_model", "SUPPORTED_ARCHS"]
