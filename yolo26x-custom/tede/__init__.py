"""TEDE — independent production object detection framework.

Built on PyTorch + torchvision; no Ultralytics, no YOLO codebase.

Quick start:
    >>> from tede import TEDE
    >>> model = TEDE(arch="retinanet")
    >>> best = model.train(data="data.yaml", epochs=50, batch=8)
    >>> dets = model.predict("image.jpg")
    >>> model.export(format="onnx")

CLI equivalent:
    $ tede train data=data.yaml arch=retinanet epochs=50
    $ tede predict model=runs/.../best.pt source=image.jpg
"""

from tede.core import TEDE

__version__ = "0.2.0"
__all__ = ["TEDE", "__version__"]
