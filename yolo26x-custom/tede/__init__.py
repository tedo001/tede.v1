"""TEDE — production object detection framework built on Ultralytics.

Quick start:
    >>> from tede import TEDE
    >>> model = TEDE("yolo26s.pt")
    >>> model.train(data="data.yaml", epochs=50)
    >>> results = model.predict("image.jpg")
    >>> model.export(format="onnx")

CLI equivalent:
    $ tede train data=data.yaml model=yolo26s.pt epochs=50
    $ tede predict model=runs/.../best.pt source=image.jpg
"""

from tede.core import TEDE

__version__ = "0.1.0"
__all__ = ["TEDE", "__version__"]
