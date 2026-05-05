# TEDE — Independent Object Detection Framework

TEDE is a single-package object detection framework for training, validating,
serving and monitoring detectors on custom datasets. It is **fully independent**
of Ultralytics and the YOLO codebase: every detection component is built on
**PyTorch** and **torchvision** primitives only.

```
[DATA] -> [TRAIN] -> [VAL] -> [REGISTER] -> [DEPLOY] -> [MONITOR]
```

- **Architectures**: `retinanet` (default), `fcos`, `fasterrcnn`, `ssd` —
  all from `torchvision.models.detection`.
- **Data**: standard YOLO label format (`class cx cy w h`, normalised).
- **Tracking**: optional MLflow.
- **Serving**: FastAPI + Uvicorn.
- **Export**: ONNX (and from there to TensorRT/OpenVINO/etc.).
- **Quality gates**: pytest + flake8 + GitHub Actions.

## Why a torchvision backend?

A from-scratch detector that matches YOLO accuracy is multi-year work for a
team. torchvision's detection models are standard PyTorch (Meta-maintained,
not YOLO), have ImageNet-pretrained backbones, and are known to converge
on small custom datasets — so we use those as the engine and own everything
above them: data loading, training loop, validation, NMS decoding, mAP
scoring, registry, monitoring, and serving.

If you later want to swap in a hand-written detector, the only file that
needs to change is `tede/nn/model.py` — everything else is architecture-
agnostic.

## Install

```bash
# core only
pip install -e .

# with MLflow + FastAPI + ONNX
pip install -e ".[all]"
```

GPU users — install CUDA-enabled torch first (pick the wheel matching your driver):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[all]"
```

## CLI

```bash
# Train (defaults to RetinaNet + ResNet-50 FPN, ImageNet-pretrained backbone)
tede train data=configs/dataset.yaml arch=retinanet epochs=50 batch=8 device=0

# Switch architecture
tede train data=configs/dataset.yaml arch=fcos epochs=50 batch=8 device=0
tede train data=configs/dataset.yaml arch=fasterrcnn epochs=50 batch=4 device=0

# Validate
tede val model=runs/detect/tede/weights/best.pt data=configs/dataset.yaml

# Predict (file, directory, or webcam index)
tede predict model=runs/detect/tede/weights/best.pt source=test.jpg
tede predict model=runs/detect/tede/weights/best.pt source=0

# Export to ONNX
tede export model=runs/detect/tede/weights/best.pt format=onnx

# Serve via FastAPI
tede serve model=runs/detect/tede/weights/best.pt port=8000

# Preprocess raw_data/{images,labels} into data/ with 70/20/10 split
tede preprocess source=raw_data/ output=data/ nc=3

# Local model registry
tede registry list
tede registry best
tede registry compare version=v1.1
```

`python -m tede ...` is equivalent if the console script isn't on `PATH`.

## Python API

```python
from tede import TEDE

# Train from scratch
model = TEDE(arch="retinanet")
best = model.train(
    data="configs/dataset.yaml",
    epochs=50, batch=8, imgsz=640, device="0", workers=2,
    pretrained=True,                       # ImageNet-pretrained backbone
    register_after=True,                   # auto-register to model registry
)

# Reload trained weights later
model = TEDE(weights="runs/detect/tede/weights/best.pt")

# Validate
metrics = model.val(data="configs/dataset.yaml")
print(metrics["mAP50"], metrics["mAP50-95"])

# Predict
results = model.predict("test.jpg", conf=0.3)
for r in results:
    for d in r["detections"]:
        print(d["class_name"], d["confidence"], d["bbox_xyxy"])

# Export & serve
model.export(format="onnx")
model.serve(port=8000)
```

## Project layout

```
yolo26x-custom/
├── tede/                         # Independent framework
│   ├── core.py                   # Public TEDE class
│   ├── cli.py                    # `tede` console script
│   ├── nn/
│   │   └── model.py              # torchvision model factory
│   ├── datasets/
│   │   ├── yolo_dataset.py       # PyTorch Dataset for YOLO labels
│   │   └── transforms.py         # detection-aware augmentations
│   ├── ops/
│   │   ├── boxes.py              # IoU / clipping
│   │   ├── nms.py                # multi-class NMS
│   │   └── metrics.py            # mAP@0.5 + mAP@0.5:0.95
│   ├── engine/
│   │   ├── trainer.py            # training loop
│   │   ├── validator.py          # validation + mAP
│   │   ├── predictor.py          # inference
│   │   └── exporter.py           # ONNX export
│   ├── data.py                   # preprocessing (validate, split, hash)
│   ├── tracking.py               # MLflow wrapper (optional)
│   ├── registry.py               # local semver model registry
│   ├── monitor.py                # JSONL drift / latency monitor
│   ├── api.py                    # FastAPI server
│   └── configs/                  # bundled defaults
├── configs/                      # project-local configs (yours to edit)
├── data/                         # images/{train,val,test} + labels/...
├── tests/test_framework.py       # pytest suite
├── pyproject.toml
├── Makefile
├── USAGE.md                      # detailed copy-paste recipes
└── README.md
```

## Quick smoke test (no dataset needed)

Use any small YOLO-format dataset. There's no built-in `coco8` since we
don't depend on Ultralytics. The fastest path is to drop a few labelled
images under `raw_data/{images,labels}/` and:

```bash
tede preprocess source=raw_data/ output=data/ nc=<your nc>
tede train data=configs/dataset.yaml arch=retinanet epochs=3 batch=2 imgsz=416 workers=0 device=0
```

## Hardware notes

| GPU VRAM | Recommended arch        | imgsz | batch |
|----------|-------------------------|-------|-------|
| ≤4 GB    | `retinanet` / `fcos`    | 416   | 2     |
| 6–8 GB   | `retinanet` / `fcos`    | 640   | 4     |
| 12 GB    | `fasterrcnn`            | 640   | 8     |
| 16+ GB   | `fasterrcnn`            | 640   | 16    |

On Windows with limited RAM, use `workers=0` to avoid the dataloader
spawning multiple Python interpreters that exhaust the paging file.

## License

AGPL-3.0-or-later.
