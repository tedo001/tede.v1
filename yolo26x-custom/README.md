# TEDE — Production Object Detection Framework

TEDE is a single-package object detection framework for training, evaluating,
serving and monitoring detectors on custom datasets. It is built on top of
Ultralytics YOLO26 and exposes both a Python API (`from tede import TEDE`)
and a CLI (`tede train ...`) — same shape as `yolo` itself, but with MLOps
hooks (MLflow tracking, local model registry, JSONL drift monitor,
ready-to-deploy FastAPI server) baked in.

```
[DATA] -> [TRAIN] -> [VAL] -> [REGISTER] -> [DEPLOY] -> [MONITOR]
```

## Install

```bash
# core only
pip install -e .

# everything (MLflow + FastAPI + ONNX export)
pip install -e ".[all]"

# pick what you need
pip install -e ".[mlops,serving]"
```

GPU users — install CUDA-enabled torch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[all]"
```

## CLI usage

```bash
# Train
tede train data=configs/dataset.yaml model=yolo26s.pt epochs=50 batch=8 device=0

# Validate
tede val model=runs/detect/tede/weights/best.pt data=configs/dataset.yaml

# Predict (file, directory, or webcam index)
tede predict model=runs/detect/tede/weights/best.pt source=test.jpg
tede predict model=runs/detect/tede/weights/best.pt source=0

# Export
tede export model=runs/detect/tede/weights/best.pt format=onnx
tede export model=runs/detect/tede/weights/best.pt format=engine

# Serve (FastAPI on :8000)
tede serve model=runs/detect/tede/weights/best.pt port=8000

# Preprocess raw_data/{images,labels} into data/ with 70/20/10 split
tede preprocess source=raw_data/ output=data/ nc=3

# Local model registry
tede registry list
tede registry best
tede registry compare version=v1.1
```

`python -m tede ...` is equivalent to `tede ...` — useful if the console
script isn't on `PATH`.

## Python API

```python
from tede import TEDE

# Train
model = TEDE("yolo26s.pt")
best = model.train(
    data="configs/dataset.yaml",
    epochs=50, batch=8, imgsz=640, device="0",
    workers=2, register_after=True,
)

# Validate
metrics = model.val(data="configs/dataset.yaml")
print(metrics["mAP50-95"])

# Predict
results = model.predict("test.jpg", conf=0.3)
for r in results:
    for d in r["detections"]:
        print(d["class_name"], d["confidence"], d["bbox_xyxy"])

# Export & serve
model.export(format="onnx")
model.serve(port=8000)        # blocks; uvicorn runs the FastAPI app

# Underlying Ultralytics object is exposed for power users
model.yolo.tune(data="...", iterations=10)
```

## Project layout

```
tede.v1/yolo26x-custom/
├── tede/                         # The pip-installable framework
│   ├── core.py                   # TEDE class
│   ├── cli.py                    # `tede` console script
│   ├── data.py                   # preprocessing + dataset YAML resolver
│   ├── utils.py                  # helpers
│   ├── tracking.py               # MLflow wrapper
│   ├── registry.py               # local model registry
│   ├── monitor.py                # production monitoring (JSONL)
│   ├── api.py                    # FastAPI inference server
│   └── configs/                  # default config YAMLs (shipped with wheel)
├── configs/                      # project-local configs (yours to edit)
├── data/                         # images/{train,val,test} + labels/...
├── tests/                        # pytest suite
├── pyproject.toml                # `pip install -e .`
├── Makefile                      # convenience wrappers
└── README.md
```

The legacy `src/` and `mlops/` modules from earlier versions still exist for
backward compatibility but new code should use `tede.*`.

## Pipeline stages

### 1. Data

Drop labelled YOLO data under `raw_data/{images,labels}/` then:

```bash
tede preprocess source=raw_data/ output=data/ nc=3
```

This validates labels, splits 70/20/10, and writes
`data/preprocess_report.json` with a SHA-256 dataset hash for lightweight
versioning. For full DVC-tracked versioning:

```bash
dvc init && dvc add data && git add data.dvc .gitignore
```

### 2. Train

```bash
tede train data=configs/dataset.yaml model=yolo26s.pt epochs=100 device=0
```

- Loads pretrained weights (auto-downloaded from Ultralytics).
- Optimizer: **`MuSGD`** (YOLO26 native).
- MLflow logging is automatic when MLflow is installed.
- Checkpoints saved every 10 epochs to `runs/detect/tede/weights/`.

### 3. Evaluate

```bash
tede val model=runs/detect/tede/weights/best.pt data=configs/dataset.yaml
```

Returns a JSON report with `mAP50`, `mAP50-95`, mean precision/recall, and
per-class breakdown.

### 4. Register

```python
from tede import TEDE
m = TEDE("runs/detect/tede/weights/best.pt")
metrics = m.val(data="configs/dataset.yaml")
m.register(metrics=metrics, dataset="configs/dataset.yaml")
```

Writes `model_registry/tede/v<MAJOR>.<MINOR>/{best.pt,metadata.json}` and
updates the `best` pointer when a new model wins on `mAP50-95`.

### 5. Deploy

```bash
tede export model=runs/detect/tede/weights/best.pt format=onnx
tede serve  model=runs/detect/tede/weights/best.pt port=8000

curl -F file=@sample.jpg http://localhost:8000/predict
```

| Method | Path                | Description                       |
|--------|---------------------|-----------------------------------|
| GET    | `/health`           | Liveness probe                    |
| GET    | `/metrics/summary`  | Last-N inference summary          |
| POST   | `/predict`          | Multipart image inference         |

### 6. Monitor

Every served prediction is logged to `monitoring/predictions.jsonl`. Latency
above `latency_threshold_ms` or average confidence below
`accuracy_threshold` raises an alert into `monitoring/alerts.jsonl`.

## Quick smoke test (no dataset needed)

```bash
tede train data=coco8.yaml model=yolo26s.pt epochs=3 batch=4 imgsz=416 workers=0 device=0
```

`coco8.yaml` is a tiny 8-image dataset Ultralytics ships with — perfect for
verifying your install end-to-end in ~1 minute on a modest GPU.

## Hardware notes

| GPU VRAM | Recommended model     | imgsz | batch |
|----------|-----------------------|-------|-------|
| ≤4 GB    | `yolo26n.pt` / `yolo26s.pt` | 416   | 4     |
| 6–8 GB   | `yolo26s.pt` / `yolo26m.pt` | 640   | 8     |
| 12 GB    | `yolo26m.pt` / `yolo26l.pt` | 640   | 16    |
| 16+ GB   | `yolo26l.pt` / `yolo26x.pt` | 640   | 16+   |

On Windows with limited RAM use `workers=0` to avoid the dataloader spawning
multiple Python interpreters that exhaust the paging file.

## Branching strategy

| Branch          | Purpose                                                 |
|-----------------|---------------------------------------------------------|
| `main`          | Production-ready; protected, deploys releases.          |
| `develop`       | Active integration branch.                              |
| `feature/<x>`   | New features merged into `develop`.                     |
| `experiment/<x>`| ML experiments and ablations; may be discarded.         |

CI (`.github/workflows/ci.yml`) runs lint + tests on every push and
auto-tags releases when `model_registry/tede/index.json` records a new best.

## References

- Ultralytics: <https://github.com/ultralytics/ultralytics>
- YOLO26 docs: <https://docs.ultralytics.com/models/yolo26>
- MLflow: <https://mlflow.org/docs/latest/index.html>

## License

AGPL-3.0-or-later (matching Ultralytics).
