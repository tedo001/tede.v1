# YOLO26x Custom Object Detection — Production MLOps Stack

End-to-end MLOps project for training, evaluating, registering, deploying, and
monitoring an Ultralytics **YOLO26x** detector on a custom dataset.

```
[DATA] -> [TRAIN] -> [EVALUATE] -> [REGISTER] -> [DEPLOY] -> [MONITOR]
```

- **Model**: `yolo26x.pt` (Ultralytics YOLO26 family)
- **Optimizer**: `MuSGD` (YOLO26 native)
- **Tracking**: MLflow
- **Serving**: FastAPI + Uvicorn
- **Export**: ONNX, TensorRT, TorchScript
- **Quality gates**: pytest, flake8, GitHub Actions

---

## 1. Project layout

```
yolo26x-custom/
├── .github/workflows/ci.yml       GitHub Actions: lint + tests + auto-tag
├── configs/
│   ├── dataset.yaml               Custom dataset config (paths, nc, names)
│   └── model.yaml                 Training hyperparameters
├── data/                          images/{train,val,test} + labels/{train,val,test}
├── src/
│   ├── train.py                   Training entrypoint (YOLO26x + MLflow)
│   ├── evaluate.py                mAP / per-class metrics / FPS benchmark
│   ├── predict.py                 CLI inference (image/dir/video/webcam)
│   ├── preprocess.py              Validate + 70/20/10 split + dataset hash
│   └── utils.py                   Shared helpers
├── mlops/
│   ├── experiment_tracker.py      MLflow wrapper
│   ├── model_registry.py          Filesystem registry with semver
│   ├── monitor.py                 Confidence/latency drift logger
│   ├── api.py                     FastAPI inference endpoint
│   └── export.py                  ONNX / TensorRT export
├── tests/test_pipeline.py         Unit tests (pure-python, runs in seconds)
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

---

## 2. Quickstart

### 2.1. PyCharm + virtualenv

1. **Open the project**: *File -> Open* -> select `yolo26x-custom/`.
2. **Create venv**: *File -> Settings -> Project -> Python Interpreter ->
   Add Interpreter -> Add Local Interpreter -> Virtualenv Environment ->
   New environment*. Base Python 3.10+.
3. **Set as project interpreter** and apply.
4. **Install deps** (PyCharm terminal):
   ```bash
   make setup
   ```
5. **Run configurations** (*Run -> Edit Configurations -> + -> Python*):
   - `train`     -> module: `src.train`     -> params: `--config configs/model.yaml`
   - `evaluate`  -> module: `src.evaluate`  -> params: `--weights runs/detect/yolo26x_custom/weights/best.pt`
   - `predict`   -> module: `src.predict`   -> params: `--weights runs/.../best.pt --source data/images/test`
   - `api`       -> module: `uvicorn`       -> params: `mlops.api:app --reload`
6. **Git**: *VCS -> Enable Version Control Integration -> Git*. Add the GitHub
   remote: `git remote add origin git@github.com:<user>/yolo26x-custom.git`.

### 2.2. Plain CLI (Linux / Colab)

```bash
python -m venv .venv && source .venv/bin/activate
make setup
```

---

## 3. Pipeline stages

### Data

```bash
# Place raw images and YOLO-format labels under raw_data/{images,labels}/
make preprocess           # validates labels, splits 70/20/10, hashes the dataset
```

The hash is written to `data/preprocess_report.json` for lightweight versioning.
For full DVC-tracked versioning:

```bash
dvc init
dvc add data
git add data.dvc .gitignore
```

### Train

```bash
make train                                # uses configs/model.yaml
python -m src.train --config configs/model.yaml --epochs 50 --batch 8
```

- Loads `yolo26x.pt` pretrained weights.
- Optimizer **`MuSGD`**, `imgsz=640`, `epochs=100`, `batch=16`,
  `patience=20`, full augmentation pipeline.
- Logs params, metrics and best/last checkpoints to MLflow.
- Saves a checkpoint every 10 epochs (`save_period: 10`).

### Evaluate

```bash
make evaluate WEIGHTS=runs/detect/yolo26x_custom/weights/best.pt
```

Produces `runs/eval/evaluation_report.json` with mAP50, mAP50-95, per-class
precision/recall and FPS on both GPU and CPU.

### Register

```python
from mlops.model_registry import ModelRegistry
reg = ModelRegistry()
info = reg.register(
    "runs/detect/yolo26x_custom/weights/best.pt",
    metrics={"mAP50-95": 0.74, "mAP50": 0.91},
    dataset="configs/dataset.yaml",
)
print(reg.compare_with_best(info["version"]))
```

### Deploy

```bash
make export FORMAT=onnx                       # ONNX
make export FORMAT=engine                     # TensorRT
make deploy WEIGHTS=runs/.../best.pt          # FastAPI on :8000
curl -F file=@sample.jpg http://localhost:8000/predict
```

Endpoints:

| Method | Path                | Description                       |
|--------|---------------------|-----------------------------------|
| GET    | `/health`           | Liveness probe                    |
| GET    | `/metrics/summary`  | Last-N inference summary          |
| POST   | `/predict`          | Multipart image inference         |

### Monitor

`mlops.monitor.Monitor` writes JSONL streams in `monitoring/`:

- `predictions.jsonl` — confidence stats, detection counts, latency per call.
- `alerts.jsonl` — alerts raised when avg confidence < `accuracy_threshold` or
  latency > `latency_threshold_ms`.

---

## 4. Testing & lint

```bash
make test     # pytest
make lint     # flake8
```

---

## 5. Branching strategy

| Branch          | Purpose                                                        |
|-----------------|----------------------------------------------------------------|
| `main`          | Production-ready code; protected, deploys releases.            |
| `develop`       | Active integration branch.                                     |
| `feature/<x>`   | New features merged into `develop`.                            |
| `experiment/<x>`| ML experiments and ablations; may be discarded.                |

CI (`.github/workflows/ci.yml`) runs lint + tests on every push to those
branches and auto-tags releases when `model_registry/yolo26x_custom/index.json`
contains a new best model.

---

## 6. Cross-platform notes

- **Windows + PyCharm**: use the Git Bash terminal for `make`. If `make` is
  unavailable, run the equivalent `python -m ...` commands directly.
- **Linux / Colab**: works as-is. For Colab, install with `pip install -r
  requirements.txt` and skip the venv step.
- **GPU**: training auto-detects CUDA; force CPU with `--device cpu`.

---

## 7. References

- Ultralytics: <https://github.com/ultralytics/ultralytics>
- YOLO26 docs: <https://docs.ultralytics.com/models/yolo26>
- MLflow: <https://mlflow.org/docs/latest/index.html>
