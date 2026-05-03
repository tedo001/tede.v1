# TEDE — convenience wrappers around the `tede` CLI.
# Most targets just invoke `tede <subcommand> ...`.

PY ?= python
PIP ?= pip
WEIGHTS ?= runs/detect/tede/weights/best.pt
DATA    ?= configs/dataset.yaml
MODEL   ?= yolo26s.pt
HOST    ?= 0.0.0.0
PORT    ?= 8000
FORMAT  ?= onnx
SOURCE  ?= data/images/test
EPOCHS  ?= 100
BATCH   ?= 8
IMGSZ   ?= 640
DEVICE  ?= 0
WORKERS ?= 2
CUDA    ?= cu121

VENV     ?= .venv
VENV_GPU ?= venv_gpu

.PHONY: help setup setup-gpu install train val predict export serve preprocess test lint clean smoke

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup:  ## Create .venv and install TEDE (CPU torch).
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && \
		pip install --upgrade pip && \
		pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
		pip install -e ".[all,dev]"
	@echo "CPU env ready. Activate: source $(VENV)/bin/activate"

setup-gpu:  ## Create venv_gpu and install TEDE (CUDA torch). Override CUDA=cu118|cu121|cu124.
	$(PY) -m venv $(VENV_GPU)
	. $(VENV_GPU)/bin/activate && \
		pip install --upgrade pip && \
		pip install torch torchvision --index-url https://download.pytorch.org/whl/$(CUDA) && \
		pip install -e ".[all,dev]" && \
		python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
	@echo "GPU env ready. Activate: source $(VENV_GPU)/bin/activate"

install:  ## Install TEDE editable into the active interpreter.
	$(PIP) install -e ".[all,dev]"

smoke:  ## Tiny end-to-end test on Ultralytics' built-in coco8 dataset.
	tede train data=coco8.yaml model=$(MODEL) epochs=3 batch=$(BATCH) imgsz=416 workers=0 device=$(DEVICE)

preprocess:  ## Validate and split raw_data/ into data/.
	tede preprocess source=raw_data output=data val=0.2 test=0.1 nc=3

train:  ## Run training. Override DATA=, MODEL=, EPOCHS=, BATCH=, DEVICE=, WORKERS=.
	tede train data=$(DATA) model=$(MODEL) epochs=$(EPOCHS) batch=$(BATCH) imgsz=$(IMGSZ) device=$(DEVICE) workers=$(WORKERS)

val:  ## Validate WEIGHTS against DATA.
	tede val model=$(WEIGHTS) data=$(DATA) device=$(DEVICE)

predict:  ## Run inference. Override SOURCE=path/to/img_or_dir.
	tede predict model=$(WEIGHTS) source=$(SOURCE) device=$(DEVICE)

export:  ## Export weights. Override FORMAT=onnx|engine|torchscript.
	tede export model=$(WEIGHTS) format=$(FORMAT) imgsz=$(IMGSZ)

serve:  ## Start the FastAPI inference server.
	tede serve model=$(WEIGHTS) host=$(HOST) port=$(PORT)

test:  ## Run unit tests.
	pytest tests/ -v --tb=short

lint:  ## Run flake8.
	flake8 tede tests --max-line-length=120 --extend-ignore=E203,W503

clean:  ## Remove caches and temporary files.
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
