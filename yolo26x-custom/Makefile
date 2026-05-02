# YOLO26x Custom — automation
# Usage: make <target>

PY ?= python
PIP ?= pip
WEIGHTS ?= runs/detect/yolo26x_custom/weights/best.pt
CONFIG  ?= configs/model.yaml
DATA    ?= configs/dataset.yaml
HOST    ?= 0.0.0.0
PORT    ?= 8000
FORMAT  ?= onnx
SOURCE  ?= data/images/test
CUDA    ?= cu121          # CUDA wheel tag for PyTorch (cu118|cu121|cu124)

VENV     ?= .venv
VENV_GPU ?= venv_gpu

.PHONY: help setup setup-gpu train evaluate export deploy test lint preprocess clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup:  ## Install CPU dependencies into .venv (creates it if missing).
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && \
		pip install --upgrade pip && \
		pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
		pip install -r requirements.txt
	@echo "CPU env ready. Activate with: source $(VENV)/bin/activate"

setup-gpu:  ## Install CUDA dependencies into venv_gpu (override CUDA=cu118|cu121|cu124).
	$(PY) -m venv $(VENV_GPU)
	. $(VENV_GPU)/bin/activate && \
		pip install --upgrade pip && \
		pip install torch torchvision --index-url https://download.pytorch.org/whl/$(CUDA) && \
		pip install -r requirements.txt && \
		python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
	@echo "GPU env ready. Activate with: source $(VENV_GPU)/bin/activate"

preprocess:  ## Validate and split a raw dataset into data/.
	$(PY) -m src.preprocess --source raw_data --output data --val 0.2 --test 0.1 --nc 3

train:  ## Run YOLO26x training.
	$(PY) -m src.train --config $(CONFIG)

evaluate:  ## Evaluate the trained model.
	$(PY) -m src.evaluate --weights $(WEIGHTS) --data $(DATA)

export:  ## Export weights (override with FORMAT=onnx|engine|torchscript).
	$(PY) -m mlops.export --weights $(WEIGHTS) --format $(FORMAT)

predict:  ## Run inference. Override SOURCE=path/to/img_or_dir.
	$(PY) -m src.predict --weights $(WEIGHTS) --source $(SOURCE)

deploy:  ## Start the FastAPI inference server.
	YOLO_WEIGHTS=$(WEIGHTS) uvicorn mlops.api:app --host $(HOST) --port $(PORT)

test:  ## Run unit tests.
	pytest tests/ -v --tb=short

lint:  ## Run flake8.
	flake8 src mlops tests --max-line-length=120 --extend-ignore=E203,W503

clean:  ## Remove caches and temporary files.
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
