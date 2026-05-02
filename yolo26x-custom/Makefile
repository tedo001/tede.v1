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

.PHONY: help setup train evaluate export deploy test lint preprocess clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup:  ## Install all Python dependencies from requirements.txt.
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

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
