"""Unit tests for the YOLO26x custom pipeline.

These tests intentionally avoid heavy training; they target pure-python
helpers (dataset config validity, label parsing, registry, monitor) so the
suite runs in a few seconds in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlops.model_registry import ModelRegistry
from mlops.monitor import Monitor
from src import preprocess
from src.utils import auto_device, hash_directory, load_yaml, save_json


# ---------- dataset.yaml ----------

REQUIRED_KEYS = {"path", "train", "val", "nc", "names"}


def test_dataset_yaml_structure() -> None:
    """dataset.yaml must define required keys with consistent class counts."""
    cfg = load_yaml("configs/dataset.yaml")
    missing = REQUIRED_KEYS - cfg.keys()
    assert not missing, f"Missing keys in dataset.yaml: {missing}"
    assert isinstance(cfg["nc"], int) and cfg["nc"] > 0
    assert isinstance(cfg["names"], (dict, list))
    name_count = len(cfg["names"]) if isinstance(cfg["names"], (dict, list)) else 0
    assert name_count == cfg["nc"], f"nc={cfg['nc']} but names has {name_count} entries"


def test_model_yaml_loads() -> None:
    """model.yaml must parse and contain the YOLO26x optimizer."""
    cfg = load_yaml("configs/model.yaml")
    assert cfg.get("optimizer") == "MuSGD"
    assert cfg.get("epochs", 0) > 0
    assert cfg.get("imgsz", 0) >= 32


# ---------- preprocess ----------

def test_validate_label_file_ok(tmp_path: Path) -> None:
    """Well-formed YOLO label files validate cleanly."""
    f = tmp_path / "ok.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.9 0.05 0.05\n", encoding="utf-8")
    assert preprocess.validate_label_file(f, num_classes=3) == []


def test_validate_label_file_errors(tmp_path: Path) -> None:
    """Out-of-range and malformed lines surface as errors."""
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 1.5 0.2 0.2\nfoo bar\n5 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    errs = preprocess.validate_label_file(f, num_classes=3)
    assert any("out of [0,1]" in e for e in errs)
    assert any("non-numeric" in e for e in errs)
    assert any("class id 5 >= nc" in e for e in errs)


def test_class_distribution(tmp_path: Path) -> None:
    """class_distribution should aggregate label counts correctly."""
    f1 = tmp_path / "a.txt"
    f1.write_text("0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    f2.write_text("0 0.3 0.3 0.3 0.3\n", encoding="utf-8")
    dist = preprocess.class_distribution([f1, f2])
    assert dist == {0: 2, 1: 1}


def test_split_dataset(tmp_path: Path) -> None:
    """split_dataset writes the expected directory structure."""
    src_img = tmp_path / "raw" / "images"
    src_lbl = tmp_path / "raw" / "labels"
    src_img.mkdir(parents=True)
    src_lbl.mkdir(parents=True)
    pairs = []
    for i in range(20):
        ip = src_img / f"img_{i}.jpg"
        ip.write_bytes(b"\xff\xd8\xff\xe0fake")
        lp = src_lbl / f"img_{i}.txt"
        lp.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        pairs.append((ip, lp))

    out = tmp_path / "out"
    counts = preprocess.split_dataset(pairs, out, val_frac=0.2, test_frac=0.1, seed=1)
    assert sum(counts.values()) == 20
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()


# ---------- utils ----------

def test_auto_device_returns_string() -> None:
    """auto_device must always return a non-empty string."""
    dev = auto_device()
    assert isinstance(dev, str) and dev


def test_hash_directory_stable(tmp_path: Path) -> None:
    """hash_directory must be deterministic for identical content."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    h1 = hash_directory(tmp_path)
    h2 = hash_directory(tmp_path)
    assert h1 == h2 and len(h1) == 64


def test_save_json_roundtrip(tmp_path: Path) -> None:
    """save_json should produce a parseable JSON file."""
    p = tmp_path / "x.json"
    save_json({"k": 1, "v": [1, 2]}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": 1, "v": [1, 2]}


# ---------- model registry ----------

def test_registry_register_and_compare(tmp_path: Path) -> None:
    """Registering two models picks the higher-mAP one as best."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")

    reg = ModelRegistry(root=str(tmp_path / "registry"), model_name="testmodel")
    v1 = reg.register(str(weights), metrics={"mAP50-95": 0.50, "mAP50": 0.70}, dataset="test")
    v2 = reg.register(str(weights), metrics={"mAP50-95": 0.55, "mAP50": 0.72}, dataset="test")

    assert v1["version"] == "v1.0"
    assert v2["version"] == "v1.1"
    best = reg.get_best()
    assert best is not None and best["version"] == "v1.1"
    cmp = reg.compare_with_best(v2["version"])
    assert "deltas" in cmp


# ---------- monitor ----------

def test_monitor_logs_and_alerts(tmp_path: Path) -> None:
    """Monitor must persist predictions and trigger threshold-based alerts."""
    m = Monitor(log_dir=str(tmp_path), accuracy_threshold=0.5, latency_threshold_ms=100.0)
    m.log_prediction([{"confidence": 0.9}, {"confidence": 0.8}], latency_ms=50.0, image_id="ok.jpg")
    m.log_prediction([{"confidence": 0.2}], latency_ms=300.0, image_id="bad.jpg")

    summary = m.summary(last_n=10)
    assert summary["records"] == 2

    alerts_path = Path(tmp_path) / "alerts.jsonl"
    assert alerts_path.is_file()
    alerts = [json.loads(l) for l in alerts_path.read_text(encoding="utf-8").splitlines()]
    types = {a["type"] for a in alerts}
    assert "high_latency" in types
    assert "low_confidence" in types


# ---------- soft import smoke tests ----------

@pytest.mark.parametrize(
    "module",
    ["src.train", "src.evaluate", "src.predict", "src.preprocess", "mlops.api", "mlops.export"],
)
def test_modules_import(module: str) -> None:
    """All pipeline modules must import without side effects."""
    __import__(module)
