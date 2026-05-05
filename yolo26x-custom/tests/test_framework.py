"""Unit tests for the independent TEDE framework.

These tests do not load weights or train — they verify the package
public surface, CLI parsing, dataset utilities, NMS/IoU, mAP scoring,
the model factory's argument validation, and the local registry/monitor.
Heavy imports (torch, torchvision) are exercised via the model factory
test which validates that a tiny RetinaNet model builds successfully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tede import TEDE, __version__
from tede.cli import _coerce, parse_kv
from tede.data import (
    class_distribution,
    split_dataset,
    validate_label_file,
)
from tede.monitor import Monitor
from tede.nn import SUPPORTED_ARCHS, build_model
from tede.nn.model import label_offset
from tede.ops import compute_map, multiclass_nms
from tede.ops.boxes import box_iou
from tede.registry import ModelRegistry
from tede.utils import auto_device, hash_directory, load_yaml


# ---------- public surface ----------

def test_version_string() -> None:
    assert isinstance(__version__, str) and __version__


def test_tede_lazy_constructor_does_not_load() -> None:
    m = TEDE(arch="retinanet")
    assert m.arch == "retinanet"
    assert m.weights is None
    assert m._predictor is None


# ---------- CLI parsing ----------

def test_coerce() -> None:
    assert _coerce("true") is True
    assert _coerce("False") is False
    assert _coerce("none") is None
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("retinanet") == "retinanet"


def test_parse_kv() -> None:
    out = parse_kv(["data=x.yaml", "epochs=5", "amp=true"])
    assert out == {"data": "x.yaml", "epochs": 5, "amp": True}


def test_parse_kv_rejects_positional() -> None:
    with pytest.raises(SystemExit):
        parse_kv(["bad"])


# ---------- model factory ----------

def test_label_offset() -> None:
    assert label_offset("retinanet") == 0
    assert label_offset("fcos") == 0
    assert label_offset("fasterrcnn") == 1
    assert label_offset("ssd") == 1


def test_build_model_rejects_unknown_arch() -> None:
    with pytest.raises(ValueError):
        build_model(num_classes=2, arch="nope")


def test_build_model_retinanet_constructs() -> None:
    """RetinaNet builds without pretrained weights and accepts dummy input."""
    model = build_model(num_classes=3, arch="retinanet", pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        out = model([torch.randn(3, 256, 256)])
    assert isinstance(out, list) and len(out) == 1
    assert {"boxes", "labels", "scores"} <= set(out[0].keys())


@pytest.mark.parametrize("arch", list(SUPPORTED_ARCHS))
def test_build_model_all_archs(arch: str) -> None:
    """Every supported arch should at least construct (no forward needed)."""
    model = build_model(num_classes=2, arch=arch, pretrained_backbone=False)
    assert isinstance(model, torch.nn.Module)


# ---------- ops ----------

def test_box_iou_basic() -> None:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    iou = box_iou(a, b)
    assert iou[0, 0].item() == pytest.approx(1.0)
    assert 0.0 < iou[0, 1].item() < 1.0


def test_multiclass_nms_filters_low_scores() -> None:
    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.01])
    labels = torch.tensor([0, 0], dtype=torch.int64)
    out_b, out_s, out_l = multiclass_nms(boxes, scores, labels, score_threshold=0.05)
    assert out_b.shape[0] == 1
    assert out_s.item() == pytest.approx(0.9)


def test_compute_map_perfect_predictions() -> None:
    """When predictions exactly match targets, mAP50 should be 1.0."""
    targets = [{"boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "labels": torch.tensor([0])}]
    preds = [{"boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
              "scores": torch.tensor([0.99]),
              "labels": torch.tensor([0])}]
    result = compute_map(preds, targets)
    assert result["mAP50"] == pytest.approx(1.0, abs=0.01)


# ---------- preprocessing ----------

def test_validate_label_file_ok(tmp_path: Path) -> None:
    f = tmp_path / "ok.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert validate_label_file(f, num_classes=3) == []


def test_validate_label_file_errors(tmp_path: Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 1.5 0.2 0.2\nfoo bar\n", encoding="utf-8")
    errs = validate_label_file(f, num_classes=3)
    assert any("out of [0,1]" in e for e in errs)
    assert any("non-numeric" in e for e in errs)


def test_class_distribution(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("0 0.3 0.3 0.3 0.3\n", encoding="utf-8")
    assert class_distribution([a, b]) == {0: 2, 1: 1}


def test_split_dataset(tmp_path: Path) -> None:
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
    counts = split_dataset(pairs, out, val_frac=0.2, test_frac=0.1, seed=1)
    assert sum(counts.values()) == 20
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()


# ---------- utils ----------

def test_auto_device() -> None:
    assert isinstance(auto_device(), str) and auto_device()
    assert auto_device("cpu") == "cpu"


def test_hash_directory_stable(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("hello", encoding="utf-8")
    h1 = hash_directory(tmp_path)
    h2 = hash_directory(tmp_path)
    assert h1 == h2


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text("key: 42\nlist: [1, 2]\n", encoding="utf-8")
    cfg = load_yaml(p)
    assert cfg == {"key": 42, "list": [1, 2]}


# ---------- registry ----------

def test_registry_register_and_compare(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    reg = ModelRegistry(root=str(tmp_path / "registry"), model_name="t")
    v1 = reg.register(str(weights), metrics={"mAP50-95": 0.50, "mAP50": 0.70})
    v2 = reg.register(str(weights), metrics={"mAP50-95": 0.55, "mAP50": 0.72})
    assert v1["version"] == "v1.0"
    assert v2["version"] == "v1.1"
    assert reg.get_best()["version"] == "v1.1"


# ---------- monitor ----------

def test_monitor_logs_and_alerts(tmp_path: Path) -> None:
    m = Monitor(log_dir=str(tmp_path), accuracy_threshold=0.5, latency_threshold_ms=100.0)
    m.log_prediction([{"confidence": 0.9}, {"confidence": 0.8}], latency_ms=50.0, image_id="ok.jpg")
    m.log_prediction([{"confidence": 0.2}], latency_ms=300.0, image_id="bad.jpg")
    summary = m.summary(last_n=10)
    assert summary["records"] == 2
    alerts = [json.loads(l) for l in (Path(tmp_path) / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    types = {a["type"] for a in alerts}
    assert {"high_latency", "low_confidence"} <= types


# ---------- import smoke ----------

@pytest.mark.parametrize(
    "module",
    [
        "tede", "tede.core", "tede.cli", "tede.data", "tede.utils",
        "tede.tracking", "tede.registry", "tede.monitor", "tede.api",
        "tede.nn", "tede.nn.model",
        "tede.datasets", "tede.datasets.yolo_dataset", "tede.datasets.transforms",
        "tede.ops", "tede.ops.boxes", "tede.ops.nms", "tede.ops.metrics",
        "tede.engine", "tede.engine.trainer", "tede.engine.validator",
        "tede.engine.predictor", "tede.engine.exporter",
    ],
)
def test_modules_import(module: str) -> None:
    __import__(module)
