"""Unit tests for the TEDE framework (CLI parsing, data, registry, monitor).

These tests do not load weights or train — they verify the public surface
area is correct and the pure-python helpers behave as documented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tede import TEDE, __version__
from tede.cli import _coerce, parse_kv
from tede.data import (
    class_distribution,
    resolve_dataset_yaml,
    split_dataset,
    validate_label_file,
)
from tede.monitor import Monitor
from tede.registry import ModelRegistry
from tede.utils import auto_device, hash_directory, load_yaml


# ---------- public surface ----------

def test_version_string() -> None:
    """The package exposes a non-empty semver-ish version."""
    assert isinstance(__version__, str) and __version__


def test_tede_class_constructs_without_loading() -> None:
    """Constructing TEDE must not load weights."""
    m = TEDE("yolo26s.pt")
    assert m.weights == "yolo26s.pt"
    assert m._yolo is None  # lazy


# ---------- CLI parsing ----------

def test_coerce() -> None:
    """_coerce maps strings to bool / None / int / float / str."""
    assert _coerce("true") is True
    assert _coerce("False") is False
    assert _coerce("none") is None
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("0.001") == 0.001
    assert _coerce("yolo26s.pt") == "yolo26s.pt"


def test_parse_kv_simple() -> None:
    """parse_kv produces a dict of coerced values."""
    out = parse_kv(["data=x.yaml", "epochs=5", "amp=true"])
    assert out == {"data": "x.yaml", "epochs": 5, "amp": True}


def test_parse_kv_rejects_positional() -> None:
    """parse_kv raises SystemExit for tokens missing '='."""
    with pytest.raises(SystemExit):
        parse_kv(["bad-token"])


# ---------- data / preprocessing ----------

def test_validate_label_file_ok(tmp_path: Path) -> None:
    f = tmp_path / "ok.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.9 0.05 0.05\n", encoding="utf-8")
    assert validate_label_file(f, num_classes=3) == []


def test_validate_label_file_errors(tmp_path: Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 1.5 0.2 0.2\nfoo bar\n5 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    errs = validate_label_file(f, num_classes=3)
    assert any("out of [0,1]" in e for e in errs)
    assert any("non-numeric" in e for e in errs)
    assert any("class id 5 >= nc" in e for e in errs)


def test_class_distribution(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    f2.write_text("0 0.3 0.3 0.3 0.3\n", encoding="utf-8")
    assert class_distribution([f1, f2]) == {0: 2, 1: 1}


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


def test_resolve_dataset_yaml_makes_path_absolute(tmp_path: Path) -> None:
    """A relative `path:` becomes absolute relative to the YAML file."""
    yml = tmp_path / "dataset.yaml"
    yml.write_text(
        "path: ../data\ntrain: images/train\nval: images/val\nnc: 1\nnames: {0: dog}\n",
        encoding="utf-8",
    )
    resolved_path = Path(resolve_dataset_yaml(str(yml)))
    assert resolved_path.is_file()
    cfg = load_yaml(resolved_path)
    assert Path(cfg["path"]).is_absolute()


def test_resolve_dataset_yaml_passes_through_pre_shipped_names() -> None:
    """Names like 'coco8.yaml' (no on-disk file) are returned unchanged."""
    assert resolve_dataset_yaml("coco8.yaml") == "coco8.yaml"


# ---------- utils ----------

def test_auto_device_returns_string() -> None:
    assert isinstance(auto_device(), str) and auto_device()


def test_auto_device_respects_explicit_choice() -> None:
    assert auto_device("cpu") == "cpu"


def test_hash_directory_stable(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")
    h1 = hash_directory(tmp_path)
    h2 = hash_directory(tmp_path)
    assert h1 == h2 and len(h1) == 64


# ---------- registry ----------

def test_registry_register_and_compare(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    reg = ModelRegistry(root=str(tmp_path / "registry"), model_name="t")
    v1 = reg.register(str(weights), metrics={"mAP50-95": 0.50, "mAP50": 0.70})
    v2 = reg.register(str(weights), metrics={"mAP50-95": 0.55, "mAP50": 0.72})
    assert v1["version"] == "v1.0"
    assert v2["version"] == "v1.1"
    best = reg.get_best()
    assert best is not None and best["version"] == "v1.1"
    cmp = reg.compare_with_best(v2["version"])
    assert cmp["is_better"] is False  # v1.1 IS the best, delta == 0


# ---------- monitor ----------

def test_monitor_logs_and_alerts(tmp_path: Path) -> None:
    m = Monitor(log_dir=str(tmp_path), accuracy_threshold=0.5, latency_threshold_ms=100.0)
    m.log_prediction([{"confidence": 0.9}, {"confidence": 0.8}], latency_ms=50.0, image_id="ok.jpg")
    m.log_prediction([{"confidence": 0.2}], latency_ms=300.0, image_id="bad.jpg")
    summary = m.summary(last_n=10)
    assert summary["records"] == 2
    alerts = [json.loads(l) for l in (Path(tmp_path) / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    types = {a["type"] for a in alerts}
    assert "high_latency" in types
    assert "low_confidence" in types


# ---------- import smoke ----------

@pytest.mark.parametrize(
    "module",
    [
        "tede",
        "tede.core",
        "tede.cli",
        "tede.data",
        "tede.utils",
        "tede.tracking",
        "tede.registry",
        "tede.monitor",
        "tede.export",
        "tede.api",
    ],
)
def test_modules_import(module: str) -> None:
    """All public modules import cleanly."""
    __import__(module)
