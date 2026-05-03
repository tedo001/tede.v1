"""Dataset preprocessing and YAML resolution for TEDE."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tede.utils import ensure_dirs, get_logger, hash_directory, load_yaml, save_json, save_yaml

LOGGER = get_logger("tede.data")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_dataset_yaml(data_path: str) -> str:
    """Materialize a sibling YAML with absolute paths.

    Ultralytics resolves a relative ``path:`` against its global ``datasets_dir``
    setting — which often points at an unrelated old project on Windows. We
    rewrite the YAML to use an absolute path so behaviour is independent of
    that global config. Pre-shipped names like ``coco8.yaml`` are passed
    through unchanged so the registry still resolves them.
    """
    src = Path(data_path)
    if not src.is_file():
        return data_path
    src = src.resolve()
    cfg = load_yaml(src)
    base = src.parent
    if "path" in cfg and cfg["path"]:
        p = Path(cfg["path"])
        if not p.is_absolute():
            cfg["path"] = str((base / p).resolve())
    out = src.with_name(f".{src.stem}.resolved.yaml")
    save_yaml(cfg, out)
    LOGGER.info("Resolved dataset YAML: %s (path=%s)", out, cfg.get("path"))
    return str(out)


def list_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair every image with its YOLO ``.txt`` label."""
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels directory not found: {labels_dir}")
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        if not label.is_file():
            LOGGER.warning("Missing label for image: %s", img.name)
            continue
        pairs.append((img, label))
    return pairs


def validate_label_file(label_path: Path, num_classes: Optional[int] = None) -> List[str]:
    """Return a list of validation errors for a YOLO label file."""
    errors: List[str] = []
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"unreadable: {exc}"]
    if not text:
        return errors
    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"L{ln}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
        except ValueError:
            errors.append(f"L{ln}: non-numeric values")
            continue
        if cls < 0:
            errors.append(f"L{ln}: negative class id {cls}")
        if num_classes is not None and cls >= num_classes:
            errors.append(f"L{ln}: class id {cls} >= nc ({num_classes})")
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not 0.0 <= val <= 1.0:
                errors.append(f"L{ln}: {name}={val} out of [0,1]")
    return errors


def class_distribution(label_files: List[Path]) -> Dict[int, int]:
    """Count YOLO class occurrences across label files."""
    counts: Counter[int] = Counter()
    for f in label_files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        counts[int(parts[0])] += 1
                    except ValueError:
                        continue
        except OSError:
            continue
    return dict(sorted(counts.items()))


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    output_root: Path,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, int]:
    """Copy image/label pairs into ``{output_root}/{images,labels}/{train,val,test}``."""
    if not 0.0 < val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("val_frac must be in (0,1) and test_frac in [0,1).")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }
    for split, items in splits.items():
        img_out = output_root / "images" / split
        lbl_out = output_root / "labels" / split
        ensure_dirs(img_out, lbl_out)
        for img, lbl in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)
    counts = {k: len(v) for k, v in splits.items()}
    LOGGER.info("Dataset split: %s", counts)
    return counts


def run(
    source: Path,
    output: Path,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    num_classes: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """End-to-end preprocess: validate -> split -> hash."""
    images_dir = source / "images"
    labels_dir = source / "labels"
    pairs = list_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError(f"No image/label pairs discovered under {source}.")
    LOGGER.info("Discovered %d image/label pairs", len(pairs))
    invalid = 0
    for _, lbl in pairs:
        errs = validate_label_file(lbl, num_classes=num_classes)
        if errs:
            invalid += 1
            LOGGER.warning("Invalid labels in %s: %s", lbl.name, "; ".join(errs))
    LOGGER.info("Label validation complete (%d invalid files).", invalid)
    distribution = class_distribution([lbl for _, lbl in pairs])
    LOGGER.info("Class distribution: %s", distribution)
    counts = split_dataset(pairs, output, val_frac=val_frac, test_frac=test_frac, seed=seed)
    version_hash = hash_directory(output)
    report = {
        "source": str(source),
        "output": str(output),
        "splits": counts,
        "class_distribution": distribution,
        "invalid_label_files": invalid,
        "dataset_hash": version_hash,
        "seed": seed,
    }
    save_json(report, output / "preprocess_report.json")
    LOGGER.info("Dataset version (sha256): %s", version_hash)
    return report
