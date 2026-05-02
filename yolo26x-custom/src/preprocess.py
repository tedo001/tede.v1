"""Dataset preprocessing: validation, balancing checks and 70/20/10 splitting.

Run as a module:
    python -m src.preprocess --source raw_data/ --output data/ --val 0.2 --test 0.1
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils import ensure_dirs, get_logger, hash_directory, save_json

LOGGER = get_logger("preprocess")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair every image with its YOLO ``.txt`` label.

    Images without a matching label are skipped with a warning.
    """
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


def validate_label_file(label_path: Path, num_classes: int | None = None) -> List[str]:
    """Validate a YOLO label file. Returns a list of human-readable errors.

    Each line must be ``class cx cy w h`` with numeric values; coordinates
    must be normalized to ``[0, 1]``; ``class`` must be a non-negative int
    (and ``< num_classes`` if provided).
    """
    errors: List[str] = []
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"unreadable: {exc}"]

    if not text:
        return errors  # empty label files are valid (negative samples)

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
    """Copy image/label pairs into ``data/{images,labels}/{train,val,test}``."""
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
    val_frac: float,
    test_frac: float,
    num_classes: int | None,
    seed: int,
) -> Dict[str, object]:
    """Top-level entrypoint: validate + split + version."""
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


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(description="Validate and split a YOLO dataset.")
    p.add_argument("--source", type=Path, required=True, help="Raw dataset root with images/ and labels/")
    p.add_argument("--output", type=Path, default=Path("data"), help="Output dataset root")
    p.add_argument("--val", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--test", type=float, default=0.1, help="Test fraction")
    p.add_argument("--nc", type=int, default=None, help="Number of classes (for label range validation)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """Module entrypoint."""
    args = parse_args()
    try:
        run(args.source, args.output, args.val, args.test, args.nc, args.seed)
    except Exception as exc:
        LOGGER.exception("Preprocessing failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
