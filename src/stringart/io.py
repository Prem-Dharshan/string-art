"""Saving and loading runs: sequence JSON (the fabrication output), images, metrics."""

import json
from pathlib import Path

import cv2
import numpy as np


def save_gray(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix, np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(path)


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not decode {path}")
    return img.astype(np.float64) / 255.0


def save_sequence(path: Path, *, sequence, pins, size, frame, opacity, meta=None) -> None:
    doc = {
        "frame": frame,
        "size": size,
        "opacity": opacity,
        "n_pins": len(pins),
        "pins": np.round(pins, 3).tolist(),
        "n_lines": len(sequence) - 1,
        "sequence": [int(p) for p in sequence],
        "meta": meta or {},
    }
    Path(path).write_text(json.dumps(doc, indent=1))


def load_sequence(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    doc["pins"] = np.asarray(doc["pins"], dtype=np.float64)
    return doc
