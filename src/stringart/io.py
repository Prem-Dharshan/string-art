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


def save_color_result(path: Path, *, steps, palette, colors, pins, size, frame, opacity,
                      meta=None) -> None:
    doc = {
        "mode": "color",
        "frame": frame,
        "size": size,
        "opacity": opacity,
        "n_pins": len(pins),
        "pins": np.round(pins, 3).tolist(),
        "palette": list(palette),
        "colors": np.round(np.asarray(colors), 4).tolist(),
        "n_lines": len(steps),
        "steps": [[int(v) for v in s] for s in steps],
        "meta": meta or {},
    }
    Path(path).write_text(json.dumps(doc, indent=1))


def save_rgb(path: Path, img: np.ndarray) -> None:
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(Path(path).suffix, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(path)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def load_sequence(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    doc["pins"] = np.asarray(doc["pins"], dtype=np.float64)
    return doc
