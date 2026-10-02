"""Fabrication outputs: a human-readable build list and the thread length."""

from collections.abc import Sequence

import numpy as np


def chord_lengths_px(sequence: Sequence[int], pins: np.ndarray) -> np.ndarray:
    p = np.asarray(pins)[np.asarray(sequence)]
    return np.hypot(*(p[1:] - p[:-1]).T)


def thread_length_mm(sequence, pins, size: int, frame_mm: float) -> float:
    """Total thread length in mm for a frame `frame_mm` across (diameter or side)."""
    return float(chord_lengths_px(sequence, pins).sum() * frame_mm / (size - 1))


def instructions(sequence, pins, size: int, frame: str, frame_mm: float | None = None,
                 block: int = 100) -> str:
    """Step-by-step winding list. Pins are numbered 0..n-1 clockwise, pin 0 at the top
    (circle) or the top-left corner (rect)."""
    n_lines = len(sequence) - 1
    px = chord_lengths_px(sequence, pins)
    if frame_mm:
        scale, unit = frame_mm / (size - 1) / 1000.0, "m"
    else:
        scale, unit = 1.0 / (size - 1), "frame widths"
    total = px.sum() * scale
    start = "top" if frame == "circle" else "top-left corner"
    lines = [
        "String art build instructions",
        f"frame: {frame}{f', {frame_mm:g} mm' if frame_mm else ''}; pins: {len(pins)} numbered "
        f"0..{len(pins) - 1} clockwise, pin 0 at the {start}",
        f"lines: {n_lines}; thread length: {total:.1f} {unit} (add ~10% for knots and slack)",
        f"start: tie the thread at pin {sequence[0]}",
        "",
    ]
    run = 0.0
    for k in range(1, n_lines + 1):
        run += px[k - 1] * scale
        if (k - 1) % block == 0:
            lines.append(f"-- lines {k}-{min(k + block - 1, n_lines)} "
                         f"(thread used so far: {run:.1f} {unit}) --")
        lines.append(f"{k:5d}  {sequence[k - 1]:4d} -> {sequence[k]:4d}")
    lines.append(f"done: tie off at pin {sequence[-1]}")
    return "\n".join(lines) + "\n"
