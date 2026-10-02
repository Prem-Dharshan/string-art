"""Fabrication outputs: a human-readable build list and the thread length."""

from collections.abc import Sequence

import numpy as np


def chord_lengths_px(sequence: Sequence[int], pins: np.ndarray) -> np.ndarray:
    p = np.asarray(pins)[np.asarray(sequence)]
    return np.hypot(*(p[1:] - p[:-1]).T)


def thread_length_mm(sequence, pins, size: int, frame_mm: float) -> float:
    """Total thread length in mm for a frame `frame_mm` across (diameter or side)."""
    return float(chord_lengths_px(sequence, pins).sum() * frame_mm / (size - 1))


def instructions(
    sequence, pins, size: int, frame: str, frame_mm: float | None = None, block: int = 100
) -> str:
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
            lines.append(
                f"-- lines {k}-{min(k + block - 1, n_lines)} "
                f"(thread used so far: {run:.1f} {unit}) --"
            )
        lines.append(f"{k:5d}  {sequence[k - 1]:4d} -> {sequence[k]:4d}")
    lines.append(f"done: tie off at pin {sequence[-1]}")
    return "\n".join(lines) + "\n"


def color_instructions(
    steps, palette: list[str], pins, size: int, frame: str, frame_mm: float | None = None
) -> str:
    """Winding list for colour art: one spool per colour, each its own continuous thread.
    Steps must be wound in the listed order (it is the layering order the solver assumed)."""
    if frame_mm:
        scale, unit = frame_mm / (size - 1) / 1000.0, "m"
    else:
        scale, unit = 1.0 / (size - 1), "frame widths"
    p = np.asarray(pins)
    length = dict.fromkeys(palette, 0.0)
    count = dict.fromkeys(palette, 0)
    first: dict[str, int] = {}
    for k, a, b in steps:
        name = palette[k]
        length[name] += float(np.hypot(*(p[b] - p[a]))) * scale
        count[name] += 1
        first.setdefault(name, a)
    start = "top" if frame == "circle" else "top-left corner"
    out = [
        "Colour string art build instructions",
        f"frame: {frame}{f', {frame_mm:g} mm' if frame_mm else ''}; pins: {len(pins)} numbered "
        f"0..{len(pins) - 1} clockwise, pin 0 at the {start}",
        f"lines: {len(steps)}; one spool per thread colour, each a single continuous thread:",
    ]
    for name in palette:
        if count[name]:
            out.append(
                f"  {name:<10} {count[name]:5d} lines, {length[name]:.1f} {unit}, "
                f"tie on at pin {first[name]}"
            )
    out += [
        "Wind in the order below: switching spools keeps the layering the solver assumed.",
        "Leave each idle spool hanging at its current pin.",
        "",
    ]
    run_start = 0
    for i in range(1, len(steps) + 1):
        if i == len(steps) or steps[i][0] != steps[run_start][0]:
            name = palette[steps[run_start][0]]
            out.append(f"-- {name} thread: lines {run_start + 1}-{i} --")
            for j in range(run_start, i):
                _, a, b = steps[j]
                out.append(f"{j + 1:5d}  {name:<10} {a:4d} -> {b:4d}")
            run_start = i
    out.append("done: tie off every thread at its last pin")
    return "\n".join(out) + "\n"
