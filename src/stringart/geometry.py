"""Pin layouts. Coordinates are float (x, y) in pixel units of the canvas."""

import numpy as np


def circle_pins(n: int, size: int) -> np.ndarray:
    """`n` pins evenly spaced on the inscribed circle of a `size`x`size` canvas, starting at top."""
    c = (size - 1) / 2
    t = 2 * np.pi * np.arange(n) / n - np.pi / 2
    return np.stack([c + c * np.cos(t), c + c * np.sin(t)], axis=1)


def rect_pins(n: int, width: int, height: int) -> np.ndarray:
    """`n` pins evenly spaced along the border of a `width`x`height` canvas, clockwise from
    the top-left corner."""
    w, h = width - 1, height - 1
    perim = 2 * (w + h)
    s = np.arange(n) * perim / n
    xy = np.empty((n, 2))
    for i, d in enumerate(s):
        if d < w:
            xy[i] = (d, 0)
        elif d < w + h:
            xy[i] = (w, d - w)
        elif d < 2 * w + h:
            xy[i] = (w - (d - w - h), h)
        else:
            xy[i] = (0, h - (d - 2 * w - h))
    return xy


def make_pins(frame: str, n: int, size: int) -> np.ndarray:
    if frame == "circle":
        return circle_pins(n, size)
    if frame == "rect":
        return rect_pins(n, size, size)
    raise ValueError(f"unknown frame {frame!r} (expected 'circle' or 'rect')")


def pin_distance(i: int, j: int, n: int) -> int:
    """Number of pins between i and j going the short way round the frame."""
    d = abs(i - j) % n
    return min(d, n - d)
