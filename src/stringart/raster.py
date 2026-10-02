"""Line rasterization.

`aa_line` is a vectorized Xiaolin-Wu style anti-aliased line: every step along the major axis
splits unit coverage between the two pixels straddling the exact line. `line_pixels` is the
aliased (nearest-pixel) version used by the baseline solver.
"""

import numpy as np


def _major_minor(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    major = np.arange(np.ceil(x0), np.floor(x1) + 1)
    if x1 == x0:
        minor = np.full_like(major, y0)
    else:
        minor = y0 + (major - x0) * (y1 - y0) / (x1 - x0)
    return steep, major.astype(np.int64), minor


def _to_yx(steep, major, minor):
    return (major, minor) if steep else (minor, major)


def aa_line(p0, p1, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anti-aliased line from p0=(x, y) to p1. Returns (ys, xs, coverage), coverage in (0, 1]."""
    steep, major, minor = _major_minor(p0, p1)
    lo = np.floor(minor).astype(np.int64)
    frac = minor - lo
    major = np.concatenate([major, major])
    minor_px = np.concatenate([lo, lo + 1])
    w = np.concatenate([1.0 - frac, frac])
    ys, xs = _to_yx(steep, major, minor_px)
    h, wd = shape
    keep = (w > 1e-9) & (ys >= 0) & (ys < h) & (xs >= 0) & (xs < wd)
    return ys[keep], xs[keep], w[keep]


def line_pixels(p0, p1, shape) -> tuple[np.ndarray, np.ndarray]:
    """Aliased line (one pixel per major-axis step). Returns (ys, xs)."""
    steep, major, minor = _major_minor(p0, p1)
    ys, xs = _to_yx(steep, major, np.rint(minor).astype(np.int64))
    h, wd = shape
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < wd)
    return ys[keep], xs[keep]
