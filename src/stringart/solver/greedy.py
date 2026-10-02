"""Improved greedy solver (PLAN.md I1-I3).

* I1 physical thread model: the solver optimizes exactly what `render.Canvas` draws. A thread
  covers pixel p with alpha a = opacity * coverage_p (anti-aliased), and darkness composes
  multiplicatively: d <- d + a (1 - d). `opacity` can be derived from thread width / pixel size.
* I2 objective: weighted squared error E = sum_p w_p (t_p - d_p)^2 against target darkness t.
  Each candidate line is scored by the exact error drop dE. The solver stops by itself once
  no line from the current pin lowers E (or after `max_lines`).
* Viewing distance: with objective="blur" the error is measured after a Gaussian blur
  (sigma ~ thread spacing seen from afar), maintained incrementally (see `_line_blur`).
* I3 speed: anti-aliased lines are rasterized on the fly inside a numba kernel (same Wu
  traversal as `raster.aa_line`), so every candidate is scored at every step with no
  line cache.
"""

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit, prange

from . import SolveResult


@dataclass
class GreedyConfig:
    max_lines: int = 8000
    opacity: float = 0.2  # per-thread alpha on one pixel (thread width / pixel size)
    min_gap: int = 10  # skip near-neighbour pins (short chords hug the rim)
    max_repeats: int = 2  # how often the same chord may be used
    stop_tol: float = 0.0  # stop when the best dE <= stop_tol
    patience: int = 1  # consecutive non-improving steps before stopping (pixel objective)
    start_pin: int = 0
    objective: str = "pixel"  # "pixel": per-pixel error; "blur": error after Gaussian blur
    blur_sigma: float = 1.5  # viewing-distance blur for the "blur" objective, in px


@njit(cache=True, inline="always")
def _visit(d, t, w, alpha, flat, cov, apply):
    """Score (and optionally apply) one pixel. Returns its contribution to dE."""
    a = alpha * cov
    d0 = d[flat]
    d1 = d0 + a * (1.0 - d0)
    e0 = t[flat] - d0
    e1 = t[flat] - d1
    if apply:
        d[flat] = d1
    return w[flat] * (e0 * e0 - e1 * e1)


@njit(cache=True)
def _line(p0x, p0y, p1x, p1y, h, wd, d, t, w, alpha, apply):
    """Walk the Wu anti-aliased line p0->p1 (identical to raster.aa_line); return dE."""
    x0, y0, x1, y1 = p0x, p0y, p1x, p1y
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    slope = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
    total = 0.0
    m = math.ceil(x0)
    end = math.floor(x1)
    while m <= end:
        minor = y0 if x1 == x0 else y0 + (m - x0) * slope
        lo = math.floor(minor)
        frac = minor - lo
        for k in range(2):
            mi = lo + k
            cov = 1.0 - frac if k == 0 else frac
            if cov <= 1e-9:
                continue
            if steep:
                yy, xx = int(m), int(mi)
            else:
                yy, xx = int(mi), int(m)
            if 0 <= yy < h and 0 <= xx < wd:
                total += _visit(d, t, w, alpha, yy * wd + xx, cov, apply)
        m += 1.0
    return total


@njit(cache=True, parallel=True)
def _score_from(pins, cur, prev, counts, min_gap, max_repeats, h, wd, d, t, w, alpha, out):
    """out[j] = dE of chord cur -> j (-inf if not allowed). Read-only on d, so parallel."""
    n = pins.shape[0]
    for j in prange(n):
        gap = abs(cur - j) % n
        if min(gap, n - gap) < min_gap or j == prev or counts[cur, j] >= max_repeats:
            out[j] = -np.inf
        else:
            out[j] = _line(pins[cur, 0], pins[cur, 1], pins[j, 0], pins[j, 1], h, wd, d, t, w,
                           alpha, False)


@njit(cache=True)
def _solve(pins, h, wd, d, t, w, alpha, max_lines, min_gap, max_repeats, stop_tol, patience,
           start):
    n = pins.shape[0]
    counts = np.zeros((n, n), dtype=np.int32)
    seq = np.empty(max_lines + 1, dtype=np.int64)
    gains = np.empty(max_lines, dtype=np.float64)
    scores = np.empty(n)
    seq[0] = start
    cur, prev, k, bad = start, -1, 0, 0
    while k < max_lines:
        _score_from(pins, cur, prev, counts, min_gap, max_repeats, h, wd, d, t, w, alpha, scores)
        best, best_j = -np.inf, -1
        for j in range(n):  # first maximum, same tie-breaking as a serial scan
            if scores[j] > best:
                best, best_j = scores[j], j
        if best_j < 0:
            break
        if best <= stop_tol:
            bad += 1
            if bad >= patience:
                break
        else:
            bad = 0
        _line(pins[cur, 0], pins[cur, 1], pins[best_j, 0], pins[best_j, 1], h, wd, d, t, w, alpha,
              True)
        counts[cur, best_j] += 1
        counts[best_j, cur] += 1
        gains[k] = best
        k += 1
        seq[k] = best_j
        prev, cur = cur, best_j
    # Drop trailing non-improving lines taken while waiting out `patience`.
    while k > 0 and gains[k - 1] <= stop_tol:
        k -= 1
    return seq[: k + 1], gains[:k]


@njit(cache=True)
def _line_blur(p0x, p0y, p1x, p1y, h, wd, d, field, w, alpha, c0, apply, scratch):
    """Viewing-distance objective for one line.

    E_blur = sum w (G*e)^2 with e = t - d. Drawing the line adds delta = a (1 - d) along it, so
    dE = 2 <delta, G*(w G*e)> - sum w (G*delta)^2. `field` holds G*(w G*e). The second term for a
    thin line is ~ w c^2 / (L 2 sqrt(pi) sigma) per major-axis step (c = darkness added in that
    step, L = step length); `c0` = 1 / (2 sqrt(pi) sigma).
    If `apply`, darken d along the line and write delta into `scratch` instead of scoring.
    """
    x0, y0, x1, y1 = p0x, p0y, p1x, p1y
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    slope = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
    inv_len = 1.0 / math.sqrt(1.0 + slope * slope)
    total = 0.0
    m = math.ceil(x0)
    end = math.floor(x1)
    while m <= end:
        minor = y0 if x1 == x0 else y0 + (m - x0) * slope
        lo = math.floor(minor)
        frac = minor - lo
        col, wcol = 0.0, 0.0
        for k in range(2):
            mi = lo + k
            cov = 1.0 - frac if k == 0 else frac
            if cov <= 1e-9:
                continue
            if steep:
                yy, xx = int(m), int(mi)
            else:
                yy, xx = int(mi), int(m)
            if 0 <= yy < h and 0 <= xx < wd:
                f = yy * wd + xx
                delta = alpha * cov * (1.0 - d[f])
                if apply:
                    d[f] += delta
                    scratch[f] += delta
                else:
                    total += 2.0 * delta * field[f]
                    col += delta
                    wcol += w[f] * cov
        total -= c0 * inv_len * wcol * col * col
        m += 1.0
    return total


@njit(cache=True, parallel=True)
def _best_blur(pins, cur, prev, counts, min_gap, max_repeats, h, wd, d, field, w, alpha, c0,
               scratch):
    n = pins.shape[0]
    scores = np.empty(n)
    for j in prange(n):  # scoring is read-only on d / field
        gap = abs(cur - j) % n
        if min(gap, n - gap) < min_gap or j == prev or counts[cur, j] >= max_repeats:
            scores[j] = -np.inf
        else:
            scores[j] = _line_blur(pins[cur, 0], pins[cur, 1], pins[j, 0], pins[j, 1], h, wd, d,
                                   field, w, alpha, c0, False, scratch)
    best, best_j = -np.inf, -1
    for j in range(n):
        if scores[j] > best:
            best, best_j = scores[j], j
    return best_j, best


def _blur0(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with zero padding, so ROI updates match a full-image blur exactly."""
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)


def _solve_blur(pins, target_dark, w, cfg: "GreedyConfig"):
    h, wd = target_dark.shape
    sigma = cfg.blur_sigma
    pad = int(math.ceil(8 * sigma)) + 2  # two blurs, each with radius ~4 sigma in OpenCV
    d = np.zeros(h * wd)
    scratch = np.zeros((h, wd))
    w2 = w.reshape(h, wd)
    field = _blur0(w2 * _blur0(target_dark, sigma), sigma).ravel()
    c0 = 1.0 / (2.0 * math.sqrt(math.pi) * sigma)
    n = len(pins)
    counts = np.zeros((n, n), dtype=np.int32)
    seq, gains = [cfg.start_pin], []
    cur, prev = cfg.start_pin, -1
    sflat = scratch.ravel()
    while len(gains) < cfg.max_lines:
        j, g = _best_blur(pins, cur, prev, counts, cfg.min_gap, cfg.max_repeats, h, wd, d, field,
                          w, cfg.opacity, c0, sflat)
        if j < 0 or g <= cfg.stop_tol:
            break
        _line_blur(pins[cur, 0], pins[cur, 1], pins[j, 0], pins[j, 1], h, wd, d, field, w,
                   cfg.opacity, c0, True, sflat)
        # field -= G*(w G*delta), computed only in the line's padded bounding box.
        (ax, ay), (bx, by) = pins[cur], pins[j]
        y0, y1 = max(0, int(min(ay, by)) - pad), min(h, int(max(ay, by)) + pad + 2)
        x0, x1 = max(0, int(min(ax, bx)) - pad), min(wd, int(max(ax, bx)) + pad + 2)
        roi = scratch[y0:y1, x0:x1]
        upd = _blur0(w2[y0:y1, x0:x1] * _blur0(roi, sigma), sigma)
        field.reshape(h, wd)[y0:y1, x0:x1] -= upd
        roi[:] = 0.0
        counts[cur, j] += 1
        counts[j, cur] += 1
        seq.append(j)
        gains.append(g)
        prev, cur = cur, j
    return seq, gains


def line_gain(target, pins, i, j, dark, opacity, weights=None) -> float:
    """dE for drawing chord i->j on the given darkness canvas (does not modify it)."""
    h, wd = target.shape
    t = (1.0 - target).ravel()
    w = np.ones_like(t) if weights is None else np.asarray(weights, np.float64).ravel()
    p = np.asarray(pins, np.float64)
    return _line(p[i, 0], p[i, 1], p[j, 0], p[j, 1], h, wd, dark.ravel().copy(), t, w, opacity,
                 False)


def solve_greedy(target, pins, cfg: GreedyConfig, weights=None, progress=True) -> SolveResult:
    """`target`: grayscale [0, 1] (1 = white). Returns the pin sequence (length chosen by I2)."""
    if cfg.min_gap * 2 >= len(pins):
        raise ValueError(f"min_gap={cfg.min_gap} leaves no candidates with {len(pins)} pins")
    if not 0 < cfg.opacity <= 1:
        raise ValueError(f"opacity must be in (0, 1], got {cfg.opacity}")
    h, wd = target.shape
    t = np.ascontiguousarray((1.0 - target).ravel(), dtype=np.float64)
    w = (np.ones_like(t) if weights is None
         else np.ascontiguousarray(np.asarray(weights, np.float64).ravel()))
    if w.shape != t.shape:
        raise ValueError("weights must have the same shape as target")
    pins = np.ascontiguousarray(pins, dtype=np.float64)
    t0 = time.perf_counter()
    if cfg.objective == "pixel":
        seq, gains = _solve(pins, h, wd, np.zeros_like(t), t, w, float(cfg.opacity),
                            cfg.max_lines, cfg.min_gap, cfg.max_repeats, cfg.stop_tol,
                            cfg.patience, cfg.start_pin)
    elif cfg.objective == "blur":
        if cfg.blur_sigma <= 0:
            raise ValueError("blur_sigma must be > 0 for the blur objective")
        seq, gains = _solve_blur(pins, t.reshape(h, wd), w, cfg)
    else:
        raise ValueError(f"unknown objective {cfg.objective!r} (expected 'pixel' or 'blur')")
    elapsed = time.perf_counter() - t0
    if progress:
        print(f"greedy[{cfg.objective}]: {len(seq) - 1} lines in {elapsed:.2f}s (auto-stop)")
    return SolveResult([int(p) for p in seq], [float(g) for g in gains], elapsed)


def opacity_from_physical(thread_mm: float, frame_mm: float, size_px: int) -> float:
    """Fraction of a canvas pixel one thread covers: thread width / pixel size, capped at 1."""
    return min(1.0, thread_mm * size_px / frame_mm)
