"""Path refinement after greedy (PLAN.md I6): local search that keeps a single thread.

Greedy commits to every line forever. Refinement revisits the path with three moves, each a
local edit of the pin sequence, so the result is still one continuous thread:

* delete   a -> b -> c   becomes  a -> c         (drop pin b: one line fewer)
* reroute  a -> b -> c   becomes  a -> b' -> c   (move pin b)
* insert   a -> b        becomes  a -> x -> b    (one line more, anywhere on the path)

Removing a thread is exact under the multiplicative model: transmittance T = 1 - d is a
product of (1 - a_i), so dropping a line divides its factor back out, d <- 1 - (1-d)/(1-a).
Each candidate is scored on the canvas with the old lines removed, the best is applied
exactly, and it is kept only if the true weighted error E = sum w (t - d)^2 went down;
otherwise it is reverted. E therefore never increases.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

ADD_SCORE, ADD_APPLY, REM_SCORE, REM_APPLY = 0, 1, 2, 3


@dataclass
class RefineConfig:
    sweeps: int = 2
    insert: bool = True  # allow insert moves (can add lines)
    min_gain: float = 1e-9  # accept a move only if E drops by more than this


@njit(cache=True)
def _walk(p0x, p0y, p1x, p1y, h, wd, d, t, w, alpha, mode):
    """Wu AA line p0->p1 (same traversal as raster.aa_line). Returns the drop in E for
    adding (modes 0/1) or removing (2/3) the line; modes 1/3 also update d."""
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
            cov = 1.0 - frac if k == 0 else frac
            if cov <= 1e-9:
                continue
            mi = lo + k
            if steep:
                yy, xx = int(m), int(mi)
            else:
                yy, xx = int(mi), int(m)
            if 0 <= yy < h and 0 <= xx < wd:
                f = yy * wd + xx
                a = alpha * cov
                d0 = d[f]
                if mode <= ADD_APPLY:
                    d1 = d0 + a * (1.0 - d0)
                else:
                    d1 = 1.0 - (1.0 - d0) / (1.0 - a)
                    if d1 < 0.0:
                        d1 = 0.0
                e0 = t[f] - d0
                e1 = t[f] - d1
                total += w[f] * (e0 * e0 - e1 * e1)
                if mode == ADD_APPLY or mode == REM_APPLY:
                    d[f] = d1
        m += 1.0
    return total


@njit(cache=True)
def _line_op(pins, i, j, h, wd, d, t, w, alpha, mode):
    return _walk(pins[i, 0], pins[i, 1], pins[j, 0], pins[j, 1], h, wd, d, t, w, alpha, mode)


@njit(cache=True)
def _ok(i, j, n, counts, min_gap, max_repeats):
    if i == j:
        return False
    gap = abs(i - j) % n
    return min(gap, n - gap) >= min_gap and counts[i, j] < max_repeats


@njit(cache=True)
def _inc(counts, i, j, v):
    counts[i, j] += v
    counts[j, i] += v


@njit(cache=True, parallel=True)
def _score_via(pins, a, c, s1, s2, s3, h, wd, d, t, w, alpha, counts, min_gap, max_repeats, out):
    """out[x] = gain of adding a -> x -> c on the current canvas (-inf if not allowed).
    Read-only on d, so candidates are scored in parallel."""
    n = pins.shape[0]
    for x in prange(n):
        if (
            x == s1
            or x == s2
            or x == s3
            or not _ok(a, x, n, counts, min_gap, max_repeats)
            or not _ok(x, c, n, counts, min_gap, max_repeats)
        ):
            out[x] = -np.inf
        else:
            out[x] = _line_op(pins, a, x, h, wd, d, t, w, alpha, ADD_SCORE) + _line_op(
                pins, x, c, h, wd, d, t, w, alpha, ADD_SCORE
            )


@njit(cache=True)
def _argmax(out):
    best, best_i = -np.inf, -1
    for i in range(out.shape[0]):
        if out[i] > best:
            best, best_i = out[i], i
    return best, best_i


@njit(cache=True)
def _sweep(
    pins,
    h,
    wd,
    d,
    t,
    w,
    alpha,
    seq,
    length,
    counts,
    min_gap,
    max_repeats,
    do_insert,
    min_gain,
    max_len,
):
    """One pass over the path. `seq[:length]` is edited in place (capacity `max_len`).
    Returns (new length, total E drop, #deletes, #reroutes, #inserts)."""
    n = pins.shape[0]
    out = np.empty(n)
    gained = 0.0
    n_del, n_rer, n_ins = 0, 0, 0
    k = 1
    while k < length:
        # ---- moves on pin k (needs neighbours on both sides): delete / reroute ----
        if k < length - 1:
            a, b, c = seq[k - 1], seq[k], seq[k + 1]
            pa = seq[k - 2] if k >= 2 else -1
            pc = seq[k + 2] if k + 2 < length else -1
            base = _line_op(pins, a, b, h, wd, d, t, w, alpha, REM_APPLY)
            base += _line_op(pins, b, c, h, wd, d, t, w, alpha, REM_APPLY)
            _inc(counts, a, b, -1)
            _inc(counts, b, c, -1)
            best, best_b = -np.inf, -2  # -1 = delete
            if _ok(a, c, n, counts, min_gap, max_repeats) and pa != c and pc != a:
                best = _line_op(pins, a, c, h, wd, d, t, w, alpha, ADD_SCORE)
                best_b = -1
            _score_via(
                pins, a, c, b, pa, pc, h, wd, d, t, w, alpha, counts, min_gap, max_repeats, out
            )
            g, x = _argmax(out)
            if x >= 0 and g > best:
                best, best_b = g, x
            kept = False
            if best_b != -2:
                if best_b == -1:
                    real = base + _line_op(pins, a, c, h, wd, d, t, w, alpha, ADD_APPLY)
                    if real > min_gain:
                        _inc(counts, a, c, 1)
                        for q in range(k, length - 1):
                            seq[q] = seq[q + 1]
                        length -= 1
                        gained += real
                        n_del += 1
                        kept = True
                        k -= 1  # re-examine the pin that slid into position k
                    else:
                        _line_op(pins, a, c, h, wd, d, t, w, alpha, REM_APPLY)
                else:
                    x = best_b
                    real = base + _line_op(pins, a, x, h, wd, d, t, w, alpha, ADD_APPLY)
                    real += _line_op(pins, x, c, h, wd, d, t, w, alpha, ADD_APPLY)
                    if real > min_gain:
                        _inc(counts, a, x, 1)
                        _inc(counts, x, c, 1)
                        seq[k] = x
                        gained += real
                        n_rer += 1
                        kept = True
                    else:
                        _line_op(pins, x, c, h, wd, d, t, w, alpha, REM_APPLY)
                        _line_op(pins, a, x, h, wd, d, t, w, alpha, REM_APPLY)
            if not kept:  # restore a -> b -> c
                _line_op(pins, a, b, h, wd, d, t, w, alpha, ADD_APPLY)
                _line_op(pins, b, c, h, wd, d, t, w, alpha, ADD_APPLY)
                _inc(counts, a, b, 1)
                _inc(counts, b, c, 1)
        # ---- insert a pin into line (k-1 -> k) ----
        if do_insert and k >= 1 and length < max_len:
            a, b = seq[k - 1], seq[k]
            pa = seq[k - 2] if k >= 2 else -1
            pb = seq[k + 1] if k + 1 < length else -1
            base = _line_op(pins, a, b, h, wd, d, t, w, alpha, REM_APPLY)
            _inc(counts, a, b, -1)
            _score_via(
                pins, a, b, pa, pb, -1, h, wd, d, t, w, alpha, counts, min_gap, max_repeats, out
            )
            best, best_x = _argmax(out)
            kept = False
            if best_x >= 0 and base + best > min_gain:
                x = best_x
                real = base + _line_op(pins, a, x, h, wd, d, t, w, alpha, ADD_APPLY)
                real += _line_op(pins, x, b, h, wd, d, t, w, alpha, ADD_APPLY)
                if real > min_gain:
                    _inc(counts, a, x, 1)
                    _inc(counts, x, b, 1)
                    for q in range(length, k, -1):
                        seq[q] = seq[q - 1]
                    seq[k] = x
                    length += 1
                    gained += real
                    n_ins += 1
                    kept = True
                    k += 1  # skip past the new pin
                else:
                    _line_op(pins, x, b, h, wd, d, t, w, alpha, REM_APPLY)
                    _line_op(pins, a, x, h, wd, d, t, w, alpha, REM_APPLY)
            if not kept:
                _line_op(pins, a, b, h, wd, d, t, w, alpha, ADD_APPLY)
                _inc(counts, a, b, 1)
        k += 1
    return length, gained, n_del, n_rer, n_ins


def _canvas(seq, pins, h, wd, t, w, alpha):
    d = np.zeros(h * wd)
    for i, j in zip(seq[:-1], seq[1:], strict=True):
        _line_op(pins, i, j, h, wd, d, t, w, alpha, ADD_APPLY)
    return d


def weighted_error(d, t, w) -> float:
    return float(np.sum(w * (t - d) ** 2))


def refine(
    target,
    pins,
    sequence,
    opacity,
    min_gap=10,
    max_repeats=2,
    weights=None,
    cfg: RefineConfig | None = None,
    progress=True,
):
    """Return (new sequence, stats dict). E never increases."""
    cfg = cfg or RefineConfig()
    if not 0 < opacity < 1:
        raise ValueError("refinement needs 0 < opacity < 1 (removal divides by 1 - opacity)")
    h, wd = target.shape
    t = np.ascontiguousarray((1.0 - target).ravel(), dtype=np.float64)
    w = (
        np.ones_like(t)
        if weights is None
        else np.ascontiguousarray(np.asarray(weights, np.float64).ravel())
    )
    pins = np.ascontiguousarray(pins, dtype=np.float64)
    n = len(pins)
    t0 = time.perf_counter()
    d = _canvas(sequence, pins, h, wd, t, w, opacity)
    e_start = weighted_error(d, t, w)
    counts = np.zeros((n, n), dtype=np.int32)
    for i, j in zip(sequence[:-1], sequence[1:], strict=True):
        counts[i, j] += 1
        counts[j, i] += 1
    max_len = 2 * len(sequence) + 16
    seq = np.zeros(max_len + 1, dtype=np.int64)
    seq[: len(sequence)] = sequence
    length = len(sequence)
    stats = {"sweeps": [], "e_start": e_start}
    for s in range(cfg.sweeps):
        length, gained, nd, nr, ni = _sweep(
            pins,
            h,
            wd,
            d,
            t,
            w,
            float(opacity),
            seq,
            length,
            counts,
            min_gap,
            max_repeats,
            cfg.insert,
            cfg.min_gain,
            max_len,
        )
        stats["sweeps"].append(
            {"gain": gained, "deleted": nd, "rerouted": nr, "inserted": ni, "lines": length - 1}
        )
        if progress:
            print(
                f"refine sweep {s + 1}: dE={gained:.2f}  -{nd} lines, ~{nr} moved, "
                f"+{ni} lines -> {length - 1}"
            )
        if nd + nr + ni == 0:
            break
    stats["e_end"] = weighted_error(d, t, w)
    stats["elapsed_s"] = time.perf_counter() - t0
    return [int(p) for p in seq[:length]], stats
