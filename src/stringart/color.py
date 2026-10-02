"""Colour string art (PLAN.md I7).

Thread model: a thread of colour c covering a pixel with alpha a = opacity * coverage is
composited *over* what is beneath it, per RGB channel: C <- C (1 - a) + c a. With a black
thread on a white board this is exactly the grayscale model (d <- d + a (1 - d)), so the
colour solver generalizes M2.

* Palette: k-means in CIELAB (perceptual) via cv2.kmeans, snapped to real thread colours.
* Joint greedy: every colour is its own physical thread with its own current pin. Each step
  takes the (colour, line) with the largest drop in weighted squared RGB error. A colour, once
  chosen, is kept for at least `min_run` lines, so the build switches spools a handful of
  times, not every line. The render order equals the build order, so "over" compositing
  matches the physical layering.
* Baseline (LessWrong): Floyd-Steinberg dither into the palette, then an independent greedy
  per colour on its dither mask, layered light to dark.
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from numba import njit, prange

from .preprocess import PreprocessConfig, Prepared, prepare

# Common sewing / embroidery thread colours (sRGB). Black and white are always available.
THREADS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128),
    "red": (196, 30, 45), "maroon": (110, 20, 35), "orange": (235, 110, 30),
    "yellow": (245, 200, 30), "tan": (210, 160, 120), "brown": (110, 70, 40),
    "green": (40, 140, 60), "dark_green": (20, 75, 40), "cyan": (40, 170, 200),
    "blue": (30, 80, 170), "navy": (20, 30, 80), "purple": (100, 50, 140),
    "pink": (235, 130, 170),
}


@dataclass
class ColorConfig:
    n_colors: int = 4  # threads, including black
    palette: list[str] | None = None  # explicit thread names; overrides k-means
    opacity: float = 0.2
    max_lines: int = 12000
    min_gap: int = 10
    max_repeats: int = 2
    min_run: int = 100  # lines before the solver may switch to another colour
    background: tuple[int, int, int] = (255, 255, 255)


@dataclass
class ColorResult:
    palette: list[str]
    colors: np.ndarray  # (K, 3) float RGB in [0, 1]
    steps: list[tuple[int, int, int]]  # (colour index, from pin, to pin), in build order
    gains: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0

    def sequences(self) -> dict[str, list[int]]:
        """Pin sequence per colour (each is one continuous thread)."""
        out: dict[str, list[int]] = {}
        for k, a, b in self.steps:
            seq = out.setdefault(self.palette[k], [a])
            seq.append(b)
        return out


# ------------------------------------------------------------------ targets and palettes
def color_target(prep: Prepared, clahe_clip: float = 2.0) -> np.ndarray:
    """RGB target in [0, 1] from the prepared crop: CLAHE on Lab lightness, white outside."""
    lab = cv2.cvtColor(prep.canvas_bgr, cv2.COLOR_BGR2LAB)
    if clahe_clip > 0:
        lab[..., 0] = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(
            lab[..., 0])
    rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float64) / 255.0
    rgb[~prep.mask] = 1.0
    return rgb


def _lab(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb01.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(
        -1, 3)


def auto_palette(target_rgb: np.ndarray, mask: np.ndarray, n_colors: int,
                 seed: int = 0, space: str = "lab") -> list[str]:
    """k-means on the image, each centre snapped to the nearest thread colour. `space="lab"`
    (perceptual, default) or `"rgb"` (as in prior work, for comparison). Black is always
    included; white never is (on a white board it can only lighten threads, and the board
    itself already supplies white)."""
    names = list(THREADS)
    if space == "lab":
        thread_lab = _lab(np.array([THREADS[n] for n in names]) / 255.0)
        px = _lab(target_rgb[mask])
    elif space == "rgb":
        thread_lab = np.array([THREADS[n] for n in names], dtype=np.float32)
        px = (target_rgb[mask] * 255).astype(np.float32)
    else:
        raise ValueError(f"unknown palette space {space!r}")
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    k = min(max(n_colors + 1, 2), len(px))  # +1: the white board usually takes one centre
    _, labels, centers = cv2.kmeans(px.astype(np.float32), k, None, crit, 3,
                                    cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=k))  # most common first
    chosen = ["black"]
    for c in centers[order]:
        name = names[int(np.argmin(np.linalg.norm(thread_lab - c, axis=1)))]
        if name not in chosen and name != "white":
            chosen.append(name)
        if len(chosen) == n_colors:
            break
    for extra in ("red", "blue", "yellow", "brown"):  # rarely needed: tiny palettes
        if len(chosen) >= n_colors:
            break
        if extra not in chosen:
            chosen.append(extra)
    return chosen[:n_colors]


def palette_rgb(names: list[str]) -> np.ndarray:
    unknown = [n for n in names if n not in THREADS]
    if unknown:
        raise ValueError(f"unknown thread colour(s) {unknown}; choose from {sorted(THREADS)}")
    return np.array([THREADS[n] for n in names], dtype=np.float64) / 255.0


# ------------------------------------------------------------------ renderer
class ColorCanvas:
    def __init__(self, shape, colors: np.ndarray, opacity: float, background=(1.0, 1.0, 1.0)):
        self.rgb = np.empty((*shape, 3))
        self.rgb[:] = background
        self.colors = np.asarray(colors, dtype=np.float64)
        self.opacity = opacity
        self.shape = shape

    def add_line(self, k: int, p0, p1) -> None:
        from .raster import aa_line

        ys, xs, w = aa_line(p0, p1, self.shape)
        a = (self.opacity * w)[:, None]
        c = self.rgb[ys, xs]
        self.rgb[ys, xs] = c + a * (self.colors[k] - c)

    def image(self) -> np.ndarray:
        return self.rgb


def render_steps(steps, pins, shape, colors, opacity, background=(1.0, 1.0, 1.0)):
    canvas = ColorCanvas(shape, colors, opacity, background)
    for k, a, b in steps:
        canvas.add_line(k, pins[a], pins[b])
    return canvas


# ------------------------------------------------------------------ joint colour greedy
@njit(cache=True)
def _walk_rgb(p0x, p0y, p1x, p1y, h, wd, C, T, w, alpha, col, apply):
    """Drop in E = sum w ||T - C||^2 for compositing a line of colour `col` over C."""
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
                g = 0.0
                for ch in range(3):
                    c0 = C[f, ch]
                    c1 = c0 + a * (col[ch] - c0)
                    e0 = T[f, ch] - c0
                    e1 = T[f, ch] - c1
                    g += e0 * e0 - e1 * e1
                    if apply:
                        C[f, ch] = c1
                total += w[f] * g
        m += 1.0
    return total


@njit(cache=True, parallel=True)
def _score_rgb(pins, cur, prev, counts, min_gap, max_repeats, h, wd, C, T, w, alpha, col, out):
    n = pins.shape[0]
    for j in prange(n):
        gap = abs(cur - j) % n
        if min(gap, n - gap) < min_gap or j == prev or counts[cur, j] >= max_repeats:
            out[j] = -np.inf
        else:
            out[j] = _walk_rgb(pins[cur, 0], pins[cur, 1], pins[j, 0], pins[j, 1], h, wd, C, T,
                               w, alpha, col, False)


@njit(cache=True)
def _best(out):
    best, bj = -np.inf, -1
    for j in range(out.shape[0]):
        if out[j] > best:
            best, bj = out[j], j
    return best, bj


@njit(cache=True)
def _solve_rgb(pins, h, wd, C, T, w, alpha, colors, max_lines, min_gap, max_repeats, min_run,
               starts):
    n, K = pins.shape[0], colors.shape[0]
    counts = np.zeros((K, n, n), dtype=np.int32)
    cur = starts.copy()
    prev = np.full(K, -1)
    steps = np.empty((max_lines, 3), dtype=np.int64)
    gains = np.empty(max_lines)
    out = np.empty(n)
    active, run, s = -1, 0, 0
    while s < max_lines:
        best, bk, bj = -np.inf, -1, -1
        if active >= 0 and run < min_run:
            _score_rgb(pins, cur[active], prev[active], counts[active], min_gap, max_repeats,
                       h, wd, C, T, w, alpha, colors[active], out)
            best, bj = _best(out)
            bk = active
        if best <= 0.0:  # run finished (or exhausted): consider every colour
            for k in range(K):
                _score_rgb(pins, cur[k], prev[k], counts[k], min_gap, max_repeats, h, wd, C, T,
                           w, alpha, colors[k], out)
                g, j = _best(out)
                if g > best:
                    best, bk, bj = g, k, j
        if bj < 0 or best <= 0.0:
            break
        _walk_rgb(pins[cur[bk], 0], pins[cur[bk], 1], pins[bj, 0], pins[bj, 1], h, wd, C, T, w,
                  alpha, colors[bk], True)
        counts[bk, cur[bk], bj] += 1
        counts[bk, bj, cur[bk]] += 1
        steps[s, 0], steps[s, 1], steps[s, 2] = bk, cur[bk], bj
        gains[s] = best
        run = run + 1 if bk == active else 1
        active = bk
        prev[bk], cur[bk] = cur[bk], bj
        s += 1
    return steps[:s], gains[:s]


def solve_color(target_rgb, pins, colors, cfg: ColorConfig, weights=None,
                names=None) -> ColorResult:
    h, wd, _ = target_rgb.shape
    if not 0 < cfg.opacity <= 1:
        raise ValueError(f"opacity must be in (0, 1], got {cfg.opacity}")
    T = np.ascontiguousarray(target_rgb.reshape(-1, 3), dtype=np.float64)
    C = np.empty_like(T)
    C[:] = np.asarray(cfg.background, dtype=np.float64) / 255.0
    w = (np.ones(h * wd) if weights is None
         else np.ascontiguousarray(np.asarray(weights, np.float64).ravel()))
    pins = np.ascontiguousarray(pins, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)
    n, K = len(pins), len(colors)
    starts = (np.arange(K) * n // max(K, 1)).astype(np.int64)  # spread the start pins
    t0 = time.perf_counter()
    steps, gains = _solve_rgb(pins, h, wd, C, T, w, float(cfg.opacity), colors, cfg.max_lines,
                              cfg.min_gap, cfg.max_repeats, cfg.min_run, starts)
    return ColorResult(names or [f"c{k}" for k in range(K)], colors,
                       [tuple(int(v) for v in s) for s in steps], gains.tolist(),
                       time.perf_counter() - t0)


# ------------------------------------------------------------------ LessWrong-style baseline
def dither(target_rgb: np.ndarray, colors_with_bg: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error diffusion in RGB onto a palette. Returns the index map."""
    img = target_rgb.astype(np.float64).copy()
    h, wd, _ = img.shape
    return _fs(img, np.ascontiguousarray(colors_with_bg, dtype=np.float64), h, wd)


@njit(cache=True)
def _fs(img, pal, h, wd):
    idx = np.zeros((h, wd), dtype=np.int64)
    for y in range(h):
        for x in range(wd):
            old = img[y, x].copy()
            best, bi = 1e18, 0
            for i in range(pal.shape[0]):
                dd = 0.0
                for ch in range(3):
                    v = old[ch] - pal[i, ch]
                    dd += v * v
                if dd < best:
                    best, bi = dd, i
            idx[y, x] = bi
            err = old - pal[bi]
            if x + 1 < wd:
                img[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += err * (3 / 16)
                img[y + 1, x] += err * (5 / 16)
                if x + 1 < wd:
                    img[y + 1, x + 1] += err * (1 / 16)
    return idx


def solve_color_baseline(target_rgb, pins, colors, cfg: ColorConfig, names=None,
                         blur_sigma: float = 1.5) -> ColorResult:
    """Dither into palette + white board; per colour, greedy on its blurred dither mask
    (independent of the other colours); build order light -> dark (LessWrong's advice)."""
    from .solver.greedy import GreedyConfig, solve_greedy

    t0 = time.perf_counter()
    bg = np.asarray(cfg.background, dtype=np.float64)[None] / 255.0
    idx = dither(target_rgb, np.vstack([colors, bg]))
    lum = colors @ np.array([0.299, 0.587, 0.114])
    steps, gains = [], []
    gcfg = GreedyConfig(opacity=cfg.opacity, min_gap=cfg.min_gap, max_repeats=cfg.max_repeats,
                        max_lines=cfg.max_lines)
    for k in np.argsort(-lum):  # lightest first, darkest on top
        mask = cv2.GaussianBlur((idx == k).astype(np.float64), (0, 0), blur_sigma)
        res = solve_greedy(1.0 - mask, pins, gcfg, progress=False)
        steps += [(int(k), a, b) for a, b in zip(res.sequence[:-1], res.sequence[1:],
                                                 strict=True)]
        gains += res.scores
    return ColorResult(names or [f"c{k}" for k in range(len(colors))], colors, steps, gains,
                       time.perf_counter() - t0)


# ------------------------------------------------------------------ metrics
def color_metrics(target_rgb, render_rgb, mask, sigmas=(2, 4)) -> dict[str, float]:
    """Mean CIEDE2000 and luminance SSIM after viewing blur, inside the frame."""
    from skimage.color import deltaE_ciede2000, rgb2lab
    from skimage.metrics import structural_similarity

    out = {}
    for s in sigmas:
        t = cv2.GaussianBlur(target_rgb, (0, 0), s) if s else target_rgb
        r = cv2.GaussianBlur(render_rgb, (0, 0), s) if s else render_rgb
        de = deltaE_ciede2000(rgb2lab(t), rgb2lab(r))
        out[f"de2000_s{s}"] = round(float(de[mask].mean()), 3)
        yt = t @ np.array([0.299, 0.587, 0.114])
        yr = r @ np.array([0.299, 0.587, 0.114])
        _, smap = structural_similarity(yt, yr, data_range=1.0, full=True)
        out[f"ssim_lum_s{s}"] = round(float(smap[mask].mean()), 4)
    return out


def prepare_color(img_bgr, size=600, frame="circle", n_colors=4, palette=None,
                  pcfg: PreprocessConfig | None = None):
    """Convenience: crop/prepare -> colour target -> palette names and RGB."""
    prep = prepare(img_bgr, pcfg or PreprocessConfig(size=size, frame=frame))
    target = color_target(prep)
    names = palette or auto_palette(target, prep.mask, n_colors)
    return prep, target, names, palette_rgb(names)
