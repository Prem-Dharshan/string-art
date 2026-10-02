"""Physical calibration: measure the real thread's effective opacity from one photo.

The simulation assumes each thread darkens a pixel by `opacity = thread width / pixel size`.
Real thread is not perfectly black, has fuzz, and casts small shadows, so the right value for
a given thread and frame is measured, not guessed:

1. `make_sheet` builds a short calibration pattern (~250 lines with regions of different
   thread density) for the user's frame, with its build sheet.
2. The user winds it and photographs it from the front, frame upright (pin 0 at the top).
3. `fit_opacity` finds the frame circle in the photo (Hough), normalizes brightness to the
   bare board, and searches for the opacity whose simulated render best matches the photo
   after viewing blur. The result converts to an effective thread width for `--thread-mm`.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .fabrication import instructions
from .geometry import make_pins
from .preprocess import frame_mask
from .render import render_sequence
from .solver.greedy import GreedyConfig, solve_greedy

SIZE = 600


def calibration_target(size: int = SIZE) -> np.ndarray:
    """Three vertical bands, dark / mid / light, so the pattern spans thread densities."""
    t = np.ones((size, size))
    third = size // 3
    t[:, :third] = 0.25
    t[:, third : 2 * third] = 0.55
    t[:, 2 * third :] = 0.85
    t[~frame_mask("circle", size)] = 1.0
    return t


def make_sheet(n_pins: int, frame_mm: float, n_lines: int = 250, opacity: float = 0.25):
    """Return (sequence, pins, instructions text) for the calibration pattern."""
    pins = make_pins("circle", n_pins, SIZE)
    gap = max(2, round(10 * n_pins / 256))
    res = solve_greedy(
        calibration_target(),
        pins,
        GreedyConfig(opacity=opacity, min_gap=gap, max_lines=n_lines, stop_tol=-np.inf),
        progress=False,
    )
    text = instructions(res.sequence, pins, SIZE, "circle", frame_mm)
    header = (
        "CALIBRATION PATTERN: wind this, photograph it from the front (frame upright, pin 0 at "
        "the top, even light, no flash), then run `stringart calibrate fit <photo>`.\n\n"
    )
    return res.sequence, pins, header + text


def find_frame(photo_gray: np.ndarray) -> tuple[float, float, float]:
    """Largest circle in the photo = the frame: (cx, cy, r) in photo pixels."""
    h, w = photo_gray.shape
    s = 800 / max(h, w)
    small = cv2.resize(photo_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2)
    m = min(small.shape)
    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=m,
        param1=120,
        param2=40,
        minRadius=int(0.25 * m),
        maxRadius=int(0.6 * m),
    )
    if circles is None:
        raise ValueError("could not find the circular frame; pass --circle cx,cy,r")
    cx, cy, r = max(circles[0], key=lambda c: c[2])
    return cx / s, cy / s, r / s


def bare_mask(sequence, n_pins: int) -> np.ndarray:
    """Canvas pixels the pattern never touches (eroded): where the photo shows bare board."""
    pins = make_pins("circle", n_pins, SIZE)
    d = 1.0 - render_sequence(sequence, pins, (SIZE, SIZE), 0.5).image()
    d = cv2.GaussianBlur(d, (0, 0), 3)
    inner = cv2.erode(frame_mask("circle", SIZE).astype(np.uint8), np.ones((25, 25), np.uint8))
    return (d < 1e-3) & inner.astype(bool)


def photo_darkness(
    photo_bgr: np.ndarray, circle=None, board_level=None, bare: np.ndarray | None = None
) -> np.ndarray:
    """Crop the frame to the 600 px canvas and return observed darkness in [0, 1]
    (0 = bare board). Board brightness = the median over `bare` (pixels the pattern never
    touches) when given, else a high percentile inside the frame."""
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY) if photo_bgr.ndim == 3 else photo_bgr
    cx, cy, r = circle or find_frame(gray)
    # Affine map: frame circle -> the canvas circle (centre (S-1)/2, radius (S-1)/2).
    c = (SIZE - 1) / 2
    k = c / r
    M = np.array([[k, 0, c - k * cx], [0, k, c - k * cy]], dtype=np.float64)
    crop = cv2.warpAffine(
        gray.astype(np.float64), M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderValue=0
    )
    mask = frame_mask("circle", SIZE)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if board_level is None and bare is not None and bare.sum() > 500:
        board_level = float(np.median(crop[bare]))
    board = board_level or float(np.percentile(crop[inner], 97))
    dark = np.clip(1.0 - crop / max(board, 1e-6), 0.0, 1.0)
    dark[~mask] = 0.0
    return dark


@dataclass
class Fit:
    opacity: float
    thread_mm: float
    error: float
    curve: list[tuple[float, float]]  # (opacity, error) samples


def fit_opacity(
    sequence, n_pins: int, frame_mm: float, observed_dark: np.ndarray, sigma: float = 2.0
) -> Fit:
    """Opacity whose render best matches the observed darkness (MSE after viewing blur)."""
    pins = make_pins("circle", n_pins, SIZE)
    mask = frame_mask("circle", SIZE)
    obs = cv2.GaussianBlur(observed_dark, (0, 0), sigma)

    def err(op: float) -> float:
        d = 1.0 - render_sequence(sequence, pins, (SIZE, SIZE), op).image()
        return float(np.mean((cv2.GaussianBlur(d, (0, 0), sigma) - obs)[mask] ** 2))

    grid = np.linspace(0.04, 0.8, 20)
    curve = [(float(o), err(o)) for o in grid]
    best = min(curve, key=lambda c: c[1])[0]
    lo, hi = max(0.01, best - 0.04), min(0.95, best + 0.04)
    for _ in range(18):  # golden-section refine
        a, b = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
        if err(a) < err(b):
            hi = b
        else:
            lo = a
    op = (lo + hi) / 2
    return Fit(op, op * frame_mm / SIZE, err(op), curve)
