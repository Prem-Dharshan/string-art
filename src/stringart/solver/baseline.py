"""Baseline greedy solver, after Vrellis and the LessWrong "Computational Thread Art" post.

Residual r starts as the target darkness. From the current pin every candidate line is scored by
its mean weighted residual (over-darkened pixels, r < 0, are penalized by `darkness_penalty`);
the best line is taken and `line_strength` is subtracted along its aliased pixels.

Speed: per pin we cache the concatenated flat pixel indices of all its candidate lines, so one
`np.add.reduceat` scores every candidate at once.
"""

import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from ..geometry import pin_distance
from ..raster import line_pixels
from . import SolveResult


@dataclass
class BaselineConfig:
    n_lines: int = 3000
    line_strength: float = 0.2
    min_gap: int = 10  # skip near-neighbour pins (short chords hug the rim)
    n_candidates: int | None = None  # random subset per step (LessWrong); None = all
    darkness_penalty: float = 0.0  # D in the LessWrong penalty
    start_pin: int = 0
    seed: int = 0


class _PinBundle:
    """All candidate lines leaving one pin, as concatenated flat pixel indices."""

    __slots__ = ("targets", "idx", "starts", "lengths")

    def __init__(self, targets, idx, starts, lengths):
        self.targets, self.idx, self.starts, self.lengths = targets, idx, starts, lengths

    def line(self, k: int) -> np.ndarray:
        return self.idx[self.starts[k] : self.starts[k] + self.lengths[k]]


def _build_bundle(i, pins, shape, min_gap) -> _PinBundle:
    n, width = len(pins), shape[1]
    targets = np.array([j for j in range(n) if pin_distance(i, j, n) >= min_gap])
    flats = []
    for j in targets:
        ys, xs = line_pixels(pins[i], pins[j], shape)
        flats.append((ys * width + xs).astype(np.int32))
    lengths = np.array([len(f) for f in flats])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return _PinBundle(targets, np.concatenate(flats), starts, lengths)


def solve_baseline(target, pins, cfg: BaselineConfig, weights=None, progress=True) -> SolveResult:
    """`target`: grayscale [0, 1] (1 = white). Returns the pin sequence."""
    if cfg.min_gap * 2 >= len(pins):
        raise ValueError(f"min_gap={cfg.min_gap} leaves no candidates with {len(pins)} pins")
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    shape = target.shape
    resid = (1.0 - target).ravel().copy()
    w = np.ones_like(resid) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    bundles: dict[int, _PinBundle] = {}

    seq, scores = [cfg.start_pin], []
    cur, prev = cfg.start_pin, -1
    for _ in tqdm(range(cfg.n_lines), disable=not progress, desc="baseline", unit="line"):
        b = bundles.get(cur)
        if b is None:
            b = bundles[cur] = _build_bundle(cur, pins, shape, cfg.min_gap)
        r = resid[b.idx]
        vals = w[b.idx] * (np.maximum(r, 0) + cfg.darkness_penalty * np.minimum(r, 0))
        s = np.add.reduceat(vals, b.starts) / b.lengths
        s[b.targets == prev] = -np.inf  # no immediate back-and-forth
        if cfg.n_candidates is not None and cfg.n_candidates < len(s):
            drop = rng.choice(len(s), len(s) - cfg.n_candidates, replace=False)
            s[drop] = -np.inf
        k = int(np.argmax(s))
        resid[b.line(k)] -= cfg.line_strength
        prev, cur = cur, int(b.targets[k])
        seq.append(cur)
        scores.append(float(s[k]))
    return SolveResult(seq, scores, time.perf_counter() - t0)
