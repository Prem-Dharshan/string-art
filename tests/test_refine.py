import numpy as np
import pytest

from stringart.geometry import pin_distance
from stringart.render import render_sequence
from stringart.solver.greedy import GreedyConfig, solve_greedy
from stringart.solver.refine import RefineConfig, refine


def _err(target, seq, pins, op, w=None):
    img = render_sequence(seq, pins, target.shape, op).image()
    w = np.ones_like(target) if w is None else w
    return float(np.sum(w * (img - target) ** 2))


def _valid(seq, n, min_gap, max_repeats):
    pairs = list(zip(seq, seq[1:], strict=False))
    assert all(pin_distance(i, j, n) >= min_gap for i, j in pairs)
    assert all(seq[k] != seq[k + 2] for k in range(len(seq) - 2))
    chords = [tuple(sorted(p)) for p in pairs]
    assert max(chords.count(c) for c in set(chords)) <= max_repeats


def test_refine_never_worse_and_tracks_true_error(pins, target):
    g = solve_greedy(target, pins, GreedyConfig(opacity=0.25, min_gap=6), progress=False)
    seq, st = refine(target, pins, g.sequence, 0.25, min_gap=6, progress=False)
    assert st["e_end"] <= st["e_start"] + 1e-9
    assert st["e_start"] == pytest.approx(_err(target, g.sequence, pins, 0.25), rel=1e-9)
    assert st["e_end"] == pytest.approx(_err(target, seq, pins, 0.25), rel=1e-6)
    _valid(seq, len(pins), 6, 2)


def test_refine_fixes_a_bad_path(pins):
    # Target: one dark vertical band. Start from a path with two useless horizontal detours.
    t = np.ones((120, 120))
    t[:, 56:64] = 0.0
    bad = [0, 32, 16, 48, 0, 32]  # 0->32 is the band; 16<->48 are horizontal, mostly wasted
    seq, st = refine(t, pins, bad, 0.4, min_gap=6, progress=False, cfg=RefineConfig(sweeps=3))
    assert st["e_end"] < st["e_start"]
    assert _err(t, seq, pins, 0.4) < _err(t, bad, pins, 0.4)
    _valid(seq, len(pins), 6, 2)


def test_refine_respects_weights(pins, target):
    w = np.full_like(target, 0.2)
    w[40:80, 40:80] = 1.0
    g = solve_greedy(target, pins, GreedyConfig(min_gap=6), weights=w, progress=False)
    seq, st = refine(target, pins, g.sequence, 0.2, min_gap=6, weights=w, progress=False)
    assert st["e_end"] == pytest.approx(_err(target, seq, pins, 0.2, w), rel=1e-6)
    assert st["e_end"] <= st["e_start"] + 1e-9


def test_refine_without_insert_never_adds_lines(pins, target):
    g = solve_greedy(target, pins, GreedyConfig(min_gap=6), progress=False)
    seq, _ = refine(
        target, pins, g.sequence, 0.2, min_gap=6, progress=False, cfg=RefineConfig(insert=False)
    )
    assert len(seq) <= len(g.sequence)


def test_refine_rejects_opaque_thread(pins, target):
    with pytest.raises(ValueError):
        refine(target, pins, [0, 20, 40], 1.0, min_gap=6, progress=False)
