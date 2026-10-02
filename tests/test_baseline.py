import numpy as np
import pytest

from stringart.geometry import pin_distance
from stringart.metrics import evaluate
from stringart.render import render_sequence
from stringart.solver.baseline import BaselineConfig, solve_baseline


def test_first_line_follows_dark_chord(pins):
    # Dark stroke exactly along the chord 0 -> 32 (top to bottom through the centre).
    t = np.ones((120, 120))
    t[:, 58:62] = 0.0
    res = solve_baseline(t, pins, BaselineConfig(n_lines=1, min_gap=5), progress=False)
    assert res.sequence == [0, 32]


def test_constraints_and_determinism(pins, target):
    cfg = BaselineConfig(n_lines=300, min_gap=6, n_candidates=20, seed=3)
    a = solve_baseline(target, pins, cfg, progress=False)
    b = solve_baseline(target, pins, cfg, progress=False)
    assert a.sequence == b.sequence and len(a.sequence) == 301
    s = a.sequence
    assert all(pin_distance(i, j, len(pins)) >= 6 for i, j in zip(s, s[1:], strict=False))
    assert all(s[k] != s[k + 2] for k in range(len(s) - 2))  # no immediate back-and-forth


def test_lines_improve_on_blank_board(pins, target):
    # Note: the baseline has no stopping rule, so *too many* lines over-darken (on this 120 px
    # canvas 400 lines score worse than 40). That is the weakness PLAN.md I2 addresses.
    res = solve_baseline(target, pins, BaselineConfig(n_lines=60, min_gap=6), progress=False)
    blank = np.ones_like(target)
    drawn = render_sequence(res.sequence, pins, target.shape, 0.2).image()
    assert (
        evaluate(target, drawn, sigmas=(2,))["psnr_s2"]
        > evaluate(target, blank, sigmas=(2,))["psnr_s2"]
    )


def test_min_gap_too_large_raises(pins, target):
    with pytest.raises(ValueError):
        solve_baseline(target, pins, BaselineConfig(n_lines=1, min_gap=32), progress=False)
