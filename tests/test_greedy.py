import math

import numpy as np
import pytest

from stringart.geometry import pin_distance
from stringart.render import Canvas, render_sequence
from stringart.solver.greedy import (
    GreedyConfig,
    _blur0,
    line_gain,
    opacity_from_physical,
    solve_greedy,
)


def _err(target, img):
    return float(np.sum((img - target) ** 2))  # (t_dark - d)^2 == (target - image)^2


def test_line_gain_matches_brute_force(pins, target):
    c = Canvas(target.shape, 0.3)
    c.add_line(pins[3], pins[40])
    c.add_line(pins[40], pins[10])
    before = _err(target, c.image())
    gain = line_gain(target, pins, 10, 45, c.dark, 0.3)
    c.add_line(pins[10], pins[45])
    assert gain == pytest.approx(before - _err(target, c.image()), rel=1e-9)


def test_gains_sum_to_error_drop_of_rendered_sequence(pins, target):
    """The solver's internal model is exactly the renderer: sum of gains == E(blank) - E(render)."""
    res = solve_greedy(target, pins, GreedyConfig(opacity=0.25, min_gap=6), progress=False)
    img = render_sequence(res.sequence, pins, target.shape, 0.25).image()
    drop = _err(target, np.ones_like(target)) - _err(target, img)
    assert len(res.sequence) > 10
    assert all(g > 0 for g in res.scores)
    assert sum(res.scores) == pytest.approx(drop, rel=1e-6)


def test_auto_stop_on_blank_target(pins):
    res = solve_greedy(np.ones((120, 120)), pins, GreedyConfig(min_gap=6), progress=False)
    assert res.sequence == [0]


def test_constraints(pins, target):
    cfg = GreedyConfig(opacity=0.5, min_gap=6, max_repeats=1, max_lines=400)
    s = solve_greedy(target, pins, cfg, progress=False).sequence
    n = len(pins)
    assert all(pin_distance(i, j, n) >= 6 for i, j in zip(s, s[1:], strict=False))
    assert all(s[k] != s[k + 2] for k in range(len(s) - 2))
    chords = [tuple(sorted(p)) for p in zip(s, s[1:], strict=False)]
    assert len(chords) == len(set(chords))  # max_repeats=1


def test_max_lines_cap(pins, target):
    res = solve_greedy(target, pins, GreedyConfig(max_lines=25, min_gap=6), progress=False)
    assert len(res.sequence) == 26


def test_blur_objective_gains_match_blurred_error_drop(pins, target):
    sigma = 1.5
    res = solve_greedy(target, pins, GreedyConfig(objective="blur", blur_sigma=sigma, min_gap=6,
                                                  opacity=0.25), progress=False)
    img = render_sequence(res.sequence, pins, target.shape, 0.25).image()
    blur_err = lambda im: float(np.sum(_blur0(im - target, sigma) ** 2))  # noqa: E731
    drop = blur_err(np.ones_like(target)) - blur_err(img)
    assert len(res.sequence) > 10 and all(g > 0 for g in res.scores)
    assert sum(res.scores) == pytest.approx(drop, rel=0.02)  # thin-line approximation


def test_weights_steer_lines(pins):
    # Two dark bands mirrored about the horizontal axis; start on the axis (pin 48, left middle)
    # so both are equally reachable. Only the top band is important -> first line goes up.
    t = np.ones((120, 120))
    t[28:34, :] = 0.0
    t[86:92, :] = 0.0
    w = np.full_like(t, 0.05)
    w[:60] = 1.0
    cfg = GreedyConfig(max_lines=1, min_gap=6, start_pin=48)
    up = solve_greedy(t, pins, cfg, weights=w, progress=False).sequence
    down = solve_greedy(t, pins, cfg, weights=w[::-1].copy(), progress=False).sequence
    assert pins[up[1]][1] < 59.5 < pins[down[1]][1]


@pytest.mark.parametrize("cfg", [GreedyConfig(opacity=0), GreedyConfig(min_gap=40),
                                 GreedyConfig(objective="nope"),
                                 GreedyConfig(objective="blur", blur_sigma=0)])
def test_invalid_config(pins, target, cfg):
    with pytest.raises(ValueError):
        solve_greedy(target, pins, cfg, progress=False)


def test_opacity_from_physical():
    assert opacity_from_physical(0.25, 500, 600) == pytest.approx(0.3)
    assert opacity_from_physical(2.0, 100, 600) == 1.0
    assert math.isclose(opacity_from_physical(0.5, 600, 600), 0.5)
