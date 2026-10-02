import numpy as np
import pytest

from stringart.color import (
    THREADS,
    ColorCanvas,
    ColorConfig,
    auto_palette,
    color_metrics,
    dither,
    palette_rgb,
    render_steps,
    solve_color,
    solve_color_baseline,
)
from stringart.render import Canvas
from stringart.solver.greedy import GreedyConfig, solve_greedy


def _rgb_err(target, img):
    return float(np.sum((target - img) ** 2))


def test_black_thread_matches_grayscale_model(pins):
    c = ColorCanvas((120, 120), palette_rgb(["black"]), 0.3)
    g = Canvas((120, 120), 0.3)
    for a, b in [(0, 30), (30, 5), (5, 40), (40, 10)]:
        c.add_line(0, pins[a], pins[b])
        g.add_line(pins[a], pins[b])
    assert np.allclose(c.image(), g.image()[..., None])


def test_black_only_palette_reproduces_grayscale_solver(pins, target):
    rgb = np.repeat(target[..., None], 3, axis=2)
    cres = solve_color(rgb, pins, palette_rgb(["black"]), ColorConfig(opacity=0.25, min_gap=6))
    gres = solve_greedy(target, pins, GreedyConfig(opacity=0.25, min_gap=6), progress=False)
    cseq = [cres.steps[0][1]] + [b for _, _, b in cres.steps]
    assert cseq == gres.sequence


def test_color_solver_gains_match_render(pins):
    rng = np.random.default_rng(0)
    target = np.ones((120, 120, 3))
    target[30:60, :, :] = (0.8, 0.1, 0.1)  # red band
    target[70:90, :, :] = (0.1, 0.2, 0.7)  # blue band
    target += rng.normal(0, 0.01, target.shape)
    names = ["black", "red", "blue"]
    res = solve_color(
        target,
        pins,
        palette_rgb(names),
        ColorConfig(opacity=0.3, min_gap=6, min_run=5),
        names=names,
    )
    img = render_steps(res.steps, pins, (120, 120), res.colors, 0.3).image()
    drop = _rgb_err(target, np.ones_like(target)) - _rgb_err(target, img)
    assert sum(res.gains) == pytest.approx(drop, rel=1e-6)
    used = {names[k] for k, _, _ in res.steps}
    assert {"red", "blue"} <= used
    for name, seq in res.sequences().items():  # every colour is one continuous thread
        assert len(seq) >= 2 and name in names


def test_min_run_limits_colour_switches(pins):
    target = np.ones((120, 120, 3))
    target[20:100, 20:60] = (0.8, 0.1, 0.1)
    target[20:100, 60:100] = (0.1, 0.2, 0.7)
    names = ["red", "blue"]
    res = solve_color(
        target,
        pins,
        palette_rgb(names),
        ColorConfig(opacity=0.3, min_gap=6, min_run=20),
        names=names,
    )
    ks = [k for k, _, _ in res.steps]
    runs = [1]
    for a, b in zip(ks, ks[1:], strict=False):
        if a == b:
            runs[-1] += 1
        else:
            runs.append(1)
    # Runs are at least min_run long, except near the end when a colour has no improving
    # line left (the solver may then switch early rather than stop).
    assert len(runs) > 1 and all(r >= 20 for r in runs[: -len(names)])


def test_auto_palette_finds_dominant_colours():
    img = np.ones((80, 80, 3))
    img[:40] = np.array(THREADS["red"]) / 255
    img[40:, :40] = np.array(THREADS["blue"]) / 255
    names = auto_palette(img, np.ones((80, 80), bool), 3)
    assert names[0] == "black" and {"red", "blue"} <= set(names) and "white" not in names


def test_dither_uses_palette_proportions():
    img = np.full((40, 40, 3), 0.5)
    idx = dither(img, np.array([[0, 0, 0], [1, 1, 1.0]]))
    assert 0.4 < (idx == 0).mean() < 0.6


def test_baseline_runs_and_orders_light_to_dark(pins):
    target = np.ones((120, 120, 3))
    target[30:90, 30:90] = (0.85, 0.75, 0.2)
    target[50:70, 50:70] = 0.05
    names = ["black", "yellow"]
    res = solve_color_baseline(
        target, pins, palette_rgb(names), ColorConfig(opacity=0.3, min_gap=6), names=names
    )
    ks = [k for k, _, _ in res.steps]
    assert ks and ks[0] == 1 and ks[-1] == 0  # yellow first, black on top


def test_color_metrics_identical():
    a = np.random.default_rng(2).random((40, 40, 3))
    m = color_metrics(a, a, np.ones((40, 40), bool))
    assert m["de2000_s2"] == pytest.approx(0, abs=1e-6) and m["ssim_lum_s2"] == pytest.approx(1)


def test_unknown_thread_rejected():
    with pytest.raises(ValueError):
        palette_rgb(["black", "unobtainium"])
