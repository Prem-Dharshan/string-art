import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent

from stringart import viz
from stringart.cli import main
from stringart.color import ColorConfig, palette_rgb, render_steps, solve_color
from stringart.render import render_sequence
from stringart.solver.baseline import BaselineConfig, solve_baseline
from stringart.viz import Source


def _src(pins, target, n=150):
    seq = solve_baseline(target, pins, BaselineConfig(n_lines=n, min_gap=6),
                         progress=False).sequence
    return seq, Source.gray(seq, pins, target.shape, 0.2)


def _color_src(pins):
    t = np.ones((120, 120, 3))
    t[30:90, 30:90] = (0.8, 0.2, 0.1)
    names = ["black", "red"]
    res = solve_color(t, pins, palette_rgb(names), ColorConfig(opacity=0.3, min_gap=6,
                                                               min_run=10), names=names)
    return t, res, Source.color(res.steps, pins, (120, 120), res.colors, 0.3, names)


def test_last_frame_is_exactly_final_render(pins, target):
    seq, src = _src(pins, target)
    final = render_sequence(seq, pins, target.shape, 0.2).image()
    frames = list(src.frames(7))
    k, img = frames[-1]
    assert k == len(seq) - 1
    assert np.array_equal(img, final)
    assert np.array_equal(src.final(), final)


def test_color_last_frame_is_exactly_final_render(pins):
    _, res, src = _color_src(pins)
    final = render_steps(res.steps, pins, (120, 120), res.colors, 0.3).image()
    assert np.array_equal(list(src.frames(9))[-1][1], final)
    assert "red" in src.label(1) or "black" in src.label(1)


def test_export_mp4_and_gif(tmp_path, pins, target):
    _, src = _src(pins, target)
    mp4 = viz.export(src, tmp_path / "b.mp4", fps=10, duration_s=1)
    gif = viz.export(src, tmp_path / "b.gif", fps=10, duration_s=1)
    cap = cv2.VideoCapture(str(mp4))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n_frames >= 10
    assert gif.stat().st_size > 0
    _, _, csrc = _color_src(pins)
    assert viz.export(csrc, tmp_path / "c.gif", fps=10, duration_s=1).stat().st_size > 0


def test_snapshot_grid(tmp_path, pins, target):
    _, src = _src(pins, target)
    out = viz.snapshot_grid(src, [10, 50, 10_000], tmp_path / "g.png", target)
    assert out.is_file()
    t, _, csrc = _color_src(pins)
    assert viz.snapshot_grid(csrc, [5, 50], tmp_path / "c.png", t).is_file()


def test_live_player_headless(tmp_path, monkeypatch, pins, target):
    """Drive the interactive player without a window: frames, pause, step, speed, jump to end."""
    seq, src = _src(pins, target)
    seen = {}

    def fake_show():
        fig = plt.gcf()
        anim = fig._stringart_anim
        key = lambda k: fig.canvas.callbacks.process(  # noqa: E731
            "key_press_event", KeyEvent("key_press_event", fig.canvas, k))
        for i in range(3):
            anim._func(i)
        key(" ")
        key("right")
        key("+")
        key("e")
        anim._func(99)
        seen["status"] = fig.texts[0].get_text()
        fig.savefig(tmp_path / "player.png")
        plt.close(fig)

    monkeypatch.setattr(plt, "show", fake_show)
    viz.play(src, target, lines_per_frame=10)
    assert f"line   {len(seq) - 1}/{len(seq) - 1}" in seen["status"]
    assert "[done]" in seen["status"] and "SSIM" in seen["status"]


def test_cli_run_then_viz(tmp_path):
    out = tmp_path / "run"
    main(["run", "sample:camera", "--size", "128", "--pins", "64", "--lines", "100",
          "--min-gap", "5", "--out", str(out), "--quiet"])
    for f in ("target.png", "render.png", "render.svg", "sequence.json", "metrics.json",
              "instructions.txt"):
        assert (out / f).is_file()
    main(["viz", str(out), "--grid", "10,100", "--save", str(out / "b.mp4")])
    assert (out / "grid.png").is_file() and (out / "b.mp4").is_file()


def test_cli_color_run_then_viz(tmp_path):
    out = tmp_path / "crun"
    main(["run", "sample:coffee", "--size", "128", "--pins", "64", "--min-gap", "5",
          "--colors", "3", "--out", str(out), "--quiet"])
    doc = json.loads((out / "sequence.json").read_text())
    assert doc["mode"] == "color" and len(doc["palette"]) == 3 and doc["steps"]
    assert "thread" in (out / "instructions.txt").read_text().lower()
    main(["viz", str(out), "--grid", "10,100", "--save", str(out / "c.mp4")])
    assert (out / "grid.png").is_file() and (out / "c.mp4").is_file()
