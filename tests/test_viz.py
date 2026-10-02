import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent

from stringart import viz
from stringart.cli import main
from stringart.render import render_sequence
from stringart.solver.baseline import BaselineConfig, solve_baseline


def _seq(pins, target, n=150):
    return solve_baseline(target, pins, BaselineConfig(n_lines=n, min_gap=6), progress=False)


def test_last_frame_is_exactly_final_render(pins, target):
    seq = _seq(pins, target).sequence
    final = render_sequence(seq, pins, target.shape, 0.2).image()
    frames = list(viz.iter_frames(seq, pins, target.shape, 0.2, lines_per_frame=7))
    k, canvas = frames[-1]
    assert k == len(seq) - 1
    assert np.array_equal(canvas.image(), final)


def test_export_mp4_and_gif(tmp_path, pins, target):
    seq = _seq(pins, target).sequence
    mp4 = viz.export(seq, pins, target.shape, 0.2, tmp_path / "b.mp4", fps=10, duration_s=1)
    gif = viz.export(seq, pins, target.shape, 0.2, tmp_path / "b.gif", fps=10, duration_s=1)
    cap = cv2.VideoCapture(str(mp4))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n_frames >= 10
    assert gif.stat().st_size > 0


def test_snapshot_grid(tmp_path, pins, target):
    seq = _seq(pins, target).sequence
    out = viz.snapshot_grid(seq, pins, target.shape, 0.2, [10, 50, 10_000], tmp_path / "g.png",
                            target)
    assert out.is_file()


def test_live_player_headless(tmp_path, monkeypatch, pins, target):
    """Drive the interactive player without a window: frames, pause, step, speed, jump to end."""
    seq = _seq(pins, target).sequence
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
    viz.play(seq, pins, target.shape, 0.2, target, lines_per_frame=10)
    assert f"line   {len(seq) - 1}/{len(seq) - 1}" in seen["status"]
    assert "[done]" in seen["status"] and "SSIM" in seen["status"]


def test_cli_run_then_viz(tmp_path):
    out = tmp_path / "run"
    main(["run", "sample:camera", "--size", "128", "--pins", "64", "--lines", "100",
          "--min-gap", "5", "--out", str(out), "--quiet"])
    for f in ("target.png", "render.png", "render.svg", "sequence.json", "metrics.json"):
        assert (out / f).is_file()
    main(["viz", str(out), "--grid", "10,100", "--save", str(out / "b.mp4")])
    assert (out / "grid.png").is_file() and (out / "b.mp4").is_file()
