"""Report figures: the pipeline on one photo, and a thread-by-thread build-up strip.

uv run python experiments/figures_report.py [--image f07_woman_smiling_closeup]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from stringart import viz
from stringart.color import (
    ColorConfig,
    color_target,
    fit_palette,
    palette_rgb,
    render_steps,
    solve_color,
)
from stringart.geometry import make_pins
from stringart.importance import auto_weights
from stringart.preprocess import PreprocessConfig, load_image, prepare
from stringart.render import render_sequence
from stringart.solver.greedy import GreedyConfig, solve_greedy
from stringart.solver.refine import RefineConfig, refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report" / "figures"


def _u8(img):
    img = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if img.ndim == 3 else cv2.COLOR_GRAY2BGR)


def _label(tile, text):
    bar = np.full((30, tile.shape[1], 3), 252, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="f07_woman_smiling_closeup")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bgr = load_image(str(ROOT / "data" / "raw" / f"{args.image}.jpg"))
    prep = prepare(bgr, PreprocessConfig())
    w, parts = auto_weights(prep, "on")
    pins = make_pins("circle", 300, 600)
    gap = round(10 * 300 / 256)
    op = 0.21  # 0.25 mm thread on a 700 mm frame at 600 px

    # 1. input crop with detection and landmarks
    inp = prep.canvas_bgr.copy()
    for f in prep.faces:
        x, y, fw, fh = f.box.astype(int)
        cv2.rectangle(inp, (x, y), (x + fw, y + fh), (60, 200, 60), 4)
        if f.landmarks is not None:
            for p in f.landmarks.astype(int):
                cv2.circle(inp, tuple(p), 5, (40, 40, 230), -1)
    g = solve_greedy(
        prep.target, pins, GreedyConfig(opacity=op, min_gap=gap), weights=w, progress=False
    )
    seq, _ = refine(
        prep.target,
        pins,
        g.sequence,
        op,
        min_gap=gap,
        weights=w,
        cfg=RefineConfig(sweeps=2),
        progress=False,
    )
    gray = render_sequence(seq, pins, prep.target.shape, op).image()
    ct = color_target(prep)
    names = fit_palette(ct, prep.mask, 4, weights=w)
    cres = solve_color(
        ct, pins, palette_rgb(names), ColorConfig(opacity=op, min_gap=gap), weights=w, names=names
    )
    color = render_steps(cres.steps, pins, (600, 600), palette_rgb(names), op).image()
    imp = cv2.applyColorMap(np.clip(w * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    tiles = [
        _label(inp, "1 face detection + landmarks"),
        _label(_u8(prep.target), "2 preprocessed target"),
        _label(imp, "3 importance map"),
        _label(_u8(gray), f"4 black thread, {len(seq) - 1} lines"),
        _label(_u8(color), f"5 colour: {', '.join(names)}"),
    ]
    fig = cv2.resize(np.hstack(tiles), None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "pipeline.png"), fig)
    print(f"wrote {OUT / 'pipeline.png'} ({len(seq) - 1} lines, palette {names})")

    src = viz.Source.gray(seq, pins, prep.target.shape, op)
    viz.snapshot_grid(
        src, [250, 750, 1500, 2500, len(seq) - 1], OUT / "buildup.png", prep.target, prep.mask
    )
    print(f"wrote {OUT / 'buildup.png'}")


if __name__ == "__main__":
    main()
