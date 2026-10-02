"""M2 experiment: improved greedy vs. the LessWrong-style baseline.

For each image, run greedy (pixel and blur objectives; the line count is chosen automatically),
then the baseline at the *same* line count with line_strength tuned over a small grid (best
blurred SSIM reported, so the baseline gets a fair shot). Writes results.csv / results.md and
renders into outputs/experiments/compare_solvers/.

    uv run python experiments/compare_solvers.py [images...] [--size 600 --pins 256]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from stringart.geometry import make_pins
from stringart.io import save_gray
from stringart.metrics import evaluate
from stringart.preprocess import PreprocessConfig, load_image, preprocess
from stringart.render import render_sequence
from stringart.solver.baseline import BaselineConfig, solve_baseline
from stringart.solver.greedy import GreedyConfig, solve_greedy

KEYS = ("ssim_s0", "ssim_s1", "ssim_s2", "ssim_s4", "psnr_s0", "psnr_s2", "psnr_s4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*",
                    default=["sample:camera", "sample:astronaut", "sample:coffee",
                             "sample:chelsea"])
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--pins", type=int, default=256)
    ap.add_argument("--opacity", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/experiments/compare_solvers")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pins = make_pins("circle", args.pins, args.size)
    op = args.opacity
    solve_greedy(np.ones((args.size, args.size)), pins, GreedyConfig(max_lines=1),
                 progress=False)  # JIT warm-up, excluded from timings

    rows = []
    for src in args.images:
        name = Path(src.split(":", 1)[-1]).stem
        # The M2 comparison is about solvers, so keep the M1/M2 preprocessing fixed.
        target, mask = preprocess(load_image(src), PreprocessConfig.legacy(size=args.size))
        save_gray(out / f"{name}_target.png", target)

        def record(method, res, extra=""):
            img = render_sequence(res.sequence, pins, target.shape, op).image()
            save_gray(out / f"{name}_{method}.png", img)
            m = evaluate(target, img, mask)
            rows.append({"image": name, "method": method, "lines": len(res.sequence) - 1,
                         "time_s": round(res.elapsed_s, 2), "note": extra,
                         **{k: m[k] for k in KEYS}})
            return m

        g = solve_greedy(target, pins, GreedyConfig(opacity=op), progress=False)
        record("greedy_pixel", g, "auto line count")
        gb = solve_greedy(target, pins, GreedyConfig(opacity=op, objective="blur",
                                                     blur_sigma=1.0), progress=False)
        record("greedy_blur1", gb, "auto line count")

        n = len(g.sequence) - 1
        best = None
        for ls in (0.05, 0.1, 0.2):
            b = solve_baseline(target, pins, BaselineConfig(n_lines=n, line_strength=ls),
                               progress=False)
            img = render_sequence(b.sequence, pins, target.shape, op).image()
            s2 = evaluate(target, img, mask, sigmas=(2,))["ssim_s2"]
            if best is None or s2 > best[0]:
                best = (s2, ls, b)
        record("baseline_tuned", best[2], f"same lines as greedy_pixel, line_strength={best[1]}")
        b3k = solve_baseline(target, pins, BaselineConfig(n_lines=3000), progress=False)
        record("baseline_3000", b3k, "fixed 3000 lines, line_strength=0.1")
        print(f"{name}: done")

    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    cols = ["image", "method", "lines", "time_s", *KEYS]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
