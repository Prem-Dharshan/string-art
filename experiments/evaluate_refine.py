"""M4 experiment: what path refinement adds on top of the full method (D), on all 30 images.

For each image: D (greedy, auto-stop), then refinement sweeps 1..3 applied one after another.
Same scoring as evaluate_dataset.py: fidelity to the plain photo of the crop (clean photo for
the synthetic exposure faults), SSIM/PSNR at viewing blur, face-ROI SSIM.

    uv run python experiments/evaluate_refine.py [--sweeps 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_dataset import (  # noqa: E402
    OPACITY,
    PINS,
    ROOT,
    SIZE,
    _clean_reference,
    _mean_std,
    face_roi,
    image,
    load_items,
    score,
    write_csv,
)

from stringart.geometry import make_pins  # noqa: E402
from stringart.importance import auto_weights  # noqa: E402
from stringart.preprocess import PreprocessConfig, prepare  # noqa: E402
from stringart.render import render_sequence  # noqa: E402
from stringart.solver.greedy import GreedyConfig, solve_greedy  # noqa: E402
from stringart.solver.refine import RefineConfig, refine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "experiments" / "m4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pins = make_pins("circle", PINS, SIZE)
    refine(np.full((SIZE, SIZE), 0.5), pins, [0, 100, 200], OPACITY, progress=False)  # JIT
    solve_greedy(np.ones((SIZE, SIZE)), pins, GreedyConfig(max_lines=1), progress=False)

    rows = []
    for it in load_items(args.limit):
        prep = prepare(image(it["id"]), PreprocessConfig(size=SIZE))
        w, _ = auto_weights(prep)
        roi = face_roi(prep)
        ref = (None if not it.get("derived_from")
               else _clean_reference(image(it["derived_from"]), prep.crop, SIZE, prep.mask))
        g = solve_greedy(prep.target, pins, GreedyConfig(opacity=OPACITY), weights=w,
                         progress=False)
        seq, elapsed = g.sequence, g.elapsed_s

        def record(stage, seq, elapsed, e=None):
            render = render_sequence(seq, pins, prep.target.shape, OPACITY).image()
            rows.append({"image": it["id"], "category": it["category"], "stage": stage,
                         "lines": len(seq) - 1, "time_s": round(elapsed, 2),
                         "weighted_error": "" if e is None else round(e, 2),
                         **score(prep, render, roi, ref)})

        record("D_greedy", seq, elapsed)
        for s in range(1, args.sweeps + 1):
            seq, st = refine(prep.target, pins, seq, OPACITY, weights=w,
                             cfg=RefineConfig(sweeps=1), progress=False)
            elapsed += st["elapsed_s"]
            record(f"D_refine{s}", seq, elapsed, st["e_end"])
        print(f"{it['id']}: " + "  ".join(f"{r['stage'][2:]}={r['ssim_s2']}"
                                          for r in rows[-(args.sweeps + 1):]), flush=True)

    write_csv(out / "refine.csv", rows)
    stages = ["D_greedy"] + [f"D_refine{s}" for s in range(1, args.sweeps + 1)]
    md = ["# M4 refinement summary (mean ± std over images)", "",
          "| stage | SSIM σ2 | SSIM σ4 | PSNR σ2 | face SSIM σ2 | lines | time s |",
          "|---|---|---|---|---|---|---|"]
    for st in stages:
        rr = [r for r in rows if r["stage"] == st]
        cells = []
        for k in ("ssim_s2", "ssim_s4", "psnr_s2", "ssim_roi_s2", "lines", "time_s"):
            m, s, _ = _mean_std([r[k] for r in rr])
            cells.append(f"{m:.3f} ± {s:.3f}" if "ssim" in k else f"{m:.2f} ± {s:.2f}")
        md.append(f"| {st} | " + " | ".join(cells) + " |")
    by = {(r["image"], r["stage"]): r for r in rows}
    ids = sorted({r["image"] for r in rows})
    md += ["", "| vs D_greedy | Δ SSIM σ2 | wins | Δ face SSIM σ2 | wins |",
           "|---|---|---|---|---|"]
    for st in stages[1:]:
        d = [by[(i, st)]["ssim_s2"] - by[(i, "D_greedy")]["ssim_s2"] for i in ids]
        f = [by[(i, st)]["ssim_roi_s2"] - by[(i, "D_greedy")]["ssim_roi_s2"] for i in ids
             if by[(i, st)]["ssim_roi_s2"] != ""]
        md.append(f"| {st} | {np.mean(d):+.4f} | {sum(x > 0 for x in d)} / {len(d)} | "
                  f"{np.mean(f):+.4f} | {sum(x > 0 for x in f)} / {len(f)} |")
    for cat in ("face", "hard", "animal", "object"):
        d = [by[(i, stages[-1])]["ssim_s2"] - by[(i, "D_greedy")]["ssim_s2"] for i in ids
             if by[(i, "D_greedy")]["category"] == cat]
        md.append(f"| {stages[-1]} ({cat}) | {np.mean(d):+.4f} | {sum(x > 0 for x in d)} / "
                  f"{len(d)} | | |")
    text = "\n".join(md) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
