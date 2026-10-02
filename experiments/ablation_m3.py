"""M3 experiment: what face-aware preprocessing and importance maps buy.

Variants (all with the M2 greedy solver, pixel objective):
  legacy        centre crop, grayscale -> CLAHE -> Gaussian, uniform weights      (M2)
  prep          face crop + auto level stretch + bilateral + CLAHE, uniform weights (I4)
  prep+imp      ... + automatic importance map                                    (I4 + I5)
  prep+imp+bg   ... + GrabCut background fade (only applied when a face is found)

Metrics are against the *plain photo* of each variant's own crop (`Prepared.plain`), not
against the preprocessed target, because preprocessing changes the target itself. Face-ROI
metrics use the landmark face oval. Writes results.md / results.csv and a figure per image.

    uv run python experiments/ablation_m3.py [images...] [--opacity 0.2]
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from stringart.geometry import make_pins
from stringart.importance import importance
from stringart.metrics import evaluate
from stringart.preprocess import PreprocessConfig, load_image, prepare
from stringart.render import render_sequence
from stringart.solver.greedy import GreedyConfig, solve_greedy

VARIANTS = {
    "legacy": (PreprocessConfig.legacy(), False),
    "prep": (PreprocessConfig(), False),
    "prep+imp": (PreprocessConfig(), True),
    "prep+imp+bg": (PreprocessConfig(background="fade"), True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*",
                    default=["sample:astronaut", "sample:camera", "sample:coffee",
                             "sample:chelsea"])
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--pins", type=int, default=256)
    ap.add_argument("--opacity", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/experiments/ablation_m3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pins = make_pins("circle", args.pins, args.size)
    op = args.opacity

    rows = []
    for src in args.images:
        name = Path(src.split(":", 1)[-1]).stem
        img = load_image(src)
        tiles = []
        for vname, (pcfg, use_w) in VARIANTS.items():
            pcfg = PreprocessConfig(**{**pcfg.__dict__, "size": args.size})
            p = prepare(img, pcfg)
            w, parts = importance(p)
            g = solve_greedy(p.target, pins, GreedyConfig(opacity=op),
                             weights=w if use_w else None, progress=False)
            r = render_sequence(g.sequence, pins, p.target.shape, op).image()
            m = evaluate(p.plain, r, p.mask, sigmas=(2, 4), roi=parts["face_roi"])
            rows.append({"image": name, "variant": vname, "faces": len(p.faces),
                         "lines": len(g.sequence) - 1, "time_s": round(g.elapsed_s, 2),
                         "ssim_s2": m["ssim_s2"], "ssim_s4": m["ssim_s4"],
                         "psnr_s2": m["psnr_s2"],
                         "face_ssim_s2": m.get("ssim_roi_s2", ""),
                         "face_psnr_s2": m.get("psnr_roi_s2", "")})
            tiles.append(np.vstack([p.target, r]))
        fig = (np.hstack(tiles) * 255 + 0.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{name}.png"),
                    cv2.resize(fig, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        print(f"{name}: done")

    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
