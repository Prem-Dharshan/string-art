"""M5 experiment: colour string art on the colourful images of the dataset.

Methods (same auto palette of 4 threads unless noted, 256 pins, opacity 0.2, white board):
  lw_baseline     LessWrong-style: Floyd-Steinberg dither into the palette, independent greedy
                  per colour on its dither mask, layered light -> dark
  joint           joint colour greedy (this work), Lab k-means palette, min_run 100
  joint_rgbpal    joint colour greedy with an RGB k-means palette (palette-space ablation)
  joint_run1      joint colour greedy without the spool-switch limit (min_run 1)
  black_only      the grayscale method rendered in black (what colour adds)

Metrics vs the colour target: mean CIEDE2000 after viewing blur (lower is better) and SSIM on
luminance (higher is better).

    uv run python experiments/evaluate_color.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_dataset import OPACITY, PINS, ROOT, SIZE, _mean_std, image, write_csv  # noqa: E402

from stringart.color import (  # noqa: E402
    ColorConfig,
    auto_palette,
    color_metrics,
    color_target,
    palette_rgb,
    render_steps,
    solve_color,
    solve_color_baseline,
)
from stringart.geometry import make_pins  # noqa: E402
from stringart.importance import auto_weights  # noqa: E402
from stringart.io import save_rgb  # noqa: E402
from stringart.preprocess import PreprocessConfig, prepare  # noqa: E402

COLOURFUL = ["f02_elderly_bearded_outdoor", "f04_bearded_man_colourful_bg", "f05_athlete_dark_bg",
             "f09_young_woman_three_quarter", "f12_child_thanaka", "f14_boy_smiling_outdoor",
             "f15_child_monk_small_face", "h01_face_covered", "a01_dog_fluffy",
             "a03_border_collie", "a06_cat_tabby", "o01_lighthouse_striped"]
METHODS = ["lw_baseline", "joint", "joint_rgbpal", "joint_run1", "black_only"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colors", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "experiments" / "m5"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    pins = make_pins("circle", PINS, SIZE)
    rows = []
    for iid in COLOURFUL:
        prep = prepare(image(iid), PreprocessConfig(size=SIZE))
        target = color_target(prep)
        w, _ = auto_weights(prep)
        lab = auto_palette(target, prep.mask, args.colors, space="lab")
        rgb = auto_palette(target, prep.mask, args.colors, space="rgb")
        cfg = ColorConfig(n_colors=args.colors, opacity=OPACITY)
        runs = {
            "lw_baseline": (lab, solve_color_baseline(target, pins, palette_rgb(lab), cfg, lab)),
            "joint": (lab, solve_color(target, pins, palette_rgb(lab), cfg, w, lab)),
            "joint_rgbpal": (rgb, solve_color(target, pins, palette_rgb(rgb), cfg, w, rgb)),
            "joint_run1": (lab, solve_color(target, pins, palette_rgb(lab),
                                            ColorConfig(opacity=OPACITY, min_run=1), w, lab)),
            "black_only": (["black"], solve_color(target, pins, palette_rgb(["black"]), cfg, w,
                                                  ["black"])),
        }
        save_rgb(out / "renders" / f"{iid}__target.png", target)
        for m in METHODS:
            names, res = runs[m]
            render = render_steps(res.steps, pins, target.shape[:2], palette_rgb(names),
                                  OPACITY).image()
            save_rgb(out / "renders" / f"{iid}__{m}.png", render)
            switches = sum(1 for a, b in zip(res.steps, res.steps[1:], strict=False)
                           if a[0] != b[0])
            rows.append({"image": iid, "method": m, "palette": "+".join(names),
                         "lines": len(res.steps), "switches": switches,
                         "time_s": round(res.elapsed_s, 2),
                         **color_metrics(target, render, prep.mask)})
        print(f"{iid}: " + "  ".join(f"{r['method']}={r['de2000_s2']}" for r in rows[-5:]),
              flush=True)

    write_csv(out / "color.csv", rows)
    md = ["# M5 colour summary (mean ± std over images)", "",
          "| method | ΔE2000 σ2 ↓ | ΔE2000 σ4 ↓ | lum SSIM σ2 ↑ | lum SSIM σ4 ↑ | lines | "
          "switches | time s |", "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        rr = [r for r in rows if r["method"] == m]
        cells = []
        for k in ("de2000_s2", "de2000_s4", "ssim_lum_s2", "ssim_lum_s4", "lines", "switches",
                  "time_s"):
            mean, sd, _ = _mean_std([r[k] for r in rr])
            cells.append(f"{mean:.3f} ± {sd:.3f}" if "ssim" in k else f"{mean:.2f} ± {sd:.2f}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    by = {(r["image"], r["method"]): r for r in rows}
    md += ["", "| comparison | Δ ΔE2000 σ2 (negative = better) | wins | Δ lum SSIM σ2 | wins |",
           "|---|---|---|---|---|"]
    for a, b in (("joint", "lw_baseline"), ("joint", "joint_rgbpal"), ("joint", "joint_run1"),
                 ("joint", "black_only")):
        d = [by[(i, a)]["de2000_s2"] - by[(i, b)]["de2000_s2"] for i in COLOURFUL]
        s = [by[(i, a)]["ssim_lum_s2"] - by[(i, b)]["ssim_lum_s2"] for i in COLOURFUL]
        md.append(f"| {a} − {b} | {np.mean(d):+.3f} | {sum(x < 0 for x in d)} / {len(d)} | "
                  f"{np.mean(s):+.4f} | {sum(x > 0 for x in s)} / {len(s)} |")
    text = "\n".join(md) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
