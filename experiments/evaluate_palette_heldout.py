"""Palette check on the held-out set: k-means palette vs gamut-fit palette, joint colour solver.

The gamut palette was chosen as the colour default from the 12 M5 images; this checks that
choice on the 12 natural images of the rule-chosen held-out set (never used for any decision).

    uv run python experiments/evaluate_palette_heldout.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_dataset import OPACITY, PINS, ROOT, SIZE, write_csv  # noqa: E402

from stringart.color import (  # noqa: E402
    ColorConfig,
    auto_palette,
    color_metrics,
    color_target,
    fit_palette,
    palette_rgb,
    render_steps,
    solve_color,
)
from stringart.geometry import make_pins  # noqa: E402
from stringart.importance import auto_weights  # noqa: E402
from stringart.preprocess import PreprocessConfig, load_image, prepare  # noqa: E402


def main() -> None:
    spec = json.loads((ROOT / "data" / "heldout.json").read_text(encoding="utf-8"))
    raw = ROOT / "data" / "raw_heldout"
    pins = make_pins("circle", PINS, SIZE)
    rows = []
    for it in spec["images"]:
        if "derived_from" in it:
            continue
        prep = prepare(load_image(str(raw / f"{it['id']}.jpg")), PreprocessConfig(size=SIZE))
        target = color_target(prep)
        w, _ = auto_weights(prep)
        for name, names in (
            ("kmeans", auto_palette(target, prep.mask, 4)),
            ("gamut", fit_palette(target, prep.mask, 4, weights=w)),
        ):
            colors = palette_rgb(names)
            res = solve_color(target, pins, colors, ColorConfig(opacity=OPACITY), w, names)
            img = render_steps(res.steps, pins, target.shape[:2], colors, OPACITY).image()
            m = color_metrics(target, img, prep.mask, sigmas=(2,))
            rows.append(
                {
                    "image": it["id"],
                    "palette": name,
                    "threads": "+".join(names),
                    "lines": len(res.steps),
                    **m,
                }
            )
        print(rows[-2:], flush=True)
    out = ROOT / "outputs" / "experiments" / "m5"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "palette_heldout.csv", rows)
    by = {(r["image"], r["palette"]): r for r in rows}
    ids = sorted({r["image"] for r in rows})
    de = {p: np.mean([by[(i, p)]["de2000_s2"] for i in ids]) for p in ("kmeans", "gamut")}
    ss = {p: np.mean([by[(i, p)]["ssim_lum_s2"] for i in ids]) for p in ("kmeans", "gamut")}
    wins_de = sum(by[(i, "gamut")]["de2000_s2"] < by[(i, "kmeans")]["de2000_s2"] for i in ids)
    wins_ss = sum(by[(i, "gamut")]["ssim_lum_s2"] > by[(i, "kmeans")]["ssim_lum_s2"] for i in ids)
    md = [
        f"# Palette on the held-out set ({len(ids)} natural images, 4 threads)",
        "",
        "| palette | ΔE2000 σ2 ↓ | luminance SSIM σ2 ↑ |",
        "|---|---|---|",
        f"| k-means, snapped | {de['kmeans']:.2f} | {ss['kmeans']:.3f} |",
        f"| gamut fit | {de['gamut']:.2f} | {ss['gamut']:.3f} |",
        "",
        f"Gamut fit has lower ΔE on {wins_de} / {len(ids)} images and higher luminance SSIM on "
        f"{wins_ss} / {len(ids)}.",
    ]
    (out / "palette_heldout_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
