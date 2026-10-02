"""Palette choice on the 12 colourful images (run evaluate_color.py first).

The final joint solve is deterministic given a palette, and evaluate_color.py already solved
every image with both candidate palettes (k-means -> `joint`, gamut fit -> `joint_fitpal`).
So the automatic selector (`color.choose_palette`, a quick 240 px preview per candidate,
scored by CIEDE2000) is run here and its choice is looked up in those results.

    uv run python experiments/evaluate_palette.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_color import COLOURFUL  # noqa: E402
from evaluate_dataset import PINS, ROOT, image  # noqa: E402

from stringart.color import choose_palette, color_target  # noqa: E402
from stringart.importance import auto_weights  # noqa: E402
from stringart.preprocess import PreprocessConfig, prepare  # noqa: E402


def main() -> None:
    src = ROOT / "outputs" / "experiments" / "m5" / "color.csv"
    with open(src, encoding="utf-8") as f:
        by = {(r["image"], r["method"]): r for r in csv.DictReader(f)}
    rows = []
    for iid in COLOURFUL:
        prep = prepare(image(iid), PreprocessConfig())
        target = color_target(prep)
        w, _ = auto_weights(prep)
        names, scores = choose_palette(target, prep.mask, 4, PINS, weights=w)
        km, fit = by[(iid, "joint")], by[(iid, "joint_fitpal")]
        pick = km if "+".join(names) == km["palette"] else fit
        rows.append(
            {
                "image": iid,
                "kmeans": km["palette"],
                "gamut": fit["palette"],
                "picked": "kmeans" if pick is km else "gamut",
                "preview_de_kmeans": round(scores["kmeans"], 2),
                "preview_de_gamut": round(scores["gamut"], 2),
                "de_kmeans": float(km["de2000_s2"]),
                "de_gamut": float(fit["de2000_s2"]),
                "de_auto": float(pick["de2000_s2"]),
                "ssim_kmeans": float(km["ssim_lum_s2"]),
                "ssim_gamut": float(fit["ssim_lum_s2"]),
                "ssim_auto": float(pick["ssim_lum_s2"]),
            }
        )
        print(rows[-1], flush=True)
    mean = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k.startswith(("de_", "ssim_"))}
    oracle = np.mean([min(r["de_kmeans"], r["de_gamut"]) for r in rows])
    right = sum(r["de_auto"] == min(r["de_kmeans"], r["de_gamut"]) for r in rows)
    md = [
        "# Palette choice (12 colourful images, 4 threads)",
        "",
        "| palette | ΔE2000 σ2 ↓ | luminance SSIM σ2 ↑ |",
        "|---|---|---|",
        f"| k-means, snapped | {mean['de_kmeans']:.2f} | {mean['ssim_kmeans']:.3f} |",
        f"| gamut fit | {mean['de_gamut']:.2f} | {mean['ssim_gamut']:.3f} |",
        f"| **auto (preview ΔE)** | **{mean['de_auto']:.2f}** | {mean['ssim_auto']:.3f} |",
        f"| oracle (better of the two per image) | {oracle:.2f} | |",
        "",
        f"The selector picked the better palette on {right} / {len(rows)} images.",
    ]
    out = ROOT / "outputs" / "experiments" / "m5"
    with open(out / "palette.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "palette_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
