"""Report figures from the M6 evaluation CSVs (run evaluate_dataset.py first).

Colours follow the dataviz reference palette (validated: adjacent CVD dE >= 9.1). Each method
keeps its slot in every figure. Aqua/yellow are under 3:1 on the light surface, so bars carry
value labels.

    uv run python experiments/figures_m6.py [--src outputs/experiments/m6] [--docs]
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                         "#c3c2b7")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
METHODS = {  # entity -> (label, colour); fixed across figures
    "A_baseline_published": ("A  baseline, 3000 lines", SERIES[0]),
    "B_baseline_equal": ("B  baseline, equal lines", SERIES[1]),
    "C_greedy_legacy": ("C  greedy solver (M2)", SERIES[2]),
    "D_full": ("D  full method (M2 + M3)", SERIES[3]),
}
CATS = [("face", "Faces"), ("hard", "Hard cases"), ("animal", "Animals"), ("object", "Objects"),
        ("all", "All")]


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else np.nan


def style(ax, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)


def new_fig(w, h, ncols=1):
    fig = Figure(figsize=(w, h), facecolor=SURFACE)
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    return fig, axes


def save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def dotplot(ax, groups, series, title, xlabel):
    """Horizontal dot plot: one row block per group, one dot per series, value-labelled.
    `groups`: [(label, {series_key: value})]; `series`: {key: (label, colour)}.
    A dot plot needs no zero baseline, unlike bars, so small differences stay honest."""
    keys = list(series)
    step = 0.8 / len(keys)
    for g, (_, vals) in enumerate(groups):
        present = [vals[k] for k in keys if not np.isnan(vals.get(k, np.nan))]
        if present:
            ax.plot([min(present), max(present)], [g, g], color=GRID, linewidth=1, zorder=1)
        for i, k in enumerate(keys):
            v = vals.get(k, np.nan)
            if np.isnan(v):
                continue
            y = g - 0.4 + step * (i + 0.5)
            ax.plot([v], [y], "o", color=series[k][1], markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3, label=series[k][0] if g == 0 else None)
            ax.text(v, y, f"   {v:.3f}", fontsize=7.5, color=INK2, va="center")
    ax.set_yticks(range(len(groups)), [g for g, _ in groups])
    ax.invert_yaxis()
    style(ax, None, xlabel)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.18)  # room for the value labels


def fig_main(rows, out: Path):
    fig, axes = new_fig(13, 5.2, 2)
    series = {m: v for m, v in METHODS.items()}
    panels = [("ssim_s2", "Whole frame: SSIM at viewing blur σ = 2 px", CATS),
              ("ssim_roi_s2", "Face region only: SSIM at σ = 2 px",
               [("face", "Faces"), ("hard", "Hard cases"), ("facehard", "Faces + hard")])]
    for ax, (metric, title, cats) in zip(axes, panels, strict=True):
        groups = []
        for c, name in cats:
            vals = {}
            for meth in METHODS:
                v = [fnum(r[metric]) for r in rows if r["method"] == meth and (
                    c == "all" or r["category"] == c
                    or (c == "facehard" and r["category"] in ("face", "hard")))]
                v = [x for x in v if not np.isnan(x)]
                vals[meth] = np.mean(v) if v else np.nan
                n = len(v)
            groups.append((f"{name} (n={n})", vals))
        dotplot(ax, groups, series, title, "mean SSIM (higher is better)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncols=2, labelcolor=INK2,
                   bbox_to_anchor=(0, -0.1))
    save(fig, out / "m6_main_comparison.png")


def fig_curves(rows, out: Path):
    imgs = sorted({r["image"] for r in rows})
    fig, axes = new_fig(4 * len(imgs), 3.6, len(imgs))
    colors = {"greedy": SERIES[2], "baseline": SERIES[0]}
    labels = {"greedy": "greedy (M2)", "baseline": "baseline"}
    for ax, img in zip(axes, imgs, strict=True):
        rr = [r for r in rows if r["image"] == img]
        for s in ("baseline", "greedy"):
            pts = sorted((int(r["lines"]), fnum(r["ssim_s2"])) for r in rr if r["solver"] == s)
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, color=colors[s], linewidth=2, label=labels[s])
            ax.text(xs[-1], ys[-1], f" {labels[s]}", color=INK2, fontsize=8, va="center")
        stop = int(rr[0]["auto_stop"])
        ax.axvline(stop, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(stop, ax.get_ylim()[0], f" auto-stop\n {stop}", color=MUTED, fontsize=8,
                va="bottom")
        style(ax, "SSIM σ = 2" if ax is axes[0] else None, "lines")
        ax.set_title(img.split("_", 1)[1].replace("_", " "), color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, max(int(r["lines"]) for r in rr) * 1.25)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=9, labelcolor=INK2, ncols=2,
               loc="lower left", bbox_to_anchor=(0.06, 0.98))
    save(fig, out / "m6_line_curves.png")


def fig_sweep(rows, param, xlabel, out: Path):
    vals = sorted({fnum(r[param]) for r in rows})
    fig, axes = new_fig(10, 3.6, 2)
    for ax, (metric, title) in zip(axes, (("ssim_s2", "whole frame"),
                                          ("ssim_roi_s2", "face region")), strict=True):
        means = [np.mean([fnum(r[metric]) for r in rows if fnum(r[param]) == v]) for v in vals]
        ax.plot(vals, means, color=SERIES[3], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for v, m in zip(vals, means, strict=True):
            ax.text(v, m, f"  {m:.3f}", fontsize=8, color=INK2, va="center")
        style(ax, f"mean SSIM σ = 2, {title}", xlabel)
        ax.set_xticks(vals)
        ax.set_title(f"Full method: {title}", color=INK, fontsize=10, loc="left")
    lines = [np.mean([fnum(r["lines"]) for r in rows if fnum(r[param]) == v]) for v in vals]
    axes[0].text(0.0, -0.3, "mean lines: " + ", ".join(f"{v:g} → {n:.0f}"
                                                         for v, n in zip(vals, lines, strict=True)),
                 transform=axes[0].transAxes, fontsize=8, color=MUTED)
    save(fig, out / f"m6_sweep_{param}.png")


def fig_robust(rows, out: Path):
    fault = [r for r in rows if r["stretch"] in ("auto", "off") and r["transform"] != "none (clean)"]
    clean = [r for r in rows if r["transform"] == "none (clean)"]
    transforms = sorted({r["transform"] for r in fault})
    fig, axes = new_fig(7, 3.4)
    ax = axes[0]
    series = {"off": ("level stretch off", SERIES[0]),
              "auto": ("auto stretch (default)", SERIES[3])}
    groups = [(t.replace("_", " "),
               {st: fnum(next(r for r in fault if r["transform"] == t and r["stretch"] == st)
                         ["ssim_s2_vs_clean"]) for st in series}) for t in transforms]
    dotplot(ax, groups, series, "Exposure faults of f07, scored against the clean photo",
            "SSIM σ = 2 vs. the clean photo")
    if clean:
        ref = fnum(clean[0]["ssim_s2_vs_clean"])
        ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(ref, -0.55, f"clean input {ref:.3f}", color=MUTED, fontsize=8, ha="center")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left",
              bbox_to_anchor=(0, -0.2), ncols=2)
    save(fig, out / "m6_robustness.png")


def fig_gallery(rows, src: Path, out: Path, ids):
    rdir = src / "renders"
    tiles = []
    for i in ids:
        cols = [cv2.imread(str(rdir / f"{i}__{s}.png"), cv2.IMREAD_GRAYSCALE)
                for s in ("plain", "A_baseline_published", "D_full")]
        if any(c is None for c in cols):
            continue
        tiles.append(np.hstack(cols))
    if not tiles:
        return
    grid = cv2.resize(np.vstack(tiles), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h = 34
    header = np.full((h, grid.shape[1]), 252, np.uint8)
    w = grid.shape[1] // 3
    for k, t in enumerate(("photo (same crop)", "A: baseline, 3000 lines", "D: full method")):
        cv2.putText(header, t, (k * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 40, 1,
                    cv2.LINE_AA)
    path = out / "m6_gallery.png"
    cv2.imwrite(str(path), np.vstack([header, grid]))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "outputs" / "experiments" / "m6"))
    ap.add_argument("--docs", action="store_true", help="also copy into docs/milestones/figures")
    args = ap.parse_args()
    src = Path(args.src)
    out = src / "figures"
    out.mkdir(exist_ok=True)
    if (src / "main.csv").is_file():
        main_rows = read(src / "main.csv")
        fig_main(main_rows, out)
        fig_gallery(main_rows, src, out, ["f07_woman_smiling_closeup", "f16_elderly_man_pipe_bw",
                                          "f13_girl_bw_smiling", "f05_athlete_dark_bg",
                                          "a04_cat_black_white", "o01_lighthouse_striped"])
    if (src / "curves.csv").is_file():
        fig_curves(read(src / "curves.csv"), out)
    if (src / "pins.csv").is_file():
        fig_sweep(read(src / "pins.csv"), "pins", "pins", out)
    if (src / "opacity.csv").is_file():
        fig_sweep(read(src / "opacity.csv"), "opacity", "thread opacity per pixel", out)
    if (src / "robust.csv").is_file():
        fig_robust(read(src / "robust.csv"), out)
    if args.docs:
        dst = ROOT / "docs" / "milestones" / "figures"
        dst.mkdir(parents=True, exist_ok=True)
        for p in out.glob("*.png"):
            shutil.copy(p, dst / p.name)
        print(f"copied figures to {dst}")


if __name__ == "__main__":
    main()
