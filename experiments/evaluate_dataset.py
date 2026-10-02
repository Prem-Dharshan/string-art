"""M6 evaluation on the dataset in data/dataset.json (fetch it first with fetch_dataset.py).

Sections (select with --sections):
  main       4 methods x all images, all with the SAME face-centred framing, so every method
             is scored against the same picture. Framing is reported separately as the face's
             share of the canvas (centre crop vs face crop).
               A baseline_published  M1 filters + baseline solver, 3000 lines, line_strength 0.1
               B baseline_equal      M1 filters + baseline solver at C's line count
               C greedy_legacy       M1 filters + greedy (solver improvement only)
               D full                M3 filters + auto importance + greedy (the method)
  robust     h02-h04 (synthetic exposure faults of f07) vs. the clean photo, auto stretch on/off
  curves     SSIM vs. line count, baseline vs. greedy (greedy run past its auto-stop)
  pins       full method with 128/200/256/320 pins on a face subset
  opacity    full method with thread opacity 0.1/0.15/0.2/0.3 on a face subset

Metrics are against the plain grayscale photo of the crop (fidelity), inside the frame.
Face-ROI metrics use the landmark face oval. Outputs go to outputs/experiments/m6/
(CSV per section, renders, summary.md).

    uv run python experiments/evaluate_dataset.py [--sections main,robust,...] [--limit N]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from stringart.face import detect_faces
from stringart.geometry import make_pins
from stringart.importance import auto_weights, face_map
from stringart.io import save_gray
from stringart.metrics import evaluate
from stringart.preprocess import PreprocessConfig, load_image, prepare
from stringart.render import Canvas, render_sequence, replay
from stringart.solver.baseline import BaselineConfig, solve_baseline
from stringart.solver.greedy import GreedyConfig, solve_greedy

ROOT = Path(__file__).resolve().parents[1]
SIZE, PINS, OPACITY = 600, 256, 0.2
FACE_SUBSET = [
    "f01_man_glasses_studio",
    "f07_woman_smiling_closeup",
    "f10_older_woman_glasses",
    "f13_girl_bw_smiling",
    "f16_elderly_man_pipe_bw",
]
CURVE_SET = [
    "f07_woman_smiling_closeup",
    "f16_elderly_man_pipe_bw",
    "a06_cat_tabby",
    "o01_lighthouse_striped",
]
METHODS = ("A_baseline_published", "B_baseline_equal", "C_greedy_legacy", "D_full")


def load_items(limit=None):
    spec = json.loads((ROOT / "data" / "dataset.json").read_text(encoding="utf-8"))
    items = spec["images"][:limit] if limit else spec["images"]
    for it in items:
        if not (ROOT / "data" / "raw" / f"{it['id']}.jpg").is_file():
            raise SystemExit("dataset missing: run `uv run python experiments/fetch_dataset.py`")
    return items


def image(item_id: str) -> np.ndarray:
    return load_image(str(ROOT / "data" / "raw" / f"{item_id}.jpg"))


def face_roi(prep) -> np.ndarray:
    faces = prep.faces or detect_faces(prep.canvas_bgr, landmarks=True)
    return face_map(faces, prep.target.shape[0])[1]


def score(prep, render, roi, ref=None) -> dict:
    """Fidelity to `ref` (default: the plain photo of the crop)."""
    ref = prep.plain if ref is None else ref
    m = evaluate(ref, render, prep.mask, sigmas=(0, 2, 4), roi=roi)
    keep = ("ssim_s0", "ssim_s2", "ssim_s4", "psnr_s2", "ssim_roi_s2", "psnr_roi_s2")
    return {k: m.get(k, "") for k in keep}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def section_main(items, out: Path, pins) -> list[dict]:
    rdir = out / "renders"
    rdir.mkdir(exist_ok=True)
    rows = []
    for it in items:
        img = image(it["id"])
        # M1 filter chain (no stretch, Gaussian) but the same face-centred framing as D.
        legacy = prepare(img, PreprocessConfig(size=SIZE, stretch="off", smooth="gaussian"))
        full = prepare(img, PreprocessConfig(size=SIZE))
        roi = face_roi(full)  # identical framing, so one ROI serves all methods
        centre = prepare(img, PreprocessConfig.legacy(size=SIZE))
        share_centre = float(face_roi(centre)[centre.mask].mean())
        share_face = float(roi[full.mask].mean())
        w_full, _ = auto_weights(full)

        runs = {}
        runs["A_baseline_published"] = (
            legacy,
            solve_baseline(
                legacy.target, pins, BaselineConfig(n_lines=3000, line_strength=0.1), progress=False
            ),
        )
        runs["C_greedy_legacy"] = (
            legacy,
            solve_greedy(legacy.target, pins, GreedyConfig(opacity=OPACITY), progress=False),
        )
        n_c = len(runs["C_greedy_legacy"][1].sequence) - 1
        runs["B_baseline_equal"] = (
            legacy,
            solve_baseline(
                legacy.target, pins, BaselineConfig(n_lines=n_c, line_strength=0.1), progress=False
            ),
        )
        runs["D_full"] = (
            full,
            solve_greedy(
                full.target, pins, GreedyConfig(opacity=OPACITY), weights=w_full, progress=False
            ),
        )

        # Synthetic exposure faults are scored against the clean photo (the true scene);
        # scoring against the degraded input would penalize correcting the exposure.
        clean = image(it["derived_from"]) if it.get("derived_from") else None
        for method in METHODS:
            prep, res = runs[method]
            ref = None if clean is None else _clean_reference(clean, prep.crop, SIZE, prep.mask)
            render = render_sequence(res.sequence, pins, prep.target.shape, OPACITY).image()
            if method in ("A_baseline_published", "D_full"):
                save_gray(rdir / f"{it['id']}__{method}.png", render)
            rows.append(
                {
                    "image": it["id"],
                    "category": it["category"],
                    "method": method,
                    "faces": len(full.faces),
                    "face_share_centre": round(share_centre, 4),
                    "face_share_facecrop": round(share_face, 4),
                    "lines": len(res.sequence) - 1,
                    "time_s": round(res.elapsed_s, 2),
                    **score(prep, render, roi, ref),
                }
            )
        save_gray(rdir / f"{it['id']}__plain.png", full.plain)
        save_gray(rdir / f"{it['id']}__plain_centre.png", centre.plain)
        print(
            f"main  {it['id']}: "
            + "  ".join(f"{r['method'][:1]}={r['ssim_s2']}" for r in rows[-4:]),
            flush=True,
        )
    return rows


def _clean_reference(clean_bgr, crop, size, mask):
    x0, y0, side = crop
    c = cv2.resize(
        clean_bgr[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_AREA
    )
    ref = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    ref[~mask] = 1.0
    return ref


def _robust_row(name, transform, stretch, res, ref, render, prep):
    m = evaluate(ref, render, prep.mask, sigmas=(2, 4), roi=face_roi(prep))
    return {
        "image": name,
        "transform": transform,
        "stretch": stretch,
        "lines": len(res.sequence) - 1,
        "ssim_s2_vs_clean": m["ssim_s2"],
        "ssim_s4_vs_clean": m["ssim_s4"],
        "psnr_s2_vs_clean": m["psnr_s2"],
        "face_ssim_s2_vs_clean": m.get("ssim_roi_s2", ""),
    }


def section_robust(items, out: Path, pins) -> list[dict]:
    rows = []
    for it in [i for i in items if i.get("derived_from")]:
        clean, degraded = image(it["derived_from"]), image(it["id"])
        for stretch in ("auto", "off"):
            prep = prepare(degraded, PreprocessConfig(size=SIZE, stretch=stretch))
            w, _ = auto_weights(prep)
            res = solve_greedy(
                prep.target, pins, GreedyConfig(opacity=OPACITY), weights=w, progress=False
            )
            render = render_sequence(res.sequence, pins, prep.target.shape, OPACITY).image()
            ref = _clean_reference(clean, prep.crop, SIZE, prep.mask)
            rows.append(_robust_row(it["id"], it["transform"], stretch, res, ref, render, prep))
            save_gray(out / "renders" / f"{it['id']}__stretch_{stretch}.png", render)
            print(
                f"robust {it['id']} stretch={stretch}: ssim2={rows[-1]['ssim_s2_vs_clean']}",
                flush=True,
            )
    if rows:
        prep = prepare(image("f07_woman_smiling_closeup"), PreprocessConfig(size=SIZE))
        w, _ = auto_weights(prep)
        res = solve_greedy(
            prep.target, pins, GreedyConfig(opacity=OPACITY), weights=w, progress=False
        )
        render = render_sequence(res.sequence, pins, prep.target.shape, OPACITY).image()
        rows.append(
            _robust_row(
                "f07_woman_smiling_closeup", "none (clean)", "auto", res, prep.plain, render, prep
            )
        )
    return rows


def _curve(prep, seq, pins, every=250):
    pts = []
    canvas = Canvas(prep.target.shape, OPACITY)
    for k in replay(seq, pins, canvas):
        if k % every == 0:
            m = evaluate(prep.plain, canvas.image(), prep.mask, sigmas=(2,))
            pts.append((k, m["ssim_s2"], m["psnr_s2"]))
    return pts


def section_curves(items, out: Path, pins, max_lines=6000) -> list[dict]:
    rows = []
    for it in [i for i in items if i["id"] in CURVE_SET]:
        prep = prepare(image(it["id"]), PreprocessConfig(size=SIZE))
        w, _ = auto_weights(prep)
        auto = solve_greedy(
            prep.target, pins, GreedyConfig(opacity=OPACITY), weights=w, progress=False
        )
        greedy = solve_greedy(
            prep.target,
            pins,
            GreedyConfig(opacity=OPACITY, max_lines=max_lines, stop_tol=-np.inf),
            weights=w,
            progress=False,
        )
        base = solve_baseline(
            prep.target,
            pins,
            BaselineConfig(n_lines=max_lines, line_strength=0.1),
            weights=w,
            progress=False,
        )
        for name, res in (("greedy", greedy), ("baseline", base)):
            for k, s2, p2 in _curve(prep, res.sequence, pins):
                rows.append(
                    {
                        "image": it["id"],
                        "solver": name,
                        "lines": k,
                        "ssim_s2": s2,
                        "psnr_s2": p2,
                        "auto_stop": len(auto.sequence) - 1,
                    }
                )
        print(f"curves {it['id']}: auto-stop at {len(auto.sequence) - 1}", flush=True)
    return rows


def section_sweep(items, out: Path, param: str, values) -> list[dict]:
    rows = []
    for it in [i for i in items if i["id"] in FACE_SUBSET]:
        prep = prepare(image(it["id"]), PreprocessConfig(size=SIZE))
        w, _ = auto_weights(prep)
        roi = face_roi(prep)
        for v in values:
            n_pins = v if param == "pins" else PINS
            op = v if param == "opacity" else OPACITY
            pins = make_pins("circle", n_pins, SIZE)
            gap = max(2, round(10 * n_pins / 256))  # same angular gap as the default
            res = solve_greedy(
                prep.target, pins, GreedyConfig(opacity=op, min_gap=gap), weights=w, progress=False
            )
            render = render_sequence(res.sequence, pins, prep.target.shape, op).image()
            rows.append(
                {
                    "image": it["id"],
                    param: v,
                    "lines": len(res.sequence) - 1,
                    "time_s": round(res.elapsed_s, 2),
                    **score(prep, render, roi),
                }
            )
        print(f"{param} {it['id']}: done", flush=True)
    return rows


def _mean_std(vals):
    v = np.asarray([float(x) for x in vals if x != ""], dtype=float)
    return (np.nan, np.nan, 0) if v.size == 0 else (v.mean(), v.std(), v.size)


def summarize(out: Path, main_rows, robust_rows, curve_rows, pins_rows, op_rows) -> str:
    md = ["# M6 evaluation summary", ""]
    if main_rows:
        cats = ["face", "hard", "animal", "object", "all"]
        md += ["## Main comparison: mean ± std (n images)", ""]
        for metric in ("ssim_s2", "ssim_s4", "psnr_s2", "ssim_roi_s2", "lines", "time_s"):
            md += [
                f"**{metric}**",
                "",
                "| method | " + " | ".join(cats) + " |",
                "|---|" + "---|" * len(cats),
            ]
            for meth in METHODS:
                cells = []
                for c in cats:
                    vals = [
                        r[metric]
                        for r in main_rows
                        if r["method"] == meth and (c == "all" or r["category"] == c)
                    ]
                    m, s, n = _mean_std(vals)
                    if n == 0:
                        cells.append("–")
                    elif metric.startswith("ssim"):
                        cells.append(f"{m:.3f} ± {s:.3f} ({n})")
                    else:
                        cells.append(f"{m:.2f} ± {s:.2f} ({n})")
                md.append(f"| {meth} | " + " | ".join(cells) + " |")
            md.append("")
        by = {(r["image"], r["method"]): r for r in main_rows}
        imgs = sorted({r["image"] for r in main_rows})
        md += [
            "## Paired differences: mean Δ, wins / n",
            "",
            "| comparison | Δ SSIM σ2 | wins | Δ face SSIM σ2 | wins |",
            "|---|---|---|---|---|",
        ]
        for a, b in (
            ("D_full", "A_baseline_published"),
            ("D_full", "C_greedy_legacy"),
            ("C_greedy_legacy", "B_baseline_equal"),
            ("C_greedy_legacy", "A_baseline_published"),
        ):
            d = [float(by[(i, a)]["ssim_s2"]) - float(by[(i, b)]["ssim_s2"]) for i in imgs]
            f = [
                float(by[(i, a)]["ssim_roi_s2"]) - float(by[(i, b)]["ssim_roi_s2"])
                for i in imgs
                if by[(i, a)]["ssim_roi_s2"] != ""
            ]
            fcell = f"{np.mean(f):+.4f} | {sum(x > 0 for x in f)} / {len(f)}" if f else "– | –"
            md.append(
                f"| {a} − {b} | {np.mean(d):+.4f} | {sum(x > 0 for x in d)} / {len(d)} | {fcell} |"
            )
        md.append("")
        shares = [
            (float(r["face_share_centre"]), float(r["face_share_facecrop"]))
            for r in main_rows
            if r["method"] == "D_full" and r["category"] in ("face", "hard")
        ]
        if shares:
            c, f = np.mean(shares, axis=0)
            md += [
                "## Framing",
                "",
                f"Face share of the canvas (face + hard images): centre crop "
                f"{c:.1%}, face crop {f:.1%} ({f / max(c, 1e-9):.1f}x).",
                "",
            ]
    for title, rows in (
        ("Robustness (vs. clean photo)", robust_rows),
        ("Pins sweep (face subset)", pins_rows),
        ("Opacity sweep (face subset)", op_rows),
    ):
        if rows:
            cols = list(rows[0])
            md += [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows] + [""]
    if curve_rows:
        md += [
            "## Line-count curves: peak SSIM σ2",
            "",
            "| image | solver | peak at | peak | auto-stop |",
            "|---|---|---|---|---|",
        ]
        for img_id in sorted({r["image"] for r in curve_rows}):
            for s in ("greedy", "baseline"):
                rr = [r for r in curve_rows if r["image"] == img_id and r["solver"] == s]
                best = max(rr, key=lambda r: float(r["ssim_s2"]))
                md.append(
                    f"| {img_id} | {s} | {best['lines']} | {float(best['ssim_s2']):.3f} | "
                    f"{rr[0]['auto_stop']} |"
                )
        md.append("")
    text = "\n".join(md)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            for cast in (int, float):
                try:
                    r[k] = cast(v)
                    break
                except ValueError:
                    pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="main,robust,curves,pins,opacity")
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    ap.add_argument(
        "--ids",
        default=None,
        help="comma-separated image ids to (re)run; their rows replace earlier ones",
    )
    ap.add_argument("--out", default=str(ROOT / "outputs" / "experiments" / "m6"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    items = load_items(args.limit)
    ids = set(args.ids.split(",")) if args.ids else None
    if ids:
        items = [it for it in items if it["id"] in ids]
    pins = make_pins("circle", PINS, SIZE)
    solve_greedy(np.ones((SIZE, SIZE)), pins, GreedyConfig(max_lines=1), progress=False)  # JIT
    sections = set(args.sections.split(","))
    t0 = time.perf_counter()

    def run(name, fn, *a):
        path = out / f"{name}.csv"
        if name in sections:
            rows = fn(*a)
            if ids and path.is_file():  # merge: keep earlier rows for the other images
                rows = [r for r in _read_csv(path) if r["image"] not in ids] + rows
            write_csv(path, rows)
            return rows
        return _read_csv(path) if path.is_file() else []  # reuse earlier results

    main_rows = run("main", section_main, items, out, pins)
    robust_rows = run("robust", section_robust, items, out, pins)
    curve_rows = run("curves", section_curves, items, out, pins)
    pins_rows = run("pins", section_sweep, items, out, "pins", (128, 200, 256, 320))
    op_rows = run("opacity", section_sweep, items, out, "opacity", (0.1, 0.15, 0.2, 0.3))
    print(summarize(out, main_rows, robust_rows, curve_rows, pins_rows, op_rows))
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
