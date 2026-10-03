"""Command line: `stringart run <image>` solves, `stringart viz <run dir>` replays,
`stringart demo` opens the web demo."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import viz
from .fabrication import color_instructions, instructions, thread_length_mm
from .geometry import make_pins
from .importance import ImportanceConfig, auto_weights
from .io import (
    load_gray,
    load_rgb,
    load_sequence,
    save_color_result,
    save_gray,
    save_rgb,
    save_sequence,
)
from .metrics import evaluate
from .models import fetch, model_path
from .preprocess import PreprocessConfig, frame_mask, load_image, prepare
from .render import render_sequence, to_svg
from .solver.baseline import BaselineConfig, solve_baseline
from .solver.greedy import GreedyConfig, opacity_from_physical, solve_greedy
from .solver.refine import RefineConfig, refine


def _run(args) -> None:
    if args.thread_mm is not None and args.frame_mm is None:
        raise ValueError("--thread-mm needs --frame-mm")
    if args.thread_mm is not None:
        args.opacity = round(opacity_from_physical(args.thread_mm, args.frame_mm, args.size), 4)
    if not args.legacy_prep and not model_path("yunet").is_file():
        print(
            "note: face model not found, so no face crop/landmarks. "
            "Run `stringart fetch-models` to enable them."
        )
    if args.legacy_prep:
        pcfg = PreprocessConfig.legacy(
            size=args.size, frame=args.frame, clahe_clip=args.clahe, blur_sigma=args.blur
        )
    else:
        pcfg = PreprocessConfig(
            size=args.size,
            frame=args.frame,
            clahe_clip=args.clahe,
            crop=args.crop,
            face_zoom=args.face_zoom,
            background=args.background,
        )
    prep = prepare(load_image(args.image), pcfg)
    target, mask = prep.target, prep.mask
    weights, parts = auto_weights(
        prep, args.importance, ImportanceConfig(floor=args.importance_floor)
    )
    pins = make_pins(args.frame, args.pins, args.size)
    if args.colors > 1 or args.palette:
        _run_color(args, prep, pcfg, weights, parts, pins)
        return
    if args.solver == "baseline":
        scfg = BaselineConfig(
            n_lines=args.lines or 3000,
            line_strength=args.line_strength,
            min_gap=args.min_gap,
            n_candidates=args.candidates,
            darkness_penalty=args.darkness_penalty,
            seed=args.seed,
        )
        res = solve_baseline(target, pins, scfg, weights=weights, progress=not args.quiet)
    else:
        scfg = GreedyConfig(
            max_lines=args.lines or 8000,
            opacity=args.opacity,
            min_gap=args.min_gap,
            max_repeats=args.max_repeats,
            objective=args.objective,
            blur_sigma=args.blur_sigma,
        )
        res = solve_greedy(target, pins, scfg, weights=weights, progress=not args.quiet)
    refine_stats = None
    if args.refine > 0:
        res.sequence, refine_stats = refine(
            target,
            pins,
            res.sequence,
            args.opacity,
            min_gap=args.min_gap,
            max_repeats=args.max_repeats,
            weights=weights,
            cfg=RefineConfig(sweeps=args.refine),
            progress=not args.quiet,
        )
        res.elapsed_s += refine_stats["elapsed_s"]
    render = render_sequence(res.sequence, pins, target.shape, args.opacity).image()
    roi = parts["face_roi"]
    metrics = {
        "vs_target": evaluate(target, render, mask, roi=roi),
        "vs_photo": evaluate(prep.plain, render, mask, roi=roi),
    }

    stem = Path(args.image.split(":", 1)[-1]).stem
    out = Path(args.out) if args.out else Path("outputs") / f"{stem}_{args.solver}"
    out.mkdir(parents=True, exist_ok=True)
    save_gray(out / "target.png", target)
    save_gray(out / "render.png", render)
    if weights is not None:
        save_gray(out / "importance.png", weights)
    (out / "render.svg").write_text(to_svg(res.sequence, pins, args.size, args.opacity))
    (out / "instructions.txt").write_text(
        instructions(res.sequence, pins, args.size, args.frame, args.frame_mm)
    )
    meta = {
        "image": args.image,
        "solver": args.solver,
        "elapsed_s": round(res.elapsed_s, 3),
        "faces": len(prep.faces),
        "crop_xyside": list(prep.crop),
        "thread_length_m": (
            round(thread_length_mm(res.sequence, pins, args.size, args.frame_mm) / 1000, 2)
            if args.frame_mm
            else None
        ),
        "preprocess": asdict(pcfg),
        "solver_config": asdict(scfg),
        "refine": refine_stats,
        "metrics": metrics,
    }
    save_sequence(
        out / "sequence.json",
        sequence=res.sequence,
        pins=pins,
        size=args.size,
        frame=args.frame,
        opacity=args.opacity,
        meta=meta,
    )
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(
        f"{len(res.sequence) - 1} lines in {res.elapsed_s:.2f}s, {len(prep.faces)} face(s) -> {out}"
    )
    for name, m in metrics.items():
        print(f"  {name}: " + "  ".join(f"{k}={v}" for k, v in m.items() if "s0" not in k))
    if args.viz:
        viz.play(viz.Source.gray(res.sequence, pins, target.shape, args.opacity), target, mask)


def _run_color(args, prep, pcfg, weights, parts, pins) -> None:
    from .color import (
        ColorConfig,
        auto_palette,
        choose_palette,
        color_metrics,
        color_target,
        fit_palette,
        palette_rgb,
        render_steps,
        solve_color,
        solve_color_baseline,
    )

    target = color_target(prep, clahe_clip=args.clahe)
    if args.palette:
        names = [n.strip() for n in args.palette.split(",")]
    elif args.palette_method == "kmeans":
        names = auto_palette(target, prep.mask, args.colors)
    elif args.palette_method == "gamut":
        names = fit_palette(target, prep.mask, args.colors, weights=weights)
    else:
        names, _ = choose_palette(
            target, prep.mask, args.colors, args.pins, weights=weights, opacity=args.opacity
        )
    colors = palette_rgb(names)
    ccfg = ColorConfig(
        n_colors=len(names),
        palette=names,
        opacity=args.opacity,
        max_lines=args.lines or 12000,
        min_gap=args.min_gap,
        max_repeats=args.max_repeats,
        min_run=args.min_run,
    )
    if args.solver == "baseline":
        res = solve_color_baseline(target, pins, colors, ccfg, names=names)
    else:
        res = solve_color(target, pins, colors, ccfg, weights=weights, names=names)
    render = render_steps(res.steps, pins, target.shape[:2], colors, args.opacity).image()
    metrics = color_metrics(target, render, prep.mask)

    stem = Path(args.image.split(":", 1)[-1]).stem
    out = Path(args.out) if args.out else Path("outputs") / f"{stem}_{args.solver}_color"
    out.mkdir(parents=True, exist_ok=True)
    save_rgb(out / "target.png", target)
    save_rgb(out / "render.png", render)
    (out / "instructions.txt").write_text(
        color_instructions(res.steps, names, pins, args.size, args.frame, args.frame_mm),
        encoding="utf-8",
    )
    switches = sum(1 for a, b in zip(res.steps, res.steps[1:], strict=False) if a[0] != b[0])
    meta = {
        "image": args.image,
        "solver": args.solver,
        "elapsed_s": round(res.elapsed_s, 3),
        "palette": names,
        "lines_per_colour": {n: sum(1 for k, _, _ in res.steps if names[k] == n) for n in names},
        "colour_switches": switches,
        "faces": len(prep.faces),
        "preprocess": asdict(pcfg),
        "colour_config": asdict(ccfg),
        "metrics": metrics,
    }
    save_color_result(
        out / "sequence.json",
        steps=res.steps,
        palette=names,
        colors=colors,
        pins=pins,
        size=args.size,
        frame=args.frame,
        opacity=args.opacity,
        meta=meta,
    )
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    per = ", ".join(f"{n} {c}" for n, c in meta["lines_per_colour"].items())
    print(
        f"{len(res.steps)} lines ({per}), {switches} spool switches, {res.elapsed_s:.2f}s -> {out}"
    )
    print("  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    if args.viz:
        viz.play(
            viz.Source.color(res.steps, pins, target.shape[:2], colors, args.opacity, names),
            target,
            prep.mask,
        )


def _viz(args) -> None:
    src = Path(args.source)
    seq_path = src / "sequence.json" if src.is_dir() else src
    doc = load_sequence(seq_path)
    color = doc.get("mode") == "color"
    target_path = Path(args.target) if args.target else seq_path.parent / "target.png"
    target = None
    if target_path.is_file():
        target = load_rgb(target_path) if color else load_gray(target_path)
    mask = frame_mask(doc["frame"], doc["size"]) if target is not None else None
    shape = (doc["size"], doc["size"])
    opacity = args.opacity or doc["opacity"]
    if color:
        source = viz.Source.color(
            doc["steps"], doc["pins"], shape, doc["colors"], opacity, doc["palette"]
        )
    else:
        source = viz.Source.gray(doc["sequence"], doc["pins"], shape, opacity)

    did_file_output = False
    if args.grid:
        counts = [int(c) for c in args.grid.split(",")]
        path = viz.snapshot_grid(source, counts, seq_path.parent / "grid.png", target, mask)
        print(f"wrote {path}")
        did_file_output = True
    for out in args.save or []:
        path = viz.export(
            source,
            Path(out),
            lines_per_frame=args.step,
            fps=args.fps,
            duration_s=args.duration,
            scale=args.scale,
        )
        print(f"wrote {path}")
        did_file_output = True
    if not did_file_output or args.show:
        viz.play(source, target, mask, lines_per_frame=args.step or 10)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stringart", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compute a pin sequence for an image")
    r.add_argument("image", help="image path, or sample:astronaut / sample:camera / ...")
    r.add_argument("--solver", choices=["greedy", "baseline"], default="greedy")
    r.add_argument("--out", help="output directory (default outputs/<image>_<solver>)")
    r.add_argument("--size", type=int, default=600, help="canvas size in px")
    r.add_argument("--frame", choices=["circle", "rect"], default="circle")
    r.add_argument("--pins", type=int, default=256)
    r.add_argument(
        "--lines",
        type=int,
        default=None,
        help="baseline: line count (default 3000); greedy: cap (default 8000)",
    )
    r.add_argument("--min-gap", type=int, default=10)
    r.add_argument(
        "--opacity",
        type=float,
        default=0.2,
        help="thread opacity per pixel (solver model and rendering)",
    )
    r.add_argument("--thread-mm", type=float, help="thread width; with --frame-mm sets opacity")
    r.add_argument(
        "--frame-mm",
        type=float,
        help="frame diameter/side in mm (thread length in instructions.txt)",
    )
    g = r.add_argument_group("greedy solver")
    g.add_argument("--objective", choices=["pixel", "blur"], default="pixel")
    g.add_argument("--blur-sigma", type=float, default=1.5, help="viewing blur for 'blur'")
    g.add_argument("--max-repeats", type=int, default=2, help="max uses of one chord")
    g.add_argument(
        "--refine",
        type=int,
        default=2,
        help="refinement sweeps after greedy (delete/reroute/insert pins; 0 = off)",
    )
    c = r.add_argument_group("colour")
    c.add_argument(
        "--colors",
        type=int,
        default=1,
        help="number of thread colours (>1 = colour mode, palette by Lab k-means)",
    )
    c.add_argument("--palette", help="explicit thread colours, e.g. black,red,tan,blue")
    c.add_argument(
        "--palette-method",
        choices=["gamut", "kmeans", "auto"],
        default="gamut",
        help="gamut: threads that can reach the photo's colours (best colour accuracy); "
        "kmeans: snapped colour clusters (slightly more light/dark structure); "
        "auto: quick preview of both (weak predictor)",
    )
    c.add_argument(
        "--min-run", type=int, default=100, help="colour mode: lines before switching spool"
    )
    b = r.add_argument_group("baseline solver")
    b.add_argument("--line-strength", type=float, default=0.1)
    b.add_argument("--candidates", type=int, default=None, help="random candidates per step")
    b.add_argument("--darkness-penalty", type=float, default=0.0)
    pp = r.add_argument_group("preprocessing / importance")
    pp.add_argument(
        "--crop",
        choices=["face", "center"],
        default="face",
        help="centre the frame on the largest face (falls back to centre)",
    )
    pp.add_argument("--face-zoom", type=float, default=1.8, help="crop side / face height")
    pp.add_argument(
        "--background",
        choices=["none", "fade"],
        default="none",
        help="fade: lighten the background with GrabCut (only when a face is found)",
    )
    pp.add_argument(
        "--importance",
        choices=["auto", "on", "off"],
        default="auto",
        help="error weights from face/edges/saliency (auto: only if a face is found)",
    )
    pp.add_argument(
        "--importance-floor", type=float, default=0.1, help="weight of unimportant regions (0..1)"
    )
    pp.add_argument("--clahe", type=float, default=2.0, help="CLAHE clip limit (0 = off)")
    pp.add_argument(
        "--legacy-prep",
        action="store_true",
        help="M1/M2 chain: centre crop, CLAHE, Gaussian (--blur)",
    )
    pp.add_argument("--blur", type=float, default=1.0, help="Gaussian sigma for --legacy-prep")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--viz", action="store_true", help="open the visualizer when done")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_run)

    v = sub.add_parser("viz", help="replay a sequence thread by thread")
    v.add_argument("source", help="run directory or sequence.json")
    v.add_argument("--target", help="target image (default: target.png next to the sequence)")
    v.add_argument("--save", action="append", help="export .mp4 or .gif (repeatable)")
    v.add_argument("--grid", help="snapshot grid at line counts, e.g. 100,500,1000,3000")
    v.add_argument("--show", action="store_true", help="also open the live player")
    v.add_argument("--step", type=int, default=None, help="lines per frame")
    v.add_argument("--fps", type=int, default=30)
    v.add_argument("--duration", type=float, default=15.0, help="export length in seconds")
    v.add_argument("--scale", type=float, default=1.0, help="export resolution scale")
    v.add_argument("--opacity", type=float, default=None, help="override thread opacity")
    v.set_defaults(func=_viz)

    d = sub.add_parser("demo", help="launch the interactive web demo (needs --extra demo)")
    d.add_argument("--port", type=int, default=7860)
    d.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container")
    d.set_defaults(
        func=lambda a: __import__("stringart.demo", fromlist=["launch"]).launch(a.port, a.host)
    )

    k = sub.add_parser("kit", help="build kit: 1:1 frame template, pin coordinates, shopping list")
    k.add_argument("source", help="run directory (with sequence.json)")
    k.add_argument("--frame-mm", type=float, required=True, help="real frame diameter / side")
    k.add_argument("--spool-m", type=float, default=500, help="metres of thread per spool")
    k.set_defaults(func=_kit)

    w = sub.add_parser("wind", help="winding assistant: step through the lines at the frame")
    w.add_argument("source", help="run directory (with sequence.json)")
    w.add_argument("--reset", action="store_true", help="start again from line 1")
    w.set_defaults(
        func=lambda a: __import__("stringart.wind", fromlist=["run"]).run(a.source, a.reset)
    )

    cal = sub.add_parser("calibrate", help="measure the real thread's opacity from a photo")
    calsub = cal.add_subparsers(dest="calcmd", required=True)
    cs = calsub.add_parser("sheet", help="write a short calibration pattern to wind")
    cs.add_argument("--pins", type=int, default=300)
    cs.add_argument("--frame-mm", type=float, default=700)
    cs.add_argument("--lines", type=int, default=250)
    cs.add_argument("--out", default="outputs/calibration")
    cs.set_defaults(func=_calibrate_sheet)
    cf = calsub.add_parser("fit", help="fit thread opacity from a photo of the wound pattern")
    cf.add_argument("photo")
    cf.add_argument("--sheet", default="outputs/calibration", help="folder from `sheet`")
    cf.add_argument("--circle", help="frame circle in the photo as cx,cy,r (if auto fails)")
    cf.set_defaults(func=_calibrate_fit)

    f = sub.add_parser("fetch-models", help="download the YuNet face and LBF landmark models")
    f.add_argument("--no-lbf", action="store_true", help="skip the 56 MB landmark model")
    f.set_defaults(func=_fetch_models)
    return p


def _kit(args) -> None:
    from .kit import write_kit

    for path in write_kit(Path(args.source), args.frame_mm, args.spool_m):
        print(f"wrote {path}")
    print((Path(args.source) / "shopping_list.txt").read_text(encoding="utf-8"))


def _calibrate_sheet(args) -> None:
    from .calibrate import calibration_target, make_sheet

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq, pins, text = make_sheet(args.pins, args.frame_mm, args.lines)
    (out / "instructions.txt").write_text(text, encoding="utf-8")
    save_sequence(
        out / "sequence.json",
        sequence=seq,
        pins=pins,
        size=600,
        frame="circle",
        opacity=0.25,
        meta={"calibration": True, "frame_mm": args.frame_mm},
    )
    save_gray(out / "pattern_preview.png", render_sequence(seq, pins, (600, 600), 0.25).image())
    save_gray(out / "pattern_target.png", calibration_target())
    print(f"calibration pattern: {len(seq) - 1} lines -> {out / 'instructions.txt'}")


def _calibrate_fit(args) -> None:
    from .calibrate import bare_mask, fit_opacity, photo_darkness

    doc = load_sequence(Path(args.sheet) / "sequence.json")
    if not doc.get("meta", {}).get("calibration"):
        raise ValueError(f"{args.sheet} does not hold a calibration pattern (run `sheet` first)")
    circle = tuple(float(v) for v in args.circle.split(",")) if args.circle else None
    bare = bare_mask(doc["sequence"], doc["n_pins"])
    dark = photo_darkness(load_image(args.photo), circle, bare=bare)
    frame_mm = doc["meta"]["frame_mm"]
    fit = fit_opacity(doc["sequence"], doc["n_pins"], frame_mm, dark)
    save_gray(Path(args.sheet) / "photo_darkness.png", 1.0 - dark)
    print(
        f"effective opacity {fit.opacity:.3f} -> use --thread-mm {fit.thread_mm:.3f} "
        f"with --frame-mm {frame_mm:g} (fit error {fit.error:.5f})"
    )


def _fetch_models(args) -> None:
    for name in ("yunet",) if args.no_lbf else ("yunet", "lbf"):
        print(f"{name}: {fetch(name)}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
