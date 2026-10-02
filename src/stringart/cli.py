"""Command line: `stringart run <image>` solves, `stringart viz <run dir or sequence.json>` replays."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import viz
from .geometry import make_pins
from .io import load_gray, load_sequence, save_gray, save_sequence
from .metrics import evaluate
from .preprocess import PreprocessConfig, frame_mask, load_image, preprocess
from .render import render_sequence, to_svg
from .solver.baseline import BaselineConfig, solve_baseline
from .solver.greedy import GreedyConfig, opacity_from_physical, solve_greedy


def _run(args) -> None:
    if (args.thread_mm is None) != (args.frame_mm is None):
        raise ValueError("--thread-mm and --frame-mm must be given together")
    if args.thread_mm is not None:
        args.opacity = round(opacity_from_physical(args.thread_mm, args.frame_mm, args.size), 4)
    pcfg = PreprocessConfig(size=args.size, frame=args.frame, clahe_clip=args.clahe,
                            blur_sigma=args.blur)
    target, mask = preprocess(load_image(args.image), pcfg)
    pins = make_pins(args.frame, args.pins, args.size)
    if args.solver == "baseline":
        scfg = BaselineConfig(n_lines=args.lines or 3000, line_strength=args.line_strength,
                              min_gap=args.min_gap, n_candidates=args.candidates,
                              darkness_penalty=args.darkness_penalty, seed=args.seed)
        res = solve_baseline(target, pins, scfg, progress=not args.quiet)
    else:
        scfg = GreedyConfig(max_lines=args.lines or 8000, opacity=args.opacity,
                            min_gap=args.min_gap, max_repeats=args.max_repeats,
                            objective=args.objective, blur_sigma=args.blur_sigma)
        res = solve_greedy(target, pins, scfg, progress=not args.quiet)
    render = render_sequence(res.sequence, pins, target.shape, args.opacity).image()
    metrics = evaluate(target, render, mask)

    stem = Path(args.image.split(":", 1)[-1]).stem
    out = Path(args.out) if args.out else Path("outputs") / f"{stem}_{args.solver}"
    out.mkdir(parents=True, exist_ok=True)
    save_gray(out / "target.png", target)
    save_gray(out / "render.png", render)
    (out / "render.svg").write_text(to_svg(res.sequence, pins, args.size, args.opacity))
    meta = {"image": args.image, "solver": args.solver, "elapsed_s": round(res.elapsed_s, 3),
            "preprocess": asdict(pcfg), "solver_config": asdict(scfg), "metrics": metrics}
    save_sequence(out / "sequence.json", sequence=res.sequence, pins=pins, size=args.size,
                  frame=args.frame, opacity=args.opacity, meta=meta)
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(res.sequence) - 1} lines in {res.elapsed_s:.2f}s -> {out}")
    print("  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    if args.viz:
        viz.play(res.sequence, pins, target.shape, args.opacity, target, mask)


def _viz(args) -> None:
    src = Path(args.source)
    seq_path = src / "sequence.json" if src.is_dir() else src
    doc = load_sequence(seq_path)
    target_path = Path(args.target) if args.target else seq_path.parent / "target.png"
    target = load_gray(target_path) if target_path.is_file() else None
    mask = frame_mask(doc["frame"], doc["size"]) if target is not None else None
    shape = (doc["size"], doc["size"])
    seq, pins, opacity = doc["sequence"], doc["pins"], args.opacity or doc["opacity"]

    did_file_output = False
    if args.grid:
        counts = [int(c) for c in args.grid.split(",")]
        path = viz.snapshot_grid(seq, pins, shape, opacity, counts,
                                 seq_path.parent / "grid.png", target, mask)
        print(f"wrote {path}")
        did_file_output = True
    for out in args.save or []:
        path = viz.export(seq, pins, shape, opacity, Path(out), lines_per_frame=args.step,
                          fps=args.fps, duration_s=args.duration, scale=args.scale)
        print(f"wrote {path}")
        did_file_output = True
    if not did_file_output or args.show:
        viz.play(seq, pins, shape, opacity, target, mask, lines_per_frame=args.step or 10)


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
    r.add_argument("--lines", type=int, default=None,
                   help="baseline: line count (default 3000); greedy: cap (default 8000)")
    r.add_argument("--min-gap", type=int, default=10)
    r.add_argument("--opacity", type=float, default=0.2,
                   help="thread opacity per pixel (solver model and rendering)")
    r.add_argument("--thread-mm", type=float, help="thread width; with --frame-mm sets opacity")
    r.add_argument("--frame-mm", type=float, help="frame diameter/side in mm")
    g = r.add_argument_group("greedy solver")
    g.add_argument("--objective", choices=["pixel", "blur"], default="pixel")
    g.add_argument("--blur-sigma", type=float, default=1.5, help="viewing blur for 'blur'")
    g.add_argument("--max-repeats", type=int, default=2, help="max uses of one chord")
    b = r.add_argument_group("baseline solver")
    b.add_argument("--line-strength", type=float, default=0.1)
    b.add_argument("--candidates", type=int, default=None, help="random candidates per step")
    b.add_argument("--darkness-penalty", type=float, default=0.0)
    r.add_argument("--clahe", type=float, default=2.0, help="CLAHE clip limit (0 = off)")
    r.add_argument("--blur", type=float, default=1.0, help="Gaussian sigma (0 = off)")
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
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
