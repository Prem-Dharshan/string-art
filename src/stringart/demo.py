"""Interactive demo (Gradio): photo in, string art + build-up animation + build sheet out.

    uv run --extra demo stringart demo          # then open http://127.0.0.1:7860

Everything runs locally; nothing is uploaded anywhere.
"""

import json
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import viz
from .color import (
    ColorConfig,
    color_metrics,
    color_target,
    fit_palette,
    palette_rgb,
    render_steps,
    solve_color,
)
from .fabrication import color_instructions, instructions, thread_length_mm
from .geometry import make_pins
from .importance import auto_weights
from .io import save_color_result, save_sequence
from .kit import write_kit
from .metrics import evaluate
from .preprocess import PreprocessConfig, prepare
from .render import render_sequence
from .solver.greedy import GreedyConfig, opacity_from_physical, solve_greedy
from .solver.refine import RefineConfig, refine

SIZE = 600


def make_art(
    image_rgb,
    n_pins=300,
    frame_mm=700.0,
    thread_mm=0.25,
    n_colors=1,
    refine_sweeps=2,
    crop="face",
    importance="auto",
    progress=None,
):
    """Run the full pipeline. Returns (target, render, gif, mp4, instructions, sequence, md)."""
    if image_rgb is None:
        raise ValueError("upload a photo first")
    say = progress or (lambda *_a, **_k: None)
    t0 = time.perf_counter()
    out = Path(tempfile.mkdtemp(prefix="stringart_"))
    bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    opacity = round(opacity_from_physical(thread_mm, frame_mm, SIZE), 4)
    if not 0 < opacity < 1:
        raise ValueError("thread too thick for this frame: thread_mm * 600 / frame_mm must be < 1")
    pins = make_pins("circle", int(n_pins), SIZE)
    min_gap = max(2, round(10 * int(n_pins) / 256))

    say(0.05, desc="preprocessing (face detection, contrast)")
    prep = prepare(bgr, PreprocessConfig(size=SIZE, crop=crop))
    weights, _ = auto_weights(prep, importance)

    if int(n_colors) <= 1:
        say(0.2, desc="solving (greedy)")
        res = solve_greedy(
            prep.target,
            pins,
            GreedyConfig(opacity=opacity, min_gap=min_gap),
            weights=weights,
            progress=False,
        )
        seq = res.sequence
        if refine_sweeps:
            say(0.45, desc=f"refining ({int(refine_sweeps)} sweeps)")
            seq, _ = refine(
                prep.target,
                pins,
                seq,
                opacity,
                min_gap=min_gap,
                weights=weights,
                cfg=RefineConfig(sweeps=int(refine_sweeps)),
                progress=False,
            )
        target_img = prep.target
        render = render_sequence(seq, pins, prep.target.shape, opacity).image()
        src = viz.Source.gray(seq, pins, prep.target.shape, opacity)
        m = evaluate(prep.plain, render, prep.mask, sigmas=(2,))
        quality = f"SSIM (viewing blur σ=2) vs photo: **{m['ssim_s2']:.3f}**"
        length = thread_length_mm(seq, pins, SIZE, frame_mm) / 1000
        sheet = instructions(seq, pins, SIZE, "circle", frame_mm)
        save_sequence(
            out / "sequence.json",
            sequence=seq,
            pins=pins,
            size=SIZE,
            frame="circle",
            opacity=opacity,
        )
        n_lines, palette_txt = len(seq) - 1, "black"
    else:
        say(0.2, desc="choosing thread colours (reachable gamut)")
        target_img = color_target(prep)
        names = fit_palette(target_img, prep.mask, int(n_colors), weights=weights)
        colors = palette_rgb(names)
        say(0.3, desc="solving (joint colour greedy)")
        res = solve_color(
            target_img,
            pins,
            colors,
            ColorConfig(opacity=opacity, min_gap=min_gap),
            weights=weights,
            names=names,
        )
        render = render_steps(res.steps, pins, target_img.shape[:2], colors, opacity).image()
        src = viz.Source.color(res.steps, pins, target_img.shape[:2], colors, opacity, names)
        m = color_metrics(target_img, render, prep.mask, sigmas=(2,))
        quality = (
            f"colour error ΔE2000 (σ=2): **{m['de2000_s2']:.1f}**, "
            f"luminance SSIM: **{m['ssim_lum_s2']:.3f}**"
        )
        p = np.asarray(pins)
        length = sum(float(np.hypot(*(p[b] - p[a]))) for _, a, b in res.steps)
        length *= frame_mm / (SIZE - 1) / 1000
        sheet = color_instructions(res.steps, names, pins, SIZE, "circle", frame_mm)
        save_color_result(
            out / "sequence.json",
            steps=res.steps,
            palette=names,
            colors=colors,
            pins=pins,
            size=SIZE,
            frame="circle",
            opacity=opacity,
        )
        n_lines, palette_txt = len(res.steps), ", ".join(names)

    say(0.75, desc="rendering the build-up animation")
    gif = viz.export(src, out / "build.gif", duration_s=10, scale=0.5)
    mp4 = viz.export(src, out / "build.mp4", duration_s=15)
    (out / "instructions.txt").write_text(sheet, encoding="utf-8")
    # Everything needed at the workbench later, in one download.
    kit_files = [out / "instructions.txt", out / "sequence.json", *write_kit(out, frame_mm)]
    kit_zip = out / "build_kit.zip"
    with zipfile.ZipFile(kit_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in kit_files:
            z.write(f, f.name)
    summary = {
        "lines": n_lines,
        "pins": int(n_pins),
        "frame_mm": frame_mm,
        "thread_mm": thread_mm,
        "opacity": opacity,
        "thread_length_m": round(length, 1),
        "faces": len(prep.faces),
        "palette": palette_txt,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = (
        f"**{n_lines} lines** · thread ≈ **{length:.0f} m** (+10% for knots) · "
        f"{int(n_pins)} pins on a {frame_mm:g} mm frame · threads: {palette_txt} · "
        f"faces found: {len(prep.faces)} · {summary['seconds']} s\n\n{quality}"
    )
    return (
        target_img,
        render,
        str(gif),
        str(mp4),
        str(kit_zip),
        str(out / "sequence.json"),
        md,
    )


def build_ui(samples_dir: Path | None = None, footer: str = ""):
    """The Gradio app. `samples_dir` (default: the repo's data/samples) feeds the examples."""
    import gradio as gr

    from .preprocess import SAMPLE_DIR

    samples = sorted(Path(samples_dir or SAMPLE_DIR).glob("*.jpg"))
    with gr.Blocks(title="String art") as ui:
        gr.Markdown(
            "# Computational string art\nUpload a photo. The pipeline crops to the "
            "face, weights the important features, chooses every thread line, and "
            "gives you a step-by-step build sheet."
        )
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(label="Photo", type="numpy", sources=["upload", "clipboard"])
                with gr.Accordion("Frame and thread", open=True):
                    n_pins = gr.Slider(128, 360, value=300, step=4, label="Pins")
                    frame_mm = gr.Slider(300, 1000, value=700, step=10, label="Frame diameter (mm)")
                    thread_mm = gr.Slider(
                        0.1, 0.6, value=0.25, step=0.01, label="Thread width (mm)"
                    )
                    n_colors = gr.Slider(
                        1, 5, value=1, step=1, label="Thread colours (1 = black only)"
                    )
                with gr.Accordion("Advanced", open=False):
                    refine_sweeps = gr.Slider(
                        0, 3, value=2, step=1, label="Refinement sweeps (black only)"
                    )
                    crop = gr.Radio(["face", "center"], value="face", label="Crop")
                    importance = gr.Radio(
                        ["auto", "on", "off"], value="auto", label="Importance weights"
                    )
                go = gr.Button("Make string art", variant="primary")
                if samples:
                    gr.Examples(
                        [[str(p)] for p in samples], inputs=[photo], label="Or try a sample photo"
                    )
            with gr.Column(scale=2):
                summary = gr.Markdown()
                with gr.Row():
                    target = gr.Image(label="Target (after preprocessing)")
                    render = gr.Image(label="String art (simulated)")
                anim = gr.Image(label="Build-up, thread by thread")
                with gr.Row():
                    mp4 = gr.File(label="Animation (mp4)")
                    sheet = gr.File(
                        label="Build kit (zip): winding list, 1:1 template, pins, shopping list"
                    )
                    seq = gr.File(label="Pin sequence (sequence.json)")

        def run(
            photo,
            n_pins,
            frame_mm,
            thread_mm,
            n_colors,
            refine_sweeps,
            crop,
            importance,
            progress=gr.Progress(),  # noqa: B008 - Gradio injects progress via this default
        ):
            try:
                return make_art(
                    photo,
                    n_pins,
                    frame_mm,
                    thread_mm,
                    n_colors,
                    refine_sweeps,
                    crop,
                    importance,
                    progress,
                )
            except ValueError as e:
                raise gr.Error(str(e)) from e

        go.click(
            run,
            [photo, n_pins, frame_mm, thread_mm, n_colors, refine_sweeps, crop, importance],
            [target, render, anim, mp4, sheet, seq, summary],
        )
        if footer:
            gr.Markdown(footer)
    return ui


def launch(port: int = 7860, host: str = "127.0.0.1", share: bool = False) -> None:
    try:
        import gradio  # noqa: F401
    except ImportError as e:
        raise SystemExit("the demo needs gradio: run `uv run --extra demo stringart demo`") from e
    build_ui().launch(server_name=host, server_port=port, share=share)
