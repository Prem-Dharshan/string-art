"""Thread-by-thread visualizer (PLAN.md §2b).

Replays a pin sequence with the same renderer the solver output uses (`render.replay`), so the
last frame is exactly the final render. Three modes:

* `play`        interactive matplotlib window: target | threads so far | error map
                keys: space = pause/resume, right = step (when paused), +/- = speed, e = jump to end
* `export`      mp4 (cv2.VideoWriter) or gif (Pillow) of the build-up
* `snapshot_grid`  one figure showing the render at several line counts (report figure)
"""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .metrics import evaluate
from .render import Canvas, replay

SPEEDS = (1, 2, 5, 10, 20, 50, 100)


def _status_metrics(target, img, mask) -> str:
    if target is None:
        return ""
    m = evaluate(target, img, mask, sigmas=(2,))
    return f"   SSIM(σ=2) {m['ssim_s2']:.3f}   PSNR(σ=2) {m['psnr_s2']:.2f} dB"


def play(
    sequence: Sequence[int],
    pins: np.ndarray,
    shape: tuple[int, int],
    opacity: float,
    target: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    lines_per_frame: int = 10,
    interval_ms: int = 30,
    metrics_every: int = 200,
) -> None:
    """Open an interactive window that draws the sequence line by line."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    canvas = Canvas(shape, opacity)
    steps = replay(sequence, pins, canvas)
    n_lines = len(sequence) - 1
    state = {"k": 0, "speed": lines_per_frame, "paused": False, "status_metrics": ""}

    panels = 3 if target is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 5.6), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_axis_off()
    if target is not None:
        axes[0].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Target (preprocessed)")
        ax_r, ax_e = axes[1], axes[2]
        err_im = ax_e.imshow(np.abs(target - canvas.image()), cmap="magma", vmin=0, vmax=1)
        ax_e.set_title("|target − render|")
    else:
        ax_r, ax_e, err_im = axes[0], None, None
    ren_im = ax_r.imshow(canvas.image(), cmap="gray", vmin=0, vmax=1)
    ax_r.scatter(pins[:, 0], pins[:, 1], s=2, c="tab:blue")
    (new_line,) = ax_r.plot([], [], color="red", lw=1.2)
    (cur_pin,) = ax_r.plot([], [], "o", color="red", ms=4)
    ax_r.set_title("Threads so far")
    status = fig.text(0.01, 0.02, "", family="monospace", fontsize=9)
    fig.text(0.99, 0.02, "space pause · → step · +/- speed · e end", ha="right", fontsize=8,
             color="gray")

    def advance(count: int) -> None:
        last = state["k"]
        for _ in range(count):
            k = next(steps, None)
            if k is None:
                break
            last = k
        state["k"] = last

    def redraw() -> None:
        k = state["k"]
        img = canvas.image()
        ren_im.set_data(img)
        if k > 0:
            a, b = pins[sequence[k - 1]], pins[sequence[k]]
            new_line.set_data([a[0], b[0]], [a[1], b[1]])
            cur_pin.set_data([b[0]], [b[1]])
        if err_im is not None:
            err_im.set_data(np.abs(target - img))
        done = k >= n_lines
        if target is not None and (done or k % metrics_every < state["speed"]):
            state["status_metrics"] = _status_metrics(target, img, mask)
        pin_txt = f"pin {sequence[k - 1]:>3} → {sequence[k]:>3}" if k > 0 else ""
        flag = "  [done]" if done else ("  [paused]" if state["paused"] else "")
        status.set_text(
            f"line {k:>5}/{n_lines}   {pin_txt}   {state['speed']} lines/frame"
            f"{state['status_metrics']}{flag}"
        )

    def on_frame(_):
        if not state["paused"] and state["k"] < n_lines:
            advance(state["speed"])
        redraw()
        return ren_im, new_line, cur_pin, status

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right" and state["paused"]:
            advance(1)
        elif event.key in ("+", "="):
            i = SPEEDS.index(state["speed"]) if state["speed"] in SPEEDS else 0
            state["speed"] = SPEEDS[min(i + 1, len(SPEEDS) - 1)]
        elif event.key == "-":
            i = SPEEDS.index(state["speed"]) if state["speed"] in SPEEDS else 0
            state["speed"] = SPEEDS[max(i - 1, 0)]
        elif event.key == "e":
            advance(n_lines)
        redraw()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    # Keep a reference so the animation isn't garbage-collected while the window is open.
    fig._stringart_anim = FuncAnimation(fig, on_frame, interval=interval_ms,
                                        cache_frame_data=False)
    plt.show()


def _frame_bgr(img: np.ndarray, a, b, k: int, n_lines: int, scale: float) -> np.ndarray:
    frame = cv2.cvtColor(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if a is not None:
        p0 = tuple(int(round(v * scale)) for v in a)
        p1 = tuple(int(round(v * scale)) for v in b)
        cv2.line(frame, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{k}/{n_lines}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 200), 1,
                cv2.LINE_AA)
    return frame


def iter_frames(sequence, pins, shape, opacity, lines_per_frame: int):
    """Yield (k, canvas) every `lines_per_frame` lines, always including the final line."""
    canvas = Canvas(shape, opacity)
    n_lines = len(sequence) - 1
    for k in replay(sequence, pins, canvas):
        if k % lines_per_frame == 0 or k == n_lines:
            yield k, canvas


def export(
    sequence: Sequence[int],
    pins: np.ndarray,
    shape: tuple[int, int],
    opacity: float,
    out_path: Path,
    lines_per_frame: int | None = None,
    fps: int = 30,
    duration_s: float = 15.0,
    hold_s: float = 2.0,
    scale: float = 1.0,
) -> Path:
    """Write the build-up as .mp4 or .gif. By default the clip lasts about `duration_s`."""
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        fps = min(fps, 10)  # GIFs balloon quickly; 10 fps keeps a 15 s clip to a few MB
    n_lines = len(sequence) - 1
    if lines_per_frame is None:
        lines_per_frame = max(1, int(np.ceil(n_lines / (fps * duration_s))))
    frames = (
        _frame_bgr(c.image(), pins[sequence[k - 1]], pins[sequence[k]], k, n_lines, scale)
        for k, c in iter_frames(sequence, pins, shape, opacity, lines_per_frame)
    )
    if suffix == ".mp4":
        h, w = int(round(shape[0] * scale)), int(round(shape[1] * scale))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")
        last = None
        for last in frames:
            writer.write(last)
        for _ in range(int(hold_s * fps)):
            writer.write(last)
        writer.release()
    elif suffix == ".gif":
        from PIL import Image

        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        durations = [int(1000 / fps)] * (len(imgs) - 1) + [int(hold_s * 1000)]
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=durations,
                     loop=0, optimize=True)
    else:
        raise ValueError(f"unsupported export format {suffix!r} (use .mp4 or .gif)")
    return out_path


def snapshot_grid(
    sequence: Sequence[int],
    pins: np.ndarray,
    shape: tuple[int, int],
    opacity: float,
    counts: Sequence[int],
    out_path: Path,
    target: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> Path:
    """Save one figure with the render after each line count in `counts` (plus target if given)."""
    from matplotlib.figure import Figure  # no pyplot: never touches the interactive backend

    n_lines = len(sequence) - 1
    wanted = sorted({min(c, n_lines) for c in counts if c > 0})
    snaps = {}
    canvas = Canvas(shape, opacity)
    for k in replay(sequence, pins, canvas):
        if k in wanted:
            snaps[k] = canvas.image().copy()
            if len(snaps) == len(wanted):
                break
    panels = [("Target", target)] if target is not None else []
    for k in wanted:
        title = f"{k} lines"
        if target is not None:
            m = evaluate(target, snaps[k], mask, sigmas=(2,))
            title += f"\nSSIM(σ=2) {m['ssim_s2']:.3f}"
        panels.append((title, snaps[k]))
    fig = Figure(figsize=(3.2 * len(panels), 3.6))
    axes = fig.subplots(1, len(panels), squeeze=False)
    for ax, (title, img) in zip(axes[0], panels, strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return Path(out_path)
