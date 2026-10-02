"""Thread-by-thread visualizer (PLAN.md §2b).

Replays a result with the same renderer the solver output uses (`render.Canvas` for black
thread, `color.ColorCanvas` for colour), so the last frame is exactly the final render.
A `Source` wraps either a pin sequence (grayscale) or colour steps. Three modes:

* `play`           interactive matplotlib window: target | threads so far | error map
                   keys: space = pause/resume, right = step (paused), +/- = speed, e = end
* `export`         mp4 (cv2.VideoWriter) or gif (Pillow) of the build-up
* `snapshot_grid`  one figure showing the render at several line counts (report figure)
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .metrics import evaluate
from .render import Canvas

SPEEDS = (1, 2, 5, 10, 20, 50, 100)
LUMA = np.array([0.299, 0.587, 0.114])


@dataclass
class Source:
    pins: np.ndarray
    shape: tuple[int, int]
    lines: list[tuple[int, int, int]]  # (colour index, from pin, to pin); colour -1 = black
    opacity: float
    colors: np.ndarray | None = None  # (K, 3) RGB in [0, 1]; None = grayscale
    names: list[str] | None = None

    @classmethod
    def gray(cls, sequence: Sequence[int], pins, shape, opacity) -> "Source":
        seq = list(sequence)
        return cls(np.asarray(pins), tuple(shape),
                   [(-1, a, b) for a, b in zip(seq[:-1], seq[1:], strict=True)], opacity)

    @classmethod
    def color(cls, steps, pins, shape, colors, opacity, names=None) -> "Source":
        return cls(np.asarray(pins), tuple(shape), [tuple(s) for s in steps], opacity,
                   np.asarray(colors, dtype=np.float64), names)

    @property
    def is_color(self) -> bool:
        return self.colors is not None

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    def new_canvas(self):
        if self.is_color:
            from .color import ColorCanvas

            return ColorCanvas(self.shape, self.colors, self.opacity)
        return Canvas(self.shape, self.opacity)

    def draw(self, canvas, k: int) -> None:
        """Draw line k (1-based) onto the canvas."""
        c, a, b = self.lines[k - 1]
        if self.is_color:
            canvas.add_line(c, self.pins[a], self.pins[b])
        else:
            canvas.add_line(self.pins[a], self.pins[b])

    def frames(self, every: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (k, image) every `every` lines, always including the last line."""
        canvas = self.new_canvas()
        for k in range(1, self.n_lines + 1):
            self.draw(canvas, k)
            if k % every == 0 or k == self.n_lines:
                yield k, canvas.image()

    def final(self) -> np.ndarray:
        canvas = self.new_canvas()
        for k in range(1, self.n_lines + 1):
            self.draw(canvas, k)
        return canvas.image()

    def label(self, k: int) -> str:
        c, a, b = self.lines[k - 1]
        thread = f"{self.names[c]:<7} " if self.is_color and self.names else ""
        return f"{thread}pin {a:>3} → {b:>3}"


def _luma(img: np.ndarray) -> np.ndarray:
    return img @ LUMA if img.ndim == 3 else img


def _status_metrics(target, img, mask) -> str:
    if target is None:
        return ""
    m = evaluate(_luma(target), _luma(img), mask, sigmas=(2,))
    return f"   SSIM(σ=2) {m['ssim_s2']:.3f}   PSNR(σ=2) {m['psnr_s2']:.2f} dB"


def _imshow_kw(img):
    return {} if img.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 1}


def play(src: Source, target: np.ndarray | None = None, mask: np.ndarray | None = None,
         lines_per_frame: int = 10, interval_ms: int = 30, metrics_every: int = 200) -> None:
    """Open an interactive window that draws the result line by line."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    canvas = src.new_canvas()
    n_lines = src.n_lines
    state = {"k": 0, "speed": lines_per_frame, "paused": False, "status_metrics": ""}
    hi = "#00c8ff" if src.is_color else "red"  # highlight that never matches a thread

    panels = 3 if target is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 5.6), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_axis_off()
    if target is not None:
        axes[0].imshow(target, **_imshow_kw(target))
        axes[0].set_title("Target (preprocessed)")
        ax_r, ax_e = axes[1], axes[2]
        err = np.abs(_luma(target) - _luma(canvas.image()))
        err_im = ax_e.imshow(err, cmap="magma", vmin=0, vmax=1)
        ax_e.set_title("|target − render| (luminance)")
    else:
        ax_r, err_im = axes[0], None
    ren_im = ax_r.imshow(canvas.image(), **_imshow_kw(canvas.image()))
    ax_r.scatter(src.pins[:, 0], src.pins[:, 1], s=2, c="tab:blue")
    (new_line,) = ax_r.plot([], [], color=hi, lw=1.2)
    (cur_pin,) = ax_r.plot([], [], "o", color=hi, ms=4)
    ax_r.set_title("Threads so far")
    status = fig.text(0.01, 0.02, "", family="monospace", fontsize=9)
    fig.text(0.99, 0.02, "space pause · → step · +/- speed · e end", ha="right", fontsize=8,
             color="gray")

    def advance(count: int) -> None:
        for _ in range(count):
            if state["k"] >= n_lines:
                break
            state["k"] += 1
            src.draw(canvas, state["k"])

    def redraw() -> None:
        k = state["k"]
        img = canvas.image()
        ren_im.set_data(img)
        if k > 0:
            _, a, b = src.lines[k - 1]
            pa, pb = src.pins[a], src.pins[b]
            new_line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            cur_pin.set_data([pb[0]], [pb[1]])
        if err_im is not None:
            err_im.set_data(np.abs(_luma(target) - _luma(img)))
        done = k >= n_lines
        if target is not None and (done or k % metrics_every < state["speed"]):
            state["status_metrics"] = _status_metrics(target, img, mask)
        what = src.label(k) if k > 0 else ""
        flag = "  [done]" if done else ("  [paused]" if state["paused"] else "")
        status.set_text(f"line {k:>5}/{n_lines}   {what}   {state['speed']} lines/frame"
                        f"{state['status_metrics']}{flag}")

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


def _to_bgr8(img: np.ndarray) -> np.ndarray:
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR if img.ndim == 3 else cv2.COLOR_GRAY2BGR)


def _frame_bgr(src: Source, img: np.ndarray, k: int, scale: float) -> np.ndarray:
    frame = _to_bgr8(img)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, a, b = src.lines[k - 1]
    p0 = tuple(int(round(v * scale)) for v in src.pins[a])
    p1 = tuple(int(round(v * scale)) for v in src.pins[b])
    color = (255, 200, 0) if src.is_color else (0, 0, 255)
    cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{k}/{src.n_lines}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (40, 40, 200), 1, cv2.LINE_AA)
    return frame


def export(src: Source, out_path: Path, lines_per_frame: int | None = None, fps: int = 30,
           duration_s: float = 15.0, hold_s: float = 2.0, scale: float = 1.0) -> Path:
    """Write the build-up as .mp4 or .gif. By default the clip lasts about `duration_s`."""
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError(f"unsupported export format {suffix!r} (use .mp4 or .gif)")
    if src.n_lines == 0:
        raise ValueError("nothing to export: the result has no lines")
    if suffix == ".gif":
        fps = min(fps, 10)  # GIFs balloon quickly; 10 fps keeps a 15 s clip to a few MB
    if lines_per_frame is None:
        lines_per_frame = max(1, int(np.ceil(src.n_lines / (fps * duration_s))))
    frames = (_frame_bgr(src, img, k, scale) for k, img in src.frames(lines_per_frame))
    if suffix == ".mp4":
        h, w = int(round(src.shape[0] * scale)), int(round(src.shape[1] * scale))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")
        last = None
        for last in frames:
            writer.write(last)
        for _ in range(int(hold_s * fps)):
            writer.write(last)
        writer.release()
    else:
        from PIL import Image

        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        durations = [int(1000 / fps)] * (len(imgs) - 1) + [int(hold_s * 1000)]
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=durations,
                     loop=0, optimize=True)
    return out_path


def snapshot_grid(src: Source, counts: Sequence[int], out_path: Path,
                  target: np.ndarray | None = None, mask: np.ndarray | None = None) -> Path:
    """Save one figure with the render after each line count in `counts` (plus target)."""
    from matplotlib.figure import Figure  # no pyplot: never touches the interactive backend

    wanted = sorted({min(c, src.n_lines) for c in counts if c > 0})
    snaps = {}
    canvas = src.new_canvas()
    for k in range(1, src.n_lines + 1):
        src.draw(canvas, k)
        if k in wanted:
            snaps[k] = canvas.image().copy()
            if len(snaps) == len(wanted):
                break
    panels = [("Target", target)] if target is not None else []
    for k in wanted:
        title = f"{k} lines"
        if target is not None:
            m = evaluate(_luma(target), _luma(snaps[k]), mask, sigmas=(2,))
            title += f"\nSSIM(σ=2) {m['ssim_s2']:.3f}"
        panels.append((title, snaps[k]))
    fig = Figure(figsize=(3.2 * len(panels), 3.6))
    axes = fig.subplots(1, len(panels), squeeze=False)
    for ax, (title, img) in zip(axes[0], panels, strict=True):
        ax.imshow(img, **_imshow_kw(img))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return Path(out_path)
