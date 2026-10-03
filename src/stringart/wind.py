"""Winding assistant: step through a run one line at a time at the frame, with progress saved.

    stringart wind outputs/<run>            # opens the assistant, resumes where you stopped

Window: the piece so far with the *next* line highlighted, and in big type
"line 1,234 / 3,053 · black · pin 87 → pin 203", plus the next few steps.
Keys: space / right / n = done, next line · left / b = back one · e = jump to the end ·
0-9 then enter = go to a line number · q = quit (progress is saved on every step).
Progress lives in `<run>/progress.json`; `--reset` starts over.
"""

import json
from pathlib import Path

import numpy as np

from .io import load_sequence
from .viz import Source


class WindSession:
    """Navigation and saved progress, independent of any window (tested headless)."""

    def __init__(self, run_dir: Path, reset: bool = False):
        self.run_dir = Path(run_dir)
        doc = load_sequence(self.run_dir / "sequence.json")
        self.doc = doc
        shape = (doc["size"], doc["size"])
        if doc.get("mode") == "color":
            self.source = Source.color(
                doc["steps"], doc["pins"], shape, doc["colors"], doc["opacity"], doc["palette"]
            )
            self.names = list(doc["palette"])
        else:
            self.source = Source.gray(doc["sequence"], doc["pins"], shape, doc["opacity"])
            self.names = ["black"]
        self.n = self.source.n_lines
        self.path = self.run_dir / "progress.json"
        self.done = 0  # lines already wound
        if self.path.is_file() and not reset:
            self.done = int(json.loads(self.path.read_text())["done"])
            self.done = max(0, min(self.done, self.n))
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps({"done": self.done, "total": self.n}))

    def line(self, k: int) -> tuple[str, int, int]:
        """Thread name, from pin, to pin of line k (1-based)."""
        c, a, b = self.source.lines[k - 1]
        return (self.names[c] if c >= 0 else "black"), a, b

    def next(self) -> None:
        self.done = min(self.n, self.done + 1)
        self.save()

    def back(self) -> None:
        self.done = max(0, self.done - 1)
        self.save()

    def goto(self, done: int) -> None:
        self.done = max(0, min(self.n, int(done)))
        self.save()

    @property
    def finished(self) -> bool:
        return self.done >= self.n

    def headline(self) -> str:
        if self.finished:
            name, _, b = self.line(self.n)
            return f"All {self.n:,} lines wound. Tie off the {name} thread at pin {b}."
        k = self.done + 1
        name, a, b = self.line(k)
        switch = ""
        if k > 1 and self.line(k - 1)[0] != name:
            switch = f"  (switch to the {name} spool)"
        return f"line {k:,} / {self.n:,} · {name} · pin {a} → pin {b}{switch}"

    def upcoming(self, count: int = 6) -> list[str]:
        out = []
        for k in range(self.done + 2, min(self.n, self.done + 1 + count) + 1):
            name, a, b = self.line(k)
            out.append(f"{k:>6,}  {name:<8} {a:>4} → {b:<4}")
        return out

    def image(self) -> np.ndarray:
        canvas = self.source.new_canvas()
        for k in range(1, self.done + 1):
            self.source.draw(canvas, k)
        return canvas.image()


def run(run_dir: Path, reset: bool = False) -> None:
    import matplotlib.pyplot as plt

    s = WindSession(run_dir, reset)
    pins = s.source.pins
    fig = plt.figure(figsize=(12, 7.2))
    ax = fig.add_axes((0.01, 0.02, 0.55, 0.96))
    ax.set_axis_off()
    img = s.image()
    im = ax.imshow(img, **({} if img.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 1}))
    ax.scatter(pins[:, 0], pins[:, 1], s=2, c="tab:blue")
    (hl,) = ax.plot([], [], color="#ff2d55", lw=2.2)
    (dot_a,) = ax.plot([], [], "o", color="#ff2d55", ms=7)
    (dot_b,) = ax.plot([], [], "s", color="#ff2d55", ms=7)
    head = fig.text(0.58, 0.88, "", fontsize=19, fontweight="bold", wrap=True)
    nxt = fig.text(0.58, 0.80, "", fontsize=12, family="monospace", va="top")
    fig.text(
        0.58,
        0.06,
        "space/→ next · ←/b back · e end · digits+enter go to line · q quit\n"
        "circle = from pin, square = to pin · progress saves on every step",
        fontsize=9,
        color="gray",
    )
    typed = {"digits": ""}

    def redraw() -> None:
        im.set_data(s.image())
        if s.finished:
            hl.set_data([], [])
            dot_a.set_data([], [])
            dot_b.set_data([], [])
        else:
            _, a, b = s.line(s.done + 1)
            pa, pb = pins[a], pins[b]
            hl.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            dot_a.set_data([pa[0]], [pa[1]])
            dot_b.set_data([pb[0]], [pb[1]])
        go = f"   go to line: {typed['digits']}" if typed["digits"] else ""
        head.set_text(s.headline() + go)
        nxt.set_text("next:\n" + "\n".join(s.upcoming()))
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        key = event.key or ""
        if key.isdigit():
            typed["digits"] += key
        elif key == "enter" and typed["digits"]:
            s.goto(int(typed["digits"]) - 1)
            typed["digits"] = ""
        elif key in (" ", "right", "n"):
            s.next()
        elif key in ("left", "b", "backspace"):
            s.back()
        elif key == "e":
            s.goto(s.n)
        elif key == "escape":
            typed["digits"] = ""
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
