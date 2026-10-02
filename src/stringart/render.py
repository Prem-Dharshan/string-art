"""Incremental thread renderer shared by solvers' final output and the visualizer.

Physical model: each thread covers a pixel with opacity `opacity * coverage`; overlapping
threads compose multiplicatively, so darkness d stays in [0, 1]:  d <- d + a * (1 - d).
"""

from collections.abc import Iterator, Sequence

import numpy as np

from .raster import aa_line


class Canvas:
    def __init__(self, shape: tuple[int, int], opacity: float = 0.2):
        self.shape = shape
        self.opacity = opacity
        self.dark = np.zeros(shape, dtype=np.float64)

    def add_line(self, p0, p1) -> None:
        ys, xs, w = aa_line(p0, p1, self.shape)
        a = self.opacity * w
        d = self.dark[ys, xs]
        self.dark[ys, xs] = d + a * (1.0 - d)

    def image(self) -> np.ndarray:
        """Grayscale render in [0, 1] (1 = white board)."""
        return 1.0 - self.dark


def replay(sequence: Sequence[int], pins: np.ndarray, canvas: Canvas) -> Iterator[int]:
    """Draw the pin sequence onto `canvas` line by line, yielding k after line k (1-based)."""
    for k in range(1, len(sequence)):
        canvas.add_line(pins[sequence[k - 1]], pins[sequence[k]])
        yield k


def render_sequence(sequence: Sequence[int], pins: np.ndarray, shape, opacity: float) -> Canvas:
    canvas = Canvas(shape, opacity)
    for _ in replay(sequence, pins, canvas):
        pass
    return canvas


def to_svg(sequence: Sequence[int], pins: np.ndarray, size: int, opacity: float,
           stroke_px: float = 1.0) -> str:
    pts = " ".join(f"{pins[i][0]:.2f},{pins[i][1]:.2f}" for i in sequence)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<polyline points="{pts}" fill="none" stroke="black" '
        f'stroke-opacity="{opacity}" stroke-width="{stroke_px}"/></svg>'
    )
