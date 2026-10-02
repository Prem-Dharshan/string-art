import numpy as np
import pytest

from stringart.fabrication import instructions, thread_length_mm
from stringart.geometry import circle_pins


def test_thread_length_diameters():
    pins = circle_pins(4, 101)  # top, right, bottom, left on a 100 px diameter
    # top->bottom is one diameter; on a 500 mm frame that's 500 mm.
    assert thread_length_mm([0, 2], pins, 101, 500) == pytest.approx(500)
    assert thread_length_mm([0, 2, 0], pins, 101, 500) == pytest.approx(1000)
    assert thread_length_mm([0, 1], pins, 101, 500) == pytest.approx(500 / np.sqrt(2))


def test_instructions_lists_every_line():
    pins = circle_pins(16, 101)
    seq = [0, 8, 3, 11, 5]
    txt = instructions(seq, pins, 101, "circle", frame_mm=400, block=2)
    for k, (a, b) in enumerate(zip(seq, seq[1:], strict=False), start=1):
        assert f"{k:5d}  {a:4d} -> {b:4d}" in txt
    assert "tie the thread at pin 0" in txt and "tie off at pin 5" in txt
    assert "lines: 4" in txt and " m " in txt
    assert "frame widths" in instructions(seq, pins, 101, "circle")
