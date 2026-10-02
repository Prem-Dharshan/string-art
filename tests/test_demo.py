import importlib.util
import json
from pathlib import Path

import cv2
import pytest

from stringart.demo import make_art
from stringart.preprocess import load_image


@pytest.fixture(scope="module")
def photo():
    return cv2.cvtColor(load_image("sample:coffee"), cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("n_colors", [1, 2])
def test_make_art_outputs(photo, n_colors):
    target, render, gif, mp4, sheet, seq, md = make_art(photo, n_pins=128, frame_mm=500,
                                                        thread_mm=0.25, n_colors=n_colors,
                                                        refine_sweeps=1)
    assert render.shape[:2] == (600, 600) and target.shape[:2] == (600, 600)
    for f in (gif, mp4, sheet, seq):
        assert Path(f).is_file() and Path(f).stat().st_size > 0
    doc = json.loads(Path(seq).read_text())
    assert ("steps" in doc) == (n_colors > 1)
    assert "lines" in md and "thread" in md


def test_make_art_rejects_bad_input(photo):
    with pytest.raises(ValueError):
        make_art(None)
    with pytest.raises(ValueError):
        make_art(photo, frame_mm=100, thread_mm=0.6)  # thread wider than a pixel


@pytest.mark.skipif(importlib.util.find_spec("gradio") is None, reason="gradio not installed")
def test_ui_builds():
    from stringart.demo import build_ui

    assert build_ui() is not None
