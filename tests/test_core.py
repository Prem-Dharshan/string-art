import cv2
import numpy as np
import pytest

from stringart.geometry import circle_pins, pin_distance, rect_pins
from stringart.metrics import evaluate
from stringart.preprocess import PreprocessConfig, frame_mask, load_image, preprocess
from stringart.raster import aa_line, line_pixels
from stringart.render import Canvas, render_sequence


def test_circle_pins_on_circle_and_in_bounds():
    p = circle_pins(200, 301)
    r = np.hypot(p[:, 0] - 150, p[:, 1] - 150)
    assert np.allclose(r, 150)
    assert p.min() >= 0 and p.max() <= 300


def test_rect_pins_on_border():
    p = rect_pins(100, 50, 50)
    on_border = (np.isclose(p, 0) | np.isclose(p, 49)).any(axis=1)
    assert on_border.all()


def test_pin_distance_wraps():
    assert pin_distance(0, 255, 256) == 1
    assert pin_distance(10, 138, 256) == 128


@pytest.mark.parametrize("p0,p1", [((3, 7), (90, 40)), ((10, 90), (40, 2)), ((50, 5), (50, 95)),
                                   ((0, 0), (99, 99))])
def test_aa_line_unit_coverage_per_step(p0, p1):
    ys, xs, w = aa_line(p0, p1, (100, 100))
    steep = abs(p1[1] - p0[1]) > abs(p1[0] - p0[0])
    major = ys if steep else xs
    sums = np.bincount(major, weights=w)
    assert np.allclose(sums[sums > 0], 1.0)
    assert len(set(zip(ys.tolist(), xs.tolist(), strict=True))) == len(ys)  # no duplicates


def test_aa_line_matches_cv2_reference():
    p0, p1 = (5.0, 12.0), (190.0, 151.0)
    ys, xs, w = aa_line(p0, p1, (200, 200))
    ours = np.zeros((200, 200))
    ours[ys, xs] = w
    ref = np.zeros((200, 200), np.uint8)
    cv2.line(ref, (5, 12), (190, 151), 255, 1, cv2.LINE_AA)
    blur = lambda a: cv2.GaussianBlur(a.astype(np.float64), (0, 0), 1.0)  # noqa: E731
    corr = np.corrcoef(blur(ours).ravel(), blur(ref).ravel())[0, 1]
    assert corr > 0.95


def test_line_pixels_connected():
    ys, xs = line_pixels((0, 0), (80, 33), (100, 100))
    assert np.all(np.abs(np.diff(xs)) <= 1) and np.all(np.abs(np.diff(ys)) <= 1)


def test_canvas_darkness_saturates_below_one():
    c = Canvas((50, 50), opacity=0.5)
    for _ in range(200):
        c.add_line((0, 25), (49, 25))
    assert c.dark.max() <= 1.0 and c.dark[25, 25] > 0.99


def test_render_sequence_empty_is_white(pins):
    assert np.all(render_sequence([0], pins, (120, 120), 0.2).image() == 1.0)


def test_preprocess_non_square_input_and_white_outside():
    img = np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    t, mask = preprocess(img, PreprocessConfig(size=128))
    assert t.shape == (128, 128) and 0 <= t.min() and t.max() <= 1
    assert np.all(t[~mask] == 1.0)
    assert np.array_equal(mask, frame_mask("circle", 128))


def test_load_image_errors():
    with pytest.raises(FileNotFoundError):
        load_image("does/not/exist.png")
    with pytest.raises(ValueError):
        load_image("sample:nope")


def test_metrics_identical_images():
    a = np.random.default_rng(1).random((64, 64))
    m = evaluate(a, a)
    assert m["ssim_s0"] == pytest.approx(1.0) and m["psnr_s2"] == float("inf")
