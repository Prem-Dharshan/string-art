import cv2
import numpy as np
import pytest

from stringart.calibrate import bare_mask, fit_opacity, make_sheet, photo_darkness
from stringart.cli import main
from stringart.render import render_sequence


def _fake_photo(seq, pins, opacity, board=0.82, offset=(140, 90), scale=1.4, seed=0):
    """A 'photo' of the wound pattern: grey board, dark frame ring, shifted and scaled."""
    dark = 1.0 - render_sequence(seq, pins, (600, 600), opacity).image()
    canvas = board * (1.0 - dark)
    big = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = 1100, 1300
    photo = np.full((h, w), 0.55)
    y0, x0 = offset
    yy, xx = np.mgrid[: big.shape[0], : big.shape[1]]
    c = (big.shape[0] - 1) / 2
    inside = (xx - c) ** 2 + (yy - c) ** 2 <= c**2
    region = photo[y0 : y0 + big.shape[0], x0 : x0 + big.shape[1]]
    region[inside] = big[inside]
    cv2.circle(photo, (int(x0 + c), int(y0 + c)), int(c + 6), 0.15, 12)  # the frame rim
    photo += np.random.default_rng(seed).normal(0, 0.01, photo.shape)
    return (np.clip(photo, 0, 1) * 255).astype(np.uint8), (x0 + c, y0 + c, c)


@pytest.fixture(scope="module")
def sheet():
    return make_sheet(n_pins=200, frame_mm=600, n_lines=200)


def test_sheet_has_instructions(sheet):
    seq, pins, text = sheet
    assert len(seq) == 201 and "CALIBRATION PATTERN" in text and "-> " in text


@pytest.mark.parametrize("true_op", [0.18, 0.3, 0.45])
def test_fit_recovers_opacity_with_known_circle(sheet, true_op):
    seq, pins, _ = sheet
    photo, circle = _fake_photo(seq, pins, true_op)
    bare = bare_mask(seq, len(pins))
    fit = fit_opacity(seq, len(pins), 600, photo_darkness(photo, circle, bare=bare))
    assert fit.opacity == pytest.approx(true_op, abs=0.015)
    assert fit.thread_mm == pytest.approx(fit.opacity * 600 / 600)


def test_fit_with_detected_frame(sheet):
    seq, pins, _ = sheet
    photo, _ = _fake_photo(seq, pins, 0.3)
    dark = photo_darkness(photo, bare=bare_mask(seq, len(pins)))  # Hough finds the frame
    fit = fit_opacity(seq, len(pins), 600, dark)
    assert fit.opacity == pytest.approx(0.3, abs=0.06)


def test_cli_sheet_then_fit(tmp_path, sheet):
    out = tmp_path / "cal"
    main(
        [
            "calibrate",
            "sheet",
            "--pins",
            "200",
            "--frame-mm",
            "600",
            "--lines",
            "200",
            "--out",
            str(out),
        ]
    )
    assert (out / "instructions.txt").is_file() and (out / "pattern_preview.png").is_file()
    import json

    doc = json.loads((out / "sequence.json").read_text())
    seq = doc["sequence"]
    pins = np.asarray(doc["pins"])
    photo, (cx, cy, r) = _fake_photo(seq, pins, 0.3)
    cv2.imwrite(str(tmp_path / "photo.png"), photo)
    main(
        [
            "calibrate",
            "fit",
            str(tmp_path / "photo.png"),
            "--sheet",
            str(out),
            "--circle",
            f"{cx},{cy},{r}",
        ]
    )
    assert (out / "photo_darkness.png").is_file()
