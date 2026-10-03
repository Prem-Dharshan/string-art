import json
import re

import numpy as np
import pytest

from stringart.cli import main
from stringart.fabrication import thread_length_mm
from stringart.kit import MARGIN_MM, frame_template_svg, pins_csv, shopping_list, write_kit
from stringart.wind import WindSession


@pytest.fixture(scope="module")
def gray_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    main(
        [
            "run",
            "sample:camera",
            "--size",
            "128",
            "--pins",
            "64",
            "--lines",
            "120",
            "--min-gap",
            "5",
            "--refine",
            "0",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    return out


@pytest.fixture(scope="module")
def color_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("crun")
    main(
        [
            "run",
            "sample:coffee",
            "--size",
            "128",
            "--pins",
            "64",
            "--min-gap",
            "5",
            "--colors",
            "3",
            "--min-run",
            "10",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    return out


def _doc(run):
    return json.loads((run / "sequence.json").read_text())


def test_template_has_every_pin_on_the_real_circle(gray_run):
    doc = _doc(gray_run)
    svg = frame_template_svg(doc, 700)
    assert 'width="760.0mm"' in svg  # 700 mm frame + 2 x 30 mm margin, 1:1
    dots = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="0.75"', svg)
    assert len(dots) == doc["n_pins"]
    xy = np.array(dots, dtype=float) - (350 + MARGIN_MM)
    assert np.allclose(np.hypot(*xy.T), 350, atol=0.01)  # every pin on the 700 mm circle
    assert ">0</text>" in svg and f">{doc['n_pins'] - 1}</text>" in svg


def test_pins_csv_angles_and_order(gray_run):
    doc = _doc(gray_run)
    rows = pins_csv(doc, 700).strip().splitlines()
    assert rows[0].startswith("pin,x_mm,y_mm,angle") and len(rows) == doc["n_pins"] + 1
    first, second = rows[1].split(","), rows[2].split(",")
    assert float(first[3]) == 0 and float(second[3]) == pytest.approx(360 / doc["n_pins"])
    assert float(first[1]) == pytest.approx(350 + MARGIN_MM, abs=0.01)  # pin 0 at the top


def test_shopping_list_thread_matches_sequence(gray_run, color_run):
    doc = _doc(gray_run)
    pins = np.asarray(doc["pins"])
    metres = thread_length_mm(doc["sequence"], pins, doc["size"], 700) / 1000
    text = shopping_list(doc, 700, spool_m=100)
    assert f"{'black':<10} {metres:7.0f} m wound" in text
    assert f"{int(np.ceil(metres * 1.1 / 100))} spool(s)" in text
    ctext = shopping_list(_doc(color_run), 700)
    for name in _doc(color_run)["palette"]:
        assert name in ctext


def test_write_kit_and_cli(gray_run, capsys):
    files = write_kit(gray_run, 500)
    assert {f.name for f in files} == {
        "frame_template.svg",
        "pins.csv",
        "shopping_list.txt",
        "frame_template_A4_tiles.pdf",
    }
    assert (gray_run / "frame_template_A4_tiles.pdf").read_bytes()[:4] == b"%PDF"
    main(["kit", str(gray_run), "--frame-mm", "500"])
    assert "shopping list" in capsys.readouterr().out.lower()


def test_wind_session_navigation_and_resume(gray_run):
    s = WindSession(gray_run, reset=True)
    n = s.n
    assert s.done == 0 and "line 1 /" in s.headline()
    s.next()
    s.next()
    s.back()
    assert s.done == 1
    s.goto(10)
    assert "line 11 /" in s.headline() and len(s.upcoming()) > 0
    again = WindSession(gray_run)  # progress was saved
    assert again.done == 10
    again.goto(n + 50)
    assert again.finished and "All" in again.headline()
    img = again.image()
    assert img.shape == (128, 128) and img.min() < 1.0
    assert WindSession(gray_run, reset=True).done == 0


def test_wind_session_colour_spool_switch(color_run):
    s = WindSession(color_run, reset=True)
    names = [s.line(k)[0] for k in range(1, s.n + 1)]
    switch = next(k for k in range(2, s.n + 1) if names[k - 1] != names[k - 2])
    s.goto(switch - 1)
    assert "switch to the" in s.headline() and s.image().ndim == 3


def test_wind_window_headless(gray_run, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    from stringart import wind

    WindSession(gray_run, reset=True)
    seen = {}

    def fake_show():
        fig = plt.gcf()

        def key(k):
            fig.canvas.callbacks.process(
                "key_press_event", KeyEvent("key_press_event", fig.canvas, k)
            )

        for k in (" ", "right", "left", "1", "2", "enter"):
            key(k)
        seen["head"] = fig.texts[0].get_text()
        plt.close(fig)

    monkeypatch.setattr(plt, "show", fake_show)
    wind.run(gray_run)
    assert seen["head"].startswith("line 12 /")
    assert WindSession(gray_run).done == 11
