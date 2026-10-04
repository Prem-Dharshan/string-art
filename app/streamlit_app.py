"""Streamlit front end for the string-art pipeline (hosted on Streamlit Community Cloud).

    uv run --extra web streamlit run app/streamlit_app.py

It wraps the same `stringart.demo.make_art` pipeline as the Gradio demo. The package is
imported from ../src, so the hosting service only installs app/requirements.txt.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("STRINGART_MODELS", str(Path(tempfile.gettempdir()) / "stringart-models"))

from stringart.demo import make_art  # noqa: E402
from stringart.preprocess import SAMPLE_DIR  # noqa: E402

DOCS = "https://prem-dharshan.github.io/string-art/"

st.set_page_config(page_title="String art", page_icon="🧵", layout="wide")


@st.cache_resource(show_spinner="Starting up: downloading face models, compiling the solver…")
def warm_up() -> bool:
    """Once per server: face models, and numba compilation so a visitor never waits on it."""
    from stringart.color import ColorConfig, palette_rgb, solve_color
    from stringart.geometry import make_pins
    from stringart.models import fetch
    from stringart.solver.greedy import GreedyConfig, solve_greedy
    from stringart.solver.refine import RefineConfig, refine

    for name in ("yunet", "lbf"):
        fetch(name)
    pins = make_pins("circle", 32, 64)
    t = np.full((64, 64), 0.5)
    g = solve_greedy(t, pins, GreedyConfig(max_lines=20, min_gap=2), progress=False)
    refine(t, pins, g.sequence, 0.2, min_gap=2, cfg=RefineConfig(sweeps=1), progress=False)
    solve_color(
        np.full((64, 64, 3), 0.5),
        pins,
        palette_rgb(["black", "red"]),
        ColorConfig(max_lines=20, min_gap=2, min_run=5),
    )
    return True


def load_rgb(data: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not read that image file")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


warm_up()

st.title("🧵 Computational string art")
st.write(
    "Turn a photo into string art: one continuous thread between pins on a circular frame, "
    "or one thread per colour. You get the simulated piece, the build-up animation and a "
    f"build kit. [How it works]({DOCS})"
)

with st.sidebar:
    st.header("Photo")
    samples = sorted(SAMPLE_DIR.glob("*.jpg"))
    source = st.radio("Source", ["Sample photo", "Upload"], horizontal=True)
    photo_bytes, photo_name = None, None
    if source == "Sample photo" and samples:
        pick = st.selectbox("Sample", [p.stem for p in samples], index=0)
        path = SAMPLE_DIR / f"{pick}.jpg"
        photo_bytes, photo_name = path.read_bytes(), pick
    else:
        up = st.file_uploader("Your photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if up is not None:
            photo_bytes, photo_name = up.getvalue(), Path(up.name).stem
    if photo_bytes:
        st.image(photo_bytes, width="stretch")

    st.header("Frame and thread")
    n_pins = st.slider("Pins", 128, 360, 300, 4, help="nails around the frame")
    frame_mm = st.slider("Frame diameter (mm)", 300, 1000, 700, 10)
    thread_mm = st.slider(
        "Thread width (mm)",
        0.10,
        0.60,
        0.25,
        0.01,
        help="with the frame size this sets how dark one thread looks: thinner thread or a "
        "bigger frame = more lines and more detail",
    )
    n_colors = st.slider("Thread colours", 1, 5, 1, help="1 = black only")
    with st.expander("Advanced"):
        refine_sweeps = st.slider("Refinement sweeps (black only)", 0, 3, 2)
        crop = st.radio("Crop", ["face", "center"], horizontal=True)
        importance = st.radio("Importance weights", ["auto", "on", "off"], horizontal=True)
    go = st.button("Make string art", type="primary", width="stretch", disabled=photo_bytes is None)

if go and photo_bytes:
    bar = st.progress(0.0, text="Starting…")

    def say(frac, desc=""):
        bar.progress(min(max(float(frac), 0.0), 1.0), text=desc)

    try:
        out = make_art(
            load_rgb(photo_bytes),
            n_pins,
            frame_mm,
            thread_mm,
            n_colors,
            refine_sweeps,
            crop,
            importance,
            say,
        )
    except ValueError as e:
        bar.empty()
        st.error(str(e))
    else:
        bar.progress(1.0, text="Done")
        target, render, gif, mp4, kit, seq, md = out
        st.session_state["result"] = {
            "name": photo_name,
            "target": target,
            "render": render,
            "gif": Path(gif).read_bytes(),
            "mp4": Path(mp4).read_bytes(),
            "kit": Path(kit).read_bytes(),
            "seq": Path(seq).read_bytes(),
            "md": md,
        }

res = st.session_state.get("result")
if res is None:
    st.info(
        "Pick a sample or upload a photo in the sidebar, then press **Make string art**. "
        "It takes about 30–90 s on this free server."
    )
else:
    st.markdown(res["md"])
    c1, c2 = st.columns(2)
    c1.image(np.clip(res["target"], 0, 1), caption="Target (after preprocessing)", width="stretch")
    c2.image(np.clip(res["render"], 0, 1), caption="String art (simulated)", width="stretch")
    st.image(res["gif"], caption="Build-up, thread by thread")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Build kit (zip)",
        res["kit"],
        f"{res['name']}_build_kit.zip",
        "application/zip",
        width="stretch",
    )
    d2.download_button(
        "Animation (mp4)", res["mp4"], f"{res['name']}_build.mp4", "video/mp4", width="stretch"
    )
    d3.download_button(
        "Pin sequence (json)",
        res["seq"],
        f"{res['name']}_sequence.json",
        "application/json",
        width="stretch",
    )
    st.caption(
        "The build kit holds the winding list, a 1:1 pin template (A4 pages + SVG), pin "
        "coordinates and a shopping list."
    )

st.caption(
    f"Docs, method and results: [{DOCS}]({DOCS}) · sample photos from Wikimedia "
    f"Commons ([credits]({DOCS}credits/))"
)
