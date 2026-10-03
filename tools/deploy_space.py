"""Package the Gradio app as a Hugging Face Space, and optionally upload it.

    uv run python tools/deploy_space.py                          # stage into .hf_space/ only
    uv run python tools/deploy_space.py --push USER/string-art   # stage, create the Space, upload

`--push` needs a Hugging Face login first: `uvx --from huggingface_hub hf auth login`.
The staged folder (.hf_space/, gitignored) holds the package, the sample photos, app.py,
requirements.txt pinned from uv.lock (OpenCV swapped for its headless build) and the Space card.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".hf_space"
DOCS_URL = "https://prem-dharshan.github.io/string-art/"

APP = '''"""Hugging Face Space entry point for the string-art demo."""

import os
from pathlib import Path

HERE = Path(__file__).parent
os.environ.setdefault("STRINGART_MODELS", str(HERE / "models"))

import numpy as np  # noqa: E402

from stringart.color import ColorConfig, palette_rgb, solve_color  # noqa: E402
from stringart.demo import build_ui  # noqa: E402
from stringart.geometry import make_pins  # noqa: E402
from stringart.models import fetch  # noqa: E402
from stringart.solver.greedy import GreedyConfig, solve_greedy  # noqa: E402
from stringart.solver.refine import RefineConfig, refine  # noqa: E402

for name in ("yunet", "lbf"):  # face models, downloaded once per container
    fetch(name)

# Compile the numba kernels now, so the first visitor doesn't wait for it.
_pins = make_pins("circle", 32, 64)
_t = np.full((64, 64), 0.5)
_g = solve_greedy(_t, _pins, GreedyConfig(max_lines=20, min_gap=2), progress=False)
refine(_t, _pins, _g.sequence, 0.2, min_gap=2, cfg=RefineConfig(sweeps=1), progress=False)
solve_color(np.full((64, 64, 3), 0.5), _pins, palette_rgb(["black", "red"]),
            ColorConfig(max_lines=20, min_gap=2, min_run=5))

FOOTER = (
    "Docs, method and results: [{docs}]({docs}) · sample photos from Wikimedia Commons "
    "(credits: [{docs}credits/]({docs}credits/)). Runs on a shared CPU: a photo takes about "
    "20–60 s."
)
build_ui(samples_dir=HERE / "samples", footer=FOOTER).launch()
'''.replace("{docs}", DOCS_URL)

CARD = f"""---
title: String Art
emoji: 🧵
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: {{gradio}}
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Photo to string art - thread sequence, animation, build kit
---

# Computational string art

Upload a photo (or pick a sample) and get string art: one continuous thread wound between pins
on a circular frame, or one thread per colour. The app gives you the simulated piece, the
build-up animation and a build kit (winding list, 1:1 pin template, shopping list).

Documentation, method and results: {DOCS_URL}

Sample photos are openly licensed images from Wikimedia Commons; see `samples/ATTRIBUTION.md`.
"""


def requirements() -> tuple[str, str]:
    out = subprocess.run(
        [
            "uv",
            "export",
            "--no-hashes",
            "--extra",
            "demo",
            "--no-dev",
            "--no-emit-project",
            "--no-header",
            "--no-annotate",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines, gradio = [], ""
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e")):
            continue
        pkg = line.split(";")[0].strip()
        if pkg.startswith("gradio=="):
            gradio = pkg.split("==")[1]
        if line.startswith("opencv-contrib-python=="):
            line = line.replace("opencv-contrib-python", "opencv-contrib-python-headless", 1)
        lines.append(line)  # keep environment markers: pip evaluates them on the Space
    return "\n".join(sorted(set(lines))) + "\n", gradio


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(
        ROOT / "src" / "stringart",
        STAGE / "stringart",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbi", "*.nbc"),
    )
    (STAGE / "samples").mkdir(parents=True)
    for f in (ROOT / "data" / "samples").glob("*"):
        if f.suffix in (".jpg", ".md"):
            shutil.copy(f, STAGE / "samples" / f.name)
    reqs, gradio = requirements()
    (STAGE / "requirements.txt").write_text(reqs, encoding="utf-8")
    (STAGE / "app.py").write_text(APP, encoding="utf-8")
    (STAGE / "README.md").write_text(CARD.replace("{gradio}", gradio), encoding="utf-8")
    return STAGE


def push(repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy string-art demo",
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="USER/SPACE", help="create/update this Space and upload")
    args = ap.parse_args()
    folder = stage()
    print(f"staged {folder}: {sorted(p.name for p in folder.iterdir())}")
    if args.push:
        print(f"deployed: {push(args.push)}")


if __name__ == "__main__":
    main()
