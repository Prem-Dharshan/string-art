# stringart

OpenCV-based computational string art: photo → optimized pin sequence for a single thread.
See [docs/PLAN.md](docs/PLAN.md) for the method and roadmap.

## Setup

```sh
uv sync
```

## Usage

```sh
# One-off: download the OpenCV face models (YuNet 0.2 MB + LBF landmarks 56 MB) into models/
uv run stringart fetch-models

# Solve and write outputs/<image>_greedy/
# (face-centred crop, auto importance map, line count chosen automatically)
uv run stringart run path/to/photo.jpg --pins 256
uv run stringart run sample:astronaut         # scikit-image test images: astronaut, camera, coffee, chelsea
uv run stringart run photo.jpg --thread-mm 0.25 --frame-mm 500   # thread opacity from real sizes
uv run stringart run photo.jpg --solver baseline --legacy-prep  # the LessWrong-style baseline

# Watch the image form thread by thread (space pause, -> step, +/- speed, e end)
uv run stringart viz outputs/camera_greedy

# Export the build-up and a snapshot grid for the report
uv run stringart viz outputs/camera_greedy --save build.mp4 --save build.gif --scale 0.5
uv run stringart viz outputs/camera_greedy --grid 100,500,1500,3000
```

Each run directory has `sequence.json` (pin coordinates and pin order, the fabrication output),
`target.png`, `render.png`, `render.svg`, `importance.png` (when used) and `metrics.json`
(PSNR/SSIM raw and blurred vs. the target and vs. the plain photo, face-region scores, timing,
full config).

## Dataset and experiments

```sh
uv run python experiments/fetch_dataset.py      # 30 openly licensed images -> data/raw/ (see data/ATTRIBUTION.md)
uv run python experiments/compare_solvers.py    # M2: greedy vs baseline (sample images)
uv run python experiments/ablation_m3.py        # M3: preprocessing / importance ablation
uv run python experiments/evaluate_dataset.py   # M6: full evaluation -> outputs/experiments/m6/
uv run python experiments/figures_m6.py --docs  # M6: report figures
```

The milestone log with results and findings is in [docs/milestones/](docs/milestones/README.md).

## Development

```sh
uv run pytest
```
