# stringart

OpenCV-based computational string art: photo → optimized pin sequence for a single thread.
The project report is [docs/report/report.md](docs/report/report.md); the roadmap is
[docs/PLAN.md](docs/PLAN.md) and the per-milestone log is [docs/milestones/](docs/milestones/README.md).

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
uv run stringart run photo.jpg --colors 4 --pins 300 --frame-mm 700  # colour, 4 thread spools
uv run stringart run photo.jpg --palette black,tan,brown,red         # choose the threads yourself

# Calibrate to your real thread: wind a ~250-line test pattern, photograph it, fit
uv run stringart calibrate sheet --pins 300 --frame-mm 700
uv run stringart calibrate fit photo.jpg       # prints the --thread-mm to use

# Interactive demo (browser UI, runs locally)
uv run --extra demo stringart demo

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
uv run python experiments/evaluate_refine.py    # M4: refinement sweeps on all 30 images
uv run python experiments/evaluate_color.py     # M5: colour vs LessWrong-style baseline
uv run python experiments/evaluate_palette.py   # M8: automatic palette choice
uv run python experiments/build_heldout.py      # M8: rule-chosen held-out set (then fetch_dataset.py --spec data/heldout.json ...)
uv run python experiments/evaluate_exposure.py  # M8: exposure handling, tuning vs held-out
uv run python experiments/figures_report.py     # report figures 1-2
```

The milestone log with results and findings is in [docs/milestones/](docs/milestones/README.md).

## Docker

A Linux image with everything (including the demo) and named volumes for datasets, face models
and results. The report's numbers were produced this way.

```sh
docker compose build
docker compose run --rm stringart stringart fetch-models
docker compose run --rm stringart python experiments/fetch_dataset.py
docker compose run --rm stringart python experiments/fetch_dataset.py --spec data/heldout.json --raw data/raw_heldout --attribution /tmp/attr.md
docker compose run --rm stringart pytest -q
docker compose run -d --name sa-exp stringart bash experiments/run_all.sh   # all experiments, detached
docker compose cp stringart:/app/outputs/experiments ./outputs/              # copy results out
docker compose up demo                                                       # http://localhost:7860
```

Memory is capped per container (`STRINGART_MEM_LIMIT`, default 6g) and numba threads by
`STRINGART_THREADS` (default 8).

## Development

```sh
uv run pytest
uv run python tools/ruff.py check          # lint (falls back to WSL if Windows blocks ruff.exe)
uv run python tools/ruff.py format --check
```
