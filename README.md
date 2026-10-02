# stringart

OpenCV-based computational string art: photo → optimized pin sequence for a single thread.
See [docs/PLAN.md](docs/PLAN.md) for the method and roadmap.

## Setup

```sh
uv sync
```

## Usage

```sh
# Solve (baseline greedy) and write outputs/<image>_baseline/
uv run stringart run path/to/photo.jpg --pins 256 --lines 3000
uv run stringart run sample:camera            # scikit-image test images: astronaut, camera, coffee, chelsea

# Watch the image form thread by thread (space pause, -> step, +/- speed, e end)
uv run stringart viz outputs/camera_baseline

# Export the build-up and a snapshot grid for the report
uv run stringart viz outputs/camera_baseline --save build.mp4 --save build.gif --scale 0.5
uv run stringart viz outputs/camera_baseline --grid 100,500,1500,3000
```

Each run directory has `sequence.json` (pin coordinates and pin order, the fabrication output),
`target.png`, `render.png`, `render.svg` and `metrics.json` (PSNR/SSIM raw and blurred, timing,
full config).

## Development

```sh
uv run pytest
```
