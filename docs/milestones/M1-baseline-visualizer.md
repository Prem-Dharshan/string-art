# M1: Baseline reproduction and thread-by-thread visualizer

**Status:** done · **Commit:** `b8e0c0e`

## Goal
Reproduce the prior-art greedy method (Vrellis / LessWrong) as a comparison baseline, and build
the visualizer that replays the image forming one thread at a time (PLAN §2b).

## What was built

| Module | Content |
|---|---|
| `geometry.py` | Circular and rectangular pin layouts; `pin_distance` (short way round the frame) |
| `raster.py` | `aa_line`: vectorized Xiaolin-Wu anti-aliased line (unit coverage per major-axis step). `line_pixels`: aliased line for the baseline |
| `render.py` | `Canvas`: thread renderer with multiplicative darkness `d ← d + a(1−d)`. `replay` and `render_sequence` share one code path; SVG export |
| `preprocess.py` | Grayscale → centre-square crop → resize (INTER_AREA) → CLAHE → Gaussian; white outside the frame |
| `solver/baseline.py` | Greedy: residual = target darkness. Score = mean weighted residual along the aliased line (`darkness_penalty` D for over-dark pixels). Subtract `line_strength`. Options: random candidate subset, `min_gap`, no immediate back-and-forth |
| `metrics.py` | PSNR and SSIM, raw and after Gaussian blur σ ∈ {0, 1, 2, 4}, inside the frame mask |
| `viz.py` | Live player, mp4/GIF export, snapshot grid |
| `cli.py` | `stringart run` / `stringart viz` |

**Speed trick (baseline):** for each pin, the flat pixel indices of all candidate lines are
concatenated once and cached. One `np.add.reduceat` then scores every candidate in a step,
instead of a Python loop over about 245 lines.

### Visualizer
- **Live player:** three panels (target | threads so far with the newest line in red and pins
  marked | error map) and a status bar (line k/N, from→to pin, SSIM/PSNR at σ=2 every
  200 lines). Keys: space = pause/resume, → = step, +/− = speed, e = jump to end.
- **Export:** `--save x.mp4` (cv2.VideoWriter, 30 fps) or `x.gif` (Pillow, capped at 10 fps).
  Lines per frame default to a ~15 s clip, and the last frame is held for 2 s.
- **Snapshot grid:** `--grid 100,500,1500,3000` gives one figure with SSIM per panel.
- **Guarantee:** the visualizer draws through the same `Canvas` as the final render. A test
  asserts the last frame equals the final render pixel for pixel.

## Results
Camera sample: 256 pins, 3,000 lines, line_strength 0.1, opacity 0.2 → **~7 s**,
SSIM σ=2 0.611, PSNR σ=2 17.1 dB.

## Findings
- **The baseline has no stopping rule, so too many lines over-darken the image.** On a 120 px
  test canvas, 400 lines scored lower than 40. This motivated the automatic stop (I2).
- Its result depends strongly on `line_strength` matching the rendering opacity, an extra
  hand-tuned parameter.
- Faces come out murky. On `astronaut`, the face is small in the frame and most threads go
  to the background (motivates M3).
- Exported GIFs are large (8.6 MB at half scale, 10 fps), so prefer mp4 for sharing.

## Tests
23 tests, including:
- the anti-aliased raster has unit coverage per step and correlates > 0.95 with
  `cv2.line(LINE_AA)`
- solver determinism and constraints
- headless player (frames, pause, step, speed, end)
- mp4/GIF export
- CLI round trip

## Reproduce
```sh
uv run stringart run sample:camera --solver baseline --legacy-prep --lines 3000
uv run stringart viz outputs/camera_baseline --grid 100,500,1500,3000 --save build.mp4
```
