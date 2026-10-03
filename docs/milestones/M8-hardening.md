# M8: Hardening the weak spots

**Status:** done · **Commit:** see the milestone log

## Goal
Fix the five open issues listed at the end of M7:
1. Results were never checked against a real build.
2. Washed-out photos recovered only partly.
3. The colour palette could miss a needed colour.
4. The exposure thresholds were tuned on the test set.
5. Lint (ruff) couldn't run on the dev machine.

## 1. Physical calibration (`calibrate.py`, `stringart calibrate`)
The simulation's one physical parameter is the thread's effective opacity per pixel. It
should be measured, not guessed: real thread isn't perfectly black, has fuzz, and casts
small shadows.

```sh
uv run stringart calibrate sheet --pins 300 --frame-mm 700   # ~250-line pattern + build sheet
#  ... wind it, photograph it from the front (frame upright, pin 0 at the top, even light) ...
uv run stringart calibrate fit photo.jpg                     # prints the --thread-mm to use
```

- **Pattern.** The greedy solver is run on a three-band target (dark, mid, light), so the
  pattern spans several thread densities in about 250 lines.
- **Fit.**
  1. The frame circle is found with `cv2.HoughCircles`; `--circle cx,cy,r` overrides it.
  2. The frame is warped onto the 600 px canvas.
  3. Brightness is normalized by the **median of board areas the pattern never touches**,
     which the sequence tells us exactly.
  4. A grid search plus golden-section refinement finds the opacity that minimizes the error
     between the simulated and photographed darkness after viewing blur.
  5. The result is printed as the `--thread-mm` to use.
- **Validation on synthetic photos** (grey board, dark frame rim, shifted and 1.4× scaled,
  noise σ = 0.01):

| true opacity | fitted, frame given | fitted, frame auto-detected |
|---|---|---|
| 0.12 | 0.121 | 0.122 |
| 0.21 | 0.211 | 0.231 |
| 0.30 | 0.301 | 0.306 |
| 0.45 | 0.451 | 0.438 |

  A first version normalized by the 97th percentile of the frame and over-estimated opacity by
  about 0.014. Switching to the bare-board median removed the bias.
- **Still open:** this is a tool, not a measurement. The project is simulation-only for now;
  the loop closes when the planned 700 mm, 300-pin piece is built (see
  [BUILD_GUIDE](../BUILD_GUIDE.md) and [M9](M9-build-ready.md)).

## 2 and 4. Gamma-aware exposure + a held-out check (`experiments/evaluate_exposure.py`)
- **Gamma.** When the exposure trigger fires, a gamma now follows the level stretch,
  moving the median toward mid-grey (clamped to 0.5–2.0). `PreprocessConfig.gamma="auto"`
  is the default; `"off"` gives stretch only.
- **Held-out set** (`data/heldout.json`, built by `experiments/build_heldout.py`). The
  images were chosen **by a fixed rule, never by looking**: for each Commons category, skip
  the first 60 members (the pool browsed when the M6 set was curated) and anything already
  used, then take the next openly licensed images in API order. Face categories also need the
  same automatic single-face filter. Six synthetic faults were made from three held-out
  faces at **strengths different from the M6 faults**. Result: 12 natural images
  (8 faces, 4 animals) + 6 faults. The children, cat and lighthouse categories gave nothing
  under the rule, and that was left as is.

| set | natural photos flagged | faults flagged |
|---|---|---|
| tuning (M6) | 0 / 27 | 3 / 3 |
| **held-out** | **0 / 12** | **5 / 6** |

SSIM σ2 against the clean photo, mean over faults:

| set | no correction | level stretch | **stretch + gamma** | clean input |
|---|---|---|---|---|
| tuning (M6, 3 faults) | 0.559 | 0.611 | **0.638** | 0.639 |
| held-out (6 faults) | 0.591 | 0.640 | **0.650** | 0.657 |

- **The thresholds generalize.** No false alarms on held-out natural photos. The one missed
  fault (x13, the mildest underexposure) scores 0.683 uncorrected, *above* its own clean
  original (0.656), so it didn't need fixing.
- **Gamma closes the washed-out gap:** fully on the tuning set (0.548 → 0.645) and mostly on
  held-out (0.604 → 0.620 and 0.595 → 0.627, vs 0.670 clean).
- **Trade-off:** gamma slightly lowers underexposed results (h02 0.661 → 0.635, x16 0.672 →
  0.666). It wins on average on both sets, so it stays on. It wasn't re-tuned after seeing the
  held-out numbers, which would have spoiled the held-out set.

## 3. Palette by reachable gamut (`color.fit_palette`)
Threads composited over the white board reach (to first order) the **convex hull of the board
and the thread colours**. `reach_error` measures each image colour's distance to that hull
(accelerated projected gradient on the capped simplex, vectorized: 6,000 pixels in 0.09 s).
`fit_palette` starts from black and greedily adds the thread that most lowers the
importance-weighted mean distance. Unlike snapping k-means centres to the nearest thread, a
saturated region pulls in the thread that can reproduce it.

**Results** (joint colour solver, 4 threads; Docker run):

| palette | M5 set, 12 images: ΔE2000 σ2 ↓ | lum SSIM σ2 ↑ | held-out, 12 images: ΔE2000 σ2 ↓ | lum SSIM σ2 ↑ |
|---|---|---|---|---|
| k-means, snapped (old default) | 12.56 | **0.663** | 12.59 | **0.687** |
| **gamut fit (new default)** | **11.34** | 0.649 | **11.14** | 0.661 |
| auto: 240 px preview of both, by ΔE | 11.75 | 0.660 | not run | |

- **Gamut fit wins on colour on both sets**: lower ΔE on 8/12 (M5) and **11/12 (held-out)**
  images. Against the LessWrong-style colour baseline it gives ΔE **−5.0** and luminance SSIM
  **+0.146**, both on 12/12.
- **Trade-off:** it gives up some light/dark structure (luminance SSIM lower on 9/12 and 11/12).
  Colour mode exists to reproduce colour, so gamut fit is the default;
  `--palette-method kmeans` keeps the old behaviour.
- **The yellow shirt:** the boy's palette now includes yellow (black, orange, yellow, blue)
  and his ΔE drops 12.2 → **10.4**.
- **The preview selector is a weak predictor** and isn't recommended. With the first
  (under-converged, 60-step) gamut solver it picked the better palette on 10/12 images. After
  the solver was fixed (FISTA, converged) the gamut palettes changed, and the 240 px preview
  picks right on only 6/12, because it systematically favours the k-means palette.
- **Honesty note:** the default was chosen from the M5 images and then confirmed on the
  held-out set, which was not used for any colour decision.

## 5. Lint (`tools/ruff.py`)
- **Cause.** `ruff.exe` is unsigned, so Windows Smart App Control may refuse to start it.
- **Helper.** `tools/ruff.py` tries the native binary first. If that is blocked, it downloads
  the official Linux build of the *same version* from PyPI (sha256-checked, into the
  gitignored `.cache/`) and runs it through WSL.
- **On this machine** the native binary started on the retry, and the code base was brought
  to **`ruff check` and `ruff format` clean**:
  - closures in the experiment loops now bind their loop variables
  - `RefineConfig()` / `ImportanceConfig()` are no longer used as call defaults
  - Gradio's progress default carries a justified `noqa`
  - 30 files were reformatted (no behaviour change; all tests pass)

## 6. Docker (`Dockerfile`, `compose.yaml`, `experiments/run_all.sh`)
A long experiment run on the host was stopped once when the PC ran critically low on memory.
Everything now also runs in a Linux container:
- `python:3.12-slim` with `uv` and the locked dependencies (including the demo).
- Named volumes for the M6 set, the held-out set, the face models and outputs.
- A memory cap (default 6 GB) and a numba thread cap (default 8).
- `run_all.sh` runs every experiment behind the report in one detached container.

All 79 tests pass in the container. **All numbers in this document and the report were
regenerated in the container**, so they come from one environment. Linux floating-point and
OpenCV builds can differ slightly from the Windows runs in earlier milestone docs, and timings
there reflect the container's 8 numba threads.

## Reproduce
```sh
uv run python tools/ruff.py check && uv run python tools/ruff.py format --check
uv run python experiments/build_heldout.py
uv run python experiments/fetch_dataset.py --spec data/heldout.json --raw data/raw_heldout --attribution data/ATTRIBUTION_heldout.md
uv run python experiments/evaluate_exposure.py
uv run python experiments/evaluate_color.py
uv run stringart calibrate sheet && uv run stringart calibrate fit photo.jpg
```
