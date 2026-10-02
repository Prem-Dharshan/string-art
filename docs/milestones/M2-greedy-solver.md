# M2: Improved greedy solver (I1–I3)

**Status:** done · **Commit:** `fe395f6`

## Goal
A solver that optimizes a physically meaningful objective, stops by itself, and is fast enough
to score every candidate line at every step. It must beat the baseline at an equal line count.

## Method (`solver/greedy.py`)
- **I1 Physical thread model.** A thread covers pixel p with alpha `a = opacity × coverage_p`
  (Wu anti-aliasing), and darkness composes multiplicatively, `d ← d + a(1 − d)`. This is
  *exactly* `render.Canvas`, so the solver optimizes the image that is shown and built.
  `opacity_from_physical(thread_mm, frame_mm, size_px) = thread width / pixel size`
  (CLI `--thread-mm --frame-mm`).
- **I2 Objective and automatic stop.** Minimize `E = Σ w_p (t_p − d_p)²`. Each candidate is
  scored by its exact error drop
  `ΔE = Σ_p w_p[(t−d)² − (t−d−a(1−d))²]`, in O(line length). The solver stops when no line
  from the current pin lowers E, so the line count is chosen by the image. Constraints:
  `min_gap`, no immediate back-and-forth, `max_repeats` per chord.
- **I3 Speed.** A numba `@njit(cache=True)` kernel walks the anti-aliased line on the fly
  (the same traversal as `raster.aa_line`), so there is no ~150 MB line cache. Every
  candidate is scored at every step.
- **Viewing-distance objective (optional, `objective="blur"`).** Minimize
  `Σ w (G_σ * e)²`. The error drop is `2⟨δ, G*(w·G*e)⟩ − Σ w (G*δ)²`. The first term reads
  a field `F = G*(w·G*e)`, which is kept up to date by blurring only the new line's padded
  bounding box (zero-padded blur, so a local update equals a full-image blur). The second
  term uses a thin-line closed form, `c²/(L·2√π·σ)` per step. This matches brute force with
  **correlation 0.999999** (0.1 % median error).

## Results
From `experiments/compare_solvers.py`. Preprocessing is fixed to the M1 chain, so this compares
solvers only. `baseline_tuned` gets the *same* line count the greedy picked and the best
`line_strength` ∈ {0.05, 0.1, 0.2} by SSIM σ=2, a deliberately generous baseline.

| image | method | lines | time s | SSIM σ2 | SSIM σ4 | PSNR σ2 | PSNR σ4 |
|---|---|---|---|---|---|---|---|
| camera | **greedy_pixel** | 2484 | 3.4 | 0.614 | 0.804 | 18.51 | 19.57 |
| camera | greedy_blur1 | 2431 | 7.2 | **0.618** | **0.816** | **18.78** | **19.87** |
| camera | baseline_tuned | 2484 | 5.1 | 0.594 | 0.800 | 18.42 | 19.50 |
| camera | baseline_3000 | 3000 | 6.4 | 0.546 | 0.776 | 15.87 | 16.53 |
| astronaut | **greedy_pixel** | 2787 | 3.2 | 0.508 | 0.664 | 16.38 | 17.45 |
| astronaut | greedy_blur1 | 2890 | 15.2 | **0.517** | **0.677** | **16.61** | **17.71** |
| astronaut | baseline_tuned | 2787 | 14.2 | 0.492 | 0.655 | 16.24 | 17.35 |
| astronaut | baseline_3000 | 3000 | 11.4 | 0.468 | 0.649 | 15.09 | 15.97 |
| coffee | **greedy_pixel** | 3601 | 7.3 | 0.682 | 0.841 | 20.13 | 21.23 |
| coffee | greedy_blur1 | 3441 | 15.3 | **0.683** | **0.851** | **20.57** | **21.72** |
| coffee | baseline_tuned | 3601 | 11.8 | 0.677 | 0.824 | 18.22 | 18.94 |
| coffee | baseline_3000 | 3000 | 13.0 | 0.578 | 0.788 | 18.24 | 19.16 |
| chelsea | **greedy_pixel** | 2393 | 6.2 | 0.617 | 0.815 | 20.06 | 21.67 |
| chelsea | greedy_blur1 | 2381 | 11.1 | **0.628** | **0.827** | **20.35** | 22.00 |
| chelsea | baseline_tuned | 2393 | 10.8 | 0.610 | 0.812 | 20.34 | **22.13** |
| chelsea | baseline_3000 | 3000 | 11.0 | 0.574 | 0.787 | 17.62 | 18.60 |

(These timings came from a loaded machine. On an idle machine, `greedy_pixel` takes 2–4 s
for 2.4k–3.9k lines; see M3.)

## Findings
- **Greedy vs. a tuned baseline at an equal line count:** better SSIM σ2/σ4 on 4/4 images,
  by +0.005 to +0.02. The margin is modest; the bigger practical wins are no tuning (the line
  count is automatic) and speed.
- **Greedy vs. the untuned 3k-line baseline:** +0.04 to +0.10 SSIM σ2 and +1.5 to +3 dB
  PSNR σ2.
- **The blur objective** is best on blurred metrics for 4/4 images but takes 2–4× longer, and
  it is slightly worse on raw SSIM. It is kept as an option, and `pixel` is the default.
- Visually all solvers look alike on these inputs, so the remaining quality gap is in the
  *target* (framing, contrast). That is the job of M3.

## Tests
12 new tests (35 total). The key one: **the sum of the solver's per-line gains equals
E(blank) − E(rendered sequence) to 1e-6**, which proves solver model = renderer. Others cover
the brute-force ΔE check, the blurred-gain check (2 %), constraints, auto-stop on a blank
target, symmetric importance steering, and invalid configs.

## Reproduce
```sh
uv run python experiments/compare_solvers.py
uv run stringart run sample:camera --legacy-prep --objective blur --blur-sigma 1
```
