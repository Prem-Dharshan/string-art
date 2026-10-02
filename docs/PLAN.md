# Plan: OpenCV-based Computational String Art

Goal: improve on existing computational string-art methods so the pipeline is
**robust** (works on any photo with no hand-tuning), **fast** (seconds, not minutes),
and **feasible** (pure Python + OpenCV + numba, runs on a laptop, produces a pin
sequence someone can actually build).

---

## 1. Prior work (what we're improving on)

| Source | Method | Weak spots |
|---|---|---|
| **Vrellis / LessWrong "Computational Thread Art"** (refs.md #2) | Greedy. From the current pin, sample random candidate lines and pick the one through the darkest pixels on average. Subtract a constant darkness along the line, then repeat. Penalty `Σ max(p·ω⁺,0) − D·Σ min(p·ω⁻,0) / N` uses ± importance weights. Color: Floyd–Steinberg dithering into one mono image per palette color, then interleaved layering. Speed came from numpy and precomputed line coordinates (~10 s). | The darkness per line is an arbitrary constant with no link to real thread width. Error is measured pixel-by-pixel, while the eye sees a blurred average. Line count is fixed by hand. Lines are aliased (Bresenham). Importance masks are drawn manually. Color is chosen by RGB Euclidean distance. |
| **Birsak et al. 2018, "String Art: Towards Computational Fabrication of String Images"** (CGF/Eurographics) | Binary optimization on a **super-sampled** canvas, compared against a **down-sampled** target, which models viewing distance. Greedy add and remove of strings. Importance map. | Slow (minutes to hours). Mostly grayscale. Heavy implementation. |
| **IEEE 10844087, "Automated String Art Creation"** (refs.md #1) | Hardware: CNC, 3D printing, Arduino tensioning, plus a CNN (transposed conv) for image processing. | Mostly about fabrication. The CNN adds training cost but no clear quality gain. We only borrow fabrication ideas from it. |

**Our position:** keep the speed and simplicity of greedy (LessWrong) but adopt the
physically meaningful objective from Birsak, and automate everything that was manual
using OpenCV (preprocessing, face and saliency importance, color quantization).

---

## 2. Proposed improvements

Each improvement gets an ablation switch so we can measure it (§5).

### I1. Physical thread model (replaces the "subtract constant darkness" rule)
- Per-pixel opacity of one thread: `α = clip(thread_width_mm / pixel_size_mm) × AA_coverage`.
  Example: 0.25 mm thread, 500 mm frame, 500 px canvas → α ≈ 0.25.
- Overlapping threads compose multiplicatively: `d_new = d + α·(1 − d)`. Threads stack
  like real occluding fibres and darkness never goes above 1.
- The low-resolution canvas itself is the "viewing distance" (Birsak's insight), with no
  super-sampled canvas, so it stays cheap.

### I2. Residual-based objective with automatic stopping
- Minimize the weighted error `E = Σ_p w_p (t_p − d_p)²`, where t is target darkness and
  d is rendered darkness.
- Score each candidate line by the drop in error `ΔE`, computed in O(line length) from the
  per-pixel closed form.
- **Stop automatically** when the best `ΔE ≤ ε`, so the line count adapts to the image.
  This fixes the hand-tuned "N lines" parameter.
- Constraints: minimum pin gap (no near-tangent chords), no immediate back-and-forth, and
  a cap on how often each chord can repeat.

### I3. Anti-aliased lines plus a numba inner loop (speed)
- Rasterize Xiaolin-Wu anti-aliased lines on the fly in a `@njit(cache=True)` kernel.
  This avoids a ~150 MB precomputed line cache: 256 pins give 32,640 chords of about
  760 AA pixels each.
- Expected cost per step: about 255 candidates × 760 px ≈ 2×10⁵ flops. For 4,000 lines
  that is about 10⁹ flops, roughly 1–3 s.
- Optional: score all candidates instead of a random subset, since it's fast enough.
- `cv2.line(..., LINE_AA)` serves as the reference renderer for the final image and for
  testing the numba rasterizer.

### I4. OpenCV preprocessing (robustness)
- Auto-orient and resize, then crop to the frame (circle or rectangle). When a face is
  found, the crop is centred on it (YuNet).
- **CLAHE** on the L channel (local contrast) instead of global histogram equalization.
- **Edge-preserving smoothing** (`cv2.bilateralFilter` or `ximgproc.guidedFilter`)
  instead of plain Gaussian, which keeps eye and lip edges sharp.
- Optional background suppression: `cv2.grabCut` seeded from the face box or saliency
  mask, which fades the background toward white so no threads are wasted there.
- Gamma and level normalization so very dark or very bright photos behave the same.

### I5. Automatic importance maps (replaces manual masks)
`W = normalize(a·W_face + b·W_edges + c·W_saliency + floor)`
- **W_face:** `cv2.FaceDetectorYN` (YuNet ONNX, ~300 KB) finds the face box and 5 points.
  `cv2.face.createFacemarkLBF` adds 68 landmarks, which become polygons for the eyes,
  brows, nose and mouth. Fill them with `cv2.fillPoly` and feather with `GaussianBlur`.
- **Negative weights (ω⁻):** eye whites and specular highlights get ω⁻ so threads avoid
  them, which automates the LessWrong trick.
- **W_edges:** Scharr gradient magnitude, smoothed.
- **W_saliency:** `cv2.saliency.StaticSaliencySpectralResidual`. This is the fallback
  when no face is detected, so it works for pets, objects and landscapes.

### I6. Refinement pass
- After greedy finishes, do a few sweeps that try **removing** or **swapping** lines when
  that lowers E (Birsak-style add/remove, but local and fast).
- Stretch goal: optimize the edge set without the continuous-path constraint, then repair
  it into one thread with an Eulerian path, routing extra links around the rim (outside
  the image).

### I7. Color
- Build the palette with `cv2.kmeans` in **CIELAB** (perceptual) rather than RGB, then
  snap it to the thread colors we actually own.
- Floyd–Steinberg dithering in Lab gives one target mask per color.
- Run greedy per color (I1 to I3) and interleave layers light→dark, so dark threads end
  on top.
- Stretch goal: joint color greedy, where each step picks the (color, line) pair with the
  largest ΔE in Lab.

---

## 2b. Visualizer: watch the image form thread by thread

A small, solver-independent tool that replays a pin sequence one line at a time. It is
useful for debugging (bad lines are obvious immediately), for the report figures, and for
demos.

**Input:** a pin sequence (`[p0, p1, p2, ...]`), the pin coordinates, the canvas size, and
optionally the target image. It works on the output of *any* solver (baseline or improved),
so both can be compared side by side.

**View (one matplotlib window, three panels):**

```
┌───────────────┬───────────────┬───────────────┐
│ Target        │ Threads so far│ Error map     │
│ (preprocessed)│ + pins as dots│ |target−render│
│               │ newest line   │ (optional)    │
│               │ in red        │               │
└───────────────┴───────────────┴───────────────┘
 line 1250 / 4000   pin 87 → 203   SSIM(blur σ=2) 0.61   [▶/❚❚] [step] [speed]
```

- Each frame adds the next line to the canvas using the **same renderer as the final
  output** (`cv2.line` with LINE_AA plus the physical thread model). The animation is
  therefore exactly what the solver produced, not a re-drawing.
- The newest line is highlighted in red for one frame, and the current pin is marked.
- The status bar shows line index, from→to pins, and a running SSIM/PSNR every k lines
  (computing it every frame is too expensive).
- Controls: play/pause (space), single-step (→), speed (lines per frame: 1, 10, 50), and
  jump to the end.

**Modes:**
1. **Live / interactive:** `stringart viz out/seq.json --target out/target.png`.
   This uses `matplotlib.animation.FuncAnimation`, drawing incrementally with `blit=True`,
   so updating one canvas image per frame stays smooth up to about 4k lines.
2. **Export:** `--save build.mp4` / `--save build.gif` writes frames with
   `cv2.VideoWriter`, or Pillow for GIF. There is a configurable "lines per frame" so a
   4,000-line piece becomes a 10–20 s clip, and the last frame is held for 2 s.
3. **Snapshot grid:** `--grid 100,500,1000,2000,4000` saves a single figure showing the
   image at those line counts. This is the report figure for "how quality grows with
   lines".

**Implementation notes:**
- `viz.py` holds about 150 lines and depends only on `render.py` (draw one line onto a
  canvas) and `metrics.py`.
- Rendering is incremental, one line per step onto a float canvas. There is no redraw from
  scratch, so frame cost is O(line length).
- Color sequences use the same replay, iterating `(color, from, to)` triples in layer
  order.
- Done when it replays a 4k-line sequence smoothly, exports an mp4 and a GIF, and the
  final frame matches the solver's final render pixel-for-pixel (covered by a test).

---

## 3. Architecture (`src/stringart/`)

```
io.py          load / save images, sequences (CSV, JSON), configs
geometry.py    pin layouts: circle, rectangle, hook-sides (2 pts per hook)
preprocess.py  crop, CLAHE, bilateral/guided filter, grabcut, gamma      (I4)
importance.py  YuNet + LBF face masks, Scharr edges, saliency, combine   (I5)
raster.py      numba Wu-AA rasterizer; cv2 reference renderer            (I3)
solver/
  baseline.py  LessWrong-style greedy (for comparison)
  greedy.py    physical-model ΔE greedy + constraints + auto-stop        (I1, I2)
  refine.py    remove/swap passes                                        (I6)
color.py       Lab k-means palette, Lab dithering, multi-color scheduling (I7)
metrics.py     PSNR, SSIM (raw + Gaussian-blurred σ ∈ {1,2,4}), timing
render.py      incremental line drawing (shared by solver output + visualizer), SVG export
viz.py         thread-by-thread visualizer: live player, mp4/GIF export, snapshot grid (§2b)
cli.py         `stringart run img.jpg --pins 256 --color 3 ...`, `stringart viz seq.json ...`
```
Tests: `tests/` (pytest) cover the rasterizer against `cv2.line`, the ΔE closed form
against brute force, deterministic seeds, and edge cases (tiny image, blank image, no
face detected).

Models go in `models/` (gitignored): `face_detection_yunet_2023mar.onnx` and
`lbfmodel.yaml`, fetched with `scripts/fetch_models.py`.

---

## 4. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Env and project setup | ✅ uv project, Python 3.12 venv, OpenCV 5.0 contrib, numba verified |
| M1 | Baseline reproduced, plus visualizer | ✅ LessWrong-style greedy (all candidates scored per step via `reduceat`, ~7 s for 256 pins / 3k lines / 600 px); metrics logged; `stringart viz` live player, mp4/GIF export, snapshot grid. Observed: no stopping rule, so extra lines over-darken (motivates I2) |
| M2 | Core solver (I1–I3) | ✅ mostly met. `experiments/compare_solvers.py`, 4 images. Greedy (pixel objective) beats a *tuned* baseline at equal line count on SSIM σ=2 and σ=4 for 4/4 images (+0.005–0.02). It beats the untuned 3k-line baseline by +0.04–0.10 SSIM σ=2 and +1.5–3 dB PSNR σ=2. It picks its own line count (2.4k–3.6k). Time: 3–7 s on a loaded machine, so the < 5 s / 4k-line target is borderline. The blur objective (σ=1) is best on blurred metrics for 4/4 images but takes ~2–4× longer, so it stays an option. |
| M3 | Preprocessing and importance (I4, I5) | ✅ partly; the 20-image set is still to come (only 4 sample images, 2 with faces). From `experiments/ablation_m3.py`, metrics vs the plain photo:<br>• **Face crop + preprocessing:** SSIM σ=2 +0.046 / +0.066 and PSNR +0.8 / +1.8 dB on the 2 face images; neutral on the 2 non-face ones (+0.003 / −0.000).<br>• **Importance map:** face-region SSIM σ=2 +0.045 / +0.088, at −0.02 global SSIM. Without a face it didn't help (coffee −0.03), so it is **auto** (on only when a face is found).<br>• **Level stretch:** hurt the well-exposed cat (−0.036), so it is **auto** (only when p99−p1 < 100). Bilateral vs Gaussian smoothing made no measurable difference.<br>• **GrabCut background fade:** an aesthetic option only. It lowers fidelity and erased the cameraman's dark coat. Saliency-seeded GrabCut without a face failed, so the fade needs a face.<br>• **Speed:** greedy solve 2–4 s for 2.4k–3.9k lines on an idle machine, which meets the M2 < 5 s target. |
| M4 | Refinement (I6) | ✅ delete/reroute/insert local search keeps one thread, E never increases; +0.010 SSIM σ2 (23/30), face +0.015 (21/22) with 2 sweeps; parallel (prange) scoring: greedy 4× faster, results bit-identical. [M4 doc](milestones/M4-refinement.md) |
| M5 | Colour (I7) | ✅ joint colour greedy with per-colour threads and spool-run limit; vs LessWrong-style dither baseline: ΔE2000 −3.8 and luminance SSIM +0.16 on 12/12 images; Lab vs RGB palette inconclusive. [M5 doc](milestones/M5-colour.md) |
| M7 | Demo app | ✅ Gradio app: photo → string art, build-up animation, build sheet; defaults = 700 mm / 300 pins / 0.25 mm thread. [M7 doc](milestones/M7-demo.md) |
| M8 | Hardening | ✅ physical calibration tool (`stringart calibrate`), gamma-aware exposure validated on a rule-chosen held-out set, automatic palette choice, ruff clean (WSL fallback). [M8 doc](milestones/M8-hardening.md) |
| M6 | Evaluation and report | ✅ 30-image openly licensed dataset (`data/dataset.json`). Solver beats the baseline at equal lines on 29/30 images; full method raises face SSIM on 22/22 face images; auto stretch recovers synthetic exposure faults; pins/opacity sweeps; SSIM-vs-lines curves; fabrication `instructions.txt`. Details: [docs/milestones/M6-evaluation.md](milestones/M6-evaluation.md) |

Order taken: M1 → M2 → M3 → M6 (minimum viable evaluation), then M4, M5, M7 (demo) and the report.

---

## 5. Evaluation protocol

- **Data:** about 20 images. That covers about 12 faces (varied lighting, skin tone and
  pose), 4 pets/objects, and 4 hard cases (low contrast, backlit, busy background). Use
  our own photos or permissively licensed ones.
- **Metrics:** PSNR and SSIM on (a) the direct render and (b) after Gaussian blur at
  σ = 1, 2, 4 px, simulating viewing distance. Also report ROI-SSIM on face landmark
  regions, wall-clock time, line count, and thread length in metres.
- **Ablations:** baseline vs. +I1 vs. +I2 vs. +I3 (AA) vs. +I4 vs. +I5 vs. +I6; importance
  on/off; pins ∈ {128, 200, 256, 300}; lines ∈ {1k…6k} curve.
- **Determinism:** fixed seeds and config dumped with every output.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OpenCV 5.0 contrib API drift (Facemark, saliency) | Wrap each behind a function with a fallback (YuNet 5 points → coarse ellipse masks; saliency → edges only) |
| LBF model is ~54 MB, download can fail | Optional; YuNet-only mode is the default fallback |
| numba JIT warm-up (~seconds) | `cache=True`; report steady-state timing separately |
| Greedy gets stuck in local minima, over-dark regions | Multiplicative darkness model, ω⁻ weights, refinement pass |
| Physical result differs from render | Calibrate α from one small test frame (photograph it, fit thread opacity) |

---

## 7. Environment

- `uv` project, Python **3.12** (pinned in `.python-version`) for wheel compatibility
  with numba and opencv.
- Runtime deps: numpy, opencv-contrib-python (do **not** also install `opencv-python`,
  because the two conflict), scipy, scikit-image, numba, matplotlib, tqdm.
- Dev deps: pytest, ruff.
- Commands: `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run stringart ...`.
