# M3: Face-aware OpenCV preprocessing and automatic importance maps (I4, I5)

**Status:** done on the 4 sample images; the dataset evaluation is part of M6 ·
**Commit:** `da3f38e`

## Goal
Make the target and the weights carry the work, with no per-image tuning. Faces must come out
recognizable.

## What was built

| Module | Content |
|---|---|
| `models.py` | On-demand download of YuNet (0.23 MB ONNX) and LBF (56 MB) into `models/` (gitignored). `stringart fetch-models` |
| `face.py` | `detect_faces`: YuNet box and 5 points (input downscaled to ≤ 800 px), plus LBF 68 landmarks. Missing model → `[]` (graceful fallback) |
| `preprocess.py` | `prepare()`: face-centred crop (side = 1.8 × face height, centre shifted down 0.15 h, clamped to the image) → resize → grayscale → *auto* level stretch → bilateral filter → CLAHE → optional GrabCut background fade → white outside the frame. Also returns `plain` (the unprocessed crop, used as the metric reference), the faces in canvas coordinates, and the crop box. `PreprocessConfig.legacy()` = the M1/M2 chain |
| `importance.py` | `W = floor + (1−floor)·clip(1.0·face + 0.5·edges + 0.25·saliency)`. Face map from the 68 landmarks (oval 0.5; eyes, brows, nose, mouth 1.0, feathered) or the 5-point fallback. Edges = Scharr magnitude at its 99th percentile, with the frame rim excluded. Saliency = spectral residual. `auto_weights`: W only when a face is found |
| `metrics.py` | `roi=` → `ssim_roi_sK`, `psnr_roi_sK` (face oval) |
| CLI | `--crop face/center`, `--face-zoom`, `--importance auto/on/off`, `--importance-floor`, `--background none/fade`, `--legacy-prep`; reports metrics vs target *and* vs the plain photo |

**Why no negative weights?** LessWrong used ω⁻ to keep threads out of eye whites. Our error
is symmetric, so a high weight on a light region already penalizes darkening it. The feature
mask therefore protects eye whites and teeth automatically.

## Results
From `experiments/ablation_m3.py`. Metrics are vs the **plain photo** of each variant's own
crop, because preprocessing changes the target itself. Greedy solver, pixel objective.

| image | variant | faces | lines | time s | SSIM σ2 | SSIM σ4 | PSNR σ2 | face SSIM σ2 | face PSNR σ2 |
|---|---|---|---|---|---|---|---|---|---|
| astronaut | legacy | 0 | 2774 | 3.1 | 0.517 | 0.681 | 16.54 | | |
| astronaut | prep | 1 | 2869 | 3.0 | **0.564** | **0.719** | **17.33** | 0.639 | 17.72 |
| astronaut | prep+imp | 1 | 2876 | 2.9 | 0.544 | 0.706 | 16.62 | **0.683** | **18.54** |
| astronaut | prep+imp+bg | 1 | 2392 | 2.1 | 0.521 | 0.681 | 15.89 | 0.672 | 18.30 |
| camera | legacy | 0 | 2476 | 2.3 | 0.593 | 0.770 | 17.22 | | |
| camera | prep | 1 | 3939 | 4.1 | **0.659** | **0.796** | **19.04** | 0.627 | 19.12 |
| camera | prep+imp | 1 | 3865 | 3.5 | 0.640 | 0.782 | 18.54 | **0.715** | **20.27** |
| camera | prep+imp+bg | 1 | 1358 | 1.2 | 0.402 | 0.544 | 8.24 | 0.634 | 15.07 |
| coffee | legacy | 0 | 3535 | 3.3 | 0.669 | 0.830 | 19.13 | | |
| coffee | prep | 0 | 3563 | 3.2 | **0.673** | **0.833** | **19.18** | | |
| coffee | prep+imp | 0 | 3500 | 3.1 | 0.639 | 0.814 | 18.59 | | |
| chelsea | legacy | 0 | 2448 | 2.4 | **0.672** | **0.870** | 20.98 | | |
| chelsea | prep | 0 | 2463 | 2.4 | 0.671 | 0.869 | 21.07 | | |
| chelsea | prep+imp | 0 | 2480 | 2.5 | 0.660 | 0.866 | **21.44** | | |

(`prep+imp` forces importance *on*, to show its effect. The default `auto` mode would use
uniform weights for coffee and chelsea. `+bg` is a no-op without a face.)

Tuning that led to the defaults, on astronaut with background fade:

| setting | lines | SSIM σ2 | face SSIM σ2 |
|---|---|---|---|
| face_zoom 2.6, floor 0.25 | 1471 | 0.516 | 0.636 |
| floor 0.1 | 1520 | 0.499 | 0.659 |
| floor 0.05, face weight 2 | 1593 | 0.492 | 0.693 |
| zoom 2.0, floor 0.1 | 1920 | 0.501 | 0.667 |
| zoom 1.7, floor 0.1 | 2290 | 0.526 | 0.677 |
| zoom 1.7, floor 0.1, opacity 0.15 | 3073 | 0.564 | 0.719 |

## Findings
- **The face-centred crop is the largest single visible improvement in the project so far.**
  It adds +0.046 / +0.066 SSIM σ2 and +0.8 / +1.8 dB on the face images, and eyes, nose and
  mouth become legible. YuNet also found the cameraman's face in profile.
- **Importance** buys +0.045 / +0.088 face SSIM for about −0.02 global SSIM, a deliberate
  trade. On face-less images the edge/saliency weights did not help (coffee −0.034), so the
  default `auto` mode applies W only when a face is found.
- **The level stretch hurt the well-exposed cat** (−0.036 SSIM, −2.9 dB). Its p1–p99 range
  is 147 (20–167). The stretch is now *auto*, applied only when p99−p1 < 100, i.e. genuinely
  poor exposure. Caveat: fidelity-to-photo metrics penalize *any* tonal change by
  construction, so the stretch's value has to be judged on badly exposed inputs (M6).
- **Bilateral vs Gaussian smoothing:** no measurable difference. Bilateral is kept for edge
  preservation.
- **GrabCut background fade** lowers fidelity everywhere (aesthetic option only). On the
  cameraman it also faded his dark coat (8.2 dB). Seeding GrabCut from saliency on face-less
  images faded most of the subject (coffee and cat dropped to ~800 lines), so the fade now
  requires a face.
- **Thinner thread (lower opacity) clearly helps faces:** at opacity 0.15, SSIM σ2 rose from
  0.526 to 0.564. Opacity is physical (thread width / pixel size), so this is a hardware
  recommendation rather than a default change.
- LBF landmarks are good on eyes, nose and mouth, but the jaw contour is loose. The oval is
  built from the convex hull of jaw and brows, so that error is tolerable.
- **Bug caught by a test:** the frame rim (photo next to white) registered as a strong edge
  and gave the rim importance. Fixed by eroding the mask before the edge map.

> **Revised in M6:** the auto-stretch trigger is now *narrow range (< 115) or no black point
> (p1 > 64) or no white point (p99 < 128)*. The range-only rule missed underexposed and
> washed-out inputs. See [M6](M6-evaluation.md#robustness-to-exposure-faults-vs-the-clean-photo).

## Tests
13 new tests (48 total):
- face and landmark sanity
- face crop centres the face
- missing models degrade to a centre crop
- crop clamping
- the legacy chain is bit-compatible with M2
- a flat image is stable (no NaNs, no rim edge)
- auto-stretch fires only for poor exposure
- `auto_weights` follows faces
- the fade needs a face
- 5-point face map
- importance range and focus
- ROI metrics
- CLI writes `importance.png` and both metric sets

## Reproduce
```sh
uv run stringart fetch-models
uv run python experiments/ablation_m3.py
uv run stringart run sample:astronaut && uv run stringart viz outputs/astronaut_greedy --grid 300,1000,2000,3000
```
