# M0: Environment and project setup

**Status:** done · **Commit:** `c8c701a`

## Goal
A reproducible Python project with the computer-vision stack verified, and a written plan
grounded in the project references.

## What was done
- Reviewed the references (`docs/references.md`):
  - **LessWrong, "Computational Thread Art"**: a greedy line choice, a penalty with ±
    importance weights and a darkness term D, per-colour Floyd–Steinberg dithering, and
    numpy speed-ups (~10 s).
  - **IEEE 10844087, "Automated String Art Creation"**: mostly the fabrication machine (CNC,
    3D printing, Arduino) plus a CNN. It's useful as fabrication context, not as an algorithm
    baseline.
  - Added **Birsak et al. 2018** (CGF) as the academic baseline. It compares a fine-scale
    canvas against a coarse target, which models viewing distance.
- Created a `uv` package project (`src/` layout), with Python **3.12** pinned in
  `.python-version`.
- Wrote `docs/PLAN.md`: improvements I1–I7, architecture, milestones, evaluation protocol,
  risks, and the visualizer spec (§2b).

## Environment

| Item | Version / note |
|---|---|
| uv | 0.11.2 |
| Python | 3.12.13 (pinned; widest wheel support for numba and OpenCV) |
| opencv-contrib-python | 5.0.0. Do **not** also install `opencv-python`; they conflict |
| numpy / scipy / scikit-image | 2.5.3 / 1.18.1 / 0.26.0 |
| numba | 0.68.0 (JIT verified) |
| dev | pytest 9.1, ruff 0.16 |

Verified OpenCV modules: `FaceDetectorYN` (YuNet), `face.createFacemarkLBF`, `saliency`,
`ximgproc`, `createCLAHE`.

## Known issues
- `ruff.exe` is blocked by Windows Smart App Control ("An Application Control policy has
  blocked this file"), so linting can't run on the dev machine. Workarounds: allow the binary,
  or run `uv tool install ruff`.

## Reproduce
```sh
uv sync
uv run python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'FaceDetectorYN'))"
```
