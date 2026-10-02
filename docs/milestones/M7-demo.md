# M7: Interactive demo app

**Status:** done · **Commit:** see the milestone log

## Goal
A demo for the project review: anyone can upload a photo, choose their frame and thread, and
get the string art, the thread-by-thread build-up and a printable build sheet. It needs no
code and runs locally.

## What was built
- `src/stringart/demo.py`: a Gradio app (optional dependency group `demo`).
  - **Inputs:** photo (upload or paste), pins (128–360), frame diameter (mm), thread width
    (mm), number of thread colours (1 = black). Advanced: refinement sweeps, crop
    (face/centre), importance (auto/on/off).
  - **Outputs:** the preprocessed target, the simulated string art, the build-up animation
    (GIF in the page; mp4 to download), `instructions.txt` (the build sheet), and
    `sequence.json`. A summary line gives line count, thread length, faces found, run time and
    a quality score.
  - Thread opacity is derived from the physical sizes (`thread_mm × 600 / frame_mm`), so the
    simulation matches the planned build. The defaults match the team's frame: **700 mm,
    300 pins, 0.25 mm thread** (opacity ≈ 0.21).
- `stringart demo [--port]` launches it on `http://127.0.0.1:7860` (local only).
- `make_art()` is the UI-independent pipeline function behind the button. It is tested
  without Gradio.

## Verification
- Tests run `make_art` headless for black and colour, check bad inputs (missing photo, thread
  wider than a pixel), and build the Gradio UI.
- End-to-end check: launched the server and called its API with `gradio_client`:
  - **f07 portrait, black, 1 refinement sweep:** 3,053 lines, about 1,545 m of thread,
    SSIM σ2 0.638, 9.5 s.
  - **f14 boy, 4 colours (black, tan, brown, grey):** 5,536 lines, about 2,918 m, ΔE2000 12.6,
    luminance SSIM 0.709, 18.4 s. The build sheet lists per-spool thread length and tie-on pin.

## Notes
- The animation is shown as a GIF because browsers generally can't play the `mp4v` codec
  that OpenCV writes. The mp4 is offered as a download.
- The thread lengths (about 1.5 km for 3,000 lines on a 700 mm frame) are realistic for this
  kind of piece, but plan spool purchases with the +10 % margin from the build sheet.

## Run it
```sh
uv sync --extra demo
uv run --extra demo stringart demo
```
