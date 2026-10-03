# M9: Build-ready tooling

**Status:** done · **Commit:** see the milestone log

## Goal
The project is simulation-only for now: no tools are available yet. A real piece is planned
later, so the software has to carry the user through it when the time comes, with nothing
left to work out by hand.

## What was built

| Command / file | What it gives you |
|---|---|
| `stringart kit <run> --frame-mm 700` (`kit.py`) | `frame_template_A4_tiles.pdf`: the 1:1 pin template over 12 A4 pages (700 mm frame), 20 mm overlap, tile ids and a 50 mm scale bar. Labels go in the corner farthest from the pins so they never cover one. `frame_template.svg`: one 760 mm sheet for a print shop. `pins.csv`: mm coordinates and angles clockwise from the top. `shopping_list.txt`: board size, pins and spacing, thread per colour + 10%, spools |
| `stringart wind <run>` (`wind.py`) | Winding assistant: the picture so far with the next line highlighted (circle = from pin, square = to pin), `line k / N · thread · pin a → pin b` in large type, the next 6 steps, and a "switch to the X spool" note in colour runs. Progress is saved to `progress.json` on every step (resume or `--reset`). Keys: space/→ next, ← back, number + enter to jump |
| Demo | the download is now a **build kit zip**: winding list, sequence, template (PDF + SVG), pins, shopping list |
| [docs/BUILD_GUIDE.md](../BUILD_GUIDE.md) | the end-to-end checklist: calibrate → mark pins → design with the calibrated thread → wind → finish |

## Verification
- **Example run:** f07, 300 pins, 700 mm, 0.25 mm thread. 3,187 lines, 1,566 m of thread
  (buy 1,722 m = 4 × 500 m spools), pins 7.3 mm apart.
- **Templates checked by rendering them** (SVG via `rsvg-convert`, PDF via `pdftoppm`, in a
  throwaway container). Pin 0 is at the top and numbers run clockwise and read outward; every
  10th is bold; pins near page edges repeat on both neighbouring tiles.
- **A defect found this way and fixed:** the page labels first covered pins 237–240 on tile 1.
- **Tests, 7 new (86 total):**
  - every template pin lies on the 700 mm circle (±0.01 mm)
  - CSV order and angles
  - shopping-list thread equals the sequence's length
  - the kit files and the CLI
  - winding navigation and resume
  - the colour spool-switch notice
  - the assistant window driven by key presses, headless

## Reproduce
```sh
uv run stringart run data/raw/f07_woman_smiling_closeup.jpg --pins 300 --thread-mm 0.25 --frame-mm 700 --out outputs/build_example
uv run stringart kit outputs/build_example --frame-mm 700
uv run stringart wind outputs/build_example
```
