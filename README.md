# stringart

Turn a photo into **string art**: one continuous thread wound between pins on a frame. This
OpenCV-based pipeline:
- crops to the face
- weights important features (eyes, mouth, edges)
- picks every thread line with a physically modelled, self-stopping solver
- refines the path
- handles colour

It outputs a simulated render, a thread-by-thread animation and a numbered build sheet, plus
a build kit (printable pin template, shopping list, winding assistant) for when you make a real
piece.

| Read more | |
|---|---|
| Project report | [docs/report/report.md](docs/report/report.md) |
| Building a real piece | [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) |
| Milestone log (results, findings) | [docs/milestones/](docs/milestones/README.md) |
| Roadmap | [docs/PLAN.md](docs/PLAN.md) |

## Contents
1. [Setup](#1-setup)
2. [Quick start](#2-quick-start)
3. [The web demo](#3-the-web-demo)
4. [Sample images](#4-sample-images)
5. [Command reference and arguments](#5-command-reference-and-arguments)
6. [Output files](#6-output-files)
7. [Recipes](#7-recipes)
8. [Docker](#8-docker)
9. [Experiments and the report](#9-experiments-and-the-report)
10. [Development](#10-development)
11. [Troubleshooting](#11-troubleshooting)
12. [Project layout](#12-project-layout)

---

## 1. Setup

**You need:**
- Windows, macOS or Linux, about 2 GB of free disk space and an internet connection for the
  first setup.
- [uv](https://docs.astral.sh/uv/), the Python package manager. It installs the right Python
  (3.12) by itself.

**Install uv** (once):
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```sh
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Get the code and install everything:**
```sh
git clone https://github.com/Prem-Dharshan/string-art.git
cd string-art
uv sync --extra demo          # creates .venv with Python 3.12 + all dependencies (incl. the web demo)
uv run stringart fetch-models # downloads the face models into models/ (0.2 MB + 56 MB)
```

- `uv sync` without `--extra demo` installs everything except the web demo (Gradio).
- `fetch-models` downloads YuNet (face detection) and LBF (68 facial landmarks). Without them
  the program still works, but it can't crop to faces or weight facial features.
  `--no-lbf` skips the 56 MB landmark model.

**Check that it works:**
```sh
uv run pytest -q              # 87 tests, about 1.5 minutes
```

The first run of anything that solves takes 10–30 s longer than usual, because the solver
compiles itself once (numba) and caches the result. Later runs start straight away.

---

## 2. Quick start

```sh
uv run stringart run sample:woman_smiling     # solve a sample photo
uv run stringart viz outputs/woman_smiling_greedy   # watch it form, thread by thread
```

Open `outputs/woman_smiling_greedy/render.png` to see the simulated piece. Every command below
works the same way on your own photo: replace `sample:woman_smiling` with a path such as
`C:\photos\me.jpg` or `~/photos/me.jpg`.

---

## 3. The web demo

```sh
uv run --extra demo stringart demo
```
Then open **http://127.0.0.1:7860**. It runs only on your computer; nothing is uploaded
anywhere. Stop it with `Ctrl+C`.

1. **Photo:** upload a file (any photo, or one from `data/samples/`) or paste from the
   clipboard.
2. **Frame and thread:**

   | control | default | what it does |
   |---|---|---|
   | Pins | 300 | nails around the frame. More = finer detail, with little gain past ~256 |
   | Frame diameter (mm) | 700 | the real frame size |
   | Thread width (mm) | 0.25 | with the frame size, sets how dark one thread looks: thinner thread or a bigger frame = more lines and more detail |
   | Thread colours | 1 | 1 = black thread only; 2–5 = colour, with the thread colours chosen automatically |

3. **Advanced (optional):**

   | control | default | what it does |
   |---|---|---|
   | Refinement sweeps | 2 | extra passes that move, add or remove pins to improve the path (black only). 0 is fastest |
   | Crop | face | `face` centres the frame on the largest face; `center` uses the middle of the photo |
   | Importance weights | auto | spends more thread on eyes, mouth and edges. `auto` = only when a face is found |

4. Click **Make string art**. It takes about 10–30 s.

You get back:
- the preprocessed target and the simulated string art
- the build-up animation (GIF), with an mp4 to download
- a summary line: number of lines, thread length, faces found, time and a quality score
- the **build kit zip**: winding list, 1:1 pin template (A4 PDF + SVG), pin coordinates,
  shopping list
- `sequence.json`

Other options: `stringart demo --port 8000` uses another port. `--host 0.0.0.0` makes it
reachable from other devices on your network; the Docker setup uses this.

---

## 4. Sample images

Eight test photos are in `data/samples/`. Credits and licences are in
[data/samples/ATTRIBUTION.md](data/samples/ATTRIBUTION.md). Use them as `sample:<name>` or by
path:

| name | good for |
|---|---|
| `woman_smiling` | black thread, clear frontal face |
| `elderly_man_bw` | black thread, lots of fine detail (beard, wrinkles) |
| `girl_bw` | black-and-white child portrait |
| `athlete_dark_bg` | face on a dark background |
| `boy_yellow_shirt` | colour: skin tones plus a saturated yellow shirt |
| `child_monk_orange` | colour: orange robe, small face |
| `border_collie` | animal (no human face), colour |
| `lighthouse` | object/scene, red and white stripes, colour |

scikit-image's built-in test images also work: `sample:astronaut`, `sample:camera`,
`sample:coffee` and `sample:chelsea` (a cat).

```sh
# black thread
uv run stringart run sample:woman_smiling
uv run stringart run sample:elderly_man_bw --pins 300

# colour (4 thread spools)
uv run stringart run sample:boy_yellow_shirt --colors 4
uv run stringart run sample:lighthouse --colors 4

# by path, same thing
uv run stringart run data/samples/border_collie.jpg --colors 3

# then look at any result
uv run stringart viz outputs/boy_yellow_shirt_greedy_color
```

To try all samples at once:
```powershell
# PowerShell
Get-ChildItem data/samples/*.jpg | ForEach-Object { uv run stringart run $_.FullName --quiet }
```
```sh
# bash
for f in data/samples/*.jpg; do uv run stringart run "$f" --quiet; done
```

---

## 5. Command reference and arguments

Every command prints its own help with defaults: `uv run stringart <command> --help`.

| command | what it does |
|---|---|
| [`run`](#stringart-run) | solve: photo → pin sequence, render, build sheet |
| [`viz`](#stringart-viz) | replay a result thread by thread; export video/GIF/snapshot grid |
| [`kit`](#stringart-kit) | build kit: 1:1 pin template, pin coordinates, shopping list |
| [`wind`](#stringart-wind) | winding assistant for a real piece, one line at a time |
| [`calibrate`](#stringart-calibrate) | measure your real thread's darkness from one photo |
| [`demo`](#stringart-demo) | the web demo (section 3) |
| [`fetch-models`](#stringart-fetch-models) | download the face models |

### `stringart run`
```
uv run stringart run IMAGE [options]
```
`IMAGE` is a photo path or `sample:<name>`.

#### The canvas and thread model
The photo is turned into a **600 × 600 px canvas** (`--size`). Each thread darkens every pixel
it crosses by `opacity`: the fraction of a pixel's width the thread covers. Overlapping
threads darken further but never past black. With a real frame and thread:

```
opacity = thread width (mm) × canvas size (px) / frame size (mm)
        = 0.25 × 600 / 700 ≈ 0.21     (0.25 mm thread on a 700 mm frame)
```

Give `--thread-mm` and `--frame-mm` and the opacity is computed for you. Lower opacity (thinner
thread or bigger frame) means more lines and finer detail.

#### Frame and thread

| argument | default | meaning | when to change it |
|---|---|---|---|
| `--frame {circle,rect}` | `circle` | pins on a circle or a square | square frames |
| `--pins N` | 256 | number of pins | your frame's pin count. 200–320 is typical; gains are small past ~256 |
| `--size PX` | 600 | canvas resolution | larger = finer and slower. Keep 600 to match the evaluation |
| `--opacity A` | 0.2 | darkness of one thread on one pixel (see above) | set it directly, or use the next two |
| `--thread-mm W` | – | real thread width, mm | with `--frame-mm`, sets `--opacity` |
| `--frame-mm D` | – | real frame diameter (or side), mm | also makes the build sheet show metres of thread |
| `--min-gap G` | 10 | never connect pins closer than G positions apart (short chords hug the rim and waste thread) | about 4% of `--pins` |

#### Solver

| argument | default | meaning | when to change it |
|---|---|---|---|
| `--solver {greedy,baseline}` | `greedy` | `greedy` = this project's solver (picks the best line each step, stops by itself). `baseline` = the prior LessWrong/Vrellis method, for comparison | comparisons only |
| `--lines N` | – | greedy: a *cap* (default 8,000; it usually stops earlier by itself). Baseline: the exact count (default 3,000). Colour: cap 12,000 | to force fewer lines |
| `--refine K` | 2 | refinement sweeps after greedy. Each sweep tries to delete, move or insert a pin anywhere on the path and keeps a change only if the picture gets closer to the target (black only) | `0` for a fast preview (~1 s); `3` for a little more quality |
| `--objective {pixel,blur}` | `pixel` | what "closer" means: per pixel (fast), or after a blur that simulates viewing distance (about 3× slower, slightly better from afar) | experiments |
| `--blur-sigma S` | 1.5 | blur radius for `--objective blur`, px | – |
| `--max-repeats R` | 2 | how often the same pin pair may be used | rarely |

#### Colour

| argument | default | meaning | when to change it |
|---|---|---|---|
| `--colors N` | 1 | number of thread colours. 1 = black only; ≥ 2 = colour mode (always includes black) | 3–4 for colour photos |
| `--palette LIST` | – | choose the threads yourself, e.g. `black,red,tan,blue` | when you already own the spools. Names: black, white, grey, red, maroon, orange, yellow, tan, brown, green, dark_green, cyan, blue, navy, purple, pink |
| `--palette-method {gamut,kmeans,auto}` | `gamut` | how threads are chosen. `gamut`: the threads that can best reproduce the photo's colours (most accurate colour). `kmeans`: the photo's main colour clusters snapped to threads (slightly more light/dark detail, duller colour). `auto`: preview both (not reliable) | `kmeans` if structure matters more than colour |
| `--min-run L` | 100 | lines in a row before switching to another colour spool | lower = more spool switches (no quality gain measured) |

#### Preprocessing and importance

| argument | default | meaning | when to change it |
|---|---|---|---|
| `--crop {face,center}` | `face` | centre the frame on the largest face (falls back to the centre) | `center` for scenes, or to keep the original framing |
| `--face-zoom Z` | 1.8 | crop size = Z × face height. Smaller = face fills more of the frame | 1.5 tighter, 2.5 head and shoulders |
| `--importance {auto,on,off}` | `auto` | spend more thread on eyes, brows, nose, mouth, edges and salient areas. `auto` = only when a face is found | `on` to use edges/saliency on non-face images |
| `--importance-floor F` | 0.1 | weight of unimportant areas (0–1). Lower = background matters less | 0.05 for even more face detail |
| `--background {none,fade}` | `none` | `fade` lightens the background (GrabCut), so fewer threads go there. Faces only | for a cleaner, more graphic look |
| `--clahe C` | 2.0 | local contrast boost (CLAHE clip limit); 0 = off | 0 for already contrasty photos |
| `--legacy-prep` | off | use the old preprocessing (centre crop, CLAHE, Gaussian blur) instead | comparisons only |
| `--blur S` | 1.0 | Gaussian blur for `--legacy-prep` | – |

Badly exposed photos (too dark, flat or washed out) are detected and corrected automatically
(level stretch + gamma). Well-exposed photos are left alone.

#### Baseline-only

| argument | default | meaning |
|---|---|---|
| `--line-strength S` | 0.1 | darkness removed from the target per line |
| `--candidates N` | all | score only N random candidate lines per step (as in the original method) |
| `--darkness-penalty D` | 0.0 | penalty for over-darkening already dark pixels |
| `--seed N` | 0 | random seed for `--candidates` |

#### Output and misc

| argument | default | meaning |
|---|---|---|
| `--out DIR` | `outputs/<image>_<solver>` (colour: `…_color`) | where to write the results |
| `--viz` | off | open the visualizer when done |
| `--quiet` | off | less console output |

**Examples:**
```sh
# your photo, for a 700 mm frame with 300 pins and 0.25 mm thread
uv run stringart run photo.jpg --pins 300 --frame-mm 700 --thread-mm 0.25

# fast preview, then the real thing
uv run stringart run photo.jpg --refine 0 --out outputs/preview
uv run stringart run photo.jpg --refine 3 --out outputs/final

# colour with the threads you own
uv run stringart run photo.jpg --palette black,tan,brown,red --frame-mm 700 --thread-mm 0.25

# square frame, 200 pins, no face crop
uv run stringart run scene.jpg --frame rect --pins 200 --crop center
```

### `stringart viz`
Replays a result thread by thread through the same renderer as the final image.
```
uv run stringart viz RUN_DIR_or_sequence.json [options]
```

| argument | default | meaning |
|---|---|---|
| `--save FILE` | – | export `.mp4` or `.gif` (repeatable: `--save a.mp4 --save a.gif`) |
| `--grid 100,500,1500,3000` | – | save `grid.png` showing the piece at those line counts |
| `--show` | off | also open the live player when exporting |
| `--step N` | auto | lines drawn per frame |
| `--fps N` | 30 | export frame rate (GIFs are capped at 10) |
| `--duration S` | 15 | export length in seconds (sets `--step` automatically) |
| `--scale X` | 1.0 | export resolution scale (0.5 = half size, smaller files) |
| `--target FILE` | `target.png` | image shown beside the render |
| `--opacity A` | from the run | override the thread darkness |

With no `--save` or `--grid`, the **live player** opens. It shows three panels (target |
threads so far with the newest line highlighted | error map), plus line number, pins and
running SSIM. Keys:

| key | action |
|---|---|
| space | pause / resume |
| → | one line (while paused) |
| + / - | faster / slower |
| e | jump to the end |

```sh
uv run stringart viz outputs/woman_smiling_greedy
uv run stringart viz outputs/woman_smiling_greedy --save build.mp4 --save build.gif --scale 0.5
uv run stringart viz outputs/woman_smiling_greedy --grid 250,1000,2000,3000
```

### `stringart kit`
Everything needed at the workbench for a real piece. Details are in
[docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md).
```
uv run stringart kit RUN_DIR --frame-mm 700 [--spool-m 500]
```

| argument | default | meaning |
|---|---|---|
| `--frame-mm D` | required | real frame diameter (or side) in mm |
| `--spool-m M` | 500 | metres of thread per spool, for the spool count |

It writes `frame_template_A4_tiles.pdf`, `frame_template.svg`, `pins.csv` and
`shopping_list.txt` into the run folder (see [Output files](#6-output-files)).

### `stringart wind`
The winding assistant. It steps through the lines one at a time and saves progress so you can
stop and resume.
```
uv run stringart wind RUN_DIR [--reset]
```

| key | action |
|---|---|
| space / → / n | line done, show the next |
| ← / b | back one line |
| digits, then enter | jump to that line number |
| e | jump to the end |
| q | quit (progress is already saved) |

On screen: the picture so far with the next line highlighted (circle = from pin, square = to
pin), `line 1,234 / 3,187 · black · pin 87 → pin 203`, a spool-switch notice in colour runs,
and the next six steps. `--reset` starts again from line 1.

### `stringart calibrate`
Measures how dark your real thread looks, so the simulation matches reality. It needs a built
frame; see [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md).

```
uv run stringart calibrate sheet [--pins 300] [--frame-mm 700] [--lines 250] [--out outputs/calibration]
uv run stringart calibrate fit PHOTO [--sheet outputs/calibration] [--circle cx,cy,r]
```
- `sheet` writes a short test pattern and its build sheet.
- `fit` takes a front photo of the wound pattern (pin 0 at the top), finds the frame
  automatically (`--circle` overrides it) and prints the `--thread-mm` to use from then on.

### `stringart demo`
```
uv run --extra demo stringart demo [--port 7860] [--host 127.0.0.1]
```

### `stringart fetch-models`
```
uv run stringart fetch-models [--no-lbf]
```
This downloads into `models/`. Set the environment variable `STRINGART_MODELS` to keep the
models elsewhere.

---

## 6. Output files

A run folder (e.g. `outputs/woman_smiling_greedy/`) contains:

| file | what it is |
|---|---|
| `render.png` | the simulated piece (`render.svg` too, for black thread) |
| `target.png` | the photo after cropping and preprocessing: what the solver aimed for |
| `importance.png` | where the solver spent extra effort (when importance was used) |
| `sequence.json` | the result: pin positions and the pin order (colour: every step with its thread) |
| `instructions.txt` | the build sheet: numbered winding list in blocks of 100 lines, with running thread length (metres if `--frame-mm` was given) |
| `metrics.json` | settings, timing, faces found, quality scores (SSIM/PSNR; colour: CIEDE2000 plus lines per colour and spool switches) and, for black runs with `--frame-mm`, thread length |

After `stringart viz --save/--grid`:

| file | what it is |
|---|---|
| `build.mp4`, `build.gif` | the build-up animation |
| `grid.png` | snapshots at chosen line counts |

After `stringart kit` and `stringart wind`:

| file | what it is |
|---|---|
| `frame_template_A4_tiles.pdf` | 1:1 pin template over A4 pages. Print at 100% / actual size and check the 50 mm bar |
| `frame_template.svg` | the same template as one sheet (print shop / plotter) |
| `pins.csv` | each pin's x/y in mm and its angle clockwise from the top |
| `shopping_list.txt` | board size, pins, thread per colour and spools |
| `progress.json` | how far you've wound (from `stringart wind`) |

---

## 7. Recipes

| goal | command |
|---|---|
| quick look at a photo | `uv run stringart run photo.jpg --refine 0 --viz` |
| best quality, black | `uv run stringart run photo.jpg --refine 3` |
| face fills more of the frame | `uv run stringart run photo.jpg --face-zoom 1.5` |
| cleaner background | `uv run stringart run photo.jpg --background fade` |
| colour | `uv run stringart run photo.jpg --colors 4` |
| colour with spools you own | `uv run stringart run photo.jpg --palette black,red,yellow,blue` |
| finer detail (thinner thread or bigger frame) | `uv run stringart run photo.jpg --frame-mm 900 --thread-mm 0.25` |
| compare with the old method | `uv run stringart run photo.jpg --solver baseline --legacy-prep --out outputs/old` |
| video for a presentation | `uv run stringart viz outputs/<run> --save talk.mp4 --duration 20` |
| everything for a real build | `uv run stringart kit outputs/<run> --frame-mm 700`, then `uv run stringart wind outputs/<run>` |

---

## 8. Docker

A Linux image with everything, including the demo. Datasets, face models and results live in
named Docker volumes. The report's numbers were produced this way.

```sh
docker compose build
docker compose run --rm stringart stringart fetch-models
docker compose run --rm stringart stringart run sample:woman_smiling --out outputs/docker_test
docker compose run --rm stringart pytest -q
docker compose up demo          # web demo at http://localhost:7860
```

- Memory per container is capped (`STRINGART_MEM_LIMIT`, default `6g`), and so are solver
  threads (`STRINGART_THREADS`, default 8).
- Copy results out with `docker compose cp stringart:/app/outputs ./outputs_docker`.

---

## 9. Experiments and the report

The 30-image evaluation set (Wikimedia Commons, openly licensed) and a separately chosen
held-out set are downloaded by script, with attribution in `data/ATTRIBUTION.md` and
`data/ATTRIBUTION_heldout.md`.

```sh
uv run python experiments/fetch_dataset.py         # evaluation set -> data/raw/
uv run python experiments/evaluate_dataset.py      # main comparison, robustness, curves, sweeps
uv run python experiments/evaluate_refine.py       # refinement
uv run python experiments/evaluate_color.py        # colour vs the prior method
uv run python experiments/evaluate_exposure.py     # exposure handling, incl. held-out set
uv run python experiments/figures_m6.py --docs     # report figures
```

Or run all of them in Docker in one go:
```sh
docker compose run -d --name sa-exp stringart bash experiments/run_all.sh
docker logs -f sa-exp
```

---

## 10. Development

```sh
uv run pytest -q                              # 87 tests
uv run python tools/ruff.py check             # lint
uv run python tools/ruff.py format --check    # formatting
```

`tools/ruff.py` runs ruff. If Windows blocks the unsigned `ruff.exe`, it runs the same version
through WSL instead.

---

## 11. Troubleshooting

| problem | fix |
|---|---|
| `note: face model not found` | run `uv run stringart fetch-models` |
| first run is slow | the solver compiles once (10–30 s) and caches it; later runs are fast |
| no face found in a portrait | very small, turned or covered faces can be missed. The photo's centre is then used and `--face-zoom` has no effect, so crop the photo closer to the face first |
| result too dark / too light overall | adjust the thread: lower `--opacity` (or a thinner `--thread-mm`) → more, lighter lines |
| demo: `port already in use` | `uv run --extra demo stringart demo --port 7861` |
| demo animation not playing in the browser | the page shows a GIF; the mp4 is offered as a download (browsers often can't play OpenCV's mp4 codec) |
| `ImportError` mentioning gradio | install the demo extra: `uv sync --extra demo` |
| OpenCV import errors after installing other packages | don't also install `opencv-python`; it conflicts with `opencv-contrib-python` |
| PC runs low on memory during long experiments | run them in Docker (memory is capped), see section 8 |
| printed template is the wrong size | print at **100% / actual size**, never "fit to page"; check the 50 mm bar with a ruler |

---

## 12. Project layout

```
src/stringart/
  preprocess.py   crop, exposure (stretch + gamma), smoothing, CLAHE, background fade
  face.py         YuNet face detection + LBF landmarks
  importance.py   face / edge / saliency importance maps
  raster.py       anti-aliased line rasterizer
  render.py       thread renderer (black)        color.py   colour model, palettes, solver
  solver/         greedy.py (main), refine.py (path refinement), baseline.py (prior method)
  viz.py          thread-by-thread player and exports
  fabrication.py  build sheets                   kit.py     template, pins, shopping list
  wind.py         winding assistant              calibrate.py  thread calibration
  metrics.py      PSNR / SSIM                    demo.py    web demo        cli.py  commands
data/samples/     8 sample photos (+ ATTRIBUTION.md)
experiments/      evaluation scripts and figures
docs/             report, plan, milestone log, build guide
tests/            87 tests
```
