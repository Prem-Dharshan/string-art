# Build guide: from simulation to a real piece

Everything so far runs in simulation. This guide is the checklist for when the tools are
available. Every step has a command; nothing needs to be worked out by hand.

The planned frame is a 700 mm circle with 300 pins and black sewing thread of about 0.25 mm.
Change the numbers below if yours differ.

## 0. What you need
- A round board or ring at least 760 mm across (frame + 30 mm margin each side).
- 300 thin nails or pins, plus a few spare.
- Thread, roughly 1.5–2 km per black portrait on this frame. The exact amount for your design
  is in its `shopping_list.txt` (step 3).
- A printer (A4 is fine), tape, and a hammer or drill.

## 1. Calibrate to your real thread (once per thread type)
The simulation assumes each thread darkens a canvas pixel by a fixed amount. Measure what
your thread really does:

```sh
uv run stringart calibrate sheet --pins 300 --frame-mm 700
#   -> outputs/calibration/instructions.txt : a short ~250-line pattern
```
1. Put in the pins (step 2) and wind that pattern.
2. Photograph it from the front: frame upright, pin 0 at the top, even light, no flash.
3. Run the fit:
   ```sh
   uv run stringart calibrate fit photo.jpg
   #   -> "effective opacity 0.23 -> use --thread-mm 0.27 with --frame-mm 700"
   ```

Unwind the pattern afterwards; the pins stay.

## 2. Mark and place the pins
```sh
uv run stringart kit outputs/<your run> --frame-mm 700
```
This writes into the run folder:

| file | use |
|---|---|
| `frame_template_A4_tiles.pdf` | the 1:1 pin template over ~12 A4 pages. Print at **100% / actual size**, check the 50 mm bar with a ruler, tape the pages so the pins line up, then nail through each dot |
| `frame_template.svg` | the same template as one 760 mm sheet, for a print shop or plotter |
| `pins.csv` | every pin's position in mm and its angle clockwise from the top, if you'd rather mark with a protractor (300 pins = 1.2° apart) |
| `shopping_list.txt` | pins, board size, thread per colour with a 10% margin, number of spools |

Pin 0 is at the top; numbers go clockwise. Number the pins on the board edge in tens
(0, 10, 20, …) so you can find them quickly while winding.

## 3. Make the design with your calibrated thread
```sh
uv run stringart run photo.jpg --pins 300 --frame-mm 700 --thread-mm 0.27   # value from step 1
uv run stringart kit outputs/photo_greedy --frame-mm 700                    # shopping list
```
For colour, add `--colors 4`. The shopping list then gives thread per colour, and the build
sheet says when to switch spools.

Check the result before winding (`render.png`, or `stringart viz outputs/photo_greedy` to watch
it form).

## 4. Wind it with the assistant
```sh
uv run stringart wind outputs/photo_greedy
```
- **The window shows** the picture so far, with the next line highlighted (circle = from pin,
  square = to pin). In big type it shows `line 1,234 / 3,187 · black · pin 87 → pin 203`,
  plus the next few steps.
- **Keys:**
  - space or → when you have wound a line
  - ← to go back
  - type a number then enter to jump to that line
  - q to quit
- **Progress** is saved on every step, so you can stop and resume any time. `--reset`
  starts over.
- **Colour pieces:** it tells you when to switch spools. Leave idle spools hanging at their
  current pin.
- **No laptop at the frame:** `instructions.txt` is the same list in 100-line blocks, with
  the thread used so far, for printing.

## 5. Finish
Tie off at the last pin (the assistant and `instructions.txt` both name it). The finished
piece should look like `render.png` viewed from a few metres away.
