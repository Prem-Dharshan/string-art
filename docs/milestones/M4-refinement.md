# M4: Path refinement and parallel solver

**Status:** done · **Commit:** see the milestone log

## Goal
Recover from greedy's irreversible choices, and from its MSE-optimal auto-stop that ends too
early on detailed images (M6, finding 4). Keep the result one continuous thread and keep the
pipeline fast.

## Method
### Refinement (`solver/refine.py`)
A local search over the pin sequence. Every move is a local edit of the path, so the output
is still one continuous thread:

| move | edit | effect |
|---|---|---|
| delete | a → b → c  ⇒  a → c | one line fewer |
| reroute | a → b → c  ⇒  a → b′ → c | moves pin b |
| insert | a → b  ⇒  a → x → b | one line more, anywhere on the path |

- **Exact removal.** Transmittance `T = 1 − d` is a product of `(1 − a_i)` under the
  multiplicative thread model, so removing a line divides its factor back out:
  `d ← 1 − (1 − d)/(1 − a)`. This needs opacity < 1.
- **Each candidate.** The old lines are removed, then every replacement pin is scored on that
  canvas. The best move is applied exactly and kept only if the true weighted error went down;
  otherwise it is reverted. **E never increases** (tested).
- **Constraints.** Min pin gap, max chord repeats and no back-and-forth are preserved,
  including across the edited neighbourhood.
- **Pass order.** A pass visits every position once. Passes repeat until no move is accepted
  or `sweeps` is reached.

### Parallel scoring (all solvers)
Candidate scoring only reads the canvas, so the per-candidate loops now use numba `prange`.
This covers greedy, the blur objective, refinement and the colour solver. The best candidate
is still chosen serially with the same tie-breaking, so **results are bit-identical** to the
serial version (same line counts and errors, checked).

| | before | after (16 threads) |
|---|---|---|
| greedy, 256 pins, ~3k lines | 3.2–4.7 s | **0.8–1.3 s** |
| refinement, 3 sweeps | 28–48 s | 10–17 s |

## Results
From `experiments/evaluate_refine.py`: all 30 M6 images, full method (D), then refinement
passes one after another. Scoring is the same as M6.

| stage | SSIM σ2 | SSIM σ4 | PSNR σ2 | face SSIM σ2 | lines | time s |
|---|---|---|---|---|---|---|
| greedy (D) | 0.634 | 0.818 | 18.83 | 0.686 | 2778 | 0.9 |
| + 1 sweep | 0.639 | 0.820 | 18.93 | 0.696 | 3133 | 4.1 |
| **+ 2 sweeps (default)** | 0.644 | 0.821 | 18.91 | 0.701 | 3227 | 7.8 |
| + 3 sweeps | **0.645** | 0.821 | 18.90 | **0.705** | 3254 | 11.7 |

| vs greedy | Δ SSIM σ2 | wins | Δ face SSIM σ2 | wins |
|---|---|---|---|---|
| + 1 sweep | +0.006 | 21 / 30 | +0.010 | 19 / 22 |
| + 2 sweeps | +0.010 | 23 / 30 | +0.015 | 21 / 22 |
| + 3 sweeps | +0.011 | 25 / 30 | +0.018 | **22 / 22** |
| + 3 sweeps, by category | faces +0.016 (15/17), hard +0.006 (3/4), animals +0.006 (6/7), objects +0.003 (1/2) | | | |

## Findings
- **Refinement helps most where greedy stopped early.** On the athlete (f05) SSIM σ2 goes
  0.660 → 0.728, and on the elderly man (f16) 0.585 → 0.659. Insert moves add lines where they
  help, not just at the end of the path. Elsewhere the typical gain is +0.005 to +0.015.
- **Faces gain the most** (face SSIM up on 22/22), because the importance-weighted error is
  what refinement minimizes.
- **It isn't uniformly positive under SSIM.** 5/30 images lose SSIM while their weighted error
  still drops. Four of them lose ≤ 0.002 (noise level). One, h02 (underexposed), loses 0.027:
  refinement fits the exposure-corrected target more closely, but the image is scored against
  the clean photo, the usual MSE-vs-SSIM gap.
- **Parallel scoring made the default pipeline (greedy + 2 sweeps) about 8 s**, close to the
  old greedy-only time.
- **Default:** `--refine 2` in the CLI (`--refine 0` turns it off for a 1 s preview).

## Tests
5 new tests:
- E never increases, and refinement's tracked E matches a fresh render (rel 1e-6)
- a deliberately bad path gets fixed
- importance weights are respected
- `insert=False` never adds lines
- opaque thread (opacity 1) is rejected

## Reproduce
```sh
uv run python experiments/evaluate_refine.py   # ~10 min
uv run stringart run sample:astronaut --refine 3
```
