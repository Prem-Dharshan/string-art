# Milestone log

One document per milestone: goal, what was built, design decisions, results, findings
(including what did *not* work) and how to reproduce. The overall roadmap is in
[../PLAN.md](../PLAN.md).

| Milestone | Status | Doc | Commit |
|---|---|---|---|
| M0 Environment and project setup | done | [M0-setup.md](M0-setup.md) | `c8c701a` |
| M1 Baseline and visualizer | done | [M1-baseline-visualizer.md](M1-baseline-visualizer.md) | `b8e0c0e` |
| M2 Improved greedy solver | done | [M2-greedy-solver.md](M2-greedy-solver.md) | `fe395f6` |
| M3 Preprocessing and importance maps | done (4-image eval) | [M3-preprocessing-importance.md](M3-preprocessing-importance.md) | `da3f38e` |
| M6 Dataset, evaluation and fabrication output | done | [M6-evaluation.md](M6-evaluation.md) | see git log |
| M4 Path refinement + parallel solver | done | [M4-refinement.md](M4-refinement.md) | see git log |
| M5 Colour string art | done | [M5-colour.md](M5-colour.md) | see git log |
| M7 Interactive demo app | done | [M7-demo.md](M7-demo.md) | see git log |
| Report | done | [../report/report.md](../report/report.md) | see git log |

**Conventions used in every results table**
- Images are rendered with the same thread model the solver uses (`render.Canvas`).
- `ssim_sK` / `psnr_sK` are computed after blurring both images with a Gaussian of σ = K px,
  which simulates viewing distance. `s0` is the raw render.
- Metrics are computed inside the frame mask only. `roi` variants are restricted to the
  landmark face oval.
- Default setup unless stated: 600 px canvas, circular frame, 256 pins, thread opacity 0.2,
  min pin gap 10.
- Timings were taken on the development laptop (Windows 11, Python 3.12). Wall-clock varies
  ±2× with background load, so compare timings only within one table.
