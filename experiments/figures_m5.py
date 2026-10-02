"""M5 report figure: colour gallery (target | LessWrong-style baseline | joint colour greedy).

    uv run python experiments/figures_m5.py [--docs]
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = ["f14_boy_smiling_outdoor", "a03_border_collie", "f15_child_monk_small_face",
       "o01_lighthouse_striped"]
COLS = [("target", "target (preprocessed photo)"), ("lw_baseline", "LessWrong-style baseline"),
        ("joint", "joint colour greedy (ours)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "outputs" / "experiments" / "m5"))
    ap.add_argument("--docs", action="store_true")
    args = ap.parse_args()
    rdir = Path(args.src) / "renders"
    rows = []
    for i in IDS:
        tiles = [cv2.imread(str(rdir / f"{i}__{c}.png")) for c, _ in COLS]
        if any(t is None for t in tiles):
            raise SystemExit(f"missing renders for {i}: run evaluate_color.py first")
        rows.append(np.hstack(tiles))
    grid = cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    header = np.full((34, grid.shape[1], 3), 252, np.uint8)
    w = grid.shape[1] // len(COLS)
    for k, (_, title) in enumerate(COLS):
        cv2.putText(header, title, (k * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (40, 40, 40), 1, cv2.LINE_AA)
    out = Path(args.src) / "figures"
    out.mkdir(exist_ok=True)
    path = out / "m5_color_gallery.png"
    cv2.imwrite(str(path), np.vstack([header, grid]))
    print(f"wrote {path}")
    if args.docs:
        dst = ROOT / "docs" / "milestones" / "figures"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dst / path.name)


if __name__ == "__main__":
    main()
