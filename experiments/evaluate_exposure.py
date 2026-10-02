"""Exposure handling on the tuning set (M6) and on the held-out set (rule-chosen, never tuned).

1. Does the auto-exposure trigger fire on every synthetic fault and on no natural photo?
2. For each fault, SSIM sigma=2 against the clean photo with exposure correction off, level
   stretch only, and stretch + gamma (default). Full method, greedy without refinement.

    uv run python experiments/evaluate_exposure.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_dataset import OPACITY, PINS, ROOT, SIZE, _clean_reference, write_csv  # noqa: E402

from stringart.geometry import make_pins  # noqa: E402
from stringart.importance import auto_weights  # noqa: E402
from stringart.metrics import evaluate  # noqa: E402
from stringart.preprocess import PreprocessConfig, load_image, prepare  # noqa: E402
from stringart.render import render_sequence  # noqa: E402
from stringart.solver.greedy import GreedyConfig, solve_greedy  # noqa: E402

SETS = {
    "tuning (M6)": (ROOT / "data" / "dataset.json", ROOT / "data" / "raw"),
    "held-out": (ROOT / "data" / "heldout.json", ROOT / "data" / "raw_heldout"),
}
MODES = {
    "off": {"stretch": "off"},
    "stretch": {"stretch": "auto", "gamma": "off"},
    "stretch+gamma": {"stretch": "auto", "gamma": "auto"},
}


def triggered(img) -> tuple[bool, float, float]:
    """Run the trigger exactly as prepare() does: on the face crop, inside the frame."""
    p = prepare(img, PreprocessConfig(size=SIZE, stretch="off"))
    g = p.plain[p.mask] * 255
    lo, hi = np.percentile(g, (1, 99))
    c = PreprocessConfig()
    return (
        bool(hi - lo < c.stretch_below or lo > c.black_point_above or hi < c.white_point_below),
        float(lo),
        float(hi),
    )


def main() -> None:
    pins = make_pins("circle", PINS, SIZE)
    trig_rows, fault_rows = [], []
    for set_name, (spec_path, raw) in SETS.items():
        items = json.loads(spec_path.read_text(encoding="utf-8"))["images"]
        load = lambda iid, raw=raw: load_image(str(raw / f"{iid}.jpg"))  # noqa: E731
        for it in items:
            fault = "derived_from" in it
            fired, lo, hi = triggered(load(it["id"]))
            trig_rows.append(
                {
                    "set": set_name,
                    "image": it["id"],
                    "fault": fault,
                    "fired": fired,
                    "p1": round(lo),
                    "p99": round(hi),
                }
            )
            if not fault:
                continue
            clean, degraded = load(it["derived_from"]), load(it["id"])
            row = {"set": set_name, "image": it["id"], "transform": it["transform"]}
            for mode, kw in MODES.items():
                prep = prepare(degraded, PreprocessConfig(size=SIZE, **kw))
                w, _ = auto_weights(prep)
                res = solve_greedy(
                    prep.target, pins, GreedyConfig(opacity=OPACITY), weights=w, progress=False
                )
                img = render_sequence(res.sequence, pins, prep.target.shape, OPACITY).image()
                ref = _clean_reference(clean, prep.crop, SIZE, prep.mask)
                row[mode] = evaluate(ref, img, prep.mask, sigmas=(2,))["ssim_s2"]
            prep = prepare(clean, PreprocessConfig(size=SIZE))
            w, _ = auto_weights(prep)
            res = solve_greedy(
                prep.target, pins, GreedyConfig(opacity=OPACITY), weights=w, progress=False
            )
            img = render_sequence(res.sequence, pins, prep.target.shape, OPACITY).image()
            row["clean_input"] = evaluate(prep.plain, img, prep.mask, sigmas=(2,))["ssim_s2"]
            fault_rows.append(row)
            print(row, flush=True)

    out = ROOT / "outputs" / "experiments" / "exposure"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "trigger.csv", trig_rows)
    write_csv(out / "faults.csv", fault_rows)
    md = ["# Exposure handling: tuning set vs held-out set", "", "## Trigger", ""]
    md += ["| set | natural photos flagged | faults flagged |", "|---|---|---|"]
    for s in SETS:
        nat = [r for r in trig_rows if r["set"] == s and not r["fault"]]
        flt = [r for r in trig_rows if r["set"] == s and r["fault"]]
        md.append(
            f"| {s} | {sum(r['fired'] for r in nat)} / {len(nat)} | "
            f"{sum(r['fired'] for r in flt)} / {len(flt)} |"
        )
    md += ["", "## Faults: SSIM sigma=2 vs the clean photo", ""]
    md += [
        "| set | image | fault | off | stretch | stretch+gamma | clean input |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in fault_rows:
        md.append(
            f"| {r['set']} | {r['image']} | {r['transform']} | {r['off']:.3f} | "
            f"{r['stretch']:.3f} | {r['stretch+gamma']:.3f} | {r['clean_input']:.3f} |"
        )
    for s in SETS:
        rr = [r for r in fault_rows if r["set"] == s]
        if rr:
            m = {k: np.mean([r[k] for r in rr]) for k in (*MODES, "clean_input")}
            md.append(
                f"| {s} | **mean** | | {m['off']:.3f} | {m['stretch']:.3f} | "
                f"{m['stretch+gamma']:.3f} | {m['clean_input']:.3f} |"
            )
    text = "\n".join(md) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
