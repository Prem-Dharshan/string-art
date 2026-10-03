"""Build kit: what you need at the workbench to make a run into a real piece.

From a run's `sequence.json` (black or colour) and the real frame size:

* `frame_template.svg`  1:1 printable template: frame outline, centre mark, every pin as a dot
  with its number (every 10th in bold), and a 100 mm bar to check the print scale.
* `pins.csv`            pin number, x / y in mm from the top-left of the template, and (circle)
  the angle clockwise from the top, for marking with a protractor instead of printing.
* `shopping_list.txt`   frame and board size, pin count and spacing, thread per colour with
  a 10% margin, and how many spools that is.
"""

from pathlib import Path

import numpy as np

from .io import load_sequence

MARGIN_MM = 30.0


def _lines_by_thread(doc: dict) -> dict[str, list[tuple[int, int]]]:
    if doc.get("mode") == "color":
        out: dict[str, list[tuple[int, int]]] = {}
        for k, a, b in doc["steps"]:
            out.setdefault(doc["palette"][k], []).append((a, b))
        return out
    seq = doc["sequence"]
    return {"black": list(zip(seq[:-1], seq[1:], strict=True))}


def pin_positions_mm(doc: dict, frame_mm: float) -> np.ndarray:
    """(n, 2) pin centres in mm, origin at the template's top-left corner (incl. margin)."""
    return np.asarray(doc["pins"], dtype=np.float64) * frame_mm / (doc["size"] - 1) + MARGIN_MM


def thread_lengths_m(doc: dict, frame_mm: float) -> dict[str, float]:
    pins = np.asarray(doc["pins"], dtype=np.float64)
    scale = frame_mm / (doc["size"] - 1) / 1000.0
    out = {}
    for name, pairs in _lines_by_thread(doc).items():
        a, b = np.array(pairs).T
        out[name] = float(np.hypot(*(pins[b] - pins[a]).T).sum() * scale)
    return out


def pins_csv(doc: dict, frame_mm: float) -> str:
    pos = pin_positions_mm(doc, frame_mm)
    n = len(pos)
    rows = ["pin,x_mm,y_mm" + (",angle_deg_clockwise_from_top" if doc["frame"] == "circle" else "")]
    for k, (x, y) in enumerate(pos):
        row = f"{k},{x:.2f},{y:.2f}"
        if doc["frame"] == "circle":
            row += f",{360.0 * k / n:.3f}"
        rows.append(row)
    return "\n".join(rows) + "\n"


def frame_template_svg(doc: dict, frame_mm: float, pin_mm: float = 1.5) -> str:
    """1:1 template in millimetres (print at 100%, no 'fit to page')."""
    side = frame_mm + 2 * MARGIN_MM
    pos = pin_positions_mm(doc, frame_mm)
    c = side / 2
    font = max(1.6, min(3.0, np.pi * frame_mm / len(pos) * 0.45))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side:.1f}mm" height="{side:.1f}mm" '
        f'viewBox="0 0 {side:.1f} {side:.1f}" font-family="sans-serif">',
        f'<rect width="{side:.1f}" height="{side:.1f}" fill="white"/>',
    ]
    if doc["frame"] == "circle":
        parts.append(
            f'<circle cx="{c:.2f}" cy="{c:.2f}" r="{frame_mm / 2:.2f}" fill="none" '
            'stroke="#999" stroke-width="0.3"/>'
        )
    else:
        parts.append(
            f'<rect x="{MARGIN_MM}" y="{MARGIN_MM}" width="{frame_mm}" height="{frame_mm}" '
            'fill="none" stroke="#999" stroke-width="0.3"/>'
        )
    parts.append(
        f'<path d="M{c - 6:.2f} {c:.2f}H{c + 6:.2f}M{c:.2f} {c - 6:.2f}V{c + 6:.2f}" '
        'stroke="#444" stroke-width="0.3"/>'
    )
    for k, (x, y) in enumerate(pos):
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{pin_mm / 2:.2f}" fill="black"/>')
        # Number just outside the pin, pointing away from the centre.
        d = np.array([x - c, y - c])
        d = d / max(np.linalg.norm(d), 1e-9)
        tx, ty = x + d[0] * (pin_mm + 2.5), y + d[1] * (pin_mm + 2.5)
        ang = np.degrees(np.arctan2(d[1], d[0]))
        bold = ' font-weight="bold"' if k % 10 == 0 else ""
        size = font * (1.25 if k % 10 == 0 else 1.0)
        parts.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="{size:.2f}"{bold} '
            f'dominant-baseline="middle" transform="rotate({ang:.1f} {tx:.2f} {ty:.2f})">{k}</text>'
        )
    y0 = side - 10
    parts += [
        f'<path d="M10 {y0}h100" stroke="black" stroke-width="0.5"/>',
        f'<path d="M10 {y0 - 2}v4M110 {y0 - 2}v4" stroke="black" stroke-width="0.5"/>',
        f'<text x="10" y="{y0 - 3}" font-size="4">100 mm: measure after printing (print at '
        "100%, no fit-to-page)</text>",
        f'<text x="{c:.1f}" y="8" font-size="4" text-anchor="middle">pin 0 = top; pins numbered '
        "clockwise</text>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


PAPER_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "letter": (215.9, 279.4)}


def template_tiles_pdf(
    doc: dict,
    frame_mm: float,
    path: Path,
    paper: str = "A4",
    overlap_mm: float = 20.0,
    pin_mm: float = 1.5,
) -> int:
    """The 1:1 template split over overlapping pages for a home printer. Returns page count.
    Each page shows its tile id, a 50 mm scale bar and the overlap; pages with nothing on them
    are skipped. Print at 100% (actual size), then tape pages so the pins line up."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    pw, ph = PAPER_MM[paper]
    side = frame_mm + 2 * MARGIN_MM
    pos = pin_positions_mm(doc, frame_mm)
    c = side / 2
    n = len(pos)
    font_pt = max(4.5, min(8.0, np.pi * frame_mm / n * 0.45 / 0.3528))  # mm -> pt
    xs = np.arange(0, side, pw - overlap_mm)
    ys = np.arange(0, side, ph - overlap_mm)
    pages = 0
    with PdfPages(path) as pdf:
        for r, y0 in enumerate(ys):
            for col, x0 in enumerate(xs):
                inside = (
                    (pos[:, 0] >= x0)
                    & (pos[:, 0] <= x0 + pw)
                    & (pos[:, 1] >= y0)
                    & (pos[:, 1] <= y0 + ph)
                )
                has_centre = x0 <= c <= x0 + pw and y0 <= c <= y0 + ph
                if not inside.any() and not has_centre:
                    continue
                fig = Figure(figsize=(pw / 25.4, ph / 25.4))
                ax = fig.add_axes((0, 0, 1, 1))
                ax.set_xlim(x0, x0 + pw)
                ax.set_ylim(y0 + ph, y0)  # y down, like the SVG
                ax.set_aspect("equal")
                ax.set_axis_off()
                if doc["frame"] == "circle":
                    ax.add_patch(Circle((c, c), frame_mm / 2, fill=False, lw=0.4, ec="#999"))
                else:
                    ax.add_patch(
                        Rectangle(
                            (MARGIN_MM, MARGIN_MM),
                            frame_mm,
                            frame_mm,
                            fill=False,
                            lw=0.4,
                            ec="#999",
                        )
                    )
                ax.plot([c - 6, c + 6], [c, c], color="#444", lw=0.5)
                ax.plot([c, c], [c - 6, c + 6], color="#444", lw=0.5)
                for k in np.flatnonzero(inside):
                    x, y = pos[k]
                    ax.add_patch(Circle((x, y), pin_mm / 2, color="black", lw=0))
                    d = np.array([x - c, y - c])
                    d = d / max(np.linalg.norm(d), 1e-9)
                    tx, ty = x + d[0] * (pin_mm + 2.5), y + d[1] * (pin_mm + 2.5)
                    ax.text(
                        tx,
                        ty,
                        str(k),
                        fontsize=font_pt * (1.25 if k % 10 == 0 else 1),
                        fontweight="bold" if k % 10 == 0 else "normal",
                        rotation=-np.degrees(np.arctan2(d[1], d[0])),
                        rotation_mode="anchor",
                        ha="left",
                        va="center",
                    )
                # Page furniture (tile id, overlap note, 50 mm bar) goes in the 80 x 22 mm
                # corner box farthest from every pin, so it never covers one.
                boxes = [
                    (x0 + 10, y0 + 8),
                    (x0 + pw - 90, y0 + 8),
                    (x0 + 10, y0 + ph - 30),
                    (x0 + pw - 90, y0 + ph - 30),
                ]

                def clearance(b):
                    bx = np.clip(pos[:, 0], b[0], b[0] + 80)
                    by = np.clip(pos[:, 1], b[1], b[1] + 22)
                    return np.min(np.hypot(pos[:, 0] - bx, pos[:, 1] - by))

                bx, by = max(boxes, key=clearance)
                ax.text(bx, by + 4, f"tile row {r + 1}, column {col + 1}", fontsize=7, color="#555")
                ax.text(
                    bx,
                    by + 9,
                    f"pages overlap {overlap_mm:g} mm: line up the pins",
                    fontsize=6.5,
                    color="#555",
                )
                ax.plot([bx, bx + 50], [by + 18] * 2, color="black", lw=0.8)
                ax.text(
                    bx, by + 15, "50 mm (print at 100% / actual size)", fontsize=6.5, color="#555"
                )
                pdf.savefig(fig)
                pages += 1
    return pages


def shopping_list(
    doc: dict, frame_mm: float, spool_m: float = 500.0, thread_mm: float | None = None
) -> str:
    n = doc["n_pins"]
    spacing = (np.pi * frame_mm if doc["frame"] == "circle" else 4 * frame_mm) / n
    measure = "diameter" if doc["frame"] == "circle" else "side"
    lines = [
        "String art shopping list",
        "",
        f"Frame: {doc['frame']}, {frame_mm:g} mm {measure}"
        f"; board at least {frame_mm + 2 * MARGIN_MM:g} mm square (the printed template's size)",
        f"Pins / nails: {n} (+ a few spare), {spacing:.1f} mm apart along the rim",
        f"Lines to wind: {sum(len(v) for v in _lines_by_thread(doc).values())}",
        "",
        f"Thread (+10% for knots and slack; spools of {spool_m:g} m):",
    ]
    for name, metres in thread_lengths_m(doc, frame_mm).items():
        need = metres * 1.1
        spools = int(np.ceil(need / spool_m))
        lines.append(f"  {name:<10} {metres:7.0f} m wound, buy {need:7.0f} m = {spools} spool(s)")
    if thread_mm:
        lines += ["", f"Simulated thread width: {thread_mm:g} mm"]
    lines += [
        "",
        "Before winding the real piece, calibrate with the thread you bought:",
        f"  stringart calibrate sheet --pins {n} --frame-mm {frame_mm:g}",
        "  (wind that short pattern, photograph it, then: stringart calibrate fit photo.jpg)",
        "and re-run the solver with the --thread-mm it prints.",
    ]
    return "\n".join(lines) + "\n"


def write_kit(run_dir: Path, frame_mm: float, spool_m: float = 500.0) -> list[Path]:
    run_dir = Path(run_dir)
    doc = load_sequence(run_dir / "sequence.json")
    doc["pins"] = np.asarray(doc["pins"]).tolist()
    thread_mm = doc.get("meta", {}).get("thread_mm")
    files = {
        "frame_template.svg": frame_template_svg(doc, frame_mm),
        "pins.csv": pins_csv(doc, frame_mm),
        "shopping_list.txt": shopping_list(doc, frame_mm, spool_m, thread_mm),
    }
    out = []
    for name, text in files.items():
        (run_dir / name).write_text(text, encoding="utf-8")
        out.append(run_dir / name)
    tiles = run_dir / "frame_template_A4_tiles.pdf"
    template_tiles_pdf(doc, frame_mm, tiles)
    out.append(tiles)
    return out
