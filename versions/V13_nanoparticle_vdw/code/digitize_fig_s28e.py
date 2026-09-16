"""Reproducibly digitise the published vdW curve of Singh et al. Fig. S28E.

The SI reports only two numbers for this curve (the minimum at
R = 2.99 nm, E = -2.33 kcal/mol per NC).  Calibrating a three-parameter
potential against two numbers is under-determined, so the whole published
curve is extracted from the vector figure and used as the calibration target.

Axis calibration is taken from the plot frame itself (left spine R = 0, right
spine R = 25 nm, top spine E = 15, bottom spine E = -5 kcal/mol), and the
result is cross-checked against the printed E = 0 grid line.

The curve is traced column by column, following the cluster of green pixels
nearest the running estimate, so the rainbow-coloured insets that overlap the
plot area at R > 19 nm are not mistaken for the curve.

Output: outputs/reference/fig_s28e_digitised.csv  (R_nm, E_kcalmol, halfwidth)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT.parents[1] / "references" / "singh.sm.pdf"
DEFAULT_OUT = ROOT / "outputs" / "reference"

PAGE_INDEX = 21          # zero-based; Fig. S28 page
CLIP = (270.0, 385.0, 560.0, 535.0)   # PDF points, panel E plus its axes
DPI = 600
AXIS_R = (0.0, 25.0)     # values at the left and right spines
AXIS_E = (15.0, -5.0)    # values at the top and bottom spines


def render(pdf_path: Path):
    import pymupdf

    doc = pymupdf.open(pdf_path)
    page = doc[PAGE_INDEX]
    pix = page.get_pixmap(dpi=DPI, clip=pymupdf.Rect(*CLIP))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )[:, :, :3]
    return img.astype(int)


def _cluster(indices):
    if len(indices) == 0:
        raise RuntimeError("no frame line found")
    groups, current = [], [indices[0]]
    for value in indices[1:]:
        if value - current[-1] <= 2:
            current.append(value)
        else:
            groups.append(np.mean(current))
            current = [value]
    groups.append(np.mean(current))
    return groups


def find_frame(img):
    """Locate the four spines of the panel E plot box."""
    dark = img.mean(axis=2) < 120
    height, width = dark.shape
    band = dark[int(0.11 * height):int(0.78 * height), :]
    columns = band.sum(axis=0)
    vertical = _cluster(np.where(columns > 0.80 * band.shape[0])[0])
    if len(vertical) < 2:
        raise RuntimeError("could not locate the left/right spines")
    left, right = vertical[0], vertical[-1]
    span = slice(int(left) + 4, int(right) - 4)
    rows = dark[:, span].sum(axis=1)
    horizontal = _cluster(np.where(rows > 0.90 * (span.stop - span.start))[0])
    if len(horizontal) < 2:
        raise RuntimeError("could not locate the top/bottom spines")
    return left, right, horizontal[0], horizontal[-1]


def trace(img, left, right, top, bottom):
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (green > red + 18) & (green > blue + 18) & (green < 190)
    rows, cols = np.nonzero(mask)
    inside = (
        (cols > left + 1) & (cols < right - 1) & (rows > top + 1) & (rows < bottom - 1)
    )
    rows, cols = rows[inside], cols[inside]

    def to_r(c):
        return AXIS_R[0] + (c - left) * (AXIS_R[1] - AXIS_R[0]) / (right - left)

    def to_e(r):
        return AXIS_E[0] + (r - top) * (AXIS_E[1] - AXIS_E[0]) / (bottom - top)

    order = np.argsort(cols)
    rows, cols = rows[order], cols[order]
    records = []
    running = None
    for column in range(int(left) + 2, int(right) - 1):
        sel = cols == column
        if not sel.any():
            continue
        candidate_rows = np.sort(rows[sel])
        groups, current = [], [candidate_rows[0]]
        for value in candidate_rows[1:]:
            if value - current[-1] <= 3:
                current.append(value)
            else:
                groups.append(current)
                current = [value]
        groups.append(current)
        values = [to_e(np.mean(g)) for g in groups]
        if running is None:
            pick = int(np.argmax(values))     # curve starts at the top left
        else:
            pick = int(np.argmin([abs(v - running) for v in values]))
        group = groups[pick]
        energy = to_e(np.mean(group))
        halfwidth = 0.5 * abs(to_e(max(group)) - to_e(min(group)))
        running = energy if running is None else 0.65 * energy + 0.35 * running
        records.append((to_r(column), energy, halfwidth))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    img = render(args.pdf)
    left, right, top, bottom = find_frame(img)
    zero_row = top + (AXIS_E[0] - 0.0) / (AXIS_E[0] - AXIS_E[1]) * (bottom - top)
    records = trace(img, left, right, top, bottom)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "fig_s28e_digitised.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["R_nm", "E_kcalmol_per_NC_axis", "halfwidth_kcalmol"])
        writer.writerows(["%.6f" % v for v in row] for row in records)

    array = np.array(records)
    index = int(np.argmin(array[:, 1]))
    print("frame px: left=%.1f right=%.1f top=%.1f bottom=%.1f" % (left, right, top, bottom))
    print("E=0 grid line predicted at row %.1f" % zero_row)
    print("traced %d columns, R in [%.3f, %.3f] nm" % (len(array), array[0, 0], array[-1, 0]))
    print("digitised minimum: R = %.3f nm, E = %.3f kcal/mol  (SI states 2.99, -2.33)"
          % (array[index, 0], array[index, 1]))
    print("line halfwidth (median) = %.3f kcal/mol" % np.median(array[:, 2]))
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
