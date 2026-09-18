"""Stage B5: does refining the SI's own 27-element rule converge, and to what?

The SI rule is an equal-weight midpoint lattice.  The 'midpoint' scheme
reproduces it exactly at n = 3 and refines it consistently for larger n:
cells whose centre falls outside the superellipsoid are dropped and the
surviving weights are renormalised, so no partial boundary cell is ever
counted as a full one.

Two questions are answered here.

1. Does U_attr converge?  Yes, to the same limit as the Gauss-Legendre rules,
   because any consistent quadrature converges to the same definite integral.
   But the staircase boundary costs an order: the midpoint error falls roughly
   like 1/n while the nested Gauss-Legendre error falls far faster at equal
   node count.
2. Does the potential well survive the refinement?  No.  It is present only at
   n = 3 and disappears as soon as the rule starts resolving the near-surface
   layer.  The well is a property of that one coarse lattice, not of Eqn. 4.

Output: outputs/convergence/B5_midpoint_refinement.csv
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "convergence"

EDGE_NM = 13.37
GAP_NM = 2.99
LEVELS = [3, 4, 6, 8, 10, 12, 16, 20]
SURFACE = dict(surface_scheme="facegl", surface_n=32, r2_mode="exact_surface")
REFERENCE = dict(volume_scheme="cartesian", volume_n=26, **SURFACE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = V.load_parameters("singh_calibrated_si_mesh")
    shape = V.Superellipsoid(EDGE_NM)

    reference = V.face_to_face_pair(EDGE_NM, GAP_NM, replace(base, **REFERENCE))
    ref_attr = reference["u_attr_kcalmol"]
    print("reference: cartesian n=26, %d nodes, U_attr = %.6f kcal/mol"
          % (reference["volume_nodes"], ref_attr))
    print()

    # the SI rule itself, for the record
    si = V.face_to_face_pair(EDGE_NM, GAP_NM, base)
    print("SI grid27 on its own surface mesh: U_attr = %.6f (%.1f %% off), "
          "U_pair = %.6f" % (si["u_attr_kcalmol"],
                             100 * abs(si["u_attr_kcalmol"] / ref_attr - 1),
                             si["u_pair_kcalmol"]))
    print()

    rows = []
    previous = None
    print("%-4s %7s %7s %13s %10s %10s %13s %s"
          % ("n", "sites", "inside", "U_attr", "err", "vs prev", "U_pair", "well"))
    for n in LEVELS:
        params = replace(base, volume_scheme="midpoint", volume_n=n, **SURFACE)
        nodes, weights = shape.volume_nodes("midpoint", n, base.kw_volume_convention)
        t0 = time.time()
        result = V.face_to_face_pair(EDGE_NM, GAP_NM, params)
        elapsed = time.time() - t0
        attr = result["u_attr_kcalmol"]
        error = abs(attr / ref_attr - 1.0)
        change = abs(attr / previous - 1.0) if previous is not None else np.nan
        previous = attr
        gap, depth, interior = V.find_well(
            EDGE_NM, params, bracket=(0.2, 14.0), samples=60
        )
        well = ("%.4f nm / %.4f" % (gap, depth)) if interior else "NONE"
        print("%-4d %7d %7d %13.6f %9.2e %9.2e %13.6f %s"
              % (n, n ** 3, len(nodes), attr, error, change,
                 result["u_pair_kcalmol"], well))
        rows.append(
            [
                n,
                n ** 3,
                len(nodes),
                "%.8e" % attr,
                "%.4e" % error,
                "%.4e" % change,
                "%.8e" % result["u_rep_kcalmol"],
                "%.8e" % result["u_pair_kcalmol"],
                "%.4f" % gap if interior else "none",
                "%.5f" % depth if interior else "",
                "%.3f" % elapsed,
            ]
        )

    header = [
        "n_per_axis", "lattice_sites", "nodes_inside", "U_attr_kcalmol",
        "rel_error_vs_cartesian26", "rel_change_vs_previous", "U_rep_kcalmol",
        "U_pair_kcalmol", "well_gap_nm", "well_depth_kcalmol", "wall_seconds",
    ]
    path = OUT / "B5_midpoint_refinement.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    # observed convergence order from the last three levels
    counts = np.array([int(r[2]) for r in rows[-3:]], dtype=float)
    errors = np.array([float(r[4]) for r in rows[-3:]])
    order = np.polyfit(np.log(counts ** (1 / 3.0)), np.log(errors), 1)[0]
    print()
    print("observed midpoint order in n over the last three levels: %.2f" % order)
    print("(a staircase boundary limits this to about first order; the nested")
    print(" Gauss-Legendre rule reaches 1.3e-3 at 1728 nodes for comparison)")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
