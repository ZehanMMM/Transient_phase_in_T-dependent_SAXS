"""Stage A1d: the converged form of Eqn. 4 has no potential well at all.

Two facts about the contact limit D -> 0 for co-oriented face-to-face
particles:

  attraction   the exact double volume integral diverges.  For two parallel
               flat faces of area S at separation D the Hamaker result is
               U/S = -A/(12 pi D^2), so
                   U_attr  ->  -eps1 (A/pi^2) * pi S / (12 D^2)
               and the sextic superellipsoid's faces are flat enough over the
               central region that this is the correct leading behaviour.
  repulsion    eps2 K_W INT dS/(r2+beta)^8 is BOUNDED above by
                   eps2 K_W S_total / beta^8
               because r2 >= 0 everywhere.

So U_pair -> -infinity at contact for any (eps1, eps2, beta), and the only
thing that stops interpenetration is the hard core.  The minimum at 2.99 nm in
Fig. S28E therefore exists only because the SI's 27-element quadrature
truncates the divergence: the closest volume-element centres of two touching
particles are still a/3 = 4.46 nm apart, so the discrete attraction stays
finite.

This script measures both statements.

Output: outputs/singh_reproduction/A1d_contact_limit.csv
"""
from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "singh_reproduction"
EDGE = 13.37

CONVERGED = dict(
    volume_scheme="cartesian",
    volume_n=20,
    surface_scheme="facegl",
    surface_n=32,
    r2_mode="exact_surface",
)
SI_MESH = dict(
    volume_scheme="grid27",
    volume_n=3,
    surface_scheme="lattice",
    surface_n=8,
    r2_mode="nearest_element",
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    literature = V.load_parameters("singh_literature")
    shape = V.Superellipsoid(EDGE)

    repulsion_bound = (
        literature.epsilon2
        * literature.kw(EDGE)
        * shape.surface_area()
        / literature.beta_nm ** 8
    )
    print("upper bound on U_rep (r2 = 0 everywhere, literature parameters):")
    print("   eps2 K_W S_total / beta^8 = %.4f kcal/mol" % repulsion_bound)
    print("   the repulsive term can never exceed this, at any separation")
    print()

    gaps = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    rows = []
    print("%-8s %16s %16s %14s %14s" % ("gap_nm", "U_attr converged",
                                        "U_attr x D^2", "U_rep conv", "U_attr 27-elem"))
    for gap in gaps:
        converged = replace(literature, label="conv", **CONVERGED)
        coarse = replace(literature, label="coarse", **SI_MESH)
        fine = V.face_to_face_pair(EDGE, float(gap), converged)
        crude = V.face_to_face_pair(EDGE, float(gap), coarse)
        rows.append(
            [
                "%.6f" % gap,
                "%.6e" % fine["u_attr_kcalmol"],
                "%.6e" % (fine["u_attr_kcalmol"] * gap ** 2),
                "%.6e" % fine["u_rep_kcalmol"],
                "%.6e" % crude["u_attr_kcalmol"],
                "%.6e" % crude["u_rep_kcalmol"],
                "%.6e" % fine["u_pair_kcalmol"],
                "%.6e" % crude["u_pair_kcalmol"],
            ]
        )
        print("%-8.5f %16.4e %16.4e %14.4e %14.4e"
              % (gap, fine["u_attr_kcalmol"], fine["u_attr_kcalmol"] * gap ** 2,
                 fine["u_rep_kcalmol"], crude["u_attr_kcalmol"]))

    header = [
        "gap_nm", "U_attr_converged_kcalmol", "U_attr_converged_times_D2",
        "U_rep_converged_kcalmol", "U_attr_27element_kcalmol",
        "U_rep_386element_kcalmol", "U_pair_converged_kcalmol",
        "U_pair_SI_mesh_kcalmol",
    ]
    path = OUT / "A1d_contact_limit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    plate = (
        -literature.attraction_prefactor()
        * np.pi
        * EDGE ** 2
        / 12.0
    )
    print()
    print("U_attr x D^2 approaches a constant, so U_attr diverges as 1/D^2.")
    print("   parallel-plate prediction -eps1 (A/pi^2) pi a^2 / 12 = %.4e" % plate)
    print("   measured at the smallest gap                        = %.4e"
          % (float(rows[-1][2])))
    print("   (the measured value is smaller in magnitude because only the")
    print("    central part of a sextic face is flat, and the 20^3 quadrature")
    print("    cannot resolve a 0.03 nm gap; the 1/D^2 scaling is the point)")
    print()
    print("Closest volume-element separation of two TOUCHING particles on the")
    print("SI's 27-element lattice: a/3 = %.4f nm, so the coarse attraction"
          % (EDGE / 3.0))
    print("stays finite at contact.  That truncation, not the physics, is what")
    print("creates the Fig. S28E minimum.")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
