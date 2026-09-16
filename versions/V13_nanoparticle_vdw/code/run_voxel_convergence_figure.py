"""Stage B6: U_attr, U_rep and U_pair against voxel count, for two volume rules.

Purpose: settle by picture whether refining the SI's midpoint lattice
converges, and show where the error actually lives.

Three facts the figure makes visible.

1. U_rep does not depend on the volume rule at all.  It is a surface integral,
   so every point in the volume sweep has an identical U_rep.  All of the
   movement in U_pair comes from U_attr.
2. Both volume rules converge to the same limit.  The midpoint rule gets there
   slowly and with visible jitter; the nested Gauss-Legendre rule gets there
   an order of magnitude faster at equal node count.
3. The jitter is a boundary-registration effect, not a failure to converge.
   As n changes, the set of lattice cells whose centre lies inside the body
   changes discontinuously, and cells near the two facing faces carry most of
   the 1/r_1^6 weight, so gaining or losing one near-surface shell moves the
   answer by percent.  Gauss-Legendre avoids this twice over: its integration
   limits are exact, and its nodes cluster towards x = +/- a/2.

Outputs:
  outputs/convergence/B6_voxel_convergence.csv
  outputs/convergence/B6_voxel_convergence.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "convergence"

EDGE_NM = 13.37
GAP_NM = 2.99

VOLUME_LEVELS = [3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
SURFACE_LEVELS = [3, 4, 6, 8, 12, 16, 24, 32]
FIXED_SURFACE = dict(surface_scheme="facegl", surface_n=32,
                     r2_mode="exact_surface")
FIXED_VOLUME = dict(volume_scheme="cartesian", volume_n=20)
REFERENCE = dict(volume_scheme="cartesian", volume_n=36, **FIXED_SURFACE)


def volume_sweep(base, scheme):
    rows = []
    for n in VOLUME_LEVELS:
        params = replace(base, volume_scheme=scheme, volume_n=n,
                         **FIXED_SURFACE)
        t0 = time.time()
        r = V.face_to_face_pair(EDGE_NM, GAP_NM, params)
        rows.append(
            {
                "scheme": scheme,
                "n": n,
                "nodes": r["volume_nodes"],
                "u_attr": r["u_attr_kcalmol"],
                "u_rep": r["u_rep_kcalmol"],
                "u_pair": r["u_pair_kcalmol"],
                "seconds": time.time() - t0,
            }
        )
        print("  %-9s n=%2d  nodes=%6d  U_attr=%11.5f  U_rep=%9.5f  "
              "U_pair=%11.5f  %6.1f s"
              % (scheme, n, r["volume_nodes"], r["u_attr_kcalmol"],
                 r["u_rep_kcalmol"], r["u_pair_kcalmol"], rows[-1]["seconds"]))
    return rows


def surface_sweep(base, r2_mode):
    rows = []
    for n in SURFACE_LEVELS:
        params = replace(base, surface_scheme="facegl", surface_n=n,
                         r2_mode=r2_mode, **FIXED_VOLUME)
        r = V.face_to_face_pair(EDGE_NM, GAP_NM, params)
        rows.append(
            {
                "r2_mode": r2_mode,
                "n": n,
                "nodes": r["surface_nodes"],
                "u_rep": r["u_rep_kcalmol"],
            }
        )
        print("  %-16s n=%2d  nodes=%5d  U_rep=%10.6f"
              % (r2_mode, n, r["surface_nodes"], r["u_rep_kcalmol"]))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = V.load_parameters("singh_calibrated_si_mesh")

    print("reference mesh ...")
    ref = V.face_to_face_pair(EDGE_NM, GAP_NM, replace(base, **REFERENCE))
    ra, rr, rp = ref["u_attr_kcalmol"], ref["u_rep_kcalmol"], ref["u_pair_kcalmol"]
    print("  cartesian n=36, %d nodes: U_attr=%.6f  U_rep=%.6f  U_pair=%.6f"
          % (ref["volume_nodes"], ra, rr, rp))

    print("\nthe SI rule itself, on its own 386-element surface mesh ...")
    si = V.face_to_face_pair(EDGE_NM, GAP_NM, base)
    print("  grid27 + lattice386: U_attr=%.6f  U_rep=%.6f  U_pair=%.6f"
          % (si["u_attr_kcalmol"], si["u_rep_kcalmol"], si["u_pair_kcalmol"]))

    print("\nvolume sweep, midpoint (the SI rule refined) ...")
    midpoint = volume_sweep(base, "midpoint")
    print("\nvolume sweep, nested Gauss-Legendre ...")
    cartesian = volume_sweep(base, "cartesian")
    print("\nsurface sweep, exact_surface ...")
    exact = surface_sweep(base, "exact_surface")
    print("\nsurface sweep, nearest_element (the SI's literal reading) ...")
    nearest = surface_sweep(base, "nearest_element")

    # U_rep must be identical across the whole volume sweep
    reps = np.array([r["u_rep"] for r in midpoint + cartesian])
    spread = float(np.max(np.abs(reps / reps[0] - 1.0)))
    print("\nU_rep spread across the entire volume sweep: %.2e  "
          "(it is a surface integral, so this must be zero)" % spread)

    header = ["block", "scheme_or_r2_mode", "n", "nodes", "U_attr_kcalmol",
              "U_rep_kcalmol", "U_pair_kcalmol", "rel_err_attr",
              "rel_err_pair", "wall_seconds"]
    rows = []
    for r in midpoint + cartesian:
        rows.append([
            "volume", r["scheme"], r["n"], r["nodes"],
            "%.8e" % r["u_attr"], "%.8e" % r["u_rep"], "%.8e" % r["u_pair"],
            "%.4e" % abs(r["u_attr"] / ra - 1.0),
            "%.4e" % abs(r["u_pair"] / rp - 1.0), "%.3f" % r["seconds"],
        ])
    for r in exact + nearest:
        rows.append([
            "surface", r["r2_mode"], r["n"], r["nodes"], "", "%.8e" % r["u_rep"],
            "", "", "%.4e" % abs(r["u_rep"] / rr - 1.0), "",
        ])
    with (OUT / "B6_voxel_convergence.csv").open("w", newline="",
                                                 encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    make_figure(midpoint, cartesian, exact, nearest, si, (ra, rr, rp))
    print("wrote %s" % OUT)


def make_figure(midpoint, cartesian, exact, nearest, si, reference):
    ra, rr, rp = reference
    mid_n = np.array([r["nodes"] for r in midpoint])
    car_n = np.array([r["nodes"] for r in cartesian])
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))

    # (a) U_attr against voxel count
    ax = axes[0, 0]
    ax.axhline(ra, color="k", ls="--", lw=1.0,
               label="converged limit %.3f" % ra)
    ax.plot(mid_n, [r["u_attr"] for r in midpoint], "o-", color="#d62728",
            ms=5, label="midpoint (SI rule refined)")
    ax.plot(car_n, [r["u_attr"] for r in cartesian], "s-", color="#1f77b4",
            ms=5, label="nested Gauss-Legendre")
    ax.plot([27], [si["u_attr_kcalmol"]], "*", color="#d62728", ms=17,
            markeredgecolor="k", zorder=5, label="SI grid27, 27 voxels")
    ax.set_xscale("log")
    ax.set_xlabel("volume nodes used per particle")
    ax.set_ylabel(r"$U_{\rm attr}$ [kcal/mol]")
    ax.set_title("(a) attraction: all the movement is here")
    ax.legend(fontsize=8, loc="lower right")

    # (b) U_rep against surface count, and its independence of the volume mesh
    ax = axes[0, 1]
    ax.axhline(rr, color="k", ls="--", lw=1.0,
               label="converged limit %.4f" % rr)
    ax.plot([r["nodes"] for r in exact], [r["u_rep"] for r in exact], "o-",
            color="#2ca02c", ms=5, label=r"$r_2$ = exact surface distance")
    ax.plot([r["nodes"] for r in nearest], [r["u_rep"] for r in nearest], "^--",
            color="#9467bd", ms=5, label=r"$r_2$ = nearest element (SI literal)")
    ax.plot([386], [si["u_rep_kcalmol"]], "*", color="#9467bd", ms=17,
            markeredgecolor="k", zorder=5, label="SI 386 elements")
    ax.set_xscale("log")
    ax.set_xlabel("surface nodes per particle")
    ax.set_ylabel(r"$U_{\rm rep}$ [kcal/mol]")
    ax.set_title("(b) repulsion: independent of the volume mesh entirely")
    ax.legend(fontsize=8, loc="lower right")

    # (c) U_pair against voxel count
    ax = axes[1, 0]
    ax.axhline(rp, color="k", ls="--", lw=1.0,
               label="converged limit %.3f" % rp)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.plot(mid_n, [r["u_pair"] for r in midpoint], "o-", color="#d62728",
            ms=5, label="midpoint (SI rule refined)")
    ax.plot(car_n, [r["u_pair"] for r in cartesian], "s-", color="#1f77b4",
            ms=5, label="nested Gauss-Legendre")
    ax.plot([27], [si["u_pair_kcalmol"]], "*", color="#d62728", ms=17,
            markeredgecolor="k", zorder=5,
            label="SI grid27, %.2f kcal/mol" % si["u_pair_kcalmol"])
    ax.set_xscale("log")
    ax.set_xlabel("volume nodes used per particle")
    ax.set_ylabel(r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_title("(c) total: the SI value is 4.8x shallower than the limit")
    ax.legend(fontsize=8, loc="lower right")

    # (d) relative errors, log-log
    ax = axes[1, 1]
    for rows, n, colour, marker, label in (
        (midpoint, mid_n, "#d62728", "o", "midpoint"),
        (cartesian, car_n, "#1f77b4", "s", "Gauss-Legendre"),
    ):
        err = np.array([abs(r["u_attr"] / ra - 1.0) for r in rows])
        keep = err > 0
        ax.loglog(n[keep], err[keep], marker + "-", color=colour, ms=5,
                  label=r"%s, $U_{\rm attr}$" % label)
        err = np.array([abs(r["u_pair"] / rp - 1.0) for r in rows])
        keep = err > 0
        ax.loglog(n[keep], err[keep], marker + ":", color=colour, ms=4,
                  alpha=0.6, label=r"%s, $U_{\rm pair}$" % label)
    ref_line = np.array([60.0, 30000.0])
    ax.loglog(ref_line, 0.6 * (ref_line / 60.0) ** (-1.0 / 3.0), color="0.5",
              lw=1.0, ls="-.", label=r"$N^{-1/3}$, i.e. first order in $n$")
    ax.axhline(0.01, color="k", ls="--", lw=0.9, label="1 % target")
    ax.set_xlabel("volume nodes used per particle")
    ax.set_ylabel("relative error")
    ax.set_title("(d) both converge; the rates differ by an order")
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Convergence in voxel count at $a$ = 13.37 nm, 2.99 nm gap, "
                 "fixed potential", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("B6_voxel_convergence.%s" % suffix), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
