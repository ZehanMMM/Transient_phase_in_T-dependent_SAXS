"""Stage B7: the whole E-r curve at each voxel level, small mesh to large.

One curve per volume-mesh level, so the well can be watched appearing and
disappearing instead of being argued about from single numbers.

Configuration: co-oriented face-to-face along [100] at a = 13.37 nm, the case
that carries the Fig. S28E minimum.  The potential is frozen
(singh_calibrated_si_mesh constants) and the surface mesh is held at the
converged setting, so the only thing changing between curves is the volume
quadrature.

A caution on mechanism, recorded because an earlier version of this analysis
got it wrong: the well does NOT require U_rep/|U_attr| to cross 1.  At 27
voxels that ratio is 0.807 at the minimum and a well still exists.  The
stationarity condition is

    d|U_attr|/dD (x - 1) + |U_attr| dx/dD = 0,     x = U_rep / |U_attr|

so a minimum needs x to be falling with D fast enough there, not to exceed 1.
What the refinement actually destroys is the near-cancellation: at 27 voxels
|U_attr| and U_rep are within 24 % of each other, so the small residue is free
to be non-monotonic; in the limit |U_attr| is 2.2x U_rep and the residue
simply tracks the attraction.

Outputs:
  outputs/convergence/B7_voxel_curve_family.csv
  outputs/convergence/B7_voxel_curve_family.{png,pdf}
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
DIRECTION = (1.0, 0.0, 0.0)
FIXED_SURFACE = dict(surface_scheme="facegl", surface_n=32,
                     r2_mode="exact_surface")

GAPS = np.concatenate([
    np.arange(0.30, 6.00, 0.15),
    np.arange(6.00, 14.01, 0.50),
])

MIDPOINT_LEVELS = [3, 4, 6, 8, 12, 16, 24]
CARTESIAN_LEVELS = [3, 4, 8, 16, 24]
REFERENCE = ("cartesian", 20)


def curve(base, scheme, n):
    params = replace(base, volume_scheme=scheme, volume_n=n, **FIXED_SURFACE)
    kernels = V.energy_kernels(EDGE_NM, GAPS, params, direction=DIRECTION)
    attr, rep, pair = V.kernel_energies(
        kernels, params.epsilon1, params.epsilon2, params.beta_nm
    )
    nodes = len(V.Superellipsoid(EDGE_NM).volume_nodes(
        scheme, n, params.kw_volume_convention)[0])
    return nodes, np.asarray(attr), np.asarray(rep), np.asarray(pair)


def interior_minimum(gaps, values):
    """Minimum on the computed grid, only if it is strictly interior."""
    index = int(np.argmin(values))
    if index == 0 or index == len(values) - 1:
        return None
    return float(gaps[index]), float(values[index])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = V.load_parameters("singh_calibrated_si_mesh")
    results = {}

    for scheme, levels in (("midpoint", MIDPOINT_LEVELS),
                           ("cartesian", CARTESIAN_LEVELS)):
        for n in levels:
            t0 = time.time()
            nodes, attr, rep, pair = curve(base, scheme, n)
            results[(scheme, n)] = (nodes, attr, rep, pair)
            well = interior_minimum(GAPS, pair)
            print("%-9s n=%2d nodes=%6d  %5.1f s  U_pair(2.99)=%9.4f  well: %s"
                  % (scheme, n, nodes, time.time() - t0,
                     float(np.interp(2.99, GAPS, pair)),
                     ("%.2f nm / %.3f kcal/mol" % well) if well else "none"))

    nodes, attr, rep, pair = curve(base, *REFERENCE)
    results[REFERENCE] = (nodes, attr, rep, pair)
    print("reference %s n=%d, %d nodes" % (REFERENCE[0], REFERENCE[1], nodes))

    header = ["scheme", "n_per_axis", "volume_nodes", "gap_nm",
              "U_attr_kcalmol", "U_rep_kcalmol", "U_pair_kcalmol"]
    rows = []
    for (scheme, n), (nodes, a, r, p) in results.items():
        for gap, ai, ri, pi in zip(GAPS, a, r, p):
            rows.append([scheme, n, nodes, "%.4f" % gap, "%.8e" % ai,
                         "%.8e" % ri, "%.8e" % pi])
    with (OUT / "B7_voxel_curve_family.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    make_figure(results)
    print("wrote %s" % OUT)


def make_figure(results):
    ref_nodes, ref_attr, ref_rep, ref_pair = results[REFERENCE]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.4))

    def plot_family(ax, scheme, levels, key, label):
        for index, n in enumerate(levels):
            nodes, attr, rep, pair = results[(scheme, n)]
            values = {"attr": attr, "rep": rep, "pair": pair}[key]
            colour = cmap(index / max(len(levels) - 1, 1))
            ax.plot(GAPS, values, color=colour, lw=1.6,
                    label="%d voxels" % nodes)
            if key == "pair":
                well = interior_minimum(GAPS, values)
                if well:
                    ax.plot([well[0]], [well[1]], "o", color=colour, ms=8,
                            markeredgecolor="k", zorder=6)
        values = {"attr": ref_attr, "rep": ref_rep, "pair": ref_pair}[key]
        ax.plot(GAPS, values, color="k", lw=2.2, ls="--",
                label="%d voxels (limit)" % ref_nodes)
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_xlabel("surface separation D [nm]")
        ax.set_ylabel(label)

    # (a) midpoint, U_pair, full range
    ax = axes[0, 0]
    plot_family(ax, "midpoint", MIDPOINT_LEVELS, "pair", r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_xlim(0, 14)
    ax.set_ylim(-70, 10)
    ax.set_title("(a) midpoint rule: $U_{\\rm pair}(D)$, full range")
    ax.legend(fontsize=7, loc="lower right")

    # (b) midpoint, U_pair, zoom on the well
    ax = axes[0, 1]
    plot_family(ax, "midpoint", MIDPOINT_LEVELS, "pair", r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(-14, 3)
    ax.axvline(2.99, color="0.6", lw=0.9, ls=":")
    ax.set_title("(b) same curves, zoomed: the 27-voxel well is the only one")
    ax.legend(fontsize=7, loc="lower right")

    # (c) midpoint, U_attr
    ax = axes[0, 2]
    plot_family(ax, "midpoint", MIDPOINT_LEVELS, "attr", r"$U_{\rm attr}$ [kcal/mol]")
    ax.set_xlim(0, 14)
    ax.set_ylim(-80, 2)
    ax.set_title("(c) attraction deepens monotonically with voxel count")
    ax.legend(fontsize=7, loc="lower right")

    # (d) U_rep, identical for every level
    ax = axes[1, 0]
    for index, n in enumerate(MIDPOINT_LEVELS):
        nodes, attr, rep, pair = results[("midpoint", n)]
        ax.plot(GAPS, rep, color=cmap(index / max(len(MIDPOINT_LEVELS) - 1, 1)),
                lw=3.0 - 0.3 * index, label="%d voxels" % nodes)
    ax.plot(GAPS, ref_rep, "k--", lw=1.2, label="limit")
    ax.set_xlim(0, 14)
    ax.set_xlabel("surface separation D [nm]")
    ax.set_ylabel(r"$U_{\rm rep}$ [kcal/mol]")
    ax.set_title("(d) repulsion: every curve is the same curve")
    ax.legend(fontsize=7)

    # (e) Gauss-Legendre, U_pair
    ax = axes[1, 1]
    plot_family(ax, "cartesian", CARTESIAN_LEVELS, "pair",
                r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(-14, 3)
    ax.axvline(2.99, color="0.6", lw=0.9, ls=":")
    ax.set_title("(e) Gauss-Legendre: no well at any level, even 27 voxels")
    ax.legend(fontsize=7, loc="lower right")

    # (f) the cancellation ratio
    ax = axes[1, 2]
    for index, n in enumerate(MIDPOINT_LEVELS):
        nodes, attr, rep, pair = results[("midpoint", n)]
        ax.plot(GAPS, rep / np.abs(attr),
                color=cmap(index / max(len(MIDPOINT_LEVELS) - 1, 1)), lw=1.6,
                label="%d voxels" % nodes)
    ax.plot(GAPS, ref_rep / np.abs(ref_attr), "k--", lw=2.2, label="limit")
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    ax.set_xlim(0, 14)
    ax.set_xlabel("surface separation D [nm]")
    ax.set_ylabel(r"$U_{\rm rep} / |U_{\rm attr}|$")
    ax.set_title("(f) the near-cancellation that the coarse mesh manufactures")
    ax.legend(fontsize=7)

    fig.suptitle("Effective vdW curve at each voxel level, $a$ = 13.37 nm, "
                 "face-to-face [100], fixed potential", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("B7_voxel_curve_family.%s" % suffix), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
