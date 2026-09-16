"""Stage B8: refine BOTH integrals together and map where the well survives.

Earlier stages refined one integral while holding the other converged.  This
one starts from the SI's own pair of discretisations, 27 volume elements and
386 surface elements with the literal nearest-element r_2, and refines them
together.  It also separates the two directions, because they push opposite
ways:

  refining the VOLUME   raises |U_attr| by 9.06 kcal/mol towards the limit,
                        which removes the near-cancellation and kills the well;
  refining the SURFACE  raises U_rep by only 0.16 kcal/mol, which pushes
                        max(U_rep/|U_attr|) slightly further above 1 and so
                        makes the well condition marginally easier to satisfy,
                        while making the well itself SHALLOWER because U_pair
                        becomes less negative.

Measured outcome: the volume term wins by a factor of 57, so joint refinement
behaves like volume refinement and the well is gone from 64 voxels upwards at
every surface resolution from 386 to 6146 elements.  Holding the volume at 27
voxels and refining only the surface leaves the well in place, moving it from
3.00 nm / -2.35 kcal/mol to 3.10 nm / -2.22 kcal/mol.

The calculation exploits an exact separability: U_attr depends only on the
volume mesh and U_rep only on the surface mesh, verified to 0.00e+00 spread in
stage B6.  Two one-dimensional sweeps therefore give every combination for
free, and the whole (volume x surface) map costs no more than the two sweeps.

Outputs:
  outputs/convergence/B8_joint_mesh_map.csv
  outputs/convergence/B8_joint_mesh_map.{png,pdf}
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
DIRECTION = np.array([1.0, 0.0, 0.0])

GAPS = np.concatenate([
    np.arange(0.30, 6.00, 0.10),
    np.arange(6.00, 14.01, 0.40),
])

VOLUME_LEVELS = [3, 4, 6, 8, 12, 16, 24]      # midpoint, the SI rule refined
SURFACE_LEVELS = [8, 10, 12, 16, 20, 24, 32]  # lattice, 6n^2+2 elements
SURFACE_REFINE = 24


def placements(gap):
    radius = V.centre_distance_for_gap(EDGE_NM, float(gap), DIRECTION)
    return (V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0]),
            V.Placement.create(EDGE_NM, radius * DIRECTION))


def attraction_sweep(base):
    out = {}
    for n in VOLUME_LEVELS:
        params = replace(base, volume_scheme="midpoint", volume_n=n)
        nodes = len(V.Superellipsoid(EDGE_NM).volume_nodes(
            "midpoint", n, params.kw_volume_convention)[0])
        t0 = time.time()
        values = np.array([
            V.attraction_kcalmol(*placements(g), params)[0] for g in GAPS
        ])
        out[nodes] = values
        print("  volume  n=%2d  %6d voxels  U_attr(2.99) = %10.4f   %5.1f s"
              % (n, nodes, np.interp(2.99, GAPS, values), time.time() - t0))
    return out


def repulsion_sweep(base, r2_mode):
    out = {}
    for n in SURFACE_LEVELS:
        params = replace(base, surface_scheme="lattice", surface_n=n,
                         surface_refine=SURFACE_REFINE, r2_mode=r2_mode)
        elements = len(V.Superellipsoid(EDGE_NM).surface_nodes(
            "lattice", n, SURFACE_REFINE, params.surface_area_mode)[0])
        t0 = time.time()
        values = np.array([
            V.repulsion_kcalmol(*placements(g), params, float(g))[0]
            for g in GAPS
        ])
        out[elements] = values
        print("  surface n=%2d  %6d elements U_rep(2.99)  = %10.4f   %5.1f s"
              % (n, elements, np.interp(2.99, GAPS, values), time.time() - t0))
    return out


def well_of(pair):
    index = int(np.argmin(pair))
    if index == 0 or index == len(pair) - 1:
        return None
    return float(GAPS[index]), float(pair[index])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = V.load_parameters("singh_calibrated_si_mesh")
    print("attraction sweep (volume mesh only) ...")
    attraction = attraction_sweep(base)
    print("repulsion sweep, nearest_element (the SI's literal r_2) ...")
    repulsion = repulsion_sweep(base, "nearest_element")
    print("repulsion limit, exact point-to-body r_2 ...")
    limit_params = replace(base, surface_scheme="facegl", surface_n=32,
                           r2_mode="exact_surface")
    rep_limit = np.array([
        V.repulsion_kcalmol(*placements(g), limit_params, float(g))[0]
        for g in GAPS
    ])
    print("  exact r_2, 6144 nodes      U_rep(2.99)  = %10.4f"
          % np.interp(2.99, GAPS, rep_limit))

    voxels = sorted(attraction)
    elements = sorted(repulsion)

    rows = []
    print()
    print("well position and depth over the (voxel x surface element) map")
    print("%9s | %s" % ("voxels", "".join("%14d" % e for e in elements)))
    for v in voxels:
        cells = []
        for e in elements:
            pair = attraction[v] + repulsion[e]
            well = well_of(pair)
            cells.append("%14s" % (("%.2f/%.2f" % well) if well else "none"))
            rows.append([
                v, e,
                "%.8e" % np.interp(2.99, GAPS, attraction[v]),
                "%.8e" % np.interp(2.99, GAPS, repulsion[e]),
                "%.8e" % np.interp(2.99, GAPS, pair),
                "%.6f" % np.max(repulsion[e] / np.abs(attraction[v])),
                "%.4f" % well[0] if well else "",
                "%.5f" % well[1] if well else "",
            ])
        print("%9d | %s" % (v, "".join(cells)))

    with (OUT / "B8_joint_mesh_map.csv").open("w", newline="",
                                              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["volume_voxels", "surface_elements",
                         "U_attr_at_2.99", "U_rep_at_2.99", "U_pair_at_2.99",
                         "max_Urep_over_Uattr", "well_gap_nm",
                         "well_depth_kcalmol"])
        writer.writerows(rows)

    make_figure(attraction, repulsion, rep_limit, voxels, elements)
    print()
    print("wrote %s" % OUT)


def make_figure(attraction, repulsion, rep_limit, voxels, elements):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
    cmap = plt.get_cmap("viridis")

    # (a) the diagonal: both meshes refined together
    ax = axes[0, 0]
    pairs = list(zip(voxels, elements))
    for index, (v, e) in enumerate(pairs):
        pair = attraction[v] + repulsion[e]
        colour = cmap(index / max(len(pairs) - 1, 1))
        ax.plot(GAPS, pair, color=colour, lw=1.7,
                label="%d vox + %d elem" % (v, e))
        well = well_of(pair)
        if well:
            ax.plot([well[0]], [well[1]], "o", color=colour, ms=9,
                    markeredgecolor="k", zorder=6)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(2.99, color="0.6", lw=0.9, ls=":")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(-14, 4)
    ax.set_xlabel("surface separation D [nm]")
    ax.set_ylabel(r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_title("(a) both meshes refined together: the well still goes")
    ax.legend(fontsize=7, loc="lower right")

    # (b) surface refined alone, volume pinned at the SI's 27
    ax = axes[0, 1]
    for index, e in enumerate(elements):
        pair = attraction[voxels[0]] + repulsion[e]
        colour = cmap(index / max(len(elements) - 1, 1))
        ax.plot(GAPS, pair, color=colour, lw=1.7, label="%d elements" % e)
        well = well_of(pair)
        if well:
            ax.plot([well[0]], [well[1]], "o", color=colour, ms=9,
                    markeredgecolor="k", zorder=6)
    pair = attraction[voxels[0]] + rep_limit
    ax.plot(GAPS, pair, "k--", lw=1.8, label="exact $r_2$ limit")
    well = well_of(pair)
    if well:
        ax.plot([well[0]], [well[1]], "ko", ms=9, zorder=6)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(2.99, color="0.6", lw=0.9, ls=":")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(-6, 3)
    ax.set_xlabel("surface separation D [nm]")
    ax.set_ylabel(r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_title("(b) surface alone, 27 voxels held: the well SURVIVES")
    ax.legend(fontsize=7, loc="lower right")

    # (c) the two components against their own mesh
    ax = axes[1, 0]
    ax.plot(voxels, [np.interp(2.99, GAPS, attraction[v]) for v in voxels],
            "o-", color="#d62728", label=r"$U_{\rm attr}$ vs voxels")
    ax.plot(elements, [np.interp(2.99, GAPS, repulsion[e]) for e in elements],
            "s-", color="#2ca02c", label=r"$U_{\rm rep}$ vs surface elements")
    ax.axhline(np.interp(2.99, GAPS, rep_limit), color="#2ca02c", ls=":",
               lw=1.2, label=r"$U_{\rm rep}$ exact-$r_2$ limit")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("nodes in the relevant integral")
    ax.set_ylabel("energy at a 2.99 nm gap [kcal/mol]")
    ax.set_title("(c) the attraction moves by 9 kcal/mol, the repulsion by 0.8")
    ax.legend(fontsize=8)

    # (d) where a well exists, over the full map
    ax = axes[1, 1]
    grid = np.zeros((len(voxels), len(elements)))
    for i, v in enumerate(voxels):
        for j, e in enumerate(elements):
            grid[i, j] = np.max(repulsion[e] / np.abs(attraction[v]))
    mesh = ax.pcolormesh(np.arange(len(elements) + 1),
                         np.arange(len(voxels) + 1), grid,
                         cmap="RdBu_r", vmin=0.2, vmax=1.8, shading="flat")
    fig.colorbar(mesh, ax=ax, label=r"max $U_{\rm rep}/|U_{\rm attr}|$")
    for i, v in enumerate(voxels):
        for j, e in enumerate(elements):
            pair = attraction[v] + repulsion[e]
            mark = "well" if well_of(pair) else "-"
            ax.text(j + 0.5, i + 0.5, "%s\n%.2f" % (mark, grid[i, j]),
                    ha="center", va="center", fontsize=6.5,
                    color="k" if grid[i, j] < 1.2 else "w")
    ax.set_xticks(np.arange(len(elements)) + 0.5)
    ax.set_xticklabels(elements, fontsize=7)
    ax.set_yticks(np.arange(len(voxels)) + 0.5)
    ax.set_yticklabels(voxels, fontsize=7)
    ax.set_xlabel("surface elements per particle")
    ax.set_ylabel("volume voxels per particle")
    ax.set_title("(d) a well needs max ratio > 1: only the 27-voxel row")

    fig.suptitle("Refining both integrals together, $a$ = 13.37 nm, "
                 "face-to-face [100], fixed potential", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("B8_joint_mesh_map.%s" % suffix), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
