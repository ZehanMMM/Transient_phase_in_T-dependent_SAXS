"""Stage A2b: how far can the unknown 27-element layout move the attraction?

The SI says the nanocube is "divided into 3^3 = 27 identical volume elements"
and shows their centres in the Fig. S28E bottom inset, but publishes no
coordinates.  Three reconstructions consistent with that sentence are tried
here, to bound the layout ambiguity against the factor that would be needed
to rescue the literature parameter triple.

  regular_lattice     centres at multiples of a/3, weights V/27
                      (the module default)
  cut_plane_centroid  the superellipsoid cut by planes at +/- a/6; centres at
                      the centroids of the 27 clipped regions, weights V/27
  equal_volume_centroid
                      cut planes chosen so each of the three slabs per axis
                      holds exactly V/3; centres at the centroids of the 27
                      equal-volume regions, weights V/27
  cut_plane_true_weight
                      as cut_plane_centroid but with the ACTUAL clipped
                      volumes as weights (so the elements are not identical,
                      contradicting the SI wording; included as a bound)

Output: outputs/singh_reproduction/A2b_node_layout_probe.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma

sys.path.insert(0, str(Path(__file__).resolve().parent))

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "singh_reproduction"
EDGE = 13.37
GAP = 2.99
SAMPLES = 4_000_000


def cross_section_area(x, half):
    """Exact area of {|y|^6+|z|^6 <= (half^6-|x|^6)}."""
    scale = max(half ** 6 - abs(x) ** 6, 0.0) ** (1.0 / 6.0)
    return 4.0 * scale ** 2 * gamma(7 / 6.0) ** 2 / gamma(1 + 2 / 6.0)


def layouts():
    shape = V.Superellipsoid(EDGE)
    half = shape.half_nm
    volume = shape.volume_literature_nm3
    out = {}

    centres, weights = shape.volume_nodes("grid27", 3, "literature")
    out["regular_lattice"] = (centres, weights)

    rng = np.random.default_rng(0)
    points = rng.uniform(-half, half, (SAMPLES, 3))
    points = points[shape.contains(points)]

    total = quad(cross_section_area, -half, half, args=(half,))[0]
    equal = brentq(
        lambda t: quad(cross_section_area, -half, t, args=(half,))[0] - total / 3.0,
        -half, 0.0, xtol=1e-14,
    )

    for name, cuts in (
        ("cut_plane_centroid", np.array([-half / 3.0, half / 3.0])),
        ("equal_volume_centroid", np.array([equal, -equal])),
    ):
        index = np.stack([np.digitize(points[:, k], cuts) for k in range(3)], axis=1)
        key = index[:, 0] * 9 + index[:, 1] * 3 + index[:, 2]
        centres = np.zeros((27, 3))
        true_volume = np.zeros(27)
        for cell in range(27):
            mask = key == cell
            centres[cell] = points[mask].mean(axis=0)
            true_volume[cell] = mask.sum() / len(points) * volume
        out[name] = (centres, np.full(27, volume / 27.0))
        if name == "cut_plane_centroid":
            out["cut_plane_true_weight"] = (centres, true_volume)
    return out


def attraction(centres, weights, params, gap, edge=EDGE):
    offset = np.array([gap + edge, 0.0, 0.0])
    delta = centres[:, None, :] - (centres[None, :, :] + offset[None, None, :])
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    return -params.attraction_prefactor() * float(
        np.sum(np.outer(weights, weights) * distance2 ** -3.0)
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = V.load_parameters("singh_literature")
    calibrated = V.load_parameters("singh_calibrated_si_mesh")
    required = params.epsilon1 / calibrated.epsilon1

    rows = []
    reference = None
    for name, (centres, weights) in layouts().items():
        value = attraction(centres, weights, params, GAP)
        if reference is None:
            reference = value
        rows.append(
            [
                name,
                "%.6f" % (np.abs(centres[:, 0]).max() / EDGE),
                "%.4f" % weights.min(),
                "%.4f" % weights.max(),
                "%.4f" % weights.sum(),
                "%.6f" % value,
                "%.4f" % (value / reference),
            ]
        )
    header = [
        "layout", "outer_layer_|x|/a", "min_weight_nm3", "max_weight_nm3",
        "weight_sum_nm3", "U_attr_at_2.99nm_kcalmol", "ratio_to_regular_lattice",
    ]
    path = OUT / "A2b_node_layout_probe.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    print("%-24s %10s %10s %14s %8s" % ("layout", "|x|max/a", "Vmin", "U_attr", "ratio"))
    for row in rows:
        print("%-24s %10s %10s %14s %8s" % (row[0], row[1], row[2], row[5], row[6]))
    spread = max(float(r[6]) for r in rows) / min(float(r[6]) for r in rows)
    print()
    print("layout ambiguity spans a factor of %.3f in U_attr." % spread)
    print("Rescuing eps1 = %.0f from the calibrated %.2f would need a factor of "
          "%.3f." % (params.epsilon1, calibrated.epsilon1, required))
    print("The layout ambiguity is %.1fx too small to explain the discrepancy."
          % (required / spread))
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
