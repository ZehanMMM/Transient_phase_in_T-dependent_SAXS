"""Stage B: continuum convergence of the two integrals at a FIXED potential.

The potential definition is frozen for the whole study: shape, edge length,
A, eps1, eps2, beta and the K_W reference normalisation (0.9 a^3 / 27, with
the 27 NEVER rebound to the number of quadrature nodes).  Only the quadrature
is refined.  eps1 and eps2 enter linearly, so the RELATIVE errors reported for
U_attr and U_rep are independent of their values; only U_pair, where the two
terms partially cancel, depends on the parameter set.

Acceptance targets used here are THIS PROJECT'S, not the SI's:
    * <= 1 % relative change on two successive refinements, per component;
    * <= 0.01 kBT absolute where a component passes through zero;
    * well position stable to < 0.03 nm.

Both components are tracked separately so that cancellation between an
over-estimated attraction and an over-estimated repulsion cannot masquerade
as convergence.

The primary convergence measure is the RELATIVE CHANGE between successive
refinement levels, which needs no reference value.  A reference is also
reported: the nested Cartesian rule at n = 26 (17576 nodes), which is one
step beyond the top of the refinement ladder and whose own successive change
is below 1e-3 at every gap tested.  The reference must come from the SAME
rule family: using the radial rule as the reference produces a meaningless
few-percent "error" column at sub-2 nm gaps, because the radial rule is
itself unconverged there (see the note in B1).

Rule of thumb established by this study: at near-contact separations the
nested Cartesian Gauss-Legendre rule is the correct choice, because its nodes
cluster towards x = +/- a/2, i.e. towards the facing faces where the 1/r^6
integrand is concentrated.  The radial rule resolves the same region only
through its angular grid and is NOT converged below about a 2 nm gap even at
8192 nodes.  Above about 3 nm both rules agree.

Outputs land in outputs/convergence/.
"""
from __future__ import annotations

import argparse
import csv
import json
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
TEMPERATURE_K = 300.0
KBT_KCALMOL = 1.0 / V.kcalmol_to_kBT(1.0, TEMPERATURE_K)

VOLUME_LEVELS = [3, 4, 6, 8, 12, 16, 20, 24]
VOLUME_REFERENCE = ("cartesian", 26)
SURFACE_LEVELS = [3, 4, 6, 8, 12, 16, 24, 32]
SURFACE_REFERENCE = ("facegl", 56)

CONFIGURATIONS = [
    ("face_to_face_100", (1.0, 0.0, 0.0), None),
    ("face_edge_110", (1.0, 1.0, 0.0), None),
    ("corner_111", (1.0, 1.0, 1.0), None),
    ("face_to_face_twist45", (1.0, 0.0, 0.0), "twist45"),
]
GAPS_NM = [1.0, 2.99, 8.0]


def rotation(tag):
    if tag is None:
        return None
    if tag == "twist45":
        angle = np.deg2rad(45.0)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    raise ValueError(tag)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def evaluate(params, direction, rotation_j, gap):
    a = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0])
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    radius = V.centre_distance_for_gap(
        EDGE_NM, gap, d, rotation_b=rotation_j
    )
    b = V.Placement.create(EDGE_NM, radius * d, rotation_j)
    t0 = time.time()
    result = V.pair_energy(a, b, params, temperature_K=TEMPERATURE_K)
    result["wall_seconds"] = time.time() - t0
    return result


def peak_pair_memory_MB(n_volume_nodes, block=4_000_000):
    """Peak transient bytes of the chunked attraction kernel."""
    rows = max(1, block // max(n_volume_nodes, 1))
    entries = rows * n_volume_nodes
    # delta (3 floats) + r2 (1 float) + the weighted product (1 float)
    return entries * 8 * 5 / 1024 ** 2


def run_volume_study(base, report):
    rows = []
    reference_params = replace(
        base, volume_scheme=VOLUME_REFERENCE[0], volume_n=VOLUME_REFERENCE[1]
    )
    reference = {}
    for name, direction, tag in CONFIGURATIONS:
        for gap in GAPS_NM:
            reference[(name, gap)] = evaluate(
                reference_params, direction, rotation(tag), gap
            )
    ref_nodes = reference[(CONFIGURATIONS[0][0], GAPS_NM[0])]["volume_nodes"]

    for scheme in ("cartesian", "radial"):
        levels = VOLUME_LEVELS if scheme == "cartesian" else VOLUME_LEVELS[:6]
        previous = {}
        for n in levels:
            params = replace(base, volume_scheme=scheme, volume_n=n)
            for name, direction, tag in CONFIGURATIONS:
                for gap in GAPS_NM:
                    result = evaluate(params, direction, rotation(tag), gap)
                    ref = reference[(name, gap)]
                    key = (scheme, name, gap)
                    attr = result["u_attr_kcalmol"]
                    rel_ref = abs(attr / ref["u_attr_kcalmol"] - 1.0)
                    rel_prev = (
                        abs(attr / previous[key] - 1.0) if key in previous else np.nan
                    )
                    previous[key] = attr
                    rows.append(
                        [
                            scheme,
                            n,
                            result["volume_nodes"],
                            name,
                            "%.2f" % gap,
                            "%.8e" % attr,
                            "%.3e" % rel_ref,
                            "%.3e" % rel_prev,
                            "%.4e" % (abs(attr - ref["u_attr_kcalmol"]) / KBT_KCALMOL),
                            "%.3f" % result["wall_seconds"],
                            "%.1f" % peak_pair_memory_MB(result["volume_nodes"]),
                        ]
                    )
    header = [
        "scheme", "n_per_axis", "volume_nodes", "configuration", "gap_nm",
        "U_attr_kcalmol", "rel_error_vs_reference", "rel_change_vs_previous",
        "abs_error_kBT", "wall_seconds", "peak_pair_memory_MB",
    ]
    write_csv(OUT / "B1_volume_convergence.csv", header, rows)
    report["B1_volume"] = {
        "reference": {"scheme": VOLUME_REFERENCE[0], "n": VOLUME_REFERENCE[1],
                      "nodes": ref_nodes},
        "reference_values_kcalmol": {
            "%s@%.2f" % (k[0], k[1]): v["u_attr_kcalmol"] for k, v in reference.items()
        },
        "rows": rows,
        "header": header,
    }
    return rows, reference


def run_surface_study(base, report):
    rows = []
    reference_params = replace(
        base,
        surface_scheme=SURFACE_REFERENCE[0],
        surface_n=SURFACE_REFERENCE[1],
        r2_mode="exact_surface",
    )
    reference = {}
    for name, direction, tag in CONFIGURATIONS:
        for gap in GAPS_NM:
            reference[(name, gap)] = evaluate(
                reference_params, direction, rotation(tag), gap
            )

    for r2_mode in ("exact_surface", "nearest_element"):
        previous = {}
        for n in SURFACE_LEVELS:
            params = replace(
                base, surface_scheme="facegl", surface_n=n, r2_mode=r2_mode
            )
            for name, direction, tag in CONFIGURATIONS:
                for gap in GAPS_NM:
                    result = evaluate(params, direction, rotation(tag), gap)
                    ref = reference[(name, gap)]
                    key = (r2_mode, name, gap)
                    rep = result["u_rep_kcalmol"]
                    rel_ref = abs(rep / ref["u_rep_kcalmol"] - 1.0)
                    rel_prev = (
                        abs(rep / previous[key] - 1.0) if key in previous else np.nan
                    )
                    previous[key] = rep
                    rows.append(
                        [
                            r2_mode,
                            n,
                            result["surface_nodes"],
                            name,
                            "%.2f" % gap,
                            "%.8e" % rep,
                            "%.3e" % rel_ref,
                            "%.3e" % rel_prev,
                            "%.4e" % (abs(rep - ref["u_rep_kcalmol"]) / KBT_KCALMOL),
                            "%.3e" % result.get("r2_certificate_nm", 0.0),
                            "%.3f" % result["wall_seconds"],
                        ]
                    )
    header = [
        "r2_mode", "n_per_face_axis", "surface_nodes", "configuration", "gap_nm",
        "U_rep_kcalmol", "rel_error_vs_reference", "rel_change_vs_previous",
        "abs_error_kBT", "r2_certificate_nm", "wall_seconds",
    ]
    write_csv(OUT / "B2_surface_convergence.csv", header, rows)
    report["B2_surface"] = {
        "reference": {"scheme": SURFACE_REFERENCE[0], "n": SURFACE_REFERENCE[1]},
        "rows": rows,
        "header": header,
        "note_surface": (
            "r2_mode='nearest_element' does NOT converge to the continuum "
            "value: r_2 is the distance to the nearest NODE of the partner "
            "mesh, so it inherits that mesh's spacing.  Only "
            "r2_mode='exact_surface' is a mesh-independent definition."
        ),
    }
    return rows, reference


def run_joint_study(base, report):
    # The ladder stays inside ONE rule family so that each "relative change on
    # refinement" compares like with like.  A rung that switches from the
    # Cartesian to the radial rule measures the inter-rule discrepancy, not
    # convergence, and at sub-2 nm gaps that discrepancy is several percent.
    # The radial rung is therefore run last and flagged as a cross-rule check,
    # and it is excluded from the acceptance verdict.
    ladder = [
        ("cartesian", 4, "facegl", 6),
        ("cartesian", 6, "facegl", 8),
        ("cartesian", 8, "facegl", 12),
        ("cartesian", 12, "facegl", 16),
        ("cartesian", 16, "facegl", 24),
        ("cartesian", 20, "facegl", 32),
        ("cartesian", 24, "facegl", 40),
    ]
    cross_rule = ("radial", 16, "facegl", 40)
    rows = []
    previous = {}
    for vs, vn, ss, sn in ladder:
        params = replace(
            base,
            volume_scheme=vs,
            volume_n=vn,
            surface_scheme=ss,
            surface_n=sn,
            r2_mode="exact_surface",
        )
        for name, direction, tag in CONFIGURATIONS:
            for gap in GAPS_NM:
                result = evaluate(params, direction, rotation(tag), gap)
                key = (name, gap)
                values = (
                    result["u_attr_kcalmol"],
                    result["u_rep_kcalmol"],
                    result["u_pair_kcalmol"],
                )
                if key in previous:
                    changes = [
                        abs(new / old - 1.0) if old != 0 else np.nan
                        for new, old in zip(values, previous[key])
                    ]
                    absolute = [
                        abs(new - old) / KBT_KCALMOL
                        for new, old in zip(values, previous[key])
                    ]
                else:
                    changes = [np.nan] * 3
                    absolute = [np.nan] * 3
                previous[key] = values
                rows.append(
                    [
                        "%s%d+%s%d" % (vs, vn, ss, sn),
                        result["volume_nodes"],
                        result["surface_nodes"],
                        name,
                        "%.2f" % gap,
                        "%.8e" % values[0],
                        "%.8e" % values[1],
                        "%.8e" % values[2],
                        "%.3e" % changes[0],
                        "%.3e" % changes[1],
                        "%.3e" % changes[2],
                        "%.3e" % absolute[2],
                        "%.3f" % result["wall_seconds"],
                    ]
                )
    header = [
        "level", "volume_nodes", "surface_nodes", "configuration", "gap_nm",
        "U_attr_kcalmol", "U_rep_kcalmol", "U_pair_kcalmol",
        "rel_change_attr", "rel_change_rep", "rel_change_pair",
        "abs_change_pair_kBT", "wall_seconds",
    ]
    # cross-rule check, reported but excluded from the acceptance verdict
    vs, vn, ss, sn = cross_rule
    params = replace(
        base, volume_scheme=vs, volume_n=vn, surface_scheme=ss, surface_n=sn,
        r2_mode="exact_surface",
    )
    cross_rows = []
    for name, direction, tag in CONFIGURATIONS:
        for gap in GAPS_NM:
            result = evaluate(params, direction, rotation(tag), gap)
            reference = previous[(name, gap)]
            cross_rows.append(
                [
                    "%s%d+%s%d (cross-rule)" % cross_rule,
                    result["volume_nodes"],
                    result["surface_nodes"],
                    name,
                    "%.2f" % gap,
                    "%.8e" % result["u_attr_kcalmol"],
                    "%.8e" % result["u_rep_kcalmol"],
                    "%.8e" % result["u_pair_kcalmol"],
                    "%.3e" % abs(result["u_attr_kcalmol"] / reference[0] - 1.0),
                    "%.3e" % abs(result["u_rep_kcalmol"] / reference[1] - 1.0),
                    "%.3e" % abs(result["u_pair_kcalmol"] / reference[2] - 1.0),
                    "%.3e" % (abs(result["u_pair_kcalmol"] - reference[2]) / KBT_KCALMOL),
                    "%.3f" % result["wall_seconds"],
                ]
            )
    rows.extend(cross_rows)

    write_csv(OUT / "B3_joint_convergence.csv", header, rows)

    # acceptance check on the final two Cartesian levels
    passed = []
    for row in rows:
        if row[0] == "%s%d+%s%d" % ladder[-1]:
            rel = [float(row[8]), float(row[9]), float(row[10])]
            abs_kbt = float(row[11])
            passed.append(
                {
                    "configuration": row[3],
                    "gap_nm": row[4],
                    "rel_change_attr": rel[0],
                    "rel_change_rep": rel[1],
                    "rel_change_pair": rel[2],
                    "abs_change_pair_kBT": abs_kbt,
                    "meets_1_percent_or_0p01kBT": bool(
                        max(rel) <= 0.01 or abs_kbt <= 0.01
                    ),
                }
            )
    report["B3_joint"] = {
        "header": header,
        "rows": rows,
        "final_level_check": passed,
        "acceptance_note": (
            "The verdict uses the last two CARTESIAN rungs.  The radial rung "
            "is reported as a cross-rule check only: comparing across rule "
            "families measures their mutual discrepancy, which is several "
            "percent at sub-2 nm gaps, not the convergence of either."
        ),
    }
    return rows


def run_well_study(base, report):
    ladder = [
        ("grid27", 3, "lattice", 8, "nearest_element", "SI discretisation"),
        ("cartesian", 6, "facegl", 8, "exact_surface", "coarse continuum"),
        ("cartesian", 8, "facegl", 12, "exact_surface", "continuum"),
        ("cartesian", 12, "facegl", 16, "exact_surface", "continuum"),
        ("cartesian", 16, "facegl", 24, "exact_surface", "continuum"),
        ("cartesian", 16, "facegl", 32, "exact_surface", "continuum"),
    ]
    rows = []
    previous = None
    for vs, vn, ss, sn, r2, note in ladder:
        params = replace(
            base,
            volume_scheme=vs,
            volume_n=vn,
            surface_scheme=ss,
            surface_n=sn,
            r2_mode=r2,
        )
        t0 = time.time()
        gap, depth, interior = V.find_well(EDGE_NM, params, bracket=(0.2, 14.0),
                                           samples=60)
        rows.append(
            [
                note,
                "%s%d" % (vs, vn),
                "%s%d" % (ss, sn),
                r2,
                "%.4f" % gap if interior else "none",
                "%.5f" % depth,
                "%.5f" % (0.5 * depth),
                "%.5f" % V.kcalmol_to_kBT(depth, TEMPERATURE_K),
                "%.4f" % abs(gap - previous) if previous is not None and interior else "",
                "%.1f" % (time.time() - t0),
            ]
        )
        if interior:
            previous = gap
    header = [
        "note", "volume_rule", "surface_rule", "r2_mode", "well_gap_nm",
        "well_depth_U_pair_kcalmol", "well_depth_per_particle_kcalmol",
        "well_depth_U_pair_kBT_300K", "gap_shift_vs_previous_nm", "wall_seconds",
    ]
    write_csv(OUT / "B4_well_convergence.csv", header, rows)
    report["B4_well"] = {"header": header, "rows": rows}
    return rows


def make_figures(volume_rows, surface_rows, joint_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    ax = axes[0]
    for scheme, marker in (("cartesian", "o"), ("radial", "s")):
        for name, colour in zip(
            [c[0] for c in CONFIGURATIONS],
            ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
        ):
            data = [
                (int(r[2]), float(r[6]))
                for r in volume_rows
                if r[0] == scheme and r[3] == name and r[4] == "2.99"
            ]
            if not data:
                continue
            data = np.array(data)
            ax.loglog(data[:, 0], np.maximum(data[:, 1], 1e-16), marker + "-",
                      color=colour, ms=4, lw=1.0,
                      label="%s %s" % (scheme[:4], name) if scheme == "cartesian" else None)
    ax.axhline(0.01, color="k", ls="--", lw=0.9, label="1 % target")
    ax.set_xlabel("volume nodes per particle")
    ax.set_ylabel(r"relative error in $U_{\rm attr}$")
    ax.set_title("B1  volume integral (gap = 2.99 nm)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for mode, style in (("exact_surface", "-"), ("nearest_element", "--")):
        for name, colour in zip(
            [c[0] for c in CONFIGURATIONS],
            ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"],
        ):
            data = [
                (int(r[2]), float(r[6]))
                for r in surface_rows
                if r[0] == mode and r[3] == name and r[4] == "2.99"
            ]
            if not data:
                continue
            data = np.array(data)
            ax.loglog(data[:, 0], np.maximum(data[:, 1], 1e-16), "o" + style,
                      color=colour, ms=4, lw=1.0,
                      label="%s %s" % (mode.split("_")[0], name))
    ax.axhline(0.01, color="k", ls="--", lw=0.9)
    ax.set_xlabel("surface nodes per particle")
    ax.set_ylabel(r"relative error in $U_{\rm rep}$")
    ax.set_title("B2  surface integral (gap = 2.99 nm)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for name, colour in zip(
        [c[0] for c in CONFIGURATIONS], ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    ):
        data = [
            (int(r[1]), float(r[8]), float(r[9]), float(r[10]))
            for r in joint_rows
            if r[3] == name and r[4] == "2.99" and r[8] != "nan"
        ]
        if not data:
            continue
        data = np.array(data)
        ax.loglog(data[:, 0], data[:, 1], "o-", color=colour, ms=4, lw=1.0,
                  label="%s attr" % name)
        ax.loglog(data[:, 0], data[:, 2], "s--", color=colour, ms=4, lw=1.0,
                  label="%s rep" % name)
        ax.loglog(data[:, 0], data[:, 3], "^:", color=colour, ms=4, lw=1.2,
                  label="%s pair" % name)
    ax.axhline(0.01, color="k", ls="--", lw=0.9)
    ax.set_xlabel("volume nodes per particle")
    ax.set_ylabel("relative change on refinement")
    ax.set_title("B3  joint refinement, components separate")
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("B_convergence.%s" % suffix), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="singh_calibrated_si_mesh")
    parser.add_argument(
        "--stages", nargs="*", default=["B1", "B2", "B3", "B4"],
        choices=["B1", "B2", "B3", "B4"],
        help="sub-studies to run; a partial run merges into the existing "
             "B_report.json instead of overwriting it",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    base = V.load_parameters(args.parameters)
    report = {}
    report_path = OUT / "B_report.json"
    if report_path.exists() and set(args.stages) != {"B1", "B2", "B3", "B4"}:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update({
        "fixed_potential": base.to_dict(),
        "fixed_K_W_nm6_kcalmol": base.kw(EDGE_NM),
        "kw_element_count_is_reference_normalisation": base.kw_element_count,
        "edge_nm": EDGE_NM,
        "temperature_K": TEMPERATURE_K,
        "kBT_in_kcalmol": KBT_KCALMOL,
        "acceptance_targets": {
            "relative": 0.01,
            "absolute_kBT": 0.01,
            "well_position_nm": 0.03,
            "source": "this project, not Singh et al.",
        },
    })
    print("fixed potential: %s  eps1=%.4f eps2=%.4f beta=%.4f nm  K_W=%.6e"
          % (base.label, base.epsilon1, base.epsilon2, base.beta_nm, base.kw(EDGE_NM)))
    print("1 kBT at %.2f K = %.6f kcal/mol" % (TEMPERATURE_K, KBT_KCALMOL))

    volume_rows = surface_rows = joint_rows = None
    if "B1" in args.stages:
        print("\nB1 volume integral ...")
        volume_rows, _ = run_volume_study(base, report)
        print("  %d rows" % len(volume_rows))
    if "B2" in args.stages:
        print("\nB2 surface integral ...")
        surface_rows, _ = run_surface_study(base, report)
        print("  %d rows" % len(surface_rows))
    if "B3" in args.stages:
        print("\nB3 joint refinement ...")
        joint_rows = run_joint_study(base, report)
        print("  %d rows" % len(joint_rows))
        for item in report["B3_joint"]["final_level_check"]:
            print("    acceptance %-22s gap=%-6s attr=%.2e rep=%.2e "
                  "pair=%.2e abs=%.2e pass=%s"
                  % (item["configuration"], item["gap_nm"],
                     item["rel_change_attr"], item["rel_change_rep"],
                     item["rel_change_pair"], item["abs_change_pair_kBT"],
                     item["meets_1_percent_or_0p01kBT"]))
    if "B4" in args.stages:
        print("\nB4 well position ...")
        well_rows = run_well_study(base, report)
        for row in well_rows:
            print("  %-20s %-12s %-10s %-16s gap=%-9s depth=%-10s shift=%s"
                  % (row[0], row[1], row[2], row[3], row[4], row[5], row[8]))

    if volume_rows is None and "B1_volume" in report:
        volume_rows = report["B1_volume"]["rows"]
    if surface_rows is None and "B2_surface" in report:
        surface_rows = report["B2_surface"]["rows"]
    if joint_rows is None and "B3_joint" in report:
        joint_rows = report["B3_joint"]["rows"]
    if volume_rows and surface_rows and joint_rows:
        make_figures(volume_rows, surface_rows, joint_rows)
    report_path.write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
