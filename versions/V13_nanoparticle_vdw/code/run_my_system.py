"""Stage C: the present particle system.

Parameters are READ FROM THE EXISTING CODE (V10/V6.20 geometry_model.py), not
retyped: edge length, Hamaker constant, reference temperature, rhombohedral
lattice (a, alpha, gamma) and the V6.20 reference gap.

Everything is reported as the FULL pair energy U_pair, with U_pair/2 alongside
for the isolated two-particle per-particle comparison.  No neighbour sums are
taken here.

Parameter transfer plans (SI parameters were fitted at a = 13.37 nm):
  plan1  eps1, eps2, beta keep their source values; K_W follows the reference
         formula eps1 (A/pi^2)(0.9 a^3/27)^2 with THIS system's a and A.
  plan2  additionally beta -> beta * (a / 13.37), a geometric-similarity
         sensitivity test.  This is NOT a statement that the ligand shell
         thickens with particle size; the oleate chain length is fixed.
  plan3  eps1, eps2, beta keep their source values, the attraction follows
         this system's a and A as in plan 1, but the REPULSIVE prefactor
         eps2*K_W is frozen at the reference size and reference Hamaker
         constant.  Motivation: K_W ~ a^6 and the surface integral already
         carries a^2, so plan 1 gives E_rep ~ a^8, which a ligand shell of
         fixed chain length cannot do.  Plan 3 leaves E_rep ~ a^2.
Overall energy enhancement uses a single dimensionless factor s applied to the
COMPLETE potential (both terms), s = 1, 2, 5.  eps1 and eps2 are never
enlarged separately and then described as an overall multiple.

Outputs land in outputs/my_system/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "my_system"
LEGACY_CODE = ROOT.parents[0] / "V10_multiphysics_nanocube" / "code"

EDGE_REF_NM = 13.37
SIZES_NM = [13.0, 16.0, 19.5]
FOCUS_EDGE_NM = 16.0
FOCUS_GAP_NM = 3.0
TEMPERATURES_K = [298.15, 300.0]
SCALES = [1.0, 2.0, 5.0]

DIRECTIONS = {
    "100": (1.0, 0.0, 0.0),
    "110": (1.0, 1.0, 0.0),
    "111": (1.0, 1.0, 1.0),
}

# Production mesh established by stage B: at gaps >= 2.99 nm this meets the
# 1 % per-component target on two successive refinements (attraction 0.20 %,
# repulsion 2e-12, pair 0.37 %).  Sub-2 nm gaps need more volume nodes; see
# outputs/convergence/.
CONVERGED = dict(
    volume_scheme="cartesian",
    volume_n=20,
    surface_scheme="facegl",
    surface_n=32,
    r2_mode="exact_surface",
)


def load_legacy():
    sys.path.insert(0, str(LEGACY_CODE))
    import geometry_model as legacy

    return legacy


def rotation_about(axis, degrees):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


ROTATIONS = {
    "co_oriented": None,
    "twist45_about_axis": rotation_about([1.0, 0.0, 0.0], 45.0),
    "tilt15_about_z": rotation_about([0.0, 0.0, 1.0], 15.0),
    "rot90_about_z": rotation_about([0.0, 0.0, 1.0], 90.0),
}


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def base_parameters(source, edge_nm, hamaker_J, plan, scale=1.0, label=None,
                    mesh=None):
    """Load a parameter set and transfer it to this system.

    The discretisation is NOT overridden by default.  Each calibrated set's
    constants were fitted on a specific mesh, so evaluating them on a
    different mesh changes the physics.  Pass ``mesh`` explicitly only when
    the point of the run is to isolate a quadrature effect.
    """
    params = V.load_parameters(source)
    if mesh:
        params = replace(params, **mesh)
    params = replace(
        params,
        label=label or ("%s_%s" % (source, plan)),
        overall_scale=scale,
    ).with_hamaker_J(hamaker_J)
    if plan == "plan2":
        params = replace(params, beta_nm=params.beta_nm * edge_nm / EDGE_REF_NM)
    elif plan == "plan3":
        # Freeze the repulsive prefactor at the reference size and reference
        # Hamaker constant.  K_W enters only E_rep, and K_W ~ a^6 while the
        # surface integral already carries a^2, so plan 1 makes the steric
        # term scale as a^8.  A ligand shell of fixed chain length cannot do
        # that.  Freezing K_W leaves E_rep ~ a^2, the natural area scaling.
        params = replace(
            params,
            kw_reference_edge_nm=EDGE_REF_NM,
            kw_reference_hamaker_kcal_per_mol=units_reference_hamaker(),
        )
    elif plan != "plan1":
        raise ValueError(plan)
    return params


def units_reference_hamaker():
    """The Hamaker constant the SI parameters were fitted with, 3 kcal/mol."""
    return V.SINGH_HAMAKER_KCAL_PER_MOL


def evaluate(edge_nm, gap_nm, params, direction, rotation_j, temperature_K):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    radius = V.centre_distance_for_gap(edge_nm, gap_nm, d, rotation_b=rotation_j)
    a = V.Placement.create(edge_nm, [0.0, 0.0, 0.0])
    b = V.Placement.create(edge_nm, radius * d, rotation_j)
    return V.pair_energy(a, b, params, temperature_K=temperature_K)


# ------------------------------------------------------------------ stage C0
def stage_c0(report):
    legacy = load_legacy()
    p = legacy.PARAMS
    reference_gap_m = legacy.v620_reference_gap_m(p)
    read = {
        "source_file": str(LEGACY_CODE / "geometry_model.py"),
        "particle_size_nm": p.particle_size_nm,
        "hamaker_J": p.hamaker_J,
        "hamaker_kcal_per_mol": V.J_to_kcalmol(p.hamaker_J),
        "voxel_count_per_axis": p.voxel_count_per_axis,
        "roundness_nm": p.roundness_nm,
        "reference_a_nm": p.reference_a_nm,
        "reference_alpha_deg": p.reference_alpha_deg,
        "reference_gamma_deg": p.reference_gamma_deg,
        "reference_temperature_K": p.reference_temperature_K,
        "v620_reference_gap_nm": reference_gap_m * 1e9,
        "saturation_magnetization_Apm": p.saturation_magnetization_Apm,
        "legacy_particle_volume_sharp_cube_nm3": p.particle_size_nm ** 3,
        "superellipsoid_volume_nm3": V.Superellipsoid(
            p.particle_size_nm
        ).volume_exact_nm3,
        "volume_ratio_superellipsoid_over_sharp_cube": V.exact_volume_coefficient(),
    }
    report["C0_parameters_read_from_code"] = read
    return legacy, read


# ------------------------------------------------------------------ stage C1
def stage_c1(report, sources):
    """Centre distances: superellipsoid vs the legacy rounded cube."""
    legacy = load_legacy()
    rows = []
    for edge in SIZES_NM:
        legacy_params = replace(legacy.PARAMS, particle_size_nm=edge)
        for name, direction in DIRECTIONS.items():
            d = np.asarray(direction, float)
            d = d / np.linalg.norm(d)
            new = V.centre_distance_for_gap(edge, FOCUS_GAP_NM, d)
            old = legacy.center_distance_at_gap_m(d, FOCUS_GAP_NM * 1e-9,
                                                  legacy_params) * 1e9
            rows.append(
                [
                    "%.2f" % edge,
                    name,
                    "%.6f" % new,
                    "%.6f" % old,
                    "%+.6f" % (new - old),
                    "%.6f" % (2.0 * V.analytic_support_height(edge, d)),
                    "%.6f" % (new / edge),
                ]
            )
    header = [
        "edge_nm", "direction", "R_superellipsoid_nm", "R_legacy_rounded_cube_nm",
        "difference_nm", "2h(d)_superellipsoid_nm", "R/a",
    ]
    write_csv(OUT / "C1_centre_distances_gap3nm.csv", header, rows)
    report["C1_centre_distances"] = {
        "header": header,
        "rows": rows,
        "note": (
            "Every 'same gap' comparison below re-solves the centre distance "
            "with the sextic superellipsoid.  Legacy rounded-cube centre "
            "distances are shown only to size the change."
        ),
    }
    return rows


# ------------------------------------------------------------------ stage C2
def stage_c2(report, sources, hamaker_J):
    """Energies at the focus gap for all sizes, directions, plans, scales."""
    rows = []
    for source in sources:
        for plan in ("plan1", "plan2", "plan3"):
            for edge in SIZES_NM:
                params = base_parameters(source, edge, hamaker_J, plan)
                for name, direction in DIRECTIONS.items():
                    for rot_name, rot in ROTATIONS.items():
                        if rot_name != "co_oriented" and name != "100":
                            continue
                        for temperature in TEMPERATURES_K:
                            result = evaluate(
                                edge, FOCUS_GAP_NM, params, direction, rot,
                                temperature,
                            )
                            for scale in SCALES:
                                rows.append(
                                    [
                                        source,
                                        plan,
                                        "%.2f" % edge,
                                        name,
                                        rot_name,
                                        "%.2f" % temperature,
                                        "%.1f" % scale,
                                        "%.4f" % params.beta_nm,
                                        "%.6e" % params.kw(edge),
                                        "%.6f" % result["centre_distance_nm"],
                                        "%.6e" % (scale * result["u_attr_kcalmol"]),
                                        "%.6e" % (scale * result["u_rep_kcalmol"]),
                                        "%.6e" % (scale * result["u_pair_kcalmol"]),
                                        "%.6e" % (scale * result["u_pair_J"]),
                                        "%.6e" % (scale * result["u_pair_kBT"]),
                                        "%.6e" % (0.5 * scale * result["u_pair_kBT"]),
                                    ]
                                )
    header = [
        "parameter_set", "transfer_plan", "edge_nm", "direction", "orientation",
        "temperature_K", "scale_s", "beta_nm", "K_W_nm6_kcalmol",
        "centre_distance_nm", "U_attr_kcalmol", "U_rep_kcalmol",
        "U_pair_kcalmol", "U_pair_J", "U_pair_kBT", "U_pair_per_particle_kBT",
    ]
    write_csv(OUT / "C2_energies_at_gap3nm.csv", header, rows)
    report["C2_focus"] = {"header": header, "n_rows": len(rows)}

    highlight = []
    for source in sources:
      for plan in ("plan1", "plan3"):
        params = base_parameters(source, FOCUS_EDGE_NM, hamaker_J, plan)
        for name, direction in DIRECTIONS.items():
            result = evaluate(FOCUS_EDGE_NM, FOCUS_GAP_NM, params, direction,
                              None, 298.15)
            highlight.append(
                {
                    "parameter_set": source,
                    "transfer_plan": plan,
                    "edge_nm": FOCUS_EDGE_NM,
                    "gap_nm": FOCUS_GAP_NM,
                    "direction": name,
                    "centre_distance_nm": result["centre_distance_nm"],
                    "U_attr_kcalmol": result["u_attr_kcalmol"],
                    "U_rep_kcalmol": result["u_rep_kcalmol"],
                    "U_pair_kcalmol": result["u_pair_kcalmol"],
                    "U_attr_J": result["u_attr_J"],
                    "U_rep_J": result["u_rep_J"],
                    "U_pair_J": result["u_pair_J"],
                    "U_pair_kBT_298p15": result["u_pair_kBT"],
                    "U_pair_per_particle_kBT_298p15": 0.5 * result["u_pair_kBT"],
                }
            )
    report["C2_highlight_16nm_3nm"] = highlight
    return rows, highlight


# ------------------------------------------------------------------ stage C3
def stage_c3(report, sources, hamaker_J):
    """Face-to-face well position and depth per size, plan, parameter set."""
    rows = []
    for source in sources:
        for plan in ("plan1", "plan2", "plan3"):
            for edge in SIZES_NM:
                params = base_parameters(source, edge, hamaker_J, plan)
                gap, depth, interior = V.find_well(
                    edge, params, direction=(1, 0, 0), bracket=(0.2, 16.0),
                    samples=48,
                )
                rows.append(
                    [
                        source,
                        plan,
                        "%.2f" % edge,
                        "%.4f" % params.beta_nm,
                        "%.4f" % gap if interior else "none",
                        "%.4f" % (gap / edge) if interior else "",
                        "%.5f" % depth,
                        "%.5f" % (0.5 * depth),
                        "%.5f" % V.kcalmol_to_kBT(depth, 298.15),
                        "%.5e" % V.kcalmol_to_J(depth),
                    ]
                )
    header = [
        "parameter_set", "transfer_plan", "edge_nm", "beta_nm",
        "well_gap_nm", "well_gap_over_a", "well_depth_U_pair_kcalmol",
        "well_depth_per_particle_kcalmol", "well_depth_U_pair_kBT_298p15",
        "well_depth_U_pair_J",
    ]
    write_csv(OUT / "C3_face_to_face_wells.csv", header, rows)
    report["C3_wells"] = {"header": header, "rows": rows}
    return rows


# ------------------------------------------------------------------ stage C4
def stage_c4(report, sources, hamaker_J):
    """Size comparison at fixed physical gap and at fixed reduced gap D/a."""
    rows = []
    reduced = FOCUS_GAP_NM / FOCUS_EDGE_NM
    for source in sources:
        params_by_edge = {
            (edge, plan): base_parameters(source, edge, hamaker_J, plan)
            for edge in SIZES_NM
            for plan in ("plan1", "plan3")
        }
        for edge in SIZES_NM:
          for plan in ("plan1", "plan3"):
            for mode, gap in (
                ("fixed_physical_gap_3nm", FOCUS_GAP_NM),
                ("fixed_reduced_gap_D_over_a", reduced * edge),
            ):
                result = evaluate(edge, gap, params_by_edge[(edge, plan)],
                                  (1, 0, 0), None, 298.15)
                rows.append(
                    [
                        source,
                        plan,
                        mode,
                        "%.2f" % edge,
                        "%.4f" % gap,
                        "%.5f" % (gap / edge),
                        "%.6f" % result["centre_distance_nm"],
                        "%.6e" % result["u_attr_kcalmol"],
                        "%.6e" % result["u_rep_kcalmol"],
                        "%.6e" % result["u_pair_kcalmol"],
                        "%.6e" % result["u_pair_kBT"],
                    ]
                )
    header = [
        "parameter_set", "transfer_plan", "comparison_mode", "edge_nm",
        "gap_nm", "gap_over_a",
        "centre_distance_nm", "U_attr_kcalmol", "U_rep_kcalmol",
        "U_pair_kcalmol", "U_pair_kBT_298p15",
    ]
    write_csv(OUT / "C4_size_comparison.csv", header, rows)
    report["C4_size"] = {"header": header, "rows": rows,
                         "reduced_gap_D_over_a": reduced}
    return rows


# ------------------------------------------------------------------ stage C5
def stage_c5(report, hamaker_J, calibrated_source):
    """Decompose the change from the legacy model, one effect at a time."""
    legacy = load_legacy()
    edge = FOCUS_EDGE_NM
    d = np.array([1.0, 0.0, 0.0])
    legacy_params = replace(legacy.PARAMS, particle_size_nm=edge)

    R_legacy = legacy.center_distance_at_gap_m(
        d, FOCUS_GAP_NM * 1e-9, legacy_params
    ) * 1e9
    R_new = V.centre_distance_for_gap(edge, FOCUS_GAP_NM, d)

    steps = []

    def record(name, value_J, note):
        steps.append(
            {
                "step": name,
                "U_pair_J": value_J,
                "U_pair_kcalmol": V.J_to_kcalmol(value_J),
                "U_pair_kBT_298p15": V.J_to_kBT(value_J, 298.15),
                "note": note,
            }
        )

    record(
        "0_legacy_as_is",
        legacy.pair_vdw_energy_J(R_legacy * 1e-9 * d, legacy_params),
        "sharp-cube 4^3 voxels, V = a^3, Minkowski rounded-cube gap, no eps1, "
        "no repulsion; R = %.4f nm" % R_legacy,
    )
    record(
        "1_legacy_at_superellipsoid_R",
        legacy.pair_vdw_energy_J(R_new * 1e-9 * d, legacy_params),
        "same legacy energy function evaluated at the superellipsoid centre "
        "distance R = %.4f nm: isolates the geometry change in the gap "
        "definition alone" % R_new,
    )

    bare = replace(
        V.load_parameters("singh_literature"),
        label="bare_hamaker",
        epsilon1=1.0,
        epsilon2=0.0,
        volume_scheme="cartesian",
        volume_n=4,
        surface_scheme="facegl",
        surface_n=8,
        r2_mode="exact_surface",
        kw_volume_convention="exact",
    ).with_hamaker_J(hamaker_J)
    record(
        "2_superellipsoid_shape_4x4x4",
        evaluate(edge, FOCUS_GAP_NM, bare, d, None, 298.15)["u_pair_J"],
        "superellipsoid body and volume (0.90096 a^3), 4^3 exact-boundary "
        "quadrature, eps1 = 1, no repulsion: isolates the shape change",
    )
    converged_bare = replace(bare, volume_n=20, surface_n=32)
    record(
        "3_converged_quadrature",
        evaluate(edge, FOCUS_GAP_NM, converged_bare, d, None, 298.15)["u_pair_J"],
        "same physics, 20^3 volume quadrature: isolates integration accuracy",
    )

    calibrated = V.load_parameters("singh_calibrated_continuum")
    with_eps1 = replace(
        converged_bare, epsilon1=calibrated.epsilon1, label="with_eps1"
    )
    record(
        "4_plus_eps1_scaling",
        evaluate(edge, FOCUS_GAP_NM, with_eps1, d, None, 298.15)["u_pair_J"],
        "eps1 = %.4f applied (energy-scale constant of the effective "
        "potential): isolates the eps1 rescaling" % calibrated.epsilon1,
    )
    full = base_parameters("singh_calibrated_continuum", edge, hamaker_J,
                           "plan1", mesh=CONVERGED)
    record(
        "5_plus_repulsion",
        evaluate(edge, FOCUS_GAP_NM, full, d, None, 298.15)["u_pair_J"],
        "steric repulsion added (eps2 = %.4f, beta = %.4f nm): isolates the "
        "repulsive term" % (full.epsilon2, full.beta_nm),
    )
    plan3 = base_parameters("singh_calibrated_continuum", edge, hamaker_J,
                            "plan3", mesh=CONVERGED)
    record(
        "5b_plan3_frozen_repulsion_prefactor",
        evaluate(edge, FOCUS_GAP_NM, plan3, d, None, 298.15)["u_pair_J"],
        "transfer plan 3: the repulsive prefactor eps2*K_W frozen at the "
        "13.37 nm reference instead of following a^6, so E_rep scales with "
        "area rather than a^8",
    )
    steps.append(
        {
            "step": "6_per_particle_convention",
            "U_pair_J": 0.5 * steps[-1]["U_pair_J"],
            "U_pair_kcalmol": 0.5 * steps[-1]["U_pair_kcalmol"],
            "U_pair_kBT_298p15": 0.5 * steps[-1]["U_pair_kBT_298p15"],
            "note": "U_pair / 2, the per-particle energy of an isolated "
                    "two-particle system.  A pure factor of two, listed here so "
                    "it is never confused with a physical effect.",
        }
    )

    for index in range(1, len(steps)):
        steps[index]["delta_from_previous_kBT"] = (
            steps[index]["U_pair_kBT_298p15"] - steps[index - 1]["U_pair_kBT_298p15"]
        )
    write_csv(
        OUT / "C5_effect_decomposition.csv",
        ["step", "U_pair_J", "U_pair_kcalmol", "U_pair_kBT_298p15",
         "delta_from_previous_kBT", "note"],
        [
            [s["step"], "%.6e" % s["U_pair_J"], "%.6e" % s["U_pair_kcalmol"],
             "%.6f" % s["U_pair_kBT_298p15"],
             "%.6f" % s.get("delta_from_previous_kBT", 0.0), s["note"]]
            for s in steps
        ],
    )
    report["C5_decomposition"] = steps
    return steps


# ------------------------------------------------------------------ stage C6
SI_MESH = dict(
    volume_scheme="grid27",
    volume_n=3,
    surface_scheme="lattice",
    surface_n=8,
    r2_mode="nearest_element",
)


def stage_c6(report, hamaker_J):
    """Each calibrated set evaluated on BOTH meshes: the constants are mesh bound.

    The SI's 27-element volume sum truncates the near-surface divergence of
    the Hamaker double integral, because the closest element centres never
    come nearer than a/3.  That truncation is what lets a bounded
    (r2+beta)^-8 repulsion hold off the attraction and create a minimum.  In
    the converged integral the attraction diverges like 1/D^2 as D -> 0 while
    the repulsion stays bounded, so no (eps1, eps2, beta) produces a well and
    only the hard core sets a contact distance.
    """
    rows = []
    for source in ("singh_calibrated_si_mesh", "singh_calibrated_continuum",
                   "singh_literature"):
        for mesh_name, mesh in (("SI_27_386", SI_MESH), ("converged", CONVERGED)):
            params = base_parameters(source, FOCUS_EDGE_NM, hamaker_J, "plan1",
                                     mesh=mesh)
            result = evaluate(FOCUS_EDGE_NM, FOCUS_GAP_NM, params, (1, 0, 0),
                              None, 298.15)
            gap, depth, interior = V.find_well(
                FOCUS_EDGE_NM, params, bracket=(0.2, 16.0), samples=40
            )
            rows.append(
                [
                    source,
                    mesh_name,
                    result["volume_nodes"],
                    result["surface_nodes"],
                    "%.6e" % result["u_attr_kcalmol"],
                    "%.6e" % result["u_rep_kcalmol"],
                    "%.6e" % result["u_pair_kcalmol"],
                    "%.4f" % result["u_pair_kBT"],
                    "%.4f" % gap if interior else "none",
                    "%.5f" % depth if interior else "",
                ]
            )
    header = [
        "parameter_set", "mesh", "volume_nodes", "surface_nodes",
        "U_attr_kcalmol", "U_rep_kcalmol", "U_pair_kcalmol",
        "U_pair_kBT_298p15", "well_gap_nm", "well_depth_kcalmol",
    ]
    write_csv(OUT / "C6_mesh_sensitivity_of_constants.csv", header, rows)
    report["C6_mesh_sensitivity"] = {"header": header, "rows": rows}
    return rows


# ------------------------------------------------------------------- figures
def make_figures(sources, hamaker_J):
    OUT.mkdir(parents=True, exist_ok=True)
    gaps = np.concatenate([np.arange(0.3, 8.0, 0.08), np.arange(8.0, 25.01, 0.3)])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    colours = {13.0: "#1f77b4", 16.0: "#d62728", 19.5: "#2ca02c"}
    source = sources[0]
    ax = axes[0]
    for edge in SIZES_NM:
        params = base_parameters(source, edge, hamaker_J, "plan3")
        kernels = V.energy_kernels(edge, gaps, params)
        a, r, t = V.kernel_energies(kernels, params.epsilon1, params.epsilon2,
                                    params.beta_nm)
        kbt = np.array([V.kcalmol_to_kBT(x, 298.15) for x in t])
        ax.plot(gaps, kbt, color=colours[edge], lw=1.7, label="a = %.1f nm" % edge)
        ax.plot(gaps, [V.kcalmol_to_kBT(x, 298.15) for x in a], color=colours[edge],
                lw=0.8, ls="--")
        ax.plot(gaps, [V.kcalmol_to_kBT(x, 298.15) for x in r], color=colours[edge],
                lw=0.8, ls=":")
    ax.axvline(FOCUS_GAP_NM, color="0.6", lw=0.9)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlim(0, 16)
    ax.set_xlabel("surface separation D [nm]")
    ax.set_ylabel(r"$U$ [$k_{\rm B}T$, 298.15 K]")
    ax.set_title("Face-to-face [100], %s, transfer plan 3\n"
                 "(solid pair, dashed attr, dotted rep)" % source)
    ax.legend(fontsize=8)

    ax = axes[1]
    styles = {"100": "-", "110": "--", "111": ":"}
    for name, direction in DIRECTIONS.items():
        params = base_parameters(source, FOCUS_EDGE_NM, hamaker_J, "plan3")
        kernels = V.energy_kernels(FOCUS_EDGE_NM, gaps, params, direction=direction)
        _, _, t = V.kernel_energies(kernels, params.epsilon1, params.epsilon2,
                                    params.beta_nm)
        ax.plot(gaps, [V.kcalmol_to_kBT(x, 298.15) for x in t], styles[name],
                color="#d62728", lw=1.7, label="[%s]" % name)
    ax.axvline(FOCUS_GAP_NM, color="0.6", lw=0.9)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlim(0, 16)
    ax.set_xlabel("surface separation D [nm]")
    ax.set_ylabel(r"$U_{\rm pair}$ [$k_{\rm B}T$]")
    ax.set_title("16 nm, centre line along [100]/[110]/[111], plan 3")
    ax.legend(fontsize=8)

    ax = axes[2]
    for edge in SIZES_NM:
        params = base_parameters(source, edge, hamaker_J, "plan3")
        kernels = V.energy_kernels(edge, gaps, params)
        _, _, t = V.kernel_energies(kernels, params.epsilon1, params.epsilon2,
                                    params.beta_nm)
        ax.plot(gaps / edge, [V.kcalmol_to_kBT(x, 298.15) for x in t],
                color=colours[edge], lw=1.7, label="a = %.1f nm" % edge)
    ax.axvline(FOCUS_GAP_NM / FOCUS_EDGE_NM, color="0.6", lw=0.9)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("reduced separation D / a")
    ax.set_ylabel(r"$U_{\rm pair}$ [$k_{\rm B}T$]")
    ax.set_title("Same curves at fixed reduced separation, plan 3")
    ax.legend(fontsize=8)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("C_my_system.%s" % suffix), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+",
        default=["singh_calibrated_si_mesh", "singh_calibrated_continuum",
                 "singh_literature"],
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    report = {}
    print("=" * 78)
    print("C0  parameters read from the existing code")
    print("=" * 78)
    legacy, read = stage_c0(report)
    for key, value in read.items():
        print("  %-48s %s" % (key, value))
    hamaker_J = read["hamaker_J"]

    print()
    print("C1  centre distances at a 3 nm gap (superellipsoid vs legacy)")
    rows = stage_c1(report, args.sources)
    print("  %-9s %-10s %14s %14s %12s" % ("edge", "dir", "R_new", "R_legacy", "diff"))
    for row in rows:
        print("  %-9s %-10s %14s %14s %12s" % (row[0], row[1], row[2], row[3], row[4]))

    print()
    print("C2  energies at the focus gap")
    _, highlight = stage_c2(report, args.sources, hamaker_J)
    for item in highlight:
        print("  %-28s %-6s [%s]  U_attr=%+.4f  U_rep=%+.4f  U_pair=%+.4f "
              "kcal/mol = %+.3f kBT"
              % (item["parameter_set"], item["transfer_plan"], item["direction"],
                 item["U_attr_kcalmol"], item["U_rep_kcalmol"],
                 item["U_pair_kcalmol"], item["U_pair_kBT_298p15"]))

    print()
    print("C3  face-to-face wells")
    for row in stage_c3(report, args.sources, hamaker_J):
        print("  %-30s %-6s a=%-6s beta=%-8s gap=%-9s depth=%-11s (%s kBT)"
              % (row[0], row[1], row[2], row[3], row[4], row[6], row[8]))

    print()
    print("C4  fixed physical gap vs fixed reduced gap")
    stage_c4(report, args.sources, hamaker_J)

    print()
    print("C5  effect decomposition versus the legacy model")
    for step in stage_c5(report, hamaker_J, args.sources[0]):
        print("  %-34s U_pair = %+10.4f kBT   delta = %+9.4f"
              % (step["step"], step["U_pair_kBT_298p15"],
                 step.get("delta_from_previous_kBT", 0.0)))

    print()
    print("C6  the same constants on both meshes")
    for row in stage_c6(report, hamaker_J):
        print("  %-30s %-10s V=%-6s S=%-6s U_pair=%-14s kBT=%-10s well=%s"
              % (row[0], row[1], row[2], row[3], row[6], row[7], row[8]))

    make_figures(args.sources, hamaker_J)
    (OUT / "C_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
