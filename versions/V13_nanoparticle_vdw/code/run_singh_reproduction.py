"""Stage A: reproduce the Singh et al. Fig. S28E effective vdW potential.

Three things happen here, kept strictly separate:

  A1  the literature form with the literature parameters, run as printed,
      with NO fitting of any kind;
  A2  a diagnostic scan over every choice the SI leaves open (volume node
      layout, surface element areas, r_2 definition, pair vs per-particle
      convention) to test whether any of them recovers the published well;
  A3  a clearly labelled CALIBRATION of (eps1, eps2, beta) against the
      digitised Fig. S28E curve, at the literature discretisation and again
      in the continuum limit.  Calibration is NOT validation: it is reported
      with its residuals and with the adjusted parameter values.

Outputs land in outputs/singh_reproduction/.
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
from scipy.optimize import least_squares

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "singh_reproduction"
REFERENCE = ROOT / "outputs" / "reference" / "fig_s28e_digitised.csv"

EDGE_REF_NM = 13.37
TEMPERATURE_REF_K = 300.0
SI_WELL_GAP_NM = 2.99
SI_WELL_DEPTH_KCALMOL = -2.33

CONTINUUM = dict(
    volume_scheme="radial",
    volume_n=14,
    surface_scheme="facegl",
    surface_n=32,
    r2_mode="exact_surface",
)


# ---------------------------------------------------------------- utilities
def load_reference():
    rows = []
    with REFERENCE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    float(row["R_nm"]),
                    float(row["E_kcalmol_per_NC_axis"]),
                    float(row["halfwidth_kcalmol"]),
                )
            )
    data = np.array(rows)
    return data


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def log_slope(x, y):
    """d ln|y| / dx by central differences."""
    return np.gradient(np.log(np.abs(y)), x)


# ------------------------------------------------------------------ stage A1
def stage_a1(report):
    params = V.load_parameters("singh_literature")
    shape = V.Superellipsoid(EDGE_REF_NM)
    gaps = np.concatenate(
        [np.arange(0.2, 6.0, 0.05), np.arange(6.0, 25.01, 0.2)]
    )
    t0 = time.time()
    kernels = V.energy_kernels(EDGE_REF_NM, gaps, params)
    attraction, repulsion, total = V.kernel_energies(
        kernels, params.epsilon1, params.epsilon2, params.beta_nm
    )
    elapsed = time.time() - t0

    volume_nodes, volume_weights = shape.volume_nodes(
        params.volume_scheme, params.volume_n, params.kw_volume_convention
    )
    surface_nodes, surface_areas = shape.surface_nodes(
        params.surface_scheme, params.surface_n, params.surface_refine,
        params.surface_area_mode,
    )

    report["A1_audit"] = {
        "edge_nm": EDGE_REF_NM,
        "epsilon1": params.epsilon1,
        "epsilon2": params.epsilon2,
        "beta_nm": params.beta_nm,
        "hamaker_kcal_per_mol": params.hamaker_kcal_per_mol,
        "hamaker_J": params.hamaker_J,
        "K_W_nm6_kcalmol": params.kw(EDGE_REF_NM),
        "K_W_SI_quoted": 2.5e5,
        "volume_exact_nm3": shape.volume_exact_nm3,
        "volume_literature_0p9a3_nm3": shape.volume_literature_nm3,
        "volume_coefficient_exact": V.exact_volume_coefficient(),
        "volume_literature_minus_exact_percent": 100.0
        * (shape.volume_literature_nm3 / shape.volume_exact_nm3 - 1.0),
        "volume_node_count": len(volume_nodes),
        "volume_weight_sum_nm3": float(volume_weights.sum()),
        "surface_node_count": len(surface_nodes),
        "surface_area_nm2": float(surface_areas.sum()),
        "surface_area_sharp_cube_6a2_nm2": 6.0 * EDGE_REF_NM**2,
        "surface_area_sphere_nm2": float(4.0 * np.pi * (EDGE_REF_NM / 2) ** 2),
        "evaluation_ms_per_gap": 1e3 * elapsed / len(gaps),
    }

    index = int(np.argmin(total))
    interior = 0 < index < len(gaps) - 1
    report["A1_result"] = {
        "minimum_gap_nm": float(gaps[index]),
        "minimum_U_pair_kcalmol": float(total[index]),
        "interior_minimum": bool(interior),
        "U_attr_at_2p99_kcalmol": float(np.interp(2.99, gaps, attraction)),
        "U_rep_at_2p99_kcalmol": float(np.interp(2.99, gaps, repulsion)),
        "U_pair_at_2p99_kcalmol": float(np.interp(2.99, gaps, total)),
        "U_pair_per_particle_at_2p99_kcalmol": 0.5
        * float(np.interp(2.99, gaps, total)),
        "SI_target_pair_or_per_NC_kcalmol": SI_WELL_DEPTH_KCALMOL,
    }

    # why there is no well: compare the logarithmic slopes of the two terms
    window = (gaps > 1.5) & (gaps < 8.0)
    slope_a = log_slope(gaps, attraction)[window]
    slope_r = log_slope(gaps, repulsion)[window]
    report["A1_slope_diagnosis"] = {
        "comment": (
            "A well requires the repulsion to decay faster than the attraction "
            "(|dln E_rep/dg| > |dln E_attr/dg|) below the well and slower above "
            "it.  With beta = 9.56 nm the two logarithmic slopes are nearly "
            "equal over the whole relevant range, so E_rep is essentially a "
            "constant multiple of E_attr and the sum never turns around."
        ),
        "gap_window_nm": [1.5, 8.0],
        "mean_attr_log_slope_per_nm": float(np.mean(slope_a)),
        "mean_rep_log_slope_per_nm": float(np.mean(slope_r)),
        "mean_difference_per_nm": float(np.mean(slope_r - slope_a)),
        "max_abs_difference_per_nm": float(np.max(np.abs(slope_r - slope_a))),
    }

    write_csv(
        OUT / "A1_literature_parameters_curve.csv",
        [
            "gap_nm",
            "centre_distance_nm",
            "U_attr_kcalmol",
            "U_rep_kcalmol",
            "U_pair_kcalmol",
            "U_pair_per_particle_kcalmol",
            "U_pair_J",
            "U_pair_kBT_300K",
        ],
        [
            [
                "%.4f" % g,
                "%.6f" % (g + EDGE_REF_NM),
                "%.6e" % a,
                "%.6e" % r,
                "%.6e" % t,
                "%.6e" % (0.5 * t),
                "%.6e" % V.kcalmol_to_J(t),
                "%.6e" % V.kcalmol_to_kBT(t, TEMPERATURE_REF_K),
            ]
            for g, a, r, t in zip(gaps, attraction, repulsion, total)
        ],
    )
    return params, gaps, (attraction, repulsion, total)


# ------------------------------------------------------------------ stage A2
def stage_a2(report):
    base = V.load_parameters("singh_literature")
    reference = load_reference()
    mask = (reference[:, 0] > 1.3) & (reference[:, 0] < 24.0)
    ref_r, ref_e = reference[mask, 0], reference[mask, 1]

    variants = []
    for volume_scheme, volume_n, vlabel in [
        ("grid27", 3, "27 elements (SI)"),
        ("cartesian", 16, "continuum volume"),
    ]:
        for area_mode in ["cube_param", "equal_solid_angle"]:
            for r2_mode in ["nearest_element", "exact_surface", "global_min"]:
                variants.append(
                    (
                        replace(
                            base,
                            label="scan",
                            volume_scheme=volume_scheme,
                            volume_n=volume_n,
                            surface_area_mode=area_mode,
                            r2_mode=r2_mode,
                        ),
                        vlabel,
                        area_mode,
                        r2_mode,
                    )
                )

    gaps = np.concatenate([np.arange(0.2, 6.0, 0.1), np.arange(6.0, 24.01, 0.4)])
    rows = []
    for params, vlabel, area_mode, r2_mode in variants:
        kernels = V.energy_kernels(EDGE_REF_NM, gaps, params)
        attraction, repulsion, total = V.kernel_energies(
            kernels, params.epsilon1, params.epsilon2, params.beta_nm
        )
        index = int(np.argmin(total))
        interior = 0 < index < len(gaps) - 1
        model = np.interp(ref_r, gaps, total)
        rms_pair = float(np.sqrt(np.mean((model - ref_e) ** 2)))
        rms_half = float(np.sqrt(np.mean((0.5 * model - ref_e) ** 2)))
        rows.append(
            [
                vlabel,
                area_mode,
                r2_mode,
                "%.3f" % gaps[index] if interior else "none",
                "%.4f" % total[index] if interior else "%.4f(edge)" % total[index],
                "%.4f" % np.interp(2.99, gaps, attraction),
                "%.4f" % np.interp(2.99, gaps, repulsion),
                "%.4f" % np.interp(2.99, gaps, total),
                "%.3f" % rms_pair,
                "%.3f" % rms_half,
            ]
        )
    header = [
        "volume_discretisation",
        "surface_area_mode",
        "r2_mode",
        "interior_well_gap_nm",
        "well_depth_kcalmol",
        "U_attr_at_2.99",
        "U_rep_at_2.99",
        "U_pair_at_2.99",
        "rms_vs_figure_as_pair",
        "rms_vs_figure_as_per_particle",
    ]
    write_csv(OUT / "A2_ambiguity_scan.csv", header, rows)
    report["A2_scan"] = {
        "header": header,
        "rows": rows,
        "conclusion": (
            "No combination of the SI's open choices produces an interior "
            "minimum with (eps1, eps2, beta) = (130, 290, 9.56).  Reading the "
            "figure ordinate as U_pair or as U_pair/2 changes the residual by "
            "a factor of two but never creates a well, so the pair vs "
            "per-particle convention is not the discrepancy."
        ),
    }
    return rows


# ------------------------------------------------------------------ stage A3
def calibrate(params, label, provenance, report_key, report):
    reference = load_reference()
    mask = (reference[:, 0] > 1.3) & (reference[:, 0] < 24.0)
    ref_r, ref_e, ref_w = reference[mask, 0], reference[mask, 1], reference[mask, 2]
    fit_gaps = ref_r[::8]
    fit_values = ref_e[::8]

    # Calibration targets, per the SI: the published curve PLUS the stated
    # well position and depth.  The last two enter as a depth residual at
    # R = 2.99 nm and a stationarity residual dU/dR = 0 there, evaluated by a
    # central difference on two extra kernel gaps.
    delta = 0.05
    extra = np.array([SI_WELL_GAP_NM - delta, SI_WELL_GAP_NM, SI_WELL_GAP_NM + delta])
    all_gaps = np.concatenate([fit_gaps, extra])
    weights = np.concatenate([np.ones(len(fit_gaps)), np.zeros(3)])
    depth_weight = 8.0
    slope_weight = 40.0

    t0 = time.time()
    kernels = V.energy_kernels(EDGE_REF_NM, all_gaps, params)
    build = time.time() - t0
    n_curve = len(fit_gaps)

    def residual(theta):
        _, _, total = V.kernel_energies(kernels, theta[0], theta[1], theta[2])
        curve = total[:n_curve] - fit_values
        depth = depth_weight * (total[n_curve + 1] - SI_WELL_DEPTH_KCALMOL)
        slope = slope_weight * (total[n_curve + 2] - total[n_curve]) / (2 * delta)
        return np.concatenate([curve, [depth, slope]])

    best = None
    for start in ([40.0, 125.0, 7.4], [130.0, 290.0, 9.56], [10.0, 50.0, 4.0]):
        try:
            solution = least_squares(
                residual,
                start,
                bounds=([1e-3, 1e-3, 1e-3], [1e5, 1e8, 60.0]),
                xtol=1e-14,
                ftol=1e-14,
            )
        except Exception:
            continue
        if best is None or solution.cost < best.cost:
            best = solution
    theta = best.x
    jac = best.jac
    curve_residual = best.fun[:n_curve]
    covariance = np.linalg.inv(jac.T @ jac) * np.mean(curve_residual**2)
    sigma = np.sqrt(np.diag(covariance))

    calibrated = replace(
        params,
        label=label,
        epsilon1=float(theta[0]),
        epsilon2=float(theta[1]),
        beta_nm=float(theta[2]),
        provenance=provenance,
    )
    V.save_parameters(calibrated, label)

    gaps = np.concatenate([np.arange(0.2, 6.0, 0.05), np.arange(6.0, 25.01, 0.2)])
    curve_kernels = V.energy_kernels(EDGE_REF_NM, gaps, calibrated)
    attraction, repulsion, total = V.kernel_energies(
        curve_kernels, theta[0], theta[1], theta[2]
    )
    index = int(np.argmin(total))
    interior = 0 < index < len(gaps) - 1
    from scipy.optimize import minimize_scalar

    if interior:
        refine = minimize_scalar(
            lambda g: V.kernel_energies(
                V.energy_kernels(EDGE_REF_NM, [g], calibrated), *theta
            )[2][0],
            bounds=(gaps[index - 1], gaps[index + 1]),
            method="bounded",
            options={"xatol": 1e-5},
        )
        well_gap, well_depth = float(refine.x), float(refine.fun)
    else:
        well_gap, well_depth = float(gaps[index]), float(total[index])

    model_on_reference = np.interp(ref_r, gaps, total)
    report[report_key] = {
        "label": label,
        "epsilon1": float(theta[0]),
        "epsilon1_sigma": float(sigma[0]),
        "epsilon2": float(theta[1]),
        "epsilon2_sigma": float(sigma[1]),
        "beta_nm": float(theta[2]),
        "beta_sigma_nm": float(sigma[2]),
        "literature_values": {"epsilon1": 130.0, "epsilon2": 290.0, "beta_nm": 9.56},
        "ratio_to_literature": {
            "epsilon1": 130.0 / float(theta[0]),
            "epsilon2": 290.0 / float(theta[1]),
            "beta_nm": 9.56 / float(theta[2]),
        },
        "K_W_nm6_kcalmol": calibrated.kw(EDGE_REF_NM),
        "rms_residual_kcalmol": float(np.sqrt(np.mean(curve_residual**2))),
        "max_residual_kcalmol": float(np.max(np.abs(curve_residual))),
        "n_curve_points": int(n_curve),
        "well_targets_in_fit": {"depth_weight": depth_weight, "slope_weight": slope_weight},
        "median_digitisation_halfwidth_kcalmol": float(np.median(ref_w)),
        "rms_over_full_reference_kcalmol": float(
            np.sqrt(np.mean((model_on_reference - ref_e) ** 2))
        ),
        "well_gap_nm": well_gap,
        "well_depth_U_pair_kcalmol": well_depth,
        "well_depth_U_pair_per_particle_kcalmol": 0.5 * well_depth,
        "well_depth_U_pair_kBT_300K": V.kcalmol_to_kBT(well_depth, 300.0),
        "well_depth_U_pair_J": V.kcalmol_to_J(well_depth),
        "SI_well_gap_nm": SI_WELL_GAP_NM,
        "SI_well_depth_kcalmol": SI_WELL_DEPTH_KCALMOL,
        "SI_well_depth_kBT_300K": V.kcalmol_to_kBT(SI_WELL_DEPTH_KCALMOL, 300.0),
        "gap_error_nm": well_gap - SI_WELL_GAP_NM,
        "depth_error_kcalmol": well_depth - SI_WELL_DEPTH_KCALMOL,
        "kernel_build_seconds": build,
        "discretisation": {
            "volume_scheme": calibrated.volume_scheme,
            "volume_n": calibrated.volume_n,
            "surface_scheme": calibrated.surface_scheme,
            "surface_n": calibrated.surface_n,
            "r2_mode": calibrated.r2_mode,
            "surface_area_mode": calibrated.surface_area_mode,
        },
    }
    write_csv(
        OUT / ("%s_curve.csv" % label),
        ["gap_nm", "U_attr_kcalmol", "U_rep_kcalmol", "U_pair_kcalmol",
         "U_pair_per_particle_kcalmol", "U_pair_kBT_300K", "figure_kcalmol"],
        [
            ["%.4f" % g, "%.6e" % a, "%.6e" % r, "%.6e" % t, "%.6e" % (0.5 * t),
             "%.6e" % V.kcalmol_to_kBT(t, 300.0), "%.6e" % np.interp(g, reference[:, 0], reference[:, 1])]
            for g, a, r, t in zip(gaps, attraction, repulsion, total)
        ],
    )
    return calibrated, gaps, (attraction, repulsion, total)


# ------------------------------------------------------------------- figures
def make_figures(lit, calibrated_lit, calibrated_cont, reference):
    OUT.mkdir(parents=True, exist_ok=True)
    gaps_l, (al, rl, tl) = lit
    gaps_c, (ac, rc, tc) = calibrated_lit
    gaps_k, (ak, rk, tk) = calibrated_cont

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    ax.plot(reference[:, 0], reference[:, 1], color="0.25", lw=3.2, alpha=0.35,
            label="Fig. S28E (digitised)")
    ax.plot(gaps_c, tc, color="#1f77b4", lw=1.8, label="calibrated, SI discretisation")
    ax.plot(gaps_c, ac, color="#1f77b4", lw=1.0, ls="--", label="  attraction")
    ax.plot(gaps_c, rc, color="#1f77b4", lw=1.0, ls=":", label="  repulsion")
    ax.plot(gaps_k, tk, color="#2ca02c", lw=1.8, label="calibrated, continuum")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot([SI_WELL_GAP_NM], [SI_WELL_DEPTH_KCALMOL], "r*", ms=14, zorder=5,
            label="SI minimum (2.99, -2.33)")
    ax.set_xlim(0, 25)
    ax.set_ylim(-5, 15)
    ax.set_xlabel("surface separation R [nm]")
    ax.set_ylabel(r"$U_{\rm pair}$ [kcal/mol]")
    ax.set_title("Calibrated reproduction of Fig. S28E ($a$ = 13.37 nm)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(reference[:, 0], reference[:, 1], color="0.25", lw=3.2, alpha=0.35,
            label="Fig. S28E (digitised)")
    ax.plot(gaps_l, tl, color="#d62728", lw=1.8,
            label=r"literature $\epsilon_1$=130, $\epsilon_2$=290, $\beta$=9.56 nm")
    ax.plot(gaps_l, al, color="#d62728", lw=1.0, ls="--", label="  attraction")
    ax.plot(gaps_l, rl, color="#d62728", lw=1.0, ls=":", label="  repulsion")
    ax.plot(gaps_l, 0.5 * tl, color="#ff7f0e", lw=1.2, ls="-.",
            label="literature $U_{\\rm pair}/2$")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot([SI_WELL_GAP_NM], [SI_WELL_DEPTH_KCALMOL], "r*", ms=14, zorder=5)
    ax.set_xlim(0, 25)
    ax.set_ylim(-60, 20)
    ax.set_xlabel("surface separation R [nm]")
    ax.set_ylabel(r"$U$ [kcal/mol]")
    ax.set_title("Literature parameters, run as printed: no minimum")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("A_fig_s28e_reproduction.%s" % suffix), dpi=170)
    plt.close(fig)

    # log-slope diagnosis figure
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    window = (gaps_l > 1.0) & (gaps_l < 12.0)
    ax.plot(gaps_l[window], -log_slope(gaps_l, al)[window], color="#d62728",
            label=r"attraction, $-d\ln|U_{\rm attr}|/dR$")
    ax.plot(gaps_l[window], -log_slope(gaps_l, rl)[window], color="#1f77b4",
            label=r"repulsion, $\beta$ = 9.56 nm (literature)")
    window_c = (gaps_c > 1.0) & (gaps_c < 12.0)
    ax.plot(gaps_c[window_c], -log_slope(gaps_c, rc)[window_c],
            color="#2ca02c", label=r"repulsion, calibrated $\beta$")
    ax.set_xlabel("surface separation R [nm]")
    ax.set_ylabel(r"logarithmic decay rate [nm$^{-1}$]")
    ax.set_title("Why (130, 290, 9.56 nm) cannot make a well")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("A_log_slope_diagnosis.%s" % suffix), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-continuum", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    print("=" * 78)
    print("STAGE A1  literature form, literature parameters, no fitting")
    print("=" * 78)
    lit_params, gaps_l, curves_l = stage_a1(report)
    audit = report["A1_audit"]
    for key in (
        "K_W_nm6_kcalmol", "volume_exact_nm3", "volume_literature_0p9a3_nm3",
        "volume_literature_minus_exact_percent", "surface_area_nm2",
        "volume_node_count", "surface_node_count", "evaluation_ms_per_gap",
    ):
        print("  %-42s %s" % (key, audit[key]))
    for key, value in report["A1_result"].items():
        print("  %-42s %s" % (key, value))
    print("  log-slope diagnosis: attr %.4f /nm, rep %.4f /nm, difference %.4f /nm"
          % (report["A1_slope_diagnosis"]["mean_attr_log_slope_per_nm"],
             report["A1_slope_diagnosis"]["mean_rep_log_slope_per_nm"],
             report["A1_slope_diagnosis"]["mean_difference_per_nm"]))

    print()
    print("=" * 78)
    print("STAGE A2  scan over every choice the SI leaves open")
    print("=" * 78)
    rows = stage_a2(report)
    print("  %-20s %-18s %-15s %10s %12s %10s" % (
        "volume", "area mode", "r2 mode", "well gap", "depth", "U@2.99"))
    for row in rows:
        print("  %-20s %-18s %-15s %10s %12s %10s" % (
            row[0], row[1], row[2], row[3], row[4], row[7]))
    print("  ->", report["A2_scan"]["conclusion"])

    print()
    print("=" * 78)
    print("STAGE A3  CALIBRATION against the digitised Fig. S28E curve")
    print("=" * 78)
    cal_lit, gaps_c, curves_c = calibrate(
        lit_params,
        "singh_calibrated_si_mesh",
        "CALIBRATED, not literature.  Eqn. 4 form and the SI's 27-element / "
        "386-element discretisation retained; (eps1, eps2, beta) least-squares "
        "fitted to the digitised Fig. S28E curve of Singh et al. "
        "Literature values were (130, 290, 9.56 nm) and do NOT reproduce that "
        "curve.  Do not cite these numbers as the published parameters.",
        "A3_calibrated_si_mesh",
        report,
    )
    for key in ("epsilon1", "epsilon2", "beta_nm", "rms_residual_kcalmol",
                "well_gap_nm", "well_depth_U_pair_kcalmol", "gap_error_nm",
                "depth_error_kcalmol", "ratio_to_literature"):
        print("  %-38s %s" % (key, report["A3_calibrated_si_mesh"][key]))

    if args.skip_continuum:
        cal_cont, gaps_k, curves_k = cal_lit, gaps_c, curves_c
    else:
        print()
        print("  continuum-limit calibration (same potential, refined quadrature)")
        cont_params = replace(lit_params, **CONTINUUM)
        cal_cont, gaps_k, curves_k = calibrate(
            cont_params,
            "singh_calibrated_continuum",
            "CALIBRATED, not literature.  Eqn. 4 form with converged "
            "quadrature (radial volume rule n=14, 6x32^2 surface rule, exact "
            "point-to-body r_2); (eps1, eps2, beta) fitted to the digitised "
            "Fig. S28E curve.  Separate from singh_calibrated_si_mesh because "
            "the SI's coarse 27-element sum is far from the continuum limit.",
            "A3_calibrated_continuum",
            report,
        )
        for key in ("epsilon1", "epsilon2", "beta_nm", "rms_residual_kcalmol",
                    "well_gap_nm", "well_depth_U_pair_kcalmol"):
            print("  %-38s %s" % (key, report["A3_calibrated_continuum"][key]))

    reference = load_reference()
    make_figures(
        (gaps_l, curves_l), (gaps_c, curves_c), (gaps_k, curves_k), reference
    )

    (OUT / "A_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
