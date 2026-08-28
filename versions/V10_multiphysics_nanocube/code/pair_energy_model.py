"""Reduced Brownian-rotation correction to the V10 SAXS-window pair model.

Every temperature is evaluated independently from a uniform 64-state prior,
as in ``protocol.py``.  The fixed-face branch retains that
Néel-only result because a cube cannot rotate relative to its neighbour while
remaining face registered.  For the tip branch, the fraction that does not
undergo an inter-well Néel event during the effective 20 s observation window
is allowed to rotate
as a rigid cube.  Its locked <111> axis is driven into head-to-tail alignment by
dipolar torque.  The free-particle Brownian time is calculated explicitly.

This is a deliberately reduced model.  It tests whether rigid-body Brownian
rotation alone is sufficient; it does not calculate the ligand/contact barrier
for the actual face-to-tip structural conversion.  The constrained-face branch
is written as its canonical pair energy plus a finite-time Neel relaxation
deficit and the magnetic energy lost by suppressing rigid-cube rotation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.constants import Boltzmann
from scipy.interpolate import PchipInterpolator


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.8,
        "axes.grid": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = (
    ROOT
    / "versions"
    / "V10_multiphysics_nanocube"
    / "code"
    / "protocol.py"
)
OUTPUT_DIR = ROOT / "versions" / "V10_multiphysics_nanocube" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIGAND_LENGTH_NM = 1.5
HYDRODYNAMIC_DIAMETER_M = (16.0 + 2.0 * LIGAND_LENGTH_NM) * 1.0e-9
FACE_SURFACE_GAP_NM = 3.0
TIP_SURFACE_GAP_NM = 2.0 * LIGAND_LENGTH_NM
EXPERIMENTAL_MIN_K = 233.15
EXPERIMENTAL_MAX_K = 293.15
TRANSIENT_MIN_K = 253.15
TRANSIENT_MAX_K = 273.15
EXPERIMENTAL_WINDOW_S = 20.0

# Saturated-liquid n-hexane viscosities from Table 4 of Michailidou et al.,
# J. Phys. Chem. Ref. Data 42, 033104 (2013), DOI 10.1063/1.4818980.
# The tabulated reference values have a stated 2% expanded uncertainty.  A
# quadratic in reciprocal temperature reproduces these values to <0.31% and
# provides a smooth continuation from 250 K down to the 233 K experimental
# limit (values below 250 K are therefore an extrapolation of the table).
HEXANE_VISCOSITY_T_K = np.arange(250.0, 351.0, 10.0)
HEXANE_VISCOSITY_UPA_S = np.array(
    [514.4, 452.7, 401.8, 359.2, 323.1, 292.1,
     265.2, 241.7, 221.0, 202.6, 186.2]
)
HEXANE_LOG_VISCOSITY_FIT = np.polyfit(
    1.0 / HEXANE_VISCOSITY_T_K,
    np.log(HEXANE_VISCOSITY_UPA_S * 1.0e-6),
    deg=2,
)

MS_BASIS = (
    "present-particle magnetometry at 300 K: 55 emu/g = 55 A m2/kg; "
    "converted with rho(Fe3O4)=5.18e3 kg/m3"
)
ATTEMPT_TIME_REFERENCE = (
    "Moreno et al., Phys. Rev. B 112, 024429 (2025), "
    "DOI 10.1103/vmwp-q427"
)
HAMAKER_REFERENCE = (
    "Faure et al., Langmuir 27, 8659-8664 (2011), "
    "DOI 10.1021/la201387d"
)
VISCOSITY_REFERENCE = (
    "Michailidou et al., J. Phys. Chem. Ref. Data 42, 033104 "
    "(2013), DOI 10.1063/1.4818980"
)


def load_protocol_module():
    spec = importlib.util.spec_from_file_location(
        "v10_17_for_brownian_rotation", PROTOCOL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def geometry_data(
    v620, direction: np.ndarray, surface_gap_nm: float
) -> dict:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    distance_m = v620.center_distance_at_gap_m(
        direction, surface_gap_nm * 1.0e-9, v620.PARAMS
    )
    vector_m = distance_m * direction
    energies_J, _, states, _ = v620.easy_axis_pair_energies_J(
        vector_m, v620.PARAMS
    )
    easy_111 = np.ones(3) / np.sqrt(3.0)
    easy_index = int(np.argmax(states @ easy_111))
    same_111_index = 8 * easy_index + easy_index
    return {
        "distance_m": distance_m,
        "surface_gap_nm": surface_gap_nm,
        "vdw_J": v620.pair_vdw_energy_J(vector_m, v620.PARAMS),
        "minimum_Edd_J": float(np.min(energies_J)),
        "same_111_Edd_J": float(energies_J[same_111_index]),
        "Edd_levels_J": energies_J,
    }


def neel_statistics(v620, temperatures_K: np.ndarray) -> dict:
    factors, weights = v620.barrier_distribution(v620.PARAMS)
    tau_all_s = np.empty((len(temperatures_K), len(factors)))
    for index, temperature_K in enumerate(temperatures_K):
        exponent = (
            v620.PARAMS.zfc_fc_activation_barrier_J
            * factors
            / (Boltzmann * temperature_K)
        )
        tau_all_s[index] = v620.PARAMS.attempt_time_s * np.exp(
            np.minimum(exponent, 700.0)
        )
    # Survival probability of an easy-axis basin at the midpoint of the
    # 150 s SAXS integration interval.
    blocked_fraction = np.sum(
        weights[None, :]
        * np.exp(-EXPERIMENTAL_WINDOW_S / tau_all_s),
        axis=1,
    )
    geometric_mean_tau_s = np.exp(
        np.sum(weights[None, :] * np.log(tau_all_s), axis=1)
    )
    return {
        "blocked_fraction": blocked_fraction,
        "geometric_mean_tau_s": geometric_mean_tau_s,
    }


def hexane_viscosity_Pa_s(temperatures_K: np.ndarray) -> np.ndarray:
    """Temperature-dependent saturated-liquid n-hexane viscosity."""
    temperatures_K = np.asarray(temperatures_K, dtype=float)
    if np.any(temperatures_K <= 0.0):
        raise ValueError("temperatures_K must be positive")
    return np.exp(
        np.polyval(HEXANE_LOG_VISCOSITY_FIT, 1.0 / temperatures_K)
    )


def brownian_time_s(temperatures_K: np.ndarray) -> np.ndarray:
    hydrodynamic_volume_m3 = (
        np.pi * HYDRODYNAMIC_DIAMETER_M**3 / 6.0
    )
    viscosity_Pa_s = hexane_viscosity_Pa_s(temperatures_K)
    return (
        3.0
        * viscosity_Pa_s
        * hydrodynamic_volume_m3
        / (Boltzmann * temperatures_K)
    )


def crossings_K(x: np.ndarray, difference: np.ndarray) -> list[float]:
    order = np.argsort(x)
    x_ordered = x[order]
    y_ordered = difference[order]
    roots = []
    for index in range(len(x_ordered) - 1):
        y0 = y_ordered[index]
        y1 = y_ordered[index + 1]
        if y0 == 0.0:
            roots.append(float(x_ordered[index]))
        elif y0 * y1 < 0.0:
            fraction = -y0 / (y1 - y0)
            roots.append(
                float(
                    x_ordered[index]
                    + fraction * (x_ordered[index + 1] - x_ordered[index])
                )
            )
    return roots


def main():
    protocol = load_protocol_module()
    # Use the midpoint-time approximation for a 150 s SAXS integration.
    protocol.EXPOSURE_DURATION_S = EXPERIMENTAL_WINDOW_S
    protocol.EXPOSURE_SUBSTEP_S = 5.0
    v620 = protocol.load_v620_module()
    temperatures_K, segments, frames = protocol.build_protocol()
    face_dynamic = protocol.propagate_geometry(
        v620,
        np.array([1.0, 0.0, 0.0]),
        segments,
        len(frames),
        surface_gap_nm=FACE_SURFACE_GAP_NM,
    )
    tip_dynamic = protocol.propagate_geometry(
        v620,
        np.ones(3) / np.sqrt(3.0),
        segments,
        len(frames),
        surface_gap_nm=TIP_SURFACE_GAP_NM,
    )
    face_geometry = geometry_data(
        v620, np.array([1.0, 0.0, 0.0]), FACE_SURFACE_GAP_NM
    )
    tip_geometry = geometry_data(
        v620, np.ones(3) / np.sqrt(3.0), TIP_SURFACE_GAP_NM
    )

    neel = neel_statistics(v620, temperatures_K)
    viscosity_Pa_s = hexane_viscosity_Pa_s(temperatures_K)
    tau_B_s = brownian_time_s(temperatures_K)
    brownian_alignment_fraction = 1.0 - np.exp(
        -EXPERIMENTAL_WINDOW_S / tau_B_s
    )
    intrawell_q = tip_dynamic["exposure_end_q"]
    tip_locked_aligned_J = (
        intrawell_q**2 * tip_geometry["minimum_Edd_J"]
    )

    face_neel_limited_J = face_dynamic["exposure_mean_Edd_J"]
    tip_neel_only_J = tip_dynamic["exposure_mean_Edd_J"]
    # The Néel-only average contains no rigid-body response.  A blocked basin
    # is initially uncorrelated (zero ensemble Edd), so its Brownian alignment
    # contribution can be added without subtracting another finite term.
    tip_magnetic_J = tip_neel_only_J + (
        neel["blocked_fraction"]
        * brownian_alignment_fraction
        * tip_locked_aligned_J
    )
    face_equilibrium_J = np.empty_like(temperatures_K)
    for index, temperature_K in enumerate(temperatures_K):
        levels_J = (
            intrawell_q[index] ** 2 * face_geometry["Edd_levels_J"]
        )
        weights = np.exp(
            -(levels_J - np.min(levels_J))
            / (Boltzmann * temperature_K)
        )
        weights /= np.sum(weights)
        face_equilibrium_J[index] = float(weights @ levels_J)
    # Neel blocking is already present in the finite-time face result.  Express
    # it explicitly as a non-negative relaxation deficit relative to canonical
    # equilibrium, rather than adding the barrier energy a second time.
    face_neel_penalty_J = np.maximum(
        face_neel_limited_J - face_equilibrium_J, 0.0
    )
    face_locked_111_J = (
        intrawell_q**2 * face_geometry["same_111_Edd_J"]
    )
    # A blocked cube would lower its magnetic energy by rotating its <111> axis
    # into the tip direction.  Facet registration suppresses that relaxation;
    # the lost lowering is the Brownian constraint penalty.
    face_brownian_constraint_penalty_J = (
        neel["blocked_fraction"]
        * brownian_alignment_fraction
        * (face_locked_111_J - tip_locked_aligned_J)
    )
    face_magnetic_J = (
        face_equilibrium_J
        + face_neel_penalty_J
        + face_brownian_constraint_penalty_J
    )
    face_vdw_J = np.full_like(temperatures_K, face_geometry["vdw_J"])
    tip_vdw_J = np.full_like(temperatures_K, tip_geometry["vdw_J"])
    kBT_J = Boltzmann * temperatures_K
    face_pair_total_J = face_magnetic_J + face_vdw_J
    tip_pair_total_J = tip_magnetic_J + tip_vdw_J
    magnetic_crossings = crossings_K(
        temperatures_K, tip_magnetic_J - face_magnetic_J
    )
    pair_crossings = crossings_K(
        temperatures_K, tip_pair_total_J - face_pair_total_J
    )

    display_temperature_K = np.linspace(200.0, 300.0, 501)
    interpolation_order = np.argsort(temperatures_K)

    def smooth_display(values: np.ndarray) -> np.ndarray:
        return PchipInterpolator(
            temperatures_K[interpolation_order],
            values[interpolation_order],
        )(display_temperature_K)

    face_magnetic_display_J = smooth_display(face_magnetic_J)
    tip_magnetic_display_J = smooth_display(tip_magnetic_J)
    neel_tau_display_s = smooth_display(neel["geometric_mean_tau_s"])
    brownian_tau_display_s = smooth_display(tau_B_s)
    blocked_fraction_display = smooth_display(neel["blocked_fraction"])

    scale = 1.0e-20
    figure, ax = plt.subplots(figsize=(9.5, 9.5))
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.85)
    ax.axvspan(
        EXPERIMENTAL_MIN_K,
        EXPERIMENTAL_MAX_K,
        color="#c9c9c9",
        alpha=0.22,
        zorder=0,
    )
    ax.axvspan(
        TRANSIENT_MIN_K,
        TRANSIENT_MAX_K,
        color="#f6e58d",
        alpha=0.48,
        zorder=0,
    )
    ax.plot(
        display_temperature_K,
        face_magnetic_display_J / scale,
        color="#d97706",
        linewidth=2.8,
        label=r"$E_{mag}^{face}$",
    )
    ax.plot(
        display_temperature_K,
        tip_magnetic_display_J / scale,
        color="#7e57c2",
        linewidth=2.8,
        label=r"$E_{mag}^{tip}$",
    )
    ax.plot(
        display_temperature_K,
        np.full_like(display_temperature_K, face_geometry["vdw_J"]) / scale,
        color="#2676b8",
        linewidth=2.8,
        label=r"$E_{vdW}^{face,pair}$",
    )
    ax.plot(
        display_temperature_K,
        Boltzmann * display_temperature_K / scale,
        color="#222222",
        linewidth=2.8,
        label=r"$k_BT$",
    )
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.set_xlim(200.0, 300.0)
    energy_plot_values = np.concatenate(
        [
            face_magnetic_display_J / scale,
            tip_magnetic_display_J / scale,
            np.full_like(display_temperature_K, face_geometry["vdw_J"])
            / scale,
            Boltzmann * display_temperature_K / scale,
            np.array([0.0]),
        ]
    )
    energy_min = float(np.min(energy_plot_values))
    energy_max = float(np.max(energy_plot_values))
    energy_span = max(energy_max - energy_min, 1.0)
    ax.set_ylim(
        energy_min - 0.45 * energy_span,
        energy_max + 0.08 * energy_span,
    )
    ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    ax.set_xlabel("Temperature (K)", fontsize=21, fontweight="bold")
    ax.set_ylabel(
        r"Energy ($10^{-20}$ J)", fontsize=21, fontweight="bold"
    )
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.6,
        length=6,
        labelsize=18,
    )
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    for gridline in ax.get_xgridlines() + ax.get_ygridlines():
        gridline.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Patch(
                facecolor="#c9c9c9",
                alpha=0.22,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.48,
                edgecolor="none",
                label="transient aggregation",
            ),
        ]
    )
    labels.extend(["experimental range", "transient aggregation"])
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.015),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        prop={"family": "Arial", "size": 14.0, "weight": "bold"},
        handlelength=2.8,
        handletextpad=0.9,
        borderpad=0.85,
        labelspacing=0.65,
        borderaxespad=0.0,
    )
    figure.suptitle(
        "Pair energies with rigid-cube Brownian rotation\n"
        f"16 nm Fe$_3$O$_4$, 1.5 nm ligand shell; {EXPERIMENTAL_WINDOW_S:g} s window",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.42,
        0.027,
        r"Each temperature starts from a random 64-state prior; blocked $\langle111\rangle$ moments may align by cube rotation."
        "\n"
        r"The face curve includes the Néel relaxation deficit and suppressed-rotation penalty; no barrier is fitted.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#555555",
    )
    energy_png = OUTPUT_DIR / "brownian_cube_rotation_pair_energies.png"
    energy_pdf = OUTPUT_DIR / "brownian_cube_rotation_pair_energies.pdf"
    figure.savefig(energy_png, dpi=600)
    figure.savefig(energy_pdf)
    plt.close(figure)

    timescale_figure, timescale_ax = plt.subplots(figsize=(10.2, 7.4))
    timescale_figure.subplots_adjust(
        left=0.13, right=0.68, bottom=0.15, top=0.86
    )
    timescale_ax.semilogy(
        display_temperature_K,
        neel_tau_display_s,
        color="#d97706",
        linewidth=2.8,
        label=r"geometric mean $\tau_N$",
    )
    timescale_ax.semilogy(
        display_temperature_K,
        brownian_tau_display_s,
        color="#7e57c2",
        linewidth=2.8,
        label=r"free-particle $\tau_B$",
    )
    timescale_ax.axhline(
        protocol.EXPOSURE_DURATION_S,
        color="#222222",
        linewidth=2.4,
        label=f"{EXPERIMENTAL_WINDOW_S:g} s effective window",
    )
    timescale_ax.set_xlim(200.0, 300.0)
    timescale_ax.set_xlabel("Temperature (K)", fontsize=14)
    timescale_ax.set_ylabel("Relaxation time (s)", fontsize=14)
    timescale_ax.tick_params(axis="both", labelsize=12)
    timescale_ax.grid(False, which="both", axis="both")
    timescale_ax.xaxis.grid(False, which="both")
    timescale_ax.yaxis.grid(False, which="both")
    fraction_ax = timescale_ax.twinx()
    fraction_ax.plot(
        display_temperature_K,
        blocked_fraction_display,
        color="#2676b8",
        linewidth=2.5,
        label=f"blocked fraction over {EXPERIMENTAL_WINDOW_S:g} s",
    )
    fraction_ax.set_ylim(0.0, 1.0)
    fraction_ax.set_ylabel("Blocked fraction", fontsize=14)
    fraction_ax.tick_params(axis="y", labelsize=12)
    handles_1, labels_1 = timescale_ax.get_legend_handles_labels()
    handles_2, labels_2 = fraction_ax.get_legend_handles_labels()
    timescale_ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False,
        fontsize=10.0,
        borderaxespad=0.0,
    )
    timescale_figure.suptitle(
        "Néel and free-cube Brownian timescales",
        fontsize=17,
    )
    timescale_png = OUTPUT_DIR / "brownian_cube_rotation_timescales.png"
    timescale_pdf = OUTPUT_DIR / "brownian_cube_rotation_timescales.pdf"
    timescale_figure.savefig(timescale_png, dpi=300)
    timescale_figure.savefig(timescale_pdf)
    plt.close(timescale_figure)

    csv_path = OUTPUT_DIR / "brownian_cube_rotation_pair_energies.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                temperatures_K,
                face_equilibrium_J,
                face_neel_limited_J,
                face_neel_penalty_J,
                face_brownian_constraint_penalty_J,
                face_magnetic_J,
                tip_neel_only_J,
                tip_magnetic_J,
                tip_locked_aligned_J,
                face_vdw_J,
                tip_vdw_J,
                face_pair_total_J,
                tip_pair_total_J,
                kBT_J,
                neel["blocked_fraction"],
                neel["geometric_mean_tau_s"],
                viscosity_Pa_s,
                tau_B_s,
                brownian_alignment_fraction,
            ]
        ),
        delimiter=",",
        header=(
            "temperature_K,face_equilibrium_Edd_J,"
            "face_Neel_limited_Edd_J,face_Neel_penalty_J,"
            "face_Brownian_constraint_penalty_J,face_constrained_Emag_J,"
            "tip_Neel_only_Emag_J,"
            "tip_Neel_plus_Brownian_Emag_J,tip_locked_aligned_Edd_J,"
            "face_vdW_pair_J,tip_vdW_pair_J,face_pair_total_J,"
            f"tip_pair_total_J,kBT_J,blocked_fraction_{EXPERIMENTAL_WINDOW_S:g}s,"
            "geometric_mean_tau_N_s,hexane_viscosity_Pa_s,tau_B_free_s,"
            f"brownian_alignment_fraction_{EXPERIMENTAL_WINDOW_S:g}s"
        ),
        comments="",
        fmt="%.12e",
    )

    summary_path = OUTPUT_DIR / "brownian_cube_rotation_summary.txt"
    magnetic_text = (
        ", ".join(f"{value:.2f} K" for value in magnetic_crossings)
        if magnetic_crossings
        else "none between 200 and 300 K"
    )
    pair_text = (
        ", ".join(f"{value:.2f} K" for value in pair_crossings)
        if pair_crossings
        else "none between 200 and 300 K"
    )
    summary_path.write_text(
        "\n".join(
            [
                "Reduced rigid-cube Brownian-rotation model",
                "Each temperature is initialized independently from a uniform 1/64 pair-state distribution.",
                "Face is facet constrained; Brownian alignment is applied only to the blocked tip-forming fraction.",
                "Face Emag = canonical face Edd + Neel relaxation deficit + suppressed-Brownian-rotation penalty.",
                "The Neel deficit is not added twice; the transition barrier is not treated as occupied-state energy.",
                "No ligand/contact rotational barrier is included.",
                "SAXS integration = 150 s",
                f"effective midpoint window = {EXPERIMENTAL_WINDOW_S:g} s",
                f"Ms(300 K sample value) = {v620.PARAMS.saturation_magnetization_Apm:.9e} A/m",
                f"Ms basis = {MS_BASIS}",
                f"Neel attempt time tau0(300 K) = {v620.PARAMS.attempt_time_s:.9e} s",
                f"attempt-time source = {ATTEMPT_TIME_REFERENCE}",
                "The 1/3 pathway factor partitions the literature total escape frequency among the three adjacent <111> wells.",
                f"Hamaker constant = {v620.PARAMS.hamaker_J:.9e} J",
                f"Hamaker source/range = {HAMAKER_REFERENCE}; representative value within 9-29 zJ for nonpolar solvents",
                f"diameter coefficient of variation = {100.0 * v620.PARAMS.diameter_coefficient_of_variation:.3f}% (temporary monodisperse baseline; replace with sample TEM result)",
                "viscosity = temperature-dependent saturated-liquid n-hexane reference correlation",
                f"viscosity source = {VISCOSITY_REFERENCE}",
                f"hexane viscosity at 250 K = {hexane_viscosity_Pa_s(np.array([250.0]))[0] * 1e3:.6f} mPa s",
                f"hexane viscosity at 300 K = {hexane_viscosity_Pa_s(np.array([300.0]))[0] * 1e3:.6f} mPa s",
                f"hydrodynamic diameter = {HYDRODYNAMIC_DIAMETER_M * 1e9:.3f} nm",
                f"face surface gap = {face_geometry['surface_gap_nm']:.3f} nm",
                f"tip surface gap = {tip_geometry['surface_gap_nm']:.3f} nm",
                f"face center distance = {face_geometry['distance_m'] * 1e9:.9f} nm",
                f"tip center distance = {tip_geometry['distance_m'] * 1e9:.9f} nm",
                f"face vdW = {face_geometry['vdw_J']:.9e} J/pair",
                f"tip vdW = {tip_geometry['vdw_J']:.9e} J/pair",
                f"face minimum Edd = {face_geometry['minimum_Edd_J']:.9e} J/pair",
                f"face same-[111] Edd = {face_geometry['same_111_Edd_J']:.9e} J/pair",
                f"tip aligned Edd = {tip_geometry['minimum_Edd_J']:.9e} J/pair",
                f"magnetic-energy crossing = {magnetic_text}",
                f"pair-total crossing including both vdW terms = {pair_text}",
                "This reduced correction is a mechanism test, not a quantitative face-to-tip rate calculation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parameter_audit_path = (
        OUTPUT_DIR / "brownian_cube_rotation_parameter_audit.md"
    )
    parameter_audit_path.write_text(
        "\n".join(
            [
                "# Parameter provenance for the V10.18 pair-energy model",
                "",
                "| Parameter | Value used | Classification | Source or required action |",
                "|---|---:|---|---|",
                f"| Saturation magnetization, `Ms` | {v620.PARAMS.saturation_magnetization_Apm:.3e} A m^-1 | Present-sample value at 300 K | {MS_BASIS}. This assumes 55 emu/g is normalized to inorganic Fe3O4 mass; if it includes ligand mass, apply the TGA inorganic-mass correction before conversion. The constant-Ms treatment over 200-300 K remains a model approximation. |",
                f"| Neel attempt time, `tau0` | {v620.PARAMS.attempt_time_s * 1e9:.2f} ns | Literature value at 300 K | {ATTEMPT_TIME_REFERENCE}; reported value is 0.98 +/- 0.13 ns for cubic-anisotropy magnetite. |",
                f"| Hamaker constant | {v620.PARAMS.hamaker_J:.2e} J | Literature-based representative value | {HAMAKER_REFERENCE} reports 9-29 zJ across hexane/toluene. Use a solvent-specific value when the medium is fixed. |",
                f"| Particle edge | {v620.PARAMS.particle_size_nm:g} nm | Experimental sample input | Must come from TEM/SAXS for the present batch; it is not a universal material constant. |",
                f"| Blocking temperature | {v620.PARAMS.blocking_temperature_K:g} K | Experimental input | Taken from the present ZFC/FC result. |",
                f"| ZFC/FC observation time | {v620.PARAMS.zfc_fc_observation_time_s:g} s | Protocol assumption | Replace by a timescale derived from the actual magnetometry sweep/settling protocol if available. |",
                f"| Diameter coefficient of variation | {100.0 * v620.PARAMS.diameter_coefficient_of_variation:.1f}% | Temporary monodisperse baseline | Replace by the present sample's TEM size-distribution fit when available. |",
                f"| Solvent viscosity | temperature-dependent n-hexane | Literature reference correlation | {VISCOSITY_REFERENCE}; the Table 4 saturation values are fitted in reciprocal temperature, with extrapolation below 250 K. |",
                f"| Ligand length per surface | {LIGAND_LENGTH_NM:g} nm | User-specified sample input | Confirm by ligand identity/chain conformation or scattering/TGA characterization. |",
                f"| Face and tip surface gaps | {FACE_SURFACE_GAP_NM:g}, {TIP_SURFACE_GAP_NM:g} nm | Geometric model inputs | Face gap was user specified; tip gap equals two ligand lengths. |",
                f"| Cube rounding radius | {v620.PARAMS.roundness_nm:g} nm | Unsupported morphology input | Replace by a TEM-derived corner radius; it materially affects the tip center distance. |",
                f"| Effective SAXS window | {EXPERIMENTAL_WINDOW_S:g} s | Experimental protocol convention | Midpoint representation of the 150 s integration, as requested. |",
                f"| Angular quadrature | {v620.PARAMS.orientation_gauss_count} x {v620.PARAMS.orientation_phi_count} | Numerical setting | Requires convergence checking, not a literature citation. |",
                f"| Barrier quadrature | {v620.PARAMS.barrier_distribution_count} points | Numerical setting | Gauss-Hermite integration; relevant only when a nonzero measured diameter CV is supplied. |",
                "",
                "The factor 1/3 in each adjacent-well rate is not an empirical parameter. A <111> minimum has three nearest <110>-saddle exits, and the cited tau0 describes the total mean residence/escape time. Therefore each equivalent path receives one third of the total isolated escape rate; the other four <111> wells require multiple elementary hops.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(energy_png)
    print(timescale_png)
    print(csv_path)
    print(summary_path)
    print(parameter_audit_path)


if __name__ == "__main__":
    main()
