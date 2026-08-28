"""Sweep the face/tip crossing over 0.1--100 s observation windows.

This calculation uses the same independent-setpoint, 64-state pair model as
the final pair-energy model, but evaluates the exposure-averaged Neel energy analytically in the
eigenmodes of the rate generator.  This makes a dense observation-time sweep
possible without time-step artifacts.  The diameter CV is temporarily zero
and the free-cube Brownian time uses temperature-dependent n-hexane viscosity.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
from scipy.linalg import eig


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    ROOT
    / "versions"
    / "V10_multiphysics_nanocube"
    / "code"
    / "pair_energy_model.py"
)
OUTPUT_DIR = ROOT / "versions" / "V10_multiphysics_nanocube" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES_K = np.linspace(200.0, 300.0, 201)
STANDARD_WINDOWS_S = np.array(
    [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0]
)
OBSERVATION_WINDOWS_S = np.unique(
    np.concatenate([np.logspace(-1.0, 2.0, 61), STANDARD_WINDOWS_S])
)


def load_model_module():
    spec = importlib.util.spec_from_file_location("v10_18_window_sweep", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def modal_mean_energy_coefficients(
    generator: np.ndarray,
    initial_probability: np.ndarray,
    energy_levels_J: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lambda and coefficients for the time-averaged observable."""
    eigenvalues, eigenvectors = eig(generator)
    inverse_eigenvectors = np.linalg.inv(eigenvectors)
    coefficients = (
        initial_probability @ eigenvectors
    ) * (inverse_eigenvectors @ energy_levels_J)
    return eigenvalues, coefficients


def modal_time_average(
    eigenvalues: np.ndarray,
    coefficients: np.ndarray,
    windows_s: np.ndarray,
) -> np.ndarray:
    z = windows_s[:, None] * eigenvalues[None, :]
    phi = np.ones_like(z, dtype=complex)
    nonzero = np.abs(z) > 1.0e-10
    phi[nonzero] = np.expm1(z[nonzero]) / z[nonzero]
    values = np.real_if_close(phi @ coefficients, tol=1000)
    if np.max(np.abs(np.imag(values))) > 1.0e-24:
        raise RuntimeError("Unexpected complex residual in modal energy average")
    return np.asarray(np.real(values), dtype=float)


def first_crossing_K(
    temperatures_K: np.ndarray, difference_J: np.ndarray
) -> float:
    for index in range(len(temperatures_K) - 1):
        y0 = difference_J[index]
        y1 = difference_J[index + 1]
        if y0 == 0.0:
            return float(temperatures_K[index])
        if y0 * y1 < 0.0:
            fraction = -y0 / (y1 - y0)
            return float(
                temperatures_K[index]
                + fraction
                * (temperatures_K[index + 1] - temperatures_K[index])
            )
    return float("nan")


def main():
    model = load_model_module()
    protocol = model.load_protocol_module()
    v620 = protocol.load_v620_module()
    if v620.PARAMS.diameter_coefficient_of_variation != 0.0:
        raise RuntimeError("This sweep requires the temporary CV=0 baseline")

    face_geometry = model.geometry_data(
        v620, np.array([1.0, 0.0, 0.0]), model.FACE_SURFACE_GAP_NM
    )
    tip_geometry = model.geometry_data(
        v620, np.ones(3) / np.sqrt(3.0), model.TIP_SURFACE_GAP_NM
    )
    geometries = (face_geometry, tip_geometry)
    directions = (
        np.array([1.0, 0.0, 0.0]),
        np.ones(3) / np.sqrt(3.0),
    )
    initial_probability = np.full(64, 1.0 / 64.0)
    window_count = len(OBSERVATION_WINDOWS_S)
    temperature_count = len(TEMPERATURES_K)
    neel_mean_J = [
        np.empty((window_count, temperature_count)),
        np.empty((window_count, temperature_count)),
    ]
    intrawell_q = np.empty(temperature_count)

    for temperature_index, temperature_K in enumerate(TEMPERATURES_K):
        q_value = protocol.intrawell_q(v620, float(temperature_K))
        intrawell_q[temperature_index] = q_value
        for geometry_index, (geometry, direction) in enumerate(
            zip(geometries, directions)
        ):
            center_vector_m = geometry["distance_m"] * direction
            base_Edd_J, base_prefactor_J, states, separation_hat = (
                v620.easy_axis_pair_energies_J(center_vector_m, v620.PARAMS)
            )
            energy_levels_J = q_value**2 * base_Edd_J
            generator = v620.neel_pair_generator(
                float(temperature_K),
                v620.PARAMS.zfc_fc_activation_barrier_J,
                energy_levels_J,
                q_value**2 * base_prefactor_J,
                states,
                separation_hat,
                v620.PARAMS,
            )
            eigenvalues, coefficients = modal_mean_energy_coefficients(
                generator, initial_probability, energy_levels_J
            )
            neel_mean_J[geometry_index][:, temperature_index] = (
                modal_time_average(
                    eigenvalues, coefficients, OBSERVATION_WINDOWS_S
                )
            )

    face_equilibrium_J = np.empty(temperature_count)
    for index, temperature_K in enumerate(TEMPERATURES_K):
        levels_J = intrawell_q[index] ** 2 * face_geometry["Edd_levels_J"]
        weights = np.exp(
            -(levels_J - np.min(levels_J)) / (Boltzmann * temperature_K)
        )
        weights /= np.sum(weights)
        face_equilibrium_J[index] = float(weights @ levels_J)

    tau_N_s = v620.PARAMS.attempt_time_s * np.exp(
        v620.PARAMS.zfc_fc_activation_barrier_J
        / (Boltzmann * TEMPERATURES_K)
    )
    blocked_fraction = np.exp(
        -OBSERVATION_WINDOWS_S[:, None] / tau_N_s[None, :]
    )
    viscosity_Pa_s = model.hexane_viscosity_Pa_s(TEMPERATURES_K)
    tau_B_s = model.brownian_time_s(TEMPERATURES_K)
    brownian_alignment_fraction = 1.0 - np.exp(
        -OBSERVATION_WINDOWS_S[:, None] / tau_B_s[None, :]
    )
    tip_locked_aligned_J = (
        intrawell_q**2 * tip_geometry["minimum_Edd_J"]
    )
    face_locked_111_J = (
        intrawell_q**2 * face_geometry["same_111_Edd_J"]
    )

    face_neel_penalty_J = np.maximum(
        neel_mean_J[0] - face_equilibrium_J[None, :], 0.0
    )
    face_brownian_penalty_J = (
        blocked_fraction
        * brownian_alignment_fraction
        * (face_locked_111_J - tip_locked_aligned_J)[None, :]
    )
    face_magnetic_J = (
        face_equilibrium_J[None, :]
        + face_neel_penalty_J
        + face_brownian_penalty_J
    )
    tip_magnetic_J = neel_mean_J[1] + (
        blocked_fraction
        * brownian_alignment_fraction
        * tip_locked_aligned_J[None, :]
    )
    face_total_J = face_magnetic_J + face_geometry["vdw_J"]
    tip_total_J = tip_magnetic_J + tip_geometry["vdw_J"]

    magnetic_crossing_K = np.array(
        [
            first_crossing_K(
                TEMPERATURES_K,
                tip_magnetic_J[index] - face_magnetic_J[index],
            )
            for index in range(window_count)
        ]
    )
    pair_total_crossing_K = np.array(
        [
            first_crossing_K(
                TEMPERATURES_K,
                tip_total_J[index] - face_total_J[index],
            )
            for index in range(window_count)
        ]
    )

    csv_path = OUTPUT_DIR / "observation_window_face_tip_crossings.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                OBSERVATION_WINDOWS_S,
                magnetic_crossing_K,
                pair_total_crossing_K,
            ]
        ),
        delimiter=",",
        header=(
            "observation_window_s,magnetic_energy_crossing_K,"
            "pair_total_energy_crossing_K"
        ),
        comments="",
        fmt="%.12e",
    )

    selected_rows = []
    for window_s in STANDARD_WINDOWS_S:
        index = int(np.argmin(np.abs(OBSERVATION_WINDOWS_S - window_s)))
        selected_rows.append(
            (
                OBSERVATION_WINDOWS_S[index],
                magnetic_crossing_K[index],
                pair_total_crossing_K[index],
            )
        )
    selected_csv_path = (
        OUTPUT_DIR / "observation_window_face_tip_crossings_selected.csv"
    )
    np.savetxt(
        selected_csv_path,
        np.asarray(selected_rows),
        delimiter=",",
        header=(
            "observation_window_s,magnetic_energy_crossing_K,"
            "pair_total_energy_crossing_K"
        ),
        comments="",
        fmt="%.9f",
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.8,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, ax = plt.subplots(figsize=(8.2, 8.2))
    figure.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.84)
    ax.semilogx(
        OBSERVATION_WINDOWS_S,
        magnetic_crossing_K,
        color="#2676b8",
        linewidth=3.0,
        label="magnetic-energy crossing",
    )
    ax.semilogx(
        OBSERVATION_WINDOWS_S,
        pair_total_crossing_K,
        color="#d97706",
        linewidth=3.0,
        label="pair-total crossing",
    )
    ax.set_xlim(0.1, 100.0)
    finite_values = np.concatenate(
        [
            magnetic_crossing_K[np.isfinite(magnetic_crossing_K)],
            pair_total_crossing_K[np.isfinite(pair_total_crossing_K)],
        ]
    )
    if finite_values.size:
        margin = max(2.0, 0.1 * np.ptp(finite_values))
        ax.set_ylim(np.min(finite_values) - margin, np.max(finite_values) + margin)
    ax.set_xlabel("Observation window (s)", fontsize=20)
    ax.set_ylabel("Face-tip crossing temperature (K)", fontsize=20)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.6,
        length=6,
        labelsize=17,
    )
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.legend(loc="best", frameon=False, fontsize=14)
    figure.suptitle(
        "Observation-window dependence of the face-tip crossing\n"
        "CV = 0; temperature-dependent n-hexane viscosity",
        fontsize=20,
    )
    png_path = OUTPUT_DIR / "observation_window_face_tip_crossings.png"
    pdf_path = OUTPUT_DIR / "observation_window_face_tip_crossings.pdf"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path)
    plt.close(figure)

    summary_path = OUTPUT_DIR / "observation_window_face_tip_crossings_summary.txt"
    summary_lines = [
        "Observation-window sweep for the reduced V10.18 face/tip model",
        "observation window range = 0.1-100 s",
        f"number of sampled windows = {window_count}",
        "temperature grid = 200-300 K in 0.5 K increments",
        "diameter coefficient of variation = 0% (temporary monodisperse baseline)",
        "viscosity = temperature-dependent saturated-liquid n-hexane",
        f"viscosity source = {model.VISCOSITY_REFERENCE}",
        f"hexane viscosity at 250 K = {model.hexane_viscosity_Pa_s(np.array([250.0]))[0] * 1e3:.6f} mPa s",
        f"hexane viscosity at 300 K = {model.hexane_viscosity_Pa_s(np.array([300.0]))[0] * 1e3:.6f} mPa s",
        "Each temperature starts independently from a uniform 64-state prior.",
        "Exposure-mean Neel energies are evaluated by exact modal integration of the master equation.",
        "",
        "window_s,magnetic_crossing_K,pair_total_crossing_K",
    ]
    summary_lines.extend(
        f"{window_s:g},{magnetic_K:.6f},{pair_K:.6f}"
        for window_s, magnetic_K, pair_K in selected_rows
    )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(png_path)
    print(csv_path)
    print(selected_csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
