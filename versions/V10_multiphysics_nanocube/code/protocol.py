"""Conditional face/tip pair energies under the stepped SAXS protocol.

Protocol assumption:
  * an initial 180 s SAXS exposure at 300 K;
  * a 120 s linear ramp for every 10 K decrease (5 K/min);
  * a 180 s isothermal SAXS exposure at every new setpoint;
  * setpoints from 300 K to 200 K.

The hypothetical face and tip pairs exist at their ligand-limited separation
throughout the calculation.  Thus there is no contact-population switch and no
factor multiplying Edd by an assembled fraction.  Every temperature setpoint
is an independent calculation initialized from the uniform random 64-state
distribution; no probability distribution is inherited from another setpoint.

The 100 s ZFC/FC time is used only inside geometry_model.py to calibrate the anisotropy
barrier.  No separate 1/10/100 s observation window is imposed.  Each plotted
energy is averaged over the actual 180 s SAXS exposure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.constants import Boltzmann
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[3]
V620_PATH = (
    Path(__file__).resolve().parent / "geometry_model.py"
)
OUTPUT_DIR = ROOT / "versions" / "V10_multiphysics_nanocube" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_TEMPERATURE_K = 300.0
FINAL_TEMPERATURE_K = 200.0
TEMPERATURE_STEP_K = 10.0
RAMP_DURATION_S = 120.0
EXPOSURE_DURATION_S = 180.0
RAMP_SUBSTEP_K = 0.5
EXPOSURE_SUBSTEP_S = 10.0
LIGAND_LENGTH_NM = 1.7
SURFACE_GAP_NM = 2.0 * LIGAND_LENGTH_NM
EXPERIMENTAL_MIN_K = 233.15
EXPERIMENTAL_MAX_K = 293.15
TRANSIENT_MIN_K = 253.15
TRANSIENT_MAX_K = 273.15


def load_v620_module():
    spec = importlib.util.spec_from_file_location("v620_saxs_protocol", V620_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V620_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def intrawell_q(v620, temperature_K: float) -> float:
    """Boltzmann mean projection onto one selected <111> easy axis."""
    directions, quadrature_weights, easy_axis = (
        v620.selected_easy_axis_basin_quadrature(v620.PARAMS)
    )
    anisotropy_J = v620.single_particle_anisotropy_J(
        directions, v620.PARAMS
    )
    shifted_J = anisotropy_J - np.min(anisotropy_J)
    weights = quadrature_weights * np.exp(
        -shifted_J / (Boltzmann * temperature_K)
    )
    weights /= np.sum(weights)
    return float(weights @ (directions @ easy_axis))


def build_protocol():
    """Return elementary ramp/hold segments and SAXS frame metadata."""
    setpoints_K = np.arange(
        INITIAL_TEMPERATURE_K,
        FINAL_TEMPERATURE_K - 0.1,
        -TEMPERATURE_STEP_K,
    )
    segments = []
    frames = []
    time_s = 0.0

    def add_exposure(setpoint_K: float, frame_index: int):
        nonlocal time_s
        start_s = time_s
        count = int(round(EXPOSURE_DURATION_S / EXPOSURE_SUBSTEP_S))
        for _ in range(count):
            segments.append(
                {
                    "start_time_s": time_s,
                    "duration_s": EXPOSURE_SUBSTEP_S,
                    "start_T_K": setpoint_K,
                    "end_T_K": setpoint_K,
                    "stage_code": 1,
                    "frame_index": frame_index,
                }
            )
            time_s += EXPOSURE_SUBSTEP_S
        frames.append(
            {
                "frame_index": frame_index,
                "setpoint_K": setpoint_K,
                "exposure_start_s": start_s,
                "exposure_end_s": time_s,
                "exposure_mid_s": 0.5 * (start_s + time_s),
            }
        )

    add_exposure(setpoints_K[0], 0)
    ramp_substeps = int(round(TEMPERATURE_STEP_K / RAMP_SUBSTEP_K))
    ramp_dt_s = RAMP_DURATION_S / ramp_substeps
    for frame_index, target_K in enumerate(setpoints_K[1:], start=1):
        start_K = setpoints_K[frame_index - 1]
        for substep in range(ramp_substeps):
            fraction_0 = substep / ramp_substeps
            fraction_1 = (substep + 1) / ramp_substeps
            segment_start_K = start_K + (target_K - start_K) * fraction_0
            segment_end_K = start_K + (target_K - start_K) * fraction_1
            segments.append(
                {
                    "start_time_s": time_s,
                    "duration_s": ramp_dt_s,
                    "start_T_K": segment_start_K,
                    "end_T_K": segment_end_K,
                    "stage_code": 0,
                    "frame_index": -1,
                }
            )
            time_s += ramp_dt_s
        add_exposure(target_K, frame_index)
    return setpoints_K, segments, frames


def propagate_geometry(
    v620,
    direction: np.ndarray,
    segments,
    frame_count: int,
    surface_gap_nm: float | None = None,
):
    """Average each setpoint from an independent uniform 64-state start."""
    params = v620.PARAMS
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    gap_nm = SURFACE_GAP_NM if surface_gap_nm is None else surface_gap_nm
    center_distance_m = v620.center_distance_at_gap_m(
        direction, gap_nm * 1.0e-9, params
    )
    center_vector_m = center_distance_m * direction
    easy_axis_Edd_J, prefactor_J, states, separation_hat = (
        v620.easy_axis_pair_energies_J(center_vector_m, params)
    )
    barrier_factors, barrier_weights = v620.barrier_distribution(params)

    # Every setpoint is reset to the same random prior.  Dipolar coupling acts
    # during that setpoint's exposure, but the resulting distribution is not
    # passed to the following temperature.
    probability_initial = np.full(64, 1.0 / 64.0)
    q_initial = intrawell_q(v620, INITIAL_TEMPERATURE_K)
    initial_energy_J = q_initial**2 * easy_axis_Edd_J
    probabilities = np.repeat(
        probability_initial[None, :], len(barrier_factors), axis=0
    )

    exposure_integral_Js = np.zeros(frame_count)
    exposure_start_Edd_J = np.full(frame_count, np.nan)
    exposure_end_Edd_J = np.full(frame_count, np.nan)
    exposure_start_q = np.full(frame_count, np.nan)
    exposure_end_q = np.full(frame_count, np.nan)
    trajectory_time_s = [0.0]
    trajectory_temperature_K = [INITIAL_TEMPERATURE_K]
    trajectory_Edd_J = [float(probability_initial @ initial_energy_J)]
    transition_cache = {}

    def ensemble_energy(temperature_K: float) -> tuple[float, float]:
        q_value = intrawell_q(v620, temperature_K)
        ensemble_probability = barrier_weights @ probabilities
        energy_J = float(
            q_value**2 * (ensemble_probability @ easy_axis_Edd_J)
        )
        return energy_J, q_value

    active_frame_index = -1
    for segment in segments:
        start_T_K = float(segment["start_T_K"])
        end_T_K = float(segment["end_T_K"])
        midpoint_K = 0.5 * (start_T_K + end_T_K)
        duration_s = float(segment["duration_s"])
        frame_index = int(segment["frame_index"])
        # Cooling ramps carry no magnetic history in the independent-setpoint
        # model.  Only the isothermal SAXS exposure at each setpoint is evolved.
        if frame_index < 0:
            continue
        if frame_index != active_frame_index:
            probabilities[:] = probability_initial
            active_frame_index = frame_index
        start_Edd_J, start_q = ensemble_energy(start_T_K)
        if frame_index >= 0 and np.isnan(exposure_start_Edd_J[frame_index]):
            exposure_start_Edd_J[frame_index] = start_Edd_J
            exposure_start_q[frame_index] = start_q

        midpoint_q = intrawell_q(v620, midpoint_K)
        scaled_Edd_J = midpoint_q**2 * easy_axis_Edd_J
        scaled_prefactor_J = midpoint_q**2 * prefactor_J
        for distribution_index, barrier_factor in enumerate(barrier_factors):
            cache_key = (
                round(midpoint_K, 8),
                round(duration_s, 8),
                distribution_index,
            )
            transition = transition_cache.get(cache_key)
            if transition is None:
                generator = v620.neel_pair_generator(
                    midpoint_K,
                    params.zfc_fc_activation_barrier_J * barrier_factor,
                    scaled_Edd_J,
                    scaled_prefactor_J,
                    states,
                    separation_hat,
                    params,
                )
                transition = expm(generator * duration_s)
                transition_cache[cache_key] = transition
            probabilities[distribution_index] = (
                probabilities[distribution_index] @ transition
            )
            probabilities[distribution_index] = np.maximum(
                probabilities[distribution_index], 0.0
            )
            probabilities[distribution_index] /= np.sum(
                probabilities[distribution_index]
            )

        end_Edd_J, end_q = ensemble_energy(end_T_K)
        if frame_index >= 0:
            exposure_integral_Js[frame_index] += (
                0.5 * (start_Edd_J + end_Edd_J) * duration_s
            )
            exposure_end_Edd_J[frame_index] = end_Edd_J
            exposure_end_q[frame_index] = end_q
        trajectory_time_s.append(segment["start_time_s"] + duration_s)
        trajectory_temperature_K.append(end_T_K)
        trajectory_Edd_J.append(end_Edd_J)

    return {
        "center_distance_m": center_distance_m,
        "vdw_pair_J": v620.pair_vdw_energy_J(center_vector_m, params),
        "exposure_mean_Edd_J": exposure_integral_Js / EXPOSURE_DURATION_S,
        "exposure_start_Edd_J": exposure_start_Edd_J,
        "exposure_end_Edd_J": exposure_end_Edd_J,
        "exposure_start_q": exposure_start_q,
        "exposure_end_q": exposure_end_q,
        "trajectory_time_s": np.asarray(trajectory_time_s),
        "trajectory_temperature_K": np.asarray(trajectory_temperature_K),
        "trajectory_Edd_J": np.asarray(trajectory_Edd_J),
    }


def main():
    v620 = load_v620_module()
    setpoints_K, segments, frames = build_protocol()
    face = propagate_geometry(
        v620, np.array([1.0, 0.0, 0.0]), segments, len(frames)
    )
    tip = propagate_geometry(
        v620, np.ones(3) / np.sqrt(3.0), segments, len(frames)
    )

    frame_mid_s = np.array([frame["exposure_mid_s"] for frame in frames])
    exposure_start_s = np.array(
        [frame["exposure_start_s"] for frame in frames]
    )
    exposure_end_s = np.array([frame["exposure_end_s"] for frame in frames])
    kBT_J = Boltzmann * setpoints_K
    scale = 1.0e-20

    figure, energy_ax = plt.subplots(figsize=(10.2, 10.2))
    figure.subplots_adjust(left=0.13, right=0.70, bottom=0.18, top=0.86)
    energy_ax.axvspan(
        EXPERIMENTAL_MIN_K,
        EXPERIMENTAL_MAX_K,
        color="#c9c9c9",
        alpha=0.22,
        zorder=0,
    )
    energy_ax.axvspan(
        TRANSIENT_MIN_K,
        TRANSIENT_MAX_K,
        color="#f6e58d",
        alpha=0.48,
        zorder=0,
    )

    energy_ax.plot(
        setpoints_K,
        face["exposure_mean_Edd_J"] / scale,
        color="#d97706",
        linewidth=2.8,
        marker="o",
        markersize=5.0,
        label=r"$\overline{E}_{mag}^{face}$ (180 s average)",
    )
    energy_ax.plot(
        setpoints_K,
        tip["exposure_mean_Edd_J"] / scale,
        color="#7e57c2",
        linewidth=2.8,
        marker="s",
        markersize=4.8,
        label=r"$\overline{E}_{mag}^{tip}$ (180 s average)",
    )
    energy_ax.plot(
        setpoints_K,
        np.full_like(setpoints_K, face["vdw_pair_J"]) / scale,
        color="#2676b8",
        linewidth=2.8,
        label=r"$E_{vdW}^{face,pair}$",
    )
    energy_ax.plot(
        setpoints_K,
        kBT_J / scale,
        color="#222222",
        linewidth=2.8,
        label=r"$k_BT$",
    )
    energy_ax.axhline(0.0, color="#777777", linewidth=0.9)
    energy_ax.set_xlim(200.0, 300.0)
    energy_ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    energy_ax.set_xlabel("Temperature (K)", fontsize=16)
    energy_ax.set_ylabel(r"Energy ($10^{-20}$ J)", fontsize=16)
    energy_ax.tick_params(axis="both", labelsize=14)
    energy_ax.grid(alpha=0.18)
    handles, labels = energy_ax.get_legend_handles_labels()
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
    energy_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=12,
        borderaxespad=0.0,
    )

    figure.suptitle(
        "Independent-setpoint conditional pair energies\n"
        "16 nm Fe$_3$O$_4$, 1.7 nm ligand shell",
        fontsize=18,
    )
    figure.text(
        0.42,
        0.027,
        r"Every temperature starts from the same uniform random 64-state distribution."
        "\n"
        r"Each curve is a separate 180 s SAXS average; no state inheritance or contact/population switch.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#555555",
    )

    base = OUTPUT_DIR / "saxs_step_protocol_pair_energies"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    frame_csv_path = base.with_suffix(".csv")
    trajectory_csv_path = OUTPUT_DIR / "saxs_step_protocol_trajectory.csv"
    summary_path = OUTPUT_DIR / "saxs_step_protocol_summary.txt"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)

    np.savetxt(
        frame_csv_path,
        np.column_stack(
            [
                setpoints_K,
                exposure_start_s,
                exposure_end_s,
                frame_mid_s,
                face["exposure_mean_Edd_J"],
                face["exposure_start_Edd_J"],
                face["exposure_end_Edd_J"],
                tip["exposure_mean_Edd_J"],
                tip["exposure_start_Edd_J"],
                tip["exposure_end_Edd_J"],
                np.full_like(setpoints_K, face["vdw_pair_J"]),
                kBT_J,
                face["exposure_start_q"],
                face["exposure_end_q"],
            ]
        ),
        delimiter=",",
        header=(
            "setpoint_temperature_K,exposure_start_s,exposure_end_s,"
            "exposure_mid_s,face_exposure_mean_Emag_J,face_exposure_start_Emag_J,"
            "face_exposure_end_Emag_J,tip_exposure_mean_Emag_J,"
            "tip_exposure_start_Emag_J,tip_exposure_end_Emag_J,"
            "face_vdW_pair_J,kBT_J,q111_exposure_start,q111_exposure_end"
        ),
        comments="",
        fmt="%.12e",
    )
    np.savetxt(
        trajectory_csv_path,
        np.column_stack(
            [
                face["trajectory_time_s"],
                face["trajectory_temperature_K"],
                face["trajectory_Edd_J"],
                tip["trajectory_Edd_J"],
            ]
        ),
        delimiter=",",
        header="time_s,temperature_K,face_Emag_J,tip_Emag_J",
        comments="",
        fmt="%.12e",
    )
    summary_path.write_text(
        "\n".join(
            [
                "Conditional fixed-geometry pair energies with independent temperature setpoints",
                f"temperature range = {INITIAL_TEMPERATURE_K:g} to {FINAL_TEMPERATURE_K:g} K",
                f"temperature step = {TEMPERATURE_STEP_K:g} K",
                f"ramp duration per step = {RAMP_DURATION_S:g} s",
                f"ramp rate = {TEMPERATURE_STEP_K / RAMP_DURATION_S * 60.0:g} K/min",
                f"isothermal SAXS exposure = {EXPOSURE_DURATION_S:g} s",
                f"point-to-point cycle after initial frame = {RAMP_DURATION_S + EXPOSURE_DURATION_S:g} s",
                f"ZFC/FC barrier calibration time = {v620.PARAMS.zfc_fc_observation_time_s:g} s",
                "No additional magnetic observation window is used.",
                "Magnetic energies are averaged over each actual SAXS exposure.",
                "Every temperature setpoint starts from a uniform 1/64 pair-state distribution.",
                "No pair-state probability is inherited from the preceding temperature.",
                "Dipolar coupling acts during each 180 s exposure; ramps are not propagated magnetically.",
                "No contact/population switch is used.",
                f"ligand length per surface = {LIGAND_LENGTH_NM:g} nm",
                f"core-surface gap = {SURFACE_GAP_NM:g} nm",
                f"face center distance = {face['center_distance_m'] * 1e9:.9f} nm",
                f"tip center distance = {tip['center_distance_m'] * 1e9:.9f} nm",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(png_path)
    print(frame_csv_path)
    print(trajectory_csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
