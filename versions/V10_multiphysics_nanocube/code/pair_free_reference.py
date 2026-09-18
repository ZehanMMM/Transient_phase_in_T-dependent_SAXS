"""Fixed-geometry, 64-well magnetic thermodynamics without rotation penalties.

The reference is TWO UNCOUPLED particles, not the other pair geometry.
All 8 <111> minima have the same anisotropy energy, subtracted to zero.
This is a deep-well approximation. No intrawell q(T) rescaling is used:
the Hamiltonian is temperature independent and is NOT a continuous-spin model.
The inherited anisotropy barrier enters Neel rates, not occupied-state energy.
Rigid-body Brownian dynamics and assembly free energy are not computed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.special import logsumexp, xlogy

import geometry_model as geometry
from pair_energy_model import brownian_time_s, hexane_viscosity_Pa_s


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "free_reference"
COLORS = {"face": "#B54435", "tip": "#E69B50", "vdw": "#287A96"}


def equilibrium(energies_J, temperature_K):
    """F relative to an equal-weight, zero-energy uncoupled state space."""
    e = np.asarray(energies_J, dtype=float) / (Boltzmann * temperature_K)
    log_z = logsumexp(-e)
    p_eq = np.exp(-e - log_z)
    return p_eq, -log_z + np.log(len(e))


def state_thermodynamics(p, energies_kBT, p_eq):
    """Return U, F-F_free and F-F_eq, all in kBT, for equal-volume states."""
    p = np.asarray(p, dtype=float)
    u = float(p @ energies_kBT)
    f = u + float(np.sum(xlogy(p, len(p) * p)))
    kl = float(np.sum(xlogy(p, p) - p * np.log(p_eq)))
    return u, f, kl


class ReversibleEvolution:
    """Exact constant-T spectral propagator of a row-generator master equation."""

    def __init__(self, generator, p_eq, p_initial):
        q = np.asarray(generator)
        self.p_eq = np.asarray(p_eq)
        self.sqrt_pi = np.sqrt(p_eq)
        scale = max(float(np.max(np.abs(q))), 1e-300)
        if np.max(np.abs(q.sum(axis=1))) > 1e-10 * scale:
            raise ValueError("Generator rows do not sum to zero")
        off_diagonal = q.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.min(off_diagonal) < -1e-12 * scale:
            raise ValueError("Negative transition rate")
        symmetric = self.sqrt_pi[:, None] * q / self.sqrt_pi[None, :]
        self.balance_error = float(np.max(np.abs(symmetric - symmetric.T)) / scale)
        if self.balance_error > 1e-10:
            raise ValueError("Rates fail detailed balance for the supplied energies")
        self.eigenvalues, self.vectors = eigh((symmetric + symmetric.T) / 2)
        if np.max(self.eigenvalues) > 1e-10 * scale:
            raise ValueError("Positive generator eigenvalue")
        self.eigenvalues = np.minimum(self.eigenvalues, 0.0)
        self.eigenvalues[-1] = 0.0
        self.coefficients = (np.asarray(p_initial) / self.sqrt_pi) @ self.vectors

    def _probability(self, factors):
        p = ((self.coefficients * factors) @ self.vectors.T) * self.sqrt_pi
        if np.min(p) < -1e-9 or abs(np.sum(p) - 1.0) > 1e-8:
            raise ArithmeticError("Probability propagation lost positivity or normalization")
        p = np.maximum(p, 0.0)
        return p / np.sum(p)

    def at(self, time_s):
        return self._probability(np.exp(self.eigenvalues * time_s))

    def mean(self, window_s):
        x = self.eigenvalues * window_s
        phi = np.ones_like(x)
        np.divide(np.expm1(x), x, out=phi, where=x != 0)
        return self._probability(phi)

    def mean_kl(self, window_s):
        """Integrate nonlinear KL, resolving fast initial modes explicitly."""
        fastest = max(-float(self.eigenvalues[0]), 1e-300)
        first = min(window_s, 0.01 / fastest)
        points = np.unique(np.r_[0.0, np.geomspace(first, window_s, 18)])

        def kl(time_s):
            p = self.at(time_s)
            return float(np.sum(xlogy(p, p) - p * np.log(self.p_eq)))

        integrals, errors = [], []
        for left, right in zip(points[:-1], points[1:]):
            value, error = quad(kl, left, right, epsabs=1e-10 * window_s,
                                epsrel=1e-9)
            integrals.append(value)
            errors.append(error)
        return sum(integrals) / window_s, sum(errors) / window_s


def compute_point(temperature_K, direction, window_s=20.0, gap_nm=3.0,
                  coupling_scale=1.0):
    params = geometry.PARAMS
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    distance = geometry.center_distance_at_gap_m(direction, gap_nm * 1e-9, params)
    energies, prefactor, states, rhat = geometry.easy_axis_pair_energies_J(
        distance * direction, params)
    energies = coupling_scale * energies
    prefactor *= coupling_scale
    kbt = Boltzmann * temperature_K
    e = energies / kbt
    p_eq, f_eq = equilibrium(energies, temperature_K)
    p_initial = np.full(64, 1 / 64)
    generator = geometry.neel_pair_generator(
        temperature_K, params.zfc_fc_activation_barrier_J, energies,
        prefactor, states, rhat, params)
    evolution = ReversibleEvolution(generator, p_eq, p_initial)
    p_end = evolution.at(window_s)
    p_mean = evolution.mean(window_s)
    u_end, f_end, kl_end = state_thermodynamics(p_end, e, p_eq)
    u_mean, f_of_mean, kl_of_mean = state_thermodynamics(p_mean, e, p_eq)
    kl_mean, integration_error = evolution.mean_kl(window_s)
    f_mean = f_eq + kl_mean
    u_eq = float(p_eq @ e)
    checks = [abs(f_end - f_eq - kl_end),
              abs(f_of_mean - f_eq - kl_of_mean)]
    if max(checks) > 1e-9 or min(kl_mean, kl_end) < -1e-9:
        raise ArithmeticError("Free-energy / KL identity failed")
    sample_times = np.r_[0.0, np.geomspace(window_s * 1e-9, window_s, 60)]
    sample_f = [state_thermodynamics(evolution.at(t), e, p_eq)[1]
                for t in sample_times]
    if np.max(np.diff(sample_f)) > 1e-8:
        raise ArithmeticError("Free energy increases under fixed-T detailed balance")
    vdw = geometry.pair_vdw_energy_J(distance * direction, params)
    result = {
        "temperature_K": temperature_K, "window_s": window_s,
        "center_distance_nm": distance * 1e9, "surface_gap_nm": gap_nm,
        "kBT_J": kbt, "U_initial_kBT": float(p_initial @ e),
        "Udd_eq_kBT": u_eq, "Udd_time_mean_kBT": u_mean,
        "Udd_end_kBT": u_end,
        "delta_Uani_kBT": 0.0,
        "delta_Fmag_eq_kBT": f_eq,
        "delta_Fmag_time_mean_kBT": f_mean,
        "delta_Fmag_end_kBT": f_end,
        "delta_Fmag_of_mean_probability_kBT": f_of_mean,
        "relaxation_free_energy_excess_time_mean_kBT": kl_mean,
        "relaxation_free_energy_excess_end_kBT": kl_end,
        "internal_energy_relaxation_deficit_kBT": u_mean - u_eq,
        "vdW_kBT": vdw / kbt,
        "partial_contact_F_eq_kBT": f_eq + vdw / kbt,
        "partial_contact_F_time_mean_kBT": f_mean + vdw / kbt,
        "tau_N_isolated_s": params.attempt_time_s * np.exp(
            params.zfc_fc_activation_barrier_J / kbt),
        "tau_B_free_s": float(brownian_time_s(np.array([temperature_K]))[0]),
        "hexane_viscosity_Pa_s": float(hexane_viscosity_Pa_s(
            np.array([temperature_K]))[0]),
        "detailed_balance_relative_error": evolution.balance_error,
        "KL_quadrature_error_kBT": integration_error,
        "free_energy_identity_error_kBT": max(checks),
        # Unknown is deliberately not replaced by a zero interaction.
        "ligand_solvent_F_J": np.nan,
        "assembly_translation_rotation_F_J": np.nan,
        "assembly_delta_G_J": np.nan,
    }
    for name, value in list(result.items()):
        if name.endswith("_kBT"):
            result[name[:-4] + "_J"] = value * kbt
    return result, p_initial, p_eq, p_end, p_mean


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_results(records, output, window_s):
    plt.rcParams.update({"font.family": "Arial", "font.size": 13,
                         "axes.labelsize": 16, "axes.linewidth": 1.8,
                         "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.grid": False, "pdf.fonttype": 42})
    groups = {g: [r for r in records if r["geometry"] == g] for g in COLORS if g != "vdw"}
    figures = {
        "magnetic_energy": ("Udd_time_mean", "Udd_eq", "Magnetic interaction energy"),
        "magnetic_free_energy": ("delta_Fmag_time_mean", "delta_Fmag_eq", "Magnetic free energy vs uncoupled pair"),
        "relaxation_excess": ("relaxation_free_energy_excess_time_mean", None, "Unrelaxed free-energy excess"),
    }
    for name, (field, eq_field, title) in figures.items():
        for unit in ("kBT", "J"):
            fig, ax = plt.subplots(figsize=(7.2, 8.8))
            ax.set_box_aspect(1)
            for g, rows in groups.items():
                t = [r["temperature_K"] for r in rows]
                ax.plot(t, [r[f"{field}_{unit}"] for r in rows], color=COLORS[g],
                        lw=2.5, label=f"{g}, {window_s:g} s mean")
                if eq_field:
                    ax.plot(t, [r[f"{eq_field}_{unit}"] for r in rows], color=COLORS[g],
                            lw=1.8, ls="--", label=f"{g}, equilibrium")
            if name == "magnetic_energy":
                rows = groups["face"]
                t = [r["temperature_K"] for r in rows]
                ax.plot(t, [r[f"vdW_{unit}"] for r in rows], color=COLORS["vdw"],
                        lw=2, label="face vdW (separate)")
                ax.plot(t, [1.0 if unit == "kBT" else r["kBT_J"] for r in rows],
                        color="black", lw=1.6, label=r"$k_BT$")
            ax.axhline(0, color="0.55", lw=0.8)
            ax.set(xlabel="Temperature (K)",
                   ylabel=r"Energy / $k_BT$" if unit == "kBT" else "Energy (J)",
                   title=title, xlim=(200, 300))
            if unit == "J":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16),
                      ncol=1, frameon=False, fontsize=12, borderaxespad=0)
            fig.subplots_adjust(left=0.17, right=0.96, top=0.92, bottom=0.34)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{name}_{unit}.{extension}", dpi=300,
                            bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--temperature-step", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.window_s <= 0 or not 0 < args.temperature_step <= 100:
        parser.error("Window and grid spacing must be positive (spacing <= 100 K)")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, probabilities = [], []
    temperatures = np.unique(np.r_[np.arange(200, 300, args.temperature_step), 250, 300])
    for g, direction in (("face", [1, 0, 0]), ("tip", [1, 1, 1])):
        for temperature in temperatures:
            row, p0, peq, pend, pmean = compute_point(temperature, direction, args.window_s)
            row = {"geometry": g, **row}
            records.append(row)
            for i in range(64):
                probabilities.append({"geometry": g, "temperature_K": temperature,
                                      "state_index": i, "spin1_index": i // 8,
                                      "spin2_index": i % 8, "p_initial": p0[i],
                                      "p_equilibrium": peq[i], "p_end": pend[i],
                                      "p_time_mean": pmean[i]})
        print(f"Completed {g}: {len(temperatures)} temperatures", flush=True)
    write_csv(output / "energies.csv", records)
    write_csv(output / "state_probabilities.csv", probabilities)
    metadata = {
        "window_s": args.window_s, "initial_state": "uniform 1/64, reset at each T",
        "reference": "two uncoupled particles with identical magnetic state spaces",
        "approximation": "64 exact easy-axis states, no intrawell q(T) rescaling",
        "anisotropy": "all easy-axis minima subtracted to zero; barrier enters rates only",
        "rigid_body_rotation": "not simulated; free Brownian times are diagnostics only",
        "temperature_step_K": args.temperature_step,
        "core_nm": geometry.PARAMS.particle_size_nm, "ligand_nm": 1.5,
        "gap_face_nm": 3.0, "gap_tip_nm": 3.0,
        "Ms_A_per_m": geometry.PARAMS.saturation_magnetization_Apm,
        "Ms_T_dependence": "constant sample 300 K value, no measured curve available",
        "tau0_s": geometry.PARAMS.attempt_time_s,
        "barrier_J": geometry.PARAMS.zfc_fc_activation_barrier_J,
        "TB_K": geometry.PARAMS.blocking_temperature_K, "TB_calibration_time_s": 100,
        "size_CV": 0, "Hamaker_J": geometry.PARAMS.hamaker_J,
        "assembly_free_energy": "not computed: ligand/solvent and concentration missing",
        "hexane_below_250K": "extrapolation, not experimentally constrained",
        "state_directions": geometry.cubic_easy_axis_states().tolist(),
    }
    (output / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(records, output, args.window_s)
    print(f"Outputs: {output}")
    for row in records:
        if row["temperature_K"] in (200, 250, 300):
            print(row["geometry"], row["temperature_K"],
                  "Umean/kBT=", round(row["Udd_time_mean_kBT"], 5),
                  "Fmean/kBT=", round(row["delta_Fmag_time_mean_kBT"], 5),
                  "Feq/kBT=", round(row["delta_Fmag_eq_kBT"], 5))


if __name__ == "__main__":
    main()
