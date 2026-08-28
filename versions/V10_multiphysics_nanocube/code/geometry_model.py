"""Two-particle temperature comparison using the V6.20 Hamiltonian.

The calculation deliberately keeps the V6.20 material parameters and model
choices:

* 16 nm Fe3O4 cubes with 1.5 nm Minkowski rounding;
* sample-specific Ms(300 K) = 55 emu/g = 55 A m2/kg, converted with
  rho(Fe3O4) = 5.18e3 kg/m3 to 2.849e5 A/m (rounded to 2.85e5 A/m);
* representative nonpolar-solvent Hamaker constant A = 2.0e-20 J, within
  the 9--29 zJ Lifshitz range reported by Faure et al., Langmuir 2011,
  DOI 10.1021/la201387d;
* 4x4x4 sharp-cube voxel quadrature for the van der Waals integral;
* the exact rounded-cube surface gap calibrated from the V6.20 reference
  state (a, alpha, gamma) = (21 nm, 74.2 deg, -15 deg).

The pair outputs count only one pair of nanocubes.  Additional coordination
outputs estimate the energy per NC as (n/2) times the pair energy, using n=6
for face-to-face and n=8 for tip-to-tip.

The D=5 nm output uses a 64-state master equation for two coupled moments in
the eight cubic <111> wells.  Adjacent-well Néel rates contain the dipolar
energy at the <110> saddle, are observed over a configurable experimental
window, and are broadened by a configurable lognormal diameter distribution.
The cubic-anisotropy coefficient is independently inferred from the 100 s
ZFC/FC condition tau(250 K)=100 s using
Delta E = K_cubic V / 12.  The older conditional +[111] equilibrium
calculation is retained below for audit and reference-gap outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.constants import Boltzmann, mu_0
from scipy.linalg import expm
from scipy.optimize import brentq


@dataclass(frozen=True)
class V620Parameters:
    particle_size_nm: float = 16.0
    # Present-particle magnetometry at 300 K: 55 emu/g = 55 A m2/kg.
    # Multiplication by rho(Fe3O4)=5.18e3 kg/m3 gives 2.849e5 A/m,
    # rounded here to 2.85e5 A/m. This conversion assumes the reported
    # emu/g is normalized to inorganic Fe3O4 mass rather than ligand-bearing
    # total sample mass.
    saturation_magnetization_mass_Am2pkg: float = 55.0
    magnetite_density_kgpm3: float = 5.18e3
    saturation_magnetization_Apm: float = 285_000.0
    saturation_magnetization_reference_temperature_K: float = 300.0
    # Representative value for magnetite across a nonpolar solvent.  Faure
    # et al. report 9--29 zJ for hexane/toluene; the exact value is
    # medium-dependent and should be replaced when the experimental solvent
    # and its optical data are specified. Langmuir 27, 8659 (2011),
    # DOI 10.1021/la201387d.
    hamaker_J: float = 2.0e-20
    voxel_count_per_axis: int = 4
    roundness_nm: float = 1.5
    vdw_d2_floor_m2: float = 1.0e-19
    reference_a_nm: float = 21.0
    reference_alpha_deg: float = 74.2
    reference_gamma_deg: float = -15.0
    reference_temperature_K: float = 298.15
    blocking_temperature_K: float = 250.0
    zfc_fc_observation_time_s: float = 100.0
    experimental_observation_time_s: float = 1.0
    # Atomistic-spin-dynamics result for magnetite with cubic anisotropy at
    # 300 K: tau0 = 0.98 +/- 0.13 ns. Moreno et al., Phys. Rev. B 112,
    # 024429 (2025), DOI 10.1103/vmwp-q427.
    attempt_time_s: float = 0.98e-9
    orientation_gauss_count: int = 40
    orientation_phi_count: int = 80
    # Monodisperse baseline. Replace this temporary CV=0 assumption by the
    # coefficient of variation measured from the present sample's TEM size
    # distribution when those data are available.
    diameter_coefficient_of_variation: float = 0.0
    barrier_distribution_count: int = 11

    @property
    def particle_volume_m3(self) -> float:
        return (self.particle_size_nm * 1e-9) ** 3

    @property
    def zfc_fc_activation_barrier_J(self) -> float:
        if self.zfc_fc_observation_time_s <= self.attempt_time_s:
            raise ValueError(
                "zfc_fc_observation_time_s must exceed attempt_time_s."
            )
        return (
            Boltzmann
            * self.blocking_temperature_K
            * np.log(
                self.zfc_fc_observation_time_s / self.attempt_time_s
            )
        )

    @property
    def effective_barrier_anisotropy_Jpm3(self) -> float:
        """K_eff defined by Delta E = K_eff V."""
        return self.zfc_fc_activation_barrier_J / self.particle_volume_m3

    @property
    def magnetocrystalline_anisotropy_Jpm3(self) -> float:
        """Coefficient in V6.20 Eani; adjacent <111> barrier is K V / 12."""
        return 12.0 * self.effective_barrier_anisotropy_Jpm3


PARAMS = V620Parameters()
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "V6.20_energy_landscape_EaAlpha"
    / "generated"
)
TEMPERATURES_K = np.unique(
    np.sort(
        np.append(
            np.arange(100.0, 850.0 + 0.1, 5.0),
            [
                PARAMS.reference_temperature_K,
                223.15,
                233.15,
                253.15,
                273.15,
                293.15,
                303.15,
            ],
        )
    )
)
DYNAMIC_TEMPERATURES_K = np.unique(
    np.sort(
        np.append(
            np.arange(200.0, 300.0 + 0.1, 2.0),
            [223.15, 233.15, 250.0, 253.15, 273.15, 293.15],
        )
    )
)
EXPERIMENTAL_WINDOWS_S = (1.0, 10.0, 100.0)
COORDINATION_NUMBERS = {
    "face_to_face": 6,
    "tip_to_tip": 8,
}


def basis_unit(alpha_deg: float) -> np.ndarray:
    """V6.20 rhombohedral primitive-vector directions."""
    alpha = np.deg2rad(alpha_deg)
    cp2 = (2.0 * np.cos(alpha) + 1.0) / 3.0
    cp = np.sqrt(max(cp2, 0.0))
    sp = np.sqrt(max(1.0 - cp**2, 0.0))
    return np.array(
        [
            [cp, sp, 0.0],
            [cp, -0.5 * sp, (np.sqrt(3.0) / 2.0) * sp],
            [cp, -0.5 * sp, -(np.sqrt(3.0) / 2.0) * sp],
        ]
    )


def rotation_about_field(gamma_deg: float) -> np.ndarray:
    """V6.20 rotation about the laboratory field axis."""
    gamma = np.deg2rad(gamma_deg)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])


def exact_rounded_cube_gap_m(
    center_vector_body_m: np.ndarray, params: V620Parameters = PARAMS
) -> float:
    """Exact signed Euclidean gap for two co-oriented V6.20 rounded cubes."""
    size_m = params.particle_size_nm * 1e-9
    roundness_m = params.roundness_nm * 1e-9
    core_difference_halfwidth_m = size_m - 2.0 * roundness_m
    z = np.abs(np.asarray(center_vector_body_m)) - core_difference_halfwidth_m
    outside = np.linalg.norm(np.maximum(z, 0.0))
    inside = min(float(np.max(z)), 0.0)
    return outside + inside - 2.0 * roundness_m


def v620_reference_gap_m(params: V620Parameters = PARAMS) -> float:
    """Reproduce D_ref from the V6.20 experimental reference state."""
    q_sc = basis_unit(90.0).T
    orientation = rotation_about_field(params.reference_gamma_deg) @ q_sc
    primitive_basis = (
        basis_unit(params.reference_alpha_deg) * params.reference_a_nm * 1e-9
    )
    grid = np.array(
        np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
    ).T.reshape(-1, 3)
    neighbour_indices = grid[np.any(grid != 0, axis=1)]
    neighbour_vectors_lab = neighbour_indices @ primitive_basis
    neighbour_vectors_body = neighbour_vectors_lab @ orientation
    gaps = np.array(
        [exact_rounded_cube_gap_m(vector, params) for vector in neighbour_vectors_body]
    )
    return float(np.min(gaps))


def center_distance_at_gap_m(
    direction_body: np.ndarray,
    target_gap_m: float,
    params: V620Parameters = PARAMS,
) -> float:
    """Solve the V6.20 exact-gap equation along a specified body direction."""
    direction_body = np.asarray(direction_body, dtype=float)
    direction_body /= np.linalg.norm(direction_body)
    size_m = params.particle_size_nm * 1e-9

    def objective(center_distance_m: float) -> float:
        return (
            exact_rounded_cube_gap_m(center_distance_m * direction_body, params)
            - target_gap_m
        )

    return float(brentq(objective, 0.25 * size_m, 4.0 * size_m, xtol=1e-18))


def voxel_centers_and_volume(
    params: V620Parameters = PARAMS,
) -> tuple[np.ndarray, float]:
    """Return the V6.20 4^3 sharp-cube voxel centers and voxel volume."""
    size_m = params.particle_size_nm * 1e-9
    count = params.voxel_count_per_axis
    step = size_m / count
    lin = np.linspace(-size_m / 2.0 + step / 2.0, size_m / 2.0 - step / 2.0, count)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return centers, step**3


def pair_vdw_energy_J(
    center_vector_body_m: np.ndarray, params: V620Parameters = PARAMS
) -> float:
    """V6.20 sharp-cube voxel Hamaker energy for one isolated pair."""
    voxel_centers, voxel_volume = voxel_centers_and_volume(params)
    voxel_delta = voxel_centers[:, None, :] - voxel_centers[None, :, :]
    separation = voxel_delta - np.asarray(center_vector_body_m)[None, None, :]
    distance_squared = np.einsum("...i,...i->...", separation, separation)
    coefficient = -(params.hamaker_J / np.pi**2) * voxel_volume**2
    return float(
        coefficient
        * np.sum(1.0 / np.maximum(distance_squared, params.vdw_d2_floor_m2) ** 3)
    )


def orientation_quadrature(
    params: V620Parameters = PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized product Gauss-Legendre/uniform-azimuth sphere quadrature."""
    cos_theta, gauss_weights = np.polynomial.legendre.leggauss(
        params.orientation_gauss_count
    )
    phi = (
        2.0
        * np.pi
        * np.arange(params.orientation_phi_count)
        / params.orientation_phi_count
    )
    z = np.repeat(cos_theta, params.orientation_phi_count)
    phi_grid = np.tile(phi, params.orientation_gauss_count)
    radial = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    directions = np.column_stack(
        [radial * np.cos(phi_grid), radial * np.sin(phi_grid), z]
    )
    # dOmega/(4*pi) = d(cos(theta))*dphi/(4*pi).
    weights = np.repeat(
        gauss_weights / (2.0 * params.orientation_phi_count),
        params.orientation_phi_count,
    )
    if not np.isclose(np.sum(weights), 1.0, atol=1e-14):
        raise RuntimeError("Orientation quadrature is not normalized.")
    return directions, weights


def single_particle_anisotropy_J(
    moment_directions_body: np.ndarray,
    params: V620Parameters = PARAMS,
) -> np.ndarray:
    """V6.20 cubic magnetocrystalline anisotropy for each direction."""
    mx, my, mz = np.asarray(moment_directions_body).T
    particle_volume_m3 = (params.particle_size_nm * 1e-9) ** 3
    return -params.magnetocrystalline_anisotropy_Jpm3 * particle_volume_m3 * (
        (mx * my) ** 2 + (mx * mz) ** 2 + (my * mz) ** 2
    )


def selected_easy_axis_basin_quadrature(
    params: V620Parameters = PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the +[111] octant without sampling its measure-zero edges."""
    polar_count = max(params.orientation_gauss_count // 2, 2)
    azimuth_count = max(params.orientation_phi_count // 4, 2)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(
        polar_count
    )
    cos_theta = 0.5 * (gauss_nodes + 1.0)
    cos_theta_weights = 0.5 * gauss_weights
    phi = (
        np.arange(azimuth_count, dtype=float) + 0.5
    ) * (0.5 * np.pi / azimuth_count)
    z = np.repeat(cos_theta, azimuth_count)
    phi_grid = np.tile(phi, polar_count)
    radial = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    basin_directions = np.column_stack(
        [
            radial * np.cos(phi_grid),
            radial * np.sin(phi_grid),
            z,
        ]
    )
    basin_weights = np.repeat(
        cos_theta_weights / azimuth_count,
        azimuth_count,
    )
    if not np.isclose(np.sum(basin_weights), 1.0, atol=1e-14):
        raise RuntimeError("Easy-axis-basin quadrature is not normalized.")
    easy_axis = np.ones(3) / np.sqrt(3.0)
    return basin_directions, basin_weights, easy_axis


def thermal_magnetic_pair_curves(
    temperatures_K: np.ndarray,
    center_vector_body_m: np.ndarray,
    params: V620Parameters = PARAMS,
) -> dict[str, np.ndarray]:
    """Jointly average a correlated two-dipole +[111] equilibrium.

    Within the selected cubic-anisotropy basins, the pair distribution is

        rho(m1,m2|T) proportional to
            exp[-(Eani1 + Eani2 + Edd)/(kB T)].

    There is no inter-basin relaxation or thermal-history variable.  Edd is
    included in the joint equilibrium weight, so the two orientations develop
    a temperature-dependent cross-correlation.
    """
    temperatures_K = np.asarray(temperatures_K, dtype=float)
    center_vector_body_m = np.asarray(center_vector_body_m, dtype=float)
    distance_m = np.linalg.norm(center_vector_body_m)
    separation_hat = center_vector_body_m / distance_m

    directions, direction_weights, easy_axis = (
        selected_easy_axis_basin_quadrature(params)
    )
    anisotropy_single_J = single_particle_anisotropy_J(directions, params)
    magnetic_moment_Am2 = (
        params.saturation_magnetization_Apm
        * (params.particle_size_nm * 1e-9) ** 3
    )
    dipole_prefactor_J = (
        (mu_0 / (4.0 * np.pi)) * magnetic_moment_Am2**2 / distance_m**3
    )
    direction_dot_matrix = directions @ directions.T
    separation_projection = directions @ separation_hat
    dipole_angular_matrix = (
        direction_dot_matrix
        - 3.0 * np.outer(separation_projection, separation_projection)
    )
    dipole_energy_matrix_J = dipole_prefactor_J * dipole_angular_matrix
    anisotropy_pair_matrix_J = (
        anisotropy_single_J[:, None] + anisotropy_single_J[None, :]
    )
    joint_energy_matrix_J = anisotropy_pair_matrix_J + dipole_energy_matrix_J
    joint_quadrature_weight = np.outer(direction_weights, direction_weights)

    result = {
        "mean_dipole_J": np.empty_like(temperatures_K),
        "uncorrelated_dipole_from_marginals_J": np.empty_like(temperatures_K),
        "dipole_correlation_correction_J": np.empty_like(temperatures_K),
        "mean_anisotropy_pair_J": np.empty_like(temperatures_K),
        "mean_anisotropy_two_isolated_J": np.empty_like(temperatures_K),
        "anisotropy_response_J": np.empty_like(temperatures_K),
        "magnetic_interaction_energy_J": np.empty_like(temperatures_K),
        "magnetic_pair_free_energy_J": np.empty_like(temperatures_K),
        "mean_total_pair_magnetic_J": np.empty_like(temperatures_K),
        "mean_moment_dot_product": np.empty_like(temperatures_K),
        "mean_easy_axis_projection": np.empty_like(temperatures_K),
        "activation_barrier_J": np.full_like(
            temperatures_K, params.zfc_fc_activation_barrier_J
        ),
        "effective_barrier_anisotropy_Jpm3": np.full_like(
            temperatures_K, params.effective_barrier_anisotropy_Jpm3
        ),
        "cubic_anisotropy_coefficient_Jpm3": np.full_like(
            temperatures_K, params.magnetocrystalline_anisotropy_Jpm3
        ),
        "orientation_temperature_K": temperatures_K.copy(),
    }

    single_minimum_J = float(np.min(anisotropy_single_J))
    for index, bath_temperature_K in enumerate(temperatures_K):
        thermal_J = Boltzmann * bath_temperature_K
        single_boltzmann = direction_weights * np.exp(
            -(anisotropy_single_J - single_minimum_J) / thermal_J
        )
        single_partition_shifted = float(np.sum(single_boltzmann))
        normalized_single_weight = single_boltzmann / single_partition_shifted
        mean_single_anisotropy_J = float(
            np.sum(normalized_single_weight * anisotropy_single_J)
        )

        joint_boltzmann = joint_quadrature_weight * np.exp(
            -(joint_energy_matrix_J - 2.0 * single_minimum_J) / thermal_J
        )
        joint_partition_shifted = float(np.sum(joint_boltzmann))
        joint_probability = joint_boltzmann / joint_partition_shifted

        mean_dipole_J = float(
            np.sum(joint_probability * dipole_energy_matrix_J)
        )
        mean_anisotropy_pair_J = float(
            np.sum(joint_probability * anisotropy_pair_matrix_J)
        )
        marginal_1 = np.sum(joint_probability, axis=1)
        marginal_2 = np.sum(joint_probability, axis=0)
        mean_moment_1 = marginal_1 @ directions
        mean_moment_2 = marginal_2 @ directions
        uncorrelated_angular_factor = (
            np.dot(mean_moment_1, mean_moment_2)
            - 3.0
            * np.dot(mean_moment_1, separation_hat)
            * np.dot(mean_moment_2, separation_hat)
        )
        uncorrelated_dipole_J = float(
            dipole_prefactor_J * uncorrelated_angular_factor
        )
        correlation_correction_J = mean_dipole_J - uncorrelated_dipole_J
        two_isolated_anisotropy_J = 2.0 * mean_single_anisotropy_J
        anisotropy_response_J = (
            mean_anisotropy_pair_J - two_isolated_anisotropy_J
        )
        mean_easy_axis_projection = float(
            0.5
            * (
                np.dot(mean_moment_1, easy_axis)
                + np.dot(mean_moment_2, easy_axis)
            )
        )
        magnetic_pair_free_energy_J = float(
            -thermal_J
            * np.log(
                joint_partition_shifted / single_partition_shifted**2
            )
        )

        result["mean_dipole_J"][index] = mean_dipole_J
        result["uncorrelated_dipole_from_marginals_J"][
            index
        ] = uncorrelated_dipole_J
        result["dipole_correlation_correction_J"][
            index
        ] = correlation_correction_J
        result["mean_anisotropy_pair_J"][index] = mean_anisotropy_pair_J
        result["mean_anisotropy_two_isolated_J"][
            index
        ] = two_isolated_anisotropy_J
        result["anisotropy_response_J"][index] = anisotropy_response_J
        result["magnetic_interaction_energy_J"][index] = (
            mean_dipole_J + anisotropy_response_J
        )
        result["magnetic_pair_free_energy_J"][
            index
        ] = magnetic_pair_free_energy_J
        result["mean_moment_dot_product"][index] = float(
            np.sum(joint_probability * direction_dot_matrix)
        )
        result["mean_easy_axis_projection"][index] = mean_easy_axis_projection

    particle_volume_m3 = (params.particle_size_nm * 1e-9) ** 3
    two_particle_easy_axis_energy_J = (
        -2.0
        * params.magnetocrystalline_anisotropy_Jpm3
        * particle_volume_m3
        / 3.0
    )
    result["mean_anisotropy_excitation_J"] = (
        result["mean_anisotropy_pair_J"] - two_particle_easy_axis_energy_J
    )
    # Absolute anisotropy energy contains an arbitrary constant offset.
    # Magnetic energy therefore uses the thermal excitation above the two
    # +[111] easy-axis minima, not the raw negative Eani value.
    result["mean_total_pair_magnetic_J"] = (
        result["mean_dipole_J"] + result["mean_anisotropy_excitation_J"]
    )

    return result


def cubic_easy_axis_states() -> np.ndarray:
    """Return the eight normalized cubic <111> easy-axis directions."""
    return np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    ) / np.sqrt(3.0)


def barrier_distribution(
    params: V620Parameters = PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Lognormal barrier factors induced by a diameter distribution.

    The nominal NC diameter is the mean diameter and Delta E scales as D^3.
    Only the activation barrier is dispersed here; the nominal pair geometry
    and dipole moment are retained so that the broadening mechanism is explicit.
    """
    count = params.barrier_distribution_count
    if count < 1:
        raise ValueError("barrier_distribution_count must be positive.")
    if params.diameter_coefficient_of_variation < 0.0:
        raise ValueError("diameter_coefficient_of_variation must be nonnegative.")
    if params.diameter_coefficient_of_variation == 0.0 or count == 1:
        return np.ones(1), np.ones(1)
    nodes, weights = np.polynomial.hermite.hermgauss(count)
    log_sigma = np.sqrt(
        np.log1p(params.diameter_coefficient_of_variation**2)
    )
    diameter_factor = np.exp(
        -0.5 * log_sigma**2 + np.sqrt(2.0) * log_sigma * nodes
    )
    barrier_factor = diameter_factor**3
    normalized_weights = weights / np.sqrt(np.pi)
    normalized_weights /= np.sum(normalized_weights)
    return barrier_factor, normalized_weights


def easy_axis_pair_energies_J(
    center_vector_body_m: np.ndarray,
    params: V620Parameters = PARAMS,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Dipole energies for all 8 x 8 pairs of <111> easy-axis states."""
    center_vector_body_m = np.asarray(center_vector_body_m, dtype=float)
    distance_m = float(np.linalg.norm(center_vector_body_m))
    separation_hat = center_vector_body_m / distance_m
    states = cubic_easy_axis_states()
    magnetic_moment_Am2 = (
        params.saturation_magnetization_Apm * params.particle_volume_m3
    )
    prefactor_J = (
        (mu_0 / (4.0 * np.pi)) * magnetic_moment_Am2**2 / distance_m**3
    )
    energy_matrix_J = prefactor_J * (
        states @ states.T
        - 3.0
        * np.outer(states @ separation_hat, states @ separation_hat)
    )
    return energy_matrix_J.reshape(-1), prefactor_J, states, separation_hat


def neel_pair_generator(
    temperature_K: float,
    activation_barrier_J: float,
    pair_energies_J: np.ndarray,
    dipole_prefactor_J: float,
    easy_axis_states: np.ndarray,
    separation_hat: np.ndarray,
    params: V620Parameters = PARAMS,
) -> np.ndarray:
    """Build the 64-state generator for coupled two-particle Néel hopping.

    One sign change connects adjacent <111> wells through a <110> saddle.
    The saddle barrier contains the dipolar energy of the other particle.
    The factor 1/3 makes the total isolated escape rate reproduce
    tau = tau0 exp(Delta E/kBT) despite the three adjacent exits.
    """
    sign_to_index = {
        tuple(np.sign(state).astype(int)): index
        for index, state in enumerate(easy_axis_states)
    }
    generator = np.zeros((64, 64))
    thermal_J = Boltzmann * temperature_K

    for state_1_index, state_1 in enumerate(easy_axis_states):
        for state_2_index, state_2 in enumerate(easy_axis_states):
            pair_index = 8 * state_1_index + state_2_index
            initial_dipole_J = pair_energies_J[pair_index]
            for particle_index in (0, 1):
                current_index = (
                    state_1_index if particle_index == 0 else state_2_index
                )
                current_state = easy_axis_states[current_index]
                other_state = state_2 if particle_index == 0 else state_1
                for component_index in range(3):
                    destination_signs = np.sign(current_state).astype(int)
                    destination_signs[component_index] *= -1
                    destination_index = sign_to_index[
                        tuple(destination_signs)
                    ]

                    saddle_state = current_state.copy()
                    saddle_state[component_index] = 0.0
                    saddle_state /= np.linalg.norm(saddle_state)
                    saddle_dipole_J = dipole_prefactor_J * (
                        np.dot(saddle_state, other_state)
                        - 3.0
                        * np.dot(saddle_state, separation_hat)
                        * np.dot(other_state, separation_hat)
                    )
                    transition_barrier_J = max(
                        activation_barrier_J
                        + saddle_dipole_J
                        - initial_dipole_J,
                        0.0,
                    )
                    transition_rate_per_s = (
                        np.exp(-transition_barrier_J / thermal_J)
                        / (3.0 * params.attempt_time_s)
                    )
                    if particle_index == 0:
                        destination_pair_index = (
                            8 * destination_index + state_2_index
                        )
                    else:
                        destination_pair_index = (
                            8 * state_1_index + destination_index
                        )
                    generator[
                        pair_index, destination_pair_index
                    ] += transition_rate_per_s
            generator[pair_index, pair_index] = -np.sum(
                generator[pair_index]
            )
    return generator


def dynamic_neel_pair_curves(
    temperatures_K: np.ndarray,
    center_vector_body_m: np.ndarray,
    observation_time_s: float | None = None,
    params: V620Parameters = PARAMS,
) -> dict[str, np.ndarray]:
    """Observed two-NC magnetic energy after coupled Néel hopping.

    Both moments start in the +[111] well.  The 64-state master equation allows
    either moment to hop among all eight <111> wells.  A lognormal diameter
    distribution broadens the activation barriers.  Magnetocrystalline
    anisotropy is measured relative to the easy-axis minima: occupied discrete
    wells have Delta Eani = 0, while K enters every transition barrier.
    """
    temperatures_K = np.asarray(temperatures_K, dtype=float)
    if observation_time_s is None:
        observation_time_s = params.experimental_observation_time_s
    if observation_time_s <= 0.0:
        raise ValueError("observation_time_s must be positive.")
    (
        pair_energies_J,
        dipole_prefactor_J,
        easy_axis_states,
        separation_hat,
    ) = easy_axis_pair_energies_J(center_vector_body_m, params)
    barrier_factors, barrier_weights = barrier_distribution(params)
    sign_to_index = {
        tuple(np.sign(state).astype(int)): index
        for index, state in enumerate(easy_axis_states)
    }
    initial_easy_axis_index = sign_to_index[(1, 1, 1)]
    initial_pair_index = 9 * initial_easy_axis_index
    initial_probability = np.zeros(64)
    initial_probability[initial_pair_index] = 1.0

    result = {
        "mean_dipole_J": np.empty_like(temperatures_K),
        "mean_anisotropy_excitation_J": np.zeros_like(temperatures_K),
        "mean_magnetic_J": np.empty_like(temperatures_K),
        "initial_pair_probability": np.empty_like(temperatures_K),
    }
    for temperature_index, temperature_K in enumerate(temperatures_K):
        ensemble_probability = np.zeros(64)
        for barrier_factor, distribution_weight in zip(
            barrier_factors, barrier_weights
        ):
            generator = neel_pair_generator(
                temperature_K,
                params.zfc_fc_activation_barrier_J * barrier_factor,
                pair_energies_J,
                dipole_prefactor_J,
                easy_axis_states,
                separation_hat,
                params,
            )
            observed_probability = initial_probability @ expm(
                generator * observation_time_s
            )
            ensemble_probability += (
                distribution_weight * observed_probability
            )
        ensemble_probability = np.maximum(ensemble_probability, 0.0)
        ensemble_probability /= np.sum(ensemble_probability)
        mean_dipole_J = float(
            np.dot(ensemble_probability, pair_energies_J)
        )
        result["mean_dipole_J"][temperature_index] = mean_dipole_J
        result["mean_magnetic_J"][temperature_index] = mean_dipole_J
        result["initial_pair_probability"][temperature_index] = (
            ensemble_probability[initial_pair_index]
        )
    return result


def calculate_dynamic_five_nm_cases(
    temperatures_K: np.ndarray = DYNAMIC_TEMPERATURES_K,
    observation_time_s: float | None = None,
    params: V620Parameters = PARAMS,
) -> dict[str, dict]:
    """Calculate the dynamic face and tip cases at an exact 5 nm gap."""
    if observation_time_s is None:
        observation_time_s = params.experimental_observation_time_s
    cases: dict[str, dict] = {}
    for key, title, direction, coordination_number in (
        (
            "face_to_face",
            "Face-to-face",
            np.array([1.0, 0.0, 0.0]),
            COORDINATION_NUMBERS["face_to_face"],
        ),
        (
            "tip_to_tip",
            "Tip-to-tip",
            np.ones(3) / np.sqrt(3.0),
            COORDINATION_NUMBERS["tip_to_tip"],
        ),
    ):
        center_distance_m = center_distance_at_gap_m(
            direction, 5.0e-9, params
        )
        center_vector_m = center_distance_m * direction
        cases[key] = {
            "title": title,
            "center_distance_m": center_distance_m,
            "vdw_pair_J": pair_vdw_energy_J(center_vector_m, params),
            "observation_time_s": float(observation_time_s),
            "coordination_number": coordination_number,
        }
        cases[key].update(
            dynamic_neel_pair_curves(
                temperatures_K,
                center_vector_m,
                observation_time_s,
                params,
            )
        )
    return cases


def equilibrium_easy_axis_pair_curves(
    temperatures_K: np.ndarray,
    center_vector_body_m: np.ndarray,
    params: V620Parameters = PARAMS,
) -> dict[str, np.ndarray]:
    """Return the exact canonical average over the 64 easy-axis pair states."""
    temperatures_K = np.asarray(temperatures_K, dtype=float)
    pair_energies_J, _, _, _ = easy_axis_pair_energies_J(
        center_vector_body_m, params
    )
    shifted_energies_J = (
        pair_energies_J - np.min(pair_energies_J)
    )
    mean_dipole_J = np.empty_like(temperatures_K)
    for temperature_index, temperature_K in enumerate(
        temperatures_K
    ):
        boltzmann_weights = np.exp(
            -shifted_energies_J / (Boltzmann * temperature_K)
        )
        probabilities = boltzmann_weights / np.sum(
            boltzmann_weights
        )
        mean_dipole_J[temperature_index] = np.dot(
            probabilities, pair_energies_J
        )
    return {
        "mean_dipole_J": mean_dipole_J,
        "mean_anisotropy_excitation_J": np.zeros_like(
            temperatures_K
        ),
        "mean_magnetic_J": mean_dipole_J.copy(),
    }


def calculate_equilibrium_five_nm_cases(
    temperatures_K: np.ndarray = DYNAMIC_TEMPERATURES_K,
    params: V620Parameters = PARAMS,
) -> dict[str, dict]:
    """Calculate continuous, observation-time-independent equilibrium at D=5 nm."""
    equilibrium_params = replace(
        params,
        orientation_gauss_count=80,
        orientation_phi_count=160,
    )
    cases: dict[str, dict] = {}
    for key, title, direction, coordination_number in (
        (
            "face_to_face",
            "Face-to-face",
            np.array([1.0, 0.0, 0.0]),
            COORDINATION_NUMBERS["face_to_face"],
        ),
        (
            "tip_to_tip",
            "Tip-to-tip",
            np.ones(3) / np.sqrt(3.0),
            COORDINATION_NUMBERS["tip_to_tip"],
        ),
    ):
        center_distance_m = center_distance_at_gap_m(
            direction, 5.0e-9, params
        )
        center_vector_m = center_distance_m * direction
        cases[key] = {
            "title": title,
            "center_distance_m": center_distance_m,
            "vdw_pair_J": pair_vdw_energy_J(
                center_vector_m, params
            ),
            "coordination_number": coordination_number,
        }
        continuous_result = thermal_magnetic_pair_curves(
            temperatures_K, center_vector_m, equilibrium_params
        )
        cases[key].update(continuous_result)
        # For an assembly-driving energy, subtract the anisotropy energy of
        # two isolated particles at the same T.  This retains the anisotropy
        # change induced by Edd without adding the unrelated ~2 kBT
        # single-particle thermal-excitation background.
        cases[key]["mean_magnetic_J"] = continuous_result[
            "magnetic_interaction_energy_J"
        ]
    return cases


def calculate_cases(
    temperatures_K: np.ndarray = TEMPERATURES_K,
    params: V620Parameters = PARAMS,
) -> tuple[float, dict[str, dict]]:
    """Calculate thermally fluctuating face-to-face and tip-to-tip pairs."""
    reference_gap_m = v620_reference_gap_m(params)
    face_direction = np.array([1.0, 0.0, 0.0])
    tip_direction = np.ones(3) / np.sqrt(3.0)

    cases: dict[str, dict] = {}
    for key, title, direction, coordination_number in (
        ("face_to_face", "Face-to-face", face_direction, 6),
        ("tip_to_tip", "Tip-to-tip", tip_direction, 8),
    ):
        center_distance_m = center_distance_at_gap_m(direction, reference_gap_m, params)
        center_vector_m = center_distance_m * direction
        cases[key] = {
            "title": title,
            "direction": direction,
            "center_distance_m": center_distance_m,
            "vdw_J": pair_vdw_energy_J(center_vector_m, params),
            "coordination_number": coordination_number,
        }
        cases[key].update(
            thermal_magnetic_pair_curves(
                temperatures_K, center_vector_m, params
            )
        )
    return reference_gap_m, cases


def export_csv(
    temperatures_K: np.ndarray,
    reference_gap_m: float,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
    filename: str = "V620_two_NC_temperature_data.csv",
) -> None:
    """Export the thermally averaged magnetic decomposition."""
    thermal_J = Boltzmann * temperatures_K
    columns = [temperatures_K, thermal_J]
    names = ["temperature_K", "kBT_J"]
    for key in ("face_to_face", "tip_to_tip"):
        case = cases[key]
        vdw = np.full_like(temperatures_K, case["vdw_J"])
        coordination_number = int(
            case.get("coordination_number", 6 if key == "face_to_face" else 8)
        )
        vdw_multi_per_particle = (
            0.5 * coordination_number * vdw
        )
        magnetic = case["magnetic_interaction_energy_J"]
        columns.extend(
            [
                vdw,
                np.abs(vdw) / thermal_J,
                vdw_multi_per_particle,
                np.abs(vdw_multi_per_particle) / thermal_J,
                case["mean_dipole_J"],
                np.abs(case["mean_dipole_J"]) / thermal_J,
                case["uncorrelated_dipole_from_marginals_J"],
                case["dipole_correlation_correction_J"],
                case["magnetic_pair_free_energy_J"],
                case["mean_anisotropy_pair_J"],
                case["mean_anisotropy_excitation_J"],
                case["mean_anisotropy_excitation_J"] / thermal_J,
                case["mean_total_pair_magnetic_J"],
                np.abs(case["mean_total_pair_magnetic_J"]) / thermal_J,
                case["mean_anisotropy_two_isolated_J"],
                case["anisotropy_response_J"],
                magnetic,
                np.abs(magnetic) / thermal_J,
                case["activation_barrier_J"],
                case["effective_barrier_anisotropy_Jpm3"],
                case["cubic_anisotropy_coefficient_Jpm3"],
                case["orientation_temperature_K"],
                case["mean_moment_dot_product"],
                case["mean_easy_axis_projection"],
            ]
        )
        names.extend(
            [
                f"{key}_vdw_J",
                f"{key}_abs_vdw_over_kBT",
                f"{key}_vdw_multi_per_particle_J",
                f"{key}_abs_vdw_multi_per_particle_over_kBT",
                f"{key}_mean_dipole_J",
                f"{key}_abs_mean_dipole_over_kBT",
                f"{key}_uncorrelated_dipole_from_marginals_J",
                f"{key}_dipole_correlation_correction_J",
                f"{key}_magnetic_pair_free_energy_J",
                f"{key}_mean_anisotropy_pair_J",
                f"{key}_mean_anisotropy_excitation_J",
                f"{key}_mean_anisotropy_excitation_over_kBT",
                f"{key}_magnetic_energy_J",
                f"{key}_abs_magnetic_energy_over_kBT",
                f"{key}_mean_anisotropy_two_isolated_J",
                f"{key}_anisotropy_response_J",
                f"{key}_magnetic_interaction_energy_J",
                f"{key}_abs_magnetic_interaction_over_kBT",
                f"{key}_activation_barrier_J",
                f"{key}_effective_barrier_anisotropy_Jpm3",
                f"{key}_cubic_anisotropy_coefficient_Jpm3",
                f"{key}_bath_temperature_K",
                f"{key}_mean_moment_dot_product",
                f"{key}_local_mean_easy_axis_projection",
            ]
        )
    table = np.column_stack(columns)
    np.savetxt(
        OUTPUT_DIR / filename,
        table,
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def add_experimental_temperature_bands(
    ax: plt.Axes,
    temperature_unit: str = "K",
    show_labels: bool = True,
) -> None:
    """Shade the experimental and transient-aggregation temperature windows."""
    offset = 0.0 if temperature_unit == "C" else 273.15
    experimental_min = -40.0 + offset
    experimental_max = 20.0 + offset
    transient_min = -20.0 + offset
    transient_max = 0.0 + offset
    ax.axvspan(
        experimental_min,
        experimental_max,
        color="#d9d9d9",
        alpha=0.38,
        linewidth=0.0,
        zorder=-10,
    )
    ax.axvspan(
        transient_min,
        transient_max,
        color="#f6e58d",
        alpha=0.62,
        linewidth=0.0,
        zorder=-9,
    )
    if show_labels:
        if temperature_unit == "K":
            ax.text(
                10.0 + offset,
                0.68,
                "experimental range",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                rotation=90,
                color="#555555",
                fontsize=6.5,
                zorder=6,
            )
            ax.text(
                -10.0 + offset,
                0.68,
                "transient aggregation",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                rotation=90,
                color="#806600",
                fontsize=6.5,
                zorder=6,
            )
        else:
            ax.text(
                10.0 + offset,
                0.97,
                "experimental range",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color="#555555",
                fontsize=7,
                zorder=6,
            )
            ax.text(
                -10.0 + offset,
                0.87,
                "transient\naggregation",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color="#806600",
                fontsize=7,
                zorder=6,
            )


def add_physical_panel(
    ax: plt.Axes,
    temperatures_K: np.ndarray,
    case: dict,
    params: V620Parameters = PARAMS,
    temperature_unit: str = "K",
    xlim: tuple[float, float] | None = None,
    show_band_labels: bool = True,
    show_reference_line: bool = True,
) -> None:
    """Plot signed physical energies in zeptojoules."""
    if temperature_unit not in {"K", "C"}:
        raise ValueError("temperature_unit must be 'K' or 'C'.")
    temperature_x = (
        temperatures_K if temperature_unit == "K" else temperatures_K - 273.15
    )
    axis_offset = 0.0 if temperature_unit == "K" else -273.15
    thermal_zJ = Boltzmann * temperatures_K / 1e-21
    vdw_zJ = case["vdw_J"] / 1e-21
    coordination_number = int(
        case.get(
            "coordination_number",
            6 if case["title"].lower().startswith("face") else 8,
        )
    )
    vdw_multi_per_particle_zJ = (
        0.5 * coordination_number * vdw_zJ
    )
    # Use the interaction free energy relative to two isolated particles.
    # This removes the approximately 2 kBT thermal excitation background that
    # appears in <Edd> + <Delta Eani,1 + Delta Eani,2>.
    magnetic_zJ = case["magnetic_pair_free_energy_J"] / 1e-21
    add_experimental_temperature_bands(
        ax, temperature_unit, show_labels=show_band_labels
    )
    ax.plot(temperature_x, thermal_zJ, color="#252525", label=r"$k_BT$", zorder=3)
    ax.axhline(
        vdw_zJ,
        color="#2b6cb0",
        linestyle="--",
        label=r"$E_{\rm vdW}^{\rm pair}$",
    )
    ax.axhline(
        vdw_multi_per_particle_zJ,
        color="#2c7fb8",
        linestyle=":",
        linewidth=2.0,
        label=rf"$E_{{\rm vdW}}^{{\rm multi}}/NC$ ($z={coordination_number}$)",
    )
    ax.plot(
        temperature_x,
        magnetic_zJ,
        color="#c53030",
        linestyle="-.",
        label=r"$\Delta F_{\rm mag}^{\rm pair}$",
        zorder=3,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.65, zorder=1)
    if show_reference_line:
        ax.axvline(
            params.reference_temperature_K + axis_offset,
            color="#777777",
            linewidth=0.7,
            alpha=0.5,
        )
    ax.set(
        title=(
            f"{case['title']}: signed physical energies\n"
            f"center distance = {case['center_distance_m'] * 1e9:.3f} nm"
        ),
        xlabel=("Temperature (K)" if temperature_unit == "K" else "Temperature (°C)"),
        ylabel=r"Signed energy ($10^{-21}$ J)",
        xlim=(
            xlim
            if xlim is not None
            else (temperature_x[0], temperature_x[-1])
        ),
    )
    ax.margins(y=0.08)
    ax.grid(alpha=0.2)


def add_normalized_panel(
    ax: plt.Axes,
    temperatures_K: np.ndarray,
    case: dict,
    params: V620Parameters = PARAMS,
    temperature_unit: str = "K",
    xlim: tuple[float, float] | None = None,
    show_band_labels: bool = True,
    show_reference_line: bool = True,
) -> None:
    """Plot signed interaction energies normalized by kBT."""
    if temperature_unit not in {"K", "C"}:
        raise ValueError("temperature_unit must be 'K' or 'C'.")
    temperature_x = (
        temperatures_K if temperature_unit == "K" else temperatures_K - 273.15
    )
    axis_offset = 0.0 if temperature_unit == "K" else -273.15
    thermal_J = Boltzmann * temperatures_K
    vdw_ratio = case["vdw_J"] / thermal_J
    coordination_number = int(
        case.get(
            "coordination_number",
            6 if case["title"].lower().startswith("face") else 8,
        )
    )
    vdw_multi_per_particle_ratio = (
        0.5 * coordination_number * vdw_ratio
    )
    magnetic_ratio = case["magnetic_pair_free_energy_J"] / thermal_J
    add_experimental_temperature_bands(
        ax, temperature_unit, show_labels=show_band_labels
    )
    ax.axhline(1.0, color="#252525", label=r"$k_BT/k_BT=1$")
    ax.plot(
        temperature_x,
        vdw_ratio,
        color="#2b6cb0",
        linestyle="--",
        label=r"$E_{\rm vdW}^{\rm pair}/k_BT$",
    )
    ax.plot(
        temperature_x,
        vdw_multi_per_particle_ratio,
        color="#2c7fb8",
        linestyle=":",
        linewidth=2.0,
        label=rf"$E_{{\rm vdW}}^{{\rm multi}}/(NC\,k_BT)$ ($z={coordination_number}$)",
    )
    ax.plot(
        temperature_x,
        magnetic_ratio,
        color="#c53030",
        linestyle="-.",
        label=r"$\Delta F_{\rm mag}^{\rm pair}/k_BT$",
    )
    if show_reference_line:
        ax.axvline(
            params.reference_temperature_K + axis_offset,
            color="#777777",
            linewidth=0.7,
            alpha=0.5,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.65, zorder=1)
    ax.set(
        title=f"{case['title']}: competition with thermal energy",
        xlabel=("Temperature (K)" if temperature_unit == "K" else "Temperature (°C)"),
        ylabel=r"Signed dimensionless energy ($E/k_BT$)",
        xlim=(
            xlim
            if xlim is not None
            else (temperature_x[0], temperature_x[-1])
        ),
    )
    ax.margins(y=0.08)
    ax.grid(alpha=0.2)


def save_figures(
    temperatures_K: np.ndarray,
    reference_gap_m: float,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
) -> None:
    """Save one combined figure and one figure for each pair geometry."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for row, key in enumerate(("face_to_face", "tip_to_tip")):
        add_physical_panel(axes[row, 0], temperatures_K, cases[key], params)
        add_normalized_panel(axes[row, 1], temperatures_K, cases[key], params)
        axes[row, 0].legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none", fontsize=7,
        )
        axes[row, 1].legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none", fontsize=7,
        )
    figure.suptitle(
        (
            "16 nm Fe$_3$O$_4$: isolated-pair energy vs temperature "
            "(V6.20 method, correlated two-dipole equilibrium)\n"
            f"same exact rounded-surface gap $D_{{ref}}$ = "
            f"{reference_gap_m * 1e9:.3f} nm; zero external field"
        ),
        fontsize=11,
    )
    figure.savefig(OUTPUT_DIR / "V620_two_NC_temperature.png")
    figure.savefig(OUTPUT_DIR / "V620_two_NC_temperature.pdf")
    plt.close(figure)

    for key in ("face_to_face", "tip_to_tip"):
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), constrained_layout=True)
        add_physical_panel(axes[0], temperatures_K, cases[key], params)
        add_normalized_panel(axes[1], temperatures_K, cases[key], params)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(
            handles, labels, loc="upper right", frameon=True,
            facecolor="white", framealpha=0.92, edgecolor="none",
        )
        axes[1].legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none",
        )
        figure.suptitle(
            (
                f"16 nm Fe$_3$O$_4$, {cases[key]['title']} "
                f"($D_{{ref}}$={reference_gap_m * 1e9:.3f} nm, "
                "correlated two-dipole $+[111]$ equilibrium)"
            ),
            fontsize=11,
        )
        figure.savefig(OUTPUT_DIR / f"V620_{key}_temperature.png")
        figure.savefig(OUTPUT_DIR / f"V620_{key}_temperature.pdf")
        plt.close(figure)


def write_summary(
    temperatures_K: np.ndarray,
    reference_gap_m: float,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
) -> None:
    """Write a compact, human-readable audit of the calculation."""
    reference_kBT_J = Boltzmann * params.reference_temperature_K
    reference_index = int(
        np.argmin(np.abs(temperatures_K - params.reference_temperature_K))
    )
    if not np.isclose(
        temperatures_K[reference_index], params.reference_temperature_K
    ):
        raise RuntimeError("Reference temperature is missing from the scan.")
    lines = [
        "V6.20 thermally fluctuating two-NC comparison",
        "================================================",
        f"Particle: Fe3O4 rounded cube, L = {params.particle_size_nm:.3f} nm",
        f"Exact reference surface gap D_ref = {reference_gap_m * 1e9:.9f} nm",
        f"Reference temperature = {params.reference_temperature_K:.2f} K",
        f"kBT at reference temperature = {reference_kBT_J:.9e} J",
        f"Ms = {params.saturation_magnetization_Apm:.6g} A/m",
        (
            "Ms basis: present-particle magnetometry at 300 K, 55 emu/g "
            "(55 A m2/kg), converted with rho(Fe3O4)=5.18e3 kg/m3"
        ),
        f"Hamaker constant = {params.hamaker_J:.6e} J",
        (
            "Hamaker basis: Faure et al., Langmuir 27, 8659-8664 "
            "(2011), DOI 10.1021/la201387d; representative value "
            "within their 9-29 zJ nonpolar-solvent range"
        ),
        (
            "ZFC/FC effective barrier anisotropy K_eff = "
            f"{params.effective_barrier_anisotropy_Jpm3:.9e} J/m^3"
        ),
        (
            "V6.20 cubic anisotropy coefficient K_cubic = "
            f"{params.magnetocrystalline_anisotropy_Jpm3:.6g} J/m^3"
        ),
        f"Blocking temperature TB = {params.blocking_temperature_K:.2f} K",
        (
            "ZFC/FC observation time = "
            f"{params.zfc_fc_observation_time_s:.6g} s"
        ),
        f"Néel attempt time tau0 = {params.attempt_time_s:.6e} s",
        (
            "Attempt-time source: Moreno et al., Phys. Rev. B 112, "
            "024429 (2025), DOI 10.1103/vmwp-q427"
        ),
        (
            "ZFC/FC-calibrated activation barrier = "
            f"{cases['face_to_face']['activation_barrier_J'][0]:.9e} J"
        ),
        "Barrier convention: Delta E = K_eff V = K_cubic V / 12.",
        "No kinetic blocking factor or thermal-history variable is used.",
        "The correlated +[111] joint Boltzmann distribution uses the bath temperature.",
        "External magnetic field = 0 T",
        (
            "Orientation quadrature = "
            f"{params.orientation_gauss_count} x "
            f"{params.orientation_phi_count} directions per dipole"
        ),
        "",
    ]
    for key in ("face_to_face", "tip_to_tip"):
        case = cases[key]
        mean_dipole_J = case["mean_dipole_J"][reference_index]
        mean_anisotropy_pair_J = case["mean_anisotropy_pair_J"][
            reference_index
        ]
        mean_anisotropy_isolated_J = case[
            "mean_anisotropy_two_isolated_J"
        ][reference_index]
        anisotropy_response_J = case["anisotropy_response_J"][reference_index]
        magnetic_interaction_J = case["magnetic_interaction_energy_J"][
            reference_index
        ]
        uncorrelated_dipole_J = case[
            "uncorrelated_dipole_from_marginals_J"
        ][reference_index]
        dipole_correlation_correction_J = case[
            "dipole_correlation_correction_J"
        ][reference_index]
        magnetic_pair_free_energy_J = case[
            "magnetic_pair_free_energy_J"
        ][reference_index]
        anisotropy_excitation_J = case["mean_anisotropy_excitation_J"][
            reference_index
        ]
        magnetic_energy_J = case["mean_total_pair_magnetic_J"][
            reference_index
        ]
        lines.extend(
            [
                case["title"],
                "-" * len(case["title"]),
                f"Center distance = {case['center_distance_m'] * 1e9:.9f} nm",
                f"E_vdW = {case['vdw_J']:.9e} J "
                f"({case['vdw_J'] / reference_kBT_J:.6f} kBT at 298.15 K)",
                f"<E_dd>_pair = {mean_dipole_J:.9e} J "
                f"({mean_dipole_J / reference_kBT_J:.6f} kBT)",
                f"E_dd from product of joint marginals = "
                f"{uncorrelated_dipole_J:.9e} J",
                f"Dipole cross-correlation correction = "
                f"{dipole_correlation_correction_J:.9e} J",
                f"Pair magnetic free-energy change = "
                f"{magnetic_pair_free_energy_J:.9e} J",
                f"<E_ani,1 + E_ani,2>_pair = "
                f"{mean_anisotropy_pair_J:.9e} J",
                f"2 <E_ani>_isolated = {mean_anisotropy_isolated_J:.9e} J",
                f"Anisotropy response Delta E_ani = "
                f"{anisotropy_response_J:.9e} J",
                f"Easy-axis-referenced anisotropy excitation = "
                f"{anisotropy_excitation_J:.9e} J",
                f"Magnetic energy <Edd> + <Delta Eani> = "
                f"{magnetic_energy_J:.9e} J "
                f"({magnetic_energy_J / reference_kBT_J:.6f} kBT)",
                f"Interaction-induced <Edd> + Delta<Eani> = "
                f"{magnetic_interaction_J:.9e} J "
                f"({magnetic_interaction_J / reference_kBT_J:.6f} kBT)",
                f"|E_vdW| = kBT at T = {abs(case['vdw_J']) / Boltzmann:.3f} K",
                f"<m1 dot m2> = "
                f"{case['mean_moment_dot_product'][reference_index]:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "Statistical convention",
            "----------------------",
            "Both particles are conditionally confined to their +[111] basins.",
            "The joint Boltzmann weight contains Eani1 + Eani2 + Edd.",
            "There is no inter-basin relaxation or thermal-history variable.",
            "TB=250 K and t_obs=100 s are used only to infer K.",
            "Magnetic energy uses Delta Eani = Eani - Eani_min.",
            "Edd therefore changes both marginal distributions and their",
            "cross-correlation.",
            "No total-energy curve is plotted.",
        ]
    )
    (OUTPUT_DIR / "V620_two_NC_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def calculate_gap_sensitivity(
    temperatures_K: np.ndarray,
    reference_gap_m: float,
    reference_cases: dict[str, dict],
    params: V620Parameters = PARAMS,
) -> dict[float, dict[str, dict]]:
    """Evaluate several larger exact surface gaps without overwriting the baseline."""
    gap_values_nm = np.array([reference_gap_m * 1e9, 5.0, 6.0, 8.0, 10.0])
    sensitivity: dict[float, dict[str, dict]] = {
        float(gap_values_nm[0]): reference_cases
    }
    directions = {
        "face_to_face": ("Face-to-face", np.array([1.0, 0.0, 0.0]), 6),
        "tip_to_tip": ("Tip-to-tip", np.ones(3) / np.sqrt(3.0), 8),
    }
    for gap_nm in gap_values_nm[1:]:
        gap_cases: dict[str, dict] = {}
        for key, (title, direction, coordination_number) in directions.items():
            center_distance_m = center_distance_at_gap_m(
                direction, gap_nm * 1e-9, params
            )
            center_vector_m = center_distance_m * direction
            gap_cases[key] = {
                "title": title,
                "direction": direction,
                "center_distance_m": center_distance_m,
                "vdw_J": pair_vdw_energy_J(center_vector_m, params),
                "coordination_number": coordination_number,
            }
            gap_cases[key].update(
                thermal_magnetic_pair_curves(
                    temperatures_K, center_vector_m, params
                )
            )
        sensitivity[float(gap_nm)] = gap_cases
    return sensitivity


def export_gap_sensitivity_csv(
    temperatures_K: np.ndarray,
    sensitivity: dict[float, dict[str, dict]],
) -> None:
    """Export total-energy distance sensitivity and its decomposition."""
    thermal_J = Boltzmann * temperatures_K
    columns = [temperatures_K, thermal_J]
    names = ["temperature_K", "kBT_J"]
    for gap_nm, gap_cases in sensitivity.items():
        gap_tag = f"{gap_nm:.3f}".replace(".", "p")
        for key in ("face_to_face", "tip_to_tip"):
            case = gap_cases[key]
            vdw = np.full_like(temperatures_K, case["vdw_J"])
            magnetic = case["magnetic_interaction_energy_J"]
            total = vdw + magnetic
            columns.extend([vdw, magnetic, total, np.abs(total) / thermal_J])
            names.extend(
                [
                    f"gap_{gap_tag}_nm_{key}_vdw_J",
                    f"gap_{gap_tag}_nm_{key}_magnetic_interaction_J",
                    f"gap_{gap_tag}_nm_{key}_total_interaction_J",
                    f"gap_{gap_tag}_nm_{key}_abs_total_over_kBT",
                ]
            )
    np.savetxt(
        OUTPUT_DIR / "V620_gap_sensitivity_data.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def save_gap_sensitivity_figure(
    temperatures_K: np.ndarray,
    sensitivity: dict[float, dict[str, dict]],
    params: V620Parameters = PARAMS,
) -> None:
    """Plot total energy for the reference and three larger surface gaps."""
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    colors = mpl.colormaps["viridis"](
        np.linspace(0.08, 0.88, len(sensitivity))
    )
    thermal_J = Boltzmann * temperatures_K
    thermal_zJ = thermal_J / 1e-21

    for row, key in enumerate(("face_to_face", "tip_to_tip")):
        physical_ax, normalized_ax = axes[row]
        physical_ax.plot(
            temperatures_K, thermal_zJ, color="#252525", label=r"$k_BT$"
        )
        normalized_ax.axhline(
            1.0, color="#252525", label=r"$k_BT/k_BT=1$"
        )
        for color, (gap_nm, gap_cases) in zip(colors, sensitivity.items()):
            case = gap_cases[key]
            total_J = case["vdw_J"] + case["magnetic_interaction_energy_J"]
            label = rf"$D={gap_nm:.3g}$ nm"
            physical_ax.plot(
                temperatures_K,
                np.abs(total_J) / 1e-21,
                color=color,
                label=label,
            )
            normalized_ax.plot(
                temperatures_K,
                np.abs(total_J) / thermal_J,
                color=color,
                label=label,
            )
        for ax in (physical_ax, normalized_ax):
            ax.axvline(
                params.reference_temperature_K,
                color="#777777",
                linewidth=0.7,
                alpha=0.5,
            )
            ax.set_xlim(temperatures_K[0], temperatures_K[-1])
            ax.grid(which="both", alpha=0.2)
        normalized_ax.set_yscale("log")
        title = sensitivity[next(iter(sensitivity))][key]["title"]
        physical_ax.set(
            title=f"{title}: total interaction at larger gaps",
            xlabel="Temperature (K)",
            ylabel=r"Energy magnitude ($10^{-21}$ J)",
            ylim=(0.0, None),
        )
        normalized_ax.set(
            title=f"{title}: total interaction relative to $k_BT$",
            xlabel="Temperature (K)",
            ylabel=r"$|E_{\rm total}|/k_BT$",
        )
        physical_ax.legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none", fontsize=7,
        )
        normalized_ax.legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none", fontsize=7,
        )

    figure.suptitle(
        (
            "16 nm Fe$_3$O$_4$: effect of larger NC surface separation\n"
            r"$E_{\rm total}=E_{\rm vdW}+\Delta U_{\rm mag}$; "
            "correlated two-dipole $+[111]$ equilibrium, zero external field"
        ),
        fontsize=11,
    )
    figure.savefig(OUTPUT_DIR / "V620_gap_sensitivity.png")
    figure.savefig(OUTPUT_DIR / "V620_gap_sensitivity.pdf")
    plt.close(figure)


def save_fixed_gap_figure(
    temperatures_K: np.ndarray,
    gap_nm: float,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
) -> None:
    """Save the full component comparison for one user-selected surface gap."""
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for row, key in enumerate(("face_to_face", "tip_to_tip")):
        add_physical_panel(
            axes[row, 0],
            temperatures_K,
            cases[key],
            params,
            temperature_unit="K",
            xlim=(100.0, 850.0),
            show_band_labels=True,
            show_reference_line=False,
        )
        add_normalized_panel(
            axes[row, 1],
            temperatures_K,
            cases[key],
            params,
            temperature_unit="K",
            xlim=(100.0, 850.0),
            show_band_labels=True,
            show_reference_line=False,
        )
        axes[row, 0].legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none", fontsize=7,
        )
        axes[row, 1].legend(
            loc="upper right", frameon=True, facecolor="white",
            framealpha=0.92, edgecolor="none", fontsize=7,
        )
    figure.suptitle(
        (
            f"16 nm Fe$_3$O$_4$, exact surface gap $D={gap_nm:g}$ nm: "
            "correlated two-dipole $+[111]$ equilibrium\n"
            r"$\Delta F_{\rm mag}^{\rm pair}="
            r"-k_BT\ln[Z_{\rm pair}/Z_{\rm isolated}^2]$"
        ),
        fontsize=11,
    )
    figure.savefig(OUTPUT_DIR / f"V620_D{gap_nm:g}nm_temperature.png")
    figure.savefig(OUTPUT_DIR / f"V620_D{gap_nm:g}nm_temperature.pdf")
    plt.close(figure)


def morphology_competition_curves(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Return positive energy scales for the competing morphology tendencies.

    The magnetic scale is the free-energy advantage of tip-to-tip over
    face-to-face.  The vdW scales are the corresponding face-to-face
    advantages for one pair and for a coordination-number estimate per NC.
    These are preference differences, not absolute-valued component energies.
    """
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    face_vdw_multi_J = (
        0.5 * face["coordination_number"] * face["vdw_J"]
    )
    tip_vdw_multi_J = (
        0.5 * tip["coordination_number"] * tip["vdw_J"]
    )
    return {
        "thermal_disorder_J": Boltzmann * temperatures_K,
        "magnetic_tip_preference_J": (
            face["magnetic_pair_free_energy_J"]
            - tip["magnetic_pair_free_energy_J"]
        ),
        "vdw_face_preference_pair_J": np.full_like(
            temperatures_K, tip["vdw_J"] - face["vdw_J"]
        ),
        "vdw_face_preference_multi_per_nc_J": np.full_like(
            temperatures_K, tip_vdw_multi_J - face_vdw_multi_J
        ),
    }


def export_morphology_competition_csv(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
) -> None:
    """Export the three morphology-selection energy scales at D=5 nm."""
    curves = morphology_competition_curves(temperatures_K, cases)
    names = ["temperature_K", *curves.keys()]
    table = np.column_stack([temperatures_K, *curves.values()])
    np.savetxt(
        OUTPUT_DIR / "V620_D5nm_morphology_competition.csv",
        table,
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def save_morphology_competition_figure(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
) -> None:
    """Plot the calculated tip, face, and thermal-disorder tendencies."""
    curves = morphology_competition_curves(temperatures_K, cases)
    thermal_zJ = curves["thermal_disorder_J"] / 1e-21
    magnetic_zJ = curves["magnetic_tip_preference_J"] / 1e-21
    vdw_pair_zJ = curves["vdw_face_preference_pair_J"] / 1e-21
    vdw_multi_zJ = (
        curves["vdw_face_preference_multi_per_nc_J"] / 1e-21
    )
    dominant_index = np.argmax(
        np.vstack([magnetic_zJ, vdw_multi_zJ, thermal_zJ]), axis=0
    )

    figure, ax = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    add_experimental_temperature_bands(ax, "K", show_labels=True)
    ax.plot(
        temperatures_K,
        magnetic_zJ,
        color="#c53030",
        linewidth=2.2,
        label=(
            r"magnetic tip preference "
            r"$(\Delta F_{\rm mag}^{face}-\Delta F_{\rm mag}^{tip})$"
        ),
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        vdw_multi_zJ,
        color="#2b6cb0",
        linestyle="--",
        linewidth=2.2,
        label=(
            r"vdW face preference per NC "
            r"$(E_{\rm vdW}^{tip}-E_{\rm vdW}^{face})$"
        ),
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        vdw_pair_zJ,
        color="#6baed6",
        linestyle=":",
        linewidth=1.7,
        label="vdW face preference, isolated pair",
        zorder=3,
    )
    ax.plot(
        temperatures_K,
        thermal_zJ,
        color="#252525",
        linewidth=2.0,
        label=r"thermal disorder $k_BT$",
        zorder=4,
    )
    ax.axvline(
        params.blocking_temperature_K,
        color="#6f42c1",
        linestyle=(0, (4, 3)),
        linewidth=1.3,
        label=rf"$T_B={params.blocking_temperature_K:g}$ K",
        zorder=3,
    )

    # A narrow strip reports the dominant scale actually produced by the
    # calculation.  It intentionally does not manufacture a vdW-dominant
    # interval when the computed curves do not contain one.
    y_max = max(
        float(np.max(thermal_zJ)),
        float(np.max(magnetic_zJ)),
        float(np.max(vdw_multi_zJ)),
    )
    strip_low = 1.34 * y_max
    strip_high = 1.43 * y_max
    regime_colors = np.array(["#f4cccc", "#cfe2f3", "#e7e7e7"])
    for regime_index in range(3):
        ax.fill_between(
            temperatures_K,
            strip_low,
            strip_high,
            where=dominant_index == regime_index,
            step="mid",
            color=regime_colors[regime_index],
            linewidth=0.0,
            zorder=2,
        )
    ax.text(
        temperatures_K[0] + 8.0,
        0.5 * (strip_low + strip_high),
        "calculated dominant scale",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#444444",
        zorder=5,
    )
    ax.text(
        0.5 * (temperatures_K[0] + temperatures_K[-1]),
        -0.16,
        (
            r"$T_B$ calibrates $K$ through the 100 s N"
            r"$\acute{\rm e}$el barrier; it is not an equilibrium transition."
        ),
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.set(
        title=(
            "16 nm Fe$_3$O$_4$, $D=5$ nm: calculated morphology-selection "
            "energy scales"
        ),
        xlabel="Temperature (K)",
        ylabel=r"Preference / disorder scale ($10^{-21}$ J)",
        xlim=(100.0, 850.0),
        ylim=(0.0, 1.48 * y_max),
    )
    ax.grid(alpha=0.2)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=7.7,
    )
    figure.savefig(OUTPUT_DIR / "V620_D5nm_morphology_competition.png")
    figure.savefig(OUTPUT_DIR / "V620_D5nm_morphology_competition.pdf")
    plt.close(figure)


def export_dynamic_five_nm_csv(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    filename: str = "V620_D5nm_temperature_data.csv",
) -> None:
    """Export the dynamic Néel result with only face-to-face pair vdW."""
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    columns = [
        temperatures_K,
        Boltzmann * temperatures_K,
        np.full_like(temperatures_K, face["vdw_pair_J"]),
        face["mean_dipole_J"],
        face["mean_anisotropy_excitation_J"],
        face["mean_magnetic_J"],
        face["initial_pair_probability"],
        tip["mean_dipole_J"],
        tip["mean_anisotropy_excitation_J"],
        tip["mean_magnetic_J"],
        tip["initial_pair_probability"],
    ]
    names = [
        "temperature_K",
        "kBT_J",
        "face_to_face_vdw_pair_J",
        "face_to_face_mean_dipole_J",
        "face_to_face_mean_anisotropy_excitation_J",
        "face_to_face_magnetic_energy_J",
        "face_to_face_initial_pair_probability_after_observation",
        "tip_to_tip_mean_dipole_J",
        "tip_to_tip_mean_anisotropy_excitation_J",
        "tip_to_tip_magnetic_energy_J",
        "tip_to_tip_initial_pair_probability_after_observation",
    ]
    np.savetxt(
        OUTPUT_DIR / filename,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def export_all_dynamic_windows_csv(
    temperatures_K: np.ndarray,
    window_cases: dict[float, dict[str, dict]],
) -> None:
    """Export all observation windows in one wide table."""
    first_cases = window_cases[next(iter(window_cases))]
    columns = [
        temperatures_K,
        Boltzmann * temperatures_K,
        np.full_like(
            temperatures_K,
            first_cases["face_to_face"]["vdw_pair_J"],
        ),
    ]
    names = [
        "temperature_K",
        "kBT_J",
        "face_to_face_vdw_pair_J",
    ]
    for observation_time_s, cases in window_cases.items():
        tag = f"{observation_time_s:g}s".replace(".", "p")
        columns.extend(
            [
                cases["face_to_face"]["mean_magnetic_J"],
                cases["tip_to_tip"]["mean_magnetic_J"],
                cases["face_to_face"]["initial_pair_probability"],
                cases["tip_to_tip"]["initial_pair_probability"],
            ]
        )
        names.extend(
            [
                f"face_to_face_magnetic_energy_{tag}_J",
                f"tip_to_tip_magnetic_energy_{tag}_J",
                f"face_to_face_initial_pair_probability_{tag}",
                f"tip_to_tip_initial_pair_probability_{tag}",
            ]
        )
    np.savetxt(
        OUTPUT_DIR / "V620_D5nm_all_experimental_windows_data.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def export_dynamic_five_nm_normalized_csv(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    filename: str,
) -> None:
    """Export the D=5 nm pair energies normalized by kBT."""
    thermal_J = Boltzmann * temperatures_K
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    columns = [
        temperatures_K,
        face["mean_magnetic_J"] / thermal_J,
        tip["mean_magnetic_J"] / thermal_J,
        np.full_like(
            temperatures_K, face["vdw_pair_J"]
        )
        / thermal_J,
        np.ones_like(temperatures_K),
    ]
    names = [
        "temperature_K",
        "face_to_face_magnetic_energy_over_kBT",
        "tip_to_tip_magnetic_energy_over_kBT",
        "face_to_face_vdw_pair_energy_over_kBT",
        "kBT_over_kBT",
    ]
    np.savetxt(
        OUTPUT_DIR / filename,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def export_all_dynamic_windows_normalized_csv(
    temperatures_K: np.ndarray,
    window_cases: dict[float, dict[str, dict]],
) -> None:
    """Export normalized energies for all observation windows."""
    thermal_J = Boltzmann * temperatures_K
    first_cases = window_cases[next(iter(window_cases))]
    columns = [
        temperatures_K,
        np.full_like(
            temperatures_K,
            first_cases["face_to_face"]["vdw_pair_J"],
        )
        / thermal_J,
        np.ones_like(temperatures_K),
    ]
    names = [
        "temperature_K",
        "face_to_face_vdw_pair_energy_over_kBT",
        "kBT_over_kBT",
    ]
    for observation_time_s, cases in window_cases.items():
        tag = f"{observation_time_s:g}s".replace(".", "p")
        columns.extend(
            [
                cases["face_to_face"]["mean_magnetic_J"]
                / thermal_J,
                cases["tip_to_tip"]["mean_magnetic_J"]
                / thermal_J,
            ]
        )
        names.extend(
            [
                (
                    "face_to_face_magnetic_energy_over_kBT_"
                    f"{tag}"
                ),
                (
                    "tip_to_tip_magnetic_energy_over_kBT_"
                    f"{tag}"
                ),
            ]
        )
    np.savetxt(
        OUTPUT_DIR
        / "V620_D5nm_all_experimental_windows_E_over_kBT_data.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def coordination_factor_per_nc(
    geometry: str,
    case: dict,
) -> float:
    """Return n/2 so each pair bond is shared by its two NCs."""
    coordination_number = int(
        case.get(
            "coordination_number",
            COORDINATION_NUMBERS[geometry],
        )
    )
    return 0.5 * coordination_number


def export_coordination_dynamic_csv(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    filename: str,
    normalized: bool = False,
) -> None:
    """Export coordination-number energy estimates per NC."""
    thermal_J = Boltzmann * temperatures_K
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    face_factor = coordination_factor_per_nc(
        "face_to_face", face
    )
    tip_factor = coordination_factor_per_nc("tip_to_tip", tip)
    face_magnetic = face_factor * face["mean_magnetic_J"]
    tip_magnetic = tip_factor * tip["mean_magnetic_J"]
    face_vdw = np.full_like(
        temperatures_K,
        face_factor * face["vdw_pair_J"],
    )
    if normalized:
        columns = [
            temperatures_K,
            face_magnetic / thermal_J,
            tip_magnetic / thermal_J,
            face_vdw / thermal_J,
            np.ones_like(temperatures_K),
        ]
        names = [
            "temperature_K",
            "face_n6_magnetic_energy_per_NC_over_kBT",
            "tip_n8_magnetic_energy_per_NC_over_kBT",
            "face_n6_vdw_energy_per_NC_over_kBT",
            "kBT_over_kBT",
        ]
    else:
        columns = [
            temperatures_K,
            thermal_J,
            face_magnetic,
            tip_magnetic,
            face_vdw,
        ]
        names = [
            "temperature_K",
            "kBT_J",
            "face_n6_magnetic_energy_per_NC_J",
            "tip_n8_magnetic_energy_per_NC_J",
            "face_n6_vdw_energy_per_NC_J",
        ]
    np.savetxt(
        OUTPUT_DIR / filename,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def export_all_coordination_windows_csv(
    temperatures_K: np.ndarray,
    window_cases: dict[float, dict[str, dict]],
    normalized: bool = False,
) -> None:
    """Export coordination-number estimates for all windows."""
    thermal_J = Boltzmann * temperatures_K
    first_cases = window_cases[next(iter(window_cases))]
    face = first_cases["face_to_face"]
    face_factor = coordination_factor_per_nc(
        "face_to_face", face
    )
    face_vdw = np.full_like(
        temperatures_K,
        face_factor * face["vdw_pair_J"],
    )
    if normalized:
        columns = [
            temperatures_K,
            face_vdw / thermal_J,
            np.ones_like(temperatures_K),
        ]
        names = [
            "temperature_K",
            "face_n6_vdw_energy_per_NC_over_kBT",
            "kBT_over_kBT",
        ]
    else:
        columns = [temperatures_K, thermal_J, face_vdw]
        names = [
            "temperature_K",
            "kBT_J",
            "face_n6_vdw_energy_per_NC_J",
        ]
    for observation_time_s, cases in window_cases.items():
        tag = f"{observation_time_s:g}s".replace(".", "p")
        face = cases["face_to_face"]
        tip = cases["tip_to_tip"]
        face_magnetic = coordination_factor_per_nc(
            "face_to_face", face
        ) * face["mean_magnetic_J"]
        tip_magnetic = coordination_factor_per_nc(
            "tip_to_tip", tip
        ) * tip["mean_magnetic_J"]
        if normalized:
            columns.extend(
                [
                    face_magnetic / thermal_J,
                    tip_magnetic / thermal_J,
                ]
            )
            names.extend(
                [
                    (
                        "face_n6_magnetic_energy_per_NC_over_kBT_"
                        f"{tag}"
                    ),
                    (
                        "tip_n8_magnetic_energy_per_NC_over_kBT_"
                        f"{tag}"
                    ),
                ]
            )
        else:
            columns.extend([face_magnetic, tip_magnetic])
            names.extend(
                [
                    f"face_n6_magnetic_energy_per_NC_{tag}_J",
                    f"tip_n8_magnetic_energy_per_NC_{tag}_J",
                ]
            )
    suffix = "_E_over_kBT" if normalized else ""
    np.savetxt(
        OUTPUT_DIR
        / (
            "V620_D5nm_all_experimental_windows_coordination"
            f"{suffix}_data.csv"
        ),
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def write_dynamic_five_nm_summary(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
    filename: str = "V620_D5nm_dynamic_summary.txt",
) -> None:
    """Write the assumptions and selected values for the dynamic D=5 nm run."""
    observation_time_s = float(
        cases["face_to_face"]["observation_time_s"]
    )
    lines = [
        "V6.20 D=5 nm coupled-Neel calculation",
        "======================================",
        "State space: 8 <111> wells per NC, 64 pair states",
        "Initial state: both moments in +[111]",
        (
            "Dynamic experimental observation time: "
            f"{observation_time_s:g} s"
        ),
        (
            "ZFC/FC time used to calibrate K: "
            f"{params.zfc_fc_observation_time_s:g} s"
        ),
        f"Attempt time: {params.attempt_time_s:.6e} s",
        (
            "Attempt-time source: Moreno et al., Phys. Rev. B 112, "
            "024429 (2025), DOI 10.1103/vmwp-q427"
        ),
        f"Nominal blocking temperature: {params.blocking_temperature_K:g} K",
        (
            "Nominal adjacent-well barrier: "
            f"{params.zfc_fc_activation_barrier_J:.9e} J"
        ),
        (
            "Barrier relation: Delta E = K_cubic V / 12; "
            f"K_cubic = {params.magnetocrystalline_anisotropy_Jpm3:.9e} J/m^3"
        ),
        (
            "Lognormal diameter coefficient of variation: "
            f"{100.0 * params.diameter_coefficient_of_variation:.3f}%"
        ),
        (
            "Barrier quadrature count: "
            f"{params.barrier_distribution_count:d}"
        ),
        "Dipole interaction is included in each transition-state barrier.",
        (
            "Anisotropy is referenced to the easy-axis minimum; occupied "
            "discrete wells have Delta Eani = 0."
        ),
        "Only face-to-face pair vdW is exported and plotted.",
        "",
        "T_K,face_Emag_zJ,tip_Emag_zJ,face_initial_probability,tip_initial_probability",
    ]
    for temperature_K in (200.0, 225.0, 250.0, 275.0, 300.0):
        index = int(np.argmin(np.abs(temperatures_K - temperature_K)))
        lines.append(
            f"{temperatures_K[index]:.2f},"
            f"{cases['face_to_face']['mean_magnetic_J'][index] / 1e-21:.6f},"
            f"{cases['tip_to_tip']['mean_magnetic_J'][index] / 1e-21:.6f},"
            f"{cases['face_to_face']['initial_pair_probability'][index]:.6f},"
            f"{cases['tip_to_tip']['initial_pair_probability'][index]:.6f}"
        )
    (OUTPUT_DIR / filename).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_signed_geometry_selection_figure(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
    filename_stem: str = "V620_D5nm_signed_geometry_selection",
) -> None:
    """Plot coupled-Néel magnetic energies from 200 to 300 K."""
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    observation_time_s = float(face["observation_time_s"])
    thermal_J = Boltzmann * temperatures_K
    face_magnetic_J = face["mean_magnetic_J"]
    tip_magnetic_J = tip["mean_magnetic_J"]
    face_vdw_pair_J = face["vdw_pair_J"]

    figure, ax = plt.subplots(figsize=(10.2, 10.2))
    figure.subplots_adjust(
        left=0.105, right=0.64, bottom=0.18, top=0.87
    )
    add_experimental_temperature_bands(ax, "K", show_labels=False)
    ax.plot(
        temperatures_K,
        face_magnetic_J,
        color="#c53030",
        linewidth=2.4,
        label=r"$E_{\rm magnetic}^{face}$",
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        tip_magnetic_J,
        color="#dd6b20",
        linestyle="-",
        linewidth=2.4,
        label=r"$E_{\rm magnetic}^{tip}$",
        zorder=4,
    )
    ax.axhline(
        face_vdw_pair_J,
        color="#2b6cb0",
        linestyle="-",
        linewidth=2.4,
        label=r"$E_{\rm vdW}^{face,\ pair}$",
        zorder=3,
    )
    ax.plot(
        temperatures_K,
        thermal_J,
        color="#252525",
        linewidth=2.0,
        label=r"$k_BT$",
        zorder=3,
    )
    ax.axvline(
        params.blocking_temperature_K,
        color="#666666",
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        zorder=2,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)

    minimum_energy_J = min(
        float(np.min(face_magnetic_J)),
        float(np.min(tip_magnetic_J)),
        face_vdw_pair_J,
    )
    ax.set(
        xlabel="Temperature (K)",
        ylabel="Energy (J)",
        xlim=(200.0, 300.0),
        ylim=(1.08 * minimum_energy_J, 6.2e-21),
    )
    ax.set_title(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\n"
            f"{observation_time_s:g} s experimental window"
        ),
        fontsize=17,
        fontweight="normal",
        pad=12,
    )
    ax.set_xlabel(
        "Temperature (K)", fontsize=16, fontweight="normal"
    )
    ax.set_ylabel("Energy (J)", fontsize=16, fontweight="normal")
    ax.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    ax.tick_params(axis="both", labelsize=14)
    plt.setp(
        ax.get_xticklabels(),
        fontweight="normal",
    )
    plt.setp(
        ax.get_yticklabels(),
        fontweight="normal",
    )
    ax.grid(alpha=0.2)
    ax.text(
        params.blocking_temperature_K + 1.0,
        0.965,
        rf"$T_B={params.blocking_temperature_K:g}$ K",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="normal",
        color="#555555",
        zorder=6,
    )
    energy_legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=12,
    )
    for legend_text in energy_legend.get_texts():
        legend_text.set_fontweight("normal")
    ax.add_artist(energy_legend)
    range_legend = ax.legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=12,
    )
    for legend_text in range_legend.get_texts():
        legend_text.set_fontweight("normal")
    figure.text(
        0.5,
        0.035,
        (
            r"$E_{\rm magnetic}=\langle E_{dd}+\Delta E_{\rm ani}\rangle$; "
            r"$\Delta E_{\rm ani}=0$ at occupied easy-axis minima."
            "\n"
            r"$K_{\rm cubic}$ enters the transition barriers; "
            rf"$t_{{obs}}={observation_time_s:g}$ s; "
            rf"diameter CV = {100.0 * params.diameter_coefficient_of_variation:.0f}%."
            "\n"
            rf"$K$ calibrated from $T_B$ at {params.zfc_fc_observation_time_s:g} s."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
        color="#555555",
    )
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.png")
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.pdf")
    plt.close(figure)


def save_all_experimental_windows_figure(
    temperatures_K: np.ndarray,
    window_cases: dict[float, dict[str, dict]],
    params: V620Parameters = PARAMS,
) -> None:
    """Save a shared-scale comparison of the 1, 10, and 100 s windows."""
    minimum_energy_J = min(
        min(
            float(np.min(cases[geometry]["mean_magnetic_J"]))
            for geometry in ("face_to_face", "tip_to_tip")
        )
        for cases in window_cases.values()
    )
    figure, axes = plt.subplots(
        len(window_cases),
        1,
        figsize=(8.6, 15.0),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.12,
        top=0.90,
        hspace=0.15,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for panel_index, (
        observation_time_s,
        cases,
    ) in enumerate(window_cases.items()):
        ax = axes[panel_index]
        face = cases["face_to_face"]
        tip = cases["tip_to_tip"]
        add_experimental_temperature_bands(
            ax, "K", show_labels=False
        )
        ax.plot(
            temperatures_K,
            face["mean_magnetic_J"],
            color="#c53030",
            linewidth=2.5,
            label=r"$E_{\rm magnetic}^{face}$",
            zorder=4,
        )
        ax.plot(
            temperatures_K,
            tip["mean_magnetic_J"],
            color="#dd6b20",
            linewidth=2.5,
            label=r"$E_{\rm magnetic}^{tip}$",
            zorder=4,
        )
        ax.axhline(
            face["vdw_pair_J"],
            color="#2b6cb0",
            linewidth=2.5,
            label=r"$E_{\rm vdW}^{face,\ pair}$",
            zorder=3,
        )
        ax.plot(
            temperatures_K,
            Boltzmann * temperatures_K,
            color="#252525",
            linewidth=2.3,
            label=r"$k_BT$",
            zorder=3,
        )
        ax.axvline(
            params.blocking_temperature_K,
            color="#666666",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            zorder=2,
        )
        ax.axhline(
            0.0, color="#777777", linewidth=0.8, alpha=0.7
        )
        ax.text(
            params.blocking_temperature_K + 1.0,
            0.965,
            rf"$T_B={params.blocking_temperature_K:g}$ K",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=12,
            color="#555555",
            zorder=6,
        )
        ax.set(
            title=f"{observation_time_s:g} s",
            xlim=(200.0, 300.0),
            ylim=(1.08 * minimum_energy_J, 6.2e-21),
        )
        ax.title.set_fontsize(15)
        ax.set_xticks(np.arange(200.0, 301.0, 10.0))
        ax.tick_params(axis="both", labelsize=12.5)
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
        ax.grid(alpha=0.2)

    energy_legend = axes[0].legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=11.5,
    )
    axes[0].add_artist(energy_legend)
    axes[0].legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=11.5,
    )
    figure.suptitle(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\n1 s, 10 s, and 100 s experimental windows"
        ),
        fontsize=18,
    )
    figure.supxlabel("Temperature (K)", fontsize=15, y=0.075)
    figure.supylabel("Energy (J)", fontsize=15, x=0.035)
    figure.text(
        0.5,
        0.014,
        (
            r"$E_{\rm magnetic}=\langle E_{dd}+\Delta E_{\rm ani}\rangle$; "
            r"$\Delta E_{\rm ani}=0$ at occupied easy-axis minima."
            "\n"
            r"$K_{\rm cubic}$ enters the transition barriers; "
            rf"diameter CV = {100.0 * params.diameter_coefficient_of_variation:.0f}%."
            "\n"
            rf"$K$ calibrated from $T_B$ at "
            rf"{params.zfc_fc_observation_time_s:g} s."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    figure.savefig(
        OUTPUT_DIR / "V620_D5nm_all_experimental_windows.png"
    )
    figure.savefig(
        OUTPUT_DIR / "V620_D5nm_all_experimental_windows.pdf"
    )
    plt.close(figure)


def save_normalized_geometry_selection_figure(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
    filename_stem: str = "V620_D5nm_E_over_kBT",
) -> None:
    """Plot the D=5 nm pair energies divided by kBT."""
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    observation_time_s = float(face["observation_time_s"])
    thermal_J = Boltzmann * temperatures_K
    face_magnetic_normalized = (
        face["mean_magnetic_J"] / thermal_J
    )
    tip_magnetic_normalized = (
        tip["mean_magnetic_J"] / thermal_J
    )
    face_vdw_pair_normalized = (
        np.full_like(temperatures_K, face["vdw_pair_J"])
        / thermal_J
    )

    figure, ax = plt.subplots(figsize=(10.2, 10.2))
    figure.subplots_adjust(
        left=0.105, right=0.64, bottom=0.18, top=0.87
    )
    add_experimental_temperature_bands(
        ax, "K", show_labels=False
    )
    ax.plot(
        temperatures_K,
        face_magnetic_normalized,
        color="#c53030",
        linewidth=2.4,
        label=r"$E_{\rm magnetic}^{face}/k_BT$",
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        tip_magnetic_normalized,
        color="#dd6b20",
        linewidth=2.4,
        label=r"$E_{\rm magnetic}^{tip}/k_BT$",
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        face_vdw_pair_normalized,
        color="#2b6cb0",
        linewidth=2.4,
        label=r"$E_{\rm vdW}^{face,\ pair}/k_BT$",
        zorder=3,
    )
    ax.axhline(
        1.0,
        color="#252525",
        linewidth=2.0,
        label=r"$k_BT/k_BT=1$",
        zorder=3,
    )
    ax.axvline(
        params.blocking_temperature_K,
        color="#666666",
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        zorder=2,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    minimum_normalized_energy = min(
        float(np.min(face_magnetic_normalized)),
        float(np.min(tip_magnetic_normalized)),
        float(np.min(face_vdw_pair_normalized)),
    )
    ax.set(
        xlim=(200.0, 300.0),
        ylim=(1.08 * minimum_normalized_energy, 1.35),
    )
    ax.set_title(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\n"
            f"{observation_time_s:g} s experimental window"
        ),
        fontsize=17,
        fontweight="normal",
        pad=12,
    )
    ax.set_xlabel(
        "Temperature (K)", fontsize=16, fontweight="normal"
    )
    ax.set_ylabel(
        r"$E/k_BT$", fontsize=16, fontweight="normal"
    )
    ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.2)
    ax.text(
        params.blocking_temperature_K + 1.0,
        0.88,
        rf"$T_B={params.blocking_temperature_K:g}$ K",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="normal",
        color="#555555",
        zorder=6,
    )
    energy_legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=12,
    )
    ax.add_artist(energy_legend)
    ax.legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.035,
        (
            r"$E_{\rm magnetic}=\langle E_{dd}+\Delta E_{\rm ani}\rangle$; "
            r"$\Delta E_{\rm ani}=0$ at occupied easy-axis minima."
            "\n"
            r"$K_{\rm cubic}$ enters the transition barriers; "
            rf"$t_{{obs}}={observation_time_s:g}$ s; "
            rf"diameter CV = {100.0 * params.diameter_coefficient_of_variation:.0f}%."
            "\n"
            rf"$K$ calibrated from $T_B$ at "
            rf"{params.zfc_fc_observation_time_s:g} s."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
        color="#555555",
    )
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.png")
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.pdf")
    plt.close(figure)


def save_all_normalized_experimental_windows_figure(
    temperatures_K: np.ndarray,
    window_cases: dict[float, dict[str, dict]],
    params: V620Parameters = PARAMS,
) -> None:
    """Save a shared-scale E/kBT comparison of all windows."""
    thermal_J = Boltzmann * temperatures_K
    minimum_normalized_energy = min(
        min(
            float(
                np.min(
                    cases[geometry]["mean_magnetic_J"]
                    / thermal_J
                )
            )
            for geometry in ("face_to_face", "tip_to_tip")
        )
        for cases in window_cases.values()
    )
    figure, axes = plt.subplots(
        len(window_cases),
        1,
        figsize=(8.6, 15.0),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.12,
        top=0.90,
        hspace=0.15,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for panel_index, (
        observation_time_s,
        cases,
    ) in enumerate(window_cases.items()):
        ax = axes[panel_index]
        face = cases["face_to_face"]
        tip = cases["tip_to_tip"]
        add_experimental_temperature_bands(
            ax, "K", show_labels=False
        )
        ax.plot(
            temperatures_K,
            face["mean_magnetic_J"] / thermal_J,
            color="#c53030",
            linewidth=2.5,
            label=r"$E_{\rm magnetic}^{face}/k_BT$",
            zorder=4,
        )
        ax.plot(
            temperatures_K,
            tip["mean_magnetic_J"] / thermal_J,
            color="#dd6b20",
            linewidth=2.5,
            label=r"$E_{\rm magnetic}^{tip}/k_BT$",
            zorder=4,
        )
        ax.plot(
            temperatures_K,
            np.full_like(
                temperatures_K, face["vdw_pair_J"]
            )
            / thermal_J,
            color="#2b6cb0",
            linewidth=2.5,
            label=r"$E_{\rm vdW}^{face,\ pair}/k_BT$",
            zorder=3,
        )
        ax.axhline(
            1.0,
            color="#252525",
            linewidth=2.3,
            label=r"$k_BT/k_BT=1$",
            zorder=3,
        )
        ax.axvline(
            params.blocking_temperature_K,
            color="#666666",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            zorder=2,
        )
        ax.axhline(
            0.0, color="#777777", linewidth=0.8, alpha=0.7
        )
        ax.text(
            params.blocking_temperature_K + 1.0,
            0.88,
            rf"$T_B={params.blocking_temperature_K:g}$ K",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=12,
            color="#555555",
            zorder=6,
        )
        ax.set(
            title=f"{observation_time_s:g} s",
            xlim=(200.0, 300.0),
            ylim=(1.08 * minimum_normalized_energy, 1.35),
        )
        ax.title.set_fontsize(15)
        ax.set_xticks(np.arange(200.0, 301.0, 10.0))
        ax.tick_params(axis="both", labelsize=12.5)
        ax.grid(alpha=0.2)

    energy_legend = axes[0].legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=11.5,
    )
    axes[0].add_artist(energy_legend)
    axes[0].legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=11.5,
    )
    figure.suptitle(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\n1 s, 10 s, and 100 s experimental windows"
        ),
        fontsize=18,
    )
    figure.supxlabel("Temperature (K)", fontsize=15, y=0.075)
    figure.supylabel(r"$E/k_BT$", fontsize=15, x=0.035)
    figure.text(
        0.5,
        0.014,
        (
            r"$E_{\rm magnetic}=\langle E_{dd}+\Delta E_{\rm ani}\rangle$; "
            r"$\Delta E_{\rm ani}=0$ at occupied easy-axis minima."
            "\n"
            r"$K_{\rm cubic}$ enters the transition barriers; "
            rf"diameter CV = {100.0 * params.diameter_coefficient_of_variation:.0f}%."
            "\n"
            rf"$K$ calibrated from $T_B$ at "
            rf"{params.zfc_fc_observation_time_s:g} s."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    figure.savefig(
        OUTPUT_DIR
        / "V620_D5nm_all_experimental_windows_E_over_kBT.png"
    )
    figure.savefig(
        OUTPUT_DIR
        / "V620_D5nm_all_experimental_windows_E_over_kBT.pdf"
    )
    plt.close(figure)


def coordination_plot_curves(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return face magnetic, tip magnetic, face vdW, and thermal curves."""
    thermal_J = Boltzmann * temperatures_K
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    face_magnetic_J = coordination_factor_per_nc(
        "face_to_face", face
    ) * face["mean_magnetic_J"]
    tip_magnetic_J = coordination_factor_per_nc(
        "tip_to_tip", tip
    ) * tip["mean_magnetic_J"]
    face_vdw_J = np.full_like(
        temperatures_K,
        coordination_factor_per_nc("face_to_face", face)
        * face["vdw_pair_J"],
    )
    if normalized:
        return (
            face_magnetic_J / thermal_J,
            tip_magnetic_J / thermal_J,
            face_vdw_J / thermal_J,
            np.ones_like(temperatures_K),
        )
    return (
        face_magnetic_J,
        tip_magnetic_J,
        face_vdw_J,
        thermal_J,
    )


def save_coordination_geometry_selection_figure(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    params: V620Parameters = PARAMS,
    normalized: bool = False,
    filename_stem: str = "V620_D5nm_coordination",
) -> None:
    """Plot coordination-number energy estimates per NC."""
    face = cases["face_to_face"]
    observation_time_s = float(face["observation_time_s"])
    (
        face_magnetic,
        tip_magnetic,
        face_vdw,
        thermal,
    ) = coordination_plot_curves(
        temperatures_K, cases, normalized
    )

    figure, ax = plt.subplots(figsize=(10.2, 10.2))
    figure.subplots_adjust(
        left=0.105, right=0.64, bottom=0.19, top=0.86
    )
    add_experimental_temperature_bands(
        ax, "K", show_labels=False
    )
    energy_suffix = (
        r"/(NC\,k_BT)" if normalized else r"/NC"
    )
    ax.plot(
        temperatures_K,
        face_magnetic,
        color="#c53030",
        linewidth=2.4,
        label=(
            rf"$E_{{\rm magnetic}}^{{face}}{energy_suffix}$ "
            r"$(n=6)$"
        ),
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        tip_magnetic,
        color="#dd6b20",
        linewidth=2.4,
        label=(
            rf"$E_{{\rm magnetic}}^{{tip}}{energy_suffix}$ "
            r"$(n=8)$"
        ),
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        face_vdw,
        color="#2b6cb0",
        linewidth=2.4,
        label=(
            rf"$E_{{\rm vdW}}^{{face}}{energy_suffix}$ "
            r"$(n=6)$"
        ),
        zorder=3,
    )
    ax.plot(
        temperatures_K,
        thermal,
        color="#252525",
        linewidth=2.0,
        label=(
            r"$k_BT/k_BT=1$"
            if normalized
            else r"$k_BT$"
        ),
        zorder=3,
    )
    ax.axvline(
        params.blocking_temperature_K,
        color="#666666",
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        zorder=2,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    minimum_energy = min(
        float(np.min(face_magnetic)),
        float(np.min(tip_magnetic)),
        float(np.min(face_vdw)),
    )
    ax.set(
        xlim=(200.0, 300.0),
        ylim=(
            1.08 * minimum_energy,
            1.35 if normalized else 6.2e-21,
        ),
    )
    ax.set_title(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\n"
            f"{observation_time_s:g} s experimental window; "
            "coordination per NC"
        ),
        fontsize=17,
        fontweight="normal",
        pad=12,
    )
    ax.set_xlabel(
        "Temperature (K)", fontsize=16, fontweight="normal"
    )
    ax.set_ylabel(
        r"$E/k_BT$" if normalized else "Energy (J)",
        fontsize=16,
        fontweight="normal",
    )
    if not normalized:
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
    ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.2)
    ax.text(
        params.blocking_temperature_K + 1.0,
        0.88,
        rf"$T_B={params.blocking_temperature_K:g}$ K",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="normal",
        color="#555555",
        zorder=6,
    )
    energy_legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=11.5,
    )
    ax.add_artist(energy_legend)
    ax.legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.033,
        (
            r"$E_{\rm coord}/NC=(n/2)E_{\rm pair}$; "
            r"face $n=6$, tip $n=8$."
            "\n"
            r"$E_{\rm magnetic}=\langle E_{dd}+\Delta E_{\rm ani}\rangle$; "
            r"$K_{\rm cubic}$ enters the transition barriers."
            "\n"
            "Independent-equivalent-bond coordination estimate; "
            rf"$K$ calibrated from $T_B$ at "
            rf"{params.zfc_fc_observation_time_s:g} s."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="normal",
        color="#555555",
    )
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.png")
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.pdf")
    plt.close(figure)


def save_all_coordination_windows_figure(
    temperatures_K: np.ndarray,
    window_cases: dict[float, dict[str, dict]],
    params: V620Parameters = PARAMS,
    normalized: bool = False,
) -> None:
    """Save shared-scale coordination estimates for all windows."""
    all_curves = {
        observation_time_s: coordination_plot_curves(
            temperatures_K, cases, normalized
        )
        for observation_time_s, cases in window_cases.items()
    }
    minimum_energy = min(
        min(
            float(np.min(curve))
            for curve in curves[:3]
        )
        for curves in all_curves.values()
    )
    figure, axes = plt.subplots(
        len(window_cases),
        1,
        figsize=(8.6, 15.0),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.12,
        top=0.90,
        hspace=0.15,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    energy_suffix = (
        r"/(NC\,k_BT)" if normalized else r"/NC"
    )
    for panel_index, (
        observation_time_s,
        cases,
    ) in enumerate(window_cases.items()):
        ax = axes[panel_index]
        face_magnetic, tip_magnetic, face_vdw, thermal = (
            all_curves[observation_time_s]
        )
        add_experimental_temperature_bands(
            ax, "K", show_labels=False
        )
        ax.plot(
            temperatures_K,
            face_magnetic,
            color="#c53030",
            linewidth=2.5,
            label=(
                rf"$E_{{\rm magnetic}}^{{face}}{energy_suffix}$ "
                r"$(n=6)$"
            ),
            zorder=4,
        )
        ax.plot(
            temperatures_K,
            tip_magnetic,
            color="#dd6b20",
            linewidth=2.5,
            label=(
                rf"$E_{{\rm magnetic}}^{{tip}}{energy_suffix}$ "
                r"$(n=8)$"
            ),
            zorder=4,
        )
        ax.plot(
            temperatures_K,
            face_vdw,
            color="#2b6cb0",
            linewidth=2.5,
            label=(
                rf"$E_{{\rm vdW}}^{{face}}{energy_suffix}$ "
                r"$(n=6)$"
            ),
            zorder=3,
        )
        ax.plot(
            temperatures_K,
            thermal,
            color="#252525",
            linewidth=2.3,
            label=(
                r"$k_BT/k_BT=1$"
                if normalized
                else r"$k_BT$"
            ),
            zorder=3,
        )
        ax.axvline(
            params.blocking_temperature_K,
            color="#666666",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            zorder=2,
        )
        ax.axhline(
            0.0, color="#777777", linewidth=0.8, alpha=0.7
        )
        ax.text(
            params.blocking_temperature_K + 1.0,
            0.88,
            rf"$T_B={params.blocking_temperature_K:g}$ K",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=12,
            color="#555555",
            zorder=6,
        )
        ax.set(
            title=f"{observation_time_s:g} s",
            xlim=(200.0, 300.0),
            ylim=(
                1.08 * minimum_energy,
                1.35 if normalized else 6.2e-21,
            ),
        )
        ax.title.set_fontsize(15)
        ax.set_xticks(np.arange(200.0, 301.0, 10.0))
        ax.tick_params(axis="both", labelsize=12.5)
        if not normalized:
            ax.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(0, 0),
                useMathText=True,
            )
        ax.grid(alpha=0.2)

    energy_legend = axes[0].legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=11,
    )
    axes[0].add_artist(energy_legend)
    axes[0].legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=11.5,
    )
    figure.suptitle(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\ncoordination energy per NC: "
            r"face $n=6$, tip $n=8$"
        ),
        fontsize=18,
    )
    figure.supxlabel("Temperature (K)", fontsize=15, y=0.075)
    figure.supylabel(
        r"$E/k_BT$" if normalized else "Energy (J)",
        fontsize=15,
        x=0.035,
    )
    figure.text(
        0.5,
        0.014,
        (
            r"$E_{\rm coord}/NC=(n/2)E_{\rm pair}$; "
            "each pair bond is counted once."
            "\n"
            r"$E_{\rm magnetic}=\langle E_{dd}+\Delta E_{\rm ani}\rangle$; "
            r"$K_{\rm cubic}$ enters the transition barriers."
            "\n"
            "Independent-equivalent-bond estimate; "
            rf"$K$ calibrated from $T_B$ at "
            rf"{params.zfc_fc_observation_time_s:g} s."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    suffix = "_E_over_kBT" if normalized else ""
    figure.savefig(
        OUTPUT_DIR
        / (
            "V620_D5nm_all_experimental_windows_coordination"
            f"{suffix}.png"
        )
    )
    figure.savefig(
        OUTPUT_DIR
        / (
            "V620_D5nm_all_experimental_windows_coordination"
            f"{suffix}.pdf"
        )
    )
    plt.close(figure)


def equilibrium_plot_curves(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    coordination: bool,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equilibrium face, tip, vdW, and thermal curves."""
    thermal_J = Boltzmann * temperatures_K
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    face_factor = (
        coordination_factor_per_nc("face_to_face", face)
        if coordination
        else 1.0
    )
    tip_factor = (
        coordination_factor_per_nc("tip_to_tip", tip)
        if coordination
        else 1.0
    )
    face_magnetic_J = face_factor * face["mean_magnetic_J"]
    tip_magnetic_J = tip_factor * tip["mean_magnetic_J"]
    face_vdw_J = np.full_like(
        temperatures_K, face_factor * face["vdw_pair_J"]
    )
    if normalized:
        return (
            face_magnetic_J / thermal_J,
            tip_magnetic_J / thermal_J,
            face_vdw_J / thermal_J,
            np.ones_like(temperatures_K),
        )
    return (
        face_magnetic_J,
        tip_magnetic_J,
        face_vdw_J,
        thermal_J,
    )


def export_equilibrium_five_nm_csv(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    coordination: bool,
    normalized: bool,
    filename: str,
) -> None:
    """Export canonical equilibrium pair or coordination energies."""
    face_magnetic, tip_magnetic, face_vdw, thermal = (
        equilibrium_plot_curves(
            temperatures_K,
            cases,
            coordination,
            normalized,
        )
    )
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    face_factor = (
        coordination_factor_per_nc("face_to_face", face)
        if coordination
        else 1.0
    )
    tip_factor = (
        coordination_factor_per_nc("tip_to_tip", tip)
        if coordination
        else 1.0
    )
    thermal_J = Boltzmann * temperatures_K
    face_dipole = face_factor * face["mean_dipole_J"]
    tip_dipole = tip_factor * tip["mean_dipole_J"]
    face_anisotropy_response = (
        face_factor * face["anisotropy_response_J"]
    )
    tip_anisotropy_response = (
        tip_factor * tip["anisotropy_response_J"]
    )
    if normalized:
        face_dipole = face_dipole / thermal_J
        tip_dipole = tip_dipole / thermal_J
        face_anisotropy_response = (
            face_anisotropy_response / thermal_J
        )
        tip_anisotropy_response = (
            tip_anisotropy_response / thermal_J
        )
        names = [
            "temperature_K",
            "face_magnetic_energy_over_kBT",
            "tip_magnetic_energy_over_kBT",
            "face_mean_Edd_over_kBT",
            "tip_mean_Edd_over_kBT",
            "face_anisotropy_response_over_kBT",
            "tip_anisotropy_response_over_kBT",
            "face_vdw_energy_over_kBT",
            "kBT_over_kBT",
        ]
    else:
        names = [
            "temperature_K",
            "face_magnetic_energy_J",
            "tip_magnetic_energy_J",
            "face_mean_Edd_J",
            "tip_mean_Edd_J",
            "face_anisotropy_response_J",
            "tip_anisotropy_response_J",
            "face_vdw_energy_J",
            "kBT_J",
        ]
    np.savetxt(
        OUTPUT_DIR / filename,
        np.column_stack(
            [
                temperatures_K,
                face_magnetic,
                tip_magnetic,
                face_dipole,
                tip_dipole,
                face_anisotropy_response,
                tip_anisotropy_response,
                face_vdw,
                thermal,
            ]
        ),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12e",
    )


def save_equilibrium_five_nm_figure(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
    coordination: bool,
    normalized: bool,
    filename_stem: str,
) -> None:
    """Plot observation-time-independent canonical equilibrium energies."""
    face_magnetic, tip_magnetic, face_vdw, thermal = (
        equilibrium_plot_curves(
            temperatures_K,
            cases,
            coordination,
            normalized,
        )
    )
    figure, ax = plt.subplots(figsize=(10.2, 10.2))
    figure.subplots_adjust(
        left=0.105, right=0.64, bottom=0.18, top=0.87
    )
    add_experimental_temperature_bands(
        ax, "K", show_labels=False
    )
    if coordination:
        energy_suffix = (
            r"/(NC\,k_BT)" if normalized else r"/NC"
        )
        face_label = (
            rf"$E_{{\rm magnetic}}^{{face}}{energy_suffix}$ "
            r"$(n=6)$"
        )
        tip_label = (
            rf"$E_{{\rm magnetic}}^{{tip}}{energy_suffix}$ "
            r"$(n=8)$"
        )
        vdw_label = (
            rf"$E_{{\rm vdW}}^{{face}}{energy_suffix}$ "
            r"$(n=6)$"
        )
    else:
        energy_suffix = r"/k_BT" if normalized else ""
        face_label = (
            rf"$E_{{\rm magnetic}}^{{face}}{energy_suffix}$"
        )
        tip_label = (
            rf"$E_{{\rm magnetic}}^{{tip}}{energy_suffix}$"
        )
        vdw_label = (
            rf"$E_{{\rm vdW}}^{{face,\ pair}}{energy_suffix}$"
        )
    ax.plot(
        temperatures_K,
        face_magnetic,
        color="#c53030",
        linewidth=2.4,
        label=face_label,
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        tip_magnetic,
        color="#dd6b20",
        linewidth=2.4,
        label=tip_label,
        zorder=4,
    )
    ax.plot(
        temperatures_K,
        face_vdw,
        color="#2b6cb0",
        linewidth=2.4,
        label=vdw_label,
        zorder=3,
    )
    ax.plot(
        temperatures_K,
        thermal,
        color="#252525",
        linewidth=2.0,
        label=(
            r"$k_BT/k_BT=1$"
            if normalized
            else r"$k_BT$"
        ),
        zorder=3,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    minimum_energy = min(
        float(np.min(face_magnetic)),
        float(np.min(tip_magnetic)),
        float(np.min(face_vdw)),
    )
    ax.set(
        xlim=(200.0, 300.0),
        ylim=(
            1.08 * minimum_energy,
            1.35 if normalized else 6.2e-21,
        ),
    )
    title_second_line = (
        r"continuous equilibrium coordination per NC: face $n=6$, tip $n=8$"
        if coordination
        else "continuous equilibrium pair energies"
    )
    ax.set_title(
        (
            r"16 nm Fe$_3$O$_4$, $D=5$ nm"
            "\n"
            f"{title_second_line}"
        ),
        fontsize=17,
        fontweight="normal",
        pad=12,
    )
    ax.set_xlabel(
        "Temperature (K)", fontsize=16, fontweight="normal"
    )
    ax.set_ylabel(
        r"$E/k_BT$" if normalized else "Energy (J)",
        fontsize=16,
        fontweight="normal",
    )
    if not normalized:
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
    ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.2)
    energy_legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.96,
        edgecolor="none",
        fontsize=11.5,
    )
    ax.add_artist(energy_legend)
    ax.legend(
        handles=[
            Patch(
                facecolor="#d9d9d9",
                alpha=0.38,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.62,
                edgecolor="none",
                label="transient aggregation",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="none",
        fontsize=12,
    )
    coordination_note = (
        r"$E_{\rm coord}/NC=(n/2)E_{\rm pair}$."
        if coordination
        else "One isolated pair is counted."
    )
    figure.text(
        0.5,
        0.033,
        (
            r"$\rho(\mathbf{m}_1,\mathbf{m}_2)\propto\;"
            r"e^{-[E_{\rm ani,1}+E_{\rm ani,2}+E_{dd}]/k_BT}$ "
            r"within the $+[111]$ basin."
            "\n"
            r"$E_{\rm magnetic}=\langle E_{dd}\rangle+"
            r"[\langle E_{\rm ani}\rangle_{\rm pair}-"
            r"2\langle E_{\rm ani}\rangle_{\rm isolated}]$. "
            f"{coordination_note}"
            "\n"
            r"No experimental $t_{\rm obs}$ or transition rate enters "
            r"the equilibrium average."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="normal",
        color="#555555",
    )
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.png")
    figure.savefig(OUTPUT_DIR / f"{filename_stem}.pdf")
    plt.close(figure)


def write_equilibrium_five_nm_summary(
    temperatures_K: np.ndarray,
    cases: dict[str, dict],
) -> None:
    """Write selected canonical-equilibrium values and assumptions."""
    lines = [
        "V6.20 D=5 nm continuous +[111]-basin canonical equilibrium",
        "================================================",
        "State space: continuous moment directions in each NC's +[111] octant",
        "Angular quadrature: 40 polar x 40 azimuthal points per NC",
        "Joint density: rho(m1,m2) proportional to exp[-(Eani1+Eani2+Edd)/kBT]",
        "No experimental observation window or transition rate enters the canonical average",
        "K_cubic is fixed once from the ZFC/FC-derived barrier; it is not switched at TB",
        "Magnetic energy = <Edd> + (<Eani>pair - 2<Eani>isolated)",
        "Only face-to-face vdW is reported in the plotted components",
        "",
        "T_K,face_pair_zJ,tip_pair_zJ,face_pair_Edd_zJ,tip_pair_Edd_zJ,"
        "face_pair_anisotropy_response_zJ,tip_pair_anisotropy_response_zJ,"
        "face_coord_per_NC_zJ,tip_coord_per_NC_zJ",
    ]
    face = cases["face_to_face"]
    tip = cases["tip_to_tip"]
    for temperature_K in (200.0, 250.0, 300.0):
        index = int(
            np.argmin(
                np.abs(temperatures_K - temperature_K)
            )
        )
        face_pair = face["mean_magnetic_J"][index] / 1e-21
        tip_pair = tip["mean_magnetic_J"][index] / 1e-21
        face_dipole = face["mean_dipole_J"][index] / 1e-21
        tip_dipole = tip["mean_dipole_J"][index] / 1e-21
        face_anisotropy = (
            face["anisotropy_response_J"][index] / 1e-21
        )
        tip_anisotropy = (
            tip["anisotropy_response_J"][index] / 1e-21
        )
        lines.append(
            f"{temperatures_K[index]:.2f},"
            f"{face_pair:.9f},{tip_pair:.9f},"
            f"{face_dipole:.9f},{tip_dipole:.9f},"
            f"{face_anisotropy:.9f},{tip_anisotropy:.9f},"
            f"{3.0 * face_pair:.9f},{4.0 * tip_pair:.9f}"
        )
    pair_difference = (
        face["mean_magnetic_J"] - tip["mean_magnetic_J"]
    )
    coordination_difference = (
        3.0 * face["mean_magnetic_J"]
        - 4.0 * tip["mean_magnetic_J"]
    )
    lines.extend(
        [
            "",
            (
                "Pair crossover in 200-300 K: "
                + (
                    "present"
                    if np.any(
                        pair_difference[:-1]
                        * pair_difference[1:]
                        <= 0.0
                    )
                    else "none"
                )
            ),
            (
                "Coordination crossover in 200-300 K: "
                + (
                    "present"
                    if np.any(
                        coordination_difference[:-1]
                        * coordination_difference[1:]
                        <= 0.0
                    )
                    else "none"
                )
            ),
        ]
    )
    (
        OUTPUT_DIR / "V620_D5nm_equilibrium_summary.txt"
    ).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_gap_sensitivity_summary(
    temperatures_K: np.ndarray,
    sensitivity: dict[float, dict[str, dict]],
    params: V620Parameters = PARAMS,
) -> None:
    """Write the room-temperature total energies for all evaluated gaps."""
    reference_index = int(
        np.argmin(np.abs(temperatures_K - params.reference_temperature_K))
    )
    reference_kBT_J = Boltzmann * params.reference_temperature_K
    lines = [
        "V6.20 larger-gap sensitivity at 298.15 K",
        "==========================================",
        "gap_nm,geometry,center_distance_nm,EvdW_kBT,DeltaU_mag_kBT,Etotal_kBT",
    ]
    for gap_nm, gap_cases in sensitivity.items():
        for key in ("face_to_face", "tip_to_tip"):
            case = gap_cases[key]
            magnetic_J = case["magnetic_interaction_energy_J"][reference_index]
            total_J = case["vdw_J"] + magnetic_J
            lines.append(
                f"{gap_nm:.6f},{key},"
                f"{case['center_distance_m'] * 1e9:.6f},"
                f"{case['vdw_J'] / reference_kBT_J:.6f},"
                f"{magnetic_J / reference_kBT_J:.6f},"
                f"{total_J / reference_kBT_J:.6f}"
            )
    (OUTPUT_DIR / "V620_gap_sensitivity_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_gap_m, cases = calculate_cases(TEMPERATURES_K, PARAMS)
    export_csv(TEMPERATURES_K, reference_gap_m, cases, PARAMS)
    save_figures(TEMPERATURES_K, reference_gap_m, cases, PARAMS)
    write_summary(TEMPERATURES_K, reference_gap_m, cases, PARAMS)
    equilibrium_cases = calculate_equilibrium_five_nm_cases(
        DYNAMIC_TEMPERATURES_K, PARAMS
    )
    for coordination, geometry_tag in (
        (False, "pair"),
        (True, "coordination"),
    ):
        for normalized, normalization_tag in (
            (False, ""),
            (True, "_E_over_kBT"),
        ):
            filename_stem = (
                f"V620_D5nm_equilibrium_{geometry_tag}"
                f"{normalization_tag}"
            )
            export_equilibrium_five_nm_csv(
                DYNAMIC_TEMPERATURES_K,
                equilibrium_cases,
                coordination,
                normalized,
                filename=f"{filename_stem}_data.csv",
            )
            save_equilibrium_five_nm_figure(
                DYNAMIC_TEMPERATURES_K,
                equilibrium_cases,
                coordination,
                normalized,
                filename_stem=filename_stem,
            )
    write_equilibrium_five_nm_summary(
        DYNAMIC_TEMPERATURES_K, equilibrium_cases
    )
    window_cases: dict[float, dict[str, dict]] = {}
    for observation_time_s in EXPERIMENTAL_WINDOWS_S:
        tag = f"{observation_time_s:g}s".replace(".", "p")
        dynamic_cases = calculate_dynamic_five_nm_cases(
            DYNAMIC_TEMPERATURES_K,
            observation_time_s,
            PARAMS,
        )
        window_cases[observation_time_s] = dynamic_cases
        export_dynamic_five_nm_csv(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            filename=f"V620_D5nm_temperature_data_{tag}.csv",
        )
        export_dynamic_five_nm_normalized_csv(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            filename=(
                f"V620_D5nm_E_over_kBT_data_{tag}.csv"
            ),
        )
        export_coordination_dynamic_csv(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            filename=(
                f"V620_D5nm_coordination_data_{tag}.csv"
            ),
        )
        export_coordination_dynamic_csv(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            filename=(
                "V620_D5nm_coordination_E_over_kBT_data_"
                f"{tag}.csv"
            ),
            normalized=True,
        )
        write_dynamic_five_nm_summary(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            PARAMS,
            filename=f"V620_D5nm_dynamic_summary_{tag}.txt",
        )
        save_signed_geometry_selection_figure(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            PARAMS,
            filename_stem=(
                f"V620_D5nm_signed_geometry_selection_{tag}"
            ),
        )
        save_normalized_geometry_selection_figure(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            PARAMS,
            filename_stem=f"V620_D5nm_E_over_kBT_{tag}",
        )
        save_coordination_geometry_selection_figure(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            PARAMS,
            filename_stem=f"V620_D5nm_coordination_{tag}",
        )
        save_coordination_geometry_selection_figure(
            DYNAMIC_TEMPERATURES_K,
            dynamic_cases,
            PARAMS,
            normalized=True,
            filename_stem=(
                f"V620_D5nm_coordination_E_over_kBT_{tag}"
            ),
        )
    export_all_dynamic_windows_csv(
        DYNAMIC_TEMPERATURES_K, window_cases
    )
    save_all_experimental_windows_figure(
        DYNAMIC_TEMPERATURES_K, window_cases, PARAMS
    )
    export_all_dynamic_windows_normalized_csv(
        DYNAMIC_TEMPERATURES_K, window_cases
    )
    save_all_normalized_experimental_windows_figure(
        DYNAMIC_TEMPERATURES_K, window_cases, PARAMS
    )
    export_all_coordination_windows_csv(
        DYNAMIC_TEMPERATURES_K, window_cases
    )
    export_all_coordination_windows_csv(
        DYNAMIC_TEMPERATURES_K,
        window_cases,
        normalized=True,
    )
    save_all_coordination_windows_figure(
        DYNAMIC_TEMPERATURES_K, window_cases, PARAMS
    )
    save_all_coordination_windows_figure(
        DYNAMIC_TEMPERATURES_K,
        window_cases,
        PARAMS,
        normalized=True,
    )
    # Keep the established filenames as aliases of the 1 s result.
    one_second_cases = window_cases[1.0]
    export_dynamic_five_nm_csv(
        DYNAMIC_TEMPERATURES_K, one_second_cases
    )
    write_dynamic_five_nm_summary(
        DYNAMIC_TEMPERATURES_K, one_second_cases, PARAMS
    )
    save_signed_geometry_selection_figure(
        DYNAMIC_TEMPERATURES_K, one_second_cases, PARAMS
    )
    export_dynamic_five_nm_normalized_csv(
        DYNAMIC_TEMPERATURES_K,
        one_second_cases,
        filename="V620_D5nm_E_over_kBT_data.csv",
    )
    save_normalized_geometry_selection_figure(
        DYNAMIC_TEMPERATURES_K,
        one_second_cases,
        PARAMS,
    )
    export_coordination_dynamic_csv(
        DYNAMIC_TEMPERATURES_K,
        one_second_cases,
        filename="V620_D5nm_coordination_data.csv",
    )
    export_coordination_dynamic_csv(
        DYNAMIC_TEMPERATURES_K,
        one_second_cases,
        filename=(
            "V620_D5nm_coordination_E_over_kBT_data.csv"
        ),
        normalized=True,
    )
    save_coordination_geometry_selection_figure(
        DYNAMIC_TEMPERATURES_K,
        one_second_cases,
        PARAMS,
    )
    save_coordination_geometry_selection_figure(
        DYNAMIC_TEMPERATURES_K,
        one_second_cases,
        PARAMS,
        normalized=True,
        filename_stem=(
            "V620_D5nm_coordination_E_over_kBT"
        ),
    )
    print((OUTPUT_DIR / "V620_two_NC_temperature.png").resolve())
    print((OUTPUT_DIR / "V620_two_NC_temperature_data.csv").resolve())
    print(
        (
            OUTPUT_DIR / "V620_D5nm_all_experimental_windows.png"
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR / "V620_D5nm_all_experimental_windows_data.csv"
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR
            / "V620_D5nm_all_experimental_windows_E_over_kBT.png"
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR
            / "V620_D5nm_all_experimental_windows_E_over_kBT_data.csv"
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR
            / "V620_D5nm_all_experimental_windows_coordination.png"
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR
            / (
                "V620_D5nm_all_experimental_windows_"
                "coordination_E_over_kBT.png"
            )
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR / "V620_D5nm_equilibrium_pair.png"
        ).resolve()
    )
    print(
        (
            OUTPUT_DIR
            / "V620_D5nm_equilibrium_coordination.png"
        ).resolve()
    )


if __name__ == "__main__":
    main()
