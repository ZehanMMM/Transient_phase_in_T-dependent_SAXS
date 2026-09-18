"""Colloid/superlattice populations from joint cube- and dipole-configuration averaging.

WHAT THIS ADDS TO THE INHERITED V10 PAIR MODELS
-----------------------------------------------
Every existing entry point evaluates ONE fixed pair geometry (face-to-face or
tip-to-tip at a 3 nm surface gap) and reports its magnetic energy.  None of
them decides whether that pair exists.  Here the contact geometry is promoted
to a label of the configuration space

    contact  c   in {free, face, tip}            <- "cube configuration"
    dipole  (i,j) in the inherited 8 x 8 lab <111> easy-axis pairs

giving 3 x 64 = 192 states.  `free` is the colloidally dispersed pair; `face`
and `tip` are the inherited contact geometries at the inherited gap.  Summing
over c is the cube-configuration average, summing over (i,j) is the
dipole-configuration average.  The colloid/SL ratio is the marginal over c,
and the "equivalent total system energy" is the configuration average of the
per-NC energy over the SAME distribution.

THE TWO RELAXATION CHANNELS ARE KEPT SEPARATE, WHICH IS THE POINT
-----------------------------------------------------------------
The lab-frame dipole label can change two ways, and the inherited parameters
put them on very different clocks over 200-300 K:

  Neel, tau_N = tau0 exp(dE/kBT) with the inherited ZFC/FC barrier:
      1.5 s at 300 K, 100 s at 250 K (= T_B by construction), 5.7e4 s at
      200 K.  This moves the moment INSIDE a fixed cube, so it can convert a
      bound attractive configuration into a repulsive one WITHOUT moving the
      cube.  That is the mechanism that dissolves a near-contact pair at
      300 K.
  Brownian rigid-cube rotation, tau_B ~ 1e-6 s at every temperature here:
      this rotates the cube with the moment frozen in the body frame.  Below
      T_B it is the only surviving channel, and while bound it is restricted
      to the rotations that preserve the contact: the two +/-90 deg quarter
      turns about the <100> bond for `face`, the two +/-120 deg turns about
      the <111> bond for `tip`.  A free cube keeps the inherited six <100>
      quarter turns.

Because tau_B stays microscopic, the lab dipole label equilibrates in BOTH
regimes, so the equilibrium colloid/SL ratio is the same function of T in
both; detailed balance guarantees that and this module does not pretend
otherwise.  What changes across T_B is the pair LIFETIME: with Neel open, a
bound pair is ejected as soon as a flip lands it in a weakly bound state,
whereas with Neel blocked, escape must first climb the dipolar barrier of a
contact-preserving rotation.  Both are computed, and `persistent_SL_fraction`
= p_bound x (20 s survival) is the population that is still the same pair at
the end of the observation window.

BINDING: ONE LENGTH AND ONE CONCENTRATION, NOTHING ELSE
-------------------------------------------------------
free(i,j) <-> c(i,j) is a docking move that preserves the dipole label:
    k_on  = z_c nu_enc exp(-max(U,0)/kBT),   nu_enc = k_diff n
    k_off =     nu_esc exp(min(U,0)/kBT),    nu_esc = D_rel / l^2
with k_diff = 8 kBT/(3 eta) the Smoluchowski rate for equal spheres, D_rel =
2 kBT/(3 pi eta d_h) the relative translational diffusivity, n = phi/V the
number density and z_c the inherited coordination numbers.  Detailed balance
then FIXES the reaction volume,
    v_site = k_diff / nu_esc = 4 pi d_h l^2   (temperature independent),
so the bound/free weight is alpha_c = z_c v_site n and the only free inputs
are the escape length l (ligand-shell compliance, 0.5 nm default, giving
v_site = 59.7 nm^3) and the volume fraction phi.  Both are ASSUMPTIONS, not
measured here, and both are swept.  phi enters alpha_c linearly, so it shifts
the colloid/SL crossover logarithmically and cannot change its slope.

ASSEMBLY MODE
-------------
`dimer`        one bond per pair.  Mass action puts the full bond energy in
               the exponent and reports U_bond/2 per NC.
`superlattice` the inherited coordination_factor_per_nc convention: a lattice
               site carries z_c/2 bonds, so the exponent, the reported per-NC
               energy and the dipolar coupling entering every rate are all
               scaled by z_c/2.  This is a MEAN-FIELD coordination scaling of
               a pair model, not a many-body calculation: all z_c neighbours
               are assumed to share the same relative configuration, and
               alpha_c keeps the dimer reaction volume.

NOT INCLUDED, DELIBERATELY
--------------------------
Ligand/solvent PMF, the face<->tip structural conversion path, positional and
librational entropy beyond v_site, many-body magnetic correlations, the
measured size distribution (CV = 0 inherited), and any Ms(T) curve.  U(free)
is set to 0; the dipolar coupling at the mean free separation is reported as
a diagnostic so the size of that approximation stays visible.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
from scipy.linalg import eigh, expm
from scipy.special import logsumexp

import geometry_model as geometry
from pair_energy_model import (HYDRODYNAMIC_DIAMETER_M, brownian_time_s,
                               hexane_viscosity_Pa_s)

OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'configuration_averaged_assembly'

CONTACTS = ('free', 'face', 'tip')
BOUND = ('face', 'tip')
LINKS = {'face': np.array([1., 0., 0.]), 'tip': np.ones(3) / np.sqrt(3)}
# Inherited geometry_model.COORDINATION_NUMBERS keys, reached through the
# short contact labels used throughout this module.
COORDINATION = {kind: geometry.COORDINATION_NUMBERS[name] for kind, name in
                (('face', 'face_to_face'), ('tip', 'tip_to_tip'))}
SURFACE_GAP_NM = 3.0
WINDOW_S = 20.0
QUARTER, THIRD = np.pi / 2, 2 * np.pi / 3
# Contact-preserving rigid rotations.  Each set is a cyclic subgroup of the
# cube group about the bond axis, so it maps <111> onto <111> and preserves
# the face/tip registry exactly.  `free` keeps the inherited six <100> turns.
MOVES = {
    'free': [(axis, sign * QUARTER) for axis in np.eye(3) for sign in (1, -1)],
    'face': [(LINKS['face'], sign * QUARTER) for sign in (1, -1)],
    'tip': [(LINKS['tip'], sign * THIRD) for sign in (1, -1)],
}
COLORS = {'free': '#287A96', 'face': '#B54535', 'tip': '#DE9945'}


@dataclass(frozen=True)
class Binding:
    volume_fraction: float = 1.0e-2
    escape_length_nm: float = 0.5
    mobility: float = 1.0
    contact_barrier_kBT: float = 0.0
    assembly: str = 'dimer'

    def __post_init__(self):
        if not 0 < self.volume_fraction < 0.74:
            raise ValueError('volume_fraction must describe a physical dispersion')
        if not 0 < self.escape_length_nm <= 5:
            raise ValueError('escape_length_nm must lie in (0, 5] nm')
        if self.mobility < 0 or self.contact_barrier_kBT < 0:
            raise ValueError('mobility and contact barrier must be nonnegative')
        if self.assembly not in ('dimer', 'superlattice'):
            raise ValueError('assembly must be dimer or superlattice')


# --------------------------------------------------------------------------
# binding kinetics, derived entirely from the inherited hexane viscosity
# --------------------------------------------------------------------------
def number_density_per_m3(volume_fraction: float) -> float:
    return volume_fraction / geometry.PARAMS.particle_volume_m3


def smoluchowski_rate_m3ps(temperature_K: float) -> float:
    """k_diff = 8 kBT / (3 eta) for equal spheres; the diameter cancels."""
    eta = float(hexane_viscosity_Pa_s(np.array([temperature_K]))[0])
    return 8.0 * Boltzmann * temperature_K / (3.0 * eta)


def relative_diffusion_m2ps(temperature_K: float) -> float:
    eta = float(hexane_viscosity_Pa_s(np.array([temperature_K]))[0])
    return 2.0 * Boltzmann * temperature_K / (3.0 * np.pi * eta * HYDRODYNAMIC_DIAMETER_M)


def site_volume_m3(escape_length_nm: float) -> float:
    """v_site = k_diff / nu_esc = 4 pi d_h l^2; temperature independent."""
    return 4.0 * np.pi * HYDRODYNAMIC_DIAMETER_M * (escape_length_nm * 1e-9) ** 2


# --------------------------------------------------------------------------
# geometry and rigid rotations
# --------------------------------------------------------------------------
def coordination_scale(kind: str, assembly: str) -> float:
    """Bond multiplicity entering the exponent and every rate."""
    if assembly == 'dimer':
        return 1.0
    return COORDINATION[kind] / 2.0


def report_scale(kind: str, assembly: str) -> float:
    """Bond multiplicity entering the reported per-NC energy."""
    return 0.5 if assembly == 'dimer' else COORDINATION[kind] / 2.0


def contact_geometry(kind: str, assembly: str = 'dimer') -> dict:
    """Inherited 3 nm-gap pair geometry, scaled by the bond multiplicity."""
    params = geometry.PARAMS
    scale = coordination_scale(kind, assembly)
    distance_m = geometry.center_distance_at_gap_m(
        LINKS[kind], SURFACE_GAP_NM * 1e-9, params)
    dipole_J, prefactor_J, states, link = geometry.easy_axis_pair_energies_J(
        distance_m * LINKS[kind], params)
    vdw_J = geometry.pair_vdw_energy_J(distance_m * LINKS[kind], params)
    return dict(kind=kind, link=link, states=states, center_distance_m=distance_m,
                bond_multiplicity=scale, dipole_J=scale * dipole_J,
                prefactor_J=scale * prefactor_J, vdw_J=scale * vdw_J,
                energy_J=scale * (vdw_J + dipole_J),
                report_factor=report_scale(kind, assembly) / scale)


def rotation_matrix(axis, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    cross = np.array([[0.0, -unit[2], unit[1]],
                      [unit[2], 0.0, -unit[0]],
                      [-unit[1], unit[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def path_maximum(prefactor_J, moment, other, axis, link, angle):
    """Exact maximum of the dipolar energy along a rigid rotation of `moment`.

    E(theta) = D + A cos(theta) + B sin(theta), the rotation acting on one
    moment with the partner and the bond held fixed.  Both endpoints are
    candidates, so the maximum never falls below either endpoint energy and
    the barrier built from it is never clipped; transition rates therefore
    satisfy detailed balance exactly.
    """
    field = np.asarray(other) - 3.0 * np.dot(other, link) * np.asarray(link)
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    parallel = np.dot(unit, moment) * unit
    a = prefactor_J * np.dot(moment - parallel, field)
    b = prefactor_J * np.dot(np.cross(unit, moment), field)
    d = prefactor_J * np.dot(parallel, field)
    low, high = sorted((0.0, float(angle)))
    stationary = np.arctan2(b, a)
    angles = [low, high]
    for turn in (-2, -1, 0, 1, 2):
        candidate = stationary + turn * np.pi
        if low <= candidate <= high:
            angles.append(candidate)
    return max(d + a * np.cos(t) + b * np.sin(t) for t in angles)


def rotation_decay_rank(moves) -> float:
    """lambda in sum_k (R_k - I) = -lambda (I - P), P the invariant projector.

    Taking attempt = mobility / (lambda tau_B) per listed move makes the
    zero-coupling first-rank decay of every relaxable moment component equal
    to 1/tau_B.  This reproduces the inherited 1/(4 tau_B) for the six <100>
    quarter turns and extends the same calibration to the restricted
    contact-preserving sets (lambda = 2 for face, 3 for tip).
    """
    total = sum(rotation_matrix(axis, angle) - np.eye(3) for axis, angle in moves)
    if not np.allclose(total, total.T, atol=1e-12):
        raise ValueError('Move set is not reversible: sum of rotations is asymmetric')
    values, vectors = np.linalg.eigh(total)
    invariant = int(np.sum(np.abs(values) < 1e-10))
    if invariant == 3:
        raise ValueError('Move set does not relax any moment component')
    rank = float(-np.trace(total) / (3 - invariant))
    projector = vectors[:, np.abs(values) < 1e-10]
    expected = -rank * (np.eye(3) - projector @ projector.T)
    if not np.allclose(total, expected, atol=1e-9 * max(rank, 1.0)):
        raise ValueError('Move set does not give an isotropic transverse decay')
    return rank


# --------------------------------------------------------------------------
# temperature-independent transition structures
# --------------------------------------------------------------------------
def registry_structure(geom: dict, moves) -> dict:
    """Rigid-rotation moves of the 64 lab dipole pairs, barriers precomputed."""
    states, energies = geom['states'], geom['dipole_J']
    prefactor, link = geom['prefactor_J'], geom['link']
    lookup = {tuple(np.sign(state).astype(int)): index
              for index, state in enumerate(states)}
    rotations = [(np.asarray(axis, dtype=float), float(angle),
                  rotation_matrix(axis, angle)) for axis, angle in moves]
    sources, destinations, barriers = [], [], []
    for index_1, moment_1 in enumerate(states):
        for index_2, moment_2 in enumerate(states):
            source = 8 * index_1 + index_2
            for particle in (0, 1):
                moment, other = ((moment_1, moment_2) if particle == 0
                                 else (moment_2, moment_1))
                for axis, angle, matrix in rotations:
                    rotated = matrix @ moment
                    moved = lookup[tuple(np.sign(rotated).astype(int))]
                    destination = (8 * moved + index_2 if particle == 0
                                   else 8 * index_1 + moved)
                    if destination == source:
                        continue  # bond-axis turn leaving the moment invariant
                    saddle = path_maximum(prefactor, moment, other, axis, link, angle)
                    sources.append(source)
                    destinations.append(destination)
                    barriers.append(max(saddle - energies[source], 0.0))
    return dict(source=np.array(sources, dtype=int),
                destination=np.array(destinations, dtype=int),
                barrier_J=np.array(barriers), rank=rotation_decay_rank(moves))


def neel_structure(geom: dict) -> dict:
    """Inherited adjacent-well Neel moves, barriers precomputed.

    Reproduces geometry_model.neel_pair_generator exactly (asserted in the
    tests); only the temperature-independent barrier list is cached here.
    """
    states, energies = geom['states'], geom['dipole_J']
    prefactor, link = geom['prefactor_J'], geom['link']
    activation_J = geometry.PARAMS.zfc_fc_activation_barrier_J
    lookup = {tuple(np.sign(state).astype(int)): index
              for index, state in enumerate(states)}
    sources, destinations, barriers = [], [], []
    for index_1, moment_1 in enumerate(states):
        for index_2, moment_2 in enumerate(states):
            source = 8 * index_1 + index_2
            for particle in (0, 1):
                current = moment_1 if particle == 0 else moment_2
                other = moment_2 if particle == 0 else moment_1
                for component in range(3):
                    signs = np.sign(current).astype(int)
                    signs[component] *= -1
                    moved = lookup[tuple(signs)]
                    saddle_state = current.copy()
                    saddle_state[component] = 0.0
                    saddle_state = saddle_state / np.linalg.norm(saddle_state)
                    saddle_J = prefactor * (np.dot(saddle_state, other)
                                            - 3.0 * np.dot(saddle_state, link)
                                            * np.dot(other, link))
                    destination = (8 * moved + index_2 if particle == 0
                                   else 8 * index_1 + moved)
                    sources.append(source)
                    destinations.append(destination)
                    barriers.append(max(activation_J + saddle_J - energies[source], 0.0))
    return dict(source=np.array(sources, dtype=int),
                destination=np.array(destinations, dtype=int),
                barrier_J=np.array(barriers))


def generator_from_structure(structure: dict, temperature_K: float,
                             attempt_per_s: float,
                             extra_barrier_J: float = 0.0) -> np.ndarray:
    matrix = np.zeros((64, 64))
    rates = attempt_per_s * np.exp(
        -(structure['barrier_J'] + extra_barrier_J) / (Boltzmann * temperature_K))
    np.add.at(matrix, (structure['source'], structure['destination']), rates)
    np.fill_diagonal(matrix, 0.0)
    np.fill_diagonal(matrix, -matrix.sum(axis=1))
    return matrix


def build_structures(assembly: str) -> dict:
    """All temperature-independent geometry and barrier data for one mode."""
    geometries = {kind: contact_geometry(kind, assembly) for kind in BOUND}
    free = dict(geometries['face'])
    free.update(kind='free', dipole_J=np.zeros(64), prefactor_J=0.0, vdw_J=0.0,
                energy_J=np.zeros(64), bond_multiplicity=1.0, report_factor=0.0,
                center_distance_m=np.nan)
    geometries['free'] = free
    return dict(assembly=assembly, geometry=geometries,
                neel={kind: neel_structure(geometries[kind]) for kind in CONTACTS},
                registry={kind: registry_structure(geometries[kind], MOVES[kind])
                          for kind in CONTACTS})


# --------------------------------------------------------------------------
# the 192-state joint model
# --------------------------------------------------------------------------
def block(kind: str) -> np.ndarray:
    return np.arange(64) + 64 * CONTACTS.index(kind)


def per_state_report_energy(structures: dict) -> np.ndarray:
    """Reported per-NC energy of all 192 states; temperature independent."""
    values = np.zeros(192)
    for kind in CONTACTS:
        geom = structures['geometry'][kind]
        values[block(kind)] = geom['report_factor'] * geom['energy_J']
    return values


def _label_generator(structures, kind, temperature_K, binding, tau_B_s,
                     channels=('neel', 'registry')) -> np.ndarray:
    """Within-label dipole dynamics: Neel and/or contact-preserving rotation."""
    total = np.zeros((64, 64))
    if 'neel' in channels:
        total += generator_from_structure(
            structures['neel'][kind], temperature_K,
            1.0 / (3.0 * geometry.PARAMS.attempt_time_s))
    if 'registry' in channels and binding.mobility > 0:
        extra = 0.0 if kind == 'free' else binding.contact_barrier_kBT * Boltzmann * temperature_K
        total += generator_from_structure(
            structures['registry'][kind], temperature_K,
            binding.mobility / (structures['registry'][kind]['rank'] * tau_B_s), extra)
    return total


def joint_model(temperature_K: float, binding: Binding, structures: dict,
                channels=('neel', 'registry')) -> dict:
    """Energies, equilibrium weights and the full detailed-balanced generator."""
    kbt = Boltzmann * temperature_K
    geometries = structures['geometry']
    density = number_density_per_m3(binding.volume_fraction)
    volume = site_volume_m3(binding.escape_length_nm)
    k_diff = smoluchowski_rate_m3ps(temperature_K)
    encounter_per_s = k_diff * density
    escape_per_s = k_diff / volume
    tau_B = float(brownian_time_s(np.array([temperature_K]))[0])

    energy_J = np.zeros(192)
    report_J = per_state_report_energy(structures)
    log_weight = np.zeros(192)
    alpha = {}
    for kind in CONTACTS:
        geom = geometries[kind]
        rows = block(kind)
        energy_J[rows] = geom['energy_J']
        alpha[kind] = (1.0 if kind == 'free'
                       else COORDINATION[kind] * volume * density)
        log_weight[rows] = np.log(alpha[kind]) - geom['energy_J'] / kbt
    equilibrium = np.exp(log_weight - logsumexp(log_weight))

    generator = np.zeros((192, 192))
    for kind in CONTACTS:
        rows = block(kind)
        generator[np.ix_(rows, rows)] = _label_generator(
            structures, kind, temperature_K, binding, tau_B, channels)
    free_rows = block('free')
    for kind in BOUND:
        bound_rows = block(kind)
        bound_energy = geometries[kind]['energy_J']
        on_per_s = (COORDINATION[kind] * encounter_per_s
                    * np.exp(-np.maximum(bound_energy, 0.0) / kbt))
        off_per_s = escape_per_s * np.exp(np.minimum(bound_energy, 0.0) / kbt)
        generator[free_rows, bound_rows] += on_per_s
        generator[bound_rows, free_rows] += off_per_s
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))

    scale = max(float(np.max(np.abs(generator))), 1e-300)
    flux = equilibrium[:, None] * generator
    imbalance = float(np.max(np.abs(flux - flux.T))
                      / (scale * float(np.max(equilibrium))))
    if imbalance > 1e-9:
        raise ArithmeticError(
            f'Joint generator violates detailed balance: {imbalance:.3e}')
    return dict(energy_J=energy_J, report_J=report_J, equilibrium=equilibrium,
                generator=generator, alpha=alpha, kbt_J=kbt, tau_B_s=tau_B,
                encounter_per_s=encounter_per_s, escape_per_s=escape_per_s,
                site_volume_m3=volume, number_density_per_m3=density,
                detailed_balance_relative_error=imbalance)


def propagate(generator, initial, window_s):
    """Endpoint and window-mean probability by scaling-and-squaring expm.

    The sqrt(pi) spectral propagator inherited from pair_free_reference cannot
    be used here.  In `superlattice` mode pi spans ~25 decades (alpha_c times
    exp(34) at 200 K against the free label), and reconstructing p from the
    symmetrised eigenbasis then divides by sqrt(pi) and breaks that module's
    1e-9 positivity guard.  The generator itself is well scaled, so the
    augmented block

        expm( [[Q, I], [0, 0]] t ) = [[exp(Qt), int_0^t exp(Qs) ds], [0, I]]

    gives the endpoint and the exact window mean in one exponential, with
    positivity and normalisation checked afterwards.  Returns
    (endpoint, window mean, normalisation drift).
    """
    count = len(initial)
    augmented = np.zeros((2 * count, 2 * count))
    augmented[:count, :count] = generator
    augmented[:count, count:] = np.eye(count)
    blocks = expm(augmented * window_s)
    end = np.asarray(initial) @ blocks[:count, :count]
    mean = np.asarray(initial) @ blocks[:count, count:] / window_s
    # ||Q|| window_s reaches ~1e10 here, so scaling-and-squaring needs ~34
    # squarings and leaks up to ~1e-4 of the total probability.  The leak is a
    # uniform loss of norm, not a distortion of the distribution (checked
    # against a composite Gauss-Legendre quadrature in the tests to 1e-9), so
    # it is renormalised away and reported as a diagnostic.  A tighter guard
    # would reject only long windows; expm_multiply is not an alternative,
    # its cost scales with ||Q|| window_s.
    drift = max(abs(end.sum() - 1.0), abs(mean.sum() - 1.0))
    for name, probability in (('endpoint', end), ('window mean', mean)):
        if probability.min() < -1e-9 or drift > 1e-3:
            raise ArithmeticError(
                f'{name} propagation lost positivity or normalisation: '
                f'min {probability.min():.2e}, sum {probability.sum():.12f}')
    end = np.maximum(end, 0.0)
    mean = np.maximum(mean, 0.0)
    return end / end.sum(), mean / mean.sum(), drift


def relaxation_spectrum(generator, equilibrium) -> np.ndarray:
    """Generator eigenvalues via the sqrt(pi) symmetrisation.

    Detailed balance makes sqrt(pi_i/pi_j) Q_ij = sqrt(Q_ij Q_ji), so the
    symmetrised matrix is bounded by max|Q| however wide pi is: the
    eigenvalues are reliable even where reconstructing p from them is not.
    """
    root = np.sqrt(equilibrium)
    symmetric = root[:, None] * generator / root[None, :]
    return np.linalg.eigvalsh((symmetric + symmetric.T) / 2.0)


def marginal(probability: np.ndarray) -> dict:
    return {kind: float(probability[block(kind)].sum()) for kind in CONTACTS}


def bound_indices() -> np.ndarray:
    return np.r_[block('face'), block('tip')]


def survival_and_lifetime(generator, equilibrium, initial_bound, window_s):
    """Bound-manifold statistics with the `free` label made absorbing.

    Returns the probability that a bound pair is still the same bound pair
    after `window_s` and its mean lifetime.  Two different solvers, because
    k_off spans ~15 decades in `superlattice` mode:

      lifetime  exact linear solve of Q_bb tau = -1.  Reconstructing tau from
                1/lambda would divide by eigenvalues that sit only one or two
                decades above the eigenvalue noise floor eps*max|Q| ~ 1e-8/s.
      survival  spectral, from the sqrt(pi) symmetrisation of Q_bb (bounded by
                max|Q| however wide pi is).  The slow modes contribute
                exp(-lambda t) ~ 1 here, so the same noise floor is harmless.

    `initial_bound` must not put weight on states of negligible pi: the
    spectral reconstruction divides by sqrt(pi), so pass either the
    conditional equilibrium or a delta on a deep state.
    """
    index = bound_indices()
    sub = generator[np.ix_(index, index)]
    start = np.asarray(initial_bound, dtype=float)
    if start.sum() <= 0:
        raise ValueError('Initial bound distribution is empty')
    start = start / start.sum()
    lifetime = float(start @ np.linalg.solve(sub, -np.ones(len(index))))
    if not np.isfinite(lifetime) or lifetime <= 0:
        raise ArithmeticError('Bound manifold does not absorb: escape is blocked')
    root = np.sqrt(equilibrium[index])
    symmetric = root[:, None] * sub / root[None, :]
    symmetric = (symmetric + symmetric.T) / 2.0
    values, vectors = eigh(symmetric)
    if np.max(values) > 0.0:
        raise ArithmeticError('Bound manifold has a growing mode')
    coefficients = (start / root) @ vectors
    survival = float(((coefficients * np.exp(values * window_s)) @ vectors.T * root).sum())
    return min(max(survival, 0.0), 1.0), lifetime


def deepest_bound_start(energy_J) -> np.ndarray:
    """Delta on the single most stable bound configuration.

    This is the state a persistent superlattice bond actually occupies, and
    it is the only initial condition for which "how long does the pair last"
    has a sharp answer.  A flux-weighted (nascent) average is NOT used: every
    encounter enters the bound manifold, including the 3/4 of dipole
    configurations that are weakly bound or repulsive and leave again within
    1/nu_esc ~ 10 ns, so the flux-weighted mean lifetime is just 1/nu_esc and
    carries no information about the deep bonds.
    """
    index = bound_indices()
    start = np.zeros(len(index))
    start[int(np.argmin(np.asarray(energy_J)[index]))] = 1.0
    return start


def regime_label(tau_N_s: float, window_s: float) -> str:
    if tau_N_s < 0.1 * window_s:
        return 'fast Neel and Brownian'
    if tau_N_s > 10.0 * window_s:
        return 'Neel blocked, Brownian only'
    return 'Neel crossover'


def compute(temperature_K: float, binding: Binding, structures: dict,
            window_s: float = WINDOW_S):
    """Populations, energies, lifetimes and timescales at one temperature."""
    params = geometry.PARAMS
    model = joint_model(temperature_K, binding, structures)
    kbt = model['kbt_J']
    equilibrium = model['equilibrium']
    # Start colloidally dispersed with a uniform dipole prior: the inherited
    # convention of an independent reset at every temperature, no history.
    initial = np.zeros(192)
    initial[block('free')] = 1.0 / 64.0
    window_end, window_mean, drift = propagate(model['generator'], initial, window_s)

    tau_N = params.attempt_time_s * np.exp(
        min(params.zfc_fc_activation_barrier_J / kbt, 700.0))
    deepest = deepest_bound_start(model['energy_J'])
    lifetimes = {}
    for name, channels in (('joint', ('neel', 'registry')),
                           ('neel_blocked', ('registry',)),
                           ('rotation_blocked', ('neel',)),
                           ('direct_only', ())):
        generator = (model['generator'] if name == 'joint' else
                     joint_model(temperature_K, binding, structures, channels)['generator'])
        lifetimes[name] = survival_and_lifetime(generator, equilibrium, deepest, window_s)
    equilibrated = survival_and_lifetime(
        model['generator'], equilibrium, equilibrium[bound_indices()], window_s)
    spectrum = relaxation_spectrum(model['generator'], equilibrium)

    face, tip = structures['geometry']['face'], structures['geometry']['tip']
    free_separation_m = (1.0 / model['number_density_per_m3']) ** (1.0 / 3.0)
    row = dict(assembly=binding.assembly, temperature_K=temperature_K,
               window_s=window_s, volume_fraction=binding.volume_fraction,
               escape_length_nm=binding.escape_length_nm,
               site_volume_nm3=model['site_volume_m3'] * 1e27,
               number_density_per_m3=model['number_density_per_m3'],
               alpha_face=model['alpha']['face'], alpha_tip=model['alpha']['tip'],
               kBT_J=kbt)
    for label, probability in (('eq', equilibrium), ('window', window_mean),
                               ('end', window_end)):
        populations = marginal(probability)
        bound = populations['face'] + populations['tip']
        for kind in CONTACTS:
            row[f'p_{kind}_{label}'] = populations[kind]
        row[f'p_bound_{label}'] = bound
        row[f'U_total_per_NC_{label}_J'] = float(probability @ model['report_J'])
        row[f'U_bound_conditional_{label}_J'] = (
            float(probability @ model['report_J']) / bound if bound > 0 else np.nan)
    row.update(
        tau_N_isolated_s=tau_N, tau_B_free_s=model['tau_B_s'],
        encounter_time_s=1.0 / (COORDINATION['face'] * model['encounter_per_s']),
        escape_attempt_time_s=1.0 / model['escape_per_s'],
        slowest_relaxation_s=-1.0 / spectrum[-2],
        strongest_bond_per_NC_J=float(np.min(model['report_J'])),
        pair_lifetime_deepest_s=lifetimes['joint'][1],
        pair_survival_deepest=lifetimes['joint'][0],
        pair_lifetime_neel_blocked_s=lifetimes['neel_blocked'][1],
        pair_survival_neel_blocked=lifetimes['neel_blocked'][0],
        pair_lifetime_rotation_blocked_s=lifetimes['rotation_blocked'][1],
        pair_survival_rotation_blocked=lifetimes['rotation_blocked'][0],
        pair_lifetime_direct_escape_s=lifetimes['direct_only'][1],
        pair_lifetime_equilibrated_s=equilibrated[1],
        pair_survival_equilibrated=equilibrated[0],
        regime=regime_label(tau_N, window_s),
        persistent_SL_fraction_eq=row['p_bound_eq'] * lifetimes['joint'][0],
        persistent_SL_fraction_window=row['p_bound_window'] * lifetimes['joint'][0],
        vdW_face_J=face['vdw_J'], vdW_tip_J=tip['vdw_J'],
        face_center_distance_nm=face['center_distance_m'] * 1e9,
        tip_center_distance_nm=tip['center_distance_m'] * 1e9,
        mean_free_separation_nm=free_separation_m * 1e9,
        free_state_dipole_diagnostic_J=abs(face['prefactor_J'])
        * (face['center_distance_m'] / free_separation_m) ** 3,
        detailed_balance_relative_error=model['detailed_balance_relative_error'],
        propagator_normalisation_drift=drift)
    for key, value in list(row.items()):
        if key.endswith('_J'):
            row[key[:-2] + '_kBT'] = value / kbt
    return row, equilibrium, window_mean


def equilibrium_populations(temperature_K, binding, structures) -> dict:
    """Closed-form colloid/SL split: p_c/p_free = alpha_c <exp(-U/kBT)>.

    Independent of the master equation and of every rate, so it is the
    reference the 192-state propagator must reproduce at a long window.
    """
    kbt = Boltzmann * temperature_K
    density = number_density_per_m3(binding.volume_fraction)
    volume = site_volume_m3(binding.escape_length_nm)
    ratios, energies = {}, {}
    for kind in BOUND:
        geom = structures['geometry'][kind]
        alpha = COORDINATION[kind] * volume * density
        log_terms = -geom['energy_J'] / kbt
        ratios[kind] = alpha * float(np.exp(logsumexp(log_terms) - np.log(64.0)))
        conditional = np.exp(log_terms - logsumexp(log_terms))
        energies[kind] = float(conditional @ (geom['report_factor'] * geom['energy_J']))
    total = sum(ratios.values())
    row = dict(assembly=binding.assembly, temperature_K=temperature_K,
               volume_fraction=binding.volume_fraction,
               escape_length_nm=binding.escape_length_nm,
               p_free=1.0 / (1.0 + total), p_bound=total / (1.0 + total), kBT_J=kbt)
    for kind in BOUND:
        row[f'p_{kind}'] = ratios[kind] / (1.0 + total)
    row['U_total_per_NC_J'] = sum(
        ratios[kind] * energies[kind] for kind in BOUND) / (1.0 + total)
    row['U_total_per_NC_kBT'] = row['U_total_per_NC_J'] / kbt
    return row


# --------------------------------------------------------------------------
# face vs tip, resolved by dipole manifold and by orientational entropy
# --------------------------------------------------------------------------
# The 192-state model above averages the dipole label over all 64 lab <111>
# pairs and gives every contact the same orientational phase-space volume.
# Both choices suppress tip-to-tip, and both are wrong in the regime the
# experiment probes.  This section resolves the competition explicitly.
#
# THE EXACT CANCELLATION.  For a <100> bond every <111> easy axis sits at the
# magic angle to it, (s.n)^2 = 1/3 exactly, so for PARALLEL moments
#     U_dd(face) = C (s1.s2 - 3 (s1.n)(s2.n)) = C (1 - 1) = 0
# identically, for all eight parallel states.  Face-to-face therefore gets NO
# dipolar binding from parallel moments at all; its -4C/3 well requires
# s1.s2 = -1/3, i.e. non-parallel moments.  For a <111> bond the two
# head-to-tail states have s.n = +/-1 and give U_dd = -2C, which is also the
# global minimum over all 64 states.  So the face/tip ordering INVERTS with
# the dipole manifold, and which manifold applies is physics, not bookkeeping.
#
# WHICH MANIFOLD, AND WHEN.  Throughout this section the CONTACT is fixed
# (face-to-face or tip-to-tip, chosen by the caller) and the DIPOLE is not:
# it always carries a Boltzmann distribution.  What the blocking temperature
# switches is WHICH STATES the dipole can reach, not whether it is thermal.
#
#   T > T_B   'free'     Neel is fast, a moment reorients inside its own cube,
#                        so all 64 lab <111> pairs are mutually reachable and
#                        the distribution is Boltzmann over all of them.
#   T < T_B   'blocked'  Neel is frozen: a moment is locked to its own crystal
#                        and the ONLY way the lab pair state can change is a
#                        contact-preserving rigid rotation.  Those rotations
#                        generate a cyclic subgroup about the bond, so the 64
#                        states break into disjoint ORBITS and a pair
#                        thermalises inside its own orbit only.
#
# The blocked case is therefore NOT "parallel moments".  It is a Boltzmann
# distribution inside each orbit, quenched-averaged over orbits, with orbit
# weights inherited from the uniform distribution the particles carry while
# still colloidally dispersed (where the dipolar coupling is negligible), so
# w_orbit = |orbit| / 64.  Low-energy states still dominate inside each orbit,
# which is why the head-to-tail tip state carries so much weight.
#
#   face, <100> bond   the 4-fold turn preserves the sign of s.n, so the two
#                      per-particle orbits are the 4 states with s_x > 0 and
#                      the 4 with s_x < 0: FOUR pair orbits of 16.  Two of
#                      them (s1.n and s2.n of the same sign) contain the
#                      -4C/3 ground state, two are purely repulsive.
#   tip, <111> bond    the 3-fold turn leaves +[111] and -[111] invariant, so
#                      the per-particle orbits have sizes 1, 3, 3, 1: SIXTEEN
#                      pair orbits.  The -2C head-to-tail state is an orbit of
#                      its own and cannot be left or entered by rotation.
#
# 'parallel' is retained only as a limiting-case diagnostic; it is the hard
# restriction to s1 = s2 and is NOT used for the T < T_B branch.
#
# THE ORIENTATIONAL ENTROPY.  A tip contact is a point contact: the vertex
# must point at the partner, but rotation about the bond is free.  A face
# contact must additionally register in twist.  Using the inherited
# rotational_entropy_model.orientation_fraction (24 face registries with a
# +/-delta twist window, 8 tip registries with free twist), the tilt cone
# cancels exactly between the two and the tip advantage is
#     2 kBT ln(f_tip / f_face) = 2 kBT ln(pi / (3 delta))
# which is 8.2, 6.8, 5.0 and 3.6 kBT for delta = 1, 2, 5 and 10 degrees.  Only
# delta survives, so this term is robust against the tilt tolerance even
# though neither tolerance is measured.
MANIFOLDS = ('free', 'blocked')
BLOCKING_TEMPERATURE_K = geometry.PARAMS.blocking_temperature_K


def manifold_for_temperature(temperature_K: float,
                             blocking_K: float = BLOCKING_TEMPERATURE_K) -> str:
    """'free' above the blocking temperature, 'blocked' below it."""
    return 'free' if temperature_K > blocking_K else 'blocked'


def single_particle_orbits(states, moves) -> list[np.ndarray]:
    """Orbits of the 8 easy axes under the group generated by `moves`."""
    lookup = {tuple(np.sign(state).astype(int)): index
              for index, state in enumerate(states)}
    matrices = [rotation_matrix(axis, angle) for axis, angle in moves]
    orbits, assigned = [], set()
    for start in range(len(states)):
        if start in assigned:
            continue
        orbit, stack = set(), [start]
        while stack:
            index = stack.pop()
            if index in orbit:
                continue
            orbit.add(index)
            for matrix in matrices:
                moved = lookup[tuple(np.sign(matrix @ states[index]).astype(int))]
                if moved not in orbit:
                    stack.append(moved)
        assigned |= orbit
        orbits.append(np.array(sorted(orbit)))
    return orbits


# WHAT A BLOCKED CUBE CAN STILL DO, taken from the SAME orientational model
# that supplies the entropy term.  rotational_entropy_model.orientation_fraction
# grants the two contacts very different freedom, and the blocked dynamics have
# to agree with it or the entropy and the kinetics contradict each other:
#
#   tip   8 cap                 -> all 8 body diagonals count as available and
#                                  the twist is FREE over the full 2 pi.  A
#                                  vertex contact is a point contact: choosing
#                                  which vertex touches is itself a rigid
#                                  rotation that maps tip-contact to
#                                  tip-contact, so a blocked tip cube can turn
#                                  its frozen moment onto the bond and reach
#                                  the head-to-tail state.  Group = the whole
#                                  cube group, ONE orbit of 64: a tip contact
#                                  is never really blocked.
#   face  24 cap delta/pi       -> the twist is confined to +/-delta, a few
#                                  degrees.  A blocked face cube therefore
#                                  cannot perform the 90 deg bond-axis turn
#                                  (that leaves the registry) and cannot roll
#                                  to present a different face without
#                                  unbinding.  Group = identity, 64 orbits of
#                                  one: a blocked face pair keeps whatever
#                                  state it docked in.
#
# 'bondaxis' keeps the intermediate convention used before this consistency
# fix (face C4 about the bond, tip C3 about the bond) for comparison.
BLOCKED_ROTATION = {'tip': 'full', 'face': 'none'}


def manifold_orbits(geom: dict, manifold: str) -> list[np.ndarray]:
    """Pair-state index sets a dipole pair can thermalise within.

    'free'      one orbit of all 64 states; Neel connects everything.
    'blocked'   per-contact, from BLOCKED_ROTATION: one orbit of 64 for tip,
                64 orbits of one for face.
    'bondaxis'  the previous convention: products of the single-particle
                orbits of the bond-axis rotations only.
    'parallel'  limiting-case diagnostic only: the 8 states with s1 = s2.
    """
    if manifold == 'free':
        return [np.arange(64)]
    if manifold == 'parallel':
        return [np.arange(8) * 9]
    if manifold == 'bondaxis':
        orbits = single_particle_orbits(geom['states'], MOVES[geom['kind']])
        return [np.add.outer(8 * first, second).ravel()
                for first in orbits for second in orbits]
    if manifold == 'blocked':
        group = BLOCKED_ROTATION[geom['kind']]
        if group == 'full':
            return [np.arange(64)]
        if group == 'none':
            return [np.array([index]) for index in range(64)]
        if group == 'bondaxis':
            return manifold_orbits(geom, 'bondaxis')
        raise ValueError(f'unknown blocked rotation group {group!r}')
    raise ValueError(f'unknown dipole manifold {manifold!r}')


def dipole_manifold_energies(geom: dict, manifold: str) -> np.ndarray:
    """Energies of every state the manifold admits, orbits concatenated."""
    energies = np.asarray(geom['energy_J'])
    return np.concatenate([energies[orbit]
                           for orbit in manifold_orbits(geom, manifold)])


def twist_entropy_advantage_kBT(twist_halfwidth_deg: float,
                                tilt_deg: float = 2.0) -> float:
    """2 ln(f_tip/f_face) from the inherited orientation_fraction.

    Positive means tip favoured.  Imported rather than re-derived so this
    stays tied to rotational_entropy_model; the tilt cone cancels, leaving
    2 ln(pi / (3 delta)).
    """
    from rotational_entropy_model import orientation_fraction

    face = orientation_fraction('face', tilt_deg, twist_halfwidth_deg)
    tip = orientation_fraction('tip', tilt_deg, twist_halfwidth_deg)
    return 2.0 * float(np.log(tip / face))


def orbit_statistics(geom: dict, manifold: str, kbt: float) -> dict:
    """Per-orbit Boltzmann weights, and the quenched average over orbits.

    Inside an orbit the dipole is thermal; across orbits it is quenched with
    w_orbit = |orbit|/64, the distribution the pair carries in from the
    dispersed state.  `free` has a single orbit, so the quenched average
    degenerates to the ordinary Boltzmann average and nothing double counts.
    """
    energies = np.asarray(geom['energy_J'])
    orbits = manifold_orbits(geom, manifold)
    total = sum(len(orbit) for orbit in orbits)
    records = []
    for orbit in orbits:
        values = energies[orbit]
        log_terms = -values / kbt
        log_mean = float(logsumexp(log_terms) - np.log(len(values)))
        probability = np.exp(log_terms - logsumexp(log_terms))
        records.append(dict(index=orbit, size=len(orbit),
                            weight=len(orbit) / total,
                            energies_J=values,
                            log_mean_boltzmann=log_mean,
                            free_energy_J=-kbt * log_mean,
                            mean_energy_J=float(probability @ values),
                            min_energy_J=float(values.min()),
                            probability=probability))
    # Two averages that answer different questions.  Do not conflate them.
    #   quenched  mean of the orbit free energies: the binding free energy of
    #             ONE pair that is stuck inside its own orbit.
    #   annealed  free energy of the weighted mean Boltzmann factor: what an
    #             ENSEMBLE of such pairs shows in the dilute limit.
    # Because w_orbit = |orbit|/64, sum_orbits w <exp(-U/kBT)>_orbit is
    # identically the mean over all 64 states, so the annealed value is the
    # SAME for every manifold that partitions those states.  That is an
    # identity, not a coincidence: blocking changes which states a given pair
    # can reach, not how many pairs sit at each energy.  The two therefore
    # disagree, and which one applies depends on saturation - see
    # restricted_channel_populations.
    quenched_F = sum(r['weight'] * r['free_energy_J'] for r in records)
    annealed_F = -kbt * float(logsumexp(
        [np.log(r['weight']) + r['log_mean_boltzmann'] for r in records]))
    return dict(orbits=records, states=total,
                quenched_free_energy_J=quenched_F,
                annealed_free_energy_J=annealed_F,
                min_energy_J=min(r['min_energy_J'] for r in records))


# --------------------------------------------------------------------------
# cooling trajectory: a superlattice that assembles above T_B and is then
# stranded by its own frozen moments
# --------------------------------------------------------------------------
# The mechanism, spelled out because it is the one thing in this module that
# is a genuine prediction rather than a reorganisation of inherited numbers:
#
#   Above the freeze-in temperature the bound population is enriched in
#   ATTRACTIVE dipole configurations.  That enrichment does not need Neel: a
#   particle whose frozen moment points the wrong way simply fails to stick,
#   unbinds, rotates freely in solution and re-docks.  Selective binding is a
#   translational process and it is what produces the Boltzmann weighting.
#
#   A particle INSIDE a formed superlattice cannot use that mechanism.  It is
#   surrounded by z neighbours, its twist is registry-locked to +/-delta, and
#   below T_B it cannot flip by Neel either.  So it is stranded with whatever
#   moment direction it happened to freeze with, and its bonds sample the
#   dipole distribution it inherited rather than the optimised one.
#
# What gets inherited is therefore the whole question, and the two limits
# bracket it:
#   'annealed'   the bound Boltzmann distribution at the freeze temperature.
#                Nothing happens on further cooling: the moments were already
#                ordered, and Neel had time to order them while tau_N was
#                still short.  Energy stays flat in J and keeps deepening in
#                kBT units.
#   'dispersed'  the uniform distribution the particles carry while still
#                colloidal, where the dipolar coupling is ~0.1 kBT.  Because
#                sum over the 64 states of U_dd is EXACTLY zero, the mean
#                dipolar binding of such a pair is exactly zero and only the
#                van der Waals term is left.  This is the disassembly branch.
#
# The freeze temperature is NOT T_B by default.  T_B = 250 K is the 100 s
# ZFC/FC definition; what matters here is where tau_N crosses the observation
# window, tau_N(T) = window, which is ~267 K for 20 s.  Both are reported.
def neel_freeze_temperature_K(window_s: float = WINDOW_S) -> float:
    """Temperature at which tau_N equals the observation window."""
    params = geometry.PARAMS
    if window_s <= params.attempt_time_s:
        raise ValueError('window must exceed the attempt time')
    return float(params.zfc_fc_activation_barrier_J
                 / (Boltzmann * np.log(window_s / params.attempt_time_s)))


def neel_blocked_fraction(temperature_K: float, window_s: float = WINDOW_S,
                          size_cv: float = 0.10, order: int = 400) -> float:
    """Fraction of particles that have NOT flipped within the window.

    Lognormal diameters of coefficient of variation `size_cv` with the
    inherited scaling dE ~ D^3, so a 10 % spread in D is a 33 % spread in the
    barrier and, on an exponential, an enormous spread in tau_N:

        b(T) = < exp(-window / tau0 exp(f dE / kBT)) >,  f = (D/Dbar)^3

    Integrated on a fixed 400-node Gauss-Legendre rule over the standard
    normal variate, NOT on the inherited 11-node Gauss-Hermite rule of
    geometry_model.barrier_distribution.  That rule is built for smooth
    integrands; exp(-window/tau) is nearly a step in f and the 11-node result
    is wrong by 15 % at 270 K (0.373 against 0.436).  The inherited rule is
    still used for the inherited figures; this one is checked against
    adaptive quadrature in the tests.
    """
    params = geometry.PARAMS
    if size_cv < 0:
        raise ValueError('size_cv must be nonnegative')
    exponent = params.zfc_fc_activation_barrier_J / (Boltzmann * temperature_K)
    if size_cv == 0.0:
        return float(np.exp(-window_s / (params.attempt_time_s
                                         * np.exp(min(exponent, 700.0)))))
    sigma = np.sqrt(np.log1p(size_cv ** 2))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    variate = 8.0 * nodes
    density = np.exp(-0.5 * variate ** 2) / np.sqrt(2.0 * np.pi) * 8.0 * weights
    factor = np.exp(-0.5 * sigma ** 2 + sigma * variate) ** 3
    tau = params.attempt_time_s * np.exp(np.minimum(factor * exponent, 700.0))
    return float(density @ np.exp(-window_s / tau) / density.sum())


def particle_freeze_temperature_K(barrier_factor, window_s: float = WINDOW_S):
    """T at which tau_N of a particle of this barrier factor equals the window."""
    params = geometry.PARAMS
    return (np.asarray(barrier_factor, dtype=float)
            * params.zfc_fc_activation_barrier_J
            / (Boltzmann * np.log(window_s / params.attempt_time_s)))


def _lognormal_barrier_grid(size_cv: float, order: int = 400):
    """Nodes, weights, and survival function of the (D/Dbar)^3 distribution."""
    from scipy.special import erfc

    nodes, weights = np.polynomial.legendre.leggauss(order)
    variate = 8.0 * nodes
    density = np.exp(-0.5 * variate ** 2) / np.sqrt(2.0 * np.pi) * 8.0 * weights
    density = density / density.sum()
    if size_cv == 0.0:
        return np.ones(1), np.ones(1), np.array([0.5])
    sigma = np.sqrt(np.log1p(size_cv ** 2))
    factor = np.exp(-0.5 * sigma ** 2 + sigma * variate) ** 3
    survival = 0.5 * erfc(variate / np.sqrt(2.0))  # P(f' > f) for each node
    return factor, density, survival


def history_cooling_trajectory(temperatures_K, structures: dict,
                               channel: str = 'face', size_cv: float = 0.10,
                               window_s: float = WINDOW_S,
                               bound_at_freeze=None,
                               binding: Binding | None = None,
                               twist_halfwidth_deg: float = 5.0) -> list[dict]:
    """Cooling with the frozen dipole distribution INHERITED, not uniform.

    Fixes the physical error in `gradual_cooling_trajectory`, which gave every
    frozen pair a uniform dipole distribution.  A pair that froze at
    temperature T_pair inherits the distribution it actually had then, and
    that depends on where it was:

        q(T_pair) = p_bound(T_pair) Boltzmann_bound(T_pair)
                  + (1 - p_bound(T_pair)) uniform

    A pair that froze while already bound keeps its attractive bias and loses
    almost nothing on further cooling.  A pair that froze while still
    colloidally dispersed inherits a uniform distribution, because a free pair
    has essentially no dipolar coupling (~0.1 kBT), and that is the branch
    that disassembles.

    T_pair is set by the SECOND moment to freeze, which is the exact
    consequence of the face identity in `gradual_cooling_trajectory`: one
    mobile moment already recovers the whole equilibrium binding, so the pair
    behaves as annealed until both are blocked.  With independent sizes the
    relevant variable is m = min(f1, f2), whose density is 2 (1-F(f)) p(f), so
    the frozen weight integrates to exactly b^2.

    `bound_at_freeze` may be None (compute p_bound from
    restricted_channel_populations at T_pair), or a float in [0, 1] to pin it.
    """
    if binding is None:
        binding = Binding()
    if bound_at_freeze is not None and not 0.0 <= float(bound_at_freeze) <= 1.0:
        raise ValueError('bound_at_freeze must be None or lie in [0, 1]')
    geom = structures['geometry'][channel]
    energies = np.asarray(geom['energy_J'])
    coordination = COORDINATION[channel]
    uniform = np.full(64, 1.0 / 64.0)
    factor, density, survival = _lognormal_barrier_grid(size_cv)
    freeze_grid = particle_freeze_temperature_K(factor, window_s)
    # Density of m = min(f1, f2): 2 (1 - F(f)) p(f).
    minimum_density = 2.0 * survival * density
    minimum_density = minimum_density / max(minimum_density.sum(), 1e-300)

    def bound_distribution(temperature_K):
        log_terms = -energies / (Boltzmann * temperature_K)
        return np.exp(log_terms - logsumexp(log_terms))

    inherited_energy, inherited_repulsive = {}, {}
    for index, freeze in enumerate(freeze_grid):
        if bound_at_freeze is None:
            fraction = restricted_channel_populations(
                float(freeze), structures, channel, 'free',
                twist_halfwidth_deg, binding)['p_pair']
        else:
            fraction = float(bound_at_freeze)
        distribution = (fraction * bound_distribution(float(freeze))
                        + (1.0 - fraction) * uniform)
        inherited_energy[index] = float(distribution @ energies)
        inherited_repulsive[index] = float(distribution[energies > 0].sum())

    # A pair is frozen once BOTH moments are, i.e. once m = min(f1,f2) exceeds
    # f_min(T).  Integrate the min-density from the top of the grid once and
    # then interpolate at T; selecting grid nodes with a hard mask instead
    # leaves a visible 1-2 K staircase on every curve.
    order = np.argsort(freeze_grid)
    sorted_freeze = freeze_grid[order]
    sorted_density = minimum_density[order]
    energy_grid = np.array([inherited_energy[i] for i in order])
    repulsive_grid = np.array([inherited_repulsive[i] for i in order])
    tail_weight = np.r_[np.cumsum(sorted_density[::-1])[::-1], 0.0]
    tail_energy = np.r_[np.cumsum((sorted_density * energy_grid)[::-1])[::-1], 0.0]
    tail_repulsive = np.r_[
        np.cumsum((sorted_density * repulsive_grid)[::-1])[::-1], 0.0]
    knots = np.r_[sorted_freeze, sorted_freeze[-1] + 1.0]

    rows = []
    for temperature in np.atleast_1d(temperatures_K).astype(float):
        kbt = Boltzmann * temperature
        annealed = bound_distribution(temperature)
        frozen_weight = float(np.interp(temperature, knots, tail_weight))
        if frozen_weight > 1e-12:
            frozen_energy = float(
                np.interp(temperature, knots, tail_energy) / frozen_weight)
            frozen_repulsive = float(
                np.interp(temperature, knots, tail_repulsive) / frozen_weight)
            above = sorted_freeze >= temperature
            mean_freeze = (float(sorted_density[above] @ sorted_freeze[above]
                                 / sorted_density[above].sum())
                           if above.any() else float(sorted_freeze[-1]))
        else:
            frozen_weight = 0.0
            frozen_energy = float(annealed @ energies)
            frozen_repulsive = float(annealed[energies > 0].sum())
            mean_freeze = np.nan
        annealed_energy = float(annealed @ energies)
        row = dict(channel=channel, temperature_K=temperature, size_cv=size_cv,
                   window_s=window_s,
                   bound_at_freeze=('computed' if bound_at_freeze is None
                                    else float(bound_at_freeze)),
                   twist_halfwidth_deg=twist_halfwidth_deg,
                   volume_fraction=binding.volume_fraction,
                   frozen_pair_weight=frozen_weight,
                   mean_pair_freeze_temperature_K=mean_freeze, kBT_J=kbt,
                   U_pair_J=((1.0 - frozen_weight) * annealed_energy
                             + frozen_weight * frozen_energy),
                   U_pair_equilibrium_J=annealed_energy,
                   U_pair_frozen_component_J=frozen_energy,
                   vdW_pair_J=geom['vdw_J'],
                   repulsive_weight=((1.0 - frozen_weight)
                                     * float(annealed[energies > 0].sum())
                                     + frozen_weight * frozen_repulsive))
        row['U_lattice_per_NC_J'] = (coordination / 2.0) * row['U_pair_J']
        row['U_pair_excess_J'] = row['U_pair_J'] - annealed_energy
        for key, value in list(row.items()):
            if key.endswith('_J') and key != 'kBT_J':
                row[key[:-2] + '_kBT'] = value / kbt
        rows.append(row)
    return rows


def gradual_cooling_trajectory(temperatures_K, structures: dict,
                               channel: str = 'face', size_cv: float = 0.10,
                               window_s: float = WINDOW_S) -> list[dict]:
    """Cooling with a GRADUAL freeze-in set by tau_N against a fixed window.

    No step at any temperature: the blocked fraction b(T) rises smoothly
    because tau_N crosses the window at a different temperature for every
    particle size.  A pair is then a three-component mixture, exact rather
    than interpolated:

      both moments still free   (1-b)^2   Boltzmann over all 64 states
      one frozen, one free      2b(1-b)   partner held at a uniformly random
                                          easy axis, the other Boltzmann over
                                          its 8 states
      both frozen               b^2       uniform over 64, mean U_dd exactly 0

    The middle term is what removes the discontinuity of cooling_trajectory:
    a half-frozen pair still recovers part of its dipolar binding, because the
    mobile moment can chase the frozen one.
    """
    geom = structures['geometry'][channel]
    energies = np.asarray(geom['energy_J'])
    matrix = energies.reshape(8, 8)
    coordination = COORDINATION[channel]
    rows = []
    for temperature in np.atleast_1d(temperatures_K).astype(float):
        kbt = Boltzmann * temperature
        blocked = neel_blocked_fraction(temperature, window_s, size_cv)
        log_terms = -energies / kbt
        annealed = np.exp(log_terms - logsumexp(log_terms))
        # One partner frozen at a uniformly random easy axis j, the other
        # Boltzmann over i given j.  Columns of the matrix, normalised.
        #
        # AN EXACT IDENTITY, FACE ONLY: for a <100> bond the conditional
        # partition function Z_j = sum_i exp(-U_ij/kBT) is the SAME for all
        # eight j, so this term equals `annealed` exactly and the mixture
        # collapses to (1 - b^2) U_annealed + b^2 U_frozen.  One mobile moment
        # recovers the whole equilibrium dipolar binding; only a pair with
        # BOTH moments blocked loses it, so the controlling variable is b^2,
        # not b.  It follows from the C4 stabiliser of the bond plus the
        # (i,j) -> (-i,-j) symmetry of a bilinear energy, and it does NOT hold
        # for tip, where Z_j differs by 2.9x between the head-to-tail states
        # and the rest.  The general three-term form is kept for that reason.
        columns = np.exp(-matrix / kbt)
        half = (columns / columns.sum(axis=0, keepdims=True) / 8.0).ravel()
        uniform = np.full(64, 1.0 / 64.0)
        mixture = ((1.0 - blocked) ** 2 * annealed
                   + 2.0 * blocked * (1.0 - blocked) * half
                   + blocked ** 2 * uniform)
        if abs(mixture.sum() - 1.0) > 1e-12:
            raise ArithmeticError('mixture weights do not normalise')
        row = dict(channel=channel, temperature_K=temperature, size_cv=size_cv,
                   window_s=window_s, blocked_fraction=blocked,
                   weight_both_free=(1.0 - blocked) ** 2,
                   weight_half_frozen=2.0 * blocked * (1.0 - blocked),
                   weight_both_frozen=blocked ** 2, kBT_J=kbt,
                   U_pair_J=float(mixture @ energies),
                   U_pair_equilibrium_J=float(annealed @ energies),
                   U_pair_both_frozen_J=float(uniform @ energies),
                   U_pair_half_frozen_J=float(half @ energies * 8.0 / 8.0),
                   vdW_pair_J=geom['vdw_J'],
                   repulsive_weight=float(mixture[energies > 0].sum()),
                   U_lattice_per_NC_J=(coordination / 2.0)
                   * float(mixture @ energies),
                   U_lattice_per_NC_equilibrium_J=(coordination / 2.0)
                   * float(annealed @ energies))
        row['U_pair_excess_J'] = row['U_pair_J'] - row['U_pair_equilibrium_J']
        for key, value in list(row.items()):
            if key.endswith('_J') and key != 'kBT_J':
                row[key[:-2] + '_kBT'] = value / kbt
        rows.append(row)
    return rows


def cooling_trajectory(temperatures_K, structures: dict, channel: str = 'face',
                       inherit: str = 'dispersed', freeze_K: float | None = None,
                       window_s: float = WINDOW_S) -> list[dict]:
    """Mean pair energy of a FIXED contact on cooling, with dipoles freezing.

    Above `freeze_K` the dipole distribution is the bound Boltzmann one at the
    current temperature.  At and below it the distribution is frozen at
    whatever `inherit` specifies and never updates again, so the mean energy
    is constant in joules while the equilibrium reference keeps deepening.
    """
    if inherit not in ('annealed', 'dispersed'):
        raise ValueError("inherit must be 'annealed' or 'dispersed'")
    if freeze_K is None:
        freeze_K = neel_freeze_temperature_K(window_s)
    geom = structures['geometry'][channel]
    energies = np.asarray(geom['energy_J'])
    coordination = COORDINATION[channel]

    def bound_distribution(temperature_K):
        log_terms = -energies / (Boltzmann * temperature_K)
        return np.exp(log_terms - logsumexp(log_terms))

    frozen = (bound_distribution(freeze_K) if inherit == 'annealed'
              else np.full(64, 1.0 / 64.0))
    rows = []
    for temperature in np.atleast_1d(temperatures_K).astype(float):
        kbt = Boltzmann * temperature
        equilibrium = bound_distribution(temperature)
        blocked = temperature <= freeze_K
        probability = frozen if blocked else equilibrium
        row = dict(channel=channel, inherit=inherit, temperature_K=temperature,
                   freeze_temperature_K=freeze_K, window_s=window_s,
                   blocked=bool(blocked), kBT_J=kbt,
                   U_pair_J=float(probability @ energies),
                   U_pair_equilibrium_J=float(equilibrium @ energies),
                   vdW_pair_J=geom['vdw_J'],
                   repulsive_weight=float(probability[energies > 0].sum()),
                   U_per_NC_J=0.5 * float(probability @ energies),
                   U_lattice_per_NC_J=(coordination / 2.0)
                   * float(probability @ energies),
                   U_lattice_per_NC_equilibrium_J=(coordination / 2.0)
                   * float(equilibrium @ energies))
        row['U_pair_excess_J'] = row['U_pair_J'] - row['U_pair_equilibrium_J']
        for key, value in list(row.items()):
            if key.endswith('_J') and key != 'kBT_J':
                row[key[:-2] + '_kBT'] = value / kbt
        rows.append(row)
    return rows


def plot_history_cooling(rows, out: Path) -> None:
    """The frozen dipole distribution is inherited, and where it froze decides."""
    style()
    figure, panels = plt.subplots(3, 1, figsize=(7.8, 13.6), sharex=True,
                                  height_ratios=[1, 2, 1])
    weight, energy, repulsive = panels
    branches = (('1.0', '#287A96', 'froze while BOUND (keeps its bias)'),
                ('computed', '#B54535', 'computed $p_{bound}(T_{freeze})$'),
                ('0.0', '#DE9945', 'froze while DISPERSED (uniform)'))
    for axes in panels:
        experimental_spans(axes)
        axes.axvline(BLOCKING_TEMPERATURE_K, color='0.45', lw=1.6,
                     ls=(0, (2, 2)), zorder=2)
    for tag, color, label in branches:
        data = sorted([r for r in rows if str(r['bound_at_freeze']) == tag],
                      key=lambda r: r['temperature_K'])
        if not data:
            continue
        grid = [r['temperature_K'] for r in data]
        energy.plot(grid, [r['U_pair_kBT'] for r in data], color=color, lw=2.9,
                    label=label)
        repulsive.plot(grid, [r['repulsive_weight'] for r in data], color=color,
                       lw=2.6, label=label)
    reference = sorted([r for r in rows if str(r['bound_at_freeze']) == '0.0'],
                       key=lambda r: r['temperature_K'])
    grid = [r['temperature_K'] for r in reference]
    weight.plot(grid, [r['frozen_pair_weight'] for r in reference],
                color='#3C3C8C', lw=2.6,
                label=r'$b^2$: both moments frozen')
    energy.plot(grid, [r['U_pair_equilibrium_kBT'] for r in reference],
                color='black', lw=1.8, ls=(0, (5, 2)),
                label='equilibrium reference (never freezes)')
    energy.plot(grid, [r['vdW_pair_kBT'] for r in reference], color='0.45',
                lw=1.8, label=r'$E_{vdW}^{face,pair}$ (uniform-dipole limit)')
    energy.axhline(0, color='0.6', lw=.8)
    weight.plot([], [], color='0.45', lw=1.6, ls=(0, (2, 2)),
                label=rf'$T_B$ = {BLOCKING_TEMPERATURE_K:g} K')
    weight.set(ylabel='Frozen pair\nweight  $b^2$', ylim=(-0.03, 1.03),
               xlim=(200, 300))
    energy.set(ylabel=r'Face pair energy / $k_BT$', xlim=(200, 300))
    repulsive.set(xlabel='Temperature (K)', xlim=(200, 300),
                  ylabel='Weight of net-\nrepulsive bonds')
    weight.set_title('Inherited, not uniform: where the pair froze decides\n'
                     rf'$CV$ = {rows[0]["size_cv"] * 100:.0f} %, '
                     rf'{rows[0]["window_s"]:g} s window, '
                     rf'$\phi$ = {rows[0]["volume_fraction"]:g}', fontsize=13)
    handles, labels = [], []
    for axes in (weight, energy):
        for handle, label in zip(*axes.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    repulsive.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -.30),
                     frameon=False, fontsize=11)
    figure.subplots_adjust(left=.17, right=.97, top=.93, bottom=.26, hspace=.10)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'history_cooling.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def plot_gradual_cooling(rows, out: Path) -> None:
    """Gradual freeze-in: no step anywhere, set by tau_N against the window."""
    style()
    figure, panels = plt.subplots(3, 1, figsize=(7.8, 13.6), sharex=True,
                                  height_ratios=[1, 2, 1])
    blocked, energy, repulsive = panels
    cvs = sorted({r['size_cv'] for r in rows})
    colors = ('#3C3C8C', '#287A96', '#B54535', '#DE9945')
    for axes in panels:
        experimental_spans(axes)
        axes.axvline(BLOCKING_TEMPERATURE_K, color='0.45', lw=1.6,
                     ls=(0, (2, 2)), zorder=2)
    for cv, color in zip(cvs, colors):
        data = sorted([r for r in rows if r['size_cv'] == cv],
                      key=lambda r: r['temperature_K'])
        grid = [r['temperature_K'] for r in data]
        label = ('monodisperse' if cv == 0 else
                 rf'lognormal $CV$ = {cv * 100:.0f} %')
        blocked.plot(grid, [r['blocked_fraction'] for r in data], color=color,
                     lw=2.6, label=label)
        energy.plot(grid, [r['U_pair_kBT'] for r in data], color=color, lw=2.8,
                    label=label)
        repulsive.plot(grid, [r['repulsive_weight'] for r in data], color=color,
                       lw=2.6, label=label)
    reference = sorted([r for r in rows if r['size_cv'] == cvs[0]],
                       key=lambda r: r['temperature_K'])
    grid = [r['temperature_K'] for r in reference]
    energy.plot(grid, [r['U_pair_equilibrium_kBT'] for r in reference],
                color='black', lw=1.8, ls=(0, (5, 2)),
                label='equilibrium reference (dipole never freezes)')
    energy.plot(grid, [r['vdW_pair_kBT'] for r in reference], color='0.45',
                lw=1.8, label=r'$E_{vdW}^{face,pair}$ (fully frozen limit)')
    energy.axhline(0, color='0.6', lw=.8)
    blocked.plot([], [], color='0.45', lw=1.6, ls=(0, (2, 2)),
                 label=rf'$T_B$ = {BLOCKING_TEMPERATURE_K:g} K')
    blocked.set(ylabel='Blocked fraction\n' r'$b(T)$', ylim=(-0.03, 1.03),
                xlim=(200, 300))
    energy.set(ylabel=r'Face pair energy / $k_BT$', xlim=(200, 300))
    repulsive.set(xlabel='Temperature (K)', xlim=(200, 300),
                  ylabel='Weight of net-\nrepulsive bonds')
    blocked.set_title('Gradual dipole freeze-in on cooling, fixed face contact\n'
                      rf'$\tau_N$ against a {rows[0]["window_s"]:g} s window, '
                      'lognormal sizes', fontsize=13)
    handles, labels = [], []
    for axes in (blocked, energy):
        for handle, label in zip(*axes.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    repulsive.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -.30),
                     frameon=False, fontsize=11)
    figure.subplots_adjust(left=.17, right=.97, top=.93, bottom=.26, hspace=.10)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'gradual_cooling.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def plot_cooling_trajectory(rows, out: Path) -> None:
    """Panel-D style: fixed face contact, cooled from 300 K, dipoles freezing."""
    style()
    freeze = rows[0]['freeze_temperature_K']
    figure, panels = plt.subplots(2, 1, figsize=(7.8, 11.0), sharex=True,
                                  height_ratios=[2, 1])
    energy, repulsive = panels
    for axes in panels:
        experimental_spans(axes)
        axes.axvline(freeze, color='#7D2E68', lw=2.0, ls=(0, (6, 3)), zorder=2)
        axes.axvline(BLOCKING_TEMPERATURE_K, color='0.45', lw=1.6,
                     ls=(0, (2, 2)), zorder=2)
    styles = {'dispersed': ('#B54535', 2.9, '-',
                            'frozen random (inherited from dispersed)'),
              'annealed': ('#287A96', 2.4, '-',
                           'frozen ordered (inherited from bound)')}
    for inherit, (color, width, dash, label) in styles.items():
        data = sorted([r for r in rows if r['inherit'] == inherit],
                      key=lambda r: r['temperature_K'])
        if not data:
            continue
        grid = [r['temperature_K'] for r in data]
        energy.plot(grid, [r['U_pair_kBT'] for r in data], color=color,
                    lw=width, ls=dash, label=label)
        repulsive.plot(grid, [r['repulsive_weight'] for r in data], color=color,
                       lw=width, ls=dash, label=label)
    reference = sorted([r for r in rows if r['inherit'] == 'dispersed'],
                       key=lambda r: r['temperature_K'])
    grid = [r['temperature_K'] for r in reference]
    energy.plot(grid, [r['U_pair_equilibrium_kBT'] for r in reference],
                color='black', lw=1.8, ls=(0, (5, 2)),
                label='equilibrium reference (dipole never freezes)')
    energy.plot(grid, [r['vdW_pair_kBT'] for r in reference], color='#DE9945',
                lw=1.8, label=r'$E_{vdW}^{face,pair}$ (dipole averages to zero)')
    energy.axhline(0, color='0.6', lw=.8)
    energy.plot([], [], color='#7D2E68', lw=2.0, ls=(0, (6, 3)),
                label=rf'freeze-in, $\tau_N$ = {rows[0]["window_s"]:g} s '
                      rf'({freeze:.0f} K)')
    energy.plot([], [], color='0.45', lw=1.6, ls=(0, (2, 2)),
                label=rf'$T_B$ = {BLOCKING_TEMPERATURE_K:g} K (100 s ZFC/FC)')
    energy.set(ylabel=r'Face pair energy / $k_BT$', xlim=(200, 300))
    repulsive.set(xlabel='Temperature (K)', xlim=(200, 300), ylim=(-0.03, 0.55),
                  ylabel='Weight of net-repulsive\nbonds')
    energy.set_title('Cooling a fixed face-to-face contact from 300 K\n'
                     'dipoles freeze when $\\tau_N$ exceeds the window',
                     fontsize=13)
    energy.legend(loc='upper left', bbox_to_anchor=(0, -1.30), frameon=False,
                  fontsize=11)
    figure.subplots_adjust(left=.17, right=.97, top=.93, bottom=.30, hspace=.10)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'cooling_trajectory.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def face_tip_competition(temperature_K: float, structures: dict,
                         manifold: str = 'free',
                         twist_halfwidth_deg: float = 5.0,
                         tilt_deg: float = 2.0) -> dict:
    """Free-energy difference between a tip and a face contact, per pair.

    Every column is (tip - face), so NEGATIVE favours tip-to-tip.  The three
    contributions are reported separately and never silently summed:
      dvdW            inherited 4^3 voxel Hamaker at the inherited gap
      dU_dd_min       deepest dipole state of the manifold
      dF_dip_annealed  ensemble dipole free energy; governs the dilute-limit
                       populations, and is manifold independent by the orbit
                       identity in orbit_statistics
      dF_dip_quenched  free energy of ONE pair locked in its own orbit; this
                       is where blocking actually bites
      dF_twist         orientational entropy, -2 ln(f_tip/f_face)
    dF_total uses the annealed branch, so it always equals -ln of the odds
    ratio p_tip/p_face in the dilute limit; dF_total_quenched is its
    single-pair counterpart.
    """
    kbt = Boltzmann * temperature_K
    terms = {}
    for kind in BOUND:
        geom = structures['geometry'][kind]
        statistics = orbit_statistics(geom, manifold, kbt)
        minimum = statistics['min_energy_J']
        terms[kind] = dict(
            vdw=geom['vdw_J'] / kbt,
            u_min=minimum / kbt,
            dipole_min=(minimum - geom['vdw_J']) / kbt,
            f_dip=statistics['annealed_free_energy_J'] / kbt,
            f_dip_quenched=statistics['quenched_free_energy_J'] / kbt,
            degeneracy=len(statistics['orbits']))
    twist = twist_entropy_advantage_kBT(twist_halfwidth_deg, tilt_deg)
    row = dict(temperature_K=temperature_K, dipole_manifold=manifold,
               manifold_states=len(dipole_manifold_energies(
                   structures['geometry']['face'], manifold)),
               twist_halfwidth_deg=twist_halfwidth_deg, tilt_deg=tilt_deg,
               kBT_J=kbt)
    for kind in BOUND:
        for key, value in terms[kind].items():
            row[f'{kind}_{key}' + ('' if key == 'degeneracy' else '_kBT')] = value
    row['d_vdW_kBT'] = terms['tip']['vdw'] - terms['face']['vdw']
    row['d_U_min_kBT'] = terms['tip']['u_min'] - terms['face']['u_min']
    row['d_dipole_min_kBT'] = terms['tip']['dipole_min'] - terms['face']['dipole_min']
    row['d_F_dip_kBT'] = terms['tip']['f_dip'] - terms['face']['f_dip']
    row['d_F_dip_quenched_kBT'] = (terms['tip']['f_dip_quenched']
                                   - terms['face']['f_dip_quenched'])
    row['d_F_twist_kBT'] = -twist
    row['d_F_energy_only_kBT'] = row['d_F_dip_kBT']
    row['d_F_total_kBT'] = row['d_F_dip_kBT'] + row['d_F_twist_kBT']
    row['d_F_total_quenched_kBT'] = (row['d_F_dip_quenched_kBT']
                                     + row['d_F_twist_kBT'])
    row['preferred_energy_only'] = 'tip' if row['d_F_dip_kBT'] < 0 else 'face'
    row['preferred_with_entropy'] = 'tip' if row['d_F_total_kBT'] < 0 else 'face'
    row['preferred_single_pair'] = ('tip' if row['d_F_total_quenched_kBT'] < 0
                                    else 'face')
    for key in ('d_vdW', 'd_U_min', 'd_dipole_min', 'd_F_dip',
                'd_F_dip_quenched', 'd_F_twist', 'd_F_energy_only',
                'd_F_total', 'd_F_total_quenched'):
        row[key + '_J'] = row[key + '_kBT'] * kbt
    return row


def twist_registry_weight(kind: str, twist_halfwidth_deg: float) -> float:
    """Orientational weight of a contact RELATIVE to tip-to-tip.

    orientation_fraction is z_c x angular measure: 24 cap delta/pi for face
    (6 normals x 4 twist registries x a +/-delta window) against 8 cap for tip
    (8 vertices, twist free).  The ratio is 3 delta/pi per particle, squared
    for the pair, and the tilt cone `cap` cancels.

    The ABSOLUTE orientational volume of a contact is not measured; it stays
    absorbed in v_site, calibrated so the tip channel reproduces the baseline
    alpha_tip = z_tip v_site n of the 192-state model.  Only the tilt-free
    face/tip ratio is applied here, so nothing in the comparison depends on
    the unmeasured tilt tolerance.
    """
    if kind == 'tip':
        return 1.0
    if kind == 'face':
        return (3.0 * np.deg2rad(twist_halfwidth_deg) / np.pi) ** 2
    raise ValueError(f'unknown contact {kind!r}')


def restricted_channel_populations(temperature_K: float, structures: dict,
                                   channel: str, manifold: str,
                                   twist_halfwidth_deg: float,
                                   binding: Binding) -> dict:
    """Pair fraction and total system energy when ONLY `channel` may form.

    Closed form, not the propagator: the 192-state run established that the
    colloid/SL partition is fully equilibrated on the 20 s window at these
    concentrations (window_scan.csv), so the Boltzmann split is exact here.

    Each orbit binds on its own, because a quenched pair cannot leave it:

        p_pair = sum_orbits w_orbit  r_orbit / (1 + r_orbit)
        r_orbit = alpha_c <exp(-U/kBT)>_orbit,  alpha_c = z_tip v_site n w_c

    With one orbit ('free') this is the ordinary two-state Boltzmann split.
    `U_total_per_NC` is the configuration average over the SAME distribution;
    the free states carry zero energy.
    """
    kbt = Boltzmann * temperature_K
    geom = structures['geometry'][channel]
    weight = twist_registry_weight(channel, twist_halfwidth_deg)
    alpha = (COORDINATION['tip'] * site_volume_m3(binding.escape_length_nm)
             * number_density_per_m3(binding.volume_fraction) * weight)
    statistics = orbit_statistics(geom, manifold, kbt)
    fraction, energy_J = 0.0, 0.0
    for orbit in statistics['orbits']:
        ratio = alpha * float(np.exp(orbit['log_mean_boltzmann']))
        bound = ratio / (1.0 + ratio)
        fraction += orbit['weight'] * bound
        energy_J += (orbit['weight'] * bound * geom['report_factor']
                     * orbit['mean_energy_J'])
    bound_J = energy_J / fraction if fraction > 0 else np.nan
    row = dict(assembly=structures['assembly'], channel=channel,
               dipole_manifold=manifold, temperature_K=temperature_K,
               twist_halfwidth_deg=twist_halfwidth_deg,
               volume_fraction=binding.volume_fraction,
               twist_registry_weight=weight,
               twist_free_energy_cost_kBT=-float(np.log(weight)),
               orbit_count=len(statistics['orbits']),
               alpha=alpha, p_pair=fraction, p_free=1.0 - fraction,
               kBT_J=kbt, vdW_pair_J=geom['vdw_J'],
               U_bound_conditional_per_NC_J=bound_J,
               U_total_per_NC_J=energy_J)
    for key, value in list(row.items()):
        if key.endswith('_J') and key != 'kBT_J':
            row[key[:-2] + '_kBT'] = value / kbt
    return row


def plot_restricted_channels(rows, out: Path, assembly: str) -> None:
    """One figure per twist tolerance, in the style of the panel-D figure.

    Rows: pair fraction (log, it spans decades), the total system energy, and
    the bound-pair energy on its own axis.  The last is the direct analogue of
    the published panel-D curves; the middle one is the new quantity and is
    p_pair x (bound energy), so it is far smaller wherever pairing is rare and
    would be invisible if the two shared an axis.

    ONE set of curves, not two: the dipole manifold switches at T_B, from
    `free` above (Neel reorients the moment inside the cube) to `blocked`
    below (only contact-preserving rotation is left, so the pair thermalises
    inside its own orbit).  The dashed continuations show what each branch
    would have given on the wrong side of T_B, so the size of the switch is
    visible rather than hidden in the kink.
    """
    style()
    data = [r for r in rows if r['assembly'] == assembly]
    for twist in sorted({r['twist_halfwidth_deg'] for r in data}):
        figure, panels = plt.subplots(3, 1, figsize=(7.8, 13.8), sharex=True)
        subset = [r for r in data if r['twist_halfwidth_deg'] == twist]
        temperatures = sorted({r['temperature_K'] for r in subset})
        switched = {t: manifold_for_temperature(t) for t in temperatures}

        def series(channel, field, manifold=None):
            picked = []
            for temperature in temperatures:
                want = manifold or switched[temperature]
                picked += [r[field] for r in subset
                           if r['channel'] == channel
                           and r['temperature_K'] == temperature
                           and r['dipole_manifold'] == want]
            return picked

        fraction, total, bound = panels
        for axes in panels:
            experimental_spans(axes)
            axes.axvline(BLOCKING_TEMPERATURE_K, color='#7D2E68', lw=2.0,
                         ls=(0, (6, 3)), zorder=2)
        for channel, label in (('face', 'face-to-face only'),
                               ('tip', 'tip-to-tip only')):
            fraction.semilogy(temperatures,
                              np.maximum(series(channel, 'p_pair'), 1e-12),
                              color=COLORS[channel], lw=3.0, label=label)
            total.plot(temperatures, series(channel, 'U_total_per_NC_kBT'),
                       color=COLORS[channel], lw=3.0, label=label)
            bound.plot(temperatures, series(channel, 'U_bound_conditional_per_NC_kBT'),
                       color=COLORS[channel], lw=3.0,
                       label=label.replace(' only', ', bound pairs only'))
            for manifold, dash in (('free', (0, (2, 2))), ('blocked', (0, (5, 2)))):
                keep = [index for index, t in enumerate(temperatures)
                        if switched[t] != manifold]
                if not keep:
                    continue
                grid = [temperatures[i] for i in keep]
                fraction.semilogy(
                    grid, np.maximum([series(channel, 'p_pair', manifold)[i]
                                      for i in keep], 1e-12),
                    color=COLORS[channel], lw=1.3, ls=dash, alpha=.75)
                total.plot(grid, [series(channel, 'U_total_per_NC_kBT', manifold)[i]
                                  for i in keep],
                           color=COLORS[channel], lw=1.3, ls=dash, alpha=.75)
        bound.plot(temperatures, series('face', 'vdW_pair_kBT'),
                   color='#287A96', lw=1.8, label=r'$E_{vdW}^{face,pair}$')
        bound.axhline(1.0, color='black', lw=1.8, label=r'$k_BT = 1$')
        for axes in (total, bound):
            axes.axhline(0.0, color='0.6', lw=.8)
        fraction.plot([], [], color='0.35', lw=2.0, ls=(0, (6, 3)),
                      label=rf'$T_B$ = {BLOCKING_TEMPERATURE_K:g} K '
                            '(free above, blocked below)')
        fraction.plot([], [], color='0.35', lw=1.3, ls=(0, (2, 2)),
                      label='other manifold, continued for comparison')
        fraction.set(ylim=(1e-7, 2.0), xlim=(200, 300),
                     ylabel='Pair fraction  $p_{pair}$')
        total.set(xlim=(200, 300),
                  ylabel='Total system energy\n' r'per NC / $k_BT$')
        bound.set(xlabel='Temperature (K)', xlim=(200, 300),
                  ylabel='Bound pairs only,\n' r'energy per NC / $k_BT$')
        handles, labels = [], []
        for axes in (fraction, bound):
            for handle, label in zip(*axes.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        bound.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -.16),
                     frameon=False, fontsize=11)
        figure.suptitle(
            f'Restricted single-channel assembly, {assembly}\n'
            rf'face twist tolerance $\delta$ = {twist:g}$\degree$  '
            rf'(costs face {-np.log(twist_registry_weight("face", twist)):.2f} $k_BT$ '
            rf'against tip),  $\phi$ = {data[0]["volume_fraction"]:g}',
            fontsize=14)
        figure.subplots_adjust(left=.17, right=.97, top=.93, bottom=.26,
                               hspace=.10)
        stem = f'restricted_channels_{assembly}_twist{twist:g}deg'
        for extension in ('png', 'pdf'):
            figure.savefig(out / f'{stem}.{extension}', dpi=300, bbox_inches='tight')
        plt.close(figure)


def plot_face_tip_competition(rows, out: Path) -> None:
    """Where tip-to-tip wins, with the manifold switched at T_B.

    Left: the ensemble branch, which the orbit identity makes continuous
    across T_B.  Right: the single-pair branch, where the switch bites - a
    frozen face pair keeps only its van der Waals binding while a frozen tip
    pair can still turn its moment onto the bond.
    """
    style()
    twists = sorted({r['twist_halfwidth_deg'] for r in rows})
    temperatures = sorted({r['temperature_K'] for r in rows})
    switched = {t: manifold_for_temperature(t) for t in temperatures}
    figure, panels = plt.subplots(1, 2, figsize=(12.6, 6.8), sharey=True)
    fields = (('d_F_dip_kBT', 'd_F_total_kBT', 'ensemble of pairs'),
              ('d_F_dip_quenched_kBT', 'd_F_total_quenched_kBT', 'one pair, quenched'))
    for axes, (dipole_field, total_field, title) in zip(panels, fields):
        experimental_spans(axes)
        axes.axvline(BLOCKING_TEMPERATURE_K, color='#7D2E68', lw=2.0,
                     ls=(0, (6, 3)), zorder=2)

        def series(field, twist=None):
            values = []
            for temperature in temperatures:
                values += [r[field] for r in rows
                           if r['temperature_K'] == temperature
                           and r['dipole_manifold'] == switched[temperature]
                           and (twist is None
                                or r['twist_halfwidth_deg'] == twist)]
            return values

        axes.plot(temperatures, series(dipole_field, twists[0]), color='black',
                  lw=2.8, label='vdW + dipole only')
        for twist, color in zip(twists, ('#7D2E68', '#B54535', '#DE9945', '#287A96')):
            axes.plot(temperatures, series(total_field, twist), color=color,
                      lw=2.2, label=rf'+ twist entropy, $\delta$ = {twist:g}$\degree$')
        axes.axhline(0, color='0.3', lw=1.2)
        axes.set(xlabel='Temperature (K)', xlim=(200, 300), title=title)
        axes.title.set_fontsize(13)
    panels[0].set_ylabel(r'$(F_{tip}-F_{face})\,/\,k_BT$')
    panels[0].text(.03, .03, 'below 0: tip-to-tip favoured',
                   transform=panels[0].transAxes, ha='left', va='bottom',
                   fontsize=11)
    panels[0].plot([], [], color='#7D2E68', lw=2.0, ls=(0, (6, 3)),
                   label=rf'$T_B$ = {BLOCKING_TEMPERATURE_K:g} K')
    handles, labels = panels[0].get_legend_handles_labels()
    panels[1].legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -.13),
                     frameon=False, fontsize=11)
    figure.suptitle('Face vs tip with the dipole manifold switched at $T_B$\n'
                    'free above (Neel reorients the moment), blocked below '
                    '(only rigid rotation)', fontsize=13)
    figure.subplots_adjust(left=.09, right=.98, top=.84, bottom=.36, wspace=.06)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'face_tip_competition.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_csv(path: Path, rows) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style() -> None:
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 13,
                         'axes.labelsize': 16, 'axes.linewidth': 1.8,
                         'axes.grid': False, 'pdf.fonttype': 42})


def experimental_spans(axes) -> None:
    axes.axvspan(233.15, 293.15, color='#EEEEEE', zorder=0,
                 label='Experimental range')
    axes.axvspan(253.15, 273.15, color='#F4EAA2', alpha=.8, zorder=1,
                 label='Transient aggregation (experiment)')


def plot_populations(rows, out: Path, assembly: str) -> None:
    style()
    data = [r for r in rows if r['assembly'] == assembly]
    temperatures = [r['temperature_K'] for r in data]
    figure, axes = plt.subplots(figsize=(7.3, 9.2))
    axes.set_box_aspect(1)
    experimental_spans(axes)
    for kind, label in (('free', 'colloidal (free)'), ('face', 'SL, face-to-face'),
                        ('tip', 'SL, tip-to-tip')):
        axes.plot(temperatures, [r[f'p_{kind}_eq'] for r in data],
                  color=COLORS[kind], lw=2.6, label=label + ', equilibrium')
        axes.plot(temperatures, [r[f'p_{kind}_window'] for r in data],
                  color=COLORS[kind], lw=1.6, ls='--',
                  label=label + f", {data[0]['window_s']:g} s mean")
    axes.plot(temperatures, [r['persistent_SL_fraction_eq'] for r in data],
              color='black', lw=1.8, ls=':',
              label='SL surviving the whole window')
    axes.set(xlabel='Temperature (K)', ylabel='Configuration probability',
             xlim=(200, 300), ylim=(-0.03, 1.03),
             title=f'Colloid / SL configuration populations\n{assembly}, '
                   rf"$\phi$ = {data[0]['volume_fraction']:g}")
    axes.title.set_fontsize(14)
    axes.legend(loc='upper left', bbox_to_anchor=(0, -.15), frameon=False, fontsize=11)
    figure.subplots_adjust(left=.17, right=.96, top=.90, bottom=.40)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'populations_{assembly}.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def plot_total_energy(rows, out: Path, assembly: str) -> None:
    style()
    data = [r for r in rows if r['assembly'] == assembly]
    temperatures = [r['temperature_K'] for r in data]
    for unit in ('kBT', 'J'):
        figure, axes = plt.subplots(figsize=(7.3, 9.2))
        axes.set_box_aspect(1)
        experimental_spans(axes)
        axes.plot(temperatures, [r[f'U_total_per_NC_eq_{unit}'] for r in data],
                  color='#3C3C8C', lw=2.8, label='configuration average, equilibrium')
        axes.plot(temperatures, [r[f'U_total_per_NC_window_{unit}'] for r in data],
                  color='#3C3C8C', lw=1.8, ls='--',
                  label=f"configuration average, {data[0]['window_s']:g} s mean")
        axes.plot(temperatures, [r[f'U_bound_conditional_eq_{unit}'] for r in data],
                  color=COLORS['face'], lw=2.0,
                  label='conditional on being bound')
        axes.plot(temperatures, [r[f'vdW_face_{unit}'] for r in data],
                  color=COLORS['free'], lw=1.8, label='face vdW (pair, separate)')
        axes.plot(temperatures, [1.0 if unit == 'kBT' else r['kBT_J'] for r in data],
                  color='black', lw=1.5, label=r'$k_BT$ (reference)')
        axes.axhline(0, color='0.6', lw=.8)
        axes.set(xlabel='Temperature (K)',
                 ylabel=r'Energy per NC / $k_BT$' if unit == 'kBT' else 'Energy per NC (J)',
                 xlim=(200, 300),
                 title='Equivalent total system energy\n'
                       f"{assembly}, configuration averaged over contact and dipole")
        axes.title.set_fontsize(14)
        if unit == 'J':
            axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes.legend(loc='upper left', bbox_to_anchor=(0, -.15), frameon=False, fontsize=11)
        figure.subplots_adjust(left=.19, right=.96, top=.90, bottom=.36)
        for extension in ('png', 'pdf'):
            figure.savefig(out / f'total_energy_{assembly}_{unit}.{extension}',
                           dpi=300, bbox_inches='tight')
        plt.close(figure)


def plot_timescales(rows, out: Path, assembly: str) -> None:
    style()
    data = [r for r in rows if r['assembly'] == assembly]
    temperatures = [r['temperature_K'] for r in data]
    figure, axes = plt.subplots(figsize=(7.3, 9.2))
    axes.set_box_aspect(1)
    experimental_spans(axes)
    series = (('tau_N_isolated_s', r'$\tau_N$ (isolated Neel)', '#B54535', 2.6, '-'),
              ('tau_B_free_s', r'$\tau_B$ (free Brownian)', '#DE9945', 2.6, '-'),
              ('encounter_time_s', 'encounter time', '#287A96', 2.0, '-'),
              ('pair_lifetime_deepest_s', 'strongest bond, both channels',
               'black', 2.6, '-'),
              ('pair_lifetime_neel_blocked_s', 'strongest bond, Neel blocked',
               'black', 1.8, '--'),
              ('pair_lifetime_rotation_blocked_s', 'strongest bond, rotation blocked',
               '0.45', 1.8, ':'),
              ('pair_lifetime_direct_escape_s', 'strongest bond, direct escape only',
               '#7D2E68', 1.6, '-.'))
    for field, label, color, width, dash in series:
        axes.semilogy(temperatures, [r[field] for r in data], color=color,
                      lw=width, ls=dash, label=label)
    axes.axhline(data[0]['window_s'], color='#7D2E68', lw=1.6,
                 label=f"{data[0]['window_s']:g} s observation window")
    axes.set(xlabel='Temperature (K)', ylabel='Time (s)', xlim=(200, 300),
             title='Relaxation, encounter and pair-lifetime clocks\n'
                   f'{assembly}')
    axes.title.set_fontsize(14)
    axes.legend(loc='upper left', bbox_to_anchor=(0, -.15), frameon=False, fontsize=11)
    figure.subplots_adjust(left=.19, right=.96, top=.90, bottom=.40)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'timescales_{assembly}.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def plot_density(densities, out: Path, assembly: str) -> None:
    """Dipole-configuration probability density inside each contact label."""
    style()
    selected = sorted(densities)
    figure, panels = plt.subplots(len(selected), 1, figsize=(7.6, 2.6 * len(selected)),
                                  sharex=True)
    panels = np.atleast_1d(panels)
    index = np.arange(64)
    for axes, temperature in zip(panels, selected):
        probability = densities[temperature]
        for kind in CONTACTS:
            values = probability[block(kind)]
            axes.plot(index, values / max(values.sum(), 1e-300), color=COLORS[kind],
                      lw=1.6, label=f'{kind} (p = {values.sum():.3f})')
        axes.set_ylabel(f'{temperature:g} K\nP(dipole | contact)')
        axes.legend(fontsize=9, frameon=False, ncol=3)
    panels[-1].set_xlabel('Dipole pair state index  (8 x spin1 + spin2)')
    panels[0].set_title(f'Configuration probability density, {assembly}\n'
                        'equilibrium, conditional on the contact label',
                        fontsize=13)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'configuration_density_{assembly}.{extension}', dpi=300)
    plt.close(figure)


def plot_persistence(rows, barrier_scan, out: Path, assembly: str) -> None:
    """The regime figure: does the bond outlive the observation window."""
    style()
    data = [r for r in rows if r['assembly'] == assembly]
    temperatures = [r['temperature_K'] for r in data]
    figure, axes = plt.subplots(figsize=(7.3, 9.2))
    axes.set_box_aspect(1)
    experimental_spans(axes)
    axes.plot(temperatures, [r['p_bound_eq'] for r in data], color=COLORS['face'],
              lw=2.6, label='bound population (thermodynamic)')
    axes.plot(temperatures, [r['pair_survival_deepest'] for r in data], color='black',
              lw=2.8, label=f"bond surviving {data[0]['window_s']:g} s (kinetic)")
    barriers = sorted({r['contact_barrier_kBT'] for r in barrier_scan} - {0.0})
    for barrier, dash in zip(barriers, ('--', ':', '-.')):
        scan = sorted([r for r in barrier_scan if r['assembly'] == assembly
                       and r['contact_barrier_kBT'] == barrier],
                      key=lambda r: r['temperature_K'])
        axes.plot([r['temperature_K'] for r in scan],
                  [r['pair_survival_deepest'] for r in scan], color='0.40',
                  lw=1.7, ls=dash,
                  label=f'survival, +{barrier:g} $k_BT$ rotation barrier')
    axes.plot(temperatures, [r['persistent_SL_fraction_eq'] for r in data],
              color=COLORS['free'], lw=2.0,
              label='persistent SL fraction (product)')
    axes.set(xlabel='Temperature (K)', ylabel='Probability', xlim=(200, 300),
             ylim=(-0.03, 1.03),
             title='Thermodynamic occupancy vs kinetic persistence\n'
                   f'{assembly}, strongest-bond configuration')
    axes.title.set_fontsize(14)
    axes.legend(loc='upper left', bbox_to_anchor=(0, -.15), frameon=False, fontsize=11)
    figure.subplots_adjust(left=.17, right=.96, top=.90, bottom=.40)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'persistence_{assembly}.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def half_crossing(data, field) -> float:
    """Highest temperature at which `field` is still >= 1/2, by interpolation."""
    ordered = sorted(data, key=lambda r: r['temperature_K'])
    temperatures = [r['temperature_K'] for r in ordered]
    values = [r[field] for r in ordered]
    for left in range(len(values) - 1, 0, -1):
        low, high = values[left - 1], values[left]
        if (low - 0.5) * (high - 0.5) <= 0 and low != high:
            fraction = (0.5 - low) / (high - low)
            return temperatures[left - 1] + fraction * (
                temperatures[left] - temperatures[left - 1])
    return float('nan')


def competition_lines(competition) -> list[str]:
    """Tabulate the face/tip competition and name the branch that inverts it."""
    lines = ['--- face vs tip: (F_tip - F_face), negative favours tip ---',
             '  The <111> easy axis sits at the magic angle to a <100> bond, so',
             '  U_dd(face) is identically ZERO for parallel moments; the face well',
             '  needs s1.s2 = -1/3.  A <111> bond gives -2C head-to-tail.',
             '']
    header = (f"{'manifold':>10}{'T/K':>6}{'d_vdW':>8}{'d_Udd_min':>11}"
              f"{'dF_dip_ens':>12}{'dF_dip_1pair':>14}{'d_F_twist':>11}"
              f"{'dF_tot_ens':>12}{'dF_tot_1pair':>14}")
    for manifold in MANIFOLDS:
        lines.append(header)
        for row in competition:
            if (row['dipole_manifold'] != manifold
                    or row['twist_halfwidth_deg'] != 5.0
                    or row['temperature_K'] not in (200.0, 250.0, 300.0)):
                continue
            lines.append(
                f"{manifold:>10}{row['temperature_K']:>6.0f}"
                f"{row['d_vdW_kBT']:>8.2f}{row['d_dipole_min_kBT']:>11.2f}"
                f"{row['d_F_dip_kBT']:>12.2f}{row['d_F_dip_quenched_kBT']:>14.2f}"
                f"{row['d_F_twist_kBT']:>11.2f}"
                f"{row['d_F_total_kBT']:>12.2f}"
                f"{row['d_F_total_quenched_kBT']:>14.2f}")
        lines.append('')
    lines.append('  twist-entropy advantage 2 ln(pi/(3 delta)) for tip; '
                 'the tilt cone cancels:')
    for twist in sorted({r['twist_halfwidth_deg'] for r in competition}):
        lines.append(f'    delta = {twist:>4.0f} deg   '
                     f'{twist_entropy_advantage_kBT(twist):.2f} kBT')
    lines.append('')
    lines.append('  With the manifold SWITCHED at T_B (free above, blocked '
                 'below), which contact')
    lines.append('  is preferred over 200-300 K:')
    for twist in sorted({r['twist_halfwidth_deg'] for r in competition}):
        picked = [r for r in competition
                  if r['twist_halfwidth_deg'] == twist
                  and r['dipole_manifold'] == manifold_for_temperature(
                      r['temperature_K'])]
        for field, label in (('d_F_total_kBT', 'ensemble '),
                             ('d_F_total_quenched_kBT', 'one pair')):
            tip = sorted(r['temperature_K'] for r in picked if r[field] < 0)
            span = (f'{min(tip):.0f}-{max(tip):.0f} K' if tip else 'nowhere')
            lines.append(f'    delta = {twist:>4.0f} deg  {label}: tip at {span}')
    return lines


def summary_lines(rows, sweep) -> list[str]:
    lines = ['Configuration-averaged colloid/SL populations', '=' * 46, '']
    for assembly in ('dimer', 'superlattice'):
        data = [r for r in rows if r['assembly'] == assembly]
        if not data:
            continue
        lines.append(f'--- assembly = {assembly} '
                     f"(phi = {data[0]['volume_fraction']:g}, "
                     f"v_site = {data[0]['site_volume_nm3']:.1f} nm^3) ---")
        header = (f"{'T/K':>6}{'regime':>30}{'p_free':>9}{'p_face':>9}{'p_tip':>9}"
                  f"{'U/NC kBT':>10}{'tau_N/s':>10}{'tau_bond/s':>12}{'survive':>9}")
        lines.append(header)
        for row in data:
            if row['temperature_K'] not in (200.0, 225.0, 250.0, 265.0, 275.0, 300.0):
                continue
            lines.append(
                f"{row['temperature_K']:>6.0f}{row['regime']:>30}"
                f"{row['p_free_eq']:>9.4f}{row['p_face_eq']:>9.4f}{row['p_tip_eq']:>9.4f}"
                f"{row['U_total_per_NC_eq_kBT']:>10.3f}"
                f"{row['tau_N_isolated_s']:>10.2e}"
                f"{row['pair_lifetime_deepest_s']:>12.2e}"
                f"{row['pair_survival_deepest']:>9.4f}")
        lines.append(f"  thermodynamic: p_bound = 1/2 at T = "
                     f"{half_crossing(data, 'p_bound_eq'):.1f} K")
        lines.append(f"  kinetic:       {data[0]['window_s']:g} s bond survival = 1/2 at T = "
                     f"{half_crossing(data, 'pair_survival_deepest'):.1f} K")
        lines.append(f"  window mean vs equilibrium, max |dp_bound| = "
                     f"{max(abs(r['p_bound_window'] - r['p_bound_eq']) for r in data):.2e}")
        lines.append('')
    lines.append('--- volume-fraction sensitivity of the thermodynamic crossover ---')
    lines.append('  (closed form on a 120-700 K grid; alpha_c is linear in phi)')
    for assembly in ('dimer', 'superlattice'):
        for phi in sorted({r['volume_fraction'] for r in sweep}):
            data = [r for r in sweep if r['assembly'] == assembly
                    and r['volume_fraction'] == phi]
            lines.append(f"  {assembly:>13}  phi = {phi:<8g}  "
                         f"T_50 = {half_crossing(data, 'p_bound'):.1f} K")
    return lines


def restricted_lines(restricted) -> list[str]:
    """Single-channel pair fraction and total energy, per twist tolerance."""
    lines = ['--- restricted single-channel assembly (dimer, only one contact '
             'allowed) ---',
             '  alpha_c = z_tip v_site n w_c(delta); w_tip = 1 and '
             'w_face = (3 delta/pi)^2,',
             '  the tilt-free part of the face/tip orientational ratio.', '']
    lines.append(f"{'delta':>6}{'manifold':>10}{'T/K':>6}{'p_face':>11}{'p_tip':>11}"
                 f"{'p_tip/p_face':>14}{'U_face':>9}{'U_tip':>9}")
    for twist in sorted({r['twist_halfwidth_deg'] for r in restricted}):
        for manifold in MANIFOLDS:
            for temperature in (200.0, 250.0, 300.0):
                pick = {r['channel']: r for r in restricted
                        if r['assembly'] == 'dimer'
                        and r['twist_halfwidth_deg'] == twist
                        and r['dipole_manifold'] == manifold
                        and r['temperature_K'] == temperature}
                if len(pick) != 2:
                    continue
                face, tip = pick['face'], pick['tip']
                lines.append(
                    f"{twist:>6.0f}{manifold:>10}{temperature:>6.0f}"
                    f"{face['p_pair']:>11.2e}{tip['p_pair']:>11.2e}"
                    f"{tip['p_pair'] / face['p_pair']:>14.1f}"
                    f"{face['U_total_per_NC_kBT']:>9.3f}"
                    f"{tip['U_total_per_NC_kBT']:>9.3f}")
    lines.append('')
    lines.append('  The odds ratio p_tip/p_face equals exp(-dF_total) of the '
                 'competition table')
    lines.append('  above, so the two analyses are the same statement in '
                 'different variables.')
    return lines


def reading_lines(rows) -> list[str]:
    """State the conclusions the numbers support, and the ones they do not."""
    dimer = [r for r in rows if r['assembly'] == 'dimer']
    lattice = [r for r in rows if r['assembly'] == 'superlattice']
    window = dimer[0]['window_s']
    return [
        'HOW TO READ THIS',
        '-' * 16,
        '1. Tip-to-tip never competes.  The inherited rounded-cube geometry '
        'puts the',
        f"   tip centres at {dimer[0]['tip_center_distance_nm']:.1f} nm against "
        f"{dimer[0]['face_center_distance_nm']:.1f} nm for face, so the tip vdW is",
        f"   {dimer[0]['vdW_tip_kBT']:.2f} kBT against {dimer[0]['vdW_face_kBT']:.2f} kBT "
        'and p_tip stays below 0.2 % at every',
        '   temperature.  The colloid/SL competition is face against free, only.',
        '',
        '2. The colloid/SL POPULATIONS are equilibrated on the observation '
        'window.',
        f"   Encounters take ~{dimer[-1]['encounter_time_s']:.0e} s at phi = "
        f"{dimer[0]['volume_fraction']:g}, so the {window:g} s mean and the",
        '   equilibrium split agree to ~1e-6 everywhere.  Detailed balance then '
        'makes',
        '   the split the same function of T whether Neel or Brownian rotation '
        'is the',
        '   channel that equilibrates the dipole label.  Regime 1, 2 and 3 do '
        'NOT',
        '   differ in their populations; they differ in the bond LIFETIME below.',
        '',
        '3. One bond is never enough.  In dimer mode the deepest bond is only '
        f"{min(r['strongest_bond_per_NC_kBT'] for r in dimer):.1f} kBT",
        '   per NC and the diffusive escape prefactor is ~1e8 /s, so no pair '
        f'survives {window:g} s',
        '   at any temperature in 200-300 K.  An isolated pair is a rapidly '
        'exchanging',
        '   contact, not a structure, even where 90 % of pairs are bound at any '
        'instant.',
        '',
        f"4. A superlattice site ({min(r['strongest_bond_per_NC_kBT'] for r in lattice):.0f} "
        f"to {max(r['strongest_bond_per_NC_kBT'] for r in lattice):.0f} kBT per NC) is where the "
        'regimes separate:',
        f"     300 K  tau_bond = {next(r['pair_lifetime_deepest_s'] for r in lattice if r['temperature_K'] == 300):.1f} s "
        f"< {window:g} s window, survival "
        f"{next(r['pair_survival_deepest'] for r in lattice if r['temperature_K'] == 300):.3f}"
        '  -> dissolves while observed',
        f"     250 K  tau_bond = {next(r['pair_lifetime_deepest_s'] for r in lattice if r['temperature_K'] == 250):.0f} s, survival "
        f"{next(r['pair_survival_deepest'] for r in lattice if r['temperature_K'] == 250):.3f}"
        '           -> persists',
        f"     200 K  tau_bond = {next(r['pair_lifetime_deepest_s'] for r in lattice if r['temperature_K'] == 200):.1e} s, survival "
        f"{next(r['pair_survival_deepest'] for r in lattice if r['temperature_K'] == 200):.3f}"
        '        -> effectively frozen',
        f"   The {window:g} s survival passes 1/2 at "
        f"{half_crossing(lattice, 'pair_survival_deepest'):.0f} K and the dimer population",
        f"   passes 1/2 at {half_crossing(dimer, 'p_bound_eq'):.0f} K.  Both land inside the "
        'experimentally',
        '   flagged 253-273 K transient-aggregation band; neither was fitted to '
        'it.',
        '',
        '5. WHICH relaxation dissolves the bond depends on the contact barrier, '
        'which',
        '   the inherited model sets to zero as an optimistic accessibility '
        'baseline.',
        '   At zero barrier the contact-preserving Brownian rotation dominates '
        'and Neel',
        '   is a ~2x modifier; at +10 kBT the rotation channel is shut and the '
        'lifetime',
        '   becomes Neel limited.  See contact_barrier_scan.csv.  The user-facing',
        '   statement "fast Neel breaks the pair up" is therefore the '
        'BARRIER-LIMITED',
        '   branch of this model, not the zero-barrier default.',
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-s', type=float, default=WINDOW_S)
    parser.add_argument('--volume-fraction', type=float, default=1.0e-2)
    parser.add_argument('--escape-length-nm', type=float, default=0.5)
    parser.add_argument('--mobility', type=float, default=1.0)
    parser.add_argument('--contact-barrier-kBT', type=float, default=0.0)
    parser.add_argument('--temperature-step', type=float, default=1.0)
    parser.add_argument('--tilt-deg', type=float, default=2.0,
                        help='face/tip cage tilt tolerance; cancels in the '
                             'tip-minus-face difference')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    arguments = parser.parse_args()
    if arguments.window_s <= 0 or not 0 < arguments.temperature_step <= 25:
        parser.error('Window must be positive and the grid spacing at most 25 K')
    out = arguments.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    temperatures = np.unique(np.r_[np.arange(200.0, 300.0 + 1e-9,
                                             arguments.temperature_step),
                                   250.0, 300.0])
    coarse = np.arange(200.0, 300.0 + 1e-9, 5.0)
    rows, probabilities, sweep, barrier_scan, window_scan = [], [], [], [], []
    densities = {}
    dimer_structures = None
    structures_by_mode = {}
    for assembly in ('dimer', 'superlattice'):
        structures = build_structures(assembly)
        structures_by_mode[assembly] = structures
        if assembly == 'dimer':
            dimer_structures = structures
        binding = Binding(volume_fraction=arguments.volume_fraction,
                          escape_length_nm=arguments.escape_length_nm,
                          mobility=arguments.mobility,
                          contact_barrier_kBT=arguments.contact_barrier_kBT,
                          assembly=assembly)
        density_snapshot = {}
        report_J = per_state_report_energy(structures)
        for temperature in temperatures:
            row, equilibrium, window_mean = compute(
                float(temperature), binding, structures, arguments.window_s)
            reference = equilibrium_populations(float(temperature), binding, structures)
            row['closed_form_p_bound_eq'] = reference['p_bound']
            row['closed_form_p_bound_error'] = abs(reference['p_bound'] - row['p_bound_eq'])
            rows.append(row)
            for state in range(192):
                probabilities.append(dict(
                    assembly=assembly, temperature_K=float(temperature),
                    state_index=state, contact=CONTACTS[state // 64],
                    spin1_index=(state % 64) // 8, spin2_index=state % 8,
                    energy_per_NC_J=report_J[state], p_equilibrium=equilibrium[state],
                    p_window_mean=window_mean[state]))
            if float(temperature) in (200.0, 265.0, 300.0):
                density_snapshot[float(temperature)] = equilibrium
        densities[assembly] = density_snapshot
        # The closed form costs nothing, so the phi sweep runs on a wider grid:
        # alpha_c is linear in phi, so the crossover shifts logarithmically and
        # leaves the inherited 200-300 K window entirely at phi = 1e-3 or 1e-1.
        for phi in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1):
            scan = Binding(volume_fraction=phi,
                           escape_length_nm=arguments.escape_length_nm,
                           mobility=arguments.mobility,
                           contact_barrier_kBT=arguments.contact_barrier_kBT,
                           assembly=assembly)
            for temperature in np.arange(120.0, 700.0 + 1e-9, 2.0):
                sweep.append(equilibrium_populations(float(temperature), scan, structures))
        # Is bond breaking controlled by Neel or by contact-preserving rotation?
        # The inherited zero contact barrier is an explicitly optimistic
        # accessibility baseline; raising it suppresses the rotation channel
        # only, so the comparison isolates the mechanism.
        for barrier in (0.0, 5.0, 10.0):
            probe = Binding(volume_fraction=arguments.volume_fraction,
                            escape_length_nm=arguments.escape_length_nm,
                            mobility=arguments.mobility,
                            contact_barrier_kBT=barrier, assembly=assembly)
            for temperature in coarse:
                row = compute(float(temperature), probe, structures,
                              arguments.window_s)[0]
                barrier_scan.append({key: row[key] for key in (
                    'assembly', 'temperature_K', 'regime',
                    'pair_lifetime_deepest_s', 'pair_survival_deepest',
                    'pair_lifetime_neel_blocked_s', 'pair_lifetime_rotation_blocked_s',
                    'pair_lifetime_direct_escape_s', 'p_bound_eq')}
                    | {'contact_barrier_kBT': barrier})
        # How long does the colloid/SL partition take to equilibrate?  At the
        # default phi the encounter time is microseconds, so the 20 s answer is
        # the equilibrium one; this scan shows where that stops being true.
        for phi in (1.0e-4, 1.0e-2):
            probe = Binding(volume_fraction=phi,
                            escape_length_nm=arguments.escape_length_nm,
                            mobility=arguments.mobility,
                            contact_barrier_kBT=arguments.contact_barrier_kBT,
                            assembly=assembly)
            for window in (1.0e-6, 1.0e-3, 1.0, 20.0, 150.0):
                for temperature in (200.0, 250.0, 300.0):
                    row = compute(temperature, probe, structures, window)[0]
                    window_scan.append(dict(
                        assembly=assembly, volume_fraction=phi, window_s=window,
                        temperature_K=temperature, encounter_time_s=row['encounter_time_s'],
                        p_bound_window=row['p_bound_window'], p_bound_end=row['p_bound_end'],
                        p_bound_eq=row['p_bound_eq'],
                        U_total_per_NC_window_kBT=row['U_total_per_NC_window_kBT'],
                        U_total_per_NC_eq_kBT=row['U_total_per_NC_eq_kBT']))
        print(f'Completed {assembly}: {len(temperatures)} temperatures', flush=True)

    write_csv(out / 'populations_and_energies.csv', rows)
    write_csv(out / 'state_probabilities.csv', probabilities)
    write_csv(out / 'volume_fraction_sweep.csv', sweep)
    write_csv(out / 'contact_barrier_scan.csv', barrier_scan)
    write_csv(out / 'window_scan.csv', window_scan)
    # Face vs tip is a pair-geometry question, not an assembly-mode one: the
    # coordination scaling multiplies both contacts, so it is evaluated once
    # on the unscaled dimer geometry.
    competition = [face_tip_competition(float(temperature), dimer_structures,
                                        manifold, twist, arguments.tilt_deg)
                   for manifold in MANIFOLDS + ('bondaxis', 'parallel')
                   for twist in (1.0, 2.0, 5.0, 10.0)
                   for temperature in temperatures]
    write_csv(out / 'face_tip_competition.csv', competition)
    plot_face_tip_competition(competition, out)
    # Restrict the system to ONE contact channel and ask what fraction is
    # paired and what the total energy is, for each twist tolerance.
    restricted = [restricted_channel_populations(
                      float(temperature), structures_by_mode[mode], channel,
                      manifold, twist,
                      Binding(volume_fraction=arguments.volume_fraction,
                              escape_length_nm=arguments.escape_length_nm,
                              mobility=arguments.mobility,
                              contact_barrier_kBT=arguments.contact_barrier_kBT,
                              assembly=mode))
                  for mode in ('dimer', 'superlattice')
                  for channel in BOUND
                  for manifold in MANIFOLDS + ('bondaxis', 'parallel')
                  for twist in (1.0, 2.0, 5.0, 10.0)
                  for temperature in temperatures]
    write_csv(out / 'restricted_channel_populations.csv', restricted)
    # Cooling a fixed face contact from 300 K, dipoles freezing at tau_N =
    # window.  The two inheritance limits bracket what a stranded lattice
    # particle carries.
    cooling = [row for inherit in ('dispersed', 'annealed')
               for row in cooling_trajectory(temperatures, dimer_structures,
                                             'face', inherit,
                                             window_s=arguments.window_s)]
    write_csv(out / 'cooling_trajectory.csv', cooling)
    plot_cooling_trajectory(cooling, out)
    # The realistic version: tau_N crosses the fixed window at a different
    # temperature for every particle size, so the freeze-in is gradual.
    gradual = [row for cv in (0.0, 0.05, 0.10, 0.15)
               for row in gradual_cooling_trajectory(
                   temperatures, dimer_structures, 'face', cv,
                   arguments.window_s)]
    write_csv(out / 'gradual_cooling.csv', gradual)
    plot_gradual_cooling(gradual, out)
    # The frozen distribution is INHERITED, not uniform: a pair that froze
    # while already bound keeps its attractive bias.
    history = [row for bound in (1.0, None, 0.0)
               for row in history_cooling_trajectory(
                   temperatures, dimer_structures, 'face', 0.10,
                   arguments.window_s, bound)]
    write_csv(out / 'history_cooling.csv', history)
    plot_history_cooling(history, out)
    for mode in ('dimer', 'superlattice'):
        plot_restricted_channels(restricted, out, mode)
    for assembly in ('dimer', 'superlattice'):
        plot_populations(rows, out, assembly)
        plot_total_energy(rows, out, assembly)
        plot_timescales(rows, out, assembly)
        plot_density(densities[assembly], out, assembly)
        plot_persistence(rows, barrier_scan, out, assembly)

    lines = summary_lines(rows, sweep)
    lines.append('')
    lines.append('--- rotation-barrier control of the bond lifetime (superlattice) ---')
    lines.append(f"{'T/K':>6}{'barrier/kBT':>13}{'tau_bond/s':>13}"
                 f"{'Neel blocked':>14}{'rotation blocked':>18}{'direct only':>13}")
    for row in barrier_scan:
        if row['assembly'] != 'superlattice' or row['temperature_K'] not in (200.0, 250.0, 300.0):
            continue
        lines.append(f"{row['temperature_K']:>6.0f}{row['contact_barrier_kBT']:>13.0f}"
                     f"{row['pair_lifetime_deepest_s']:>13.2e}"
                     f"{row['pair_lifetime_neel_blocked_s']:>14.2e}"
                     f"{row['pair_lifetime_rotation_blocked_s']:>18.2e}"
                     f"{row['pair_lifetime_direct_escape_s']:>13.2e}")
    lines.append('')
    lines += competition_lines(competition)
    lines.append('')
    lines += restricted_lines(restricted)
    lines.append('')
    lines += reading_lines(rows)
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    config = dict(
        state_space='3 contact labels x 64 lab <111> dipole pairs = 192 states',
        contacts=list(CONTACTS), surface_gap_nm=SURFACE_GAP_NM,
        window_s=arguments.window_s, volume_fraction=arguments.volume_fraction,
        escape_length_nm=arguments.escape_length_nm,
        site_volume_nm3=site_volume_m3(arguments.escape_length_nm) * 1e27,
        vdw_model='INHERITED 4^3 sharp-cube voxel Hamaker sum, unchanged',
        gap_model='INHERITED exact rounded-cube (1.5 nm Minkowski) centre distance',
        coordination_numbers=COORDINATION,
        bound_rotations={'face': '+/-90 deg about the <100> bond',
                         'tip': '+/-120 deg about the <111> bond',
                         'free': 'inherited six <100> quarter turns'},
        rotation_attempt='mobility / (lambda tau_B) per move, lambda from '
                         'sum(R-I) = -lambda (I-P); first-rank decay = 1/tau_B',
        neel_rates='inherited neel_pair_generator barriers, verified identical',
        binding='k_on = z_c k_diff n exp(-U+/kBT), k_off = (k_diff/v_site) '
                'exp(-(-U)+/kBT); detailed balance exact',
        free_state_energy='0; dipolar coupling at the mean free separation '
                          'reported as free_state_dipole_diagnostic',
        initial_condition='uniform over the 64 free-label dipole states, reset '
                          'independently at every temperature',
        assumptions=['phi and l are not measured here',
                     'superlattice mode is a mean-field z_c/2 scaling of a pair model',
                     'no ligand/solvent PMF, no face<->tip conversion path',
                     'constant Ms, CV = 0, inherited effective cubic K'],
        equilibrium_check='closed_form_p_bound_error column; the propagator and '
                          'the analytic Boltzmann split must agree',
        scans=dict(face_tip_competition='free vs parallel dipole manifold x '
                                        'face twist tolerance 1, 2, 5, 10 deg; '
                                        'resolves which contact is preferred',
                   blocked_rotation=BLOCKED_ROTATION,
                   history_cooling='frozen dipole distribution inherited from '
                                   'the pair freeze temperature, weighted by '
                                   'p_bound there; bound / computed / '
                                   'dispersed branches',
                   gradual_cooling='same, but tau_N against a fixed window '
                                   'with lognormal sizes CV = 0, 5, 10, 15 %; '
                                   'three-component pair mixture, no step',
                   cooling_trajectory='fixed face contact cooled from 300 K; '
                                      'dipole frozen at tau_N = window; '
                                      'inherit = dispersed (random) or '
                                      'annealed (ordered)',
                   restricted_channels='only face, or only tip, allowed to '
                                       'form; pair fraction and total energy '
                                       'per twist tolerance and manifold',
                   volume_fraction_sweep='phi = 1e-4 .. 1e-1, closed form',
                   contact_barrier_scan='extra rotation barrier 0, 5, 10 kBT; '
                                        'isolates Neel from rotation control of '
                                        'the bond lifetime',
                   window_scan='1 us .. 150 s at phi = 1e-4 and 1e-2; shows '
                               'where the colloid/SL partition stops being '
                               'equilibrated within the window'))
    (out / 'model_config.json').write_text(json.dumps(config, indent=2, default=str),
                                           encoding='utf-8')
    print('\n'.join(lines))
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
