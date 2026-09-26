"""Re-entrant fluid-fluid binodal for the blocked-dipole nanocube suspension.

WHY THIS IS NOT A SUPERLATTICE CALCULATION
------------------------------------------
The SAXS evidence is a single BROAD peak, not the sharp family a superlattice
would give, with an interparticle spacing of 19 nm for a 16 nm cube.  So the
dense phase is an amorphous condensate, and the right framework is colloidal
fluid-fluid phase separation, not crystallisation.  Every lattice concept used
in `configuration_averaged_assembly` - coordination registry, twist entropy,
tip and edge contacts - is dropped here.

THE MEASURED SPACING CLOSES THE ONE FREE PARAMETER
--------------------------------------------------
19 nm centre-to-centre on a 16 nm cube is a 3 nm surface gap, which is also
the gap every inherited pair energy is evaluated at, so the energies are at
the right separation.  It also leaves each particle only 9.50 nm of half
extent, against a rounded-cube support height of 8.00 nm along <100>, 10.69 nm
along <110> and 12.76 nm along <111>.  Therefore:

  * the dense phase MUST be face-registered; edge and vertex contacts do not
    fit at this spacing;
  * a particle can tilt only +/-15.5 deg before its corner hits a neighbour,
    against the 54.7 deg needed to bring a different <111> onto the bond and
    the 90 deg needed to swap faces.

So inside the condensate the cube cannot reorient.  The orientational sampling
that a free particle gets from Brownian rotation is unavailable there, and the
only remaining way to sample the moment is a Neel flip.

THE MECHANISM
-------------
What drives condensation is not the mean pair energy but the Mayer factor.
exp is convex, so even though the dipolar energy averages to EXACTLY zero over
the 64 lab <111> pair states, the orientational average of exp(-U_dd/kBT) is
large and contributes a real attraction:

    eps_pair(T) = -U_vdW/kBT + ln <exp(-U_dd/kBT)>

That second term is 4.4 kBT at 300 K rising to 7.5 kBT at 200 K, against 1.2
to 1.8 kBT from van der Waals.  It is available only while the moment can be
re-sampled.  In the condensate rotation is blocked, so a bond keeps the term
only while Neel is running; once the moment blocks, that bond is worth its
van der Waals energy alone.  Writing b(T) for the blocked fraction, the
cohesive energy per bond in the dense phase is LINEAR in the blocked weight,
because in a condensate every bond is realised at once rather than sampled:

    eps_dense(T) = -U_vdW/kBT + (1 - b(T)^2) ln <exp(-U_dd/kBT)>

b^2 rather than b: one mobile moment already re-samples the pair, an exact
identity for a <100> bond proved in configuration_averaged_assembly.

eps_dense(T) is NON-MONOTONIC.  Cooling deepens both terms, but it also
freezes moments, and past the freeze-in the loss of the dipolar term wins.
The binodal built on it is therefore re-entrant: its dilute branch has a
minimum, and a horizontal line at the experimental concentration crosses it
TWICE, which is the observed aggregation window.

THE BLOCKING WINDOW IS THE DWELL TIME, NOT THE SAXS FRAME
----------------------------------------------------------
On a stepped ramp the moment population tracks temperature with time constant
tau_N, so what matters is how long the sample sits near each temperature, not
how long the detector integrates.  For the reported protocol - 10 K in 2 min
then 2.5 min of counting - the dwell is 150 to 270 s and the freeze-in falls
at 246 to 241 K.  This is a strong lever: 12 s of dwell would put it at 273 K
and 300 s at 240 K, so the lower edge should move with ramp rate while the
upper edge, a thermodynamic crossing, should not.

FREE ENERGY
-----------
Carnahan-Starling hard spheres plus a mean-field attraction,

    f(eta)/kBT = ln eta + eta(4 - 3 eta)/(1 - eta)^2 - a eta,
    a = (z_max/2) eps_dense / eta_cp,

with coexistence from equal chemical potential and pressure.  This is a
van der Waals level treatment: it gives the topology and the concentration
scale, not quantitative binodal compositions.  z_max and eta_cp are the
close-packed coordination and packing fraction, stated rather than fitted.

NOT INCLUDED
------------
Nucleation barriers and therefore the observed ~10 K hysteresis, which is
rate dependent and so is a lag rather than a boundary; the capillary wall,
which the anisotropic 2D pattern shows is where nucleation actually happens;
and any ramp-rate trajectory.  This module gives the quasi-static boundaries
the hysteresis loops should straddle.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
from scipy.optimize import brentq, fsolve
from scipy.special import logsumexp

import geometry_model as geometry
from configuration_averaged_assembly import (LINKS, SURFACE_GAP_NM,
                                             neel_blocked_fraction)

OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'binodal_phase_diagram'

# Close-packed references for the mean-field attraction; stated, not fitted.
MAX_COORDINATION = 6.0
CLOSE_PACKED_FRACTION = 0.64
# Converged sharp-cube Hamaker value for the same integral the inherited 4^3
# voxel sum evaluates; see the audit in the configuration_averaged_assembly
# report.  Selectable so the sensitivity is explicit.
CONVERGED_VDW_J = -9.156e-21
# 1.25 % inorganic core volume fraction (65 mg/mL Fe3O4), converted to the
# effective fraction the free energy uses via the measured 19 nm spacing.
SAMPLE_VOLUME_FRACTION = 0.0125 * (19.0 / 16.0) ** 3


def pair_energies(vdw: str = 'inherited') -> dict:
    """Inherited face-pair dipolar spectrum and van der Waals energy."""
    params = geometry.PARAMS
    direction = LINKS['face']
    distance = geometry.center_distance_at_gap_m(
        direction, SURFACE_GAP_NM * 1e-9, params)
    dipole_J, prefactor_J, states, link = geometry.easy_axis_pair_energies_J(
        distance * direction, params)
    if vdw == 'inherited':
        vdw_J = geometry.pair_vdw_energy_J(distance * direction, params)
    elif vdw == 'converged':
        vdw_J = CONVERGED_VDW_J
    else:
        raise ValueError("vdw must be 'inherited' or 'converged'")
    return dict(dipole_J=dipole_J, prefactor_J=prefactor_J, vdw_J=vdw_J,
                centre_distance_nm=distance * 1e9, vdw_choice=vdw)


def dipolar_mayer_term(temperature_K: float, energies: dict) -> float:
    """ln <exp(-U_dd/kBT)> over the 64 states.  Positive despite <U_dd> = 0."""
    kbt = Boltzmann * temperature_K
    return float(logsumexp(-np.asarray(energies['dipole_J']) / kbt)
                 - np.log(64.0))


# --------------------------------------------------------------------------
# what a bond keeps when its moments freeze
# --------------------------------------------------------------------------
# The blocked fraction of bonds contributes ZERO only if the frozen moment
# directions are uncorrelated with the bond, which holds when the moment froze
# in the dilute phase: there is no partner there, so it freezes into one of the
# eight body-frame <111> easy axes with equal weight (cubic anisotropy, no
# field), and which face later registers is set by Brownian tumbling.  Summed
# over the 64 lab states U_dd is identically zero, so such a bond keeps nothing.
#
# A moment that freezes while ALREADY BONDED is different: until it blocks it
# samples the Boltzmann distribution over the pair states, which is weighted
# towards the eight head-to-tail states at -7.675 kBT, and it freezes out of
# that biased distribution.  It therefore keeps an attraction.  What limits the
# bias is geometry, not statistics -- see `frustrated_quench_kBT`.
NEIGHBOUR_DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                        (0, -1, 0), (0, 0, 1), (0, 0, -1))


_QUENCH_TOTALS: dict[int, np.ndarray] = {}


def _frozen_neighbour_totals(coordination: int) -> np.ndarray:
    """Energy in joules of each centre axis against every frozen neighbour set.

    Shape (8, 8**z): rows are the centre's easy axes, columns one configuration
    of the z frozen neighbours.  None of it depends on temperature, so it is
    built once per coordination and reused; rebuilding it was most of the cost
    of a cooling sweep.
    """
    if coordination not in _QUENCH_TOTALS:
        params = geometry.PARAMS
        matrices = []
        for vector in NEIGHBOUR_DIRECTIONS[:coordination]:
            direction = np.asarray(vector, float)
            distance = geometry.center_distance_at_gap_m(
                direction, SURFACE_GAP_NM * 1e-9, params)
            energies, _, _, _ = geometry.easy_axis_pair_energies_J(
                distance * direction, params)
            matrices.append(np.asarray(energies).reshape(8, 8))
        index = np.indices((8,) * coordination).reshape(coordination, -1)
        _QUENCH_TOTALS[coordination] = sum(
            matrices[k][:, index[k]] for k in range(coordination))
    return _QUENCH_TOTALS[coordination]


def frustrated_quench_kBT(temperature_K: float, coordination: int = 6) -> float:
    """Attraction per bond kept by a pair that froze inside the condensate.

    One moment serves `coordination` bonds at once and cannot sit in the
    deepest state of all of them, so the retained energy per bond collapses
    with the number of neighbours: 7.51 kBT for an isolated pair, 2.64 kBT at
    z = 6 (250 K).  That frustration, not any averaging, is what keeps the
    frozen contribution small.

    Exhaustive over all 8**z frozen neighbour configurations -- no sampling and
    no seed, so the value is reproducible to machine precision.
    """
    if not 1 <= coordination <= 6:
        raise ValueError('coordination must be between 1 and 6')
    total = _frozen_neighbour_totals(coordination) / (Boltzmann * temperature_K)
    weight = np.exp(-(total - total.min(axis=0)))
    weight /= weight.sum(axis=0)
    return float(-(weight * total).sum(axis=0).mean() / coordination)


def loss_fraction(temperature_K: float, energies: dict,
                  correlated_fraction: float = 0.0,
                  coordination: int = 6) -> float:
    """lambda: the share of the dipolar attraction a fully blocked bond loses.

    `correlated_fraction` (f) is the share of blocked moments that froze while
    already inside the condensate rather than free in the dilute phase.  The
    lever rule bounds it: at phi_eff = 0.0209 against phi_dense ~ 0.31 at most
    6.7 % of particles are in the dense phase at any instant, so f <~ 0.07 if
    exchange is fast compared with the ramp.

        lambda = 1 - f * Q_z / D

    f = 0 gives lambda = 1, the attraction vanishing outright; f = 1 gives
    lambda = 0.53, because Q_z / D is about 0.47 and nearly independent of
    temperature.  So the dipolar term is at worst HALVED by freezing, never
    reversed -- nothing here is ever repulsive.
    """
    if correlated_fraction < 0.0 or correlated_fraction > 1.0:
        raise ValueError('correlated_fraction must be in [0, 1]')
    if correlated_fraction == 0.0:
        return 1.0
    retained = frustrated_quench_kBT(temperature_K, coordination)
    return 1.0 - correlated_fraction * retained / dipolar_mayer_term(
        temperature_K, energies)


def inherited_frozen_energy(temperatures_K, energies: dict,
                            aggregated_fraction, dwell_s: float = 100.0,
                            size_cv: float = 0.048480,
                            coordination: int = 6,
                            ceiling_K: float = 340.0,
                            step_K: float = 0.5) -> dict:
    """b^2 <U>_P accumulated over the freezing history: no scalar f, no ansatz.

    Replaces `loss_fraction`.  The share of bonds that freeze between T' and
    T' + dT' is -d(b^2)/dT', so the frozen energy carried at temperature T is

        E_frozen(T) = int_T^inf  (-d b^2/dT')  x(T')  u_kept(T')  dT'

    with x(T') the fraction of particles inside the dense phase at T' and
    u_kept the energy a bond keeps if it froze in place.  The uniform branch
    contributes exactly zero -- every row of U sums to zero -- so only the
    aggregated share appears.

    u_kept = -Q_z(T') k_B T' is a FIXED energy in joules (Q_z(T) T is constant
    to 0.1 % over 200-300 K), so once inherited it strengthens on further
    cooling exactly as van der Waals does.  The scalar `loss_fraction` could
    not express that, nor the fact that bonds frozen at different temperatures
    inherit different biases.

    b(T) comes from `neel_blocked_fraction`, NOT from the inherited
    lognormal grid: that grid is too coarse here and misses b^2 by ~30 %.
    """
    grid = np.asarray(temperatures_K, float)
    history = np.arange(min(grid.min(), 200.0), ceiling_K + step_K, step_K)
    blocked = np.array([neel_blocked_fraction(float(t), dwell_s, size_cv) ** 2
                        for t in history])

    if callable(aggregated_fraction):
        bound = np.array([float(aggregated_fraction(float(t)))
                          for t in history])
    else:
        bound = np.interp(history, grid, np.asarray(aggregated_fraction, float))
    bound = np.clip(bound, 0.0, 1.0)

    kept_J = np.array([-frustrated_quench_kBT(float(t), coordination)
                       * Boltzmann * float(t) for t in history])

    # freshly frozen share in each interval; b^2 falls as T rises
    fresh = -np.diff(blocked)
    midpoint = 0.5 * (history[:-1] + history[1:])
    weight = 0.5 * (bound[:-1] + bound[1:]) * 0.5 * (kept_J[:-1] + kept_J[1:])
    # energy carried by everything that froze ABOVE each temperature
    above = np.r_[np.cumsum((fresh * weight)[::-1])[::-1], 0.0]

    frozen_energy_J = np.interp(grid, history, above)
    return dict(temperature_K=grid,
                blocked_weight=np.interp(grid, history, blocked),
                frozen_energy_J=frozen_energy_J,
                frozen_kBT=-frozen_energy_J / (Boltzmann * grid),
                dwell_s=dwell_s, size_cv=size_cv, coordination=coordination)


def dense_phase_epsilon(temperature_K: float, energies: dict,
                        dwell_s: float = 150.0, size_cv: float = 0.10,
                        loss_fraction: float = 1.0) -> dict:
    """Cohesive energy per bond in the condensate, in kBT.

    Linear in the blocked weight b^2: inside a condensate every bond is
    realised simultaneously, so the dipolar term is lost in proportion rather
    than log-averaged away.
    """
    kbt = Boltzmann * temperature_K
    blocked = neel_blocked_fraction(temperature_K, dwell_s, size_cv)
    vdw = -energies['vdw_J'] / kbt
    dipole = dipolar_mayer_term(temperature_K, energies)
    return dict(temperature_K=temperature_K, blocked_fraction=blocked,
                blocked_weight=blocked ** 2, vdw_kBT=vdw, dipole_kBT=dipole,
                epsilon_kBT=vdw + (1.0 - loss_fraction * blocked ** 2) * dipole,
                epsilon_unblocked_kBT=vdw + dipole,
                epsilon_frozen_kBT=vdw + (1.0 - loss_fraction) * dipole,
                loss_fraction=loss_fraction)


# --------------------------------------------------------------------------
# van der Waals free energy and coexistence
# --------------------------------------------------------------------------
def free_energy(eta, attraction: float):
    """f/kBT per particle: ideal + Carnahan-Starling + mean-field attraction."""
    eta = np.asarray(eta, dtype=float)
    return (np.log(eta) + eta * (4.0 - 3.0 * eta) / (1.0 - eta) ** 2
            - attraction * eta)


def chemical_potential(eta, attraction: float):
    """mu/kBT = d(eta f)/d eta."""
    eta = np.asarray(eta, dtype=float)
    carnahan = (8.0 * eta - 9.0 * eta ** 2 + 3.0 * eta ** 3) / (1.0 - eta) ** 3
    return np.log(eta) + carnahan - 2.0 * attraction * eta


def pressure(eta, attraction: float):
    """P/(n kBT) x eta, i.e. the osmotic pressure in units of kBT/volume."""
    eta = np.asarray(eta, dtype=float)
    return (eta * (1.0 + eta + eta ** 2 - eta ** 3) / (1.0 - eta) ** 3
            - attraction * eta ** 2)


def coexistence(attraction: float, guess=(1e-4, 0.45)):
    """Dilute and dense branches from equal mu and equal P."""
    def residual(logs):
        dilute, dense = np.exp(logs)
        if not (0 < dilute < dense < 0.74):
            return [1e3, 1e3]
        return [float(chemical_potential(dilute, attraction)
                      - chemical_potential(dense, attraction)),
                float(pressure(dilute, attraction) - pressure(dense, attraction))]

    solution, _, flag, _ = fsolve(residual, np.log(guess), full_output=True)
    dilute, dense = np.exp(solution)
    if flag != 1 or dense - dilute < 1e-3 or dense > 0.70 or dilute <= 0:
        return None
    # fsolve will happily report a point where the residual is merely small
    # because both branches ran off to the same place, or where the Newton
    # step stalled.  Demand that the equations are actually satisfied and
    # that a van der Waals loop exists between the two roots.
    if max(abs(v) for v in residual(np.log((dilute, dense)))) > 1e-8:
        return None
    interior = np.linspace(dilute, dense, 64)[1:-1]
    if np.all(np.diff(chemical_potential(interior, attraction)) > 0):
        return None
    return float(dilute), float(dense)


def coexistence_by_continuation(attractions):
    """Solve along increasing attraction, reusing the previous root as guess.

    The dilute branch falls roughly as exp(-2 a eta_dense), so it spans many
    decades over the temperature range and a fixed initial guess fails.  Walk
    up from a weak attraction where the root is easy to find.
    """
    ordered = np.argsort(attractions)
    results = [None] * len(attractions)
    guess = (1e-3, 0.40)
    for index in ordered:
        value = float(attractions[index])
        found = coexistence(value, guess)
        if found is None:               # retry from a fresh bracket
            for trial in ((1e-2, 0.35), (1e-4, 0.45), (1e-6, 0.50),
                          (1e-9, 0.55), (1e-12, 0.60)):
                found = coexistence(value, trial)
                if found is not None:
                    break
        results[index] = found
        if found is not None:
            guess = found
    return results


def critical_attraction(low: float = 1.0, high: float = 12.0,
                        count: int = 220) -> float:
    """Smallest attraction that still yields two coexisting phases.

    Scanned with the same continuation the binodal uses, because a fixed
    initial guess fails once the dilute branch drops below ~1e-6.
    """
    grid = np.linspace(low, high, count)
    split = [value for value, branches
             in zip(grid, coexistence_by_continuation(grid)) if branches]
    if not split:
        raise RuntimeError('no coexistence anywhere in the bracket')
    return float(min(split))


def binodal_curve(temperatures_K, energies: dict, dwell_s: float = 150.0,
                  size_cv: float = 0.10) -> list[dict]:
    """Coexistence branches against temperature."""
    temperatures = np.atleast_1d(temperatures_K).astype(float)
    terms = [dense_phase_epsilon(float(t), energies, dwell_s, size_cv)
             for t in temperatures]
    attractions = np.array([MAX_COORDINATION / 2.0 * t['epsilon_kBT']
                            / CLOSE_PACKED_FRACTION for t in terms])
    branches = coexistence_by_continuation(attractions)
    rows = []
    for term, attraction, found in zip(terms, attractions, branches):
        rows.append(dict(term, attraction=float(attraction), dwell_s=dwell_s,
                         size_cv=size_cv, vdw_choice=energies['vdw_choice'],
                         phi_dilute=found[0] if found else np.nan,
                         phi_dense=found[1] if found else np.nan,
                         two_phase=found is not None))
    return rows


def aggregation_window(rows, volume_fraction: float):
    """Temperatures where the sample concentration lies inside the dome."""
    ordered = sorted(rows, key=lambda r: r['temperature_K'])
    inside = [r['temperature_K'] for r in ordered
              if r['two_phase'] and r['phi_dilute'] < volume_fraction
              < r['phi_dense']]
    if not inside:
        return None
    return min(inside), max(inside)


# --------------------------------------------------------------------------
# orientational cost of registering a face bond
# --------------------------------------------------------------------------
# The measured 19 nm spacing leaves a +/-15.5 deg tilt cone.  Twisting about
# the bond does NOT change the support height along it, so twist is
# geometrically free at a face contact.  The allowed fraction of SO(3) per
# particle is then 6 face normals times that cone, and the pair pays it twice:
#
#     f_face = 6 (1 - cos theta_tilt)/2,   cost = -2 kBT ln f_face
#
# which is 4.43 kBT at 15.5 deg.  This is the term the condensation picture
# was missing, and it is what sets the concentration scale of the binodal.
def registration_cost_kBT(tilt_deg: float = 15.5) -> float:
    """Orientational free-energy cost of holding a pair in face registry."""
    if not 0 < tilt_deg <= 90:
        raise ValueError('tilt tolerance must lie in (0, 90] degrees')
    fraction = 6.0 * (1.0 - np.cos(np.deg2rad(tilt_deg))) / 2.0
    if fraction >= 1.0:
        return 0.0
    return float(-2.0 * np.log(fraction))


def orientational_cost_kBT(tilt_deg: float = 15.5, coordination: int = 1,
                           samples: int = 600_000) -> dict:
    """Orientational entropy cost, counted per PARTICLE and per bond.

    `registration_cost_kBT` above is the isolated-pair value and is exactly
    right for a pair: one neighbour, any of six faces, a cone of half-angle
    `tilt_deg`, twist free.  It is NOT right inside a condensate, because a
    particle has one orientation serving every neighbour at once.  Two
    non-collinear neighbours already lock all three rotational degrees of
    freedom, so the per-particle cost SATURATES at about 3.57 kBT however many
    neighbours follow, while the per-bond share keeps falling as 1/z:

        z = 1  ->  2.22 kBT per particle,  4.43 per bond   (the pair value)
        z = 2  ->  3.44                    3.44
        z = 6  ->  3.57                    1.19

    Using the z = 1 cost together with z = 6 coordination -- which is what the
    cohesion currently does -- overcharges the orientational entropy by 3.6x.
    The total is nevertheless close to what the observed upper edge demands, so
    the surplus is standing in for something real that genuinely does scale per
    contact; ligand conformational entropy at a 3 nm gap is the obvious
    candidate at 1-3 kBT per contact.  `surplus_per_bond_kBT` reports exactly
    how much of the fitted cost the orientational term cannot explain.

    Monte Carlo over SO(3), seeded, so the value is reproducible; the z = 1
    case is checked against the analytic -2 ln[6(1-cos t)/2] in the tests.
    """
    from scipy.spatial.transform import Rotation

    if not 1 <= coordination <= 6:
        raise ValueError('coordination must be between 1 and 6')
    # Perpendicular directions first, so that `coordination` counts
    # INDEPENDENT constraints: a collinear pair (+x, -x) is one constraint,
    # not two, because presenting a face along +x presents one along -x too.
    faces = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    rotated = np.einsum('nij,kj->nki',
                        Rotation.random(samples, random_state=0).as_matrix(),
                        faces)
    allowed = np.ones(samples, bool)
    for bond in faces[:coordination]:
        allowed &= (rotated @ bond > np.cos(np.deg2rad(tilt_deg))).any(axis=1)
    fraction = float(allowed.mean())
    per_particle = -np.log(fraction) if fraction > 0 else np.inf
    return dict(tilt_deg=tilt_deg, coordination=coordination,
                allowed_fraction=fraction,
                per_particle_kBT=per_particle,
                per_bond_kBT=2.0 * per_particle / coordination,
                pair_cost_kBT=registration_cost_kBT(tilt_deg),
                surplus_per_bond_kBT=registration_cost_kBT(tilt_deg)
                - 2.0 * per_particle / coordination)


def critical_epsilon_kBT(critical: float) -> float:
    """Attraction per bond at the critical point, in kBT."""
    return critical * CLOSE_PACKED_FRACTION / (MAX_COORDINATION / 2.0)


def sweep(temperatures, dwell_s, size_cv, tilt_deg):
    """Binodal branches for both vdW choices, with and without the registry cost."""
    critical = critical_attraction(1.0, 30.0, 300)
    rows, curves = [], {}
    for vdw in ('inherited', 'converged'):
        energies = pair_energies(vdw)
        for label, penalty in (('none', 0.0),
                               (f'{tilt_deg:g}deg', registration_cost_kBT(tilt_deg))):
            terms = [dense_phase_epsilon(float(t), energies, dwell_s, size_cv)
                     for t in temperatures]
            attraction = np.array([
                MAX_COORDINATION / 2.0 * (t['epsilon_kBT'] - penalty)
                / CLOSE_PACKED_FRACTION for t in terms])
            branches = coexistence_by_continuation(attraction)
            series = []
            for term, value, found in zip(terms, attraction, branches):
                record = dict(term, vdw_choice=vdw, registration=label,
                              registration_cost_kBT=penalty,
                              epsilon_net_kBT=term['epsilon_kBT'] - penalty,
                              attraction=float(value),
                              critical_attraction=critical,
                              dwell_s=dwell_s, size_cv=size_cv,
                              phi_dilute=found[0] if found else np.nan,
                              phi_dense=found[1] if found else np.nan,
                              two_phase=found is not None)
                rows.append(record)
                series.append(record)
            curves[(vdw, label)] = series
    return rows, curves, critical


# --------------------------------------------------------------------------
# concentration bookkeeping
# --------------------------------------------------------------------------
# The volume fraction the free energy uses is the EFFECTIVE one, built on the
# measured 19 nm spacing, not on the 16 nm inorganic core.  They differ by
# (19/16)^3 = 1.67, which matters when comparing against a weighed-out mg/mL.
CORE_EDGE_NM = 16.0
EFFECTIVE_DIAMETER_NM = 19.0


def concentration_table(effective_fraction):
    """Effective phi -> core phi, Fe3O4 mg/mL, number density."""
    effective_fraction = np.atleast_1d(effective_fraction).astype(float)
    core = effective_fraction * CORE_EDGE_NM ** 3 / EFFECTIVE_DIAMETER_NM ** 3
    density = geometry.PARAMS.magnetite_density_kgpm3 / 1000.0   # g/cm^3
    return dict(effective_fraction=effective_fraction, core_fraction=core,
                mg_per_mL=core * density * 1000.0,
                number_density_per_m3=core / (CORE_EDGE_NM ** 3 * 1e-27))


def plot_energy_and_binodal(curves, critical, out: Path, dwell_s, size_cv,
                            tilt_deg, samples=(0.02, 0.03, 0.05)):
    """Energy and binodal on a SHARED temperature axis, read across.

    Temperature is vertical in both panels so a horizontal cut gives the
    cohesion on the left and the coexisting compositions on the right at the
    same temperature.
    """
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 13,
                         'axes.labelsize': 15, 'axes.linewidth': 1.8,
                         'axes.grid': False, 'pdf.fonttype': 42})
    figure, panels = plt.subplots(1, 2, figsize=(13.6, 8.2), sharey=True,
                                  gridspec_kw={'width_ratios': [1.0, 1.35]})
    energy, diagram = panels
    for axes in panels:
        axes.axhspan(250, 270, color='#F4EAA2', alpha=.85, zorder=0)
        axes.axhline(250, color='#C9A227', lw=1.0, zorder=1)
        axes.axhline(270, color='#C9A227', lw=1.0, zorder=1)

    styles = {'none': ('#B54535', 2.8, '-', 'no registry cost'),
              f'{tilt_deg:g}deg': ('#3C3C8C', 3.0, '-',
                                   rf'$-{registration_cost_kBT(tilt_deg):.1f}\,k_BT$ '
                                   rf'registry ({tilt_deg:g}$\degree$ cone)')}
    for label, (colour, width, dash, text) in styles.items():
        series = curves[('converged', label)]
        grid = [r['temperature_K'] for r in series]
        energy.plot([r['epsilon_net_kBT'] for r in series], grid, color=colour,
                    lw=width, ls=dash, label=text)
        finite = [r for r in series if r['two_phase']]
        if finite:
            diagram.plot([r['phi_dilute'] for r in finite],
                         [r['temperature_K'] for r in finite], color=colour,
                         lw=width, ls=dash, label=text)
            diagram.plot([r['phi_dense'] for r in finite],
                         [r['temperature_K'] for r in finite], color=colour,
                         lw=width, ls=dash)
            diagram.fill_betweenx([r['temperature_K'] for r in finite],
                                  [r['phi_dilute'] for r in finite],
                                  [r['phi_dense'] for r in finite],
                                  color=colour, alpha=.10)
    threshold = critical_epsilon_kBT(critical)
    energy.axvline(threshold, color='black', lw=2.0, ls=(0, (5, 2)),
                   label=rf'condensation threshold $\epsilon_c$ = {threshold:.2f} $k_BT$')
    energy.set(xlabel=r'Cohesion per bond  $\epsilon$ / $k_BT$',
               ylabel='Temperature (K)', ylim=(200, 300))
    energy.set_title('Dense-phase cohesion', fontsize=13)
    energy.legend(loc='upper left', bbox_to_anchor=(0, -.12), frameon=False,
                  fontsize=10)

    for sample, dash in zip(samples, ((0, (1, 1.6)), (0, (4, 2)), (0, (7, 2)))):
        diagram.axvline(sample, color='0.35', lw=1.4, ls=dash)
        table = concentration_table(sample)
        diagram.text(sample * 0.93, 298,
                     rf'$\phi$={sample:g}, {table["mg_per_mL"][0]:.0f} mg/mL',
                     rotation=90, fontsize=9, ha='right', va='top',
                     color='0.25')
    diagram.set_xscale('log')
    diagram.set(xlabel=r'Effective volume fraction  $\phi$  (19 nm spacing)',
                xlim=(2e-3, 0.7))
    diagram.set_title('Fluid-fluid binodal, re-entrant', fontsize=13, pad=10)
    diagram.text(.025, .03, 'shaded: two phases\nyellow band: observed window',
                 transform=diagram.transAxes, fontsize=11, ha='left', va='bottom')
    diagram.legend(loc='upper left', bbox_to_anchor=(0, -.12), frameon=False,
                   fontsize=10)
    figure.suptitle('Blocked-dipole condensation, converged van der Waals\n'
                    rf'dwell {dwell_s:g} s, size $CV$ = {size_cv * 100:.0f} %, '
                    '3 nm gap and 19 nm spacing (both measured)', fontsize=14)
    figure.subplots_adjust(left=.09, right=.97, top=.82, bottom=.28, wspace=.10)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'energy_and_binodal.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def epsilon_threshold(volume_fraction: float, span=(11.0, 30.0), count=120):
    """Cohesion at which the dilute binodal branch sits at `volume_fraction`.

    Calibrated once: phi_dilute falls monotonically with the attraction, so
    "inside the dome" is equivalent to a threshold on the cohesion, which
    turns every later scan into arithmetic instead of a root solve.
    """
    grid = np.linspace(*span, count)
    solved = coexistence_by_continuation(grid)
    attraction = [a for a, s in zip(grid, solved) if s]
    dilute = [s[0] for s in solved if s]
    star = float(np.interp(-np.log(volume_fraction),
                           [-np.log(x) for x in dilute], attraction))
    return star * CLOSE_PACKED_FRACTION / (MAX_COORDINATION / 2.0), star


def fit_size_cv(volume_fraction, lower_edge_K, upper_edge_K, dwell_s,
                energies, bracket=(0.02, 0.13), loss_fraction=1.0,
                threshold_kBT=None, grid_step_K=0.25):
    """CV that reproduces both observed edges at a fixed dwell.

    Pinning the upper edge fixes the cut level at eps(upper_edge), which makes
    the WIDTH independent of the registry cost: the lower edge is simply where
    the cohesion returns to that level.  CV is therefore the only thing
    setting the width, and the registry cost that falls out is a free check
    against the tilt cone the measured 19 nm spacing allows.
    """
    from scipy.optimize import brentq

    threshold = (epsilon_threshold(volume_fraction)[0]
                 if threshold_kBT is None else threshold_kBT)
    grid = np.arange(200.0, 300.01, grid_step_K)

    def edges(size_cv):
        cohesion = np.array([dense_phase_epsilon(float(t), energies, dwell_s,
                                                 size_cv, loss_fraction)['epsilon_kBT']
                             for t in grid])
        level = float(np.interp(upper_edge_K, grid, cohesion))
        cold = grid < grid[int(np.argmax(cohesion))]
        if not cold.any() or cohesion[cold].min() > level:
            # The cohesion never falls back to the cut level inside the grid,
            # so the lower edge sits at or below its floor.  Reporting the
            # floor keeps `miss` decreasing through this regime; a positive
            # sentinel here would collide in sign with the small-CV branch and
            # make the bracket look rootless.
            return float(grid[0]), level, cohesion
        return float(np.interp(level, cohesion[cold], grid[cold])), level, cohesion

    def miss(size_cv):
        return edges(size_cv)[0] - lower_edge_K

    if miss(bracket[0]) * miss(bracket[1]) > 0.0:
        return None
    size_cv = float(brentq(miss, *bracket, xtol=1e-4))
    low, level, cohesion = edges(size_cv)
    cost = level - threshold
    tilt = float(brentq(lambda t: registration_cost_kBT(t) - cost, 5.0, 60.0))
    return dict(size_cv=size_cv, dwell_s=dwell_s, lower_edge_K=low,
                upper_edge_K=upper_edge_K, registry_cost_kBT=cost,
                fitted_tilt_deg=tilt, geometric_tilt_deg=15.5,
                geometric_cost_kBT=registration_cost_kBT(15.5),
                epsilon_threshold_kBT=threshold, temperatures_K=grid,
                cohesion_kBT=cohesion, volume_fraction=volume_fraction,
                loss_fraction=loss_fraction)


def plot_fitted_diagram(fit, out: Path):
    """One panel: binodal on the bottom axis, cohesion on the top axis.

    Temperature is the shared vertical axis, so a horizontal cut reads the
    coexisting compositions on the bottom scale and the cohesion driving them
    on the top scale.
    """
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 13,
                         'axes.labelsize': 15, 'axes.linewidth': 1.8,
                         'axes.grid': False, 'pdf.fonttype': 42})
    grid = fit['temperatures_K']
    net = fit['cohesion_kBT'] - fit['registry_cost_kBT']
    attraction = MAX_COORDINATION / 2.0 * net / CLOSE_PACKED_FRACTION
    branches = coexistence_by_continuation(attraction)

    figure, axes = plt.subplots(figsize=(9.4, 8.8))
    top = axes.twiny()
    axes.axhspan(fit['lower_edge_K'], fit['upper_edge_K'], color='#F4EAA2',
                 alpha=.85, zorder=0)
    for edge in (fit['lower_edge_K'], fit['upper_edge_K']):
        axes.axhline(edge, color='#C9A227', lw=1.2, zorder=1)

    dilute = [(s[0], t) for s, t in zip(branches, grid) if s]
    dense = [(s[1], t) for s, t in zip(branches, grid) if s]
    if dilute:
        axes.plot([p[0] for p in dilute], [p[1] for p in dilute],
                  color='#3C3C8C', lw=3.0, zorder=4, label='binodal')
        axes.plot([p[0] for p in dense], [p[1] for p in dense],
                  color='#3C3C8C', lw=3.0, zorder=4)
        axes.fill_betweenx([p[1] for p in dilute], [p[0] for p in dilute],
                           [p[0] for p in dense], color='#3C3C8C', alpha=.13,
                           zorder=2)
    axes.axvline(fit['volume_fraction'], color='#1B7A3E', lw=2.6,
                 ls=(0, (7, 3)), zorder=5,
                 label=rf"sample $\phi$ = {fit['volume_fraction']:.4f}"
                       "\n(1.25 % core, 65 mg/mL)")
    axes.set_xscale('log')
    axes.set(xlabel=r'Effective volume fraction  $\phi$   (19 nm spacing)',
             ylabel='Temperature (K)', xlim=(3e-3, 0.7), ylim=(200, 300))

    top.plot(net, grid, color='#C0552F', lw=2.8, zorder=3)
    top.axvline(fit['epsilon_threshold_kBT'], color='#C0552F', lw=1.8,
                ls=(0, (5, 2)), zorder=3)
    top.set_xlim(0.0, 4.2)
    top.set_xlabel('Cohesion per bond  '
                   r'$\epsilon$ / $k_BT$    (orange, dashed = threshold)',
                   color='#C0552F')
    top.tick_params(axis='x', colors='#C0552F')
    top.spines['top'].set_color('#C0552F')

    top.plot([], [], color='#C0552F', lw=1.8, ls=(0, (5, 2)),
             label=rf"threshold $\epsilon_c$ = {fit['epsilon_threshold_kBT']:.2f}"
                   ' $k_BT$ (top axis)')
    top.plot([], [], color='#C0552F', lw=2.8,
             label=r'cohesion $\epsilon(T)$ (top axis)')
    handles, labels = axes.get_legend_handles_labels()
    extra, extra_labels = top.get_legend_handles_labels()
    axes.legend(handles + extra, labels + extra_labels, loc='lower right',
                frameon=False, fontsize=10.5)
    axes.text(0.035, 0.035,
              f"fitted   CV = {fit['size_cv'] * 100:.2f} %\n"
              f"fixed    dwell = {fit['dwell_s']:.0f} s\n"
              f"window   {fit['lower_edge_K']:.0f}-{fit['upper_edge_K']:.0f} K, "
              "both edges matched",
              transform=axes.transAxes, fontsize=11, va='bottom', ha='left',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='none', alpha=.85))
    axes.text(0.035, 0.82,
              'registry cost was NOT fitted:\n'
              f"   needed    {fit['registry_cost_kBT']:.2f} $k_BT$ = "
              f"{fit['fitted_tilt_deg']:.2f}$\\degree$ cone\n"
              f"   geometry  {fit['geometric_cost_kBT']:.2f} $k_BT$ = "
              f"{fit['geometric_tilt_deg']:.2f}$\\degree$ cone\n"
              f"   differ by {fit['fitted_tilt_deg'] - fit['geometric_tilt_deg']:+.2f}$\\degree$",
              transform=axes.transAxes, fontsize=10.5, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                        edgecolor='0.7'))
    figure.suptitle('Blocked-dipole condensation of 16 nm magnetite nanocubes\n'
                    'converged vdW; 3 nm gap and 19 nm spacing both measured',
                    fontsize=14)
    figure.subplots_adjust(left=.12, right=.96, top=.84, bottom=.10)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'fitted_phase_diagram.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)


def aggregated_particle_fraction(epsilon_kBT: float,
                                 volume_fraction: float = SAMPLE_VOLUME_FRACTION):
    """Lever rule: share of PARTICLES sitting in the dense phase.

    Note this is not the share of the sample VOLUME the dense phase occupies,
    which is smaller by phi / phi_dense; conflating the two is what produced
    the bogus "at most 6.7 % of particles" bound earlier.  With phi_dilute -> 0
    the particle share goes to 1, not to 6.7 %.
    """
    attraction = MAX_COORDINATION / 2.0 * epsilon_kBT / CLOSE_PACKED_FRACTION
    branches = coexistence_by_continuation([attraction])[0]
    if branches is None:
        return 0.0
    dilute, dense = branches
    if not dilute < volume_fraction < dense:
        return 0.0
    volume_share = (volume_fraction - dilute) / (dense - dilute)
    return float(np.clip(dense * volume_share / volume_fraction, 0.0, 1.0))


def self_consistent_cooling(energies: dict, dwell_s: float = 100.0,
                            size_cv: float = 0.048480,
                            tilt_deg: float = 15.5,
                            coordination: int = 6,
                            volume_fraction: float = SAMPLE_VOLUME_FRACTION,
                            start_K: float = 320.0, stop_K: float = 200.0,
                            step_K: float = 2.0) -> list[dict]:
    """March down in temperature with the frozen bias inherited, not assumed.

    This closes the loop that `loss_fraction` left open.  At each step the
    bonds that freeze inherit the bias of wherever they were AT THAT MOMENT,
    and where they were is set by the phase equilibrium one step earlier:

        x(T)  <- lever rule at eps(T)
        E_frozen(T) <- E_frozen(T + dT) + [b^2(T) - b^2(T + dT)] x u_kept
        eps(T) <- vdW + (1 - b^2) D + E_frozen / kBT - registration

    Marching rather than iterating to a fixed point is deliberate: the
    dependence is causal in temperature, so a downward sweep is the physical
    solution and an upward sweep is a genuinely different one.  f never
    appears -- it was only ever a summary of this integral.
    """
    registration = registration_cost_kBT(tilt_deg)
    grid = np.arange(start_K, stop_K - 1e-9, -abs(step_K))
    frozen_J, previous_b2, previous_x = 0.0, None, 0.0
    trajectory = []
    for temperature in grid:
        kbt = Boltzmann * float(temperature)
        blocked = neel_blocked_fraction(float(temperature), dwell_s, size_cv) ** 2
        if previous_b2 is not None and blocked > previous_b2:
            kept_J = (-frustrated_quench_kBT(float(temperature), coordination)
                      * kbt)
            frozen_J += (blocked - previous_b2) * previous_x * kept_J
        previous_b2 = blocked
        dipole = dipolar_mayer_term(float(temperature), energies)
        epsilon = (-energies['vdw_J'] / kbt + (1.0 - blocked) * dipole
                   - frozen_J / kbt - registration)
        aggregated = aggregated_particle_fraction(epsilon, volume_fraction)
        previous_x = aggregated
        trajectory.append(dict(
            temperature_K=float(temperature), blocked_weight=blocked,
            dipole_kBT=dipole, vdw_kBT=-energies['vdw_J'] / kbt,
            frozen_kBT=-frozen_J / kbt, epsilon_kBT=epsilon,
            aggregated_fraction=aggregated, two_phase=aggregated > 0.0))
    return trajectory


def pair_energy_table(energies: dict, dwell_s: float = 100.0,
                      size_cv: float = 0.048480, tilt_deg: float = 15.5,
                      grid=np.arange(200.0, 300.01, 0.5)) -> dict:
    """Signed bond free energy of ONE pair, negative = binding.

    The registration cost here is unambiguously the z = 1 value, so that term
    carries none of the per-bond / per-particle bookkeeping the dense phase
    forces.  The b^2 suppression, however, is NOT a free-pair effect: a
    particle floating in solution re-randomises its lab-frame moment by
    Brownian tumbling (tau_B ~ 6 us, near enough temperature independent)
    whether or not Neel is blocked, so an isolated pair stays annealed at every
    temperature and shows no re-entrance at all.  What this figure describes is
    one bond whose particles are CAGED by neighbours -- at 19 nm spacing a cube
    can tilt only +/-15.5 deg, far short of the 54.7 deg needed to bring a
    different <111> onto the bond -- which is why Neel is then the only way
    left to re-sample.

    The frozen bond keeps between none and half of the dipolar attraction --
    lambda = 1 if the moment froze free in the dilute phase, lambda = lambda(T)
    from `loss_fraction` at f = 1 if it froze inside an aggregate -- so the
    dipolar term is returned as a pair of bounds, not a single curve.
    """
    grid = np.asarray(grid, float)
    vdw = np.array([energies['vdw_J'] / (Boltzmann * t) for t in grid])
    dipole = np.array([dipolar_mayer_term(float(t), energies) for t in grid])
    blocked = np.array([neel_blocked_fraction(float(t), dwell_s, size_cv)
                        for t in grid])
    registration = registration_cost_kBT(tilt_deg)
    halved = np.array([loss_fraction(float(t), energies, 1.0, 6) for t in grid])
    bounds = {}
    for name, lam in (('lost', np.ones_like(grid)), ('halved', halved)):
        collected = (1.0 - lam * blocked ** 2) * dipole
        bounds[name] = dict(dipolar=-collected,
                            total=vdw - collected + registration)
    blocking_K = float(np.interp(0.5, blocked[::-1], grid[::-1]))
    return dict(temperature_K=grid, vdw_kBT=vdw, dipole_kBT=dipole,
                blocked=blocked, blocked_weight=blocked ** 2,
                registration_kBT=registration, bounds=bounds,
                blocking_temperature_K=blocking_K,
                dwell_s=dwell_s, size_cv=size_cv)


def _dipole_cartoon(axes, mode: str, colour: str):
    """One schematic of a cube pair and what its moments are doing.

    Three states, distinguished by two glyphs that stay legible when small: a
    curved arrow over a cube means the moment can still be re-sampled, and a
    cross through it means Neel blocking has stopped that.  Ghost arrows were
    tried first and turned into an unreadable blob at this size.

    'warm'   free to re-sample, but every well is shallow against kBT
    'window' free to re-sample, and the pair sits head-to-tail -> bound
    'frozen' blocked at inherited, mutually uncorrelated angles
    """
    from matplotlib.patches import FancyArrow, Rectangle

    axes.set(xlim=(-1.08, 1.08), ylim=(-0.95, 1.12))
    axes.set_aspect('equal')
    axes.axis('off')

    gap = 0.12 if mode == 'window' else 0.46
    side = 0.62
    centres = (-(gap / 2 + side / 2), gap / 2 + side / 2)
    for cx in centres:
        axes.add_patch(Rectangle((cx - side / 2, -side / 2), side, side,
                                 facecolor='#DCDCE6', edgecolor='#33334D',
                                 lw=1.8, zorder=2))

    angles = {'frozen': (143.0, -24.0),      # inherited, uncorrelated
              'window': (55.0, -55.0),       # head-to-tail, 110 deg apart
              'warm': (25.0, 155.0)}[mode]
    length = 0.46
    for cx, angle in zip(centres, angles):
        radian = np.deg2rad(angle)
        axes.add_patch(FancyArrow(
            cx - length / 2 * np.cos(radian), -length / 2 * np.sin(radian),
            length * np.cos(radian), length * np.sin(radian),
            width=0.040, head_width=0.24, head_length=0.20,
            length_includes_head=True, color=colour, zorder=4))

        spin = '0.55' if mode == 'frozen' else colour
        axes.annotate('', xy=(cx + 0.26, 0.52), xytext=(cx - 0.26, 0.52),
                      annotation_clip=False,
                      arrowprops=dict(arrowstyle='<|-|>', color=spin, lw=1.5,
                                      mutation_scale=9,
                                      connectionstyle='arc3,rad=-0.55'))
        if mode == 'frozen':
            axes.plot([cx - 0.12, cx + 0.12], [0.60, 0.84], color='#B54535',
                      lw=2.2, zorder=6)
            axes.plot([cx - 0.12, cx + 0.12], [0.84, 0.60], color='#B54535',
                      lw=2.2, zorder=6)

    if mode == 'warm':
        axes.annotate('', xy=(0.20, -0.02), xytext=(-0.20, -0.02),
                      arrowprops=dict(arrowstyle='<|-|>', color='#B54535',
                                      lw=2.0, mutation_scale=10))
        axes.text(0, -0.30, r'$k_BT$', fontsize=9, color='#B54535',
                  ha='center', va='top', zorder=6)
    elif mode == 'window':
        axes.plot([0, 0], [-side / 2, side / 2], color='#8A6D1F', lw=2.4,
                  zorder=5)


def add_dipole_cartoons(figure, main, table: dict, window):
    """Three cartoons above the energy axes, aligned to the temperature axis."""
    orange, dark = '#C0552F', '#1A1A1A'
    cold, warm = window
    panels = (
        (0.5 * (200 + cold), 'frozen', orange, f'below {cold:.0f} K',
         'blocked at inherited random angles\n'
         r'$\rightarrow$ no selection, attraction gone'),
        (0.5 * (cold + warm), 'window', dark, f'{cold:.0f}-{warm:.0f} K',
         'still re-sampling, picks head-to-tail\n'
         r'$\rightarrow$ aggregates'),
        (0.5 * (warm + 300), 'warm', orange, f'above {warm:.0f} K',
         're-sampling, but the well is shallow\n'
         r'$\rightarrow$ single phase'),
    )
    box = main.get_position()
    width = 0.235
    height = width * figure.get_figwidth() / figure.get_figheight()
    for temperature, mode, colour, heading, caption in panels:
        centre = box.x0 + (temperature - 200.0) / 100.0 * box.width
        centre = min(max(centre, box.x0 + width / 2), box.x1 - width / 2)
        axes = figure.add_axes([centre - width / 2, box.y1 + 0.088,
                                width, height])
        _dipole_cartoon(axes, mode, colour)
        axes.set_title(heading, fontsize=11, fontweight='bold', pad=4,
                       color=colour)
        axes.text(0, -1.02, caption, fontsize=8.8, ha='center', va='top',
                  color='0.25', transform=axes.transData)
        figure.add_artist(plt.Line2D(
            [centre, centre], [box.y1 + 0.080, box.y1],
            color=colour, lw=1.0, ls=':', alpha=.7))


def plot_pair_energies(table: dict, out: Path):
    """One pair, signed energies, no phase diagram anywhere.

    Negative binds.  Van der Waals falls monotonically on cooling because a
    fixed joule energy divided by kBT grows; the dipolar term turns and rises
    because Neel blocking removes the orientational re-sampling it is made of.
    Their sum has a minimum, and the binding window opens around it.

    The binding criterion is ONE chosen number, not a derived one: it is set so
    the cold edge of the window coincides with the blocking temperature.  Where
    the warm edge then lands is not chosen, and is the figure's only claim.
    """
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
                         'axes.labelsize': 13.5, 'axes.linewidth': 1.6,
                         'axes.grid': False, 'pdf.fonttype': 42})
    grid = table['temperature_K']
    lost, halved = table['bounds']['lost'], table['bounds']['halved']
    blocking = table['blocking_temperature_K']
    blue, orange, grey, dark, red = '#3C3C8C', '#C0552F', '0.45', '#1A1A1A', '#B54535'

    criterion = float(np.interp(blocking, grid, lost['total']))
    inside = grid[lost['total'] <= criterion]
    warm = inside.max()

    figure, (main, lower) = plt.subplots(
        2, 1, figsize=(10.4, 11.2), sharex=True,
        gridspec_kw={'height_ratios': [2.5, 1.0]})
    for axes in (main, lower):
        axes.axvspan(blocking, warm, color='#F4EAA2', alpha=.75, zorder=0)
        axes.axvline(blocking, color='#C9A227', lw=2.0, zorder=1)

    main.axhline(0, color='black', lw=1.0, zorder=2)
    main.fill_between(grid, lost['dipolar'], halved['dipolar'], color=orange,
                      alpha=.22, lw=0, zorder=2)
    main.plot(grid, halved['dipolar'], color=orange, lw=1.6, ls=(0, (5, 2)),
              zorder=3)
    main.plot(grid, lost['dipolar'], color=orange, lw=3.2, zorder=4,
              label=r'dipolar   $-(1-\lambda b^2)\,\ln\langle e^{-U_{dd}/k_BT}\rangle$')
    main.plot(grid, table['vdw_kBT'], color=blue, lw=3.2, zorder=4,
              label=r'van der Waals   $U_{vdW}/k_BT$')
    main.plot(grid, np.full_like(grid, table['registration_kBT']), color=grey,
              lw=2.4, zorder=3,
              label=rf'registration   $+{table["registration_kBT"]:.2f}\,k_BT$'
                    '  (pair value, exact here)')
    main.fill_between(grid, lost['total'], halved['total'], color=dark,
                      alpha=.16, lw=0, zorder=4)
    main.plot(grid, halved['total'], color=dark, lw=1.8, ls=(0, (5, 2)),
              zorder=5)
    main.plot(grid, lost['total'], color=dark, lw=3.6, zorder=6,
              label='total pair bond free energy')
    main.axhline(criterion, color=red, lw=2.0, ls=(0, (6, 3)), zorder=5,
                 label='binding criterion (chosen, see caption)')

    # Everything above zero is empty, so the trend labels and their
    # leaders live there and no annotation crosses another.
    main.annotate('dipolar WEAKENS on cooling\n'
                  '(blocking removes the re-sampling)',
                  xy=(202.5, lost['dipolar'][0]), xytext=(206, 3.1),
                  fontsize=10.5, color=orange, ha='left', va='center',
                  arrowprops=dict(arrowstyle='->', color=orange, lw=1.5))
    main.annotate('van der Waals STRENGTHENS on cooling\n'
                  r'(fixed $U$ divided by a smaller $k_BT$)',
                  xy=(210, float(np.interp(210, grid, table['vdw_kBT']))),
                  xytext=(206, 1.0), fontsize=10.5, color=blue,
                  ha='left', va='center',
                  arrowprops=dict(arrowstyle='->', color=blue, lw=1.5))
    main.text(.5 * (blocking + warm), 3.75,
              f'bound  {blocking:.0f}-{warm:.0f} K', fontsize=12,
              color='#8A6D1F', ha='center', va='center', fontweight='bold')
    main.text(298, 2.55,
              'dashed: moments frozen inside the bond\n'
              r'($\lambda \approx 0.5$) - the pair never unbinds',
              fontsize=10, color='0.3', ha='right', va='center')
    main.set(ylabel=r'Bond free energy  ($k_BT$,  negative binds)',
             ylim=(-8.8, 5.0))
    main.legend(loc='lower left', bbox_to_anchor=(.01, .015),
                frameon=False, fontsize=10.5)
    main.set_title('One bond, its particles caged so they cannot tumble',
                   loc='left', fontsize=12.5, fontweight='bold')

    lower.plot(grid, table['blocked'], color=orange, lw=2.0, ls=(0, (5, 2)),
               label='$b$')
    lower.plot(grid, table['blocked_weight'], color=orange, lw=3.0,
               label='$b^2$')
    lower.axhline(0.5, color=grey, lw=1.0, ls=':')
    lower.annotate(rf'$T_B$ = {blocking:.0f} K', xy=(blocking, .5),
                   xytext=(blocking - 34, .70), fontsize=11, color='#8A6D1F',
                   fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#C9A227', lw=1.6))
    lower.set(xlabel='Temperature (K)', ylabel='Blocked fraction',
              xlim=(200, 300), ylim=(0, 1.05))
    lower.legend(loc='upper right', frameon=False, fontsize=10.5)
    lower.set_title(f"Neel blocking at {table['dwell_s']:.0f} s dwell, "
                    f"CV {table['size_cv'] * 100:.2f} %", loc='left',
                    fontsize=11.5)

    figure.subplots_adjust(left=.10, right=.975, top=.73, bottom=.06,
                           hspace=.16)
    add_dipole_cartoons(figure, main, table, (blocking, warm))
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'pair_energy_vs_temperature.{extension}',
                       dpi=300, bbox_inches='tight')
    plt.close(figure)
    return dict(criterion_kBT=criterion, cold_edge_K=blocking,
                warm_edge_K=float(warm))


def cooling_energy_table(energies: dict, dwell_s: float = 100.0,
                         size_cv: float = 0.048480, tilt_deg: float = 15.65,
                         coordination: int = 6,
                         volume_fraction: float = SAMPLE_VOLUME_FRACTION,
                         step_K: float = 1.0) -> dict:
    """Signed bond free energy along a cooling sweep, negative = binding.

    Everything is the self-consistent trajectory: the frozen dipolar bias is
    inherited from wherever the moments were when they blocked, and where they
    were is the lever rule applied to the cohesion one step earlier.  No
    lambda, no f, and no chosen binding criterion -- the window is simply where
    the lever rule puts a finite share of the particles in the dense phase.

    The two bounds kept for the band are the extremes that closure chooses
    between: nothing freezes in place (x = 0) and everything does (x = 1).
    """
    trajectory = self_consistent_cooling(
        energies, dwell_s=dwell_s, size_cv=size_cv, tilt_deg=tilt_deg,
        coordination=coordination, volume_fraction=volume_fraction,
        start_K=320.0, stop_K=200.0, step_K=step_K)
    trajectory = trajectory[::-1]                      # ascending in T
    grid = np.array([r['temperature_K'] for r in trajectory])
    inside = grid <= 300.0
    grid = grid[inside]

    def column(key):
        return np.array([r[key] for r in trajectory])[inside]

    registration = registration_cost_kBT(tilt_deg)
    dipole, blocked = column('dipole_kBT'), column('blocked_weight')
    annealed = (1.0 - blocked) * dipole
    frozen = column('frozen_kBT')
    ceiling = inherited_frozen_energy(grid, energies, lambda t: 1.0,
                                      dwell_s, size_cv, coordination)
    aggregated = column('aggregated_fraction')
    window = grid[aggregated > 0.0]
    return dict(
        temperature_K=grid, vdw_kBT=-column('vdw_kBT'),
        dipole_kBT=dipole, blocked=np.sqrt(blocked), blocked_weight=blocked,
        aggregated_fraction=aggregated, registration_kBT=registration,
        annealed_kBT=annealed,
        dipolar_self=-(annealed + frozen),
        dipolar_ceiling=-(annealed + ceiling['frozen_kBT']),
        total_self=-(column('vdw_kBT') + annealed + frozen),
        total_ceiling=-(column('vdw_kBT') + annealed
                        + ceiling['frozen_kBT']),
        epsilon_net_kBT=column('epsilon_kBT'),
        frozen_kBT=frozen, frozen_ceiling_kBT=ceiling['frozen_kBT'],
        window=(float(window.min()), float(window.max())) if window.size else None,
        threshold_sum_kBT=(
            float(np.interp(float(window.min()), grid,
                            column('vdw_kBT') + annealed + frozen))
            if window.size else np.nan),
        blocking_temperature_K=float(np.interp(
            0.5, np.sqrt(blocked)[::-1], grid[::-1])),
        dwell_s=dwell_s, size_cv=size_cv, tilt_deg=tilt_deg,
        coordination=coordination)


def plot_cooling_energies(table: dict, out: Path):
    """Three panels: the totals, the dipolar term split in two, and the switches.

    The middle panel is the one that earns its place: it separates the part of
    the dipolar attraction still being re-sampled, (1-b^2) D, from the part
    inherited by bonds that have already frozen, E_frozen.  The self-consistent
    E_frozen hugs zero while its ceiling climbs to 2.7 kBT, which is the whole
    content of the closure -- the sweep picks the bottom of the available range
    because the aggregate is never more than a small share of the sample.
    """
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
                         'axes.labelsize': 13.5, 'axes.linewidth': 1.6,
                         'axes.grid': False, 'pdf.fonttype': 42})
    grid = table['temperature_K']
    blue, orange, grey, dark = '#3C3C8C', '#C0552F', '0.45', '#1A1A1A'
    cold, warm = table['window'] if table['window'] else (np.nan, np.nan)
    blocking = table['blocking_temperature_K']

    figure, (main, split, lower) = plt.subplots(
        3, 1, figsize=(10.4, 13.6), sharex=True,
        gridspec_kw={'height_ratios': [2.3, 1.15, 1.0]})
    for axes in (main, split, lower):
        axes.axvspan(cold, warm, color='#F4EAA2', alpha=.75, zorder=0)
        axes.axvline(blocking, color='#C9A227', lw=2.0, ls=(0, (5, 2)), zorder=1)

    # ---- (a) the two attractions and their sum ---------------------------
    main.axhline(0, color='black', lw=1.0, zorder=2)
    main.fill_between(grid, table['dipolar_self'], table['dipolar_ceiling'],
                      color=orange, alpha=.20, lw=0, zorder=2)
    main.plot(grid, table['dipolar_ceiling'], color=orange, lw=1.5,
              ls=(0, (5, 2)), zorder=3)
    main.plot(grid, table['dipolar_self'], color=orange, lw=3.2, zorder=4,
              label=r'dipolar  $-[(1-b^2)D + E_{frozen}/k_BT]$')
    main.plot(grid, table['vdw_kBT'], color=blue, lw=3.2, zorder=4,
              label=r'van der Waals   $U_{vdW}/k_BT$')
    level = -table['threshold_sum_kBT']
    main.axhline(level, color='#B54535', lw=2.2, ls=(0, (6, 3)), zorder=3,
                 label=rf'phase-separation threshold  ${level:.2f}\,k_BT$')
    main.fill_between(grid, table['total_self'], table['total_ceiling'],
                      color=dark, alpha=.14, lw=0, zorder=4)
    main.plot(grid, table['total_ceiling'], color=dark, lw=1.6, ls=(0, (5, 2)),
              zorder=5,
              label='dashed: the same, if EVERY moment froze bonded')
    main.plot(grid, table['total_self'], color=dark, lw=3.6, zorder=6,
              label='total = van der Waals + dipolar')
    main.annotate('dipolar WEAKENS on cooling\n(blocking removes the re-sampling)',
                  xy=(202.5, table['dipolar_self'][0]), xytext=(206, 2.9),
                  fontsize=10.5, color=orange, ha='left', va='center',
                  arrowprops=dict(arrowstyle='->', color=orange, lw=1.5))
    main.annotate('van der Waals STRENGTHENS on cooling\n'
                  r'(fixed $U$ divided by a smaller $k_BT$)',
                  xy=(210, float(np.interp(210, grid, table['vdw_kBT']))),
                  xytext=(206, 0.8), fontsize=10.5, color=blue, ha='left',
                  va='center',
                  arrowprops=dict(arrowstyle='->', color=blue, lw=1.5))
    main.text(.5 * (cold + warm), 2.9, f'two phases\n{cold:.0f}-{warm:.0f} K',
              fontsize=12, color='#8A6D1F', ha='center', va='center',
              fontweight='bold')
    main.text(298, -0.9,
              'the threshold carries the constant that used to be\n'
              f"subtracted as registration ({table['registration_kBT']:.2f} "
              r'$k_BT$); moving it changes nothing',
              fontsize=9.5, color='#B54535', ha='right', va='center')
    main.set(ylabel=r'Interaction free energy  ($k_BT$)', ylim=(-11.4, 4.2))
    main.legend(loc='lower left', bbox_to_anchor=(.005, .005), frameon=False,
                fontsize=10.5, ncol=2, columnspacing=2.4)
    main.set_title('(a)  Total = the two attractions, nothing subtracted',
                   loc='left', fontsize=12.5, fontweight='bold')

    # ---- (b) the dipolar term split ---------------------------------------
    split.axhline(0, color='black', lw=1.0, zorder=2)
    split.plot(grid, table['annealed_kBT'], color=orange, lw=3.2, zorder=4,
               label=r'$(1-b^2)\,D$   still being re-sampled')
    split.fill_between(grid, table['frozen_kBT'], table['frozen_ceiling_kBT'],
                       color=dark, alpha=.16, lw=0, zorder=2)
    split.plot(grid, table['frozen_ceiling_kBT'], color=dark, lw=1.6,
               ls=(0, (5, 2)), zorder=3,
               label=r'$E_{frozen}/k_BT$   ceiling, if every moment froze bonded')
    split.plot(grid, table['frozen_kBT'], color=dark, lw=3.2, zorder=5,
               label=r'$E_{frozen}/k_BT$   self-consistent')
    peak_frozen = float(np.max(table['frozen_kBT']))
    split.annotate(f'self-consistent frozen term never exceeds '
                   f'{peak_frozen:.3f} ' r'$k_BT$',
                   xy=(228, float(np.interp(228, grid, table['frozen_kBT']))),
                   xytext=(232, 1.45), fontsize=10, color=dark, ha='left',
                   arrowprops=dict(arrowstyle='->', color=dark, lw=1.3))
    split.set(ylabel=r'Contribution  ($k_BT$)', ylim=(-0.25, 5.9))
    split.legend(loc='upper left', frameon=False, fontsize=10.5)
    split.set_title('(b)  The dipolar term split: re-sampled vs inherited',
                    loc='left', fontsize=12.5, fontweight='bold')

    # ---- (c) the switches --------------------------------------------------
    lower.plot(grid, table['blocked'], color=orange, lw=1.8, ls=(0, (5, 2)),
               label='$b$  one moment blocked')
    lower.plot(grid, table['blocked_weight'], color=orange, lw=3.0,
               label='$b^2$  both blocked')
    lower.fill_between(grid, 0, table['aggregated_fraction'], color=blue,
                       alpha=.25, lw=0)
    lower.plot(grid, table['aggregated_fraction'], color=blue, lw=3.0,
               label='$x$  particles in the dense phase')
    lower.annotate(rf'$T_B$ = {blocking:.0f} K', xy=(blocking, .52),
                   xytext=(blocking - 36, .74), fontsize=11, color='#8A6D1F',
                   fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#C9A227', lw=1.6))
    peak = table['aggregated_fraction'].max()
    lower.text(.5 * (cold + warm), peak + .06, f'peak {100 * peak:.0f} %',
               fontsize=10.5, color=blue, ha='center', va='bottom')
    lower.set(xlabel='Temperature (K)', ylabel='Fraction',
              xlim=(200, 300), ylim=(0, 1.05))
    lower.legend(loc='upper right', frameon=False, fontsize=10.5)
    lower.set_title(f"(c)  Neel blocking at {table['dwell_s']:.0f} s dwell, "
                    f"CV {table['size_cv'] * 100:.2f} %, and the aggregated "
                    "fraction it produces", loc='left', fontsize=12.5,
                    fontweight='bold')

    figure.subplots_adjust(left=.10, right=.975, top=.745, bottom=.05,
                           hspace=.17)
    add_dipole_cartoons(figure, main, table, (cold, warm))
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'cooling_energy_vs_temperature.{extension}',
                       dpi=300, bbox_inches='tight')
    plt.close(figure)
    return dict(window=table['window'], peak_aggregated=float(peak),
                peak_frozen_kBT=peak_frozen,
                blocking_temperature_K=blocking)


def plot_two_energies(table: dict, out: Path, blocking_K: float = 250.0,
                      show_threshold: bool = True):
    """Signed energies throughout, over the split and the switches.

    Everything that is an energy is plotted with its own sign, negative for
    binding, in panels (a) and (b) alike -- an earlier version drew panel (b)
    as positive magnitudes and the two panels could not be read against each
    other.  The panel heights are set from the y-ranges so that a kBT is the
    same height in both.

    `blocking_K` is the measured blocking temperature, 250 K, which is also
    the calibration input: K_cubic is set so tau_N = 100 s at 250 K for the
    nominal 16 nm cube.  The ensemble is half blocked a little lower, at
    244 K, for two definitional reasons -- b is the survival probability
    exp(-t/tau_N), which is 1/e rather than 1/2 at tau_N = t, and the
    lognormal is parameterised by the MEAN diameter, so the median particle is
    slightly smaller.  Both are marked so neither is mistaken for the other.
    """
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
                         'axes.labelsize': 13.5, 'axes.linewidth': 1.6,
                         'axes.grid': False, 'pdf.fonttype': 42})
    grid = table['temperature_K']
    blue, orange, dark = '#3C3C8C', '#C0552F', '#1A1A1A'
    red = '#B54535'
    cold, warm = table['window'] if table['window'] else (np.nan, np.nan)
    blocking = float(blocking_K)
    modelled = table['blocking_temperature_K']

    # The panel heights follow the y-ranges so a kBT is the same height in
    # both; at z = 1 the ceiling curve runs to -7.7 and would be clipped by
    # a fixed range.
    floor = min(float(np.min(-table['annealed_kBT'])),
                float(np.min(-table['frozen_ceiling_kBT'])))
    top_span = (min(-8.1, float(np.min(table['total_self'])) - 0.8), 2.7)
    split_span = (floor - 0.5, 0.4)
    heights = [top_span[1] - top_span[0], split_span[1] - split_span[0], 5.0]
    figure, (main, split, lower) = plt.subplots(
        3, 1, figsize=(10.4, 15.2), sharex=True,
        gridspec_kw={'height_ratios': heights})
    for axes in (main, split, lower):
        axes.axvspan(cold, warm, color='#F4EAA2', alpha=.75, zorder=0)
        axes.axvline(blocking, color='#C9A227', lw=2.0, ls=(0, (5, 2)), zorder=1)

    # ---- (a) the two attractions, their sum, and the level it must clear --
    main.plot(grid, table['dipolar_self'], color=orange, lw=3.4, zorder=4,
              label=r'dipolar   $-[(1-b^2)D + E_{frozen}/k_BT]$')
    main.plot(grid, table['vdw_kBT'], color=blue, lw=3.4, zorder=4,
              label=r'van der Waals   $U_{vdW}/k_BT$')
    if show_threshold:
        level = -table['threshold_sum_kBT']
        main.axhline(level, color=red, lw=2.2, ls=(0, (6, 3)), zorder=3,
                     label=rf'aggregation threshold   ${level:.2f}\,k_BT$')
    main.plot(grid, table['total_self'], color=dark, lw=3.8, zorder=6,
              label='total = van der Waals + dipolar')
    main.annotate('dipolar WEAKENS on cooling\n'
                  '(blocking removes the re-sampling)',
                  xy=(203, table['dipolar_self'][0]), xytext=(211, -0.30),
                  fontsize=11.5, color=orange, ha='left', va='center',
                  arrowprops=dict(arrowstyle='->', color=orange, lw=1.6))
    main.annotate('van der Waals STRENGTHENS on cooling',
                  xy=(213, float(np.interp(213, grid, table['vdw_kBT']))),
                  xytext=(237, -1.25), fontsize=11.5, color=blue, ha='left',
                  va='center',
                  arrowprops=dict(arrowstyle='->', color=blue, lw=1.6))
    main.text(.5 * (cold + warm), -6.35, f'{cold:.0f}-{warm:.0f} K',
              fontsize=12.5, color='#8A6D1F', ha='center', va='center',
              fontweight='bold')
    main.text(blocking - 1.6, -3.4,
              rf'$T_B$ = {blocking:.0f} K', rotation=90,
              fontsize=11.5, color='#8A6D1F', ha='right', va='center',
              fontweight='bold')
    main.text(298, -0.30, f'ensemble half blocked ($b$ = 1/2) at {modelled:.0f} K',
              fontsize=10, color='0.4', ha='right', va='center')
    main.set(ylabel=r'Interaction free energy  ($k_BT$)', ylim=top_span)
    main.legend(loc='upper left', bbox_to_anchor=(.005, .995),
                frameon=False, fontsize=11, ncol=2, columnspacing=2.2)
    main.set_title(f"(a)  The two attractions and their sum   "
                   f"(coordination z = {table['coordination']})",
                   loc='left', fontsize=12.5, fontweight='bold')

    # ---- (b) where the dipolar term goes, same sign convention ------------
    split.axhline(0, color='black', lw=1.0, zorder=2)
    split.plot(grid, -table['annealed_kBT'], color=orange, lw=3.2, zorder=4,
               label=r'$-(1-b^2)\,D$   still being re-sampled')
    split.fill_between(grid, -table['frozen_kBT'], -table['frozen_ceiling_kBT'],
                       color=dark, alpha=.16, lw=0, zorder=2)
    split.plot(grid, -table['frozen_ceiling_kBT'], color=dark, lw=1.6,
               ls=(0, (5, 2)), zorder=3,
               label=r'$-E_{frozen}/k_BT$   ceiling, if every moment froze bonded')
    split.plot(grid, -table['frozen_kBT'], color=dark, lw=3.2, zorder=5,
               label=r'$-E_{frozen}/k_BT$   self-consistent')
    peak_frozen = float(np.max(table['frozen_kBT']))
    split.annotate('the self-consistent frozen term never goes below '
                   f'{-peak_frozen:.3f} ' r'$k_BT$',
                   xy=(226, -float(np.interp(226, grid, table['frozen_kBT']))),
                   xytext=(232, -1.45), fontsize=10, color=dark, ha='left',
                   arrowprops=dict(arrowstyle='->', color=dark, lw=1.3))
    split.set(ylabel=r'Contribution  ($k_BT$)', ylim=split_span)
    split.legend(loc='lower left', bbox_to_anchor=(.005, .01), frameon=False,
                 fontsize=10.5)
    split.set_title('(b)  The dipolar term split: re-sampled vs inherited',
                    loc='left', fontsize=12.5, fontweight='bold')

    # ---- (c) the switches --------------------------------------------------
    lower.plot(grid, table['blocked'], color=orange, lw=1.8, ls=(0, (5, 2)),
               label='$b$  one moment blocked')
    lower.plot(grid, table['blocked_weight'], color=orange, lw=3.0,
               label='$b^2$  both blocked')
    lower.fill_between(grid, 0, table['aggregated_fraction'], color=blue,
                       alpha=.25, lw=0)
    lower.plot(grid, table['aggregated_fraction'], color=blue, lw=3.0,
               label='$x$  particles in the dense phase')
    peak = table['aggregated_fraction'].max()
    lower.text(.5 * (cold + warm), peak + .06, f'peak {100 * peak:.0f} %',
               fontsize=10.5, color=blue, ha='center', va='bottom')
    lower.set(xlabel='Temperature (K)', ylabel='Fraction',
              xlim=(200, 300), ylim=(0, 1.05))
    lower.legend(loc='upper right', frameon=False, fontsize=10.5)
    lower.set_title(f"(c)  Neel blocking at {table['dwell_s']:.0f} s dwell, "
                    f"CV {table['size_cv'] * 100:.2f} %, and the aggregated "
                    "fraction it produces", loc='left', fontsize=12.5,
                    fontweight='bold')

    figure.subplots_adjust(left=.10, right=.975, top=.775, bottom=.045,
                           hspace=.15)
    add_dipole_cartoons(figure, main, table, (cold, warm))
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'two_energies_z{table["coordination"]}.{extension}',
                       dpi=300, bbox_inches='tight')
    plt.close(figure)


def energy_decomposition(energies: dict, dwell_s: float, size_cv: float,
                         tilt_deg: float = 15.5,
                         grid=np.arange(200.0, 300.01, 0.5),
                         loss_fraction: float = 1.0) -> dict:
    """Every term of eps(T) on one temperature grid, ready to plot.

    Separates the two things cooling does, because they point opposite ways:
    `vdw_kBT` and `dipole_kBT` are the well depths, which grow monotonically
    on cooling simply because they are fixed joule energies divided by kBT,
    while `blocked_weight` is the fraction of that dipolar depth the
    condensate can no longer collect.
    """
    rows = [dense_phase_epsilon(float(t), energies, dwell_s, size_cv,
                                loss_fraction)
            for t in grid]
    cost = registration_cost_kBT(tilt_deg)
    collected = np.array([(1.0 - loss_fraction * r['blocked_weight'])
                          * r['dipole_kBT'] for r in rows])
    vdw = np.array([r['vdw_kBT'] for r in rows])
    return dict(temperature_K=np.asarray(grid, float), rows=rows,
                vdw_kBT=vdw,
                dipole_kBT=np.array([r['dipole_kBT'] for r in rows]),
                collected_kBT=collected,
                blocked=np.array([r['blocked_fraction'] for r in rows]),
                blocked_weight=np.array([r['blocked_weight'] for r in rows]),
                net_kBT=vdw + collected - cost,
                net_unfrozen_kBT=vdw
                + np.array([r['dipole_kBT'] for r in rows]) - cost,
                registration_kBT=cost, tilt_deg=tilt_deg,
                dwell_s=dwell_s, size_cv=size_cv,
                loss_fraction=loss_fraction)


def plot_energy_vs_temperature(decomposition: dict, threshold_kBT: float,
                               out: Path, observed=(250.0, 270.0)):
    """Three panels on a shared temperature axis: terms, freeze, net.

    Lines only.  An earlier version stacked translucent areas for the three
    terms and they were unreadable where they overlapped; the mechanism is a
    solid curve peeling away from a dashed one, which needs no fill at all.
    The single filled region is the sliver where the net clears the threshold.

    Sign convention throughout: POSITIVE binds.  So the van der Waals curve is
    -U_vdW/kBT, the dipolar curve is +ln<exp(-U_dd/kBT)> -- attractive despite
    <U_dd> being exactly zero, because exp is convex -- and the registration
    entropy is the one negative term.
    """
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
                         'axes.labelsize': 13, 'axes.linewidth': 1.6,
                         'axes.grid': False, 'pdf.fonttype': 42})
    grid = decomposition['temperature_K']
    cost = decomposition['registration_kBT']
    net = decomposition['net_kBT']
    blue, orange, grey, dark = '#3C3C8C', '#C0552F', '0.45', '#1A1A1A'

    figure, panels = plt.subplots(3, 1, figsize=(8.8, 11.6), sharex=True,
                                  gridspec_kw={'height_ratios': [1.15, .7, 1.05]})
    terms, freeze, result = panels
    for axes in panels:
        for edge in observed:
            axes.axvline(edge, color='#C9A227', lw=1.4, ls=(0, (4, 3)),
                         zorder=1)

    # (a) every term of eps, signed, as lines
    terms.axhline(0, color='black', lw=1.0, zorder=2)
    terms.plot(grid, decomposition['dipole_kBT'], color=orange, lw=2.0,
               ls=(0, (5, 2)), zorder=3,
               label=r'dipolar if free to re-sample   $+\ln\langle e^{-U_{dd}/k_BT}\rangle$')
    terms.plot(grid, decomposition['collected_kBT'], color=orange, lw=3.4,
               zorder=5,
               label=r'dipolar actually collected   $(1-b^2)\times$ the above')
    terms.plot(grid, decomposition['vdw_kBT'], color=blue, lw=3.4, zorder=4,
               label=r'van der Waals   $-U_{vdW}/k_BT$')
    terms.plot(grid, np.full_like(grid, -cost), color=grey, lw=3.4, zorder=4,
               label=rf'registration entropy   $-{cost:.2f}\,k_BT$ '
                     rf'({decomposition["tilt_deg"]:g}$\degree$ cone)')
    terms.annotate('freezing peels the solid curve\n'
                   'away from the dashed one:\n'
                   'that gap is the whole mechanism',
                   xy=(232, 3.55), xytext=(244, -2.3), fontsize=10.5,
                   color=orange, ha='left',
                   arrowprops=dict(arrowstyle='->', color=orange, lw=1.4,
                                   connectionstyle='arc3,rad=-.25'))
    terms.set(ylabel=r'Contribution to $\epsilon$  ($k_BT$)', ylim=(-5.6, 8.4))
    terms.legend(loc='lower left', frameon=False, fontsize=10)
    terms.set_title('(a)  The four terms.  Positive binds; '
                    r'$\langle U_{dd}\rangle$ is exactly zero',
                    loc='left', fontsize=12.5, fontweight='bold')

    # (b) the only term that points the other way
    freeze.plot(grid, decomposition['blocked_weight'], color=orange, lw=3.4,
                zorder=4, label=r'$b^2$   both moments blocked')
    freeze.plot(grid, decomposition['blocked'], color=orange, lw=2.0,
                ls=(0, (5, 2)), zorder=3, label=r'$b$   one moment blocked')
    freeze.set(ylabel='Blocked fraction', ylim=(0, 1.06))
    freeze.legend(loc='upper right', frameon=False, fontsize=10)
    freeze.set_title(f"(b)  Néel freezing at {decomposition['dwell_s']:.0f} s "
                     f"dwell, CV {decomposition['size_cv'] * 100:.2f} %",
                     loc='left', fontsize=12.5, fontweight='bold')

    # (c) the sum, against the threshold
    result.plot(grid, decomposition['net_unfrozen_kBT'], color=grey, lw=2.0,
                ls=(0, (2, 2)), zorder=3,
                label='net if nothing froze (ordinary colloid)')
    result.axhline(threshold_kBT, color='#B54535', lw=2.2, ls=(0, (6, 3)),
                   zorder=4,
                   label=rf'condensation threshold  {threshold_kBT:.2f} $k_BT$')
    result.fill_between(grid, threshold_kBT, net, where=net > threshold_kBT,
                        color='#B54535', alpha=.30, lw=0, zorder=2,
                        interpolate=True)
    result.plot(grid, net, color=dark, lw=3.6, zorder=5,
                label=r'net cohesion  $\epsilon$  (sum of panel a)')

    inside = net > threshold_kBT
    if inside.any():
        edges = grid[inside]
        for edge in (edges.min(), edges.max()):
            result.plot([edge, edge], [0, threshold_kBT], color='#B54535',
                        lw=1.2, ls=':', zorder=4)
        result.annotate('condensing\n'
                        f'{edges.min():.0f}-{edges.max():.0f} K',
                        xy=(.5 * (edges.min() + edges.max()), 1.95),
                        fontsize=10.5, color='#B54535', fontweight='bold',
                        ha='center')
        result.annotate('peak clears the threshold by only '
                        f'{net.max() - threshold_kBT:.3f} '
                        r'$k_BT$', xy=(203, 3.62), fontsize=10,
                        color='0.3', ha='left')
    result.set(xlabel='Temperature (K)',
               ylabel=r'Net cohesion  $\epsilon$  ($k_BT$ per bond)',
               xlim=(200, 300), ylim=(0, 6.8))
    result.legend(loc='lower right', frameon=False, fontsize=10)
    result.set_title('(c)  Sum: non-monotonic, crosses the threshold twice',
                     loc='left', fontsize=12.5, fontweight='bold')

    figure.suptitle('Where the cooling window comes from\n'
                    'dashed gold lines = observed aggregation window, '
                    '250-270 K', fontsize=13.5)
    figure.subplots_adjust(left=.11, right=.97, top=.92, bottom=.06, hspace=.17)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'energy_vs_temperature.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dwell-s', type=float, default=150.0)
    parser.add_argument('--size-cv', type=float, default=0.10)
    parser.add_argument('--tilt-deg', type=float, default=15.5)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    arguments = parser.parse_args()
    out = arguments.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    temperatures = np.arange(200.0, 300.0 + 1e-9, 1.0)
    rows, curves, critical = sweep(temperatures, arguments.dwell_s,
                                   arguments.size_cv, arguments.tilt_deg)

    with (out / 'binodal.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 13,
                         'axes.labelsize': 16, 'axes.linewidth': 1.8,
                         'axes.grid': False, 'pdf.fonttype': 42})
    figure, panels = plt.subplots(1, 2, figsize=(13.4, 7.6))
    energy, diagram = panels
    tag = f'{arguments.tilt_deg:g}deg'
    colours = {('inherited', 'none'): '#B54535',
               ('inherited', tag): '#7D2E68',
               ('converged', 'none'): '#287A96',
               ('converged', tag): '#3C3C8C'}
    energy.axvspan(250, 270, color='#F4EAA2', alpha=.85, zorder=0,
                   label='observed aggregation window')
    diagram.axhspan(250, 270, color='#F4EAA2', alpha=.85, zorder=0)
    for key, series in curves.items():
        grid = [r['temperature_K'] for r in series]
        label = (f"{key[0]} vdW, "
                 + ('no registry cost' if key[1] == 'none'
                    else f'-{series[0]["registration_cost_kBT"]:.1f} $k_BT$ registry'))
        energy.plot(grid, [r['epsilon_net_kBT'] for r in series],
                    color=colours[key], lw=2.6, label=label)
        finite = [r for r in series if r['two_phase']]
        if finite:
            diagram.semilogx([r['phi_dilute'] for r in finite],
                             [r['temperature_K'] for r in finite],
                             color=colours[key], lw=2.6, label=label)
            diagram.semilogx([r['phi_dense'] for r in finite],
                             [r['temperature_K'] for r in finite],
                             color=colours[key], lw=2.6)
    energy.axhline(critical_epsilon_kBT(critical), color='black', lw=2.0,
                   ls=(0, (5, 2)), label=r'condensation threshold $\epsilon_c$')
    energy.set(xlabel='Temperature (K)', xlim=(200, 300),
               ylabel=r'Dense-phase cohesion / $k_BT$ per bond')
    energy.set_title('Cohesion against the condensation threshold', fontsize=13)
    energy.legend(loc='upper left', bbox_to_anchor=(0, -.14), frameon=False,
                  fontsize=10)
    diagram.set(xlabel=r'Volume fraction $\phi$', ylabel='Temperature (K)',
                ylim=(200, 300), xlim=(1e-7, 0.7))
    diagram.set_title('Fluid-fluid binodal (re-entrant)', fontsize=13)
    diagram.text(.03, .03, 'inside a dome: two phases', fontsize=11,
                 transform=diagram.transAxes, ha='left', va='bottom')
    figure.suptitle('Blocked-dipole condensation of 16 nm nanocubes\n'
                    rf'dwell {arguments.dwell_s:g} s, size $CV$ = '
                    rf'{arguments.size_cv * 100:.0f} %, 3 nm gap (measured)',
                    fontsize=14)
    figure.subplots_adjust(left=.08, right=.97, top=.85, bottom=.30, wspace=.24)
    for extension in ('png', 'pdf'):
        figure.savefig(out / f'binodal_phase_diagram.{extension}', dpi=300,
                       bbox_inches='tight')
    plt.close(figure)
    plot_energy_and_binodal(curves, critical, out, arguments.dwell_s,
                            arguments.size_cv, arguments.tilt_deg)
    fit = fit_size_cv(SAMPLE_VOLUME_FRACTION, 250.0, 270.0, 100.0,
                      pair_energies('converged'))
    plot_fitted_diagram(fit, out)
    decomposition = energy_decomposition(pair_energies('converged'),
                                         fit['dwell_s'], fit['size_cv'],
                                         arguments.tilt_deg)
    plot_energy_vs_temperature(decomposition,
                               fit['epsilon_threshold_kBT'], out)
    (out / 'fit.json').write_text(json.dumps(
        {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
         for k, v in fit.items() if not isinstance(v, np.ndarray)},
        indent=2), encoding='utf-8')
    print(json.dumps({k: float(v) for k, v in fit.items()
                      if isinstance(v, (int, float, np.floating))}, indent=2))

    summary = dict(
        critical_attraction=critical,
        critical_epsilon_kBT=critical_epsilon_kBT(critical),
        registration_cost_kBT=registration_cost_kBT(arguments.tilt_deg),
        tilt_deg=arguments.tilt_deg, dwell_s=arguments.dwell_s,
        size_cv=arguments.size_cv,
        epsilon_peak={key[0] + '/' + key[1]:
                      max(r['epsilon_net_kBT'] for r in series)
                      for key, series in curves.items()},
        epsilon_peak_temperature_K={
            key[0] + '/' + key[1]:
            max(series, key=lambda r: r['epsilon_net_kBT'])['temperature_K']
            for key, series in curves.items()},
        two_phase_temperatures={key[0] + '/' + key[1]:
                                sum(1 for r in series if r['two_phase'])
                                for key, series in curves.items()})
    (out / 'model_config.json').write_text(json.dumps(summary, indent=2),
                                           encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
