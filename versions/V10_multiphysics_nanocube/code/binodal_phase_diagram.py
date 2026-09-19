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


def dense_phase_epsilon(temperature_K: float, energies: dict,
                        dwell_s: float = 150.0, size_cv: float = 0.10) -> dict:
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
                epsilon_kBT=vdw + (1.0 - blocked ** 2) * dipole,
                epsilon_unblocked_kBT=vdw + dipole, epsilon_frozen_kBT=vdw)


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
                energies, bracket=(0.0, 0.075)):
    """CV that reproduces both observed edges at a fixed dwell.

    Pinning the upper edge fixes the cut level at eps(upper_edge), which makes
    the WIDTH independent of the registry cost: the lower edge is simply where
    the cohesion returns to that level.  CV is therefore the only thing
    setting the width, and the registry cost that falls out is a free check
    against the tilt cone the measured 19 nm spacing allows.
    """
    from scipy.optimize import brentq

    threshold, _ = epsilon_threshold(volume_fraction)
    grid = np.arange(200.0, 300.01, 0.25)

    def edges(size_cv):
        cohesion = np.array([dense_phase_epsilon(float(t), energies, dwell_s,
                                                 size_cv)['epsilon_kBT']
                             for t in grid])
        level = float(np.interp(upper_edge_K, grid, cohesion))
        cold = grid < grid[int(np.argmax(cohesion))]
        if not cold.any() or cohesion[cold].min() > level:
            return None, level, cohesion
        return float(np.interp(level, cohesion[cold], grid[cold])), level, cohesion

    def miss(size_cv):
        low, _, _ = edges(size_cv)
        return 1e3 if low is None else low - lower_edge_K

    size_cv = float(brentq(miss, *bracket, xtol=1e-4))
    low, level, cohesion = edges(size_cv)
    cost = level - threshold
    tilt = float(brentq(lambda t: registration_cost_kBT(t) - cost, 5.0, 60.0))
    return dict(size_cv=size_cv, dwell_s=dwell_s, lower_edge_K=low,
                upper_edge_K=upper_edge_K, registry_cost_kBT=cost,
                fitted_tilt_deg=tilt, geometric_tilt_deg=15.5,
                geometric_cost_kBT=registration_cost_kBT(15.5),
                epsilon_threshold_kBT=threshold, temperatures_K=grid,
                cohesion_kBT=cohesion, volume_fraction=volume_fraction)


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
