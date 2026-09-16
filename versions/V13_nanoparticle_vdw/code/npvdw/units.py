"""Unit system for the nanoparticle vdW module.

INTERNAL UNIT SYSTEM (used everywhere inside npvdw):
    length  : nm
    energy  : kcal/mol   (molar energy per interacting pair)
    area    : nm^2
    volume  : nm^3

Rationale: Singh et al. state every vdW parameter in this system
(A = 3 kcal/mol, beta = 9.56 nm, K_W = 2.5e5 nm^6 kcal/mol), so keeping it
internally avoids conversion errors while reproducing the reference.
Conversion to J / kBT happens only at the output boundary.

"1 kcal/mol" is a molar energy.  One pair interaction of energy
U [kcal/mol] corresponds to U * KCAL_PER_MOL_IN_J joule for that single pair.
"""
from __future__ import annotations

# CODATA 2018 / SI-2019 exact values.
AVOGADRO = 6.022_140_76e23           # 1/mol  (exact)
BOLTZMANN_J_PER_K = 1.380_649e-23    # J/K    (exact)
CAL_IN_J = 4.184                     # thermochemical calorie (exact by definition)

# Energy of one pair when the molar energy is 1 kcal/mol.
KCAL_PER_MOL_IN_J = 1.0e3 * CAL_IN_J / AVOGADRO   # 6.947695e-21 J

# Singh et al. Hamaker constant A = 3 kcal/mol.
SINGH_HAMAKER_KCAL_PER_MOL = 3.0
SINGH_HAMAKER_J = SINGH_HAMAKER_KCAL_PER_MOL * KCAL_PER_MOL_IN_J  # 2.084309e-20 J


def kcalmol_to_J(value):
    """Molar kcal/mol -> joule per single pair interaction."""
    return value * KCAL_PER_MOL_IN_J


def J_to_kcalmol(value):
    return value / KCAL_PER_MOL_IN_J


def kcalmol_to_kBT(value, temperature_K):
    return kcalmol_to_J(value) / (BOLTZMANN_J_PER_K * temperature_K)


def J_to_kBT(value, temperature_K):
    return value / (BOLTZMANN_J_PER_K * temperature_K)


def hamaker_J_to_kcalmol(hamaker_J):
    """Convert a Hamaker constant in J to the module's kcal/mol convention."""
    return J_to_kcalmol(hamaker_J)
