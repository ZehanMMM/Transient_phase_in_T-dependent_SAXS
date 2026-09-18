"""Minimal runnable example of the npvdw public API.

python example_usage.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import npvdw as V

EDGE_NM = 16.0
GAP_NM = 3.0
TEMPERATURE_K = 298.15

print("available parameter sets:", V.available_parameters())

# Recommended working configuration for this project: the SI-mesh calibration
# on its own 27/386 discretisation, transferred to a different size with plan 3
# (repulsive prefactor frozen at the 13.37 nm / 3 kcal/mol reference).  See
# FINDINGS.md sections 4 and C2 for why, and why plan 1 must not be used above
# the reference size.
params = replace(
    V.load_parameters("singh_calibrated_si_mesh"),
    kw_reference_edge_nm=13.37,
    kw_reference_hamaker_kcal_per_mol=V.SINGH_HAMAKER_KCAL_PER_MOL,
).with_hamaker_J(2.0e-20)
print("using %s with A = %.3e J (%.4f kcal/mol), K_W(%.1f nm) = %.4e "
      "nm^6 kcal/mol" % (params.label, params.hamaker_J,
                         params.hamaker_kcal_per_mol, EDGE_NM,
                         params.kw(EDGE_NM)))
print("eps1 = %.4f, eps2 = %.4f, beta = %.4f nm, mesh = %s/%s, r2 = %s"
      % (params.epsilon1, params.epsilon2, params.beta_nm,
         params.volume_scheme, params.surface_scheme, params.r2_mode))
print()

# 1. face-to-face at a prescribed surface separation ------------------------
print("1. co-oriented pair, centre line along [100], 3 nm surface separation")
radius = V.centre_distance_for_gap(EDGE_NM, GAP_NM, [1.0, 0.0, 0.0])
i = V.Placement.create(EDGE_NM, [0.0, 0.0, 0.0])
j = V.Placement.create(EDGE_NM, [radius, 0.0, 0.0])
result = V.pair_energy(i, j, params, temperature_K=TEMPERATURE_K)
print("   centre distance  %.6f nm   GJK gap %.6f nm   overlap %s"
      % (result["centre_distance_nm"], result["gap_nm"], result["overlap"]))
for name in ("u_attr", "u_rep", "u_pair"):
    print("   %-7s %+12.6f kcal/mol   %+12.5e J   %+9.4f kBT"
          % (name, result["%s_kcalmol" % name], result["%s_J" % name],
             result["%s_kBT" % name]))
print("   U_pair/2 (per particle, isolated pair only) %+9.4f kBT"
      % result["u_pair_per_particle_kBT"])
print()

# 2. arbitrary independent orientations -------------------------------------
print("2. particle j rotated 37 deg about [0,1,1], centre line along [1,1,0.3]")
axis = np.array([0.0, 1.0, 1.0])
axis /= np.linalg.norm(axis)
angle = np.deg2rad(37.0)
K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
direction = np.array([1.0, 1.0, 0.3])
direction /= np.linalg.norm(direction)
radius = V.centre_distance_for_gap(EDGE_NM, GAP_NM, direction, rotation_b=rotation)
j = V.Placement.create(EDGE_NM, radius * direction, rotation)
result = V.pair_energy(i, j, params, temperature_K=TEMPERATURE_K)
print("   centre distance  %.6f nm   GJK gap %.6f nm" % (result["centre_distance_nm"], result["gap_nm"]))
print("   closest points   i %s   j %s"
      % (np.round(result["closest_point_i_nm"], 4), np.round(result["closest_point_j_nm"], 4)))
print("   U_pair %+.6f kcal/mol = %+.4f kBT" % (result["u_pair_kcalmol"], result["u_pair_kBT"]))
print()

# 3. overlap is rejected by the hard core -----------------------------------
print("3. overlapping pair")
bad = V.pair_energy(i, V.Placement.create(EDGE_NM, [0.8 * EDGE_NM, 0, 0]), params)
print("   overlap %s   hard_core_rejected %s   U_pair %s"
      % (bad["overlap"], bad["hard_core_rejected"], bad["u_pair_kcalmol"]))
print()

# 4. the well, and the geometry diagnostics --------------------------------
print("4. face-to-face well along [100]")
gap, depth, interior = V.find_well(EDGE_NM, params, samples=40)
print("   interior well: %s   gap %.4f nm   depth %+.5f kcal/mol = %+.4f kBT"
      % (interior, gap, depth, V.kcalmol_to_kBT(depth, TEMPERATURE_K)))
print()

# 5. a whole curve through the reusable kernels -----------------------------
print("5. curve via energy_kernels (geometry computed once, parameters free)")
gaps = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
kernels = V.energy_kernels(EDGE_NM, gaps, params)
attraction, repulsion, total = V.kernel_energies(
    kernels, params.epsilon1, params.epsilon2, params.beta_nm
)
print("   gap_nm   U_attr      U_rep       U_pair   [kcal/mol]")
for g, a, r, t in zip(gaps, attraction, repulsion, total):
    print("   %6.2f  %+10.5f  %+10.5f  %+10.5f" % (g, a, r, t))
print()
print("   the same numbers with the overall scale s = 5")
a5, r5, t5 = V.kernel_energies(kernels, params.epsilon1, params.epsilon2,
                               params.beta_nm, overall_scale=5.0)
print("   U_pair x5 =", np.round(t5, 5))
