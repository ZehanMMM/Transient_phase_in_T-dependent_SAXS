# Blocked-dipole condensation: energies, mechanism, and the fitted phase diagram

Produced by `code/binodal_phase_diagram.py`. Verified by
`code/test_binodal_phase_diagram.py`, 17 tests.

This model replaces the superlattice framing used in
`outputs/configuration_averaged_assembly/`. The revision history is at the end
of this file, because three earlier conclusions in this project were wrong and
the reasoning is not reconstructible without them.

---

## 1. Which van der Waals is in use

**The converged one.** `CONVERGED_VDW_J = -9.156e-21 J` for the face pair at
the 3 nm gap, against the inherited 4³ voxel value of −4.883e-21 J. The
inherited sum is under-converged by a factor 1.875: its 4 nm voxel edge cannot
resolve a 3 nm gap on a 1/r⁶ kernel, and its `vdw_d2_floor_m2 = 1e-19` guard
(3.2 nm) truncates exactly the dominant near-contact term. Refining the *same*
sharp-cube Hamaker integral converges to −9.156e-21 J.

`geometry_model.py` is **not modified**. The converged value is a module-level
constant here, and `pair_energies('inherited')` is retained so the sensitivity
stays visible.

---

## 2. What energies the model contains

Everything is per face-to-face bond, in k_BT, at the **measured** geometry:
19 nm centre spacing, 16 nm cube, hence a 3 nm surface gap.

| term | origin | 300 K | 250 K | 200 K |
|---|---|---|---|---|
| van der Waals | converged sharp-cube Hamaker, A = 2.0e-20 J | −2.21 | −2.65 | −3.32 |
| dipolar **Mayer** term, ln⟨e^(−U_dd/k_BT)⟩ | inherited 64-state ⟨111⟩ spectrum | −4.40 | −5.64 | −7.53 |
| registration entropy | measured ±15.5° tilt cone | **+4.43** | +4.43 | +4.43 |

and one temperature-dependent switch:

| quantity | meaning | 300 K | 250 K | 200 K |
|---|---|---|---|---|
| b²(T) | fraction of pairs with **both** moments Néel-blocked | 0.008 | 0.19 | 0.81 |

combined as

```
eps(T) = -U_vdW/kBT  +  (1 - b(T)^2) * ln<exp(-U_dd/kBT)>  -  registration cost
```

Three features of this expression carry the whole result.

**(a) The dipolar term is an attraction even though its mean is exactly zero.**
Summed over the 64 lab ⟨111⟩ pair states, U_dd is identically 0 — asserted in
the tests to 1e-12. But condensation is set by the Mayer factor, and exp is
convex, so ⟨exp(−U_dd/k_BT)⟩ ≫ 1. That contributes 4.4 k_BT at 300 K rising to
7.5 k_BT at 200 K, i.e. **twice the van der Waals term and more**. No magnetic
field is needed: the attraction comes from the *fluctuation* of the dipolar
energy, not its average.

**(b) That attraction is only collected if the moment can be re-sampled.** In
the dilute fluid a particle rotates freely (τ_B ~ µs), so it always is. Inside
the condensate it is not: at 19 nm spacing each particle has 9.50 nm of half
extent against a rounded-cube support height of 8.00 nm along ⟨100⟩, 10.69 nm
along ⟨110⟩ and 12.76 nm along ⟨111⟩. So the dense phase **must** be
face-registered — edge and vertex contacts do not fit — and a cube can tilt
only **±15.5°** before its corner meets a neighbour, against the 54.7° needed
to bring a different ⟨111⟩ onto the bond and the 90° needed to swap faces. The
only remaining way to re-sample is a Néel flip.

**(c) The loss is linear in b², not b.** In a condensate every bond is realised
at once rather than sampled, so the dipolar term is lost in proportion. It is
b² rather than b because one mobile moment already re-samples the pair — an
exact identity for a ⟨100⟩ bond, proved in `configuration_averaged_assembly`:
the conditional partition function is the same for all eight frozen partner
states.

The registration term is the orientational free-energy cost of holding a pair
in face registry. Twisting about the bond does not change the support height
along it, so twist is geometrically free at a face contact, and the allowed
fraction of SO(3) per particle is 6 face normals times the tilt cone,
f = 6(1 − cos θ)/2 = 0.109 at 15.5°, paid twice: −2 ln f = 4.43 k_BT.

---

## 3. Why it aggregates on cooling — and why it stops

Cooling does two opposite things at once.

**It deepens every well.** Both the van der Waals and the dipolar terms are
fixed energies in joules, so dividing by k_BT makes them grow monotonically on
cooling: −2.21 → −3.32 and −4.40 → −7.53 k_BT between 300 and 200 K. On its
own this would make the suspension aggregate more and more as it cools, which
is the ordinary colloidal behaviour reported for comparable systems.

**It freezes the moments.** τ_N = τ₀ exp(ΔE/k_BT) crosses the time the sample
spends at each temperature, and past that point the moment can no longer be
re-sampled. Because ΔE ∝ D³, a 5 % spread in diameter is a 15 % spread in the
barrier, and on an exponential that smears the freeze-in over tens of kelvin —
there is no step anywhere. b² climbs 0.008 → 0.19 → 0.81 from 300 to 200 K.

The net cohesion is therefore **non-monotonic**:

| T | vdW | (1−b²)·dipolar | eps | eps − registration |
|---|---|---|---|---|
| 300 K | 2.21 | 4.36 | 6.57 | 2.21 |
| 280 K | 2.37 | 4.68 | 7.04 | 2.67 |
| **270 K** | 2.46 | 4.76 | 7.21 | **2.84** |
| **260 K** | 2.55 | 4.73 | 7.29 | **2.92** |
| **250 K** | 2.65 | 4.56 | 7.21 | **2.84** |
| 240 K | 2.76 | 4.19 | 6.96 | 2.59 |
| 220 K | 3.01 | 2.91 | 5.93 | 1.56 |
| 200 K | 3.32 | 1.41 | 4.72 | 0.35 |

The condensation threshold at the sample concentration is 2.84 k_BT, and the
cohesion peaks at 2.92 k_BT at 260 K. So:

- **at 300 K** the cohesion is 2.21 k_BT, below threshold — the suspension is a
  single phase, colloidal;
- **between 270 and 250 K** it exceeds the threshold — two phases, the observed
  aggregate;
- **below 250 K** the moments freeze faster than the wells deepen, the dipolar
  contribution collapses from 4.8 to 1.4 k_BT, and the suspension **redissolves**
  even though every individual energy is still growing.

Both edges are crossings of the same curve, one on the way up and one on the
way down. That is the whole mechanism, and it is why the binodal dome closes at
both ends instead of opening downwards as an ordinary colloid's would.

---

## 4. The fitted diagram

`fitted_phase_diagram.png`. One panel, temperature vertical and shared: volume
fraction on the bottom axis with the binodal, cohesion on the top axis.

| | | |
|---|---|---|
| **fitted** | size CV | **4.85 %** |
| fixed | dwell | 100 s |
| fixed | φ (effective) | 0.0209 = 1.25 % core = 65 mg/mL |
| matched | window | 250–270 K, both edges |

Concentration bookkeeping: the free energy uses the **effective** fraction
built on the measured 19 nm spacing, which exceeds the inorganic core fraction
by (19/16)³ = 1.67. Both are reported by `concentration_table`.

**The registration cost was not fitted.** Pinning the upper edge fixes the cut
level at eps(270 K), which makes the window *width* independent of that cost —
the lower edge is simply where the cohesion returns to the same level. CV alone
sets the width, so the cost falls out as a free quantity:

| | value | equivalent tilt cone |
|---|---|---|
| needed by the fit | 4.37 k_BT | **15.74°** |
| from the measured 19 nm spacing | 4.43 k_BT | **15.50°** |
| difference | 0.06 k_BT | **+0.24°** |

The sign is right as well as the magnitude: soft ligands can only widen the
cone relative to rigid cores. This is the one genuinely independent check in
the construction.

**Free energy.** Carnahan–Starling hard spheres plus a mean-field attraction,
`f/k_BT = ln η + η(4−3η)/(1−η)² − aη` with `a = (z_max/2)·eps/η_cp`, coexistence
from equal chemical potential and pressure. The critical attraction comes out
at **ε_c = 2.28 k_BT**, which matches the Noro–Frenkel expectation of 2–3 k_BT
for a short-range attractive colloid — a consistency check on the free energy
that was not imposed.

---

## 5. What would falsify this

- **Size distribution.** CV = 4.85 % is now a prediction. At CV = 10 % the
  window widens past 45 K against the observed 20 K (asserted in the tests).
  A TEM CV of 9–10 % kills the model.
- **Ramp rate.** The lower edge is set by the dwell time and should move
  systematically with it — roughly 273 K at 12 s of dwell and 240 K at 300 s.
  The upper edge is a thermodynamic crossing and should not. If both edges move
  together with rate, the mechanism is wrong.
- **Concentration.** The window should close entirely below φ_eff ≈ 0.012 and
  widen above it. A dilution series is a direct test of the dome.

---

## 6. What is not in the model

- **Nucleation, and therefore the observed ~10 K hysteresis.** The hysteresis
  grows with ramp rate, so it is a kinetic lag rather than a boundary; this
  module gives the quasi-static boundaries the loops should straddle.
- **The capillary wall.** The anisotropic 2D pattern shows nucleation is
  heterogeneous and wall-oriented. Nothing here describes that.
- **Ramp trajectories.** Everything is evaluated at a fixed dwell, not along a
  path.
- **Structure of the dense phase** beyond its spacing. The free energy is
  van der Waals level: it gives the topology and the concentration scale, not
  quantitative binodal compositions. `z_max = 6` and `η_cp = 0.64` are stated,
  not fitted.

---

## 7. Revision history — three conclusions that were wrong

Recorded because the reasoning is not reconstructible without them.

1. **The superlattice framing was wrong.** Everything in
   `configuration_averaged_assembly` — coordination registry, twist entropy,
   tip and edge contacts, the 192-state master equation — assumed a crystalline
   superlattice. The SAXS evidence is a single **broad** peak, so the dense
   phase is amorphous and the right framework is fluid–fluid phase separation.
   That module's face/tip analysis does not apply to this experiment.

2. **The van der Waals conclusion reversed twice.** First the inherited value
   was used throughout; then, comparing against an assumed 2.5 k_BT threshold,
   the converged value was called marginal and the V13 `npvdw` value "ruled
   out"; finally, once the registration cost was included and the threshold was
   derived rather than assumed, the **converged** value turned out to be the
   one that works and the inherited one fails to condense at all. The earlier
   statements were wrong because the threshold was assumed instead of computed.

3. **A stated prediction was falsified.** The mechanism was predicted to give
   little or no hysteresis at the lower edge, since the dense phase there just
   becomes unstable. The measured lower edge has ~10 K of hysteresis, the same
   as the upper edge, and it grows with ramp rate. That killed the
   "barrier-free spinodal" reading and forced the kinetic-lag interpretation
   now recorded in section 6.

Two further corrections were made inside the present framework: the frozen
dipole distribution must be *inherited* from the pair's freeze temperature
rather than assumed uniform, and the blocked fraction must not be computed with
`geometry_model.barrier_distribution`'s 11-node Gauss–Hermite rule, which is
15 % wrong at 270 K because the survival factor is nearly a step in the barrier.
