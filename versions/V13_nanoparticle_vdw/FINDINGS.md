# Findings

Numbers here are produced by the scripts in `code/` and mirrored in
`outputs/*/`. Energies are the **full pair** interaction `U_pair` unless a
row says per particle.

## The five questions, answered with numbers

**1. How well is the literature result reproduced?**

The *form* of Eqn. 4 reproduces Fig. S28E to 0.057 kcal/mol rms, which is the
digitisation noise floor of 0.060, and hits the published minimum to +0.0021 nm
and −0.016 kcal/mol — **but only with refitted constants**
(ε₁, ε₂, β) = (33.752, 94.797, 7.0375 nm) and only on the SI's own 27/386
mesh. The *printed* constants (130, 290, 9.56 nm) give U_pair(2.99 nm) =
−23.884 kcal/mol, 10.3× too deep, and **no minimum at any separation**. In the
well region the reconstruction sits about 0.14 kcal/mol (6 %) below the
published curve, which is the honest measure of how close it gets without the
authors' node coordinates. Separately, the published minimum is a **quadrature
artifact**: it disappears at every continuum refinement level, because the
converged attraction diverges as 1/D² at contact while the repulsion is
bounded by ε₂K_W·S/β⁸ = 913.8 kcal/mol.

**2. How many integration nodes are needed?**

| integral | nodes per particle | accuracy |
|---|---|---|
| surface, `exact_surface` | 216 (6×6²) | 0.1 % |
| surface, `exact_surface` | 384 (6×8²) | 0.002 % |
| surface, `exact_surface` | 6144 (6×32²) | machine precision |
| volume, Cartesian | 1728 (12³) | 0.86 % at a 2.99 nm gap |
| volume, Cartesian | 8000 (20³) | 0.13 % at 2.99 nm, 0.50 % at 1 nm |
| volume, Cartesian | 13824 (24³) | 0.030 % at 2.99 nm, 0.11 % at 1 nm |

**Production mesh: Cartesian n = 20 (8000 volume nodes) + facegl n = 32 (6144
surface nodes), `r2_mode='exact_surface'`**, at about 2 s per pair. The radial
rule must not be used below about a 2 nm gap: at 8192 nodes it is 4.9 % off
there, against 0.50 % for the Cartesian rule at 8000 nodes.

**3. The 16 nm particles at a 3 nm gap** (A = 2.0e-20 J, SI-mesh constants,
transfer plan 3, T = 298.15 K, centre line along [100], R = 19.000000 nm):

| component | kcal/mol | J | kBT |
|---|---|---|---|
| U_attr | −14.6364 | −1.01689e-19 | −24.703 |
| U_rep | +11.6771 | +8.11288e-20 | +19.709 |
| **U_pair** | **−2.9593** | **−2.05601e-20** | **−4.995** |
| U_pair / 2 | −1.4797 | −1.02800e-20 | −2.497 |

Along [110]: U_pair = −1.4675 kcal/mol = −2.477 kBT. Along [111]: −0.9262
kcal/mol = −1.563 kBT.

**4. Where is the well after parameter transfer?**

| edge | plan 1 | plan 2 | plan 3 (recommended) |
|---|---|---|---|
| 13.0 nm | none | 2.9093 nm / −2.2515 | 3.3001 nm / −1.74945 |
| **16.0 nm** | 9.727 nm / −0.341 (spurious secondary) | 3.5807 nm / −2.2515 | **3.6224 nm / −3.21876** |
| 19.5 nm | none | 4.3640 nm / −2.2515 | 3.9732 nm / −4.92141 |

Depths in kcal/mol. Plan 1, the literal SI transfer, turns 16 and 19.5 nm
particles **net repulsive** (+30.8 and +211.8 kBT at a 3 nm gap) because K_W
grows as a⁶ and the surface integral already carries a², so the steric term
grows as a⁸. Plan 3 freezes the repulsive prefactor at the 13.37 nm reference
and is the physically defensible choice for a fixed oleate chain length.

**5. What has to change in the three models?**

The whole vdW surface is **two functions** in `geometry_model.py`:
`center_distance_at_gap_m` and `pair_vdw_energy_J`. Replace those and every
entry point follows. Five signature or behaviour changes are required: accept
two rotation matrices, return the attractive and repulsive components
separately, guard `+inf` on overlap in every Boltzmann weight, delete the
second vdW implementation in `rotational_entropy_model.central_vdw`, and keep
the Minkowski rounded-cube gap only for audit. Switching the body also forces
a magnetic audit in the same commit: the volume falls by 9.90 %, so m_s falls
9.90 %, E_dd falls 18.83 %, and the inferred anisotropy constant rises
10.99 %. Full detail and an ordered work list are in `INTEGRATION.md`.

## 0. Sources

`references/singh.sm.pdf` was read: Eqn. 4 (p. 23, rendered equation checked
against the text), Fig. S28E and its caption (p. 22), the superellipsoid and
the support function (pp. 23–25), K_W and the well statement (p. 24).

The **Lee–Arya SI is not in this repository.** `references/` contains only
`singh.sm.pdf`. A full-text search for "Arya" across the tree hits only the
bibliographies of three unrelated cached papers under `tmp/`. Nothing in this
deliverable is cross-checked against a Lee–Arya document, and no result
depends on one. If that SI is supposed to constrain the repulsion form, the
ligand thickness, or the Hamaker constant, those points remain open.

## 1. What the existing code is

One shared library, many entry points. See `INTEGRATION.md` §1. The two
points that matter for this stage:

- `pair_energy_model.py` is **not** an energy model. It supplies hexane
  viscosity and Brownian times.
- `finite_window_rotational_entropy.py` calls no geometry function directly;
  it composes `pair_free_reference.compute_point` with
  `rotational_entropy_model.orientation_fraction`.

The legacy vdW is a sharp-cube 4³ = 64 voxel Hamaker sum with **no repulsive
term**, evaluated at a centre distance derived from a *different* body: a
Minkowski-rounded cube (box half-width 6.5 nm ⊕ ball 1.5 nm). So the legacy
model already mixes two geometries: a rounded cube for the gap and a sharp
cube for the integral, with the sharp-cube volume a³ in the weights.

## 2. Stage A — reproducing Fig. S28E

### A1. Literature parameters, run exactly as printed

ε₁ = 130, ε₂ = 290, β = 9.56 nm, A = 3 kcal/mol, 27 volume elements, 386
surface elements, a = 13.37 nm.

| Quantity | Value |
|---|---|
| K_W | 2.50790e5 nm⁶·kcal/mol (SI quotes ≈2.5e5) ✓ |
| exact volume | 2153.273 nm³ = 0.9009589 a³ |
| SI's 0.9 a³ | 2150.982 nm³, 0.106 % below exact |
| surface area | 876.646 nm² (sphere 561.58, sharp cube 1072.54) |
| U_attr at 2.99 nm | **−43.840 kcal/mol** |
| U_rep at 2.99 nm | **+19.956 kcal/mol** |
| U_pair at 2.99 nm | **−23.884 kcal/mol** |
| U_pair/2 at 2.99 nm | −11.942 kcal/mol |
| SI target | −2.33 kcal/mol |
| interior minimum | **none — the curve is monotonic** |

The result is 10.3× too deep as a pair energy, 5.1× too deep as a
per-particle energy, and has no well at any separation.

### A1 diagnosis: why β = 9.56 nm cannot produce a well

A minimum requires the repulsion to fall off faster than the attraction below
the well. Measured over 1.5–8 nm:

| | mean d ln\|U\|/dR |
|---|---|
| attraction | −0.5327 nm⁻¹ |
| repulsion, β = 9.56 nm | −0.5464 nm⁻¹ |
| difference | **−0.0136 nm⁻¹** |

With β = 9.56 nm the repulsive term is very nearly a *constant multiple* of
the attractive term across the whole physically relevant range, so their sum
cannot turn around. This is independent of ε₁ and ε₂, which are linear
prefactors. It is a property of β relative to a = 13.37 nm.

### A1c. No ε₂ can rescue β = 9.56 nm (`A1c_beta_diagnosis.csv`)

Write U_pair = |U_attr|·(x − 1) with x = U_rep/|U_attr|. Since ε₁ and ε₂ are
linear prefactors, x is proportional to ε₂/ε₁ and its *shape* is set by β
alone. Measured on the SI mesh at a = 13.37 nm:

| ε₂ | max x | gap at max | x = 1 crossings | interior minima | interior maxima |
|---|---|---|---|---|---|
| 290 (literature) | **0.4552** | **3.000 nm** | 0 | none | none |
| 637.07 | 1.0000 | 3.000 nm | 2 | 6.500 nm / −0.905 | 3.000 nm / +0.000 |
| 955.61 | 1.5000 | 3.000 nm | 1 | 15.200 nm / −0.099 | 0.660 nm / +62.82 |
| 1911.21 | 3.0000 | 3.000 nm | 1 | 31.300 nm / −0.006 | none |

Two things follow.

**β = 9.56 nm is not arbitrary.** The balance extremum sits at gap = 3.000 nm,
essentially the published 2.99 nm. So β and the ratio ε₂/ε₁ = 2.231 do encode
the right *position*. That is unlikely to be chance and suggests the printed
β is meaningful.

**But the repulsion is short by a factor of 2.197.** Because x < 1 everywhere,
U_pair < 0 everywhere and strictly increasing in R: no well. And raising ε₂
until the peak reaches 1 does not recover the published well either — x then
crosses 1 twice, producing a *maximum* of exactly 0 at 3.000 nm plus a shallow
secondary minimum at 6.5 nm. Larger ε₂ pushes that minimum out to 15 and
31 nm with negligible depth. **No ε₂ at β = 9.56 nm produces a simple
−2.33 kcal/mol well at 2.99 nm.** β itself has to shrink, which is what the
calibration does (7.0375 nm).

### A1d. The converged potential has no well, for an exact reason (`A1d_contact_limit.csv`)

The repulsive integral is **bounded above** by ε₂ K_W S_total/β⁸ = 913.84
kcal/mol, because r₂ ≥ 0 everywhere. The attractive double integral is
**unbounded**. Measured at a = 13.37 nm on the converged mesh:

| gap (nm) | U_attr converged | U_attr × D² | U_rep converged | U_attr, 27 elements |
|---|---|---|---|---|
| 2.0 | −1.988e2 | −7.95e2 | +3.70e1 | −8.18e1 |
| 1.0 | −8.365e2 | −8.37e2 | +7.19e1 | −1.73e2 |
| 0.5 | −3.151e3 | −7.88e2 | +1.03e2 | −2.69e2 |
| 0.25 | −1.336e4 | −8.35e2 | +1.23e2 | −3.43e2 |
| 0.0625 | −5.685e5 | −2.22e3 | +1.42e2 | −4.16e2 |
| 0.03125 | −2.121e6 | −2.07e3 | +1.45e2 | −4.30e2 |

U_attr × D² is flat over 2 → 0.25 nm, confirming the 1/D² parallel-plate
divergence, against a repulsion saturating near 145 kcal/mol. The 27-element
sum instead stays at −430 kcal/mol even at a 0.03 nm gap, because the closest
volume-element centres of two *touching* particles are still a/3 = 4.4567 nm
apart. **That truncation, not the physics, is what creates the Fig. S28E
minimum.**

### A2. Every choice the SI leaves open

Twelve combinations of volume discretisation × surface-element area
reconstruction × r₂ definition, all at the literature (130, 290, 9.56):

- **No combination produces an interior minimum.** The only one that does,
  `global_min`, puts it at 16.4 nm with a depth of −0.099 kcal/mol, and that
  mode is not a surface-resolved integral at all.
- Reading the figure ordinate as `U_pair` or as `U_pair/2` changes the
  residual by exactly two and never creates a well. **The pair vs
  per-particle convention is not the discrepancy.**
- The `cube_param` vs `equal_solid_angle` area reconstruction moves U_pair at
  2.99 nm by 0.15 %.
- The unknown 27-element layout spans a factor of ~1.17 in U_attr across four
  reconstructions (`A2b_node_layout_probe.csv`), against the factor of ~3.9
  that would be needed. The layout ambiguity is far too small to explain it.

Conclusion: with Eqn. 4 as printed and any defensible reconstruction of the
node sets, **(ε₁, ε₂, β) = (130, 290, 9.56 nm) does not reproduce Fig. S28E.**
Either the published constants belong to a normalisation the SI does not
state, or at least one of them is a typographical error. The functional form
is not at fault — see A3.

### A3. Calibration (clearly labelled: not validation)

Same functional form, same geometry, same K_W reference normalisation, same
node sets. Only (ε₁, ε₂, β) are refitted, against the digitised Fig. S28E
curve **plus** the published minimum (position and depth as explicit
residuals). The digitisation itself recovers the printed minimum: (2.900 nm,
−2.324 kcal/mol) versus the stated (2.99, −2.33), with a median line
half-width of 0.060 kcal/mol.

| | ε₁ | ε₂ | β (nm) | curve rms (kcal/mol) | well (nm, kcal/mol) |
|---|---|---|---|---|---|
| literature | 130 | 290 | 9.56 | — | **none** |
| `singh_calibrated_si_mesh` | **33.752 ± 0.309** | **94.797 ± 1.845** | **7.0375 ± 0.0229** | **0.0574** | **(2.9921, −2.3464)** |
| `singh_calibrated_continuum` | 14.173 ± 0.174 | 7.4489 ± 0.1452 | 3.7296 ± 0.0151 | 0.2287 | **none** |

Uncertainties are the least-squares parameter standard errors scaled by the
curve residual. The well depth of the SI-mesh set is −2.3464 kcal/mol =
−1.63020e-20 J = −3.9358 kBT at 300 K, against the SI's −2.33 kcal/mol
≈ −3.91 kBT.

The SI-mesh calibration reproduces the published curve to 0.057 kcal/mol rms,
which is the digitisation noise floor (0.060), and hits the published minimum
to **+0.0021 nm and −0.0164 kcal/mol**. Ratios to the literature values are
3.85× in ε₁, 3.06× in ε₂ and 1.36× in β.

**Residual tension, stated rather than smoothed over.** The calibration targets
both the digitised curve and the printed minimum, and those two targets are not
perfectly compatible with this node reconstruction. Fitting the curve alone
gives rms 0.032 kcal/mol but puts the well at 3.155 nm, 0.165 nm too far out.
Adding the well constraints moves the well to 2.9921 nm and raises the rms to
0.057, with the residual concentrated in the well region:

| gap | model | digitised figure | difference |
|---|---|---|---|
| 1.50 nm | −0.6466 | −0.5269 | −0.120 |
| 2.00 nm | −1.7477 | −1.5957 | −0.152 |
| 3.00 nm | −2.3464 | −2.2043 | −0.142 |
| 4.00 nm | −2.0941 | −2.0848 | −0.009 |
| 6.00 nm | −1.2742 | −1.3680 | +0.094 |
| 10.00 nm | −0.4153 | −0.4958 | +0.081 |
| 15.00 nm | −0.1210 | −0.1374 | +0.016 |

So around the well the reconstruction sits about 0.14 kcal/mol, roughly 6 %,
below the published curve while matching its stated minimum. That residual is
the price of not knowing the authors' node coordinates, and it is the honest
measure of how close this reconstruction gets. Full-grid rms over
1.3–24 nm is 0.067 kcal/mol; the largest single deviation is 0.250 kcal/mol at
a 1.35 nm gap, in the steep rise where the digitised line is thickest.

So the *form* of Eqn. 4 is right and reproduces Fig. S28E to within the
reconstruction uncertainty; the *printed constants* do not. Both calibrated
sets are stored in separate files and the literature file is never
overwritten.

The two calibrated sets differ by a lot because the SI's 27-element sum is far
from the continuum limit of the same integral. Constants tuned on the coarse
mesh are discretisation-specific and must not be carried into a converged
calculation, which is why they are kept apart.

## 3. Stage B — continuum convergence at a fixed potential

Potential frozen throughout: `singh_calibrated_si_mesh` constants, a = 13.37 nm,
K_W = 6.51131e4 nm⁶·kcal/mol with the 27 kept as the reference normalisation.
Only the quadrature changes. 1 kBT at 300 K = 0.596161 kcal/mol.

### Surface integral — converges very fast

With `r2_mode='exact_surface'` and the analytic surface Jacobian, the relative
error of U_rep against the facegl n = 56 reference (18816 nodes):

| nodes per particle | face-to-face 2.99 nm | face-to-face 1.0 nm | corner 2.99 nm |
|---|---|---|---|
| 54 | 5.0e-2 | 7.4e-2 | 1.3e-1 |
| 96 | 1.9e-2 | 2.1e-2 | 4.4e-2 |
| 216 | 9.7e-4 | 9.5e-4 | 6.0e-3 |
| 384 | 2.4e-5 | 1.6e-5 | 6.4e-4 |
| 864 | 1.3e-6 | 3.3e-6 | 5.7e-6 |
| 1536 | 2.5e-8 | 7.3e-8 | 3.5e-8 |
| 3456 | 2.1e-12 | 2.4e-11 | 1.8e-12 |

**216 surface nodes already give 0.1 %, and 384 give 0.002 %.** Convergence is
effectively spectral, which is what the analytic Jacobian plus Gauss-Legendre
should deliver for a smooth integrand. The point-to-body distance carries a
two-sided certificate of ~1e-14 nm, so r₂ itself is exact to machine precision.

With `r2_mode='nearest_element'` the same quantity converges only
algebraically: 2.1e-2 at 384 nodes, 7.2e-3 at 1536, and still 2.9e-3 at 6144,
approaching the continuum value from below. This is the numerical confirmation
that `nearest_element` is bound to the partner's mesh and is not a
mesh-independent definition.

### Volume integral — the expensive one, and the rule matters

Relative error of U_attr against the Cartesian n = 26 reference (17576 nodes),
nested Cartesian rule:

| nodes (n) | face-to-face 2.99 nm | face-to-face 1.0 nm | corner 1.0 nm | cost |
|---|---|---|---|---|
| 512 (8) | 2.78e-2 | 1.87e-1 | 2.48e-2 | 0.01 s |
| 1728 (12) | 8.56e-3 | 4.56e-2 | 6.33e-3 | 0.10 s |
| 4096 (16) | 3.30e-3 | 1.42e-2 | 2.26e-4 | 0.53 s |
| **8000 (20)** | **1.26e-3** | **4.97e-3** | **1.62e-4** | **2.0 s** |
| 13824 (24) | 2.98e-4 | 1.13e-3 | 1.33e-5 | 5.6 s |

Same quantity for the radial rule, which has comparable node counts:

| nodes (n) | face-to-face 2.99 nm | face-to-face 1.0 nm | corner 1.0 nm |
|---|---|---|---|
| 3456 (12) | 3.55e-3 | 1.75e-1 | 1.27e-1 |
| 8192 (16) | 9.22e-4 | **4.85e-2** | **4.08e-2** |

**The radial rule must not be used below about a 2 nm gap.** At 8192 nodes it
is 4.9 % off at a 1 nm face-to-face gap and 4.1 % off at a 1 nm corner
approach, where the Cartesian rule at 8000 nodes is 0.50 % and 0.016 %. At a
2.99 nm gap the two rules agree to 0.1 % and either is fine. The reason is
structural:
Gauss–Legendre nodes cluster towards x = ±a/2, i.e. towards the facing faces
where the 1/r₁⁶ integrand concentrates, whereas the radial rule reaches that
layer only through its angular grid. Above ~3 nm the two rules agree to 0.2 %.
My first attempt used a radial reference and produced a meaningless 4–6 %
"error" column; the reference is now Cartesian.

### Production mesh

**Cartesian n = 20 (8000 volume nodes) + facegl n = 32 (6144 surface nodes)
with `r2_mode='exact_surface'`.** Relative change on the next refinement, to
Cartesian n = 24 + facegl n = 40 — **all twelve configuration and gap
combinations pass the 1 % criterion per component**:

| configuration | gap | ΔU_attr | ΔU_rep | ΔU_pair |
|---|---|---|---|---|
| face-to-face [100] | 1.00 nm | 3.83e-3 | 1.3e-14 | 4.90e-3 |
| face-to-face [100] | 2.99 nm | 9.65e-4 | 2.3e-15 | 1.75e-3 |
| face-to-face [100] | 8.00 nm | 3.04e-4 | 4.6e-15 | 4.57e-4 |
| face–edge [110] | 1.00 nm | 1.23e-3 | 1.1e-15 | 2.09e-3 |
| face–edge [110] | 2.99 nm | 3.79e-4 | 6.7e-16 | 1.22e-3 |
| face–edge [110] | 8.00 nm | 1.58e-4 | 1.3e-14 | 2.82e-4 |
| corner [111] | 1.00 nm | 1.49e-4 | 2.9e-15 | 4.26e-4 |
| corner [111] | 2.99 nm | 1.47e-5 | 1.6e-15 | 1.02e-4 |
| corner [111] | 8.00 nm | 3.14e-5 | 0 | 6.07e-5 |
| face-to-face, 45° twist | 1.00 nm | 3.75e-3 | 2.4e-15 | 4.79e-3 |
| face-to-face, 45° twist | 2.99 nm | 9.72e-4 | 2.3e-15 | 1.74e-3 |
| face-to-face, 45° twist | 8.00 nm | 3.04e-4 | 1.7e-15 | 4.55e-4 |

The previous step, n = 16 to n = 20, is also under 1 % per component (for
example 2.0e-3 on U_attr at a 2.99 nm face-to-face gap), so the "two
successive refinements" form of the criterion is satisfied as well.

U_pair is about −19 kBT here, far from zero, so the 1 % relative criterion is
the operative one rather than the 0.01 kBT absolute one.

One methodological point about this table. The acceptance verdict compares the
last two **Cartesian** rungs. A refinement rung that switches rule family
measures the mutual discrepancy of the two rules rather than the convergence of
either. Measured: radial n = 16 (8192 nodes) against Cartesian n = 24 (13824
nodes), U_attr disagreement

| configuration | 1.00 nm | 2.99 nm |
|---|---|---|
| face-to-face [100] | 4.74e-2 | 1.22e-3 |
| face–edge [110] | 5.01e-2 | 6.23e-3 |
| corner [111] | 4.08e-2 | 1.14e-4 |
| face-to-face, 45° twist | 6.55e-3 | 1.18e-3 |

so 4–5 % at a 1 nm gap and 0.1–0.6 % at 2.99 nm. The radial rung is therefore
reported in `B3_joint_convergence.csv` as a cross-rule check and excluded from
the verdict.

Stage D measures the same mesh against a Cartesian n = 28 reference (21952
nodes) and finds it comfortably inside 1 % at every tested separation:

| direction | gap | rel. error U_attr | rel. error U_rep |
|---|---|---|---|
| [100] | 1.00 nm | **5.81e-3** | 4e-15 |
| [100] | 2.99 nm | 1.49e-3 | 2e-15 |
| [100] | 8.00 nm | 4.71e-4 | 7e-15 |
| [110] | 1.00 nm | 1.94e-3 | 6e-16 |
| [110] | 2.99 nm | 5.97e-4 | 4e-16 |
| [110] | 8.00 nm | 2.48e-4 | 1e-14 |
| [111] | 1.00 nm | 1.89e-4 | 2e-15 |
| [111] | 2.99 nm | 1.88e-5 | 4e-16 |
| [111] | 8.00 nm | 4.71e-5 | 4e-16 |

The worst case anywhere is 0.58 %, at the hardest configuration: a 1 nm
face-to-face gap. Both acceptance checks pass at the 1 % level.

An earlier version of this study reported a 5.7 % error at a 1 nm gap. That
figure was wrong: it came from using the radial rule as the reference, and the
radial rule is itself unconverged there. The correction is recorded because it
changes the conclusion, not just a digit. The reference must come from the same
rule family; stage B now uses Cartesian n = 26 (17576 nodes) and stage D uses
Cartesian n = 28 (21952 nodes), both one step beyond the top of the refinement
ladder. All the convergence numbers quoted above are reference-free relative
changes between successive levels, so they do not depend on that choice at all.

### The well position does not converge — because there is no well

| discretisation | well |
|---|---|
| SI 27 volume / 386 surface elements | **2.9921 nm, −2.34639 kcal/mol** |
| cartesian 6 + facegl 8 | none |
| cartesian 8 + facegl 12 | none |
| cartesian 12 + facegl 16 | none |
| cartesian 16 + facegl 24 | none |
| cartesian 16 + facegl 32 | none |

**This is the central result of stage B.** The minimum in Fig. S28E is an
artifact of the 27-element quadrature, not a property of Eqn. 4. The reason is
exact, not numerical:

- the attractive double volume integral **diverges** at contact. For two
  parallel faces of area S at separation D the Hamaker result is
  U/S = −A/(12πD²), so U_attr ~ −1/D²;
- the repulsive surface integral is **bounded above** by
  ε₂ K_W S_total / β⁸, because r₂ ≥ 0 everywhere.

So for any (ε₁, ε₂, β) the converged Eqn. 4 is monotonically attractive down
to contact, and only the hard core defines a contact distance. The SI's
27-element lattice hides this because the closest volume-element centres of
two *touching* particles are still a/3 = 4.46 nm apart, which keeps the
discrete attraction finite.

A second, independent quantitative confirmation of the same point comes from
the literature parameters themselves. Writing U_pair = |U_attr|·(x − 1) with
x = U_rep/|U_attr|, on the SI mesh x peaks at **gap = 3.00 nm** — so β = 9.56 nm
and ε₂/ε₁ = 2.231 do place the attraction–repulsion balance extremum at the
published well position, which is unlikely to be chance. But the peak value is
only **0.455**, so x < 1 everywhere, U_pair < 0 everywhere and strictly
monotonic. Raising ε₂ until the peak exceeds 1 does not recover the published
well either: x then crosses 1 twice, giving a repulsive *barrier* plus a
shallow secondary minimum at large separation, with the deep primary minimum
still at contact. β itself has to shrink, which is what the calibration does.

## 4. Recommendation on which potential to use

Two defensible choices, and they answer different questions:

1. **`singh_calibrated_si_mesh` on its own 27/386 mesh.** Use this when the
   point is to be consistent with the effective potential Singh et al.
   actually used in their Monte Carlo. It reproduces their published curve to
   0.057 kcal/mol rms and their stated minimum to 0.002 nm. Accept that its
   well is a property of the coarse quadrature.
2. **Converged Eqn. 4 plus the GJK hard core.** Use this when the integrals
   have to be right. There is then no soft minimum, and the contact distance
   comes from the hard core rather than from the potential.

If a *physically* converged steric repulsion is wanted, the (r₂+β)⁻⁸ form has
to be replaced by one that diverges at contact — dropping β, or using an
Alexander–de Gennes brush term. That is a modelling decision beyond this
stage, and it is flagged rather than taken.

**Size transfer is a separate choice from the parameter set.** It is made at
the call site, and the recommendation is transfer plan 3: freeze the repulsive
prefactor ε₂K_W at the 13.37 nm reference so the steric term scales with area
rather than a⁸. See C2 for why plan 1, the literal reading of the SI, turns
16 nm and 19.5 nm particles net repulsive.

So the working default for this project is **`singh_calibrated_si_mesh`
constants, on the 27/386 mesh, transferred with plan 3.** That is what the
stage C headline numbers use.

## 5. Stage C — the present particle system

Parameters read programmatically from
`V10_multiphysics_nanocube/code/geometry_model.py`:

| Read from code | Value |
|---|---|
| edge length | 16.0 nm (compared against 13.0 and 19.5 nm) |
| Hamaker constant | 2.0e-20 J = 2.878652 kcal/mol |
| legacy voxels per axis | 4 |
| legacy Minkowski rounding | 1.5 nm |
| rhombohedral reference (a, alpha, gamma) | 21.0 nm, 74.2 deg, −15.0 deg |
| V6.20 reference gap | 4.16797 nm |
| reference temperature | 298.15 K |
| M_s | 2.85e5 A/m |
| legacy sharp-cube volume | 4096.0 nm³ |
| superellipsoid volume | 3690.327 nm³ (ratio 0.9009587) |

### C1. Centre distances must be re-solved, and the change is large

At a 3 nm surface gap, superellipsoid versus the legacy Minkowski-rounded cube:

| edge | [100] | [110] | [111] |
|---|---|---|---|
| 13.0 nm | 16.0000 / 16.0000 (0.000) | 19.3790 / 20.1421 (**−0.763**) | 21.7492 / 23.3205 (**−1.571**) |
| 16.0 nm | 19.0000 / 19.0000 (0.000) | 23.1587 / 24.3848 (**−1.226**) | 26.0760 / 28.5167 (**−2.441**) |
| 19.5 nm | 22.5000 / 22.5000 (0.000) | 27.5685 / 29.3345 (**−1.766**) | 31.1239 / 34.5788 (**−3.455**) |

Along [100] the two bodies have the same support height a/2, so the centre
distance is identical. Along [110] and [111] the superellipsoid protrudes much
less than a Minkowski-rounded cube, and the centre distance at equal gap falls
by up to 3.5 nm. Any "same gap" comparison inherited from the old model is
therefore wrong along those directions, which is why every comparison here
re-solves the centre distance.

### C2. Transfer plans, and why Plan 1 breaks

| plan | what it does | U_rep scaling |
|---|---|---|
| 1 | eps1, eps2, beta keep their source values; K_W follows the reference formula with this system's a and A | proportional to a⁸ |
| 2 | additionally beta scaled by a/13.37, full geometric similarity | proportional to a⁰ |
| 3 | eps2 K_W frozen at the 13.37 nm / 3 kcal/mol reference | proportional to a² |

**Plan 1, the literal reading of the SI, does not transfer.** K_W grows as a⁶
and the surface integral already carries a², so the steric term grows as a⁸.
With the SI-mesh constants at a 3 nm gap along [100]:

| edge | U_attr | U_rep | U_pair | U_pair |
|---|---|---|---|---|
| 13.0 nm | −10.3314 | +6.9815 | −3.3498 kcal/mol | −5.65 kBT |
| 16.0 nm | −14.6364 | **+32.9104** | **+18.2741 kcal/mol** | **+30.84 kBT** |
| 19.5 nm | −19.6122 | **+145.1237** | **+125.5115 kcal/mol** | **+211.84 kBT** |

A ligand shell of fixed oleate chain length cannot have a steric energy that
grows as the eighth power of the core size. Plan 1 is reported because it is
what the SI's own definition gives, not because it should be used above the
reference size.

**Plan 2 is exact geometric similarity.** With beta proportional to a, every
length scales together and the potential becomes invariant: both E_attr and
E_rep scale as a⁰. The well depth is then identical at all sizes and the well
position is strictly proportional to a:

| edge | beta | well gap | well depth |
|---|---|---|---|
| 13.0 nm | 6.8427 nm | 2.9093 nm | −2.25148 kcal/mol |
| 16.0 nm | 8.4218 nm | 3.5807 nm | −2.25148 kcal/mol |
| 19.5 nm | 10.2641 nm | 4.3640 nm | −2.25148 kcal/mol |

The gaps are exactly 0.22374 a, and the common depth is the reference depth
times A_new/A_ref = 0.95955. Clean, but only a sensitivity test: it requires
the ligand layer to thicken in proportion to the core, which it does not.

**Plan 3 is the physically defensible transfer.** Freezing the repulsive
prefactor leaves E_rep scaling with area. With the SI-mesh constants:

| edge | U_attr | U_rep | U_pair at 3 nm | well gap | well depth |
|---|---|---|---|---|---|
| 13.0 nm | −10.3314 | +8.6102 | −1.7212 kcal/mol (−2.91 kBT) | 3.3001 nm | −1.74945 kcal/mol (−2.953 kBT) |
| **16.0 nm** | **−14.6364** | **+11.6771** | **−2.9593 kcal/mol (−4.995 kBT)** | **3.6224 nm** | **−3.21876 kcal/mol (−5.433 kBT)** |
| 19.5 nm | −19.6122 | +15.7127 | −3.8995 kcal/mol (−6.582 kBT) | 3.9732 nm | −4.92141 kcal/mol (−8.306 kBT) |

Well gaps as a fraction of the edge: 0.2539, 0.2264, 0.2038. So the well moves
outward in absolute terms but inward relative to the particle, which is the
signature of a fixed-thickness shell on a growing core.

The well deepens and moves outward monotonically with size, which is the
expected behaviour for a fixed ligand shell on a growing core.

### C3. The 16 nm working point

Full pair energy, edge 16 nm, 3 nm surface gap, A = 2.0e-20 J, SI-mesh
constants, transfer plan 3, T = 298.15 K:

| centre line | R | U_attr | U_rep | U_pair | U_pair | U_pair/2 |
|---|---|---|---|---|---|---|
| **[100]** | **19.0000 nm** | **−14.6364** | **+11.6771** | **−2.9593 kcal/mol = −2.0560e-20 J** | **−4.995 kBT** | **−2.497 kBT** |
| [110] | 23.1587 nm | −5.2255 | +3.7580 | −1.4675 kcal/mol = −1.0195e-20 J | −2.477 kBT | −1.238 kBT |
| [111] | 26.0760 nm | −2.3806 | +1.4545 | −0.9262 kcal/mol = −6.4346e-21 J | −1.563 kBT | −0.782 kBT |

Energies in kcal/mol. At 300 K instead of 298.15 K the [100] pair energy is
−4.9639 kBT, a 0.62 % change; the kcal/mol and joule values are unchanged
because the potential itself is temperature independent.

Relative rotations, [100] centre line, 16 nm, 3 nm gap:

| orientation of particle j | R | U_attr | U_rep | U_pair |
|---|---|---|---|---|
| co-oriented | 19.0000 nm | −14.6364 | +11.6771 | −2.9593 (−4.995 kBT) |
| 45° twist about [100] | 19.0000 nm | −14.2189 | +10.7577 | −3.4612 (−5.842 kBT) |
| 15° tilt about [001] | 20.0196 nm | −9.9096 | +6.8986 | −3.0110 (−5.082 kBT) |
| 90° about [001] | 19.0000 nm | −14.6364 | +11.6771 | −2.9593 (−4.995 kBT) |

The 90° rotation reproduces the co-oriented energy to machine precision,
which is the cubic symmetry of the body acting as a free consistency check.
The 15° tilt needs a 1.02 nm larger centre distance to keep the same 3 nm
gap, and the net energy still deepens, so the anisotropy of this potential is
not a simple function of the centre distance.

### C4. Fixed physical gap versus fixed reduced gap

SI-mesh constants, transfer plan 3, [100], T = 298.15 K. The reduced gap is
held at D/a = 0.1875, the value the 16 nm particle has at a 3 nm gap:

| edge | fixed D = 3 nm | fixed D/a = 0.1875 |
|---|---|---|
| 13.0 nm | D/a = 0.2308, U_pair = −2.905 kBT | D = 2.4375 nm, U_pair = −2.443 kBT |
| 16.0 nm | D/a = 0.1875, U_pair = −4.995 kBT | D = 3.0000 nm, U_pair = −4.995 kBT |
| 19.5 nm | D/a = 0.1538, U_pair = −6.582 kBT | D = 3.6562 nm, U_pair = −8.174 kBT |

Fitting a power law in the edge length over 13 → 19.5 nm:

- at fixed physical gap, U_pair scales as a^2.02;
- at fixed reduced gap, U_pair scales as a^2.98.

The two comparisons therefore disagree by a full power of a across this size
range, which is why the comparison basis has to be stated with any size trend.

### C4b. Limits of applicability

- The constants were fitted at a single size, a = 13.37 nm, on a single mesh.
  Nothing in the SI constrains their size dependence, so Plans 1, 2 and 3 are
  three different extrapolations of the same fit, not three predictions.
- Do **not** require every size to have its well at 2.99 nm. Only Plan 2
  forces a fixed reduced well position, and it does so by assuming the ligand
  shell scales with the core.
- Overall energy enhancement uses one dimensionless factor s applied to the
  complete potential, s = 1, 2, 5. eps1 and eps2 are never enlarged separately
  and then described as an overall multiple. A unit test enforces that s
  multiplies both terms identically.

### C5. Effect decomposition versus the legacy model (16 nm, 3 nm gap, [100])

Continuum-calibrated constants throughout, converged mesh, T = 298.15 K:

| step | U_pair | change from previous |
|---|---|---|
| legacy as is (sharp cube, 4³ voxels, no repulsion) | −1.1863 kBT | — |
| legacy evaluated at the superellipsoid centre distance | −1.1863 kBT | 0.0000 |
| superellipsoid body and volume, 4³ exact-boundary rule | −1.6282 kBT | −0.4418 |
| converged quadrature (20³) | −1.4918 kBT | +0.1363 |
| plus eps1 scaling | −21.1436 kBT | −19.6518 |
| plus the repulsive term, transfer plan 1 | +17.1952 kBT | +38.3388 |
| transfer plan 3 instead (frozen repulsive prefactor) | **−7.5114 kBT** | −24.7067 |
| per-particle convention, U_pair/2 | −3.7557 kBT | +3.7557 |

Ranked by size, the effects are: the eps1 energy scale (19.7 kBT), the
repulsive term (24.7 to 38.3 kBT depending on the transfer plan), the shape
change (0.44 kBT), and the integration accuracy (0.14 kBT). The pair versus
per-particle convention is a factor of two and nothing else.

The second row shows no change because along [100] both bodies have support
height a/2, so the centre distance is unchanged; the gap-definition effect
appears only along [110] and [111], where it is −1.23 and −2.44 nm (C1). The
dominant effects are the eps1 energy scale and the repulsive term, not the
shape change and not the integration accuracy. The last row is a pure factor
of two and is listed separately so it is never mistaken for a physical effect.

### C6. The constants are bound to their mesh

Same constants on both meshes, 16 nm at a 3 nm gap:

| parameter set | mesh | nodes (V / S) | U_attr | U_rep | U_pair | well |
|---|---|---|---|---|---|---|
| si_mesh calibration | SI 27/386 | 27 / 386 | −14.636 | +32.910 | +18.274 | 9.727 nm |
| si_mesh calibration | converged | 8000 / 6144 | **−29.770** | +33.565 | +3.795 | none |
| continuum calibration | SI 27/386 | 27 / 386 | −6.146 | +22.094 | +15.948 | 7.471 nm |
| continuum calibration | converged | 8000 / 6144 | **−12.501** | +22.689 | +10.188 | none |
| literature | SI 27/386 | 27 / 386 | −56.374 | +72.334 | +15.960 | 8.825 nm |
| literature | converged | 8000 / 6144 | **−114.661** | +73.520 | −41.142 | none |

Refining the mesh multiplies the attraction by **2.034** while leaving the
repulsion within 2 %. The factor is identical for all three parameter sets, as
it must be: eps1 is a linear prefactor, so the mesh ratio is purely geometric. The attraction is the
mesh-sensitive term, and no interior well survives on the converged mesh for
any of the three sets. The apparent wells on the coarse mesh at 7.5–9.7 nm are
the spurious secondary minima of section A1c, not the published 2.99 nm well;
they appear because these rows use transfer plan 1, whose over-strong
repulsion pushes the balance point outwards. Constants fitted on one mesh must
not be evaluated on the other.

### C7. Legacy benchmark at identical geometry and parameters

Same Hamaker constant, eps1 = 1, no repulsion, 3 nm gap, 16 nm edge:

| direction | legacy R / U | superellipsoid R / U | ratio |
|---|---|---|---|
| [100] | 19.0000 nm / −4.8834e-21 J | 19.0000 nm / −6.1410e-21 J | 1.258 |
| [110] | 24.3848 nm / −9.2615e-22 J | 23.1587 nm / −1.3569e-21 J | 1.465 |
| [111] | 28.5167 nm / −2.4532e-22 J | 26.0760 nm / −4.4736e-22 J | 1.824 |

An independent re-implementation of the legacy 4³ sharp-cube sum reproduces
`geometry_model.pair_vdw_energy_J` to 1.9e-15 relative, so this compares two
verified implementations. The superellipsoid is 26 % to 82 % more attractive at
equal gap, the enhancement growing from [100] to [111] because that is where
the centre distance shrinks most.

## 6. Stage D — verification

`outputs/validation/D_validation.json`: **34 checks, 0 failures.** Highlights,
each with its measured deviation:

| check | measured |
|---|---|
| kcal/mol to J round trip | 0 |
| A = 3 kcal/mol equals 2.084309e-20 J | 1.7e-7 rel |
| K_W within 1 % of the SI's quoted 2.5e5 | 3.2e-3 rel |
| U_pair(i,j) = U_pair(j,i), random orientations | 3.0e-16 rel |
| symmetrisation adds no factor of two | 0 |
| invariance under global translation | 2.1e-14 rel |
| invariance under global rotation | 2.2e-15 rel |
| GJK equals R − 2h(d) on the symmetry axes | 6.2e-15 nm |
| GJK is never below R − 2h(d) off axis | 0 violations in 200 |
| point-to-body two-sided certificate | 7.1e-15 nm |
| volume weights sum to the exact volume | 2.2e-16 rel |
| all volume nodes strictly inside the body | 0 outside |
| surface weights sum to the surface area | 4.3e-6 rel |
| 386 surface elements from the 8×8×8 lattice | exact |
| second moment equals the closed form 3a²/40 | 2.4e-4 rel |
| far-field excess is 2.25(a/R)² | 7.7e-3 abs |
| far-field exponent equals 6 | 1.5e-3 abs (6.00153) |
| components within 1 % at gaps ≥ 2.99 nm | 1.5e-3 |
| components within 1 % at a 1 nm gap | 5.8e-3 |
| overlapping pair rejected and returns +inf | exact |
| legacy re-implementation matches `pair_vdw_energy_J` | 1.9e-15 rel |

The support heights come out as exactly 1, 2^(1/3) and 3^(1/3) times a/2 along
[100], [110] and [111]. The far-field law
U_attr = −eps1 (A/pi²) V²/R⁶ [1 + 2.25(a/R)² + ...] is confirmed with the 2.25
independently derived as 30 times the closed-form second moment 3a²/40.

`pytest -q tests`: **46 passed.**

Two corrections were made during the work and are recorded because they
changed conclusions rather than digits:

1. The projection of a point onto the body was first written as a fixed-point
   iteration on the surface normal. It oscillates above the nearly flat faces
   of a sextic superellipsoid and was replaced by a batched damped Newton solve
   on the KKT system, now accurate to ~1e-14 nm and 17 times faster.
2. `R − 2h(d)` was first used as a general analytic benchmark for the
   co-oriented gap. It is exact only along the symmetry directions, where the
   closest-point normal is forced parallel to the centre offset; off axis it is
   a strict lower bound. GJK was right and the benchmark was wrong.
