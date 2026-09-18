# `npvdw` — standalone nanoparticle van der Waals module

Sextic-superellipsoid cores, GJK contact geometry, and the Singh et al.
effective vdW pair potential with its attractive and repulsive terms kept
separate. Runs on its own; designed to be dropped into the existing V10
models later (see `INTEGRATION.md`). **Nothing under
`versions/V10_multiphysics_nanocube/` was modified.**

## Sources actually used

| Source | Status |
|---|---|
| `references/singh.sm.pdf` | Read. Eqn. 4 (p. 23), Fig. S28E and caption (p. 22), the superellipsoid and support function (pp. 23–25), the K_W and well statements (p. 24). |
| `versions/V10_multiphysics_nanocube/code/` | Read. `geometry_model.py`, `pair_energy_model.py`, `pair_free_reference.py`, `finite_window_rotational_entropy.py`, `rotational_entropy_model.py` and the entry points around them. Parameters are read programmatically, not retyped. |
| Lee–Arya SI | **Not present in this repository.** The only PDF under `references/` is `singh.sm.pdf`; a full-text search for "Arya" finds it only in the bibliography of three unrelated papers cached under `tmp/`. Nothing here is cross-checked against it, and no claim in this deliverable depends on it. |

## Layout

```
code/
  npvdw/
    units.py           nm / kcal/mol internal system, J and kBT at the boundary
    superellipsoid.py  body, support map, volume and surface quadratures
    gjk.py             GJK distance and overlap, analytic benchmarks,
                       batched exact point-to-body distance
    potential.py       Eqn. 4, the pair_energy API, kernels for recalibration
    config.py          JSON parameter sets
  configs/*.json       parameter sets; literature and calibrated never mix
  digitize_fig_s28e.py        reproducible extraction of the published curve
  run_singh_reproduction.py   stage A: literature run, ambiguity scan, calibration
  run_beta_diagnosis.py       stage A1c: why beta = 9.56 nm cannot make a well
  run_contact_limit.py        stage A1d: the converged potential has no well
  run_node_layout_probe.py    stage A2b: how far the unknown 27-element layout moves things
  run_convergence.py          stage B: continuum convergence at a fixed potential
  run_my_system.py            stage C: the present particle system
  run_validation.py           stage D: 34-check verification suite
  run_all.py                  driver for the whole sequence
  example_usage.py            minimal API example
  tests/test_npvdw.py         pytest suite (46 tests)
outputs/
  reference/           digitised Fig. S28E
  singh_reproduction/  stage A csv, json, figures
  convergence/         stage B
  my_system/           stage C
  validation/          stage D
```

## Quick start

```bash
cd code
python run_all.py
```

or one stage at a time:

```bash
cd code
python digitize_fig_s28e.py
python run_validation.py
python run_singh_reproduction.py
python run_beta_diagnosis.py
python run_contact_limit.py
python run_node_layout_probe.py
python run_convergence.py
python run_my_system.py
python example_usage.py
pytest -q tests
```

```python
from dataclasses import replace
import npvdw as V

# Recommended default: the SI-mesh calibration, transferred with plan 3
# (repulsive prefactor frozen at the 13.37 nm reference).  See FINDINGS.md C2.
params = replace(
    V.load_parameters("singh_calibrated_si_mesh"),
    kw_reference_edge_nm=13.37,
    kw_reference_hamaker_kcal_per_mol=3.0,
).with_hamaker_J(2.0e-20)

radius = V.centre_distance_for_gap(16.0, 3.0, [1, 0, 0])   # 19.0 nm
i = V.Placement.create(16.0, [0, 0, 0])
j = V.Placement.create(16.0, [radius, 0, 0])
r = V.pair_energy(i, j, params, temperature_K=298.15)
r["u_attr_J"], r["u_rep_J"], r["u_pair_J"], r["gap_nm"], r["overlap"]
# -> U_pair = -2.9593 kcal/mol = -2.0560e-20 J = -4.995 kBT
```

`pair_energy` always returns the **full single-pair** interaction `U_pair`.
`u_pair_per_particle_*` is `U_pair / 2` and is meaningful only for an isolated
two-particle system. Neighbour summation and per-particle normalisation are
the caller's job.

## Formulae and provenance

| Item | Value / form | Source |
|---|---|---|
| Body | \|x\|⁶+\|y\|⁶+\|z\|⁶ ≤ (a/2)⁶ | SI p. 23 |
| Support map | xᵢ = (a/2)·sgn(dᵢ)·\|dᵢ\|^(1/5) / (Σ\|dⱼ\|^(6/5))^(1/6) | SI Eqn. 7 |
| Support height | h(d̂) = (a/2)(Σ\|d̂ᵢ\|^(6/5))^(5/6) | derived from Eqn. 7 |
| Exact volume | 0.9009589 a³ | SI p. 22 closed form |
| Volume in K_W | 0.9 a³ (the SI's rounded value, 0.106 % below exact) | SI p. 24 |
| Attraction | −ε₁(A/π²)∬dVᵢdVⱼ/r₁⁶ | SI Eqn. 4 |
| Repulsion | +ε₂ K_W ∫dSᵢ/(r₂+β)⁸ | SI Eqn. 4 |
| K_W | ε₁(A/π²)(0.9a³/27)² = 2.5079e5 nm⁶·kcal/mol at a = 13.37 nm | SI p. 24 (quoted ≈2.5e5) |
| A | 3 kcal/mol = 2.084309e-20 J | SI p. 23 |
| ε₁, ε₂, β | 130, 290, 9.56 nm | SI p. 23 |
| Overlap test | GJK on the support map | SI p. 24 |
| Far-field law | U_attr = −ε₁(A/π²)V²/R⁶·[1 + 2.25(a/R)² + O((a/R)⁴)] | derived here; the 2.25 is 30⟨x²⟩/a² with ⟨x²⟩ = 3a²/40 exactly for this body. Used as a validation target, not an approximation inside the module. |

The **27** in K_W is a fixed reference normalisation. It is never rebound to
the number of quadrature nodes; a unit test enforces this.

### Implementation assumptions (the SI does not determine these)

1. **Volume node coordinates.** The SI says "27 identical volume elements" and
   the Fig. S28E bottom inset shows a 3×3×3 array of centres viewed down
   [111]. Assumed: centres on the regular 3×3×3 lattice at multiples of a/3,
   equal weights V/27. No coordinates are published.
2. **Surface node coordinates and areas.** The SI says "386 surface elements …
   the elements have different surface areas". 6·8²+2 = 386 is exactly the
   number of surface lattice points of an 8×8×8 cube grid, so those points,
   radially projected, are used. Areas are offered two ways, because the SI
   publishes none:
   - `cube_param` — Voronoi cells of the cube-surface parameter domain.
     Largest elements at face centres. **This ordering is opposite to the
     Fig. S28E top-inset colour scale.**
   - `equal_solid_angle` — equal solid angle per node times the exact
     R(û)²/cos γ factor. Smallest at face centres, largest at the [111]
     corners, with 3-fold corner cells. **Matches the inset.**
   Both are reported; the choice moves U_pair at a 2.99 nm gap by 0.15 %.
3. **r₂.** The SI sentence names a surface element on each particle, but
   E_rep carries only one surface measure dSᵢ, and a double sum would break
   the units of K_W (nm⁶·kcal/mol × nm² / nm⁸ = kcal/mol only for a single
   integral). Three candidates are implemented:
   - `nearest_element` — distance from element i to the nearest surface
     element of the partner. The literal reading; mesh dependent.
   - `exact_surface` — exact distance from element i to the partner's
     surface. The mesh-independent continuum limit, required for refinement.
   - `global_min` — the single GJK particle-particle distance for every
     element. Provided only for comparison; it is not a surface-resolved
     integral and is not recommended.
4. **Symmetry.** E_rep over Sᵢ alone is not exchange symmetric. The module
   returns the average of the two one-sided integrals. This is an
   implementation choice and introduces no double counting: for identical
   particles in an exchange-symmetric configuration the two one-sided
   integrals are equal, so the average equals either one and equals the
   literature expression. A unit test checks this.
5. **Hard core.** Overlap is forbidden by GJK and returns `+inf`. The soft
   repulsion is bounded (≈10³ kcal/mol at contact) and is never relied on to
   prevent interpenetration.

## Parameter sets

| File | Origin |
|---|---|
| `singh_literature.json` | ε₁, ε₂, β, A exactly as printed in the SI. Reproduces no potential well; see stage A. |
| `singh_calibrated_si_mesh.json` | Same form and same 27/386 discretisation; ε₁, ε₂, β fitted to the digitised Fig. S28E curve plus its published minimum. **Calibrated, not literature.** |
| `singh_calibrated_continuum.json` | Same form with converged quadrature; refitted separately so the coarse-mesh set is never overwritten. Fits the mid-range curve but has **no potential well**, for the reason given in `FINDINGS.md` A1d. **Calibrated, not literature.** |

Size transfer is a separate choice from the parameter set, made at the call
site through `kw_reference_edge_nm` / `kw_reference_hamaker_kcal_per_mol`:

| plan | K_W in the repulsion | E_rep size scaling | use |
|---|---|---|---|
| 1 | follows this system's a and A, per the SI formula | a⁸ | what the SI literally says; **breaks above 13.37 nm** |
| 2 | as plan 1, plus β ∝ a | a⁰ | exact geometric similarity, sensitivity test only |
| 3 | frozen at 13.37 nm / 3 kcal/mol | a² | **recommended**: fixed ligand chain length |

## Headline results

See `FINDINGS.md` for the full account. In one paragraph: the *form* of Eqn. 4
reproduces Fig. S28E to the digitisation noise floor, but only with refitted
constants and only on the SI's own 27/386 mesh; the printed
(eps1, eps2, beta) = (130, 290, 9.56 nm) produce no minimum at any separation;
and the published minimum at 2.99 nm is an artifact of that coarse quadrature,
because the converged attraction diverges as 1/D^2 at contact while the
repulsion is bounded. `INTEGRATION.md` is the handover document for wiring
this into the existing models.

## Numerical standards

These are this project's acceptance targets, not the SI's:

- ≤ 1 % relative change on two successive refinements, per component;
- ≤ 0.01 kBT absolute where a component passes through zero;
- well position stable to < 0.03 nm.

Attraction and repulsion are always tracked separately so that cancelling
errors cannot look like convergence.
