# Integrating `npvdw` into the three existing V10 models

This is the handover document for the next stage. Nothing in
`versions/V10_multiphysics_nanocube/` has been modified.

## 1. What the existing code actually is

The names are misleading, so this is spelled out before anything else.

| File | What it really is |
|---|---|
| `geometry_model.py` | The shared library. Holds `V620Parameters`, the rounded-cube gap, the centre-distance solver, the 4³ voxel vdW sum, the magnetic Hamiltonian and its own `main()`. Byte-identical copy also sits at `v620_pair_model.py`. |
| `pair_energy_model.py` | **Not** an energy model. It supplies `hexane_viscosity_Pa_s` and `brownian_time_s`, plus a `geometry_data` helper that forwards to `geometry_model`. |
| `pair_free_reference.py` | Entry point. 64-well magnetic thermodynamics at fixed geometry. |
| `rotational_entropy_model.py` | Entry point. Cage-orientation sensitivity. Also carries its **own second** vdW implementation, `central_vdw`, a sharp-cube convolution sum used only as a diagnostic. |
| `finite_window_rotational_entropy.py` | Entry point. Calls **no** geometry function directly; it composes `pair_free_reference.compute_point` and `rotational_entropy_model.orientation_fraction`. |
| `axial_face_free_tip.py`, `brownian_configuration_hopping.py`, `coupled_axial_brownian_neel.py`, `neel_cooling_history.py`, `protocol.py`, `v10_11`…`v10_20` | Further entry points, all routed through the same two `geometry_model` functions. |

So there are not three independent models. There is **one** geometry and
energy library with many entry points, and the whole vdW surface is two
functions.

## 2. The integration surface is two functions

Every vdW and gap call site in `V10_multiphysics_nanocube/code` reduces to:

```python
geometry_model.center_distance_at_gap_m(direction_body, target_gap_m, params)
geometry_model.pair_vdw_energy_J(center_vector_body_m, params)
```

Call sites (verified by grep over the package):

- `center_distance_at_gap_m` — `pair_free_reference.py:117`,
  `rotational_entropy_model.py:96,107`, `axial_face_free_tip.py:133`,
  `brownian_configuration_hopping.py:83`, `coupled_axial_brownian_neel.py:75`,
  `neel_cooling_history.py:44`, `pair_energy_model.py:123`, `protocol.py:155`,
  and six `v10_1x` scripts.
- `pair_vdw_energy_J` — `pair_free_reference.py:146`,
  `axial_face_free_tip.py:137`, `brownian_configuration_hopping.py:106`,
  `coupled_axial_brownian_neel.py:114`, `neel_cooling_history.py:46`,
  `pair_energy_model.py:136`, `protocol.py:256`, and six `v10_1x` scripts.
- Third, separate implementation: `rotational_entropy_model.central_vdw`.

Replace those two functions and every entry point picks up the new physics
with no other edit. Concretely, add to `geometry_model.py`:

```python
import dataclasses
import npvdw

# Recommended working configuration: the SI-mesh calibration on its own
# 27/386 discretisation, with the repulsive prefactor frozen at the 13.37 nm
# reference (transfer plan 3).  Do NOT use singh_calibrated_continuum here:
# it has no potential well, and do NOT let K_W follow a^6 above 13.37 nm.
# See FINDINGS.md sections 4, A1d and C2.
VDW_PARAMETERS = dataclasses.replace(
    npvdw.load_parameters("singh_calibrated_si_mesh"),
    kw_reference_edge_nm=13.37,
    kw_reference_hamaker_kcal_per_mol=npvdw.SINGH_HAMAKER_KCAL_PER_MOL,
)

def superellipsoid_gap_m(center_vector_body_m, params=PARAMS):
    a = npvdw.Placement.create(params.particle_size_nm, [0.0, 0.0, 0.0])
    b = npvdw.Placement.create(params.particle_size_nm,
                               np.asarray(center_vector_body_m) * 1e9)
    return npvdw.gjk_distance(a, b)["distance_nm"] * 1e-9

def center_distance_at_gap_m(direction_body, target_gap_m, params=PARAMS):
    return 1e-9 * npvdw.centre_distance_for_gap(
        params.particle_size_nm, target_gap_m * 1e9, direction_body)

def pair_vdw_energy_J(center_vector_body_m, params=PARAMS):
    a = npvdw.Placement.create(params.particle_size_nm, [0.0, 0.0, 0.0])
    b = npvdw.Placement.create(params.particle_size_nm,
                               np.asarray(center_vector_body_m) * 1e9)
    vdw = VDW_PARAMETERS.with_hamaker_J(params.hamaker_J)
    return npvdw.pair_energy(a, b, vdw)["u_pair_J"]
```

Sanity value for the shim: at `particle_size_nm = 16.0`,
`hamaker_J = 2.0e-20` and a centre vector of 19.0 nm along [100] this returns
**-2.0560e-20 J**, i.e. -2.9593 kcal/mol or -4.995 kBT at 298.15 K. The
legacy function returns -4.8834e-21 J for the same geometry, so expect the
inherited figures to change by roughly a factor of four once the repulsive
term and the eps1 energy scale are both in.

### Required signature changes

The current signatures cannot express the new physics and must widen:

1. **Orientation.** `pair_vdw_energy_J` takes only a centre vector, so both
   particles are implicitly co-oriented with the body frame. The superellipsoid
   potential is orientation dependent. Add two rotation matrices, defaulting to
   identity so existing calls keep working:
   `pair_vdw_energy_J(center_vector_m, rotation_i=None, rotation_j=None, params=PARAMS)`.
2. **Component return.** Callers now need the attraction and repulsion
   separately, plus the overlap flag. Prefer a new
   `pair_vdw_components(...) -> dict` returning `u_attr_J`, `u_rep_J`,
   `u_pair_J`, `gap_m`, `overlap`, and keep `pair_vdw_energy_J` as a thin
   wrapper on `u_pair_J`.
3. **Overlap.** `npvdw` returns `+inf` for interpenetrating pairs. Every
   Boltzmann factor and every master-equation rate in the callers must treat
   `+inf` as zero weight rather than propagating a NaN. `pair_free_reference`
   and `brownian_configuration_hopping` currently have no such guard.
4. **`central_vdw`.** Delete it or make it call `npvdw`. Keeping a second,
   sharp-cube vdW in the tree invites silent disagreement.
5. **`exact_rounded_cube_gap_m`.** Keep it, renamed, for audit reproductions
   only. It is a different body (Minkowski box ⊕ ball, 1.5 nm radius) and gives
   different centre distances: at a 3 nm gap and a = 16 nm it differs from the
   superellipsoid by 0.0 nm along [100], −1.23 nm along [110] and −2.44 nm
   along [111].

### Interface contract

`npvdw.pair_energy` returns the **full single-pair** interaction. Neighbour
sums and per-particle normalisation stay in the caller. The existing
`coordination_factor_per_nc` convention, `(n/2) × U_pair`, is unaffected and
should keep living in the caller.

## 3. Volume: the change that propagates furthest

`V620Parameters.particle_volume_m3` returns `(a·1e-9)³`, the **sharp cube**.
The superellipsoid volume is 0.900959 a³. That 9.90 % reduction is not
confined to the vdW term. For a = 16 nm:

| Quantity | Sharp cube a³ | Superellipsoid 0.900959 a³ | Change |
|---|---|---|---|
| V (m³) | 4.096000e-24 | 3.690327e-24 | −9.90 % |
| m_s = M_s V (A·m²) | 1.167360e-18 | 1.051743e-18 | −9.90 % |
| E_dd ∝ m_s² | — | — | **−18.83 %** |
| K_eff = ΔE/V (J/m³) | 2.136082e4 | 2.370899e4 | +10.99 % |
| `magnetocrystalline_anisotropy_Jpm3` = 12 K_eff | 2.563299e5 | 2.845079e5 | +10.99 % |

So switching the geometry silently rescales the dipolar coupling by −18.8 %
and the inferred anisotropy constant by +11.0 %. **Audit `M_s V` and `K_1 V`
in the same commit that changes the body.** Do not leave the sharp-cube volume
in the magnetic terms while the vdW term uses the superellipsoid.

## 4. Magnetic anisotropy: sign convention and what K₁ may be called

Adopt explicitly:

```
E_ani = K1 · V · (ax² ay² + ay² az² + az² ax²)
```

with `a` the moment direction cosines in the particle frame. For magnetite at
room temperature the starting default is **K1 = −1.25e4 J/m³**; with K1 < 0
the easy axes are ⟨111⟩ and the barrier to an adjacent ⟨111⟩ well through the
⟨110⟩ saddle is |K1| V / 12. Verify and cite the primary source before use;
the value is temperature dependent and not a constant across a wide sweep.

The existing code writes `E = −K_code V (ax²ay² + ax²az² + ay²az²)` with
`K_code > 0`, so `K1 = −K_code`. The two forms agree, but the stored constant
has the opposite sign to the convention above; converting without flipping the
sign will invert easy and hard axes.

**Keep the intrinsic constant and the effective barrier separate.** The code
currently back-solves K from a blocking condition:
`ΔE = k_B · 250 K · ln(100 s / 0.98 ns) = 8.749e-20 J`, then
`K_code = 12 ΔE / V`. With the superellipsoid volume that gives
K_code = 2.845e5 J/m³, i.e. |K1| = 2.845e5 J/m³, which is **22.8×** the bulk
room-temperature value. The bulk constant alone would give a barrier of
|K1| V/12 = 3.844e-21 J = 0.93 k_BT at 300 K, against 8.749e-20 J = 21.1 k_BT
from the ZFC/FC condition. That factor of 22.8 cannot be intrinsic
magnetocrystalline anisotropy; surface and shape anisotropy, interparticle
coupling and the size distribution all contribute. Store the two numbers in
different fields, name the effective one as effective, and state the
constant-K approximation's validity range wherever a temperature sweep uses it.

## 5. Configurational entropy, if it is added later

Before writing any entropy term, state which degrees of freedom it covers:
particle positions, rigid-body orientations, or moment orientations. Then fix
a partition function and a common reference state, and report free energy and
mean energy in separate columns. Two specific traps in the present tree:

- If a Monte Carlo run already samples a degree of freedom, do not add a cage
  entropy for that same degree of freedom on top. `finite_window_rotational_entropy`
  adds an SO(3) cage measure to an inherited generator whose body orientations
  are frozen; that is consistent only because the generator does not sample
  them. Once rigid-body Brownian motion is sampled, the cage term must go.
- A finite-time Néel relaxation result is not an equilibrium configurational
  free energy. The 20 s and 100 s window outputs are observation-window
  quantities and must not be relabelled as free energies.

## 6. Suggested order of work

1. Add `npvdw` to the path and write the three shim functions above, keeping
   the old ones under `legacy_` names for the audit reproductions.
2. Add the `+inf` overlap guards in every caller that Boltzmann-weights a
   pair energy.
3. Switch `particle_volume_m3` to the superellipsoid volume and, in the same
   commit, re-derive `m_s`, `K_eff` and every figure that depends on them.
4. Flip the anisotropy constant to the `E_ani = K1 V (…)` convention with
   K1 < 0 and split intrinsic from effective.
5. Widen `pair_vdw_energy_J` to accept orientations and return components.
6. Delete `rotational_entropy_model.central_vdw`.
7. Re-run `run_validation.py`, which contains the legacy cross-check, and
   confirm the D9 legacy-vs-superellipsoid ratios still match this report.

## 7. Performance, for whoever wires this into a Monte Carlo loop

Measured on this machine, single-threaded:

| operation | cost |
|---|---|
| `pair_energy` on the SI 27/386 mesh | 0.7 ms |
| `pair_energy` on the production mesh (8000 volume + 6144 surface nodes) | ~1.8 s |
| `gjk_distance` alone | ~0.1 ms |
| `point_to_body_distance`, 6144 points, batched | 21 ms |

Meshes are cached in `potential._MESH_CACHE`, keyed on (edge, scheme, n,
convention), so the per-pair cost is only the distance arithmetic.

Two consequences for the Markov chain Monte Carlo stage:

1. **Do not call the production mesh inside an MC loop.** At 1.8 s per pair a
   27-particle system needs about 10 minutes for one full energy evaluation.
   Either run the MC on the SI 27/386 mesh, which is what Singh et al. did, or
   precompute the potential on a grid of (gap, relative orientation) and
   interpolate, validating the interpolation against direct evaluation.
2. **Use incremental updates.** A single-particle move changes only that
   particle's pair terms, so recompute those and reuse the rest.

If a tabulated potential is built, tabulate `u_attr` and `u_rep` separately.
They have different ranges and different smoothness, and keeping them apart
preserves the ability to rescale eps1, eps2 or the overall factor s without
rebuilding the table.
