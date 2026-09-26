# Fe3O4 Nanocube Assembly: Temperature- and Field-Driven SAXS Models

Energetic models for 16 nm Fe3O4 nanocubes (oleic acid, 1.25 vol.% core in
hexane) supporting the in situ SAXS/WAXS study of temperature- and
field-driven assembly. The code and outputs live under
`versions/V10_multiphysics_nanocube/`.

**For the zero-field, temperature-dependent SAXS experiment the current model
is the self-consistent blocked-dipole condensation model in
`code/binodal_phase_diagram.py`.** It supersedes the face-to-face versus
tip-to-tip pair-energy crossover used in earlier drafts of SI Section S8; the
earlier models are kept below for reference.

## Current model: blocked-dipole condensation

### What it explains

On cooling in zero field the dispersion is a single colloidal phase at 300 K,
aggregates between about 270 and 250 K (a single broad SAXS peak at 19 nm
spacing), and redissolves on further cooling. Aggregation is reversible, with
roughly 10 K of rate-dependent hysteresis at both edges. Under a static field
the same particles form a rhombohedral superlattice that only strengthens on
cooling. The model accounts for the re-entrant window, places its cold edge at
the blocking temperature, and explains why the field case is monotonic.

### Mechanism

Van der Waals attraction is a fixed energy in joules, so in units of kBT it
strengthens monotonically on cooling. The dipolar attraction behaves the
opposite way below the blocking temperature. Averaged over the 64 easy-axis
states of a face-registered pair, the dipolar energy is exactly zero, yet the
pair is attracted: while the moments can be re-sampled, the pair spends most
of its time in the head-to-tail states, and the Boltzmann-weighted average
exp(-U_dd/kBT) exceeds one. That attraction is a fluctuation effect and exists
only while the moments can still be re-sampled. At the measured 19 nm spacing a
cube in the dense phase can tilt only +/-15.5 degrees, far short of the 54.7
degrees needed to bring a different <111> axis onto the bond, so Neel
relaxation is the only way left to re-sample. As the moments block, the
dipolar attraction is removed rather than turned into a repulsion, and the sum
of the two terms passes through a minimum. The aggregate exists only where
that minimum dips below the condensation threshold.

In a field the moments are locked into a correlated, parallel arrangement
instead: parallel moments on a <100> bond contribute exactly zero, the field
supplies the orientational registry, and nothing is left to be lost on
cooling. Locking per se does not weaken the bond; locking into uncorrelated
orientations does.

### Energy expression (per face-registered bond, positive binds)

```
eps(T) = -U_vdW/kBT + [1 - b(T)^2] D(T) + Phi(T) - C_contact

D(T)    = ln[ (1/64) sum_ij exp(-U_ij/kBT) ]            annealed dipolar term
b(T)    = < exp(-t_obs / tau_N) >_size                  Neel-blocked fraction
Phi(T)  = (1/kBT) int_T^inf (-d b^2/dT') x(T') Q_z(T') kB T' dT'
                                                        inherited frozen term
Q_z(T)  = energy per bond kept by a bond frozen inside an aggregate,
          with one moment shared by z neighbours (exhaustive over 8^z)
x(T)    = particle fraction in the dense phase, from the lever rule
```

- **b^2, not b.** For a <100> bond the conditional partition function is the
  same for every frozen partner state, so one mobile moment recovers the full
  annealed value exactly.
- **Frozen bonds inherit their history.** A bond that freezes in the dilute
  phase is uncorrelated with any later bond and contributes exactly zero
  (every row of U_ij sums to zero). Only bonds that freeze inside an aggregate
  keep an attraction, and geometric frustration limits it to Q_6 = 2.64 kBT
  at 250 K (7.51 kBT for an isolated pair).
- **Self-consistent and causal.** Phi(T) depends only on x at temperatures
  already passed, so a single downward sweep closes the loop. The share of
  frozen bonds that froze inside an aggregate is an output, not a parameter.
- **Van der Waals** is the converged sharp-cube Hamaker sum, -9.156e-21 J for
  the face pair (A = 2.0e-20 J). The inherited 4^3 voxel value,
  -4.883e-21 J, is under-converged by a factor 1.875. `geometry_model.py` is
  not modified.

### Inputs

| Quantity | Value | Status |
|---|---|---|
| Core edge, spacing, gap | 16 nm, 19 nm, 3 nm | measured (TEM, SAXS) |
| Volume fraction | 1.25 % core (2.09 % effective) | measured |
| Ms | 2.85e5 A/m (55 emu/g, 300 K) | measured |
| Blocking temperature | 250 K at t = 100 s, tau0 = 0.98 ns | measured; calibrates K_cubic = 2.56e5 J/m^3 |
| Dwell per temperature step | 100 s | protocol |
| Size CV | 4.85 % | fitted to the observed window |
| Contact cost C_contact | 4.39 kBT | fitted to the observed upper edge |
| Coordination z | 6 | face-registered dense phase |

Two quantities are fitted against two observed edges, so the window position
itself is not a prediction. What is not fitted: the non-monotonic shape of the
cohesion, the placement of the cold edge at the blocking temperature (the
energy minimum sits 12-16 K above T_B for any CV between 1 and 5 %), the
smallness of the inherited frozen term, and the magnitude of the aggregated
fraction.

The contact cost deserves a caveat. Its magnitude matches the orientational
registration cost of an isolated pair at the measured tilt tolerance, but once
the six contacts of a particle share one orientation, geometry accounts for
only about 1.2 kBT per bond (`orientational_cost_kBT`). The remaining ~3.2 kBT
is attributed to ligand-shell compression and conformational entropy at the
3 nm gap, which does scale per contact. It is a constrained constant, not a
derived one.

### Key results (z = 6)

| | |
|---|---|
| Two-phase window (lever rule) | 250-268 K (observed 250-270 K) |
| Peak aggregated particle fraction | 12.8 % at 258 K |
| Deepest bond free energy | -7.30 kBT at 258 K; -6.57 at 300 K; -4.76 at 200 K |
| Inherited frozen term | at most 0.036 kBT, against a ceiling of 2.68 kBT |
| Ensemble half blocked (b = 1/2) | 244 K |
| Coordination z = 1 instead | window 239-268 K, frozen term 0.38 kBT |
| Feedback runaway | above ~18.5 degrees effective tilt (aggregate > ~88 % of particles) the dense phase never redissolves above 200 K |

The ensemble is half blocked at 244 K rather than 250 K for definitional
reasons only: b is the survival probability exp(-t/tau_N), which equals 1/e
rather than 1/2 at tau_N = t, and the lognormal is parameterised by the mean
diameter.

### Run it

```powershell
Set-Location versions/V10_multiphysics_nanocube/code
python -m pytest test_binodal_phase_diagram.py -q
python -c "from pathlib import Path; import binodal_phase_diagram as B; E = B.pair_energies('converged'); out = Path('../outputs/binodal_phase_diagram'); B.plot_two_energies(B.cooling_energy_table(E, coordination=6), out); B.plot_two_energies(B.cooling_energy_table(E, coordination=1), out, show_threshold=False)"
```

Main entry points in `binodal_phase_diagram.py`:

| Function | Role |
|---|---|
| `self_consistent_cooling` | downward sweep; returns b^2, D, Phi, eps and x at each step |
| `cooling_energy_table` | signed energies for plotting, plus the x = 1 ceiling |
| `plot_two_energies` | the figure: (a) energies and total, (b) dipolar split, (c) b and x |
| `inherited_frozen_energy` | the history integral for a prescribed x(T) |
| `frustrated_quench_kBT` | Q_z, exhaustive over 8^z neighbour configurations |
| `orientational_cost_kBT` | per-particle vs per-bond registration entropy |
| `aggregated_particle_fraction` | lever rule, particle share (not volume share) |

`loss_fraction`, `energy_decomposition`, `pair_energy_table` and their plotting
functions are the earlier scalar-lambda and single-pair versions. They are kept
because the tests use them to check that the history integral reproduces the
scalar result in its limits.

### Outputs

`versions/V10_multiphysics_nanocube/outputs/binodal_phase_diagram/`

- `two_energies_z6.png/.pdf` - the model figure, with the condensation threshold
- `two_energies_z1.png/.pdf` - the same without frustration, for comparison
- `cooling_energy_vs_temperature.png/.pdf` - panel (a) with the x = 1 ceiling shown
- `REPORT.md` - derivation, checks, limitations and revision history

### Tests

`test_binodal_phase_diagram.py`, 46 tests. Among them: the exact identities of
the 64-state spectrum; parallel moments contributing exactly zero on a <100>
bond; the history integral vanishing for uniform freezing and reproducing the
scalar result for complete in-place freezing; Q_z(T) T constant to 1 % at
z = 6; the lever rule's particle share running to one while the volume share
stays below phi/phi_dense; the self-consistent frozen term staying below 5 % of
its ceiling; and the feedback running away at 18.5 degrees. The full V10 suite
is 144 tests.

### Limitations

- Nucleation is not modelled, so the observed rate-dependent hysteresis is not
  reproduced; the model gives quasi-static boundaries.
- The dense phase is treated in a van der Waals-level (Carnahan-Starling plus
  mean-field) free energy, which gives the topology and concentration scale,
  not quantitative binodal compositions.
- The anisotropic 2D pattern points to wall-oriented, heterogeneous nucleation
  in the capillary, which is not described.
- The sweep is quasi-static at each temperature and uses the equilibrium lever
  rule for x(T); it does not follow individual particles in and out of the
  aggregate.

## Repository structure

```text
versions/V10_multiphysics_nanocube/
├── code/
│   ├── binodal_phase_diagram.py              current temperature model
│   ├── test_binodal_phase_diagram.py
│   ├── configuration_averaged_assembly.py    192-state populations, Neel quadrature
│   ├── geometry_model.py                     geometry, dipolar and vdW kernels
│   └── ...                                   earlier models, see below
└── outputs/
    ├── binodal_phase_diagram/
    └── ...
```

## Installation

Python 3.10 or later.

```powershell
python -m pip install -r requirements.txt
```

---

## Earlier models (kept for reference)

These evaluate one prescribed pair geometry and compare face-to-face and
tip-to-tip energies. The finite-time crossover they produce was the basis of
earlier drafts of SI Section S8. For the temperature-dependent SAXS experiment
they are superseded by the model above: the single broad peak indicates an
amorphous dense phase rather than a superlattice, and a pair model cannot
produce an observable aggregate at 1.25 vol.% (an isolated pair binds only
about 0.6 % of the particles at the window peak).

### Original calculation chain

```powershell
Set-Location versions/V10_multiphysics_nanocube/code
python pair_energy_model.py
python plot_energy_over_kbt.py
python observation_window_sweep.py
```

`pair_energy_model.py` generates the 20 s pair energies, `plot_energy_over_kbt.py`
converts them to kBT, and `observation_window_sweep.py` evaluates the
observation-window dependence. These use a diameter CV of zero and the
inherited voxel van der Waals sum. The crossings are finite-time pair-energy
crossovers, not thermodynamic phase boundaries.

### Free-reference audit model

The legacy rotation corrections in `pair_energy_model.py` are phenomenological.
They are not independently derived constrained free energies. The original
calculation and outputs are preserved for comparison.

An independent, penalty-free baseline is now available:

```powershell
python versions/V10_multiphysics_nanocube/code/pair_free_reference.py --window-s 20
python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_pair_free_reference.py
```

Results and a detailed Chinese explanation are in
`versions/V10_multiphysics_nanocube/outputs/free_reference/`.
This baseline computes a fixed-geometry 64-state master equation, exact
time-averaged dipolar energies, magnetic free energies relative to two uncoupled
particles, and the separate non-equilibrium free-energy excess. It does not add
or subtract a Brownian energy penalty. The 64-state approximation places each
moment exactly at an easy-axis minimum and omits intrawell thermal fluctuations.
Rigid-body rotations, ligand/solvent free energies and assembly entropy are not
simulated. Missing assembly free-energy terms are exported as NaN, not zero.
The plots are not an assembly phase diagram. No claim about SL stability follows
from their crossings.

### Continuous spins and rotational-entropy sensitivity

```powershell
python versions/V10_multiphysics_nanocube/code/rotational_entropy_model.py
python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_rotational_entropy_model.py
```

This additional equilibrium test jointly integrates continuous magnetic-moment
directions and local rigid-body cage orientations. Face cages restrict all three
orientational coordinates. Tip cages allow axial twist. Unmeasured cage widths
are scanned, with proper cubic symmetry factors, not fitted to a crossover.
Outputs are in `outputs/rotational_entropy/`. This is not a many-particle SC
calculation or a finite-time Brownian trajectory. Main curves exclude vdW and
ligand interactions. A separately labeled nominal-posture vdW diagnostic is
provided, not an orientation-dependent contact free energy.

To add the same local-cage entropy to the penalty-free 20 s Neel baseline:

```powershell
python versions/V10_multiphysics_nanocube/code/finite_window_rotational_entropy.py --window-s 20
```

Outputs are in `outputs/finite_window_rotational_entropy/`. This is a factorized
finite-time magnetic plus equilibrated-cage-entropy approximation, not coupled
Brownian/Neel dynamics. It leaves magnetic energies and probabilities unchanged.
Its illustrative 5-degree cage crossings depend on unmeasured angle widths and
retain the legacy coarse vdW calculation to isolate the added entropy term.

### Updated constraint: face axial rotation, tip full rotation

```powershell
python versions/V10_multiphysics_nanocube/code/axial_face_free_tip.py
python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_axial_face_free_tip.py
```

This model allows face cubes to rotate around the centre-to-centre axis and
tip-branch cubes to rotate in 3D. Fast allowed body rotations are integrated
adiabatically, so blocked intrinsic spins can still have negative dipolar
energies. Face slow axial-sign populations retain a 20 s Neel evolution with
angularly averaged well and saddle free energies. Three stages use the same
probability model: Udd, addition of interaction-induced orientational
entropy, and addition of the separate geometric cage-measure cost. This is not
an explicit Brownian trajectory or a literal freely rotating tip-contact
structure. See `outputs/axial_face_free_tip/REPORT_ZH.md` for assumptions.


### Cooling history with Neel dynamics only

Run `python versions/V10_multiphysics_nanocube/code/neel_cooling_history.py`.
The uniform 64-state prior is initialized once at 300 K, then inherited through
120 s / 10 K ramps and each hold down to 200 K. Both 20 s comparison holds and
150 s SAXS exposure averages are exported. No particle-body rotation, entropy
or penalty is included. Face/tip Udd, face vdW and kBT are plotted separately.
The 11 frame averages are connected with display-only PCHIP interpolation.
Raw frame averages and full state-history audit are in `outputs/neel_cooling_history/`.

### Whole-particle configuration hopping (magnetic energy only)

Run `python versions/V10_multiphysics_nanocube/code/brownian_configuration_hopping.py`.
This adds Brownian-driven transitions between equivalent cubic contact registries
to the old 64-state Neel generator. The quarter-turn network is calibrated to
the isolated Brownian orientational correlation time. Main curves contain only
Udd, with vdW shown separately. No entropy or energy-deficit penalty is added.
The baseline omits additional ligand/contact barriers and does not verify
intermediate hard-core path accessibility. It is not the earlier axial-twist
diffusion model or a prediction of assembly kinetics. Details are in
`versions/V10_multiphysics_nanocube/outputs/brownian_configuration_hopping/REPORT_ZH.md`.

The latest concise draft is located at:

```text
versions/V10_multiphysics_nanocube/outputs/
JACS_SI_temperature_dependent_pair_energy_model_20s_concise.docx
```

### Configuration-averaged colloid / superlattice populations

```powershell
python versions/V10_multiphysics_nanocube/code/configuration_averaged_assembly.py
python -m unittest discover -s versions/V10_multiphysics_nanocube/code -p test_configuration_averaged_assembly.py
```

Every model above evaluates one fixed pair geometry and reports its energy;
none of them decides whether the pair exists. This one promotes the contact
geometry to a label of the configuration space, `{free, face, tip}` times the
inherited 8 x 8 lab <111> dipole pairs, and solves the resulting 192-state
detailed-balanced master equation. It answers what fraction of the system is
colloidal against superlattice, and what the equivalent total system energy is,
over 200-300 K.

The van der Waals and gap models are the inherited ones; `geometry_model.py`
is not modified and the V13 `npvdw` potential is not used. Binding needs one
length and one concentration: detailed balance fixes the reaction volume from
the Smoluchowski encounter rate and the diffusive escape frequency, leaving the
ligand-shell escape length and the volume fraction as the only free inputs.
Both are assumptions rather than measurements and both are swept.

Results, figures and a detailed discussion are in
`versions/V10_multiphysics_nanocube/outputs/configuration_averaged_assembly/REPORT.md`,
which also records two audit findings that hold independently of the model:
the inherited 4^3 voxel van der Waals sum is under-converged by a factor 1.9,
and `barrier_distribution`'s 11-node Gauss-Hermite rule is 15 % wrong for the
blocked fraction because the survival factor is nearly a step in the barrier.

The populations are equilibrium quantities at a stated concentration, not a
phase diagram, and the superlattice mode is a mean-field coordination scaling
of a pair model rather than a many-body calculation. Quantities that the
inputs cannot support are reported as NaN rather than zero.
