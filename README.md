# V10 Magnetic Nanocube Multiphysics Model

Finite-time interaction and assembly calculations for 16 nm Fe3O4 nanocubes. The model compares face-to-face and tip-to-tip configurations during cooling.

## Model components

- Rounded-cube geometry
- Pairwise van der Waals attraction
- Dipole-dipole interaction
- Cubic magnetocrystalline anisotropy
- A 64-state Néel master equation
- Temperature-dependent n-hexane viscosity
- Brownian rotation
- Finite experimental observation windows

The current baseline uses a 16 nm magnetic core, a 1.5 nm ligand shell, a magnetization of 55 emu/g at 300 K, and a diameter coefficient of variation of zero.

## Repository structure

```text
versions/V10_multiphysics_nanocube/
├── code/
│   ├── geometry_model.py
│   ├── protocol.py
│   ├── pair_energy_model.py
│   ├── plot_energy_over_kbt.py
│   └── observation_window_sweep.py
└── outputs/
```

The repository contains one final calculation chain. `pair_energy_model.py` generates the 20 s pair energies. `plot_energy_over_kbt.py` converts them to units of `kBT`. `observation_window_sweep.py` evaluates the observation-window dependence. `protocol.py` and `geometry_model.py` provide the required protocol and geometry functions.

## Installation

Python 3.10 or later is recommended.

```powershell
python -m pip install -r requirements.txt
```

## Usage

```powershell
Set-Location versions/V10_multiphysics_nanocube/code
python pair_energy_model.py
python plot_energy_over_kbt.py
python observation_window_sweep.py
```

New results are written to `versions/V10_multiphysics_nanocube/outputs`.

## Interpretation

The calculated crossings are finite-time pair-energy crossovers rather than thermodynamic phase boundaries. The pair model assumes a prescribed separation and does not describe particle encounter kinetics or full many-particle assembly.

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

## Supporting Information

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
