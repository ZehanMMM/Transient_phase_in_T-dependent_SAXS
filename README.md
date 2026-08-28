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

## Supporting Information

The latest concise draft is located at:

```text
versions/V10_multiphysics_nanocube/outputs/
JACS_SI_temperature_dependent_pair_energy_model_20s_concise.docx
```
