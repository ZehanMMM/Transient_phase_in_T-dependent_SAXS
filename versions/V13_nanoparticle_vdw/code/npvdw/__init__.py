"""Standalone nanoparticle van der Waals module (sextic superellipsoid cores).

Public surface intended for the three existing V10 models:

    from npvdw import Placement, VdwParameters, pair_energy, load_parameters

    result = pair_energy(particle_i, particle_j, params, temperature_K=298.15)
    result["u_attr_J"], result["u_rep_J"], result["u_pair_J"]

``pair_energy`` always returns the FULL single-pair interaction.  Neighbour
summation and per-particle normalisation stay with the caller.
"""
from .units import (  # noqa: F401
    AVOGADRO,
    BOLTZMANN_J_PER_K,
    KCAL_PER_MOL_IN_J,
    SINGH_HAMAKER_J,
    SINGH_HAMAKER_KCAL_PER_MOL,
    J_to_kBT,
    J_to_kcalmol,
    hamaker_J_to_kcalmol,
    kcalmol_to_J,
    kcalmol_to_kBT,
)
from .superellipsoid import Superellipsoid, exact_volume_coefficient  # noqa: F401
from .gjk import (  # noqa: F401
    Placement,
    SYMMETRY_DIRECTIONS,
    analytic_gap_coaligned,
    analytic_support_height,
    centre_distance_for_gap,
    centre_distance_for_gap_coaligned,
    gjk_distance,
    is_symmetry_direction,
    overlaps,
    point_to_body_distance,
)
from .potential import (
    energy_kernels,
    kernel_energies,  # noqa: F401
    VdwParameters,
    attraction_kcalmol,
    energy_curve,
    face_to_face_pair,
    find_well,
    pair_energy,
    repulsion_kcalmol,
)
from .config import (  # noqa: F401
    CONFIG_DIR,
    available_parameters,
    load_parameters,
    save_parameters,
)

__all__ = [n for n in dir() if not n.startswith("_")]
