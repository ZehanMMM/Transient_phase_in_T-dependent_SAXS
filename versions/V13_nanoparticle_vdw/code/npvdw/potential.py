"""Singh et al. effective van der Waals pair potential for nanocubes.

Singh et al. SI, Eqn. 4 (verbatim from the rendered equation):

    E_ij^vdW = E_ij^attr + E_ij^rep

    E_W^attr = - eps1 * (A / pi^2) * INT INT_{V_i,V_j} dV_i dV_j / r_1^6

    E_W^rep  = + eps2 * K_W * INT_{S_i} dS_i / (r_2 + beta)^8

with (SI p. 23-24)
    A     = 3 kcal/mol            Hamaker constant of magnetite in hexane
    eps1  = 130, eps2 = 290       dimensionless fitting constants
    beta  = 9.56 nm               fitting length
    K_W   = eps1 * (A/pi^2) * (0.9 a^3 / 27)^2  ~ 2.5e5 nm^6 kcal/mol
    r_1   = distance between the centres of two volume elements in
            different nanocubes
    r_2   = "the distance between the center of a surface element of a chosen
            nanocube and the surface element of the interacting nanocube"

UNITS: nm and kcal/mol throughout (see units.py).

WHAT THE SI DOES NOT DETERMINE (each is a documented switch here):

1. r_2.  The quoted sentence names a surface element on each particle but
   E_W^rep carries only ONE surface measure dS_i, so it cannot be a double
   sum over element pairs without changing the units of K_W (nm^6 kcal/mol
   times nm^2 over nm^8 is kcal/mol only for a single integral).  Candidates:
     'nearest_element' (default for the literature discretisation) - r_2 of
         element i is the distance to the NEAREST surface element of the other
         particle.  Mesh dependent, and the literal reading of the sentence.
     'exact_surface'   - r_2 of element i is the exact distance from i to the
         other particle's SURFACE.  This is the mesh-independent continuum
         limit of 'nearest_element' and is the correct choice for refinement
         studies.
     'global_min'      - r_2 replaced by the single GJK particle-particle
         minimum distance for every element.  Provided ONLY as a comparison;
         it is not a surface-resolved integral and is not recommended.

2. Symmetry.  E_W^rep integrates over S_i only, which is not symmetric under
   i <-> j for unequal particles or asymmetric configurations.  With
   symmetrise_repulsion=True (default) the module returns the AVERAGE of the
   two one-sided integrals.  This is an IMPLEMENTATION CHOICE.  It introduces
   no double counting: for identical particles in a configuration that is
   symmetric under exchange the two one-sided integrals are equal, so the
   average equals either one of them and equals the literature expression.

3. Node layouts and element areas: see superellipsoid.py.

K_W NORMALISATION: the 27 in K_W is a fixed literature reference
normalisation tied to 0.9 a^3 / 27.  It is NEVER replaced by the actual
number of quadrature nodes.  Refining the mesh changes only r_1, r_2 and the
quadrature weights.

HARD CORE: overlap is forbidden separately via GJK.  The bounded soft
repulsion is not relied upon to prevent interpenetration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from . import units
from .gjk import Placement, gjk_distance, point_to_body_distance
from .superellipsoid import Superellipsoid


@dataclass(frozen=True)
class VdwParameters:
    """Parameter set for the Singh effective vdW potential."""

    label: str = "unnamed"
    # --- physics ---
    epsilon1: float = 130.0
    epsilon2: float = 290.0
    beta_nm: float = 9.56
    hamaker_kcal_per_mol: float = units.SINGH_HAMAKER_KCAL_PER_MOL
    overall_scale: float = 1.0  # dimensionless s, scales the COMPLETE potential
    # --- K_W reference normalisation (never rebound to the mesh) ---
    kw_element_count: int = 27
    kw_volume_convention: str = "literature"  # 'literature' = 0.9 a^3
    # K_W enters ONLY the repulsive term in this implementation (the attraction
    # is built from the quadrature weights directly).  Setting these pins the
    # repulsive prefactor to a reference size and Hamaker constant instead of
    # letting it follow a^6, which is what "transfer plan 3" does.  Leave them
    # at None to reproduce the SI's own definition.
    kw_reference_edge_nm: Optional[float] = None
    kw_reference_hamaker_kcal_per_mol: Optional[float] = None
    # --- discretisation ---
    volume_scheme: str = "grid27"
    volume_n: int = 3
    surface_scheme: str = "lattice"
    surface_n: int = 8
    surface_area_mode: str = "cube_param"
    surface_refine: int = 48
    # --- ambiguity switches ---
    r2_mode: str = "nearest_element"
    symmetrise_repulsion: bool = True
    forbid_overlap: bool = True
    provenance: str = ""

    @property
    def hamaker_J(self) -> float:
        return units.kcalmol_to_J(self.hamaker_kcal_per_mol)

    def attraction_prefactor(self) -> float:
        """eps1 * A / pi^2  [kcal/mol]."""
        return self.epsilon1 * self.hamaker_kcal_per_mol / np.pi ** 2

    def kw(self, edge_nm: float) -> float:
        """K_W = eps1 (A/pi^2) (V_ref(a)/27)^2  [nm^6 kcal/mol].

        If kw_reference_edge_nm / kw_reference_hamaker_kcal_per_mol are set,
        those are used in place of this particle's edge and Hamaker constant,
        which freezes the repulsive prefactor across sizes.
        """
        edge = self.kw_reference_edge_nm or edge_nm
        hamaker = (
            self.kw_reference_hamaker_kcal_per_mol
            if self.kw_reference_hamaker_kcal_per_mol is not None
            else self.hamaker_kcal_per_mol
        )
        prefactor = self.epsilon1 * hamaker / np.pi ** 2
        volume = Superellipsoid(edge).volume(self.kw_volume_convention)
        return prefactor * (volume / self.kw_element_count) ** 2

    def with_hamaker_J(self, hamaker_J: float) -> "VdwParameters":
        return replace(self, hamaker_kcal_per_mol=units.J_to_kcalmol(hamaker_J))

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


# --------------------------------------------------------------------------
# cached meshes
# --------------------------------------------------------------------------
_MESH_CACHE: dict = {}


def volume_mesh(edge_nm, params: VdwParameters):
    key = ("V", edge_nm, params.volume_scheme, params.volume_n, params.kw_volume_convention)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = Superellipsoid(edge_nm).volume_nodes(
            params.volume_scheme, params.volume_n, params.kw_volume_convention
        )
    return _MESH_CACHE[key]


def surface_mesh(edge_nm, params: VdwParameters):
    key = (
        "S",
        edge_nm,
        params.surface_scheme,
        params.surface_n,
        params.surface_area_mode,
        params.surface_refine,
    )
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = Superellipsoid(edge_nm).surface_nodes(
            params.surface_scheme,
            params.surface_n,
            params.surface_refine,
            params.surface_area_mode,
        )
    return _MESH_CACHE[key]


# --------------------------------------------------------------------------
# the two integrals
# --------------------------------------------------------------------------
def attraction_kcalmol(a: Placement, b: Placement, params: VdwParameters,
                       block: int = 4_000_000):
    """E^attr in kcal/mol plus the bare dimensionless double sum.

    Chunked over the nodes of particle A so that the pairwise distance array
    never exceeds ``block`` entries.  Each node carries its own quadrature
    weight, so no volume factor is applied twice and boundary nodes are never
    treated as full interior voxels (the 'cartesian' and 'radial' schemes use
    the exact superellipsoid limits; see superellipsoid.py).
    """
    ca, wa = volume_mesh(a.shape.edge_nm, params)
    cb, wb = volume_mesh(b.shape.edge_nm, params)
    xa = a.to_world(ca)
    xb = b.to_world(cb)
    rows = max(1, int(block // max(len(xb), 1)))
    total = 0.0
    for start in range(0, len(xa), rows):
        chunk = xa[start:start + rows]
        delta = chunk[:, None, :] - xb[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", delta, delta)
        total += float(
            np.sum(wa[start:start + rows, None] * wb[None, :] * r2 ** -3.0)
        )
    energy = -params.attraction_prefactor() * total
    return energy, total


def _r2_values(source: Placement, target: Placement, params: VdwParameters,
               gap_nm: float):
    """r_2 for every surface element of ``source`` relative to ``target``."""
    cs, areas = surface_mesh(source.shape.edge_nm, params)
    xs = source.to_world(cs)
    certificate = 0.0
    if params.r2_mode == "nearest_element":
        from scipy.spatial import cKDTree

        ct, _ = surface_mesh(target.shape.edge_nm, params)
        xt = target.to_world(ct)
        r2 = cKDTree(xt).query(xs)[0]
    elif params.r2_mode == "exact_surface":
        r2, certificate = point_to_body_distance(xs, target)
    elif params.r2_mode == "global_min":
        r2 = np.full(len(xs), gap_nm)
    else:
        raise ValueError("unknown r2_mode %r" % (params.r2_mode,))
    return r2, areas, certificate


def repulsion_kcalmol(a: Placement, b: Placement, params: VdwParameters,
                      gap_nm: float):
    """E^rep in kcal/mol, symmetrised over the two one-sided surface integrals."""
    terms = []
    diagnostics = {}
    certificate = 0.0
    pairs = [(a, b, "i")] if not params.symmetrise_repulsion else [(a, b, "i"), (b, a, "j")]
    for source, target, tag in pairs:
        r2, areas, cert = _r2_values(source, target, params, gap_nm)
        certificate = max(certificate, cert)
        kw = params.kw(source.shape.edge_nm)
        value = params.epsilon2 * kw * float(
            np.sum(areas / (r2 + params.beta_nm) ** 8)
        )
        terms.append(value)
        diagnostics["r2_min_%s_nm" % tag] = float(r2.min())
        diagnostics["r2_area_weighted_mean_%s_nm" % tag] = float(
            np.sum(areas * r2) / np.sum(areas)
        )
        diagnostics["E_rep_one_sided_%s_kcalmol" % tag] = value
    energy = float(np.mean(terms))
    diagnostics["r2_certificate_nm"] = certificate
    return energy, diagnostics


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def pair_energy(
    particle_i: Placement,
    particle_j: Placement,
    params: VdwParameters,
    temperature_K: Optional[float] = None,
) -> dict:
    """Effective vdW energy of ONE pair of nanoparticles.

    Returns the FULL pair interaction U_pair.  Neighbour summation and any
    per-particle normalisation are the caller's responsibility; the
    'u_pair_per_particle_*' fields are U_pair / 2, which is the per-particle
    energy of an ISOLATED TWO-PARTICLE system only.

    Keys (energies in kcal/mol, J and kBT when ``temperature_K`` is given):
        u_attr_*, u_rep_*, u_pair_*, u_pair_per_particle_*
        overlap, gap_nm, centre_distance_nm, closest_point_i/j_nm
        plus quadrature and r_2 diagnostics.
    """
    contact = gjk_distance(particle_i, particle_j)
    gap = contact["distance_nm"]
    centre_vector = particle_j.centre_nm - particle_i.centre_nm
    out = {
        "label": params.label,
        "overlap": bool(contact["overlap"]),
        "gap_nm": gap,
        "centre_distance_nm": float(np.linalg.norm(centre_vector)),
        "closest_point_i_nm": contact["point_a"],
        "closest_point_j_nm": contact["point_b"],
        "gjk_iterations": contact["iterations"],
        "edge_i_nm": particle_i.shape.edge_nm,
        "edge_j_nm": particle_j.shape.edge_nm,
        "kw_nm6_kcalmol": params.kw(particle_i.shape.edge_nm),
        "volume_nodes": len(volume_mesh(particle_i.shape.edge_nm, params)[0]),
        "surface_nodes": len(surface_mesh(particle_i.shape.edge_nm, params)[0]),
    }

    if out["overlap"] and params.forbid_overlap:
        out.update(
            {
                "u_attr_kcalmol": np.nan,
                "u_rep_kcalmol": np.inf,
                "u_pair_kcalmol": np.inf,
                "hard_core_rejected": True,
            }
        )
        _emit_units(out, temperature_K)
        return out

    attraction, bare_sum = attraction_kcalmol(particle_i, particle_j, params)
    repulsion, rep_diagnostics = repulsion_kcalmol(
        particle_i, particle_j, params, gap
    )
    scale = params.overall_scale
    out.update(rep_diagnostics)
    out.update(
        {
            "hard_core_rejected": False,
            "attraction_bare_double_sum_nm6": bare_sum,
            "u_attr_kcalmol": scale * attraction,
            "u_rep_kcalmol": scale * repulsion,
            "u_pair_kcalmol": scale * (attraction + repulsion),
        }
    )
    _emit_units(out, temperature_K)
    return out


def _emit_units(out: dict, temperature_K):
    for name in ("u_attr", "u_rep", "u_pair"):
        value = out["%s_kcalmol" % name]
        out["%s_J" % name] = units.kcalmol_to_J(value)
        if temperature_K is not None:
            out["%s_kBT" % name] = units.kcalmol_to_kBT(value, temperature_K)
    for suffix in ("kcalmol", "J") + (("kBT",) if temperature_K is not None else ()):
        out["u_pair_per_particle_%s" % suffix] = 0.5 * out["u_pair_%s" % suffix]
    if temperature_K is not None:
        out["temperature_K"] = temperature_K


def face_to_face_pair(
    edge_nm: float,
    gap_nm: float,
    params: VdwParameters,
    direction=(1.0, 0.0, 0.0),
    rotation_i=None,
    rotation_j=None,
    temperature_K: Optional[float] = None,
):
    """Convenience: place two particles at surface separation ``gap_nm``."""
    from .gjk import centre_distance_for_gap

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    radius = centre_distance_for_gap(
        edge_nm, gap_nm, d, rotation_a=rotation_i, rotation_b=rotation_j
    )
    a = Placement.create(edge_nm, [0.0, 0.0, 0.0], rotation_i)
    b = Placement.create(edge_nm, radius * d, rotation_j)
    return pair_energy(a, b, params, temperature_K)


def energy_curve(
    edge_nm: float,
    gaps_nm,
    params: VdwParameters,
    direction=(1.0, 0.0, 0.0),
    rotation_i=None,
    rotation_j=None,
    temperature_K: Optional[float] = None,
):
    """Pair energy versus surface separation along ``direction``."""
    return [
        face_to_face_pair(
            edge_nm, float(g), params, direction, rotation_i, rotation_j, temperature_K
        )
        for g in gaps_nm
    ]


def energy_kernels(
    edge_nm: float,
    gaps_nm,
    params: VdwParameters,
    direction=(1.0, 0.0, 0.0),
    rotation_i=None,
    rotation_j=None,
):
    """Precompute the geometry-only kernels of Eqn. 4 for fast recalibration.

    Returns dict with, for every gap:
        attraction_sum  S_a = sum_mn w_m w_n / r_1^6      [nm^6 -> dimensionless]
        r2, areas       the per-element r_2 and dS_i of both one-sided integrals
    so that for any (eps1, eps2, beta)
        U_attr = -eps1 (A/pi^2) S_a
        U_rep  = eps2 K_W(eps1) mean_sides sum_i dS_i / (r_2 + beta)^8
    reproduces pair_energy() exactly.  eps1, eps2 and beta never enter the
    geometry, so calibration cannot hide a quadrature error.
    """
    from .gjk import centre_distance_for_gap

    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    attraction_sums = []
    sides = []
    gaps_out = []
    for gap in gaps_nm:
        radius = centre_distance_for_gap(
            edge_nm, float(gap), d, rotation_a=rotation_i, rotation_b=rotation_j
        )
        a = Placement.create(edge_nm, [0.0, 0.0, 0.0], rotation_i)
        b = Placement.create(edge_nm, radius * d, rotation_j)
        _, bare = attraction_kcalmol(a, b, params)
        attraction_sums.append(bare)
        pairs = (
            [(a, b), (b, a)] if params.symmetrise_repulsion else [(a, b)]
        )
        side = []
        for source, target in pairs:
            r2, areas, _ = _r2_values(source, target, params, float(gap))
            side.append((r2, areas, params.kw(source.shape.edge_nm)))
        sides.append(side)
        gaps_out.append(float(gap))
    return {
        "gaps_nm": np.array(gaps_out),
        "attraction_sum": np.array(attraction_sums),
        "sides": sides,
        "hamaker_kcal_per_mol": params.hamaker_kcal_per_mol,
        "params": params,
    }


def kernel_energies(kernels, epsilon1, epsilon2, beta_nm, overall_scale=1.0):
    """Evaluate (U_attr, U_rep, U_pair) from precomputed kernels."""
    base = kernels["params"]
    prefactor = epsilon1 * kernels["hamaker_kcal_per_mol"] / np.pi ** 2
    scale_kw = epsilon1 / base.epsilon1
    attraction = -prefactor * kernels["attraction_sum"]
    repulsion = np.empty_like(attraction)
    for index, side in enumerate(kernels["sides"]):
        values = [
            epsilon2 * (kw * scale_kw) * float(np.sum(areas / (r2 + beta_nm) ** 8))
            for r2, areas, kw in side
        ]
        repulsion[index] = float(np.mean(values))
    return (
        overall_scale * attraction,
        overall_scale * repulsion,
        overall_scale * (attraction + repulsion),
    )


def find_well(
    edge_nm: float,
    params: VdwParameters,
    direction=(1.0, 0.0, 0.0),
    bracket=(0.15, 14.0),
    samples: int = 140,
    rotation_i=None,
    rotation_j=None,
):
    """Locate the minimum of U_pair(gap): coarse scan then golden refinement.

    Returns (gap_nm, depth_kcalmol, interior) where ``interior`` is False when
    the minimum sits on the scan boundary, i.e. there is no potential well.
    """
    from scipy.optimize import minimize_scalar

    grid = np.linspace(bracket[0], bracket[1], samples)
    values = np.array(
        [
            face_to_face_pair(
                edge_nm, float(g), params, direction, rotation_i, rotation_j
            )["u_pair_kcalmol"]
            for g in grid
        ]
    )
    index = int(np.argmin(values))
    if index == 0 or index == len(grid) - 1:
        return float(grid[index]), float(values[index]), False
    lo, hi = grid[index - 1], grid[index + 1]
    result = minimize_scalar(
        lambda g: face_to_face_pair(
            edge_nm, float(g), params, direction, rotation_i, rotation_j
        )["u_pair_kcalmol"],
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x), float(result.fun), True
